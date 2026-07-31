"""Auditable runtime primitives for the paired active Poisson canary.

The real SafeLIBERO entry point lives in ``scripts/run_poisson_active_canary.py``.
This module deliberately contains the pieces whose semantics must be testable
without LIBERO, CUDA, or a MuJoCo renderer: exact fixed-exposure accounting,
terminal-command tracking, Table-1 CAR, and fail-closed QP classification.

Counters use *entered-prefix* semantics.  A high-level or inner update is
counted immediately before its provider is entered; a physics substep is
counted only after ``mj_step`` returns.  That distinction permits a final
artifact to state exactly where a fail-closed decision occurred.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


HIGH_LEVEL_DT_SECONDS = 0.05
INNER_DT_SECONDS = 0.01
PHYSICS_DT_SECONDS = 0.002
INNER_UPDATES_PER_HIGH_LEVEL = 5
PHYSICS_SUBSTEPS_PER_INNER = 5
PAPER_CAR_THRESHOLD_M = 0.001


class ActiveCanaryError(RuntimeError):
    """The active canary violated an apparatus or scientific invariant."""


class ControllerTrackingInvalid(ActiveCanaryError):
    """The empirical command-tracking validity gate was crossed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def ledger_sha256(records: Iterable[Any]) -> str:
    """Hash an ordered ledger with an explicit active-canary domain tag."""

    payload = list(records)
    digest = hashlib.sha256()
    digest.update(b"vlsa-poisson-active-ledger-v1\0")
    digest.update(canonical_json_bytes(payload))
    return digest.hexdigest()


def finite_vector(value: Sequence[float], length: int, label: str) -> Tuple[float, ...]:
    try:
        output = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ActiveCanaryError("%s must be a finite length-%d vector" % (label, length)) from error
    if len(output) != length or any(not math.isfinite(item) for item in output):
        raise ActiveCanaryError("%s must be a finite length-%d vector" % (label, length))
    return output


@dataclass(frozen=True)
class FailClosedDecision:
    completion_class: str
    terminal_reason: str
    optimizer_outcome: str
    diagnostics: Mapping[str, Any]


def classify_filter_failure(
    reason: str,
    diagnostics: Mapping[str, Any],
    *,
    initial_preflight: bool = False,
) -> FailClosedDecision:
    """Map one unsolved hard-QP result onto the registered v2 outcome.

    There is intentionally no pass-through or cached-command category.  Every
    returned value means that the caller must stop before another physics step.
    """

    reason = str(reason)
    record = dict(diagnostics)
    if reason == "unsafe_or_invalid_field_start":
        if not initial_preflight:
            return FailClosedDecision(
                "barrier_invariance_lost",
                reason,
                "not_reached",
                record,
            )
        return FailClosedDecision(
            "safe_start_inadmissible",
            reason,
            "not_reached",
            record,
        )
    if reason == "no_cbf_samples":
        return FailClosedDecision(
            "field_invalid",
            reason,
            "not_reached",
            record,
        )
    if reason == "qp_postcheck_failed":
        return FailClosedDecision(
            "fail_closed_runtime",
            reason,
            "postcheck_failure",
            record,
        )
    status = str(record.get("status", "")).lower()
    infeasible = reason == "uncontrollable_cbf_constraint" or (
        reason == "qp_not_solved" and "infeasible" in status
    )
    if infeasible:
        return FailClosedDecision(
            "qp_infeasible",
            reason,
            "infeasible",
            record,
        )
    return FailClosedDecision(
        "qp_solver_failure",
        reason,
        "solver_failure",
        record,
    )


@dataclass(frozen=True)
class TrackingObservation:
    high_level_index: int
    inner_control_index: int
    physics_substep_index: int
    command: Tuple[float, ...]
    measured: Tuple[float, ...]
    error_linf_rad_s: float
    error_rmse_rad_s: float


class TrackingAudit:
    """Audit command fidelity at every completed 2 ms physics substep.

    The registered Linf and cumulative-RMSE thresholds are empirical apparatus
    validity gates, not a CBF safety certificate.  The pending command is
    cleared at substep four, which also proves that the fifth command of the
    final high-level action was observed instead of being lost at termination.
    """

    def __init__(self, *, maximum_linf_rad_s: float, maximum_rmse_rad_s: float) -> None:
        linf = float(maximum_linf_rad_s)
        rmse = float(maximum_rmse_rad_s)
        if not math.isfinite(linf) or linf < 0.0:
            raise ValueError("maximum_linf_rad_s must be finite and nonnegative")
        if not math.isfinite(rmse) or rmse < 0.0:
            raise ValueError("maximum_rmse_rad_s must be finite and nonnegative")
        self.maximum_linf_rad_s = linf
        self.maximum_rmse_rad_s = rmse
        self._pending: Optional[Tuple[int, int, Tuple[float, ...]]] = None
        self._observations: List[TrackingObservation] = []
        self._registered_command_count = 0
        self._verified_terminal_count = 0
        self._first_threshold_crossing: Optional[Dict[str, Any]] = None
        self._squared_error_sum = 0.0
        self._component_count = 0
        self._maximum_linf = 0.0

    @property
    def command_count(self) -> int:
        return self._registered_command_count

    @property
    def verified_count(self) -> int:
        return self._verified_terminal_count

    @property
    def maximum_observed_linf_rad_s(self) -> float:
        return self._maximum_linf

    @property
    def cumulative_rmse_rad_s(self) -> float:
        if self._component_count == 0:
            return 0.0
        return math.sqrt(self._squared_error_sum / self._component_count)

    def register_command(
        self,
        *,
        high_level_index: int,
        inner_control_index: int,
        qdot_command_rad_s: Sequence[float],
    ) -> None:
        if self._pending is not None:
            raise ActiveCanaryError("a new command was registered before the prior fifth substep")
        if not 0 <= int(inner_control_index) < PHYSICS_SUBSTEPS_PER_INNER:
            raise ActiveCanaryError("inner_control_index is outside the registered 0..4 range")
        command = finite_vector(qdot_command_rad_s, 7, "qdot_command_rad_s")
        self._pending = (int(high_level_index), int(inner_control_index), command)
        self._registered_command_count += 1

    def observe_post_integration(
        self,
        *,
        high_level_index: int,
        inner_control_index: int,
        physics_substep_index: int,
        measured_qvel_rad_s: Sequence[float],
    ) -> Optional[TrackingObservation]:
        if self._pending is None:
            raise ActiveCanaryError("physics was observed without a pending command")
        expected_high, expected_inner, command = self._pending
        if (int(high_level_index), int(inner_control_index)) != (
            expected_high,
            expected_inner,
        ):
            raise ActiveCanaryError("tracking callback indexes differ from the pending command")
        physics = int(physics_substep_index)
        if not 0 <= physics < PHYSICS_SUBSTEPS_PER_INNER:
            raise ActiveCanaryError("physics_substep_index is outside the registered 0..4 range")
        measured = finite_vector(measured_qvel_rad_s, 7, "measured_qvel_rad_s")
        errors = tuple(observed - requested for observed, requested in zip(measured, command))
        linf = max(abs(value) for value in errors)
        squared = sum(value * value for value in errors)
        interval_rmse = math.sqrt(squared / len(errors))
        self._squared_error_sum += squared
        self._component_count += len(errors)
        self._maximum_linf = max(self._maximum_linf, linf)
        observation = TrackingObservation(
            high_level_index=expected_high,
            inner_control_index=expected_inner,
            physics_substep_index=physics,
            command=command,
            measured=measured,
            error_linf_rad_s=linf,
            error_rmse_rad_s=interval_rmse,
        )
        self._observations.append(observation)
        if physics == PHYSICS_SUBSTEPS_PER_INNER - 1:
            self._pending = None
            self._verified_terminal_count += 1
        cumulative = self.cumulative_rmse_rad_s
        if linf > self.maximum_linf_rad_s or cumulative > self.maximum_rmse_rad_s:
            if self._first_threshold_crossing is None:
                self._first_threshold_crossing = {
                    "high_level_index": expected_high,
                    "inner_control_index": expected_inner,
                    "physics_substep_index": physics,
                    "error_linf_rad_s": linf,
                    "cumulative_rmse_rad_s": cumulative,
                    "linf_threshold_rad_s": self.maximum_linf_rad_s,
                    "rmse_threshold_rad_s": self.maximum_rmse_rad_s,
                    "command_rad_s": list(command),
                    "measured_rad_s": list(measured),
                }
            raise ControllerTrackingInvalid(
                "joint velocity tracking threshold crossed at physics substep %d "
                "(linf=%.17g, cumulative_rmse=%.17g)"
                % (physics, linf, cumulative)
            )
        return observation

    def require_complete(self, expected_commands: int) -> None:
        if self._pending is not None:
            raise ActiveCanaryError("the terminal command did not complete five physics substeps")
        if self.verified_count != int(expected_commands):
            raise ActiveCanaryError(
                "tracking verified %d commands, expected %d"
                % (self.verified_count, int(expected_commands))
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "command_count": self.command_count,
            "verified_terminal_command_count": self.verified_count,
            "observed_physics_substep_count": len(self._observations),
            "maximum_linf_error_rad_s": self.maximum_observed_linf_rad_s,
            "cumulative_rmse_rad_s": self.cumulative_rmse_rad_s,
            "maximum_linf_threshold_rad_s": self.maximum_linf_rad_s,
            "maximum_rmse_threshold_rad_s": self.maximum_rmse_rad_s,
            "final_fifth_command_verified": bool(
                self._pending is None and self.verified_count > 0
            ),
            "first_threshold_crossing": self._first_threshold_crossing,
        }


class PrefixCounters:
    """Strict 20/100/500 Hz entered/completed prefix accounting."""

    def __init__(self) -> None:
        self.high_level_steps = 0
        self.inner_control_steps = 0
        self.physics_substeps = 0

    def enter_high_level(self, index: int) -> None:
        if int(index) != self.high_level_steps:
            raise ActiveCanaryError("high-level updates are not contiguous")
        if self.inner_control_steps != INNER_UPDATES_PER_HIGH_LEVEL * self.high_level_steps:
            raise ActiveCanaryError("prior high-level update did not finish five controls")
        self.high_level_steps += 1

    def enter_inner(self, high_index: int, inner_index: int) -> None:
        expected = self.inner_control_steps
        if int(high_index) != expected // INNER_UPDATES_PER_HIGH_LEVEL:
            raise ActiveCanaryError("inner update belongs to the wrong high-level action")
        if int(inner_index) != expected % INNER_UPDATES_PER_HIGH_LEVEL:
            raise ActiveCanaryError("inner updates are not contiguous")
        if int(high_index) >= self.high_level_steps:
            raise ActiveCanaryError("inner update entered before its high-level action")
        if self.physics_substeps != PHYSICS_SUBSTEPS_PER_INNER * expected:
            raise ActiveCanaryError("prior inner command did not finish five physics substeps")
        self.inner_control_steps += 1

    def complete_physics(self, high_index: int, inner_index: int, physics_index: int) -> None:
        expected = self.physics_substeps
        expected_inner_global = expected // PHYSICS_SUBSTEPS_PER_INNER
        observed_inner_global = (
            int(high_index) * INNER_UPDATES_PER_HIGH_LEVEL + int(inner_index)
        )
        if observed_inner_global != expected_inner_global:
            raise ActiveCanaryError("physics callback belongs to the wrong inner command")
        if int(physics_index) != expected % PHYSICS_SUBSTEPS_PER_INNER:
            raise ActiveCanaryError("physics substeps are not contiguous")
        if observed_inner_global >= self.inner_control_steps:
            raise ActiveCanaryError("physics completed before its inner update was entered")
        self.physics_substeps += 1

    def exposure_complete(self, registered_high_level_steps: int) -> bool:
        high = int(registered_high_level_steps)
        return (
            self.high_level_steps == high
            and self.inner_control_steps == high * INNER_UPDATES_PER_HIGH_LEVEL
            and self.physics_substeps
            == high * INNER_UPDATES_PER_HIGH_LEVEL * PHYSICS_SUBSTEPS_PER_INNER
        )

    def record(self) -> Dict[str, Any]:
        return {
            "high_level_steps": self.high_level_steps,
            "inner_control_steps": self.inner_control_steps,
            "physics_substeps": self.physics_substeps,
            "physics_exposure_seconds": self.physics_substeps * PHYSICS_DT_SECONDS,
        }


def paper_car_endpoint(
    settled_position_m: Sequence[float],
    completed_post_step_positions_m: Iterable[Sequence[float]],
    *,
    fixed_exposure_complete: bool,
) -> Dict[str, Any]:
    """Compute the exact SafeLIBERO Table-1 CAR endpoint.

    Contact, proxy barrier, OBB drift, and intermediate physics states are not
    substitutes.  The endpoint is the maximum L1 displacement of the active
    obstacle in completed high-level post-step observations from the settled
    reference, with collision defined by strict ``> 0.001 m``.
    """

    settled = finite_vector(settled_position_m, 3, "settled_position_m")
    maximum = 0.0
    for index, position_value in enumerate(completed_post_step_positions_m):
        position = finite_vector(position_value, 3, "post_step_position[%d]" % index)
        displacement = sum(abs(value - reference) for value, reference in zip(position, settled))
        maximum = max(maximum, displacement)
    if not isinstance(fixed_exposure_complete, bool):
        raise TypeError("fixed_exposure_complete must be Boolean")
    if fixed_exposure_complete:
        collision = maximum > PAPER_CAR_THRESHOLD_M
        return {
            "available": True,
            "maximum_active_obstacle_l1_displacement_m": maximum,
            "threshold_m": PAPER_CAR_THRESHOLD_M,
            "collision": collision,
            "avoidance": not collision,
            "reason": None,
        }
    return {
        "available": False,
        "maximum_active_obstacle_l1_displacement_m": maximum,
        "threshold_m": PAPER_CAR_THRESHOLD_M,
        "collision": None,
        "avoidance": None,
        "reason": "fixed_exposure_incomplete",
    }


def fixed_exposure_action_contract(
    actions: Iterable[Sequence[float]],
    *,
    expected_count: int,
    expected_sha256: str,
) -> Tuple[Tuple[float, ...], ...]:
    """Freeze and re-hash the exact historical 7-D action sequence."""

    frozen = tuple(
        finite_vector(action, 7, "actions[%d]" % index)
        for index, action in enumerate(actions)
    )
    if len(frozen) != int(expected_count):
        raise ActiveCanaryError(
            "historical action count differs: expected %d, observed %d"
            % (int(expected_count), len(frozen))
        )
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ActiveCanaryError("expected action ledger SHA-256 is invalid")
    historical_digest = hashlib.sha256(
        canonical_json_bytes([list(action) for action in frozen])
    ).hexdigest()
    if historical_digest != expected_sha256:
        raise ActiveCanaryError(
            "frozen actions do not match the historical executed-sequence SHA-256"
        )
    return frozen


__all__ = [
    "ActiveCanaryError",
    "ControllerTrackingInvalid",
    "FailClosedDecision",
    "HIGH_LEVEL_DT_SECONDS",
    "INNER_DT_SECONDS",
    "INNER_UPDATES_PER_HIGH_LEVEL",
    "PAPER_CAR_THRESHOLD_M",
    "PHYSICS_DT_SECONDS",
    "PHYSICS_SUBSTEPS_PER_INNER",
    "PrefixCounters",
    "TrackingAudit",
    "canonical_json_bytes",
    "classify_filter_failure",
    "fixed_exposure_action_contract",
    "ledger_sha256",
    "paper_car_endpoint",
]
