"""Paired transport-only CFS-00A canary layered over the frozen R05A runner.

The historical :func:`run_r05a_canary` remains the owner of source restoration,
target construction, Arm-A execution, and its before/after regression checks.
This module only wraps the already supplied WebSocket client.  Every one of
the two historical teacher requests is returned unchanged to the legacy
runner, while one additional request opts into the separate constrained-flow
server adapter.  No generated action is ever sent to the simulator.

The comparison response contains two distinct candidates:

* Arm B is the linearized same-budget schedule in the nested
  ``crfs_trace.linearized_warm_start`` diagnostic.
* Arm C is the historical 128-update result initialized by Arm B and remains
  in the ordinary outer inverse-flow trace.

Both candidates are replayed through ordinary ``residual_schedule`` requests.
Finite failure is recorded as a frozen-method negative, never as
infeasibility.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from crfs_harness.artifacts import atomic_write_json, content_hash, file_sha256, load_json

from .r02_runner import _array_from_record, _array_record, _validate_array_record
from .r05a_canary import (
    CASE_ID,
    EXPECTED_RESULT_STATUSES,
    PAYLOAD_TYPE as LEGACY_PAYLOAD_TYPE,
    _array_exact,
    _replay_summary,
    _schedule_diagnostics,
    run_r05a_canary,
)


SCHEMA_VERSION = "1.0"
PAYLOAD_TYPE = "r05a_constrained_flow_transport_canary_payload"
EXPERIMENT_ARM = "linearized_warm_start"
NESTED_TRACE_KEY = "linearized_warm_start"
EXPECTED_COMPARISON_CALLS = 2
DT_FLOAT32 = np.float32(-0.1)
FIDELITY_LIMITS: Mapping[str, float] = {
    "xyz_max_abs": 0.010,
    "xyz_rms": 0.005,
    "full_max_abs": 0.050,
    "full_rms": 0.015,
}
OUTCOME_STATUSES = {
    "mechanism_pass",
    "frozen_method_negative",
    "apparatus_inconclusive",
}
SCIENTIFIC_CONFIG_HASH = (
    "7dc2c8f63838ae4e22db8a927d87cae89daf0025c931b33e225c946abe8dc915"
)


def constrained_flow_scientific_config_hash(value: Mapping[str, Any]) -> str:
    """Hash every frozen scientific choice while excluding release bookkeeping."""

    scientific = copy.deepcopy(dict(value))
    for key in ("config_status", "ready_to_run", "blocked_on", "execution_release"):
        scientific.pop(key, None)
    preregistration = scientific.get("preregistration")
    if isinstance(preregistration, Mapping):
        preregistration = dict(preregistration)
        preregistration.pop("h100_submission_authorized", None)
        scientific["preregistration"] = preregistration
    return content_hash(scientific)


class ConstrainedFlowCanaryError(RuntimeError):
    """The paired CFS transport apparatus is malformed or incomplete."""


@dataclass(frozen=True)
class _PairedTeacherCall:
    ordinary_request: Mapping[str, Any]
    comparison_request: Mapping[str, Any]
    ordinary_reply: Mapping[str, Any]
    comparison_reply: Mapping[str, Any]
    comparison_elapsed_seconds: float


class PairedConstrainedFlowClient:
    """Add one opt-in comparison beside each of the two legacy teacher calls."""

    def __init__(self, client: Any):
        self.client = client
        self.paired_calls: list[_PairedTeacherCall] = []
        self.canonical_replies: dict[str, tuple[Mapping[str, Any], float]] = {}

    def infer(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        controls = request.get("__crfs__") if isinstance(request, Mapping) else None
        is_historical_teacher = bool(
            isinstance(controls, Mapping)
            and controls.get("intervention_mode") == "inverse_flow_teacher"
            and "experiment_arm" not in controls
        )
        ordinary_reply = self.client.infer(request)
        if not is_historical_teacher:
            return ordinary_reply
        if len(self.paired_calls) >= EXPECTED_COMPARISON_CALLS:
            raise ConstrainedFlowCanaryError(
                "legacy runner issued more than two inverse-flow teacher calls"
            )
        comparison_request = copy.deepcopy(dict(request))
        comparison_controls = copy.deepcopy(dict(controls))
        comparison_controls["experiment_arm"] = EXPERIMENT_ARM
        comparison_request["__crfs__"] = comparison_controls
        started = time.perf_counter_ns()
        comparison_reply = self.client.infer(comparison_request)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
        if not isinstance(comparison_reply, Mapping):
            raise ConstrainedFlowCanaryError("comparison policy reply is not a mapping")
        self.paired_calls.append(
            _PairedTeacherCall(
                ordinary_request=copy.deepcopy(dict(request)),
                comparison_request=copy.deepcopy(comparison_request),
                ordinary_reply=ordinary_reply,
                comparison_reply=comparison_reply,
                comparison_elapsed_seconds=float(elapsed),
            )
        )
        if len(self.paired_calls) == EXPECTED_COMPARISON_CALLS:
            # These replays intentionally occur here, before the frozen legacy
            # runner performs zero-control/eager/compiled after checks.  Any
            # process-state leak caused by either new arm is therefore caught
            # by the existing immutable R05A regression tail.
            for replicate, paired_call in zip(
                ("first", "duplicate"), self.paired_calls, strict=True
            ):
                comparison_trace, nested = _nested_diagnostic(
                    paired_call.comparison_reply
                )
                ordinary_trace = paired_call.ordinary_reply.get("crfs_trace")
                if not isinstance(ordinary_trace, Mapping):
                    raise ConstrainedFlowCanaryError(
                        "ordinary Arm-A reply has no trace for canonical replay"
                    )
                schedules = {
                    "A_historical_run_b": _finite_array(
                        ordinary_trace.get("solver_schedule"),
                        name=f"Arm A {replicate} auto-replay schedule",
                        shape=(10, 10, 32),
                        dtype=np.float32,
                    ),
                    "B_linearized_candidate": _diagnostic_schedule(
                        nested.get("schedule"),
                        name=f"Arm B {replicate} auto-replay schedule",
                    ),
                    "C_linearized_then_historical_adam": _finite_array(
                        comparison_trace.get("solver_schedule"),
                        name=f"Arm C {replicate} auto-replay schedule",
                        shape=(10, 10, 32),
                        dtype=np.float32,
                    ),
                }
                source_controls = paired_call.ordinary_request.get("__crfs__")
                if not isinstance(source_controls, Mapping):
                    raise ConstrainedFlowCanaryError(
                        "paired teacher request lost replay budget"
                    )
                budget = np.float32(source_controls.get("model_l2_path_budget"))
                for arm, schedule in schedules.items():
                    self.canonical_replies[f"{arm}:{replicate}"] = self.replay(
                        schedule, budget
                    )
        # The historical runner must receive the exact object returned by its
        # ordinary request.  The comparison is strictly side evidence.
        return ordinary_reply

    def replay(self, schedule: np.ndarray, budget: np.float32) -> tuple[Mapping[str, Any], float]:
        """Replay a recorded candidate without the experiment adapter or simulator."""

        if len(self.paired_calls) != EXPECTED_COMPARISON_CALLS:
            raise ConstrainedFlowCanaryError("canonical replay requires exactly two paired calls")
        source = self.paired_calls[0].ordinary_request
        source_controls = source.get("__crfs__")
        if not isinstance(source_controls, Mapping):
            raise ConstrainedFlowCanaryError("paired teacher request lost its controls")
        controls = {
            "noise": np.asarray(source_controls.get("noise"), dtype=np.float32).copy(),
            "intervention_mode": "residual_schedule",
            "intervention_step": 5,
            "return_trace": True,
            "return_normalized_final": True,
            "schedule": np.asarray(schedule, dtype=np.float32).copy(),
            "schedule_space": "model",
            "model_l2_path_budget": np.float32(budget),
        }
        request = copy.deepcopy(dict(source))
        request["__crfs__"] = controls
        started = time.perf_counter_ns()
        reply = self.client.infer(request)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
        if not isinstance(reply, Mapping):
            raise ConstrainedFlowCanaryError("canonical replay reply is not a mapping")
        return reply, float(elapsed)


def _scalar(value: Any, *, name: str) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ConstrainedFlowCanaryError(f"{name} must be scalar")
    item = array.reshape(()).item()
    if isinstance(item, float) and not math.isfinite(item):
        raise ConstrainedFlowCanaryError(f"{name} must be finite")
    return item


def _typed_scalar(value: Any, *, name: str, dtype: Any) -> Any:
    array = np.asarray(value)
    if array.size != 1 or array.dtype != np.dtype(dtype):
        raise ConstrainedFlowCanaryError(
            f"{name} must preserve scalar {np.dtype(dtype)}"
        )
    item = array.reshape(()).item()
    if isinstance(item, float) and not math.isfinite(item):
        raise ConstrainedFlowCanaryError(f"{name} must be finite")
    return item


def _zero_dimensional_typed_scalar(value: Any, *, name: str, dtype: Any) -> Any:
    """Decode exact unbatched Arm-B metadata without scalar coercion.

    Arm-C sampler trace values retain their one-element model batch and continue
    to use :func:`_typed_scalar`.  Arm-B dataclass and adapter metadata are
    genuinely unbatched, so accepting a length-one vector here would weaken the
    wire contract.
    """

    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype(dtype):
        raise ConstrainedFlowCanaryError(
            f"{name} must preserve zero-dimensional {np.dtype(dtype)}"
        )
    item = array.item()
    if isinstance(item, float) and not math.isfinite(item):
        raise ConstrainedFlowCanaryError(f"{name} must be finite")
    return item


def _integer_scalar(value: Any, *, name: str) -> int:
    """Return only the registered scalar int64 wire representation.

    Python and NumPy integer scalars both become zero-dimensional ``int64``
    arrays on the supported client/publisher platforms.  Requiring that exact
    representation prevents booleans and fractional floats from being silently
    accepted by ``int(...)`` truncation.
    """

    item = _zero_dimensional_typed_scalar(
        value, name=name, dtype=np.int64
    )
    if type(item) is not int:
        raise ConstrainedFlowCanaryError(f"{name} must be an exact integer")
    return item


def _boolean_scalar(value: Any, *, name: str) -> bool:
    """Return only the registered scalar bool wire representation."""

    item = _zero_dimensional_typed_scalar(
        value, name=name, dtype=np.bool_
    )
    if type(item) is not bool:
        raise ConstrainedFlowCanaryError(f"{name} must be an exact boolean")
    return item


def _finite_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ConstrainedFlowCanaryError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or not bool(np.isfinite(array).all()):
        raise ConstrainedFlowCanaryError(f"{name} must be a finite numeric array")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise ConstrainedFlowCanaryError(
            f"{name} must preserve {np.dtype(dtype)}, got {array.dtype}"
        )
    return np.ascontiguousarray(array)


def _batched_array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Remove the adapter diagnostic's retained model batch dimension."""

    array = np.asarray(value)
    if array.shape == (1, *shape):
        return _finite_array(
            array[0], name=name, shape=shape, dtype=np.float32
        )
    raise ConstrainedFlowCanaryError(
        f"{name} must have retained batch shape {(1, *shape)}, got {array.shape}"
    )


def _diagnostic_schedule(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    expected = (10, 1, 10, 32)
    if array.shape != expected:
        raise ConstrainedFlowCanaryError(f"{name} must have shape {expected}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not bool(np.isfinite(array).all()):
        raise ConstrainedFlowCanaryError(f"{name} must be finite")
    if array.dtype != np.dtype(np.float32):
        raise ConstrainedFlowCanaryError(f"{name} must preserve float32")
    return np.ascontiguousarray(array[:, 0])


def _tree_record(value: Any) -> Any:
    """Create a deterministic JSON record, hashing every numeric array leaf."""

    if isinstance(value, Mapping):
        return {str(key): _tree_record(item) for key, item in sorted(value.items())}
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise ConstrainedFlowCanaryError("diagnostic tree contains a nonfinite array")
        return _array_record(value)
    if isinstance(value, np.generic):
        # Preserve scalar dtype bytes (notably the float32 source budget) in
        # the transport record instead of widening through Python ``float``.
        return _array_record(np.asarray(value))
    if isinstance(value, (tuple, list)):
        return [_tree_record(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ConstrainedFlowCanaryError("diagnostic tree contains a nonfinite scalar")
        return value
    # WebSocket implementations can deserialize zero-dimensional leaves as a
    # scalar-like object accepted by numpy without making it an ndarray.
    array = np.asarray(value)
    if array.dtype != np.dtype("O"):
        return _array_record(array)
    raise ConstrainedFlowCanaryError(
        f"unsupported diagnostic value type {type(value).__name__}"
    )


def _fidelity_metrics(error: np.ndarray) -> dict[str, Any]:
    error = _finite_array(error, name="physical fidelity error", shape=(10, 32))
    # Match the frozen sampler's float32 metric algebra while independently
    # recomputing every value from the serialized physical error tensor.
    xyz = np.asarray(error[:5, :3], dtype=np.float32)
    full = np.asarray(error[:5, :7], dtype=np.float32)
    metrics: dict[str, Any] = {
        "xyz_max_abs": float(np.max(np.abs(xyz))),
        "xyz_rms": float(np.sqrt(np.mean(np.square(xyz), dtype=np.float32))),
        "full_max_abs": float(np.max(np.abs(full))),
        "full_rms": float(np.sqrt(np.mean(np.square(full), dtype=np.float32))),
    }
    checks = {
        key: bool(metrics[key] <= limit) for key, limit in FIDELITY_LIMITS.items()
    }
    metrics["checks"] = checks
    metrics["passed"] = all(checks.values())
    return metrics


def _reported_metrics_match(reported: Any, computed: Mapping[str, Any]) -> bool:
    if not isinstance(reported, Mapping):
        return False
    try:
        reported_values = {
            key: float(
                _typed_scalar(
                    reported.get(key),
                    name=f"reported {key}",
                    dtype=np.float32,
                )
            )
            for key in FIDELITY_LIMITS
        }
        reported_feasible = bool(
            _typed_scalar(
                reported.get("feasible"),
                name="reported feasible",
                dtype=np.bool_,
            )
        )
    except (ConstrainedFlowCanaryError, TypeError, ValueError):
        return False
    return bool(
        all(reported_values[key] == float(computed[key]) for key in FIDELITY_LIMITS)
        and reported_feasible is bool(computed["passed"])
    )


def _candidate_constraints(
    schedule: np.ndarray, *, budget: np.float32
) -> tuple[Mapping[str, Any], list[str]]:
    diagnostics, errors = _schedule_diagnostics(
        np.asarray(schedule, dtype=np.float32),
        source_budget_float32=np.float32(budget),
    )
    return diagnostics, list(errors)


def _nested_diagnostic(reply: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    trace = reply.get("crfs_trace")
    if not isinstance(trace, Mapping):
        raise ConstrainedFlowCanaryError("comparison reply has no outer CRFS trace")
    nested = trace.get(NESTED_TRACE_KEY)
    if not isinstance(nested, Mapping):
        raise ConstrainedFlowCanaryError("comparison reply has no constrained-flow diagnostic")
    return trace, nested


def _adapter_checks(nested: Mapping[str, Any]) -> Mapping[str, bool]:
    def integer_equals(name: str, expected: int) -> bool:
        try:
            return _integer_scalar(nested.get(name), name=name) == expected
        except (ConstrainedFlowCanaryError, TypeError, ValueError):
            return False

    def boolean_equals(name: str, expected: bool) -> bool:
        try:
            return _boolean_scalar(nested.get(name), name=name) is expected
        except (ConstrainedFlowCanaryError, TypeError, ValueError):
            return False

    return {
        "experiment_arm_exact": nested.get("experiment_arm") == EXPERIMENT_ARM,
        "solve_calls_one": integer_equals("adapter_solve_calls", 1),
        "historical_projection_calls_129": integer_equals(
            "adapter_historical_projection_calls", 129
        ),
        "injection_count_one": integer_equals("adapter_injection_count", 1),
        "projection_check_one": integer_equals(
            "adapter_projection_idempotence_checks", 1
        ),
        "projection_idempotent": boolean_equals(
            "adapter_projection_idempotence_exact", True
        ),
        "only_historical_initialization_changed": boolean_equals(
            "adapter_only_historical_initialization_changed", True
        ),
        "target_pairing_exact": boolean_equals("target_pairing_exact", True),
        "schedule_valid": boolean_equals("schedule_valid", True),
        "jacobian_valid": boolean_equals("jacobian_valid", True),
        "no_terminal_overwrite": boolean_equals("terminal_overwrite_used", False),
        "no_optimality_certificate": boolean_equals(
            "optimality_certificate", False
        ),
        "no_infeasibility_certificate": boolean_equals(
            "infeasibility_certificate", False
        ),
        "no_nonlinear_feasibility_certificate": boolean_equals(
            "nonlinear_feasibility_certificate", False
        ),
    }


def _project_compact_float64(value: np.ndarray, radius: float) -> np.ndarray:
    compact = np.asarray(value, dtype=np.float64).reshape(5, 15)
    norms = np.linalg.norm(compact, axis=1)
    scales = np.minimum(
        1.0,
        radius / np.maximum(norms, np.finfo(np.float64).tiny),
    )
    return np.ascontiguousarray(compact * scales[:, None])


def _independent_linear_checks(
    nested: Mapping[str, Any],
    *,
    jacobian: np.ndarray,
    selected64: np.ndarray,
    pre_projection: np.ndarray,
    post_projection: np.ndarray,
    raw_singular_values: np.ndarray,
    weighted_singular_values: np.ndarray,
    source_baseline_target_error: np.ndarray,
    executed_increments: np.ndarray,
    budget: np.float32,
) -> tuple[Mapping[str, bool], Mapping[str, Any]]:
    """Recompute the registered convex algebra with independent NumPy code."""

    baseline_error = _finite_array(
        nested.get("baseline_target_error"),
        name="Arm B baseline target error",
        shape=(35,),
        dtype=np.float32,
    )
    weights = _finite_array(
        nested.get("weights"),
        name="Arm B physical objective weights",
        shape=(35,),
        dtype=np.float64,
    )
    linear_error = _finite_array(
        nested.get("linear_predicted_target_error"),
        name="Arm B recorded linear target error",
        shape=(35,),
        dtype=np.float32,
    )
    fista = nested.get("fista")
    if not isinstance(fista, Mapping):
        raise ConstrainedFlowCanaryError("Arm B has no FISTA diagnostic")

    xyz_rows = np.zeros(35, dtype=np.bool_)
    for row in range(5):
        xyz_rows[row * 7 : row * 7 + 3] = True
    expected_weights = np.where(xyz_rows, 1.0 / 0.005, 1.0 / 0.015).astype(
        np.float64
    )
    source_baseline_target_error = _finite_array(
        source_baseline_target_error,
        name="source-derived Arm B baseline target error",
        shape=(35,),
        dtype=np.float32,
    )
    executed_increments = _finite_array(
        executed_increments,
        name="Arm B exact executed increments",
        shape=(10, 10, 32),
        dtype=np.float32,
    )
    matrix = np.asarray(jacobian, dtype=np.float64) * expected_weights[:, None]
    offset = (
        np.asarray(source_baseline_target_error, dtype=np.float64)
        * expected_weights
    )
    independent_raw_singular = np.linalg.svd(
        np.asarray(jacobian, dtype=np.float64), compute_uv=False
    )
    independent_weighted_singular = np.linalg.svd(matrix, compute_uv=False)
    sigma_max = float(independent_weighted_singular[0])
    exact_zero = bool(np.count_nonzero(matrix) == 0)
    if not math.isfinite(sigma_max) or sigma_max < 0.0 or (
        sigma_max == 0.0 and not exact_zero
    ):
        raise ConstrainedFlowCanaryError(
            "independent weighted Jacobian spectral norm is invalid"
        )
    step = 0.0 if exact_zero else 1.0 / (sigma_max * sigma_max)
    radius = float(np.asarray(budget, dtype=np.float32)) / 5.0

    zero = np.zeros((5, 15), dtype=np.float64)

    def mse(candidate: np.ndarray) -> float:
        residual = matrix @ candidate.reshape(-1) + offset
        return float(np.mean(np.square(residual), dtype=np.float64))

    best = zero.copy()
    best_iteration = 0
    best_mse = mse(best)
    if not exact_zero:
        x_value = zero.copy()
        y_value = zero.copy()
        momentum = 1.0
        for update in range(1, 4097):
            residual = matrix @ y_value.reshape(-1) + offset
            gradient = (matrix.T @ residual).reshape(5, 15)
            next_x = _project_compact_float64(y_value - step * gradient, radius)
            next_mse = mse(next_x)
            if not (
                np.isfinite(residual).all()
                and np.isfinite(gradient).all()
                and np.isfinite(next_x).all()
                and math.isfinite(next_mse)
            ):
                raise ConstrainedFlowCanaryError(
                    "independent FISTA recomputation became nonfinite"
                )
            if next_mse < best_mse:
                best = next_x.copy()
                best_mse = next_mse
                best_iteration = update
            next_momentum = (
                1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)
            ) / 2.0
            y_value = next_x + ((momentum - 1.0) / next_momentum) * (
                next_x - x_value
            )
            x_value = next_x
            momentum = next_momentum

    selected_residual = matrix @ selected64.reshape(-1) + offset
    selected_mse = float(np.mean(np.square(selected_residual), dtype=np.float64))
    selected_half = float(0.5 * np.sum(np.square(selected_residual), dtype=np.float64))
    if exact_zero:
        selected_mapping_norm = 0.0
    else:
        selected_gradient = (matrix.T @ selected_residual).reshape(5, 15)
        selected_mapped = _project_compact_float64(
            selected64 - step * selected_gradient, radius
        )
        selected_mapping_norm = float(
            np.linalg.norm(((selected64 - selected_mapped) / step).reshape(-1))
        )
    executed_compact64 = np.asarray(
        executed_increments[5:, :5, :3], dtype=np.float64
    ).reshape(-1)
    executed_linear64 = np.asarray(
        source_baseline_target_error, dtype=np.float64
    ) + np.asarray(jacobian, dtype=np.float64) @ executed_compact64
    executed_weighted_mse = float(
        np.mean(np.square(executed_linear64 * expected_weights), dtype=np.float64)
    )

    reported_selected_iteration = _integer_scalar(
        fista.get("selected_iteration"), name="FISTA selected iteration"
    )
    reported_step = float(
        _zero_dimensional_typed_scalar(
            fista.get("step_size"), name="FISTA step size", dtype=np.float64
        )
    )
    reported_half = float(
        _zero_dimensional_typed_scalar(
            fista.get("objective_half_squared_l2"),
            name="FISTA half-squared objective",
            dtype=np.float64,
        )
    )
    reported_mse = float(
        _zero_dimensional_typed_scalar(
            fista.get("historical_weighted_mse"),
            name="FISTA weighted MSE",
            dtype=np.float64,
        )
    )
    reported_executed_mse = float(
        _zero_dimensional_typed_scalar(
            fista.get("executed_historical_weighted_mse"),
            name="FISTA executed weighted MSE",
            dtype=np.float64,
        )
    )
    reported_mapping = float(
        _zero_dimensional_typed_scalar(
            fista.get("projected_gradient_mapping_norm"),
            name="FISTA projected-gradient mapping",
            dtype=np.float64,
        )
    )

    candidate_max_abs = float(np.max(np.abs(best - selected64)))
    candidate_relative_l2 = float(
        np.linalg.norm((best - selected64).reshape(-1))
        / max(
            np.linalg.norm(best.reshape(-1)),
            np.linalg.norm(selected64.reshape(-1)),
            1.0e-12,
        )
    )
    production_numpy = _project_compact_float64(
        np.asarray(pre_projection[:, :5, :3], dtype=np.float64), radius
    ).astype(np.float32)
    production_post = np.asarray(
        post_projection[:, :5, :3], dtype=np.float32
    ).reshape(5, 15)
    production_max_abs = float(np.max(np.abs(production_numpy - production_post)))

    close64 = dict(rtol=1.0e-10, atol=1.0e-12)
    checks = {
        "weights_exact": _array_exact(weights, expected_weights),
        "baseline_target_error_source_exact": _array_exact(
            baseline_error, source_baseline_target_error
        ),
        "raw_singular_values_independently_recomputed": bool(
            np.allclose(
                independent_raw_singular,
                raw_singular_values,
                **close64,
            )
        ),
        "weighted_singular_values_independently_recomputed": bool(
            np.allclose(
                independent_weighted_singular,
                weighted_singular_values,
                **close64,
            )
        ),
        "step_size_independently_recomputed": bool(
            math.isclose(reported_step, step, rel_tol=1.0e-10, abs_tol=1.0e-12)
        ),
        "selected_candidate_independently_recomputed": bool(
            candidate_max_abs <= 1.0e-8 or candidate_relative_l2 <= 1.0e-8
        ),
        "selected_half_objective_independently_recomputed": bool(
            math.isclose(reported_half, selected_half, rel_tol=1.0e-10, abs_tol=1.0e-10)
        ),
        "selected_mse_independently_recomputed": bool(
            math.isclose(reported_mse, selected_mse, rel_tol=1.0e-10, abs_tol=1.0e-10)
        ),
        "executed_mse_independently_recomputed": bool(
            math.isclose(
                reported_executed_mse,
                executed_weighted_mse,
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            )
        ),
        "projected_gradient_mapping_independently_recomputed": bool(
            math.isclose(
                reported_mapping,
                selected_mapping_norm,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
        ),
        "linear_prediction_independently_recomputed": bool(
            np.allclose(
                np.asarray(linear_error, dtype=np.float64),
                executed_linear64,
                rtol=1.0e-6,
                atol=1.0e-6,
            )
        ),
        "production_projection_independently_recomputed": production_max_abs
        <= 2.0e-7,
    }
    audit = {
        "linear_prediction_increment_owner": "exact_float32_dt_times_schedule",
        "independent_selected_iteration": best_iteration,
        "reported_selected_iteration": reported_selected_iteration,
        "independent_selected_iteration_exact_diagnostic_only": (
            best_iteration == reported_selected_iteration
        ),
        "independent_selected_weighted_mse": best_mse,
        "candidate_max_abs_difference": candidate_max_abs,
        "candidate_relative_l2_difference": candidate_relative_l2,
        "production_projection_max_abs_difference": production_max_abs,
        "numeric_tolerances_frozen_before_real_run": {
            "svd_relative": 1.0e-10,
            "svd_absolute": 1.0e-12,
            "step_size_relative": 1.0e-10,
            "step_size_absolute": 1.0e-12,
            "objective_relative": 1.0e-10,
            "objective_absolute": 1.0e-10,
            "projected_gradient_mapping_relative": 1.0e-9,
            "projected_gradient_mapping_absolute": 1.0e-9,
            "candidate_max_abs": 1.0e-8,
            "candidate_relative_l2": 1.0e-8,
            "linear_prediction_relative": 1.0e-6,
            "linear_prediction_absolute": 1.0e-6,
            "production_projection_max_abs": 2.0e-7,
        },
    }
    return checks, audit


def _arm_b_summary(
    nested: Mapping[str, Any],
    *,
    budget: np.float32,
    source_target: np.ndarray,
    source_baseline: np.ndarray,
    source_scale: np.ndarray,
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray]:
    source_target = _finite_array(
        source_target,
        name="source Arm B target",
        shape=(10, 32),
        dtype=np.float32,
    )
    source_baseline = _finite_array(
        source_baseline,
        name="source Arm B baseline",
        shape=(10, 32),
        dtype=np.float32,
    )
    source_scale = _finite_array(
        source_scale,
        name="source Arm B physical scale",
        shape=(10, 32),
        dtype=np.float32,
    )
    target_mask = np.zeros((10, 32), dtype=np.bool_)
    target_mask[:5, :7] = True
    source_baseline_fidelity_error = np.asarray(
        (source_baseline - source_target) * source_scale,
        dtype=np.float32,
    )
    source_baseline_target_error = np.ascontiguousarray(
        source_baseline_fidelity_error[target_mask]
    )
    nested_target = _batched_array(
        nested.get("target"), name="Arm B paired target", shape=(10, 32)
    )
    nested_baseline = _batched_array(
        nested.get("baseline_final"),
        name="Arm B paired baseline",
        shape=(10, 32),
    )
    schedule = _diagnostic_schedule(nested.get("schedule"), name="Arm B schedule")
    active = np.asarray(nested.get("active_increments"))
    if active.shape != (5, 1, 10, 32):
        raise ConstrainedFlowCanaryError(
            f"Arm B active increments have shape {active.shape}, expected (5, 1, 10, 32)"
        )
    if not bool(np.isfinite(active).all()):
        raise ConstrainedFlowCanaryError("Arm B active increments are nonfinite")
    if active.dtype != np.dtype(np.float32):
        raise ConstrainedFlowCanaryError("Arm B active increments must preserve float32")
    active = np.ascontiguousarray(active[:, 0])
    injected_raw = np.asarray(nested.get("adapter_injected_initial_increments"))
    if (
        injected_raw.shape != (5, 1, 10, 32)
        or injected_raw.dtype != np.dtype(np.float32)
        or not bool(np.isfinite(injected_raw).all())
    ):
        raise ConstrainedFlowCanaryError(
            "Arm C injected initialization must preserve finite float32 (5,1,10,32)"
        )
    injected = np.ascontiguousarray(injected_raw[:, 0])
    selected64 = _finite_array(
        nested.get("selected_candidate_float64"),
        name="Arm B selected float64 candidate",
        shape=(5, 15),
    )
    if selected64.dtype != np.dtype(np.float64):
        raise ConstrainedFlowCanaryError("Arm B selected candidate must preserve float64")
    pre_projection_raw = np.asarray(nested.get("model_candidate_pre_projection"))
    post_projection_raw = np.asarray(nested.get("model_candidate_post_projection"))
    expected_model_shape = (5, 1, 10, 32)
    if pre_projection_raw.shape != expected_model_shape or post_projection_raw.shape != expected_model_shape:
        raise ConstrainedFlowCanaryError(
            "Arm B model candidates must preserve shape (5, 1, 10, 32)"
        )
    if not bool(np.isfinite(pre_projection_raw).all()) or not bool(
        np.isfinite(post_projection_raw).all()
    ):
        raise ConstrainedFlowCanaryError("Arm B model candidates must be finite")
    if (
        pre_projection_raw.dtype != np.dtype(np.float32)
        or post_projection_raw.dtype != np.dtype(np.float32)
    ):
        raise ConstrainedFlowCanaryError(
            "Arm B model candidates must preserve float32"
        )
    pre_projection = np.ascontiguousarray(pre_projection_raw[:, 0])
    post_projection = np.ascontiguousarray(post_projection_raw[:, 0])
    mask = np.zeros((10, 32), dtype=np.bool_)
    mask[:5, :3] = True
    expected_pre_projection = np.zeros((5, 10, 32), dtype=pre_projection.dtype)
    expected_pre_projection[:, mask] = selected64.reshape(5, 15).astype(
        pre_projection.dtype, copy=False
    )
    expected_schedule = np.ascontiguousarray(
        np.asarray(active / DT_FLOAT32, dtype=np.float32)
    )
    candidate_schedule_exact = _array_exact(
        np.asarray(schedule[5:], dtype=np.float32), expected_schedule
    )
    executed_increments = np.ascontiguousarray(
        np.asarray(schedule * DT_FLOAT32, dtype=np.float32)
    )
    recorded_increments_raw = np.asarray(nested.get("increments"))
    if recorded_increments_raw.shape != (10, 1, 10, 32):
        raise ConstrainedFlowCanaryError(
            "Arm B recorded executed increments must preserve shape (10, 1, 10, 32)"
        )
    if recorded_increments_raw.dtype != np.dtype(np.float32):
        raise ConstrainedFlowCanaryError(
            "Arm B recorded executed increments must preserve float32"
        )
    recorded_increments = np.ascontiguousarray(recorded_increments_raw[:, 0])
    constraints, constraint_errors = _candidate_constraints(schedule, budget=budget)
    linear_error = _batched_array(
        nested.get("linear_fidelity_error"),
        name="Arm B linear fidelity error",
        shape=(10, 32),
    )
    nonlinear_error = _batched_array(
        nested.get("nonlinear_fidelity_error"),
        name="Arm B nonlinear fidelity error",
        shape=(10, 32),
    )
    linear_target_error = _finite_array(
        nested.get("linear_predicted_target_error"),
        name="Arm B linear predicted target error",
        shape=(35,),
        dtype=np.float32,
    )
    nonlinear_target_error = _finite_array(
        nested.get("nonlinear_target_error"),
        name="Arm B nonlinear target error",
        shape=(35,),
        dtype=np.float32,
    )
    linearization_error = _finite_array(
        nested.get("linearization_error"),
        name="Arm B linearization error",
        shape=(35,),
        dtype=np.float32,
    )
    rollout_final = _batched_array(
        nested.get("rollout", {}).get("final"),
        name="Arm B nonlinear rollout final",
        shape=(10, 32),
    )
    expected_linear_fidelity_error = source_baseline_fidelity_error.copy()
    expected_linear_fidelity_error[target_mask] = linear_target_error
    expected_nonlinear_fidelity_error = np.asarray(
        (rollout_final - source_target) * source_scale,
        dtype=np.float32,
    )
    expected_nonlinear_target_error = np.ascontiguousarray(
        expected_nonlinear_fidelity_error[target_mask]
    )
    expected_linearization_error = np.asarray(
        expected_nonlinear_target_error - linear_target_error,
        dtype=np.float32,
    )
    linear_metrics = _fidelity_metrics(linear_error)
    nonlinear_metrics = _fidelity_metrics(nonlinear_error)
    jacobian = _finite_array(
        nested.get("jacobian"), name="Arm B Jacobian", shape=(35, 75)
    )
    if jacobian.dtype != np.dtype(np.float32):
        raise ConstrainedFlowCanaryError("Arm B Jacobian must preserve model float32")
    jacobian_singular_values = _finite_array(
        nested.get("jacobian_singular_values"),
        name="Arm B raw-Jacobian singular values",
        shape=(35,),
    )
    weighted_singular_values = _finite_array(
        nested.get("weighted_matrix_singular_values"),
        name="Arm B weighted-matrix singular values",
        shape=(35,),
    )
    if (
        jacobian_singular_values.dtype != np.dtype(np.float64)
        or weighted_singular_values.dtype != np.dtype(np.float64)
    ):
        raise ConstrainedFlowCanaryError(
            "Arm B singular-value diagnostics must preserve solver float64"
        )
    finite_difference = nested.get("finite_difference")
    if not isinstance(finite_difference, Mapping):
        raise ConstrainedFlowCanaryError("Arm B has no finite-difference diagnostic")
    fd_directions = _finite_array(
        finite_difference.get("directions"),
        name="finite-difference directions",
        shape=(3, 75),
        dtype=np.float32,
    )
    fd_epsilon = _finite_array(
        finite_difference.get("epsilon_values"),
        name="finite-difference epsilon values",
        shape=(2,),
        dtype=np.float32,
    )
    fd_plus = _finite_array(
        finite_difference.get("plus_target_physical"),
        name="finite-difference plus physical targets",
        shape=(3, 2, 35),
        dtype=np.float32,
    )
    fd_minus = _finite_array(
        finite_difference.get("minus_target_physical"),
        name="finite-difference minus physical targets",
        shape=(3, 2, 35),
        dtype=np.float32,
    )
    fd_autograd = _finite_array(
        finite_difference.get("autograd_directional_derivatives"),
        name="finite-difference autograd products",
        shape=(3, 35),
        dtype=np.float32,
    )
    fd_central = _finite_array(
        finite_difference.get("central_directional_derivatives"),
        name="finite-difference central products",
        shape=(3, 2, 35),
        dtype=np.float32,
    )
    fd_absolute = _finite_array(
        finite_difference.get("absolute_l2_errors"),
        name="finite-difference absolute errors",
        shape=(3, 2),
        dtype=np.float32,
    )
    fd_relative = _finite_array(
        finite_difference.get("relative_l2_errors"),
        name="finite-difference relative errors",
        shape=(3, 2),
        dtype=np.float32,
    )
    fd_checks = np.asarray(finite_difference.get("checks_passed"))
    fd_directions_passed = np.asarray(finite_difference.get("directions_passed"))
    if fd_checks.shape != (3, 2) or fd_checks.dtype != np.dtype(np.bool_):
        raise ConstrainedFlowCanaryError("finite-difference checks must be bool[3,2]")
    if fd_directions_passed.shape != (3,) or fd_directions_passed.dtype != np.dtype(np.bool_):
        raise ConstrainedFlowCanaryError("finite-difference direction status must be bool[3]")
    registered_indices = np.arange(75, dtype=np.int64)
    registered_directions64 = np.stack(
        (
            np.ones(75, dtype=np.float64),
            np.where(registered_indices % 2 == 0, 1.0, -1.0),
            np.where(((registered_indices * 17 + 3) % 31) < 15, 1.0, -1.0),
        )
    )
    registered_directions64 /= np.linalg.norm(
        registered_directions64, axis=1, keepdims=True
    )
    expected_directions = registered_directions64.astype(fd_directions.dtype)
    expected_epsilon = (
        np.asarray(budget, dtype=np.float32)
        / np.asarray(5.0, dtype=np.float32)
        * np.asarray((1.0 / 256.0, 1.0 / 512.0), dtype=np.float32)
    ).astype(fd_epsilon.dtype)
    # Independently bind all reported finite-difference summaries to J and to
    # the recorded central evaluations.  The central evaluations themselves
    # are allocation/model evidence; their internal arithmetic and decision
    # maps must nevertheless be self-consistent without trusting GPU booleans.
    fd_j64 = np.asarray(jacobian, dtype=np.float64)
    fd_direction64 = np.asarray(fd_directions, dtype=np.float64)
    fd_autograd64 = np.asarray(fd_autograd, dtype=np.float64)
    fd_central64 = np.asarray(fd_central, dtype=np.float64)
    expected_central64 = (
        np.asarray(fd_plus, dtype=np.float64)
        - np.asarray(fd_minus, dtype=np.float64)
    ) / (2.0 * np.asarray(fd_epsilon, dtype=np.float64)[None, :, None])
    expected_autograd64 = fd_direction64 @ fd_j64.T
    float32_reduction_slack = 128.0 * np.finfo(np.float32).eps
    autograd_bound = float32_reduction_slack * np.maximum(
        1.0, np.abs(fd_direction64) @ np.abs(fd_j64).T
    )
    fd_autograd_recomputed = bool(
        np.all(np.abs(fd_autograd64 - expected_autograd64) <= autograd_bound)
    )
    central_bound = float32_reduction_slack * np.maximum(
        1.0, np.abs(expected_central64)
    )
    fd_central_recomputed = bool(
        np.all(np.abs(fd_central64 - expected_central64) <= central_bound)
    )
    expected_absolute64 = np.linalg.norm(
        fd_central64 - fd_autograd64[:, None, :], axis=2
    )
    expected_denominator64 = np.maximum(
        np.maximum(
            np.linalg.norm(fd_central64, axis=2),
            np.linalg.norm(fd_autograd64, axis=1)[:, None],
        ),
        1.0e-6,
    )
    expected_relative64 = expected_absolute64 / expected_denominator64
    summary_bound_absolute = float32_reduction_slack * np.maximum(
        1.0, expected_absolute64
    )
    summary_bound_relative = float32_reduction_slack * np.maximum(
        1.0, expected_relative64
    )
    fd_absolute_recomputed = bool(
        np.all(
            np.abs(np.asarray(fd_absolute, dtype=np.float64) - expected_absolute64)
            <= summary_bound_absolute
        )
    )
    fd_relative_recomputed = bool(
        np.all(
            np.abs(np.asarray(fd_relative, dtype=np.float64) - expected_relative64)
            <= summary_bound_relative
        )
    )
    expected_fd_checks = (expected_relative64 <= 0.10) | (
        expected_absolute64 <= 1.0e-3
    )
    fista = nested.get("fista")
    if not isinstance(fista, Mapping):
        raise ConstrainedFlowCanaryError("Arm B has no FISTA diagnostic")
    status = _integer_scalar(nested.get("status"), name="Arm B status")
    updates = _integer_scalar(fista.get("updates"), name="FISTA updates")
    selected_iteration = _integer_scalar(
        fista.get("selected_iteration"), name="FISTA selected iteration"
    )
    fista_effective_rank = _integer_scalar(
        fista.get("effective_rank"), name="effective rank"
    )
    fista_early_stopping = _boolean_scalar(
        fista.get("early_stopping_used"), name="early stopping"
    )
    fista_adaptive_restart = _boolean_scalar(
        fista.get("adaptive_restart_used"), name="adaptive restart"
    )
    projected_mapping = float(
        _zero_dimensional_typed_scalar(
            fista.get("projected_gradient_mapping_norm"),
            name="projected gradient mapping norm",
            dtype=np.float64,
        )
    )
    fista_spectral_norm = float(
        _zero_dimensional_typed_scalar(
            fista.get("spectral_norm"),
            name="weighted spectral norm",
            dtype=np.float64,
        )
    )
    fista_step_size = float(
        _zero_dimensional_typed_scalar(
            fista.get("step_size"), name="FISTA step size", dtype=np.float64
        )
    )
    fista_optimality_certificate = _boolean_scalar(
        fista.get("optimality_certificate"), name="FISTA optimality certificate"
    )
    fista_infeasibility_certificate = _boolean_scalar(
        fista.get("infeasibility_certificate"),
        name="FISTA infeasibility certificate",
    )
    fista_checks = {
        "status_registered": status in {0, 1},
        "updates_registered": (status == 0 and updates == 4096)
        or (status == 1 and updates == 0),
        "selected_iteration_in_range": 0 <= selected_iteration <= updates,
        "effective_rank_in_range": 0 <= fista_effective_rank <= 35,
        "early_stopping_false": fista_early_stopping is False,
        "adaptive_restart_false": fista_adaptive_restart is False,
        "finite_numeric_diagnostic": math.isfinite(projected_mapping),
    }
    adapter_checks = _adapter_checks(nested)
    independent_checks, independent_audit = _independent_linear_checks(
        nested,
        jacobian=jacobian,
        selected64=selected64,
        pre_projection=pre_projection,
        post_projection=post_projection,
        raw_singular_values=jacobian_singular_values,
        weighted_singular_values=weighted_singular_values,
        source_baseline_target_error=source_baseline_target_error,
        executed_increments=executed_increments,
        budget=budget,
    )
    jacobian_sigma_max = float(jacobian_singular_values[0])
    raw_rank_threshold = (
        np.finfo(np.float64).eps * 75 * jacobian_sigma_max
    )
    recomputed_jacobian_rank = int(
        np.count_nonzero(jacobian_singular_values > raw_rank_threshold)
    )
    weighted_sigma_max = float(weighted_singular_values[0])
    weighted_rank_threshold = np.finfo(np.float64).eps * 75 * weighted_sigma_max
    recomputed_weighted_rank = int(
        np.count_nonzero(weighted_singular_values > weighted_rank_threshold)
    )
    jacobian_is_exactly_zero = bool(np.count_nonzero(jacobian) == 0)
    status_matches_jacobian = bool(
        (status == 1 and jacobian_is_exactly_zero)
        or (status == 0 and not jacobian_is_exactly_zero)
    )
    zero_branch_candidate_exact = bool(
        status != 1
        or (
            _array_exact(selected64, np.zeros_like(selected64))
            and _array_exact(pre_projection, np.zeros_like(pre_projection))
            and _array_exact(post_projection, np.zeros_like(post_projection))
            and _array_exact(active, np.zeros_like(active))
        )
    )
    zero_branch_solver_metadata_exact = bool(
        status != 1
        or (
            updates == 0
            and selected_iteration == 0
            and fista_spectral_norm == 0.0
            and fista_step_size == 0.0
            and recomputed_jacobian_rank == 0
            and recomputed_weighted_rank == 0
        )
    )
    checks = {
        **adapter_checks,
        **fista_checks,
        **independent_checks,
        "nested_target_source_exact": _array_exact(nested_target, source_target),
        "nested_baseline_source_exact": _array_exact(
            nested_baseline, source_baseline
        ),
        "linear_fidelity_error_source_reconstructed": _array_exact(
            linear_error, expected_linear_fidelity_error
        ),
        "nonlinear_fidelity_error_source_reconstructed": _array_exact(
            nonlinear_error, expected_nonlinear_fidelity_error
        ),
        "nonlinear_target_error_source_reconstructed": _array_exact(
            nonlinear_target_error, expected_nonlinear_target_error
        ),
        "linearization_error_source_reconstructed": _array_exact(
            linearization_error, expected_linearization_error
        ),
        "solver_status_matches_exact_jacobian_zero_state": status_matches_jacobian,
        "zero_jacobian_candidate_exact_zero": zero_branch_candidate_exact,
        "zero_jacobian_solver_metadata_exact": zero_branch_solver_metadata_exact,
        "fista_convex_dtype_exact": fista.get("convex_dtype") == "torch.float64",
        "fista_convex_device_exact": fista.get("convex_device") == "cpu",
        "fista_optimality_certificate_false": fista_optimality_certificate is False,
        "fista_infeasibility_certificate_false": fista_infeasibility_certificate
        is False,
        "post_projection_candidate_exact_schedule": candidate_schedule_exact,
        "recorded_executed_increments_exact_schedule": _array_exact(
            recorded_increments, executed_increments
        ),
        "selected_float64_single_cast_exact": _array_exact(
            pre_projection, expected_pre_projection
        ),
        "post_projection_exact_active": _array_exact(post_projection, active),
        "arm_c_injected_initialization_exact_arm_b": _array_exact(
            injected, active
        ),
        "constraints_passed": not constraint_errors,
        "linear_reported_metrics_exact": _reported_metrics_match(
            nested.get("linear_metrics"), linear_metrics
        ),
        "nonlinear_reported_metrics_exact": _reported_metrics_match(
            nested.get("nonlinear_metrics"), nonlinear_metrics
        ),
        "jacobian_finite": bool(np.isfinite(jacobian).all()),
        "raw_singular_values_nonnegative_descending": bool(
            np.all(jacobian_singular_values >= 0)
            and np.all(jacobian_singular_values[:-1] >= jacobian_singular_values[1:])
        ),
        "weighted_singular_values_nonnegative_descending": bool(
            np.all(weighted_singular_values >= 0)
            and np.all(weighted_singular_values[:-1] >= weighted_singular_values[1:])
        ),
        "raw_jacobian_rank_exact": _integer_scalar(
            nested.get("jacobian_effective_rank"), name="Jacobian effective rank"
        )
        == recomputed_jacobian_rank,
        "weighted_matrix_rank_exact": fista_effective_rank
        == recomputed_weighted_rank,
        "weighted_spectral_norm_exact": fista_spectral_norm == weighted_sigma_max,
        "finite_difference_passed": _boolean_scalar(
            finite_difference.get("passed"), name="finite difference passed"
        ),
        "finite_difference_all_directions_passed": bool(fd_directions_passed.all()),
        "finite_difference_status_recomputed": bool(
            np.array_equal(fd_directions_passed, np.any(fd_checks, axis=1))
        ),
        "finite_difference_autograd_products_recomputed": fd_autograd_recomputed,
        "finite_difference_central_products_recomputed": fd_central_recomputed,
        "finite_difference_absolute_errors_recomputed": fd_absolute_recomputed,
        "finite_difference_relative_errors_recomputed": fd_relative_recomputed,
        "finite_difference_threshold_map_recomputed": bool(
            np.array_equal(fd_checks, expected_fd_checks)
        ),
        "finite_difference_registered_directions_exact": _array_exact(
            fd_directions, expected_directions
        ),
        "finite_difference_registered_epsilon_exact": _array_exact(
            fd_epsilon, expected_epsilon
        ),
        "finite_difference_relative_tolerance_exact": float(
            _zero_dimensional_typed_scalar(
                finite_difference.get("relative_l2_tolerance"),
                name="finite difference relative tolerance",
                dtype=np.float64,
            )
        )
        == 0.10,
        "finite_difference_absolute_tolerance_exact": float(
            _zero_dimensional_typed_scalar(
                finite_difference.get("absolute_l2_tolerance"),
                name="finite difference absolute tolerance",
                dtype=np.float64,
            )
        )
        == 1.0e-3,
    }
    record = _tree_record(nested)
    return (
        {
            "status_code": status,
            "schedule": _array_record(schedule, dtype=np.float32),
            "active_increments": _array_record(active, dtype=np.float32),
            "arm_c_injected_initialization": _array_record(
                injected, dtype=np.float32
            ),
            "executed_increments": _array_record(
                executed_increments, dtype=np.float32
            ),
            "selected_candidate_float64": _array_record(selected64, dtype=np.float64),
            "model_candidate_pre_projection": _array_record(pre_projection),
            "model_candidate_post_projection": _array_record(post_projection),
            "jacobian": _array_record(jacobian),
            "jacobian_singular_values": _array_record(jacobian_singular_values),
            "weighted_matrix_singular_values": _array_record(weighted_singular_values),
            "jacobian_effective_rank": recomputed_jacobian_rank,
            "weighted_matrix_effective_rank": recomputed_weighted_rank,
            "finite_difference": _tree_record(finite_difference),
            "independent_linear_audit": independent_audit,
            "linear_metrics": linear_metrics,
            "nonlinear_metrics": nonlinear_metrics,
            "constraints": constraints,
            "constraint_errors": constraint_errors,
            "checks": checks,
            "passed": all(checks.values()),
            "diagnostic_sha256": content_hash(record),
            "diagnostic": record,
            "finite_failure_is_infeasibility": False,
        },
        schedule,
        nonlinear_error,
    )


def _arm_c_summary(
    trace: Mapping[str, Any], *, budget: np.float32, returned_actions: Any
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray]:
    schedule = _finite_array(
        trace.get("solver_schedule"),
        name="Arm C solver schedule",
        shape=(10, 10, 32),
        dtype=np.float32,
    )
    fidelity_error = _finite_array(
        trace.get("solver_fidelity_error"),
        name="Arm C solver fidelity error",
        shape=(10, 32),
        dtype=np.float32,
    )
    metrics = _fidelity_metrics(fidelity_error)
    constraints, constraint_errors = _candidate_constraints(schedule, budget=budget)
    reply_actions = _finite_array(
        returned_actions,
        name="Arm C comparison reply actions",
        shape=(10, 7),
    )
    audited_actions = _finite_array(
        trace.get("final_normalized_physical"),
        name="Arm C audited physical actions",
        shape=(10, 7),
    )
    status = int(
        _typed_scalar(
            trace.get("solver_status"),
            name="Arm C solver status",
            dtype=np.int64,
        )
    )
    reported = {
        "xyz_max_abs": trace.get("fidelity_xyz_max_abs"),
        "xyz_rms": trace.get("fidelity_xyz_rms"),
        "full_max_abs": trace.get("fidelity_full_max_abs"),
        "full_rms": trace.get("fidelity_full_rms"),
        "feasible": trace.get("solver_converged"),
    }
    solver_config = {
        "max_iterations": int(
            _typed_scalar(
                trace.get("solver_config_max_iterations"),
                name="Arm C max iterations",
                dtype=np.int64,
            )
        ),
        "learning_rate": float(
            _typed_scalar(
                trace.get("solver_config_learning_rate"),
                name="Arm C learning rate",
                dtype=np.float32,
            )
        ),
        "adam_beta1": float(
            _typed_scalar(
                trace.get("solver_config_adam_beta1"),
                name="Arm C Adam beta1",
                dtype=np.float32,
            )
        ),
        "adam_beta2": float(
            _typed_scalar(
                trace.get("solver_config_adam_beta2"),
                name="Arm C Adam beta2",
                dtype=np.float32,
            )
        ),
        "adam_epsilon": float(
            _typed_scalar(
                trace.get("solver_config_adam_epsilon"),
                name="Arm C Adam epsilon",
                dtype=np.float32,
            )
        ),
        "xyz_max_abs_tolerance": float(
            _typed_scalar(
                trace.get("solver_config_xyz_max_abs_tolerance"),
                name="Arm C XYZ max tolerance",
                dtype=np.float32,
            )
        ),
        "xyz_rms_tolerance": float(
            _typed_scalar(
                trace.get("solver_config_xyz_rms_tolerance"),
                name="Arm C XYZ RMS tolerance",
                dtype=np.float32,
            )
        ),
        "full_max_abs_tolerance": float(
            _typed_scalar(
                trace.get("solver_config_full_max_abs_tolerance"),
                name="Arm C full max tolerance",
                dtype=np.float32,
            )
        ),
        "full_rms_tolerance": float(
            _typed_scalar(
                trace.get("solver_config_full_rms_tolerance"),
                name="Arm C full RMS tolerance",
                dtype=np.float32,
            )
        ),
        "constraint_slack_ulps": int(
            _typed_scalar(
                trace.get("solver_config_constraint_slack_ulps"),
                name="Arm C constraint slack ULPs",
                dtype=np.int64,
            )
        ),
        "stop_on_first_feasible": bool(
            _typed_scalar(
                trace.get("solver_config_stop_on_first_feasible"),
                name="Arm C early stop flag",
                dtype=np.bool_,
            )
        ),
    }
    expected_solver_config = {
        "max_iterations": 128,
        "learning_rate": float(np.float32(0.02)),
        "adam_beta1": float(np.float32(0.9)),
        "adam_beta2": float(np.float32(0.999)),
        "adam_epsilon": float(np.float32(1.0e-8)),
        "xyz_max_abs_tolerance": float(np.float32(0.010)),
        "xyz_rms_tolerance": float(np.float32(0.005)),
        "full_max_abs_tolerance": float(np.float32(0.050)),
        "full_rms_tolerance": float(np.float32(0.015)),
        "constraint_slack_ulps": 8,
        "stop_on_first_feasible": False,
    }
    checks = {
        "finite_registered_status": status in {0, 3},
        "not_nonfinite": bool(
            _typed_scalar(
                trace.get("solver_nonfinite"),
                name="solver nonfinite",
                dtype=np.bool_,
            )
        )
        is False,
        "fields_available": bool(
            _typed_scalar(
                trace.get("solver_fields_available"),
                name="solver fields available",
                dtype=np.bool_,
            )
        ),
        "schedule_valid": bool(
            _typed_scalar(
                trace.get("solver_schedule_valid"),
                name="solver schedule valid",
                dtype=np.bool_,
            )
        ),
        "target_pairing_exact": bool(
            _typed_scalar(
                trace.get("target_pairing_exact"),
                name="target pairing exact",
                dtype=np.bool_,
            )
        ),
        "fixed_128_updates": int(
            _typed_scalar(
                trace.get("solver_iterations"),
                name="solver iterations",
                dtype=np.int64,
            )
        )
        == 128,
        "constraints_passed": not constraint_errors,
        "reported_metrics_exact": _reported_metrics_match(reported, metrics),
        "historical_solver_config_exact": solver_config
        == expected_solver_config,
        "parameter_requires_grad_restored": bool(
            _typed_scalar(
                trace.get("parameter_requires_grad_restored"),
                name="parameter requires-grad restored",
                dtype=np.bool_,
            )
        ),
        "parameter_grads_none_before": bool(
            _typed_scalar(
                trace.get("parameter_grads_none_before"),
                name="grads none before",
                dtype=np.bool_,
            )
        ),
        "parameter_grads_none_after": bool(
            _typed_scalar(
                trace.get("parameter_grads_none_after"),
                name="grads none after",
                dtype=np.bool_,
            )
        ),
        "parameter_grad_check_performed": bool(
            _typed_scalar(
                trace.get("parameter_grad_check_performed"),
                name="parameter grad check performed",
                dtype=np.bool_,
            )
        ),
        "returned_actions_exact_audited_physical": _array_exact(
            reply_actions, audited_actions
        ),
    }
    selected_final = _finite_array(
        trace.get("solver_internal_replay_final"),
        name="Arm C selected final",
        shape=(10, 32),
        dtype=np.float32,
    )
    return (
        {
            "status_code": status,
            "converged": bool(
                _typed_scalar(
                    trace.get("solver_converged"),
                    name="solver converged",
                    dtype=np.bool_,
                )
            ),
            "schedule": _array_record(schedule, dtype=np.float32),
            "metrics": metrics,
            "constraints": constraints,
            "constraint_errors": constraint_errors,
            "selected_final_normalized": _array_record(selected_final, dtype=np.float32),
            "historical_solver_config": solver_config,
            "comparison_trace": _tree_record(trace),
            "returned_actions": _array_record(reply_actions),
            "checks": checks,
            "passed": all(checks.values()),
            "finite_failure_is_infeasibility": False,
        },
        schedule,
        selected_final,
    )


def _legacy_arm_a_summary(
    legacy_payload: Mapping[str, Any], config: Mapping[str, Any], legacy_status: str
) -> Mapping[str, Any]:
    if legacy_payload.get("payload_type") != LEGACY_PAYLOAD_TYPE:
        raise ConstrainedFlowCanaryError("legacy canary payload identity changed")
    result = legacy_payload.get("result_without_memory")
    if not isinstance(result, Mapping):
        raise ConstrainedFlowCanaryError("legacy canary payload has no result")
    binding = config.get("historical_run_b_binding")
    if not isinstance(binding, Mapping):
        raise ConstrainedFlowCanaryError("new config has no historical Run-B binding")
    first = result.get("solver", {}).get("first")
    duplicate = result.get("solver", {}).get("duplicate")
    if not isinstance(first, Mapping) or not isinstance(duplicate, Mapping):
        raise ConstrainedFlowCanaryError("legacy payload has no duplicate Arm-A summaries")
    schedule = first.get("schedule")
    actions = first.get("actions")
    fidelity = first.get("fidelity")
    if not isinstance(schedule, Mapping) or not isinstance(actions, Mapping) or not isinstance(fidelity, Mapping):
        raise ConstrainedFlowCanaryError("legacy Arm-A diagnostic is incomplete")
    expected_metrics = {
        "xyz_max_abs": binding.get("expected_xyz_max_abs_physical"),
        "xyz_rms": binding.get("expected_xyz_rms_physical"),
        "full_max_abs": binding.get("expected_full_max_abs_physical"),
        "full_rms": binding.get("expected_full_rms_physical"),
    }
    checks = {
        "legacy_return_status_exact": legacy_status == binding.get("expected_status"),
        "payload_status_exact": result.get("status") == binding.get("expected_status"),
        "schedule_sha256_exact": schedule.get("sha256")
        == binding.get("expected_schedule_sha256"),
        "returned_actions_sha256_exact": actions.get("sha256")
        == binding.get("expected_returned_actions_sha256"),
        "metrics_exact": all(
            fidelity.get(key) == expected for key, expected in expected_metrics.items()
        ),
        "duplicate_schedule_exact": duplicate.get("schedule", {}).get("sha256")
        == schedule.get("sha256"),
        "duplicate_actions_exact": duplicate.get("actions", {}).get("sha256")
        == actions.get("sha256"),
        "legacy_determinism_passed": result.get("determinism", {}).get("passed") is True,
        "zero_generated_simulator_steps": all(
            result.get("simulator_use", {}).get(key) == expected
            for key, expected in (
                ("policy_generated_action_steps_executed", 0),
                ("teacher_generated_action_steps_executed", 0),
                ("efficacy_rollouts_executed", 0),
                ("simulator_efficacy_evaluated", False),
            )
        ),
    }
    expected_status = binding.get("expected_status")
    if expected_status == "completed_nonconverged":
        checks["legacy_clean_nonconvergence"] = (
            result.get("apparatus", {}).get("clean_nonconvergence") is True
        )
    else:
        checks["legacy_apparatus_passed"] = (
            result.get("apparatus", {}).get("passed_before_memory_finalization") is True
        )
    return {
        "status": result.get("status"),
        "schedule": schedule,
        "returned_actions": actions,
        "metrics": {key: fidelity.get(key) for key in FIDELITY_LIMITS},
        "checks": checks,
        "passed": all(checks.values()),
    }


def _duplicate_checks(
    first_trace: Mapping[str, Any],
    first_nested: Mapping[str, Any],
    duplicate_trace: Mapping[str, Any],
    duplicate_nested: Mapping[str, Any],
) -> Mapping[str, bool]:
    def exact_outer(name: str) -> bool:
        if name not in first_trace or name not in duplicate_trace:
            return False
        return _array_exact(first_trace[name], duplicate_trace[name])

    first_record = _tree_record(first_nested)
    duplicate_record = _tree_record(duplicate_nested)
    return {
        "arm_b_full_diagnostic_exact": content_hash(first_record)
        == content_hash(duplicate_record),
        "arm_b_jacobian_exact": _array_exact(
            first_nested.get("jacobian"), duplicate_nested.get("jacobian")
        ),
        "arm_b_candidate_exact": _array_exact(
            first_nested.get("schedule"), duplicate_nested.get("schedule")
        ),
        "arm_b_selected_iteration_exact": _integer_scalar(
            first_nested.get("fista", {}).get("selected_iteration"),
            name="first selected iteration",
        )
        == _integer_scalar(
            duplicate_nested.get("fista", {}).get("selected_iteration"),
            name="duplicate selected iteration",
        ),
        "arm_c_status_exact": exact_outer("solver_status"),
        "arm_c_schedule_exact": exact_outer("solver_schedule"),
        "arm_c_selected_final_exact": exact_outer("solver_internal_replay_final"),
        "arm_c_fidelity_exact": exact_outer("solver_fidelity_error"),
        "arm_c_returned_actions_exact": _array_exact(
            first_trace.get("final_normalized_physical"),
            duplicate_trace.get("final_normalized_physical"),
        ),
    }


def _comparison_pairing_checks(
    paired_calls: list[_PairedTeacherCall],
    *,
    target: np.ndarray,
    frozen_baseline: np.ndarray,
    noise: np.ndarray,
    scale: np.ndarray,
    budget: np.float32,
) -> Mapping[str, bool]:
    """Bind both new-arm requests and traces to the legacy paired source."""

    checks: dict[str, bool] = {}
    for replicate, paired_call in zip(
        ("first", "duplicate"), paired_calls, strict=True
    ):
        ordinary_request = copy.deepcopy(dict(paired_call.ordinary_request))
        comparison_request = copy.deepcopy(dict(paired_call.comparison_request))
        comparison_controls = comparison_request.get("__crfs__")
        ordinary_controls = ordinary_request.get("__crfs__")
        if not isinstance(comparison_controls, Mapping) or not isinstance(
            ordinary_controls, Mapping
        ):
            checks[f"{replicate}_request_controls_present"] = False
            continue
        comparison_without_arm = copy.deepcopy(comparison_request)
        sanitized_controls = dict(comparison_controls)
        observed_arm = sanitized_controls.pop("experiment_arm", None)
        comparison_without_arm["__crfs__"] = sanitized_controls
        checks[f"{replicate}_request_diff_only_experiment_arm"] = bool(
            observed_arm == EXPERIMENT_ARM
            and content_hash(_tree_record(comparison_without_arm))
            == content_hash(_tree_record(ordinary_request))
        )
        checks[f"{replicate}_request_target_exact"] = _array_exact(
            ordinary_controls.get("target"), target
        )
        checks[f"{replicate}_request_noise_exact"] = _array_exact(
            ordinary_controls.get("noise"), noise
        )
        try:
            request_budget = np.asarray(
                ordinary_controls.get("model_l2_path_budget")
            )
            checks[f"{replicate}_request_budget_exact"] = bool(
                request_budget.shape == ()
                and request_budget.dtype == np.dtype(np.float32)
                and request_budget.tobytes()
                == np.asarray(budget, dtype=np.float32).tobytes()
            )
        except (ConstrainedFlowCanaryError, TypeError, ValueError):
            checks[f"{replicate}_request_budget_exact"] = False

        outer, nested = _nested_diagnostic(paired_call.comparison_reply)
        try:
            outer_target = _finite_array(
                outer.get("inverse_target"),
                name=f"{replicate} comparison target",
                shape=(10, 32),
                dtype=np.float32,
            )
            outer_noise = _finite_array(
                outer.get("initial_noise"),
                name=f"{replicate} comparison noise",
                shape=(10, 32),
                dtype=np.float32,
            )
            outer_scale = _finite_array(
                outer.get("model_to_physical_scale"),
                name=f"{replicate} comparison scale",
                shape=(10, 32),
                dtype=np.float32,
            )
            outer_baseline = _finite_array(
                outer.get("solver_baseline_final"),
                name=f"{replicate} Arm-C baseline",
                shape=(10, 32),
                dtype=np.float32,
            )
            nested_target = _batched_array(
                nested.get("target"),
                name=f"{replicate} Arm-B target",
                shape=(10, 32),
            )
            nested_baseline = _batched_array(
                nested.get("baseline_final"),
                name=f"{replicate} Arm-B baseline",
                shape=(10, 32),
            )
            outer_budget = np.asarray(outer.get("source_control_budget"))
            nested_budget = np.asarray(nested.get("budget"))
        except (ConstrainedFlowCanaryError, TypeError, ValueError):
            checks[f"{replicate}_comparison_trace_fields_valid"] = False
            continue
        checks[f"{replicate}_comparison_trace_fields_valid"] = True
        checks[f"{replicate}_outer_target_exact"] = _array_exact(outer_target, target)
        checks[f"{replicate}_nested_target_exact"] = _array_exact(nested_target, target)
        checks[f"{replicate}_outer_noise_exact"] = _array_exact(outer_noise, noise)
        checks[f"{replicate}_outer_scale_exact"] = _array_exact(outer_scale, scale)
        checks[f"{replicate}_outer_baseline_exact"] = _array_exact(
            outer_baseline, frozen_baseline
        )
        checks[f"{replicate}_nested_baseline_exact"] = _array_exact(
            nested_baseline, frozen_baseline
        )
        checks[f"{replicate}_outer_budget_exact"] = bool(
            outer_budget.size == 1
            and outer_budget.dtype == np.dtype(np.float32)
            and outer_budget.reshape(()).tobytes()
            == np.asarray(budget, dtype=np.float32).tobytes()
        )
        checks[f"{replicate}_nested_budget_exact"] = bool(
            nested_budget.size == 1
            and nested_budget.dtype == np.dtype(np.float32)
            and nested_budget.reshape(()).tobytes()
            == np.asarray(budget, dtype=np.float32).tobytes()
        )
    return checks


def _comparison_request_records(
    paired_calls: list[_PairedTeacherCall],
) -> Mapping[str, Any]:
    records: dict[str, Any] = {}
    for replicate, paired_call in zip(
        ("first", "duplicate"), paired_calls, strict=True
    ):
        ordinary = copy.deepcopy(dict(paired_call.ordinary_request))
        comparison = copy.deepcopy(dict(paired_call.comparison_request))
        ordinary_controls = ordinary.pop("__crfs__", None)
        comparison_controls = comparison.pop("__crfs__", None)
        records[replicate] = {
            "ordinary_observation": _tree_record(ordinary),
            "comparison_observation": _tree_record(comparison),
            "ordinary_observation_sha256": content_hash(_tree_record(ordinary)),
            "comparison_observation_sha256": content_hash(
                _tree_record(comparison)
            ),
            "ordinary_controls": _tree_record(ordinary_controls),
            "comparison_controls": _tree_record(comparison_controls),
        }
    return records


def _canonical_replay_record(
    reply: Mapping[str, Any],
    *,
    schedule: np.ndarray,
    budget: np.float32,
    noise: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
    selected_final: np.ndarray,
    elapsed: float,
) -> Mapping[str, Any]:
    replay = dict(
        _replay_summary(
            reply,
            requested_schedule=np.asarray(schedule, dtype=np.float32),
            source_budget_float32=np.float32(budget),
            paired_noise=np.asarray(noise, dtype=np.float32),
            elapsed_seconds=elapsed,
        )
    )
    trace = reply.get("crfs_trace")
    if not isinstance(trace, Mapping):
        raise ConstrainedFlowCanaryError("canonical replay has no trace")
    final = _finite_array(
        trace.get("final_normalized"),
        name="canonical replay final",
        shape=(10, 32),
        dtype=np.float32,
    )
    returned_actions = _finite_array(
        reply.get("actions"), name="canonical replay returned actions", shape=(10, 7)
    )
    audited_physical = _finite_array(
        trace.get("final_normalized_physical"),
        name="canonical replay audited physical actions",
        shape=(10, 7),
    )
    fidelity_error = np.asarray(
        (final - np.asarray(target, dtype=np.float32))
        * np.asarray(scale, dtype=np.float32),
        dtype=np.float32,
    )
    metrics = _fidelity_metrics(fidelity_error)
    checks = dict(replay.get("checks", {}))
    checks.update(
        {
            "selected_final_exact": _array_exact(
                np.asarray(final, dtype=np.float32),
                np.asarray(selected_final, dtype=np.float32),
            ),
            "returned_actions_exact_audited_physical": _array_exact(
                returned_actions, audited_physical
            ),
            "zero_simulator_steps": True,
        }
    )
    replay.update(
        checks=checks,
        passed=all(checks.values()),
        fidelity_error=_array_record(fidelity_error, dtype=np.float32),
        metrics=metrics,
        returned_actions=_array_record(returned_actions),
    )
    return replay


def validate_constrained_flow_config(value: Mapping[str, Any]) -> list[str]:
    """Validate only the experiment choices consumed by this client layer."""

    errors: list[str] = []
    if constrained_flow_scientific_config_hash(value) != SCIENTIFIC_CONFIG_HASH:
        errors.append("constrained-flow scientific config hash changed")
    expected = {
        "schema_version": "1.0",
        "experiment_identity": "IFT-00B/CFS-00A",
        "new_experiment_not_retry": True,
        "historical_run_id_reuse_forbidden": True,
        "training": False,
        "allow_test_tuning": False,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            errors.append(f"{key} must equal {wanted!r}")
    arms = value.get("arms", {})
    if not isinstance(arms, Mapping) or set(arms) != {
        "A_historical_run_b",
        "B_linearized_candidate",
        "C_linearized_then_historical_adam",
    }:
        errors.append("config must retain exactly the three registered arms")
    execution = value.get("execution_boundary", {})
    if not isinstance(execution, Mapping) or any(
        execution.get(key) != wanted
        for key, wanted in (
            ("policy_generated_action_steps_executed", 0),
            ("teacher_generated_action_steps_executed", 0),
            ("efficacy_rollouts_executed", 0),
            ("simulator_efficacy_evaluated", False),
            ("mlp_training_authorized", False),
        )
    ):
        errors.append("execution boundary must forbid generated simulator steps and training")
    acceptance = value.get("acceptance_contract", {})
    if not isinstance(acceptance, Mapping) or acceptance.get(
        "finite_nonconvergence_is_infeasibility"
    ) is not False:
        errors.append("finite nonconvergence must not be called infeasibility")
    flow = value.get("flow_contract", {})
    if not isinstance(flow, Mapping) or flow.get("decoded_action_budget_B_action") != (
        "forbidden_not_part_of_run_b"
    ):
        errors.append("decoded action budget must remain forbidden")
    fidelity = value.get("fidelity_contract", {})
    for key, wanted in (
        ("units", "physical_action_after_checkpoint_scale"),
        ("output_dimension", 35),
        ("xyz_max_abs_tolerance", 0.01),
        ("xyz_rms_tolerance", 0.005),
        ("full_max_abs_tolerance", 0.05),
        ("full_rms_tolerance", 0.015),
    ):
        if not isinstance(fidelity, Mapping) or fidelity.get(key) != wanted:
            errors.append(f"fidelity_contract.{key} must equal {wanted!r}")
    numeric_audit = value.get("independent_numeric_audit", {})
    expected_numeric_audit = {
        "implementation": "numpy_reconstruction_of_W_A_b_svd_fista_objectives_prediction_and_projection",
        "singular_value_relative_tolerance": 1.0e-10,
        "singular_value_absolute_tolerance": 1.0e-12,
        "step_size_relative_tolerance": 1.0e-10,
        "step_size_absolute_tolerance": 1.0e-12,
        "objective_relative_tolerance": 1.0e-10,
        "objective_absolute_tolerance": 1.0e-10,
        "projected_gradient_mapping_relative_tolerance": 1.0e-9,
        "projected_gradient_mapping_absolute_tolerance": 1.0e-9,
        "candidate_max_abs_tolerance": 1.0e-8,
        "candidate_relative_l2_tolerance": 1.0e-8,
        "linear_prediction_relative_tolerance": 1.0e-6,
        "linear_prediction_absolute_tolerance": 1.0e-6,
        "production_projection_max_abs_tolerance": 2.0e-7,
        "independent_selected_iteration_is_acceptance_gate": False,
        "server_selected_iteration_exact_duplicate_required": True,
        "real_outcome_tuning_allowed": False,
    }
    if numeric_audit != expected_numeric_audit:
        errors.append("independent numeric audit contract changed")
    return errors


def _classify_outcome(
    *, apparatus_valid: bool, arm_b_nonlinear_pass: bool, arm_c_nonlinear_pass: bool
) -> str:
    """Apply the preregistered three-way result rule without scientific inflation."""

    if not apparatus_valid:
        return "apparatus_inconclusive"
    if arm_b_nonlinear_pass or arm_c_nonlinear_pass:
        return "mechanism_pass"
    return "frozen_method_negative"


def run_r05a_constrained_flow_canary(
    case: Mapping[str, Any],
    constrained_config: Mapping[str, Any],
    legacy_config: Any,
    *,
    repo_root: str | Path,
    input_manifest_sha256: str,
    constrained_config_path: str | Path,
    legacy_config_path: str | Path,
    client: Any,
    environment: Any = None,
) -> tuple[Path, str]:
    """Run paired A/B/C transport diagnostics and write a separate payload."""

    errors = validate_constrained_flow_config(constrained_config)
    if errors:
        raise ConstrainedFlowCanaryError("invalid constrained-flow config: " + "; ".join(errors))
    if dict(case).get("case_id") != CASE_ID:
        raise ConstrainedFlowCanaryError("constrained-flow canary case changed")
    paired = PairedConstrainedFlowClient(client)
    legacy_path, legacy_status = run_r05a_canary(
        case,
        legacy_config,
        repo_root=repo_root,
        input_manifest_sha256=input_manifest_sha256,
        client=paired,
        environment=environment,
    )
    if legacy_status not in EXPECTED_RESULT_STATUSES:
        raise ConstrainedFlowCanaryError(f"unexpected legacy status {legacy_status!r}")
    if len(paired.paired_calls) != EXPECTED_COMPARISON_CALLS:
        raise ConstrainedFlowCanaryError(
            f"expected two paired teacher calls, observed {len(paired.paired_calls)}"
        )
    legacy_payload = load_json(legacy_path)
    if not isinstance(legacy_payload, Mapping):
        raise ConstrainedFlowCanaryError("legacy output is not a mapping")
    arm_a = _legacy_arm_a_summary(legacy_payload, constrained_config, legacy_status)

    first_trace, first_nested = _nested_diagnostic(
        paired.paired_calls[0].comparison_reply
    )
    duplicate_trace, duplicate_nested = _nested_diagnostic(
        paired.paired_calls[1].comparison_reply
    )
    legacy_result = legacy_payload["result_without_memory"]
    budget = np.float32(legacy_result["target"]["source_budget_float32"])
    target = _array_from_record(legacy_result["target"]["target_normalized"])
    noise = _array_from_record(legacy_result["provenance"]["noise"])
    frozen_baseline = _array_from_record(
        legacy_result["target"]["fresh_frozen_normalized"]
    )
    scale = _array_from_record(
        legacy_result["solver"]["first"]["checkpoint_model_to_physical_scale"]
    )
    for name, value in (
        ("legacy target", target),
        ("legacy noise", noise),
        ("legacy frozen baseline", frozen_baseline),
        ("legacy checkpoint scale", scale),
    ):
        _finite_array(
            value,
            name=name,
            shape=(10, 32),
            dtype=np.float32,
        )

    arm_b, arm_b_schedule, _arm_b_error = _arm_b_summary(
        first_nested,
        budget=budget,
        source_target=target,
        source_baseline=frozen_baseline,
        source_scale=scale,
    )
    arm_b_duplicate, arm_b_duplicate_schedule, _arm_b_duplicate_error = (
        _arm_b_summary(
            duplicate_nested,
            budget=budget,
            source_target=target,
            source_baseline=frozen_baseline,
            source_scale=scale,
        )
    )
    arm_c, arm_c_schedule, arm_c_final = _arm_c_summary(
        first_trace,
        budget=budget,
        returned_actions=paired.paired_calls[0].comparison_reply.get("actions"),
    )
    arm_c_duplicate, arm_c_duplicate_schedule, arm_c_duplicate_final = (
        _arm_c_summary(
            duplicate_trace,
            budget=budget,
            returned_actions=paired.paired_calls[1].comparison_reply.get("actions"),
        )
    )
    arm_b_final = _batched_array(
        first_nested.get("rollout", {}).get("final"),
        name="Arm B nonlinear selected final",
        shape=(10, 32),
    )
    arm_b_duplicate_final = _batched_array(
        duplicate_nested.get("rollout", {}).get("final"),
        name="Arm B duplicate nonlinear selected final",
        shape=(10, 32),
    )
    arm_a_schedules: dict[str, np.ndarray] = {}
    arm_a_finals: dict[str, np.ndarray] = {}
    for replicate, paired_call in zip(
        ("first", "duplicate"), paired.paired_calls, strict=True
    ):
        ordinary_trace = paired_call.ordinary_reply.get("crfs_trace")
        if not isinstance(ordinary_trace, Mapping):
            raise ConstrainedFlowCanaryError(
                f"Arm A {replicate} reply has no trace"
            )
        arm_a_schedules[replicate] = _finite_array(
            ordinary_trace.get("solver_schedule"),
            name=f"Arm A {replicate} selected schedule",
            shape=(10, 10, 32),
            dtype=np.float32,
        )
        arm_a_finals[replicate] = _finite_array(
            ordinary_trace.get("solver_internal_replay_final"),
            name=f"Arm A {replicate} selected final",
            shape=(10, 32),
            dtype=np.float32,
        )
    duplicate_checks = _duplicate_checks(
        first_trace, first_nested, duplicate_trace, duplicate_nested
    )
    comparison_pairing_checks = _comparison_pairing_checks(
        paired.paired_calls,
        target=target,
        frozen_baseline=frozen_baseline,
        noise=noise,
        scale=scale,
        budget=budget,
    )

    expected_replay_keys = {
        f"{arm}:{replicate}"
        for arm in (
            "A_historical_run_b",
            "B_linearized_candidate",
            "C_linearized_then_historical_adam",
        )
        for replicate in ("first", "duplicate")
    }
    if set(paired.canonical_replies) != expected_replay_keys:
        raise ConstrainedFlowCanaryError(
            "paired client did not complete all six canonical replays before legacy after-checks"
        )
    selected_schedules = {
        ("A_historical_run_b", "first"): arm_a_schedules["first"],
        ("A_historical_run_b", "duplicate"): arm_a_schedules["duplicate"],
        ("B_linearized_candidate", "first"): arm_b_schedule,
        ("B_linearized_candidate", "duplicate"): arm_b_duplicate_schedule,
        ("C_linearized_then_historical_adam", "first"): arm_c_schedule,
        ("C_linearized_then_historical_adam", "duplicate"): arm_c_duplicate_schedule,
    }
    selected_finals = {
        ("A_historical_run_b", "first"): arm_a_finals["first"],
        ("A_historical_run_b", "duplicate"): arm_a_finals["duplicate"],
        ("B_linearized_candidate", "first"): arm_b_final,
        ("B_linearized_candidate", "duplicate"): arm_b_duplicate_final,
        ("C_linearized_then_historical_adam", "first"): arm_c_final,
        ("C_linearized_then_historical_adam", "duplicate"): arm_c_duplicate_final,
    }
    canonical_replays: dict[str, Any] = {}
    for arm in (
        "A_historical_run_b",
        "B_linearized_candidate",
        "C_linearized_then_historical_adam",
    ):
        records: dict[str, Any] = {}
        for replicate in ("first", "duplicate"):
            reply, elapsed = paired.canonical_replies[f"{arm}:{replicate}"]
            records[replicate] = _canonical_replay_record(
                reply,
                schedule=selected_schedules[(arm, replicate)],
                budget=budget,
                noise=noise,
                target=target,
                scale=scale,
                selected_final=selected_finals[(arm, replicate)],
                elapsed=elapsed,
            )
        replay_duplicate_checks = {
            "schedule_exact": _array_exact(
                selected_schedules[(arm, "first")],
                selected_schedules[(arm, "duplicate")],
            ),
            "selected_final_exact": _array_exact(
                selected_finals[(arm, "first")],
                selected_finals[(arm, "duplicate")],
            ),
            "returned_actions_exact": records["first"]["actions"]["sha256"]
            == records["duplicate"]["actions"]["sha256"],
            "replay_final_exact": records["first"]["final_normalized"]["sha256"]
            == records["duplicate"]["final_normalized"]["sha256"],
            "recurrence_trace_exact": content_hash(records["first"]["trace"])
            == content_hash(records["duplicate"]["trace"]),
        }
        canonical_replays[arm] = {
            "first": records["first"],
            "duplicate": records["duplicate"],
            "duplicate_checks": replay_duplicate_checks,
            "passed": bool(
                records["first"]["passed"] is True
                and records["duplicate"]["passed"] is True
                and all(replay_duplicate_checks.values())
            ),
        }

    arm_b_replay = canonical_replays["B_linearized_candidate"]["first"]
    arm_c_replay = canonical_replays[
        "C_linearized_then_historical_adam"
    ]["first"]

    paired_request_checks = {
        "exactly_two_comparison_calls": len(paired.paired_calls) == 2,
        "ordinary_teacher_requests_exact": content_hash(
            _tree_record(paired.paired_calls[0].ordinary_request)
        )
        == content_hash(_tree_record(paired.paired_calls[1].ordinary_request)),
        "comparison_adapter_duplicates_exact": all(duplicate_checks.values()),
        "comparison_source_pairing_exact": all(
            comparison_pairing_checks.values()
        ),
    }
    apparatus_checks = {
        "arm_a_historical_reproduced": arm_a["passed"] is True,
        "arm_b_valid": arm_b["passed"] is True,
        "arm_c_valid": arm_c["passed"] is True,
        "arm_a_canonical_replays": canonical_replays[
            "A_historical_run_b"
        ]["passed"]
        is True,
        "arm_b_canonical_replays": canonical_replays[
            "B_linearized_candidate"
        ]["passed"]
        is True,
        "arm_c_canonical_replays": canonical_replays[
            "C_linearized_then_historical_adam"
        ]["passed"]
        is True,
        "duplicates_valid": all(duplicate_checks.values()),
        "paired_requests_valid": all(paired_request_checks.values()),
        "zero_policy_generated_simulator_steps": legacy_result["simulator_use"][
            "policy_generated_action_steps_executed"
        ]
        == 0,
        "zero_teacher_generated_simulator_steps": legacy_result["simulator_use"][
            "teacher_generated_action_steps_executed"
        ]
        == 0,
        "zero_efficacy_rollouts": legacy_result["simulator_use"]["efficacy_rollouts_executed"]
        == 0,
    }
    apparatus_valid = all(apparatus_checks.values())
    arm_b_nonlinear_pass = bool(arm_b_replay["metrics"]["passed"])
    arm_c_nonlinear_pass = bool(arm_c_replay["metrics"]["passed"])
    status = _classify_outcome(
        apparatus_valid=apparatus_valid,
        arm_b_nonlinear_pass=arm_b_nonlinear_pass,
        arm_c_nonlinear_pass=arm_c_nonlinear_pass,
    )
    if status not in OUTCOME_STATUSES:
        raise AssertionError("unregistered constrained-flow status")

    config_path = Path(constrained_config_path)
    legacy_config_file = Path(legacy_config_path)
    output = Path(legacy_path).with_name("constrained-flow-payload.json")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "payload_type": PAYLOAD_TYPE,
        "payload_variant": "complete_comparison",
        "status": status,
        "case_id": CASE_ID,
        "run_id": legacy_result.get("run_id"),
        "legacy_payload_path": str(legacy_path),
        "legacy_payload_sha256": file_sha256(legacy_path),
        "config": {
            "constrained_flow_path": str(config_path),
            "constrained_flow_sha256": file_sha256(config_path),
            "constrained_flow_scientific_hash": constrained_flow_scientific_config_hash(
                constrained_config
            ),
            "legacy_path": str(legacy_config_file),
            "legacy_sha256": file_sha256(legacy_config_file),
        },
        "source_pairing": {
            "manifest_sha256": input_manifest_sha256,
            "source_r02_sha256": legacy_result["source_evidence"]["r02_case_sha256"],
            "checkpoint_sha256": legacy_result["provenance"]["checkpoint_sha256"],
            "case_record_sha256": legacy_result["provenance"]["case_record_sha256"],
            "target_normalized": legacy_result["target"]["target_normalized"],
            "noise": legacy_result["provenance"]["noise"],
            "budget_float32": float(budget),
            "checkpoint_model_to_physical_scale": _array_record(scale, dtype=np.float32),
            "comparison_checks": comparison_pairing_checks,
            "comparison_checks_passed": all(comparison_pairing_checks.values()),
            "comparison_request_records": _comparison_request_records(
                paired.paired_calls
            ),
        },
        "arms": {
            "A_historical_run_b": arm_a,
            "B_linearized_candidate": {
                "first": arm_b,
                "duplicate": arm_b_duplicate,
            },
            "C_linearized_then_historical_adam": {
                "first": arm_c,
                "duplicate": arm_c_duplicate,
            },
        },
        "duplicates": {
            "checks": duplicate_checks,
            "passed": all(duplicate_checks.values()),
            "comparison_elapsed_seconds": [
                call.comparison_elapsed_seconds for call in paired.paired_calls
            ],
        },
        "canonical_replays": canonical_replays,
        "apparatus": {
            "checks": apparatus_checks,
            "paired_request_checks": paired_request_checks,
            "passed": apparatus_valid,
        },
        "outcome": {
            "status": status,
            "arm_b_linear_prediction_passed": bool(arm_b["linear_metrics"]["passed"]),
            "arm_b_nonlinear_replay_passed": arm_b_nonlinear_pass,
            "arm_c_nonlinear_replay_passed": arm_c_nonlinear_pass,
            "finite_nonconvergence_is_infeasibility": False,
            "optimality_certificate": False,
            "infeasibility_certificate": False,
            "simulator_efficacy_evaluated": False,
            "collision_or_progress_claim_allowed": False,
            "student_or_generalization_claim_allowed": False,
            "probe_training_authorized": False,
            "automatic_next_gate_authorized": False,
        },
        "simulator_use": {
            "setup_only": True,
            "policy_generated_action_steps_executed": 0,
            "teacher_generated_action_steps_executed": 0,
            "efficacy_rollouts_executed": 0,
            "simulator_efficacy_evaluated": False,
        },
    }
    atomic_write_json(output, payload)
    return output, status


__all__ = [
    "ConstrainedFlowCanaryError",
    "PairedConstrainedFlowClient",
    "SCIENTIFIC_CONFIG_HASH",
    "constrained_flow_scientific_config_hash",
    "run_r05a_constrained_flow_canary",
    "validate_constrained_flow_config",
]
