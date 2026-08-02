"""Minimal sampled-data Poisson-CBF shield for a nominal OSC torque.

This module is deliberately independent of robosuite, MuJoCo, and the policy.
The caller supplies the nominal seven-arm-joint torque and an ordered
``M``-dimensional velocity reached by an exact nominal one-step clone.  It also
supplies a local sensitivity ``S = d(v_next) / d(tau)`` with shape ``(M, 7)``
that already includes the controller interval and any local contact effects.
Consequently ``S`` is *not* multiplied by ``dt`` again here.

For Poisson samples ``h`` and joint rows ``a = grad(h)^T J``, the shield solves

    min 0.5 ||delta_tau||^2

subject to

    a @ (v_nom_next + S @ delta_tau) + alpha * h >= margin
    tau_lower <= tau_nominal + delta_tau <= tau_upper.

There is no slack, clipping, cached-command reuse, or pass-through after an
error.  A caller must stop before physics whenever ``valid`` is false.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class TorqueShieldStatus(str, Enum):
    """Typed terminal status for one shield decision."""

    NOMINAL_SAFE = "nominal_safe"
    SOLVED = "solved"
    INVALID_INPUT = "invalid_input"
    ILL_CONDITIONED_SENSITIVITY = "ill_conditioned_sensitivity"
    UNCONTROLLABLE_CONSTRAINT = "uncontrollable_constraint"
    QP_INFEASIBLE = "qp_infeasible"
    QP_SOLVER_FAILURE = "qp_solver_failure"
    QP_POSTCHECK_FAILED = "qp_postcheck_failed"


@dataclass(frozen=True)
class TorqueShieldResult:
    """One fail-closed torque-shield result.

    ``torque_command`` and ``delta_torque`` are absent on every failure.  In
    the nominal-safe fast path, ``torque_command`` is an exact float64 copy of
    the input and therefore has byte-identical values without an optimizer
    round trip.
    """

    valid: bool
    status: TorqueShieldStatus
    torque_command: Optional[Any]
    delta_torque: Optional[Any]
    diagnostics: Mapping[str, Any]

    @property
    def reason(self) -> str:
        """String alias compatible with existing fail-closed call sites."""

        return self.status.value


def _numeric_modules() -> Tuple[Any, Any, Any]:
    try:
        import numpy as np
        import osqp
        import scipy.sparse as sparse
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError(
            "NumPy, SciPy, and OSQP are required for the post-OSC torque shield"
        ) from error
    return np, osqp, sparse


def _finite_vector(value: Sequence[float], length: int, label: str, np: Any) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError("%s must be a finite vector of length %d" % (label, length))
    return array


def _minimum_or_none(values: Any) -> Optional[float]:
    return float(values.min()) if values.size else None


def evaluate_torque_shield_residuals(
    *,
    nominal_torque: Sequence[float],
    candidate_torque: Sequence[float],
    nominal_next_qvel: Sequence[float],
    h: Sequence[float],
    joint_gradient_rows: Any,
    torque_to_next_qvel_sensitivity: Any,
    torque_lower: Sequence[float],
    torque_upper: Sequence[float],
    alpha: float,
    margin: float,
    normalized_cbf_tolerance: float,
    torque_bound_tolerance: float,
) -> Dict[str, Any]:
    """Independently reconstruct physical residuals for a candidate torque.

    This function intentionally starts from the original dynamics inputs.  It
    does not accept the QP's normalized matrix or lower-bound vector, so a
    construction/sign error in the optimizer path is exposed by the postcheck.
    """

    np, _, _ = _numeric_modules()
    nominal = _finite_vector(nominal_torque, 7, "nominal_torque", np)
    candidate = _finite_vector(candidate_torque, 7, "candidate_torque", np)
    nominal_velocity = np.asarray(nominal_next_qvel, dtype=np.float64)
    if (
        nominal_velocity.ndim != 1
        or nominal_velocity.size == 0
        or not np.all(np.isfinite(nominal_velocity))
    ):
        raise ValueError("nominal_next_qvel must be a nonempty finite vector")
    velocity_dimension = int(nominal_velocity.size)
    lower = _finite_vector(torque_lower, 7, "torque_lower", np)
    upper = _finite_vector(torque_upper, 7, "torque_upper", np)
    h_array = np.asarray(h, dtype=np.float64)
    rows = np.asarray(joint_gradient_rows, dtype=np.float64)
    sensitivity = np.asarray(torque_to_next_qvel_sensitivity, dtype=np.float64)
    if h_array.ndim != 1 or h_array.size == 0:
        raise ValueError("h must be a nonempty finite vector")
    if rows.shape != (h_array.size, velocity_dimension):
        raise ValueError("joint_gradient_rows must have shape (N, M)")
    if sensitivity.shape != (velocity_dimension, 7):
        raise ValueError(
            "torque_to_next_qvel_sensitivity must have shape (M, 7)"
        )
    if not (
        np.all(np.isfinite(h_array))
        and np.all(np.isfinite(rows))
        and np.all(np.isfinite(sensitivity))
    ):
        raise ValueError("h, joint_gradient_rows, and sensitivity must be finite")
    scalar_values = (
        ("alpha", alpha, True),
        ("margin", margin, False),
        ("normalized_cbf_tolerance", normalized_cbf_tolerance, False),
        ("torque_bound_tolerance", torque_bound_tolerance, False),
    )
    for label, value, strictly_positive in scalar_values:
        if isinstance(value, bool):
            raise ValueError("%s must be finite" % label)
        numeric = float(value)
        if not math.isfinite(numeric) or (
            numeric <= 0.0 if strictly_positive else numeric < 0.0
        ):
            qualifier = "positive" if strictly_positive else "nonnegative"
            raise ValueError("%s must be finite and %s" % (label, qualifier))
    if np.any(lower > upper):
        raise ValueError("torque limits are contradictory")

    delta = candidate - nominal
    safe_next_velocity = nominal_velocity + sensitivity @ delta
    raw_residual = (
        rows @ safe_next_velocity + float(alpha) * h_array - float(margin)
    )
    torque_gain_rows = rows @ sensitivity
    row_scales = np.max(np.abs(torque_gain_rows), axis=1)
    nonzero = row_scales > 0.0
    normalized_residual = np.full(h_array.shape, np.nan, dtype=np.float64)
    normalized_residual[nonzero] = raw_residual[nonzero] / row_scales[nonzero]

    # A zero-gain row cannot be repaired by this local torque input.  Its raw
    # inequality therefore remains hard and receives no arbitrary scaling.
    zero_gain_violation = bool(np.any((~nonzero) & (raw_residual < 0.0)))
    normalized_violation = bool(
        np.any(
            normalized_residual[nonzero]
            < -float(normalized_cbf_tolerance)
        )
    )
    lower_violation = float(np.max(np.maximum(lower - candidate, 0.0)))
    upper_violation = float(np.max(np.maximum(candidate - upper, 0.0)))
    maximum_bound_violation = max(lower_violation, upper_violation)
    finite_derived = bool(
        np.all(np.isfinite(safe_next_velocity))
        and np.all(np.isfinite(raw_residual))
        and np.all(np.isfinite(torque_gain_rows))
        and np.all(np.isfinite(row_scales))
        and np.all(np.isfinite(normalized_residual[nonzero]))
    )
    feasible = bool(
        finite_derived
        and not zero_gain_violation
        and not normalized_violation
        and maximum_bound_violation <= float(torque_bound_tolerance)
    )
    return {
        "feasible": feasible,
        "finite_derived": finite_derived,
        "safe_next_qvel_rad_s": safe_next_velocity.tolist(),
        "raw_cbf_residuals": raw_residual.tolist(),
        "minimum_raw_cbf_residual": _minimum_or_none(raw_residual),
        "normalized_cbf_residuals": [
            float(value) if math.isfinite(float(value)) else None
            for value in normalized_residual
        ],
        "minimum_normalized_cbf_residual": (
            _minimum_or_none(normalized_residual[nonzero])
            if np.any(nonzero)
            else None
        ),
        "zero_gain_constraint_count": int(np.count_nonzero(~nonzero)),
        "zero_gain_violation": zero_gain_violation,
        "maximum_torque_bound_violation": maximum_bound_violation,
        "correction_l2_nm": float(np.linalg.norm(delta)),
        "safe_next_qvel_change_l2_rad_s": float(
            np.linalg.norm(safe_next_velocity - nominal_velocity)
        ),
    }


class SampledDataPostOscTorqueShield:
    """Hard minimum-change torque shield with a nominal-safe fast path."""

    def __init__(
        self,
        *,
        eps_abs: float = 1e-8,
        eps_rel: float = 1e-8,
        max_iter: int = 20000,
        maximum_sensitivity_condition_number: float = 1e8,
        postcheck_normalized_cbf_tolerance: float = 5e-7,
        postcheck_torque_bound_tolerance: float = 5e-8,
    ) -> None:
        positive_values = (
            eps_abs,
            eps_rel,
            maximum_sensitivity_condition_number,
            postcheck_normalized_cbf_tolerance,
            postcheck_torque_bound_tolerance,
        )
        if any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in positive_values
        ):
            raise ValueError("shield tolerances and condition limit must be positive")
        if float(maximum_sensitivity_condition_number) < 1.0:
            raise ValueError("maximum sensitivity condition number must be at least one")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.max_iter = int(max_iter)
        self.maximum_sensitivity_condition_number = float(
            maximum_sensitivity_condition_number
        )
        self.postcheck_normalized_cbf_tolerance = float(
            postcheck_normalized_cbf_tolerance
        )
        self.postcheck_torque_bound_tolerance = float(
            postcheck_torque_bound_tolerance
        )

    @staticmethod
    def _failure(
        status: TorqueShieldStatus, diagnostics: Mapping[str, Any]
    ) -> TorqueShieldResult:
        return TorqueShieldResult(False, status, None, None, dict(diagnostics))

    def solve(
        self,
        *,
        nominal_torque: Sequence[float],
        current_qvel: Sequence[float],
        nominal_next_qvel: Sequence[float],
        h: Sequence[float],
        joint_gradient_rows: Any,
        dt_seconds: float,
        torque_to_next_qvel_sensitivity: Any,
        sensitivity_max_absolute_error: float = 0.0,
        torque_lower: Sequence[float],
        torque_upper: Sequence[float],
        alpha: float,
        margin: float = 0.0,
    ) -> TorqueShieldResult:
        """Return the nominal torque, a solved correction, or a typed failure."""

        np, osqp, sparse = _numeric_modules()
        try:
            nominal = _finite_vector(nominal_torque, 7, "nominal_torque", np)
            nominal_velocity = np.asarray(nominal_next_qvel, dtype=np.float64)
            if (
                nominal_velocity.ndim != 1
                or nominal_velocity.size == 0
                or not np.all(np.isfinite(nominal_velocity))
            ):
                raise ValueError("nominal_next_qvel must be a nonempty finite vector")
            velocity_dimension = int(nominal_velocity.size)
            current_velocity = _finite_vector(
                current_qvel, velocity_dimension, "current_qvel", np
            )
            lower_torque = _finite_vector(torque_lower, 7, "torque_lower", np)
            upper_torque = _finite_vector(torque_upper, 7, "torque_upper", np)
            h_array = np.asarray(h, dtype=np.float64)
            rows = np.asarray(joint_gradient_rows, dtype=np.float64)
            sensitivity = np.asarray(
                torque_to_next_qvel_sensitivity, dtype=np.float64
            )
            if h_array.ndim != 1 or h_array.size == 0:
                raise ValueError("h must be a nonempty finite vector")
            if rows.shape != (h_array.size, velocity_dimension):
                raise ValueError("joint_gradient_rows must have shape (N, M)")
            if sensitivity.shape != (velocity_dimension, 7):
                raise ValueError(
                    "torque_to_next_qvel_sensitivity must have shape (M, 7)"
                )
            if not (
                np.all(np.isfinite(h_array))
                and np.all(np.isfinite(rows))
                and np.all(np.isfinite(sensitivity))
            ):
                raise ValueError(
                    "h, joint_gradient_rows, and sensitivity must be finite"
                )
            if np.any(lower_torque > upper_torque):
                raise ValueError("torque limits are contradictory")
            if isinstance(dt_seconds, bool) or not math.isfinite(float(dt_seconds)):
                raise ValueError("dt_seconds must be finite and positive")
            if float(dt_seconds) <= 0.0:
                raise ValueError("dt_seconds must be finite and positive")
            if isinstance(alpha, bool) or not math.isfinite(float(alpha)):
                raise ValueError("alpha must be finite and positive")
            if float(alpha) <= 0.0:
                raise ValueError("alpha must be finite and positive")
            if isinstance(margin, bool) or not math.isfinite(float(margin)):
                raise ValueError("margin must be finite and nonnegative")
            if float(margin) < 0.0:
                raise ValueError("margin must be finite and nonnegative")
            if (
                isinstance(sensitivity_max_absolute_error, bool)
                or not math.isfinite(float(sensitivity_max_absolute_error))
                or float(sensitivity_max_absolute_error) < 0.0
            ):
                raise ValueError(
                    "sensitivity_max_absolute_error must be finite and nonnegative"
                )
        except (TypeError, ValueError) as error:
            return self._failure(
                TorqueShieldStatus.INVALID_INPUT,
                {"error_type": type(error).__name__, "error": str(error)},
            )

        try:
            singular_values = np.linalg.svd(sensitivity, compute_uv=False)
        except np.linalg.LinAlgError as error:
            return self._failure(
                TorqueShieldStatus.ILL_CONDITIONED_SENSITIVITY,
                {"error_type": type(error).__name__, "error": str(error)},
            )
        largest_singular = float(singular_values[0])
        smallest_singular = float(singular_values[-1])
        numerical_rank_threshold = (
            np.finfo(np.float64).eps
            * float(max(velocity_dimension, 7))
            * largest_singular
        )
        condition_number = (
            largest_singular / smallest_singular
            if smallest_singular > 0.0
            else math.inf
        )
        conditioning = {
            "sensitivity_singular_values": singular_values.tolist(),
            "sensitivity_condition_number": (
                float(condition_number) if math.isfinite(condition_number) else None
            ),
            "maximum_sensitivity_condition_number": (
                self.maximum_sensitivity_condition_number
            ),
            "numerical_rank_threshold": float(numerical_rank_threshold),
        }
        if not np.all(np.isfinite(singular_values)):
            return self._failure(
                TorqueShieldStatus.ILL_CONDITIONED_SENSITIVITY, conditioning
            )
        # Full-matrix rank is not the control authority for this QP.  A
        # contact constraint only needs its own row ``a @ S`` to be
        # controllable.  Rejecting a rank-deficient Mx7 S would discard valid
        # corrections in directions unaffected by the deficient mode (and
        # would even reject an already-safe nominal command).  Preserve the
        # condition number as an audit diagnostic; the exact per-row gain
        # checks below are the fail-closed authority.
        conditioning["global_condition_limit_exceeded"] = bool(
            not math.isfinite(condition_number)
            or condition_number > self.maximum_sensitivity_condition_number
        )
        conditioning["global_condition_is_advisory_only"] = True

        with np.errstate(over="ignore", invalid="ignore"):
            torque_gain_rows = rows @ sensitivity
            required_gain = (
                float(margin)
                - float(alpha) * h_array
                - rows @ nominal_velocity
            )
        if not (
            np.all(np.isfinite(torque_gain_rows))
            and np.all(np.isfinite(required_gain))
        ):
            return self._failure(
                TorqueShieldStatus.INVALID_INPUT,
                {"error": "derived torque constraints are non-finite", **conditioning},
            )

        base_diagnostics: Dict[str, Any] = {
            "schema": "vlsa_poisson_post_osc_torque_shield.v1",
            "constraint_equation": (
                "a@(v_nom_next+S@delta_tau)+alpha*h>=margin"
            ),
            "sample_count": int(h_array.size),
            "dt_seconds": float(dt_seconds),
            "alpha": float(alpha),
            "margin": float(margin),
            "sensitivity_already_includes_dt_and_contact_effects": True,
            "sensitivity_max_absolute_error": float(
                sensitivity_max_absolute_error
            ),
            "nominal_torque_nm": nominal.tolist(),
            "torque_lower_nm": lower_torque.tolist(),
            "torque_upper_nm": upper_torque.tolist(),
            "current_qvel_rad_s": current_velocity.tolist(),
            "nominal_next_qvel_rad_s": nominal_velocity.tolist(),
            "nominal_average_acceleration_l2_rad_s2": float(
                np.linalg.norm(nominal_velocity - current_velocity)
                / float(dt_seconds)
            ),
            **conditioning,
        }
        if velocity_dimension != 7:
            base_diagnostics.update(
                {
                    "velocity_dimension": velocity_dimension,
                    "torque_dimension": 7,
                }
            )
        try:
            nominal_postcheck = evaluate_torque_shield_residuals(
                nominal_torque=nominal,
                candidate_torque=nominal,
                nominal_next_qvel=nominal_velocity,
                h=h_array,
                joint_gradient_rows=rows,
                torque_to_next_qvel_sensitivity=sensitivity,
                torque_lower=lower_torque,
                torque_upper=upper_torque,
                alpha=float(alpha),
                margin=float(margin),
                # The fast path is exact: tolerance does not turn an unsafe
                # nominal command into a pass-through command.
                normalized_cbf_tolerance=0.0,
                torque_bound_tolerance=0.0,
            )
        except (TypeError, ValueError) as error:  # defensive independent path
            return self._failure(
                TorqueShieldStatus.INVALID_INPUT,
                {**base_diagnostics, "postcheck_error": str(error)},
            )
        if nominal_postcheck["feasible"]:
            command = nominal.copy()
            delta = np.zeros(7, dtype=np.float64)
            diagnostics = {
                **base_diagnostics,
                "solver_attempted": False,
                "nominal_bytes_preserved": bool(
                    command.tobytes(order="C") == nominal.tobytes(order="C")
                ),
                "nominal_postcheck": nominal_postcheck,
                "postcheck": nominal_postcheck,
            }
            return TorqueShieldResult(
                True,
                TorqueShieldStatus.NOMINAL_SAFE,
                command,
                delta,
                diagnostics,
            )

        row_scales = np.max(np.abs(torque_gain_rows), axis=1)
        # Propagate the registered elementwise sensitivity disagreement into
        # each CBF row: |a @ dS|_inf <= ||a||_1 * ||dS||_max.  A row is treated
        # as controllable only when its observed gain is strictly larger than
        # that numerical uncertainty.  The later exact nonlinear clone remains
        # the candidate authority.
        row_gain_uncertainty = (
            np.sum(np.abs(rows), axis=1)
            * float(sensitivity_max_absolute_error)
        )
        effective_row_gain = row_scales - row_gain_uncertainty
        uncontrollable = effective_row_gain <= 0.0
        impossible = uncontrollable & (required_gain > 0.0)
        if np.any(impossible):
            return self._failure(
                TorqueShieldStatus.UNCONTROLLABLE_CONSTRAINT,
                {
                    **base_diagnostics,
                    "solver_attempted": False,
                    "constraint_indexes": (
                        np.flatnonzero(impossible).astype(int).tolist()
                    ),
                    "required_torque_gains": required_gain[impossible].tolist(),
                    "observed_torque_gain_row_scales": row_scales[
                        impossible
                    ].tolist(),
                    "torque_gain_row_uncertainty_bounds": row_gain_uncertainty[
                        impossible
                    ].tolist(),
                    "nominal_postcheck": nominal_postcheck,
                },
            )
        keep = ~uncontrollable
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            normalized_rows = torque_gain_rows[keep] / row_scales[keep, None]
            normalized_lower = required_gain[keep] / row_scales[keep]
        if not (
            np.all(np.isfinite(normalized_rows))
            and np.all(np.isfinite(normalized_lower))
        ):
            return self._failure(
                TorqueShieldStatus.INVALID_INPUT,
                {
                    **base_diagnostics,
                    "solver_attempted": False,
                    "error": "normalized torque constraints are non-finite",
                },
            )

        delta_lower = lower_torque - nominal
        delta_upper = upper_torque - nominal
        identity = sparse.eye(7, format="csc", dtype=np.float64)
        if normalized_rows.shape[0]:
            constraint_matrix = sparse.vstack(
                (sparse.csc_matrix(normalized_rows), identity), format="csc"
            )
            qp_lower = np.concatenate((normalized_lower, delta_lower))
            qp_upper = np.concatenate(
                (
                    np.full(normalized_rows.shape[0], np.inf, dtype=np.float64),
                    delta_upper,
                )
            )
        else:
            constraint_matrix = identity
            qp_lower = delta_lower
            qp_upper = delta_upper

        solver = osqp.OSQP()
        try:
            solver.setup(
                P=identity,
                q=np.zeros(7, dtype=np.float64),
                A=constraint_matrix,
                l=qp_lower,
                u=qp_upper,
                verbose=False,
                eps_abs=self.eps_abs,
                eps_rel=self.eps_rel,
                max_iter=self.max_iter,
                polishing=True,
                warm_starting=False,
                adaptive_rho=True,
            )
            solution = solver.solve()
        except Exception as error:  # OSQP exposes backend-specific exceptions
            return self._failure(
                TorqueShieldStatus.QP_SOLVER_FAILURE,
                {
                    **base_diagnostics,
                    "solver_attempted": True,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

        solver_status = str(getattr(solution.info, "status", "unknown")).lower()
        solver_diagnostics = {
            **base_diagnostics,
            "solver_attempted": True,
            "solver": "osqp",
            "solver_status": solver_status,
            "solver_status_value": int(getattr(solution.info, "status_val", -1)),
            "solver_iterations": int(getattr(solution.info, "iter", -1)),
            "input_constraint_count": int(h_array.size),
            "solved_constraint_count": int(normalized_rows.shape[0]),
            "numerically_uncontrollable_constraint_count": int(
                np.count_nonzero(uncontrollable)
            ),
            "maximum_torque_gain_row_uncertainty": float(
                np.max(row_gain_uncertainty)
            ),
            "minimum_effective_controllable_row_gain": (
                float(np.min(effective_row_gain[keep])) if np.any(keep) else None
            ),
            "minimum_nonzero_torque_gain_row_scale": (
                float(np.min(row_scales[keep])) if np.any(keep) else None
            ),
            "maximum_nonzero_torque_gain_row_scale": (
                float(np.max(row_scales[keep])) if np.any(keep) else None
            ),
            "nominal_postcheck": nominal_postcheck,
        }
        if solver_status not in ("solved", "solved inaccurate") or solution.x is None:
            failure_status = (
                TorqueShieldStatus.QP_INFEASIBLE
                if "infeasible" in solver_status
                else TorqueShieldStatus.QP_SOLVER_FAILURE
            )
            return self._failure(failure_status, solver_diagnostics)

        delta = np.asarray(solution.x, dtype=np.float64)
        if delta.shape != (7,) or not np.all(np.isfinite(delta)):
            return self._failure(
                TorqueShieldStatus.QP_SOLVER_FAILURE,
                {**solver_diagnostics, "error": "solver returned an invalid delta"},
            )
        candidate = nominal + delta
        try:
            postcheck = evaluate_torque_shield_residuals(
                nominal_torque=nominal,
                candidate_torque=candidate,
                nominal_next_qvel=nominal_velocity,
                h=h_array,
                joint_gradient_rows=rows,
                torque_to_next_qvel_sensitivity=sensitivity,
                torque_lower=lower_torque,
                torque_upper=upper_torque,
                alpha=float(alpha),
                margin=float(margin),
                normalized_cbf_tolerance=(
                    self.postcheck_normalized_cbf_tolerance
                ),
                torque_bound_tolerance=self.postcheck_torque_bound_tolerance,
            )
        except (TypeError, ValueError) as error:
            return self._failure(
                TorqueShieldStatus.QP_POSTCHECK_FAILED,
                {**solver_diagnostics, "postcheck_error": str(error)},
            )
        diagnostics = {**solver_diagnostics, "postcheck": postcheck}
        if not postcheck["feasible"]:
            return self._failure(
                TorqueShieldStatus.QP_POSTCHECK_FAILED, diagnostics
            )
        return TorqueShieldResult(
            True,
            TorqueShieldStatus.SOLVED,
            candidate,
            delta,
            diagnostics,
        )


__all__ = [
    "SampledDataPostOscTorqueShield",
    "TorqueShieldResult",
    "TorqueShieldStatus",
    "evaluate_torque_shield_residuals",
]
