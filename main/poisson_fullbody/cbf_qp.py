"""Hard seven-joint CBF-QP for the static Poisson feasibility pilot.

The module deliberately contains no SafeLIBERO or policy code.  Inputs and
outputs use physical SI units: joint velocities are radians / second, Poisson
values are square metres, Poisson gradients are metres, and the controller
interval is seconds.
Numerical dependencies are imported lazily so the repository's dependency-
light structural gate can still inspect this module.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FilterResult:
    """One fail-closed filter decision.

    ``qdot_safe`` is absent whenever ``valid`` is false.  Callers must
    terminate before another physics step in that state; they must not reuse a
    previous command or reinterpret the missing command as a solved zero.
    """

    valid: bool
    reason: str
    qdot_safe: Optional[Any]
    diagnostics: Dict[str, Any]


def _numeric_modules() -> Tuple[Any, Any, Any]:
    try:
        import numpy as np
        import osqp
        import scipy.sparse as sparse
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError(
            "NumPy, SciPy, and OSQP are required for the Poisson CBF-QP"
        ) from error
    return np, osqp, sparse


def _finite_vector(value: Sequence[float], length: int, label: str, np: Any) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError("%s must be a finite vector of length %d" % (label, length))
    return array


def joint_velocity_bounds(
    q: Sequence[float],
    q_min: Sequence[float],
    q_max: Sequence[float],
    physical_min: Sequence[float],
    physical_max: Sequence[float],
    *,
    alpha_joint: float,
    control_dt_seconds: float,
    position_margin_rad: float = 0.0,
) -> Tuple[Any, Any]:
    """Intersect physical, continuous-CBF, and one-step position bounds."""

    np, _, _ = _numeric_modules()
    q_array = _finite_vector(q, 7, "q", np)
    lower_position = _finite_vector(q_min, 7, "q_min", np)
    upper_position = _finite_vector(q_max, 7, "q_max", np)
    lower_physical = _finite_vector(physical_min, 7, "physical_min", np)
    upper_physical = _finite_vector(physical_max, 7, "physical_max", np)
    if not math.isfinite(alpha_joint) or alpha_joint <= 0.0:
        raise ValueError("alpha_joint must be finite and positive")
    if not math.isfinite(control_dt_seconds) or control_dt_seconds <= 0.0:
        raise ValueError("control_dt_seconds must be finite and positive")
    if not math.isfinite(position_margin_rad) or position_margin_rad < 0.0:
        raise ValueError("position_margin_rad must be finite and non-negative")
    if np.any(lower_position >= upper_position):
        raise ValueError("every q_min must be strictly less than q_max")
    if np.any(lower_physical > upper_physical):
        raise ValueError("physical velocity bounds are contradictory")

    allowed_lower = lower_position + float(position_margin_rad)
    allowed_upper = upper_position - float(position_margin_rad)
    if np.any(allowed_lower >= allowed_upper):
        raise ValueError("joint-position margin removes the admissible range")
    tolerance = 1e-10
    if np.any(q_array < allowed_lower - tolerance) or np.any(
        q_array > allowed_upper + tolerance
    ):
        raise ValueError("joint state is outside the registered admissible range")

    cbf_lower = -float(alpha_joint) * (q_array - lower_position)
    cbf_upper = float(alpha_joint) * (upper_position - q_array)
    one_step_lower = (allowed_lower - q_array) / float(control_dt_seconds)
    one_step_upper = (allowed_upper - q_array) / float(control_dt_seconds)
    lower = np.maximum.reduce((lower_physical, cbf_lower, one_step_lower))
    upper = np.minimum.reduce((upper_physical, cbf_upper, one_step_upper))
    if np.any(lower > upper + tolerance):
        raise ValueError("joint velocity bounds are infeasible")
    return lower, upper


def static_cbf_rows(
    h: Sequence[float],
    gradients_world: Any,
    point_jacobians: Any,
    *,
    alpha: float,
) -> Tuple[Any, Any]:
    """Return rows ``A`` and lower bounds ``b`` for ``A qdot >= b``."""

    np, _, _ = _numeric_modules()
    h_array = np.asarray(h, dtype=np.float64)
    gradients = np.asarray(gradients_world, dtype=np.float64)
    jacobians = np.asarray(point_jacobians, dtype=np.float64)
    if h_array.ndim != 1:
        raise ValueError("h must be one-dimensional")
    count = int(h_array.shape[0])
    if gradients.shape != (count, 3):
        raise ValueError("gradients_world must have shape (N, 3)")
    if jacobians.shape != (count, 3, 7):
        raise ValueError("point_jacobians must have shape (N, 3, 7)")
    if not (
        np.all(np.isfinite(h_array))
        and np.all(np.isfinite(gradients))
        and np.all(np.isfinite(jacobians))
    ):
        raise ValueError("CBF inputs must be finite")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    rows = np.einsum("ni,nij->nj", gradients, jacobians)
    lower = -float(alpha) * h_array
    return rows, lower


class HardCbfQp:
    """Minimum-change hard CBF-QP with independent residual post-checks."""

    def __init__(
        self,
        *,
        eps_abs: float = 1e-7,
        eps_rel: float = 1e-7,
        max_iter: int = 10000,
        postcheck_cbf_tolerance: float = 5e-7,
        postcheck_bound_tolerance: float = 5e-8,
    ) -> None:
        values = (
            eps_abs,
            eps_rel,
            postcheck_cbf_tolerance,
            postcheck_bound_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("QP tolerances must be finite and positive")
        if (
            isinstance(max_iter, bool)
            or not isinstance(max_iter, int)
            or max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer")
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.max_iter = int(max_iter)
        self.postcheck_cbf_tolerance = float(postcheck_cbf_tolerance)
        self.postcheck_bound_tolerance = float(postcheck_bound_tolerance)

    def solve_from_field(
        self,
        qdot_nominal: Sequence[float],
        h: Sequence[float],
        gradients_world: Any,
        point_jacobians: Any,
        velocity_lower: Sequence[float],
        velocity_upper: Sequence[float],
        *,
        alpha: float,
        weight_diagonal: Optional[Sequence[float]] = None,
        require_safe_start: bool = True,
        safe_start_tolerance: float = 1e-12,
    ) -> FilterResult:
        np, _, _ = _numeric_modules()
        if not isinstance(require_safe_start, bool):
            raise TypeError("require_safe_start must be Boolean")
        if isinstance(safe_start_tolerance, bool):
            raise ValueError(
                "safe_start_tolerance must be finite and nonnegative"
            )
        safe_start_tolerance = float(safe_start_tolerance)
        if not math.isfinite(safe_start_tolerance) or safe_start_tolerance < 0.0:
            raise ValueError(
                "safe_start_tolerance must be finite and nonnegative"
            )
        h_array = np.asarray(h, dtype=np.float64)
        if h_array.ndim == 1 and h_array.size == 0:
            return FilterResult(
                valid=False,
                reason="no_cbf_samples",
                qdot_safe=None,
                diagnostics={"sample_count": 0},
            )
        if require_safe_start and (
            h_array.ndim != 1
            or not np.all(np.isfinite(h_array))
            or np.any(h_array < -safe_start_tolerance)
        ):
            return FilterResult(
                valid=False,
                reason="unsafe_or_invalid_field_start",
                qdot_safe=None,
                diagnostics={
                    "minimum_h": (
                        float(np.min(h_array))
                        if h_array.size and np.all(np.isfinite(h_array))
                        else None
                    )
                },
            )
        try:
            rows, row_lower = static_cbf_rows(
                h_array,
                gradients_world,
                point_jacobians,
                alpha=alpha,
            )
        except ValueError as error:
            return FilterResult(False, "invalid_cbf_inputs", None, {"error": str(error)})
        return self.solve_rows(
            qdot_nominal,
            rows,
            row_lower,
            velocity_lower,
            velocity_upper,
            weight_diagonal=weight_diagonal,
        )

    def solve_rows(
        self,
        qdot_nominal: Sequence[float],
        cbf_rows: Any,
        cbf_lower: Sequence[float],
        velocity_lower: Sequence[float],
        velocity_upper: Sequence[float],
        *,
        weight_diagonal: Optional[Sequence[float]] = None,
    ) -> FilterResult:
        np, osqp, sparse = _numeric_modules()
        try:
            nominal = _finite_vector(qdot_nominal, 7, "qdot_nominal", np)
            lower_velocity = _finite_vector(
                velocity_lower, 7, "velocity_lower", np
            )
            upper_velocity = _finite_vector(
                velocity_upper, 7, "velocity_upper", np
            )
            rows = np.asarray(cbf_rows, dtype=np.float64)
            lower_rows = np.asarray(cbf_lower, dtype=np.float64)
            if rows.ndim != 2 or rows.shape[1] != 7:
                raise ValueError("cbf_rows must have shape (N, 7)")
            if lower_rows.shape != (rows.shape[0],):
                raise ValueError("cbf_lower must have shape (N,)")
            if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(lower_rows)):
                raise ValueError("CBF rows and bounds must be finite")
            if np.any(lower_velocity > upper_velocity):
                raise ValueError("velocity bounds are contradictory")
            weights = (
                np.ones(7, dtype=np.float64)
                if weight_diagonal is None
                else _finite_vector(weight_diagonal, 7, "weight_diagonal", np)
            )
            if np.any(weights <= 0.0):
                raise ValueError("weight_diagonal must be strictly positive")
        except ValueError as error:
            return FilterResult(False, "invalid_qp_inputs", None, {"error": str(error)})

        # Normalize every mathematically nonzero row by its max coefficient.
        # Absolute "near-zero" thresholds are not scale invariant: multiplying
        # an infeasible halfspace by 1e-18 must not make it disappear.  Exact
        # zero rows are the only rows that may be dropped.
        row_scales = np.max(np.abs(rows), axis=1) if rows.shape[0] else np.empty(0)
        if not np.all(np.isfinite(row_scales)):
            return FilterResult(
                False,
                "invalid_qp_scaling",
                None,
                {"error": "non-finite derived row scale"},
            )
        zero_rows = row_scales == 0.0
        impossible_zero = zero_rows & (lower_rows > 0.0)
        if np.any(impossible_zero):
            indexes = np.flatnonzero(impossible_zero).astype(int).tolist()
            return FilterResult(
                False,
                "uncontrollable_cbf_constraint",
                None,
                {
                    "constraint_indexes": indexes,
                    "positive_lower_bounds": lower_rows[impossible_zero].tolist(),
                },
            )
        keep = ~zero_rows
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scaled_rows = rows[keep] / row_scales[keep, None]
            scaled_lower = lower_rows[keep] / row_scales[keep]
        if not np.all(np.isfinite(scaled_rows)) or not np.all(
            np.isfinite(scaled_lower)
        ):
            return FilterResult(
                False,
                "invalid_qp_scaling",
                None,
                {"error": "non-finite normalized CBF constraint"},
            )
        cbf_count = int(scaled_rows.shape[0])

        identity = sparse.eye(7, format="csc", dtype=np.float64)
        if cbf_count:
            constraint_matrix = sparse.vstack(
                (sparse.csc_matrix(scaled_rows), identity), format="csc"
            )
            lower = np.concatenate((scaled_lower, lower_velocity))
            upper = np.concatenate(
                (np.full(cbf_count, np.inf, dtype=np.float64), upper_velocity)
            )
        else:
            constraint_matrix = identity
            lower = lower_velocity.copy()
            upper = upper_velocity.copy()

        quadratic = sparse.diags(weights, format="csc")
        with np.errstate(over="ignore", invalid="ignore"):
            linear = -weights * nominal
        if not np.all(np.isfinite(linear)):
            return FilterResult(
                False,
                "invalid_qp_objective",
                None,
                {"error": "non-finite derived linear objective"},
            )
        solver = osqp.OSQP()
        try:
            solver.setup(
                P=quadratic,
                q=linear,
                A=constraint_matrix,
                l=lower,
                u=upper,
                verbose=False,
                eps_abs=self.eps_abs,
                eps_rel=self.eps_rel,
                max_iter=self.max_iter,
                polishing=True,
                warm_starting=True,
                adaptive_rho=True,
            )
            solution = solver.solve()
        except Exception as error:  # OSQP exposes backend-specific exceptions
            return FilterResult(
                False,
                "qp_solver_exception",
                None,
                {"error_type": type(error).__name__, "error": str(error)},
            )

        status = str(getattr(solution.info, "status", "unknown")).lower()
        status_value = int(getattr(solution.info, "status_val", -1))
        diagnostics = {
            "solver": "osqp",
            "status": status,
            "status_value": status_value,
            "iterations": int(getattr(solution.info, "iter", -1)),
            "solve_time_seconds": float(getattr(solution.info, "solve_time", 0.0)),
            "input_constraint_count": int(rows.shape[0]),
            "solved_constraint_count": cbf_count,
            "trivial_zero_constraint_count": int(np.count_nonzero(zero_rows)),
        }
        if status not in ("solved", "solved inaccurate") or solution.x is None:
            return FilterResult(False, "qp_not_solved", None, diagnostics)
        candidate = np.asarray(solution.x, dtype=np.float64)
        if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
            diagnostics["candidate_finite"] = False
            return FilterResult(False, "invalid_qp_solution", None, diagnostics)

        # OSQP may return a solution a few ulps beyond an active velocity
        # bound while remaining inside the registered postcheck tolerance.
        # Canonicalize the command to the actual hard bounds before computing
        # CBF residuals or returning it to the controller.  The raw violation
        # remains audited below; the postcheck therefore still fails closed if
        # it exceeds the registered tolerance.
        raw_candidate = candidate.copy()
        candidate = np.clip(raw_candidate, lower_velocity, upper_velocity)

        if cbf_count:
            normalized_residual = scaled_rows @ candidate - scaled_lower
            minimum_residual = float(np.min(normalized_residual))
            raw_residual = rows[keep] @ candidate - lower_rows[keep]
            minimum_raw_residual = float(np.min(raw_residual))
        else:
            normalized_residual = np.empty(0, dtype=np.float64)
            minimum_residual = None
            raw_residual = np.empty(0, dtype=np.float64)
            minimum_raw_residual = None
        lower_violation = float(
            np.max(np.maximum(lower_velocity - raw_candidate, 0.0))
        )
        upper_violation = float(
            np.max(np.maximum(raw_candidate - upper_velocity, 0.0))
        )
        bound_violation = max(lower_violation, upper_violation)
        diagnostics.update(
            {
                "minimum_normalized_cbf_residual": minimum_residual,
                "minimum_raw_cbf_residual_m2_per_s": minimum_raw_residual,
                "minimum_nonzero_row_scale_m2_per_rad": (
                    float(np.min(row_scales[keep])) if cbf_count else None
                ),
                "maximum_nonzero_row_scale_m2_per_rad": (
                    float(np.max(row_scales[keep])) if cbf_count else None
                ),
                "maximum_velocity_bound_violation_rad_s": bound_violation,
                "nominal_minimum_normalized_cbf_residual": (
                    float(np.min(scaled_rows @ nominal - scaled_lower))
                    if cbf_count
                    else None
                ),
                "nominal_minimum_raw_cbf_residual_m2_per_s": (
                    float(np.min(rows[keep] @ nominal - lower_rows[keep]))
                    if cbf_count
                    else None
                ),
                "correction_l2_rad_s": float(np.linalg.norm(candidate - nominal)),
                "nominal_feasible": bool(
                    (
                        not cbf_count
                        or np.all(
                            scaled_rows @ nominal - scaled_lower
                            >= -self.postcheck_cbf_tolerance
                        )
                    )
                    and np.all(
                        nominal
                        >= lower_velocity - self.postcheck_bound_tolerance
                    )
                    and np.all(
                        nominal
                        <= upper_velocity + self.postcheck_bound_tolerance
                    )
                ),
            }
        )
        if (
            minimum_residual is not None
            and minimum_residual < -self.postcheck_cbf_tolerance
        ) or bound_violation > self.postcheck_bound_tolerance:
            return FilterResult(False, "qp_postcheck_failed", None, diagnostics)
        return FilterResult(True, "solved", candidate, diagnostics)


def solve_reference_cvxpy(
    qdot_nominal: Sequence[float],
    cbf_rows: Any,
    cbf_lower: Sequence[float],
    velocity_lower: Sequence[float],
    velocity_upper: Sequence[float],
    *,
    weight_diagonal: Optional[Sequence[float]] = None,
) -> Any:
    """Declarative reference solve used only by validation fixtures."""

    np, _, _ = _numeric_modules()
    try:
        import cvxpy as cp
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("CVXPY is required for the reference QP") from error
    nominal = _finite_vector(qdot_nominal, 7, "qdot_nominal", np)
    rows = np.asarray(cbf_rows, dtype=np.float64)
    lower_rows = np.asarray(cbf_lower, dtype=np.float64)
    lower_velocity = _finite_vector(velocity_lower, 7, "velocity_lower", np)
    upper_velocity = _finite_vector(velocity_upper, 7, "velocity_upper", np)
    if rows.ndim != 2 or rows.shape[1] != 7 or lower_rows.shape != (rows.shape[0],):
        raise ValueError("reference CBF rows have inconsistent shapes")
    weights = (
        np.ones(7, dtype=np.float64)
        if weight_diagonal is None
        else _finite_vector(weight_diagonal, 7, "weight_diagonal", np)
    )
    variable = cp.Variable(7)
    constraints = [variable >= lower_velocity, variable <= upper_velocity]
    if rows.shape[0]:
        constraints.append(rows @ variable >= lower_rows)
    objective = cp.Minimize(0.5 * cp.sum(cp.multiply(weights, cp.square(variable - nominal))))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, polish=True)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or variable.value is None:
        raise RuntimeError("reference QP was not solved: %s" % problem.status)
    return np.asarray(variable.value, dtype=np.float64)
