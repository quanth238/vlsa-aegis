"""Exact hard multi-constraint QP with explicit timing and postchecks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Optional, Sequence


def _modules() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import osqp
        import scipy.sparse as sparse
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("the multi-link QP requires NumPy, SciPy, and OSQP") from error
    return np, osqp, sparse


@dataclass(frozen=True)
class QpResult:
    valid: bool
    reason: str
    qdot_safe: Optional[Any]
    diagnostics: Mapping[str, Any]


class MultiConstraintQp:
    """Solve min 1/2 (qdot-qdot0)' G (qdot-qdot0), A qdot >= b."""

    def __init__(
        self,
        *,
        eps_abs: float = 1.0e-7,
        eps_rel: float = 1.0e-7,
        max_iter: int = 10000,
        residual_tolerance: float = 5.0e-7,
        bound_tolerance: float = 5.0e-8,
    ) -> None:
        for value in (eps_abs, eps_rel, residual_tolerance, bound_tolerance):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError("QP tolerances must be finite and positive")
        if isinstance(max_iter, bool) or int(max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.max_iter = int(max_iter)
        self.residual_tolerance = float(residual_tolerance)
        self.bound_tolerance = float(bound_tolerance)

    def solve(
        self,
        qdot_nominal: Sequence[float],
        metric: Any,
        rows: Any,
        lower_rows: Sequence[float],
        velocity_lower: Sequence[float],
        velocity_upper: Sequence[float],
    ) -> QpResult:
        np, osqp, sparse = _modules()
        total_started = time.perf_counter_ns()

        def early_timing(
            *, setup_wall_seconds: float = 0.0, solve_wall_seconds: float = 0.0
        ) -> dict[str, float]:
            return {
                "setup_wall_seconds": float(setup_wall_seconds),
                "solve_wall_seconds": float(solve_wall_seconds),
                "osqp_setup_seconds": 0.0,
                "osqp_solve_seconds": 0.0,
                "osqp_run_seconds": 0.0,
                "total_wall_seconds": (
                    time.perf_counter_ns() - total_started
                )
                * 1.0e-9,
            }

        try:
            nominal = np.asarray(qdot_nominal, dtype=np.float64)
            hessian = np.asarray(metric, dtype=np.float64)
            matrix = np.asarray(rows, dtype=np.float64)
            row_lower = np.asarray(lower_rows, dtype=np.float64)
            velocity_min = np.asarray(velocity_lower, dtype=np.float64)
            velocity_max = np.asarray(velocity_upper, dtype=np.float64)
            dimension = int(nominal.shape[0]) if nominal.ndim == 1 else -1
            if dimension < 1:
                raise ValueError("qdot_nominal must be one-dimensional and nonempty")
            if hessian.shape != (dimension, dimension):
                raise ValueError("metric shape differs from qdot dimension")
            if matrix.ndim != 2 or matrix.shape[1] != dimension:
                raise ValueError("constraint rows have inconsistent shape")
            if row_lower.shape != (matrix.shape[0],):
                raise ValueError("constraint lower bounds have inconsistent shape")
            if velocity_min.shape != (dimension,) or velocity_max.shape != (dimension,):
                raise ValueError("velocity bounds have inconsistent shape")
            values = (nominal, hessian, matrix, row_lower, velocity_min, velocity_max)
            if any(not np.all(np.isfinite(value)) for value in values):
                raise ValueError("QP inputs must be finite")
            hessian = 0.5 * (hessian + hessian.T)
            minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hessian)))
            if minimum_eigenvalue <= 0.0:
                raise ValueError("metric must be positive definite")
            if np.any(velocity_min > velocity_max):
                raise ValueError("velocity bounds are contradictory")
        except ValueError as error:
            return QpResult(
                False,
                "invalid_qp_inputs",
                None,
                {"error": str(error), "timing": early_timing()},
            )

        row_scales = np.max(np.abs(matrix), axis=1) if matrix.shape[0] else np.empty(0)
        zero_rows = row_scales == 0.0
        impossible = zero_rows & (row_lower > 0.0)
        if np.any(impossible):
            return QpResult(
                False,
                "uncontrollable_constraint",
                None,
                {
                    "constraint_indexes": np.flatnonzero(impossible).astype(int).tolist(),
                    "timing": early_timing(),
                },
            )
        keep = ~zero_rows
        scaled_rows = matrix[keep] / row_scales[keep, None]
        scaled_lower = row_lower[keep] / row_scales[keep]
        identity = sparse.eye(dimension, format="csc", dtype=np.float64)
        if scaled_rows.shape[0]:
            constraint_matrix = sparse.vstack(
                (sparse.csc_matrix(scaled_rows), identity), format="csc"
            )
            lower = np.concatenate((scaled_lower, velocity_min))
            upper = np.concatenate(
                (np.full(scaled_rows.shape[0], np.inf), velocity_max)
            )
        else:
            constraint_matrix = identity
            lower = velocity_min.copy()
            upper = velocity_max.copy()
        quadratic = sparse.csc_matrix(hessian)
        linear = -hessian @ nominal

        setup_started = time.perf_counter_ns()
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
                polish=True,
                warm_start=True,
                adaptive_rho=True,
            )
        except Exception as error:
            setup_elapsed = (time.perf_counter_ns() - setup_started) * 1.0e-9
            return QpResult(
                False,
                "qp_setup_exception",
                None,
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "timing": early_timing(setup_wall_seconds=setup_elapsed),
                },
            )
        setup_finished = time.perf_counter_ns()
        solve_started = time.perf_counter_ns()
        try:
            solution = solver.solve()
        except Exception as error:
            solve_elapsed = (time.perf_counter_ns() - solve_started) * 1.0e-9
            return QpResult(
                False,
                "qp_solver_exception",
                None,
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "timing": early_timing(
                        setup_wall_seconds=(setup_finished - setup_started) * 1.0e-9,
                        solve_wall_seconds=solve_elapsed,
                    ),
                },
            )
        solve_finished = time.perf_counter_ns()
        status = str(getattr(solution.info, "status", "unknown")).lower()
        diagnostics: dict[str, Any] = {
            "solver": "osqp",
            "status": status,
            "status_value": int(getattr(solution.info, "status_val", -1)),
            "iterations": int(getattr(solution.info, "iter", -1)),
            "input_constraint_count": int(matrix.shape[0]),
            "solved_constraint_count": int(scaled_rows.shape[0]),
            "zero_constraint_count": int(np.count_nonzero(zero_rows)),
            "timing": {
                "setup_wall_seconds": (setup_finished - setup_started) * 1.0e-9,
                "solve_wall_seconds": (solve_finished - solve_started) * 1.0e-9,
                "osqp_setup_seconds": float(getattr(solution.info, "setup_time", 0.0)),
                "osqp_solve_seconds": float(getattr(solution.info, "solve_time", 0.0)),
                "osqp_run_seconds": float(getattr(solution.info, "run_time", 0.0)),
            },
        }
        if status not in ("solved", "solved inaccurate") or solution.x is None:
            diagnostics["timing"]["total_wall_seconds"] = (
                time.perf_counter_ns() - total_started
            ) * 1.0e-9
            return QpResult(False, "qp_not_solved", None, diagnostics)

        raw_candidate = np.asarray(solution.x, dtype=np.float64)
        if raw_candidate.shape != (dimension,) or not np.all(np.isfinite(raw_candidate)):
            diagnostics["timing"]["total_wall_seconds"] = (
                time.perf_counter_ns() - total_started
            ) * 1.0e-9
            return QpResult(False, "invalid_qp_solution", None, diagnostics)
        candidate = np.clip(raw_candidate, velocity_min, velocity_max)
        normalized_residual = scaled_rows @ candidate - scaled_lower
        raw_residual = matrix[keep] @ candidate - row_lower[keep]
        minimum_normalized = (
            None if not normalized_residual.size else float(np.min(normalized_residual))
        )
        minimum_raw = None if not raw_residual.size else float(np.min(raw_residual))
        raw_bound_violation = max(
            float(np.max(np.maximum(velocity_min - raw_candidate, 0.0))),
            float(np.max(np.maximum(raw_candidate - velocity_max, 0.0))),
        )
        delta = candidate - nominal
        diagnostics.update(
            {
                "minimum_normalized_residual": minimum_normalized,
                "minimum_raw_residual_m_per_s": minimum_raw,
                "maximum_velocity_bound_violation_rad_s": raw_bound_violation,
                "correction_l2_rad_s": float(np.linalg.norm(delta)),
                "objective": float(0.5 * delta @ hessian @ delta),
                "nominal_minimum_normalized_residual": (
                    None
                    if not normalized_residual.size
                    else float(np.min(scaled_rows @ nominal - scaled_lower))
                ),
                "nominal_minimum_raw_residual_m_per_s": (
                    None
                    if not raw_residual.size
                    else float(np.min(matrix[keep] @ nominal - row_lower[keep]))
                ),
            }
        )
        diagnostics["timing"]["total_wall_seconds"] = (
            time.perf_counter_ns() - total_started
        ) * 1.0e-9
        if (
            minimum_normalized is not None
            and minimum_normalized < -self.residual_tolerance
        ) or raw_bound_violation > self.bound_tolerance:
            return QpResult(False, "qp_postcheck_failed", None, diagnostics)
        return QpResult(True, "solved", candidate, diagnostics)
