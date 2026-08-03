"""Identity-preserving Poisson-CBF reference governor for native OSC.

The released SafeLIBERO ``OSC_POSE`` controller remains the executor.  This
module only changes its six pose-reference channels when the current nominal
command violates a link-aware CBF constraint.  A nominally feasible command is
returned byte-for-byte, so enabling the filter cannot silently replace the
baseline controller as the earlier direct joint-velocity pilot did.

The governor is an empirical task-space realization of the paper's kinematic
filter.  It maps OSC pose-reference rates through a damped end-effector
Jacobian, retains measured motion outside that task-space map, and enforces
the full-body Poisson rows in the resulting affine subspace.
Simulator contact, motion, CAR, and task success remain the outcome authority;
this mapping alone is not a formal joint-velocity tracking guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence


def _numeric_modules() -> Any:
    try:
        import numpy as np
        import osqp
        from scipy import sparse
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError(
            "NumPy, SciPy, and OSQP are required for the OSC reference governor"
        ) from error
    return np, osqp, sparse


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("NumPy is required for the OSC reference governor") from error
    return np


def _finite_vector(value: Any, length: int, label: str) -> Any:
    np = _numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError("%s must be a finite %d-vector" % (label, length))
    return np.array(array, dtype=np.float64, copy=True)


def damped_twist_to_joint_map(eef_jacobian: Any, damping: float) -> Any:
    """Return the 7x6 damped least-squares map from twist to joint rate."""

    np = _numpy()
    jacobian = np.asarray(eef_jacobian, dtype=np.float64)
    damping = float(damping)
    if jacobian.shape != (6, 7) or not np.all(np.isfinite(jacobian)):
        raise ValueError("eef_jacobian must be a finite 6x7 matrix")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive")
    regularized = jacobian @ jacobian.T + damping * damping * np.eye(6)
    mapping = jacobian.T @ np.linalg.solve(regularized, np.eye(6))
    if mapping.shape != (7, 6) or not np.all(np.isfinite(mapping)):
        raise RuntimeError("damped twist-to-joint map is invalid")
    return mapping


def osc_pose_velocity_from_action(
    action: Any,
    *,
    output_scale: Any,
    control_dt_seconds: float,
) -> Any:
    """Convert normalized OSC pose channels to a six-dimensional pose rate."""

    pose_action = _finite_vector(action, 6, "OSC pose action")
    scale = _finite_vector(output_scale, 6, "OSC output scale")
    dt = float(control_dt_seconds)
    if np_any_nonpositive(scale) or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("OSC scale and control interval must be positive")
    return pose_action * scale / dt


def np_any_nonpositive(value: Any) -> bool:
    """Dependency-light helper kept separate for structural tests."""

    np = _numpy()
    return bool(np.any(np.asarray(value, dtype=np.float64) <= 0.0))


def osc_output_scale(controller: Any) -> Any:
    """Read the live Robosuite symmetric six-channel OSC output scale."""

    np = _numpy()
    output_min = np.asarray(getattr(controller, "output_min", None), dtype=np.float64)
    output_max = np.asarray(getattr(controller, "output_max", None), dtype=np.float64)
    if (
        output_min.shape != (6,)
        or output_max.shape != (6,)
        or not np.all(np.isfinite(output_min))
        or not np.all(np.isfinite(output_max))
        or np.any(output_max <= 0.0)
        or not np.array_equal(output_min, -output_max)
    ):
        raise ValueError("live OSC output limits must be finite, positive, and symmetric")
    return output_max.copy()


@dataclass(frozen=True)
class OscGovernorResult:
    valid: bool
    reason: str
    action: Optional[Any]
    nominal_qdot_rad_per_s: Any
    safe_qdot_rad_per_s: Optional[Any]
    diagnostics: Any


class OscPoseReferenceGovernor:
    """Hard CBF-QP in the pose-reference subspace of native ``OSC_POSE``."""

    def __init__(
        self,
        *,
        damping: float = 0.05,
        eps_abs: float = 1.0e-7,
        eps_rel: float = 1.0e-7,
        max_iter: int = 10000,
        postcheck_tolerance: float = 5.0e-7,
        material_action_correction: float = 1.0e-6,
    ) -> None:
        values = (damping, eps_abs, eps_rel, postcheck_tolerance)
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
            raise ValueError("governor numeric tolerances must be finite and positive")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if (
            not math.isfinite(float(material_action_correction))
            or float(material_action_correction) <= 0.0
        ):
            raise ValueError("material_action_correction must be finite and positive")
        self.damping = float(damping)
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.max_iter = int(max_iter)
        self.postcheck_tolerance = float(postcheck_tolerance)
        self.material_action_correction = float(material_action_correction)

    def filter_action(
        self,
        source_action: Sequence[float],
        *,
        eef_jacobian: Any,
        measured_arm_qvel_rad_per_s: Sequence[float],
        cbf_rows_qdot: Any,
        cbf_lower_m2_per_s: Any,
        qdot_lower_rad_per_s: Optional[Sequence[float]],
        qdot_upper_rad_per_s: Optional[Sequence[float]],
        osc_output_scale: Sequence[float],
        control_dt_seconds: float,
        pose_weight_diagonal: Optional[Sequence[float]] = None,
    ) -> OscGovernorResult:
        """Filter one native OSC action without a stop or zero fallback."""

        np, osqp, sparse = _numeric_modules()
        source = _finite_vector(source_action, 7, "source_action")
        if np.any(source[:6] < -1.0) or np.any(source[:6] > 1.0):
            raise ValueError("source OSC pose action must already lie within [-1, 1]")
        scale = _finite_vector(osc_output_scale, 6, "osc_output_scale")
        dt = float(control_dt_seconds)
        if np.any(scale <= 0.0) or not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("OSC scale and control interval must be positive")
        mapping = damped_twist_to_joint_map(eef_jacobian, self.damping)
        measured_qdot = _finite_vector(
            measured_arm_qvel_rad_per_s,
            7,
            "measured_arm_qvel_rad_per_s",
        )
        measured_eef_twist = (
            np.asarray(eef_jacobian, dtype=np.float64) @ measured_qdot
        )
        qdot_affine_offset = measured_qdot - mapping @ measured_eef_twist
        nominal_twist = source[:6] * scale / dt
        nominal_qdot = qdot_affine_offset + mapping @ nominal_twist
        rows = np.asarray(cbf_rows_qdot, dtype=np.float64)
        lower = np.asarray(cbf_lower_m2_per_s, dtype=np.float64)
        if (qdot_lower_rad_per_s is None) != (qdot_upper_rad_per_s is None):
            raise ValueError("joint-velocity bounds must be both present or both absent")
        if qdot_lower_rad_per_s is None:
            # Native OSC does not expose a direct joint-velocity command
            # limit. The exact image of its bounded pose-reference box keeps
            # this constraint redundant, avoiding unrelated changes to the
            # baseline while still bounding the QP numerically.
            maximum_realisable_qdot = np.abs(mapping) @ (scale / dt)
            qdot_lower = qdot_affine_offset - maximum_realisable_qdot
            qdot_upper = qdot_affine_offset + maximum_realisable_qdot
            qdot_bounds_source = (
                "measured_velocity_affine_offset_plus_exact_osc_pose_envelope"
            )
        else:
            qdot_lower = _finite_vector(
                qdot_lower_rad_per_s, 7, "qdot_lower"
            )
            qdot_upper = _finite_vector(
                qdot_upper_rad_per_s, 7, "qdot_upper"
            )
            qdot_bounds_source = "caller_registered"
        if rows.ndim != 2 or rows.shape[1] != 7:
            raise ValueError("cbf_rows_qdot must have shape (N, 7)")
        if lower.shape != (rows.shape[0],):
            raise ValueError("cbf_lower_m2_per_s must have shape (N,)")
        if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(lower)):
            raise ValueError("CBF rows and bounds must be finite")
        if np.any(qdot_lower > qdot_upper):
            raise ValueError("joint-velocity bounds are contradictory")

        nominal_cbf_residual = rows @ nominal_qdot - lower
        nominal_bound_violation = max(
            float(np.max(np.maximum(qdot_lower - nominal_qdot, 0.0))),
            float(np.max(np.maximum(nominal_qdot - qdot_upper, 0.0))),
        )
        nominal_minimum = (
            float(np.min(nominal_cbf_residual)) if rows.shape[0] else None
        )
        nominal_feasible = bool(
            (not rows.shape[0] or nominal_minimum >= -self.postcheck_tolerance)
            and nominal_bound_violation <= self.postcheck_tolerance
        )
        common = {
            "nominal_feasible": nominal_feasible,
            "nominal_minimum_cbf_residual_m2_per_s": nominal_minimum,
            "nominal_joint_bound_violation_rad_per_s": nominal_bound_violation,
            "nominal_pose_velocity": nominal_twist.tolist(),
            "nominal_qdot_rad_per_s": nominal_qdot.tolist(),
            "measured_arm_qvel_rad_per_s": measured_qdot.tolist(),
            "measured_eef_twist": measured_eef_twist.tolist(),
            "qdot_affine_offset_rad_per_s": qdot_affine_offset.tolist(),
            "damping": self.damping,
            "control_dt_seconds": dt,
            "osc_output_scale": scale.tolist(),
            "sample_count": int(rows.shape[0]),
            "qdot_bounds_source": qdot_bounds_source,
            "qdot_lower_rad_per_s": qdot_lower.tolist(),
            "qdot_upper_rad_per_s": qdot_upper.tolist(),
        }
        if nominal_feasible:
            # Do not reconstruct or round the baseline command. This exact
            # branch is the scientific identity condition of the integration.
            returned = source.copy()
            return OscGovernorResult(
                valid=True,
                reason="nominal_safe_exact_passthrough",
                action=returned,
                nominal_qdot_rad_per_s=nominal_qdot,
                safe_qdot_rad_per_s=nominal_qdot.copy(),
                diagnostics={
                    **common,
                    "solver_attempted": False,
                    "action_byte_identical_to_source": bool(
                        returned.tobytes(order="C") == source.tobytes(order="C")
                    ),
                    "correction_l2_normalized_action": 0.0,
                    "material_correction": False,
                },
            )

        pose_map = rows @ mapping
        pose_lower = lower - rows @ qdot_affine_offset
        row_scale = (
            np.max(np.abs(pose_map), axis=1)
            if pose_map.shape[0]
            else np.empty(0, dtype=np.float64)
        )
        zero_rows = row_scale == 0.0
        impossible = zero_rows & (pose_lower > 0.0)
        if np.any(impossible):
            return OscGovernorResult(
                valid=False,
                reason="cbf_not_controllable_in_osc_pose_subspace",
                action=None,
                nominal_qdot_rad_per_s=nominal_qdot,
                safe_qdot_rad_per_s=None,
                diagnostics={
                    **common,
                    "solver_attempted": False,
                    "uncontrollable_constraint_indexes": (
                        np.flatnonzero(impossible).astype(int).tolist()
                    ),
                },
            )
        keep = ~zero_rows
        scaled_pose_rows = pose_map[keep] / row_scale[keep, None]
        scaled_pose_lower = pose_lower[keep] / row_scale[keep]
        twist_lower = -scale / dt
        twist_upper = scale / dt
        identity = np.eye(6, dtype=np.float64)
        constraint = np.vstack((scaled_pose_rows, mapping, identity))
        constraint_lower = np.concatenate(
            (
                scaled_pose_lower,
                qdot_lower - qdot_affine_offset,
                twist_lower,
            )
        )
        constraint_upper = np.concatenate(
            (
                np.full(scaled_pose_rows.shape[0], np.inf, dtype=np.float64),
                qdot_upper - qdot_affine_offset,
                twist_upper,
            )
        )
        weights = (
            np.square(dt / scale)
            if pose_weight_diagonal is None
            else _finite_vector(pose_weight_diagonal, 6, "pose_weight_diagonal")
        )
        if np.any(weights <= 0.0):
            raise ValueError("pose weights must be positive")
        solver = osqp.OSQP()
        try:
            solver.setup(
                P=sparse.diags(weights, format="csc"),
                q=-weights * nominal_twist,
                A=sparse.csc_matrix(constraint),
                l=constraint_lower,
                u=constraint_upper,
                verbose=False,
                eps_abs=self.eps_abs,
                eps_rel=self.eps_rel,
                max_iter=self.max_iter,
                polishing=True,
                warm_starting=True,
                adaptive_rho=True,
            )
            solved = solver.solve()
        except Exception as error:
            return OscGovernorResult(
                valid=False,
                reason="qp_solver_exception",
                action=None,
                nominal_qdot_rad_per_s=nominal_qdot,
                safe_qdot_rad_per_s=None,
                diagnostics={
                    **common,
                    "solver_attempted": True,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        status = str(getattr(solved.info, "status", "unknown")).lower()
        if status not in ("solved", "solved inaccurate") or solved.x is None:
            return OscGovernorResult(
                valid=False,
                reason="qp_not_solved",
                action=None,
                nominal_qdot_rad_per_s=nominal_qdot,
                safe_qdot_rad_per_s=None,
                diagnostics={
                    **common,
                    "solver_attempted": True,
                    "solver_status": status,
                    "solver_iterations": int(getattr(solved.info, "iter", -1)),
                },
            )
        raw_twist = np.asarray(solved.x, dtype=np.float64)
        if raw_twist.shape != (6,) or not np.all(np.isfinite(raw_twist)):
            return OscGovernorResult(
                valid=False,
                reason="invalid_qp_solution",
                action=None,
                nominal_qdot_rad_per_s=nominal_qdot,
                safe_qdot_rad_per_s=None,
                diagnostics={**common, "solver_attempted": True},
            )
        safe_twist = np.clip(raw_twist, twist_lower, twist_upper)
        safe_qdot = qdot_affine_offset + mapping @ safe_twist
        safe_cbf_residual = rows @ safe_qdot - lower
        minimum_safe_residual = (
            float(np.min(safe_cbf_residual)) if rows.shape[0] else None
        )
        bound_violation = max(
            float(np.max(np.maximum(qdot_lower - safe_qdot, 0.0))),
            float(np.max(np.maximum(safe_qdot - qdot_upper, 0.0))),
        )
        if (
            (minimum_safe_residual is not None and minimum_safe_residual < -self.postcheck_tolerance)
            or bound_violation > self.postcheck_tolerance
        ):
            return OscGovernorResult(
                valid=False,
                reason="qp_postcheck_failed",
                action=None,
                nominal_qdot_rad_per_s=nominal_qdot,
                safe_qdot_rad_per_s=None,
                diagnostics={
                    **common,
                    "solver_attempted": True,
                    "solver_status": status,
                    "minimum_safe_cbf_residual_m2_per_s": minimum_safe_residual,
                    "safe_joint_bound_violation_rad_per_s": bound_violation,
                },
            )
        safe_pose_action = np.clip(safe_twist * dt / scale, -1.0, 1.0)
        returned = np.concatenate((safe_pose_action, source[6:7]))
        correction = returned[:6] - source[:6]
        correction_norm = float(np.linalg.norm(correction))
        nominal_translation = nominal_twist[:3]
        safe_translation = safe_twist[:3]
        nominal_translation_norm_sq = float(nominal_translation @ nominal_translation)
        progress_ratio = (
            None
            if nominal_translation_norm_sq == 0.0
            else float(safe_translation @ nominal_translation / nominal_translation_norm_sq)
        )
        return OscGovernorResult(
            valid=True,
            reason="solved",
            action=returned,
            nominal_qdot_rad_per_s=nominal_qdot,
            safe_qdot_rad_per_s=safe_qdot,
            diagnostics={
                **common,
                "solver_attempted": True,
                "solver_status": status,
                "solver_iterations": int(getattr(solved.info, "iter", -1)),
                "solve_time_seconds": float(getattr(solved.info, "solve_time", 0.0)),
                "safe_pose_velocity": safe_twist.tolist(),
                "safe_qdot_rad_per_s": safe_qdot.tolist(),
                "minimum_safe_cbf_residual_m2_per_s": minimum_safe_residual,
                "safe_joint_bound_violation_rad_per_s": bound_violation,
                "action_byte_identical_to_source": bool(
                    returned.tobytes(order="C") == source.tobytes(order="C")
                ),
                "correction_l2_normalized_action": correction_norm,
                "material_correction": bool(
                    correction_norm >= self.material_action_correction
                ),
                "translation_progress_ratio_along_nominal": progress_ratio,
                "gripper_command_byte_preserved": bool(
                    returned[6:7].tobytes(order="C")
                    == source[6:7].tobytes(order="C")
                ),
            },
        )


__all__ = [
    "OscGovernorResult",
    "OscPoseReferenceGovernor",
    "damped_twist_to_joint_map",
    "osc_output_scale",
    "osc_pose_velocity_from_action",
]
