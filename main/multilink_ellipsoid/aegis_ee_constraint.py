"""Released AEGIS end-effector-only constraint for paired controller tests.

This module deliberately constructs no link-5/link-6/link-7 geometry.  The
Cartesian arm calls the released Table-1 six-variable translational CBF-QP.
The Direct-Joint arm calls that identical Cartesian QP, then lifts only its
translation correction to the nominal joint target with the live EE
Jacobian.  Raw MuJoCo contacts remain the simulator safety authority.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - H100 dependency
        raise RuntimeError("AEGIS-EE filtering requires NumPy") from error
    return np


class ReleasedAegisEEFilter:
    """Stateful wrapper around the released AEGIS virtual-direction QP."""

    def __init__(
        self,
        *,
        runtime: Mapping[str, Any],
        config: Mapping[str, Any],
        initial_proxy: Mapping[str, Any],
    ) -> None:
        np = _numpy()

        guidance = config["collision_guidance"]
        if guidance.get("l5_l6_l7_constraints_enabled") is not False:
            raise ValueError("AEGIS-EE filter requires L5--L7 constraints disabled")
        released = guidance["released_aegis"]
        obstacle = released["obstacle_mvee"]
        proxy = {
            "p1": np.asarray(initial_proxy["p1"], dtype=np.float64).copy(),
            "R1": np.asarray(initial_proxy["R1"], dtype=np.float64).copy(),
            "quaternion": np.asarray(
                initial_proxy["quaternion"], dtype=np.float64
            ).copy(),
        }
        if (
            proxy["p1"].shape != (3,)
            or proxy["R1"].shape != (3, 3)
            or proxy["quaternion"].shape != (4,)
            or not all(np.all(np.isfinite(value)) for value in proxy.values())
        ):
            raise ValueError("released initial end-effector proxy differs")
        p2 = np.asarray(obstacle["center_m"], dtype=np.float64)
        z_fixed = p2 - np.asarray(proxy["p1"], dtype=np.float64)
        norm = float(np.linalg.norm(z_fixed))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError("initial released AEGIS virtual direction is degenerate")
        self.runtime = runtime
        self.config = config
        self.proxy = proxy
        self.initial_proxy = {
            key: value.copy() for key, value in proxy.items()
        }
        self.q1_diag = np.asarray(
            released["end_effector_proxy"]["semiaxes_m"], dtype=np.float64
        )
        self.geometry = {
            "enabled": True,
            "p2": p2,
            "R2": np.asarray(obstacle["rotation"], dtype=np.float64),
            "Q2_diag": np.asarray(obstacle["semiaxes_m"], dtype=np.float64),
            "z_fixed": z_fixed / norm,
        }
        self.translation_scale_m = float(
            guidance["joint_adapter"]["translation_scale_m_per_action_unit"]
        )
        self.jacobian_damping = float(
            guidance["joint_adapter"]["damped_jacobian_lambda"]
        )
        self.qp_times_s: list[float] = []

    def _solve(
        self,
        observation: Mapping[str, Any],
        nominal_action: Sequence[float],
    ) -> tuple[Any, dict[str, Any]]:
        np = _numpy()
        from main.evaluate_safelibero_aegis import _aegis_action

        nominal = np.asarray(nominal_action, dtype=np.float64)
        if nominal.shape != (7,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal AEGIS-EE action must be finite length seven")
        translational = np.zeros(7, dtype=np.float64)
        translational[:3] = nominal[:3]
        translational[6] = nominal[6]
        started = time.perf_counter_ns()
        executed, diagnostics = _aegis_action(
            self.runtime,
            nominal_translational=translational,
            proxy=self.proxy,
            geometry=self.geometry,
            q1_diag=self.q1_diag,
            diagnostics_enabled=True,
        )
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        self.qp_times_s.append(elapsed)
        safe = np.asarray(executed, dtype=np.float64)
        record = {
            "constraint_count": 1,
            "protected_geometry": "released_aegis_end_effector_proxy_only",
            "l5_l6_l7_constraints_enabled": False,
            "nominal_translational_action": translational.tolist(),
            "filtered_translational_action": safe.tolist(),
            "translation_correction_l2_action_units": float(
                np.linalg.norm(safe[:3] - translational[:3])
            ),
            "qp_wall_seconds": elapsed,
            "qp": diagnostics,
        }
        return safe, record

    def observe_post_step(self, observation: Mapping[str, Any]) -> None:
        """Advance the released cached EE proxy after an executed env.step."""

        from main.evaluate_safelibero_aegis import _eef_proxy

        self.proxy = _eef_proxy(self.runtime, observation)

    def filter_cartesian(
        self,
        observation: Mapping[str, Any],
        nominal_action: Sequence[float],
    ) -> tuple[Any, dict[str, Any]]:
        safe, record = self._solve(observation, nominal_action)
        record["application"] = "released_cartesian_table1_translational_filter"
        return safe, record

    def filter_joint_target(
        self,
        observation: Mapping[str, Any],
        *,
        nominal_pose_action: Sequence[float],
        nominal_joint_target: Sequence[float],
        current_joint_position: Sequence[float],
        end_effector_jacobian: Any,
        lower: Sequence[float],
        upper: Sequence[float],
        gripper: float,
    ) -> tuple[Any, dict[str, Any]]:
        np = _numpy()
        from main.evaluate_safelibero_aegis import array_sha256
        from main.multilink_ellipsoid.embodisteer_joint_baseline import (
            damped_pseudoinverse,
        )

        pose = np.asarray(nominal_pose_action, dtype=np.float64)
        target = np.asarray(nominal_joint_target, dtype=np.float64)
        current = np.asarray(current_joint_position, dtype=np.float64)
        jacobian = np.asarray(end_effector_jacobian, dtype=np.float64)
        low = np.asarray(lower, dtype=np.float64)
        high = np.asarray(upper, dtype=np.float64)
        if (
            pose.shape != (6,)
            or target.shape != (7,)
            or current.shape != (7,)
            or jacobian.shape != (6, 7)
            or low.shape != (7,)
            or high.shape != (7,)
            or not all(
                np.all(np.isfinite(value))
                for value in (pose, target, current, jacobian, low, high)
            )
        ):
            raise ValueError("Direct-Joint AEGIS-EE adapter input differs")
        nominal = np.concatenate((pose, [float(gripper)]))
        filtered, record = self._solve(observation, nominal)
        translation_correction_m = self.translation_scale_m * (
            filtered[:3] - pose[:3]
        )
        correction_twist = np.concatenate(
            (translation_correction_m, np.zeros(3, dtype=np.float64))
        )
        joint_correction = (
            damped_pseudoinverse(jacobian, self.jacobian_damping)
            @ correction_twist
        )
        safe_unclipped = target + joint_correction
        safe_target = np.clip(safe_unclipped, low, high)
        record.update(
            {
                "application": "posthoc_released_aegis_translation_correction_lifted_to_direct_joint_target",
                "nominal_pose_action": pose.tolist(),
                "translation_correction_m": translation_correction_m.tolist(),
                "joint_correction_rad": joint_correction.tolist(),
                "joint_correction_l2_rad": float(np.linalg.norm(joint_correction)),
                "nominal_joint_target_rad": target.tolist(),
                "safe_joint_target_before_limits_rad": safe_unclipped.tolist(),
                "safe_joint_target_rad": safe_target.tolist(),
                "joint_limit_clipped_count": int(
                    np.count_nonzero(safe_unclipped != safe_target)
                ),
                "live_jacobian_shape": [6, 7],
                "live_jacobian_sha256": array_sha256(jacobian),
            }
        )
        return safe_target, record

    def geometry_record(self) -> dict[str, Any]:
        return {
            "constraint_count": 1,
            "protected_body_names": ["robot0_end_effector"],
            "end_effector_proxy": {
                "center": "released_cached_pose_pre_settle_for_first_qp_then_post_step_updates",
                "semiaxes_m": self.q1_diag.tolist(),
                "initial_pre_settle_center_m": self.initial_proxy["p1"].tolist(),
                "initial_pre_settle_rotation": self.initial_proxy["R1"].tolist(),
            },
            "obstacle_mvee": {
                "center_m": self.geometry["p2"].tolist(),
                "rotation": self.geometry["R2"].tolist(),
                "semiaxes_m": self.geometry["Q2_diag"].tolist(),
            },
            "l5_l6_l7_ellipsoids_constructed": False,
            "l5_l6_l7_constraints_built": False,
        }

    def timing_summary(self) -> dict[str, Any]:
        np = _numpy()
        values = np.asarray(self.qp_times_s, dtype=np.float64)
        if values.size == 0:
            raise ValueError("AEGIS-EE timing summary requires at least one QP")
        return {
            "qp_count": int(values.size),
            "mean_qp_wall_ms": float(1000.0 * np.mean(values)),
            "median_qp_wall_ms": float(1000.0 * np.median(values)),
            "maximum_qp_wall_ms": float(1000.0 * np.max(values)),
        }
