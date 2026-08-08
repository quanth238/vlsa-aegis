"""Live Cartesian-policy to joint-velocity distal safety demonstration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .barrier import build_pair_constraint
from .joint_velocity import _one_step_joint_bounds
from .qp import MultiConstraintQp
from .shadow import (
    DISTAL_SLABBED_SCHEMA,
    MultilinkEllipsoidShadow,
    _arm_dof_indices,
    _eef_jacobian,
    _geom_jacobians,
    _released_aegis_end_effector_ellipsoid,
)


DEMO_SCHEMA = "vlsa_distal_joint_velocity_demo.v1"
STEP_SCHEMA = "vlsa_distal_joint_velocity_demo_step.v1"


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("joint-velocity demonstration requires NumPy") from error
    return np


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_demo_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("distal joint-velocity demo config is invalid JSON") from error
    required = {
        "action_protocol",
        "arms",
        "case_ids",
        "claim_scope",
        "controller",
        "end_effector_geometry",
        "nominal_policy",
        "obstacle_geometry",
        "optimizer",
        "pairing",
        "protocol_id",
        "robot_geometry",
        "schema_version",
        "success_definition",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("distal joint-velocity demo config keys differ")
    if config["schema_version"] != DEMO_SCHEMA:
        raise ValueError("distal joint-velocity demo schema differs")
    if config["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("distal joint-velocity demo case differs")
    if config["arms"] != [
        "joint_velocity_live_base",
        "joint_velocity_live_l5_l7_ee",
    ]:
        raise ValueError("distal joint-velocity demo arms differ")
    protocol = config["action_protocol"]
    if protocol != {
        "control_frequency_hz": 20,
        "execute_actions_per_query": 5,
        "max_actions": 300,
        "model_action_horizon": 10,
        "rotation_execution": "zero_translational_table1_protocol",
        "translation_velocity_m_per_s_per_action_unit": 0.2,
    }:
        raise ValueError("distal joint-velocity action protocol differs")
    controller = config["controller"]
    if controller.get("type") != "JOINT_VELOCITY" or any(
        float(controller[key]) != expected
        for key, expected in (
            ("input_min", -1.0),
            ("input_max", 1.0),
            ("output_min", -1.0),
            ("output_max", 1.0),
        )
    ):
        raise ValueError("distal joint-velocity controller differs")
    geometry = config["robot_geometry"]
    if geometry != {
        "source": "compiled_mujoco_collision_mesh_vertices",
        "fit": "convex_hull_axis_slab_khachiyan_mvee_exact_inflation",
        "partition_axis": "dominant_pca_axis",
        "slab_partition": "uniform_projection_span_contiguous_no_gaps",
        "clipped_polytope_vertices": (
            "original_hull_vertices_plus_hull_edge_plane_intersections"
        ),
        "part_counts": {
            "robot0_link5": 3,
            "robot0_link6": 2,
            "robot0_link7": 2,
        },
        "relative_numerical_padding": 1.0e-9,
        "khachiyan_tolerance": 1.0e-4,
        "khachiyan_max_iterations": 20000,
    }:
        raise ValueError("distal joint-velocity robot geometry differs")
    if config["end_effector_geometry"] != {
        "center_and_orientation": (
            "authoritative_robot0_grip_site_pose_plus_released_"
            "minus_0.08m_local_z_offset"
        ),
        "semiaxes_m": [0.06, 0.12, 0.11],
        "source": "released_aegis_end_effector_proxy",
    }:
        raise ValueError("released AEGIS EE proxy differs")
    optimizer = config["optimizer"]
    expected_optimizer_keys = {
        "alpha_s_inv",
        "bound_tolerance_rad_s",
        "cartesian_tracking_weight_diagonal",
        "eps_abs",
        "eps_rel",
        "joint_position_margin_rad",
        "joint_regularization",
        "joint_velocity_limit_rad_s",
        "material_correction_tolerance_rad_s",
        "max_iter",
        "optimizer_clearance_m",
        "residual_tolerance",
    }
    if not isinstance(optimizer, dict) or set(optimizer) != expected_optimizer_keys:
        raise ValueError("distal joint-velocity optimizer differs")
    weights = optimizer["cartesian_tracking_weight_diagonal"]
    if not isinstance(weights, list) or len(weights) != 6 or any(
        not math.isfinite(float(value)) or float(value) <= 0.0 for value in weights
    ):
        raise ValueError("Cartesian tracking weights differ")
    for key in expected_optimizer_keys - {
        "cartesian_tracking_weight_diagonal",
        "max_iter",
        "optimizer_clearance_m",
    }:
        if not math.isfinite(float(optimizer[key])) or float(optimizer[key]) <= 0.0:
            raise ValueError("optimizer.%s must be finite and positive" % key)
    if int(optimizer["max_iter"]) < 1:
        raise ValueError("optimizer.max_iter must be positive")
    clearance = float(optimizer["optimizer_clearance_m"])
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("optimizer clearance must be finite and nonnegative")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
    return output


def _shadow_config(config: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = config["optimizer"]
    return {
        "schema_version": DISTAL_SLABBED_SCHEMA,
        "protocol_id": "%s-geometry" % config["protocol_id"],
        "case_ids": list(config["case_ids"]),
        "control_effect": "read_only_no_executed_action_change",
        "obstacle_geometry": "released_aegis_frozen_perception_mvee",
        "protected_body_names": [
            "robot0_link5",
            "robot0_link6",
            "robot0_link7",
        ],
        "robot_geometry": dict(config["robot_geometry"]),
        "end_effector_geometry": dict(config["end_effector_geometry"]),
        "optimizer": {
            "alpha_s_inv": float(optimizer["alpha_s_inv"]),
            "optimizer_clearance_m": float(optimizer["optimizer_clearance_m"]),
            "joint_velocity_limit_rad_s": float(
                optimizer["joint_velocity_limit_rad_s"]
            ),
            "resolved_rate_damping": math.sqrt(
                float(optimizer["joint_regularization"])
            ),
            "eef_preservation_weight": 1.0,
            "eef_weight_diagonal": list(
                optimizer["cartesian_tracking_weight_diagonal"]
            ),
            "eps_abs": float(optimizer["eps_abs"]),
            "eps_rel": float(optimizer["eps_rel"]),
            "max_iter": int(optimizer["max_iter"]),
            "residual_tolerance": float(optimizer["residual_tolerance"]),
            "bound_tolerance_rad_s": float(optimizer["bound_tolerance_rad_s"]),
        },
        "simulator_verification": {
            "D_sim": (
                "raw_mujoco_nonpositive_contact_distance_and_active_obstacle_"
                "displacement"
            ),
            "distinct_from_D_opt": True,
        },
        "claim_scope": config["claim_scope"],
    }


class DistalJointVelocityDemoFilter:
    """Track a Cartesian intention directly in bounded joint velocity."""

    def __init__(
        self,
        config: Mapping[str, Any],
        geometry: MultilinkEllipsoidShadow,
    ) -> None:
        if config.get("schema_version") != DEMO_SCHEMA:
            raise ValueError("distal joint-velocity config was not validated")
        self.config = dict(config)
        self.geometry = geometry
        optimizer = config["optimizer"]
        self.qp = MultiConstraintQp(
            eps_abs=float(optimizer["eps_abs"]),
            eps_rel=float(optimizer["eps_rel"]),
            max_iter=int(optimizer["max_iter"]),
            residual_tolerance=float(optimizer["residual_tolerance"]),
            bound_tolerance=float(optimizer["bound_tolerance_rad_s"]),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DistalJointVelocityDemoFilter":
        obstacle = config["obstacle_geometry"]
        carrier = MultilinkEllipsoidShadow.from_aegis_geometry(
            _shadow_config(config),
            {
                "p2": obstacle["center_m"],
                "R2": obstacle["rotation"],
                "Q2_diag": obstacle["semiaxes_m"],
                "record": {"label": "frozen_primary_obstacle"},
            },
        )
        return cls(config, carrier)

    def geometry_record(self, env: Any) -> dict[str, Any]:
        return self.geometry.geometry_record(env)

    def solve(
        self,
        env: Any,
        cartesian_action: Sequence[float],
        *,
        constraints_enabled: bool,
        step: int,
    ) -> tuple[list[float] | None, dict[str, Any]]:
        np = _numpy()
        total_started = time.perf_counter_ns()
        action = np.asarray(cartesian_action, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("live pi0.5 action must be finite length seven")
        optimizer = self.config["optimizer"]
        arm_dofs = _arm_dof_indices(env)
        eef_jacobian = _eef_jacobian(env, arm_dofs)
        desired_twist = np.concatenate(
            (
                float(
                    self.config["action_protocol"][
                        "translation_velocity_m_per_s_per_action_unit"
                    ]
                )
                * np.clip(action[:3], -1.0, 1.0),
                np.zeros(3, dtype=np.float64),
            )
        )
        weights = np.diag(
            np.asarray(
                optimizer["cartesian_tracking_weight_diagonal"], dtype=np.float64
            )
        )
        metric = (
            eef_jacobian.T @ weights @ eef_jacobian
            + float(optimizer["joint_regularization"]) * np.eye(7)
        )
        nominal = np.linalg.solve(
            metric, eef_jacobian.T @ weights @ desired_twist
        )
        lower, upper, joint_bounds = _one_step_joint_bounds(
            env,
            arm_dofs,
            control_period_seconds=1.0
            / float(self.config["action_protocol"]["control_frequency_hz"]),
            joint_velocity_limit=float(optimizer["joint_velocity_limit_rad_s"]),
            joint_position_margin=float(optimizer["joint_position_margin_rad"]),
        )
        constraints = []
        geometry_started = time.perf_counter_ns()
        if constraints_enabled:
            ellipsoids = self.geometry._slabbed_links(env)
            ellipsoids.append(_released_aegis_end_effector_ellipsoid(env))
            for ellipsoid in ellipsoids:
                jac_position, jac_rotation = _geom_jacobians(
                    env, ellipsoid, arm_dofs
                )
                constraints.append(
                    build_pair_constraint(
                        ellipsoid,
                        self.geometry.obstacle,
                        jac_position,
                        jac_rotation,
                        alpha=float(optimizer["alpha_s_inv"]),
                        optimizer_clearance_m=float(
                            optimizer["optimizer_clearance_m"]
                        ),
                    )
                )
            if len(constraints) != 8:
                raise ValueError("active distal demo requires exactly eight rows")
        geometry_finished = time.perf_counter_ns()
        rows = (
            np.stack([item.row for item in constraints], axis=0)
            if constraints
            else np.empty((0, 7), dtype=np.float64)
        )
        row_lower = np.asarray(
            [item.lower for item in constraints], dtype=np.float64
        )
        result = self.qp.solve(
            nominal,
            metric,
            rows,
            row_lower,
            lower,
            upper,
        )
        safe = None if result.qdot_safe is None else result.qdot_safe.tolist()
        material = bool(
            safe is not None
            and np.linalg.norm(np.asarray(safe) - nominal)
            > float(optimizer["material_correction_tolerance_rad_s"])
        )
        record = {
            "schema_version": STEP_SCHEMA,
            "step": int(step),
            "constraints_enabled": bool(constraints_enabled),
            "solver_variable": "physical_panda_joint_velocity_rad_s",
            "source_cartesian_action": action.tolist(),
            "desired_translational_twist_m_per_s": desired_twist[:3].tolist(),
            "desired_angular_twist_rad_s": desired_twist[3:].tolist(),
            "unbounded_task_tracking_joint_velocity_rad_s": nominal.tolist(),
            "constraint_count": len(constraints),
            "constraints": [item.to_record() for item in constraints],
            "joint_position_bounds": joint_bounds,
            "qp": {
                "valid": bool(result.valid),
                "reason": result.reason,
                "executed_safe_joint_velocity_rad_s": safe,
                "material_intervention": material,
                "diagnostics": dict(result.diagnostics),
            },
            "timing": {
                "geometry_and_jacobian_wall_seconds": (
                    geometry_finished - geometry_started
                )
                * 1.0e-9,
                "total_filter_wall_seconds": (
                    time.perf_counter_ns() - total_started
                )
                * 1.0e-9,
            },
        }
        return safe, record
