"""Opt-in active three-link multi-CBF filter for translational AEGIS actions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .barrier import build_pair_constraint
from .qp import MultiConstraintQp
from .shadow import (
    DISTAL_ELLIPSOID_SCHEMA,
    MultilinkEllipsoidShadow,
    _arm_dof_indices,
    _eef_jacobian,
    _geom_jacobians,
    _numpy,
)


ACTIVE_SCHEMA = "vlsa_distal_three_ellipsoid_multicbf.v1"
ACTIVE_STEP_SCHEMA = "vlsa_distal_three_ellipsoid_multicbf_step.v1"
_PROTECTED_BODIES = ["robot0_link5", "robot0_link6", "robot0_link7"]
_ROBOT_GEOMETRY = {
    "source": "compiled_mujoco_collision_mesh_vertices",
    "fit": "khachiyan_mvee_exact_vertex_inflation",
    "relative_numerical_padding": 1.0e-9,
    "khachiyan_tolerance": 1.0e-4,
    "khachiyan_max_iterations": 20000,
}
_SIMULATOR_VERIFICATION = {
    "D_sim": "raw_mujoco_nonpositive_contact_distance_and_active_obstacle_displacement",
    "distinct_from_D_opt": True,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_active_config(path: Path) -> dict[str, Any]:
    """Load the preregistered active three-link multi-CBF configuration."""

    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("active multi-CBF config is invalid JSON") from error
    required = {
        "schema_version",
        "protocol_id",
        "case_ids",
        "control_effect",
        "nominal_action_source",
        "output_action_contract",
        "obstacle_geometry",
        "protected_body_names",
        "robot_geometry",
        "optimizer",
        "simulator_verification",
        "success_definition",
        "claim_scope",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("active multi-CBF config keys differ")
    if config.get("schema_version") != ACTIVE_SCHEMA:
        raise ValueError("active multi-CBF config schema differs")
    if config["control_effect"] != "modify_archived_aegis_xyz_before_env_step":
        raise ValueError("active multi-CBF control effect differs")
    if config["nominal_action_source"] != "immutable_archived_released_aegis_env_step_input":
        raise ValueError("active multi-CBF nominal action source differs")
    if config["output_action_contract"] != {
        "xyz": "jointly_filtered_by_three_link_cbf_qp",
        "rotation": "zero_translational_protocol",
        "gripper": "preserve_archived_aegis_command",
    }:
        raise ValueError("active multi-CBF action contract differs")
    if config["obstacle_geometry"] != "released_aegis_frozen_perception_mvee":
        raise ValueError("active multi-CBF obstacle geometry differs")
    if config["protected_body_names"] != _PROTECTED_BODIES:
        raise ValueError("active multi-CBF must protect exactly link5-link7")
    if config["robot_geometry"] != _ROBOT_GEOMETRY:
        raise ValueError("active multi-CBF robot geometry differs")
    case_ids = config["case_ids"]
    if not isinstance(case_ids, list) or not case_ids or any(
        not isinstance(value, str) or not value for value in case_ids
    ):
        raise ValueError("active multi-CBF case_ids must be nonempty strings")
    optimizer = config["optimizer"]
    optimizer_keys = {
        "alpha_s_inv",
        "optimizer_clearance_m",
        "resolved_rate_damping",
        "action_limit",
        "action_weight_diagonal",
        "eps_abs",
        "eps_rel",
        "max_iter",
        "residual_tolerance",
        "bound_tolerance_action",
    }
    if not isinstance(optimizer, dict) or set(optimizer) != optimizer_keys:
        raise ValueError("active multi-CBF optimizer keys differ")
    positive = (
        "alpha_s_inv",
        "resolved_rate_damping",
        "action_limit",
        "eps_abs",
        "eps_rel",
        "residual_tolerance",
        "bound_tolerance_action",
    )
    for key in positive:
        value = optimizer[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("optimizer.%s must be finite and positive" % key)
    clearance = optimizer["optimizer_clearance_m"]
    if isinstance(clearance, bool) or not math.isfinite(float(clearance)) or float(clearance) < 0.0:
        raise ValueError("optimizer.optimizer_clearance_m must be nonnegative")
    weights = optimizer["action_weight_diagonal"]
    if not isinstance(weights, list) or len(weights) != 3 or any(
        isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0
        for value in weights
    ):
        raise ValueError("optimizer.action_weight_diagonal must contain three positive values")
    if isinstance(optimizer["max_iter"], bool) or not isinstance(optimizer["max_iter"], int):
        raise ValueError("optimizer.max_iter must be an integer")
    if config["simulator_verification"] != _SIMULATOR_VERIFICATION:
        raise ValueError("active multi-CBF must keep D_sim distinct from D_opt")
    if config["success_definition"] != {
        "protected_link_contact": "none_for_robot0_link5_link6_link7",
        "paper_car": "maximum_active_obstacle_l1_displacement_at_most_0.001_m",
        "task": "native_goal_satisfied_within_archived_237_action_horizon",
    }:
        raise ValueError("active multi-CBF success definition differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _geometry_shadow_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create the validated read-only carrier used only for shared geometry."""

    optimizer = config["optimizer"]
    return {
        "schema_version": DISTAL_ELLIPSOID_SCHEMA,
        "protocol_id": "%s-geometry" % config["protocol_id"],
        "case_ids": list(config["case_ids"]),
        "control_effect": "read_only_no_executed_action_change",
        "obstacle_geometry": config["obstacle_geometry"],
        "protected_body_names": list(config["protected_body_names"]),
        "robot_geometry": dict(config["robot_geometry"]),
        "optimizer": {
            "alpha_s_inv": float(optimizer["alpha_s_inv"]),
            "optimizer_clearance_m": float(optimizer["optimizer_clearance_m"]),
            "joint_velocity_limit_rad_s": 1.0,
            "resolved_rate_damping": float(optimizer["resolved_rate_damping"]),
            "eef_preservation_weight": 1.0,
            "eef_weight_diagonal": [1.0] * 6,
            "eps_abs": float(optimizer["eps_abs"]),
            "eps_rel": float(optimizer["eps_rel"]),
            "max_iter": int(optimizer["max_iter"]),
            "residual_tolerance": float(optimizer["residual_tolerance"]),
            "bound_tolerance_rad_s": float(optimizer["bound_tolerance_action"]),
        },
        "simulator_verification": dict(config["simulator_verification"]),
        "claim_scope": config["claim_scope"],
    }


def resolved_rate_action_map(eef_jacobian: Any, *, damping: float) -> Any:
    """Map normalized translational OSC action to estimated arm joint velocity."""

    np = _numpy()
    jacobian = np.asarray(eef_jacobian, dtype=np.float64)
    if jacobian.shape != (6, 7) or not np.all(np.isfinite(jacobian)):
        raise ValueError("end-effector Jacobian must be finite with shape (6, 7)")
    regularization = float(damping)
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("resolved-rate damping must be finite and positive")
    translation_selector = np.zeros((6, 3), dtype=np.float64)
    translation_selector[:3] = np.eye(3)
    mapping = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + regularization * regularization * np.eye(6),
        translation_selector,
    )
    if mapping.shape != (7, 3) or not np.all(np.isfinite(mapping)):
        raise ValueError("resolved-rate action map is invalid")
    return mapping


class DistalThreeEllipsoidMultiCbf:
    """Jointly filter XYZ using three link barriers, then execute that XYZ."""

    def __init__(
        self,
        config: Mapping[str, Any],
        geometry: MultilinkEllipsoidShadow,
    ) -> None:
        if config.get("schema_version") != ACTIVE_SCHEMA:
            raise ValueError("active multi-CBF configuration was not validated")
        self.config = dict(config)
        self.geometry = geometry
        optimizer = config["optimizer"]
        self.qp = MultiConstraintQp(
            eps_abs=float(optimizer["eps_abs"]),
            eps_rel=float(optimizer["eps_rel"]),
            max_iter=int(optimizer["max_iter"]),
            residual_tolerance=float(optimizer["residual_tolerance"]),
            bound_tolerance=float(optimizer["bound_tolerance_action"]),
        )

    @classmethod
    def from_aegis_geometry(
        cls,
        config: Mapping[str, Any],
        geometry: Mapping[str, Any],
    ) -> "DistalThreeEllipsoidMultiCbf":
        carrier = MultilinkEllipsoidShadow.from_aegis_geometry(
            _geometry_shadow_config(config),
            geometry,
        )
        return cls(config, carrier)

    def geometry_record(self, env: Any) -> dict[str, Any]:
        return self.geometry.geometry_record(env)

    def filter(
        self,
        env: Any,
        aegis_action: Sequence[float],
        *,
        step: int,
    ) -> tuple[list[float] | None, dict[str, Any]]:
        np = _numpy()
        started = time.perf_counter_ns()
        optimizer = self.config["optimizer"]
        action = np.asarray(aegis_action, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("released AEGIS action must be finite with length seven")
        if not np.allclose(action[3:6], 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("active multi-CBF requires the translational AEGIS protocol")
        limit = float(optimizer["action_limit"])
        nominal_xyz = np.clip(action[:3], -limit, limit)
        arm_dofs = _arm_dof_indices(env)
        eef_jacobian = _eef_jacobian(env, arm_dofs)
        action_to_qdot = resolved_rate_action_map(
            eef_jacobian,
            damping=float(optimizer["resolved_rate_damping"]),
        )
        geometry_started = time.perf_counter_ns()
        links = self.geometry._distal_links(env)
        constraints = []
        action_rows = []
        for link in links:
            jac_position, jac_rotation = _geom_jacobians(env, link, arm_dofs)
            constraint = build_pair_constraint(
                link,
                self.geometry.obstacle,
                jac_position,
                jac_rotation,
                alpha=float(optimizer["alpha_s_inv"]),
                optimizer_clearance_m=float(optimizer["optimizer_clearance_m"]),
            )
            constraints.append(constraint)
            action_rows.append(constraint.row @ action_to_qdot)
        geometry_finished = time.perf_counter_ns()
        if len(constraints) != 3:
            raise ValueError("active multi-CBF requires exactly three link constraints")
        rows = np.stack(action_rows, axis=0)
        lower = np.asarray([item.lower for item in constraints], dtype=np.float64)
        metric = np.diag(
            np.asarray(optimizer["action_weight_diagonal"], dtype=np.float64)
        )
        result = self.qp.solve(
            nominal_xyz,
            metric,
            rows,
            lower,
            -limit * np.ones(3),
            limit * np.ones(3),
        )
        violated = [
            index
            for index in range(3)
            if float(rows[index] @ nominal_xyz - lower[index]) < 0.0
        ]
        closest_index = int(np.argmin([item.h_opt_m for item in constraints]))
        safe_action = None
        correction_l2 = None
        if result.valid and result.qdot_safe is not None:
            safe_xyz = np.asarray(result.qdot_safe, dtype=np.float64)
            output = action.copy()
            output[:3] = safe_xyz
            output[3:6] = 0.0
            safe_action = output.tolist()
            correction_l2 = float(np.linalg.norm(safe_xyz - nominal_xyz))
        constraint_records = []
        for constraint, action_row in zip(constraints, rows):
            record = constraint.to_record()
            record["cbf_row_per_normalized_xyz_action"] = action_row.tolist()
            constraint_records.append(record)
        record = {
            "schema_version": ACTIVE_STEP_SCHEMA,
            "step": int(step),
            "control_effect": "modify_archived_aegis_xyz_before_env_step",
            "solver_variable": "normalized_translational_osc_xyz_action",
            "arm_dof_indices": list(arm_dofs),
            "nominal_aegis_action": action.tolist(),
            "nominal_xyz_after_controller_clip": nominal_xyz.tolist(),
            "resolved_rate_action_to_qdot_rad_s": action_to_qdot.tolist(),
            "nominal_qdot_estimate_rad_s": (action_to_qdot @ nominal_xyz).tolist(),
            "constraint_count": 3,
            "constraints": constraint_records,
            "constraint_derivative": {
                "source": (
                    "analytic_rigid_link_twist_through_mujoco_jacobian_"
                    "and_damped_resolved_rate_action_map"
                )
            },
            "closest_constraint_index": closest_index,
            "closest_body_name": constraints[closest_index].body_name,
            "minimum_h_opt_m": float(constraints[closest_index].h_opt_m),
            "nominal_violated_constraint_indexes": violated,
            "nominal_violated_constraint_count": len(violated),
            "qp": {
                "valid": bool(result.valid),
                "reason": result.reason,
                "safe_xyz_action": (
                    None if result.qdot_safe is None else result.qdot_safe.tolist()
                ),
                "diagnostics": dict(result.diagnostics),
            },
            "executed_action": safe_action,
            "active_correction_l2": correction_l2,
            "modified": bool(correction_l2 is not None and correction_l2 > 1.0e-12),
            "D_opt": {
                "value_m": float(optimizer["optimizer_clearance_m"]),
                "semantics": "optimizer_support_gap_buffer",
            },
            "D_sim": {
                "available": False,
                "value": {
                    "minimum_protected_link_contact_distance_m": None,
                    "active_obstacle_l1_displacement_m": None,
                    "protected_link_contact_count": None,
                },
                "semantics": self.config["simulator_verification"]["D_sim"],
                "source": "post_step_raw_simulator_evidence_not_optimizer_geometry",
            },
            "timing": {
                "geometry_and_jacobian_wall_seconds": (
                    geometry_finished - geometry_started
                )
                * 1.0e-9,
                "total_filter_wall_seconds": (
                    time.perf_counter_ns() - started
                )
                * 1.0e-9,
            },
        }
        return safe_action, record


def summarize_active_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    if not records:
        return {"status": "no_records", "step_count": 0}

    def stats(values: Any) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError("active multi-CBF timing is invalid")
        return {
            "mean_seconds": float(np.mean(array)),
            "median_seconds": float(np.median(array)),
            "p95_seconds": float(np.quantile(array, 0.95)),
            "maximum_seconds": float(np.max(array)),
        }

    first_intervention = next(
        (int(item["step"]) for item in records if item["modified"]),
        None,
    )
    return {
        "status": "complete" if all(item["qp"]["valid"] for item in records) else "qp_failure",
        "step_count": len(records),
        "constraint_count_min": min(int(item["constraint_count"]) for item in records),
        "constraint_count_max": max(int(item["constraint_count"]) for item in records),
        "all_qps_valid": all(item["qp"]["valid"] for item in records),
        "qp_failure_count": sum(not item["qp"]["valid"] for item in records),
        "intervention_count": sum(bool(item["modified"]) for item in records),
        "first_intervention_step": first_intervention,
        "maximum_action_correction_l2": max(
            float(item["active_correction_l2"] or 0.0) for item in records
        ),
        "minimum_h_opt_m": min(float(item["minimum_h_opt_m"]) for item in records),
        "qp_total_wall_timing": stats(
            [item["qp"]["diagnostics"]["timing"]["total_wall_seconds"] for item in records]
        ),
        "qp_solve_wall_timing": stats(
            [item["qp"]["diagnostics"]["timing"]["solve_wall_seconds"] for item in records]
        ),
        "total_filter_wall_timing": stats(
            [item["timing"]["total_filter_wall_seconds"] for item in records]
        ),
    }
