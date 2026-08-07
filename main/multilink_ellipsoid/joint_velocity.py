"""Direct joint-velocity L5/L6/L7 multi-CBF controller."""

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
    _raw_model_data,
)


JOINT_VELOCITY_PAIR_SCHEMA = "vlsa_pi05_droid_joint_velocity_pair.v1"
JOINT_VELOCITY_STEP_SCHEMA = "vlsa_pi05_droid_joint_velocity_multicbf_step.v1"
PROTECTED_BODIES = ["robot0_link5", "robot0_link6", "robot0_link7"]


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


def load_joint_velocity_pair_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("joint-velocity pair config is invalid JSON") from error
    required = {
        "arms",
        "case_ids",
        "claim_scope",
        "controller",
        "observation_adapter",
        "obstacle_geometry",
        "optimizer",
        "pairing",
        "policy",
        "protected_body_names",
        "protocol_id",
        "robot_geometry",
        "schema_version",
        "simulator_verification",
        "success_definition",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("joint-velocity pair config keys differ")
    if config["schema_version"] != JOINT_VELOCITY_PAIR_SCHEMA:
        raise ValueError("joint-velocity pair schema differs")
    if config["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("joint-velocity pilot case differs")
    if config["protected_body_names"] != PROTECTED_BODIES:
        raise ValueError("joint-velocity pilot must protect exactly L5-L7")
    pairing = config["pairing"]
    if pairing != {
        "action_horizon": 225,
        "control_frequency_hz": 15,
        "model_action_horizon": 15,
        "open_loop_execution_steps": 8,
        "physical_time_horizon_s": 15.0,
        "policy_noise": "same_manifest_base_seed_plus_query_index_for_both_arms",
        "settle_actions": 15,
    }:
        raise ValueError("joint-velocity pairing protocol differs")
    controller = config["controller"]
    if controller.get("type") != "JOINT_VELOCITY":
        raise ValueError("joint-velocity controller type differs")
    if any(float(controller[key]) != expected for key, expected in (
        ("input_max", 1.0),
        ("input_min", -1.0),
        ("output_max", 1.0),
        ("output_min", -1.0),
    )):
        raise ValueError("joint-velocity controller scaling is not one-to-one")
    optimizer = config["optimizer"]
    expected_optimizer_keys = {
        "alpha_s_inv",
        "bound_tolerance_rad_s",
        "eef_preservation_weight",
        "eef_weight_diagonal",
        "eps_abs",
        "eps_rel",
        "joint_position_margin_rad",
        "joint_velocity_limit_rad_s",
        "material_correction_tolerance_rad_s",
        "max_iter",
        "optimizer_clearance_m",
        "residual_tolerance",
    }
    if not isinstance(optimizer, dict) or set(optimizer) != expected_optimizer_keys:
        raise ValueError("joint-velocity optimizer keys differ")
    for key, value in optimizer.items():
        if key == "eef_weight_diagonal":
            if not isinstance(value, list) or len(value) != 6:
                raise ValueError("eef weight diagonal must have six values")
            continue
        if key == "max_iter":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("max_iter must be positive integer")
            continue
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("optimizer.%s must be finite and positive" % key)
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _geometry_config(config: Mapping[str, Any]) -> dict[str, Any]:
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
            "joint_velocity_limit_rad_s": float(optimizer["joint_velocity_limit_rad_s"]),
            "resolved_rate_damping": 0.05,
            "eef_preservation_weight": float(optimizer["eef_preservation_weight"]),
            "eef_weight_diagonal": list(optimizer["eef_weight_diagonal"]),
            "eps_abs": float(optimizer["eps_abs"]),
            "eps_rel": float(optimizer["eps_rel"]),
            "max_iter": int(optimizer["max_iter"]),
            "residual_tolerance": float(optimizer["residual_tolerance"]),
            "bound_tolerance_rad_s": float(optimizer["bound_tolerance_rad_s"]),
        },
        "simulator_verification": dict(config["simulator_verification"]),
        "claim_scope": config["claim_scope"],
    }


def _one_step_joint_bounds(
    env: Any,
    arm_dofs: Sequence[int],
    *,
    control_period_seconds: float,
    joint_velocity_limit: float,
    joint_position_margin: float,
) -> tuple[Any, Any, dict[str, Any]]:
    np = _numpy()
    model, data = _raw_model_data(env.sim)
    lower = -joint_velocity_limit * np.ones(len(arm_dofs), dtype=np.float64)
    upper = joint_velocity_limit * np.ones(len(arm_dofs), dtype=np.float64)
    records = []
    for index, dof_id in enumerate(arm_dofs):
        joint_id = int(model.dof_jntid[int(dof_id)])
        if not bool(model.jnt_limited[joint_id]):
            raise ValueError("Panda arm joint is unexpectedly unlimited")
        qpos_address = int(model.jnt_qposadr[joint_id])
        q = float(data.qpos[qpos_address])
        q_min, q_max = [float(value) for value in model.jnt_range[joint_id]]
        safe_min = q_min + joint_position_margin
        safe_max = q_max - joint_position_margin
        lower[index] = max(lower[index], (safe_min - q) / control_period_seconds)
        upper[index] = min(upper[index], (safe_max - q) / control_period_seconds)
        records.append(
            {
                "dof_id": int(dof_id),
                "joint_id": joint_id,
                "qpos_address": qpos_address,
                "q_rad": q,
                "range_rad": [q_min, q_max],
                "safe_range_rad": [safe_min, safe_max],
                "velocity_bounds_rad_s": [float(lower[index]), float(upper[index])],
            }
        )
    if np.any(lower > upper):
        raise ValueError("one-step joint-position bounds are contradictory")
    return lower, upper, {
        "control_period_seconds": float(control_period_seconds),
        "joint_position_margin_rad": float(joint_position_margin),
        "joints": records,
    }


class DistalThreeJointVelocityCbf:
    """Filter a physical seven-joint velocity with three link barriers."""

    def __init__(self, config: Mapping[str, Any], geometry: MultilinkEllipsoidShadow) -> None:
        if config.get("schema_version") != JOINT_VELOCITY_PAIR_SCHEMA:
            raise ValueError("joint-velocity configuration was not validated")
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
    def from_aegis_geometry(
        cls,
        config: Mapping[str, Any],
        geometry: Mapping[str, Any],
    ) -> "DistalThreeJointVelocityCbf":
        carrier = MultilinkEllipsoidShadow.from_aegis_geometry(
            _geometry_config(config), geometry
        )
        return cls(config, carrier)

    def geometry_record(self, env: Any) -> dict[str, Any]:
        return self.geometry.geometry_record(env)

    def filter(
        self,
        env: Any,
        nominal_joint_velocity: Sequence[float],
        *,
        step: int,
    ) -> tuple[list[float] | None, dict[str, Any]]:
        np = _numpy()
        started = time.perf_counter_ns()
        optimizer = self.config["optimizer"]
        nominal_raw = np.asarray(nominal_joint_velocity, dtype=np.float64)
        if nominal_raw.shape != (7,) or not np.all(np.isfinite(nominal_raw)):
            raise ValueError("DROID nominal joint velocity must be finite length seven")
        limit = float(optimizer["joint_velocity_limit_rad_s"])
        nominal = np.clip(nominal_raw, -limit, limit)
        arm_dofs = _arm_dof_indices(env)
        eef_jacobian = _eef_jacobian(env, arm_dofs)
        links = self.geometry._distal_links(env)
        constraints = []
        for link in links:
            jac_position, jac_rotation = _geom_jacobians(env, link, arm_dofs)
            constraints.append(
                build_pair_constraint(
                    link,
                    self.geometry.obstacle,
                    jac_position,
                    jac_rotation,
                    alpha=float(optimizer["alpha_s_inv"]),
                    optimizer_clearance_m=float(optimizer["optimizer_clearance_m"]),
                )
            )
        if len(constraints) != 3:
            raise ValueError("joint-velocity multi-CBF requires exactly three rows")
        rows = np.stack([item.row for item in constraints], axis=0)
        row_lower = np.asarray([item.lower for item in constraints], dtype=np.float64)
        weights = np.diag(np.asarray(optimizer["eef_weight_diagonal"], dtype=np.float64))
        metric = np.eye(7) + float(optimizer["eef_preservation_weight"]) * (
            eef_jacobian.T @ weights @ eef_jacobian
        )
        velocity_lower, velocity_upper, joint_bounds = _one_step_joint_bounds(
            env,
            arm_dofs,
            control_period_seconds=1.0 / float(self.config["pairing"]["control_frequency_hz"]),
            joint_velocity_limit=limit,
            joint_position_margin=float(optimizer["joint_position_margin_rad"]),
        )
        result = self.qp.solve(
            nominal,
            metric,
            rows,
            row_lower,
            velocity_lower,
            velocity_upper,
        )
        nominal_violated = [
            index for index in range(3) if float(rows[index] @ nominal - row_lower[index]) < 0.0
        ]
        nominal_bound_violated = [
            index
            for index in range(7)
            if nominal[index] < velocity_lower[index] or nominal[index] > velocity_upper[index]
        ]
        closest_index = int(np.argmin([item.h_opt_m for item in constraints]))
        safe = None
        correction = None
        exact_snap = False
        if result.valid and result.qdot_safe is not None:
            safe_array = np.asarray(result.qdot_safe, dtype=np.float64)
            if not nominal_violated and not nominal_bound_violated:
                safe_array = nominal.copy()
                exact_snap = True
            safe = safe_array.tolist()
            correction = float(np.linalg.norm(safe_array - nominal))
        record = {
            "schema_version": JOINT_VELOCITY_STEP_SCHEMA,
            "step": int(step),
            "solver_variable": "physical_panda_joint_velocity_rad_s",
            "arm_dof_indices": list(arm_dofs),
            "nominal_joint_velocity_raw_rad_s": nominal_raw.tolist(),
            "nominal_joint_velocity_clipped_rad_s": nominal.tolist(),
            "constraint_count": 3,
            "constraints": [item.to_record() for item in constraints],
            "closest_constraint_index": closest_index,
            "closest_body_name": constraints[closest_index].body_name,
            "minimum_h_opt_m": float(constraints[closest_index].h_opt_m),
            "nominal_violated_constraint_indexes": nominal_violated,
            "nominal_violated_joint_bound_indexes": nominal_bound_violated,
            "joint_position_bounds": joint_bounds,
            "qp": {
                "valid": bool(result.valid),
                "reason": result.reason,
                "solver_safe_joint_velocity_rad_s": (
                    None if result.qdot_safe is None else result.qdot_safe.tolist()
                ),
                "executed_safe_joint_velocity_rad_s": safe,
                "nominal_feasible_exact_snap": exact_snap,
                "diagnostics": dict(result.diagnostics),
            },
            "active_correction_l2_rad_s": correction,
            "material_correction_tolerance_rad_s": float(
                optimizer["material_correction_tolerance_rad_s"]
            ),
            "modified": bool(
                correction is not None
                and correction > float(optimizer["material_correction_tolerance_rad_s"])
            ),
            "D_opt": {
                "value_m": float(optimizer["optimizer_clearance_m"]),
                "semantics": "optimizer_support_gap_buffer",
            },
            "D_sim": {
                "available": False,
                "value": None,
                "semantics": self.config["simulator_verification"]["D_sim"],
            },
            "timing": {
                "total_filter_wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
            },
        }
        return safe, record
