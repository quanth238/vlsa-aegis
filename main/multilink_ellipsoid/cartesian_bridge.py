"""Bridge released AEGIS Cartesian commands to direct joint velocity."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .active import resolved_rate_action_map
from .joint_velocity import (
    JOINT_VELOCITY_PAIR_SCHEMA,
    DistalThreeJointVelocityCbf,
    _one_step_joint_bounds,
)
from .shadow import _arm_dof_indices, _eef_jacobian, _numpy


CARTESIAN_BRIDGE_SCHEMA = "vlsa_aegis_cartesian_joint_velocity_bridge_pair.v1"
CARTESIAN_BRIDGE_STEP_SCHEMA = "vlsa_aegis_cartesian_joint_velocity_bridge_step.v1"
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


def load_cartesian_bridge_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Cartesian joint-velocity bridge config is invalid JSON") from error
    required = {
        "arms",
        "bridge",
        "case_ids",
        "claim_scope",
        "controller",
        "nominal_action_source",
        "obstacle_geometry",
        "optimizer",
        "output_action_contract",
        "pairing",
        "protected_body_names",
        "protocol_id",
        "robot_geometry",
        "schema_version",
        "simulator_verification",
        "success_definition",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Cartesian joint-velocity bridge config keys differ")
    if config["schema_version"] != CARTESIAN_BRIDGE_SCHEMA:
        raise ValueError("Cartesian joint-velocity bridge schema differs")
    if config["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("Cartesian joint-velocity bridge case differs")
    if config["arms"] != [
        "aegis_cartesian_joint_velocity_bridge",
        "aegis_cartesian_joint_velocity_l5_l7_multicbf",
    ]:
        raise ValueError("Cartesian joint-velocity bridge arms differ")
    if config["protected_body_names"] != PROTECTED_BODIES:
        raise ValueError("Cartesian bridge must protect exactly L5-L7")
    if config["nominal_action_source"] != "immutable_archived_released_aegis_env_step_input":
        raise ValueError("Cartesian bridge nominal source differs")
    if config["output_action_contract"] != {
        "arm": "execute_physical_joint_velocity_qp_variable_directly",
        "gripper": "preserve_archived_aegis_command_exactly",
    }:
        raise ValueError("Cartesian bridge output contract differs")
    if config["pairing"] != {
        "action_horizon": 237,
        "control_frequency_hz": 20,
        "settle_actions": 20,
        "source": "same_immutable_archived_aegis_action_ledger_for_both_arms",
    }:
        raise ValueError("Cartesian bridge pairing differs")
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
        raise ValueError("Cartesian bridge controller is not one-to-one joint velocity")
    bridge = config["bridge"]
    expected_bridge_keys = {
        "cartesian_action_limit",
        "cartesian_delta_m_per_action_unit",
        "cartesian_velocity_m_per_s_per_action_unit",
        "calibration_source",
        "max_task_velocity_m_per_s",
        "resolved_rate_damping",
        "rotation",
    }
    if not isinstance(bridge, dict) or set(bridge) != expected_bridge_keys:
        raise ValueError("Cartesian bridge parameters differ")
    for key in (
        "cartesian_action_limit",
        "cartesian_delta_m_per_action_unit",
        "cartesian_velocity_m_per_s_per_action_unit",
        "max_task_velocity_m_per_s",
        "resolved_rate_damping",
    ):
        if isinstance(bridge[key], bool) or not math.isfinite(float(bridge[key])) or float(bridge[key]) <= 0.0:
            raise ValueError("bridge.%s must be finite and positive" % key)
    if float(bridge["cartesian_velocity_m_per_s_per_action_unit"]) != 0.2:
        raise ValueError("Cartesian bridge velocity scale differs from preregistration")
    if bridge["rotation"] != "zero_angular_velocity_translational_protocol":
        raise ValueError("Cartesian bridge rotation contract differs")
    optimizer = config["optimizer"]
    expected_optimizer = {
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
    if not isinstance(optimizer, dict) or set(optimizer) != expected_optimizer:
        raise ValueError("Cartesian bridge optimizer differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def internal_joint_velocity_cbf_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the preregistered bridge contract to the shared joint QP."""

    return {
        "schema_version": JOINT_VELOCITY_PAIR_SCHEMA,
        "protocol_id": "%s-shared-joint-cbf" % config["protocol_id"],
        "case_ids": list(config["case_ids"]),
        "claim_scope": config["claim_scope"],
        "obstacle_geometry": config["obstacle_geometry"],
        "optimizer": dict(config["optimizer"]),
        "pairing": {
            "control_frequency_hz": int(config["pairing"]["control_frequency_hz"]),
        },
        "protected_body_names": list(config["protected_body_names"]),
        "robot_geometry": dict(config["robot_geometry"]),
        "simulator_verification": dict(config["simulator_verification"]),
    }


class AegisCartesianJointVelocityBridge:
    """Map one released translational AEGIS action to bounded physical qdot."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        if config.get("schema_version") != CARTESIAN_BRIDGE_SCHEMA:
            raise ValueError("Cartesian bridge configuration was not validated")
        self.config = dict(config)

    def nominal(
        self,
        env: Any,
        current_eef_position_m: Sequence[float],
        aegis_action: Sequence[float],
        *,
        step: int,
    ) -> tuple[list[float], dict[str, Any]]:
        np = _numpy()
        action = np.asarray(aegis_action, dtype=np.float64)
        current = np.asarray(current_eef_position_m, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("released AEGIS action must be finite length seven")
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise ValueError("current end-effector position must be finite length three")
        if not np.allclose(action[3:6], 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("Cartesian bridge requires zero archived rotation")
        bridge = self.config["bridge"]
        optimizer = self.config["optimizer"]
        xyz = np.clip(
            action[:3],
            -float(bridge["cartesian_action_limit"]),
            float(bridge["cartesian_action_limit"]),
        )
        target = current + float(bridge["cartesian_delta_m_per_action_unit"]) * xyz
        desired_linear = float(bridge["cartesian_velocity_m_per_s_per_action_unit"]) * xyz
        speed = float(np.linalg.norm(desired_linear))
        max_speed = float(bridge["max_task_velocity_m_per_s"])
        speed_clipped = False
        if speed > max_speed:
            desired_linear *= max_speed / speed
            speed_clipped = True
        arm_dofs = _arm_dof_indices(env)
        jacobian = _eef_jacobian(env, arm_dofs)
        mapping = resolved_rate_action_map(
            jacobian,
            damping=float(bridge["resolved_rate_damping"]),
        )
        raw_qdot = mapping @ desired_linear
        lower, upper, joint_bounds = _one_step_joint_bounds(
            env,
            arm_dofs,
            control_period_seconds=1.0 / float(self.config["pairing"]["control_frequency_hz"]),
            joint_velocity_limit=float(optimizer["joint_velocity_limit_rad_s"]),
            joint_position_margin=float(optimizer["joint_position_margin_rad"]),
        )
        bounded_qdot = np.clip(raw_qdot, lower, upper)
        return bounded_qdot.tolist(), {
            "schema_version": CARTESIAN_BRIDGE_STEP_SCHEMA,
            "step": int(step),
            "source_aegis_action": action.tolist(),
            "source_xyz_after_clip": xyz.tolist(),
            "current_eef_position_m": current.tolist(),
            "instantaneous_target_eef_position_m": target.tolist(),
            "desired_linear_velocity_m_per_s": desired_linear.tolist(),
            "desired_angular_velocity_rad_s": [0.0, 0.0, 0.0],
            "task_speed_clipped": speed_clipped,
            "resolved_rate_mapping": mapping.tolist(),
            "raw_joint_velocity_rad_s": raw_qdot.tolist(),
            "bounded_joint_velocity_rad_s": bounded_qdot.tolist(),
            "joint_bound_clipped_indexes": [
                int(index)
                for index in np.flatnonzero(raw_qdot != bounded_qdot)
            ],
            "joint_position_bounds": joint_bounds,
            "gripper_command_preserved": float(action[6]),
        }


def build_joint_velocity_filter(
    config: Mapping[str, Any], geometry: Mapping[str, Any]
) -> DistalThreeJointVelocityCbf:
    return DistalThreeJointVelocityCbf.from_aegis_geometry(
        internal_joint_velocity_cbf_config(config), geometry
    )
