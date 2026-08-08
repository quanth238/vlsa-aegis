"""Barrier-free EmbodiSteer joint-denoising baseline primitives.

The paper uses Cartesian poses relative to the action-chunk start.  LIBERO's
frozen pi0.5 checkpoint instead predicts incremental OSC delta poses.  This
pilot therefore represents the horizon as a sequential joint-configuration
trajectory while retaining EmbodiSteer's Eq. (3) initialization and Eq. (4)
damped-Jacobian residual update.  No collision geometry or guidance appears
in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Tuple


SCHEMA = "vlsa_embodisteer_joint_baselines.v1"
SCHEMA_V2 = "vlsa_embodisteer_joint_baselines.v2"
SCHEMA_V3 = "vlsa_embodisteer_joint_baselines.v3"
SCHEMA_AEGIS_EE = "vlsa_embodisteer_aegis_ee_pair.v1"
SCHEMA_AEGIS_EE_V2 = "vlsa_embodisteer_aegis_ee_pair.v2"
CONTROL_SCHEMA = "crfs_embodisteer_joint_denoising.v1"


AEGIS_EE_SCHEMAS = {SCHEMA_AEGIS_EE, SCHEMA_AEGIS_EE_V2}


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("EmbodiSteer joint baseline requires NumPy") from error
    return np


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


def load_joint_baseline_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version",
        "protocol_id",
        "paper_reference",
        "case_ids",
        "claim_scope",
        "arms",
        "nominal_policy",
        "collision_guidance",
        "action_protocol",
        "joint_denoising",
        "pairing",
        "success_definition",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("EmbodiSteer joint-baseline config keys differ")
    if config["schema_version"] not in {
        SCHEMA,
        SCHEMA_V2,
        SCHEMA_V3,
        SCHEMA_AEGIS_EE,
        SCHEMA_AEGIS_EE_V2,
    }:
        raise ValueError("EmbodiSteer joint-baseline schema differs")
    if config["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("EmbodiSteer joint-baseline case differs")
    expected_arms = (
        ["cartesian_ee_with_aegis_ee", "joint_denoising_with_aegis_ee"]
        if config["schema_version"] in AEGIS_EE_SCHEMAS
        else ["cartesian_ee_no_guidance", "joint_denoising_no_guidance"]
    )
    if config["arms"] != expected_arms:
        raise ValueError("EmbodiSteer joint-baseline arms differ")
    if config["paper_reference"] != {
        "arxiv": "2606.12965v1",
        "baseline_definition": "appendix_B4_EE_and_Joint",
        "code_status_at_preregistration": "project_page_says_code_coming_soon",
        "joint_denoising_equations": ["3", "4", "8", "9", "10"],
    }:
        raise ValueError("EmbodiSteer joint-baseline paper mapping differs")
    if config["schema_version"] in AEGIS_EE_SCHEMAS:
        guidance = config["collision_guidance"]
        if set(guidance) != {
            "barrier_projection_enabled",
            "ellipsoid_constraints_enabled",
            "joint_adapter",
            "l5_l6_l7_constraints_enabled",
            "qp_enabled",
            "released_aegis",
        }:
            raise ValueError("AEGIS-EE guidance keys differ")
        if not all(
            guidance[key]
            for key in (
                "barrier_projection_enabled",
                "ellipsoid_constraints_enabled",
                "qp_enabled",
            )
        ) or guidance["l5_l6_l7_constraints_enabled"] is not False:
            raise ValueError("AEGIS-EE must enable one EE QP and disable L5--L7")
        released = guidance["released_aegis"]
        if set(released) != {
            "alpha_gain",
            "end_effector_proxy",
            "formulation",
            "internal_dt_s",
            "obstacle_mvee",
            "rotation_execution",
            "source_table1_artifact",
        } or float(released["alpha_gain"]) != 10.0 or float(
            released["internal_dt_s"]
        ) != 0.05:
            raise ValueError("released AEGIS parameter contract differs")
        if released.get("formulation") != (
            "released_table1_six_variable_translational_cbf_qp"
        ) or released.get("rotation_execution") != (
            "zero_as_released_table1_translational_protocol"
        ):
            raise ValueError("released AEGIS formulation differs")
        proxy = released.get("end_effector_proxy", {})
        if proxy != {
            "center": "robot0_eef_pose_plus_minus_0.08m_local_z",
            "semiaxes_m": [0.06, 0.12, 0.11],
        }:
            raise ValueError("released AEGIS end-effector proxy differs")
        obstacle = released.get("obstacle_mvee", {})
        if set(obstacle) != {"center_m", "rotation", "semiaxes_m"}:
            raise ValueError("frozen AEGIS obstacle MVEE keys differ")
        np = _numpy()
        center = np.asarray(obstacle["center_m"], dtype=np.float64)
        rotation = np.asarray(obstacle["rotation"], dtype=np.float64)
        semiaxes = np.asarray(obstacle["semiaxes_m"], dtype=np.float64)
        if (
            center.shape != (3,)
            or rotation.shape != (3, 3)
            or semiaxes.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(semiaxes))
            or np.any(semiaxes <= 0.0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        ):
            raise ValueError("frozen AEGIS obstacle MVEE is invalid")
        expected_center = np.asarray(
            [-0.058366719778605226, 0.14301469629436583, 1.0230830180351789]
        )
        expected_rotation = np.asarray(
            [
                [-0.020790474824008438, 0.058184500032663435, -0.9980893347353939],
                [0.1576830297250622, 0.9860014894880728, 0.054195247614777],
                [0.9872708940778703, -0.15625500530623276, -0.02967414739182496],
            ]
        )
        expected_semiaxes = np.asarray(
            [0.1393188890560937, 0.12054261654746469, 0.07910671170710154]
        )
        if not (
            np.array_equal(center, expected_center)
            and np.array_equal(rotation, expected_rotation)
            and np.array_equal(semiaxes, expected_semiaxes)
        ):
            raise ValueError("frozen AEGIS obstacle MVEE values differ")
        adapter = guidance["joint_adapter"]
        if adapter != {
            "damped_jacobian_lambda": 0.001,
            "method": "posthoc_released_aegis_cartesian_translation_correction_lifted_to_direct_joint_target",
            "preserve_nominal_joint_orientation": True,
            "translation_scale_m_per_action_unit": 0.05,
        }:
            raise ValueError("AEGIS-EE direct-joint adapter differs")
        if released["source_table1_artifact"] != {
            "active_obstacle_initial_position_m": [
                -0.060000286232933644,
                0.14002376109395798,
                1.005775853852426,
            ],
            "active_obstacle_position_tolerance_m": 1e-09,
            "case_id": "vlsa-t1-goal-ii-t0-e05",
            "file_sha256": "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b",
            "result_payload_sha256": "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c",
            "settled_agentview_array_sha256": "b66103d274cbd2db02ef7fe84771725819fb74062ef36de18b6af6119da7865f",
            "settled_simulator_state_sha256": "5a72a870b8368d0a6508428bb89dce75ccc6f28e86a2349edae35b2618918741",
        }:
            raise ValueError("frozen Table-1 AEGIS source artifact differs")
    elif config["collision_guidance"] != {
        "barrier_projection_enabled": False,
        "ellipsoid_constraints_enabled": False,
        "qp_enabled": False,
        "reason": "isolate_cartesian_policy_to_joint_denoising_fidelity_before_geometry",
    }:
        raise ValueError("EmbodiSteer joint baseline must disable all guidance")
    protocol = config["action_protocol"]
    paper_rate = config["schema_version"] in {SCHEMA_V3, SCHEMA_AEGIS_EE}
    expected_rate = 10 if paper_rate else 20
    expected_execution = 10 if paper_rate else 5
    if any(
        int(protocol[key]) != expected
        for key, expected in (
            ("control_frequency_hz", expected_rate),
            ("execute_actions_per_query", expected_execution),
            ("max_actions", 300),
            ("model_action_horizon", 10),
        )
    ):
        raise ValueError("EmbodiSteer joint-baseline action protocol differs")
    if protocol["cartesian_controller"] != "OSC_POSE":
        raise ValueError("Cartesian baseline controller differs")
    controller = protocol["joint_controller"]
    if controller.get("type") != "JOINT_POSITION" or controller.get(
        "impedance_mode"
    ) != "fixed":
        raise ValueError("Joint baseline controller differs")
    if config["schema_version"] == SCHEMA:
        if "joint_target_encoding" in protocol or float(controller["output_max"]) != 0.05:
            raise ValueError("v1 joint delta adapter differs")
    else:
        if protocol.get("joint_target_encoding") != {
            "formula": "env_action=(q_target-q_current)/controller_output_max",
            "maximum_panda_joint_range_rad": 5.8,
            "purpose": "encode_absolute_Q0_target_without_delta_saturation",
        }:
            raise ValueError("v2 absolute joint-target adapter differs")
        if float(controller["output_max"]) != 6.0 or float(controller["output_min"]) != -6.0:
            raise ValueError("v2 absolute joint-target range differs")
        if config["schema_version"] in {SCHEMA_V3, SCHEMA_AEGIS_EE}:
            if protocol.get("paper_rate_adaptation") != {
                "paper_control_frequency_hz": 10,
                "paper_execution_horizon": 16,
                "available_pi05_action_horizon": 10,
                "adaptation": "execute_all_10_available_actions_at_paper_control_frequency",
            }:
                raise ValueError("v3 paper-rate adaptation differs")
    expected_joint = {
        "action_representation_adaptation": "libero_incremental_delta_pose_as_sequential_joint_configuration_trajectory",
        "cartesian_rotation_scale_rad_per_action_unit": 0.5,
        "cartesian_translation_scale_m_per_action_unit": 0.05,
        "flow_euler_steps": 10,
        "jacobian_damping_lambda_pinv": 0.001,
        "joint_limit_margin_rad": 0.02,
        "joint_update_clip_rad": 0.5,
        "noise_initialization_scale_alpha": 0.1,
        "pose_dimensions": 6,
    }
    if config["joint_denoising"] != expected_joint:
        raise ValueError("EmbodiSteer joint-denoising parameters differ")
    if config["nominal_policy"] != {
        "checkpoint": "pi05_libero",
        "frozen": True,
        "same_checkpoint_both_arms": True,
    }:
        raise ValueError("EmbodiSteer joint-baseline checkpoint differs")
    if config["pairing"] != {
        "same_initial_state": True,
        "same_policy_noise_seed_by_query_index": True,
        "same_task_and_obstacle_scene": True,
        "sampler_regression_tolerances": {
            "executed_first_five_gripper_signs_must_match": True,
            "raw_action_units": 0.005,
            "rotation_rad": 0.001,
            "translation_m": 0.0001,
        },
        "settle_actions": 20,
    }:
        raise ValueError("EmbodiSteer joint-baseline pairing contract differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def damped_pseudoinverse(jacobian: Any, damping: float) -> Any:
    """Paper Eq. (3)--(4) damped right pseudoinverse."""

    np = _numpy()
    matrix = np.asarray(jacobian, dtype=np.float64)
    value = float(damping)
    if matrix.shape != (6, 7) or not np.all(np.isfinite(matrix)):
        raise ValueError("Panda end-effector Jacobian must be finite 6x7")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Jacobian damping must be finite and positive")
    regularized = matrix @ matrix.T + value * np.eye(6)
    result = matrix.T @ np.linalg.solve(regularized, np.eye(6))
    if result.shape != (7, 6) or not np.all(np.isfinite(result)):
        raise ValueError("damped pseudoinverse is invalid")
    return result


def rotation_vector_to_matrix(vector: Sequence[float]) -> Any:
    np = _numpy()
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("rotation vector must be finite length three")
    angle = float(np.linalg.norm(value))
    if angle <= 1.0e-12:
        skew = np.array(
            [[0.0, -value[2], value[1]], [value[2], 0.0, -value[0]], [-value[1], value[0], 0.0]]
        )
        return np.eye(3) + skew
    axis = value / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def matrix_to_rotation_vector(matrix: Any) -> Any:
    np = _numpy()
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation matrix must be finite 3x3")
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1.0e-10:
        return 0.5 * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
    if math.pi - angle <= 1.0e-6:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        axis /= np.linalg.norm(axis)
        return angle * axis
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return angle * axis


Kinematics = Callable[[Any], Tuple[Any, Any, Any]]


def _bounded_configuration(
    configuration: Any,
    lower: Any,
    upper: Any,
) -> Any:
    np = _numpy()
    value = np.asarray(configuration, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if value.shape != (7,) or low.shape != (7,) or high.shape != (7,):
        raise ValueError("joint configuration bounds must have length seven")
    return np.clip(value, low, high)


def initialize_joint_trajectory(
    noise_actions: Any,
    start_configuration: Any,
    kinematics: Kinematics,
    *,
    alpha: float,
    damping: float,
    update_clip_rad: float,
    translation_scale_m: float,
    rotation_scale_rad: float,
    lower: Any,
    upper: Any,
) -> tuple[Any, dict[str, Any]]:
    """LIBERO incremental-action adaptation of paper Eq. (3).

    The paper initializes every horizon configuration locally around the
    chunk-start configuration.  pi0.5-LIBERO emits incremental OSC deltas, so
    those deltas are first composed into horizon poses relative to the chunk
    start.  The same start Jacobian is then used for all ten samples, exactly
    as in Eq. (3).
    """

    np = _numpy()
    actions = np.asarray(noise_actions, dtype=np.float64)
    current = np.asarray(start_configuration, dtype=np.float64)
    if actions.shape != (10, 6) or current.shape != (7,):
        raise ValueError("joint initialization input shape differs")
    start_position, start_rotation, start_jacobian = kinematics(current)
    target_positions, target_rotations = incremental_actions_to_world_poses(
        actions,
        start_position,
        start_rotation,
        translation_scale_m=translation_scale_m,
        rotation_scale_rad=rotation_scale_rad,
    )
    pseudoinverse = damped_pseudoinverse(start_jacobian, damping)
    output = []
    raw_norms = []
    clipped_counts = []
    for position, rotation in zip(target_positions, target_rotations):
        twist = np.concatenate(
            (
                position - start_position,
                matrix_to_rotation_vector(rotation @ start_rotation.T),
            )
        )
        raw = pseudoinverse @ twist
        clipped = np.clip(raw, -float(update_clip_rad), float(update_clip_rad))
        sample = _bounded_configuration(
            current + float(alpha) * clipped, lower, upper
        )
        output.append(sample.copy())
        raw_norms.append(float(np.linalg.norm(raw)))
        clipped_counts.append(int(np.count_nonzero(raw != clipped)))
    return np.asarray(output), {
        "maximum_raw_joint_update_l2_rad": float(max(raw_norms)),
        "total_clipped_joint_dimensions": int(sum(clipped_counts)),
    }


def incremental_actions_to_world_poses(
    actions: Any,
    start_position: Any,
    start_rotation: Any,
    *,
    translation_scale_m: float,
    rotation_scale_rad: float,
) -> tuple[Any, Any]:
    """Compose normalized LIBERO OSC deltas into chunk-start pose targets."""

    np = _numpy()
    commands = np.asarray(actions, dtype=np.float64)
    position = np.asarray(start_position, dtype=np.float64).copy()
    rotation = np.asarray(start_rotation, dtype=np.float64).copy()
    if (
        commands.shape != (10, 6)
        or position.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.all(np.isfinite(commands))
    ):
        raise ValueError("incremental pose action input shape differs")
    positions = []
    rotations = []
    for command in commands:
        position = position + float(translation_scale_m) * command[:3]
        rotation = (
            rotation_vector_to_matrix(
                float(rotation_scale_rad) * command[3:6]
            )
            @ rotation
        )
        positions.append(position.copy())
        rotations.append(rotation.copy())
    return np.asarray(positions), np.asarray(rotations)


def joint_trajectory_to_pose_actions(
    trajectory: Any,
    start_configuration: Any,
    kinematics: Kinematics,
    *,
    translation_scale_m: float,
    rotation_scale_rad: float,
) -> Any:
    """Map a sequential joint trajectory to LIBERO incremental OSC actions."""

    np = _numpy()
    joints = np.asarray(trajectory, dtype=np.float64)
    previous = np.asarray(start_configuration, dtype=np.float64)
    if joints.shape != (10, 7) or previous.shape != (7,):
        raise ValueError("joint trajectory shape differs")
    previous_position, previous_rotation, _ = kinematics(previous)
    actions = []
    for configuration in joints:
        position, rotation, _ = kinematics(configuration)
        translation = (position - previous_position) / float(translation_scale_m)
        rotation_error = rotation @ previous_rotation.T
        rotation_action = matrix_to_rotation_vector(rotation_error) / float(
            rotation_scale_rad
        )
        actions.append(np.concatenate((translation, rotation_action)))
        previous = configuration
        previous_position = position
        previous_rotation = rotation
    result = np.asarray(actions)
    if result.shape != (10, 6) or not np.all(np.isfinite(result)):
        raise ValueError("joint trajectory produced invalid pose actions")
    return result


def apply_joint_denoising_residual(
    trajectory: Any,
    start_configuration: Any,
    current_actions: Any,
    next_actions: Any,
    kinematics: Kinematics,
    *,
    damping: float,
    update_clip_rad: float,
    translation_scale_m: float,
    rotation_scale_rad: float,
    lower: Any,
    upper: Any,
) -> tuple[Any, dict[str, Any]]:
    """Apply paper Eq. (4) to one pi0.5 flow-Euler residual."""

    np = _numpy()
    joints = np.asarray(trajectory, dtype=np.float64)
    start = np.asarray(start_configuration, dtype=np.float64)
    before = np.asarray(current_actions, dtype=np.float64)
    after = np.asarray(next_actions, dtype=np.float64)
    if (
        joints.shape != (10, 7)
        or start.shape != (7,)
        or before.shape != (10, 6)
        or after.shape != (10, 6)
    ):
        raise ValueError("joint denoising residual input shape differs")
    start_position, start_rotation, _ = kinematics(start)
    before_positions, before_rotations = incremental_actions_to_world_poses(
        before,
        start_position,
        start_rotation,
        translation_scale_m=translation_scale_m,
        rotation_scale_rad=rotation_scale_rad,
    )
    after_positions, after_rotations = incremental_actions_to_world_poses(
        after,
        start_position,
        start_rotation,
        translation_scale_m=translation_scale_m,
        rotation_scale_rad=rotation_scale_rad,
    )
    output = []
    raw_norms = []
    clipped_counts = []
    for configuration, position_t, rotation_t, position_next, rotation_next in zip(
        joints,
        before_positions,
        before_rotations,
        after_positions,
        after_rotations,
    ):
        _, _, jacobian = kinematics(configuration)
        twist = np.concatenate(
            (
                position_next - position_t,
                matrix_to_rotation_vector(rotation_next @ rotation_t.T),
            )
        )
        raw = damped_pseudoinverse(jacobian, damping) @ twist
        clipped = np.clip(raw, -float(update_clip_rad), float(update_clip_rad))
        updated = _bounded_configuration(configuration + clipped, lower, upper)
        output.append(updated)
        raw_norms.append(float(np.linalg.norm(raw)))
        clipped_counts.append(int(np.count_nonzero(raw != clipped)))
    return np.asarray(output), {
        "maximum_raw_joint_update_l2_rad": float(max(raw_norms)),
        "total_clipped_joint_dimensions": int(sum(clipped_counts)),
    }
