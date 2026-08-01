from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

from main.poisson_fullbody import one_step_counterfactual as core
from main.poisson_fullbody.surface_sampling import COVERAGE_SEMANTICS


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs/vlsa_poisson_one_step_counterfactual.v1.json"
RUNTIME_PATH = REPO_ROOT / "configs/vlsa_poisson_runtime_protocol.canary.v2.json"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _state(boundary: int, qpos: list[float], qvel: list[float]) -> dict:
    cur_time = boundary * 0.002
    integration = [cur_time] + list(qpos) + list(qvel)
    flattened = list(integration)
    return {
        "physical_boundary": boundary,
        "mujoco_state_specification": "mjSTATE_INTEGRATION",
        "integration_state": integration,
        "integration_state_sha256": core._float64_array_sha256(integration),
        "flattened_simulator_state": flattened,
        "flattened_simulator_state_sha256": core._float64_array_sha256(
            flattened
        ),
        "qpos": list(qpos),
        "qpos_sha256": core._float64_array_sha256(qpos),
        "qvel": list(qvel),
        "qvel_sha256": core._float64_array_sha256(qvel),
        "wrapper_bookkeeping": {
            "timestep": boundary,
            "cur_time_s": cur_time,
            "done": False,
        },
    }


def _full_robot_sampling() -> dict:
    sample = {
        "sample_id": 0,
        "body_id": 1,
        "body_name": "robot0_link5",
        "geom_id": 10,
        "geom_name": "robot0_link5_collision",
        "point_body_local_m": [0.0, 0.0, 0.0],
        "source": "exact_box_faces",
    }
    return {
        "sample_count": 1,
        "sample_ledger": [sample],
        "sample_ledger_sha256": core._sha([sample]),
        "geom_records": [
            {
                "geom_id": 10,
                "geom_name": "robot0_link5_collision",
                "body_id": 1,
                "body_name": "robot0_link5",
                "geom_type_id": 6,
                "geom_type_name": "box",
                "geometry_kind": "exact_box_faces",
                "certificate_kind": "analytic_triangle_lattice_covering_bound",
                "contype": 1,
                "conaffinity": 1,
                "mask_collision_enabled": True,
                "selection_authority": "authoritative_resolved_geom_ids",
                "sample_count": 1,
                "surface_element_count": 1,
                "certified_surface_cover_radius_m": 0.01,
                "certificate_parameters": {
                    "requested_epsilon_m": 0.05,
                    "triangle_count": 1,
                },
            }
        ],
        "epsilon_m": 0.05,
        "maximum_surface_cover_radius_m": 0.01,
        "coverage_semantics": COVERAGE_SEMANTICS,
        "rigid_roundtrip": {
            "sample_count": 1,
            "maximum_roundtrip_error_m": 0.0,
            "tolerance_m": 1e-12,
            "passed": True,
        },
    }


def _contact(boundary: int, *, shifted: bool = False) -> dict:
    robot_geom = 11 if shifted else 10
    robot_name = "robot0_link6_collision" if shifted else "robot0_link5_collision"
    robot_body = 3 if shifted else 1
    robot_body_name = "robot0_link6" if shifted else "robot0_link5"
    return {
        "source_phase": "post_integration_recomputed",
        "physical_boundary": boundary,
        "mujoco_contact_index": 0,
        "mujoco_geom1_id": robot_geom,
        "mujoco_geom1_name": robot_name,
        "mujoco_body1_id": robot_body,
        "mujoco_body1_name": robot_body_name,
        "mujoco_geom2_id": 20,
        "mujoco_geom2_name": "moka_pot_collision",
        "mujoco_body2_id": 2,
        "mujoco_body2_name": "moka_pot_obstacle_1",
        "contact_distance_m": -0.001,
        "is_physical_nonpositive_distance_contact": True,
        "solver_constraint_active": True,
        "efc_address": 0,
        "contact_includemargin_m": 0.0,
        "position_world_m": [0.0, 0.0, 0.0],
    }


def _measurement(
    *,
    boundary: int,
    command: list[float],
    gripper: float,
    qpos: list[float],
    qvel: list[float],
    h: float,
    full_distance: float,
    contact: bool = False,
    elapsed_s: float | None = None,
    h_at_B: float = 0.001,
) -> dict:
    integration = [boundary * 0.002] + list(qpos) + list(qvel)
    contacts = [_contact(boundary)] if contact else []
    lower = full_distance - 0.01
    dsim = min([lower, 0.0, -0.001]) if contact else lower
    row = {
        "physical_boundary": boundary,
        "integration_state_sha256": core._float64_array_sha256(integration),
        "integration_state": integration,
        "qpos": list(qpos),
        "qvel": list(qvel),
        "commanded_arm_qdot": list(command),
        "normalized_joint_velocity_action_8d": [
            value / 0.5 for value in command
        ]
        + [gripper],
        "measured_arm_qdot": list(qvel),
        "ordered_sample_h_m2": [h],
        "ordered_sample_grad_h_m": [[1.0, 0.0, 0.0]],
        "ordered_field_query_validity_and_reason": [
            {"sample_index": 0, "sample_id": 0, "valid": True, "reason": None}
        ],
        "ordered_sample_D_opt_m": [0.02],
        "minimum_D_opt_m": 0.02,
        "ordered_full_robot_sample_distance_m": [full_distance],
        "registered_full_robot_sample_count": 1,
        "registered_full_robot_sample_ledger_sha256": _full_robot_sampling()[
            "sample_ledger_sha256"
        ],
        "minimum_D_sim_m": dsim,
        "interval_union_minimum_D_sim_m": dsim,
        "post_state_minimum_D_sim_m": dsim,
        "minimum_exact_full_robot_sample_to_current_obstacle_m": full_distance,
        "certified_full_robot_coverage_radius_m": 0.01,
        "all_mujoco_contact_pairs_with_distances": contacts,
        "exact_ordered_nonpositive_physical_contact_subset": contacts,
        "invalid_field_query_count": 0,
        "selected_obstacle_translation_drift_m": 0.0,
        "selected_obstacle_rotation_drift_rad": 0.0,
        "selected_obstacle_surface_drift_m": 0.0,
        "selected_obstacle_body_velocity_records": [
            {
                "body_id": 2,
                "body_name": "moka_pot_obstacle_1",
                "angular_velocity_world_rad_per_s": [0.0, 0.0, 0.0],
                "linear_velocity_world_m_per_s": [0.0, 0.0, 0.0],
                "angular_speed_rad_per_s": 0.0,
                "linear_speed_m_per_s": 0.0,
            }
        ],
        "selected_obstacle_max_body_linear_speed_m_s": 0.0,
        "selected_obstacle_max_body_angular_speed_rad_s": 0.0,
        "velocity_reference": "mj_objectVelocity_world_orientation_rot_then_lin",
        "target_contact_present": contact,
        "any_robot_to_selected_obstacle_contact_present": contact,
        "any_shifted_robot_to_selected_obstacle_contact_present": False,
        "target_physical_contact_records": contacts,
        "robot_to_selected_obstacle_physical_contact_records": contacts,
        "shifted_robot_to_selected_obstacle_physical_contact_records": [],
    }
    if elapsed_s is not None:
        predicted = h_at_B + elapsed_s * (-command[0])
        row["first_order_predicted_h_m2"] = [predicted]
        row["prediction_error_m2"] = [h - predicted]
    return row


def _restore(state_B: dict, runtime: dict) -> dict:
    field_names = (
        "initial_joint",
        "goal_vel",
        "current_vel",
        "last_err",
        "summed_err",
        "last_joint_vel",
        "torques",
        "new_update",
        "saturated",
        "joint_pos",
        "joint_vel",
        "torque_compensation",
    )
    software_payload = {
        "schema_version": "vlsa_poisson_joint_velocity_controller_software_state.v1",
        "fields": {
            **{
                field: {"available": True, "value": None}
                for field in field_names
            },
            "derivative_ring_buffer": {"available": True, "fields": {}},
        },
    }
    software = dict(software_payload, sha256=core._sha(software_payload))
    return {
        "schema_version": "vlsa_poisson_controller_state_restore.v1",
        "source_controller": "OSC_POSE",
        "target_controller": "JOINT_VELOCITY",
        "model_topology_sha256": _digest("topology"),
        "physical_model_sha256": _digest("physical-model"),
        "compiled_mjb_sha256": _digest("compiled-mjb"),
        "official_integration_state_available": True,
        "official_integration_state_sha256": state_B["integration_state_sha256"],
        "target_official_integration_state_sha256": state_B[
            "integration_state_sha256"
        ],
        "settled_state_sha256": state_B["flattened_simulator_state_sha256"],
        "target_state_sha256": state_B["flattened_simulator_state_sha256"],
        "exact_flattened_state": True,
        "maximum_arm_qpos_error_rad": 0.0,
        "maximum_arm_qvel_error_rad_s": 0.0,
        "max_arm_qpos_error_tolerance_rad": runtime["admissibility"][
            "max_state_restore_qpos_error_rad"
        ],
        "max_arm_qvel_error_tolerance_rad_s": runtime["admissibility"][
            "max_state_restore_qvel_error_rad_s"
        ],
        "copied_timestep": state_B["wrapper_bookkeeping"]["timestep"],
        "copied_cur_time_s": state_B["wrapper_bookkeeping"]["cur_time_s"],
        "copied_done": state_B["wrapper_bookkeeping"]["done"],
        "controller": json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))[
            "joint_velocity_controller_authority"
        ]["expected_restore_controller_contract"],
        "controller_software_state": software,
        "pid_memory_reset": {
            "goal_velocity_zero": True,
            "current_velocity_zero": True,
            "last_error_zero": True,
            "summed_error_zero": True,
            "derivative_buffer_size": 0,
            "saturated": False,
        },
    }


def _arm(
    *,
    name: str,
    boundary: int,
    command: list[float],
    gripper: float,
    state_B: dict,
    runtime: dict,
) -> dict:
    is_nominal = name == "nominal"
    start_qpos = list(state_B["qpos"])
    start = _measurement(
        boundary=boundary,
        command=command,
        gripper=gripper,
        qpos=start_qpos,
        qvel=[0.0] * 7,
        h=0.001,
        full_distance=0.08,
    )
    ledger = []
    nominal_distances = [0.06, 0.04, 0.02, 0.005, 0.005]
    for step in range(1, 6):
        elapsed = step * 0.002
        qpos = [
            start_qpos[index] + command[index] * elapsed
            for index in range(7)
        ]
        h = 0.001 + elapsed * (-command[0])
        ledger.append(
            _measurement(
                boundary=boundary + step,
                command=command,
                gripper=gripper,
                qpos=qpos,
                qvel=list(command),
                h=h,
                full_distance=(
                    nominal_distances[step - 1] if is_nominal else 0.08
                ),
                contact=is_nominal and step == 4,
                elapsed_s=elapsed,
            )
        )
    errors = [[0.0] * 7 for _ in ledger]
    tangent = [
        ledger[-1]["qpos"][index] - start_qpos[index]
        for index in range(7)
    ]
    motion_norm = math.sqrt(sum(value * value for value in tangent))
    command_norm = math.sqrt(sum(value * value for value in command))
    command_integral = command_norm * 0.01
    maximum_drift = {
        "selected_obstacle_translation_drift_m": 0.0,
        "selected_obstacle_rotation_drift_rad": 0.0,
        "selected_obstacle_surface_drift_m": 0.0,
        "selected_obstacle_max_body_linear_speed_m_s": 0.0,
        "selected_obstacle_max_body_angular_speed_rad_s": 0.0,
    }
    return {
        "arm_name": name,
        "controller": "JOINT_VELOCITY_100Hz",
        "filter_solve_count": 0 if is_nominal else 1,
        "held_command_physics_substeps": 5,
        "commanded_arm_qdot": list(command),
        "normalized_joint_velocity_action_8d": [
            value / 0.5 for value in command
        ]
        + [gripper],
        "source_gripper_command": gripper,
        "hidden_clipping_applied": False,
        "restore": _restore(state_B, runtime),
        "start_boundary": start,
        "physics_substep_ledger": ledger,
        "tracking": {
            "error_ledger_rad_s": errors,
            "linf_rad_s": 0.0,
            "rmse_rad_s": 0.0,
            "registered_linf_max_rad_s": runtime["admissibility"][
                "max_joint_velocity_tracking_linf_rad_s"
            ],
            "registered_rmse_max_rad_s": runtime["admissibility"][
                "max_joint_velocity_tracking_rmse_rad_s"
            ],
            "within_registered_thresholds": True,
        },
        "motion": {
            "measured_arm_tangent_displacement": tangent,
            "measured_arm_motion_l2_rad": motion_norm,
            "integrated_measured_joint_speed_l2_rad": command_integral,
            "command_norm_rad_s": command_norm,
            "command_integral_over_horizon_rad": command_integral,
            "measured_motion_to_command_integral_ratio": (
                motion_norm / command_integral
            ),
            "nonzero_command": command_norm > 0.0,
            "nonzero_measured_joint_motion": motion_norm > 0.0,
        },
        "maximum_selected_obstacle_drift": maximum_drift,
        "static_field_admissible_for_complete_interval": True,
        "minimum_h_m2": min(row["ordered_sample_h_m2"][0] for row in ledger),
        "invalid_field_query_count": 0,
        "minimum_D_sim_m": min(row["minimum_D_sim_m"] for row in ledger),
        "any_target_contact": is_nominal,
        "any_robot_to_selected_obstacle_contact": is_nominal,
        "any_shifted_robot_to_selected_obstacle_contact": False,
    }


def _resolved_geometry() -> dict:
    return {
        "robot_root_body_ids": [1],
        "robot_body_ids": [1],
        "obstacle_root_body_ids": [2],
        "obstacle_body_ids": [2],
        "link56_body_ids": [1],
        "robot_geom_ids": [10],
        "obstacle_geom_ids": [20],
        "link56_geom_ids": [10],
        "collision_enabled_pairs": [[10, 20]],
        "robot_body_names": ["robot0_link5"],
        "obstacle_body_names": ["moka_pot_obstacle_1"],
        "robot_geom_names": ["robot0_link5_collision"],
        "obstacle_geom_names": ["moka_pot_collision"],
        "link56_geom_names": ["robot0_link5_collision"],
    }


def build_valid_one_step_fixture() -> tuple[dict, dict, str, dict]:
    """Return a complete strong fixture and its three external authorities.

    This is intentionally public to the test suite: the independent artifact
    consumer tests reuse it so both validators exercise the real core contract.
    """

    protocol_raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_raw)
    protocol_raw_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    runtime = json.loads(RUNTIME_PATH.read_text())
    boundary = 5
    contact_boundary = 9
    gripper = 1.0
    nominal_command = [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    safe_command = [0.005, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    qpos_B = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
    state_B = _state(boundary, qpos_B, [0.0] * 7)
    state_B5 = _state(
        boundary + 5,
        [
            qpos_B[index] + nominal_command[index] * 0.01
            for index in range(7)
        ],
        [0.0] * 7,
    )
    callback_ledger = [_digest("callback-%d" % index) for index in range(10)]
    callback_ledger[boundary - 1] = state_B["integration_state_sha256"]
    callback_ledger[boundary + 5 - 1] = state_B5["integration_state_sha256"]

    protected_sample = {
        "sample_id": 0,
        "body_id": 1,
        "body_name": "robot0_link5",
        "geom_id": 10,
        "geom_name": "robot0_link5_collision",
        "point_body_local_m": [0.0, 0.0, 0.0],
        "source": "exact_box_faces",
    }
    protected_component = {
        "geom_id": 10,
        "geom_name": "robot0_link5_collision",
        "body_id": 1,
        "body_name": "robot0_link5",
        "geometry_kind": "exact_box_faces",
        "certificate_kind": "analytic_triangle_lattice_covering_bound",
        "surface_element_count": 1,
        "sample_count": 1,
        "certified_surface_cover_radius_m": 0.01,
    }
    protected_payload = {
        "epsilon_m": 0.05,
        "maximum_surface_cover_radius_m": 0.01,
        "coverage_semantics": "fixture_strict_surface_cover",
        "components": [protected_component],
        "samples": [protected_sample],
    }
    protected_hash = core._sha(protected_payload)
    field_hashes = {
        "bundle_sha256": _digest("field-bundle"),
        "protected_samples_sha256": protected_hash,
    }
    differential_validation = {
        "schema_version": core.DIFFERENTIAL_AUDIT_SCHEMA,
        "audit_payload_sha256": _digest("differential-audit"),
        "ordered_sample_identity_sha256": core._sha([protected_sample]),
        "integration_state_sha256": _digest("settled-state"),
        "specification_sha256": _digest("differential-spec"),
        "binding_sha256": _digest("differential-binding"),
        "classification_ledger_sha256": _digest("differential-classification"),
        "counts": {"sample_count": 1, "passed_sample_count": 1},
        "passed": True,
    }
    full_sampling = _full_robot_sampling()
    resolved = _resolved_geometry()
    physical_model = {
        "schema_version": "vlsa_poisson_physical_model.v3",
        "sha256": _digest("physical-model"),
        "field_count": 100,
        "option_field_count": 10,
        "compiled_mjb_sha256": _digest("compiled-mjb"),
        "compiled_mjb_bytes": 4096,
        "nq": 7,
        "nv": 7,
        "na": 0,
        "mjstate_integration_size": 15,
        "robosuite_flattened_state_size": 15,
        "robosuite_flattened_state_layout": "time_qpos_qvel_act_no_udd_tail",
    }
    target_contact = {
        "observation_index": contact_boundary - 1,
        "source_phase": "post_integration_recomputed",
        "mujoco_contact_index": 0,
        "robot_geom_id": 10,
        "robot_geom_name": "robot0_link5_collision",
        "robot_body_id": 1,
        "robot_body_name": "robot0_link5",
        "obstacle_geom_id": 20,
        "obstacle_geom_name": "moka_pot_collision",
        "obstacle_body_id": 2,
        "obstacle_body_name": "moka_pot_obstacle_1",
        "contact_distance_m": -0.001,
        "is_physical_nonpositive_distance_contact": True,
    }
    warning = {
        "observation_index": 0,
        "geom_id": 10,
        "geom_name": "robot0_link5_collision",
        "body_id": 1,
        "body_name": "robot0_link5",
        "signal_kind": "cbf_lhs_negative",
        "evidence": {
            "sample_id": 0,
            "geom_id": 10,
            "body_id": 1,
            "observed_cbf_lhs_m2_per_s": -0.1,
        },
    }
    full_sampling_external = {
        key: value for key, value in full_sampling.items() if key != "sample_ledger"
    }
    identification = {
        "first_link56_contact": target_contact,
        "primary_registered_warning": warning,
        "contact_model_authority_sha256": _digest("contact-authority"),
        "physical_model": physical_model,
        "resolved_geometry": resolved,
        "field_bundle_hashes": field_hashes,
        "ordered_protected_sample_ledger_sha256": protected_hash,
        "protected_sample_count": 1,
        "arm_dof_indices": list(range(7)),
        "settled_mjstate_integration_sha256": _digest("settled-state"),
        "differential_binding_sha256": _digest("differential-binding"),
        "differential_classification_ledger_sha256": _digest(
            "differential-classification"
        ),
        "differential_validation": differential_validation,
        "full_robot_measurement_sampling": full_sampling_external,
        "callback_state_read_only_count": len(callback_ledger),
        "callback_state_read_only_after_sha256_ledger": callback_ledger,
        "callback_state_sequence_sha256": core._sha(callback_ledger),
    }
    prerequisites = protocol["prerequisites"]
    authority = {
        "expected_code_commit": "a" * 40,
        "expected_run_id": "fixture-one-step-strong",
        "expected_slurm_job_id": "12345",
        "expected_host": "h100-worker-fixture",
        "expected_device": "H100",
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "file_sha256": protocol_raw_sha256,
            "semantic_sha256": core._sha(protocol),
            "schema_version": core.PROTOCOL_SCHEMA,
            "protocol_id": protocol["protocol_id"],
        },
        "numeric_prerequisite": {
            "path": "/fixture/numeric.json",
            "file_sha256": _digest("numeric-file"),
            "payload_sha256": _digest("numeric-payload"),
            "schema_version": "vlsa_poisson_numeric_validation.v1",
            "status": "passed",
        },
        "parity_prerequisite": {
            "path": "/fixture/parity.json",
            "file_sha256": _digest("parity-file"),
            "payload_sha256": _digest("parity-payload"),
            "schema_version": "vlsa_poisson_shadow_parity.v2",
            "status": "passed",
        },
        "identification_prerequisite": {
            "path": "/fixture/identification.json",
            "file_sha256": _digest("identification-file"),
            "payload_sha256": _digest("identification-payload"),
            "schema_version": "vlsa_poisson_shadow_identification.v3",
            "status": "passed",
        },
        "dynamic_authority": {
            "case": {
                "case_id": protocol["case"]["case_id"],
                "manifest_file_sha256": _digest("manifest-file"),
                "manifest_row_sha256": _digest("manifest-row"),
            },
            "selection": {
                "relative_path": prerequisites["selection_protocol_relative_path"],
                "schema_version": prerequisites["selection_schema_version"],
                "file_sha256": prerequisites["selection_raw_file_sha256"],
                "protocol_id": prerequisites["selection_protocol_id"],
            },
            "historical": {
                "result_file_sha256": _digest("historical-file"),
                "result_payload_sha256": _digest("historical-payload"),
                "action_count": 1,
                "executed_action_sequence_sha256": _digest("historical-actions"),
                "policy_noise_schedule_sha256": _digest("policy-noise"),
            },
            "runtime": {
                "raw_file_sha256": prerequisites["runtime_raw_file_sha256"],
                "semantic_sha256": prerequisites[
                    "runtime_semantic_protocol_sha256"
                ],
                "parameter_block_sha256": prerequisites[
                    "runtime_parameter_block_sha256"
                ],
                "schema_version": prerequisites["runtime_schema_version"],
                "protocol_id": prerequisites["runtime_protocol_id"],
                "registered_parameters": {
                    key: runtime[key]
                    for key in ("admissibility", "coverage", "cbf", "qp", "cadence")
                },
            },
            "parity": {
                "executed_action_count": 1,
                "action_boundary_state_sha256_ledger": [_digest("action-state")],
                "state_sequence_sha256": core._sha([_digest("action-state")]),
                "official_integration_state_count": len(callback_ledger),
                "official_integration_state_sha256_ledger": callback_ledger,
                "official_integration_state_sequence_sha256": core._sha(
                    callback_ledger
                ),
            },
            "identification": identification,
        },
    }
    provenance = {
        "source": {
            "commit": authority["expected_code_commit"],
            "branch": "codex/full-body-poisson-cbf-feasibility",
            "status_short": [],
        },
        "allocation": {
            "execution_environment": "slurm_allocation",
            "slurm_job_id": authority["expected_slurm_job_id"],
            "host_name": authority["expected_host"],
            "device": {
                "device_type": "cuda",
                "visible_device_ids": ["0"],
                "model": "NVIDIA H100 80GB HBM3",
                "uuid": "GPU-fixture",
                "driver_version": "fixture-driver",
                "cuda_runtime_version": "fixture-cuda",
            },
        },
        "manifest": {
            "path": "/fixture/manifest.jsonl",
            "file_sha256": authority["dynamic_authority"]["case"][
                "manifest_file_sha256"
            ],
            "row_sha256": authority["dynamic_authority"]["case"][
                "manifest_row_sha256"
            ],
            "case_id": protocol["case"]["case_id"],
        },
        "selection": {
            "path": "/fixture/" + prerequisites["selection_protocol_relative_path"],
            "schema_version": prerequisites["selection_schema_version"],
            "file_sha256": prerequisites["selection_raw_file_sha256"],
            "protocol_id": prerequisites["selection_protocol_id"],
        },
        "runtime_protocol": {
            "path": "/fixture/" + prerequisites["runtime_protocol_relative_path"],
            "raw_file_sha256": prerequisites["runtime_raw_file_sha256"],
            "semantic_sha256": prerequisites[
                "runtime_semantic_protocol_sha256"
            ],
            "parameter_block_sha256": prerequisites[
                "runtime_parameter_block_sha256"
            ],
            "schema_version": prerequisites["runtime_schema_version"],
            "protocol_id": prerequisites["runtime_protocol_id"],
        },
        "historical_result": {
            "path": "/fixture/historical/result.json",
            "file_sha256": authority["dynamic_authority"]["historical"][
                "result_file_sha256"
            ],
            "payload_sha256": authority["dynamic_authority"]["historical"][
                "result_payload_sha256"
            ],
            "action_count": 1,
            "executed_action_sequence_sha256": authority["dynamic_authority"][
                "historical"
            ]["executed_action_sequence_sha256"],
            "policy_noise_schedule_sha256": authority["dynamic_authority"][
                "historical"
            ]["policy_noise_schedule_sha256"],
        },
        "physical_model": physical_model,
        "joint_velocity_controller": protocol[
            "joint_velocity_controller_authority"
        ]["expected_restore_controller_contract"],
        "contact_model": {
            "authority_sha256": _digest("contact-authority"),
            "active_obstacle_name": protocol["case"]["selected_obstacle_name"],
            "resolved_geometry": resolved,
        },
        "field_bundle": {
            "bundle_sha256": field_hashes["bundle_sha256"],
            "protected_samples_sha256": protected_hash,
            "protocol_sha256": prerequisites[
                "runtime_semantic_protocol_sha256"
            ],
            "parameter_block_sha256": prerequisites[
                "runtime_parameter_block_sha256"
            ],
        },
        "execution": {
            "active_policy_query_count": 0,
            "policy_server_started": False,
            "source_prefix_replayed_once": True,
        },
        "python": {"executable": "/fixture/python", "version": "3.fixture"},
    }

    prefix_hash = core._sha(callback_ledger)
    exact_prefix = {
        "execution": "one_exact_historical_OSC_prefix_no_policy_query",
        "observed_callback_count": len(callback_ledger),
        "expected_callback_count": len(callback_ledger),
        "observed_callback_sha256_ledger": callback_ledger,
        "observed_callback_prefix_sha256": prefix_hash,
        "parity_callback_prefix_sha256": prefix_hash,
        "identification_callback_prefix_sha256": prefix_hash,
        "historical_executed_action_sequence_sha256": authority[
            "dynamic_authority"
        ]["historical"]["executed_action_sequence_sha256"],
        "state_at_B": state_B,
        "state_at_B_plus_5": state_B5,
    }
    source_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]
    gripper_evidence = {
        "source_action_index": 0,
        "source_action_7d": source_action,
        "source_action_sha256": core._float64_array_sha256(source_action),
        "exact_gripper_value": gripper,
        "exact_gripper_value_sha256": core._sha(gripper),
    }
    shadow_binding = {
        "contact_model_authority_sha256": _digest("contact-authority"),
        "physical_model": physical_model,
        "resolved_geometry": resolved,
        "field_bundle_hashes": field_hashes,
        "field_bundle_sha256": field_hashes["bundle_sha256"],
        "ordered_protected_sample_ledger": [protected_sample],
        "ordered_protected_sample_ledger_sha256": protected_hash,
        "protected_sample_count": 1,
        "protected_surface_components": [protected_component],
        "protected_sampling_epsilon_m": 0.05,
        "protected_sampling_maximum_surface_cover_radius_m": 0.01,
        "protected_sampling_coverage_semantics": "fixture_strict_surface_cover",
        "full_robot_measurement_sampling": full_sampling,
        "arm_dof_indices": list(range(7)),
        "settled_mjstate_integration_sha256": _digest("settled-state"),
        "settled_link56_differential_audit_binding_sha256": _digest(
            "differential-binding"
        ),
        "settled_link56_differential_audit_classification_ledger_sha256": _digest(
            "differential-classification"
        ),
        "settled_link56_differential_audit_validation": differential_validation,
        "fresh_stable_construction_binding_equals_identification": True,
    }
    boundary_measurement = _measurement(
        boundary=boundary,
        command=nominal_command,
        gripper=gripper,
        qpos=list(state_B["qpos"]),
        qvel=[0.0] * 7,
        h=0.001,
        full_distance=0.08,
    )
    static_thresholds = {
        "translation_m": runtime["admissibility"][
            "max_selected_geom_translation_drift_m"
        ],
        "rotation_rad": runtime["admissibility"][
            "max_selected_geom_rotation_drift_rad"
        ],
        "surface_m": runtime["admissibility"][
            "max_selected_geom_surface_drift_m"
        ],
        "linear_speed_m_s": runtime["admissibility"][
            "max_selected_body_linear_speed_m_s"
        ],
        "angular_speed_rad_s": runtime["admissibility"][
            "max_selected_body_angular_speed_rad_s"
        ],
    }
    boundary_preflight = {
        "strict_all_sample_h_positive": True,
        "strict_D_sim_positive": True,
        "zero_robot_selected_obstacle_contact": True,
        "invalid_field_query_count": 0,
        "selected_obstacle_static_observed": {
            "translation_m": 0.0,
            "rotation_rad": 0.0,
            "surface_m": 0.0,
            "linear_speed_m_s": 0.0,
            "angular_speed_rad_s": 0.0,
        },
        "selected_obstacle_static_thresholds": static_thresholds,
        "selected_obstacle_static_admissible": True,
        "QP_or_arm_physics_executed_before_this_gate": False,
        "passed": True,
        "measurement": boundary_measurement,
    }
    source_boundary = {
        "target_contact": target_contact,
        "contact_physical_boundary_C": contact_boundary,
        "filter_boundary_B": boundary,
        "exact_prefix": exact_prefix,
        "source_restore": {
            "method": "mj_setState_forward_mj_setState_plus_wrapper_bookkeeping",
            "integration_state_sha256": state_B["integration_state_sha256"],
            "flattened_simulator_state_sha256": state_B[
                "flattened_simulator_state_sha256"
            ],
            "exact_official_integration_state": True,
            "exact_flattened_simulator_state": True,
            "wrapper_bookkeeping": state_B["wrapper_bookkeeping"],
        },
        "gripper_evidence": gripper_evidence,
        "warning_authorization": {
            "primary_registered_warning": warning,
            "warning_physical_boundary": 1,
            "contact_physical_boundary": contact_boundary,
            "lead_physics_substeps": contact_boundary - 1,
            "lead_time_s": (contact_boundary - 1) * 0.002,
            "next_scheduled_filter_boundary": 5,
            "next_filter_boundary_strictly_before_contact": True,
            "invalid_query_warning_used_for_authorization": False,
            "passed": True,
        },
        "boundary_B_admissibility": boundary_preflight,
        "shadow_construction_binding": shadow_binding,
    }
    nominal_velocity = {
        "method": "mujoco_mj_differentiatePos_full_nv",
        "interval_s": 0.01,
        "qpos_B_sha256": state_B["qpos_sha256"],
        "qpos_B_plus_5_sha256": state_B5["qpos_sha256"],
        "full_nv_count": 7,
        "arm_dof_indices": list(range(7)),
        "arm_qpos_indices": list(range(7)),
        "arm_joint_types": ["hinge"] * 7,
        "arm_joint_qpos_widths": [1] * 7,
        "qdot_nom_full_nv": nominal_command,
        "qdot_nom_arm_slice": nominal_command,
        "instantaneous_source_qvel_full_nv_at_B": [0.0] * 7,
        "instantaneous_source_arm_qvel_at_B": [0.0] * 7,
        "estimated_minus_instantaneous_arm_qvel": nominal_command,
        "registered_lower_rad_s": runtime["qp"]["velocity_lower_rad_s"],
        "registered_upper_rad_s": runtime["qp"]["velocity_upper_rad_s"],
        "finite": True,
        "arm_velocity_within_registered_bounds": True,
        "bound_violation_indices": [],
        "lower_bound_excess_rad_s": [0.0] * 7,
        "upper_bound_excess_rad_s": [0.0] * 7,
        "maximum_bound_excess_rad_s": 0.0,
        "out_of_bounds_policy": "inadmissible_no_hidden_clipping",
        "hidden_clipping_applied": False,
    }
    nominal_velocity["nominal_derivation_sha256"] = core._sha(nominal_velocity)

    q_min = protocol["qp_execution"]["expected_arm_q_min_rad"]
    q_max = protocol["qp_execution"]["expected_arm_q_max_rad"]
    q_arm = list(state_B["qpos"])
    margin = runtime["qp"]["joint_position_margin_rad"]
    alpha_joint = runtime["qp"]["joint_limit_alpha_per_s"]
    allowed_lower = [value + margin for value in q_min]
    allowed_upper = [value - margin for value in q_max]
    continuous_lower = [
        -alpha_joint * (q_arm[index] - q_min[index]) for index in range(7)
    ]
    continuous_upper = [
        alpha_joint * (q_max[index] - q_arm[index]) for index in range(7)
    ]
    one_step_lower = [
        (allowed_lower[index] - q_arm[index]) / 0.01 for index in range(7)
    ]
    one_step_upper = [
        (allowed_upper[index] - q_arm[index]) / 0.01 for index in range(7)
    ]
    physical_lower = runtime["qp"]["velocity_lower_rad_s"]
    physical_upper = runtime["qp"]["velocity_upper_rad_s"]
    final_lower = [
        max(physical_lower[index], continuous_lower[index], one_step_lower[index])
        for index in range(7)
    ]
    final_upper = [
        min(physical_upper[index], continuous_upper[index], one_step_upper[index])
        for index in range(7)
    ]
    nominal_residual = -nominal_command[0] + 5.0 * 0.001
    safe_residual = -safe_command[0] + 5.0 * 0.001
    correction = math.sqrt(
        sum(
            (safe_command[index] - nominal_command[index]) ** 2
            for index in range(7)
        )
    )
    boundary_filter = {
        "arm_dof_indices": list(range(7)),
        "ordered_sample_identity": [protected_sample],
        "ordered_sample_count": 1,
        "ordered_sample_ledger_sha256": protected_hash,
        "ordered_sample_points_world": [[0.0, 0.0, 0.0]],
        "ordered_sample_point_jacobians": [
            [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 7, [0.0] * 7]
        ],
        "ordered_sample_h_m2": [0.001],
        "ordered_sample_grad_h_m": [[1.0, 0.0, 0.0]],
        "ordered_sample_D_opt_m": [0.02],
        "velocity_lower_rad_s": final_lower,
        "velocity_upper_rad_s": final_upper,
        "one_CBF_row_per_exact_bound_sample": {
            "row_count": 1,
            "sample_count": 1,
            "rows_m_per_rad": [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "lower_bounds_m2_per_s": [-0.005],
            "alpha_gain_per_s": 5.0,
            "no_slack": True,
        },
        "joint_velocity_bound_rows": {
            "physical_lower_rad_s": physical_lower,
            "physical_upper_rad_s": physical_upper,
            "final_lower_rad_s": final_lower,
            "final_upper_rad_s": final_upper,
        },
        "joint_position_constraint_rows": {
            "q_arm_rad": q_arm,
            "q_min_rad": q_min,
            "q_max_rad": q_max,
            "position_margin_rad": margin,
            "allowed_lower_rad": allowed_lower,
            "allowed_upper_rad": allowed_upper,
            "joint_limit_alpha_per_s": alpha_joint,
            "continuous_lower_rad_s": continuous_lower,
            "continuous_upper_rad_s": continuous_upper,
            "one_step_dt_s": 0.01,
            "one_step_lower_rad_s": one_step_lower,
            "one_step_upper_rad_s": one_step_upper,
        },
        "nominal_CBF_residuals_m2_per_s": [nominal_residual],
        "safe_CBF_residuals_m2_per_s": [safe_residual],
        "qdot_safe": safe_command,
        "correction_norm": correction,
        "QP_status": "solved",
        "QP_iterations": 25,
        "QP_postcheck": {
            "solver": "osqp",
            "status": "solved",
            "status_value": 1,
            "iterations": 25,
            "solve_time_seconds": 0.001,
            "input_constraint_count": 1,
            "solved_constraint_count": 1,
            "trivial_zero_constraint_count": 0,
            "minimum_normalized_cbf_residual": safe_residual,
            "minimum_raw_cbf_residual_m2_per_s": safe_residual,
            "minimum_nonzero_row_scale_m2_per_rad": 1.0,
            "maximum_nonzero_row_scale_m2_per_rad": 1.0,
            "maximum_velocity_bound_violation_rad_s": 0.0,
            "nominal_minimum_normalized_cbf_residual": nominal_residual,
            "nominal_minimum_raw_cbf_residual_m2_per_s": nominal_residual,
            "correction_l2_rad_s": correction,
            "nominal_feasible": False,
        },
        "QP_reason": "solved",
        "trend_sample_index": 0,
        "trend_sample_id": 0,
        "trend_sample_robot_geom_id": 10,
        "trend_sample_nominal_residual_m2_per_s": nominal_residual,
    }
    nominal_arm = _arm(
        name="nominal",
        boundary=boundary,
        command=nominal_command,
        gripper=gripper,
        state_B=state_B,
        runtime=runtime,
    )
    psf_arm = _arm(
        name="psf",
        boundary=boundary,
        command=safe_command,
        gripper=gripper,
        state_B=state_B,
        runtime=runtime,
    )
    static_scopes = protocol["acceptance"]["static_field_admissibility_scopes"]
    tracking_scopes = protocol["acceptance"]["tracking_admissibility_scopes"]
    zero_maxima = {
        "translation_m": 0.0,
        "rotation_rad": 0.0,
        "surface_m": 0.0,
        "linear_speed_m_s": 0.0,
        "angular_speed_rad_s": 0.0,
    }
    static_windows = {
        "scope_protocol": static_scopes,
        "nominal": {
            "scope": static_scopes["nominal_with_selected_obstacle_contact"],
            "cutoff_physical_boundary_exclusive": contact_boundary,
            "included_physical_boundaries": [5, 6, 7, 8],
            "observed_maxima": zero_maxima,
            "thresholds": static_thresholds,
            "admissible": True,
        },
        "psf": {
            "scope": static_scopes["poisson_filtered_velocity"],
            "cutoff_physical_boundary_exclusive": None,
            "included_physical_boundaries": [5, 6, 7, 8, 9, 10],
            "observed_maxima": zero_maxima,
            "thresholds": static_thresholds,
            "admissible": True,
        },
    }
    tracking_thresholds = {
        "linf_rad_s": protocol["acceptance"][
            "joint_velocity_tracking_linf_max_rad_s"
        ],
        "rmse_rad_s": protocol["acceptance"][
            "joint_velocity_tracking_rmse_max_rad_s"
        ],
    }
    tracking_windows = {
        "scope_protocol": tracking_scopes,
        "nominal": {
            "scope": tracking_scopes[
                "nominal_with_selected_obstacle_contact"
            ],
            "cutoff_physical_boundary_exclusive": contact_boundary,
            "included_physical_boundaries": [6, 7, 8],
            "linf_rad_s": 0.0,
            "rmse_rad_s": 0.0,
            "thresholds": tracking_thresholds,
            "admissible": True,
        },
        "psf": {
            "scope": tracking_scopes["poisson_filtered_velocity"],
            "cutoff_physical_boundary_exclusive": None,
            "included_physical_boundaries": [6, 7, 8, 9, 10],
            "linf_rad_s": 0.0,
            "rmse_rad_s": 0.0,
            "thresholds": tracking_thresholds,
            "admissible": True,
        },
    }
    nominal_norm = nominal_arm["motion"]["command_norm_rad_s"]
    safe_norm = psf_arm["motion"]["command_norm_rad_s"]
    threshold_fields = (
        "minimum_filter_correction_norm_rad_s",
        "minimum_safe_command_norm_rad_s",
        "minimum_safe_to_nominal_command_norm_ratio",
        "minimum_safe_measured_joint_motion_rad",
        "minimum_safe_measured_motion_to_command_integral_ratio",
    )
    diagnostics = {
        "trend_sample_index": 0,
        "trend_sample_id": 0,
        "h_at_B_m2": 0.001,
        "first_order_predicted_h_after_horizon_m2": 0.0,
        "actual_h_after_each_physics_substep_m2": [
            row["ordered_sample_h_m2"][0]
            for row in nominal_arm["physics_substep_ledger"]
        ],
        "psf_h_after_horizon_m2": psf_arm["physics_substep_ledger"][-1][
            "ordered_sample_h_m2"
        ][0],
        "psf_minus_nominal_h_after_horizon_m2": psf_arm[
            "physics_substep_ledger"
        ][-1]["ordered_sample_h_m2"][0],
        "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon": True,
        "prediction_error_m2_without_acceptance_threshold": [0.0] * 5,
        "prediction_error_policy": "diagnostic_only_no_post_hoc_tolerance",
        "nominal_D_sim_at_B_m": 0.07,
        "nominal_minimum_D_sim_over_horizon_m": -0.005,
        "trend_only_conditions_all_true_without_target_contact": False,
        "historical_identification_contact_boundary_C": contact_boundary,
        "historical_contact_boundary_within_horizon": True,
        "counterfactual_nominal_first_target_contact_boundary_C_nom": contact_boundary,
        "counterfactual_nominal_first_any_robot_selected_obstacle_contact_boundary_C_any_nom": contact_boundary,
        "counterfactual_nominal_target_contact_within_horizon": True,
        "nominal_clearance_at_C_nom_m": -0.005,
        "psf_clearance_at_C_nom_m": 0.07,
        "static_field_admissibility_windows": static_windows,
        "tracking_admissibility_windows": tracking_windows,
        "local_motion_retention": {
            "filter_correction_norm_rad_s": correction,
            "nominal_command_norm_rad_s": nominal_norm,
            "safe_command_norm_rad_s": safe_norm,
            "safe_to_nominal_command_norm_ratio": safe_norm / nominal_norm,
            "safe_measured_joint_motion_rad": psf_arm["motion"][
                "measured_arm_motion_l2_rad"
            ],
            "safe_command_integral_over_horizon_rad": psf_arm["motion"][
                "command_integral_over_horizon_rad"
            ],
            "safe_measured_motion_to_command_integral_ratio": psf_arm["motion"][
                "measured_motion_to_command_integral_ratio"
            ],
            "thresholds": {
                key: protocol["acceptance"][key] for key in threshold_fields
            },
            "interpretation": protocol["acceptance"][
                "local_motion_interpretation"
            ],
        },
    }
    local_keys = (
        "nominal_velocity_within_registered_controller_envelope",
        "no_hidden_clipping",
        "boundary_B_preflight_admissible",
        "complete_exposure",
        "exact_paired_start",
        "qp_valid",
        "both_tracking_valid",
        "static_field_admissible",
        "filter_correction_above_registered_minimum",
        "safe_command_above_registered_minimum",
        "safe_command_retains_registered_nominal_norm_fraction",
        "safe_measured_motion_above_registered_minimum",
        "safe_measured_motion_retains_registered_command_integral_fraction",
        "safe_all_queries_valid",
        "safe_minimum_h_nonnegative",
        "safe_minimum_D_sim_strictly_positive",
        "safe_all_robot_selected_obstacle_contact_absent",
    )
    classification = {
        "label": protocol["acceptance"]["strong_result_label"],
        "stage_13_passed": True,
        "contact_prevention_observed": True,
        "claim_scope": (
            "one_100hz_interval_local_geometry_field_jacobian_qp_and_tracking_feasibility_not_task_success_or_full_episode_safety"
        ),
        "local_feasibility_conditions": {key: True for key in local_keys},
        "nominal_reference": {
            "kind": "target_contact_within_horizon",
            "conditions": {
                "nominal_target_contact_within_horizon": True,
                "nominal_first_selected_obstacle_contact_matches_target_boundary": True,
                "safe_clearance_greater_at_nominal_target_contact_boundary": True,
            },
            "passed": True,
        },
        "typed_reasons": [],
    }
    counts = {
        "bound_sample_count": 1,
        "source_prefix_callback_count": len(callback_ledger),
        "nominal_physics_substep_count": 5,
        "psf_physics_substep_count": 5,
        "qp_solve_count": 1,
        "nominal_invalid_field_query_count": 0,
        "psf_invalid_field_query_count": 0,
        "nominal_target_contact_record_count": 1,
        "psf_robot_selected_obstacle_contact_record_count": 0,
    }
    protocol_binding = {
        "raw_file_sha256": protocol_raw_sha256,
        "semantic_sha256": core._sha(protocol),
        "schema_version": core.PROTOCOL_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "result_schema_version": core.RESULT_SCHEMA,
    }
    result = {
        "schema_version": core.RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "run_id": authority["expected_run_id"],
        "case_id": protocol["case"]["case_id"],
        "execution": {
            "outcome_kind": "paired_counterfactual_complete",
            "source_prefix_complete": True,
            "qp_executed": True,
            "paired_joint_velocity_physics_executed": True,
            "no_hidden_clipping": True,
        },
        "protocol_binding": protocol_binding,
        "authority": authority,
        "provenance": provenance,
        "source_boundary": source_boundary,
        "nominal_velocity_estimate": nominal_velocity,
        "boundary_B_filter": boundary_filter,
        "arms": [nominal_arm, psf_arm],
        "diagnostics": diagnostics,
        "counts": counts,
        "classification": classification,
        "passed": True,
    }
    _rehash(result)
    return result, protocol, protocol_raw_sha256, authority


def _rehash(result: dict) -> None:
    nominal = result.get("nominal_velocity_estimate")
    if isinstance(nominal, dict) and "nominal_derivation_sha256" in nominal:
        unhashed = dict(nominal)
        unhashed.pop("nominal_derivation_sha256")
        nominal["nominal_derivation_sha256"] = core._sha(unhashed)
    section_fields = {
        "protocol_binding_sha256": "protocol_binding",
        "authority_sha256": "authority",
        "provenance_sha256": "provenance",
        "nominal_velocity_estimate_sha256": "nominal_velocity_estimate",
        "boundary_B_filter_sha256": "boundary_B_filter",
        "arm_ledger_sha256": "arms",
        "classification_ledger_sha256": "classification",
    }
    for digest_field, section_field in section_fields.items():
        result[digest_field] = core._sha(result[section_field])
    payload = dict(result)
    payload.pop("result_payload_sha256", None)
    result["result_payload_sha256"] = core._sha(payload)


def build_oob_one_step_fixture() -> tuple[dict, dict, str, dict]:
    result, protocol, protocol_raw_sha256, authority = (
        build_valid_one_step_fixture()
    )
    prefix = result["source_boundary"]["exact_prefix"]
    endpoint = prefix["state_at_B_plus_5"]
    endpoint["qpos"][0] = 0.006
    endpoint["qpos_sha256"] = core._float64_array_sha256(endpoint["qpos"])
    endpoint["integration_state"][1] = 0.006
    endpoint["integration_state_sha256"] = core._float64_array_sha256(
        endpoint["integration_state"]
    )
    endpoint["flattened_simulator_state"][1] = 0.006
    endpoint["flattened_simulator_state_sha256"] = core._float64_array_sha256(
        endpoint["flattened_simulator_state"]
    )
    prefix["observed_callback_sha256_ledger"][-1] = endpoint[
        "integration_state_sha256"
    ]
    prefix_hash = core._sha(prefix["observed_callback_sha256_ledger"])
    for field in (
        "observed_callback_prefix_sha256",
        "parity_callback_prefix_sha256",
        "identification_callback_prefix_sha256",
    ):
        prefix[field] = prefix_hash
    parity = result["authority"]["dynamic_authority"]["parity"]
    parity["official_integration_state_sha256_ledger"] = list(
        prefix["observed_callback_sha256_ledger"]
    )
    parity["official_integration_state_sequence_sha256"] = prefix_hash
    identification = result["authority"]["dynamic_authority"]["identification"]
    identification["callback_state_read_only_after_sha256_ledger"] = list(
        prefix["observed_callback_sha256_ledger"]
    )
    identification["callback_state_sequence_sha256"] = prefix_hash

    nominal = result["nominal_velocity_estimate"]
    nominal["qpos_B_plus_5_sha256"] = endpoint["qpos_sha256"]
    nominal["qdot_nom_full_nv"][0] = 0.6
    nominal["qdot_nom_arm_slice"][0] = 0.6
    nominal["estimated_minus_instantaneous_arm_qvel"][0] = 0.6
    nominal["arm_velocity_within_registered_bounds"] = False
    nominal["bound_violation_indices"] = [0]
    nominal["lower_bound_excess_rad_s"] = [0.0] * 7
    nominal["upper_bound_excess_rad_s"] = [0.1] + [0.0] * 6
    nominal["maximum_bound_excess_rad_s"] = 0.1

    result["execution"] = {
        "outcome_kind": "preflight_inadmissible_nominal_velocity",
        "source_prefix_complete": True,
        "qp_executed": False,
        "paired_joint_velocity_physics_executed": False,
        "no_hidden_clipping": True,
    }
    result["source_boundary"]["boundary_B_admissibility"] = None
    result["boundary_B_filter"] = None
    result["arms"] = []
    result["diagnostics"] = None
    result["counts"] = {
        "bound_sample_count": 1,
        "source_prefix_callback_count": 10,
        "nominal_physics_substep_count": 0,
        "psf_physics_substep_count": 0,
        "qp_solve_count": 0,
        "nominal_invalid_field_query_count": 0,
        "psf_invalid_field_query_count": 0,
        "nominal_target_contact_record_count": 0,
        "psf_robot_selected_obstacle_contact_record_count": 0,
    }
    result["classification"] = {
        "label": (
            "inadmissible_nominal_velocity_outside_registered_controller_envelope"
        ),
        "stage_13_passed": False,
        "contact_prevention_observed": False,
        "claim_scope": (
            "registered_controller_envelope_inadmissibility_only_not_poisson_qp_or_contact_prevention"
        ),
        "local_feasibility_conditions": {
            "nominal_velocity_within_registered_controller_envelope": False,
            "no_hidden_clipping": True,
        },
        "nominal_reference": {
            "kind": "not_evaluated_preflight_inadmissible",
            "conditions": {},
            "passed": False,
        },
        "typed_reasons": [
            "nominal_velocity_outside_registered_controller_envelope"
        ],
    }
    result["passed"] = False
    _rehash(result)
    return result, protocol, protocol_raw_sha256, authority


class BoundarySemanticsTests(unittest.TestCase):
    def test_post_contact_observation_maps_to_next_physical_boundary(self) -> None:
        self.assertEqual(
            core.contact_physical_boundary_index(
                8, "post_integration_recomputed"
            ),
            9,
        )

    def test_live_contact_observation_maps_to_same_physical_boundary(self) -> None:
        self.assertEqual(
            core.contact_physical_boundary_index(
                9, "live_solver_phase_preintegration_geometry"
            ),
            9,
        )

    def test_first_filter_boundary_is_at_or_after_warning(self) -> None:
        self.assertEqual(core.first_filter_boundary_at_or_after_warning(10, 5), 10)
        self.assertEqual(core.first_filter_boundary_at_or_after_warning(11, 5), 15)
        self.assertEqual(core.first_filter_boundary_at_or_after_warning(0, 5), 0)

    def test_invalid_boundary_inputs_fail_closed(self) -> None:
        for value in (True, -1, 1.0):
            with self.subTest(value=value), self.assertRaises(
                core.OneStepCounterfactualError
            ):
                core.contact_physical_boundary_index(
                    value, "post_integration_recomputed"
                )
        for value in (True, -1, 1.0):
            with self.subTest(warning=value), self.assertRaises(
                core.OneStepCounterfactualError
            ):
                core.first_filter_boundary_at_or_after_warning(value, 5)


def _final_protocol_schema_available() -> bool:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    return bool(
        protocol.get("qp_execution", {}).get("expected_arm_q_min_rad")
        and protocol.get("qp_execution", {}).get("expected_arm_q_max_rad")
        and protocol.get("acceptance", {}).get("tracking_admissibility_scopes")
        and "nominal_with_selected_obstacle_contact"
        in protocol.get("acceptance", {}).get(
            "static_field_admissibility_scopes", {}
        )
    )


@unittest.skipUnless(
    _final_protocol_schema_available(),
    "final Stage-13 contact/trend protocol schema is still under audit",
)
class CoreArtifactValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        (
            self.result,
            self.protocol,
            self.protocol_raw_sha256,
            self.authority,
        ) = build_valid_one_step_fixture()

    def validate(self, result: dict | None = None) -> dict:
        return core.validate_one_step_counterfactual_result(
            self.result if result is None else result,
            protocol=self.protocol,
            protocol_raw_sha256=self.protocol_raw_sha256,
            expected_authority=self.authority,
        )

    def assert_rehashed_tamper_rejected(self, mutate) -> None:
        attacked = copy.deepcopy(self.result)
        mutate(attacked)
        _rehash(attacked)
        with self.assertRaises(core.OneStepCounterfactualError):
            self.validate(attacked)

    def test_complete_strong_fixture_validates(self) -> None:
        validation = self.validate()
        self.assertTrue(validation["passed"])
        self.assertTrue(validation["stage_13_passed"])
        self.assertFalse(validation["complete_honest_negative"])

    def test_complete_oob_fixture_validates_as_honest_negative(self) -> None:
        result, protocol, protocol_raw_sha256, authority = (
            build_oob_one_step_fixture()
        )
        validation = core.validate_one_step_counterfactual_result(
            result,
            protocol=protocol,
            protocol_raw_sha256=protocol_raw_sha256,
            expected_authority=authority,
        )
        self.assertTrue(validation["passed"])
        self.assertFalse(validation["stage_13_passed"])
        self.assertTrue(validation["complete_honest_negative"])
        self.assertEqual(
            validation["classification_label"],
            "inadmissible_nominal_velocity_outside_registered_controller_envelope",
        )

    def test_rehashed_oob_clipping_or_violation_forgery_is_rejected(self) -> None:
        mutations = (
            lambda value: value["nominal_velocity_estimate"].__setitem__(
                "hidden_clipping_applied", True
            ),
            lambda value: value["nominal_velocity_estimate"].__setitem__(
                "bound_violation_indices", []
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                result, protocol, protocol_raw_sha256, authority = (
                    build_oob_one_step_fixture()
                )
                mutate(result)
                _rehash(result)
                with self.assertRaises(core.OneStepCounterfactualError):
                    core.validate_one_step_counterfactual_result(
                        result,
                        protocol=protocol,
                        protocol_raw_sha256=protocol_raw_sha256,
                        expected_authority=authority,
                    )

    def test_rehashed_oob_qp_or_arm_evidence_is_rejected(self) -> None:
        mutations = (
            lambda value: value.__setitem__("boundary_B_filter", {}),
            lambda value: value.__setitem__(
                "arms", [{"arm_name": "forged_execution"}]
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                result, protocol, protocol_raw_sha256, authority = (
                    build_oob_one_step_fixture()
                )
                mutate(result)
                _rehash(result)
                with self.assertRaises(core.OneStepCounterfactualError):
                    core.validate_one_step_counterfactual_result(
                        result,
                        protocol=protocol,
                        protocol_raw_sha256=protocol_raw_sha256,
                        expected_authority=authority,
                    )

    def test_rehashed_oob_classification_forgery_is_rejected(self) -> None:
        result, protocol, protocol_raw_sha256, authority = (
            build_oob_one_step_fixture()
        )
        result["classification"]["stage_13_passed"] = True
        result["classification"]["label"] = protocol["acceptance"][
            "trend_only_label"
        ]
        _rehash(result)
        with self.assertRaises(core.OneStepCounterfactualError):
            core.validate_one_step_counterfactual_result(
                result,
                protocol=protocol,
                protocol_raw_sha256=protocol_raw_sha256,
                expected_authority=authority,
            )

    def test_protocol_rejects_claim_evidence_mislabeled_as_diagnostic(self) -> None:
        attacked = copy.deepcopy(self.protocol)
        evidence = attacked["required_evidence"]
        moved = evidence["trend_reference_acceptance_evidence"].pop(0)
        evidence["diagnostic_only"].append(moved)
        with self.assertRaisesRegex(
            core.OneStepCounterfactualError,
            "trend acceptance and diagnostic-only evidence scopes differ",
        ):
            core.validate_one_step_counterfactual_result(
                self.result,
                protocol=attacked,
                protocol_raw_sha256=self.protocol_raw_sha256,
                expected_authority=self.authority,
            )

    def test_protocol_rejects_preflight_or_gripper_contract_mislabeling(self) -> None:
        mutations = (
            lambda protocol: protocol["boundary_B_preflight"].__setitem__(
                "failure_policy", "inadmissible_no_QP_and_no_arm_physics"
            ),
            lambda protocol: protocol["arms"]["gripper_policy"][
                "required_evidence"
            ].remove("source_action_7d"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                attacked = copy.deepcopy(self.protocol)
                mutate(attacked)
                with self.assertRaises(core.OneStepCounterfactualError):
                    core.validate_one_step_counterfactual_result(
                        self.result,
                        protocol=attacked,
                        protocol_raw_sha256=self.protocol_raw_sha256,
                        expected_authority=self.authority,
                    )

    def test_partial_result_is_rejected_without_interpretation(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value.__setitem__("status", "running")
        )

    def test_rehashed_alternate_qpos_indexes_are_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["nominal_velocity_estimate"].__setitem__(
                "arm_qpos_indices", [1, 0, 2, 3, 4, 5, 6]
            )
        )

    def test_rehashed_cross_arm_controller_state_mismatch_is_rejected(self) -> None:
        def mutate(value: dict) -> None:
            software = value["arms"][1]["restore"]["controller_software_state"]
            software["fields"]["goal_vel"]["value"] = [0.1] + [0.0] * 6
            unhashed = {key: item for key, item in software.items() if key != "sha256"}
            software["sha256"] = core._sha(unhashed)

        self.assert_rehashed_tamper_rejected(mutate)

    def test_collusively_rehashed_controller_contract_substitution_is_rejected(self) -> None:
        def mutate(value: dict) -> None:
            forged = copy.deepcopy(value["provenance"]["joint_velocity_controller"])
            forged["physical_output_upper_rad_s"] = [0.7] * 7
            value["provenance"]["joint_velocity_controller"] = forged
            for arm in value["arms"]:
                arm["restore"]["controller"] = copy.deepcopy(forged)

        self.assert_rehashed_tamper_rejected(mutate)

    def test_oob_rehashed_controller_contract_substitution_is_rejected(self) -> None:
        result, protocol, protocol_raw_sha256, authority = (
            build_oob_one_step_fixture()
        )
        result["provenance"]["joint_velocity_controller"][
            "control_frequency_hz"
        ] = 20
        _rehash(result)
        with self.assertRaisesRegex(
            core.OneStepCounterfactualError,
            "controller",
        ):
            core.validate_one_step_counterfactual_result(
                result,
                protocol=protocol,
                protocol_raw_sha256=protocol_raw_sha256,
                expected_authority=authority,
            )

    def test_collusively_rehashed_flattened_state_appended_tail_is_rejected(self) -> None:
        def mutate(value: dict) -> None:
            for key in ("state_at_B", "state_at_B_plus_5"):
                state = value["source_boundary"]["exact_prefix"][key]
                state["flattened_simulator_state"].append(123.0)
                state["flattened_simulator_state_sha256"] = (
                    core._float64_array_sha256(
                        state["flattened_simulator_state"]
                    )
                )
            restore = value["source_boundary"]["source_restore"]
            restore["flattened_simulator_state_sha256"] = value[
                "source_boundary"
            ]["exact_prefix"]["state_at_B"][
                "flattened_simulator_state_sha256"
            ]

        self.assert_rehashed_tamper_rejected(mutate)

    def test_oob_rehashed_flattened_state_appended_tail_is_rejected(self) -> None:
        result, protocol, protocol_raw_sha256, authority = (
            build_oob_one_step_fixture()
        )
        state = result["source_boundary"]["exact_prefix"]["state_at_B"]
        state["flattened_simulator_state"].append(-456.0)
        state["flattened_simulator_state_sha256"] = core._float64_array_sha256(
            state["flattened_simulator_state"]
        )
        result["source_boundary"]["source_restore"][
            "flattened_simulator_state_sha256"
        ] = state["flattened_simulator_state_sha256"]
        _rehash(result)
        with self.assertRaisesRegex(
            core.OneStepCounterfactualError, "state layout"
        ):
            core.validate_one_step_counterfactual_result(
                result,
                protocol=protocol,
                protocol_raw_sha256=protocol_raw_sha256,
                expected_authority=authority,
            )

    def test_protocol_rejects_full_nv_independent_audit_overclaim(self) -> None:
        attacked = copy.deepcopy(self.protocol)
        attacked["nominal_velocity_estimator"][
            "nonarm_values_validation_scope"
        ] = "independently_reconstructed_and_claim_bearing"
        with self.assertRaisesRegex(
            core.OneStepCounterfactualError, "overclaims"
        ):
            core.validate_one_step_counterfactual_result(
                self.result,
                protocol=attacked,
                protocol_raw_sha256=self.protocol_raw_sha256,
                expected_authority=self.authority,
            )

    def test_rehashed_mjstate_qpos_projection_mismatch_is_rejected(self) -> None:
        def mutate(value: dict) -> None:
            state = value["source_boundary"]["exact_prefix"]["state_at_B"]
            state["qpos"][0] = 0.2
            state["qpos_sha256"] = core._float64_array_sha256(state["qpos"])

        self.assert_rehashed_tamper_rejected(mutate)

    def test_rehashed_full_robot_distance_population_drop_is_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["arms"][1]["physics_substep_ledger"][0].__setitem__(
                "ordered_full_robot_sample_distance_m", []
            )
        )

    def test_rehashed_full_robot_minimum_forgery_is_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["arms"][1]["physics_substep_ledger"][0].__setitem__(
                "minimum_exact_full_robot_sample_to_current_obstacle_m", 0.09
            )
        )

    def test_rehashed_qp_row_arithmetic_forgery_is_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["boundary_B_filter"][
                "one_CBF_row_per_exact_bound_sample"
            ]["rows_m_per_rad"][0].__setitem__(0, -0.5)
        )

    def test_rehashed_qp_boundary_configuration_forgery_is_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["boundary_B_filter"][
                "joint_position_constraint_rows"
            ]["q_arm_rad"].__setitem__(0, 0.1)
        )

    def test_rehashed_alternate_joint_limits_are_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["boundary_B_filter"][
                "joint_position_constraint_rows"
            ]["q_min_rad"].__setitem__(0, -3.0)
        )

    def test_rehashed_gripper_substitution_is_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["arms"][1].__setitem__(
                "source_gripper_command", -1.0
            )
        )

    def test_rehashed_prefix_substitution_is_rejected_by_external_ledger(self) -> None:
        def mutate(value: dict) -> None:
            prefix = value["source_boundary"]["exact_prefix"]
            prefix["observed_callback_sha256_ledger"][0] = _digest("forged")
            digest = core._sha(prefix["observed_callback_sha256_ledger"])
            prefix["observed_callback_prefix_sha256"] = digest
            prefix["parity_callback_prefix_sha256"] = digest
            prefix["identification_callback_prefix_sha256"] = digest

        self.assert_rehashed_tamper_rejected(mutate)

    def test_rehashed_tracking_summary_forgery_is_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["arms"][1]["tracking"].__setitem__(
                "linf_rad_s", 0.01
            )
        )

    def test_rehashed_endpoint_motion_forgery_is_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["arms"][1]["motion"][
                "measured_arm_tangent_displacement"
            ].__setitem__(0, 0.2)
        )

    def test_rehashed_static_threshold_inflation_is_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["source_boundary"]["boundary_B_admissibility"][
                "selected_obstacle_static_thresholds"
            ].__setitem__("translation_m", 1.0)
        )

    def test_rehashed_classification_forgery_is_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["classification"][
                "local_feasibility_conditions"
            ].__setitem__(
                "qp_valid", False
            )
        )

    def test_rehashed_top_level_passed_forgery_is_rejected(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value.__setitem__("passed", False)
        )

    def test_rehashed_runtime_parameter_rewrite_is_rejected_externally(self) -> None:
        self.assert_rehashed_tamper_rejected(
            lambda value: value["authority"]["dynamic_authority"]["runtime"][
                "registered_parameters"
            ]["cbf"].__setitem__("alpha_gain_per_s", 6.0)
        )
