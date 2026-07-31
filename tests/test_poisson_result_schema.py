from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from main.poisson_fullbody.contracts import (
    ArtifactContractError,
    attach_payload_hash,
    canonical_json_bytes,
    sha256_bytes,
)
from main.poisson_fullbody.result_schema import (
    publish_episode_result,
    validate_episode_result,
)


def measurement(value, units):
    return {"available": True, "value": value, "units": units, "reason": None}


def make_car_unavailable(payload):
    execution = payload["execution"]
    high = execution["high_level_steps"]
    inner = execution["inner_control_steps"]
    physics = execution["physics_substeps"]
    execution.update(
        {
            "completed_high_level_steps": max(0, high - 1),
            "terminal_entered_high_level_index": high - 1 if high else None,
            "terminal_entered_inner_control_index": (inner - 1) % 5 if inner else None,
            "completed_physics_substeps_in_terminal_inner_control": (
                physics - 5 * (inner - 1) if inner else 0
            ),
        }
    )
    payload["endpoints"]["safety"]["paper_car"].update(
        {
            "available": False,
            "maximum_active_obstacle_l1_displacement_m": 0.0,
            "collision": None,
            "avoidance": None,
            "reason": "fixed_exposure_incomplete",
        }
    )
    payload["endpoints"]["task"].update(
        {
            "fixed_exposure_available": False,
            "ever_task_success_within_registered_source_exposure": None,
            "terminal_task_success_within_registered_source_exposure": None,
            "first_task_success_high_level_index": None,
            "unavailable_reason": "fixed_exposure_incomplete",
            "initial_task_success": False,
            "prefix_task_success_latched": False,
            "prefix_first_task_success_high_level_index": None,
            "prefix_terminal_task_success": False,
            "terminal_goal_fraction": 0.0,
            "maximum_goal_fraction": 0.0,
        }
    )
    payload["endpoints"]["validity"].update(
        {
            "velocity_tracking_observed_physics_substep_count": physics,
            "realized_cbf_observed_physics_substep_count": physics,
            "realized_cbf_residual_evaluation_count": 0,
            "negative_realized_cbf_residual_count": 0,
            "first_negative_realized_cbf_residual": None,
            "first_velocity_tracking_threshold_crossing": None,
        }
    )
    payload["endpoints"]["safety"]["minimum_realized_cbf_residual"] = {
        "available": False,
        "value": None,
        "units": "poisson_cbf_m2_per_s",
        "reason": "no_valid_completed_post_state_query",
    }
    if physics == 0:
        payload["endpoints"]["validity"].update(
            {
                "maximum_velocity_tracking_error_rad_s": 0.0,
                "velocity_tracking_rmse_rad_s": 0.0,
            }
        )
        payload["endpoints"]["usefulness"].update(
            {
                "nominal_joint_motion_integral_rad": 0.0,
                "safe_joint_motion_integral_rad": 0.0,
                "measured_joint_motion_integral_rad": 0.0,
                "eef_path_length_m": 0.0,
                "correction_integral_rad": 0.0,
                "motion_retention_ratio": 0.0,
                "zero_motion_fraction": 1.0,
                "all_issued_arm_joint_commands_zero": False,
                "safety_by_no_execution": True,
            }
        )
    safety = payload["endpoints"]["safety"]
    safety["contact_observation_complete"] = False
    safety["contact_absence_right_censored"] = not safety["any_robot_obstacle_contact"]


def valid_payload():
    seed_entries = [
        {"scope": "simulator", "name": "environment", "value": 7},
        {"scope": "policy_noise", "name": "pi05_query_noise", "value": 2026122900},
    ]
    return {
        "schema_version": "vlsa_poisson_link56_episode_result.v2",
        "status": "complete",
        "scientific_result": True,
        "completion_class": "executed",
        "run_id": "poisson-canary-a",
        "protocol_id": "vlsa-poisson-link56-aegis-car-109-v1",
        "case_id": "vlsa-t1-goal-ii-t0-e05",
        "arm": "joint_velocity_psf_link56",
        "provenance": {
            "code_repository": "quanth238/vlsa-aegis",
            "code_commit": "a" * 40,
            "code_dirty": False,
            "code_dirty_state_sha256": sha256_bytes(b""),
            "code_dirty_state_semantics": "sha256_empty_bytes_for_required_clean_git_worktree_v1",
            "run_contract_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "manifest_record_sha256": "d" * 64,
            "protocol_config_sha256": "e" * 64,
            "runtime_protocol_raw_sha256": "6" * 64,
            "runtime_protocol_semantic_sha256": "7" * 64,
            "baseline_repository": "THU-RCSCT/vlsa-aegis",
            "baseline_commit": "1592aa59361f431ba96c6ddcbebcb596f6c20853",
        },
        "case_identity": {
            "dataset": "SafeLIBERO",
            "manifest_case_id": "vlsa-t1-goal-ii-t0-e05",
            "manifest_case_ordinal": 5,
            "pair_group_id": "vlsa-poisson-link56:vlsa-t1-goal-ii-t0-e05",
            "suite": "safelibero_goal",
            "safety_level": "II",
            "logical_task_index": 0,
            "resolved_task_index": 0,
            "episode_index": 5,
            "task_name": "put_the_bowl_on_the_plate",
            "task_family_id": "8" * 64,
            "task_level_group_id": "vlsa-t1-goal-ii-t0",
            "bddl_relative_path": "safelibero/task.bddl",
            "bddl_sha256": "9" * 64,
            "initial_states_relative_path": "safelibero/task.pruned_init",
            "initial_states_sha256": "a" * 64,
            "initial_state_record_sha256": "b" * 64,
            "active_obstacle_name": "moka_pot_obstacle_1",
            "split": "bringup_canary",
            "stratum": "primary_static_positive_aegis_barrier",
            "suite_horizon_high_level_steps": 300,
            "registered_source_exposure_high_level_steps": 2,
        },
        "seeds": {
            "inventory_complete": True,
            "entries": seed_entries,
            "ledger_sha256": sha256_bytes(canonical_json_bytes(seed_entries)),
        },
        "pairing": {
            "pair_group_id": "vlsa-poisson-link56:vlsa-t1-goal-ii-t0-e05",
            "pairing_key_sha256": "0" * 64,
            "restored_settled_state_sha256": "1" * 64,
            "source_settled_observation_sha256": "2" * 64,
            "active_joint_velocity_initial_observation_sha256": "8" * 64,
            "policy_noise_schedule_sha256": "3" * 64,
            "policy_query_schedule_sha256": "5" * 64,
            "nominal_high_level_action_ledger_sha256": "9" * 64,
            "settling_action_ledger_sha256": "a" * 64,
            "controller_initial_state_sha256": "b" * 64,
            "compiled_physical_model_sha256": "d" * 64,
            "compiled_mjb_sha256": "e" * 64,
            "field_sha256": "4" * 64,
            "source_exposure_high_level_steps": 2,
            "inner_updates_per_high_level_action": 5,
            "high_level_dt_seconds": 0.05,
            "inner_dt_seconds": 0.01,
            "physics_dt_seconds": 0.002,
        },
        "runtime": {
            "protocol_parameter_block_sha256": "c" * 64,
            "model": {
                "policy_name": "pi0.5",
                "model_configuration_sha256": "d" * 64,
                "checkpoint_identifier": "openpi-pi05-libero",
                "checkpoint_sha256": "e" * 64,
                "checkpoint_hash_semantics": "sha256_content_tree_v1",
                "normalization_statistics_sha256": "f" * 64,
            },
            "sampler": {
                "implementation": "frozen_historical_pi05_plus_aegis_executed_action_replay",
                "configuration_sha256": "1" * 64,
                "source_action_horizon": 10,
                "source_replan_steps": 5,
                "source_policy_query_count": 1,
                "active_policy_query_count": 0,
                "active_policy_rng_exercised": False,
                "source_action_semantics": "historical_pi05_plus_aegis_translational_exact_actions_executed_not_nominal_raw",
                "deterministic_replay": True,
            },
            "intervention": {
                "enabled": True,
                "mode": "joint_velocity_psf_link56",
                "configuration_sha256": "2" * 64,
                "protected_robot_bodies": ["robot0_link5", "robot0_link6"],
                "static_selected_obstacle_only": True,
                "fail_closed": True,
            },
            "controller": {
                "name": "JOINT_VELOCITY",
                "implementation_version": "robosuite-1.4.1",
                "configuration_sha256": "3" * 64,
                "controlled_joint_count": 7,
                "command_dimension": 8,
                "normalized_limit_abs": 1.0,
                "physical_velocity_limit_rad_s": 0.5,
            },
            "action_space": {
                "source_policy_frame": "mujoco_world_frame_cartesian_delta",
                "controller_command_frame": "panda_joint_velocity_coordinates",
                "source_policy_normalization_space": "openpi_output_after_libero_action_unnormalization",
                "controller_normalization_space": "robosuite_joint_velocity_normalized_minus1_plus1",
                "physical_displacement_conversion": "scale_only_never_subtract_normalization_mean",
                "gripper_semantics": "unchanged_vla_command",
            },
            "measurement": {
                "schema_version": "vlsa_poisson_simulator_measurement.v1",
                "implementation_sha256": "4" * 64,
                "configuration_sha256": "5" * 64,
                "simulator": "MuJoCo",
                "simulator_version": "3.2.3",
                "robosuite_version": "1.4.1",
                "physics_timestep_seconds": 0.002,
                "contact_definition": "mujoco_contact_dist_le_zero",
                "contact_sources": [
                    "settled_state",
                    "live_solver_state",
                    "forwarded_post_integration_state",
                ],
                "D_opt_semantics": "minimum_registered_surface_sample_signed_distance_to_selected_obstacle_obb_union",
                "D_sim_semantics": "contact_authority_plus_coverage_lower_bound",
                "optimizer_and_simulator_clearance_distinct": True,
            },
        },
        "optimizer": {
            "enabled": True,
            "solver": "osqp",
            "solver_version": "1.0.5",
            "configuration_sha256": "6" * 64,
            "attempt_count": 10,
            "solved_count": 10,
            "infeasible_count": 0,
            "solver_failure_count": 0,
            "postcheck_failure_count": 0,
            "terminal_status": "solved",
            "minimum_normalized_cbf_residual": -1e-9,
            "minimum_raw_cbf_residual_m2_per_s": -1e-8,
            "normalized_cbf_postcheck_tolerance": 5e-7,
        },
        "allocation": {
            "execution_environment": "slurm_allocation",
            "slurm_job_id": "33250",
            "slurm_array_job_id": None,
            "slurm_array_task_id": None,
            "slurm_step_id": "33250.0",
            "host_name": "worker-mig-3g40gb-0",
            "process_id": 1234,
            "device": {
                "device_type": "cuda",
                "visible_device_ids": ["0"],
                "model": "NVIDIA H100 80GB HBM3",
                "uuid": "GPU-00000000-0000-0000-0000-000000000000",
                "driver_version": "550.54.15",
                "cuda_runtime_version": "12.4",
            },
        },
        "execution": {
            "high_level_steps": 2,
            "inner_control_steps": 10,
            "physics_substeps": 50,
            "physics_exposure_seconds": 0.1,
            "exposure_complete": True,
            "fail_closed_no_further_physics": False,
            "terminal_reason": "registered_horizon_exhausted",
            "completed_high_level_steps": 2,
            "terminal_entered_high_level_index": 1,
            "terminal_entered_inner_control_index": 4,
            "completed_physics_substeps_in_terminal_inner_control": 5,
            "source_action_prefix_semantics": "entered_hash_counts_provider_entered_actions_completed_hash_counts_only_fully_returned_high_level_post_steps",
            "planned_source_action_ledger_sha256": "9" * 64,
            "entered_source_action_prefix_sha256": "9" * 64,
            "completed_high_level_source_action_prefix_sha256": "9" * 64,
            "nominal_joint_velocity_ledger_sha256": "7" * 64,
            "executed_controller_action_ledger_sha256": "8" * 64,
            "terminal_simulator_state_sha256": "9" * 64,
            "terminal_simulator_state_semantics": "complete_official_mujoco_mjSTATE_INTEGRATION",
            "terminal_observation_sha256": "a" * 64,
            "terminal_observation_semantics": "current_state_synchronized_without_additional_physics",
        },
        "endpoints": {
            "safety": {
                "h_min": measurement(0.004, "poisson_field_m2"),
                "minimum_nominal_cbf_residual": measurement(
                    -0.2, "poisson_cbf_m2_per_s"
                ),
                "minimum_safe_cbf_residual": measurement(
                    -1e-8, "poisson_cbf_m2_per_s"
                ),
                "minimum_realized_cbf_residual": measurement(
                    1e-6, "poisson_cbf_m2_per_s"
                ),
                "D_opt_min_m": measurement(0.03, "m"),
                "D_sim_min_m": measurement(0.01, "m"),
                "D_sim_semantics": "contact_authority_plus_coverage_lower_bound",
                "any_robot_obstacle_contact": False,
                "link56_obstacle_contact": False,
                "contact_observation_complete": True,
                "contact_absence_right_censored": False,
                "total_physical_contact_point_record_count": 0,
                "settled_physical_contact_point_record_count": 0,
                "rollout_phase_physical_contact_point_record_count": 0,
                "live_solver_nonpositive_contact_point_record_count": 0,
                "post_state_physical_contact_point_record_count": 0,
                "first_physical_contact_point_record": None,
                "first_settled_physical_contact_point_record": None,
                "first_live_solver_physical_contact_point_record": None,
                "first_post_state_physical_contact_point_record": None,
                "paper_car": {
                    "available": True,
                    "maximum_active_obstacle_l1_displacement_m": 0.001,
                    "threshold_m": 0.001,
                    "collision": False,
                    "avoidance": True,
                    "reason": None,
                    "exposure_scope": "registered_historical_source_exposure_not_full_table1_suite_horizon",
                    "position_ledger_sha256": "1" * 64,
                },
            },
            "task": {
                "fixed_exposure_available": True,
                "outcome_scope": "registered_source_outcome_dependent_exposure_not_full_table1_suite_horizon",
                "ever_task_success_within_registered_source_exposure": True,
                "terminal_task_success_within_registered_source_exposure": True,
                "first_task_success_high_level_index": 1,
                "initial_task_success": False,
                "prefix_task_success_latched": True,
                "prefix_first_task_success_high_level_index": 1,
                "prefix_terminal_task_success": True,
                "terminal_goal_fraction": 1.0,
                "maximum_goal_fraction": 1.0,
                "goal_regression_count": 0,
                "unavailable_reason": None,
            },
            "usefulness": {
                "nominal_joint_motion_integral_rad": 0.8,
                "safe_joint_motion_integral_rad": 0.7,
                "measured_joint_motion_integral_rad": 0.68,
                "eef_path_length_m": 0.2,
                "correction_integral_rad": 0.1,
                "motion_retention_ratio": 0.875,
                "zero_motion_fraction": 0.0,
                "all_issued_arm_joint_commands_zero": False,
                "safety_by_no_execution": False,
            },
            "validity": {
                "poisson_safe_start": True,
                "static_admissible": True,
                "coverage_audit_passed": True,
                "obstacle_translation_drift_m": 1e-6,
                "obstacle_rotation_drift_rad": 2e-6,
                "invalid_field_query_count": 0,
                "nonpositive_runtime_field_query_count": 0,
                "negative_realized_cbf_residual_count": 0,
                "realized_cbf_observed_physics_substep_count": 50,
                "realized_cbf_residual_evaluation_count": 100,
                "first_negative_realized_cbf_residual": None,
                "qp_infeasible_count": 0,
                "qp_solver_failure_count": 0,
                "qp_postcheck_failure_count": 0,
                "maximum_velocity_tracking_error_rad_s": 0.02,
                "velocity_tracking_rmse_rad_s": 0.01,
                "velocity_tracking_linf_threshold_rad_s": 0.05,
                "velocity_tracking_rmse_threshold_rad_s": 0.02,
                "velocity_tracking_gate_semantics": "empirical_controller_fidelity_gate_not_cbf_safety_certificate",
                "velocity_tracking_observed_physics_substep_count": 50,
                "first_velocity_tracking_threshold_crossing": None,
            },
        },
        "artifact_references": [],
    }


def valid_partial_payload(completion):
    payload = valid_payload()
    payload["completion_class"] = completion
    physics = 1 if completion in (
        "barrier_invariance_lost",
        "controller_tracking_invalid",
    ) else 0
    preflight = completion in (
        "safe_start_inadmissible",
        "static_obstacle_inadmissible",
    )
    high = 0 if preflight else 1
    inner = 0 if preflight else 1
    payload["execution"].update(
        {
            "high_level_steps": high,
            "inner_control_steps": inner,
            "physics_substeps": physics,
            "physics_exposure_seconds": physics * 0.002,
            "exposure_complete": False,
            "fail_closed_no_further_physics": True,
            "terminal_reason": "registered_fail_closed:" + completion,
            "completed_high_level_steps": 0,
            "terminal_entered_high_level_index": 0 if high else None,
            "terminal_entered_inner_control_index": 0 if inner else None,
            "completed_physics_substeps_in_terminal_inner_control": physics,
        }
    )
    payload["optimizer"].update(
        {
            "attempt_count": 0,
            "solved_count": 0,
            "infeasible_count": 0,
            "solver_failure_count": 0,
            "postcheck_failure_count": 0,
            "terminal_status": "not_reached",
            "minimum_normalized_cbf_residual": None,
            "minimum_raw_cbf_residual_m2_per_s": None,
        }
    )
    for name in ("minimum_nominal_cbf_residual", "minimum_safe_cbf_residual"):
        payload["endpoints"]["safety"][name] = {
            "available": False,
            "value": None,
            "units": "poisson_cbf_m2_per_s",
            "reason": "no_solved_poisson_qp",
        }
    payload["endpoints"]["validity"].update(
        {
            "invalid_field_query_count": 0,
            "nonpositive_runtime_field_query_count": 0,
            "negative_realized_cbf_residual_count": 0,
            "qp_infeasible_count": 0,
            "qp_solver_failure_count": 0,
            "qp_postcheck_failure_count": 0,
        }
    )
    make_car_unavailable(payload)

    if completion == "safe_start_inadmissible":
        payload["endpoints"]["validity"]["poisson_safe_start"] = False
        payload["endpoints"]["safety"]["h_min"] = measurement(
            -1e-6, "poisson_field_m2"
        )
    elif completion == "static_obstacle_inadmissible":
        payload["endpoints"]["validity"].update(
            {
                "static_admissible": False,
                "obstacle_translation_drift_m": 2e-6,
            }
        )
    elif completion == "field_invalid":
        payload["endpoints"]["validity"]["invalid_field_query_count"] = 1
    elif completion == "qp_infeasible":
        payload["optimizer"].update(
            {"attempt_count": 1, "infeasible_count": 1, "terminal_status": "infeasible"}
        )
        payload["endpoints"]["validity"]["qp_infeasible_count"] = 1
    elif completion == "qp_solver_failure":
        payload["optimizer"].update(
            {
                "attempt_count": 1,
                "solver_failure_count": 1,
                "terminal_status": "solver_failure",
            }
        )
        payload["endpoints"]["validity"]["qp_solver_failure_count"] = 1
    elif completion == "fail_closed_runtime":
        payload["optimizer"].update(
            {
                "attempt_count": 1,
                "postcheck_failure_count": 1,
                "terminal_status": "postcheck_failure",
            }
        )
        payload["endpoints"]["validity"]["qp_postcheck_failure_count"] = 1
    else:
        payload["optimizer"].update(
            {
                "attempt_count": 1,
                "solved_count": 1,
                "terminal_status": "solved",
                "minimum_normalized_cbf_residual": -1e-9,
                "minimum_raw_cbf_residual_m2_per_s": -1e-8,
            }
        )
        payload["endpoints"]["safety"]["minimum_nominal_cbf_residual"] = measurement(
            -0.2, "poisson_cbf_m2_per_s"
        )
        payload["endpoints"]["safety"]["minimum_safe_cbf_residual"] = measurement(
            -1e-8, "poisson_cbf_m2_per_s"
        )
        payload["endpoints"]["safety"]["minimum_realized_cbf_residual"] = measurement(
            -1e-6 if completion == "barrier_invariance_lost" else 1e-6,
            "poisson_cbf_m2_per_s",
        )
        payload["endpoints"]["validity"].update(
            {
                "realized_cbf_observed_physics_substep_count": 1,
                "realized_cbf_residual_evaluation_count": 1,
                "velocity_tracking_observed_physics_substep_count": 1,
            }
        )
        payload["endpoints"]["usefulness"].update(
            {
                "nominal_joint_motion_integral_rad": 0.001,
                "safe_joint_motion_integral_rad": 0.001,
                "measured_joint_motion_integral_rad": 0.001,
                "eef_path_length_m": 0.0001,
                "correction_integral_rad": 0.0,
                "motion_retention_ratio": 1.0,
                "zero_motion_fraction": 0.0,
                "all_issued_arm_joint_commands_zero": False,
                "safety_by_no_execution": False,
            }
        )
        if completion == "barrier_invariance_lost":
            crossing = {
                "high_level_index": 0,
                "inner_control_index": 0,
                "physics_substep_index": 0,
                "minimum_realized_cbf_residual_m2_per_s": -1e-6,
                "minimum_sample_index": 0,
                "minimum_h_m2": 0.001,
            }
            payload["endpoints"]["validity"].update(
                {
                    "negative_realized_cbf_residual_count": 1,
                    "first_negative_realized_cbf_residual": crossing,
                    "maximum_velocity_tracking_error_rad_s": 0.01,
                    "velocity_tracking_rmse_rad_s": 0.005,
                }
            )
        else:
            tracking_crossing = {
                "high_level_index": 0,
                "inner_control_index": 0,
                "physics_substep_index": 0,
                "error_linf_rad_s": 0.06,
                "cumulative_rmse_rad_s": 0.023,
                "linf_threshold_rad_s": 0.05,
                "rmse_threshold_rad_s": 0.02,
                "command_rad_s": [0.0] * 7,
                "measured_rad_s": [0.06] + [0.0] * 6,
            }
            payload["endpoints"]["validity"].update(
                {
                    "maximum_velocity_tracking_error_rad_s": 0.06,
                    "velocity_tracking_rmse_rad_s": 0.023,
                    "first_velocity_tracking_threshold_crossing": tracking_crossing,
                }
            )
    return payload


def hashed(payload):
    return attach_payload_hash(payload)


class PoissonResultSchemaTest(unittest.TestCase):
    def test_complete_executed_result_is_valid(self):
        validate_episode_result(hashed(valid_payload()))

    def test_executed_result_rejects_tracking_threshold_crossing(self):
        payload = valid_payload()
        payload["endpoints"]["validity"].update(
            {
                "maximum_velocity_tracking_error_rad_s": 0.06,
                "velocity_tracking_rmse_rad_s": 0.023,
                "first_velocity_tracking_threshold_crossing": {
                    "high_level_index": 0,
                    "inner_control_index": 0,
                    "physics_substep_index": 0,
                    "error_linf_rad_s": 0.06,
                    "cumulative_rmse_rad_s": 0.023,
                    "linf_threshold_rad_s": 0.05,
                    "rmse_threshold_rad_s": 0.02,
                    "command_rad_s": [0.0] * 7,
                    "measured_rad_s": [0.06] + [0.0] * 6,
                },
            }
        )
        with self.assertRaisesRegex(
            ArtifactContractError, "tracking-threshold crossing"
        ):
            validate_episode_result(hashed(payload))

    def test_every_registered_fail_closed_completion_is_directly_validated(self):
        completions = (
            "safe_start_inadmissible",
            "static_obstacle_inadmissible",
            "field_invalid",
            "barrier_invariance_lost",
            "qp_infeasible",
            "qp_solver_failure",
            "controller_tracking_invalid",
            "fail_closed_runtime",
        )
        for completion in completions:
            with self.subTest(completion=completion):
                validate_episode_result(hashed(valid_partial_payload(completion)))

    def test_h_is_not_accepted_with_distance_units(self):
        payload = valid_payload()
        payload["endpoints"]["safety"]["h_min"]["units"] = "m"
        with self.assertRaisesRegex(ArtifactContractError, "poisson_field_m2"):
            validate_episode_result(hashed(payload))

    def test_dopt_cannot_replace_dsim(self):
        payload = valid_payload()
        del payload["endpoints"]["safety"]["D_sim_min_m"]
        with self.assertRaisesRegex(ArtifactContractError, "D_sim_min_m"):
            validate_episode_result(hashed(payload))

    def test_car_uses_strict_greater_than_one_millimetre(self):
        validate_episode_result(hashed(valid_payload()))
        payload = valid_payload()
        payload["endpoints"]["safety"]["paper_car"]["collision"] = True
        payload["endpoints"]["safety"]["paper_car"]["avoidance"] = False
        with self.assertRaisesRegex(ArtifactContractError, "strict >"):
            validate_episode_result(hashed(payload))

    def test_partial_exposure_cannot_claim_car_avoidance(self):
        payload = valid_payload()
        payload["completion_class"] = "qp_solver_failure"
        payload["execution"].update(
            {
                "high_level_steps": 1,
                "inner_control_steps": 1,
                "physics_substeps": 0,
                "physics_exposure_seconds": 0.0,
                "exposure_complete": False,
                "fail_closed_no_further_physics": True,
                "terminal_reason": "osqp_solver_failure",
                "completed_high_level_steps": 0,
                "terminal_entered_high_level_index": 0,
                "terminal_entered_inner_control_index": 0,
                "completed_physics_substeps_in_terminal_inner_control": 0,
            }
        )
        payload["optimizer"].update(
            {
                "attempt_count": 1,
                "solved_count": 0,
                "solver_failure_count": 1,
                "terminal_status": "solver_failure",
                "minimum_normalized_cbf_residual": None,
                "minimum_raw_cbf_residual_m2_per_s": None,
            }
        )
        payload["endpoints"]["validity"]["qp_solver_failure_count"] = 1
        with self.assertRaisesRegex(ArtifactContractError, "partial|fixed-exposure"):
            validate_episode_result(hashed(payload))

    def test_adapter_execution_reports_static_drift_without_requiring_static_field(self):
        payload = valid_payload()
        payload["arm"] = "joint_velocity_adapter_only"
        payload["runtime"]["intervention"].update(
            {
                "enabled": False,
                "mode": "joint_velocity_adapter_only",
                "protected_robot_bodies": [],
            }
        )
        payload["optimizer"].update(
            {
                "enabled": False,
                "attempt_count": 0,
                "solved_count": 0,
                "terminal_status": "not_applicable",
                "minimum_normalized_cbf_residual": None,
                "minimum_raw_cbf_residual_m2_per_s": None,
            }
        )
        payload["endpoints"]["safety"]["minimum_realized_cbf_residual"] = {
            "available": False,
            "value": None,
            "units": "poisson_cbf_m2_per_s",
            "reason": "adapter_intervention_disabled",
        }
        payload["endpoints"]["validity"].update(
            {
                "poisson_safe_start": False,
                "static_admissible": False,
                "coverage_audit_passed": False,
                "obstacle_translation_drift_m": 0.02,
                "realized_cbf_observed_physics_substep_count": 0,
                "realized_cbf_residual_evaluation_count": 0,
            }
        )
        payload["endpoints"]["usefulness"].update(
            {
                "safe_joint_motion_integral_rad": 0.8,
                "correction_integral_rad": 0.0,
                "motion_retention_ratio": 1.0,
            }
        )
        validate_episode_result(hashed(payload))

    def test_poisson_runtime_drift_retains_the_exact_executed_prefix(self):
        payload = valid_payload()
        payload["completion_class"] = "static_obstacle_inadmissible"
        payload["execution"].update(
            {
                "high_level_steps": 1,
                "inner_control_steps": 1,
                "physics_substeps": 1,
                "physics_exposure_seconds": 0.002,
                "exposure_complete": False,
                "fail_closed_no_further_physics": True,
                "terminal_reason": "static_obstacle_drift",
            }
        )
        payload["optimizer"].update(
            {
                "attempt_count": 1,
                "solved_count": 1,
                "terminal_status": "solved",
            }
        )
        payload["endpoints"]["validity"].update(
            {"static_admissible": False, "obstacle_translation_drift_m": 2e-6}
        )
        make_car_unavailable(payload)
        validate_episode_result(hashed(payload))

    def test_positive_clearance_conflicts_with_observed_contact(self):
        payload = valid_payload()
        payload["endpoints"]["safety"]["any_robot_obstacle_contact"] = True
        payload["endpoints"]["safety"].update(
            {
                "total_physical_contact_point_record_count": 1,
                "rollout_phase_physical_contact_point_record_count": 1,
                "live_solver_nonpositive_contact_point_record_count": 1,
                "first_physical_contact_point_record": {"source_phase": "live"},
                "first_live_solver_physical_contact_point_record": {
                    "source_phase": "live"
                },
            }
        )
        with self.assertRaisesRegex(ArtifactContractError, "cannot be positive"):
            validate_episode_result(hashed(payload))

    def test_executed_result_must_preserve_five_by_five_timing(self):
        payload = valid_payload()
        payload["execution"]["physics_substeps"] = 49
        payload["execution"]["physics_exposure_seconds"] = 0.098
        with self.assertRaisesRegex(ArtifactContractError, "20/100/500"):
            validate_episode_result(hashed(payload))

    def test_unavailable_measurement_requires_explicit_reason(self):
        payload = valid_payload()
        payload["completion_class"] = "safe_start_inadmissible"
        payload["execution"].update(
            {
                "high_level_steps": 0,
                "inner_control_steps": 0,
                "physics_substeps": 0,
                "physics_exposure_seconds": 0.0,
                "exposure_complete": False,
                "fail_closed_no_further_physics": True,
                "terminal_reason": "safe_start_inadmissible",
            }
        )
        payload["endpoints"]["safety"]["D_sim_min_m"] = {
            "available": False,
            "value": None,
            "units": "m",
            "reason": "no_control_physics_executed",
        }
        payload["endpoints"]["validity"]["poisson_safe_start"] = False
        payload["optimizer"].update(
            {
                "attempt_count": 0,
                "solved_count": 0,
                "terminal_status": "not_reached",
                "minimum_normalized_cbf_residual": None,
                "minimum_raw_cbf_residual_m2_per_s": None,
            }
        )
        make_car_unavailable(payload)
        validate_episode_result(hashed(payload))
        broken = copy.deepcopy(payload)
        broken["endpoints"]["safety"]["D_sim_min_m"]["reason"] = None
        with self.assertRaisesRegex(ArtifactContractError, "reason"):
            validate_episode_result(hashed(broken))

    def test_self_hash_mismatch_is_rejected(self):
        result = hashed(valid_payload())
        result["case_id"] = "tampered"
        with self.assertRaisesRegex(ArtifactContractError, "does not match"):
            validate_episode_result(result)

    def test_contact_fields_must_describe_the_same_event_population(self):
        payload = valid_payload()
        payload["endpoints"]["safety"].update(
            {
                "total_physical_contact_point_record_count": 7,
                "rollout_phase_physical_contact_point_record_count": 7,
                "live_solver_nonpositive_contact_point_record_count": 7,
                "first_physical_contact_point_record": {"physics_substep": 1},
                "first_live_solver_physical_contact_point_record": {
                    "physics_substep": 1
                },
            }
        )
        with self.assertRaisesRegex(
            ArtifactContractError, "missing required fields|exactly when"
        ):
            validate_episode_result(hashed(payload))

        payload = valid_payload()
        payload["endpoints"]["safety"]["link56_obstacle_contact"] = True
        with self.assertRaisesRegex(ArtifactContractError, "without robot-obstacle"):
            validate_episode_result(hashed(payload))

    def test_fail_closed_completion_must_match_flags_and_reason_count(self):
        payload = valid_payload()
        payload["completion_class"] = "qp_infeasible"
        with self.assertRaisesRegex(ArtifactContractError, "fail-closed results"):
            validate_episode_result(hashed(payload))

        payload["execution"].update(
            {
                "high_level_steps": 1,
                "inner_control_steps": 1,
                "physics_substeps": 0,
                "physics_exposure_seconds": 0.0,
                "exposure_complete": False,
                "fail_closed_no_further_physics": True,
                "terminal_reason": "qp_infeasible",
            }
        )
        make_car_unavailable(payload)
        with self.assertRaisesRegex(ArtifactContractError, "infeasible QP"):
            validate_episode_result(hashed(payload))

    def test_partial_execution_counters_must_form_an_exact_prefix(self):
        payload = valid_payload()
        payload["completion_class"] = "field_invalid"
        payload["execution"].update(
            {
                "high_level_steps": 2,
                "inner_control_steps": 10,
                "physics_substeps": 0,
                "physics_exposure_seconds": 0.0,
                "exposure_complete": False,
                "fail_closed_no_further_physics": True,
                "terminal_reason": "invalid_field_query",
            }
        )
        payload["endpoints"]["validity"]["invalid_field_query_count"] = 1
        with self.assertRaisesRegex(ArtifactContractError, "exact inner-update prefix"):
            validate_episode_result(hashed(payload))

    def test_missing_or_ambiguous_provenance_is_rejected(self):
        payload = valid_payload()
        del payload["runtime"]["model"]["checkpoint_sha256"]
        with self.assertRaisesRegex(ArtifactContractError, "checkpoint_sha256"):
            validate_episode_result(hashed(payload))

        payload = valid_payload()
        payload["provenance"]["code_dirty"] = True
        with self.assertRaisesRegex(ArtifactContractError, "must be false"):
            validate_episode_result(hashed(payload))

        payload = valid_payload()
        del payload["provenance"]["code_dirty_state_sha256"]
        with self.assertRaisesRegex(ArtifactContractError, "code_dirty_state_sha256"):
            validate_episode_result(hashed(payload))

    def test_seed_inventory_is_complete_unique_and_self_bound(self):
        payload = valid_payload()
        payload["seeds"]["inventory_complete"] = False
        with self.assertRaisesRegex(ArtifactContractError, "must be true"):
            validate_episode_result(hashed(payload))

        payload = valid_payload()
        payload["seeds"]["entries"][0]["value"] += 1
        with self.assertRaisesRegex(ArtifactContractError, "ordered seed entries"):
            validate_episode_result(hashed(payload))

    def test_pairing_action_ledger_and_horizon_are_cross_checked(self):
        payload = valid_payload()
        payload["execution"]["planned_source_action_ledger_sha256"] = "0" * 64
        with self.assertRaisesRegex(ArtifactContractError, "paired frozen"):
            validate_episode_result(hashed(payload))

        payload = valid_payload()
        payload["case_identity"]["registered_source_exposure_high_level_steps"] = 3
        with self.assertRaisesRegex(ArtifactContractError, "registered_source"):
            validate_episode_result(hashed(payload))

    def test_runtime_action_measurement_and_allocation_are_not_optional(self):
        payload = valid_payload()
        payload["runtime"]["action_space"]["physical_displacement_conversion"] = (
            "subtract_mean_then_scale"
        )
        with self.assertRaisesRegex(ArtifactContractError, "scale_only"):
            validate_episode_result(hashed(payload))

        payload = valid_payload()
        payload["runtime"]["measurement"]["D_sim_semantics"] = (
            "union_of_settled_live_solver_and_forwarded_post_state_nonpositive_contacts_plus_exact_obb_coverage_lower_bound"
        )
        with self.assertRaisesRegex(ArtifactContractError, "must equal runtime"):
            validate_episode_result(hashed(payload))

        payload = valid_payload()
        payload["allocation"]["slurm_job_id"] = "local"
        with self.assertRaisesRegex(ArtifactContractError, "decimal digits"):
            validate_episode_result(hashed(payload))

    def test_optimizer_counts_and_terminal_status_are_cross_checked(self):
        payload = valid_payload()
        payload["optimizer"]["attempt_count"] = 11
        with self.assertRaisesRegex(ArtifactContractError, "classified attempt"):
            validate_episode_result(hashed(payload))

        payload = valid_payload()
        payload["completion_class"] = "qp_solver_failure"
        payload["execution"].update(
            {
                "high_level_steps": 1,
                "inner_control_steps": 1,
                "physics_substeps": 0,
                "physics_exposure_seconds": 0.0,
                "exposure_complete": False,
                "fail_closed_no_further_physics": True,
                "terminal_reason": "osqp_solver_failure",
            }
        )
        payload["optimizer"].update(
            {
                "attempt_count": 1,
                "solved_count": 0,
                "solver_failure_count": 1,
                "terminal_status": "solver_failure",
                "minimum_normalized_cbf_residual": None,
                "minimum_raw_cbf_residual_m2_per_s": None,
            }
        )
        payload["endpoints"]["validity"]["qp_solver_failure_count"] = 1
        make_car_unavailable(payload)
        validate_episode_result(hashed(payload))

    def test_unknown_top_level_fields_are_rejected(self):
        payload = valid_payload()
        payload["unregistered_debug_state"] = {"partial": True}
        with self.assertRaisesRegex(ArtifactContractError, "unsupported fields"):
            validate_episode_result(hashed(payload))

        payload = valid_payload()
        payload["endpoints"]["validity"]["unregistered_override"] = True
        with self.assertRaisesRegex(ArtifactContractError, "unsupported fields"):
            validate_episode_result(hashed(payload))

    def test_v2_json_schema_is_closed_and_configured(self):
        root = Path(__file__).resolve().parents[1]
        schema_path = root / "schemas" / "vlsa_poisson_case_result.v2.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            "vlsa_poisson_link56_episode_result.v2",
        )
        self.assertFalse(schema["additionalProperties"])
        for definition in (
            "provenance",
            "caseIdentity",
            "seeds",
            "pairing",
            "runtime",
            "optimizer",
            "allocation",
            "execution",
            "endpoints",
        ):
            self.assertFalse(schema["$defs"][definition]["additionalProperties"])
        self.assertFalse(
            (root / "schemas" / "vlsa_poisson_case_result.v1.schema.json").exists()
        )
        config = json.loads(
            (root / "configs" / "vlsa_poisson_link56_feasibility.v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            config["result_contract"]["json_schema_relative_path"],
            "schemas/vlsa_poisson_case_result.v2.schema.json",
        )

    def test_episode_publisher_never_links_an_invalid_or_partial_final(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            invalid = valid_payload()
            del invalid["runtime"]["model"]["checkpoint_sha256"]
            with self.assertRaisesRegex(ArtifactContractError, "checkpoint_sha256"):
                publish_episode_result(path, invalid)
            self.assertFalse(path.exists())

            self.assertEqual(
                publish_episode_result(path, valid_payload()),
                "published",
            )
            validate_episode_result(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
