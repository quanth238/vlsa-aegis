import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from main.poisson_fullbody.contracts import publish_hashed_json
from scripts.validate_poisson_direct_joint_velocity_feasibility_artifact import (
    ADAPTER_MODULE,
    CBF_MODULE,
    CONTACT_MONITOR_MODULE,
    ARM_RUNNER_MODULE,
    PRODUCER_MODULE,
    PROTOCOL_MODULE,
    RESULT_SCHEMA,
    _Audit,
    _audit_source_text,
    _canonical_sha256,
    _float64_matrix_sha256,
    validate_payload as _validate_payload,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    ROOT
    / "scripts/validate_poisson_direct_joint_velocity_feasibility_artifact.py"
)
SHA = "1" * 64


def validate_payload(payload, **kwargs):
    kwargs.setdefault("_test_expected_action_count_override", 2)
    return _validate_payload(payload, **kwargs)


def _controller_contract():
    return {
        "controller": "JOINT_VELOCITY",
        "system_model": "qdot_equals_v",
        "decision_variable": "joint_velocity_rad_s",
        "cbf_constraint": "grad_h_dot_J_qdot_plus_alpha_h_ge_0",
        "static_obstacle_partial_t_h": 0.0,
        "control_frequency_hz": 100,
        "physics_frequency_hz": 500,
        "controller_updates_per_high_level_action": 5,
        "physics_substeps_per_controller_update": 5,
        "hard_constraints": True,
        "safety_slack_used": False,
        "fallback_policy": "none_fail_closed_as_negative",
        "zero_velocity_is_feasibility_witness_not_acceptance": True,
        "acceptance_thresholds": {
            "cbf_postcheck_tolerance_m2_per_s": 5.0e-7,
            "velocity_bound_tolerance_rad_s": 5.0e-8,
            "material_correction_minimum_rad_s": 1.0e-4,
            "material_correction_integral_minimum_rad": 1.0e-6,
            "post_correction_joint_motion_minimum_rad": 1.0e-4,
            "post_correction_eef_path_minimum_m": 1.0e-3,
            "post_correction_executed_integral_minimum_rad": 1.0e-4,
            "post_correction_zero_command_fraction_maximum": 0.99,
            "tracking_linf_maximum_rad_s": 0.05,
            "tracking_rmse_maximum_rad_s": 0.02,
        },
    }


def _restore():
    return {
        "source_controller": "OSC_POSE",
        "target_controller": "JOINT_VELOCITY",
        "official_integration_state_sha256": SHA,
        "target_official_integration_state_sha256": SHA,
        "settled_state_sha256": SHA,
        "target_state_sha256": SHA,
        "exact_flattened_state": True,
        "controller": {
            "controller_name": "JOINT_VELOCITY",
            "controller_class_qualname": "JointVelocityController",
            "environment_action_dim": 8,
            "arm_control_dim": 7,
            "control_frequency_hz": 100,
            "control_timestep_s": 0.01,
            "physics_timestep_s": 0.002,
            "physics_substeps_per_control": 5,
        },
    }


def _task():
    rows = [
        {
            "snapshot_kind": "branch_boundary_pre_action",
            "source_action_index": -1,
            "values": [False],
            "all_satisfied": False,
            "fraction": 0.0,
            "returned_done": False,
            "returned_observation_sha256": SHA,
        },
        {
            "snapshot_kind": "completed_high_level_post_step",
            "source_action_index": 0,
            "values": [False],
            "all_satisfied": False,
            "fraction": 0.0,
            "returned_done": False,
            "returned_observation_sha256": SHA,
        },
        {
            "snapshot_kind": "completed_high_level_post_step",
            "source_action_index": 1,
            "values": [True],
            "all_satisfied": True,
            "fraction": 1.0,
            "returned_done": True,
            "returned_observation_sha256": SHA,
        },
    ]
    return {
        "source": "native_bddl_goal_predicates",
        "goal_progress_ledger": rows,
        "registered_source_action_count": 2,
        "completed_source_action_count": 2,
        "initial_task_success_at_branch": False,
        "ever_task_success_at_or_after_branch": True,
        "first_task_success_source_action_index": 1,
        "terminal_task_success": True,
        "returned_observation_sha256_ledger": [SHA, SHA],
    }


def _registered_scope():
    scope = {
        "schema_version": "vlsa_poisson_registered_contact_scope.v1",
        "robot_geom_ids": [41],
        "robot_owned_geom_ids": [41],
        "selected_obstacle_geom_ids": [101],
        "link56_geom_ids": [41],
        "external_nonrobot_geom_ids": [101, 102],
        "geom_identities": [],
    }
    scope["identity_sha256"] = _canonical_sha256(scope)
    return scope


def _selected_measurement(contact, *, boundary=10):
    live = []
    if contact:
        live.append(
            {
                "source_phase": "live_solver_phase_preintegration_geometry",
                "observation_index": boundary,
                "high_level_index": boundary // 25,
                "inner_control_index": (boundary % 25) // 5,
                "physics_substep_index": 0,
                "robot_body_name": "robot0_link5",
                "robot_geom_id": 41,
                "obstacle_geom_id": 101,
                "contact_distance_m": 0.0,
                "is_physical_nonpositive_distance_contact": True,
            }
        )
    return {
        "settled_state": {
            "any_robot_obstacle_contact": False,
            "link56_obstacle_contact": False,
            "physical_contact_point_records": [],
        },
        "live_solver_phase_contact_point_records": live,
        "post_state_candidate_contact_point_records": [],
        "live_solver_nonpositive_contact_point_record_count": len(live),
        "post_state_physical_contact_point_record_count": 0,
        "total_physical_contact_point_record_count": len(live),
        "any_robot_obstacle_contact": bool(contact),
        "link56_obstacle_contact": bool(contact),
    }


def _registered_record(*, selected=True, shifted=False, boundary=10):
    categories = []
    if selected:
        categories.append("any_robot_vs_selected_obstacle")
    categories.append("link56_vs_external_nonrobot")
    record = {
        "schema_version": "vlsa_poisson_registered_forbidden_contact.v2",
        "source_phase": "post_integration_recomputed",
        "physical_boundary": boundary,
        "executed_transition_start_boundary": boundary,
        "observed_state_boundary": boundary + 1,
        "source_action_index": boundary // 25,
        "physics_substep_index": boundary % 25,
        "controller_update_index": (boundary % 25) // 5,
        "physics_substep_within_controller_update": boundary % 5,
        "callback_endpoint_action_inner_substep": [
            boundary // 25,
            (boundary % 25) // 5,
            boundary % 5,
        ],
        "contact_distance_m": 0.0,
        "robot_geom_id": 41,
        "robot_body_name": "robot0_link5",
        "external_geom_id": 102 if shifted else 101,
        "external_body_name": "other_object" if shifted else "selected_obstacle",
        "contact_categories": categories,
        "any_robot_selected_obstacle_contact": selected,
        "link56_external_nonrobot_contact": True,
        "link56_nonselected_external_contact": shifted,
    }
    record["record_sha256"] = _canonical_sha256(record)
    return record


def _registered_measurement(contact=False, shifted=False, *, physics_count=50, boundary=10):
    rows = []
    if contact or shifted:
        rows.append(
            _registered_record(
                selected=contact, shifted=shifted, boundary=boundary
            )
        )
    scope = _registered_scope()
    return {
        "schema_version": "vlsa_poisson_registered_contact_measurement.v2",
        "scope_identity_sha256": scope["identity_sha256"],
        "scope": scope,
        "observed_physics_substeps": physics_count,
        "callback_cadence": {
            "controller_updates_per_action": 5,
            "physics_substeps_per_controller_update": 5,
            "physics_substeps_per_action": 25,
        },
        "settled_forbidden_contact": False,
        "settled_contact_records": [],
        "rollout_forbidden_contact": bool(rows),
        "rollout_contact_records": rows,
        "any_registered_forbidden_contact": bool(rows),
        "any_robot_selected_obstacle_contact": bool(contact),
        "any_link56_external_nonrobot_contact": bool(rows),
        "any_link56_nonselected_external_contact": bool(shifted),
    }


def _arm(psf_enabled):
    command_rows = []
    physics_rows = []
    for command_index in range(10):
        source = command_index // 5
        inner = command_index % 5
        boundary = source * 25 + inner * 5
        nominal = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        material = psf_enabled and command_index == 1
        executed = [0.2 if material else 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        correction = math.sqrt(
            sum((safe - raw) ** 2 for safe, raw in zip(executed, nominal))
        )
        qp = None
        if psf_enabled:
            qp = {
                "status": "solved",
                "status_value": 1,
                "input_constraint_count": 2,
                "solved_constraint_count": 2,
                "minimum_raw_cbf_residual_m2_per_s": 0.01,
                "maximum_velocity_bound_violation_rad_s": 0.0,
            }
        command_rows.append(
            {
                "local_action_index": source,
                "source_action_index": source,
                "source_action": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                "inner_control_index": inner,
                "physical_boundary": boundary,
                "nominal_qdot_rad_s": nominal,
                "executed_qdot_rad_s": executed,
                "correction_l2_rad_s": correction,
                "safe_cbf_residual_minimum_m2_per_s": 0.01 if psf_enabled else None,
                "nominal_cbf_residual_minimum_m2_per_s": (
                    -0.01 if material else (0.01 if psf_enabled else None)
                ),
                "nominal_argmin_protected_sample": (
                    {"sample_id": 0, "body_name": "robot0_link5"}
                    if psf_enabled
                    else None
                ),
                "nominal_dynamic_lower_bound_rad_s": [-0.5] * 7,
                "nominal_dynamic_upper_bound_rad_s": [0.5] * 7,
                "nominal_within_dynamic_joint_bounds": True,
                "eef_position_before_update_world_m": [boundary * 0.001, 0.0, 0.0],
                "qp": qp,
            }
        )
        for substep in range(5):
            observation = command_index * 5 + substep
            measured = list(executed)
            physics_rows.append(
                {
                    "observation_index": observation,
                    "post_state_physical_boundary": boundary + substep + 1,
                    "local_action_index": source,
                    "source_action_index": source,
                    "inner_control_index": inner,
                    "physics_substep_index": substep,
                    "measured_qvel_rad_s": measured,
                    "issued_qvel_rad_s": list(executed),
                    "tracking_error_rad_s": [0.0] * 7,
                    "eef_position_world_m": [(observation + 1) * 0.001, 0.0, 0.0],
                    "literal_contact_observed": bool(not psf_enabled and observation == 10),
                    "registered_forbidden_contact_observed": bool(
                        not psf_enabled and observation == 10
                    ),
                }
            )
    contact = not psf_enabled
    return {
        "arm_name": (
            "joint_velocity_adapter_plus_link56_psf"
            if psf_enabled
            else "joint_velocity_adapter_only"
        ),
        "restore": _restore(),
        "start_official_raw_bytes_sha256": SHA,
        "execution_cadence": {
            "control_timestep_s": 0.01,
            "wrapper_model_timestep_s": 0.002,
            "mujoco_model_timestep_s": 0.002,
        },
        "filter_update_count": 10,
        "physics_substep_count": 50,
        "physics_trace_row_count": 50,
        "monitor_observed_physics_substep_count": 50,
        "physics_monitor_trace_counts_match": True,
        "exposure_complete": True,
        "contact_terminated_early": False,
        "method_terminated_early": False,
        "method_stop": None,
        "precontact_execution_valid": True,
        "qp_solve_count": 10 if psf_enabled else 0,
        "qp_postcheck_count": 10 if psf_enabled else 0,
        "joint_limit_postcheck_count": 10 if psf_enabled else 0,
        "issued_command_bound_check_count": 10,
        "all_issued_commands_within_physical_bounds": True,
        "nominal_dynamic_bound_check_count": 10,
        "nominal_dynamic_bound_violation_count": 0,
        "all_nominal_commands_within_dynamic_joint_bounds": True,
        "protected_sample_count": 2,
        "invalid_field_query_count": 0,
        "pre_filter_field_observation_count": 10 if psf_enabled else 0,
        "pre_filter_field_query_count": 20 if psf_enabled else 0,
        "nonpositive_post_state_field_query_count": 0,
        "command_trace": command_rows,
        "physics_trace": physics_rows,
        "measurement": _selected_measurement(contact),
        "literal_contact": {
            "any_robot_selected_obstacle_present": contact,
            "link56_present": contact,
            "first_any_robot_physical_boundary": 10 if contact else None,
            "first_link56_physical_boundary": 10 if contact else None,
        },
        "registered_forbidden_contact": _registered_measurement(contact=contact),
        "paper_car": {
            "enabled": True,
            "active_obstacle_name": "selected_obstacle",
            "initial_active_obstacle_position_m": [0.0, 0.0, 0.0],
            "endpoint_ledger": [
                {
                    "source_action_index": index,
                    "returned_observation_sha256": SHA,
                    "active_obstacle_position_m": [0.0, 0.0, 0.0],
                    "l1_displacement_from_settled_m": 0.0,
                    "paper_collision_threshold_m": 0.001,
                    "paper_collision": False,
                }
                for index in range(2)
            ],
            "expected_endpoint_count": 2,
            "endpoint_ledger_complete": True,
            "maximum_active_obstacle_l1_displacement_m": 0.0,
            "collision_first_source_action_index": None,
            "paper_collision_threshold_m": 0.001,
            "paper_collision_avoidance": True,
        },
        "task": _task(),
    }


def _video(action_count=2, *, terminal=False, completed_action_count=None):
    completed = action_count if completed_action_count is None else completed_action_count
    frame_count = action_count + 1 if not terminal else completed + 2
    source_end = action_count - 1 if not terminal else completed
    snapshots = []
    for index in range(frame_count):
        snapshots.append(
            {
                "frame_index": index,
                "snapshot_kind": (
                    "branch_boundary" if index == 0 else "completed_high_level_post_step"
                ),
                "local_action_index": None if index == 0 else index - 1,
                "source_action_index": -1 if index == 0 else index - 1,
                "source_array_sha256": SHA,
                "height": 1024,
                "width": 1024,
            }
        )
    return {
        "decoded_successfully": True,
        "real_simulation_frames": True,
        "two_dimensional_safety_overlay": False,
        "source_action_index_start": -1,
        "source_action_index_end": source_end,
        "frame_count": frame_count,
        "decoded_frame_count": frame_count,
        "snapshot_trace": snapshots,
    }


def _fixture():
    samples = [
        {
            "sample_id": 0,
            "body_name": "robot0_link5",
            "geom_name": "link5_collision",
            "point_body_local_m": [0.0, 0.0, 0.0],
        },
        {
            "sample_id": 1,
            "body_name": "robot0_link6",
            "geom_name": "link6_collision",
            "point_body_local_m": [0.0, 0.0, 0.0],
        },
    ]
    returned_actions = [
        [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        for _ in range(10)
    ]
    returned_hash = _float64_matrix_sha256(
        _Audit(), returned_actions, "fixture_actions"
    )
    policy_fingerprint = "2" * 64
    first_action = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    released_proxy_diagnostic = {
        "fresh_settled_p1": [0.0, 0.0, 0.0],
        "fresh_settled_R1": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "released_pre_settle_p1": [0.001, 0.0, 0.0],
        "released_pre_settle_R1": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "p1_l2_difference_m": 0.001,
    }
    baseline_provider = {
        "source_start_action": 0,
        "first_aegis_proxy_source": (
            "released_pre_settle_proxy_from_historical_action0_context"
        ),
        "first_fresh_proxy_diagnostic": copy.deepcopy(
            released_proxy_diagnostic
        ),
        "historical_first_action_required_aegis_inputs": {
            "p1": copy.deepcopy(
                released_proxy_diagnostic["released_pre_settle_p1"]
            ),
            "R1": copy.deepcopy(
                released_proxy_diagnostic["released_pre_settle_R1"]
            ),
        },
        "first_aegis_input_binding_matches_historical": True,
        "recorded_suffix_actions_executed": False,
        "policy_queries": [
            {
                "query_index": 0,
                "rng_seed": 1234,
                "source_action_index": 0,
                "local_action_index": 0,
                "policy_input_fingerprint_sha256": policy_fingerprint,
                "returned_actions": returned_actions,
                "returned_action_shape": [10, 7],
                "returned_actions_sha256": returned_hash,
                "inference_performed": True,
                "paired_cache_source_returned_actions_sha256": None,
            }
        ],
        "high_level_action_trace": [
            {
                "local_action_index": index,
                "source_action_index": index,
                "query_index": index // 5,
                "query_chunk_offset": index % 5,
                "native_observation_sha256": SHA,
                "aegis_proxy_source": (
                    "released_pre_settle_proxy_from_historical_action0_context"
                    if index == 0
                    else "current_observation_proxy"
                ),
                "aegis_executed": list(first_action),
                "aegis_qp": {
                    "status": "solved",
                    "context": {
                        "p1": copy.deepcopy(
                            released_proxy_diagnostic[
                                "released_pre_settle_p1"
                            ]
                        ),
                        "R1": copy.deepcopy(
                            released_proxy_diagnostic[
                                "released_pre_settle_R1"
                            ]
                        ),
                    },
                },
            }
            for index in range(2)
        ],
    }
    treatment_provider = {
        "source_start_action": 0,
        "first_aegis_proxy_source": (
            "released_pre_settle_proxy_from_historical_action0_context"
        ),
        "first_fresh_proxy_diagnostic": copy.deepcopy(
            released_proxy_diagnostic
        ),
        "historical_first_action_required_aegis_inputs": {
            "p1": copy.deepcopy(
                released_proxy_diagnostic["released_pre_settle_p1"]
            ),
            "R1": copy.deepcopy(
                released_proxy_diagnostic["released_pre_settle_R1"]
            ),
        },
        "first_aegis_input_binding_matches_historical": True,
        "recorded_suffix_actions_executed": False,
        "policy_queries": [
            {
                "query_index": 0,
                "rng_seed": 1234,
                "source_action_index": 0,
                "local_action_index": 0,
                "policy_input_fingerprint_sha256": policy_fingerprint,
                "returned_actions": returned_actions,
                "returned_action_shape": [10, 7],
                "returned_actions_sha256": returned_hash,
                "inference_performed": False,
                "paired_cache_source_returned_actions_sha256": returned_hash,
            }
        ],
        "high_level_action_trace": [
            {
                "local_action_index": index,
                "source_action_index": index,
                "query_index": index // 5,
                "query_chunk_offset": index % 5,
                "native_observation_sha256": SHA,
                "aegis_proxy_source": (
                    "released_pre_settle_proxy_from_historical_action0_context"
                    if index == 0
                    else "current_observation_proxy"
                ),
                "aegis_executed": list(first_action),
                "aegis_qp": {
                    "status": "solved",
                    "context": {
                        "p1": copy.deepcopy(
                            released_proxy_diagnostic[
                                "released_pre_settle_p1"
                            ]
                        ),
                        "R1": copy.deepcopy(
                            released_proxy_diagnostic[
                                "released_pre_settle_R1"
                            ]
                        ),
                    },
                },
            }
            for index in range(2)
        ],
    }
    payload = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "protocol_id": "fixture-direct-jv-v1",
        "run_id": "fixture-run",
        "case_id": "fixture-case",
        "partial_output_interpreted": False,
        "provenance": {
            "source": {"commit": SHA, "dirty": False},
            "allocation": {"job_id": "123"},
        },
        "controller_contract": _controller_contract(),
        "field": {
            "protected_robot_body_names": ["robot0_link5", "robot0_link6"],
            "resolved_geometry": {
                "robot_geom_ids": [41],
                "obstacle_geom_ids": [101],
                "link56_geom_ids": [41],
            },
            "selection_rule": "fixed_shared_link5_link6_surface_union_for_all_cases",
            "task_conditioned_link_selection": False,
            "protected_samples": {
                "sample_count": len(samples),
                "sample_ledger_sha256": _canonical_sha256(samples),
                "samples": samples,
            },
        },
        "pairing": {
            "settled_state_sha256": SHA,
            "same_initial_state": True,
            "same_first_policy_query": True,
            "first_query_cache": {
                "schema_version": "vlsa_poisson_paired_first_query_cache.v1",
                "first_query_index": 0,
                "store_count": 1,
                "reuse_count": 1,
                "producer": {
                    "producer_arm": "baseline",
                    "query_index": 0,
                    "rng_seed": 1234,
                    "policy_input_fingerprint_sha256": policy_fingerprint,
                    "returned_actions": returned_actions,
                    "returned_actions_sha256": returned_hash,
                },
                "reuse": {
                    "consumer_arm": "psf",
                    "query_index": 0,
                    "rng_seed": 1234,
                    "policy_input_fingerprint_sha256": policy_fingerprint,
                },
                "contract_valid": True,
            },
        },
        "providers": {
            "implementation": {
                "producer_module": PRODUCER_MODULE,
                "protocol_module": PROTOCOL_MODULE,
                "cbf_qp_module": CBF_MODULE,
                "joint_velocity_adapter_module": ADAPTER_MODULE,
                "contact_monitor_module": CONTACT_MONITOR_MODULE,
                "arm_runner_module": ARM_RUNNER_MODULE,
                "controller": "JOINT_VELOCITY",
                "decision_variable": "joint_velocity_rad_s",
                "source_sha256": {
                    "producer": SHA,
                    "protocol": SHA,
                    "arm_runner": SHA,
                    "cbf_qp": SHA,
                    "joint_velocity_adapter": SHA,
                    "contact_monitor": SHA,
                },
                "runtime_imports": [
                    CBF_MODULE,
                    ADAPTER_MODULE,
                    CONTACT_MONITOR_MODULE,
                ],
            },
            "adapter_only": baseline_provider,
            "adapter_plus_psf": treatment_provider,
        },
        "arms": {
            "adapter_only": _arm(False),
            "adapter_plus_psf": _arm(True),
        },
        "videos": {"baseline": _video(), "psf": _video()},
        "metrics": {},
        "classification": {},
    }
    payload["metrics"] = {
        "exact_settled_pair_start": True,
        "baseline_exposure_complete": True,
        "treatment_exposure_complete": True,
        "first_live_policy_query_paired": True,
        "live_policy_contract_valid": True,
        "released_aegis_contract_valid": True,
        "own_observation_chains_valid": True,
        "direct_joint_velocity_controller_used": True,
        "fixed_link56_protected_set_used": True,
        "hard_psf_rows_enforced": True,
        "shared_nominal_joint_bounds_inactive": True,
        "joint_velocity_tracking_valid": True,
        "no_unregistered_fallback_executed": True,
        "every_physics_substep_contact_checked": True,
        "paper_car_endpoint_ledger_complete": True,
        "baseline_link56_contact_present": True,
        "treatment_any_robot_selected_obstacle_contact_present": False,
        "treatment_link56_shifted_external_contact_present": False,
        "treatment_method_stop": False,
        "treatment_stalled_after_correction": False,
        "treatment_native_task_success": True,
        "treatment_task_success_after_correction": True,
        "treatment_paper_car_avoided": True,
        "material_correction_present": True,
        "material_correction_before_baseline_contact": True,
        "maximum_correction_norm_rad_s": 0.2,
        "correction_integral_rad": 0.002,
        "post_correction_measured_joint_motion_integral_rad": 0.03399999999999999,
        "post_correction_eef_path_length_m": 0.045000000000000005,
        "post_correction_executed_command_integral_rad": 0.033999999999999996,
        "post_correction_zero_command_fraction": 0.0,
    }
    payload["classification"] = {
        "schema_version": "vlsa_poisson_direct_joint_velocity_classification.v1",
        "classification": "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        "feasible": True,
        "pair_complete": True,
        "typed_terminal_negative": False,
        "baseline_reproduced": True,
        "contact_prevented": True,
        "material_correction": True,
        "useful_motion": True,
        "car_safe": True,
        "task_success": True,
        "method_stop": False,
        "safety_by_stopping": False,
    }
    return payload


def _terminal_fixture(kind):
    payload = _fixture()
    arm = payload["arms"]["adapter_plus_psf"]
    command_count = 5 if kind == "method_stop" else 6
    physics_count = 25 if kind == "method_stop" else 26
    arm["command_trace"] = arm["command_trace"][:command_count]
    arm["physics_trace"] = arm["physics_trace"][:physics_count]
    for key in (
        "filter_update_count",
        "qp_solve_count",
        "qp_postcheck_count",
        "joint_limit_postcheck_count",
        "issued_command_bound_check_count",
        "nominal_dynamic_bound_check_count",
        "pre_filter_field_observation_count",
    ):
        arm[key] = command_count
    arm["pre_filter_field_query_count"] = command_count * 2
    arm["physics_substep_count"] = physics_count
    arm["physics_trace_row_count"] = physics_count
    arm["monitor_observed_physics_substep_count"] = physics_count
    arm["exposure_complete"] = False
    arm["contact_terminated_early"] = kind == "contact"
    arm["method_terminated_early"] = kind == "method_stop"
    arm["method_stop"] = None
    task = arm["task"]
    terminal_row = copy.deepcopy(task["goal_progress_ledger"][-1])
    terminal_row.update(
        {
            "snapshot_kind": (
                "method_stop_prephysics_terminal_state_diagnostic"
                if kind == "method_stop"
                else "partial_action_terminal_state_diagnostic"
            ),
            "source_action_index": 1,
            "values": [False],
            "all_satisfied": False,
            "fraction": 0.0,
            "returned_done": None,
        }
    )
    task["goal_progress_ledger"] = task["goal_progress_ledger"][:2] + [terminal_row]
    task.update(
        {
            "completed_source_action_count": 1,
            "ever_task_success_at_or_after_branch": False,
            "first_task_success_source_action_index": None,
            "terminal_task_success": False,
            "returned_observation_sha256_ledger": [SHA],
        }
    )
    car = arm["paper_car"]
    car["endpoint_ledger"] = car["endpoint_ledger"][:1]
    car["endpoint_ledger_complete"] = False
    payload["videos"]["psf"] = _video(
        terminal=True, completed_action_count=1
    )
    if kind == "method_stop":
        arm["measurement"] = _selected_measurement(False)
        arm["registered_forbidden_contact"] = _registered_measurement(
            physics_count=physics_count
        )
        arm["literal_contact"].update(
            {
                "any_robot_selected_obstacle_present": False,
                "link56_present": False,
                "first_any_robot_physical_boundary": None,
                "first_link56_physical_boundary": None,
            }
        )
        arm["method_stop"] = {
            "local_action_index": 1,
            "source_action_index": 1,
            "source_action": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            "inner_control_index": 0,
            "physical_boundary": 25,
            "valid": False,
            "reason": "primal_infeasible",
            "diagnostics": {"status": "primal infeasible"},
        }
    else:
        arm["measurement"] = _selected_measurement(True, boundary=25)
        arm["registered_forbidden_contact"] = _registered_measurement(
            contact=True, physics_count=physics_count, boundary=25
        )
        arm["literal_contact"].update(
            {
                "any_robot_selected_obstacle_present": True,
                "link56_present": True,
                "first_any_robot_physical_boundary": 25,
                "first_link56_physical_boundary": 25,
            }
        )
        arm["physics_trace"][-1]["literal_contact_observed"] = True
        arm["physics_trace"][-1]["registered_forbidden_contact_observed"] = True
    metrics = dict(payload["metrics"])
    metrics.update(
        {
            "treatment_exposure_complete": False,
            "live_policy_contract_valid": False,
            "released_aegis_contract_valid": True,
            "hard_psf_rows_enforced": False,
            "joint_velocity_tracking_valid": False,
            "no_unregistered_fallback_executed": False,
            "every_physics_substep_contact_checked": False,
            "paper_car_endpoint_ledger_complete": False,
            "treatment_any_robot_selected_obstacle_contact_present": kind
            == "contact",
            "treatment_method_stop": kind == "method_stop",
            "treatment_native_task_success": False,
            "treatment_task_success_after_correction": False,
            "post_correction_measured_joint_motion_integral_rad": (
                0.014 if kind == "method_stop" else 0.0148
            ),
            "post_correction_eef_path_length_m": 0.02,
            "post_correction_executed_command_integral_rad": (
                0.014 if kind == "method_stop" else 0.018
            ),
        }
    )
    payload["metrics"] = metrics
    payload["classification"] = {
        "schema_version": "vlsa_poisson_direct_joint_velocity_classification.v1",
        "classification": (
            "METHOD_STOP" if kind == "method_stop" else "CONTACT_REMAINS_OR_SHIFTED"
        ),
        "feasible": False,
        "pair_complete": False,
        "typed_terminal_negative": True,
        "baseline_reproduced": True,
        "contact_prevented": kind == "method_stop",
        "material_correction": True,
        "useful_motion": True,
        "car_safe": True,
        "task_success": False,
        "method_stop": kind == "method_stop",
        "safety_by_stopping": kind == "method_stop",
    }
    return payload


class DirectJointVelocityArtifactValidatorTests(unittest.TestCase):
    def test_valid_fixture_is_useful_safe_task_success(self):
        result = validate_payload(_fixture(), inspect_source=False)
        self.assertTrue(result["artifact_valid"], result["discrepancies"])
        self.assertTrue(result["feasible"])
        self.assertEqual(
            result["outcome"], "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"
        )

    def test_real_case_binding_rejects_shortened_frozen_horizon(self):
        payload = _fixture()
        payload["protocol_id"] = "vlsa-poisson-direct-joint-velocity-action0-link56-v1"
        payload["case_id"] = "vlsa-t1-goal-ii-t3-e42"
        result = _validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "registered_action_count_differs_from_frozen_horizon",
            result["discrepancies"],
        )

    def test_method_stop_is_complete_scientific_negative(self):
        result = validate_payload(_terminal_fixture("method_stop"), inspect_source=False)
        self.assertTrue(result["artifact_valid"], result["discrepancies"])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["outcome"], "METHOD_STOP")
        self.assertTrue(result["recomputed_classification"]["typed_terminal_negative"])

    def test_contact_terminal_is_complete_scientific_negative(self):
        result = validate_payload(_terminal_fixture("contact"), inspect_source=False)
        self.assertTrue(result["artifact_valid"], result["discrepancies"])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["outcome"], "CONTACT_REMAINS_OR_SHIFTED")
        self.assertTrue(result["recomputed_classification"]["typed_terminal_negative"])

    def test_validator_does_not_import_producer_classifier(self):
        source = VALIDATOR.read_text(encoding="utf-8")
        self.assertNotIn("classify_direct_joint_velocity", source)
        self.assertNotIn("from main.poisson_fullbody.direct_joint_velocity", source)

    def test_source_audit_rejects_post_osc_torque_import(self):
        source = "\n".join(
            (
                "from main.poisson_fullbody.cbf_qp import HardCbfQp",
                "from main.poisson_fullbody.post_osc_torque_shield import SampledDataPostOscTorqueShield",
            )
        )
        audit = _audit_source_text(source)
        self.assertTrue(audit["forbidden_semantics"])

    def test_source_audit_accepts_direct_joint_velocity_imports(self):
        source = "\n".join(
            (
                "from main.poisson_fullbody.cbf_qp import HardCbfQp",
                "from main.poisson_fullbody.joint_velocity_adapter import TranslationalJointVelocityAdapter",
            )
        )
        audit = _audit_source_text(source)
        self.assertFalse(audit["forbidden_semantics"])
        self.assertTrue(audit["has_direct_joint_velocity_dependencies"])

    def test_source_audit_binds_all_six_serialized_hashes(self):
        payload = _fixture()
        sources = {
            "protocol": "PROTOCOL = 'direct_joint_velocity'\n",
            "producer": "from scripts import run_poisson_fast_feasibility\n",
            "arm_runner": (
                "from main.poisson_fullbody.cbf_qp import HardCbfQp\n"
                "from main.poisson_fullbody.joint_velocity_adapter import "
                "TranslationalJointVelocityAdapter\n"
            ),
            "contact_monitor": "MONITOR = 'registered_contact'\n",
            "cbf_qp": "QP = 'hard'\n",
            "joint_velocity_adapter": "ADAPTER = 'joint_velocity'\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for key, source in sources.items():
                path = Path(directory) / (key + ".py")
                path.write_text(source, encoding="utf-8")
                paths[key] = path
                payload["providers"]["implementation"]["source_sha256"][key] = (
                    hashlib.sha256(source.encode("utf-8")).hexdigest()
                )
            result = validate_payload(
                payload,
                source_path=paths["protocol"],
                runner_source_path=paths["producer"],
                arm_runner_source_path=paths["arm_runner"],
                contact_monitor_source_path=paths["contact_monitor"],
                cbf_source_path=paths["cbf_qp"],
                adapter_source_path=paths["joint_velocity_adapter"],
                inspect_source=True,
            )
        self.assertTrue(result["artifact_valid"], result["discrepancies"])

    def test_artifact_rejects_false_post_osc_torque_field(self):
        payload = _fixture()
        payload["controller_contract"]["post_osc_torque_conversion_used"] = False
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "artifact_contains_post_osc_torque_semantics", result["discrepancies"]
        )

    def test_controller_contract_rejects_stop_fallback(self):
        payload = _fixture()
        payload["controller_contract"]["fallback_policy"] = "zero_command"
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "controller_contract_fallback_policy_differs",
            result["discrepancies"],
        )

    def test_released_action_zero_proxy_source_mutation_is_rejected(self):
        payload = _fixture()
        payload["providers"]["adapter_plus_psf"][
            "first_aegis_proxy_source"
        ] = "current_observation_proxy"
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "provider_adapter_plus_psf_released_action0_proxy_source_differs",
            result["discrepancies"],
        )

    def test_released_action_zero_proxy_offset_mutation_is_rejected(self):
        payload = _fixture()
        payload["providers"]["adapter_only"][
            "first_fresh_proxy_diagnostic"
        ]["p1_l2_difference_m"] = 0.002
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "provider_adapter_only_p1_difference_reconstruction_failed",
            result["discrepancies"],
        )

    def test_released_action_zero_rotation_mutation_is_rejected(self):
        payload = _fixture()
        payload["providers"]["adapter_only"][
            "first_fresh_proxy_diagnostic"
        ]["released_pre_settle_R1"][0][0] = 9.0
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "provider_adapter_only_released_proxy_not_historical_authority",
            result["discrepancies"],
        )

    def test_post_action_zero_proxy_source_mutation_is_rejected(self):
        payload = _fixture()
        payload["providers"]["adapter_plus_psf"][
            "high_level_action_trace"
        ][1]["aegis_proxy_source"] = "wrong_source"
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "provider_adapter_plus_psf_proxy_source_trace_differs",
            result["discrepancies"],
        )

    def test_qp_count_mutation_is_rejected_despite_stale_metrics(self):
        payload = _fixture()
        payload["arms"]["adapter_plus_psf"]["qp_solve_count"] = 9
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn("treatment_qp_solve_count_differs", result["discrepancies"])

    def test_qp_residual_mutation_is_rejected(self):
        payload = _fixture()
        payload["arms"]["adapter_plus_psf"]["command_trace"][3][
            "safe_cbf_residual_minimum_m2_per_s"
        ] = -0.1
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "treatment_command_3_cbf_postcheck_failed", result["discrepancies"]
        )

    def test_executed_velocity_bound_mutation_is_rejected(self):
        payload = _fixture()
        payload["arms"]["adapter_plus_psf"]["command_trace"][4][
            "executed_qdot_rad_s"
        ][0] = 0.8
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "treatment_command_4_executed_outside_bounds",
            result["discrepancies"],
        )

    def test_task_conditioned_single_link_sampling_is_rejected(self):
        payload = _fixture()
        payload["field"]["task_conditioned_link_selection"] = True
        payload["field"]["protected_robot_body_names"] = ["robot0_link5"]
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "protected_robot_body_names_not_fixed_link5_link6",
            result["discrepancies"],
        )
        self.assertIn(
            "protected_surface_selection_is_task_conditioned",
            result["discrepancies"],
        )

    def test_pair_state_mutation_is_rejected(self):
        payload = _fixture()
        payload["arms"]["adapter_plus_psf"][
            "start_official_raw_bytes_sha256"
        ] = "2" * 64
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn("pair_initial_state_differs", result["discrepancies"])

    def test_first_query_action_mutation_is_rejected(self):
        payload = _fixture()
        payload["providers"]["adapter_plus_psf"]["high_level_action_trace"][0][
            "aegis_executed"
        ][0] = 0.02
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn("pair_first_executed_action_differs", result["discrepancies"])

    def test_shifted_treatment_contact_is_rejected_from_raw_ledger(self):
        payload = _fixture()
        measurement = payload["arms"]["adapter_plus_psf"][
            "registered_forbidden_contact"
        ]
        measurement.update(_registered_measurement(shifted=True))
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertFalse(result["recomputed_classification"]["contact_prevented"])

    def test_registered_contact_live_phase_uses_start_boundary(self):
        payload = _fixture()
        row = payload["arms"]["adapter_only"]["registered_forbidden_contact"][
            "rollout_contact_records"
        ][0]
        row["source_phase"] = "live_solver_phase_preintegration_geometry"
        row["observed_state_boundary"] = row["physical_boundary"]
        unhashed = dict(row)
        unhashed.pop("record_sha256")
        row["record_sha256"] = _canonical_sha256(unhashed)
        result = validate_payload(payload, inspect_source=False)
        self.assertTrue(result["artifact_valid"], result["discrepancies"])

    def test_registered_scope_must_match_resolved_geometry(self):
        payload = _fixture()
        for arm in payload["arms"].values():
            scope = arm["registered_forbidden_contact"]["scope"]
            scope["selected_obstacle_geom_ids"] = [102]
            unhashed = dict(scope)
            unhashed.pop("identity_sha256")
            scope["identity_sha256"] = _canonical_sha256(unhashed)
            arm["registered_forbidden_contact"]["scope_identity_sha256"] = scope[
                "identity_sha256"
            ]
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertIn(
            "registered_contact_scope_differs_from_resolved_geometry",
            result["discrepancies"],
        )

    def test_stalled_treatment_is_not_feasible(self):
        payload = _fixture()
        arm = payload["arms"]["adapter_plus_psf"]
        for row in arm["command_trace"]:
            row["executed_qdot_rad_s"] = [0.0] * 7
            row["correction_l2_rad_s"] = 0.4
        for row in arm["physics_trace"]:
            row["measured_qvel_rad_s"] = [0.0] * 7
            row["issued_qvel_rad_s"] = [0.0] * 7
            row["tracking_error_rad_s"] = [0.0] * 7
            row["eef_position_world_m"] = [0.0, 0.0, 0.0]
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertFalse(result["recomputed_classification"]["useful_motion"])

    def test_native_task_failure_is_rejected(self):
        payload = _fixture()
        task = payload["arms"]["adapter_plus_psf"]["task"]
        task["goal_progress_ledger"][-1].update(
            {"values": [False], "all_satisfied": False, "fraction": 0.0}
        )
        task.update(
            {
                "ever_task_success_at_or_after_branch": False,
                "first_task_success_source_action_index": None,
                "terminal_task_success": False,
            }
        )
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertFalse(result["recomputed_classification"]["task_success"])

    def test_tracking_threshold_crossing_is_rejected(self):
        payload = _fixture()
        row = payload["arms"]["adapter_plus_psf"]["physics_trace"][8]
        row["measured_qvel_rad_s"][0] += 0.2
        row["tracking_error_rad_s"][0] += 0.2
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertFalse(
            result["recomputed_metrics"]["joint_velocity_tracking_valid"]
        )

    def test_paper_car_crossing_is_reconstructed_from_raw_positions(self):
        payload = _fixture()
        car = payload["arms"]["adapter_plus_psf"]["paper_car"]
        row = car["endpoint_ledger"][1]
        row["active_obstacle_position_m"] = [0.002, 0.0, 0.0]
        row["l1_displacement_from_settled_m"] = 0.002
        row["paper_collision"] = True
        car["maximum_active_obstacle_l1_displacement_m"] = 0.002
        car["collision_first_source_action_index"] = 1
        car["paper_collision_avoidance"] = False
        result = validate_payload(payload, inspect_source=False)
        self.assertFalse(result["artifact_valid"])
        self.assertFalse(result["recomputed_classification"]["car_safe"])
        self.assertEqual(result["recomputed_classification"]["classification"], "CAR_FAILURE")

    def test_cli_rejects_nonfrozen_short_horizon_fixture(self):
        payload = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / payload["run_id"]
            run_root.mkdir()
            result_path = run_root / "result.json"
            publish_hashed_json(result_path, payload)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--result",
                    str(result_path),
                    "--skip-source-inspection",
                ],
                cwd=str(ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn("case_id_not_frozen", summary["discrepancies"])


if __name__ == "__main__":
    unittest.main()
