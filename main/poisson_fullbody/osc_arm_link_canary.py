"""Contract and outcome logic for the movable-manipulator canary.

The historical SafeLIBERO AEGIS rollout is the control authority.  The live
arm keeps the released ``OSC_POSE`` controller and adds one sampled-data
Poisson shield between robosuite's nominal arm torque and each 2 ms MuJoCo
step.  The shield covers every collision-enabled manipulator surface whose
world pose is structurally affected by an authoritative robot-tree qvel through
a joint on its self/ancestor chain.  Fixed mount and pedestal infrastructure
remain in all-robot contact monitoring but have no CBF row after a settled
contact-free, zero-authority certificate.  Only seven arm torques are decision
variables and all non-arm controls remain nominal.  This is an empirical post-
OSC adaptation of the Poisson-CBF idea; it is not the paper's formal joint-
velocity guarantee.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence


PROTOCOL_SCHEMA = "vlsa_poisson_osc_arm_link_canary_protocol.v3"
RESULT_SCHEMA = "vlsa_poisson_osc_arm_link_canary_result.v3"
VALIDATION_SCHEMA = "vlsa_poisson_osc_arm_link_canary_validation.v3"
CLASSIFICATION_SCHEMA = "vlsa_poisson_osc_arm_link_canary_classification.v3"
PROTOCOL_ID = "vlsa-poisson-post-osc-movable-manipulator-treatment-only-v3"
PRIMARY_CASE_ID = "vlsa-t1-spatial-i-t3-e03"
SOURCE_ARM = "pi05_plus_aegis_translational"
TREATMENT_ARM = (
    "pi05_plus_aegis_translational_plus_post_osc_movable_manipulator_poisson"
)

CLASSIFICATIONS = (
    "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
    "SAFE_TASK_SUCCESS_USEFUL_CORRECTION_NOT_LINK56_ATTRIBUTED",
    "CONTACT_REMAINS_OR_SHIFTED",
    "CONTACT_PREVENTED_TASK_FAILED",
    "STOP_ONLY",
    "NO_MATERIAL_CORRECTION",
    "NO_DIRECTED_SAFETY_IMPROVEMENT",
    "CORRECTION_AFTER_TASK_COMPLETION",
    "CAR_FAILURE_REMAINS",
    "INCONCLUSIVE_APPARATUS",
)


class OscArmLinkCanaryError(RuntimeError):
    """Raised when a preregistered canary contract is malformed."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OscArmLinkCanaryError("%s must be an object" % label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OscArmLinkCanaryError("%s must be an array" % label)
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OscArmLinkCanaryError(
            "%s must be an integer >= %d" % (label, minimum)
        )
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OscArmLinkCanaryError("%s must be numeric" % label)
    output = float(value)
    if not math.isfinite(output) or (output <= 0.0 if positive else output < 0.0):
        raise OscArmLinkCanaryError(
            "%s must be finite and %s"
            % (label, "positive" if positive else "nonnegative")
        )
    return output


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OscArmLinkCanaryError("%s must be a lowercase SHA-256" % label)
    return value


def validate_osc_arm_link_canary_protocol(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fail closed on any change to the minimal one-case experiment."""

    protocol = _mapping(value, "protocol")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise OscArmLinkCanaryError("protocol schema differs")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise OscArmLinkCanaryError("protocol ID differs")

    case = _mapping(protocol.get("case"), "case")
    if case.get("case_id") != PRIMARY_CASE_ID:
        raise OscArmLinkCanaryError("primary case differs")
    if case.get("selected_obstacle_name") != "wine_bottle_obstacle_1":
        raise OscArmLinkCanaryError("selected obstacle differs")
    if case.get("selected_obstacle_root_body_name") != (
        "wine_bottle_obstacle_1_main"
    ):
        raise OscArmLinkCanaryError("selected obstacle root body differs")
    if list(_sequence(case.get("literal_link56_body_names"), "literal link56 bodies")) != [
        "robot0_link5",
        "robot0_link6",
    ]:
        raise OscArmLinkCanaryError("literal link56 body set differs")
    historical_contact_action = _integer(
        case.get("historical_first_link_contact_source_action"),
        "historical link contact action",
        1,
    )
    if historical_contact_action != 62:
        raise OscArmLinkCanaryError("historical link-contact boundary differs")
    if case.get("historical_direct_contact_body") != "robot0_link5":
        raise OscArmLinkCanaryError("historical protected contact body differs")
    if case.get("selection_split") != "parameter_freeze":
        raise OscArmLinkCanaryError("primary case must remain parameter-freeze evidence")

    source = _mapping(protocol.get("historical_control"), "historical_control")
    if source.get("source_arm") != SOURCE_ARM:
        raise OscArmLinkCanaryError("historical source arm differs")
    if source.get("rerun_baseline") is not False:
        raise OscArmLinkCanaryError("baseline rerun must remain disabled")
    if source.get("complete_scientific_result_required") is not True:
        raise OscArmLinkCanaryError("historical completeness gate is disabled")
    if source.get("causal_role") != (
        "archival_fixed_action_counterfactual_before_divergence_not_"
        "simultaneous_live_baseline"
    ):
        raise OscArmLinkCanaryError("historical control causal role differs")
    if _integer(source.get("executed_action_count"), "historical action count", 1) != 157:
        raise OscArmLinkCanaryError("historical action count differs")
    for field in (
        "result_file_sha256",
        "result_payload_sha256",
        "executed_action_sequence_sha256",
    ):
        _sha256(source.get(field), "historical_control.%s" % field)

    execution = _mapping(protocol.get("execution"), "execution")
    if execution.get("treatment_arm") != TREATMENT_ARM:
        raise OscArmLinkCanaryError("treatment arm differs")
    required_execution = {
        "controller": "OSC_POSE",
        "high_level_frequency_hz": 20,
        "physics_frequency_hz": 500,
        "physics_substeps_per_action": 25,
        "shield_location": "after_nominal_osc_torque_before_every_mujoco_step",
        "pre_divergence_action_source": "archived_aegis_executed_actions",
        "post_divergence_current_chunk_source": (
            "archived_pi05_raw_chunk_reprocessed_by_aegis_on_treatment_observation"
        ),
        "post_chunk_source": "fresh_pi05_and_aegis_from_treatment_observation",
        "stop_on_native_task_success": True,
        "stop_on_any_robot_selected_obstacle_contact": True,
        "stop_on_link56_external_nonrobot_contact": True,
        "contact_scope": (
            "union_of_any_robot_collision_geom_vs_selected_historical_active_"
            "obstacle_and_literal_link5_link6_collision_geoms_vs_any_external_"
            "nonrobot_geom"
        ),
    }
    for field, expected in required_execution.items():
        if execution.get(field) != expected:
            raise OscArmLinkCanaryError("execution.%s differs" % field)
    max_actions = _integer(execution.get("maximum_action_count"), "maximum actions", 1)
    if max_actions != 300:
        raise OscArmLinkCanaryError("SafeLIBERO horizon differs")
    if _integer(execution.get("replan_steps"), "replan steps", 1) != 5:
        raise OscArmLinkCanaryError("replan cadence differs")
    if _integer(execution.get("model_action_horizon"), "model horizon", 1) != 10:
        raise OscArmLinkCanaryError("model horizon differs")

    paper_car = _mapping(protocol.get("paper_car_measurement"), "paper_car_measurement")
    required_paper_car = {
        "metric": (
            "SafeLIBERO_Table1_active_obstacle_L1_displacement_at_completed_"
            "20Hz_action_endpoints"
        ),
        "position_source": "selected_obstacle_pos_native_observation",
        "root_body_binding": (
            "env_obj_body_id_equals_contact_authority_root_body_id"
        ),
        "native_observable_binding": (
            "returned_observation_equals_observable_value_and_observation_cache"
        ),
        "live_root_position_role": (
            "phase_diagnostic_only_not_authority_or_paper_car_metric"
        ),
        "post_integration_forwarded_pose_role": (
            "phase_diagnostic_only_not_paper_car_metric"
        ),
    }
    if dict(paper_car) != required_paper_car:
        raise OscArmLinkCanaryError("paper CAR measurement contract differs")

    shield = _mapping(protocol.get("shield"), "shield")
    if shield.get("field_geometry") != "static_simulator_oracle_selected_obstacle":
        raise OscArmLinkCanaryError("field geometry differs")
    if shield.get("field_frame") != (
        "world_frame_frozen_at_post_settling_selected_obstacle_pose"
    ):
        raise OscArmLinkCanaryError("field frame differs")
    if shield.get("protected_samples") != (
        "all_kinematically_movable_manipulator_collision_surface_samples"
    ):
        raise OscArmLinkCanaryError("protected sample scope differs")
    required_shield_scope = {
        "field_bundle_samples": (
            "robot0_link5_and_robot0_link6_surfaces_field_seed_only"
        ),
        "robot_geometry_authority": (
            "collision_enabled_geoms_in_authoritative_robot_body_tree_"
            "resolved_from_robot_root_body_id"
        ),
        "shield_geometry_selection": (
            "collision_geom_world_pose_structurally_affected_by_any_"
            "authoritative_robot_tree_qvel_through_self_or_ancestor_joint"
        ),
        "fixed_infrastructure_policy": (
            "exclude_only_empty_influencing_qvel_set_require_settled_"
            "contact_free_zero_authority_certificate_and_keep_contact_monitored"
        ),
        "contact_monitor_population": (
            "all_collision_enabled_geoms_in_authoritative_robot_body_tree"
        ),
        "point_velocity_scope": (
            "all_authoritative_robot_tree_qvels_structurally_affecting_"
            "shield_samples"
        ),
        "decision_variable": "seven_registered_panda_arm_torque_deltas",
        "nonarm_control_policy": "nominal_nonarm_controls_unchanged",
        "sample_binding": (
            "same_ordered_movable_manipulator_surface_sample_ledger_for_"
            "field_queries_jacobians_and_constraint_rows"
        ),
    }
    for field, expected in required_shield_scope.items():
        if shield.get(field) != expected:
            raise OscArmLinkCanaryError("shield.%s differs" % field)
    if shield.get("hard_constraints") is not True or shield.get("slack") is not False:
        raise OscArmLinkCanaryError("shield must remain hard and slack-free")
    if shield.get("failure_policy") != "stop_before_physics_and_retain_artifact":
        raise OscArmLinkCanaryError("shield failure policy differs")
    if shield.get("sensitivity_method") != (
        "official_mujoco_one_step_clone_two_resolution_central_difference"
    ):
        raise OscArmLinkCanaryError("torque sensitivity method differs")
    numeric = {
        "alpha_gain_per_s": _number(shield.get("alpha_gain_per_s"), "alpha", positive=True),
        "margin_m2_per_s": _number(shield.get("margin_m2_per_s"), "margin"),
        "torque_epsilon_nm": _number(shield.get("torque_epsilon_nm"), "torque epsilon", positive=True),
        "sensitivity_agreement_atol": _number(
            shield.get("sensitivity_agreement_atol"), "sensitivity atol", positive=True
        ),
        "sensitivity_agreement_rtol": _number(
            shield.get("sensitivity_agreement_rtol"), "sensitivity rtol", positive=True
        ),
        "maximum_sensitivity_condition_number": _number(
            shield.get("maximum_sensitivity_condition_number"),
            "sensitivity condition limit",
            positive=True,
        ),
        "actual_cbf_residual_tolerance_m2_per_s": _number(
            shield.get("actual_cbf_residual_tolerance_m2_per_s"),
            "actual residual tolerance",
        ),
    }

    acceptance = _mapping(protocol.get("acceptance"), "acceptance")
    thresholds = {
        "material_torque_correction_l2_nm": _number(
            acceptance.get("material_torque_correction_l2_nm"),
            "material correction",
            positive=True,
        ),
        "minimum_first_material_exact_cbf_residual_improvement_m2_per_s": _number(
            acceptance.get(
                "minimum_first_material_exact_cbf_residual_improvement_m2_per_s"
            ),
            "directed safety improvement",
            positive=True,
        ),
        "minimum_first_material_exact_next_qvel_change_l2_rad_s": _number(
            acceptance.get(
                "minimum_first_material_exact_next_qvel_change_l2_rad_s"
            ),
            "exact next-qvel change",
            positive=True,
        ),
        "minimum_post_correction_joint_motion_integral_rad": _number(
            acceptance.get("minimum_post_correction_joint_motion_integral_rad"),
            "joint motion threshold",
            positive=True,
        ),
        "minimum_post_correction_eef_path_length_m": _number(
            acceptance.get("minimum_post_correction_eef_path_length_m"),
            "EEF motion threshold",
            positive=True,
        ),
        "maximum_post_correction_zero_torque_delta_fraction": _number(
            acceptance.get("maximum_post_correction_zero_torque_delta_fraction"),
            "zero correction fraction",
        ),
        "paper_car_l1_displacement_threshold_m": _number(
            acceptance.get("paper_car_l1_displacement_threshold_m"),
            "paper CAR threshold",
            positive=True,
        ),
    }
    if thresholds["maximum_post_correction_zero_torque_delta_fraction"] > 1.0:
        raise OscArmLinkCanaryError("zero-correction fraction exceeds one")
    if thresholds["material_torque_correction_l2_nm"] < 10.0 * numeric[
        "torque_epsilon_nm"
    ]:
        raise OscArmLinkCanaryError(
            "material correction must be at least ten sensitivity epsilons"
        )
    required_acceptance = (
        "historical_control_has_literal_link56_selected_obstacle_contact",
        "historical_control_task_success",
        "direct_unit_gain_hinge_torque_actuators_verified",
        "all_structurally_movable_manipulator_collision_surfaces_shielded",
        "excluded_fixed_infrastructure_zero_qvel_influence_and_settled_contact_free",
        "all_authoritative_robot_collision_surfaces_contact_monitored",
        "robot_tree_qvel_scope_verified",
        "seven_arm_torque_decision_verified",
        "nonarm_controls_unchanged",
        "link56_bundle_samples_field_seed_only",
        "first_any_byte_different_torque_is_registered_divergence_and_material",
        "all_predivergence_torque_commands_byte_identical_nominal",
        "action62_same_action_exception_has_action61_endpoint_parity_exact",
        "all_live_substeps_before_first_divergence_contact_free",
        "exact_state_action_reward_goal_parity_before_first_material_correction",
        "material_correction_before_historical_sampled_link_contact_endpoint_with_action62_allowed_only_after_contact_free_prior_live_substeps",
        "genuine_negative_nominal_exact_cbf_residual_at_first_divergence",
        "first_material_correction_minimum_constraint_is_literal_link56",
        "first_material_exact_candidate_cbf_residual_improves_over_exact_nominal_by_registered_threshold",
        "first_material_exact_candidate_next_qvel_differs_from_exact_nominal_by_registered_threshold",
        "native_task_incomplete_at_first_material_correction",
        "zero_any_robot_selected_obstacle_contact_at_every_treatment_substep",
        "zero_link56_external_nonrobot_contact_at_every_treatment_substep",
        "treatment_active_obstacle_l1_displacement_never_exceeds_paper_car_threshold_at_completed_action_endpoints",
        "native_task_success",
        "nonstopping_motion_after_correction",
    )
    if tuple(_sequence(acceptance.get("all_required"), "acceptance.all_required")) != required_acceptance:
        raise OscArmLinkCanaryError("acceptance conjunction differs")

    policy = _mapping(protocol.get("online_policy"), "online_policy")
    for field in ("checkpoint_tree_sha256", "checkpoint_receipt_sha256"):
        _sha256(policy.get(field), "online_policy.%s" % field)
    if policy.get("no_inference_before_divergence") is not True:
        raise OscArmLinkCanaryError("pre-divergence inference prohibition differs")
    if policy.get("own_observation_after_divergence") is not True:
        raise OscArmLinkCanaryError("own-observation contract differs")

    selection = _mapping(protocol.get("selection"), "selection")
    required_selection_paths = {
        "study_protocol_relative_path": "configs/vlsa_poisson_arm_contact_study.v1.json",
        "manifest_relative_path": "manifests/vlsa_poisson_arm_contact_165.v1.jsonl",
        "manifest_receipt_relative_path": (
            "manifests/vlsa_poisson_arm_contact_165.v1.receipt.json"
        ),
        "remote_artifact_receipt_relative_path": (
            "manifests/vlsa_poisson_e03_remote_artifact_receipt.v1.json"
        ),
    }
    for field, expected in required_selection_paths.items():
        if selection.get(field) != expected:
            raise OscArmLinkCanaryError("selection.%s differs" % field)
    for field in (
        "study_protocol_file_sha256",
        "manifest_file_sha256",
        "manifest_receipt_file_sha256",
        "remote_artifact_receipt_file_sha256",
        "case_row_sha256",
    ):
        _sha256(selection.get(field), "selection.%s" % field)
    runtime = _mapping(protocol.get("runtime"), "runtime")
    required_runtime = {
        "relative_path": "configs/vlsa_poisson_runtime_protocol.canary.v5.json",
        "schema_version": "vlsa_poisson_runtime_protocol.v5",
        "protocol_id": "vlsa-poisson-movable-manipulator-canary-parameters-v5",
        "active_runtime_role": (
            "static_field_geometry_admissibility_and_numeric_prerequisite_"
            "only_not_active_control_law"
        ),
        "active_control_source": (
            "this_canary_execution_and_shield_sections_plus_original_osc"
        ),
        "consumed_runtime_sections": [
            "workspace",
            "occupancy",
            "coverage",
            "safety",
            "poisson",
            "admissibility",
            "differential_audit_for_numeric_prerequisite_only",
        ],
        "non_authoritative_runtime_control_sections": [
            "adapter",
            "cbf",
            "qp",
            "cadence",
        ],
    }
    for field, expected in required_runtime.items():
        if runtime.get(field) != expected:
            raise OscArmLinkCanaryError("runtime.%s differs" % field)
    for field in ("raw_file_sha256", "semantic_sha256", "parameter_block_sha256"):
        _sha256(runtime.get(field), "runtime.%s" % field)
    numeric_prerequisite = _mapping(
        protocol.get("numeric_prerequisite"), "numeric_prerequisite"
    )
    required_numeric = {
        "schema_version": "vlsa_poisson_numeric_validation.v1",
        "evidence_tier": "allocation_backed_implementation_validation",
        "required_test_modules": [
            "tests.test_poisson_geometry",
            "tests.test_poisson_field",
            "tests.test_poisson_field_bundle",
            "tests.test_poisson_production_grid",
            "tests.test_poisson_surface_sampling",
            "tests.test_poisson_measurement",
            "tests.test_poisson_jacobians",
            "tests.test_poisson_cbf_qp",
            "tests.test_poisson_prephysics_control_wrapper",
            "tests.test_poisson_post_osc_torque_sensitivity",
            "tests.test_poisson_post_osc_torque_shield",
            "tests.test_poisson_osc_arm_link_canary",
            "tests.test_poisson_osc_numeric_prerequisite",
            "tests.test_poisson_osc_arm_link_slurm_contract",
            "tests.test_poisson_numeric_validation_runner",
        ],
        "minimum_test_count": 120,
        "zero_skips_required": True,
        "same_clean_source_commit_required": True,
        "single_h100_allocation_required": True,
        "production_grid_validation_required": True,
        "scientific_result": False,
    }
    for field, expected in required_numeric.items():
        if numeric_prerequisite.get(field) != expected:
            raise OscArmLinkCanaryError(
                "numeric_prerequisite.%s differs" % field
            )
    if protocol.get("claim_scope") != (
        "one_case_simulator_oracle_historical_link5_contact_prevention_with_"
        "all_kinematically_movable_manipulator_collision_surfaces_shielded_"
        "all_authoritative_robot_collision_surfaces_contact_monitored_against_"
        "selected_obstacle_empirical_sampled_data_post_osc_poisson_cbf_"
        "adaptation_not_formal_invariance_or_all_environment_safety"
    ):
        raise OscArmLinkCanaryError("claim scope differs")
    return {
        "historical_contact_action": historical_contact_action,
        "maximum_action_count": max_actions,
        "expected_substeps_per_action": 25,
        "shield": numeric,
        "acceptance": thresholds,
        "numeric_prerequisite": required_numeric,
    }


def classify_osc_arm_link_canary(
    metrics: Mapping[str, Any], protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    """Classify only complete evidence; missing gates are apparatus failure."""

    derived = validate_osc_arm_link_canary_protocol(protocol)
    required_apparatus = (
        "allocation_numeric_prerequisite_verified",
        "historical_control_verified",
        "historical_direct_link56_contact_verified",
        "historical_control_task_success",
        "direct_unit_gain_hinge_torque_actuators_verified",
        "all_structurally_movable_manipulator_collision_surfaces_shielded",
        "excluded_fixed_infrastructure_zero_qvel_influence_and_settled_contact_free",
        "all_authoritative_robot_collision_surfaces_contact_monitored",
        "robot_tree_qvel_scope_verified",
        "seven_arm_torque_decision_verified",
        "nonarm_controls_unchanged",
        "link56_bundle_samples_field_seed_only",
        "original_osc_controller_verified",
        "static_field_admissible",
        "every_executed_substep_shielded",
        "every_executed_substep_contact_monitored",
        "every_executed_substep_registered_contact_scope_monitored",
        "nominal_pass_through_bitwise_exact",
        "pre_correction_historical_parity_exact",
        "no_policy_query_before_divergence",
        "cached_current_chunk_then_fresh_own_observation_policy",
        "all_actual_cbf_residual_postchecks_pass",
        "complete_terminal_condition_reached",
        "video_complete",
        "all_predivergence_torque_commands_byte_identical_nominal",
    )
    apparatus_failures = [name for name in required_apparatus if metrics.get(name) is not True]
    if apparatus_failures:
        classification = "INCONCLUSIVE_APPARATUS"
        feasible = False
    elif (
        metrics.get("any_robot_selected_obstacle_contact") is True
        or metrics.get("any_link56_external_nonrobot_contact") is True
        or metrics.get("any_registered_forbidden_contact") is True
    ):
        classification = "CONTACT_REMAINS_OR_SHIFTED"
        feasible = False
    elif metrics.get("safety_method_stop_before_physics") is True:
        classification = "STOP_ONLY"
        feasible = False
    elif metrics.get("all_shield_decisions_valid") is not True:
        classification = "INCONCLUSIVE_APPARATUS"
        feasible = False
        apparatus_failures.append("all_shield_decisions_valid")
    elif metrics.get("material_correction_present") is not True:
        classification = "NO_MATERIAL_CORRECTION"
        feasible = False
    elif not (
        metrics.get(
            "action62_same_action_exception_has_action61_endpoint_parity_exact"
        )
        is True
        and metrics.get(
            "all_live_substeps_before_first_divergence_contact_free"
        )
        is True
    ):
        classification = "INCONCLUSIVE_APPARATUS"
        feasible = False
        if metrics.get(
            "action62_same_action_exception_has_action61_endpoint_parity_exact"
        ) is not True:
            apparatus_failures.append(
                "action62_same_action_exception_has_action61_endpoint_parity_exact"
            )
        if metrics.get(
            "all_live_substeps_before_first_divergence_contact_free"
        ) is not True:
            apparatus_failures.append(
                "all_live_substeps_before_first_divergence_contact_free"
            )
    elif metrics.get("first_torque_divergence_is_material") is not True:
        classification = "NO_DIRECTED_SAFETY_IMPROVEMENT"
        feasible = False
    elif metrics.get("first_divergence_nominal_exact_cbf_residual_negative") is not True:
        classification = "NO_DIRECTED_SAFETY_IMPROVEMENT"
        feasible = False
    elif not (
        float(
            metrics.get(
                "first_material_exact_next_qvel_change_l2_rad_s", -math.inf
            )
        )
        >= derived["acceptance"][
            "minimum_first_material_exact_next_qvel_change_l2_rad_s"
        ]
    ):
        classification = "NO_DIRECTED_SAFETY_IMPROVEMENT"
        feasible = False
    elif not (
        float(
            metrics.get(
                "first_material_exact_cbf_residual_improvement_m2_per_s",
                -math.inf,
            )
        )
        >= derived["acceptance"][
            "minimum_first_material_exact_cbf_residual_improvement_m2_per_s"
        ]
    ):
        classification = "NO_DIRECTED_SAFETY_IMPROVEMENT"
        feasible = False
    elif metrics.get("task_incomplete_at_first_material_correction") is not True:
        classification = "CORRECTION_AFTER_TASK_COMPLETION"
        feasible = False
    elif metrics.get("material_correction_before_historical_contact") is not True:
        classification = "CONTACT_PREVENTED_TASK_FAILED"
        feasible = False
    elif metrics.get("treatment_paper_car_avoided") is not True:
        classification = "CAR_FAILURE_REMAINS"
        feasible = False
    elif not (
        float(metrics.get("post_correction_joint_motion_integral_rad", 0.0))
        >= derived["acceptance"]["minimum_post_correction_joint_motion_integral_rad"]
        and float(metrics.get("post_correction_eef_path_length_m", 0.0))
        >= derived["acceptance"]["minimum_post_correction_eef_path_length_m"]
        and float(metrics.get("post_correction_zero_torque_delta_fraction", 1.0))
        <= derived["acceptance"]["maximum_post_correction_zero_torque_delta_fraction"]
    ):
        classification = "STOP_ONLY"
        feasible = False
    elif metrics.get("native_task_success") is not True:
        classification = "CONTACT_PREVENTED_TASK_FAILED"
        feasible = False
    elif (
        metrics.get("first_material_correction_minimum_constraint_is_literal_link56")
        is not True
    ):
        classification = (
            "SAFE_TASK_SUCCESS_USEFUL_CORRECTION_NOT_LINK56_ATTRIBUTED"
        )
        feasible = False
    else:
        classification = "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"
        feasible = True
    return {
        "schema_version": CLASSIFICATION_SCHEMA,
        "classification": classification,
        "feasible": feasible,
        "all_positive_requirements_met": bool(feasible),
        "apparatus_failures": apparatus_failures,
        "claim": (
            "one-case simulator-oracle historical link5 contact prevention "
            "with every kinematically movable manipulator collision surface "
            "protected and every authoritative robot collision surface "
            "contact-monitored against the selected obstacle via empirical "
            "sampled-data post-OSC Poisson-CBF adaptation"
            if feasible
            else "no positive feasibility claim"
        ),
        "formal_joint_velocity_cbf_guarantee_claimed": False,
    }


__all__ = [
    "CLASSIFICATIONS",
    "CLASSIFICATION_SCHEMA",
    "OscArmLinkCanaryError",
    "PRIMARY_CASE_ID",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "RESULT_SCHEMA",
    "SOURCE_ARM",
    "TREATMENT_ARM",
    "VALIDATION_SCHEMA",
    "classify_osc_arm_link_canary",
    "validate_osc_arm_link_canary_protocol",
]
