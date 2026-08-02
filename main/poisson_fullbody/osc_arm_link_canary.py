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

The additive v4 experiment constrains the same common link-5/link-6 collision
surface set in both e05 and e42.  The case-specific ``target_link_body_names``
records identify only the archived contact and first-material-correction
attribution used for causal evaluation; they never select the field, samples,
Jacobians, or QP rows.  Whole-robot contact with the selected obstacle and
literal link-5/link-6 contact with every external nonrobot geometry remain
monitored, so shifting contact is a registered failure rather than a safety
success.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence


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


# The v4 contract is intentionally additive.  The completed e03 v3 experiment
# remains immutable evidence for a different (whole-manipulator) treatment.
TARGET_LINK_PROTOCOL_SCHEMA = "vlsa_poisson_osc_target_link_canary_protocol.v4"
TARGET_LINK_RESULT_SCHEMA = "vlsa_poisson_osc_target_link_canary_result.v4"
TARGET_LINK_VALIDATION_SCHEMA = (
    "vlsa_poisson_osc_target_link_canary_validation.v4"
)
TARGET_LINK_CLASSIFICATION_SCHEMA = (
    "vlsa_poisson_osc_target_link_canary_classification.v4"
)
TARGET_LINK_PAIR_CLASSIFICATION_SCHEMA = (
    "vlsa_poisson_osc_target_link_pair_classification.v4"
)
TARGET_LINK_PROTOCOL_ID = (
    "vlsa-poisson-post-osc-common-link56-treatment-only-v4"
)
TARGET_LINK_TREATMENT_ARM = (
    "pi05_plus_aegis_translational_plus_post_osc_common_link56_poisson"
)
TARGET_LINK_FROZEN_RUNTIME_PARAMETER_SHA256 = (
    "61ac3790e704aec03283624990c5874913907140b5cb0b8bd776e8fcae6d4eca"
)
TARGET_LINK_ALLOWED_CASE_IDS = (
    "vlsa-t1-goal-ii-t0-e05",
    "vlsa-t1-goal-ii-t3-e42",
)
TARGET_LINK_LITERAL_LINK56_BODY_NAMES = ("robot0_link5", "robot0_link6")
TARGET_LINK_PROTECTED_LINK_BODY_NAMES = TARGET_LINK_LITERAL_LINK56_BODY_NAMES


def _frozen_case(**values: Any) -> Mapping[str, Any]:
    return MappingProxyType(values)


TARGET_LINK_ALLOWED_CASE_REGISTRY = MappingProxyType(
    {
        "vlsa-t1-goal-ii-t0-e05": _frozen_case(
            case_id="vlsa-t1-goal-ii-t0-e05",
            selected_obstacle_name="moka_pot_obstacle_1",
            selected_obstacle_root_body_name="moka_pot_obstacle_1_main",
            target_link_body_names=("robot0_link5",),
            historical_literal_link_contact_body_names=(
                "robot0_link5",
                "robot0_link6",
            ),
            historical_first_target_link_contact_source_action=187,
            historical_prior_completed_action_endpoint=186,
            historical_car_first_crossing_source_action=188,
            historical_executed_action_count=237,
            historical_maximum_active_obstacle_l1_displacement_m=(
                0.022805614327524107
            ),
            historical_result_file_sha256=(
                "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
            ),
            historical_result_payload_sha256=(
                "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c"
            ),
            historical_executed_action_sequence_sha256=(
                "e5e2df4efee667fee0b0b7596dbc203a3cce84e67cb3507cb839132d2ee67481"
            ),
            initial_state_sha256=(
                "714334cdae0ad6cd8540115802e9bf9b3591449e29d712ccc4651d3e89c22cf6"
            ),
            initial_observation_sha256=(
                "ca4d9aafa10674b7a2faefacb0c115f25072ff4bedffbe5d8f54ad87c4809775"
            ),
            settled_simulator_state_sha256=(
                "5a72a870b8368d0a6508428bb89dce75ccc6f28e86a2349edae35b2618918741"
            ),
            policy_noise_schedule_sha256=(
                "b50949ebc3b6a774f3800922487c13b00ee76d1f26cec5d8d7cdd8d0197f63e5"
            ),
            source_table1_manifest_row_sha256=(
                "d7a11ccf75af4b9f9823cad820fe261a4a791fc9891fe1f605cbed23648c88df"
            ),
            selection_case_row_sha256=(
                "058c679f40f13698eef629a69829d7ea98cfc824f64e2486dec9a0b1eb2b3dd6"
            ),
            remote_artifact_receipt_relative_path=(
                "manifests/vlsa_poisson_e05_remote_artifact_receipt.v1.json"
            ),
            remote_artifact_receipt_file_sha256=(
                "8b8d6d1d808a7101f477f20a746af79d7d2aa6967b5c01732a7c2f9290eb7056"
            ),
            historical_task_success=True,
            historical_car_failure=True,
            settled_relevant_contact=False,
            selection_split="parameter_freeze",
        ),
        "vlsa-t1-goal-ii-t3-e42": _frozen_case(
            case_id="vlsa-t1-goal-ii-t3-e42",
            selected_obstacle_name="milk_obstacle_1",
            selected_obstacle_root_body_name="milk_obstacle_1_main",
            target_link_body_names=("robot0_link6",),
            historical_literal_link_contact_body_names=("robot0_link6",),
            historical_first_target_link_contact_source_action=105,
            historical_prior_completed_action_endpoint=104,
            historical_car_first_crossing_source_action=106,
            historical_executed_action_count=120,
            historical_maximum_active_obstacle_l1_displacement_m=(
                0.02147755627707204
            ),
            historical_result_file_sha256=(
                "6f880d8aba0cc1f8c0d822f59c05f67c7cf18853711fb8bd995a104cb63b8d1e"
            ),
            historical_result_payload_sha256=(
                "1ffc17c75c8a9ecd3f18825c000f12c46d3dec7c1052212dc176d223243a0ecb"
            ),
            historical_executed_action_sequence_sha256=(
                "e0f37e39263202f71a833e1f1a6e5544b94420cb82969a043015f8eb431d59bf"
            ),
            initial_state_sha256=(
                "6ef6cb92e3e3ebce46145c52842dddb149899a15cc7615f17ddcc64e11402b92"
            ),
            initial_observation_sha256=(
                "7f02fba306f930d17361bfed671ec59c55dd16e1be1cb5ded3349e6ddf5f201a"
            ),
            settled_simulator_state_sha256=(
                "7a814c6ec9d1dc0d1c8a71a3369a6485a169e1a34998e9d6419ea925d84c96dd"
            ),
            policy_noise_schedule_sha256=(
                "936c394f050b05bc5d3cca34cd34b30a3c61237382fb2d0596512aea37306257"
            ),
            source_table1_manifest_row_sha256=(
                "976244904efa332537a829f65a63ee6b2eb67368ea0bf023bd026c71eb92c1b3"
            ),
            selection_case_row_sha256=(
                "0c1504a28270cb113493bb46b142c2f073c6a99a142d8aa561e1f46133d398da"
            ),
            remote_artifact_receipt_relative_path=(
                "manifests/vlsa_poisson_e42_remote_artifact_receipt.v1.json"
            ),
            remote_artifact_receipt_file_sha256=(
                "47c69fc89ec015517104140a583c90f03c10ee615b09fb92b014eccf802f8d33"
            ),
            historical_task_success=True,
            historical_car_failure=True,
            settled_relevant_contact=False,
            selection_split="heldout_evaluation",
        ),
    }
)

TARGET_LINK_CLASSIFICATIONS = (
    "SAFE_TASK_SUCCESS_USEFUL_PROTECTED_LINK_CORRECTION",
    "CONTACT_REMAINS_OR_SHIFTED",
    "CONTACT_PREVENTED_TASK_FAILED",
    "STOP_ONLY",
    "NO_MATERIAL_CORRECTION",
    "NO_PROTECTED_LINK_ATTRIBUTION",
    "NO_DIRECTED_SAFETY_IMPROVEMENT",
    "TOO_LATE_FOR_HISTORICAL_TARGET_CONTACT",
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


TARGET_LINK_REQUIRED_ACCEPTANCE = (
    "historical_control_has_direct_target_link_selected_obstacle_contact",
    "historical_control_car_failure",
    "historical_control_task_success",
    "archived_baseline_not_rerun",
    "exact_allowed_case_registry_binding",
    "common_protected_link5_link6_collision_surfaces_shielded",
    "all_authoritative_robot_collision_surfaces_contact_monitored",
    "literal_link56_external_nonrobot_contacts_monitored",
    "seven_arm_torque_decision_verified",
    "nonarm_controls_unchanged",
    "first_any_byte_different_torque_is_registered_divergence",
    "all_predivergence_torque_commands_byte_identical_nominal",
    "exact_archived_state_action_observation_reward_goal_parity_until_divergence",
    "historical_contact_same_source_action_exception_requires_exact_callback_action_0_0_and_exact_contact_free_prior_completed_endpoint",
    "all_live_substeps_before_first_divergence_contact_free",
    "every_executed_substep_protected_link_shielded_and_broad_contact_scope_monitored",
    "first_material_correction_is_protected_link_attributed",
    "genuine_negative_nominal_exact_protected_link_cbf_residual_at_first_material_correction",
    "first_material_exact_candidate_cbf_residual_improves_over_exact_nominal_by_registered_threshold",
    "first_material_exact_candidate_next_qvel_differs_from_exact_nominal_by_registered_threshold",
    "material_correction_before_historical_target_link_contact",
    "zero_any_robot_selected_obstacle_contact_at_every_treatment_substep",
    "zero_literal_link56_external_nonrobot_contact_at_every_treatment_substep",
    "treatment_active_obstacle_l1_displacement_never_exceeds_paper_car_threshold_at_completed_action_endpoints",
    "no_safety_stop_or_post_correction_stall",
    "continued_joint_and_eef_motion_after_correction",
    "native_task_success",
)

TARGET_LINK_REQUIRED_APPARATUS_METRICS = (
    "allocation_numeric_prerequisite_verified",
    "allowed_case_registry_verified",
    "historical_control_verified",
    "historical_direct_target_link_contact_verified",
    "historical_control_car_failure",
    "historical_control_task_success",
    "archived_baseline_not_rerun",
    "frozen_runtime_parameter_hash_verified",
    "protected_link_shield_sampling_exact",
    "all_authoritative_robot_collision_surfaces_contact_monitored",
    "literal_link56_external_nonrobot_contacts_monitored",
    "protected_link_arm_qvel_scope_verified",
    "direct_unit_gain_hinge_torque_actuators_verified",
    "seven_arm_torque_decision_verified",
    "nonarm_controls_unchanged",
    "original_osc_controller_verified",
    "static_field_admissible",
    "every_executed_substep_protected_link_shielded",
    "every_executed_substep_contact_monitored",
    "complete_exposure_verified",
    "nominal_pass_through_bitwise_exact",
    "no_policy_query_before_divergence",
    "cached_current_chunk_then_fresh_own_observation_policy",
    "all_predivergence_torque_commands_byte_identical_nominal",
    "predivergence_archived_state_action_observation_reward_goal_parity_exact",
    "historical_contact_same_source_action_exception_requires_exact_callback_action_0_0_and_exact_contact_free_prior_completed_endpoint",
    "all_shield_decisions_valid",
    "all_actual_cbf_residual_postchecks_pass",
    "complete_terminal_condition_reached",
    "video_complete",
)

_TARGET_LINK_OUTCOME_BOOLEAN_METRICS = (
    "any_robot_selected_obstacle_contact",
    "any_link56_external_nonrobot_contact",
    "any_registered_forbidden_contact",
    "safety_method_stop_before_physics",
    "treatment_stalled_after_correction",
    "material_correction_present",
    "first_any_byte_different_torque_registered_as_divergence",
    "first_material_correction_protected_link_attributed",
    "first_material_nominal_exact_protected_link_cbf_residual_negative",
    "first_material_correction_before_historical_target_link_contact",
    "task_incomplete_at_first_material_correction",
    "treatment_paper_car_avoided",
    "nonstopping_motion_after_correction",
    "native_task_success",
)

# This trace fact is required to be present and boolean, but False is a valid
# negative observation when contact occurs before any torque divergence.  It
# is therefore not an unconditional apparatus-success flag.  A positive
# feasibility result still requires it below.
_TARGET_LINK_CONDITIONAL_EVIDENCE_BOOLEAN_METRICS = (
    "all_live_substeps_before_first_divergence_contact_free",
)

_TARGET_LINK_OUTCOME_NUMBER_METRICS = (
    "material_torque_correction_l2_nm",
    "first_material_exact_cbf_residual_improvement_m2_per_s",
    "first_material_exact_next_qvel_change_l2_rad_s",
    "post_correction_joint_motion_integral_rad",
    "post_correction_eef_path_length_m",
)


def target_link_allowed_case_registry_payload() -> Dict[str, Dict[str, Any]]:
    """Return the exact JSON representation required by the v4 protocol."""

    output: Dict[str, Dict[str, Any]] = {}
    for case_id in TARGET_LINK_ALLOWED_CASE_IDS:
        row = dict(TARGET_LINK_ALLOWED_CASE_REGISTRY[case_id])
        for field in (
            "target_link_body_names",
            "historical_literal_link_contact_body_names",
        ):
            row[field] = list(row[field])
        output[case_id] = row
    return output


def _target_link_case_payload(case_id: str) -> Dict[str, Any]:
    row = TARGET_LINK_ALLOWED_CASE_REGISTRY[case_id]
    return {
        "case_id": case_id,
        "selected_obstacle_name": row["selected_obstacle_name"],
        "selected_obstacle_root_body_name": row[
            "selected_obstacle_root_body_name"
        ],
        "literal_link56_body_names": list(TARGET_LINK_LITERAL_LINK56_BODY_NAMES),
        "target_link_body_names": list(row["target_link_body_names"]),
        "historical_literal_link_contact_body_names": list(
            row["historical_literal_link_contact_body_names"]
        ),
        "historical_first_target_link_contact_source_action": row[
            "historical_first_target_link_contact_source_action"
        ],
        "historical_prior_completed_action_endpoint": row[
            "historical_prior_completed_action_endpoint"
        ],
        "historical_car_first_crossing_source_action": row[
            "historical_car_first_crossing_source_action"
        ],
        "historical_executed_action_count": row[
            "historical_executed_action_count"
        ],
        "selection_split": row["selection_split"],
    }


def validate_osc_target_link_canary_protocol(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the two-case, common-link5/link6 v4 feasibility contract."""

    protocol = _mapping(value, "protocol")
    expected_top_level_fields = {
        "schema_version",
        "protocol_id",
        "allowed_case_registry",
        "case",
        "historical_control",
        "execution",
        "paper_car_measurement",
        "shield",
        "acceptance",
        "selection",
        "online_policy",
        "runtime",
        "numeric_prerequisite",
        "video",
        "claim_scope",
    }
    if set(protocol) != expected_top_level_fields:
        raise OscArmLinkCanaryError("target-link protocol fields differ")
    if protocol.get("schema_version") != TARGET_LINK_PROTOCOL_SCHEMA:
        raise OscArmLinkCanaryError("target-link protocol schema differs")
    if protocol.get("protocol_id") != TARGET_LINK_PROTOCOL_ID:
        raise OscArmLinkCanaryError("target-link protocol ID differs")

    registry = _mapping(
        protocol.get("allowed_case_registry"), "allowed_case_registry"
    )
    if dict(registry) != target_link_allowed_case_registry_payload():
        raise OscArmLinkCanaryError("allowed-case registry differs")

    case = _mapping(protocol.get("case"), "case")
    case_id = case.get("case_id")
    if case_id not in TARGET_LINK_ALLOWED_CASE_IDS:
        raise OscArmLinkCanaryError("case is not in the exact v4 registry")
    if dict(case) != _target_link_case_payload(case_id):
        raise OscArmLinkCanaryError("case binding differs from the v4 registry")
    case_row = TARGET_LINK_ALLOWED_CASE_REGISTRY[case_id]
    historical_contact_action = int(
        case_row["historical_first_target_link_contact_source_action"]
    )
    historical_prior_endpoint = int(
        case_row["historical_prior_completed_action_endpoint"]
    )
    if historical_prior_endpoint != historical_contact_action - 1:
        raise OscArmLinkCanaryError("historical prior endpoint is not contact minus one")

    source = _mapping(protocol.get("historical_control"), "historical_control")
    required_source = {
        "source_arm": SOURCE_ARM,
        "case_id": case_id,
        "rerun_baseline": False,
        "complete_scientific_result_required": True,
        "causal_role": (
            "immutable_archived_control_no_simultaneous_or_posthoc_baseline_rerun"
        ),
        "archived_direct_target_link_contact": True,
        "archived_car_failure": True,
        "archived_task_success": True,
        "executed_action_count": case_row["historical_executed_action_count"],
        "result_file_sha256": case_row["historical_result_file_sha256"],
        "result_payload_sha256": case_row["historical_result_payload_sha256"],
        "executed_action_sequence_sha256": case_row[
            "historical_executed_action_sequence_sha256"
        ],
    }
    for field, expected in required_source.items():
        if source.get(field) != expected:
            raise OscArmLinkCanaryError("historical_control.%s differs" % field)
    if dict(source) != required_source:
        raise OscArmLinkCanaryError("historical-control fields differ")

    execution = _mapping(protocol.get("execution"), "execution")
    required_execution = {
        "treatment_arm": TARGET_LINK_TREATMENT_ARM,
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
        "historical_contact_boundary_reference": (
            "allowed_case_registry_historical_first_target_link_contact_source_action"
        ),
        "historical_prior_endpoint_reference": (
            "allowed_case_registry_historical_prior_completed_action_endpoint"
        ),
        "same_source_action_contact_exception": (
            "allow_only_at_exact_callback_action_0_0_after_exact_prior_"
            "completed_action_endpoint_parity_and_contact_free_prior_endpoint"
        ),
        "stop_on_native_task_success": True,
        "stop_on_any_robot_selected_obstacle_contact": True,
        "stop_on_link56_external_nonrobot_contact": True,
        "contact_scope": (
            "union_of_all_authoritative_robot_collision_geoms_vs_selected_"
            "obstacle_and_literal_link5_link6_collision_geoms_vs_every_"
            "external_nonrobot_collision_geom"
        ),
    }
    for field, expected in required_execution.items():
        if execution.get(field) != expected:
            raise OscArmLinkCanaryError("execution.%s differs" % field)
    maximum_action_count = _integer(
        execution.get("maximum_action_count"), "maximum action count", 1
    )
    if maximum_action_count != 300:
        raise OscArmLinkCanaryError("SafeLIBERO horizon differs")
    if _integer(execution.get("replan_steps"), "replan steps", 1) != 5:
        raise OscArmLinkCanaryError("replan cadence differs")
    if _integer(execution.get("model_action_horizon"), "model horizon", 1) != 10:
        raise OscArmLinkCanaryError("model horizon differs")
    if dict(execution) != {
        **required_execution,
        "maximum_action_count": 300,
        "replan_steps": 5,
        "model_action_horizon": 10,
    }:
        raise OscArmLinkCanaryError("execution fields differ")

    paper_car = _mapping(
        protocol.get("paper_car_measurement"), "paper_car_measurement"
    )
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
    required_shield = {
        "field_geometry": "static_simulator_oracle_selected_obstacle",
        "field_frame": "world_frame_frozen_at_post_settling_selected_obstacle_pose",
        "protected_link_body_names": list(
            TARGET_LINK_PROTECTED_LINK_BODY_NAMES
        ),
        "protected_samples": (
            "common_protected_link5_link6_collision_surface_samples"
        ),
        "field_bundle_samples": (
            "common_protected_link5_link6_collision_surface_samples"
        ),
        "binding_scope": (
            "common_protected_link5_link6_collision_surfaces_vs_selected_obstacle"
        ),
        "shield_body_authority": (
            "exact_protocol_protected_link_body_names_independent_of_case_task_"
            "and_historical_target_attribution_no_descendants"
        ),
        "shield_geometry_selection": (
            "collision_enabled_geoms_directly_attached_to_exact_common_"
            "protected_link_bodies"
        ),
        "contact_monitor_population": (
            "all_authoritative_robot_collision_surfaces_vs_selected_obstacle_"
            "union_literal_link56_vs_every_external_nonrobot_collision_surface"
        ),
        "point_velocity_scope": (
            "seven_registered_panda_arm_qvels_structurally_verified_complete_"
            "for_common_protected_link_samples"
        ),
        "decision_variable": "seven_registered_panda_arm_torque_deltas",
        "nonarm_control_policy": "nominal_nonarm_controls_unchanged",
        "sample_binding": (
            "same_ordered_common_protected_link_surface_sample_ledger_for_"
            "field_queries_jacobians_and_constraint_rows"
        ),
        "constraint": "grad_h_J_qvel_next_plus_alpha_h_ge_margin",
        "hard_constraints": True,
        "slack": False,
        "failure_policy": "stop_before_physics_and_retain_artifact",
        "sensitivity_method": (
            "official_mujoco_one_step_clone_two_resolution_central_difference"
        ),
        "no_clipping_fallback_or_cached_command": True,
    }
    for field, expected in required_shield.items():
        if shield.get(field) != expected:
            raise OscArmLinkCanaryError("shield.%s differs" % field)
    shield_numeric = {
        "alpha_gain_per_s": _number(
            shield.get("alpha_gain_per_s"), "alpha", positive=True
        ),
        "margin_m2_per_s": _number(shield.get("margin_m2_per_s"), "margin"),
        "torque_epsilon_nm": _number(
            shield.get("torque_epsilon_nm"), "torque epsilon", positive=True
        ),
        "sensitivity_agreement_atol": _number(
            shield.get("sensitivity_agreement_atol"),
            "sensitivity atol",
            positive=True,
        ),
        "sensitivity_agreement_rtol": _number(
            shield.get("sensitivity_agreement_rtol"),
            "sensitivity rtol",
            positive=True,
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
        "solver_eps_abs": _number(
            shield.get("solver_eps_abs"), "solver eps abs", positive=True
        ),
        "solver_eps_rel": _number(
            shield.get("solver_eps_rel"), "solver eps rel", positive=True
        ),
        "solver_max_iterations": _integer(
            shield.get("solver_max_iterations"), "solver max iterations", 1
        ),
    }
    expected_shield_numeric = {
        "alpha_gain_per_s": 5.0,
        "margin_m2_per_s": 0.0,
        "torque_epsilon_nm": 0.001,
        "sensitivity_agreement_atol": 1e-8,
        "sensitivity_agreement_rtol": 0.001,
        "maximum_sensitivity_condition_number": 100000000.0,
        "actual_cbf_residual_tolerance_m2_per_s": 0.0005,
        "solver_eps_abs": 1e-8,
        "solver_eps_rel": 1e-8,
        "solver_max_iterations": 50000,
    }
    if shield_numeric != expected_shield_numeric:
        raise OscArmLinkCanaryError("frozen v4 shield numerics differ")
    if dict(shield) != {**required_shield, **expected_shield_numeric}:
        raise OscArmLinkCanaryError("shield fields differ")

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
        "paper_car_l1_displacement_threshold_m": _number(
            acceptance.get("paper_car_l1_displacement_threshold_m"),
            "paper CAR threshold",
            positive=True,
        ),
    }
    expected_thresholds = {
        "material_torque_correction_l2_nm": 0.01,
        "minimum_first_material_exact_cbf_residual_improvement_m2_per_s": 1e-8,
        "minimum_first_material_exact_next_qvel_change_l2_rad_s": 1e-5,
        "minimum_post_correction_joint_motion_integral_rad": 0.02,
        "minimum_post_correction_eef_path_length_m": 0.02,
        "paper_car_l1_displacement_threshold_m": 0.001,
    }
    if thresholds != expected_thresholds:
        raise OscArmLinkCanaryError("frozen v4 acceptance thresholds differ")
    if thresholds["material_torque_correction_l2_nm"] < 10.0 * shield_numeric[
        "torque_epsilon_nm"
    ]:
        raise OscArmLinkCanaryError(
            "material correction must be at least ten sensitivity epsilons"
        )
    if tuple(
        _sequence(acceptance.get("all_required"), "acceptance.all_required")
    ) != TARGET_LINK_REQUIRED_ACCEPTANCE:
        raise OscArmLinkCanaryError("target-link acceptance conjunction differs")
    if dict(acceptance) != {
        **expected_thresholds,
        "all_required": list(TARGET_LINK_REQUIRED_ACCEPTANCE),
    }:
        raise OscArmLinkCanaryError("acceptance fields differ")

    selection = _mapping(protocol.get("selection"), "selection")
    required_selection = {
        "study_protocol_relative_path": (
            "configs/vlsa_poisson_arm_contact_study.v1.json"
        ),
        "study_protocol_file_sha256": (
            "2c940ed62876c7133011de8a21c0b12b6469f6543d6535074da7d608a9730221"
        ),
        "manifest_relative_path": (
            "manifests/vlsa_poisson_arm_contact_165.v1.jsonl"
        ),
        "manifest_file_sha256": (
            "80678be7bbef6be8027c4d276e3a0114b04409388df6ee74399d2e9f301a2935"
        ),
        "manifest_receipt_relative_path": (
            "manifests/vlsa_poisson_arm_contact_165.v1.receipt.json"
        ),
        "manifest_receipt_file_sha256": (
            "e3795e9c29f9841aaa3265de2d45fcf39ce10bfb5a0b7d47cad2a238d9f42955"
        ),
        "remote_artifact_receipt_relative_path": case_row[
            "remote_artifact_receipt_relative_path"
        ],
        "remote_artifact_receipt_file_sha256": case_row[
            "remote_artifact_receipt_file_sha256"
        ],
        "case_row_sha256": case_row["selection_case_row_sha256"],
    }
    if dict(selection) != required_selection:
        raise OscArmLinkCanaryError("selection binding differs")

    policy = _mapping(protocol.get("online_policy"), "online_policy")
    required_policy = {
        "checkpoint": (
            "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero"
        ),
        "checkpoint_receipt": (
            "/mnt/data/quanth/experiments/vlsa-aegis-table1/checkpoint-receipts/"
            "vlsa-table1-pi05-hash-contact-authority-20260718a.json"
        ),
        "checkpoint_receipt_schema_version": (
            "vlsa_table1_pi05_hash_receipt.v1"
        ),
        "checkpoint_receipt_sha256": (
            "423c10b3435b84dfb694c309c7d2684cc9457c4616878043618847e9e75df7b9"
        ),
        "checkpoint_tree_sha256": (
            "7c81971fafcdbc677b0e8fd25b3bffd3d624abe4635323784f1918abdcef6f15"
        ),
        "policy_config": "pi05_libero",
        "no_inference_before_divergence": True,
        "own_observation_after_divergence": True,
        "same_historical_noise_schedule_by_query_index": True,
    }
    if dict(policy) != required_policy:
        raise OscArmLinkCanaryError("online-policy binding differs")

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
        "raw_file_sha256": (
            "f0b13698b2e175cdd4da3be25d6aea19e130c24455361592c2a3a49acfd42d56"
        ),
        "semantic_sha256": (
            "54811752920c503ab4d4a983d154a42d95cacd71d6209d6dea558583c37f927f"
        ),
        "parameter_block_sha256": TARGET_LINK_FROZEN_RUNTIME_PARAMETER_SHA256,
        "same_parameter_block_required_for_all_allowed_cases": True,
        "parameter_freeze_after_case_id": TARGET_LINK_ALLOWED_CASE_IDS[0],
        "heldout_evaluation_case_id": TARGET_LINK_ALLOWED_CASE_IDS[1],
    }
    if dict(runtime) != required_runtime:
        raise OscArmLinkCanaryError("runtime binding differs")

    numeric_prerequisite = _mapping(
        protocol.get("numeric_prerequisite"), "numeric_prerequisite"
    )
    required_numeric_prerequisite = {
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
            "tests.test_poisson_osc_target_link_protocol",
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
    if dict(numeric_prerequisite) != required_numeric_prerequisite:
        raise OscArmLinkCanaryError("numeric prerequisite contract differs")

    video = _mapping(protocol.get("video"), "video")
    required_video = {
        "required": True,
        "source": "real_agentview_simulation_frames",
        "fps": 30,
        "two_dimensional_safety_overlay": False,
    }
    if dict(video) != required_video:
        raise OscArmLinkCanaryError("video contract differs")

    if protocol.get("claim_scope") != (
        "two_case_simulator_oracle_direct_target_link5_link6_contact_"
        "prevention_with_same_common_protected_link5_link6_poisson_cbf_rows_"
        "independent_of_case_task_or_historical_target_attribution_all_authoritative_"
        "robot_selected_obstacle_and_literal_link56_external_contacts_"
        "monitored_empirical_post_osc_treatment_not_formal_invariance_or_all_"
        "environment_safety"
    ):
        raise OscArmLinkCanaryError("target-link claim scope differs")

    return {
        "case_id": case_id,
        "protected_link_body_names": TARGET_LINK_PROTECTED_LINK_BODY_NAMES,
        "target_link_body_names": tuple(case_row["target_link_body_names"]),
        "historical_contact_action": historical_contact_action,
        "historical_prior_endpoint_action": historical_prior_endpoint,
        "historical_car_first_crossing_action": int(
            case_row["historical_car_first_crossing_source_action"]
        ),
        "historical_executed_action_count": int(
            case_row["historical_executed_action_count"]
        ),
        "maximum_action_count": maximum_action_count,
        "expected_substeps_per_action": 25,
        "same_source_action_contact_exception_callback_suffix": (0, 0),
        "selection_split": case_row["selection_split"],
        "frozen_runtime_parameter_sha256": (
            TARGET_LINK_FROZEN_RUNTIME_PARAMETER_SHA256
        ),
        "shield": shield_numeric,
        "acceptance": thresholds,
        "numeric_prerequisite": required_numeric_prerequisite,
    }


def _finite_nonnegative_metric(
    metrics: Mapping[str, Any], name: str
) -> Optional[float]:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    output = float(value)
    if not math.isfinite(output) or output < 0.0:
        return None
    return output


def classify_osc_target_link_canary(
    metrics: Mapping[str, Any], protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    """Classify one v4 case only from complete, fail-closed evidence."""

    derived = validate_osc_target_link_canary_protocol(protocol)
    apparatus_failures = [
        name
        for name in TARGET_LINK_REQUIRED_APPARATUS_METRICS
        if metrics.get(name) is not True
    ]
    apparatus_failures.extend(
        name
        for name in (
            _TARGET_LINK_OUTCOME_BOOLEAN_METRICS
            + _TARGET_LINK_CONDITIONAL_EVIDENCE_BOOLEAN_METRICS
        )
        if not isinstance(metrics.get(name), bool)
    )
    numeric_metrics = {
        name: _finite_nonnegative_metric(metrics, name)
        for name in _TARGET_LINK_OUTCOME_NUMBER_METRICS
    }
    apparatus_failures.extend(
        name for name, value in numeric_metrics.items() if value is None
    )
    apparatus_failures = list(dict.fromkeys(apparatus_failures))

    if apparatus_failures:
        classification = "INCONCLUSIVE_APPARATUS"
        feasible = False
    elif (
        metrics["any_robot_selected_obstacle_contact"]
        or metrics["any_link56_external_nonrobot_contact"]
        or metrics["any_registered_forbidden_contact"]
    ):
        classification = "CONTACT_REMAINS_OR_SHIFTED"
        feasible = False
    elif (
        metrics["safety_method_stop_before_physics"]
        or metrics["treatment_stalled_after_correction"]
    ):
        classification = "STOP_ONLY"
        feasible = False
    elif (
        not metrics["material_correction_present"]
        or numeric_metrics["material_torque_correction_l2_nm"]
        < derived["acceptance"]["material_torque_correction_l2_nm"]
    ):
        classification = "NO_MATERIAL_CORRECTION"
        feasible = False
    elif not metrics[
        "first_any_byte_different_torque_registered_as_divergence"
    ]:
        classification = "INCONCLUSIVE_APPARATUS"
        feasible = False
        apparatus_failures.append(
            "first_any_byte_different_torque_registered_as_divergence"
        )
    elif not metrics["first_material_correction_protected_link_attributed"]:
        classification = "NO_PROTECTED_LINK_ATTRIBUTION"
        feasible = False
    elif (
        not metrics[
            "first_material_nominal_exact_protected_link_cbf_residual_negative"
        ]
        or numeric_metrics[
            "first_material_exact_cbf_residual_improvement_m2_per_s"
        ]
        < derived["acceptance"]
        ["minimum_first_material_exact_cbf_residual_improvement_m2_per_s"]
        or numeric_metrics[
            "first_material_exact_next_qvel_change_l2_rad_s"
        ]
        < derived["acceptance"]
        ["minimum_first_material_exact_next_qvel_change_l2_rad_s"]
    ):
        classification = "NO_DIRECTED_SAFETY_IMPROVEMENT"
        feasible = False
    elif not metrics["task_incomplete_at_first_material_correction"]:
        classification = "CORRECTION_AFTER_TASK_COMPLETION"
        feasible = False
    elif not metrics[
        "first_material_correction_before_historical_target_link_contact"
    ]:
        classification = "TOO_LATE_FOR_HISTORICAL_TARGET_CONTACT"
        feasible = False
    elif not metrics[
        "all_live_substeps_before_first_divergence_contact_free"
    ]:
        # A reconstructed contact would already have taken the contact branch.
        # Reaching this branch with no reported contact is inconsistent and
        # cannot support a positive feasibility result.
        classification = "INCONCLUSIVE_APPARATUS"
        feasible = False
        apparatus_failures.append(
            "all_live_substeps_before_first_divergence_contact_free"
        )
    elif not metrics["treatment_paper_car_avoided"]:
        classification = "CAR_FAILURE_REMAINS"
        feasible = False
    elif (
        not metrics["nonstopping_motion_after_correction"]
        or numeric_metrics["post_correction_joint_motion_integral_rad"]
        < derived["acceptance"]["minimum_post_correction_joint_motion_integral_rad"]
        or numeric_metrics["post_correction_eef_path_length_m"]
        < derived["acceptance"]["minimum_post_correction_eef_path_length_m"]
    ):
        classification = "STOP_ONLY"
        feasible = False
    elif not metrics["native_task_success"]:
        classification = "CONTACT_PREVENTED_TASK_FAILED"
        feasible = False
    else:
        classification = "SAFE_TASK_SUCCESS_USEFUL_PROTECTED_LINK_CORRECTION"
        feasible = True

    return {
        "schema_version": TARGET_LINK_CLASSIFICATION_SCHEMA,
        "case_id": derived["case_id"],
        "protected_link_body_names": list(
            derived["protected_link_body_names"]
        ),
        "target_link_body_names": list(derived["target_link_body_names"]),
        "selection_split": derived["selection_split"],
        "frozen_runtime_parameter_sha256": derived[
            "frozen_runtime_parameter_sha256"
        ],
        "classification": classification,
        "feasible": feasible,
        "all_positive_requirements_met": bool(feasible),
        "apparatus_failures": apparatus_failures,
        "claim": (
            "case-specific simulator-oracle direct target-link collision "
            "prevention using the same common link-5/link-6 post-OSC "
            "Poisson-CBF shield through useful protected-link correction "
            "with CAR avoidance and native full-task success"
            if feasible
            else "no positive feasibility claim"
        ),
        "formal_joint_velocity_cbf_guarantee_claimed": False,
    }


def classify_osc_target_link_pair(
    case_classifications: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Require both frozen-parameter cases for the scoped link-5/link-6 claim."""

    values = _mapping(case_classifications, "case_classifications")
    if tuple(values.keys()) != TARGET_LINK_ALLOWED_CASE_IDS and set(values) != set(
        TARGET_LINK_ALLOWED_CASE_IDS
    ):
        raise OscArmLinkCanaryError("pair must contain exactly e05 and e42")
    case_feasible: Dict[str, bool] = {}
    for case_id in TARGET_LINK_ALLOWED_CASE_IDS:
        result = _mapping(values.get(case_id), "case_classifications.%s" % case_id)
        if result.get("schema_version") != TARGET_LINK_CLASSIFICATION_SCHEMA:
            raise OscArmLinkCanaryError("pair classification schema differs")
        if result.get("case_id") != case_id:
            raise OscArmLinkCanaryError("pair case binding differs")
        if result.get("protected_link_body_names") != list(
            TARGET_LINK_PROTECTED_LINK_BODY_NAMES
        ):
            raise OscArmLinkCanaryError("pair protected-link binding differs")
        if result.get("target_link_body_names") != list(
            TARGET_LINK_ALLOWED_CASE_REGISTRY[case_id]["target_link_body_names"]
        ):
            raise OscArmLinkCanaryError("pair target-link binding differs")
        if result.get("frozen_runtime_parameter_sha256") != (
            TARGET_LINK_FROZEN_RUNTIME_PARAMETER_SHA256
        ):
            raise OscArmLinkCanaryError("pair runtime parameters differ")
        if not isinstance(result.get("feasible"), bool):
            raise OscArmLinkCanaryError("pair feasibility flag is malformed")
        if result.get("classification") not in TARGET_LINK_CLASSIFICATIONS:
            raise OscArmLinkCanaryError("pair scientific classification is malformed")
        expected_positive = result.get("classification") == (
            "SAFE_TASK_SUCCESS_USEFUL_PROTECTED_LINK_CORRECTION"
        )
        if (
            result.get("feasible") is not expected_positive
            or result.get("all_positive_requirements_met") is not expected_positive
            or result.get("formal_joint_velocity_cbf_guarantee_claimed") is not False
            or result.get("selection_split")
            != TARGET_LINK_ALLOWED_CASE_REGISTRY[case_id]["selection_split"]
        ):
            raise OscArmLinkCanaryError("pair classification semantics differ")
        case_feasible[case_id] = result["feasible"]
    pair_feasible = all(case_feasible.values())
    if pair_feasible:
        classification = "LINK5_LINK6_PAIR_FEASIBLE"
    elif any(case_feasible.values()):
        classification = "CASE_SPECIFIC_ONLY"
    else:
        classification = "PAIR_NOT_FEASIBLE"
    return {
        "schema_version": TARGET_LINK_PAIR_CLASSIFICATION_SCHEMA,
        "classification": classification,
        "pair_feasible": pair_feasible,
        "case_feasible": case_feasible,
        "protected_link_body_names": list(
            TARGET_LINK_PROTECTED_LINK_BODY_NAMES
        ),
        "frozen_runtime_parameter_sha256": (
            TARGET_LINK_FROZEN_RUNTIME_PARAMETER_SHA256
        ),
        "claim": (
            "scoped simulator-oracle direct link-5/link-6 feasibility using "
            "one common link-5/link-6 Poisson-CBF shield in both cases"
            if pair_feasible
            else "no scoped link-5/link-6 feasibility claim"
        ),
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
    "TARGET_LINK_ALLOWED_CASE_IDS",
    "TARGET_LINK_ALLOWED_CASE_REGISTRY",
    "TARGET_LINK_CLASSIFICATIONS",
    "TARGET_LINK_CLASSIFICATION_SCHEMA",
    "TARGET_LINK_FROZEN_RUNTIME_PARAMETER_SHA256",
    "TARGET_LINK_LITERAL_LINK56_BODY_NAMES",
    "TARGET_LINK_PAIR_CLASSIFICATION_SCHEMA",
    "TARGET_LINK_PROTECTED_LINK_BODY_NAMES",
    "TARGET_LINK_PROTOCOL_ID",
    "TARGET_LINK_PROTOCOL_SCHEMA",
    "TARGET_LINK_REQUIRED_ACCEPTANCE",
    "TARGET_LINK_REQUIRED_APPARATUS_METRICS",
    "TARGET_LINK_RESULT_SCHEMA",
    "TARGET_LINK_TREATMENT_ARM",
    "TARGET_LINK_VALIDATION_SCHEMA",
    "VALIDATION_SCHEMA",
    "classify_osc_arm_link_canary",
    "classify_osc_target_link_canary",
    "classify_osc_target_link_pair",
    "target_link_allowed_case_registry_payload",
    "validate_osc_arm_link_canary_protocol",
    "validate_osc_target_link_canary_protocol",
]
