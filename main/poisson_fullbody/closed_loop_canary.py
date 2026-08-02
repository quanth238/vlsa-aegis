"""Frozen contract and outcome logic for the one-case closed-loop canary.

The canary shares the already validated AEGIS / OSC prefix through source
action 179.  From source action 180 onward, both matched joint-velocity arms
query pi0.5 live.  Once their states differ, each arm constructs every later
policy request from its own camera, wrist camera, and robot state.  The only
low-level arm difference is the link-5/link-6 Poisson-CBF filter.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence


PROTOCOL_SCHEMA = "vlsa_poisson_closed_loop_canary_protocol.v1"
RESULT_SCHEMA = "vlsa_poisson_closed_loop_canary_result.v1"
CLASSIFICATION_SCHEMA = "vlsa_poisson_closed_loop_canary_classification.v1"
PROTOCOL_ID = "vlsa-poisson-link56-closed-loop-suffix-v1"
CASE_ID = "vlsa-t1-goal-ii-t0-e05"
SOURCE_ARM = "pi05_plus_aegis_translational"

BASELINE_ARM = "pi05_plus_aegis_joint_velocity_adapter"
PSF_ARM = "pi05_plus_aegis_joint_velocity_adapter_plus_link56_psf"

CLASSIFICATIONS = (
    "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
    "CONTACT_PREVENTED_TASK_FAILED",
    "STOP_ONLY",
    "CONTACT_REMAINS_OR_SHIFTED",
    "BASELINE_CONTACT_NOT_REPRODUCED",
    "INCONCLUSIVE_APPARATUS",
)


class ClosedLoopCanaryError(RuntimeError):
    """Raised when the frozen canary contract is malformed."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClosedLoopCanaryError("%s must be an object" % label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ClosedLoopCanaryError("%s must be an array" % label)
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ClosedLoopCanaryError(
            "%s must be an integer >= %d" % (label, minimum)
        )
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClosedLoopCanaryError("%s must be numeric" % label)
    output = float(value)
    if not math.isfinite(output):
        raise ClosedLoopCanaryError("%s must be finite" % label)
    return output


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ClosedLoopCanaryError("%s must be a lowercase SHA-256" % label)
    return value


def validate_closed_loop_canary_protocol(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the preregistered experiment and return derived constants."""

    protocol = _mapping(value, "protocol")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ClosedLoopCanaryError("closed-loop protocol schema differs")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ClosedLoopCanaryError("closed-loop protocol ID differs")

    case = _mapping(protocol.get("case"), "protocol.case")
    if case.get("case_id") != CASE_ID:
        raise ClosedLoopCanaryError("closed-loop case differs")
    if case.get("selected_obstacle_name") != "moka_pot_obstacle_1":
        raise ClosedLoopCanaryError("selected obstacle differs")
    if list(
        _sequence(case.get("protected_robot_body_names"), "protected bodies")
    ) != ["robot0_link5", "robot0_link6"]:
        raise ClosedLoopCanaryError("protected link set differs")

    source = _mapping(protocol.get("source"), "protocol.source")
    if source.get("source_arm") != SOURCE_ARM:
        raise ClosedLoopCanaryError("source prefix arm differs")
    for field in (
        "historical_result_file_sha256",
        "historical_result_payload_sha256",
        "historical_executed_action_sequence_sha256",
        "post_action_179_flattened_state_sha256",
        "first_live_query_expected_action_chunk_sha256",
    ):
        _sha256(source.get(field), "protocol.source.%s" % field)
    if _integer(source.get("historical_executed_action_count"), "source count", 1) != 237:
        raise ClosedLoopCanaryError("historical source action count differs")

    episode = _mapping(protocol.get("episode"), "protocol.episode")
    start = _integer(episode.get("suffix_start_action_index"), "suffix start")
    end = _integer(episode.get("suffix_end_action_index_inclusive"), "suffix end")
    action_count = _integer(episode.get("suffix_action_count"), "suffix count", 1)
    if (
        start != 180
        or end != 236
        or action_count != 57
        or end - start + 1 != action_count
        or episode.get("branch_physical_boundary") != 4500
        or episode.get("terminal_physical_boundary") != 5925
        or list(
            _sequence(
                episode.get("expected_boundary_goal_values"),
                "expected boundary goal values",
            )
        )
        != [False]
        or episode.get("continue_after_transient_task_success") is not True
        or episode.get("task_success_is_latched_like_the_benchmark") is not True
        or episode.get("terminal_native_task_success_is_diagnostic") is not True
    ):
        raise ClosedLoopCanaryError("closed-loop suffix boundary differs")

    online = _mapping(protocol.get("online_policy"), "protocol.online_policy")
    _sha256(
        online.get("checkpoint_tree_sha256"),
        "protocol.online_policy.checkpoint_tree_sha256",
    )
    _sha256(
        online.get("checkpoint_receipt_sha256"),
        "protocol.online_policy.checkpoint_receipt_sha256",
    )
    replan = _integer(online.get("replan_steps"), "replan steps", 1)
    first_query = _integer(online.get("first_query_index"), "first query")
    query_count = _integer(online.get("queries_per_arm"), "query count", 1)
    if (
        replan != 5
        or online.get("model_action_horizon") != 10
        or first_query != 36
        or query_count != 12
        or query_count != (action_count + replan - 1) // replan
        or online.get("same_query_seed_by_replan_index") is not True
        or online.get("own_observation_after_divergence") is not True
        or online.get("recorded_suffix_actions_prohibited") is not True
        or online.get("checkpoint")
        != "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero"
        or online.get("checkpoint_receipt")
        != "/mnt/data/quanth/experiments/vlsa-aegis-table1/checkpoint-receipts/vlsa-table1-pi05-hash-contact-authority-20260718a.json"
        or online.get("checkpoint_receipt_schema_version")
        != "vlsa_table1_pi05_hash_receipt.v1"
    ):
        raise ClosedLoopCanaryError("online policy contract differs")
    if list(_sequence(online.get("policy_observation_fields"), "policy fields")) != [
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    ]:
        raise ClosedLoopCanaryError("policy observation field set differs")

    execution = _mapping(protocol.get("execution"), "protocol.execution")
    if list(_sequence(execution.get("arms"), "execution arms")) != [
        BASELINE_ARM,
        PSF_ARM,
    ]:
        raise ClosedLoopCanaryError("closed-loop arm order differs")
    if execution.get("shared_low_level_path") != (
        "same_fresh_100hz_translational_joint_velocity_adapter"
    ):
        raise ClosedLoopCanaryError("shared execution adapter differs")
    if execution.get("only_low_level_difference") != (
        "link56_static_poisson_cbf_constraints_in_treatment"
    ):
        raise ClosedLoopCanaryError("causal execution difference differs")
    if execution.get("aegis_state") != (
        "independent_copy_per_arm_initialized_from_historical_action_179_z_after"
    ):
        raise ClosedLoopCanaryError("AEGIS state initialization differs")
    if (
        execution.get("released_eef_marker_update_after_each_completed_action")
        is not True
    ):
        raise ClosedLoopCanaryError("released AEGIS marker-update cadence differs")

    cadence = _mapping(protocol.get("cadence"), "protocol.cadence")
    updates = _integer(cadence.get("expected_filter_updates_per_arm"), "updates", 1)
    substeps = _integer(cadence.get("expected_physics_substeps_per_arm"), "substeps", 1)
    if (
        cadence.get("high_level_frequency_hz") != 20
        or cadence.get("filter_frequency_hz") != 100
        or cadence.get("physics_frequency_hz") != 500
        or cadence.get("inner_updates_per_high_level_action") != 5
        or cadence.get("physics_substeps_per_inner_update") != 5
        or updates != action_count * 5
        or substeps != updates * 5
        or cadence.get("literal_contact_observation_stride_physics_substeps")
        != 1
        or cadence.get(
            "full_robot_clearance_diagnostic_stride_physics_substeps"
        )
        != 25
    ):
        raise ClosedLoopCanaryError("closed-loop cadence differs")

    videos = _mapping(protocol.get("videos"), "protocol.videos")
    if (
        videos.get("required_per_arm") is not True
        or videos.get("source") != "real_agentview_simulation_frames"
        or videos.get("fps") != 30
        or videos.get("coverage_scope")
        != "branch_boundary_through_live_suffix_terminal"
        or videos.get("playback_not_wall_clock") is not True
        or videos.get("minimum_complete_positive_frames") != action_count + 1
        or videos.get("two_dimensional_safety_overlay") is not False
    ):
        raise ClosedLoopCanaryError("video contract differs")

    acceptance = _mapping(protocol.get("acceptance"), "protocol.acceptance")
    thresholds = {
        "minimum_filter_correction_norm_rad_s": _number(
            acceptance.get("minimum_filter_correction_norm_rad_s"),
            "correction norm threshold",
        ),
        "minimum_filter_correction_integral_rad": _number(
            acceptance.get("minimum_filter_correction_integral_rad"),
            "correction integral threshold",
        ),
        "minimum_post_correction_measured_joint_motion_integral_rad": _number(
            acceptance.get("minimum_post_correction_measured_joint_motion_integral_rad"),
            "measured motion threshold",
        ),
        "minimum_post_correction_cartesian_path_length_m": _number(
            acceptance.get("minimum_post_correction_cartesian_path_length_m"),
            "Cartesian path threshold",
        ),
        "minimum_post_correction_executed_command_integral_rad": _number(
            acceptance.get("minimum_post_correction_executed_command_integral_rad"),
            "command motion threshold",
        ),
        "maximum_post_correction_zero_command_fraction": _number(
            acceptance.get("maximum_post_correction_zero_command_fraction"),
            "zero-command threshold",
        ),
        "maximum_nominal_cbf_residual_for_activation_m2_per_s": _number(
            acceptance.get("maximum_nominal_cbf_residual_for_activation_m2_per_s"),
            "activation threshold",
        ),
    }
    if (
        thresholds["minimum_filter_correction_norm_rad_s"] <= 0.0
        or thresholds["minimum_filter_correction_integral_rad"] <= 0.0
        or thresholds["minimum_post_correction_measured_joint_motion_integral_rad"] <= 0.0
        or thresholds["minimum_post_correction_cartesian_path_length_m"] <= 0.0
        or thresholds["minimum_post_correction_executed_command_integral_rad"] <= 0.0
        or not 0.0 <= thresholds["maximum_post_correction_zero_command_fraction"] < 1.0
        or acceptance.get("require_literal_contact_check_at_every_physics_substep")
        is not True
        or acceptance.get("full_robot_clearance_is_diagnostic_not_acceptance_gate")
        is not True
        or acceptance.get("require_baseline_link56_contact") is not True
        or acceptance.get("require_no_treatment_contact_from_any_robot_geometry")
        is not True
        or acceptance.get("require_material_correction_before_baseline_contact")
        is not True
        or acceptance.get("require_both_native_task_success_ever") is not True
        or acceptance.get("require_live_own_observation_feedback_after_divergence")
        is not True
        or acceptance.get("require_two_complete_decodable_videos") is not True
    ):
        raise ClosedLoopCanaryError("useful-correction thresholds differ")

    result_contract = _mapping(
        protocol.get("result_contract"), "protocol.result_contract"
    )
    if (
        result_contract.get("schema_version") != RESULT_SCHEMA
        or result_contract.get("write_policy")
        != "atomic_final_json_only_never_overwrite"
        or result_contract.get("partial_output_policy") != "never_interpret"
        or result_contract.get("independent_consumer_required") is not True
        or result_contract.get("claim_scope")
        != "one_case_static_simulator_oracle_hybrid_prefix_live_closed_loop_suffix_feasibility_not_population_safety_not_realtime_or_tracking_certified"
    ):
        raise ClosedLoopCanaryError("closed-loop result contract differs")

    return {
        "start_action": start,
        "end_action": end,
        "action_count": action_count,
        "first_query_index": first_query,
        "queries_per_arm": query_count,
        "replan_steps": replan,
        "expected_updates": updates,
        "expected_substeps": substeps,
        "clearance_diagnostic_stride": int(
            cadence["full_robot_clearance_diagnostic_stride_physics_substeps"]
        ),
        "thresholds": thresholds,
    }


def classify_closed_loop_canary(
    metrics: Mapping[str, Any], protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    """Classify a complete pair without confusing stopping with safety."""

    derived = validate_closed_loop_canary_protocol(protocol)
    apparatus_complete = all(
        metrics.get(field) is True
        for field in (
            "exact_paired_start",
            "shared_prefix_complete",
            "live_policy_contract_valid",
            "aegis_contract_valid",
            "both_nominal_commands_within_dynamic_joint_bounds",
            "videos_complete_and_decodable",
            "psf_qp_contract_valid",
            "static_selected_obstacle_admissible",
            "literal_contact_checked_at_every_physics_substep",
            "released_eef_marker_update_contract_valid",
        )
    )
    feedback_valid = all(
        metrics.get(field) is True
        for field in (
            "first_live_query_identical",
            "own_observation_chain_valid",
            "no_recorded_suffix_action_replay",
        )
    )
    post_divergence_feedback = bool(
        metrics.get("post_divergence_own_observations_used") is True
    )
    baseline_reproduced = bool(
        metrics.get("baseline_link56_contact_present") is True
        and metrics.get("baseline_first_selected_obstacle_contact_is_link56") is True
    )
    contact_prevented = bool(
        metrics.get("psf_any_robot_selected_obstacle_contact_present") is False
    )
    corrected = bool(
        metrics.get("material_correction_before_baseline_contact") is True
        and float(metrics.get("maximum_correction_norm_rad_s", 0.0))
        >= derived["thresholds"]["minimum_filter_correction_norm_rad_s"]
        and float(metrics.get("filter_correction_integral_rad", 0.0))
        >= derived["thresholds"]["minimum_filter_correction_integral_rad"]
    )
    useful_motion = bool(
        corrected
        and float(metrics.get("post_correction_measured_joint_motion_integral_rad", 0.0))
        >= derived["thresholds"][
            "minimum_post_correction_measured_joint_motion_integral_rad"
        ]
        and float(metrics.get("post_correction_cartesian_path_length_m", 0.0))
        >= derived["thresholds"]["minimum_post_correction_cartesian_path_length_m"]
        and float(metrics.get("post_correction_executed_command_integral_rad", 0.0))
        >= derived["thresholds"][
            "minimum_post_correction_executed_command_integral_rad"
        ]
        and float(metrics.get("post_correction_zero_command_fraction", 1.0))
        <= derived["thresholds"]["maximum_post_correction_zero_command_fraction"]
    )
    task_success = bool(
        metrics.get("baseline_task_success_ever") is True
        and metrics.get("psf_task_success_ever") is True
        and metrics.get("psf_task_success_after_material_correction") is True
    )

    psf_contact = metrics.get(
        "psf_any_robot_selected_obstacle_contact_present"
    ) is True
    positive_exposure_complete = bool(
        metrics.get("baseline_exposure_complete") is True
        and metrics.get("psf_exposure_complete") is True
    )
    pair_complete = bool(
        apparatus_complete
        and metrics.get("baseline_exposure_complete") is True
        and (metrics.get("psf_exposure_complete") is True or psf_contact)
    )

    if not pair_complete or not feedback_valid:
        label = "INCONCLUSIVE_APPARATUS"
    elif not baseline_reproduced:
        label = "BASELINE_CONTACT_NOT_REPRODUCED"
    elif psf_contact:
        label = "CONTACT_REMAINS_OR_SHIFTED"
    elif not post_divergence_feedback:
        label = "INCONCLUSIVE_APPARATUS"
    elif metrics.get("fresh_policy_query_after_material_correction") is not True:
        label = "INCONCLUSIVE_APPARATUS"
    elif not positive_exposure_complete or not contact_prevented:
        label = "INCONCLUSIVE_APPARATUS"
    elif not useful_motion:
        label = "STOP_ONLY"
    elif not task_success:
        label = "CONTACT_PREVENTED_TASK_FAILED"
    else:
        label = "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"

    return {
        "schema_version": CLASSIFICATION_SCHEMA,
        "classification": label,
        "feasible": label == "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        "pair_complete": pair_complete,
        "closed_loop_feedback_valid": feedback_valid,
        "post_divergence_feedback_observed": post_divergence_feedback,
        "baseline_reproduced": baseline_reproduced,
        "contact_prevented": contact_prevented,
        "useful_correction": useful_motion,
        "stop_only": bool(contact_prevented and not useful_motion),
        "task_successful": task_success,
    }


__all__ = [
    "BASELINE_ARM",
    "CASE_ID",
    "CLASSIFICATIONS",
    "CLASSIFICATION_SCHEMA",
    "ClosedLoopCanaryError",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "PSF_ARM",
    "RESULT_SCHEMA",
    "SOURCE_ARM",
    "classify_closed_loop_canary",
    "validate_closed_loop_canary_protocol",
]
