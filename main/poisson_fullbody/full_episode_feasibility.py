"""Frozen protocol and outcome logic for the full-episode Poisson test.

The experiment is intentionally small: one exact historical AEGIS prefix is
shared, then two fresh joint-velocity arms execute the complete remaining
recorded trajectory from the same pre-contact MuJoCo state.  A positive label
requires literal contact prevention, useful post-correction motion, and native
SafeLIBERO task success.  It is an offline single-case controller-feasibility
test, not a closed-loop policy or population-safety claim.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence


PROTOCOL_SCHEMA = "vlsa_poisson_full_episode_feasibility_protocol.v1"
METRICS_SCHEMA = "vlsa_poisson_full_episode_feasibility_metrics.v1"
CLASSIFICATION_SCHEMA = "vlsa_poisson_full_episode_feasibility_classification.v1"
RESULT_SCHEMA = "vlsa_poisson_full_episode_feasibility_result.v1"
PROTOCOL_ID = "vlsa-poisson-link56-full-episode-hybrid-v1"
CASE_ID = "vlsa-t1-goal-ii-t0-e05"
SOURCE_ARM = "pi05_plus_aegis_translational"

CLASSIFICATIONS = (
    "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
    "CONTACT_PREVENTED_TASK_FAILED",
    "STOP_ONLY",
    "CONTACT_REMAINS_OR_SHIFTED",
    "INCONCLUSIVE_APPARATUS",
)

METRIC_KEYS = (
    "schema_version",
    "exact_paired_start",
    "shared_prefix_complete",
    "full_recorded_episode_complete",
    "adapter_exposure_complete",
    "psf_exposure_complete",
    "adapter_physics_monitor_trace_counts_match",
    "psf_physics_monitor_trace_counts_match",
    "adapter_filter_update_count",
    "psf_filter_update_count",
    "adapter_physics_substep_count",
    "psf_physics_substep_count",
    "adapter_completed_suffix_action_count",
    "psf_completed_suffix_action_count",
    "psf_qp_count_complete",
    "psf_qp_postchecks_complete",
    "psf_joint_limit_postchecks_complete",
    "all_issued_commands_within_physical_bounds",
    "both_nominal_commands_within_dynamic_joint_bounds",
    "psf_invalid_field_query_count",
    "psf_all_post_state_field_queries_valid_and_positive",
    "static_selected_obstacle_admissible",
    "boundary_goal_unsatisfied",
    "adapter_link56_contact_present",
    "adapter_first_selected_obstacle_contact_is_link56",
    "psf_link56_contact_present",
    "psf_any_robot_selected_obstacle_contact_present",
    "psf_clearance_certified",
    "material_correction_before_adapter_contact",
    "first_material_correction_physical_boundary",
    "adapter_first_link56_contact_physical_boundary",
    "material_correction_update_count",
    "maximum_correction_norm_rad_s",
    "filter_correction_integral_rad",
    "post_correction_measured_joint_motion_integral_rad",
    "post_correction_cartesian_path_length_m",
    "post_correction_executed_command_integral_rad",
    "post_correction_zero_command_fraction",
    "psf_task_success_ever",
    "psf_terminal_task_success",
    "psf_first_task_success_source_action_index",
    "psf_task_success_after_material_correction",
    "adapter_task_success_ever",
    "adapter_terminal_task_success",
)


class FullEpisodeFeasibilityError(RuntimeError):
    """Raised when frozen protocol or compact evidence is malformed."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FullEpisodeFeasibilityError("%s must be an object" % label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FullEpisodeFeasibilityError("%s must be an array" % label)
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise FullEpisodeFeasibilityError("%s must be Boolean" % label)
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FullEpisodeFeasibilityError(
            "%s must be an integer >= %d" % (label, minimum)
        )
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FullEpisodeFeasibilityError("%s must be numeric" % label)
    output = float(value)
    if not math.isfinite(output):
        raise FullEpisodeFeasibilityError("%s must be finite" % label)
    return output


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FullEpisodeFeasibilityError("%s must be a lowercase SHA-256" % label)
    return value


def validate_full_episode_feasibility_protocol(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the frozen hybrid episode and return runner constants."""

    protocol = _mapping(value, "protocol")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise FullEpisodeFeasibilityError("full-episode protocol schema differs")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise FullEpisodeFeasibilityError("full-episode protocol ID differs")

    case = _mapping(protocol.get("case"), "protocol.case")
    if case.get("case_id") != CASE_ID:
        raise FullEpisodeFeasibilityError("full-episode case differs")
    if case.get("selected_obstacle_name") != "moka_pot_obstacle_1":
        raise FullEpisodeFeasibilityError("selected obstacle differs")
    if list(
        _sequence(case.get("protected_robot_body_names"), "protected bodies")
    ) != ["robot0_link5", "robot0_link6"]:
        raise FullEpisodeFeasibilityError("protected link set differs")

    source = _mapping(protocol.get("source"), "protocol.source")
    if source.get("source_arm") != SOURCE_ARM:
        raise FullEpisodeFeasibilityError("source arm differs")
    if source.get("action_field") != "actions[*].executed":
        raise FullEpisodeFeasibilityError("executed actions are required")
    if source.get("nominal_translational_field_prohibited") is not True:
        raise FullEpisodeFeasibilityError("nominal action substitution is prohibited")
    for field in (
        "historical_result_file_sha256",
        "historical_result_payload_sha256",
        "historical_executed_action_sequence_sha256",
        "suffix_action_record_sha256",
        "suffix_action_array_sha256",
        "post_action_179_flattened_state_sha256",
    ):
        _sha256(source.get(field), "protocol.source.%s" % field)
    if _integer(
        source.get("historical_executed_action_count"),
        "historical action count",
        1,
    ) != 237:
        raise FullEpisodeFeasibilityError("historical action count differs")

    episode = _mapping(protocol.get("episode"), "protocol.episode")
    start_boundary = _integer(
        episode.get("branch_physical_boundary"), "branch boundary"
    )
    end_boundary = _integer(
        episode.get("terminal_physical_boundary"), "terminal boundary", 1
    )
    start_action = _integer(
        episode.get("suffix_start_action_index"), "suffix start action"
    )
    end_action = _integer(
        episode.get("suffix_end_action_index_inclusive"), "suffix end action"
    )
    action_count = _integer(
        episode.get("suffix_action_count"), "suffix action count", 1
    )
    prefix_count = _integer(
        episode.get("shared_prefix_action_count"), "shared prefix count"
    )
    if (
        prefix_count != 180
        or start_boundary != 4500
        or end_boundary != 5925
        or episode.get("no_event_physical_boundary_sentinel") != 5926
        or start_action != 180
        or end_action != 236
        or action_count != 57
        or start_boundary != start_action * 25
        or end_boundary != (end_action + 1) * 25
        or action_count != end_action - start_action + 1
        or prefix_count + action_count != 237
    ):
        raise FullEpisodeFeasibilityError("full recorded episode boundaries differ")
    if episode.get("execution_kind") != "shared_prefix_branched_complete_suffix":
        raise FullEpisodeFeasibilityError("hybrid execution disclosure differs")
    if list(
        _sequence(
            episode.get("expected_boundary_goal_values"),
            "expected branch-boundary goal values",
        )
    ) != [False]:
        raise FullEpisodeFeasibilityError("branch-boundary goal authority differs")
    if episode.get("continue_fixed_suffix_after_task_success") is not True:
        raise FullEpisodeFeasibilityError("fixed suffix must continue after success")

    pairing = _mapping(protocol.get("pairing"), "protocol.pairing")
    if pairing.get("settle_actions") != 20 or pairing.get("online_policy_queries") != 0:
        raise FullEpisodeFeasibilityError("pairing/reset contract differs")
    if pairing.get("state") != "one_exact_full_mjstate_integration_snapshot_restored_into_both_suffix_arms":
        raise FullEpisodeFeasibilityError("paired-state authority differs")
    if pairing.get("controller_state") != "fresh_identically_reset_joint_velocity_pid_in_both_suffix_arms":
        raise FullEpisodeFeasibilityError("controller-state pairing differs")
    if list(_sequence(protocol.get("arms"), "protocol.arms")) != [
        "joint_velocity_adapter_only",
        "joint_velocity_adapter_plus_link56_psf",
    ]:
        raise FullEpisodeFeasibilityError("paired arm order differs")

    cadence = _mapping(protocol.get("cadence"), "protocol.cadence")
    inner = _integer(
        cadence.get("inner_updates_per_high_level_action"), "inner cadence", 1
    )
    physics = _integer(
        cadence.get("physics_substeps_per_inner_update"), "physics cadence", 1
    )
    expected_updates = _integer(
        cadence.get("expected_filter_updates_per_arm"), "expected updates", 1
    )
    expected_substeps = _integer(
        cadence.get("expected_physics_substeps_per_arm"), "expected substeps", 1
    )
    if (
        cadence.get("high_level_frequency_hz") != 20
        or cadence.get("filter_frequency_hz") != 100
        or cadence.get("physics_frequency_hz") != 500
        or inner != 5
        or physics != 5
        or expected_updates != action_count * inner
        or expected_substeps != expected_updates * physics
        or not math.isclose(
            _number(cadence.get("physics_timestep_s"), "physics timestep"),
            0.002,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise FullEpisodeFeasibilityError("full-episode cadence differs")

    exploratory = _mapping(
        protocol.get("exploratory_execution"), "protocol.exploratory_execution"
    )
    qp_max_iterations = _integer(
        exploratory.get("qp_max_iterations"), "QP iteration budget", 1
    )
    if qp_max_iterations != 50000:
        raise FullEpisodeFeasibilityError("QP iteration budget differs")
    if exploratory.get("tracking_policy") != "diagnostic_only_not_feasibility_gate":
        raise FullEpisodeFeasibilityError("tracking claim differs")
    if exploratory.get("solver_claim") != "offline_controller_feasibility_not_realtime_100hz":
        raise FullEpisodeFeasibilityError("timing claim differs")

    acceptance = _mapping(protocol.get("acceptance"), "protocol.acceptance")
    thresholds = {
        "minimum_safe_cbf_residual_m2_per_s": _number(
            acceptance.get("minimum_safe_cbf_residual_m2_per_s"),
            "safe CBF residual threshold",
        ),
        "maximum_nominal_cbf_residual_for_activation_m2_per_s": _number(
            acceptance.get(
                "maximum_nominal_cbf_residual_for_activation_m2_per_s"
            ),
            "nominal activation threshold",
        ),
        "maximum_invalid_field_queries": _integer(
            acceptance.get("maximum_invalid_field_queries"),
            "invalid field-query threshold",
        ),
        "minimum_filter_correction_norm_rad_s": _number(
            acceptance.get("minimum_filter_correction_norm_rad_s"),
            "correction-norm threshold",
        ),
        "minimum_filter_correction_integral_rad": _number(
            acceptance.get("minimum_filter_correction_integral_rad"),
            "correction-integral threshold",
        ),
        "minimum_post_correction_measured_joint_motion_integral_rad": _number(
            acceptance.get(
                "minimum_post_correction_measured_joint_motion_integral_rad"
            ),
            "joint-motion threshold",
        ),
        "minimum_post_correction_cartesian_path_length_m": _number(
            acceptance.get("minimum_post_correction_cartesian_path_length_m"),
            "Cartesian-motion threshold",
        ),
        "minimum_post_correction_executed_command_integral_rad": _number(
            acceptance.get(
                "minimum_post_correction_executed_command_integral_rad"
            ),
            "executed-command threshold",
        ),
        "maximum_post_correction_zero_command_fraction": _number(
            acceptance.get("maximum_post_correction_zero_command_fraction"),
            "zero-command threshold",
        ),
    }
    if not 0.0 <= thresholds["maximum_post_correction_zero_command_fraction"] < 1.0:
        raise FullEpisodeFeasibilityError("zero-command threshold must be in [0,1)")
    for field in (
        "require_exact_paired_start",
        "require_adapter_link56_contact",
        "require_adapter_first_selected_obstacle_contact_is_link56",
        "require_psf_all_robot_selected_obstacle_contact_absent",
        "require_psf_conservative_full_robot_clearance_strictly_positive",
        "require_material_correction_before_adapter_contact",
        "require_post_correction_motion",
        "require_native_task_success_ever_and_terminal",
        "require_task_success_after_material_correction",
    ):
        if _boolean(acceptance.get(field), "acceptance.%s" % field) is not True:
            raise FullEpisodeFeasibilityError("required acceptance gate is disabled")

    classification = _mapping(
        protocol.get("classification"), "protocol.classification"
    )
    if list(_sequence(classification.get("labels"), "classification labels")) != list(
        CLASSIFICATIONS
    ):
        raise FullEpisodeFeasibilityError("classification labels differ")
    result_contract = _mapping(protocol.get("result_contract"), "result contract")
    if (
        result_contract.get("schema_version") != RESULT_SCHEMA
        or result_contract.get("metrics_schema_version") != METRICS_SCHEMA
        or result_contract.get("classification_schema_version")
        != CLASSIFICATION_SCHEMA
        or result_contract.get("partial_output_policy") != "never_interpret"
        or result_contract.get("independent_consumer_required") is not True
    ):
        raise FullEpisodeFeasibilityError("full-episode result contract differs")

    for section_name in ("runtime", "selection"):
        section = _mapping(protocol.get(section_name), "protocol.%s" % section_name)
        for field in (
            key
            for key in section
            if key.endswith("sha256")
        ):
            _sha256(section.get(field), "protocol.%s.%s" % (section_name, field))

    return {
        "start_boundary": start_boundary,
        "end_boundary": end_boundary,
        "start_action": start_action,
        "end_action": end_action,
        "action_count": action_count,
        "expected_updates": expected_updates,
        "expected_substeps": expected_substeps,
        "qp_max_iterations": qp_max_iterations,
        "thresholds": thresholds,
    }


def classify_full_episode_feasibility(
    metrics_value: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    """Classify a complete pair without treating stopping as success."""

    derived = validate_full_episode_feasibility_protocol(protocol)
    metrics = _mapping(metrics_value, "metrics")
    if set(metrics) != set(METRIC_KEYS):
        missing = sorted(set(METRIC_KEYS) - set(metrics))
        unknown = sorted(set(metrics) - set(METRIC_KEYS))
        raise FullEpisodeFeasibilityError(
            "compact metric keys differ; missing=%s unknown=%s" % (missing, unknown)
        )
    if metrics.get("schema_version") != METRICS_SCHEMA:
        raise FullEpisodeFeasibilityError("compact metrics schema differs")

    boolean_fields = (
        "exact_paired_start",
        "shared_prefix_complete",
        "full_recorded_episode_complete",
        "adapter_exposure_complete",
        "psf_exposure_complete",
        "adapter_physics_monitor_trace_counts_match",
        "psf_physics_monitor_trace_counts_match",
        "psf_qp_count_complete",
        "psf_qp_postchecks_complete",
        "psf_joint_limit_postchecks_complete",
        "all_issued_commands_within_physical_bounds",
        "both_nominal_commands_within_dynamic_joint_bounds",
        "psf_all_post_state_field_queries_valid_and_positive",
        "static_selected_obstacle_admissible",
        "boundary_goal_unsatisfied",
        "adapter_link56_contact_present",
        "adapter_first_selected_obstacle_contact_is_link56",
        "psf_link56_contact_present",
        "psf_any_robot_selected_obstacle_contact_present",
        "psf_clearance_certified",
        "material_correction_before_adapter_contact",
        "psf_task_success_ever",
        "psf_terminal_task_success",
        "psf_task_success_after_material_correction",
        "adapter_task_success_ever",
        "adapter_terminal_task_success",
    )
    booleans = {
        field: _boolean(metrics.get(field), "metrics.%s" % field)
        for field in boolean_fields
    }
    integer_fields = (
        "adapter_filter_update_count",
        "psf_filter_update_count",
        "adapter_physics_substep_count",
        "psf_physics_substep_count",
        "adapter_completed_suffix_action_count",
        "psf_completed_suffix_action_count",
        "psf_invalid_field_query_count",
        "first_material_correction_physical_boundary",
        "adapter_first_link56_contact_physical_boundary",
        "material_correction_update_count",
    )
    integers = {
        field: _integer(metrics.get(field), "metrics.%s" % field)
        for field in integer_fields
    }
    first_success_raw = metrics.get("psf_first_task_success_source_action_index")
    first_success = (
        None
        if first_success_raw is None
        else _integer(first_success_raw, "metrics.psf_first_task_success_source_action_index")
    )
    number_fields = (
        "maximum_correction_norm_rad_s",
        "filter_correction_integral_rad",
        "post_correction_measured_joint_motion_integral_rad",
        "post_correction_cartesian_path_length_m",
        "post_correction_executed_command_integral_rad",
        "post_correction_zero_command_fraction",
    )
    numbers = {
        field: _number(metrics.get(field), "metrics.%s" % field)
        for field in number_fields
    }

    expected_updates = derived["expected_updates"]
    expected_substeps = derived["expected_substeps"]
    expected_actions = derived["action_count"]
    psf_contact_present = bool(
        booleans["psf_link56_contact_present"]
        or booleans["psf_any_robot_selected_obstacle_contact_present"]
    )

    apparatus_reasons = []
    for field, reason in (
        ("exact_paired_start", "paired_start_not_exact"),
        ("shared_prefix_complete", "shared_prefix_incomplete"),
        ("adapter_exposure_complete", "adapter_exposure_incomplete"),
        (
            "adapter_physics_monitor_trace_counts_match",
            "adapter_physics_monitor_trace_count_mismatch",
        ),
        (
            "psf_physics_monitor_trace_counts_match",
            "psf_physics_monitor_trace_count_mismatch",
        ),
        (
            "all_issued_commands_within_physical_bounds",
            "issued_command_physical_bounds_failed",
        ),
        (
            "both_nominal_commands_within_dynamic_joint_bounds",
            "nominal_dynamic_joint_bounds_confounded",
        ),
        ("static_selected_obstacle_admissible", "static_obstacle_inadmissible"),
        ("boundary_goal_unsatisfied", "task_was_already_complete_at_branch"),
    ):
        if not booleans[field]:
            apparatus_reasons.append(reason)
    for field, expected in (
        ("adapter_filter_update_count", expected_updates),
        ("adapter_physics_substep_count", expected_substeps),
        ("adapter_completed_suffix_action_count", expected_actions),
    ):
        if integers[field] != expected:
            apparatus_reasons.append("%s_differs" % field)
    if integers["psf_invalid_field_query_count"] != derived["thresholds"][
        "maximum_invalid_field_queries"
    ]:
        apparatus_reasons.append("psf_invalid_field_query_count_differs")

    # A literal PSF contact is a valid negative even though execution stops at
    # that contact.  No-contact claims, in contrast, require the entire suffix.
    if psf_contact_present:
        if not (0 < integers["psf_filter_update_count"] <= expected_updates):
            apparatus_reasons.append("psf_contact_prefix_filter_count_invalid")
        if not (0 < integers["psf_physics_substep_count"] <= expected_substeps):
            apparatus_reasons.append("psf_contact_prefix_physics_count_invalid")
        if not (
            0 <= integers["psf_completed_suffix_action_count"] <= expected_actions
        ):
            apparatus_reasons.append("psf_contact_prefix_action_count_invalid")
    else:
        for field, reason in (
            ("full_recorded_episode_complete", "full_recorded_episode_incomplete"),
            ("psf_exposure_complete", "psf_exposure_incomplete"),
            ("psf_qp_count_complete", "psf_qp_count_incomplete"),
            ("psf_qp_postchecks_complete", "psf_qp_postchecks_incomplete"),
            (
                "psf_joint_limit_postchecks_complete",
                "psf_joint_limit_postchecks_incomplete",
            ),
            (
                "psf_all_post_state_field_queries_valid_and_positive",
                "psf_post_state_field_queries_invalid",
            ),
        ):
            if not booleans[field]:
                apparatus_reasons.append(reason)
        for field, expected in (
            ("psf_filter_update_count", expected_updates),
            ("psf_physics_substep_count", expected_substeps),
            ("psf_completed_suffix_action_count", expected_actions),
        ):
            if integers[field] != expected:
                apparatus_reasons.append("%s_differs" % field)

    baseline_reproduced = bool(
        booleans["adapter_link56_contact_present"]
        and booleans["adapter_first_selected_obstacle_contact_is_link56"]
    )
    contact_prevented = bool(
        baseline_reproduced
        and not psf_contact_present
        and booleans["psf_clearance_certified"]
    )
    thresholds = derived["thresholds"]
    correction_attributed = bool(
        booleans["material_correction_before_adapter_contact"]
        and integers["material_correction_update_count"] > 0
        and integers["first_material_correction_physical_boundary"]
        < integers["adapter_first_link56_contact_physical_boundary"]
        and numbers["maximum_correction_norm_rad_s"]
        >= thresholds["minimum_filter_correction_norm_rad_s"]
        and numbers["filter_correction_integral_rad"]
        >= thresholds["minimum_filter_correction_integral_rad"]
    )
    motion_checks = {
        "measured_joint_motion": numbers[
            "post_correction_measured_joint_motion_integral_rad"
        ]
        >= thresholds[
            "minimum_post_correction_measured_joint_motion_integral_rad"
        ],
        "cartesian_motion": numbers["post_correction_cartesian_path_length_m"]
        >= thresholds["minimum_post_correction_cartesian_path_length_m"],
        "issued_motion": numbers[
            "post_correction_executed_command_integral_rad"
        ]
        >= thresholds["minimum_post_correction_executed_command_integral_rad"],
        "not_nearly_all_zero": numbers["post_correction_zero_command_fraction"]
        <= thresholds["maximum_post_correction_zero_command_fraction"],
    }
    useful_correction = bool(correction_attributed and all(motion_checks.values()))
    task_successful = bool(
        booleans["psf_task_success_ever"]
        and booleans["psf_terminal_task_success"]
        and booleans["psf_task_success_after_material_correction"]
        and first_success is not None
        and derived["start_action"] <= first_success <= derived["end_action"]
        and (first_success + 1) * 25
        > integers["first_material_correction_physical_boundary"]
    )
    pair_complete = bool(
        not apparatus_reasons
        and booleans["adapter_exposure_complete"]
        and (booleans["psf_exposure_complete"] or psf_contact_present)
    )
    apparatus_valid = not apparatus_reasons
    stop_only = bool(
        apparatus_valid
        and baseline_reproduced
        and not psf_contact_present
        and correction_attributed
        and not useful_correction
    )

    if not apparatus_valid:
        classification = "INCONCLUSIVE_APPARATUS"
        reasons = apparatus_reasons
    elif not baseline_reproduced:
        classification = "INCONCLUSIVE_APPARATUS"
        reasons = ["adapter_did_not_reproduce_first_link56_contact"]
    elif psf_contact_present:
        classification = "CONTACT_REMAINS_OR_SHIFTED"
        reasons = [
            "psf_link56_contact_present"
            if booleans["psf_link56_contact_present"]
            else "psf_shifted_contact_to_other_robot_geometry"
        ]
    elif not booleans["psf_clearance_certified"]:
        classification = "INCONCLUSIVE_APPARATUS"
        reasons = ["psf_no_contact_but_full_robot_clearance_uncertified"]
    elif not correction_attributed:
        classification = "INCONCLUSIVE_APPARATUS"
        reasons = ["contact_prevention_not_attributable_to_material_psf_correction"]
    elif not useful_correction:
        classification = "STOP_ONLY"
        reasons = [
            "post_correction_motion_gate_failed:%s" % field
            for field, passed in motion_checks.items()
            if not passed
        ]
    elif not task_successful:
        classification = "CONTACT_PREVENTED_TASK_FAILED"
        reasons = ["native_safelibero_goal_not_satisfied_after_useful_correction"]
    else:
        classification = "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"
        reasons = []

    return {
        "schema_version": CLASSIFICATION_SCHEMA,
        "classification": classification,
        "goal_achieved": classification
        == "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        "apparatus_valid": apparatus_valid,
        "pair_complete": pair_complete,
        "baseline_reproduced": baseline_reproduced,
        "contact_prevented": contact_prevented,
        "task_successful": task_successful,
        "useful_correction": useful_correction,
        "stop_only": stop_only,
        "apparatus_failure_reasons": apparatus_reasons,
        "classification_reasons": reasons,
        "gates": {
            "correction_attributed": correction_attributed,
            "motion": motion_checks,
            "native_task_success_ever": booleans["psf_task_success_ever"],
            "native_task_success_terminal": booleans[
                "psf_terminal_task_success"
            ],
            "native_task_success_after_correction": booleans[
                "psf_task_success_after_material_correction"
            ],
            "no_psf_robot_obstacle_contact": not psf_contact_present,
            "full_robot_clearance_certified": booleans[
                "psf_clearance_certified"
            ],
        },
        "diagnostics": {
            "tracking_is_feasibility_gate": False,
            "timing_is_realtime_claim_gate": False,
            "adapter_task_success_ever": booleans[
                "adapter_task_success_ever"
            ],
            "adapter_terminal_task_success": booleans[
                "adapter_terminal_task_success"
            ],
        },
        "claim_scope": (
            "one_offline_single_case_hybrid_full_recorded_episode_controller_"
            "feasibility_not_closed_loop_policy_not_population_safety_not_realtime"
        ),
    }


__all__ = [
    "CASE_ID",
    "CLASSIFICATIONS",
    "CLASSIFICATION_SCHEMA",
    "FullEpisodeFeasibilityError",
    "METRIC_KEYS",
    "METRICS_SCHEMA",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "RESULT_SCHEMA",
    "SOURCE_ARM",
    "classify_full_episode_feasibility",
    "validate_full_episode_feasibility_protocol",
]
