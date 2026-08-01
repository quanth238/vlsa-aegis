"""Pure protocol and outcome logic for the fast paired Poisson canary.

This module deliberately makes a narrow claim.  The window is selected from a
known historical collision, so a positive result is local implementation
evidence rather than an early-warning, full-episode, or population result.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence


PROTOCOL_SCHEMA = "vlsa_poisson_fast_feasibility_protocol.v1"
RESULT_SCHEMA = "vlsa_poisson_fast_feasibility_result.v1"
CASE_ID = "vlsa-t1-goal-ii-t0-e05"
SOURCE_ARM = "pi05_plus_aegis_translational"
LABELS = (
    "CONTACT_PREVENTION_FEASIBLE",
    "CONTACT_PREVENTION_FAILED",
    "UNCERTIFIED_CLEARANCE",
    "MOTION_PRESERVING_CORRECTION",
    "STOP_ONLY",
    "INCONCLUSIVE",
    "APPARATUS_FAILURE",
)


class FastFeasibilityError(RuntimeError):
    """Raised when the frozen protocol or compact evidence is malformed."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FastFeasibilityError("%s must be an object" % label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FastFeasibilityError("%s must be an array" % label)
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise FastFeasibilityError("%s must be Boolean" % label)
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FastFeasibilityError(
            "%s must be an integer >= %d" % (label, minimum)
        )
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FastFeasibilityError("%s must be numeric" % label)
    output = float(value)
    if not math.isfinite(output):
        raise FastFeasibilityError("%s must be finite" % label)
    return output


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FastFeasibilityError("%s must be a lowercase SHA-256" % label)
    return value


def validate_fast_feasibility_protocol(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the preregistered window and return its derived constants."""

    protocol = _mapping(value, "protocol")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise FastFeasibilityError("fast protocol schema differs")
    if protocol.get("protocol_id") != "vlsa-poisson-link56-fast-window-v1":
        raise FastFeasibilityError("fast protocol ID differs")
    case = _mapping(protocol.get("case"), "protocol.case")
    if case.get("case_id") != CASE_ID:
        raise FastFeasibilityError("fast protocol case differs")
    if case.get("selected_obstacle_name") != "moka_pot_obstacle_1":
        raise FastFeasibilityError("fast protocol selected obstacle differs")
    if list(_sequence(case.get("protected_robot_body_names"), "protected bodies")) != [
        "robot0_link5",
        "robot0_link6",
    ]:
        raise FastFeasibilityError("fast protocol protected bodies differ")

    source = _mapping(protocol.get("source"), "protocol.source")
    if source.get("source_arm") != SOURCE_ARM:
        raise FastFeasibilityError("fast protocol source arm differs")
    if source.get("action_field") != "actions[*].executed":
        raise FastFeasibilityError("fast protocol must consume executed actions")
    if source.get("nominal_translational_field_prohibited") is not True:
        raise FastFeasibilityError("nominal-translational source must be prohibited")
    for field in (
        "historical_result_file_sha256",
        "historical_result_payload_sha256",
        "historical_executed_action_sequence_sha256",
        "window_action_record_sha256",
        "post_action_179_flattened_state_sha256",
    ):
        _sha256(source.get(field), "protocol.source.%s" % field)
    if _integer(
        source.get("historical_executed_action_count"),
        "historical action count",
        1,
    ) != 237:
        raise FastFeasibilityError("historical action count differs")

    window = _mapping(protocol.get("window"), "protocol.window")
    start_boundary = _integer(
        window.get("start_physical_boundary"), "window start boundary"
    )
    end_boundary = _integer(
        window.get("end_physical_boundary"), "window end boundary", 1
    )
    start_action = _integer(window.get("start_action_index"), "start action")
    end_action = _integer(
        window.get("end_action_index_inclusive"), "end action"
    )
    action_count = _integer(window.get("action_count"), "window action count", 1)
    if (
        start_boundary != 4500
        or end_boundary != 4700
        or start_action != 180
        or end_action != 187
        or action_count != 8
        or start_boundary != start_action * 25
        or end_boundary != (end_action + 1) * 25
        or action_count != end_action - start_action + 1
    ):
        raise FastFeasibilityError("fast source window differs")
    contact_boundary = _integer(
        window.get("historical_contact_boundary_diagnostic"),
        "historical contact boundary",
    )
    if not start_boundary < contact_boundary <= end_boundary:
        raise FastFeasibilityError("historical contact is outside the fixed window")
    if window.get("selection_kind") != "post_hoc_known_collision_window":
        raise FastFeasibilityError("post-hoc window disclosure is missing")
    if window.get("no_boundary_or_horizon_change_after_outcome") is not True:
        raise FastFeasibilityError("window freeze declaration is missing")

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
        cadence.get("expected_physics_substeps_per_arm"),
        "expected substeps",
        1,
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
        raise FastFeasibilityError("fast cadence differs")

    pairing = _mapping(protocol.get("pairing"), "protocol.pairing")
    if (
        pairing.get("settle_actions") != 20
        or pairing.get("online_policy_queries") != 0
        or "same_historical_aegis_executed_actions_180_through_187"
        not in str(pairing.get("nominal_input"))
    ):
        raise FastFeasibilityError("fast pairing contract differs")
    arms = list(_sequence(protocol.get("arms"), "protocol.arms"))
    if arms != [
        "joint_velocity_adapter_only",
        "joint_velocity_adapter_plus_link56_psf",
    ]:
        raise FastFeasibilityError("fast arm order differs")

    acceptance = _mapping(protocol.get("acceptance"), "protocol.acceptance")
    thresholds = {
        "minimum_safe_cbf_residual_m2_per_s": _number(
            acceptance.get("minimum_safe_cbf_residual_m2_per_s"),
            "safe CBF threshold",
        ),
        "maximum_nominal_cbf_residual_for_activation_m2_per_s": _number(
            acceptance.get(
                "maximum_nominal_cbf_residual_for_activation_m2_per_s"
            ),
            "nominal CBF activation threshold",
        ),
        "maximum_invalid_field_queries": _integer(
            acceptance.get("maximum_invalid_field_queries"),
            "invalid-query threshold",
        ),
        "maximum_tracking_linf_rad_s": _number(
            acceptance.get("maximum_tracking_linf_rad_s"),
            "tracking Linf threshold",
        ),
        "maximum_tracking_rmse_rad_s": _number(
            acceptance.get("maximum_tracking_rmse_rad_s"),
            "tracking RMSE threshold",
        ),
        "minimum_filter_correction_norm_rad_s": _number(
            acceptance.get("minimum_filter_correction_norm_rad_s"),
            "correction threshold",
        ),
        "minimum_safe_command_norm_rad_s": _number(
            acceptance.get("minimum_safe_command_norm_rad_s"),
            "safe-command threshold",
        ),
        "minimum_safe_to_nominal_command_motion_ratio": _number(
            acceptance.get("minimum_safe_to_nominal_command_motion_ratio"),
            "command-retention threshold",
        ),
        "minimum_safe_measured_joint_motion_integral_rad": _number(
            acceptance.get("minimum_safe_measured_joint_motion_integral_rad"),
            "measured-motion threshold",
        ),
        "minimum_active_interval_cartesian_path_length_m": _number(
            acceptance.get("minimum_active_interval_cartesian_path_length_m"),
            "active-interval Cartesian-motion threshold",
        ),
    }
    for field in (
        "require_adapter_link56_contact",
        "require_adapter_first_selected_obstacle_contact_is_link56",
        "require_psf_link56_contact_absent",
        "require_psf_all_robot_selected_obstacle_contact_absent",
        "require_psf_protected_link_coverage_lower_bound_strictly_positive",
        "require_psf_all_post_state_field_queries_valid_and_positive",
        "require_static_selected_obstacle",
        "require_exact_paired_start",
        "require_complete_fixed_exposure_for_no_contact_claim",
        "require_valid_psf_precontact_execution_for_contact_failure",
        "require_both_arms_nominal_commands_within_same_dynamic_joint_bounds",
        "require_exhaustive_post_state_field_query_population",
        "require_material_poisson_activation_before_adapter_contact",
    ):
        if _boolean(acceptance.get(field), "acceptance.%s" % field) is not True:
            raise FastFeasibilityError("required acceptance gate is disabled")
    labels = list(
        _sequence(
            _mapping(protocol.get("classification"), "classification").get(
                "labels"
            ),
            "classification labels",
        )
    )
    if labels != list(LABELS):
        raise FastFeasibilityError("classification labels differ")
    classification = _mapping(
        protocol.get("classification"), "protocol.classification"
    )
    if list(
        _sequence(classification.get("primary_outcomes"), "primary outcomes")
    ) != [
        "APPARATUS_FAILURE",
        "INCONCLUSIVE",
        "CONTACT_PREVENTION_FAILED",
        "UNCERTIFIED_CLEARANCE",
        "CONTACT_PREVENTION_FEASIBLE",
    ]:
        raise FastFeasibilityError("primary outcome labels differ")
    if list(
        _sequence(classification.get("safety_mechanisms"), "safety mechanisms")
    ) != ["NOT_APPLICABLE", "MOTION_PRESERVING_CORRECTION", "STOP_ONLY"]:
        raise FastFeasibilityError("safety-mechanism labels differ")
    result_contract = _mapping(protocol.get("result_contract"), "result contract")
    if (
        result_contract.get("schema_version") != RESULT_SCHEMA
        or result_contract.get("independent_consumer_required") is not False
        or result_contract.get("partial_output_policy") != "never_interpret"
    ):
        raise FastFeasibilityError("fast result contract differs")
    return {
        "start_boundary": start_boundary,
        "end_boundary": end_boundary,
        "start_action": start_action,
        "end_action": end_action,
        "action_count": action_count,
        "expected_updates": expected_updates,
        "expected_substeps": expected_substeps,
        "thresholds": thresholds,
    }


def classify_fast_feasibility(
    metrics_value: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    """Classify one complete compact pair without trusting producer labels."""

    derived = validate_fast_feasibility_protocol(protocol)
    metrics = _mapping(metrics_value, "metrics")
    required_boolean_fields = (
        "exact_paired_start",
        "adapter_exposure_complete",
        "psf_exposure_complete",
        "adapter_precontact_static_obstacle_admissible",
        "psf_static_obstacle_admissible",
        "psf_all_qp_solved",
        "psf_all_qp_postchecks_passed",
        "psf_all_joint_limit_postchecks_passed",
        "psf_all_post_state_field_queries_valid_and_positive",
        "psf_precontact_execution_valid",
        "psf_precontact_static_obstacle_admissible",
        "psf_precontact_field_queries_valid_and_positive",
        "psf_nominal_cbf_activation_with_material_correction_before_adapter_contact",
        "psf_active_interval_motion_complete",
        "all_issued_commands_within_physical_bounds",
        "adapter_all_nominal_commands_within_dynamic_joint_bounds",
        "psf_all_nominal_commands_within_dynamic_joint_bounds",
        "adapter_link56_contact_present",
        "adapter_first_selected_obstacle_contact_is_link56",
        "psf_link56_contact_present",
        "psf_any_robot_selected_obstacle_contact_present",
        "psf_protected_link_clearance_lower_bound_available",
    )
    booleans = {
        field: _boolean(metrics.get(field), "metrics.%s" % field)
        for field in required_boolean_fields
    }
    expected_updates = derived["expected_updates"]
    expected_substeps = derived["expected_substeps"]
    integers = {
        field: _integer(metrics.get(field), "metrics.%s" % field)
        for field in (
            "adapter_filter_update_count",
            "psf_filter_update_count",
            "adapter_physics_substep_count",
            "psf_physics_substep_count",
            "psf_qp_solve_count",
            "psf_qp_postcheck_count",
            "psf_joint_limit_postcheck_count",
            "adapter_issued_command_bound_check_count",
            "psf_issued_command_bound_check_count",
            "adapter_precontact_tracking_observation_count",
            "psf_tracking_observation_count",
            "psf_post_state_field_observation_count",
            "psf_post_state_field_query_count",
            "psf_nonpositive_post_state_field_query_count",
            "psf_invalid_field_query_count",
            "adapter_nominal_dynamic_bound_check_count",
            "psf_nominal_dynamic_bound_check_count",
            "adapter_nominal_dynamic_bound_violation_count",
            "psf_nominal_dynamic_bound_violation_count",
            "psf_protected_sample_count",
            "psf_precontact_tracking_observation_count",
            "psf_activation_update_count_before_adapter_contact",
            "adapter_first_link56_contact_physical_boundary",
            "psf_first_activation_physical_boundary",
        )
    }
    numbers = {
        field: _number(metrics.get(field), "metrics.%s" % field)
        for field in (
            "psf_minimum_D_sim_m",
            "psf_minimum_safe_cbf_residual_m2_per_s",
            "adapter_precontact_tracking_linf_rad_s",
            "adapter_precontact_tracking_rmse_rad_s",
            "psf_tracking_linf_rad_s",
            "psf_tracking_rmse_rad_s",
            "maximum_active_filter_correction_norm_rad_s",
            "maximum_active_safe_command_norm_rad_s",
            "active_safe_to_nominal_command_motion_ratio",
            "active_safe_measured_joint_motion_integral_rad",
            "active_cartesian_path_length_m",
            "psf_protected_link_full_surface_clearance_lower_bound_m",
            "psf_precontact_tracking_linf_rad_s",
            "psf_precontact_tracking_rmse_rad_s",
            "psf_minimum_nominal_cbf_residual_before_adapter_contact_m2_per_s",
            "psf_maximum_activation_correction_norm_before_adapter_contact_rad_s",
        )
    }
    thresholds = derived["thresholds"]
    apparatus_reasons = []
    baseline_reproduced = booleans["adapter_link56_contact_present"]
    baseline_target_clean = booleans[
        "adapter_first_selected_obstacle_contact_is_link56"
    ]
    psf_contact_present = bool(
        booleans["psf_link56_contact_present"]
        or booleans["psf_any_robot_selected_obstacle_contact_present"]
    )
    if not booleans["exact_paired_start"]:
        apparatus_reasons.append("paired_start_not_exact")
    if not booleans["adapter_exposure_complete"]:
        apparatus_reasons.append("adapter_exposure_incomplete")
    if not booleans["adapter_precontact_static_obstacle_admissible"]:
        apparatus_reasons.append("adapter_precontact_static_obstacle_inadmissible")
    if not booleans["all_issued_commands_within_physical_bounds"]:
        apparatus_reasons.append("issued_command_physical_bounds_failed")
    if not booleans["adapter_all_nominal_commands_within_dynamic_joint_bounds"]:
        apparatus_reasons.append("adapter_nominal_dynamic_joint_bounds_confounded")
    common_expected_counts = {
        "adapter_filter_update_count": expected_updates,
        "adapter_physics_substep_count": expected_substeps,
        "adapter_issued_command_bound_check_count": expected_updates,
        "adapter_nominal_dynamic_bound_check_count": expected_updates,
    }
    for field, expected in common_expected_counts.items():
        if integers[field] != expected:
            apparatus_reasons.append("%s_differs" % field)
    if integers["adapter_precontact_tracking_observation_count"] <= 0:
        apparatus_reasons.append("adapter_precontact_tracking_population_empty")
    if integers["adapter_nominal_dynamic_bound_violation_count"] != 0:
        apparatus_reasons.append("adapter_nominal_dynamic_joint_bound_violation")
    if integers["psf_protected_sample_count"] <= 0:
        apparatus_reasons.append("psf_protected_sample_population_empty")
    if integers["psf_invalid_field_query_count"] > thresholds[
        "maximum_invalid_field_queries"
    ]:
        apparatus_reasons.append("psf_invalid_field_query")
    for arm, linf_field, rmse_field in (
        ("adapter_precontact", "adapter_precontact_tracking_linf_rad_s", "adapter_precontact_tracking_rmse_rad_s"),
    ):
        if numbers[linf_field] > thresholds["maximum_tracking_linf_rad_s"]:
            apparatus_reasons.append("%s_tracking_linf_failed" % arm)
        if numbers[rmse_field] > thresholds["maximum_tracking_rmse_rad_s"]:
            apparatus_reasons.append("%s_tracking_rmse_failed" % arm)

    if psf_contact_present:
        if not booleans["psf_all_nominal_commands_within_dynamic_joint_bounds"]:
            apparatus_reasons.append("psf_nominal_dynamic_joint_bounds_confounded")
        if not booleans["psf_precontact_execution_valid"]:
            apparatus_reasons.append("psf_precontact_execution_invalid")
        if not booleans["psf_precontact_static_obstacle_admissible"]:
            apparatus_reasons.append("psf_precontact_static_obstacle_inadmissible")
        if not booleans["psf_precontact_field_queries_valid_and_positive"]:
            apparatus_reasons.append("psf_precontact_field_invalid")
        if integers["psf_filter_update_count"] <= 0:
            apparatus_reasons.append("psf_contact_without_filter_update")
        for field in (
            "psf_qp_solve_count",
            "psf_qp_postcheck_count",
            "psf_joint_limit_postcheck_count",
            "psf_issued_command_bound_check_count",
            "psf_nominal_dynamic_bound_check_count",
        ):
            if integers[field] != integers["psf_filter_update_count"]:
                apparatus_reasons.append("%s_differs_from_entered_prefix" % field)
        if integers["psf_nominal_dynamic_bound_violation_count"] != 0:
            apparatus_reasons.append("psf_nominal_dynamic_joint_bound_violation")
        if integers["psf_post_state_field_query_count"] != (
            integers["psf_post_state_field_observation_count"]
            * integers["psf_protected_sample_count"]
        ):
            apparatus_reasons.append("psf_precontact_field_query_population_differs")
        if integers["psf_precontact_tracking_observation_count"] > 0:
            if numbers["psf_precontact_tracking_linf_rad_s"] > thresholds[
                "maximum_tracking_linf_rad_s"
            ]:
                apparatus_reasons.append("psf_precontact_tracking_linf_failed")
            if numbers["psf_precontact_tracking_rmse_rad_s"] > thresholds[
                "maximum_tracking_rmse_rad_s"
            ]:
                apparatus_reasons.append("psf_precontact_tracking_rmse_failed")
    else:
        if not booleans["psf_all_nominal_commands_within_dynamic_joint_bounds"]:
            apparatus_reasons.append("psf_nominal_dynamic_joint_bounds_confounded")
        for field in (
            "psf_all_qp_solved",
            "psf_all_qp_postchecks_passed",
            "psf_all_joint_limit_postchecks_passed",
            "psf_all_post_state_field_queries_valid_and_positive",
        ):
            if not booleans[field]:
                apparatus_reasons.append("%s_false" % field)
        if not booleans["psf_exposure_complete"]:
            apparatus_reasons.append("psf_exposure_incomplete")
        if not booleans["psf_static_obstacle_admissible"]:
            apparatus_reasons.append("psf_static_obstacle_inadmissible")
        if (
            booleans[
                "psf_nominal_cbf_activation_with_material_correction_before_adapter_contact"
            ]
            and not booleans["psf_active_interval_motion_complete"]
        ):
            apparatus_reasons.append("psf_active_interval_motion_incomplete")
        for field, expected in {
            "psf_filter_update_count": expected_updates,
            "psf_physics_substep_count": expected_substeps,
            "psf_qp_solve_count": expected_updates,
            "psf_qp_postcheck_count": expected_updates,
            "psf_joint_limit_postcheck_count": expected_updates,
            "psf_issued_command_bound_check_count": expected_updates,
            "psf_nominal_dynamic_bound_check_count": expected_updates,
            "psf_tracking_observation_count": expected_substeps,
            "psf_post_state_field_observation_count": expected_substeps,
        }.items():
            if integers[field] != expected:
                apparatus_reasons.append("%s_differs" % field)
        if integers["psf_post_state_field_query_count"] <= 0:
            apparatus_reasons.append("psf_post_state_field_population_empty")
        if integers["psf_post_state_field_query_count"] != (
            integers["psf_post_state_field_observation_count"]
            * integers["psf_protected_sample_count"]
        ):
            apparatus_reasons.append("psf_post_state_field_query_population_differs")
        if integers["psf_nominal_dynamic_bound_violation_count"] != 0:
            apparatus_reasons.append("psf_nominal_dynamic_joint_bound_violation")
        if integers["psf_nonpositive_post_state_field_query_count"] != 0:
            apparatus_reasons.append("psf_post_state_nonpositive_h")
        if numbers["psf_tracking_linf_rad_s"] > thresholds[
            "maximum_tracking_linf_rad_s"
        ]:
            apparatus_reasons.append("psf_tracking_linf_failed")
        if numbers["psf_tracking_rmse_rad_s"] > thresholds[
            "maximum_tracking_rmse_rad_s"
        ]:
            apparatus_reasons.append("psf_tracking_rmse_failed")
    if numbers["psf_minimum_safe_cbf_residual_m2_per_s"] < thresholds[
        "minimum_safe_cbf_residual_m2_per_s"
    ]:
        apparatus_reasons.append("psf_qp_residual_failed")

    prevention_failure_reasons = []
    if booleans["psf_link56_contact_present"]:
        prevention_failure_reasons.append("psf_link56_contact_present")
    if booleans["psf_any_robot_selected_obstacle_contact_present"]:
        prevention_failure_reasons.append("psf_shifted_or_other_robot_contact_present")
    clearance_certified = bool(
        booleans["psf_protected_link_clearance_lower_bound_available"]
        and numbers["psf_protected_link_full_surface_clearance_lower_bound_m"] > 0.0
    )
    activation_attributed = bool(
        booleans[
            "psf_nominal_cbf_activation_with_material_correction_before_adapter_contact"
        ]
        and integers["psf_activation_update_count_before_adapter_contact"] > 0
        and integers["psf_first_activation_physical_boundary"]
        < integers["adapter_first_link56_contact_physical_boundary"]
        and numbers[
            "psf_minimum_nominal_cbf_residual_before_adapter_contact_m2_per_s"
        ]
        <= thresholds[
            "maximum_nominal_cbf_residual_for_activation_m2_per_s"
        ]
        and numbers[
            "psf_maximum_activation_correction_norm_before_adapter_contact_rad_s"
        ]
        >= thresholds["minimum_filter_correction_norm_rad_s"]
    )

    motion_preservation_checks = {
        "material_filter_correction": numbers[
            "maximum_active_filter_correction_norm_rad_s"
        ]
        >= thresholds["minimum_filter_correction_norm_rad_s"],
        "material_safe_command": numbers["maximum_active_safe_command_norm_rad_s"]
        >= thresholds["minimum_safe_command_norm_rad_s"],
        "material_command_retention": numbers[
            "active_safe_to_nominal_command_motion_ratio"
        ]
        >= thresholds["minimum_safe_to_nominal_command_motion_ratio"],
        "material_realized_joint_motion": numbers[
            "active_safe_measured_joint_motion_integral_rad"
        ]
        >= thresholds["minimum_safe_measured_joint_motion_integral_rad"],
        "material_cartesian_motion": numbers[
            "active_cartesian_path_length_m"
        ]
        >= thresholds["minimum_active_interval_cartesian_path_length_m"],
    }

    apparatus_valid = not apparatus_reasons
    contact_prevention_failed = bool(
        apparatus_valid
        and baseline_reproduced
        and baseline_target_clean
        and prevention_failure_reasons
    )
    uncertified_clearance = bool(
        apparatus_valid
        and baseline_reproduced
        and baseline_target_clean
        and not prevention_failure_reasons
        and not clearance_certified
    )
    contact_prevention_feasible = bool(
        apparatus_valid
        and baseline_reproduced
        and baseline_target_clean
        and not prevention_failure_reasons
        and clearance_certified
        and activation_attributed
    )
    motion_preserving = contact_prevention_feasible and all(
        motion_preservation_checks.values()
    )
    stop_only = contact_prevention_feasible and not motion_preserving
    if not apparatus_valid:
        primary_outcome = "APPARATUS_FAILURE"
        safety_mechanism = "NOT_APPLICABLE"
        reasons = apparatus_reasons
    elif not baseline_reproduced:
        primary_outcome = "INCONCLUSIVE"
        safety_mechanism = "NOT_APPLICABLE"
        reasons = ["adapter_did_not_reproduce_link56_contact"]
    elif not booleans["adapter_first_selected_obstacle_contact_is_link56"]:
        primary_outcome = "INCONCLUSIVE"
        safety_mechanism = "NOT_APPLICABLE"
        reasons = ["adapter_link56_contact_was_not_the_first_selected_obstacle_contact"]
    elif contact_prevention_failed:
        primary_outcome = "CONTACT_PREVENTION_FAILED"
        safety_mechanism = "NOT_APPLICABLE"
        reasons = prevention_failure_reasons
    elif uncertified_clearance:
        primary_outcome = "UNCERTIFIED_CLEARANCE"
        safety_mechanism = "NOT_APPLICABLE"
        reasons = ["psf_protected_link_clearance_not_strictly_positive"]
    elif not activation_attributed:
        primary_outcome = "INCONCLUSIVE"
        safety_mechanism = "NOT_APPLICABLE"
        reasons = [
            "poisson_cbf_not_materially_activated_before_adapter_contact"
        ]
    elif motion_preserving:
        primary_outcome = "CONTACT_PREVENTION_FEASIBLE"
        safety_mechanism = "MOTION_PRESERVING_CORRECTION"
        reasons = []
    else:
        primary_outcome = "CONTACT_PREVENTION_FEASIBLE"
        safety_mechanism = "STOP_ONLY"
        reasons = [
            "motion_preservation_gate_failed:%s" % name
            for name, passed in motion_preservation_checks.items()
            if not passed
        ]
    return {
        "primary_outcome": primary_outcome,
        "safety_mechanism": safety_mechanism,
        "allowed_labels": list(LABELS),
        "apparatus_valid": apparatus_valid,
        "contact_prevention_label": (
            "CONTACT_PREVENTION_FEASIBLE"
            if contact_prevention_feasible
            else None
        ),
        "contact_prevention_feasible": contact_prevention_feasible,
        "contact_prevention_failed": contact_prevention_failed,
        "uncertified_clearance": uncertified_clearance,
        "motion_preserving_correction": motion_preserving,
        "stop_only": stop_only,
        "apparatus_failure_reasons": apparatus_reasons,
        "contact_prevention_reasons": prevention_failure_reasons,
        "motion_preservation_checks": motion_preservation_checks,
        "classification_reasons": reasons,
        "claim_scope": (
            "one_outcome_conditioned_0.4_second_window_only_not_task_success"
        ),
    }


__all__ = [
    "CASE_ID",
    "FastFeasibilityError",
    "LABELS",
    "PROTOCOL_SCHEMA",
    "RESULT_SCHEMA",
    "SOURCE_ARM",
    "classify_fast_feasibility",
    "validate_fast_feasibility_protocol",
]
