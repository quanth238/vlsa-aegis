"""Minimal event-triggered direct-qdot Poisson-CBF rescue contract.

The accepted e05 simulation showed that a link-aware CBF works when its
decision variable is the physical joint velocity that is actually executed.
This module deliberately does not model native OSC target-to-velocity
dynamics.  Native OSC is kept byte-identical before the first unsafe bounded
direct-qdot preview; from that state, paired joint-velocity arms compare the
unfiltered adapter with the hard Poisson-CBF QP.

The trigger is geometric and controller-aligned.  It contains no case-specific
action number and the protected link set is fixed once for every registered
case.  The first experiment remains an offline recorded-suffix controller
feasibility test, not a closed-loop VLA or population-safety claim.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence


PROTOCOL_SCHEMA = "vlsa_poisson_triggered_rescue_protocol.v1"
PROTOCOL_ID = "vlsa-poisson-triggered-direct-qdot-link56-v1"
RESULT_SCHEMA = "vlsa_poisson_triggered_rescue_result.v1"
METRICS_SCHEMA = "vlsa_poisson_triggered_rescue_metrics.v1"
CLASSIFICATION_SCHEMA = "vlsa_poisson_triggered_rescue_classification.v1"

PROTECTED_BODY_NAMES = ("robot0_link5", "robot0_link6")
REGISTERED_CASE_IDS = (
    "vlsa-t1-goal-ii-t0-e05",
    "vlsa-t1-goal-ii-t3-e42",
)

CLASSIFICATIONS = (
    "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
    "NO_ACTIONABLE_DIRECT_QDOT_WARNING",
    "BASELINE_CONTACT_NOT_REPRODUCED",
    "CONTACT_REMAINS_OR_SHIFTED",
    "CONTACT_PREVENTED_TASK_FAILED",
    "STOP_OR_METHOD_FAILURE",
    "INCONCLUSIVE_APPARATUS",
)


class TriggeredRescueError(RuntimeError):
    """Raised when protocol or compact scientific evidence is malformed."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TriggeredRescueError("%s must be an object" % label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TriggeredRescueError("%s must be an array" % label)
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TriggeredRescueError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise TriggeredRescueError("%s must be finite" % label)
    return result


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TriggeredRescueError(
            "%s must be an integer >= %d" % (label, minimum)
        )
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TriggeredRescueError("%s must be a lowercase SHA-256" % label)
    return value


def validate_triggered_rescue_protocol(
    value: Mapping[str, Any],
    *,
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate the method-level contract and return one case's constants."""

    protocol = _mapping(value, "protocol")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise TriggeredRescueError("triggered-rescue protocol schema differs")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise TriggeredRescueError("triggered-rescue protocol ID differs")

    method = _mapping(protocol.get("method"), "protocol.method")
    if tuple(
        _sequence(method.get("protected_robot_body_names"), "protected links")
    ) != PROTECTED_BODY_NAMES:
        raise TriggeredRescueError("protected link set must be fixed link 5 plus link 6")
    exact_method_literals = {
        "link_selection": "fixed_once_across_all_cases_not_task_conditioned",
        "pretrigger_controller": "released_native_OSC_POSE_byte_exact",
        "trigger_decision_variable": "bounded_physical_arm_joint_velocity_rad_s",
        "trigger_preview": "fresh_translation_only_DLS_first_100Hz_command",
        "trigger_cadence": "before_each_20Hz_source_action_boundary",
        "trigger_latching": "first_actionable_candidate_once",
        "posttrigger_controller": "JOINT_VELOCITY_100Hz",
        "posttrigger_pair": "adapter_only_vs_adapter_plus_link56_psf",
        "native_OSC_pose_to_qdot_surrogate": "prohibited",
        "fixed_trigger_action_index": "prohibited",
        "stop_or_zero_fallback": "prohibited",
    }
    for key, expected in exact_method_literals.items():
        if method.get(key) != expected:
            raise TriggeredRescueError("method.%s differs" % key)

    cases_raw = _sequence(protocol.get("cases"), "protocol.cases")
    cases: Dict[str, Mapping[str, Any]] = {}
    for index, raw_case in enumerate(cases_raw):
        case = _mapping(raw_case, "protocol.cases[%d]" % index)
        observed_case_id = case.get("case_id")
        if observed_case_id in cases or observed_case_id not in REGISTERED_CASE_IDS:
            raise TriggeredRescueError("registered case identity differs")
        horizon = _integer(case.get("historical_action_count"), "case horizon", 1)
        contact_action = _integer(
            case.get("historical_first_link56_contact_action"),
            "historical contact action",
            1,
        )
        if contact_action >= horizon:
            raise TriggeredRescueError("historical contact must occur inside the horizon")
        if not isinstance(case.get("selected_obstacle_name"), str):
            raise TriggeredRescueError("selected obstacle name is absent")
        for field in (
            "historical_result_file_sha256",
            "historical_result_payload_sha256",
            "historical_executed_action_sequence_sha256",
            "source_manifest_row_sha256",
        ):
            _sha256(case.get(field), "case.%s" % field)
        if case.get("historical_aegis_task_success") is not True:
            raise TriggeredRescueError("historical task-success authority differs")
        if case.get("historical_aegis_car_collision") is not True:
            raise TriggeredRescueError("historical collision authority differs")
        cases[str(observed_case_id)] = case
    if tuple(cases) != REGISTERED_CASE_IDS:
        raise TriggeredRescueError("registered cases must be ordered e05 then e42")

    pairing = _mapping(protocol.get("pairing"), "protocol.pairing")
    if pairing.get("settle_actions") != 20:
        raise TriggeredRescueError("settling cadence differs")
    if pairing.get("prefix") != "one_exact_OSC_replay_until_online_trigger":
        raise TriggeredRescueError("prefix pairing differs")
    if pairing.get("state") != "one_exact_trigger_state_restored_into_both_joint_velocity_arms":
        raise TriggeredRescueError("trigger-state pairing differs")
    if pairing.get("suffix_actions") != "same_remaining_recorded_actions_in_both_arms":
        raise TriggeredRescueError("post-trigger action pairing differs")
    if pairing.get("online_policy_queries_after_trigger") != 0:
        raise TriggeredRescueError("first causal test must use no post-trigger policy query")

    cadence = _mapping(protocol.get("cadence"), "protocol.cadence")
    if (
        cadence.get("high_level_frequency_hz") != 20
        or cadence.get("filter_frequency_hz") != 100
        or cadence.get("physics_frequency_hz") != 500
        or cadence.get("filter_updates_per_high_level_action") != 5
        or cadence.get("physics_substeps_per_filter_update") != 5
    ):
        raise TriggeredRescueError("control cadence differs")

    trigger = _mapping(protocol.get("trigger"), "protocol.trigger")
    nominal_threshold = _number(
        trigger.get("maximum_nominal_cbf_residual_m2_per_s"),
        "nominal trigger threshold",
    )
    correction_threshold = _number(
        trigger.get("minimum_material_correction_norm_rad_s"),
        "material correction threshold",
    )
    qp_max_iterations = _integer(
        trigger.get("qp_max_iterations"), "trigger QP iteration budget", 1
    )
    if nominal_threshold >= 0.0 or correction_threshold <= 0.0:
        raise TriggeredRescueError("trigger thresholds must be strict")
    if qp_max_iterations != 50000:
        raise TriggeredRescueError("trigger QP iteration budget differs")
    if trigger.get("require_valid_positive_field") is not True:
        raise TriggeredRescueError("trigger requires a valid positive field")
    if trigger.get("require_solved_hard_qp") is not True:
        raise TriggeredRescueError("trigger requires a solved hard QP")
    if trigger.get("expected_action_index") is not None:
        raise TriggeredRescueError("trigger action must not be preregistered")
    if (
        trigger.get("case_identity_used_by_trigger") is not False
        or trigger.get("historical_contact_action_used_by_trigger") is not False
    ):
        raise TriggeredRescueError(
            "trigger must not use case identity or historical contact timing"
        )

    runtime = _mapping(protocol.get("runtime_binding"), "protocol.runtime_binding")
    selection = _mapping(
        protocol.get("selection_binding"), "protocol.selection_binding"
    )
    for container, fields, label in (
        (
            runtime,
            ("file_sha256", "semantic_sha256", "parameter_block_sha256"),
            "runtime",
        ),
        (selection, ("protocol_config_sha256", "manifest_sha256"), "selection"),
    ):
        for field in fields:
            _sha256(container.get(field), "%s.%s" % (label, field))

    acceptance = _mapping(protocol.get("acceptance"), "protocol.acceptance")
    numeric_thresholds = {
        "minimum_correction_integral_rad": _number(
            acceptance.get("minimum_correction_integral_rad"),
            "minimum correction integral",
        ),
        "minimum_post_correction_measured_joint_motion_integral_rad": _number(
            acceptance.get(
                "minimum_post_correction_measured_joint_motion_integral_rad"
            ),
            "minimum measured motion",
        ),
        "minimum_post_correction_eef_path_length_m": _number(
            acceptance.get("minimum_post_correction_eef_path_length_m"),
            "minimum EEF path",
        ),
        "maximum_post_correction_zero_command_fraction": _number(
            acceptance.get("maximum_post_correction_zero_command_fraction"),
            "maximum zero-command fraction",
        ),
        "paper_car_displacement_threshold_m": _number(
            acceptance.get("paper_car_displacement_threshold_m"),
            "paper CAR threshold",
        ),
    }
    if any(value <= 0.0 for value in numeric_thresholds.values()):
        raise TriggeredRescueError("acceptance thresholds must be positive")
    if numeric_thresholds["maximum_post_correction_zero_command_fraction"] >= 1.0:
        raise TriggeredRescueError("zero-command fraction must be strictly below one")
    for field in (
        "require_exact_trigger_state_pairing",
        "require_baseline_link56_contact",
        "require_zero_psf_any_robot_selected_obstacle_contact",
        "require_zero_psf_shifted_link56_external_contact",
        "require_material_correction_before_baseline_contact",
        "require_paper_car_avoidance",
        "require_useful_motion",
        "require_native_task_success_after_correction_and_terminal",
        "stop_stall_shifted_contact_or_task_failure_is_failure",
    ):
        if acceptance.get(field) is not True:
            raise TriggeredRescueError("acceptance.%s must remain true" % field)

    chosen = REGISTERED_CASE_IDS[0] if case_id is None else case_id
    if chosen not in cases:
        raise TriggeredRescueError("requested case is not registered")
    chosen_case = cases[chosen]
    return {
        "case": dict(chosen_case),
        "case_ids": tuple(cases),
        "protected_robot_body_names": PROTECTED_BODY_NAMES,
        "nominal_trigger_threshold_m2_per_s": nominal_threshold,
        "material_correction_threshold_rad_s": correction_threshold,
        "qp_max_iterations": qp_max_iterations,
        "horizon_action_count": int(chosen_case["historical_action_count"]),
        "expected_filter_updates_per_suffix_action": 5,
        "expected_physics_substeps_per_suffix_action": 25,
        "acceptance": numeric_thresholds,
    }


def evaluate_direct_qdot_trigger_candidate(
    *,
    source_action_index: int,
    physical_boundary: int,
    nominal_qdot: Any,
    safe_qdot: Any,
    h: Any,
    gradients: Any,
    jacobians: Any,
    alpha_per_s: float,
    nominal_threshold_m2_per_s: float,
    material_correction_threshold_rad_s: float,
    sample_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate one solved direct-qdot preview without changing simulation.

    The hard QP solve occurs in the runtime.  This function independently
    reconstructs the trigger from its bounded nominal and safe commands.  A
    candidate is actionable only when the nominal CBF residual is unsafe and
    the solved command materially differs from nominal.
    """

    action = _integer(source_action_index, "source action index")
    boundary = _integer(physical_boundary, "physical boundary")
    if boundary != action * 25:
        raise TriggeredRescueError("trigger scan must occur at a 20 Hz action boundary")

    def finite_vector(raw: Any, length: int, label: str) -> Sequence[float]:
        sequence = _sequence(raw, label)
        if len(sequence) != length:
            raise TriggeredRescueError("%s length differs" % label)
        return tuple(_number(value, label) for value in sequence)

    nominal = finite_vector(nominal_qdot, 7, "nominal qdot")
    safe = finite_vector(safe_qdot, 7, "safe qdot")
    value_rows = _sequence(h, "h")
    sample_count = len(value_rows)
    if sample_count <= 0 or len(sample_records) != sample_count:
        raise TriggeredRescueError("direct-qdot trigger sample count differs")
    values = tuple(_number(value, "h") for value in value_rows)
    gradient_rows = _sequence(gradients, "gradients")
    jacobian_rows = _sequence(jacobians, "jacobians")
    if len(gradient_rows) != sample_count or len(jacobian_rows) != sample_count:
        raise TriggeredRescueError("direct-qdot trigger arrays differ")
    gradient_array = tuple(
        finite_vector(row, 3, "gradient") for row in gradient_rows
    )
    jacobian_array = []
    for row in jacobian_rows:
        coordinate_rows = _sequence(row, "point Jacobian")
        if len(coordinate_rows) != 3:
            raise TriggeredRescueError("point Jacobian shape differs")
        jacobian_array.append(
            tuple(
                finite_vector(coordinate, 7, "point Jacobian coordinate")
                for coordinate in coordinate_rows
            )
        )
    tolerance = 64.0 * 2.220446049250313e-16
    if any(abs(value) > 0.5 + tolerance for value in nominal):
        raise TriggeredRescueError("nominal preview is outside physical qdot bounds")
    if any(abs(value) > 0.5 + tolerance for value in safe):
        raise TriggeredRescueError("safe preview is outside physical qdot bounds")
    if any(value <= 0.0 for value in values):
        raise TriggeredRescueError("trigger preview requires strict positive h")
    alpha = _number(alpha_per_s, "CBF alpha")
    nominal_threshold = _number(
        nominal_threshold_m2_per_s, "nominal trigger threshold"
    )
    correction_threshold = _number(
        material_correction_threshold_rad_s, "material correction threshold"
    )
    if alpha <= 0.0 or nominal_threshold >= 0.0 or correction_threshold <= 0.0:
        raise TriggeredRescueError("trigger numeric contract differs")

    cbf_rows = []
    for gradient, jacobian in zip(gradient_array, jacobian_array):
        cbf_rows.append(
            tuple(
                sum(gradient[axis] * jacobian[axis][joint] for axis in range(3))
                for joint in range(7)
            )
        )
    nominal_residuals = tuple(
        sum(row[joint] * nominal[joint] for joint in range(7))
        + alpha * value
        for row, value in zip(cbf_rows, values)
    )
    safe_residuals = tuple(
        sum(row[joint] * safe[joint] for joint in range(7))
        + alpha * value
        for row, value in zip(cbf_rows, values)
    )
    argmin = min(range(sample_count), key=nominal_residuals.__getitem__)
    correction = math.sqrt(
        sum((safe[joint] - nominal[joint]) ** 2 for joint in range(7))
    )
    nominal_minimum = nominal_residuals[argmin]
    safe_minimum = min(safe_residuals)
    sample = _mapping(sample_records[argmin], "argmin sample")
    sample_identity = {
        key: sample.get(key)
        for key in ("sample_id", "body_id", "body_name", "geom_id", "geom_name")
    }
    if any(value is None for value in sample_identity.values()):
        raise TriggeredRescueError("argmin sample identity is incomplete")
    return {
        "source_action_index": action,
        "physical_boundary": boundary,
        "decision_variable": "bounded_physical_arm_joint_velocity_rad_s",
        "nominal_minimum_cbf_residual_m2_per_s": nominal_minimum,
        "safe_minimum_cbf_residual_m2_per_s": safe_minimum,
        "nominal_threshold_m2_per_s": nominal_threshold,
        "filter_correction_l2_rad_s": correction,
        "material_correction_threshold_rad_s": correction_threshold,
        "argmin_protected_sample": sample_identity,
        "triggered": bool(
            nominal_minimum <= nominal_threshold
            and correction >= correction_threshold
        ),
    }


def select_first_actionable_trigger(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Latch the first actionable row from an ordered online scan."""

    previous_boundary = -1
    first: Optional[Dict[str, Any]] = None
    for index, raw in enumerate(_sequence(rows, "trigger rows")):
        row = _mapping(raw, "trigger rows[%d]" % index)
        boundary = _integer(row.get("physical_boundary"), "trigger boundary")
        if boundary <= previous_boundary:
            raise TriggeredRescueError("trigger rows must be strictly ordered")
        previous_boundary = boundary
        if not isinstance(row.get("triggered"), bool):
            raise TriggeredRescueError("triggered flag must be Boolean")
        if row["triggered"] and first is None:
            first = dict(row)
    return first


def classify_triggered_rescue(
    metrics_value: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    case_id: str,
) -> Dict[str, Any]:
    """Classify the causal paired rescue without weakening negative outcomes."""

    derived = validate_triggered_rescue_protocol(protocol, case_id=case_id)
    metrics = _mapping(metrics_value, "metrics")
    if metrics.get("schema_version") != METRICS_SCHEMA:
        raise TriggeredRescueError("triggered-rescue metrics schema differs")

    required_booleans = (
        "trigger_scan_complete",
        "trigger_found",
        "native_prefix_exact",
        "exact_paired_trigger_state",
        "full_recorded_episode_complete",
        "baseline_exposure_complete",
        "psf_exposure_complete",
        "baseline_link56_contact_present",
        "psf_any_robot_selected_obstacle_contact_present",
        "psf_shifted_link56_external_contact_present",
        "psf_contact_terminated",
        "material_correction_before_baseline_contact",
        "psf_paper_car_avoided",
        "psf_useful_post_correction_motion",
        "psf_task_success_after_correction",
        "psf_terminal_task_success",
        "psf_method_stop_or_stall",
        "all_psf_field_queries_valid",
        "all_psf_qps_solved_and_postchecked",
    )
    flags: Dict[str, bool] = {}
    for field in required_booleans:
        value = metrics.get(field)
        if not isinstance(value, bool):
            raise TriggeredRescueError("metrics.%s must be Boolean" % field)
        flags[field] = value

    apparatus_valid = bool(
        flags["trigger_scan_complete"]
        and flags["native_prefix_exact"]
        and (
            not flags["trigger_found"]
            or (
                flags["exact_paired_trigger_state"]
                and flags["baseline_exposure_complete"]
                and (
                    flags["psf_method_stop_or_stall"]
                    or (
                        flags["all_psf_field_queries_valid"]
                        and flags["all_psf_qps_solved_and_postchecked"]
                    )
                )
            )
        )
    )
    if not apparatus_valid:
        classification = "INCONCLUSIVE_APPARATUS"
    elif not flags["trigger_found"]:
        classification = "NO_ACTIONABLE_DIRECT_QDOT_WARNING"
    elif (
        flags["psf_contact_terminated"]
        or flags["psf_any_robot_selected_obstacle_contact_present"]
        or flags["psf_shifted_link56_external_contact_present"]
    ):
        classification = "CONTACT_REMAINS_OR_SHIFTED"
    elif flags["psf_method_stop_or_stall"]:
        classification = "STOP_OR_METHOD_FAILURE"
    elif not flags["baseline_link56_contact_present"]:
        classification = "BASELINE_CONTACT_NOT_REPRODUCED"
    elif not (
        flags["full_recorded_episode_complete"]
        and flags["psf_exposure_complete"]
    ):
        classification = "INCONCLUSIVE_APPARATUS"
    elif not (
        flags["psf_task_success_after_correction"]
        and flags["psf_terminal_task_success"]
    ):
        classification = "CONTACT_PREVENTED_TASK_FAILED"
    elif not (
        flags["material_correction_before_baseline_contact"]
        and flags["psf_paper_car_avoided"]
        and flags["psf_useful_post_correction_motion"]
    ):
        classification = "STOP_OR_METHOD_FAILURE"
    else:
        classification = "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"

    return {
        "schema_version": CLASSIFICATION_SCHEMA,
        "classification": classification,
        "feasible": classification == "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        "apparatus_valid": apparatus_valid,
        "case_id": case_id,
        "protected_robot_body_names": list(
            derived["protected_robot_body_names"]
        ),
        "fixed_link_selection_across_cases": True,
        "stop_only_is_failure": True,
        "claim_scope": (
            "two_case_offline_event_triggered_direct_joint_velocity_"
            "controller_feasibility_not_closed_loop_policy_not_population_"
            "safety_not_formal_invariance"
        ),
    }


__all__ = [
    "CLASSIFICATIONS",
    "CLASSIFICATION_SCHEMA",
    "METRICS_SCHEMA",
    "PROTECTED_BODY_NAMES",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "REGISTERED_CASE_IDS",
    "RESULT_SCHEMA",
    "TriggeredRescueError",
    "classify_triggered_rescue",
    "evaluate_direct_qdot_trigger_candidate",
    "select_first_actionable_trigger",
    "validate_triggered_rescue_protocol",
]
