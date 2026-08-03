#!/usr/bin/env python3
"""Independent, claim-focused consumer for triggered direct-qdot rescue."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.poisson_fullbody.contracts import (  # noqa: E402
    load_hashed_json,
    publish_hashed_json,
    sha256_file,
)
from main.poisson_fullbody.shadow_replay import (  # noqa: E402
    load_historical_action_replay,
)
from main.poisson_fullbody.triggered_rescue import (  # noqa: E402
    PROTOCOL_ID,
    RESULT_SCHEMA,
    validate_triggered_rescue_protocol,
)


RECEIPT_SCHEMA = "vlsa_poisson_triggered_rescue_independent_validation.v1"
INNER_DT_S = 0.01
PHYSICS_DT_S = 0.002


class _Audit:
    def __init__(self) -> None:
        self.discrepancies: List[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition and message not in self.discrepancies:
            self.discrepancies.append(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _mapping(audit: _Audit, value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        audit.check(False, "%s_not_object" % label)
        return {}
    return value


def _sequence(audit: _Audit, value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        audit.check(False, "%s_not_array" % label)
        return ()
    return value


def _number(audit: _Audit, value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        audit.check(False, "%s_not_finite" % label)
        return 0.0
    return float(value)


def _vector(
    audit: _Audit, value: Any, length: int, label: str
) -> Tuple[float, ...]:
    row = _sequence(audit, value, label)
    audit.check(len(row) == length, "%s_length_differs" % label)
    if len(row) != length:
        return tuple(0.0 for _ in range(length))
    return tuple(
        _number(audit, item, "%s_%d" % (label, index))
        for index, item in enumerate(row)
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) ** 2 for item in value))


def _close(left: Any, right: Any, tolerance: float = 1.0e-9) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _plain_json(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("JSON input is missing or symlinked: %s" % path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON input must contain one object: %s" % path)
    return value


def _trigger(
    audit: _Audit,
    result: Mapping[str, Any],
    derived: Mapping[str, Any],
    horizon: int,
) -> Dict[str, Any]:
    scan = _mapping(audit, result.get("trigger_scan"), "trigger_scan")
    rows = _sequence(audit, scan.get("rows"), "trigger_rows")
    threshold = float(derived["nominal_trigger_threshold_m2_per_s"])
    correction_threshold = float(derived["material_correction_threshold_rad_s"])
    audit.check(scan.get("complete") is True, "trigger_scan_not_complete")
    audit.check(
        scan.get("decision_variable")
        == "bounded_physical_arm_joint_velocity_rad_s",
        "trigger_decision_variable_differs",
    )
    for key in (
        "fixed_trigger_action_index_used",
        "case_identity_used_by_trigger",
        "historical_contact_action_used_by_trigger",
    ):
        audit.check(key not in scan or scan.get(key) is False, "trigger_%s" % key)

    first: Optional[int] = None
    for index, raw in enumerate(rows):
        row = _mapping(audit, raw, "trigger_row_%d" % index)
        nominal = _vector(
            audit,
            row.get("bounded_nominal_qdot_rad_s"),
            7,
            "trigger_row_%d_nominal" % index,
        )
        residual = _number(
            audit,
            row.get("nominal_minimum_cbf_residual_m2_per_s"),
            "trigger_row_%d_residual" % index,
        )
        attempted = residual <= threshold
        correction = (
            _number(
                audit,
                row.get("filter_correction_l2_rad_s"),
                "trigger_row_%d_correction" % index,
            )
            if attempted
            else 0.0
        )
        reconstructed = bool(attempted and correction >= correction_threshold)
        audit.check(
            row.get("source_action_index") == index
            and row.get("physical_boundary") == index * 25,
            "trigger_row_%d_cadence_differs" % index,
        )
        audit.check(
            all(abs(value) <= 0.5 + 1.0e-12 for value in nominal),
            "trigger_row_%d_nominal_unbounded" % index,
        )
        audit.check(
            row.get("field_queries_all_valid_positive") is True
            and isinstance(row.get("field_query_count"), int)
            and row.get("field_query_count") > 0,
            "trigger_row_%d_field_invalid" % index,
        )
        audit.check(
            row.get("hard_qp_attempted") is attempted,
            "trigger_row_%d_qp_attempt_differs" % index,
        )
        if attempted:
            audit.check(
                row.get("hard_qp_valid") is True,
                "trigger_row_%d_qp_invalid" % index,
            )
        audit.check(
            row.get("triggered") is reconstructed,
            "trigger_row_%d_flag_differs" % index,
        )
        if reconstructed and first is None:
            first = index

    selected = scan.get("first_actionable_trigger")
    found = first is not None
    audit.check((selected is not None) is found, "selected_trigger_presence_differs")
    if not found:
        audit.check(len(rows) == horizon, "no_trigger_scan_horizon_differs")
        return {"found": False, "action": None, "boundary": None}
    selected_map = _mapping(audit, selected, "selected_trigger")
    audit.check(
        selected_map.get("source_action_index") == first
        and selected_map.get("physical_boundary") == first * 25
        and selected_map.get("triggered") is True,
        "selected_trigger_not_first",
    )
    audit.check(len(rows) == first + 1, "trigger_scan_continued_after_latch")
    return {"found": True, "action": first, "boundary": first * 25}


def _prefix_exact(
    audit: _Audit,
    result: Mapping[str, Any],
    replay: Any,
    trigger: Mapping[str, Any],
) -> bool:
    prefix = _mapping(audit, result.get("native_osc_prefix"), "native_prefix")
    rows = _sequence(audit, prefix.get("rows"), "native_prefix_rows")
    expected_count = int(trigger["action"]) if trigger["found"] else len(replay.steps)
    exact = bool(
        prefix.get("completed_action_count") == expected_count
        and len(rows) == expected_count
    )
    for index, raw in enumerate(rows):
        row = _mapping(audit, raw, "prefix_row_%d" % index)
        expected = replay.steps[index]
        exact = bool(
            exact
            and row.get("source_action_index") == index
            and row.get("simulator_state_sha256")
            == expected.simulator_state_sha256
            and tuple(row.get("native_goal_values", ())) == expected.goal_values
            and _is_sha256(row.get("returned_observation_sha256"))
        )
    audit.check(exact, "native_prefix_not_exact")
    if trigger["found"]:
        action = int(trigger["action"])
        expected_state = (
            replay.settled_simulator_state_sha256
            if action == 0
            else replay.steps[action - 1].simulator_state_sha256
        )
        selected = _mapping(
            audit,
            _mapping(audit, result.get("trigger_scan"), "trigger_scan").get(
                "first_actionable_trigger"
            ),
            "selected_trigger",
        )
        audit.check(
            prefix.get("trigger_state_sha256") == expected_state
            == selected.get("simulator_state_sha256"),
            "trigger_state_not_prefix_boundary",
        )
    return exact


def _pair_exact(
    audit: _Audit, adapter: Mapping[str, Any], psf: Mapping[str, Any]
) -> bool:
    left = _mapping(audit, adapter.get("restore"), "adapter_restore")
    right = _mapping(audit, psf.get("restore"), "psf_restore")
    fields = (
        "model_topology_sha256",
        "physical_model_sha256",
        "compiled_mjb_sha256",
        "official_integration_state_sha256",
        "target_official_integration_state_sha256",
        "settled_state_sha256",
        "target_state_sha256",
        "controller",
        "controller_software_state",
        "pid_memory_reset",
    )
    exact = bool(
        all(left.get(key) == right.get(key) for key in fields)
        and adapter.get("start_official_raw_bytes_sha256")
        == psf.get("start_official_raw_bytes_sha256")
        and adapter.get("fresh_adapter") is True
        and psf.get("fresh_adapter") is True
    )
    audit.check(exact, "paired_trigger_state_differs")
    return exact


def _contacts(
    audit: _Audit,
    arm: Mapping[str, Any],
    label: str,
    start_action: int,
) -> Dict[str, Any]:
    value = _mapping(
        audit, arm.get("registered_forbidden_contact"), "%s_contacts" % label
    )
    rows = _sequence(
        audit, value.get("rollout_contact_records"), "%s_contact_rows" % label
    )
    selected = False
    selected_link_boundaries: List[int] = []
    shifted = False
    for index, raw in enumerate(rows):
        row = _mapping(audit, raw, "%s_contact_%d" % (label, index))
        unhashed = {key: item for key, item in row.items() if key != "record_sha256"}
        source = row.get("source_action_index")
        substep = row.get("physics_substep_index")
        audit.check(
            _sha256(unhashed) == row.get("record_sha256"),
            "%s_contact_%d_hash_differs" % (label, index),
        )
        audit.check(
            isinstance(source, int)
            and not isinstance(source, bool)
            and isinstance(substep, int)
            and not isinstance(substep, bool)
            and source >= start_action
            and 0 <= substep < 25
            and row.get("physical_boundary") == source * 25 + substep,
            "%s_contact_%d_cadence_differs" % (label, index),
        )
        is_selected = row.get("any_robot_selected_obstacle_contact") is True
        is_link = row.get("link56_external_nonrobot_contact") is True
        selected = bool(selected or is_selected)
        shifted = bool(
            shifted
            or row.get("link56_nonselected_external_contact") is True
        )
        if is_selected and is_link:
            selected_link_boundaries.append(int(row["physical_boundary"]))
    selected_link = bool(selected_link_boundaries)
    audit.check(
        value.get("settled_forbidden_contact") is False
        and value.get("any_robot_selected_obstacle_contact") is selected
        and value.get("any_link56_nonselected_external_contact") is shifted
        and value.get("observed_physics_substeps")
        == arm.get("physics_trace_row_count"),
        "%s_contact_summary_differs" % label,
    )
    literal = _mapping(audit, arm.get("literal_contact"), "%s_literal" % label)
    audit.check(
        literal.get("link56_present") is selected_link
        and literal.get("any_robot_selected_obstacle_present") is selected,
        "%s_literal_contact_differs" % label,
    )
    first = min(selected_link_boundaries) if selected_link_boundaries else None
    audit.check(
        literal.get("first_link56_physical_boundary") == first,
        "%s_first_link_contact_differs" % label,
    )
    return {"selected": selected, "selected_link": selected_link, "shifted": shifted, "first": first}


def _paper_car(
    audit: _Audit, arm: Mapping[str, Any], label: str, expected_count: int
) -> Dict[str, Any]:
    value = _mapping(audit, arm.get("paper_car"), "%s_car" % label)
    initial = _vector(
        audit, value.get("initial_active_obstacle_position_m"), 3, "%s_car_initial" % label
    )
    rows = _sequence(audit, value.get("endpoint_ledger"), "%s_car_rows" % label)
    maximum = 0.0
    for index, raw in enumerate(rows):
        row = _mapping(audit, raw, "%s_car_%d" % (label, index))
        current = _vector(
            audit, row.get("active_obstacle_position_m"), 3, "%s_car_position" % label
        )
        displacement = sum(abs(current[axis] - initial[axis]) for axis in range(3))
        maximum = max(maximum, displacement)
        audit.check(
            _close(displacement, row.get("l1_displacement_from_settled_m"), 1.0e-12),
            "%s_car_%d_displacement_differs" % (label, index),
        )
    threshold = _number(audit, value.get("paper_collision_threshold_m"), "car_threshold")
    complete = len(rows) == expected_count
    audit.check(
        value.get("enabled") is True
        and value.get("expected_endpoint_count") == expected_count
        and value.get("endpoint_ledger_complete") is complete
        and _close(value.get("maximum_active_obstacle_l1_displacement_m"), maximum, 1.0e-12)
        and value.get("paper_collision_avoidance") is (maximum <= threshold),
        "%s_car_summary_differs" % label,
    )
    return {"complete": complete, "maximum": maximum, "avoided": complete and maximum <= threshold}


def _arm(
    audit: _Audit,
    arm: Mapping[str, Any],
    label: str,
    replay: Any,
    start_action: int,
    expected_actions: int,
    derived: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    is_psf = label == "psf"
    commands = _sequence(audit, arm.get("command_trace"), "%s_commands" % label)
    physics = _sequence(audit, arm.get("physics_trace"), "%s_physics" % label)
    trigger_threshold = float(derived["nominal_trigger_threshold_m2_per_s"])
    correction_threshold = float(derived["material_correction_threshold_rad_s"])
    qp_cfg = _mapping(audit, runtime.get("qp"), "runtime_qp")
    residual_tolerance = float(qp_cfg.get("postcheck_cbf_tolerance", 0.0))
    first_material: Optional[int] = None
    for index, raw in enumerate(commands):
        row = _mapping(audit, raw, "%s_command_%d" % (label, index))
        source = row.get("source_action_index")
        inner = row.get("inner_control_index")
        nominal = _vector(audit, row.get("nominal_qdot_rad_s"), 7, "nominal_qdot")
        executed = _vector(audit, row.get("executed_qdot_rad_s"), 7, "executed_qdot")
        source_action = _vector(audit, row.get("source_action"), 7, "source_action")
        correction = _norm(
            tuple(executed[joint] - nominal[joint] for joint in range(7))
        )
        audit.check(
            isinstance(source, int)
            and not isinstance(source, bool)
            and isinstance(inner, int)
            and not isinstance(inner, bool)
            and start_action <= source < len(replay.actions)
            and 0 <= inner < 5
            and row.get("physical_boundary") == source * 25 + inner * 5,
            "%s_command_%d_cadence_differs" % (label, index),
        )
        if isinstance(source, int) and 0 <= source < len(replay.actions):
            audit.check(source_action == replay.actions[source], "%s_source_action_differs" % label)
        audit.check(
            all(abs(value) <= 0.5 + 1.0e-10 for value in executed)
            and _close(correction, row.get("correction_l2_rad_s")),
            "%s_command_%d_execution_differs" % (label, index),
        )
        if is_psf:
            residual = _number(
                audit,
                row.get("safe_cbf_residual_minimum_m2_per_s"),
                "safe_cbf_residual",
            )
            audit.check(
                isinstance(row.get("qp"), Mapping)
                and residual >= -residual_tolerance - 1.0e-12,
                "psf_command_%d_qp_postcheck_failed" % index,
            )
            nominal_residual = row.get("nominal_cbf_residual_minimum_m2_per_s")
            if (
                isinstance(nominal_residual, (int, float))
                and not isinstance(nominal_residual, bool)
                and float(nominal_residual) <= trigger_threshold
                and correction >= correction_threshold
                and first_material is None
            ):
                first_material = int(row["physical_boundary"])
        else:
            audit.check(
                row.get("qp") is None and correction <= 1.0e-12,
                "adapter_command_%d_filtered" % index,
            )
    for index, raw in enumerate(physics):
        row = _mapping(audit, raw, "%s_physics_%d" % (label, index))
        source = row.get("source_action_index")
        inner = row.get("inner_control_index")
        substep = row.get("physics_substep_index")
        audit.check(
            isinstance(source, int)
            and isinstance(inner, int)
            and isinstance(substep, int)
            and row.get("post_state_physical_boundary")
            == source * 25 + inner * 5 + substep + 1,
            "%s_physics_%d_cadence_differs" % (label, index),
        )
    expected_updates = expected_actions * 5
    expected_substeps = expected_actions * 25
    task = _mapping(audit, arm.get("task"), "%s_task" % label)
    complete = bool(
        len(commands) == expected_updates
        and len(physics) == expected_substeps
        and arm.get("exposure_complete") is True
        and task.get("completed_source_action_count") == expected_actions
    )
    contacts = _contacts(audit, arm, label, start_action)
    car = _paper_car(audit, arm, label, expected_actions)
    return {
        "commands": commands,
        "physics": physics,
        "complete": complete,
        "first_material": first_material,
        "contacts": contacts,
        "car": car,
        "task": task,
    }


def _post_motion(
    audit: _Audit,
    commands: Sequence[Any],
    physics: Sequence[Any],
    first_boundary: Optional[int],
) -> Dict[str, float]:
    if first_boundary is None:
        return {
            "filter_correction_integral_rad": 0.0,
            "executed_command_integral_rad": 0.0,
            "measured_joint_motion_integral_rad": 0.0,
            "cartesian_path_length_m": 0.0,
            "zero_command_fraction": 1.0,
        }
    selected_commands = [
        row for row in commands
        if isinstance(row, Mapping) and row.get("physical_boundary", -1) >= first_boundary
    ]
    keys = {
        (row.get("source_action_index"), row.get("inner_control_index"))
        for row in selected_commands
    }
    selected_physics = [
        row for row in physics
        if isinstance(row, Mapping)
        and (row.get("source_action_index"), row.get("inner_control_index")) in keys
    ]
    command_norms = [
        _norm(_vector(audit, row.get("executed_qdot_rad_s"), 7, "motion_command"))
        for row in selected_commands
    ]
    cartesian_path = 0.0
    for command in selected_commands:
        key = (command.get("source_action_index"), command.get("inner_control_index"))
        rows = sorted(
            (row for row in selected_physics
             if (row.get("source_action_index"), row.get("inner_control_index")) == key
             and row.get("eef_position_world_m") is not None),
            key=lambda row: row.get("physics_substep_index", -1),
        )
        if len(rows) != 5:
            continue
        previous = _vector(audit, command.get("eef_position_before_update_world_m"), 3, "eef_before")
        for row in rows:
            current = _vector(audit, row.get("eef_position_world_m"), 3, "eef_after")
            cartesian_path += _norm(tuple(current[i] - previous[i] for i in range(3)))
            previous = current
    return {
        "filter_correction_integral_rad": sum(
            float(row["correction_l2_rad_s"]) * INNER_DT_S
            for row in selected_commands
        ),
        "executed_command_integral_rad": sum(command_norms) * INNER_DT_S,
        "measured_joint_motion_integral_rad": sum(
            _norm(_vector(audit, row.get("measured_qvel_rad_s"), 7, "measured_qvel"))
            * PHYSICS_DT_S
            for row in selected_physics
        ),
        "cartesian_path_length_m": cartesian_path,
        "zero_command_fraction": (
            1.0 if not command_norms
            else sum(value <= 1.0e-8 for value in command_norms) / len(command_norms)
        ),
    }


def _classify(metrics: Mapping[str, bool]) -> Tuple[str, bool]:
    apparatus = bool(
        metrics["trigger_scan_complete"]
        and metrics["native_prefix_exact"]
        and (
            not metrics["trigger_found"]
            or (
                metrics["exact_paired_trigger_state"]
                and metrics["baseline_exposure_complete"]
                and (
                    metrics["psf_method_stop_or_stall"]
                    or (
                        metrics["all_psf_field_queries_valid"]
                        and metrics["all_psf_qps_solved_and_postchecked"]
                    )
                )
            )
        )
    )
    if not apparatus:
        return "INCONCLUSIVE_APPARATUS", False
    if not metrics["trigger_found"]:
        return "NO_ACTIONABLE_DIRECT_QDOT_WARNING", True
    if (
        metrics["psf_contact_terminated"]
        or metrics["psf_any_robot_selected_obstacle_contact_present"]
        or metrics["psf_shifted_link56_external_contact_present"]
    ):
        return "CONTACT_REMAINS_OR_SHIFTED", True
    if metrics["psf_method_stop_or_stall"]:
        return "STOP_OR_METHOD_FAILURE", True
    if not metrics["baseline_link56_contact_present"]:
        return "BASELINE_CONTACT_NOT_REPRODUCED", True
    if not metrics["full_recorded_episode_complete"]:
        return "INCONCLUSIVE_APPARATUS", False
    if not (
        metrics["psf_task_success_after_correction"]
        and metrics["psf_terminal_task_success"]
    ):
        return "CONTACT_PREVENTED_TASK_FAILED", True
    if not (
        metrics["material_correction_before_baseline_contact"]
        and metrics["psf_paper_car_avoided"]
        and metrics["psf_useful_post_correction_motion"]
    ):
        return "STOP_OR_METHOD_FAILURE", True
    return "SAFE_TASK_SUCCESS_USEFUL_CORRECTION", True


def validate_artifact(
    result_path: Path,
    protocol_path: Path,
    runtime_path: Path,
    historical_path: Path,
    *,
    expected_code_commit: Optional[str] = None,
    expected_run_id: Optional[str] = None,
    expected_slurm_job_id: Optional[str] = None,
    expected_host: Optional[str] = None,
) -> Dict[str, Any]:
    audit = _Audit()
    result = load_hashed_json(result_path)
    protocol = _plain_json(protocol_path)
    runtime = _plain_json(runtime_path)
    case_id = str(result.get("case_id"))
    derived = validate_triggered_rescue_protocol(protocol, case_id=case_id)
    replay = load_historical_action_replay(
        historical_path,
        expected_case_id=case_id,
        expected_arm="pi05_plus_aegis_translational",
    )
    provenance = _mapping(audit, result.get("provenance"), "provenance")
    source = _mapping(audit, provenance.get("source"), "source")
    allocation = _mapping(audit, provenance.get("allocation"), "allocation")
    contract = derived["case"]
    audit.check(result.get("schema_version") == RESULT_SCHEMA, "result_schema_differs")
    audit.check(result.get("status") == "complete", "producer_not_complete")
    audit.check(result.get("protocol_id") == PROTOCOL_ID, "result_protocol_differs")
    audit.check(result.get("partial_output_interpreted") is False, "partial_output_interpreted")
    audit.check(provenance.get("online_policy_query_count") == 0, "online_policy_query_count_differs")
    audit.check("H100" in str(allocation.get("gpu_name", "")), "producer_not_h100")
    audit.check(
        sha256_file(protocol_path) == provenance.get("protocol_file_sha256")
        and sha256_file(runtime_path) == protocol["runtime_binding"]["file_sha256"],
        "protocol_or_runtime_hash_differs",
    )
    audit.check(
        sha256_file(historical_path) == contract["historical_result_file_sha256"]
        and replay.result_payload_sha256 == contract["historical_result_payload_sha256"]
        and replay.executed_sequence_sha256
        == contract["historical_executed_action_sequence_sha256"],
        "historical_binding_differs",
    )
    if expected_code_commit is not None:
        audit.check(source.get("commit") == expected_code_commit, "producer_commit_differs")
    if expected_run_id is not None:
        audit.check(
            result.get("run_id") == expected_run_id
            and result_path.parent.name == expected_run_id,
            "run_id_differs",
        )
    if expected_slurm_job_id is not None:
        audit.check(allocation.get("slurm_job_id") == expected_slurm_job_id, "producer_job_differs")
    if expected_host is not None:
        audit.check(allocation.get("host") == expected_host, "producer_host_differs")

    trigger = _trigger(audit, result, derived, len(replay.actions))
    prefix = _prefix_exact(audit, result, replay, trigger)
    reported = _mapping(audit, result.get("metrics"), "metrics")
    motion: Dict[str, Any] = {}
    arm_summary: Dict[str, Any] = {}
    if not trigger["found"]:
        metrics = {
            key: False for key in (
                "trigger_found", "exact_paired_trigger_state",
                "full_recorded_episode_complete", "baseline_exposure_complete",
                "psf_exposure_complete", "baseline_link56_contact_present",
                "psf_any_robot_selected_obstacle_contact_present",
                "psf_shifted_link56_external_contact_present", "psf_contact_terminated",
                "material_correction_before_baseline_contact", "psf_paper_car_avoided",
                "psf_useful_post_correction_motion", "psf_task_success_after_correction",
                "psf_terminal_task_success", "psf_method_stop_or_stall",
                "all_psf_field_queries_valid", "all_psf_qps_solved_and_postchecked",
            )
        }
        metrics.update({"trigger_scan_complete": True, "native_prefix_exact": prefix})
    else:
        start = int(trigger["action"])
        suffix_actions = [list(action) for action in replay.actions[start:]]
        suffix = _mapping(audit, result.get("paired_suffix"), "paired_suffix")
        suffix_record = {
            "source_action_index_start": start,
            "source_action_index_end_inclusive": len(replay.actions) - 1,
            "action_count": len(suffix_actions),
            "actions": suffix_actions,
        }
        audit.check(
            suffix.get("source_action_index_start") == start
            and suffix.get("source_action_index_end_inclusive") == len(replay.actions) - 1
            and suffix.get("action_count") == len(suffix_actions)
            and suffix.get("action_array_sha256") == _sha256(suffix_actions)
            and suffix.get("action_record_sha256") == _sha256(suffix_record)
            and suffix.get("online_policy_queries") == 0,
            "paired_suffix_differs",
        )
        arms = _mapping(audit, result.get("arms"), "arms")
        adapter_raw = _mapping(audit, arms.get("adapter_only"), "adapter")
        psf_raw = _mapping(audit, arms.get("adapter_plus_psf"), "psf")
        pair = _pair_exact(audit, adapter_raw, psf_raw)
        adapter = _arm(
            audit, adapter_raw, "adapter", replay, start, len(suffix_actions), derived, runtime
        )
        psf = _arm(
            audit, psf_raw, "psf", replay, start, len(suffix_actions), derived, runtime
        )
        first = psf["first_material"]
        motion = _post_motion(audit, psf["commands"], psf["physics"], first)
        for key, value in motion.items():
            audit.check(
                _close(_mapping(audit, result.get("post_correction_motion"), "motion").get(key), value),
                "motion_%s_differs" % key,
            )
        acceptance = derived["acceptance"]
        useful = bool(
            first is not None
            and motion["filter_correction_integral_rad"] >= acceptance["minimum_correction_integral_rad"]
            and motion["measured_joint_motion_integral_rad"]
            >= acceptance["minimum_post_correction_measured_joint_motion_integral_rad"]
            and motion["cartesian_path_length_m"]
            >= acceptance["minimum_post_correction_eef_path_length_m"]
            and motion["zero_command_fraction"]
            <= acceptance["maximum_post_correction_zero_command_fraction"]
        )
        task = psf["task"]
        first_success = task.get("first_task_success_source_action_index")
        method_stop = bool(
            psf_raw.get("method_terminated_early")
            or (
                not psf_raw.get("contact_terminated_early")
                and first is not None
                and not useful
            )
        )
        if psf_raw.get("method_terminated_early"):
            audit.check(isinstance(psf_raw.get("method_stop"), Mapping), "method_stop_record_absent")
        qp_valid = bool(
            psf_raw.get("qp_solve_count") == len(psf["commands"])
            and psf_raw.get("qp_postcheck_count") == len(psf["commands"])
            and psf_raw.get("joint_limit_postcheck_count") == len(psf["commands"])
        )
        field_valid = bool(
            psf_raw.get("invalid_field_query_count") == 0
            and psf_raw.get("precontact_field_queries_valid_and_positive") is True
        )
        baseline_boundary = adapter["contacts"]["first"]
        metrics = {
            "trigger_scan_complete": True,
            "trigger_found": True,
            "native_prefix_exact": prefix,
            "exact_paired_trigger_state": pair,
            "full_recorded_episode_complete": bool(adapter["complete"] and psf["complete"]),
            "baseline_exposure_complete": adapter["complete"],
            "psf_exposure_complete": psf["complete"],
            "baseline_link56_contact_present": adapter["contacts"]["selected_link"],
            "psf_any_robot_selected_obstacle_contact_present": psf["contacts"]["selected"],
            "psf_shifted_link56_external_contact_present": psf["contacts"]["shifted"],
            "psf_contact_terminated": bool(psf_raw.get("contact_terminated_early")),
            "material_correction_before_baseline_contact": bool(
                first is not None and baseline_boundary is not None and first < baseline_boundary
            ),
            "psf_paper_car_avoided": bool(
                psf["car"]["avoided"]
                and psf["car"]["maximum"] <= acceptance["paper_car_displacement_threshold_m"]
            ),
            "psf_useful_post_correction_motion": useful,
            "psf_task_success_after_correction": bool(
                first is not None
                and isinstance(first_success, int)
                and not isinstance(first_success, bool)
                and (first_success + 1) * 25 > first
            ),
            "psf_terminal_task_success": bool(task.get("terminal_task_success")),
            "psf_method_stop_or_stall": method_stop,
            "all_psf_field_queries_valid": field_valid,
            "all_psf_qps_solved_and_postchecked": qp_valid,
        }
        arm_summary = {
            "suffix_action_count": len(suffix_actions),
            "adapter_complete": adapter["complete"],
            "psf_complete": psf["complete"],
            "adapter_first_link56_contact_boundary": baseline_boundary,
            "first_material_correction_boundary": first,
            "psf_any_selected_contact": psf["contacts"]["selected"],
            "psf_shifted_link56_contact": psf["contacts"]["shifted"],
            "psf_terminal_task_success": bool(task.get("terminal_task_success")),
        }

    for key, value in metrics.items():
        audit.check(reported.get(key) is value, "reported_metric_%s_differs" % key)
    classification, interpretable = _classify(metrics)
    producer_classification = _mapping(audit, result.get("classification"), "classification")
    audit.check(
        producer_classification.get("classification") == classification
        and producer_classification.get("feasible")
        is (classification == "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"),
        "producer_classification_differs",
    )
    audit.check(interpretable, "scientific_record_inconclusive")
    valid = not audit.discrepancies
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "validated" if valid else "rejected",
        "artifact_valid": valid,
        "case_id": case_id,
        "run_id": result.get("run_id"),
        "result_file_sha256": sha256_file(result_path),
        "result_payload_sha256": result.get("result_payload_sha256"),
        "producer_commit": source.get("commit"),
        "producer_slurm_job_id": allocation.get("slurm_job_id"),
        "independent_classification": classification,
        "feasible": bool(valid and classification == "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"),
        "trigger": trigger,
        "metrics": metrics,
        "post_correction_motion": motion,
        "arm_summary": arm_summary,
        "discrepancies": audit.discrepancies,
        "producer_classifier_imported": False,
        "partial_output_interpreted": False,
        "claim_scope": "offline_targeted_controller_feasibility_not_policy_or_population_safety",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runtime-protocol", required=True, type=Path)
    parser.add_argument("--historical-result", required=True, type=Path)
    parser.add_argument("--expected-code-commit")
    parser.add_argument("--expected-run-id")
    parser.add_argument("--expected-slurm-job-id")
    parser.add_argument("--expected-host")
    arguments = parser.parse_args()
    try:
        summary = validate_artifact(
            arguments.result,
            arguments.protocol,
            arguments.runtime_protocol,
            arguments.historical_result,
            expected_code_commit=arguments.expected_code_commit,
            expected_run_id=arguments.expected_run_id,
            expected_slurm_job_id=arguments.expected_slurm_job_id,
            expected_host=arguments.expected_host,
        )
        publish_hashed_json(arguments.receipt, summary)
    except Exception as error:
        print("triggered-rescue validation failed: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["artifact_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
