"""Contracts for the bounded cross-case repulsion generalization pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_repulsion_generalization_pilot.v1"
CASE_SCHEMA = "vlsa_distal_repulsion_generalization_case.v1"
TASK_VALID_CONFIG_SCHEMA = "vlsa_distal_repulsion_task_valid_case.v1"
TASK_VALID_CASE_SCHEMA = "vlsa_distal_repulsion_task_valid_selection.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "action_space",
        "analytical_field",
        "claim_scope",
        "comparators",
        "exact_search",
        "field_estimation",
        "gate",
        "internal_verification",
        "population_gate",
        "protected_geometry",
        "protocol_id",
        "schema_version",
        "source_population",
        "state_protocol",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("repulsion-generalization config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("repulsion-generalization schema differs")
    if value["protocol_id"] != "vlsa-distal-repulsion-generalization-pilot-v1":
        raise ValueError("repulsion-generalization protocol differs")
    if value["comparators"]["arms"] != [
        "raw_aegis",
        "short_horizon_analytical_repulsion",
        "smooth_long_horizon_counterfactual_repulsion",
        "exact_derivative_free_search",
    ]:
        raise ValueError("repulsion-generalization comparator set differs")
    if value["action_space"] != {
        "action_limit": 1.0,
        "corrected_dimensions": [0, 1, 2],
        "corrected_horizon": 5,
        "line_search_fractions": [1.0, 0.5, 0.25],
        "maximum_iterations": 10,
        "maximum_total_correction_l2_action": 1.0,
        "maximum_total_path_length_action": 1.0,
        "trust_radius_action": 0.1,
    }:
        raise ValueError("repulsion-generalization action space differs")
    if value["state_protocol"] != {
        "evaluation_horizon_actions": 20,
        "intervention_lead_actions": 5,
        "policy_actions": "immutable_table1_executed_actions",
        "policy_query_disabled": True,
        "replay_prefix_from_settled_state": True,
    }:
        raise ValueError("repulsion-generalization state protocol differs")
    if value["gate"]["internal_substep_clearance_buffers_m"] != [0.0, 0.001]:
        raise ValueError("repulsion-generalization clearance gates differ")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def load_cases(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 3:
        raise ValueError("repulsion-generalization pilot must contain three cases")
    if hashlib.sha256(raw).hexdigest() != (
        "2f63d1513d8ffb333c2d73b83feef5523351aed809966ab379c81f2bbc427b16"
    ):
        raise ValueError("repulsion-generalization selection manifest differs")
    case_ids = [row.get("case_id") for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("repulsion-generalization cases are not unique")
    required = {
        "active_obstacle_name",
        "archived_result_file_sha256",
        "archived_result_payload_sha256",
        "archived_result_relative_path",
        "case_id",
        "collision_first_step",
        "episode_index",
        "evaluation_last_step",
        "first_relevant_contact_step",
        "intervention_step",
        "policy_noise_schedule_sha256",
        "protected_contact_bodies",
        "schema_version",
        "selection_reason",
        "source_manifest_row_sha256",
        "suite",
        "table1_case_ordinal",
        "task_level_group_id",
    }
    horizon = int(config["state_protocol"]["evaluation_horizon_actions"])
    lead = int(config["state_protocol"]["intervention_lead_actions"])
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("repulsion-generalization case keys differ")
        if row["schema_version"] != CASE_SCHEMA:
            raise ValueError("repulsion-generalization case schema differs")
        if int(row["intervention_step"]) != int(row["first_relevant_contact_step"]) - lead:
            raise ValueError("repulsion-generalization intervention lead differs")
        if int(row["evaluation_last_step"]) != int(row["intervention_step"]) + horizon - 1:
            raise ValueError("repulsion-generalization evaluation horizon differs")
        if not row["selection_reason"].startswith("previously_unused_"):
            raise ValueError("repulsion-generalization selection reason differs")
    output = json.loads(_canonical(rows).decode("utf-8"))
    for row in output:
        row["selection_manifest_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return output


def load_task_valid_config(path: Path) -> dict[str, Any]:
    """Load the focused task-valid replacement experiment without mutating v1."""

    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "action_space",
        "analytical_field",
        "claim_scope",
        "comparators",
        "eligibility_gate",
        "exact_search",
        "field_estimation",
        "gate",
        "internal_verification",
        "population_gate",
        "protected_geometry",
        "protocol_id",
        "schema_version",
        "source_population",
        "state_protocol",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("task-valid repulsion config keys differ")
    if value["schema_version"] != TASK_VALID_CONFIG_SCHEMA:
        raise ValueError("task-valid repulsion schema differs")
    if value["protocol_id"] != "vlsa-distal-repulsion-task-valid-e38-v1":
        raise ValueError("task-valid repulsion protocol differs")
    reference = load_config(
        Path(path).with_name("vlsa_distal_repulsion_generalization_pilot.v1.json")
    )
    for key in (
        "action_space",
        "analytical_field",
        "comparators",
        "exact_search",
        "field_estimation",
        "gate",
        "internal_verification",
        "protected_geometry",
        "state_protocol",
    ):
        if value[key] != {
            name: item
            for name, item in reference[key].items()
            if not str(name).endswith("_sha256")
        }:
            raise ValueError("task-valid repulsion algorithm differs: %s" % key)
    if value["eligibility_gate"] != {
        "aegis_native_task_success": True,
        "baseline_native_task_success": True,
        "initial_protected_contact_count": 0,
        "maximum_task_object_eef_distance_m": 0.02,
        "minimum_gripper_command": 0.5,
        "minimum_initial_proxy_clearance_m": 0.0,
    }:
        raise ValueError("task-valid repulsion eligibility gate differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def load_task_valid_cases(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("task-valid repulsion must contain exactly one case")
    required = {
        "active_obstacle_name",
        "aegis_result_file_sha256",
        "aegis_result_payload_sha256",
        "aegis_result_relative_path",
        "baseline_result_file_sha256",
        "baseline_result_payload_sha256",
        "baseline_result_relative_path",
        "case_id",
        "collision_first_step",
        "episode_index",
        "evaluation_last_step",
        "first_relevant_contact_step",
        "intervention_step",
        "policy_noise_schedule_sha256",
        "protected_contact_bodies",
        "schema_version",
        "selection_reason",
        "source_manifest_row_sha256",
        "suite",
        "table1_case_ordinal",
        "task_level_group_id",
        "task_object_name",
    }
    row = rows[0]
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("task-valid repulsion case keys differ")
    if row["schema_version"] != TASK_VALID_CASE_SCHEMA:
        raise ValueError("task-valid repulsion case schema differs")
    if row["case_id"] != "vlsa-t1-goal-ii-t3-e38":
        raise ValueError("task-valid repulsion case differs")
    lead = int(config["state_protocol"]["intervention_lead_actions"])
    horizon = int(config["state_protocol"]["evaluation_horizon_actions"])
    if int(row["intervention_step"]) != int(row["first_relevant_contact_step"]) - lead:
        raise ValueError("task-valid repulsion intervention lead differs")
    if int(row["evaluation_last_step"]) != int(row["intervention_step"]) + horizon - 1:
        raise ValueError("task-valid repulsion horizon differs")
    output = json.loads(_canonical(rows).decode("utf-8"))
    output[0]["selection_manifest_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return output


def load_experiment_contract(config_path: Path, selection_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header = json.loads(Path(config_path).read_text())
    if header.get("schema_version") == TASK_VALID_CONFIG_SCHEMA:
        config = load_task_valid_config(config_path)
        return config, load_task_valid_cases(selection_path, config)
    config = load_config(config_path)
    return config, load_cases(selection_path, config)


def acceptance(record: Mapping[str, Any], config: Mapping[str, Any], buffer_m: float) -> bool:
    return bool(
        float(record["minimum_clearance_m"]) >= float(buffer_m)
        and len(record["protected_contacts"])
        == int(config["gate"]["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(config["gate"]["paper_car_threshold_m"])
    )


def aggregate_results(results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"raw_aegis", "analytical", "smooth", "exact_search"}
    if len(results) != 3 or len({item["case_id"] for item in results}) != 3:
        raise ValueError("repulsion-generalization aggregate case set differs")
    for result in results:
        if set(result["arms"]) != expected:
            raise ValueError("repulsion-generalization aggregate arms differ")
    smooth_gain = sum(float(item["arms"]["smooth"]["clearance_gain_m"]) > 0.0 for item in results)
    smooth_safe_0 = sum(bool(item["arms"]["smooth"]["gates"]["buffer_0mm"]) for item in results)
    smooth_safe_1 = sum(bool(item["arms"]["smooth"]["gates"]["buffer_1mm"]) for item in results)
    fixed_safe_0 = sum(bool(item["arms"]["analytical"]["gates"]["buffer_0mm"]) for item in results)
    oracle_safe_0 = sum(bool(item["arms"]["exact_search"]["gates"]["buffer_0mm"]) for item in results)
    minimum = int(config["population_gate"]["mechanism_minimum_case_count"])
    return {
        "case_count": len(results),
        "smooth_positive_clearance_gain_count": smooth_gain,
        "smooth_safe_0mm_count": smooth_safe_0,
        "smooth_safe_1mm_count": smooth_safe_1,
        "analytical_safe_0mm_count": fixed_safe_0,
        "exact_search_safe_0mm_count": oracle_safe_0,
        "mechanism_gate_pass": bool(smooth_gain == 3 and smooth_safe_0 >= minimum),
        "strict_generalization_gate_pass": bool(smooth_safe_1 == 3),
        "smooth_beats_or_matches_fixed_safe_rate": bool(smooth_safe_0 >= fixed_safe_0),
    }
