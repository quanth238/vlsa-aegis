"""Contracts for the first-warning calibrated complete-episode diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_first_warning_calibrated_repulsion.v1"
RESULT_SCHEMA = "vlsa_distal_first_warning_calibrated_repulsion_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "base_episode_config",
        "risk_curve_run", "risk_curve_summary_payload_sha256", "case_ids",
        "requested_radius_by_case", "selection_rule", "reuse_rule",
        "comparison", "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("calibrated-repulsion config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("calibrated-repulsion schema differs")
    if value["protocol_id"] != "vlsa-distal-first-warning-calibrated-repulsion-v1":
        raise ValueError("calibrated-repulsion protocol differs")
    expected_cases = [
        "vlsa-t1-goal-ii-t2-e42",
        "vlsa-t1-goal-ii-t3-e42",
        "vlsa-t1-goal-ii-t3-e44",
    ]
    if value["case_ids"] != expected_cases:
        raise ValueError("calibrated-repulsion cases differ")
    if [float(item) for item in value["requested_radius_by_case"]] != [1.25, 0.75, 1.75]:
        raise ValueError("calibrated-repulsion radii differ")
    if value["reuse_rule"] != (
        "freeze_selected_case_radius_for_every_later_warning_in_same_complete_episode"
    ):
        raise ValueError("calibrated-repulsion reuse rule differs")
    if value["comparison"] != {
        "controller": "validated_fixed_radius_2_complete_episode_run_40303_40304",
        "summary_payload_sha256": "421adf1150c74fa7f194f14c54d6a0fa3c8b037efa834ca63a1ba139be871ea5",
        "intervention_count_by_case": [5, 39, 31],
        "total_requested_correction_l2_action_by_case": [10.0, 78.0, 62.0],
        "native_task_success_by_case": [True, False, False],
        "raw_L5_L7_contact_pass_by_case": [True, True, True],
        "paper_car_pass_by_case": [True, True, True],
    }:
        raise ValueError("calibrated-repulsion comparator differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def aggregate(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(rows) != int(config["gate"]["required_case_count"]):
        raise ValueError("calibrated-repulsion aggregate count differs")
    expected = list(config["case_ids"])
    if [str(row["case_id"]) for row in rows] != expected:
        raise ValueError("calibrated-repulsion aggregate order differs")
    contact = [bool(row["raw_L5_L7_contact_pass"]) for row in rows]
    car = [bool(row["paper_car_pass"]) for row in rows]
    task = [bool(row["native_task_success"]) for row in rows]
    no_timeout = [not bool(row["timeout"]) for row in rows]
    requested = [
        sum(float(item["proposal"]["requested_correction_l2_action"])
            for item in row["interventions"])
        for row in rows
    ]
    fixed_radius_two = [
        float(item)
        for item in config["comparison"][
            "total_requested_correction_l2_action_by_case"
        ]
    ]
    smaller = [left < right for left, right in zip(requested, fixed_radius_two)]
    primary = [
        a and b and c and d and e
        for a, b, c, d, e in zip(contact, car, task, no_timeout, smaller)
    ]
    return {
        "case_count": len(rows),
        "raw_L5_L7_contact_pass_count": sum(contact),
        "paper_car_pass_count": sum(car),
        "native_task_success_count": sum(task),
        "no_timeout_count": sum(no_timeout),
        "total_intervention_count": sum(int(row["intervention_count"]) for row in rows),
        "total_requested_correction_l2_action": sum(requested),
        "per_case_requested_correction_l2_action": requested,
        "per_case_validated_radius_two_requested_correction_l2_action": fixed_radius_two,
        "strictly_smaller_requested_correction_count": sum(smaller),
        "safe_task_success_count": sum(primary),
        "strict_gate_pass": all(primary),
    }
