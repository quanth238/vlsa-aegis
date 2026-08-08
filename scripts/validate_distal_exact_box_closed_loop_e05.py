#!/usr/bin/env python3
"""Validate the preregistered distal-only exact-box closed-loop E05 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


VALIDATION_SCHEMA = "vlsa_distal_exact_box_closed_loop_e05_validation.v1"
RESULT_SCHEMAS = {
    "vlsa_distal_exact_box_closed_loop_e05_result.v1",
    "vlsa_distal_exact_box_closed_loop_e05_result.v2",
    "vlsa_distal_exact_box_closed_loop_e05_result.v3",
}
CASE_ID = "vlsa-t1-goal-ii-t0-e05"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("candidate is not a JSON object")
    return value


def validate(candidate_path: Path) -> dict[str, Any]:
    result = _load(candidate_path)
    payload = dict(result)
    recorded_payload_sha256 = payload.pop("result_payload_sha256", None)
    checks = {
        "schema_and_case": bool(
            result.get("schema_version") in RESULT_SCHEMAS
            and result.get("case_id") == CASE_ID
            and result.get("scientific_result") is True
        ),
        "source_clean_and_h100": bool(
            result.get("source", {}).get("dirty") is False
            and "H100"
            in str(result.get("allocation", {}).get("device", {}).get("name", ""))
            and isinstance(
                result.get("allocation", {}).get("device", {}).get("uuid"), str
            )
        ),
        "payload_hash": bool(
            isinstance(recorded_payload_sha256, str)
            and hashlib.sha256(_canonical(payload)).hexdigest()
            == recorded_payload_sha256
        ),
        "table1_read_only": bool(
            result.get("archived_table1", {}).get("read_only") is True
            and result.get("archived_table1", {}).get("file_sha256")
            == "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
        ),
        "exact_obstacle_boxes": bool(
            result.get("exact_obstacle_geometry", {}).get("exact_box_count") == 15
            and result.get("exact_obstacle_geometry", {}).get("geometric_inflation")
            == 0.0
            and result.get("exact_obstacle_geometry", {}).get(
                "all_enclosure_certificates_verified"
            )
            is True
        ),
    }
    actions = result.get("actions")
    action_records = actions if isinstance(actions, list) else []
    target = [0.008] * 7 + [
        -1.0
        if result.get("schema_version")
        == "vlsa_distal_exact_box_closed_loop_e05_result.v3"
        else 0.0
    ]
    tolerance = 1.0e-6
    checks["nonempty_closed_loop"] = bool(
        action_records
        and result.get("policy_query_count", 0) > 0
        and result.get("action_count") == len(action_records)
    )
    checks["distal_only_target_vector"] = bool(
        action_records
        and all(
            item.get("filter", {}).get("clearance_target_m") == target
            and item.get("filter", {}).get("activation_clearance_target_m")
            == target
            for item in action_records
        )
    )
    if result.get("schema_version") == "vlsa_distal_exact_box_closed_loop_e05_result.v2":
        checks["original_aegis_ee_geometry_retained"] = bool(
            result.get("config", {}).get("protected_geometry", {}).get(
                "end_effector_target"
            )
            == "zero_margin_frozen_aegis_mvee_clearance_after_released_aegis_qp"
            and result.get("config", {}).get("obstacle_geometry", {}).get(
                "end_effector_obstacle_source"
            )
            == "frozen_released_aegis_perception_mvee"
        )
    if result.get("schema_version") == "vlsa_distal_exact_box_closed_loop_e05_result.v3":
        checks["released_aegis_ee_qp_unaugmented"] = bool(
            result.get("config", {}).get("protected_geometry", {}).get(
                "end_effector_target"
            )
            == "released_aegis_nominal_qp_then_raw_simulator_contact_and_displacement_veto"
            and result.get("config", {}).get("obstacle_geometry", {}).get(
                "end_effector_constraint_mode"
            )
            == "unchanged_released_aegis_qp_without_second_discrete_EE_target"
        )
    checks["accepted_exact_substep_clearance"] = bool(
        action_records
        and all(
            item.get("filter", {}).get("verification", {}).get("accepted") is True
            and item.get("filter", {}).get("verification", {}).get(
                "main_env_post_step_checked"
            )
            is True
            and len(
                item.get("filter", {}).get("verification", {}).get(
                    "accepted_next_clearance_m", []
                )
            )
            == 8
            and all(
                math.isfinite(float(clearance))
                and float(clearance) >= target[index] - tolerance
                for index, clearance in enumerate(
                    item.get("filter", {}).get("verification", {}).get(
                        "accepted_next_clearance_m", []
                    )
                )
            )
            for item in action_records
        )
    )
    raw = result.get("raw_simulation_evidence", {})
    outcome_consistent = bool(
        result.get("primary_problem_solved")
        is bool(
            result.get("failure") is None
            and result.get("filter_summary", {}).get("status") == "complete"
            and result.get("filter_summary", {}).get(
                "all_executed_transitions_match_clone"
            )
            is True
            and raw.get("protected_link_contact_pass") is True
            and raw.get("paper_car_pass") is True
            and raw.get("native_task_success") is True
        )
    )
    checks["outcome_consistent"] = outcome_consistent
    passed = all(checks.values())
    return {
        "schema_version": VALIDATION_SCHEMA,
        "candidate": str(candidate_path),
        "candidate_file_sha256": _file_sha256(candidate_path),
        "candidate_payload_sha256": recorded_payload_sha256,
        "checks": checks,
        "passed": passed,
        "primary_problem_solved": result.get("primary_problem_solved"),
        "scientific_status": result.get("status"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(args.candidate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_bytes(_canonical(receipt) + b"\n")
    temporary.replace(args.output)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
