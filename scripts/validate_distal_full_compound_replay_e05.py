#!/usr/bin/env python3
"""Independently validate the complete registered E05 compound replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(result_path: Path, expected_commit: str, validator_commit: str) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    _require(
        result["schema_version"] == "vlsa_distal_full_compound_replay_e05_result.v1",
        "result schema differs",
    )
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    computed = _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    _require(claimed == computed, "result self-hash differs")
    _require(result["source"]["commit"] == expected_commit, "producer commit differs")
    _require(result["scientific_result"] is True, "result is not scientific")
    _require("H100" in result["allocation"]["device"]["name"], "producer was not H100")
    _require(result["case_id"] == "vlsa-t1-goal-ii-t0-e05", "case differs")
    _require(result["action_count"] == len(result["actions"]), "action count differs")
    trace = result["trace"]
    rows = trace["clearance_trace_m"]
    _require(len(rows) == 1 + 25 * result["action_count"], "sample count differs")
    _require(trace["sample_count"] == len(rows), "recorded sample count differs")
    _require(trace["substep_counts"] == [25] * result["action_count"], "substeps differ")
    _require(all(len(row) == 7 for row in rows), "clearance row count differs")
    exact_min = min(min(row) for row in rows)
    row_mins = [min(row[index] for row in rows) for index in range(7)]
    _require(abs(exact_min - float(trace["minimum_clearance_m"])) <= 1.0e-15, "minimum differs")
    _require(
        max(abs(value - float(claimed_value)) for value, claimed_value in zip(row_mins, trace["row_minimum_clearance_m"]))
        <= 1.0e-15,
        "row minima differ",
    )
    _require(
        float(trace["maximum_boundary_equivalence_error_m"]) <= 1.0e-12,
        "boundary equivalence differs",
    )
    gate = result["config"]["gate"]
    task_success = result["goal_progress"]["summary"]["first_all_satisfied_step"] is not None
    maximum_displacement = max(trace["active_obstacle_l1_displacement_trace_m"])
    passed = bool(
        exact_min >= float(gate["internal_substep_clearance_buffer_m"])
        and len(trace["protected_contacts"]) == int(gate["protected_raw_contact_count"])
        and maximum_displacement <= float(gate["paper_car_threshold_m"])
        and (not bool(gate["require_native_task_success"]) or task_success)
    )
    _require(result["primary_problem_solved"] == passed, "primary gate differs")
    expected_interpretation = (
        "complete_compound_positive_control_safe_task_success"
        if passed
        else "complete_compound_positive_control_no_go"
    )
    _require(result["interpretation"] == expected_interpretation, "interpretation differs")
    evidence = result["raw_simulation_evidence"]
    _require(evidence["native_task_success"] == task_success, "task success differs")
    _require(
        abs(float(evidence["maximum_active_obstacle_l1_displacement_m"]) - maximum_displacement)
        <= 1.0e-15,
        "CAR maximum differs",
    )
    video_path = Path(result["video"]["path"])
    final_path = Path(result["final_jpg"]["path"])
    _require(video_path.is_file() and final_path.is_file(), "media artifact missing")
    _require(_sha256(video_path.read_bytes()) == result["video"]["file_sha256"], "video hash differs")
    _require(_sha256(final_path.read_bytes()) == result["final_jpg"]["file_sha256"], "frame hash differs")
    validation = {
        "schema_version": "vlsa_distal_full_compound_replay_e05_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "producer_commit": expected_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _sha256(result_path.read_bytes()),
        "result_payload_sha256": claimed,
        "validation_payload_sha256": None,
        "primary_problem_solved": passed,
        "interpretation": expected_interpretation,
        "minimum_internal_clearance_m": exact_min,
        "protected_contact_count": len(trace["protected_contacts"]),
        "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
        "native_task_success_step": evidence["native_task_success_step"],
        "video_file_sha256": result["video"]["file_sha256"],
    }
    validation.pop("validation_payload_sha256")
    validation["validation_payload_sha256"] = _sha256(
        json.dumps(validation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(args.result.resolve(), args.expected_commit, args.validator_commit)
    _atomic_write(args.output.resolve(), value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
