#!/usr/bin/env python3
"""Independently validate the terminal-buffer repair result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def _require(value: bool, message: str) -> None:
    if not value:
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


def _candidate_safe(item: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    record = item["record"]
    _require(record["sample_count"] == 51, "candidate sample count differs")
    _require(record["substep_counts"] == [25, 25], "candidate substeps differ")
    _require(len(record["row_minimum_clearance_m"]) == 7, "candidate rows differ")
    return bool(
        float(record["minimum_clearance_m"]) >= float(gate["internal_substep_clearance_buffer_m"])
        and len(record["protected_contacts"]) == int(gate["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"]) <= float(gate["paper_car_threshold_m"])
        and (not bool(gate["require_native_goal_after_candidate"]) or bool(item["goal"]["all_satisfied"]))
    )


def validate(result_path: Path, expected_commit: str, validator_commit: str) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    _require(result["schema_version"] == "vlsa_distal_terminal_buffer_repair_e05_result.v1", "schema differs")
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(claimed == _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()), "self-hash differs")
    _require(result["source"]["commit"] == expected_commit, "producer commit differs")
    _require(result["scientific_result"] is True and "H100" in result["allocation"]["device"]["name"], "producer evidence differs")
    candidates = result["candidates"]
    _require(result["candidate_count"] == len(candidates), "candidate count differs")
    gate = result["config"]["gate"]
    safe = []
    for item in candidates:
        recomputed = _candidate_safe(item, gate)
        _require(item["verification_gate"] == recomputed, "candidate gate differs")
        if recomputed:
            safe.append(item)
    _require(result["safe_candidate_count"] == len(safe), "safe support differs")
    expected_selected = (
        None
        if not safe
        else min(safe, key=lambda item: (float(item["applied_correction_l2_action"]), -float(item["record"]["minimum_clearance_m"])))
    )
    _require(result["selected"] == expected_selected, "selected candidate differs")
    task_success = result["goal_progress"]["summary"]["first_all_satisfied_step"] == 205
    solved = bool(expected_selected is not None and task_success)
    _require(result["primary_problem_solved"] == solved, "primary gate differs")
    if solved:
        _require(float(result["executed_clone_state_max_abs_error"]) <= 1.0e-10, "clone fidelity differs")
    video = Path(result["video"]["path"])
    frame = Path(result["final_jpg"]["path"])
    _require(video.is_file() and frame.is_file(), "media missing")
    _require(_sha256(video.read_bytes()) == result["video"]["file_sha256"], "video hash differs")
    _require(_sha256(frame.read_bytes()) == result["final_jpg"]["file_sha256"], "frame hash differs")
    validation = {
        "schema_version": "vlsa_distal_terminal_buffer_repair_e05_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "producer_commit": expected_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _sha256(result_path.read_bytes()),
        "result_payload_sha256": claimed,
        "primary_problem_solved": solved,
        "safe_candidate_count": len(safe),
        "selected_correction_l2_action": None if expected_selected is None else expected_selected["applied_correction_l2_action"],
        "selected_minimum_clearance_m": None if expected_selected is None else expected_selected["record"]["minimum_clearance_m"],
        "native_task_success_step": result["goal_progress"]["summary"]["first_all_satisfied_step"],
        "video_file_sha256": result["video"]["file_sha256"],
    }
    validation["validation_payload_sha256"] = _sha256(json.dumps(validation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
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
