#!/usr/bin/env python3
"""Independently validate the paired E05 compound-scale feedback audit."""

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


def _validate_arm(path: Path, expected_commit: str, expected_scale: float) -> dict[str, Any]:
    result = json.loads(path.read_text())
    _require(
        result["schema_version"] == "vlsa_distal_scale_feedback_audit_e05_result.v1",
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
    _require(
        float(result["registered_input"]["correction_scale"]) == float(expected_scale),
        "scale differs",
    )
    _require(
        float(result["registered_input"]["prelude_trace_max_abs_error_m"]) <= 1.0e-12,
        "registered prelude differs",
    )
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
        max(
            abs(value - float(claimed_value))
            for value, claimed_value in zip(row_mins, trace["row_minimum_clearance_m"])
        )
        <= 1.0e-15,
        "row minima differ",
    )
    _require(
        float(trace["maximum_boundary_equivalence_error_m"]) <= 1.0e-12,
        "boundary equivalence differs",
    )
    _require(
        result["action_205_task_threshold_audit"]["native_goal"]["all_satisfied"] is False,
        "action-205 task threshold differs",
    )
    xy = float(result["action_205_task_threshold_audit"]["horizontal_center_distance_m"])
    margin = float(result["action_205_task_threshold_audit"]["horizontal_threshold_margin_m"])
    _require(abs((0.03 - xy) - margin) <= 1.0e-15 and margin < 0.0, "task threshold geometry differs")
    queries = result["policy_queries"]
    _require(result["policy_query_count"] == len(queries), "policy query count differs")
    _require(
        [item["query_index"] for item in queries]
        == list(range(41, 41 + len(queries))),
        "policy query schedule differs",
    )
    _require(
        [item["step"] for item in queries]
        == list(range(206, 206 + 5 * len(queries), 5)),
        "policy execution schedule differs",
    )
    for window in result["windows"]:
        lookahead = window["lookahead"]
        _require(lookahead["substep_counts"] == [25] * 5, "lookahead substeps differ")
        future_minimum = min(min(row) for row in lookahead["clearance_trace_m"][1:])
        _require(
            abs(future_minimum - float(window["future_minimum_clearance_m"])) <= 1.0e-15,
            "lookahead future minimum differs",
        )
        _require(
            window["buffer_warning"]
            == (future_minimum < float(result["config"]["gate"]["internal_substep_clearance_buffer_m"])),
            "lookahead buffer warning differs",
        )
    live_tolerance = float(
        result["config"]["internal_verification"][
            "live_execution_clearance_equivalence_tolerance"
        ]
    )
    maximum_live_error = max(
        (
            float(action.get("lookahead_clearance_max_abs_error_m", 0.0))
            for action in result["actions"]
        ),
        default=0.0,
    )
    _require(maximum_live_error <= live_tolerance, "live execution equivalence differs")
    task_success = result["goal_progress"]["summary"]["first_all_satisfied_step"] is not None
    first_contact = min((item["step"] for item in trace["protected_contacts"]), default=None)
    maximum_displacement = max(trace["active_obstacle_l1_displacement_trace_m"])
    first_car = next(
        (
            identity["step"]
            for identity, value in zip(
                trace["sample_identities"],
                trace["active_obstacle_l1_displacement_trace_m"],
            )
            if value > float(result["config"]["gate"]["paper_car_threshold_m"])
        ),
        None,
    )
    evidence = result["raw_simulation_evidence"]
    _require(evidence["native_task_success"] == task_success, "task success differs")
    _require(evidence["first_protected_link_contact_step"] == first_contact, "contact step differs")
    _require(evidence["first_paper_car_step"] == first_car, "CAR step differs")
    _require(
        abs(float(evidence["maximum_active_obstacle_l1_displacement_m"]) - maximum_displacement)
        <= 1.0e-15,
        "CAR maximum differs",
    )
    collision_free_success = bool(task_success and first_contact is None and first_car is None)
    strict_pass = bool(
        collision_free_success
        and exact_min >= float(result["config"]["gate"]["internal_substep_clearance_buffer_m"])
        and result["failure"] is None
    )
    _require(result["collision_free_task_success"] == collision_free_success, "collision-free task gate differs")
    _require(result["primary_problem_solved"] == strict_pass, "strict gate differs")
    video_path = Path(result["video"]["path"])
    final_path = Path(result["final_jpg"]["path"])
    _require(video_path.is_file() and final_path.is_file(), "media artifact missing")
    _require(_sha256(video_path.read_bytes()) == result["video"]["file_sha256"], "video hash differs")
    _require(_sha256(final_path.read_bytes()) == result["final_jpg"]["file_sha256"], "frame hash differs")
    return {
        "path": str(path),
        "result_file_sha256": _sha256(path.read_bytes()),
        "result_payload_sha256": claimed,
        "correction_scale": expected_scale,
        "action_205_horizontal_threshold_margin_m": margin,
        "minimum_internal_clearance_m": exact_min,
        "maximum_live_execution_clearance_error_m": maximum_live_error,
        "protected_contact_count": len(trace["protected_contacts"]),
        "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
        "native_task_success_step": evidence["native_task_success_step"],
        "first_buffer_warning_step": result["lookahead_summary"]["first_buffer_warning_step"],
        "first_protected_contact_warning_step": result["lookahead_summary"]["first_protected_contact_warning_step"],
        "collision_free_task_success": collision_free_success,
        "primary_problem_solved": strict_pass,
        "interpretation": result["interpretation"],
        "video_file_sha256": result["video"]["file_sha256"],
    }


def validate(
    result_root: Path,
    expected_commit: str,
    validator_commit: str,
) -> dict[str, Any]:
    arms = {
        "scale_1p01": _validate_arm(
            result_root / "scale_1p01" / "result.json", expected_commit, 1.01
        ),
        "scale_1p02": _validate_arm(
            result_root / "scale_1p02" / "result.json", expected_commit, 1.02
        ),
    }
    raw = {
        label: json.loads((result_root / label / "result.json").read_text())
        for label in arms
    }
    left_queries = raw["scale_1p01"]["policy_queries"]
    right_queries = raw["scale_1p02"]["policy_queries"]
    paired_count = min(len(left_queries), len(right_queries))
    _require(paired_count > 0, "paired feedback queries missing")
    for left, right in zip(left_queries[:paired_count], right_queries[:paired_count]):
        _require(left["query_index"] == right["query_index"], "paired query index differs")
        _require(left["rng_seed"] == right["rng_seed"], "paired policy seed differs")
        _require(left["step"] == right["step"], "paired query step differs")
    validation = {
        "schema_version": "vlsa_distal_scale_feedback_audit_e05_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "producer_commit": expected_commit,
        "validator_commit": validator_commit,
        "paired_policy_query_count": paired_count,
        "paired_policy_noise_seed_gate": True,
        "arms": arms,
    }
    validation["validation_payload_sha256"] = _sha256(
        json.dumps(validation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        args.result_root.resolve(), args.expected_commit, args.validator_commit
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
