#!/usr/bin/env python3
"""Independent validator for the full receding persistent-route E05 oracle."""

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


def _record_gate(record: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    _require(int(record["sample_count"]) == 126, "internal sample count differs")
    _require(record["substep_counts"] == [25] * 5, "substep counts differ")
    _require(float(record["maximum_boundary_equivalence_error_m"]) <= 1.0e-12, "boundary equivalence differs")
    _require(len(record["row_minimum_clearance_m"]) == 7, "row count differs")
    return bool(
        float(record["minimum_clearance_m"])
        >= float(gate["internal_substep_clearance_buffer_m"])
        and not record["protected_contacts"]
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
    )


def validate(
    *, result_path: Path, expected_commit: str, validator_commit: str
) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    _require(
        result["schema_version"] == "vlsa_distal_receding_route_oracle_e05_result.v1",
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
    gate = result["config"]["gate"]
    windows = result["windows"]
    corrected = 0
    for window in windows:
        nominal_safe = _record_gate(window["nominal"], gate)
        _require(bool(window["nominal_safe"]) == nominal_safe, "nominal gate differs")
        if window["selected"] is not None:
            _require(_record_gate(window["selected"], gate), "selected window is unsafe")
            _require(window["executed_prefix_actions"] == 1, "execution prefix differs")
            _require(float(window["clone_state_max_abs_error"]) <= 1.0e-10, "clone fidelity differs")
            corrected += int(
                not window["selected_source"].startswith("fresh_exact_safe_nominal_window")
            )
        for item in window["persistent_routes"]:
            safe = _record_gate(item["record"], gate)
            _require(bool(item["verification_gate"]) == safe, "route gate differs")
        field = window["field_refinement"]
        if field is not None:
            exact_field_safe = _record_gate(field["record"], gate)
            if field["verification_gate"]:
                _require(exact_field_safe, "verified field is physically unsafe")
                _require(
                    float(field["total_correction_l2_action"])
                    <= float(result["config"]["derivative_free"]["maximum_correction_l2_action"])
                    + 1.0e-10,
                    "verified field exceeds correction bound",
                )
        derivative = window["derivative_free"]
        if derivative is not None:
            _record_gate(derivative["best"]["record"], gate)
    _require(int(result["receding_summary"]["window_count"]) == len(windows), "window count differs")
    _require(int(result["receding_summary"]["corrected_window_count"]) == corrected, "corrected count differs")
    evidence = result["raw_simulation_evidence"]
    recomputed_solved = bool(
        result["failure"] is None
        and evidence["first_robot_contact_step"] is None
        and evidence["first_protected_link_contact_step"] is None
        and evidence["first_paper_car_step"] is None
        and evidence["native_task_success"] is True
    )
    _require(bool(result["primary_problem_solved"]) == recomputed_solved, "primary gate differs")
    if recomputed_solved:
        interpretation = "receding_persistent_route_oracle_safe_task_success"
    elif result["failure"] is not None:
        interpretation = "receding_persistent_route_oracle_deadlock"
    else:
        interpretation = "receding_persistent_route_oracle_safe_but_task_failed"
    _require(result["interpretation"] == interpretation, "interpretation differs")
    video_path = Path(result["video"]["path"])
    final_path = Path(result["final_jpg"]["path"])
    _require(video_path.is_file() and final_path.is_file(), "media artifact missing")
    _require(_sha256(video_path.read_bytes()) == result["video"]["file_sha256"], "video hash differs")
    _require(_sha256(final_path.read_bytes()) == result["final_jpg"]["file_sha256"], "final frame hash differs")
    validation = {
        "schema_version": "vlsa_distal_receding_route_oracle_e05_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "producer_commit": expected_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _sha256(result_path.read_bytes()),
        "result_payload_sha256": claimed,
        "primary_problem_solved": recomputed_solved,
        "interpretation": interpretation,
        "window_count": len(windows),
        "corrected_window_count": corrected,
        "native_task_success_step": evidence["native_task_success_step"],
        "first_protected_link_contact_step": evidence["first_protected_link_contact_step"],
        "first_paper_car_step": evidence["first_paper_car_step"],
        "maximum_active_obstacle_l1_displacement_m": evidence[
            "maximum_active_obstacle_l1_displacement_m"
        ],
        "video_file_sha256": result["video"]["file_sha256"],
    }
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
    value = validate(
        result_path=args.result.resolve(),
        expected_commit=args.expected_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
