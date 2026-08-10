#!/usr/bin/env python3
"""Independently validate the conditional learned E05 closed loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.region_aware_mlp import (
    CLOSED_LOOP_RESULT_SCHEMA, CLOSED_LOOP_VALIDATION_SCHEMA, load_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _load(args.result.resolve())
    config = load_config(args.config.resolve())
    _require(
        result.get("schema_version") == CLOSED_LOOP_RESULT_SCHEMA
        and result.get("scientific_result") is True
        and result.get("case_id") == config["closed_loop"]["case_id"]
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"]
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and result.get("video", {}).get("file_sha256")
        == _file_sha256(Path(result["video"]["path"]))
        and result.get("final_jpg", {}).get("file_sha256")
        == _file_sha256(Path(result["final_jpg"]["path"])),
        "region-aware closed-loop identity differs",
    )
    actions = [item for item in result["actions"] if item.get("executed") is not False]
    measurements = [item["filter"]["exact_measurement"] for item in actions]
    _require(
        len(actions) == len(measurements)
        and all(item["clone_execution_state_match"] for item in actions),
        "region-aware clone/execution evidence differs",
    )
    minimum_distal = (
        None if not measurements
        else min(min(item["minimum_distal_margin_m"]) for item in measurements)
    )
    exact_safe_count = sum(bool(item["true_distal_safe"]) for item in measurements)
    first_robot = next((
        int(item["step"]) for item in actions if item["robot_contact_events"]
    ), None)
    first_protected = next((
        int(item["step"]) for item in actions if item["protected_link_contact_events"]
    ), None)
    first_car = next((
        int(item["step"]) for item in actions
        if float(item["active_obstacle_l1_displacement_m"]) > 0.001
    ), None)
    maximum_displacement = max(
        (float(item["active_obstacle_l1_displacement_m"]) for item in actions),
        default=0.0,
    )
    raw = result["raw_simulation_evidence"]
    _require(
        raw["first_robot_contact_step"] == first_robot
        and raw["first_protected_link_contact_step"] == first_protected
        and raw["first_paper_car_step"] == first_car
        and abs(float(raw["maximum_active_obstacle_l1_displacement_m"])
                - maximum_displacement) <= 1.0e-15
        and raw["minimum_exact_two_step_distal_margin_m"] == minimum_distal
        and int(raw["exact_safe_measurement_count"]) == exact_safe_count,
        "region-aware raw simulation summary differs",
    )
    native_success = raw["native_task_success"] is True
    primary = bool(
        result["failure"] is None and first_protected is None and first_car is None
        and native_success and minimum_distal is not None and minimum_distal >= 0.0
        and exact_safe_count == len(actions)
    )
    _require(result["primary_problem_solved"] is primary,
             "region-aware primary decision differs")
    output = {
        "schema_version": CLOSED_LOOP_VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": False, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "action_count": len(actions),
        "minimum_exact_two_step_distal_margin_m": minimum_distal,
        "exact_safe_measurement_count": exact_safe_count,
        "first_protected_link_contact_step": first_protected,
        "first_paper_car_step": first_car,
        "native_task_success": native_success,
        "primary_problem_solved": primary,
        "video_file_sha256": result["video"]["file_sha256"],
        "final_jpg_file_sha256": result["final_jpg"]["file_sha256"],
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
