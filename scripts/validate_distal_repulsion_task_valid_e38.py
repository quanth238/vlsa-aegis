#!/usr/bin/env python3
"""Independent validation of the focused task-valid E38 repulsion result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import _load, _require, _sha256


SCHEMA = "vlsa_distal_repulsion_task_valid_validation.v1"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(result_path: Path, expected_commit: str) -> dict[str, Any]:
    result = _load(result_path)
    _require(
        result.get("schema_version")
        == "vlsa_distal_repulsion_generalization_case_result.v1",
        "task-valid result schema differs",
    )
    _require(result.get("status") == "complete", "task-valid result is incomplete")
    _require(result.get("case_id") == "vlsa-t1-goal-ii-t3-e38", "task-valid case differs")
    _require(
        result.get("source", {}).get("commit") == expected_commit,
        "task-valid source differs",
    )
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(
        _sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        == claimed,
        "task-valid self-hash differs",
    )
    eligibility = result["eligibility"]
    _require(eligibility["aegis_native_task_success"], "AEGIS task validity differs")
    _require(eligibility["baseline_native_task_success"], "baseline task validity differs")
    _require(
        eligibility["initial_protected_contact_count"] == 0,
        "task-valid initial contact differs",
    )
    _require(
        eligibility["initial_proxy_clearance_m"] >= 0.0,
        "task-valid initial proxy margin differs",
    )
    _require(
        eligibility["task_object_eef_distance_m"] <= 0.02,
        "task-valid object retention differs",
    )
    _require(eligibility["gripper_command"] >= 0.5, "task-valid gripper differs")
    raw = result["arms"]["raw_aegis"]
    _require(result["gate"]["raw_reproduces_collision"], "raw collision did not reproduce")
    _require(not raw["gates"]["buffer_0mm"], "raw E38 is not unsafe")
    _require(
        all(count == 25 for count in raw["final"]["record"]["substep_counts"]),
        "task-valid internal substep count differs",
    )
    output = {
        "schema_version": SCHEMA,
        "status": "passed",
        "scientific_result": True,
        "producer_source_commit": expected_commit,
        "case_id": result["case_id"],
        "result_payload_sha256": claimed,
        "eligibility": eligibility,
        "arms": {
            name: {
                "minimum_clearance_m": arm["final"]["record"]["minimum_clearance_m"],
                "clearance_gain_m": arm["clearance_gain_m"],
                "correction_l2_action": arm["correction_l2_action"],
                "gates": arm["gates"],
            }
            for name, arm in result["arms"].items()
        },
        "interpretation": result["interpretation"],
    }
    output["validation_payload_sha256"] = _sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(args.result.resolve(), args.expected_commit)
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
