#!/usr/bin/env python3
"""Independently validate the E38 repeated-warning repulsion artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import _load, _require, _sha256


SCHEMA = "vlsa_distal_repulsion_task_valid_e38_receding_validation.v1"


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
        == "vlsa_distal_repulsion_task_valid_e38_receding_result.v1",
        "E38 receding result schema differs",
    )
    _require(result.get("scientific_result") is True, "E38 result is not scientific")
    _require(result.get("case_id") == "vlsa-t1-goal-ii-t3-e38", "E38 case differs")
    _require(result.get("source", {}).get("commit") == expected_commit, "E38 source differs")
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(
        _sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        == claimed,
        "E38 result self-hash differs",
    )
    windows = result["windows"]
    _require(windows and windows[0]["step"] == 108, "E38 first window differs")
    _require(
        all(
            all(count == 25 for count in item["selected_prefix"]["record"]["substep_counts"])
            for item in windows
        ),
        "E38 internal substep count differs",
    )
    _require(
        all(
            item["field"] is None or item["field_heldout_gate"] is True
            for item in windows
        ),
        "E38 field direction validation differs",
    )
    physical = result["physical_mujoco_authority"]
    proxy = result["ellipsoid_certification"]
    task = result["task_authority"]
    _require(
        result["summary"]["physical_collision_free_task_completion"]
        == bool(physical["physical_safe"] and task["native_task_success"]),
        "E38 physical/task conjunction differs",
    )
    _require(
        result["summary"]["ellipsoid_certified_task_completion_zero_margin"]
        == bool(
            physical["physical_safe"]
            and task["native_task_success"]
            and proxy["zero_margin_pass"]
        ),
        "E38 zero-margin conjunction differs",
    )
    output = {
        "schema_version": SCHEMA,
        "status": "passed",
        "scientific_result": True,
        "producer_source_commit": expected_commit,
        "case_id": result["case_id"],
        "result_payload_sha256": claimed,
        "physical_mujoco_authority": physical,
        "ellipsoid_certification": proxy,
        "task_authority": task,
        "summary": result["summary"],
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
