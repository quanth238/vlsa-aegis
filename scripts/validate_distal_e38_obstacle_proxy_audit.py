#!/usr/bin/env python3
"""Independently validate the matched E38 obstacle-proxy audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _load,
    _require,
    _sha256,
)


SCHEMA = "vlsa_distal_e38_obstacle_proxy_audit_validation.v1"


def validate(result_path: Path, expected_commit: str) -> dict[str, Any]:
    result = _load(result_path)
    _require(
        result.get("schema_version") == "vlsa_distal_e38_obstacle_proxy_audit_result.v1",
        "E38 obstacle proxy result schema differs",
    )
    _require(result.get("scientific_result") is True, "result is not scientific")
    _require(result.get("case_id") == "vlsa-t1-goal-ii-t3-e38", "case differs")
    _require(result.get("source", {}).get("commit") == expected_commit, "source differs")
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(
        _sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        == claimed,
        "result self-hash differs",
    )
    measurement = result["measurement"]
    _require(measurement["sample_count"] == len(measurement["samples"]), "sample count differs")
    _require(
        len(measurement["substep_counts"]) == 13
        and all(value == 25 for value in measurement["substep_counts"]),
        "E38 action/substep replay differs",
    )
    summary = result["summary"]
    samples = measurement["samples"]
    perceived_min = min(item["perceived_mvee_minimum_support_gap_m"] for item in samples)
    compiled_min = min(
        item["compiled_box_vertex_mvee_minimum_support_gap_m"] for item in samples
    )
    exact_overlap = sum(
        bool(item["compiled_box_union_any_exact_solid_overlap"]) for item in samples
    )
    raw_contact = sum(bool(item["raw_protected_contact_count"]) for item in samples)
    _require(summary["perceived_mvee_minimum_support_gap_m"] == perceived_min, "perceived minimum differs")
    _require(summary["compiled_box_vertex_mvee_minimum_support_gap_m"] == compiled_min, "compiled MVEE minimum differs")
    _require(summary["compiled_box_union_exact_overlap_sample_count"] == exact_overlap, "exact overlap count differs")
    _require(summary["raw_protected_contact_sample_count"] == raw_contact, "raw contact count differs")
    partial = bool(perceived_min < 0.0 and raw_contact == 0 and exact_overlap == 0)
    strong = bool(partial and compiled_min >= 0.0)
    unresolved_robot = bool(exact_overlap > 0 and raw_contact == 0)
    _require(summary["partial_proxy_mismatch"] == partial, "partial decision differs")
    _require(summary["strong_perception_mvee_isolation"] == strong, "strong decision differs")
    _require(summary["unresolved_robot_proxy"] == unresolved_robot, "robot-proxy decision differs")
    output = {
        "schema_version": SCHEMA,
        "status": "passed",
        "scientific_result": True,
        "producer_source_commit": expected_commit,
        "case_id": result["case_id"],
        "result_payload_sha256": claimed,
        "summary": summary,
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
