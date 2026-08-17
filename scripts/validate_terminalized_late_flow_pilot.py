#!/usr/bin/env python3
"""Validate independent selected-action terminalized late-flow pilot replicas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _load
from main.multilink_ellipsoid.terminal_branching import (
    PILOT_RESULT_SCHEMA,
    PILOT_VALIDATION_SCHEMA,
    payload_sha256,
)


def validate(producer_path: Path, replay_path: Path) -> dict:
    producer = _load(producer_path)
    replay = _load(replay_path)
    checks = {
        "producer_schema": producer.get("schema_version") == PILOT_RESULT_SCHEMA,
        "replay_schema": replay.get("schema_version") == PILOT_RESULT_SCHEMA,
        "producer_payload": producer.get("result_payload_sha256")
        == payload_sha256(producer, "result_payload_sha256"),
        "replay_payload": replay.get("result_payload_sha256")
        == payload_sha256(replay, "result_payload_sha256"),
        "producer_complete": producer.get("status") == "complete",
        "replay_complete": replay.get("status") == "complete",
        "producer_replica": producer.get("replica") == "producer",
        "replay_replica": replay.get("replica") == "replay",
        "scientific_view_exact": producer.get("scientific_view")
        == replay.get("scientific_view"),
    }
    gates = producer.get("scientific_view", {}).get("gates", {})
    checks["all_apparatus_gates"] = bool(gates) and all(
        value is True for value in gates.values()
    )
    outcome = producer.get("scientific_view", {}).get("outcome", {})
    records = outcome.get("records", [])
    by_arm = {row.get("arm"): row for row in records}
    required = {
        "nominal", "terminal_compact_selector",
        "late_flow_terminalized_compact_selector",
    }
    checks["three_outcomes_present"] = set(by_arm) == required
    if set(by_arm) == required:
        nominal = bool(by_arm["nominal"]["physical_safe"])
        posthoc = bool(by_arm["terminal_compact_selector"]["physical_safe"])
        late = bool(by_arm[
            "late_flow_terminalized_compact_selector"
        ]["physical_safe"])
        if late and not posthoc:
            comparison = "late_flow_better_than_posthoc"
        elif posthoc and not late:
            comparison = "posthoc_better_than_late_flow"
        elif late and posthoc:
            comparison = "both_selectors_safe"
        else:
            comparison = "neither_selector_safe"
        mechanism = {
            "ordinary_physical_safe": nominal,
            "posthoc_physical_safe": posthoc,
            "late_flow_physical_safe": late,
            "ordinary_raw_physical_collision": bool(
                by_arm["nominal"]["raw_physical_collision"]
            ),
            "posthoc_raw_physical_collision": bool(
                by_arm["terminal_compact_selector"]["raw_physical_collision"]
            ),
            "late_flow_raw_physical_collision": bool(by_arm[
                "late_flow_terminalized_compact_selector"
            ]["raw_physical_collision"]),
            "comparison": comparison,
            "late_flow_collision_avoided": bool(
                not nominal and late
            ),
            "posthoc_collision_avoided": bool(
                not nominal and posthoc
            ),
            "late_flow_raw_collision_avoided": bool(
                by_arm["nominal"]["raw_physical_collision"]
                and not by_arm[
                    "late_flow_terminalized_compact_selector"
                ]["raw_physical_collision"]
            ),
            "posthoc_raw_collision_avoided": bool(
                by_arm["nominal"]["raw_physical_collision"]
                and not by_arm["terminal_compact_selector"][
                    "raw_physical_collision"
                ]
            ),
        }
    else:
        mechanism = None
    result = {
        "schema_version": PILOT_VALIDATION_SCHEMA,
        "status": "passing" if all(checks.values()) else "apparatus_failure",
        "scientific_result": True,
        "producer_path": str(producer_path),
        "replay_path": str(replay_path),
        "producer_payload_sha256": producer.get("result_payload_sha256"),
        "replay_payload_sha256": replay.get("result_payload_sha256"),
        "checks": checks,
        "mechanism": mechanism,
        "scientific_view": producer.get("scientific_view"),
        "correction_safety_authorized": False,
        "formal_safety_claim": False,
    }
    result["validation_payload_sha256"] = payload_sha256(
        result, "validation_payload_sha256"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.producer.resolve(), args.replay.resolve())
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "mechanism": result["mechanism"],
        "validation_payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
