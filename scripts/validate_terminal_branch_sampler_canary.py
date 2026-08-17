#!/usr/bin/env python3
"""Independently validate paired terminal-branch sampler canary results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _load
from main.multilink_ellipsoid.terminal_branching import (
    RESULT_SCHEMA,
    VALIDATION_SCHEMA,
    payload_sha256,
)


def validate(producer_path: Path, replay_path: Path) -> dict:
    producer = _load(producer_path)
    replay = _load(replay_path)
    checks = {
        "producer_schema": producer.get("schema_version") == RESULT_SCHEMA,
        "replay_schema": replay.get("schema_version") == RESULT_SCHEMA,
        "producer_payload": producer.get("result_payload_sha256")
        == payload_sha256(producer, "result_payload_sha256"),
        "replay_payload": replay.get("result_payload_sha256")
        == payload_sha256(replay, "result_payload_sha256"),
        "producer_replica": producer.get("replica") == "producer",
        "replay_replica": replay.get("replica") == "replay",
        "producer_passing": producer.get("status") == "passing",
        "replay_passing": replay.get("status") == "passing",
        "scientific_view_exact": producer.get("scientific_view")
        == replay.get("scientific_view"),
    }
    gates = producer.get("scientific_view", {}).get("gates", {})
    checks["all_scientific_gates"] = bool(gates) and all(
        value is True for value in gates.values()
    )
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passing" if all(checks.values()) else "scientific_no_go",
        "scientific_result": True,
        "producer_path": str(producer_path),
        "replay_path": str(replay_path),
        "producer_payload_sha256": producer.get("result_payload_sha256"),
        "replay_payload_sha256": replay.get("result_payload_sha256"),
        "checks": checks,
        "scientific_view": producer.get("scientific_view"),
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
        "validation_payload_sha256": result["validation_payload_sha256"],
        "checks": result["checks"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
