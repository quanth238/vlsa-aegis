#!/usr/bin/env python3
"""Validate the 85-state initial protected-contact classification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.initial_contact_audit import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_config, summarize,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {
        "result": args.result, "config": args.config, "output": args.output,
    }.items()}
    result = _load(paths["result"])
    config = load_config(paths["config"])
    semantic_hashes = {
        item["semantic_protected_group_sha256"]
        for item in result.get("state_results", [])
    }
    recomputed = summarize(
        result.get("state_results", []), config, len(semantic_hashes)
    )
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"]
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and result.get("aggregates") == recomputed["aggregates"]
        and result.get("decision") == recomputed["decision"]
        and result["decision"]["training_executed"] is False
        and result["decision"]["QP_executed"] is False
        and result["decision"]["closed_loop_E05_authorized"] is False,
        "initial-contact result differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": True, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "aggregates": result["aggregates"],
        "initial_contact_audit_complete": result["decision"][
            "initial_contact_audit_complete"
        ],
        "closed_loop_E05_authorized": False,
    }
    _atomic_write(paths["output"], output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
