#!/usr/bin/env python3
"""Independently recompute the targeted expansion support decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.targeted_boundary_expansion import (
    RESULT_SCHEMA, SCAN_SCHEMA, SELECTION_SCHEMA, VALIDATION_SCHEMA, evaluate,
    load_config, select_boundary_steps,
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
    parser.add_argument("--base-expanded-dataset", type=Path, required=True)
    parser.add_argument("--base-expanded-oracle", type=Path, required=True)
    parser.add_argument("--scan-records", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--additional-dataset", type=Path, required=True)
    parser.add_argument("--collection-result", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _load(args.result.resolve())
    config = load_config(args.config.resolve())
    scan = _load(args.scan_records.resolve())
    selection = _load(args.selection.resolve())
    recomputed_selection = select_boundary_steps(scan["records"], config)
    _require(
        scan.get("schema_version") == SCAN_SCHEMA
        and scan.get("source", {}).get("commit") == args.expected_commit
        and scan.get("source", {}).get("dirty") is False
        and scan.get("scan_payload_sha256")
        == _hash_without(scan, "scan_payload_sha256")
        and selection.get("schema_version") == SELECTION_SCHEMA
        and selection.get("source_scan_payload_sha256")
        == scan["scan_payload_sha256"]
        and selection.get("selection_payload_sha256")
        == _hash_without(selection, "selection_payload_sha256"),
        "targeted expansion scan or selection identity differs",
    )
    for key, value in recomputed_selection.items():
        _require(
            selection.get(key) == value,
            "targeted expansion selection recomputation differs: %s" % key,
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
        == _hash_without(result, "result_payload_sha256"),
        "targeted expansion result identity differs",
    )
    if not selection["all_episode_windows_valid"]:
        _require(
            result.get("aggregates", {}).get("all_episode_windows_valid") is False
            and result.get("decision", {}).get(
                "current_region_aware_MLP_retraining_authorized"
            ) is False
            and result.get("decision", {}).get(
                "closed_loop_E05_remains_blocked"
            ) is True,
            "targeted expansion selection NO-GO decision differs",
        )
        output = {
            "schema_version": VALIDATION_SCHEMA, "status": "valid",
            "scientific_result": True,
            "expected_commit": args.expected_commit,
            "result_file_sha256": _file_sha256(args.result.resolve()),
            "result_payload_sha256": result["result_payload_sha256"],
            "scan_file_sha256": _file_sha256(args.scan_records.resolve()),
            "selection_file_sha256": _file_sha256(args.selection.resolve()),
            "aggregates": result["aggregates"],
            "decision": result["decision"],
        }
        _atomic_write(args.output.resolve(), output)
        print(json.dumps(output, sort_keys=True), flush=True)
        return 0
    base_dataset = _load(args.base_expanded_dataset.resolve())
    base_oracle = _load(args.base_expanded_oracle.resolve())
    additional = _load(args.additional_dataset.resolve())
    collection = _load(args.collection_result.resolve())
    dataset, oracle, audit = evaluate(
        base_dataset, additional, base_oracle, config,
    )
    _require(
        collection.get("source", {}).get("commit") == args.expected_commit
        and collection.get("source", {}).get("dirty") is False
        and collection.get("result_payload_sha256")
        == _hash_without(collection, "result_payload_sha256")
        and result.get("additional_collection", {}).get("result_file_sha256")
        == _file_sha256(args.collection_result.resolve()),
        "targeted expansion collection identity differs",
    )
    _require(
        _load(args.expanded_dataset.resolve()) == dataset
        and _load(args.expanded_oracle.resolve()) == oracle,
        "targeted expansion derived artifact recomputation differs",
    )
    for key in (
        "feature_shift", "reference", "validation_state_results",
        "test_state_results", "aggregates", "decision",
    ):
        _require(
            result.get(key) == audit[key],
            "targeted expansion audit differs: %s" % key,
        )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": True, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "scan_file_sha256": _file_sha256(args.scan_records.resolve()),
        "selection_file_sha256": _file_sha256(args.selection.resolve()),
        "expanded_dataset_file_sha256": _file_sha256(
            args.expanded_dataset.resolve()
        ),
        "expanded_oracle_file_sha256": _file_sha256(
            args.expanded_oracle.resolve()
        ),
        "aggregates": result["aggregates"], "decision": result["decision"],
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
