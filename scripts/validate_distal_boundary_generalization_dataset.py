#!/usr/bin/env python3
"""Validate the no-training grouped boundary dataset gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from main.multilink_ellipsoid.boundary_generalization import (
    GENERALIZATION_DATASET_RESULT_SCHEMA, GENERALIZATION_DATASET_SCHEMA,
    load_generalization_config, load_selected_manifest,
)
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _file_sha256, _load, _require


def _hash_without(value, key):
    payload = dict(value); payload.pop(key, None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_generalization_config(args.config.resolve())
    selected = load_selected_manifest(args.selected_manifest.resolve(), config)
    result = _load(args.result.resolve()); dataset = _load(args.dataset.resolve())
    _require(result.get("schema_version") == GENERALIZATION_DATASET_RESULT_SCHEMA, "dataset result schema differs")
    _require(dataset.get("schema_version") == GENERALIZATION_DATASET_SCHEMA, "dataset schema differs")
    _require(result.get("source", {}).get("commit") == args.expected_commit == dataset.get("source_commit"), "dataset commit differs")
    _require(dataset.get("dataset_payload_sha256") == _hash_without(dataset, "dataset_payload_sha256"), "dataset payload differs")
    _require(result.get("result_payload_sha256") == _hash_without(result, "result_payload_sha256"), "result payload differs")
    _require(result.get("dataset", {}).get("file_sha256") == _file_sha256(args.dataset.resolve()), "dataset file binding differs")
    records = dataset.get("records", [])
    selected_by_id = {row["case_id"]: row for row in selected}
    _require(set(item["case_id"] for item in records) <= set(selected_by_id), "unknown dataset case")
    _require(all(item["episode_split"] == selected_by_id[item["case_id"]]["split"] for item in records), "episode split leakage")
    _require(all(item["split"] in {item["episode_split"], "excluded_far"} for item in records), "record split leakage")
    _require(all(item["case_id"] != "vlsa-t1-goal-ii-t0-e05" or item["split"] in {"test", "excluded_far"} for item in records), "E05 leakage")
    group_splits = {}
    for row in selected:
        group = row["task_level_group_id"]
        _require(group not in group_splits or group_splits[group] == row["split"], "task group leakage")
        group_splits[group] = row["split"]
    gate = bool(result.get("decision", {}).get("dataset_gate_pass") and dataset.get("summary", {}).get("dataset_gate_pass"))
    validation = {
        "schema_version": "vlsa_distal_boundary_generalization_moka10_dataset_validation.v1",
        "status": "valid", "source_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "dataset_file_sha256": _file_sha256(args.dataset.resolve()),
        "selected_case_count": len(selected), "task_group_splits": group_splits,
        "episode_and_task_group_leakage_count": 0,
        "primary_e05_test_only": True,
        "neural_training_authorized": gate,
    }
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps({"status": "valid", "neural_training_authorized": gate}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
