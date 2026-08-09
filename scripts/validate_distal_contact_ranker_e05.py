#!/usr/bin/env python3
"""Independent structural verifier for the exact-contact ranker artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from main.multilink_ellipsoid.contact_ranker import load_contact_ranker_config
from scripts.evaluate_distal_contact_ranker_e05 import DATASET_SCHEMA, RESULT_SCHEMA
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value, key):
    payload = dict(value); payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_contact_ranker_config(args.config.resolve())
    dataset = _load(args.dataset.resolve()); result = _load(args.result.resolve())
    _require(dataset.get("schema_version") == DATASET_SCHEMA, "dataset schema differs")
    _require(result.get("schema_version") == RESULT_SCHEMA, "result schema differs")
    _require(dataset.get("source", {}).get("commit") == args.expected_commit, "dataset commit differs")
    _require(result.get("source", {}).get("commit") == args.expected_commit, "result commit differs")
    _require(dataset.get("source", {}).get("dirty") is False and result.get("source", {}).get("dirty") is False, "source is dirty")
    _require(dataset.get("config") == config and result.get("config") == config, "embedded config differs")
    _require(dataset.get("dataset_payload_sha256") == _hash_without(dataset, "dataset_payload_sha256"), "dataset payload differs")
    _require(result.get("result_payload_sha256") == _hash_without(result, "result_payload_sha256"), "result payload differs")
    _require(result.get("dataset_file_sha256") == _file_sha256(args.dataset.resolve()), "dataset file identity differs")
    records = dataset.get("records", []); states = dataset.get("states", [])
    _require(len(states) == 9 and {item["state_step"] for item in states} == set(range(184, 193)), "state population differs")
    _require(records and all(len(item.get("feature_vector", [])) == 33 and len(item.get("link_contact_labels", [])) == 3 for item in records), "record shape differs")
    _require(all(item.get("split") == next(
        name.replace("_steps", "") for name in ("train_steps", "validation_steps", "test_steps")
        if item["state_step"] in config["state_groups"][name]
    ) for item in records), "state-group split differs")
    data_gate = bool(dataset.get("dataset_gate_pass"))
    _require(result.get("dataset_gate_pass") is data_gate, "dataset gate receipt differs")
    if data_gate:
        _require(args.model.resolve().is_file(), "model artifact is missing")
        _require(result.get("training_performed") is True, "training receipt differs")
        _require(result.get("training", {}).get("model_file_sha256") == _file_sha256(args.model.resolve()), "model identity differs")
        ranking = bool(
            result["metrics"]["test"]["false_safe_count"] == 0
            and result["metrics"]["test"]["predicted_safe_count"] > 0
            and all(item.get("fresh_exact_verification")
                    and item["fresh_exact_verification"]["D_sim_raw_safe"]
                    and item["fresh_exact_verification"]["matches_collected_transition"]
                    for item in result["test_selection"])
        )
        gradient = bool(all(item["at_least_one_exact_safe"] for item in result["gradient_audit"]))
        _require(result["decision"]["ranking_feasibility_go"] is ranking, "ranking decision differs")
        _require(result["decision"]["gradient_steering_go"] is gradient, "gradient decision differs")
    validation = {
        "schema_version": "vlsa_distal_contact_ranker_e05_validation.v1",
        "status": "passed", "scientific_result": True,
        "expected_commit": args.expected_commit,
        "dataset_file_sha256": _file_sha256(args.dataset.resolve()),
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "model_file_sha256": _file_sha256(args.model.resolve()) if args.model.resolve().is_file() else None,
        "dataset_gate_pass": data_gate,
        "ranking_feasibility_go": result["decision"]["ranking_feasibility_go"],
        "gradient_steering_go": result["decision"]["gradient_steering_go"],
    }
    validation["validation_payload_sha256"] = _hash_without(validation, "validation_payload_sha256")
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
