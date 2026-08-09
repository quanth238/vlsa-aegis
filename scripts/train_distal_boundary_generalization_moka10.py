#!/usr/bin/env python3
"""Train paired grouped-boundary residual MLP arms on an H100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from main.multilink_ellipsoid.boundary_capacity import train_boundary_capacity_model
from main.multilink_ellipsoid.boundary_generalization import (
    GENERALIZATION_DATASET_SCHEMA, GENERALIZATION_TRAINING_SCHEMA,
    load_generalization_config,
)
from main.multilink_ellipsoid.execution_margin_nn import save_model_artifact
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _file_sha256, _git_identity, _load, _require


def _hash_without(value, key):
    payload = dict(value); payload.pop(key, None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_generalization_config(args.config.resolve())
    dataset = _load(args.dataset.resolve()); result = _load(args.dataset_result.resolve()); validation = _load(args.dataset_validation.resolve())
    _require(dataset.get("schema_version") == GENERALIZATION_DATASET_SCHEMA and dataset.get("dataset_payload_sha256") == _hash_without(dataset, "dataset_payload_sha256"), "training dataset differs")
    _require(dataset.get("source_commit") == args.expected_commit and result.get("source", {}).get("commit") == args.expected_commit, "training source differs")
    _require(validation.get("status") == "valid" and validation.get("neural_training_authorized") is True, "no-training gate did not authorize training")
    _require(validation.get("dataset_file_sha256") == _file_sha256(args.dataset.resolve()) and validation.get("result_file_sha256") == _file_sha256(args.dataset_result.resolve()), "training gate binding differs")
    learning = [item for item in dataset["records"] if item["split"] in {"train", "validation", "test"}]
    arms = {}
    args.model_directory.resolve().mkdir(parents=True, exist_ok=True)
    for arm in config["training"]["arms"]:
        model, state, audit = train_boundary_capacity_model(learning, config, arm=arm)
        del model
        identity = save_model_artifact(args.model_directory.resolve() / (arm + ".npz"), state)
        arms[arm] = {"model_artifact": identity, "training": audit}
    output = {
        "schema_version": GENERALIZATION_TRAINING_SCHEMA, "status": "complete",
        "scientific_result": False, "source": _git_identity(args.repo_root.resolve(), args.expected_commit),
        "config_file_sha256": config["config_file_sha256"],
        "dataset": {"path": str(args.dataset.resolve()), "file_sha256": _file_sha256(args.dataset.resolve()), "payload_sha256": dataset["dataset_payload_sha256"], "learning_record_count": len(learning)},
        "dataset_gate": {"validation_file_sha256": _file_sha256(args.dataset_validation.resolve()), "neural_training_authorized": True},
        "arms": arms,
    }
    output["training_payload_sha256"] = _hash_without(output, "training_payload_sha256")
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({arm: value["training"]["split_metrics"]["test"] for arm, value in arms.items()}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
