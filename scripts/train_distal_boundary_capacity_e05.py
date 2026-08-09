#!/usr/bin/env python3
"""Train the paired action-188 boundary-capacity MLP arms on an H100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.boundary_capacity import (
    BOUNDARY_DATASET_SCHEMA,
    load_boundary_capacity_config,
    train_boundary_capacity_model,
)
from main.multilink_ellipsoid.execution_margin_nn import save_model_artifact
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


TRAINING_SCHEMA = "vlsa_distal_boundary_capacity_e05_training.v1"


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_boundary_capacity_config(args.config.resolve())
    dataset_path = args.dataset.resolve()
    dataset_result_path = args.dataset_result.resolve()
    dataset_validation_path = args.dataset_validation.resolve()
    dataset = _load(dataset_path)
    dataset_result = _load(dataset_result_path)
    validation = _load(dataset_validation_path)
    _require(
        dataset.get("schema_version") == BOUNDARY_DATASET_SCHEMA
        and dataset.get("config_file_sha256") == config["config_file_sha256"]
        and dataset.get("config_payload_sha256") == config["config_payload_sha256"]
        and dataset.get("source_commit") == args.expected_commit
        and dataset.get("dataset_payload_sha256")
        == _hash_without(dataset, "dataset_payload_sha256")
        and len(dataset.get("records", [])) == 1384,
        "boundary-capacity training dataset differs",
    )
    _require(
        dataset_result.get("status") == "complete"
        and dataset_result.get("source", {}).get("commit") == args.expected_commit
        and dataset_result.get("dataset", {}).get("file_sha256")
        == _file_sha256(dataset_path)
        and validation.get("status") == "valid"
        and validation.get("source_commit") == args.expected_commit
        and validation.get("result_file_sha256")
        == _file_sha256(dataset_result_path)
        and validation.get("dataset_file_sha256") == _file_sha256(dataset_path)
        and validation.get("neural_training_authorized") is True,
        "boundary-capacity no-training gate did not authorize training",
    )
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    model_directory = args.model_directory.resolve()
    model_directory.mkdir(parents=True, exist_ok=True)
    arms: dict[str, Any] = {}
    for arm in config["training"]["arms"]:
        model, state, audit = train_boundary_capacity_model(
            dataset["records"], config, arm=arm
        )
        del model
        model_path = model_directory / (arm + ".npz")
        model_identity = save_model_artifact(model_path, state)
        arms[arm] = {
            "model_artifact": model_identity,
            "training": audit,
        }
    result = {
        "schema_version": TRAINING_SCHEMA,
        "status": "complete",
        "scientific_result": False,
        "source": source,
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "dataset": {
            "path": str(dataset_path),
            "file_sha256": _file_sha256(dataset_path),
            "payload_sha256": dataset["dataset_payload_sha256"],
            "record_count": len(dataset["records"]),
        },
        "dataset_gate": {
            "result_path": str(dataset_result_path),
            "result_file_sha256": _file_sha256(dataset_result_path),
            "validation_path": str(dataset_validation_path),
            "validation_file_sha256": _file_sha256(dataset_validation_path),
            "neural_training_authorized": True,
        },
        "arms": arms,
    }
    result["training_payload_sha256"] = _hash_without(
        result, "training_payload_sha256"
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                arm: {
                    "model_capacity_gate_pass": value["training"][
                        "model_capacity_gate_pass"
                    ],
                    "test": value["training"]["split_metrics"]["test"],
                }
                for arm, value in arms.items()
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
