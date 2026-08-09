#!/usr/bin/env python3
"""Train paired global and factorized two-step residual MLPs on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.two_step_margin import (
    TWO_STEP_DATASET_SCHEMA,
    TWO_STEP_TRAINING_SCHEMA,
    load_two_step_config,
    save_model,
    train_model,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


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
    config = load_two_step_config(args.config.resolve())
    dataset = _load(args.dataset.resolve())
    result = _load(args.dataset_result.resolve())
    validation = _load(args.dataset_validation.resolve())
    _require(
        dataset.get("schema_version") == TWO_STEP_DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == _hash_without(dataset, "dataset_payload_sha256"),
        "two-step training dataset differs",
    )
    _require(
        dataset.get("source_commit") == args.expected_commit
        and result.get("source", {}).get("commit") == args.expected_commit,
        "two-step training source differs",
    )
    _require(
        validation.get("status") == "valid"
        and validation.get("neural_training_authorized") is True
        and validation.get("dataset_file_sha256")
        == _file_sha256(args.dataset.resolve())
        and validation.get("result_file_sha256")
        == _file_sha256(args.dataset_result.resolve()),
        "two-step no-training gate did not authorize training",
    )
    learning = [
        item
        for item in dataset["records"]
        if item["split"] in {"train", "validation", "test"}
    ]
    directory = args.model_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm in config["training"]["arms"]:
        model, state, audit = train_model(learning, config, arm=arm)
        del model
        identity = save_model(directory / (arm + ".npz"), state)
        arms[arm] = {"model_artifact": identity, "training": audit}
    output = {
        "schema_version": TWO_STEP_TRAINING_SCHEMA,
        "status": "complete", "scientific_result": False,
        "source": _git_identity(args.repo_root.resolve(), args.expected_commit),
        "config_file_sha256": config["config_file_sha256"],
        "dataset": {
            "path": str(args.dataset.resolve()),
            "file_sha256": _file_sha256(args.dataset.resolve()),
            "payload_sha256": dataset["dataset_payload_sha256"],
            "learning_record_count": len(learning),
        },
        "dataset_gate": {
            "validation_file_sha256": _file_sha256(
                args.dataset_validation.resolve()
            ),
            "neural_training_authorized": True,
        },
        "arms": arms,
    }
    output["training_payload_sha256"] = _hash_without(
        output, "training_payload_sha256"
    )
    _atomic_write(args.output.resolve(), output)
    print(
        json.dumps(
            {
                arm: {
                    "gate": value["training"]["held_out_model_gate_pass"],
                    "metrics": value["training"]["per_episode_metrics"],
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
