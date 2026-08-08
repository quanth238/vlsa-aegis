#!/usr/bin/env python3
"""Train the E05 execution-margin residual MLP in the H100 OpenPI runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

from main.multilink_ellipsoid.execution_margin_nn import (
    load_execution_margin_config,
    save_model_artifact,
    train_execution_margin_model,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


def _dataset_payload(dataset: dict) -> str:
    payload = dict(dataset)
    payload.pop("dataset_payload_sha256", None)
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
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_execution_margin_config(args.config.resolve())
    dataset_path = args.dataset.resolve()
    dataset = _load(dataset_path)
    _require(
        dataset.get("schema_version")
        == "vlsa_distal_execution_margin_nn_e05_dataset.v1"
        and dataset.get("config_file_sha256") == config["config_file_sha256"]
        and dataset.get("config_payload_sha256") == config["config_payload_sha256"]
        and dataset.get("source_commit") == args.expected_commit
        and dataset.get("dataset_payload_sha256") == _dataset_payload(dataset)
        and len(dataset.get("records", [])) == 1125,
        "execution-margin training dataset differs",
    )
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    model, model_state, audit = train_execution_margin_model(
        dataset["records"], config
    )
    del model
    model_identity = save_model_artifact(args.model.resolve(), model_state)
    result = {
        "schema_version": "vlsa_distal_execution_margin_nn_e05_training.v1",
        "status": "complete",
        "scientific_result": False,
        "source": source,
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "dataset": {
            "path": str(dataset_path),
            "file_sha256": _file_sha256(dataset_path),
            "payload_sha256": dataset["dataset_payload_sha256"],
            "sample_count": len(dataset["records"]),
        },
        "model_artifact": model_identity,
        "training": audit,
    }
    result["training_payload_sha256"] = hashlib.sha256(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "model_learnability_gate_pass": audit[
                    "model_learnability_gate_pass"
                ],
                "best_epoch": audit["best_epoch"],
                "test": audit["split_metrics"]["test"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
