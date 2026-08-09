#!/usr/bin/env python3
"""Train the grouped direct signed two-step rollout-margin model on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.direct_rollout_margin import (
    DIRECT_MARGIN_TRAINING_SCHEMA,
    load_direct_margin_config,
    save_direct_margin_model,
    train_direct_margin_model,
)
from main.multilink_ellipsoid.two_step_margin import TWO_STEP_DATASET_SCHEMA
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
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_direct_margin_config(args.config.resolve())
    dataset = _load(args.dataset.resolve())
    dataset_result = _load(args.dataset_result.resolve())
    validation = _load(args.dataset_validation.resolve())
    identities = config["immutable_sources"]
    for name, path, expected in (
        ("dataset", args.dataset, identities["dataset_file_sha256"]),
        ("dataset_result", args.dataset_result, identities["dataset_result_file_sha256"]),
        ("dataset_validation", args.dataset_validation, identities["dataset_validation_file_sha256"]),
    ):
        _require(
            path.is_file() and not path.is_symlink()
            and _file_sha256(path.resolve()) == expected,
            "direct-margin immutable %s differs" % name,
        )
    _require(
        dataset.get("schema_version") == TWO_STEP_DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == identities["dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256")
        and dataset.get("source_commit") == identities["dataset_source_commit"],
        "direct-margin dataset identity differs",
    )
    _require(
        dataset_result.get("status") == "complete"
        and validation.get("status") == "valid"
        and validation.get("neural_training_authorized") is True,
        "direct-margin dataset gate did not authorize training",
    )
    learning = [
        item for item in dataset["records"]
        if item.get("split") in {"train", "validation", "test"}
    ]
    model, state, audit = train_direct_margin_model(learning, config)
    del model
    model_identity = save_direct_margin_model(args.model.resolve(), state)
    output = {
        "schema_version": DIRECT_MARGIN_TRAINING_SCHEMA,
        "status": "complete", "scientific_result": False,
        "source": _git_identity(args.repo_root.resolve(), args.expected_commit),
        "config_file_sha256": config["config_file_sha256"],
        "dataset": {
            "path": str(args.dataset.resolve()),
            "file_sha256": identities["dataset_file_sha256"],
            "payload_sha256": identities["dataset_payload_sha256"],
            "source_commit": identities["dataset_source_commit"],
            "learning_record_count": len(learning),
        },
        "model_artifact": model_identity,
        "training": audit,
    }
    output["training_payload_sha256"] = _hash_without(
        output, "training_payload_sha256"
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "e05_model_gate_pass": audit["e05_model_gate_pass"],
        "test_false_safe_gate_pass": audit["test_false_safe_gate_pass"],
        "e05": audit["per_episode_metrics"][config["case_id"]],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
