#!/usr/bin/env python3
"""Independent replay of the matched L5 state-generalization diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_action_risk import _arrays, _load_samples


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path,
    producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from main.multilink_ellipsoid.l5_action_risk import (
        load_config as load_model_config, load_frozen_bundle, predict,
    )
    from main.multilink_ellipsoid.l5_state_generalization import (
        RESULT_SCHEMA, canonical, load_config, payload_sha256,
        prediction_diagnostic, prediction_gate,
    )

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == RESULT_SCHEMA,
             "L5 state-generalization result schema differs")
    _require(result["source"]["commit"] == producer_commit,
             "L5 state-generalization producer differs")
    _require(result["config"] == config,
             "L5 state-generalization config differs")
    _require(result["result_payload_sha256"] == payload_sha256(result),
             "L5 state-generalization payload differs")
    model_config = load_model_config(
        repo_root / config["source"]["base_model_config"]
    )
    samples = _load_samples(model_config)
    split_samples = {
        "fit": [
            item for item in samples["train"]
            if item["temporal_profile"] in {None, "constant"}
        ],
        "test_A": [
            item for item in samples["train"]
            if item["temporal_profile"] == "front_loaded"
        ],
        "test_B": [
            item for item in samples["validation"]
            if item["temporal_profile"] == "front_loaded"
        ],
    }
    _require(
        {name: len(items) for name, items in split_samples.items()}
        == result["split"]["sample_counts"],
        "L5 state-generalization split count differs",
    )
    state_payload = result["model"]["state_payload"]
    _require(
        hashlib.sha256(canonical(state_payload)).hexdigest()
        == result["model"]["model_sha256"],
        "L5 state-generalization model hash differs",
    )
    bundle = load_frozen_bundle(torch, state_payload, model_config["model"])
    metrics = {}
    gates = {}
    maximum_prediction_replay_error_m = 0.0
    for split_name, split_items in split_samples.items():
        features, _ = _arrays(split_items)
        replay = predict(bundle, features)
        stored = np.asarray(result["predictions"][split_name], dtype=np.float64)
        _require(replay.shape == stored.shape,
                 "L5 state-generalization prediction shape differs")
        maximum_prediction_replay_error_m = max(
            maximum_prediction_replay_error_m,
            float(np.max(np.abs(replay - stored))),
        )
        metrics[split_name] = prediction_diagnostic(replay, split_items)
        gates[split_name] = prediction_gate(
            metrics[split_name], config["diagnostic_gate"]
        )
    _require(maximum_prediction_replay_error_m <= 1.0e-9,
             "L5 state-generalization prediction replay differs")
    _require(metrics == result["metrics"],
             "L5 state-generalization metric replay differs")
    _require(gates == result["gates"],
             "L5 state-generalization gates differ")
    _require(all(gates["test_A"].values()) is result["test_A_pass"]
             and all(gates["test_B"].values()) is result["test_B_pass"],
             "L5 state-generalization pass differs")
    output = {
        "schema_version": "vlsa_distal_l5_state_generalization_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "model_result_file_sha256": _file_sha256(result_path),
        "model_result_payload_sha256": result["result_payload_sha256"],
        "model_sha256": result["model"]["model_sha256"],
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "metrics": metrics,
        "gates": gates,
        "maximum_prediction_replay_error_m":
        maximum_prediction_replay_error_m,
        "test_A_pass": result["test_A_pass"],
        "test_B_pass": result["test_B_pass"],
        "interpretation": result["interpretation"],
        "AEGIS_EE_compatibility_evaluated": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["validation_payload_sha256"] = hashlib.sha256(
        canonical(output)
    ).hexdigest()
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root.resolve(), result_path=args.result.resolve(),
        config_path=args.config.resolve(), producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "interpretation": output["interpretation"],
        "maximum_prediction_replay_error_m":
        output["maximum_prediction_replay_error_m"],
        "test_A_pass": output["test_A_pass"],
        "test_B_pass": output["test_B_pass"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
