#!/usr/bin/env python3
"""Independently replay the frozen L5 row-0/1 weight-decay ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_row01_input_ablation import (
    _complete_arrays, load_augmented_samples,
)
from scripts.train_distal_l5_row01_weight_decay_ablation import _source


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path,
    producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from main.multilink_ellipsoid.l5_row01_input_ablation import (
        ablation_metrics, load_complete_bundle,
    )
    from main.multilink_ellipsoid.l5_row01_selection import predict
    from main.multilink_ellipsoid.l5_row01_weight_decay_ablation import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, classify_effect,
        load_config, payload_sha256,
    )

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == RESULT_SCHEMA,
             "weight-decay result schema differs")
    _require(result["source"]["commit"] == producer_commit,
             "weight-decay producer differs")
    _require(result["result_payload_sha256"] == payload_sha256(result),
             "weight-decay result payload differs")
    _require(result["config"] == config, "weight-decay embedded config differs")
    input_config, samples, _ = _source(config, repo_root)
    maximum_error = 0.0
    replay_metrics = {}
    for name in ("no_weight_decay", "established_weight_decay"):
        arm = result["arms"][name]
        payload = arm["model"]["state_payload"]
        _require(
            arm["model"]["model_sha256"]
            == hashlib.sha256(canonical(payload)).hexdigest(),
            "weight-decay model hash differs",
        )
        bundle = load_complete_bundle(torch, payload, arm["model_config"])
        replay_metrics[name] = {}
        for split in ("train", "validation"):
            features, _ = _complete_arrays(samples[split])
            replay = predict(bundle, features)
            stored = np.asarray(arm["predictions"][split], dtype=np.float64)
            maximum_error = max(
                maximum_error, float(np.max(np.abs(replay - stored)))
            )
            replay_metrics[name][split] = ablation_metrics(
                replay, samples[split], random_seed=20260814,
                random_draws=1024,
            )
    _require(maximum_error <= 1.0e-9, "weight-decay prediction replay differs")
    _require(
        replay_metrics
        == {name: arm["metrics"] for name, arm in result["arms"].items()},
        "weight-decay replay metrics differ",
    )
    interpretation, comparison = classify_effect(
        replay_metrics["no_weight_decay"]["validation"],
        replay_metrics["established_weight_decay"]["validation"],
        minimum_relative_validation_improvement=float(
            config["decision"]["minimum_relative_validation_RMSE_improvement"]
        ),
    )
    _require(interpretation == result["interpretation"],
             "weight-decay interpretation differs")
    _require(comparison == result["comparison"],
             "weight-decay comparison differs")
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "maximum_prediction_replay_error_m": maximum_error,
        "metrics": replay_metrics,
        "comparison": comparison,
        "interpretation": interpretation,
        "prediction_or_control_gate_pass": False,
        "fresh_exact_replay_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["validation_payload_sha256"] = payload_sha256(output)
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
        "comparison": output["comparison"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
