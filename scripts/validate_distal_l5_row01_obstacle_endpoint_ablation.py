#!/usr/bin/env python3
"""Independently replay the 9D obstacle-relative endpoint ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_row01_obstacle_endpoint_ablation import arrays, load_samples


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path,
    producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.l5_row01_input_ablation import ablation_metrics
    from main.multilink_ellipsoid.l5_row01_obstacle_endpoint_ablation import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, classify, load_bundle,
        load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.l5_row01_selection import predict

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == RESULT_SCHEMA, "obstacle-endpoint schema differs")
    _require(result["source"]["commit"] == producer_commit, "obstacle-endpoint producer differs")
    _require(result["result_payload_sha256"] == payload_sha256(result), "obstacle-endpoint payload differs")
    _require(result["config"] == config, "obstacle-endpoint embedded config differs")
    samples, complete_frozen, endpoint_frozen = load_samples(config, repo_root)
    arm = result["obstacle_relative_endpoint_9D"]
    payload = arm["model"]["state_payload"]
    _require(
        arm["model"]["model_sha256"] == hashlib.sha256(canonical(payload)).hexdigest(),
        "obstacle-endpoint model hash differs",
    )
    bundle = load_bundle(torch, payload, arm["model_config"])
    maximum_error = 0.0
    metrics = {}
    for split in ("train", "validation"):
        features, _ = arrays(samples[split])
        replay = predict(bundle, features)
        stored = np.asarray(arm["predictions"][split], dtype=np.float64)
        maximum_error = max(maximum_error, float(np.max(np.abs(replay - stored))))
        metrics[split] = ablation_metrics(
            replay, samples[split], random_seed=20260814, random_draws=1024,
        )
    _require(maximum_error <= 1.0e-9, "obstacle-endpoint replay differs")
    _require(metrics == arm["metrics"], "obstacle-endpoint metrics differ")
    endpoint_metrics = endpoint_frozen["endpoint_6D"]["metrics"]
    complete_metrics = complete_frozen["arms"]["complete_physical_OSC_354D"]["metrics"]
    _require(endpoint_metrics == result["frozen_endpoint_6D_metrics"], "endpoint comparator differs")
    _require(complete_metrics == result["frozen_complete_354D_metrics"], "complete comparator differs")
    interpretation, comparison = classify(
        metrics["validation"], endpoint_metrics["validation"],
        minimum_relative_rmse_improvement=float(config["decision"][
            "minimum_relative_validation_RMSE_improvement_over_endpoint_6D"
        ]),
    )
    _require(interpretation == result["interpretation"], "obstacle-endpoint interpretation differs")
    _require(comparison == result["comparison"], "obstacle-endpoint comparison differs")
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "validated",
        "scientific_result": True, "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "maximum_prediction_replay_error_m": maximum_error,
        "metrics": metrics, "comparison": comparison,
        "interpretation": interpretation,
        "prediction_or_control_gate_pass": False,
        "fresh_exact_replay_authorized": False,
        "QP_authorized": False, "closed_loop_authorized": False,
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
