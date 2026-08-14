#!/usr/bin/env python3
"""Independently replay the restricted L5 capacity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_action_risk_diagnostic import _arrays, load_samples


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path,
    producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from main.multilink_ellipsoid.l5_action_risk import (
        load_frozen_bundle, predict,
    )
    from main.multilink_ellipsoid.l5_action_risk_diagnostic import (
        MODEL_SCHEMA, VALIDATION_SCHEMA, canonical, load_config,
        payload_sha256, report_metrics,
    )

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == MODEL_SCHEMA,
             "L5 diagnostic model schema differs")
    _require(result["source"]["commit"] == producer_commit,
             "L5 diagnostic producer differs")
    _require(result["result_payload_sha256"] == payload_sha256(result),
             "L5 diagnostic model payload differs")
    _require(result["config"] == config,
             "L5 diagnostic embedded config differs")
    samples = load_samples(config)
    payload = result["model"]["state_payload"]
    _require(
        result["model"]["model_sha256"]
        == hashlib.sha256(canonical(payload)).hexdigest(),
        "L5 diagnostic model hash differs",
    )
    bundle = load_frozen_bundle(torch, payload, config["model"])
    near_m = float(config["reporting"]["near_boundary_absolute_risk_m"])
    reports = {}
    maximum_error = 0.0
    for name, subset in samples.items():
        features, _ = _arrays(subset)
        replay = predict(bundle, features)
        stored = np.asarray(result["predictions"][name], dtype=np.float64)
        _require(replay.shape == stored.shape,
                 "L5 diagnostic replay shape differs")
        maximum_error = max(maximum_error, float(np.max(np.abs(replay - stored))))
        reports[name] = report_metrics(replay, subset, near_m=near_m)
    _require(maximum_error <= 1.0e-9,
             "L5 diagnostic prediction replay differs")
    _require(reports == result["metrics"],
             "L5 diagnostic metric replay differs")
    _require(
        result["diagnostic_training_completed"] is True
        and result["generalizable_safety_filter_authorized"] is False
        and result["QP_authorized"] is False
        and result["closed_loop_authorized"] is False,
        "L5 diagnostic scope differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "model_result_file_sha256": _file_sha256(result_path),
        "model_result_payload_sha256": result["result_payload_sha256"],
        "model_sha256": result["model"]["model_sha256"],
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "sample_counts": result["dataset"]["sample_counts"],
        "metrics": reports,
        "maximum_prediction_replay_error_m": maximum_error,
        "diagnostic_training_completed": True,
        "generalizable_safety_filter_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": result["interpretation"],
    }
    output["validation_payload_sha256"] = payload_sha256(output)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root.resolve(), result_path=args.model_result.resolve(),
        config_path=args.config.resolve(), producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "interpretation": output["interpretation"],
        "maximum_prediction_replay_error_m":
        output["maximum_prediction_replay_error_m"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
