#!/usr/bin/env python3
"""Independently replay the two-output L5 row01 prediction and selection gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_row01_selection import _arrays, load_samples


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path,
    producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from main.multilink_ellipsoid.l5_row01_selection import (
        MODEL_SCHEMA, VALIDATION_SCHEMA, canonical, load_config,
        load_frozen_bundle, payload_sha256, predict, selection_metrics,
    )

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == MODEL_SCHEMA,
             "L5 row01 result schema differs")
    _require(result["source"]["commit"] == producer_commit,
             "L5 row01 producer commit differs")
    _require(result["result_payload_sha256"] == payload_sha256(result),
             "L5 row01 result payload differs")
    _require(result["config"] == config, "L5 row01 embedded config differs")
    payload = result["model"]["state_payload"]
    _require(
        result["model"]["model_sha256"]
        == hashlib.sha256(canonical(payload)).hexdigest(),
        "L5 row01 model hash differs",
    )
    samples = load_samples(config)
    bundle = load_frozen_bundle(torch, payload, config["model"])
    maximum_error = 0.0
    reports = {}
    for split in ("train", "validation"):
        features, _ = _arrays(samples[split])
        replay = predict(bundle, features)
        stored = np.asarray(result["predictions"][split], dtype=np.float64)
        maximum_error = max(maximum_error, float(np.max(np.abs(replay - stored))))
        reports[split] = selection_metrics(
            replay, samples[split],
            random_seed=int(config["selection"]["random_seed"]),
            random_draws=int(config["selection"]["random_draws_per_state"]),
        )
    _require(maximum_error <= 1.0e-9, "L5 row01 prediction replay differs")
    _require(reports == result["metrics"], "L5 row01 metrics replay differs")
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "model_result_file_sha256": _file_sha256(result_path),
        "model_result_payload_sha256": result["result_payload_sha256"],
        "model_sha256": result["model"]["model_sha256"],
        "maximum_prediction_replay_error_m": maximum_error,
        "metrics": reports,
        "gates": result["gates"],
        "prediction_gate_pass": result["prediction_gate_pass"],
        "fresh_exact_replay_authorized": result["fresh_exact_replay_authorized"],
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
        "maximum_prediction_replay_error_m": output["maximum_prediction_replay_error_m"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
