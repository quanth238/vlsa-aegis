#!/usr/bin/env python3
"""Independent recomputation of the clean action-risk prediction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from main.multilink_ellipsoid.clean_action_risk import (
    MODEL_RESULT_SCHEMA, canonical, load_cases, load_config, prediction_metrics,
    sha256,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_clean_action_risk import arrays, load_dataset


def main(argv: Sequence[str] | None = None) -> int:
    import numpy as np
    import torch

    from main.multilink_ellipsoid.action_risk_model import (
        load_frozen_bundle, predict,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _git_identity(args.repo_root.resolve(), args.validator_commit)
    config = load_config(args.config.resolve())
    cases = load_cases(args.manifest.resolve(), config)
    result = _load(args.model_result.resolve())
    _require(result["schema_version"] == MODEL_RESULT_SCHEMA, "model result schema differs")
    _require(result["source"]["commit"] == args.producer_commit, "model producer differs")
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(sha256(canonical(payload)) == claimed, "model payload hash differs")
    by_split = load_dataset(args.run_root.resolve(), cases)
    bundle = load_frozen_bundle(torch, result["model"]["state_payload"], config["model"])
    reports = {}
    maximum_prediction_replay_error_m = 0.0
    for split in ("train", "validation", "test", "diagnostic"):
        x, _ = arrays(by_split[split])
        replay_prediction = predict(bundle, x)
        stored_prediction = np.asarray(result["predictions"][split], dtype=np.float64)
        _require(replay_prediction.shape == stored_prediction.shape, "model prediction shape differs")
        maximum_prediction_replay_error_m = max(
            maximum_prediction_replay_error_m,
            float(np.max(np.abs(replay_prediction - stored_prediction))),
        )
        reports[split] = prediction_metrics(replay_prediction, by_split[split])
    _require(maximum_prediction_replay_error_m <= 1.0e-9, "model prediction replay differs")
    _require(reports == result["metrics"], "model metric recomputation differs")
    test = reports["test"]
    gates = {
        "zero_observed_false_safe_candidates": test["false_safe_count"] == 0,
        "minimum_global_safe_recall": test["safe_recall"]
        >= float(config["prediction_gate"]["minimum_global_safe_recall"]),
        "safe_action_support_every_recoverable_state":
        test["supported_recoverable_state_count"] == test["recoverable_state_count"],
        "untouched_test_states_all_recoverable": test["recoverable_state_count"] == 3,
    }
    _require(gates == result["gates"], "model gates differ")
    _require(bool(all(gates.values())) is result["prediction_gate_pass"], "model pass differs")
    validation = {
        "schema_version": "vlsa_distal_clean_action_risk_model_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "model_result_file_sha256": _file_sha256(args.model_result.resolve()),
        "model_result_payload_sha256": claimed,
        "producer_commit": args.producer_commit,
        "validator_commit": args.validator_commit,
        "metrics": reports,
        "maximum_prediction_replay_error_m": maximum_prediction_replay_error_m,
        "gates": gates,
        "prediction_gate_pass": result["prediction_gate_pass"],
        "interpretation": result["interpretation"],
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    validation["validation_payload_sha256"] = sha256(canonical(validation))
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
