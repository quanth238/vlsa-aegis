#!/usr/bin/env python3
"""Independent replay of the frozen three-output L5 prediction gate."""

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
        MODEL_SCHEMA, canonical, load_config, load_frozen_bundle,
        payload_sha256, predict, prediction_metrics,
    )

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == MODEL_SCHEMA, "L5 model schema differs")
    _require(result["source"]["commit"] == producer_commit,
             "L5 model producer differs")
    _require(result["result_payload_sha256"] == payload_sha256(result),
             "L5 model payload differs")
    _require(result["config"] == config, "L5 model config differs")
    _require(result["MLP_scope"] == "L5_rows_0_1_2_only",
             "L5 model scope differs")
    _require(result["rows_3_to_6"]
             == "exact_diagnostic_and_physical_veto_only",
             "L5 model physical veto differs")
    samples = _load_samples(config)
    state_payload = result["model"]["state_payload"]
    _require(
        result["model"]["model_sha256"]
        == hashlib.sha256(canonical(state_payload)).hexdigest(),
        "L5 model hash differs",
    )
    bundle = load_frozen_bundle(torch, state_payload, config["model"])
    reports = {}
    maximum_prediction_replay_error_m = 0.0
    for split_name in ("train", "validation"):
        features, _ = _arrays(samples[split_name])
        replay = predict(bundle, features)
        stored = np.asarray(result["predictions"][split_name], dtype=np.float64)
        _require(replay.shape == stored.shape, "L5 prediction shape differs")
        maximum_prediction_replay_error_m = max(
            maximum_prediction_replay_error_m,
            float(np.max(np.abs(replay - stored))),
        )
        reports[split_name] = prediction_metrics(replay, samples[split_name])
    _require(maximum_prediction_replay_error_m <= 1.0e-9,
             "L5 prediction replay differs")
    _require(reports == result["metrics"], "L5 metric replay differs")
    validation_metrics = reports["validation"]
    gates = {
        "zero_observed_L5_false_safe_candidates":
        validation_metrics["L5_false_safe_count"] == 0,
        "minimum_exact_safe_candidate_recall":
        validation_metrics["exact_safe_candidate_recall"]
        >= float(config["prediction_gate"][
            "minimum_exact_safe_candidate_recall"
        ]),
        "safe_action_support_every_recoverable_state":
        validation_metrics["supported_recoverable_state_count"]
        == validation_metrics["recoverable_state_count"],
        "validation_contains_recoverable_states":
        validation_metrics["recoverable_state_count"] > 0,
        "selected_candidates_require_exact_all_seven_physical_acceptance": True,
        "all_selected_candidates_exact_all_seven_safe":
        validation_metrics["all_selected_candidates_exact_all_seven_safe"],
        "reserved_test_episodes_unopened": True,
    }
    _require(gates == result["gates"], "L5 prediction gates differ")
    _require(bool(all(gates.values())) is result["prediction_gate_pass"],
             "L5 prediction pass differs")
    output = {
        "schema_version": "vlsa_distal_l5_action_risk_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "model_result_file_sha256": _file_sha256(result_path),
        "model_result_payload_sha256": result["result_payload_sha256"],
        "model_sha256": result["model"]["model_sha256"],
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "metrics": reports,
        "maximum_prediction_replay_error_m":
        maximum_prediction_replay_error_m,
        "gates": gates,
        "prediction_gate_pass": result["prediction_gate_pass"],
        "interpretation": result["interpretation"],
        "QP_authorized": False,
        "closed_loop_authorized": False,
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
    result = validate(
        repo_root=args.repo_root.resolve(),
        result_path=args.model_result.resolve(),
        config_path=args.config.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "gates": result["gates"],
        "maximum_prediction_replay_error_m":
        result["maximum_prediction_replay_error_m"],
        "validation_payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
