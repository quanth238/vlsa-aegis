#!/usr/bin/env python3
"""Independently retrain and validate the prospective L5 Q-only diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_prospective_l5_q_only_diagnostic import load_samples


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.generic_l5_9d_capacity import diagnostic_metrics
    from main.multilink_ellipsoid.prospective_l5_q_only_diagnostic import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, classify_diagnostic,
        load_config, payload_sha256,
    )
    from scripts.train_distal_generic_l5_9d_capacity import (
        arrays, load_bundle, predict, train_fold,
    )

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == RESULT_SCHEMA, "prospective Q-only result schema differs")
    _require(result["source"]["commit"] == expected_commit, "prospective Q-only producer differs")
    _require(result["result_payload_sha256"] == payload_sha256(result, "result_payload_sha256"), "prospective Q-only result payload differs")
    _require(result["config"] == config, "prospective Q-only embedded config differs")
    samples, source_record = load_samples(config)
    _require(result["source_artifacts"] == source_record, "prospective Q-only source record differs")
    payload = result["model"]["state_payload"]
    _require(result["model"]["model_sha256"] == hashlib.sha256(canonical(payload)).hexdigest(), "prospective Q-only model hash differs")
    bundle = load_bundle(torch, payload, config["model"])
    replay_maximum_error = 0.0
    replay_metrics = {}
    for split in ("train", "validation", "test"):
        frozen = predict(bundle, arrays(samples[split])[0])
        stored = np.asarray(result["predictions"][split], dtype=np.float64)
        replay_maximum_error = max(replay_maximum_error, float(np.max(np.abs(frozen - stored))))
        replay_metrics[split] = diagnostic_metrics(
            samples[split], frozen,
            near_boundary_abs_risk=float(config["metrics"]["near_boundary_abs_risk"]),
        )
        _require(replay_metrics[split] == result["metrics"][split], f"prospective Q-only {split} metrics differ")
    retrained = train_fold(samples["train"], samples["validation"], config["model"])
    _require(retrained["model_sha256"] == result["model"]["model_sha256"], "prospective Q-only independent model differs")
    retrain_maximum_error = 0.0
    retrain_predictions = {
        "train": retrained["train_prediction"],
        "validation": retrained["validation_prediction"],
        "test": predict(retrained["bundle"], arrays(samples["test"])[0]).tolist(),
    }
    for split in ("train", "validation", "test"):
        retrain_maximum_error = max(
            retrain_maximum_error,
            float(np.max(np.abs(
                np.asarray(retrain_predictions[split], dtype=np.float64)
                - np.asarray(result["predictions"][split], dtype=np.float64)
            ))),
        )
    _require(replay_maximum_error <= 1.0e-9, "prospective Q-only frozen replay differs")
    _require(retrain_maximum_error <= 1.0e-9, "prospective Q-only independent training differs")
    interpretation, checks = classify_diagnostic(
        replay_metrics["validation"], replay_metrics["test"], config["decision"],
    )
    _require(interpretation == result["interpretation"] and checks == result["decision_checks"], "prospective Q-only decision differs")
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "source": identity,
        "producer_commit": expected_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "frozen_prediction_replay_maximum_error": replay_maximum_error,
        "independent_retrain_prediction_maximum_error": retrain_maximum_error,
        "metrics": replay_metrics,
        "decision_checks": checks,
        "interpretation": interpretation,
        "training_dataset_gate_pass": False,
        "correction_authorized": False,
        "QP_authorized": False,
        "calibration_authorized": False,
        "closed_loop_authorized": False,
    }
    output["validation_payload_sha256"] = payload_sha256(output, "validation_payload_sha256")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root.resolve(), result_path=args.result.resolve(),
        config_path=args.config.resolve(), expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "interpretation": output["interpretation"],
        "metrics": output["metrics"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
