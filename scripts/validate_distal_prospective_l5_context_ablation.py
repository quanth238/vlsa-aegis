#!/usr/bin/env python3
"""Independently retrain and validate the prospective L5 context ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_prospective_l5_context_ablation import (
    arrays, feature_distribution_audit, load_ablation_samples, load_bundle,
    predict, train_arm,
)


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.generic_l5_9d_capacity import diagnostic_metrics
    from main.multilink_ellipsoid.prospective_l5_context_ablation import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, classify_ablation,
        load_config, payload_sha256,
    )

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == RESULT_SCHEMA, "context ablation result schema differs")
    _require(result["source"]["commit"] == expected_commit, "context ablation producer differs")
    _require(result["result_payload_sha256"] == payload_sha256(result, "result_payload_sha256"), "context ablation result payload differs")
    _require(result["config"] == config, "context ablation embedded config differs")
    samples, source_record, frozen_result = load_ablation_samples(repo_root=repo_root, config=config)
    _require(result["source_artifacts"] == source_record, "context ablation source record differs")
    base_model_config = frozen_result["config"]["model"]
    replay_maximum_error = 0.0
    retrain_maximum_error = 0.0
    replay_metrics: dict[str, Any] = {}
    independent_model_hashes: dict[str, str] = {}
    for arm_name, arm_config in config["arms"].items():
        stored_arm = result["arms"][arm_name]
        payload = stored_arm["model"]["state_payload"]
        _require(
            stored_arm["model"]["model_sha256"] == hashlib.sha256(canonical(payload)).hexdigest(),
            "context ablation model hash differs",
        )
        bundle = load_bundle(torch, payload, base_model_config)
        arm_metrics = {}
        for split in ("train", "validation", "test"):
            frozen = predict(bundle, arrays(samples[split], arm_name)[0])
            stored = np.asarray(stored_arm["predictions"][split], dtype=np.float64)
            replay_maximum_error = max(replay_maximum_error, float(np.max(np.abs(frozen - stored))))
            arm_metrics[split] = diagnostic_metrics(
                samples[split], frozen,
                near_boundary_abs_risk=float(frozen_result["config"]["metrics"]["near_boundary_abs_risk"]),
            )
            _require(arm_metrics[split] == stored_arm["metrics"][split], f"context ablation {arm_name} {split} metrics differ")
        _require(
            feature_distribution_audit(samples, arm_name) == stored_arm["feature_distribution_audit"],
            f"context ablation {arm_name} feature audit differs",
        )
        replay_metrics[arm_name] = arm_metrics
        retrained = train_arm(
            samples["train"], samples["validation"], arm_name=arm_name,
            model_config=base_model_config, input_dimension=int(arm_config["input_dimension"]),
        )
        independent_model_hashes[arm_name] = retrained["model_sha256"]
        _require(retrained["model_sha256"] == stored_arm["model"]["model_sha256"], f"context ablation {arm_name} independent model differs")
        retrain_predictions = {
            "train": retrained["train_prediction"],
            "validation": retrained["validation_prediction"],
            "test": predict(retrained["bundle"], arrays(samples["test"], arm_name)[0]).tolist(),
        }
        for split in ("train", "validation", "test"):
            retrain_maximum_error = max(retrain_maximum_error, float(np.max(np.abs(
                np.asarray(retrain_predictions[split], dtype=np.float64)
                - np.asarray(stored_arm["predictions"][split], dtype=np.float64)
            ))))
    _require(replay_maximum_error <= 1.0e-9, "context ablation frozen replay differs")
    _require(retrain_maximum_error <= 1.0e-9, "context ablation independent training differs")
    interpretation, checks = classify_ablation(
        replay_metrics["relative_endpoint_9D"],
        replay_metrics["direct_L5_OSC_33D"], config["decision"],
    )
    _require(interpretation == result["interpretation"] and checks == result["decision_checks"], "context ablation decision differs")
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
        "independent_model_hashes": independent_model_hashes,
        "metrics": replay_metrics,
        "decision_checks": checks,
        "interpretation": interpretation,
        "new_generalization_evidence": False,
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
