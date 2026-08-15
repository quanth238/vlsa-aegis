#!/usr/bin/env python3
"""Independently validate the generic L5 9D capacity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_generic_l5_9d_capacity import (
    arrays, load_bundle, load_samples, predict, train_fold,
)


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path,
    producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.generic_l5_9d_capacity import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, diagnostic_metrics,
        load_config, payload_sha256,
    )

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == RESULT_SCHEMA, "generic L5 result schema differs")
    _require(result["source"]["commit"] == producer_commit, "generic L5 producer differs")
    _require(
        result["result_payload_sha256"]
        == payload_sha256(result, "result_payload_sha256"),
        "generic L5 result payload differs",
    )
    _require(result["config"] == config, "generic L5 embedded config differs")
    grouped, recovery_samples, source_record = load_samples(config)
    _require(result["source_artifacts"] == source_record, "generic L5 source record differs")
    replay_maximum_error = 0.0
    retrain_maximum_error = 0.0
    aggregate_samples = []
    aggregate_predictions = []
    for stored, held_out in zip(result["folds"], config["dataset"]["prevention_case_ids"]):
        _require(stored["held_out_state_id"] == held_out, "generic L5 fold order differs")
        train_samples = [
            sample for state_id, samples in grouped.items() if state_id != held_out
            for sample in samples
        ]
        validation_samples = list(grouped[held_out])
        model_record = stored["model"]
        payload = model_record["state_payload"]
        _require(
            model_record["model_sha256"] == hashlib.sha256(canonical(payload)).hexdigest(),
            "generic L5 model hash differs",
        )
        bundle = load_bundle(torch, payload, config["model"])
        frozen_train_prediction = predict(bundle, arrays(train_samples)[0])
        stored_train_prediction = np.asarray(stored["train_prediction"], dtype=np.float64)
        replay_maximum_error = max(
            replay_maximum_error,
            float(np.max(np.abs(frozen_train_prediction - stored_train_prediction))),
        )
        train_metrics = diagnostic_metrics(
            train_samples, frozen_train_prediction,
            near_boundary_abs_risk=float(config["metrics"]["near_boundary_abs_risk"]),
        )
        _require(train_metrics == stored["train_metrics"], "generic L5 train metrics differ")
        frozen_prediction = predict(bundle, arrays(validation_samples)[0])
        stored_prediction = np.asarray(stored["validation_prediction"], dtype=np.float64)
        replay_maximum_error = max(
            replay_maximum_error,
            float(np.max(np.abs(frozen_prediction - stored_prediction))),
        )
        metrics = diagnostic_metrics(
            validation_samples, frozen_prediction,
            near_boundary_abs_risk=float(config["metrics"]["near_boundary_abs_risk"]),
        )
        _require(metrics == stored["metrics"], "generic L5 fold metrics differ")
        retrained = train_fold(train_samples, validation_samples, config["model"])
        retrained_prediction = np.asarray(
            retrained["validation_prediction"], dtype=np.float64,
        )
        retrain_maximum_error = max(
            retrain_maximum_error,
            float(np.max(np.abs(retrained_prediction - stored_prediction))),
        )
        _require(
            retrained["model_sha256"] == model_record["model_sha256"],
            "generic L5 independent model differs",
        )
        aggregate_samples.extend(validation_samples)
        aggregate_predictions.extend(frozen_prediction.tolist())
    aggregate = diagnostic_metrics(
        aggregate_samples, aggregate_predictions,
        near_boundary_abs_risk=float(config["metrics"]["near_boundary_abs_risk"]),
    )
    _require(
        aggregate == result["leave_one_state_out_metrics"],
        "generic L5 aggregate metrics differ",
    )
    all_prevention = [sample for samples in grouped.values() for sample in samples]
    recovery_record = result["recovery_diagnostic"]
    recovery_bundle = load_bundle(
        torch, recovery_record["model"]["state_payload"], config["model"],
    )
    recovery_prediction = predict(recovery_bundle, arrays(recovery_samples)[0])
    recovery_stored = np.asarray(recovery_record["prediction"], dtype=np.float64)
    replay_maximum_error = max(
        replay_maximum_error,
        float(np.max(np.abs(recovery_prediction - recovery_stored))),
    )
    recovery_metrics = diagnostic_metrics(
        recovery_samples, recovery_prediction,
        near_boundary_abs_risk=float(config["metrics"]["near_boundary_abs_risk"]),
    )
    _require(recovery_metrics == recovery_record["metrics"], "recovery metrics differ")
    recovery_retrained = train_fold(all_prevention, recovery_samples, config["model"])
    retrain_maximum_error = max(
        retrain_maximum_error,
        float(np.max(np.abs(
            np.asarray(recovery_retrained["validation_prediction"], dtype=np.float64)
            - recovery_stored
        ))),
    )
    _require(
        recovery_retrained["model_sha256"] == recovery_record["model"]["model_sha256"],
        "recovery independent model differs",
    )
    _require(replay_maximum_error <= 1.0e-9, "generic L5 frozen replay differs")
    _require(retrain_maximum_error <= 1.0e-9, "generic L5 independent training differs")
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "validated",
        "scientific_result": True, "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "frozen_prediction_replay_maximum_error": replay_maximum_error,
        "independent_retrain_prediction_maximum_error": retrain_maximum_error,
        "leave_one_state_out_metrics": aggregate,
        "recovery_metrics": recovery_metrics,
        "interpretation": "validated_capacity_diagnostic_only_no_control_authority",
        "training_dataset_gate_pass": False,
        "correction_authorized": False, "QP_authorized": False,
        "calibration_authorized": False, "closed_loop_authorized": False,
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256",
    )
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
        "metrics": output["leave_one_state_out_metrics"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
