#!/usr/bin/env python3
"""Independently validate the matched direct-L5 Q versus Q-plus-V diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)
from scripts.train_distal_generic_l5_qv_diagnostic import (
    load_bundle, load_samples, predict_q, q_arrays, train_arm,
)


def validate(
    *, repo_root: Path, config_path: Path, result_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.generic_l5_qv_diagnostic import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, diagnostic_metrics, load_config,
        payload_sha256,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("generic L5 Q/V validation requires exactly one H100")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("generic L5 Q/V validation requires an H100")
    config = load_config(config_path)
    result = _load(result_path)
    _require(
        result["schema_version"] == RESULT_SCHEMA
        and result["source_commit"] == expected_commit
        and result["config_file_sha256"] == config["config_file_sha256"]
        and result["config_payload_sha256"] == config["config_payload_sha256"]
        and result["result_payload_sha256"]
        == payload_sha256(result, "result_payload_sha256"),
        "generic L5 Q/V result binding differs",
    )
    cases, source_record = load_samples(config)
    _require(source_record == result["source"], "generic L5 Q/V source record differs")
    maximum_frozen_error = 0.0
    maximum_retrain_error = 0.0
    aggregate = {arm: {"samples": [], "predictions": []} for arm in config["model"]["arms"]}
    for fold in result["folds"]:
        held_out = str(fold["held_out_state_id"])
        training_ids = [str(item) for item in fold["training_state_ids"]]
        q_train = [sample for case_id in training_ids for sample in cases[case_id]["query"]]
        v_train = [sample for case_id in training_ids for sample in cases[case_id]["value"]]
        q_validation = cases[held_out]["query"]
        validation_state, validation_action, _, _ = q_arrays(q_validation)
        for arm in config["model"]["arms"]:
            stored = fold["arms"][arm]
            bundle = load_bundle(torch, stored["model"]["state_payload"], config["model"])
            frozen = predict_q(bundle, validation_state, validation_action)
            stored_prediction = np.asarray(stored["validation_prediction"], dtype=np.float64)
            maximum_frozen_error = max(
                maximum_frozen_error,
                float(np.max(np.abs(frozen - stored_prediction))),
            )
            _require(
                diagnostic_metrics(q_validation, frozen.tolist())
                == stored["validation_metrics"],
                "generic L5 Q/V frozen metrics differ",
            )
            retrained = train_arm(
                q_train, v_train, q_validation, config["model"], arm=arm,
            )
            retrained_prediction = np.asarray(
                retrained["validation_prediction"], dtype=np.float64,
            )
            maximum_retrain_error = max(
                maximum_retrain_error,
                float(np.max(np.abs(retrained_prediction - stored_prediction))),
            )
            _require(
                retrained["model_sha256"] == stored["model"]["model_sha256"],
                "generic L5 Q/V independent model differs",
            )
            aggregate[arm]["samples"].extend(q_validation)
            aggregate[arm]["predictions"].extend(stored_prediction.tolist())
    aggregate_metrics = {
        arm: diagnostic_metrics(record["samples"], record["predictions"])
        for arm, record in aggregate.items()
    }
    _require(aggregate_metrics == result["aggregate_metrics"], "generic L5 Q/V aggregate metrics differ")
    boundary_ids = [str(item) for item in config["dataset"]["boundary_case_ids"]]
    auxiliary_ids = [str(item) for item in config["dataset"]["safe_auxiliary_case_ids"]]
    recovery_ids = [str(item) for item in config["dataset"]["recovery_case_ids"]]
    fit_ids = boundary_ids + auxiliary_ids
    q_train = [sample for case_id in fit_ids for sample in cases[case_id]["query"]]
    v_train = [sample for case_id in fit_ids for sample in cases[case_id]["value"]]
    q_recovery = [sample for case_id in recovery_ids for sample in cases[case_id]["query"]]
    for arm in config["model"]["arms"]:
        stored = result["recovery_diagnostic"][arm]
        retrained = train_arm(q_train, v_train, q_recovery, config["model"], arm=arm)
        difference = np.max(np.abs(
            np.asarray(retrained["validation_prediction"], dtype=np.float64)
            - np.asarray(stored["prediction"], dtype=np.float64)
        ))
        maximum_retrain_error = max(maximum_retrain_error, float(difference))
        _require(
            retrained["model_sha256"] == stored["model"]["model_sha256"],
            "generic L5 Q/V recovery model differs",
        )
    _require(maximum_frozen_error <= 1.0e-9, "generic L5 Q/V frozen prediction differs")
    _require(maximum_retrain_error <= 1.0e-9, "generic L5 Q/V independent retraining differs")
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "claim_scope": config["claim_scope"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "result_file": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "source_commit": expected_commit,
        "maximum_frozen_prediction_error": maximum_frozen_error,
        "maximum_independent_retrain_prediction_error": maximum_retrain_error,
        "aggregate_metrics": aggregate_metrics,
        "q_plus_v_mechanism_pass": bool(result["q_plus_v_mechanism_pass"]),
        "interpretation": result["interpretation"],
        "training_dataset_gate_pass": False,
        "correction_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "validation_payload_sha256": "",
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256",
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root, config_path=args.config, result_path=args.result,
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output, output)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
