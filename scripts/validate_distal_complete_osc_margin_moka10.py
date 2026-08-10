#!/usr/bin/env python3
"""Independently validate the complete-OSC paired-data and MLP result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.multilink_ellipsoid.complete_osc_margin import (
    COLLECTION_RESULT_SCHEMA, DATASET_SCHEMA, RESULT_SCHEMA, VALIDATION_SCHEMA,
    load_config, load_weights, payload_sha256, predict, prediction_metrics,
    training_arrays,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-result", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {
        "config": args.config, "dataset": args.dataset,
        "collection": args.collection_result, "result": args.result,
        "model": args.model, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    dataset = _load(paths["dataset"])
    collection = _load(paths["collection"])
    result = _load(paths["result"])
    _require(
        dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == payload_sha256(dataset, "dataset_payload_sha256")
        and collection.get("schema_version") == COLLECTION_RESULT_SCHEMA
        and collection.get("result_payload_sha256")
        == payload_sha256(collection, "result_payload_sha256")
        and result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit
        and collection.get("source", {}).get("commit") == args.expected_commit
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"],
        "complete-OSC result identity differs",
    )
    expected_pairs = int(config["immutable_source"]["expected_pair_count"])
    sufficiency = collection["input_sufficiency"]
    sufficiency_recomputed = bool(
        int(dataset["summary"]["pair_count"]) == expected_pairs
        and int(dataset["summary"]["replay_count"]) == expected_pairs * 2
        and int(dataset["summary"]["nondeterministic_pair_count"]) == 0
        and all(
            candidate["replay_deterministic"]
            for state in dataset["state_records"]
            for candidate in state["candidates"]
        )
    )
    _require(
        sufficiency_recomputed == bool(sufficiency["input_sufficiency_pass"])
        == bool(collection["decision"]["input_sufficiency_pass"]),
        "complete-OSC input sufficiency decision differs",
    )
    if not sufficiency_recomputed:
        _require(
            result["training"]["executed"] is False
            and result["decision"]["prediction_gate_pass"] is False
            and not paths["model"].exists(),
            "complete-OSC failed sufficiency still trained",
        )
        metrics = None
    else:
        _require(
            result["training"]["executed"] is True
            and paths["model"].is_file()
            and _file_sha256(paths["model"]) == result["model"]["file_sha256"],
            "complete-OSC model artifact differs",
        )
        arrays = training_arrays(dataset)
        models, model_state = load_weights(paths["model"])
        predictions = predict(
            models, model_state, arrays["features"], arrays["current_margin_m"]
        )
        metrics = prediction_metrics(arrays, predictions, config)
        recorded = result["prediction_metrics"]
        for key in (
            "test_state_count", "test_action_count",
            "test_proxy_false_safe_action_count",
            "test_state_safe_support_count", "test_near_boundary_row_count",
            "prediction_gate_pass",
        ):
            _require(metrics[key] == recorded[key], "complete-OSC metric differs: %s" % key)
        for key in (
            "test_exact_safe_action_recall", "test_overall_RMSE_m",
            "test_near_boundary_RMSE_m",
        ):
            _require(abs(float(metrics[key]) - float(recorded[key])) <= 1.0e-12, "complete-OSC metric differs: %s" % key)
        _require(
            bool(metrics["prediction_gate_pass"])
            == bool(result["decision"]["prediction_gate_pass"])
            and result["forbidden_action_receipt"] == {
                "calibration_executed": False, "QP_executed": False,
                "closed_loop_E05_executed": False, "pi05_features_used": False,
            },
            "complete-OSC decision or forbidden action receipt differs",
        )
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "dataset_file_sha256": _file_sha256(paths["dataset"]),
        "dataset_payload_sha256": dataset["dataset_payload_sha256"],
        "input_sufficiency_pass": sufficiency_recomputed,
        "prediction_gate_pass": bool(result["decision"]["prediction_gate_pass"]),
        "separate_calibration_QP_authorized": bool(
            result["decision"]["separate_calibration_QP_authorized"]
        ),
        "recomputed_prediction_metrics": metrics,
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
