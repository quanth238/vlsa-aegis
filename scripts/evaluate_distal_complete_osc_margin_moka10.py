#!/usr/bin/env python3
"""Train and test the complete-input action-conditioned margin MLP only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.complete_osc_margin import (
    COLLECTION_RESULT_SCHEMA, DATASET_SCHEMA, RESULT_SCHEMA, load_config,
    payload_sha256, predict, prediction_metrics, save_weights, train_ensemble,
    training_arrays,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def main() -> int:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config, "dataset": args.dataset,
        "collection": args.collection_result, "model": args.model,
        "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    dataset = _load(paths["dataset"])
    collection = _load(paths["collection"])
    _require(
        dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == payload_sha256(dataset, "dataset_payload_sha256")
        and dataset.get("config_file_sha256") == config["config_file_sha256"]
        and collection.get("schema_version") == COLLECTION_RESULT_SCHEMA
        and collection.get("result_payload_sha256")
        == payload_sha256(collection, "result_payload_sha256")
        and collection.get("dataset", {}).get("file_sha256") == _file_sha256(paths["dataset"])
        and collection.get("dataset", {}).get("payload_sha256") == dataset["dataset_payload_sha256"]
        and collection.get("source", {}).get("commit") == args.expected_commit
        and dataset.get("source_commit") == args.expected_commit,
        "complete-OSC collection identity differs",
    )
    source = _git_identity(paths["repo"], args.expected_commit)
    allocation = allocation_record()
    if not bool(collection.get("decision", {}).get("input_sufficiency_pass")):
        result = {
            "schema_version": RESULT_SCHEMA, "status": "complete",
            "scientific_result": True, "claim_scope": config["claim_scope"],
            "source": source, "allocation": allocation, "config": config,
            "collection_result": {
                "path": str(paths["collection"]),
                "file_sha256": _file_sha256(paths["collection"]),
                "payload_sha256": collection["result_payload_sha256"],
            },
            "dataset": {
                "path": str(paths["dataset"]),
                "file_sha256": _file_sha256(paths["dataset"]),
                "payload_sha256": dataset["dataset_payload_sha256"],
            },
            "training": {"executed": False, "reason": "input_sufficiency_gate_failed"},
            "prediction_metrics": None,
            "decision": {
                "input_sufficiency_pass": False,
                "prediction_gate_pass": False,
                "separate_calibration_QP_authorized": False,
                "conclusion": "recorded_inputs_do_not_uniquely_determine_rollout",
            },
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = payload_sha256(result, "result_payload_sha256")
        _atomic_write(paths["output"], result)
        print(json.dumps(result["decision"], sort_keys=True), flush=True)
        return 0
    _require(
        int(dataset["summary"]["nondeterministic_pair_count"]) == 0
        and int(dataset["summary"]["pair_count"])
        == int(config["immutable_source"]["expected_pair_count"]),
        "complete-OSC input sufficiency receipt differs",
    )
    torch.set_num_threads(8)
    arrays = training_arrays(dataset)
    models, model_state, training_audit = train_ensemble(arrays, config)
    predictions = predict(models, model_state, arrays["features"], arrays["current_margin_m"])
    metrics = prediction_metrics(arrays, predictions, config)
    model_artifact = save_weights(paths["model"], model_state)
    test_mask = np.asarray(arrays["split"], dtype=object) == "test"
    test_predictions = {
        "state_index": np.asarray(arrays["state_index"])[test_mask].astype(int).tolist(),
        "action_index": np.asarray(arrays["action_index"])[test_mask].astype(int).tolist(),
        "constraint_index": np.asarray(arrays["constraint_index"])[test_mask].astype(int).tolist(),
        "predicted_margin_m": np.asarray(predictions)[test_mask].tolist(),
        "exact_margin_m": np.asarray(arrays["margin_m"])[test_mask].tolist(),
    }
    gate_pass = bool(metrics["prediction_gate_pass"])
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": source, "allocation": allocation, "config": config,
        "collection_result": {
            "path": str(paths["collection"]),
            "file_sha256": _file_sha256(paths["collection"]),
            "payload_sha256": collection["result_payload_sha256"],
        },
        "dataset": {
            "path": str(paths["dataset"]),
            "file_sha256": _file_sha256(paths["dataset"]),
            "payload_sha256": dataset["dataset_payload_sha256"],
            "complete_input_dimension": int(dataset["complete_input_dimension"]),
            "model_input_dimension": int(model_state["input_dimension"]),
        },
        "training": {"executed": True, **training_audit},
        "model": model_artifact,
        "prediction_metrics": metrics,
        "test_predictions": test_predictions,
        "forbidden_action_receipt": {
            "calibration_executed": False, "QP_executed": False,
            "closed_loop_E05_executed": False, "pi05_features_used": False,
        },
        "decision": {
            "input_sufficiency_pass": True,
            "prediction_gate_pass": gate_pass,
            "separate_calibration_QP_authorized": gate_pass,
            "conclusion": (
                "complete_OSC_inputs_sufficient_for_registered_prediction_gate"
                if gate_pass else
                "missing_OSC_inputs_not_sufficient_model_or_data_representation_remains_blocker"
            ),
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(result, "result_payload_sha256")
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "decision": result["decision"], "prediction_metrics": metrics,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
