#!/usr/bin/env python3
"""Run the preregistered old56 versus complete-OSC input ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.complete_osc_margin import (
    COLLECTION_RESULT_SCHEMA, DATASET_SCHEMA, payload_sha256, predict,
    train_ensemble, training_arrays,
)
from main.multilink_ellipsoid.matched_input_ablation import (
    RESULT_SCHEMA, load_config, matched_decision, old56_training_arrays,
    paired_row_receipt, save_arm_weights, split_metrics,
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
    parser.add_argument("--old56-model", type=Path, required=True)
    parser.add_argument("--complete-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config, "dataset": args.dataset,
        "collection": args.collection_result, "old56_model": args.old56_model,
        "complete_model": args.complete_model, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    dataset = _load(paths["dataset"])
    collection = _load(paths["collection"])
    immutable = config["immutable_source"]
    _require(
        _file_sha256(paths["dataset"]) == immutable["dataset_file_sha256"]
        and dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == immutable["dataset_payload_sha256"]
        == payload_sha256(dataset, "dataset_payload_sha256")
        and _file_sha256(paths["collection"])
        == immutable["collection_result_file_sha256"]
        and collection.get("schema_version") == COLLECTION_RESULT_SCHEMA
        and collection.get("result_payload_sha256")
        == immutable["collection_result_payload_sha256"]
        == payload_sha256(collection, "result_payload_sha256")
        and bool(collection.get("decision", {}).get("input_sufficiency_pass"))
        and int(dataset["summary"]["state_count"])
        == int(immutable["expected_state_count"])
        and int(dataset["summary"]["pair_count"])
        == int(immutable["expected_candidate_count"]),
        "matched-input immutable source differs",
    )
    source = _git_identity(paths["repo"], args.expected_commit)
    allocation = allocation_record()
    torch.set_num_threads(8)
    complete_arrays = training_arrays(dataset)
    old_arrays, reconstruction = old56_training_arrays(
        dataset,
        float(config["paired_arms"]["old56_geometry_reconstruction_tolerance_m"]),
    )
    receipt = paired_row_receipt(old_arrays, complete_arrays)
    _require(
        receipt["exact_row_identity_and_label_match"]
        and receipt["row_count"] == int(immutable["expected_row_count"])
        and old_arrays["features"].shape[1]
        == int(config["paired_arms"]["old56"]["dimension"])
        and complete_arrays["features"].shape[1]
        == int(config["paired_arms"]["completeOSC"]["dimension"]),
        "matched-input row pairing differs",
    )
    arms = {}
    arm_metrics = {}
    arm_paths = {
        "old56": paths["old56_model"], "completeOSC": paths["complete_model"],
    }
    arrays_by_arm = {"old56": old_arrays, "completeOSC": complete_arrays}
    for name in ("old56", "completeOSC"):
        arm_started = time.perf_counter_ns()
        arrays = arrays_by_arm[name]
        models, model_state, training_audit = train_ensemble(arrays, config)
        prediction = predict(
            models, model_state, arrays["features"], arrays["current_margin_m"]
        )
        metrics = {
            split: split_metrics(arrays, prediction, config, split)
            for split in ("train", "validation", "test")
        }
        model_artifact = save_arm_weights(arm_paths[name], model_state)
        arms[name] = {
            "input_dimension": int(arrays["features"].shape[1]),
            "training": training_audit, "model": model_artifact,
            "metrics": metrics,
            "wall_seconds": (time.perf_counter_ns() - arm_started) * 1.0e-9,
        }
        arm_metrics[name] = metrics
    decision = matched_decision(arm_metrics)
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": source, "allocation": allocation, "config": config,
        "immutable_dataset": {
            "path": str(paths["dataset"]),
            "file_sha256": _file_sha256(paths["dataset"]),
            "payload_sha256": dataset["dataset_payload_sha256"],
        },
        "paired_row_receipt": receipt,
        "old56_reconstruction": reconstruction,
        "arms": arms, "decision": decision,
        "forbidden_action_receipt": {
            "additional_simulation_executed": False,
            "new_data_collection_executed": False,
            "poisson_fields_used": False, "calibration_executed": False,
            "QP_executed": False, "closed_loop_E05_executed": False,
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "paired_row_receipt": receipt, "decision": decision,
        "old56": arm_metrics["old56"],
        "completeOSC": arm_metrics["completeOSC"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
