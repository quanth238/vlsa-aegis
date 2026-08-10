#!/usr/bin/env python3
"""Independently recompute the matched-input ablation result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.multilink_ellipsoid.complete_osc_margin import (
    DATASET_SCHEMA, payload_sha256, predict, training_arrays,
)
from main.multilink_ellipsoid.matched_input_ablation import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_arm_weights, load_config,
    matched_decision, old56_training_arrays, paired_row_receipt, split_metrics,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--old56-model", type=Path, required=True)
    parser.add_argument("--complete-model", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {
        "config": args.config, "dataset": args.dataset, "result": args.result,
        "old56": args.old56_model, "completeOSC": args.complete_model,
        "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    dataset = _load(paths["dataset"])
    result = _load(paths["result"])
    source = config["immutable_source"]
    _require(
        _file_sha256(paths["dataset"]) == source["dataset_file_sha256"]
        and dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source["dataset_payload_sha256"]
        and result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"],
        "matched-input result identity differs",
    )
    complete_arrays = training_arrays(dataset)
    old_arrays, reconstruction = old56_training_arrays(
        dataset,
        float(config["paired_arms"]["old56_geometry_reconstruction_tolerance_m"]),
    )
    receipt = paired_row_receipt(old_arrays, complete_arrays)
    _require(
        receipt == result["paired_row_receipt"]
        and reconstruction == result["old56_reconstruction"],
        "matched-input pairing receipt differs",
    )
    arrays_by_arm = {"old56": old_arrays, "completeOSC": complete_arrays}
    metrics = {}
    for name in ("old56", "completeOSC"):
        _require(
            _file_sha256(paths[name]) == result["arms"][name]["model"]["file_sha256"],
            "matched-input model differs: %s" % name,
        )
        models, model_state = load_arm_weights(paths[name])
        arrays = arrays_by_arm[name]
        prediction = predict(
            models, model_state, arrays["features"], arrays["current_margin_m"]
        )
        metrics[name] = {
            split: split_metrics(arrays, prediction, config, split)
            for split in ("train", "validation", "test")
        }
        _require(
            metrics[name] == result["arms"][name]["metrics"],
            "matched-input metrics differ: %s" % name,
        )
    decision = matched_decision(metrics)
    _require(
        decision == result["decision"]
        and result["forbidden_action_receipt"] == {
            "additional_simulation_executed": False,
            "new_data_collection_executed": False,
            "poisson_fields_used": False, "calibration_executed": False,
            "QP_executed": False, "closed_loop_E05_executed": False,
        },
        "matched-input decision or forbidden receipt differs",
    )
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "dataset_file_sha256": _file_sha256(paths["dataset"]),
        "dataset_payload_sha256": dataset["dataset_payload_sha256"],
        "paired_row_receipt": receipt, "recomputed_metrics": metrics,
        "recomputed_decision": decision,
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
