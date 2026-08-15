#!/usr/bin/env python3
"""Train the 12D relative endpoint plus current L5 clearance ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _file_sha256, _git_identity, _load, _require
from scripts.train_distal_l5_row01_input_ablation import _model_config, _model_record
from scripts.train_distal_l5_row01_obstacle_endpoint_ablation import load_samples as load_9d_samples


def load_samples(config: Mapping[str, Any], repo_root: Path) -> tuple[Any, Any, Any]:
    from main.multilink_ellipsoid.l5_row01_clearance_endpoint_ablation import clearance_endpoint_feature_vector
    from main.multilink_ellipsoid.l5_row01_obstacle_endpoint_ablation import payload_sha256 as prior_payload_sha256

    samples, complete_frozen, _endpoint_frozen = load_9d_samples(config, repo_root)
    source = config["source"]
    prior_path = Path(source["frozen_obstacle_endpoint_result"])
    _require(_file_sha256(prior_path) == source["frozen_obstacle_endpoint_result_file_sha256"], "clearance frozen 9D file differs")
    prior = _load(prior_path)
    _require(prior["result_payload_sha256"] == source["frozen_obstacle_endpoint_result_payload_sha256"] == prior_payload_sha256(prior), "clearance frozen 9D payload differs")
    cache = {}
    for split in ("train", "validation"):
        for sample in samples[split]:
            path = Path(sample["source_result_path"])
            if path not in cache:
                cache[path] = _load(path)
            rows = cache[path]["state"]["physical_context"]["geometry_rows"]
            _require(
                len(rows) == 7
                and all(row["body_name"] == "robot0_link5" for row in rows[:3]),
                "clearance L5 rows differ",
            )
            clearances = [float(row["current_clearance_m"]) for row in rows[:3]]
            sample["clearance_endpoint_feature_vector"] = clearance_endpoint_feature_vector(
                obstacle_endpoint_feature=sample["obstacle_endpoint_feature_vector"],
                current_L5_clearances_m=clearances,
            )
    return samples, complete_frozen, prior


def arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    import numpy as np
    return np.asarray([item["clearance_endpoint_feature_vector"] for item in samples], dtype=np.float64), np.asarray([item["risk_row01"] for item in samples], dtype=np.float64)


def run(*, repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_row01_clearance_endpoint_ablation import INPUT_DIMENSION, RESULT_SCHEMA, classify, load_config, payload_sha256
    from main.multilink_ellipsoid.l5_row01_input_ablation import ablation_metrics
    from main.multilink_ellipsoid.l5_row01_selection import predict, train_model
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples, complete_frozen, prior = load_samples(config, repo_root)
    train_x, train_y = arrays(samples["train"])
    validation_x, validation_y = arrays(samples["validation"])
    _require(train_x.shape == (43, INPUT_DIMENSION) and validation_x.shape == (29, INPUT_DIMENSION), "clearance arrays differ")
    model_config = _model_config(complete_frozen["config"], INPUT_DIMENSION)
    model_config.update(config["matched_model"])
    bundle = train_model(train_x, train_y, validation_x, validation_y, model_config)
    predictions, metrics = {}, {}
    for split, features in (("train", train_x), ("validation", validation_x)):
        prediction = predict(bundle, features)
        predictions[split] = prediction.tolist()
        metrics[split] = ablation_metrics(prediction, samples[split], random_seed=20260814, random_draws=1024)
    prior_metrics = prior["obstacle_relative_endpoint_9D"]["metrics"]
    interpretation, comparison = classify(metrics["validation"], prior_metrics["validation"], minimum_relative_rmse_improvement=float(config["decision"]["minimum_relative_validation_RMSE_improvement_over_9D"]))
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete", "scientific_result": True,
        "claim_scope": config["claim_scope"], "source": identity, "allocation": allocation_record(), "config": config,
        "dataset": {"train_label_count": len(samples["train"]), "grouped_validation_label_count": len(samples["validation"]), "labels_and_splits_identical": True, "reserved_test_episodes_unopened": True},
        "clearance_endpoint_12D": {"model_config": model_config, "model": _model_record(bundle), "predictions": predictions, "metrics": metrics},
        "frozen_obstacle_endpoint_9D_metrics": prior_metrics,
        "comparison": comparison, "interpretation": interpretation,
        "prediction_or_control_gate_pass": False, "fresh_exact_replay_authorized": False, "QP_authorized": False, "closed_loop_authorized": False,
    }
    result["result_payload_sha256"] = payload_sha256(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(repo_root=args.repo_root.resolve(), config_path=args.config.resolve(), expected_commit=args.expected_commit)
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({"interpretation": result["interpretation"], "comparison": result["comparison"], "result_payload_sha256": result["result_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
