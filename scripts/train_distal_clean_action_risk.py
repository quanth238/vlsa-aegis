#!/usr/bin/env python3
"""Train the prediction-only seven-output clean action-risk MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.clean_action_risk import (
    MODEL_RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, load_cases, load_config,
    prediction_metrics, sha256,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def load_dataset(
    run_root: Path,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "test": [], "diagnostic": [],
    }
    for index, case in enumerate(cases):
        result = _load(run_root / ("case-%02d" % index) / "result.json")
        _require(result["case"]["case_id"] == case["case_id"], "model case identity differs")
        _require(result["training_authorized_for_case"] is True, "model case was not authorized")
        split = str(case["split"])
        for state in result["states"]:
            for candidate in state["candidates"]:
                by_split[split].append({
                    "case_id": case["case_id"],
                    "state_id": state["state_id"],
                    "feature_vector": candidate["feature_vector"],
                    "risk": candidate["risk"],
                    "exact_safe": candidate["exact_safe"],
                    "candidate_name": candidate["name"],
                })
    return by_split


def arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    import numpy as np

    return (
        np.asarray([item["feature_vector"] for item in samples], dtype=np.float64),
        np.asarray([item["risk"] for item in samples], dtype=np.float64),
    )


def train(
    *, repo_root: Path, run_root: Path, validation_path: Path,
    config_path: Path, manifest_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.action_risk_model import predict, train_model
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    cases = load_cases(manifest_path, config)
    validation = _load(validation_path)
    _require(validation["schema_version"] == VALIDATION_SCHEMA, "dataset validation schema differs")
    _require(validation["status"] == "validated", "dataset validation incomplete")
    _require(validation["MLP_training_authorized"] is True, "dataset did not authorize training")
    _require(all(validation["gates"].values()), "dataset validation gate failed")
    _require(len(validation["case_artifacts"]) == len(cases), "dataset validation cases differ")
    for index, artifact in enumerate(validation["case_artifacts"]):
        path = run_root / ("case-%02d" % index) / "result.json"
        _require(_file_sha256(path) == artifact["file_sha256"], "dataset artifact hash differs")
    by_split = load_dataset(run_root, cases)
    train_x, train_y = arrays(by_split["train"])
    validation_x, validation_y = arrays(by_split["validation"])
    bundle = train_model(
        train_x, train_y, validation_x, validation_y, config["model"]
    )
    reports = {}
    predictions = {}
    for split in ("train", "validation", "test", "diagnostic"):
        x, _ = arrays(by_split[split])
        prediction = predict(bundle, x)
        reports[split] = prediction_metrics(prediction, by_split[split])
        predictions[split] = prediction.tolist()
    test = reports["test"]
    gates = {
        "zero_observed_false_safe_candidates": test["false_safe_count"] == 0,
        "minimum_global_safe_recall": test["safe_recall"]
        >= float(config["prediction_gate"]["minimum_global_safe_recall"]),
        "safe_action_support_every_recoverable_state":
        test["supported_recoverable_state_count"] == test["recoverable_state_count"],
        "untouched_test_states_all_recoverable": test["recoverable_state_count"] == 12,
    }
    passed = all(gates.values())
    result = {
        "schema_version": MODEL_RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "source": source,
        "allocation": allocation_record(),
        "config": config,
        "dataset": {
            "validation_file_sha256": _file_sha256(validation_path),
            "validation_payload_sha256": validation["validation_payload_sha256"],
            "producer_commit": validation["producer_commit"],
            "split_sample_counts": {
                key: len(value) for key, value in by_split.items()
            },
            "diagnostic_excluded_from_training_and_test": True,
        },
        "model": {
            "best_epoch": bundle["best_epoch"],
            "epochs_completed": bundle["epochs_completed"],
            "best_validation_loss": bundle["best_validation_loss"],
            "parameter_count": bundle["parameter_count"],
            "model_sha256": bundle["model_sha256"],
            "device_name": bundle["device_name"],
            "state_payload": bundle["state_payload"],
        },
        "metrics": reports,
        "predictions": predictions,
        "gates": gates,
        "prediction_gate_pass": passed,
        "interpretation": (
            "clean_action_risk_prediction_pass_control_experiment_may_be_preregistered"
            if passed else "clean_action_risk_prediction_no_go"
        ),
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    result["result_payload_sha256"] = sha256(canonical(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = train(
        repo_root=args.repo_root.resolve(), run_root=args.run_root.resolve(),
        validation_path=args.validation.resolve(), config_path=args.config.resolve(),
        manifest_path=args.manifest.resolve(), expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "gates": result["gates"],
        "test_metrics": result["metrics"]["test"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
