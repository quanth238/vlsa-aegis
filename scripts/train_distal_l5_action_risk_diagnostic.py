#!/usr/bin/env python3
"""Train the restricted three-output L5 capacity diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    import numpy as np

    return (
        np.asarray([item["feature_vector"] for item in samples], dtype=np.float64),
        np.asarray([item["risk_l5"] for item in samples], dtype=np.float64),
    )


def load_samples(config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    from main.multilink_ellipsoid.l5_action_risk import feature_vector
    from main.multilink_ellipsoid.l5_action_risk_diagnostic import (
        split_train_actions, state_has_row_boundary,
    )

    source = config["source"]
    summary_path = Path(source["summary"])
    _require(_file_sha256(summary_path) == source["summary_file_sha256"],
             "L5 diagnostic summary file differs")
    summary = _load(summary_path)
    _require(summary["result_payload_sha256"]
             == source["summary_payload_sha256"],
             "L5 diagnostic summary payload differs")
    _require(summary["schema_version"]
             == "vlsa_distal_l5_aegis_adaptive_population_summary.v3",
             "L5 diagnostic summary schema differs")
    all_samples: list[dict[str, Any]] = []
    classifications = {
        str(item["state_id"]): str(item["classification"])
        for item in summary["states"]
    }
    for state in summary["states"]:
        _require(state["split"] in ("diagnostic", "train", "validation"),
                 "L5 diagnostic opened a reserved split")
        result_path = Path(state["result_path"])
        validation_path = Path(state["validation_path"])
        _require(
            _file_sha256(result_path) == state["result_file_sha256"]
            and _file_sha256(validation_path) == state["validation_file_sha256"],
            "L5 diagnostic source artifact differs",
        )
        result = _load(result_path)
        validation = _load(validation_path)
        _require(
            result["source"]["commit"] == source["population_producer_commit"]
            and validation["producer"]["result_payload_sha256"]
            == result["result_payload_sha256"]
            and validation["validation_payload_sha256"]
            == state["validation_payload_sha256"],
            "L5 diagnostic source validation differs",
        )
        target_row = result["adaptive_boundary_sampling"]["target_row"]
        for candidate in result["candidates"]:
            if candidate["terminal_status"] == "UNKNOWN_TIMEOUT":
                continue
            _require(candidate["terminal_status"] in config["sample_protocol"][
                "known_terminal_statuses"
            ], "L5 diagnostic terminal status differs")
            all_samples.append({
                "case_id": state["case_id"],
                "state_id": state["state_id"],
                "split": state["split"],
                "classification": state["classification"],
                "candidate_name": candidate["name"],
                "candidate_order": int(candidate["order"]),
                "target_row": target_row,
                "feature_vector": feature_vector(
                    initial_clearance=result["state"]["initial_clearance_m"],
                    local_frame=result["state"]["local_frame"],
                    nominal_actions=result["nominal_five_action_chunk"],
                    candidate_actions=candidate["actions"],
                ),
                "risk_l5": [float(value) for value in candidate["combined_risk"][:3]],
                "risk_all_rows": [float(value) for value in candidate["combined_risk"]],
                "exact_safe": bool(candidate["exact_safe"]),
                "physical_veto": bool(candidate["physical_veto"]),
                "applied_correction_l2_action": float(
                    candidate["applied_correction_l2_action"]
                ),
            })
    train = [item for item in all_samples if item["split"] == "train"]
    train_fit, action_heldout = split_train_actions(train)
    grouped_validation = [
        item for item in all_samples if item["split"] == "validation"
    ]
    by_state: dict[str, list[dict[str, Any]]] = {}
    for item in grouped_validation:
        by_state.setdefault(str(item["state_id"]), []).append(item)
    near_m = float(config["reporting"]["near_boundary_absolute_risk_m"])
    row0_state_ids = {
        state_id for state_id, items in by_state.items()
        if state_has_row_boundary(items, row=0, near_m=near_m)
        and any(bool(item["exact_safe"]) for item in items)
    }
    grouped_row0 = [
        item for item in grouped_validation if item["state_id"] in row0_state_ids
    ]
    diagnostic_positive = [
        item for item in all_samples
        if item["split"] == "diagnostic"
        and classifications[item["state_id"]] == "usable_mixed_support"
    ]
    output = {
        "train_fit": train_fit,
        "action_heldout": action_heldout,
        "grouped_validation": grouped_validation,
        "grouped_row_0": grouped_row0,
        "diagnostic_positive_control": diagnostic_positive,
    }
    observed = {key: len(value) for key, value in output.items()}
    _require(observed == config["sample_protocol"]["expected_sample_counts"],
             "L5 diagnostic sample population differs")
    _require(
        {item["target_row"] for item in action_heldout} == {0, 1, 2},
        "L5 diagnostic action holdout lacks a target row",
    )
    return output


def _baseline_predictions(
    train_targets: Any, count: int,
) -> dict[str, Any]:
    import numpy as np

    target = np.asarray(train_targets, dtype=np.float64)
    return {
        "train_mean": np.repeat(np.mean(target, axis=0)[None, :], count, axis=0),
        "zero": np.zeros((count, 3), dtype=np.float64),
    }


def train(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_action_risk import predict, train_model
    from main.multilink_ellipsoid.l5_action_risk_diagnostic import (
        MODEL_SCHEMA, load_config, payload_sha256, report_metrics,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    source_identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples = load_samples(config)
    train_x, train_y = _arrays(samples["train_fit"])
    heldout_x, heldout_y = _arrays(samples["action_heldout"])
    bundle = train_model(
        train_x, train_y, heldout_x, heldout_y, config["model"]
    )
    near_m = float(config["reporting"]["near_boundary_absolute_risk_m"])
    predictions = {}
    metrics = {}
    baselines = {}
    for name, subset in samples.items():
        features, _ = _arrays(subset)
        prediction = predict(bundle, features)
        predictions[name] = prediction.tolist()
        metrics[name] = report_metrics(prediction, subset, near_m=near_m)
        baselines[name] = {
            baseline_name: report_metrics(values, subset, near_m=near_m)
            for baseline_name, values in _baseline_predictions(
                train_y, len(subset)
            ).items()
        }
    diagnostic_checks = {
        "source_population_hash_locked": True,
        "unknown_timeouts_excluded": True,
        "proxy_invalid_excluded": True,
        "action_holdout_contains_target_rows_0_1_2":
        metrics["action_heldout"]["target_row_counts"] == [3, 1, 2],
        "grouped_row_0_contains_two_states":
        metrics["grouped_row_0"]["recoverable_state_count"] == 2,
        "diagnostic_cases_excluded_from_fit": all(
            item["split"] == "train" for item in samples["train_fit"]
        ),
        "reserved_test_episodes_unopened": True,
    }
    _require(all(diagnostic_checks.values()),
             "L5 diagnostic apparatus check failed")
    result = {
        "schema_version": MODEL_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": source_identity,
        "allocation": allocation_record(),
        "config": config,
        "dataset": {
            "sample_counts": {key: len(value) for key, value in samples.items()},
            "sample_identities": {
                key: [{
                    "case_id": item["case_id"],
                    "state_id": item["state_id"],
                    "candidate_name": item["candidate_name"],
                    "candidate_order": item["candidate_order"],
                    "target_row": item["target_row"],
                } for item in value]
                for key, value in samples.items()
            },
        },
        "model": {
            key: value for key, value in bundle.items()
            if key not in {
                "model", "device", "feature_mean", "feature_scale",
                "target_mean", "target_scale",
            }
        },
        "predictions": predictions,
        "metrics": metrics,
        "baselines": baselines,
        "diagnostic_checks": diagnostic_checks,
        "diagnostic_training_completed": True,
        "generalizable_safety_filter_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": "capacity_diagnostic_complete_generalization_unresolved",
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
    result = train(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "sample_counts": result["dataset"]["sample_counts"],
        "action_heldout": result["metrics"]["action_heldout"],
        "grouped_row_0": result["metrics"]["grouped_row_0"],
        "model_sha256": result["model"]["model_sha256"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
