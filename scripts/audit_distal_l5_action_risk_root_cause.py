#!/usr/bin/env python3
"""Audit data coverage versus frozen-model capacity without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_action_risk_diagnostic import _arrays, load_samples


def analyze(model_result: dict[str, Any], samples: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.l5_action_risk_root_cause import (
        feature_support, root_cause_decision,
    )

    train_x, _ = _arrays(samples["train_fit"])
    train_ids = [str(item["state_id"]) for item in samples["train_fit"]]
    scale = model_result["model"]["state_payload"]["feature_scale"]
    support = {}
    for name, subset in samples.items():
        features, _ = _arrays(subset)
        support[name] = feature_support(
            train_x, features, train_ids,
            [str(item["state_id"]) for item in subset], scale,
        )

    metrics = model_result["metrics"]
    summary_path = Path(model_result["config"]["source"]["summary"])
    population = _load(summary_path)
    train_boundary_states = population["useful_boundary_state_count_by_split"][
        "train"
    ][:3]
    validation_boundary_states = population[
        "useful_boundary_state_count_by_split"
    ]["validation"][:3]
    active_witnesses = [
        int(item["active_witness_known_candidate_count"])
        for item in population["fit_row_coverage"][:3]
    ]
    train_false_safe = [
        int(item["false_safe_count"]) for item in metrics["train_fit"]["per_row"]
    ]
    decision = root_cause_decision(
        train_rmse_m=float(metrics["train_fit"]["rmse_m"]),
        action_holdout_rmse_m=float(metrics["action_heldout"]["rmse_m"]),
        grouped_validation_rmse_m=float(metrics["grouped_validation"]["rmse_m"]),
        train_boundary_states=train_boundary_states,
        validation_boundary_states=validation_boundary_states,
        train_active_witnesses=active_witnesses,
        train_per_row_false_safe=train_false_safe,
    )
    ratios = {
        "grouped_validation_over_action_holdout_rmse": float(
            metrics["grouped_validation"]["rmse_m"]
            / metrics["action_heldout"]["rmse_m"]
        ),
        "grouped_row0_over_action_holdout_row0_near_rmse": float(
            metrics["grouped_row_0"]["per_row"][0]["near_boundary_rmse_m"]
            / metrics["action_heldout"]["per_row"][0]["near_boundary_rmse_m"]
        ),
        "E05_row1_over_action_holdout_row1_near_rmse": float(
            metrics["diagnostic_positive_control"]["per_row"][1]["near_boundary_rmse_m"]
            / metrics["action_heldout"]["per_row"][1]["near_boundary_rmse_m"]
        ),
    }
    return {
        "split_metrics": {
            name: {
                "sample_count": int(metrics[name]["sample_count"]),
                "rmse_m": float(metrics[name]["rmse_m"]),
                "near_boundary_rmse_m": metrics[name]["near_boundary_rmse_m"],
                "L5_false_safe_count": int(metrics[name]["L5_false_safe_count"]),
                "per_row": [{
                    key: item[key] for key in (
                        "row", "rmse_m", "near_boundary_rmse_m",
                        "boundary_side_accuracy", "false_safe_count",
                        "false_unsafe_count",
                    )
                } for item in metrics[name]["per_row"]],
            } for name in metrics
        },
        "error_ratios": ratios,
        "population_coverage": {
            "known_train_candidate_count": int(
                population["known_candidate_count_by_split"]["train"]
            ),
            "useful_train_boundary_state_counts": list(train_boundary_states),
            "useful_validation_boundary_state_counts":
            list(validation_boundary_states),
            "active_witness_candidate_counts": list(active_witnesses),
        },
        "feature_support": support,
        "input_contract": {
            "included": [
                "seven_current_clearances", "local_normal_and_tangents",
                "nominal_five_action_chunk", "candidate_minus_nominal_chunk",
            ],
            "recorded_but_omitted": [
                "joint_position", "joint_velocity", "EEF_pose",
                "OSC_goal_position_and_orientation", "controller_memory",
                "obstacle_pose_and_extent", "per_row_ellipsoid_transforms",
            ],
            "omitted_input_hypothesis_status":
            "plausible_but_not_identified_by_this_frozen_audit",
        },
        "decision": decision,
    }


def audit(
    *, repo_root: Path, expected_commit: str, config_path: Path,
    model_result_path: Path, model_validation_path: Path,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_action_risk_diagnostic import (
        MODEL_SCHEMA, VALIDATION_SCHEMA as MODEL_VALIDATION_SCHEMA, load_config,
        payload_sha256 as model_payload_sha256,
    )
    from main.multilink_ellipsoid.l5_action_risk_root_cause import (
        AUDIT_SCHEMA, payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    model_result = _load(model_result_path)
    model_validation = _load(model_validation_path)
    _require(model_result["schema_version"] == MODEL_SCHEMA, "model result schema differs")
    _require(model_result["result_payload_sha256"] == model_payload_sha256(model_result),
             "model result payload differs")
    _require(model_validation["schema_version"] == MODEL_VALIDATION_SCHEMA,
             "model validation schema differs")
    _require(model_validation["status"] == "validated"
             and model_validation["maximum_prediction_replay_error_m"] == 0.0,
             "model validation is not exact")
    _require(model_validation["model_result_file_sha256"] == _file_sha256(model_result_path),
             "model validation file binding differs")
    samples = load_samples(config)
    analysis = analyze(model_result, samples)
    output = {
        "schema_version": AUDIT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": "frozen_no_training_root_cause_audit_no_QP_no_control",
        "source": source,
        "allocation": allocation_record(),
        "model_result_file_sha256": _file_sha256(model_result_path),
        "model_result_payload_sha256": model_result["result_payload_sha256"],
        "model_validation_file_sha256": _file_sha256(model_validation_path),
        "model_validation_payload_sha256": model_validation["validation_payload_sha256"],
        "analysis": analysis,
        "training_performed": False,
        "reserved_test_episodes_opened": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["result_payload_sha256"] = payload_sha256(output)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-result", type=Path, required=True)
    parser.add_argument("--model-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        repo_root=args.repo_root.resolve(), expected_commit=args.expected_commit,
        config_path=args.config.resolve(), model_result_path=args.model_result.resolve(),
        model_validation_path=args.model_validation.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "decision": result["analysis"]["decision"],
        "error_ratios": result["analysis"]["error_ratios"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
