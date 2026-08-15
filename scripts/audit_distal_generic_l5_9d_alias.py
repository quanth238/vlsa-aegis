#!/usr/bin/env python3
"""Audit frozen 9D models for risk-offset and omitted-state failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_generic_l5_9d_capacity import load_samples


def load_sources(
    *, repo_root: Path, audit_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    from main.multilink_ellipsoid.generic_l5_9d_capacity import (
        RESULT_SCHEMA as CAPACITY_RESULT_SCHEMA,
        VALIDATION_SCHEMA as CAPACITY_VALIDATION_SCHEMA,
        load_config as load_capacity_config,
        payload_sha256 as capacity_payload_sha256,
    )

    source = audit_config["sources"]
    capacity_config_path = repo_root / source["capacity_config"]
    _require(
        _file_sha256(capacity_config_path) == source["capacity_config_file_sha256"],
        "alias-audit capacity config file differs",
    )
    capacity_config = load_capacity_config(capacity_config_path)
    _require(
        capacity_config["config_payload_sha256"]
        == source["capacity_config_payload_sha256"],
        "alias-audit capacity config payload differs",
    )
    result_path = Path(source["capacity_result"])
    validation_path = Path(source["capacity_validation"])
    _require(
        _file_sha256(result_path) == source["capacity_result_file_sha256"]
        and _file_sha256(validation_path) == source["capacity_validation_file_sha256"],
        "alias-audit capacity files differ",
    )
    result = _load(result_path)
    validation = _load(validation_path)
    _require(
        result["schema_version"] == CAPACITY_RESULT_SCHEMA
        and result["source"]["commit"] == source["capacity_producer_commit"]
        and result["result_payload_sha256"]
        == source["capacity_result_payload_sha256"]
        == capacity_payload_sha256(result, "result_payload_sha256")
        and validation["schema_version"] == CAPACITY_VALIDATION_SCHEMA
        and validation["validation_payload_sha256"]
        == source["capacity_validation_payload_sha256"]
        == capacity_payload_sha256(validation, "validation_payload_sha256")
        and float(validation["frozen_prediction_replay_maximum_error"]) == 0.0
        and float(validation["independent_retrain_prediction_maximum_error"]) == 0.0,
        "alias-audit capacity evidence differs",
    )
    grouped, _, _ = load_samples(capacity_config)
    return result, capacity_config, grouped


def compute_audit(
    *, repo_root: Path, audit_config: Mapping[str, Any],
    result: Mapping[str, Any], capacity_config: Mapping[str, Any],
    grouped: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.generic_l5_9d_alias_audit import (
        correlation, direct_l5_context_vector, oracle_anchor_predictions,
        rms_distance, standardized_rows,
    )
    from main.multilink_ellipsoid.generic_l5_9d_capacity import diagnostic_metrics

    near = float(audit_config["audit"]["near_boundary_abs_risk"])
    anchored_predictions = []
    anchored_samples = []
    state_records = []
    state_raw = {}
    for fold in result["folds"]:
        state_id = str(fold["held_out_state_id"])
        samples = list(grouped[state_id])
        prediction = fold["validation_prediction"]
        anchored = oracle_anchor_predictions(samples, prediction)
        anchored_predictions.extend(anchored)
        anchored_samples.extend(samples)
        nominal_index = next(
            index for index, sample in enumerate(samples)
            if sample["candidate_name"] == "nominal"
        )
        case_binding = capacity_config["sources"]["case_artifacts"][state_id]
        case_path = Path(capacity_config["sources"]["producer_root"]) / case_binding["filename"]
        case = _load(case_path)
        context = case["exact_case"]["physical_context"]
        q_qdot = (
            [float(item) for item in context["arm_joint_position_rad"]]
            + [float(item) for item in context["arm_joint_velocity_rad_s"]]
        )
        actual_nominal = [float(item) for item in samples[nominal_index]["risk_rows"]]
        predicted_nominal = [float(item) for item in prediction[nominal_index]]
        state_raw[state_id] = {
            "samples": samples,
            "nominal_9D": [float(item) for item in samples[nominal_index]["feature"]],
            "direct_L5": direct_l5_context_vector(context),
            "q_qdot": q_qdot,
            "actual_nominal_rows": actual_nominal,
            "predicted_nominal_rows": predicted_nominal,
        }
        state_records.append({
            "state_id": state_id,
            "actual_nominal_global_risk": max(actual_nominal),
            "predicted_nominal_global_risk": max(predicted_nominal),
            "nominal_global_offset_error": max(predicted_nominal) - max(actual_nominal),
            "unanchored_rank_spearman": fold["metrics"]["global_rank_spearman"],
            "unanchored_improvement_direction_accuracy": fold["metrics"][
                "improvement_direction_accuracy"
            ],
        })
    original = result["leave_one_state_out_metrics"]
    anchored = diagnostic_metrics(
        anchored_samples, anchored_predictions, near_boundary_abs_risk=near,
    )
    state_ids = [str(item) for item in capacity_config["dataset"]["prevention_case_ids"]]
    nine = standardized_rows([state_raw[state_id]["nominal_9D"] for state_id in state_ids])
    direct = standardized_rows([state_raw[state_id]["direct_L5"] for state_id in state_ids])
    joints = standardized_rows([state_raw[state_id]["q_qdot"] for state_id in state_ids])
    pairs = []
    for left_index, left_id in enumerate(state_ids):
        for right_index in range(left_index + 1, len(state_ids)):
            right_id = state_ids[right_index]
            left = state_raw[left_id]
            right = state_raw[right_id]
            left_by_name = {sample["candidate_name"]: sample for sample in left["samples"]}
            right_by_name = {sample["candidate_name"]: sample for sample in right["samples"]}
            names = sorted(set(left_by_name) & set(right_by_name))
            left_nominal = np.asarray(left_by_name["nominal"]["risk_rows"], dtype=np.float64)
            right_nominal = np.asarray(right_by_name["nominal"]["risk_rows"], dtype=np.float64)
            response_squared = []
            global_response_squared = []
            for name in names:
                left_response = np.asarray(left_by_name[name]["risk_rows"]) - left_nominal
                right_response = np.asarray(right_by_name[name]["risk_rows"]) - right_nominal
                response_squared.extend(((left_response - right_response) ** 2).tolist())
                left_global = max(left_by_name[name]["risk_rows"]) - max(left_nominal)
                right_global = max(right_by_name[name]["risk_rows"]) - max(right_nominal)
                global_response_squared.append((left_global - right_global) ** 2)
            pairs.append({
                "left_state_id": left_id, "right_state_id": right_id,
                "common_known_candidate_count": len(names),
                "nominal_9D_z_RMS_distance": rms_distance(nine[left_index], nine[right_index]),
                "direct_L5_z_RMS_distance": rms_distance(direct[left_index], direct[right_index]),
                "q_qdot_z_RMS_distance": rms_distance(joints[left_index], joints[right_index]),
                "nominal_global_risk_absolute_gap": abs(
                    max(left["actual_nominal_rows"]) - max(right["actual_nominal_rows"])
                ),
                "per_row_action_response_RMSE": float(np.sqrt(np.mean(response_squared))),
                "global_action_response_RMSE": float(
                    np.sqrt(np.mean(global_response_squared))
                ),
            })
    for record in state_records:
        state_id = record["state_id"]
        candidates = [
            pair for pair in pairs
            if state_id in (pair["left_state_id"], pair["right_state_id"])
        ]
        nearest = min(candidates, key=lambda pair: pair["nominal_9D_z_RMS_distance"])
        record["nearest_9D_state_id"] = (
            nearest["right_state_id"] if nearest["left_state_id"] == state_id
            else nearest["left_state_id"]
        )
        record["nearest_9D_distance"] = nearest["nominal_9D_z_RMS_distance"]
        record["nearest_9D_pair_direct_L5_distance"] = nearest["direct_L5_z_RMS_distance"]
        record["nearest_9D_pair_nominal_risk_gap"] = nearest[
            "nominal_global_risk_absolute_gap"
        ]
        record["nearest_9D_pair_action_response_RMSE"] = nearest[
            "global_action_response_RMSE"
        ]
    ratio = float(anchored["global_RMSE"]) / max(float(original["global_RMSE"]), 1e-12)
    offset_dominant = bool(
        ratio <= float(audit_config["decision"][
            "offset_dominant_requires_anchored_RMSE_ratio_at_most"
        ])
        and int(anchored["false_safe_count"]) < int(original["false_safe_count"])
        and float(anchored["safe_recall"] or 0.0) > float(original["safe_recall"] or 0.0)
    )
    return {
        "unanchored_metrics": original,
        "oracle_anchored_metrics": anchored,
        "oracle_anchor_comparison": {
            "anchored_over_unanchored_global_RMSE_ratio": ratio,
            "false_safe_reduction": int(original["false_safe_count"])
            - int(anchored["false_safe_count"]),
            "false_unsafe_reduction": int(original["false_unsafe_count"])
            - int(anchored["false_unsafe_count"]),
            "safe_recall_increase": float(anchored["safe_recall"] or 0.0)
            - float(original["safe_recall"] or 0.0),
        },
        "state_records": state_records,
        "pairwise_state_audit": pairs,
        "pairwise_associations": {
            "correlation_9D_distance_with_nominal_risk_gap": correlation(
                [pair["nominal_9D_z_RMS_distance"] for pair in pairs],
                [pair["nominal_global_risk_absolute_gap"] for pair in pairs],
            ),
            "correlation_direct_L5_distance_with_nominal_risk_gap": correlation(
                [pair["direct_L5_z_RMS_distance"] for pair in pairs],
                [pair["nominal_global_risk_absolute_gap"] for pair in pairs],
            ),
            "correlation_9D_distance_with_action_response_RMSE": correlation(
                [pair["nominal_9D_z_RMS_distance"] for pair in pairs],
                [pair["global_action_response_RMSE"] for pair in pairs],
            ),
            "correlation_direct_L5_distance_with_action_response_RMSE": correlation(
                [pair["direct_L5_z_RMS_distance"] for pair in pairs],
                [pair["global_action_response_RMSE"] for pair in pairs],
            ),
        },
        "offset_dominant": offset_dominant,
        "interpretation": (
            "state_offset_is_primary_but_action_response_still_requires_coverage"
            if offset_dominant else
            "oracle_anchor_does_not_rescue_threshold_response_representation_also_fails"
        ),
    }


def run(*, repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.generic_l5_9d_alias_audit import (
        RESULT_SCHEMA, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    result, capacity_config, grouped = load_sources(
        repo_root=repo_root, audit_config=config,
    )
    audit = compute_audit(
        repo_root=repo_root, audit_config=config, result=result,
        capacity_config=capacity_config, grouped=grouped,
    )
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": identity, "allocation": allocation_record(), "config": config,
        "audit": audit,
        "new_simulation_or_training_performed": False,
        "correction_authorized": False, "QP_authorized": False,
        "calibration_authorized": False, "closed_loop_authorized": False,
    }
    output["result_payload_sha256"] = payload_sha256(output, "result_payload_sha256")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "interpretation": output["audit"]["interpretation"],
        "comparison": output["audit"]["oracle_anchor_comparison"],
        "result_payload_sha256": output["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
