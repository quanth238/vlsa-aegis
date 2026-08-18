#!/usr/bin/env python3
"""Train matched compact 7D and obstacle-frame 17D prefix-risk critics."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_whole_body_q_only_diagnostic import (
    _arrays, _predict, train_arm,
)
from scripts.train_tight_prefix_risk_q_diagnostic import _metrics


def load_samples(
    *, repo_root: Path, config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load only train/validation artifacts and construct matched features."""
    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        safety_coordinate_feature,
    )
    from main.multilink_ellipsoid.tight_prefix_obstacle_frame_q import (
        MODEL_ROWS, obstacle_frame_feature,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        CASE_SCHEMA, VALIDATION_SCHEMA as DATASET_VALIDATION_SCHEMA,
        load_config as load_dataset_config, payload_sha256 as dataset_payload,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        file_sha256, row_future_risks,
    )

    source = baseline_config["source"]
    dataset_config_path = repo_root / source["dataset_config"]
    _require(
        file_sha256(dataset_config_path) == source["dataset_config_file_sha256"],
        "obstacle-frame Q dataset config file differs",
    )
    dataset_config = load_dataset_config(dataset_config_path, repo_root=repo_root)
    _require(
        dataset_config["config_payload_sha256"]
        == source["dataset_config_payload_sha256"],
        "obstacle-frame Q dataset config payload differs",
    )
    validation_path = Path(source["validation"])
    _require(
        _file_sha256(validation_path) == source["validation_file_sha256"],
        "obstacle-frame Q dataset validation file differs",
    )
    validation = _load(validation_path)
    _require(
        validation.get("schema_version") == DATASET_VALIDATION_SCHEMA
        and validation.get("validation_payload_sha256")
        == source["validation_payload_sha256"]
        == dataset_payload(validation)
        and validation.get("dataset_gate_pass") is True
        and validation.get("diagnostic_Q_only_training_authorized") is True,
        "obstacle-frame Q dataset validation gate differs",
    )

    allowed_splits = tuple(config["dataset"]["evaluation_splits"])
    outputs = {split: [] for split in allowed_splits}
    cases = []
    fit_eligible_case_ids = []
    recovery_case_ids = []
    producer_dir = Path(source["producer_dir"])
    for item in dataset_config["cases"]:
        case_id = str(item["case_id"])
        split = str(item["split"])
        if split not in outputs:
            continue
        record_path = producer_dir / (case_id + ".json")
        record = _load(record_path)
        _require(
            record.get("schema_version") == CASE_SCHEMA
            and record.get("case_id") == case_id
            and record.get("split") == split
            and record.get("source", {}).get("commit")
            == source["artifact_commit"]
            and record.get("result_payload_sha256") == dataset_payload(record),
            "obstacle-frame Q case artifact differs",
        )
        exact = record["case"]
        _require(
            exact.get("source_replay_exact") is True
            and exact.get("rollout_scope") == dataset_config["rollout_scope"]
            and exact["exact_group_target"][
                "robot_primitive_certificate_pass"
            ] is True,
            "obstacle-frame Q case replay differs",
        )
        initial = exact["exact_group_target"]
        initially_safe = all(
            float(initial["initial_group_normalized_radial_slack"][group]) > 0.0
            and int(initial["initial_group_contact_sample_count"][group]) == 0
            for group in (
                "palm", "finger1_base", "finger1_pad", "finger2_base",
                "finger2_pad", "L5",
            )
        )
        fit_eligible = bool(split == "train" and initially_safe)
        if fit_eligible:
            fit_eligible_case_ids.append(case_id)
        elif split == "train":
            recovery_case_ids.append(case_id)
        for order, candidate in enumerate(exact["candidates"]):
            risks = row_future_risks(candidate, MODEL_ROWS)
            for row in MODEL_ROWS:
                compact = safety_coordinate_feature(
                    exact, candidate, row,
                    translation_scale=float(config["representations"][
                        "translation_scale_m_per_action_unit"
                    ]),
                    model_rows=MODEL_ROWS,
                )
                obstacle = obstacle_frame_feature(
                    exact, candidate, row,
                    translation_scale=float(config["representations"][
                        "translation_scale_m_per_action_unit"
                    ]),
                    model_rows=MODEL_ROWS,
                )
                _require(
                    math.isclose(compact[0], obstacle[0], abs_tol=1.0e-12)
                    and math.isclose(compact[1], obstacle[1], abs_tol=1.0e-12)
                    and all(
                        math.isclose(
                            compact[2 + step], obstacle[2 + 3 * step],
                            abs_tol=1.0e-12,
                        )
                        for step in range(5)
                    ),
                    "obstacle-frame Q matched normal coordinates differ",
                )
                outputs[split].append({
                    "state_id": case_id,
                    "split": split,
                    "candidate_name": str(candidate["name"]),
                    "candidate_order": int(order),
                    "row_index": int(row),
                    "balance_id": "%s|row-%d" % (case_id, row),
                    "feature_7D": compact,
                    "feature_17D": obstacle,
                    "risk": float(risks[row]),
                    "fit_eligible": fit_eligible,
                    "applied_correction_l2_action": float(candidate[
                        "source_effective_post_AEGIS_correction_l2_action"
                    ]),
                })
        cases.append({
            "case_id": case_id,
            "split": split,
            "initially_primary_safe": initially_safe,
            "fit_eligible": fit_eligible,
            "file_sha256": _file_sha256(record_path),
            "payload_sha256": record["result_payload_sha256"],
        })
    _require(
        len(cases) == 18
        and len(fit_eligible_case_ids) == 12
        and len(recovery_case_ids) == 2
        and all(outputs[split] for split in outputs),
        "obstacle-frame Q eligible population differs",
    )
    return outputs, {
        "dataset_config": {
            "path": str(dataset_config_path),
            "file_sha256": source["dataset_config_file_sha256"],
            "payload_sha256": source["dataset_config_payload_sha256"],
        },
        "validation": {
            "path": str(validation_path),
            "file_sha256": source["validation_file_sha256"],
            "payload_sha256": source["validation_payload_sha256"],
        },
        "cases": cases,
        "fit_eligible_case_ids": sorted(fit_eligible_case_ids),
        "excluded_recovery_case_ids": sorted(recovery_case_ids),
        "test_artifacts_accessed": False,
        "new_simulator_rollout_count": 0,
    }


def _sign(value: float, tolerance: float = 1.0e-10) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _directional_audit(
    *, repo_root: Path, config: Mapping[str, Any],
    trained: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        safety_coordinate_feature,
    )
    from main.multilink_ellipsoid.tight_prefix_action_observability import (
        CASE_SCHEMA as OBSERVABILITY_CASE_SCHEMA,
        VALIDATION_SCHEMA as OBSERVABILITY_VALIDATION_SCHEMA,
        load_config as load_observability_config,
        payload_sha256 as observability_payload,
        row_future_risk,
    )
    from main.multilink_ellipsoid.tight_prefix_obstacle_frame_q import (
        MODEL_ROWS, obstacle_frame_feature,
    )

    source = config["source"]
    observability_config_path = repo_root / source[
        "action_observability_config"
    ]
    _require(
        _file_sha256(observability_config_path)
        == source["action_observability_config_file_sha256"],
        "obstacle-frame Q observability config file differs",
    )
    observability_config = load_observability_config(
        observability_config_path
    )
    _require(
        observability_config["config_payload_sha256"]
        == source["action_observability_config_payload_sha256"],
        "obstacle-frame Q observability config payload differs",
    )
    validation_path = Path(source["action_observability_validation"])
    _require(
        _file_sha256(validation_path)
        == source["action_observability_validation_file_sha256"],
        "obstacle-frame Q observability validation file differs",
    )
    validation = _load(validation_path)
    _require(
        validation.get("schema_version") == OBSERVABILITY_VALIDATION_SCHEMA
        and validation.get("validation_payload_sha256")
        == source["action_observability_validation_payload_sha256"]
        == observability_payload(validation, "validation_payload_sha256")
        and validation["summary"]["seventeen_D_ablation_authorized"] is True,
        "obstacle-frame Q observability validation differs",
    )
    producer_records = {
        row["case_id"]: row for row in validation["producer_records"]
    }
    case_reports = []
    all_slopes = {arm: [] for arm in trained}
    tangent_slopes = {arm: [] for arm in trained}
    true_all = []
    true_tangent = []
    producer_dir = Path(source["action_observability_producer_dir"])
    for selected in observability_config["cases"]:
        case_id = selected["case_id"]
        path = producer_dir / (case_id + ".json")
        record = _load(path)
        expected = producer_records[case_id]
        _require(
            record.get("schema_version") == OBSERVABILITY_CASE_SCHEMA
            and record.get("result_payload_sha256")
            == expected["payload_sha256"]
            == observability_payload(record, "result_payload_sha256")
            and _file_sha256(path) == expected["file_sha256"],
            "obstacle-frame Q observability case differs",
        )
        exact = record["case"]
        active_row = int(record["active_frame"]["active_row"])
        radius = float(record["symmetry_audit"][
            "common_translation_l2_action"
        ])
        candidates = {row["name"]: row for row in exact["candidates"]}
        features = {"compact_7D": [], "obstacle_frame_17D": []}
        names = list(candidates)
        for name in names:
            candidate = candidates[name]
            features["compact_7D"].append(safety_coordinate_feature(
                exact, candidate, active_row,
                translation_scale=float(config["representations"][
                    "translation_scale_m_per_action_unit"
                ]), model_rows=MODEL_ROWS,
            ))
            features["obstacle_frame_17D"].append(obstacle_frame_feature(
                exact, candidate, active_row,
                translation_scale=float(config["representations"][
                    "translation_scale_m_per_action_unit"
                ]), model_rows=MODEL_ROWS,
            ))
        risks = {
            name: row_future_risk(candidates[name], active_row) for name in names
        }
        predictions = {}
        for arm, feature_rows in features.items():
            predictions[arm] = {
                name: float(value)
                for name, value in zip(
                    names,
                    _predict(
                        trained[arm]["bundle"],
                        np.asarray(feature_rows, dtype=np.float64),
                    ).reshape(-1),
                )
            }
        slopes = {"true": {}, **{arm: {} for arm in trained}}
        for axis in ("normal", "tangent_up", "tangent_side"):
            positive = axis + "_pos"
            negative = axis + "_neg"
            slopes["true"][axis] = float(
                (risks[positive] - risks[negative]) / (2.0 * radius)
            )
            true_all.append(slopes["true"][axis])
            if axis != "normal":
                true_tangent.append(slopes["true"][axis])
            for arm in trained:
                value = float(
                    (predictions[arm][positive] - predictions[arm][negative])
                    / (2.0 * radius)
                )
                slopes[arm][axis] = value
                all_slopes[arm].append(value)
                if axis != "normal":
                    tangent_slopes[arm].append(value)
        case_reports.append({
            "case_id": case_id,
            "active_row": active_row,
            "common_translation_l2_action": radius,
            "slopes": slopes,
        })

    def metrics(predicted: Sequence[float], actual: Sequence[float]) -> dict:
        prediction = np.asarray(predicted, dtype=np.float64)
        target = np.asarray(actual, dtype=np.float64)
        signs = [
            _sign(float(left)) == _sign(float(right))
            for left, right in zip(prediction, target)
        ]
        return {
            "count": len(target),
            "RMSE": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "sign_accuracy": float(np.mean(np.asarray(signs, dtype=np.float64))),
        }

    return {
        "case_reports": case_reports,
        "all_direction_metrics": {
            arm: metrics(all_slopes[arm], true_all) for arm in trained
        },
        "tangent_direction_metrics": {
            arm: metrics(tangent_slopes[arm], true_tangent) for arm in trained
        },
        "source": {
            "validation_path": str(validation_path),
            "validation_file_sha256": source[
                "action_observability_validation_file_sha256"
            ],
            "validation_payload_sha256": source[
                "action_observability_validation_payload_sha256"
            ],
        },
    }


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.tight_prefix_obstacle_frame_q import (
        COMPACT_INPUT_DIMENSION, OBSTACLE_FRAME_INPUT_DIMENSION,
        PRIMARY_ROWS, RESULT_SCHEMA, file_sha256, load_config,
        payload_sha256, same_state_pairwise_rank_accuracy,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        load_config as load_baseline_config,
    )

    config = load_config(config_path)
    source = config["source"]
    baseline_path = repo_root / source["baseline_training_config"]
    _require(
        file_sha256(baseline_path)
        == source["baseline_training_config_file_sha256"],
        "obstacle-frame Q baseline config file differs",
    )
    baseline = load_baseline_config(baseline_path)
    _require(
        baseline["config_payload_sha256"]
        == source["baseline_training_config_payload_sha256"]
        and baseline["model"] == config["model"],
        "obstacle-frame Q baseline config payload differs",
    )
    samples, source_artifacts = load_samples(
        repo_root=repo_root, config=config, baseline_config=baseline,
    )
    fit_samples = [
        sample for sample in samples["train"] if sample["fit_eligible"]
    ]
    dimensions = {
        "compact_7D": ("feature_7D", COMPACT_INPUT_DIMENSION),
        "obstacle_frame_17D": ("feature_17D", OBSTACLE_FRAME_INPUT_DIMENSION),
    }
    trained = {}
    arms = {}
    for arm, (feature_key, dimension) in dimensions.items():
        trained[arm] = train_arm(
            fit_samples, feature_key=feature_key, target_key="risk",
            input_dimension=dimension, output_dimension=1,
            model_config=config["model"],
        )
        predictions = {}
        metrics = {}
        ranking = {}
        for split in config["dataset"]["evaluation_splits"]:
            split_x, _, _ = _arrays(samples[split], feature_key, "risk")
            split_prediction = _predict(
                trained[arm]["bundle"], split_x,
            ).tolist()
            predictions[split] = split_prediction
            metric_samples = samples[split]
            metric_predictions = split_prediction
            if split == "train":
                keep = [sample["fit_eligible"] for sample in metric_samples]
                metric_samples = [
                    sample for sample, selected in zip(metric_samples, keep)
                    if selected
                ]
                metric_predictions = [
                    prediction for prediction, selected
                    in zip(metric_predictions, keep) if selected
                ]
            metrics[split] = _metrics(
                metric_samples, metric_predictions, config=baseline,
            )
            ranking[split] = same_state_pairwise_rank_accuracy(
                metric_samples, metric_predictions,
                rows=PRIMARY_ROWS,
            )
        arms[arm] = {
            "model": {
                key: value for key, value in trained[arm].items()
                if key != "bundle"
            },
            "predictions": predictions,
            "metrics": metrics,
            "same_state_pairwise_primary_rank": ranking,
        }
    directional = _directional_audit(
        repo_root=repo_root, config=config, trained=trained,
    )
    compact_validation = arms["compact_7D"]["metrics"]["validation"][
        "primary"
    ]["global"]
    frame_validation = arms["obstacle_frame_17D"]["metrics"]["validation"][
        "primary"
    ]["global"]
    compact_rank = float(arms["compact_7D"][
        "same_state_pairwise_primary_rank"
    ]["validation"]["accuracy"])
    frame_rank = float(arms["obstacle_frame_17D"][
        "same_state_pairwise_primary_rank"
    ]["validation"]["accuracy"])
    compact_near = float(compact_validation["near_boundary_RMSE"])
    frame_near = float(frame_validation["near_boundary_RMSE"])
    tangent = directional["tangent_direction_metrics"]
    gate_config = config["gate"]
    gates = {
        "validation_pairwise_rank_improvement": bool(
            frame_rank - compact_rank
            >= float(gate_config[
                "minimum_validation_pairwise_rank_improvement"
            ])
        ),
        "validation_near_boundary_RMSE_not_materially_worse": bool(
            frame_near / compact_near
            <= float(gate_config["maximum_validation_near_boundary_RMSE_ratio"])
        ),
        "validation_false_safes_not_increased": bool(
            int(frame_validation["false_safe_count"])
            <= int(compact_validation["false_safe_count"])
        ),
        "tangent_slope_sign_accuracy": bool(
            float(tangent["obstacle_frame_17D"]["sign_accuracy"])
            >= float(gate_config[
                "minimum_17D_tangent_slope_sign_accuracy"
            ])
        ),
        "tangent_slope_RMSE_improved": bool(
            float(tangent["obstacle_frame_17D"]["RMSE"])
            < float(tangent["compact_7D"]["RMSE"])
        ),
    }
    authorized = all(gates.values())
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_matched_obstacle_frame_Q_training",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": allocation_record(),
        "config": config,
        "source_artifacts": source_artifacts,
        "dataset": {
            "case_count": 18,
            "fit_train_case_count": 12,
            "excluded_recovery_train_case_count": 2,
            "candidate_count": 234,
            "fit_row_sample_count": len(fit_samples),
            "normalization_fit_on_train_only": True,
            "test_artifacts_accessed": False,
            "new_simulator_rollout_count": 0,
        },
        "arms": arms,
        "directional_observability_validation": directional,
        "gate": {
            "checks": gates,
            "validation_pairwise_rank_accuracy": {
                "compact_7D": compact_rank,
                "obstacle_frame_17D": frame_rank,
                "improvement": frame_rank - compact_rank,
            },
            "validation_near_boundary_RMSE": {
                "compact_7D": compact_near,
                "obstacle_frame_17D": frame_near,
                "ratio": frame_near / compact_near,
            },
            "validation_false_safe_count": {
                "compact_7D": int(compact_validation["false_safe_count"]),
                "obstacle_frame_17D": int(frame_validation["false_safe_count"]),
            },
            "exact_gradient_probe_authorized": authorized,
        },
        "directional_loss_used": False,
        "test_split_accessed": False,
        "action_correction_or_full_episode_performed": False,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "dataset": result["dataset"],
        "gate": result["gate"],
        "tangent_direction_metrics": result[
            "directional_observability_validation"
        ]["tangent_direction_metrics"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
