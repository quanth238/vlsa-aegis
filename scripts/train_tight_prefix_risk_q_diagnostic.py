#!/usr/bin/env python3
"""Train one compact shared MLP on immutable tight prefix-risk labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_whole_body_q_only_diagnostic import (
    _arrays, _predict, shared_metrics, train_arm,
)


def load_samples(
    *, repo_root: Path, config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        safety_coordinate_feature,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        CASE_SCHEMA, VALIDATION_SCHEMA as DATASET_VALIDATION_SCHEMA,
        load_config as load_dataset_config, payload_sha256 as dataset_payload,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        MODEL_ROWS, file_sha256, row_future_risks,
    )

    source = config["source"]
    dataset_config_path = repo_root / source["dataset_config"]
    _require(
        file_sha256(dataset_config_path) == source["dataset_config_file_sha256"],
        "tight prefix Q dataset config file differs",
    )
    dataset_config = load_dataset_config(dataset_config_path, repo_root=repo_root)
    _require(
        dataset_config["config_payload_sha256"]
        == source["dataset_config_payload_sha256"],
        "tight prefix Q dataset config payload differs",
    )
    validation_path = Path(source["validation"])
    _require(
        _file_sha256(validation_path) == source["validation_file_sha256"],
        "tight prefix Q dataset validation file differs",
    )
    validation = _load(validation_path)
    _require(
        validation.get("schema_version") == DATASET_VALIDATION_SCHEMA
        and validation.get("validation_payload_sha256")
        == source["validation_payload_sha256"]
        == dataset_payload(validation)
        and validation.get("dataset_gate_pass") is True
        and validation.get("diagnostic_Q_only_training_authorized") is True,
        "tight prefix Q dataset validation gate differs",
    )

    outputs = {
        split: [] for split in config["dataset"]["evaluation_splits"]
    }
    cases = []
    fit_eligible_case_ids = []
    recovery_case_ids = []
    producer_dir = Path(source["producer_dir"])
    for item in dataset_config["cases"]:
        case_id = str(item["case_id"])
        split = str(item["split"])
        record_path = producer_dir / (case_id + ".json")
        record = _load(record_path)
        _require(
            record.get("schema_version") == CASE_SCHEMA
            and record.get("case_id") == case_id
            and record.get("split") == split
            and record.get("source", {}).get("commit")
            == source["artifact_commit"]
            and record.get("result_payload_sha256") == dataset_payload(record),
            "tight prefix Q case artifact differs",
        )
        exact = record["case"]
        _require(
            exact.get("source_replay_exact") is True
            and exact.get("rollout_scope")
            == dataset_config["rollout_scope"]
            and exact["exact_group_target"][
                "robot_primitive_certificate_pass"
            ] is True,
            "tight prefix Q case replay differs",
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
            trace = candidate["exact_group_target"]["trace"]
            _require(
                candidate["action_count"] == 5
                and len(trace) > 0
                and all(
                    len(sample["row_normalized_radial_slack"])
                    == len(MODEL_ROWS)
                    for sample in trace
                ),
                "tight prefix Q candidate target differs",
            )
            risks = row_future_risks(candidate, MODEL_ROWS)
            for row in MODEL_ROWS:
                outputs[split].append({
                    "state_id": case_id,
                    "split": split,
                    "candidate_name": str(candidate["name"]),
                    "candidate_order": int(order),
                    "row_index": int(row),
                    "balance_id": "%s|row-%d" % (case_id, row),
                    "feature": safety_coordinate_feature(
                        exact, candidate, row,
                        translation_scale=float(config["feature"][
                            "translation_scale_m_per_action_unit"
                        ]),
                        model_rows=MODEL_ROWS,
                    ),
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
        len(cases) == 24
        and all(outputs[split] for split in outputs)
        and len(fit_eligible_case_ids) == 12
        and len(recovery_case_ids) == 2,
        "tight prefix Q eligible population differs",
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
        "new_simulator_rollout_count": 0,
        "test_artifacts_already_opened": True,
    }


def _metrics(
    samples: Sequence[Mapping[str, Any]], predictions: Sequence[Sequence[float]],
    *, config: Mapping[str, Any],
) -> dict[str, Any]:
    primary_rows = set(int(row) for row in config["primary_rows"])
    diagnostic_rows = set(int(row) for row in config["diagnostic_rows"])
    primary_samples = []
    primary_predictions = []
    diagnostic_samples = []
    diagnostic_predictions = []
    for sample, prediction in zip(samples, predictions):
        row = int(sample["row_index"])
        if row in primary_rows:
            primary_samples.append(sample)
            primary_predictions.append(prediction)
        elif row in diagnostic_rows:
            diagnostic_samples.append(sample)
            diagnostic_predictions.append(prediction)
        else:
            raise ValueError("tight prefix Q metric row differs")
    primary_groups = {
        key: value for key, value in config["group_rows"].items()
        if key != "L6_diagnostic"
    }
    return {
        "primary": shared_metrics(
            primary_samples, primary_predictions,
            group_rows=primary_groups,
            near_boundary_abs_risk=float(
                config["evaluation"]["near_boundary_abs_risk"]
            ),
        ),
        "L6_diagnostic": shared_metrics(
            diagnostic_samples, diagnostic_predictions,
            group_rows={"L6_diagnostic": config["group_rows"]["L6_diagnostic"]},
            near_boundary_abs_risk=float(
                config["evaluation"]["near_boundary_abs_risk"]
            ),
        ),
    }


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        INPUT_DIMENSION, RESULT_SCHEMA, load_config, payload_sha256,
    )

    config = load_config(config_path)
    samples, source_artifacts = load_samples(repo_root=repo_root, config=config)
    fit_samples = [
        sample for sample in samples["train"] if sample["fit_eligible"]
    ]
    trained = train_arm(
        fit_samples, feature_key="feature", target_key="risk",
        input_dimension=INPUT_DIMENSION, output_dimension=1,
        model_config=config["model"],
    )
    predictions = {}
    metrics = {}
    for split in config["dataset"]["evaluation_splits"]:
        split_x, _, _ = _arrays(samples[split], "feature", "risk")
        split_prediction = _predict(trained["bundle"], split_x).tolist()
        predictions[split] = split_prediction
        metric_samples = samples[split]
        metric_predictions = split_prediction
        if split == "train":
            selected = [sample["fit_eligible"] for sample in metric_samples]
            metric_samples = [
                sample for sample, keep in zip(metric_samples, selected) if keep
            ]
            metric_predictions = [
                prediction for prediction, keep in zip(metric_predictions, selected)
                if keep
            ]
        metrics[split] = _metrics(
            metric_samples, metric_predictions, config=config,
        )
    model = {key: value for key, value in trained.items() if key != "bundle"}
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_diagnostic_Q_only_training",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": allocation_record(),
        "config": config,
        "source_artifacts": source_artifacts,
        "dataset": {
            "case_count": 24,
            "fit_train_case_count": 12,
            "excluded_recovery_train_case_count": 2,
            "candidate_count": 312,
            "fit_row_sample_count": len(fit_samples),
            "normalization_fit_on_train_only": True,
            "test_is_already_opened_diagnostic": True,
            "new_simulator_rollout_count": 0,
        },
        "compact_shared_7D": {
            "model": model,
            "predictions": predictions,
            "metrics": metrics,
        },
        "interpretation": {
            "L5_boundary_prediction_supported_for_diagnostic_evaluation": True,
            "tight_EE_boundary_transfer_supported": False,
            "L6_boundary_transfer_supported": False,
            "paper_or_untouched_test_claim_authorized": False,
            "correction_or_QP_authorized": False,
        },
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
        "validation_L5": result["compact_shared_7D"]["metrics"][
            "validation"
        ]["primary"]["per_group"]["L5"],
        "test_L5": result["compact_shared_7D"]["metrics"]["test"][
            "primary"
        ]["per_group"]["L5"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
