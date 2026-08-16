#!/usr/bin/env python3
"""Train the frozen prospective grouped L5 Q-only diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def load_samples(config: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, VALIDATION_SCHEMA as SOURCE_VALIDATION_SCHEMA,
        payload_sha256 as source_payload_sha256,
    )
    from main.multilink_ellipsoid.generic_l5_9d_capacity import feature_vector

    source = config["sources"]
    validation_path = Path(source["validation_file"])
    _require(_file_sha256(validation_path) == source["validation_file_sha256"], "prospective source validation file differs")
    validation = _load(validation_path)
    _require(
        validation["schema_version"] == SOURCE_VALIDATION_SCHEMA
        and validation["validation_payload_sha256"] == source["validation_payload_sha256"]
        == source_payload_sha256(validation, "validation_payload_sha256")
        and bool(validation["summary"]["apparatus_pass"])
        and bool(validation["independent_replay"]["exact_scientific_reproduction"])
        and int(validation["summary"]["physical_false_safe_count"]) == 0,
        "prospective source validation differs",
    )
    split_lookup = {
        case_id: split for split, case_ids in config["dataset"]["split_case_ids"].items()
        for case_id in case_ids
    }
    row_indices = [int(item) for item in config["dataset"]["L5_row_indices"]]
    scale = float(config["features"]["translation_scale_m_per_action_unit"])
    output = {"train": [], "validation": [], "test": []}
    case_records = []
    for case_id in sorted(split_lookup):
        binding = source["case_artifacts"][case_id]
        split = split_lookup[case_id]
        _require(binding["split"] == split, "prospective split binding differs")
        path = Path(source["producer_root"]) / binding["filename"]
        _require(_file_sha256(path) == binding["file_sha256"], "prospective case file differs")
        result = _load(path)
        _require(
            result["schema_version"] == CASE_SCHEMA
            and result["case_id"] == case_id
            and result["selection"]["split"] == split
            and result["result_payload_sha256"] == binding["payload_sha256"]
            == source_payload_sha256(result),
            "prospective case payload differs",
        )
        exact = result["exact_case"]
        context = exact["physical_context"]
        _require(
            bool(exact["source_replay_exact"])
            and bool(exact["state_hash_matches"])
            and not bool(exact["initial_compiled_box_any_exact_overlap"])
            and int(exact["initial_raw_protected_contact_count"]) == 0
            and bool(exact["exact_group_target"]["robot_primitive_certificate_pass"]),
            "prospective case replay differs",
        )
        known = 0
        for order, candidate in enumerate(exact["candidates"]):
            target = candidate["exact_group_target"]
            if not bool(target["known_outcome"]):
                continue
            row_slack = candidate["compiled_box_row_minimum_normalized_radial_slack"]
            _require(len(row_slack) == 7, "prospective row target shape differs")
            risk_rows = [-float(row_slack[index]) for index in row_indices]
            sample = {
                "state_id": case_id,
                "split": split,
                "candidate_name": str(candidate["name"]),
                "candidate_order": int(order),
                "feature": feature_vector(
                    eef_position_m=context["eef_position_m"],
                    candidate_actions=candidate["source_executed_actions"],
                    obstacle_center_m=context["obstacle"]["center_m"],
                    obstacle_semiaxes_m=context["obstacle"]["semiaxes_m"],
                    translation_scale_m_per_action_unit=scale,
                ),
                "risk_rows": risk_rows,
                "global_risk": max(risk_rows),
                "source_terminal_status": candidate["source_terminal_status"],
                "applied_correction_l2_action": float(candidate["source_effective_post_AEGIS_correction_l2_action"]),
            }
            output[split].append(sample)
            known += 1
        _require(known == int(binding["known_candidate_count"]), "prospective known candidate count differs")
        case_records.append({
            "case_id": case_id, "split": split, "file": str(path),
            "file_sha256": binding["file_sha256"],
            "payload_sha256": binding["payload_sha256"],
            "known_candidate_count": known,
        })
    return output, {
        "source_validation_file_sha256": source["validation_file_sha256"],
        "source_validation_payload_sha256": source["validation_payload_sha256"],
        "cases": case_records,
    }


def run(*, repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.generic_l5_9d_capacity import diagnostic_metrics
    from main.multilink_ellipsoid.prospective_l5_q_only_diagnostic import (
        RESULT_SCHEMA, classify_diagnostic, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record
    from scripts.train_distal_generic_l5_9d_capacity import arrays, predict, train_fold

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples, source_record = load_samples(config)
    trained = train_fold(samples["train"], samples["validation"], config["model"])
    test_prediction = predict(trained["bundle"], arrays(samples["test"])[0]).tolist()
    predictions = {
        "train": trained["train_prediction"],
        "validation": trained["validation_prediction"],
        "test": test_prediction,
    }
    metrics = {
        split: diagnostic_metrics(
            samples[split], predictions[split],
            near_boundary_abs_risk=float(config["metrics"]["near_boundary_abs_risk"]),
        ) for split in ("train", "validation", "test")
    }
    interpretation, checks = classify_diagnostic(
        metrics["validation"], metrics["test"], config["decision"],
    )
    model_record = {key: value for key, value in trained.items() if key not in (
        "bundle", "train_prediction", "validation_prediction",
    )}
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": identity,
        "allocation": allocation_record(),
        "config": config,
        "source_artifacts": source_record,
        "dataset": {
            "state_count": {split: len(config["dataset"]["split_case_ids"][split]) for split in ("train", "validation", "test")},
            "known_sample_count": {split: len(samples[split]) for split in ("train", "validation", "test")},
            "unknown_timeouts_masked": True,
            "candidate_groups_preserved": True,
            "normalization_fit_on_train_only": True,
            "fixed_final_epoch_model": True,
            "test_accessed_once_after_model_frozen": True,
        },
        "model": model_record,
        "predictions": predictions,
        "metrics": metrics,
        "decision_checks": checks,
        "interpretation": interpretation,
        "training_dataset_gate_pass": False,
        "correction_authorized": False,
        "QP_authorized": False,
        "calibration_authorized": False,
        "closed_loop_authorized": False,
    }
    result["result_payload_sha256"] = payload_sha256(result, "result_payload_sha256")
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
    print(json.dumps({
        "interpretation": result["interpretation"],
        "metrics": result["metrics"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
