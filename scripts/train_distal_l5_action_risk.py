#!/usr/bin/env python3
"""Train the fixed prediction-only three-output L5 risk MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    import numpy as np

    return (
        np.asarray([item["feature_vector"] for item in samples], dtype=np.float64),
        np.asarray([item["risk_l5"] for item in samples], dtype=np.float64),
    )


def _load_samples(config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    import numpy as np
    from main.multilink_ellipsoid.l5_action_risk import feature_vector

    root = Path(config["source"]["frozen_dataset_root"])
    manifest = _read_jsonl(root / "candidate_manifest.jsonl")
    split = _load(root / "split_manifest.json")
    artifact_by_case = {
        item["case_id"]: item for item in split["source_artifacts"]
    }
    result_cache = {}
    output = {"train": [], "validation": []}
    for sample in manifest:
        split_name = str(sample["split"])
        if split_name not in output:
            continue
        artifact = artifact_by_case[sample["case_id"]]
        if sample["case_id"] not in result_cache:
            path = Path(artifact["path"])
            _require(_file_sha256(path) == artifact["file_sha256"],
                     "L5 action-risk source result file differs")
            result = _load(path)
            _require(result["result_payload_sha256"]
                     == artifact["result_payload_sha256"],
                     "L5 action-risk source result payload differs")
            result_cache[sample["case_id"]] = result
        result = result_cache[sample["case_id"]]
        _require(result["population_binding"]["retained_state"]["state_id"]
                 == sample["state_id"], "L5 action-risk state differs")
        matches = [
            item for item in result["candidates"]
            if item["name"] == sample["candidate_name"]
            and int(item["order"]) == int(sample["candidate_order"])
        ]
        _require(len(matches) == 1, "L5 action-risk candidate differs")
        candidate = matches[0]
        _require(np.array_equal(np.asarray(candidate["actions"]), np.asarray(sample["actions"]))
                 and candidate["combined_risk"] == sample["risk"]
                 and candidate["exact_safe"] == sample["exact_safe"],
                 "L5 action-risk candidate target differs")
        output[split_name].append({
            "case_id": sample["case_id"],
            "state_id": sample["state_id"],
            "candidate_name": sample["candidate_name"],
            "candidate_order": sample["candidate_order"],
            "feature_vector": feature_vector(
                initial_clearance=result["state"]["initial_clearance_m"],
                local_frame=result["state"]["local_frame"],
                nominal_actions=result["nominal_five_action_chunk"],
                candidate_actions=candidate["actions"],
            ),
            "risk_l5": sample["risk"][:3],
            "risk_all_rows": sample["risk"],
            "exact_safe": sample["exact_safe"],
            "physical_veto": sample["physical_veto"],
            "applied_correction_l2_action": candidate[
                "applied_correction_l2_action"
            ],
        })
    return output


def train(
    *, repo_root: Path, config_path: Path, dataset_audit_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_action_risk import (
        AUDIT_SCHEMA, MODEL_SCHEMA, load_config, payload_sha256, predict,
        prediction_metrics, train_model,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    audit = _load(dataset_audit_path)
    _require(audit["schema_version"] == AUDIT_SCHEMA
             and audit["status"] == "complete"
             and audit["MLP_training_authorized"] is True
             and all(audit["gates"].values())
             and audit["config"] == config,
             "L5 action-risk dataset audit did not authorize training")
    _require(audit["result_payload_sha256"] == payload_sha256(audit),
             "L5 action-risk dataset audit payload differs")
    samples = _load_samples(config)
    train_x, train_y = _arrays(samples["train"])
    validation_x, validation_y = _arrays(samples["validation"])
    bundle = train_model(
        train_x, train_y, validation_x, validation_y, config["model"]
    )
    predictions = {}
    metrics = {}
    for split_name in ("train", "validation"):
        x, _ = _arrays(samples[split_name])
        prediction = predict(bundle, x)
        predictions[split_name] = prediction.tolist()
        metrics[split_name] = prediction_metrics(prediction, samples[split_name])
    validation = metrics["validation"]
    gates = {
        "zero_observed_L5_false_safe_candidates":
        validation["L5_false_safe_count"] == 0,
        "minimum_exact_safe_candidate_recall":
        validation["exact_safe_candidate_recall"]
        >= float(config["prediction_gate"]["minimum_exact_safe_candidate_recall"]),
        "safe_action_support_every_recoverable_state":
        validation["supported_recoverable_state_count"]
        == validation["recoverable_state_count"],
        "validation_contains_recoverable_states":
        validation["recoverable_state_count"] > 0,
        "selected_candidates_require_exact_all_seven_physical_acceptance": True,
        "all_selected_candidates_exact_all_seven_safe":
        validation["all_selected_candidates_exact_all_seven_safe"],
        "reserved_test_episodes_unopened": True,
    }
    passed = all(gates.values())
    result = {
        "schema_version": MODEL_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation_record(),
        "config": config,
        "dataset": {
            "audit_path": str(dataset_audit_path),
            "audit_file_sha256": _file_sha256(dataset_audit_path),
            "audit_payload_sha256": audit["result_payload_sha256"],
            "sample_counts": {key: len(value) for key, value in samples.items()},
            "diagnostic_excluded": True,
            "reserved_test_unopened": True,
        },
        "model": {
            key: value for key, value in bundle.items()
            if key not in {
                "model", "device", "feature_mean", "feature_scale",
                "target_mean", "target_scale",
            }
        },
        "metrics": metrics,
        "predictions": predictions,
        "gates": gates,
        "prediction_gate_pass": passed,
        "MLP_scope": "L5_rows_0_1_2_only",
        "rows_3_to_6": "exact_diagnostic_and_physical_veto_only",
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": (
            "three_output_L5_action_risk_prediction_gate_pass"
            if passed else "three_output_L5_action_risk_prediction_no_go"
        ),
    }
    result["result_payload_sha256"] = payload_sha256(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = train(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        dataset_audit_path=args.dataset_audit.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "gates": result["gates"],
        "validation_metrics": result["metrics"]["validation"],
        "model_sha256": result["model"]["model_sha256"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
