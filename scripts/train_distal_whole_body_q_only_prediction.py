#!/usr/bin/env python3
"""Run the frozen Q-only train/validation/test prediction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)
from scripts.train_distal_whole_body_q_only_diagnostic import (
    _arrays, _predict, load_samples, shared_metrics, train_arm,
)


def run(
    *, repo_root: Path, protocol_path: Path, binding_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.generic_l5_9d_capacity import diagnostic_metrics
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.whole_body_q_only_prediction import (
        RESULT_SCHEMA, combined_training_config, evaluate_prediction_gate,
        file_sha256, load_binding, load_protocol, payload_sha256,
    )
    from main.multilink_ellipsoid.whole_body_combined_audit import (
        RESULT_SCHEMA as COVERAGE_SCHEMA,
        payload_sha256 as coverage_payload_sha256,
    )

    protocol = load_protocol(protocol_path)
    binding = load_binding(binding_path, protocol)
    coverage_binding = binding["coverage_audit"]
    coverage_path = Path(coverage_binding["path"])
    _require(
        file_sha256(coverage_path) == coverage_binding["file_sha256"],
        "whole-body prediction coverage file differs",
    )
    coverage = _load(coverage_path)
    _require(
        coverage.get("schema_version") == COVERAGE_SCHEMA
        and coverage.get("audit_payload_sha256")
        == coverage_binding["payload_sha256"]
        == coverage_payload_sha256(coverage, "audit_payload_sha256"),
        "whole-body prediction coverage payload differs",
    )
    config = combined_training_config(protocol, binding)
    samples, source_record = load_samples(repo_root=repo_root, config=config)
    model_config = protocol["model"]
    arms = {}
    for arm_name, feature_key in (
        ("relative_endpoint_9D", "feature_9d"),
        ("direct_L5_OSC_33D", "feature_33d"),
    ):
        arm_config = protocol["arms"][arm_name]
        trained = train_arm(
            samples["train"]["l5"], feature_key=feature_key,
            target_key="risk_rows",
            input_dimension=int(arm_config["input_dimension"]),
            output_dimension=int(arm_config["output_dimension"]),
            model_config=model_config,
        )
        predictions = {}
        metrics = {}
        for split in protocol["dataset"]["evaluation_splits"]:
            x, _, _ = _arrays(samples[split]["l5"], feature_key, "risk_rows")
            prediction = _predict(trained["bundle"], x).tolist()
            predictions[split] = prediction
            metrics[split] = diagnostic_metrics(
                samples[split]["l5"], prediction,
                near_boundary_abs_risk=float(
                    protocol["metrics"]["near_boundary_abs_risk"]
                ),
            )
        arms[arm_name] = {
            "model": {
                key: value for key, value in trained.items() if key != "bundle"
            },
            "predictions": predictions,
            "metrics": metrics,
        }

    shared_config = protocol["arms"]["shared_constraint_135D"]
    trained = train_arm(
        samples["train"]["shared"], feature_key="feature", target_key="risk",
        input_dimension=int(shared_config["input_dimension"]),
        output_dimension=int(shared_config["output_dimension"]),
        model_config=model_config,
    )
    shared_predictions = {}
    shared_split_metrics = {}
    for split in protocol["dataset"]["evaluation_splits"]:
        x, _, _ = _arrays(samples[split]["shared"], "feature", "risk")
        prediction = _predict(trained["bundle"], x).tolist()
        shared_predictions[split] = prediction
        shared_split_metrics[split] = shared_metrics(
            samples[split]["shared"], prediction,
            group_rows=shared_config["group_rows"],
            near_boundary_abs_risk=float(
                protocol["metrics"]["near_boundary_abs_risk"]
            ),
        )
    arms["shared_constraint_135D"] = {
        "model": {
            key: value for key, value in trained.items() if key != "bundle"
        },
        "predictions": shared_predictions,
        "metrics": shared_split_metrics,
    }

    source_replay_exact = bool(coverage["gate"]["source_replay_exact"])
    prediction_gate = evaluate_prediction_gate(
        shared_split_metrics, protocol,
        source_replay_exact=source_replay_exact,
    )
    coverage_pass = bool(coverage["training_authorized"])
    prediction_pass = bool(prediction_gate["passes"])
    correction_authorized = bool(coverage_pass and prediction_pass)
    if correction_authorized:
        status = "coverage_and_prediction_pass_correction_authorized"
    elif not coverage_pass:
        status = "complete_diagnostic_undercovered_correction_no_go"
    else:
        status = "scientific_no_go_heldout_prediction_failed"
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "scientific_result": True,
        "claim_scope": protocol["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": allocation_record(),
        "protocol": protocol,
        "binding": binding,
        "coverage_audit": {
            "path": str(coverage_path),
            "audit_payload_sha256": coverage["audit_payload_sha256"],
            "training_authorized": coverage_pass,
            "gate": coverage["gate"],
        },
        "source_artifacts": source_record,
        "dataset": {
            "eligible_state_count": {
                split: len({
                    sample["state_id"] for sample in samples[split]["l5"]
                })
                for split in protocol["dataset"]["evaluation_splits"]
            },
            "known_candidate_count": {
                split: len(samples[split]["l5"])
                for split in protocol["dataset"]["evaluation_splits"]
            },
            "shared_row_sample_count": {
                split: len(samples[split]["shared"])
                for split in protocol["dataset"]["evaluation_splits"]
            },
            "unknown_timeouts_censored": True,
            "normalization_fit_on_train_only": True,
            "fixed_model_before_test": True,
            "test_opened_once": True,
        },
        "arms": arms,
        "prediction_gate": prediction_gate,
        "coverage_gate_pass": coverage_pass,
        "heldout_prediction_gate_pass": prediction_pass,
        "diagnostic_only": not coverage_pass,
        "correction_authorized": correction_authorized,
        "QP_authorized": False,
        "denoising_authorized": False,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repo_root=args.repo_root.resolve(), protocol_path=args.protocol.resolve(),
        binding_path=args.binding.resolve(), expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "dataset": result["dataset"],
        "coverage_gate_pass": result["coverage_gate_pass"],
        "heldout_prediction_gate_pass": result["heldout_prediction_gate_pass"],
        "prediction_gate": result["prediction_gate"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

