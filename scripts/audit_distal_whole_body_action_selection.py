#!/usr/bin/env python3
"""Audit frozen whole-body MLP correction rules without new simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def run(
    *, repo_root: Path, prediction_path: Path, expected_file_sha256: str,
    expected_payload_sha256: str, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.whole_body_action_selection_audit import (
        candidate_records, evaluate_rule, validation_optimistic_margin,
    )
    from main.multilink_ellipsoid.whole_body_q_only_prediction import (
        combined_training_config,
    )
    from scripts.train_distal_whole_body_q_only_diagnostic import load_samples

    _require(
        _file_sha256(prediction_path) == expected_file_sha256,
        "selection prediction file differs",
    )
    prediction = _load(prediction_path)
    _require(
        prediction["result_payload_sha256"] == expected_payload_sha256,
        "selection prediction payload differs",
    )
    config = combined_training_config(
        prediction["protocol"], prediction["binding"],
    )
    samples, source = load_samples(repo_root=repo_root, config=config)
    arm = prediction["arms"]["shared_constraint_135D"]
    candidates = {
        split: candidate_records(
            samples[split]["shared"], arm["predictions"][split],
        )
        for split in ("train", "validation", "test")
    }
    margin = validation_optimistic_margin(candidates["validation"])
    rules = {
        split: {
            "nominal": evaluate_rule(rows, rule="nominal"),
            "zero_threshold_least_intervention": evaluate_rule(
                rows, rule="least_intervention_predicted_safe",
            ),
            "validation_margin_least_intervention": evaluate_rule(
                rows, rule="least_intervention_predicted_safe", margin=margin,
            ),
            "minimum_predicted_risk": evaluate_rule(
                rows, rule="minimum_predicted_risk",
            ),
            "exact_oracle_minimum_intervention": evaluate_rule(
                rows, rule="exact_oracle_minimum_intervention",
            ),
        }
        for split, rows in candidates.items()
    }
    heldout = ("validation", "test")
    correction_rule_pass = {
        name: all(
            rules[split][name]["false_safe_selected_state_count"] == 0
            and rules[split][name]["abstained_recoverable_state_count"] == 0
            for split in heldout
        )
        for name in (
            "zero_threshold_least_intervention",
            "validation_margin_least_intervention",
            "minimum_predicted_risk",
        )
    }
    original_coverage_pass = bool(
        prediction["coverage_audit"]["training_authorized"]
    )
    original_prediction_pass = bool(prediction["heldout_prediction_gate_pass"])
    best_rule_pass = any(correction_rule_pass.values())
    value = {
        "schema_version": "vlsa_distal_whole_body_action_selection_audit.v1",
        "status": "complete",
        "scientific_result": True,
        "claim_scope": "Post-hoc offline selection audit over frozen MLP predictions and already simulated candidate outcomes; no new label, simulation, controller, or safety authorization.",
        "source": _git_identity(repo_root, expected_commit),
        "allocation": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(), "device": "cpu_read_only_audit",
        },
        "prediction_result": {
            "path": str(prediction_path),
            "file_sha256": expected_file_sha256,
            "payload_sha256": expected_payload_sha256,
        },
        "source_artifacts": source,
        "validation_optimistic_margin": margin,
        "rules": rules,
        "correction_rule_pass": correction_rule_pass,
        "original_coverage_gate_pass": original_coverage_pass,
        "original_prediction_gate_pass": original_prediction_pass,
        "offline_rule_identified": best_rule_pass,
        "model_only_live_correction_authorized": bool(
            original_coverage_pass and original_prediction_pass and best_rule_pass
        ),
        "exact_verified_pilot_recommended": bool(best_rule_pass),
    }
    value["result_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--prediction-file-sha256", required=True)
    parser.add_argument("--prediction-payload-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repo_root=args.repo_root.resolve(), prediction_path=args.prediction.resolve(),
        expected_file_sha256=args.prediction_file_sha256,
        expected_payload_sha256=args.prediction_payload_sha256,
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "validation_optimistic_margin": result["validation_optimistic_margin"],
        "correction_rule_pass": result["correction_rule_pass"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
