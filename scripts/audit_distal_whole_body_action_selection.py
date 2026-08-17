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


def _all_candidate_samples(
    *, repo_root: Path, config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build causal shared-model features for known and timeout candidates."""

    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, load_cases, load_config as load_cohort_config,
        payload_sha256 as source_payload_sha256,
    )
    from main.multilink_ellipsoid.whole_body_q_only_diagnostic import (
        CLAIMED_ROWS, row_future_risk, shared_constraint_feature,
    )

    output = {
        split: [] for split in config["dataset"]["evaluation_splits"]
    }
    case_records = []
    for source in config["sources"]:
        cohort_path = repo_root / source["cohort_config"]
        _require(
            _file_sha256(cohort_path) == source["cohort_config_file_sha256"],
            "selection cohort config differs",
        )
        cohort = load_cohort_config(cohort_path)
        selections = load_cases(repo_root / cohort["selection_manifest"], cohort)
        root = Path(source["artifact_root"])
        for selection in selections:
            split = str(selection["split"])
            if split not in output:
                continue
            case_id = str(selection["case_id"])
            record = _load(root / "producer" / (case_id + ".json"))
            if record.get("status") != "complete":
                continue
            _require(
                record.get("schema_version") == CASE_SCHEMA
                and record.get("case_id") == case_id
                and record.get("result_payload_sha256")
                == source_payload_sha256(record),
                "selection candidate case payload differs",
            )
            exact = record["exact_case"]
            initial = exact["exact_group_target"]
            prevention_safe = all(
                float(initial["initial_group_normalized_radial_slack"][group]) > 0.0
                and int(initial["initial_group_contact_sample_count"][group]) == 0
                for group in ("palm", "L5", "L6", "L7")
            )
            if not prevention_safe:
                continue
            known_count = 0
            unknown_count = 0
            for order, candidate in enumerate(exact["candidates"]):
                known = bool(candidate["exact_group_target"]["known_outcome"])
                known_count += int(known)
                unknown_count += int(not known)
                common = {
                    "state_id": case_id,
                    "split": split,
                    "candidate_name": str(candidate["name"]),
                    "candidate_order": int(order),
                    "applied_correction_l2_action": float(candidate[
                        "source_effective_post_AEGIS_correction_l2_action"
                    ]),
                    "known_outcome": known,
                }
                for row in CLAIMED_ROWS:
                    sample = {
                        **common,
                        "row_index": int(row),
                        "feature": shared_constraint_feature(exact, candidate, row),
                    }
                    if known:
                        sample["risk"] = row_future_risk(candidate, row)
                    output[split].append(sample)
            case_records.append({
                "state_id": case_id, "split": split,
                "known_candidate_count": known_count,
                "unknown_candidate_count": unknown_count,
            })
    return output, {"cases": case_records}


def run(
    *, repo_root: Path, prediction_path: Path, expected_file_sha256: str,
    expected_payload_sha256: str, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.whole_body_action_selection_audit import (
        candidate_records, evaluate_ranked_exact_verification, evaluate_rule,
        predict_serialized_mlp, validation_optimistic_margin,
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
    all_samples, all_source = _all_candidate_samples(
        repo_root=repo_root, config=config,
    )
    all_predictions = {
        split: predict_serialized_mlp(
            [sample["feature"] for sample in all_samples[split]],
            arm["model"]["state_payload"],
        )
        for split in ("train", "validation", "test")
    }
    serialized_prediction_replay = {}
    for split in ("train", "validation", "test"):
        known = [
            (sample, predicted)
            for sample, predicted in zip(all_samples[split], all_predictions[split])
            if bool(sample["known_outcome"])
        ]
        frozen = list(zip(samples[split]["shared"], arm["predictions"][split]))
        _require(len(known) == len(frozen),
                 "selection serialized replay sample count differs")
        differences = []
        for (all_sample, all_predicted), (sample, predicted) in zip(known, frozen):
            _require(
                (
                    all_sample["state_id"], all_sample["candidate_name"],
                    all_sample["row_index"],
                ) == (
                    sample["state_id"], sample["candidate_name"],
                    sample["row_index"],
                ),
                "selection serialized replay identity differs",
            )
            differences.append(abs(float(all_predicted[0]) - float(predicted[0])))
        maximum = max(differences, default=0.0)
        _require(maximum <= 1.0e-5,
                 "selection serialized replay prediction differs")
        serialized_prediction_replay[split] = {
            "known_row_count": len(known), "maximum_absolute_error": maximum,
        }
    all_candidates = {
        split: candidate_records(all_samples[split], all_predictions[split])
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
    all_candidate_rules = {
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
        for split, rows in all_candidates.items()
    }
    heldout = ("validation", "test")
    correction_rule_pass = {
        name: all(
            all_candidate_rules[split][name]["false_safe_selected_state_count"] == 0
            and all_candidate_rules[split][name]["unknown_selected_state_count"] == 0
            and all_candidate_rules[split][name]["abstained_recoverable_state_count"] == 0
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
    ranked_known_only = {
        split: evaluate_ranked_exact_verification(candidates[split])
        for split in ("train", "validation", "test")
    }
    ranked_verification = {
        split: evaluate_ranked_exact_verification(all_candidates[split])
        for split in ("train", "validation", "test")
    }
    exact_verified_top5 = all(
        ranked_verification[split]["top_k_safe_support"]["5"]
        == ranked_verification[split]["recoverable_state_count"]
        for split in heldout
    )
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
        "all_candidate_source_artifacts": all_source,
        "serialized_prediction_replay": serialized_prediction_replay,
        "validation_optimistic_margin": margin,
        "known_outcome_only_rules": rules,
        "all_candidate_rules": all_candidate_rules,
        "correction_rule_pass": correction_rule_pass,
        "ranked_exact_verification_known_outcome_lower_bound": ranked_known_only,
        "ranked_exact_verification": ranked_verification,
        "original_coverage_gate_pass": original_coverage_pass,
        "original_prediction_gate_pass": original_prediction_pass,
        "offline_rule_identified": best_rule_pass,
        "model_only_live_correction_authorized": bool(
            original_coverage_pass and original_prediction_pass and best_rule_pass
        ),
        "exact_verified_top5_support_pass": exact_verified_top5,
        "exact_verified_pilot_recommended": bool(exact_verified_top5),
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
