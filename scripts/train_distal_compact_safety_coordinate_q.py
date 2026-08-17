#!/usr/bin/env python3
"""Train the compact Q model and audit inference-only bank selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_whole_body_q_only_diagnostic import (
    _arrays, _predict, load_samples, shared_metrics, train_arm,
)


def _load_all_candidate_samples(
    *, repo_root: Path, training_config: Mapping[str, Any],
    compact_config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load every known/UNKNOWN candidate without producing a new label."""
    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        MODEL_ROWS, safety_coordinate_feature,
    )
    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, load_cases, load_config as load_cohort_config,
        payload_sha256 as source_payload_sha256,
    )
    from main.multilink_ellipsoid.whole_body_q_only_diagnostic import (
        row_future_risk,
    )

    output = {
        split: [] for split in training_config["dataset"]["evaluation_splits"]
    }
    cases = []
    excluded = []
    rejected = []
    for source in training_config["sources"]:
        cohort_path = repo_root / source["cohort_config"]
        _require(
            _file_sha256(cohort_path) == source["cohort_config_file_sha256"],
            "compact safety-coordinate cohort config differs",
        )
        cohort = load_cohort_config(cohort_path)
        selections = load_cases(repo_root / cohort["selection_manifest"], cohort)
        root = Path(source["artifact_root"])
        validation = _load(root / "validation.json")
        progressive = validation.get("progressive_case_source_commits", {})
        for selection in selections:
            split = str(selection["split"])
            if split not in output:
                continue
            case_id = str(selection["case_id"])
            record = _load(root / "producer" / (case_id + ".json"))
            expected_commit = str(
                progressive.get(case_id, source["artifact_commit"])
            )
            _require(
                record.get("schema_version") == CASE_SCHEMA
                and record.get("case_id") == case_id
                and record.get("source", {}).get("commit") == expected_commit
                and record.get("result_payload_sha256")
                == source_payload_sha256(record),
                "compact safety-coordinate case payload differs",
            )
            if record.get("status") != "complete":
                rejection = record.get("rejection") or {}
                _require(
                    record.get("status")
                    == "retained_scientific_rejection_initial_CAR"
                    and rejection.get("retained") is True
                    and rejection.get("candidate_outcomes_observed") is False,
                    "compact safety-coordinate retained rejection differs",
                )
                rejected.append({
                    "state_id": case_id, "split": split,
                    "reason": rejection["code"],
                })
                continue
            exact = record["exact_case"]
            initial = exact["exact_group_target"]
            prevention_safe = all(
                float(initial["initial_group_normalized_radial_slack"][group])
                > 0.0
                and int(initial["initial_group_contact_sample_count"][group]) == 0
                for group in ("palm", "L5", "L6", "L7")
            )
            if not prevention_safe:
                excluded.append({
                    "state_id": case_id, "split": split,
                    "reason": "initially_unsafe_recovery_audit",
                })
                continue
            known_count = 0
            unknown_count = 0
            for order, candidate in enumerate(exact["candidates"]):
                target = candidate["exact_group_target"]
                known = bool(target["known_outcome"])
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
                    "physical_veto": bool(candidate["replayed_physical_veto"]),
                }
                for row in MODEL_ROWS:
                    sample = {
                        **common,
                        "row_index": int(row),
                        "balance_id": f"{case_id}|row-{row}",
                        "feature": safety_coordinate_feature(
                            exact, candidate, row,
                            translation_scale=float(compact_config["feature"][
                                "translation_scale_m_per_action_unit"
                            ]),
                        ),
                    }
                    if known:
                        sample["risk"] = row_future_risk(candidate, row)
                    output[split].append(sample)
            cases.append({
                "state_id": case_id,
                "split": split,
                "source": source["name"],
                "known_candidate_count": known_count,
                "unknown_candidate_count": unknown_count,
                "payload_sha256": record["result_payload_sha256"],
            })
    _require(output["train"] and output["validation"] and output["test"],
             "compact safety-coordinate samples are empty")
    return output, {
        "cases": cases,
        "excluded_initially_unsafe_cases": excluded,
        "excluded_retained_scientific_rejections": rejected,
        "test_artifacts_accessed": True,
    }


def _known(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(sample) for sample in samples if bool(sample["known_outcome"])]


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        RESULT_SCHEMA, candidate_records, evaluate_selector, file_sha256,
        load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.whole_body_q_only_prediction import (
        combined_training_config, load_binding, load_protocol,
    )

    compact = load_config(config_path)
    protocol_path = repo_root / compact["base_protocol"]["path"]
    _require(
        file_sha256(protocol_path) == compact["base_protocol"]["file_sha256"],
        "compact safety-coordinate base protocol differs",
    )
    protocol = load_protocol(protocol_path)
    binding_path = Path(compact["binding"]["path"])
    _require(
        file_sha256(binding_path) == compact["binding"]["file_sha256"],
        "compact safety-coordinate binding differs",
    )
    binding = load_binding(binding_path, protocol)
    baseline_prediction_path = Path(compact["baseline_prediction"]["path"])
    baseline_action_path = Path(compact["baseline_action_audit"]["path"])
    _require(
        file_sha256(baseline_prediction_path)
        == compact["baseline_prediction"]["file_sha256"]
        and file_sha256(baseline_action_path)
        == compact["baseline_action_audit"]["file_sha256"],
        "compact safety-coordinate baseline file differs",
    )
    baseline_prediction = _load(baseline_prediction_path)
    baseline_action = _load(baseline_action_path)
    _require(
        baseline_prediction.get("result_payload_sha256")
        == compact["baseline_prediction"]["payload_sha256"]
        and baseline_action.get("result_payload_sha256")
        == compact["baseline_action_audit"]["payload_sha256"],
        "compact safety-coordinate baseline payload differs",
    )

    training_config = combined_training_config(protocol, binding)
    # Reuse the frozen loader first so source validation and eligible-state
    # identities remain exactly aligned with the established 9D/33D/135D run.
    established, established_source = load_samples(
        repo_root=repo_root, config=training_config,
    )
    samples, source = _load_all_candidate_samples(
        repo_root=repo_root, training_config=training_config,
        compact_config=compact,
    )
    for split in compact["evaluation"]["splits"]:
        established_states = {
            sample["state_id"] for sample in established[split]["shared"]
        }
        compact_states = {sample["state_id"] for sample in samples[split]}
        _require(established_states == compact_states,
                 "compact safety-coordinate eligible states differ")

    train_samples = _known(samples["train"])
    trained = train_arm(
        train_samples, feature_key="feature", target_key="risk",
        input_dimension=int(compact["feature"]["input_dimension"]),
        output_dimension=1, model_config=compact["model"],
    )
    known_predictions = {}
    all_predictions = {}
    metrics = {}
    selectors = {}
    records = {}
    for split in compact["evaluation"]["splits"]:
        known = _known(samples[split])
        known_x, _, _ = _arrays(known, "feature", "risk")
        known_prediction = _predict(trained["bundle"], known_x).tolist()
        all_x = [sample["feature"] for sample in samples[split]]
        all_prediction = _predict(trained["bundle"], all_x).tolist()
        known_predictions[split] = known_prediction
        all_predictions[split] = all_prediction
        metrics[split] = shared_metrics(
            known, known_prediction,
            group_rows=compact["group_rows"],
            near_boundary_abs_risk=float(
                compact["evaluation"]["near_boundary_abs_risk"]
            ),
        )
        records[split] = candidate_records(
            samples[split], all_prediction,
            primary_rows=compact["primary_inference_rows"],
            physical_rows=compact["all_physical_audit_rows"],
        )
        selectors[split] = {
            rule: evaluate_selector(records[split], rule=rule)
            for rule in ["nominal", *compact["selectors"]]
        }

    model = {key: value for key, value in trained.items() if key != "bundle"}
    baseline_rules = baseline_action["all_candidate_rules"]
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_inference_only_diagnostic",
        "scientific_result": True,
        "claim_scope": compact["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": allocation_record(),
        "config": compact,
        "binding": binding,
        "source_artifacts": source,
        "established_source_artifacts": established_source,
        "dataset": {
            "eligible_state_count": {
                split: len({sample["state_id"] for sample in samples[split]})
                for split in compact["evaluation"]["splits"]
            },
            "known_candidate_count": {
                split: len(_known(samples[split])) // 7
                for split in compact["evaluation"]["splits"]
            },
            "unknown_candidate_count": {
                split: sum(
                    not sample["known_outcome"] for sample in samples[split]
                ) // 7
                for split in compact["evaluation"]["splits"]
            },
            "known_row_sample_count": {
                split: len(_known(samples[split]))
                for split in compact["evaluation"]["splits"]
            },
            "all_candidates_scored_at_inference": True,
            "new_simulator_rollout_count": 0,
            "normalization_fit_on_train_only": True,
            "test_reused_posthoc_diagnostic": True,
        },
        "compact_shared_7D": {
            "model": model,
            "known_predictions": known_predictions,
            "all_candidate_predictions": all_predictions,
            "metrics": metrics,
            "selectors": selectors,
        },
        "baseline_reference": {
            "relative_endpoint_9D": baseline_prediction["arms"][
                "relative_endpoint_9D"
            ]["metrics"],
            "direct_L5_OSC_33D": baseline_prediction["arms"][
                "direct_L5_OSC_33D"
            ]["metrics"],
            "shared_constraint_135D": baseline_prediction["arms"][
                "shared_constraint_135D"
            ]["metrics"],
            "shared_135D_all_candidate_rules": {
                split: {
                    rule: baseline_rules[split][rule]
                    for rule in (
                        "nominal", "zero_threshold_least_intervention",
                        "minimum_predicted_risk",
                    )
                }
                for split in compact["evaluation"]["splits"]
            },
        },
        "inference_only": True,
        "simulation_rollouts": False,
        "diagnostic_only": True,
        "correction_authorized": False,
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
        "validation": result["compact_shared_7D"]["selectors"]["validation"],
        "test": result["compact_shared_7D"]["selectors"]["test"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
