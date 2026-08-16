#!/usr/bin/env python3
"""Build a frozen policy-conditioned Q/V dataset from validated rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def build(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> Mapping[str, Any]:
    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, VALIDATION_SCHEMA as SOURCE_VALIDATION_SCHEMA,
        payload_sha256 as source_payload_sha256,
    )
    from main.multilink_ellipsoid.policy_conditioned_value_dataset import (
        DATASET_SCHEMA, coverage_summary, extract_case_samples, load_config,
        payload_sha256, readiness_checks,
    )

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    source = config["source"]
    validation_path = Path(source["validation_file"])
    _require(
        _file_sha256(validation_path) == source["validation_file_sha256"],
        "policy-value source validation file differs",
    )
    validation = _load(validation_path)
    _require(
        validation["schema_version"] == SOURCE_VALIDATION_SCHEMA
        and validation["validation_payload_sha256"]
        == source["validation_payload_sha256"]
        == source_payload_sha256(validation, "validation_payload_sha256")
        and bool(validation["independent_replay"]["exact_scientific_reproduction"])
        and bool(validation["summary"]["trajectory_policy_value_apparatus_pass"])
        and bool(validation["summary"]["trajectory_context_complete"])
        and float(validation["summary"]["trajectory_maximum_bellman_residual"]) == 0.0
        and int(validation["summary"]["physical_false_safe_count"]) == 0,
        "policy-value source validation differs",
    )
    split_lookup = {
        str(case_id): str(split)
        for split, case_ids in config["dataset"]["split_case_ids"].items()
        for case_id in case_ids
    }
    case_samples = []
    bindings = []
    for case_id in sorted(split_lookup):
        binding = source["case_artifacts"][case_id]
        path = Path(source["producer_root"]) / binding["filename"]
        _require(_file_sha256(path) == binding["file_sha256"], "policy-value case file differs")
        case = _load(path)
        _require(
            case["schema_version"] == CASE_SCHEMA
            and case["case_id"] == case_id
            and case["result_payload_sha256"] == binding["payload_sha256"]
            == source_payload_sha256(case)
            and str(case["selection"].get("split")) == split_lookup[case_id]
            and bool(case["exact_case"]["state_hash_matches"])
            and bool(case["exact_case"]["source_replay_exact"]),
            "policy-value case payload differs",
        )
        extracted = extract_case_samples(
            case,
            learned_groups=config["dataset"]["learned_groups"],
            group_order=config["dataset"]["group_order"],
            group_rows=config["dataset"]["group_rows"],
            eligible_value_phases=config["dataset"]["eligible_value_phases"],
            prefix_action_count=int(config["features"]["prefix_action_count"]),
            translation_scale_m_per_action_unit=float(
                config["features"]["translation_scale_m_per_action_unit"]
            ),
        )
        case_samples.append(extracted)
        bindings.append({
            "case_id": case_id, "split": split_lookup[case_id],
            "file": str(path), "file_sha256": binding["file_sha256"],
            "payload_sha256": binding["payload_sha256"],
            "unknown_candidate_count": extracted["unknown_candidate_count"],
        })
    coverage = coverage_summary(
        case_samples, learned_groups=config["dataset"]["learned_groups"],
    )
    checks = readiness_checks(
        coverage, learned_groups=config["dataset"]["learned_groups"],
        required_two_sided_states=config["gate"]["required_two_sided_states"],
        required_value_episode_groups=config["gate"]["required_value_episode_groups"],
    )
    result = {
        "schema_version": DATASET_SCHEMA,
        "status": "ready_for_matched_Q_and_policy_value_training"
        if checks["training_ready"] else "coverage_no_go",
        "scientific_result": False,
        "claim_scope": config["claim_scope"],
        "source": identity,
        "config_file": str(config_path),
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "source_validation": {
            "file": str(validation_path),
            "file_sha256": source["validation_file_sha256"],
            "payload_sha256": source["validation_payload_sha256"],
        },
        "artifact_bindings": bindings,
        "sign_convention": "positive_is_unsafe",
        "learned_object": "policy_conditioned_future_violation_not_OSC_execution",
        "internal_substeps": "label_authority_only",
        "case_samples": case_samples,
        "coverage": coverage,
        "readiness_checks": checks,
        "training_authorized": checks["training_ready"],
        "correction_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    result["dataset_payload_sha256"] = payload_sha256(
        result, "dataset_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "coverage": result["coverage"],
        "readiness_checks": result["readiness_checks"],
        "dataset_payload_sha256": result["dataset_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
