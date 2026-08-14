#!/usr/bin/env python3
"""Aggregate validated moka adaptive-v3 labels with prior L5 coverage."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)
from scripts.validate_distal_l5_aegis_consistent_risk import canonical


SUMMARY_SCHEMA = "vlsa_distal_l5_moka_noise_adaptive_summary_result.v1"


def _blank_rows() -> list[dict[str, int]]:
    return [{
        "known_safe_candidate_count": 0,
        "known_unsafe_candidate_count": 0,
        "near_boundary_known_candidate_count": 0,
        "active_witness_known_candidate_count": 0,
        "unknown_timeout_candidate_count": 0,
    } for _ in range(7)]


def _add(left: list[dict[str, int]], right: list[dict[str, int]]) -> None:
    for aggregate, record in zip(left, right):
        for key, value in record.items():
            aggregate[key] += int(value)


def summarize(
    *, repo_root: Path, producer_root: Path, boundary_root: Path,
    discovery_root: Path, parent_summary_path: Path,
    config_path: Path, manifest_path: Path,
    producer_commit: str, validator_commit: str, summary_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.adaptive_query_action_risk_v2 import RESULT_SCHEMA_V3
    from main.multilink_ellipsoid.l5_aegis_grouped_summary import (
        row_coverage, state_classification, training_eligible_state,
        training_readiness,
    )
    from main.multilink_ellipsoid.moka_noise_boundary import (
        load_cases, load_config, validate_discovery_binding,
    )

    summary_config = _load(config_path)
    _require(
        summary_config["schema_version"]
        == "vlsa_distal_l5_moka_noise_adaptive_summary.v1",
        "moka adaptive summary config differs",
    )
    boundary_config = load_config(
        repo_root / "configs/vlsa_distal_l5_moka_noise_boundary.v1.json",
        repo_root=repo_root,
    )
    cases = load_cases(manifest_path, boundary_config)
    validate_discovery_binding(
        config=boundary_config, cases=cases, discovery_root=discovery_root
    )
    boundary_validation_path = boundary_root / "validation.json"
    source = summary_config["source"]
    _require(
        _file_sha256(boundary_validation_path)
        == source["boundary_validation_file_sha256"],
        "moka boundary validation file differs",
    )
    boundary_validation = _load(boundary_validation_path)
    _require(
        boundary_validation["result_payload_sha256"]
        == source["boundary_validation_payload_sha256"],
        "moka boundary validation payload differs",
    )
    parent = _load(parent_summary_path)
    _require(_file_sha256(parent_summary_path) == source["parent_summary_file_sha256"],
             "moka adaptive parent file differs")
    _require(parent["result_payload_sha256"] == source["parent_summary_payload_sha256"],
             "moka adaptive parent payload differs")

    coverage_by_split = {
        split: _blank_rows() for split in ("train", "validation", "diagnostic")
    }
    eligible_by_split = {
        split: _blank_rows() for split in ("train", "validation", "diagnostic")
    }
    useful_by_split = {
        split: [0] * 7 for split in ("train", "validation", "diagnostic")
    }
    known_by_split = {split: 0 for split in ("train", "validation", "diagnostic")}
    all_known_by_split = dict(known_by_split)
    states = []
    for index, case in enumerate(cases):
        result_path = producer_root / ("case-%02d" % index) / "result.json"
        validation_path = producer_root / ("case-%02d" % index) / "validation.json"
        result = _load(result_path)
        validation = _load(validation_path)
        _require(result["schema_version"] == RESULT_SCHEMA_V3,
                 "moka adaptive result schema differs")
        _require(result["source"]["commit"] == producer_commit,
                 "moka adaptive producer commit differs")
        _require(validation["status"] == "complete"
                 and validation["scientific_result"] is True,
                 "moka adaptive validation incomplete")
        _require(validation["validator_source"]["commit"] == validator_commit,
                 "moka adaptive validator commit differs")
        _require(validation["producer"]["file_sha256"] == _file_sha256(result_path),
                 "moka adaptive validation binding differs")
        _require(result["population_binding"]["case"] == case,
                 "moka adaptive case binding differs")
        split = case["split"]
        candidates = result["candidates"]
        coverage = row_coverage(candidates)
        _add(coverage_by_split[split], coverage)
        known_count = sum(
            item["terminal_status"] != "UNKNOWN_TIMEOUT" for item in candidates
        )
        all_known_by_split[split] += known_count
        classification = state_classification(candidates)
        if training_eligible_state(classification):
            _add(eligible_by_split[split], coverage)
            known_by_split[split] += known_count
            for row, record in enumerate(coverage):
                if (
                    record["known_safe_candidate_count"] > 0
                    and record["known_unsafe_candidate_count"] > 0
                    and record["near_boundary_known_candidate_count"] > 0
                ):
                    useful_by_split[split][row] += 1
        states.append({
            "case_id": case["case_id"],
            "split": split,
            "trajectory_group_id": case["trajectory_group_id"],
            "state_id": result["population_binding"]["retained_state"]["state_id"],
            "target_row": result["adaptive_boundary_sampling"]["target_row"],
            "classification": classification,
            "safe_candidate_count": sum(bool(item["exact_safe"]) for item in candidates),
            "known_unsafe_candidate_count": sum(
                item["terminal_status"] != "UNKNOWN_TIMEOUT"
                and not bool(item["exact_safe"]) for item in candidates
            ),
            "unknown_timeout_count": sum(
                item["terminal_status"] == "UNKNOWN_TIMEOUT" for item in candidates
            ),
            "row_coverage": coverage,
            "result_file_sha256": _file_sha256(result_path),
            "validation_file_sha256": _file_sha256(validation_path),
            "validation_payload_sha256": validation["validation_payload_sha256"],
        })

    cumulative_eligible = {
        split: copy.deepcopy(parent["cumulative_training_eligible_coverage_by_split"][split])
        for split in ("train", "validation")
    }
    cumulative_useful = {
        split: list(parent["cumulative_useful_boundary_state_count_by_split"][split])
        for split in ("train", "validation")
    }
    cumulative_known = {
        split: int(parent["cumulative_known_candidate_count_by_split"][split])
        for split in ("train", "validation")
    }
    for split in ("train", "validation"):
        _add(cumulative_eligible[split], eligible_by_split[split])
        cumulative_useful[split] = [
            old + new for old, new in zip(
                cumulative_useful[split], useful_by_split[split]
            )
        ]
        cumulative_known[split] += known_by_split[split]
    cumulative_fit = _blank_rows()
    _add(cumulative_fit, cumulative_eligible["train"])
    _add(cumulative_fit, cumulative_eligible["validation"])
    sample_gates, row_gates, all_three_pass = training_readiness(
        aggregate_rows=cumulative_fit,
        useful_boundary_states=cumulative_useful,
        known_candidates=cumulative_known,
        thresholds=summary_config["thresholds"],
    )
    row01_pass = bool(
        all(sample_gates.values())
        and all(record["passes"] for record in row_gates if record["row"] in (0, 1))
    )
    output = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "source": _git_identity(repo_root, summary_commit),
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "summary_config": summary_config,
        "parent_summary": {
            "path": str(parent_summary_path),
            "file_sha256": _file_sha256(parent_summary_path),
            "result_payload_sha256": parent["result_payload_sha256"],
        },
        "states": states,
        "extension_coverage_by_split": coverage_by_split,
        "extension_training_eligible_coverage_by_split": eligible_by_split,
        "extension_useful_boundary_state_count_by_split": useful_by_split,
        "extension_known_candidate_count_by_split": known_by_split,
        "extension_all_known_candidate_count_by_split": all_known_by_split,
        "cumulative_training_eligible_coverage_by_split": cumulative_eligible,
        "cumulative_useful_boundary_state_count_by_split": cumulative_useful,
        "cumulative_known_candidate_count_by_split": cumulative_known,
        "cumulative_fit_row_coverage": cumulative_fit,
        "sample_gates": sample_gates,
        "learned_row_gates": row_gates,
        "row01_diagnostic_feature_audit_supported": row01_pass,
        "three_output_L5_feature_audit_supported": bool(all_three_pass),
        "all_failures_retained": True,
        "timeouts_censored_unknown": True,
        "training_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "next_gate": (
            "preregister_matched_row01_feature_audit_row2_exact_monitor_only"
            if row01_pass and not all_three_pass else
            "preregister_matched_three_output_feature_audit"
            if all_three_pass else
            "collect_more_distinct_supported_boundary_states"
        ),
    }
    output["result_payload_sha256"] = _sha256(canonical(output))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--summary-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(
        repo_root=args.repo_root.resolve(), producer_root=args.producer_root.resolve(),
        boundary_root=args.boundary_root.resolve(), discovery_root=args.discovery_root.resolve(),
        parent_summary_path=args.parent_summary.resolve(), config_path=args.config.resolve(),
        manifest_path=args.manifest.resolve(), producer_commit=args.producer_commit,
        validator_commit=args.validator_commit, summary_commit=args.summary_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "states": result["states"],
        "cumulative_useful_boundary_state_count_by_split": result[
            "cumulative_useful_boundary_state_count_by_split"
        ],
        "learned_row_gates": result["learned_row_gates"],
        "next_gate": result["next_gate"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
