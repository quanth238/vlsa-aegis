#!/usr/bin/env python3
"""Aggregate the targeted L5 extension with its immutable parent population."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)
from scripts.validate_distal_l5_aegis_consistent_risk import canonical


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
    *, repo_root: Path, parent_summary_path: Path,
    result_paths: Sequence[Path], validation_paths: Sequence[Path],
    producer_commit: str, validator_commit: str, summary_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.adaptive_query_action_risk_v2 import (
        RESULT_SCHEMA_V3,
    )
    from main.multilink_ellipsoid.l5_aegis_grouped_summary import (
        row_coverage, state_classification, training_eligible_state,
        training_readiness,
    )
    from main.multilink_ellipsoid.targeted_l5_boundary_extension import (
        load_cases, load_config,
    )

    target_config = load_config(
        repo_root / "configs/vlsa_distal_l5_targeted_extension.v1.json",
        repo_root=repo_root,
    )
    target_cases = load_cases(
        repo_root / "manifests/vlsa_distal_l5_targeted_extension.v1.jsonl",
        target_config,
    )
    parent = _load(parent_summary_path)
    parent_binding = target_config["parent_population"]
    _require(_file_sha256(parent_summary_path)
             == parent_binding["summary_file_sha256"],
             "targeted parent summary file differs")
    _require(parent["result_payload_sha256"]
             == parent_binding["summary_payload_sha256"],
             "targeted parent summary payload differs")
    _require(len(result_paths) == len(validation_paths) == len(target_cases),
             "targeted extension artifact count differs")

    coverage_by_split = {"train": _blank_rows(), "validation": _blank_rows()}
    eligible_by_split = {"train": _blank_rows(), "validation": _blank_rows()}
    useful_by_split = {"train": [0] * 7, "validation": [0] * 7}
    known_by_split = {"train": 0, "validation": 0}
    all_known_by_split = {"train": 0, "validation": 0}
    states = []
    for index, (result_path, validation_path, expected_case) in enumerate(
        zip(result_paths, validation_paths, target_cases)
    ):
        result = _load(result_path)
        validation = _load(validation_path)
        _require(result["schema_version"] == RESULT_SCHEMA_V3,
                 "targeted result schema differs")
        _require(result["source"]["commit"] == producer_commit,
                 "targeted producer commit differs")
        _require(validation["status"] == "complete"
                 and validation["scientific_result"] is True,
                 "targeted validation incomplete")
        _require(validation["validator_source"]["commit"] == validator_commit,
                 "targeted validator commit differs")
        _require(validation["producer"]["file_sha256"]
                 == _file_sha256(result_path),
                 "targeted validation file binding differs")
        binding = result["population_binding"]["targeted_extension"]
        _require(int(binding["target_case_index"]) == index
                 and binding["target_case"] == expected_case
                 and result["population_binding"]["case"] == expected_case,
                 "targeted case binding differs")
        split = expected_case["split"]
        candidates = result["candidates"]
        coverage = row_coverage(candidates)
        _add(coverage_by_split[split], coverage)
        known_count = sum(
            candidate["terminal_status"] != "UNKNOWN_TIMEOUT"
            for candidate in candidates
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
            "case_id": expected_case["case_id"],
            "split": split,
            "state_id": result["population_binding"]["retained_state"]["state_id"],
            "target_row": result["adaptive_boundary_sampling"]["target_row"],
            "classification": classification,
            "safe_candidate_count": sum(
                bool(candidate["exact_safe"]) for candidate in candidates
            ),
            "known_unsafe_candidate_count": sum(
                candidate["terminal_status"] != "UNKNOWN_TIMEOUT"
                and not bool(candidate["exact_safe"])
                for candidate in candidates
            ),
            "unknown_timeout_count": sum(
                candidate["terminal_status"] == "UNKNOWN_TIMEOUT"
                for candidate in candidates
            ),
            "result_path": str(result_path),
            "result_file_sha256": _file_sha256(result_path),
            "validation_path": str(validation_path),
            "validation_file_sha256": _file_sha256(validation_path),
            "validation_payload_sha256": validation[
                "validation_payload_sha256"
            ],
        })

    cumulative_eligible = {
        split: copy.deepcopy(parent["training_eligible_coverage_by_split"][split])
        for split in ("train", "validation")
    }
    cumulative_useful = {
        split: list(parent["useful_boundary_state_count_by_split"][split])
        for split in ("train", "validation")
    }
    cumulative_known = {
        split: int(parent["known_candidate_count_by_split"][split])
        for split in ("train", "validation")
    }
    for split in ("train", "validation"):
        _add(cumulative_eligible[split], eligible_by_split[split])
        cumulative_useful[split] = [
            int(left) + int(right) for left, right in zip(
                cumulative_useful[split], useful_by_split[split]
            )
        ]
        cumulative_known[split] += known_by_split[split]
    cumulative_fit = _blank_rows()
    _add(cumulative_fit, cumulative_eligible["train"])
    _add(cumulative_fit, cumulative_eligible["validation"])
    registered = parent["training_readiness_thresholds"]
    sample_gates, row_gates, coverage_passes = training_readiness(
        aggregate_rows=cumulative_fit,
        useful_boundary_states=cumulative_useful,
        known_candidates=cumulative_known,
        thresholds=registered,
    )
    requested = {
        "additional_row0_train_state": useful_by_split["train"][0] >= 1,
        "row1_train_boundary_states": cumulative_useful["train"][1] >= 3,
        "row1_validation_boundary_states": cumulative_useful["validation"][1] >= 1,
        "row2_train_boundary_states": cumulative_useful["train"][2] >= 3,
        "row2_validation_boundary_states": cumulative_useful["validation"][2] >= 1,
    }
    output = {
        "schema_version": "vlsa_distal_l5_targeted_extension_summary.v1",
        "status": "complete",
        "scientific_result": True,
        "source": _git_identity(repo_root, summary_commit),
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "target_config": target_config,
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
        "requested_coverage_gates": requested,
        "requested_coverage_passes": all(requested.values()),
        "coverage_authorizes_matched_feature_audit": bool(coverage_passes),
        "all_failures_retained": True,
        "timeouts_censored_unknown": True,
        "training_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "next_gate": (
            "run_preregistered_no_training_matched_complete_input_feature_audit"
            if coverage_passes else
            "target_more_distinct_episode_states_for_remaining_row_gates"
        ),
    }
    output["result_payload_sha256"] = _sha256(canonical(output))
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--validation", action="append", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--summary-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(
        repo_root=args.repo_root.resolve(),
        parent_summary_path=args.parent_summary.resolve(),
        result_paths=[path.resolve() for path in args.result],
        validation_paths=[path.resolve() for path in args.validation],
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
        summary_commit=args.summary_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "states": result["states"],
        "requested_coverage_gates": result["requested_coverage_gates"],
        "coverage_authorizes_matched_feature_audit": result[
            "coverage_authorizes_matched_feature_audit"
        ],
        "next_gate": result["next_gate"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
