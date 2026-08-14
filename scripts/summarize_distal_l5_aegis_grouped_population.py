#!/usr/bin/env python3
"""Summarize the validated 15-state AEGIS-consistent L5 population."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _load, _require, _sha256,
)
from scripts.summarize_distal_l5_aegis_grouped_canary import summarize
from scripts.validate_distal_l5_aegis_consistent_risk import canonical


def _add_coverage(left: list[dict[str, int]], right: list[dict[str, int]]) -> None:
    for aggregate, record in zip(left, right):
        for key, value in record.items():
            aggregate[key] += int(value)


def summarize_population(
    *, repo_root: Path, producer_root: Path, producer_commit: str,
    validator_commit: str, summary_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_aegis_grouped_summary import (
        row_coverage, state_classification, training_eligible_state,
        training_readiness,
    )

    indices = list(range(15))
    result_paths = [producer_root / ("case-%02d" % index) / "result.json"
                    for index in indices]
    validation_paths = [producer_root / ("case-%02d" % index) / "validation.json"
                        for index in indices]
    output = summarize(
        repo_root=repo_root,
        result_paths=result_paths,
        validation_paths=validation_paths,
        producer_commit=producer_commit,
        validator_commit=validator_commit,
        summary_commit=summary_commit,
        expected_count=15,
        required_splits=("diagnostic", "train", "validation"),
    )
    by_split: dict[str, list[dict[str, int]]] = {}
    eligible_by_split: dict[str, list[dict[str, int]]] = {}
    useful_by_split: dict[str, list[int]] = {}
    all_known_candidates_by_split: dict[str, int] = {}
    known_candidates_by_split: dict[str, int] = {}
    for result_path in result_paths:
        result = _load(result_path)
        split = result["population_binding"]["case"]["split"]
        coverage = row_coverage(result["candidates"])
        if split not in by_split:
            by_split[split] = [{key: 0 for key in record} for record in coverage]
            eligible_by_split[split] = [
                {key: 0 for key in record} for record in coverage
            ]
            useful_by_split[split] = [0] * 7
            all_known_candidates_by_split[split] = 0
            known_candidates_by_split[split] = 0
        _add_coverage(by_split[split], coverage)
        known_count = sum(
            candidate["terminal_status"] != "UNKNOWN_TIMEOUT"
            for candidate in result["candidates"]
        )
        all_known_candidates_by_split[split] += known_count
        classification = state_classification(result["candidates"])
        if not training_eligible_state(classification):
            continue
        _add_coverage(eligible_by_split[split], coverage)
        known_candidates_by_split[split] += known_count
        for row, record in enumerate(coverage):
            if (record["known_safe_candidate_count"] > 0
                    and record["known_unsafe_candidate_count"] > 0
                    and record["near_boundary_known_candidate_count"] > 0):
                useful_by_split[split][row] += 1

    thresholds = {
        "minimum_train_samples": 100,
        "minimum_validation_samples": 40,
        "minimum_known_safe_candidates_per_row": 20,
        "minimum_known_unsafe_candidates_per_row": 20,
        "minimum_near_boundary_candidates_per_row": 20,
        "minimum_train_useful_boundary_states_per_row": 3,
        "minimum_validation_useful_boundary_states_per_row": 1,
    }
    fit_row_coverage = [
        {key: 0 for key in eligible_by_split["train"][row]}
        for row in range(7)
    ]
    _add_coverage(fit_row_coverage, eligible_by_split["train"])
    _add_coverage(fit_row_coverage, eligible_by_split["validation"])
    sample_gates, row_gates, training_authorized = training_readiness(
        # Diagnostic cases remain positive controls and cannot satisfy a
        # train/validation learning gate.
        aggregate_rows=fit_row_coverage,
        useful_boundary_states=useful_by_split,
        known_candidates=known_candidates_by_split,
        thresholds=thresholds,
    )
    output.update({
        "schema_version": "vlsa_distal_l5_aegis_grouped_population_summary.v1",
        "coverage_by_split": by_split,
        "training_eligible_coverage_by_split": eligible_by_split,
        "fit_row_coverage": fit_row_coverage,
        "useful_boundary_state_count_by_split": useful_by_split,
        "all_known_candidate_count_by_split": all_known_candidates_by_split,
        "known_candidate_count_by_split": known_candidates_by_split,
        "training_readiness_thresholds": thresholds,
        "sample_gates": sample_gates,
        "learned_row_gates": row_gates,
        "training_authorized": training_authorized,
        "next_gate": (
            "freeze_episode_grouped_dataset_before_training"
            if training_authorized else
            "target_additional_episode_states_for_failed_L5_row_gates"
        ),
    })
    output.pop("result_payload_sha256", None)
    output["result_payload_sha256"] = _sha256(canonical(output))
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--summary-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = summarize_population(
        repo_root=args.repo_root.resolve(),
        producer_root=args.producer_root.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
        summary_commit=args.summary_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print({
        "status": output["status"],
        "state_classifications": {
            category: sum(state["classification"] == category
                          for state in output["states"])
            for category in sorted({state["classification"]
                                    for state in output["states"]})
        },
        "learned_row_gates": output["learned_row_gates"],
        "training_authorized": output["training_authorized"],
        "next_gate": output["next_gate"],
        "result_payload_sha256": output["result_payload_sha256"],
    }, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
