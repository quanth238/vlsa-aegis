#!/usr/bin/env python3
"""Aggregate independently validated grouped AEGIS-consistent L5 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)
from scripts.validate_distal_l5_aegis_consistent_risk import canonical


def summarize(
    *, repo_root: Path, result_paths: Sequence[Path],
    validation_paths: Sequence[Path], producer_commit: str,
    validator_commit: str, summary_commit: str, expected_count: int = 2,
    required_splits: Sequence[str] = ("train", "validation"),
) -> dict[str, Any]:
    from main.multilink_ellipsoid.grouped_query_action_risk import RESULT_SCHEMA
    from main.multilink_ellipsoid.l5_aegis_grouped_summary import (
        LEARNED_L5_ROWS, row_coverage, state_classification,
    )

    _require(len(result_paths) == len(validation_paths) == int(expected_count),
             "grouped summary paired artifact count differs")
    states = []
    rows = [None] * 7
    for result_path, validation_path in zip(result_paths, validation_paths):
        result = _load(result_path)
        validation = _load(validation_path)
        _require(result["schema_version"] == RESULT_SCHEMA,
                 "grouped result schema differs")
        _require(result["source"]["commit"] == producer_commit,
                 "grouped producer commit differs")
        _require(validation["status"] == "complete"
                 and validation["scientific_result"] is True,
                 "grouped validation is incomplete")
        _require(validation["validator_source"]["commit"] == validator_commit,
                 "grouped validator commit differs")
        _require(validation["producer"]["file_sha256"]
                 == _file_sha256(result_path),
                 "grouped validation result binding differs")
        _require(validation["producer"]["result_payload_sha256"]
                 == result["result_payload_sha256"],
                 "grouped validation payload binding differs")
        candidates = result["candidates"]
        coverage = row_coverage(candidates)
        if rows[0] is None:
            rows = [{**record, "state_count": 0, "episode_count": 0}
                    for record in coverage]
        else:
            for aggregate, record in zip(rows, coverage):
                for key, value in record.items():
                    aggregate[key] += value
        for aggregate, record in zip(rows, coverage):
            if (record["known_safe_candidate_count"] > 0
                    or record["known_unsafe_candidate_count"] > 0):
                aggregate["state_count"] += 1
                aggregate["episode_count"] += 1
        binding = result["population_binding"]
        case = binding["case"]
        classification = state_classification(candidates)
        states.append({
            "case_id": case["case_id"],
            "split": case["split"],
            "state_id": binding["retained_state"]["state_id"],
            "classification": classification,
            "safe_candidate_count": sum(bool(c["exact_safe"]) for c in candidates),
            "unsafe_candidate_count": sum(
                c["terminal_status"] != "UNKNOWN_TIMEOUT" and not bool(c["exact_safe"])
                for c in candidates
            ),
            "unknown_timeout_count": sum(
                c["terminal_status"] == "UNKNOWN_TIMEOUT" for c in candidates
            ),
            "result_path": str(result_path),
            "result_file_sha256": _file_sha256(result_path),
            "validation_path": str(validation_path),
            "validation_file_sha256": _file_sha256(validation_path),
            "validation_payload_sha256": validation["validation_payload_sha256"],
        })
    observed_splits = {state["split"] for state in states}
    _require(set(required_splits).issubset(observed_splits),
             "grouped summary required splits differ")
    _require(len({state["case_id"] for state in states}) == len(states),
             "grouped summary contains duplicate episodes")
    _require(len({state["state_id"] for state in states}) == len(states),
             "grouped summary contains duplicate states")
    mixed = sum(state["classification"] == "usable_mixed_support" for state in states)
    output = {
        "schema_version": "vlsa_distal_l5_aegis_grouped_canary_summary.v1",
        "status": "complete",
        "scientific_result": True,
        "source": _git_identity(repo_root, summary_commit),
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "learned_claim_rows": list(LEARNED_L5_ROWS),
        "diagnostic_rows": [3, 4, 5, 6],
        "states": states,
        "row_coverage": rows,
        "usable_mixed_support_state_count": mixed,
        "all_failures_retained": True,
        "training_authorized": False,
        "next_gate": (
            "expand_episode_grouped_collection_with_identical_action_contract"
            if mixed == len(states) else
            "revise_state_selection_or_candidate_support_before_expansion"
        ),
    }
    output["result_payload_sha256"] = _sha256(canonical(output))
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--validation", action="append", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--summary-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = summarize(
        repo_root=args.repo_root.resolve(),
        result_paths=[path.resolve() for path in args.result],
        validation_paths=[path.resolve() for path in args.validation],
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
        summary_commit=args.summary_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print({
        "status": output["status"],
        "states": output["states"],
        "next_gate": output["next_gate"],
        "result_payload_sha256": output["result_payload_sha256"],
    }, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
