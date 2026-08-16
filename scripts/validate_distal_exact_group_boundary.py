#!/usr/bin/env python3
"""Independently replay and validate the palm/L5/L6 boundary canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def _scientific_view(record: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(record))
    for key in ("allocation", "source", "result_payload_sha256"):
        value.pop(key, None)
    curve = value.get("source_curve") or {}
    for key in ("path", "file_sha256", "result_payload_sha256"):
        curve.pop(key, None)
    exact = value.get("exact_case") or {}
    for key in (
        "source_result", "source_result_file_sha256",
        "source_result_payload_sha256",
    ):
        exact.pop(key, None)
    return value


def validate(
    *, repo_root: Path, config_path: Path, producer_dir: Path,
    replay_dir: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, VALIDATION_SCHEMA, canonical, load_cases, load_config,
        payload_sha256, summarize_cases,
    )

    config = load_config(config_path)
    cases = load_cases(repo_root / config["selection_manifest"], config)

    def load_directory(path: Path) -> list[dict[str, Any]]:
        output = []
        for case in cases:
            record_path = path / (case["case_id"] + ".json")
            _require(record_path.is_file(), "exact-group case result is missing")
            record = _load(record_path)
            _require(record.get("schema_version") == CASE_SCHEMA,
                     "exact-group case result schema differs")
            _require(record.get("case_id") == case["case_id"],
                     "exact-group case identity differs")
            historical = config.get("progressive_historical_source_commits", {})
            case_expected_commit = str(historical.get(case["case_id"], expected_commit))
            _require(record.get("source", {}).get("commit") == case_expected_commit,
                     "exact-group source commit differs")
            _require(record.get("result_payload_sha256") == payload_sha256(record),
                     "exact-group result payload differs")
            expected_aegis = bool(
                config["candidate_bank"][
                    "released_AEGIS_EE_applied_to_every_candidate"
                ]
            )
            _require(record.get("original_AEGIS_EE_QP_enabled") is expected_aegis,
                     "original AEGIS EE-QP execution mode differs")
            _require(record.get("learned_correction_QP_enabled") is False,
                     "learned correction QP was enabled")
            output.append(record)
        return output

    producer = load_directory(producer_dir)
    replay = load_directory(replay_dir)
    mismatches = [
        left["case_id"]
        for left, right in zip(producer, replay)
        if canonical(_scientific_view(left)) != canonical(_scientific_view(right))
    ]
    _require(not mismatches, "independent exact-group replay differs")
    producer_summary = summarize_cases(producer, config)
    replay_summary = summarize_cases(replay, config)
    _require(canonical(producer_summary) == canonical(replay_summary),
             "independent exact-group summary differs")
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": (
            "passed_q_only_prediction_coverage"
            if producer_summary["q_only_prediction_gate_authorized"]
            else "passed_same_bank" if producer_summary[
                "same_bank_grouped_collection_authorized"
            ] else "passed_targeted_coverage_canary"
            if producer_summary["targeted_coverage_canary_pass"]
            else "apparatus_pass_bank_revision_required"
            if producer_summary["apparatus_pass"] else "apparatus_no_go"
        ),
        "scientific_result": False,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "progressive_case_source_commits": {
            case["case_id"]: str(
                config.get("progressive_historical_source_commits", {}).get(
                    case["case_id"], expected_commit,
                )
            )
            for case in cases
        },
        "allocation": producer[0]["allocation"],
        "config": config,
        "independent_replay": {
            "case_count": len(cases),
            "mismatch_case_ids": mismatches,
            "exact_scientific_reproduction": not mismatches,
        },
        "summary": producer_summary,
        "same_bank_grouped_collection_authorized": producer_summary[
            "same_bank_grouped_collection_authorized"
        ],
        "targeted_coverage_canary_pass": producer_summary[
            "targeted_coverage_canary_pass"
        ],
        "training_authorized": producer_summary[
            "q_only_prediction_gate_authorized"
        ],
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": (
            "prospective_initially_safe_two_sided_split_coverage_authorizes_Q_only_prediction_gate"
            if producer_summary["q_only_prediction_gate_authorized"]
            else "exact_group_boundary_apparatus_and_three_group_bank_pass"
            if producer_summary["same_bank_grouped_collection_authorized"]
            else "earlier_query_targeted_L5_coverage_canary_pass"
            if producer_summary["targeted_coverage_canary_pass"]
            else "exact_group_boundary_apparatus_pass_but_candidate_bank_needs_revision"
            if producer_summary["apparatus_pass"]
            else "exact_group_boundary_apparatus_no_go"
        ),
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, key="validation_payload_sha256",
    )
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        producer_dir=args.producer_dir.resolve(),
        replay_dir=args.replay_dir.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "summary": value["summary"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
