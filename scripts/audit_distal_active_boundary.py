#!/usr/bin/env python3
"""Audit row identifiability over validated, already-executed candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import _allocation_record
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    return _sha256(_canonical(payload))


def audit(
    *, repo_root: Path, producer_root: Path, coverage_validation_path: Path,
    config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.active_boundary_audit import (
        RESULT_SCHEMA, audit_state_row, load_config, row_interpretation,
    )
    from main.multilink_ellipsoid.grouped_query_action_risk import RESULT_SCHEMA as SOURCE_SCHEMA
    from main.multilink_ellipsoid.query_boundary_coverage import ROW_IDENTITIES

    config = load_config(config_path)
    _require(str(producer_root) == config["source"]["producer_root"],
             "active-boundary producer root differs")
    _require(str(coverage_validation_path)
             == config["source"]["validated_coverage_result"],
             "active-boundary validation path differs")
    _require(_file_sha256(coverage_validation_path)
             == config["source"]["validated_coverage_file_sha256"],
             "active-boundary validation file differs")
    coverage = _load(coverage_validation_path)
    _require(coverage["result_payload_sha256"]
             == config["source"]["validated_coverage_payload_sha256"],
             "active-boundary validation payload differs")
    _require(coverage["status"] == "validated"
             and coverage["gates"]["all_15_artifacts_complete_and_hash_valid"]
             and coverage["gates"]["all_replays_deterministic_and_bound"],
             "active-boundary source is not validated")
    _require(coverage["split_manifest"]["reserved_test_labels_absent"] is True,
             "active-boundary reserved test was touched")

    robust = float(config["finite_search"]["robust_boundary_margin_m"])
    variation = float(config["finite_search"]["minimum_meaningful_variation_m"])
    state_records = []
    artifact_records = []
    for artifact, state in zip(
        coverage["artifact_records"], coverage["state_classifications"]
    ):
        _require(artifact["case_id"] == state["case_id"],
                 "active-boundary state identity differs")
        result_path = Path(artifact["path"])
        try:
            result_path.relative_to(producer_root)
            inside_producer_root = True
        except ValueError:
            inside_producer_root = False
        _require(inside_producer_root,
                 "active-boundary artifact leaves producer root")
        _require(_file_sha256(result_path) == artifact["file_sha256"],
                 "active-boundary source artifact differs")
        result = _load(result_path)
        _require(result["schema_version"] == SOURCE_SCHEMA
                 and result["status"] == "complete"
                 and result["scientific_result"] is True,
                 "active-boundary source artifact incomplete")
        _require(result["result_payload_sha256"] == artifact["result_payload_sha256"]
                 == _payload_sha256(result),
                 "active-boundary source payload differs")
        _require(result["source"]["commit"] == config["source"]["producer_commit"]
                 and result["source"]["dirty"] is False,
                 "active-boundary producer source differs")
        _require(len(result["candidates"])
                 == int(config["source"]["finite_candidate_count_per_state"]),
                 "active-boundary source candidate count differs")
        split = str(state["split"])
        _require(split != "test", "active-boundary test split was accessed")
        rows = [
            {
                **ROW_IDENTITIES[row],
                **audit_state_row(
                    result["candidates"], row,
                    robust_margin_m=robust,
                    minimum_variation_m=variation,
                ),
            }
            for row in range(7)
        ]
        state_records.append({
            "case_id": state["case_id"],
            "split": split,
            "state_id": state["state_id"],
            "source_primary_category": state["primary_category"],
            "row_records": rows,
        })
        artifact_records.append({
            "case_id": state["case_id"],
            "split": split,
            "file_sha256": artifact["file_sha256"],
            "result_payload_sha256": artifact["result_payload_sha256"],
        })

    primary_splits = set(config["source"]["primary_splits"])
    primary = [item for item in state_records if item["split"] in primary_splits]
    diagnostic = [item for item in state_records if item["split"] == "diagnostic"]
    row_summary = []
    for identity in ROW_IDENTITIES:
        row = int(identity["row"])
        records = [item["row_records"][row] for item in primary]
        row_summary.append({
            **identity,
            "audited_state_count": len(records),
            "meaningful_variation_state_count": sum(
                bool(item["meaningful_variation"]) for item in records
            ),
            "robust_crossing_state_count": sum(
                bool(item["robust_crossing_observed"]) for item in records
            ),
            "global_safe_side_state_count": sum(
                bool(item["globally_safe_robust_negative_side_observed"])
                for item in records
            ),
            "useful_boundary_state_count": sum(
                bool(item["useful_boundary_observed"]) for item in records
            ),
            "independent_violation_state_count": sum(
                bool(item["independent_violation_observed"]) for item in records
            ),
            "proxy_invalid_candidate_count": sum(
                int(item["proxy_invalid_candidate_count"]) for item in records
            ),
            "interpretation": row_interpretation(records),
            "no_independent_violation_interpretation": (
                None if any(bool(item["independent_violation_observed"])
                            for item in records)
                else config["finite_search"]["no_independent_violation_wording"]
            ),
        })

    next_rows = [
        item["row"] for item in row_summary
        if item["interpretation"] not in {
            "USEFUL_BOUNDARY_WITNESSED_IN_FINITE_SET",
            "PROXY_INVALID_IN_AUDITED_SET",
        }
    ]
    output = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "source": _git_identity(repo_root, expected_commit),
        "allocation": _allocation_record(),
        "config": config,
        "validated_coverage_binding": {
            "path": str(coverage_validation_path),
            "file_sha256": _file_sha256(coverage_validation_path),
            "result_payload_sha256": coverage["result_payload_sha256"],
        },
        "artifact_records": artifact_records,
        "state_records": state_records,
        "primary_split_state_count": len(primary),
        "diagnostic_state_count": len(diagnostic),
        "reserved_test_episode_count_accessed": 0,
        "row_summary": row_summary,
        "next_active_search_rows": next_rows,
        "interpretation": (
            "finite_set_audit_complete_targeted_active_search_required"
            if next_rows else "finite_set_audit_complete_all_rows_have_useful_boundaries"
        ),
        "MLP_training_authorized": False,
        "new_collection_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["result_payload_sha256"] = _sha256(_canonical(output))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--coverage-validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        repo_root=args.repo_root.resolve(),
        producer_root=args.producer_root.resolve(),
        coverage_validation_path=args.coverage_validation.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "row_summary": result["row_summary"],
        "next_active_search_rows": result["next_active_search_rows"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
