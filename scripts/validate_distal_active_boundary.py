#!/usr/bin/env python3
"""Independently recompute the finite-set active-boundary audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Set

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


def _row_summary(
    state_records: Sequence[Mapping[str, Any]], identities: Sequence[Mapping[str, Any]],
    primary_splits: Set[str], no_independent_wording: str,
) -> list[dict[str, Any]]:
    from main.multilink_ellipsoid.active_boundary_audit import row_interpretation

    primary = [item for item in state_records if item["split"] in primary_splits]
    output = []
    for identity in identities:
        row = int(identity["row"])
        records = [item["row_records"][row] for item in primary]
        independent = any(
            bool(item["independent_violation_observed"]) for item in records
        )
        output.append({
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
                None if independent else no_independent_wording
            ),
        })
    return output


def validate(
    *, repo_root: Path, result_path: Path, producer_root: Path,
    coverage_validation_path: Path, config_path: Path,
    expected_producer_commit: str, expected_validation_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.active_boundary_audit import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, audit_state_row, load_config,
    )
    from main.multilink_ellipsoid.query_boundary_coverage import ROW_IDENTITIES

    config = load_config(config_path)
    result = _load(result_path)
    coverage = _load(coverage_validation_path)
    _require(result["schema_version"] == RESULT_SCHEMA
             and result["status"] == "complete"
             and result["scientific_result"] is True,
             "active-boundary result incomplete")
    _require(result["source"]["commit"] == expected_producer_commit
             and result["source"]["dirty"] is False,
             "active-boundary producer source differs")
    _require(result["result_payload_sha256"] == _payload_sha256(result),
             "active-boundary result payload differs")
    _require(result["config"] == config,
             "active-boundary result config differs")
    _require(_file_sha256(coverage_validation_path)
             == config["source"]["validated_coverage_file_sha256"],
             "active-boundary coverage file differs")
    _require(coverage["result_payload_sha256"]
             == config["source"]["validated_coverage_payload_sha256"],
             "active-boundary coverage payload differs")
    _require(result["reserved_test_episode_count_accessed"] == 0,
             "active-boundary reserved tests were accessed")

    robust = float(config["finite_search"]["robust_boundary_margin_m"])
    variation = float(config["finite_search"]["minimum_meaningful_variation_m"])
    recomputed_states = []
    artifact_hashes_match = True
    for artifact, state, observed in zip(
        coverage["artifact_records"], coverage["state_classifications"],
        result["state_records"],
    ):
        source_path = Path(artifact["path"])
        artifact_hashes_match = bool(
            artifact_hashes_match
            and _file_sha256(source_path) == artifact["file_sha256"]
        )
        source = _load(source_path)
        _require(source["result_payload_sha256"] == artifact["result_payload_sha256"]
                 == _payload_sha256(source),
                 "active-boundary source payload differs")
        rows = [
            {
                **ROW_IDENTITIES[row],
                **audit_state_row(
                    source["candidates"], row,
                    robust_margin_m=robust,
                    minimum_variation_m=variation,
                ),
            }
            for row in range(7)
        ]
        recomputed = {
            "case_id": state["case_id"],
            "split": state["split"],
            "state_id": state["state_id"],
            "source_primary_category": state["primary_category"],
            "row_records": rows,
        }
        _require(recomputed == observed,
                 "active-boundary state audit differs")
        recomputed_states.append(recomputed)
    _require(len(recomputed_states) == 15,
             "active-boundary state count differs")
    summary = _row_summary(
        recomputed_states, ROW_IDENTITIES,
        set(config["source"]["primary_splits"]),
        config["finite_search"]["no_independent_violation_wording"],
    )
    _require(summary == result["row_summary"],
             "active-boundary row summary differs")
    next_rows = [
        item["row"] for item in summary
        if item["interpretation"] not in {
            "USEFUL_BOUNDARY_WITNESSED_IN_FINITE_SET",
            "PROXY_INVALID_IN_AUDITED_SET",
        }
    ]
    _require(next_rows == result["next_active_search_rows"],
             "active-boundary next rows differ")
    gates = {
        "producer_result_hash_and_payload_valid": True,
        "validated_coverage_binding_valid": True,
        "all_15_source_artifact_hashes_valid": artifact_hashes_match,
        "all_state_row_metrics_reproduced_exactly": True,
        "row_summary_reproduced_exactly": True,
        "reserved_test_episodes_untouched": True,
        "no_learning_or_control_executed": (
            result["MLP_training_authorized"] is False
            and result["new_collection_authorized"] is False
            and result["QP_authorized"] is False
            and result["closed_loop_authorized"] is False
        ),
    }
    _require(all(gates.values()), "active-boundary validation failed")
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "source": _git_identity(repo_root, expected_validation_commit),
        "allocation": _allocation_record(),
        "producer_result": {
            "path": str(result_path),
            "file_sha256": _file_sha256(result_path),
            "result_payload_sha256": result["result_payload_sha256"],
            "commit": expected_producer_commit,
        },
        "gates": gates,
        "row_summary": summary,
        "next_active_search_rows": next_rows,
        "interpretation": result["interpretation"],
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
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--coverage-validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-validation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(), result_path=args.result.resolve(),
        producer_root=args.producer_root.resolve(),
        coverage_validation_path=args.coverage_validation.resolve(),
        config_path=args.config.resolve(),
        expected_producer_commit=args.expected_producer_commit,
        expected_validation_commit=args.expected_validation_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "gates": result["gates"],
        "next_active_search_rows": result["next_active_search_rows"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
