#!/usr/bin/env python3
"""Validate and aggregate grouped query-boundary coverage artifacts."""

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


def validate(
    *, repo_root: Path, producer_root: Path, selection_manifest: Path,
    experiment_config: Path, coverage_config: Path,
    expected_producer_commit: str, expected_validation_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.clean_action_risk import load_cases, load_config
    from main.multilink_ellipsoid.query_boundary_coverage import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, initially_safe,
        load_config as load_coverage_config, nominal_prefix_unsafe, row_coverage,
    )

    selection_cfg = load_config(experiment_config)
    config = load_coverage_config(coverage_config)
    cases = [
        item for item in load_cases(selection_manifest, selection_cfg)
        if item["split"] in config["source"]["included_splits"]
    ]
    _require(len(cases) == 15, "query-boundary validation case count differs")
    records = []
    retained = []
    for case_index, expected_case in enumerate(cases):
        path = producer_root / ("case-%02d" % case_index) / "result.json"
        result = _load(path)
        _require(result.get("schema_version") == RESULT_SCHEMA,
                 "query-boundary result schema differs")
        _require(result.get("status") == "complete", "query-boundary result incomplete")
        _require(result.get("case_index") == case_index, "query-boundary result index differs")
        _require(result.get("case") == expected_case, "query-boundary result case differs")
        _require(result.get("result_payload_sha256") == _payload_sha256(result),
                 "query-boundary result payload differs")
        _require(
            result.get("source", {}).get("commit") == expected_producer_commit
            and result.get("source", {}).get("dirty") is False,
            "query-boundary producer source differs",
        )
        _require(result.get("config") == config, "query-boundary config differs")
        _require(
            result.get("candidate_execution_performed") is False
            and result.get("MLP_training_authorized") is False
            and result.get("calibration_authorized") is False
            and result.get("QP_authorized") is False
            and result.get("closed_loop_authorized") is False,
            "query-boundary authorization differs",
        )
        recomputed_retained = []
        for item in result["boundary_records"]:
            current = item["current"]
            prefix = item["nominal_prefix"]
            current_ok = initially_safe(
                current["row_clearance_m"],
                protected_contact_count=current["protected_contact_count"],
                active_obstacle_l1_displacement_m=(
                    current["active_obstacle_l1_displacement_m"]
                ),
                config=config,
            )
            prefix_unsafe = nominal_prefix_unsafe(
                prefix["row_minimum_clearance_m"],
                protected_contact_count=prefix["protected_contact_count"],
                maximum_active_obstacle_l1_displacement_m=(
                    prefix["maximum_active_obstacle_l1_displacement_m"]
                ),
                config=config,
            )
            _require(current["strictly_safe"] == current_ok,
                     "query-boundary current eligibility differs")
            _require(item["nominal_prefix_unsafe"] == prefix_unsafe,
                     "query-boundary prefix status differs")
            _require(item["retained"] == bool(current_ok and prefix_unsafe),
                     "query-boundary retention differs")
            if item["retained"]:
                recomputed_retained.append(item)
        _require(result["retained_states"] == recomputed_retained,
                 "query-boundary retained states differ")
        _require(result["retained_state_count"] == len(recomputed_retained),
                 "query-boundary retained count differs")
        _require(result["row_coverage"] == row_coverage(recomputed_retained, config),
                 "query-boundary row coverage differs")
        retained.extend({**item, "case_id": expected_case["case_id"],
                         "split": expected_case["split"]}
                        for item in recomputed_retained)
        records.append({
            "case_id": expected_case["case_id"],
            "split": expected_case["split"],
            "source_result_file_sha256": _file_sha256(path),
            "source_result_payload_sha256": result["result_payload_sha256"],
            "query_boundary_count": result["query_boundary_count"],
            "retained_state_count": result["retained_state_count"],
        })
    aggregate_rows = row_coverage(retained, config)
    active_rows = [
        item["row"] for item in aggregate_rows
        if item["nominal_active_witness_count"] > 0
    ]
    violated_rows = [
        item["row"] for item in aggregate_rows
        if item["nominal_violated_row_count"] > 0
    ]
    split_summary = {}
    for split in config["source"]["included_splits"]:
        chosen = [item for item in records if item["split"] == split]
        split_summary[split] = {
            "episode_count": len(chosen),
            "retained_state_count": sum(item["retained_state_count"] for item in chosen),
        }
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "source": _git_identity(repo_root, expected_validation_commit),
        "allocation": _allocation_record(),
        "producer": {
            "root": str(producer_root),
            "expected_commit": expected_producer_commit,
        },
        "episode_count": len(records),
        "retained_state_count": len(retained),
        "split_summary": split_summary,
        "records": records,
        "row_coverage": aggregate_rows,
        "active_witness_rows": active_rows,
        "violated_rows": violated_rows,
        "unsupported_active_witness_rows": [
            row for row in range(7) if row not in active_rows
        ],
        "gates": {
            "all_requested_artifacts_complete": True,
            "at_least_one_retained_state": bool(retained),
            "all_seven_rows_have_active_witness_support": len(active_rows) == 7,
            "all_seven_rows_have_violation_support": len(violated_rows) == 7,
            "candidate_safe_unsafe_support_evaluated": False,
        },
        "interpretation": (
            "query_boundary_coverage_pass_ready_for_frozen_candidate_collection"
            if retained and len(active_rows) == 7 and len(violated_rows) == 7
            else "query_boundary_coverage_incomplete_do_not_train"
        ),
        "MLP_training_authorized": False,
        "calibration_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["result_payload_sha256"] = _sha256(_canonical(output))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--coverage-config", type=Path, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-validation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(), producer_root=args.producer_root.resolve(),
        selection_manifest=args.selection_manifest.resolve(),
        experiment_config=args.experiment_config.resolve(),
        coverage_config=args.coverage_config.resolve(),
        expected_producer_commit=args.expected_producer_commit,
        expected_validation_commit=args.expected_validation_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "retained_state_count": result["retained_state_count"],
        "row_coverage": result["row_coverage"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
