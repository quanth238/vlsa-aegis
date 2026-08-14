#!/usr/bin/env python3
"""Aggregate independently validated bounded active-boundary search jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
    *, repo_root: Path, producer_root: Path, search_config_path: Path,
    expected_producer_commit: str, expected_validator_commit: str,
    expected_auditor_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.active_boundary_search import (
        RESULT_SCHEMA, aggregate_search_rows, load_config, target_job,
    )

    config = load_config(search_config_path)
    source = _git_identity(repo_root, expected_auditor_commit)
    allocation = _allocation_record()
    results = []
    artifacts = []
    for array_index in range(len(config["target_jobs"])):
        expected_job = target_job(config, array_index)
        result_path = producer_root / ("job-%02d" % array_index) / "result.json"
        validation_path = producer_root / ("job-%02d" % array_index) / "validation.json"
        result = _load(result_path)
        validation = _load(validation_path)
        _require(result["schema_version"] == RESULT_SCHEMA
                 and result["status"] == "complete"
                 and result["scientific_result"] is True,
                 "active search aggregate producer incomplete")
        _require(result["source"]["commit"] == expected_producer_commit
                 and result["source"]["dirty"] is False,
                 "active search aggregate producer source differs")
        _require(result["result_payload_sha256"] == _payload_sha256(result),
                 "active search aggregate producer payload differs")
        _require(result["active_boundary_job"] == expected_job,
                 "active search aggregate job differs")
        _require(validation["status"] == "complete"
                 and validation["scientific_result"] is True
                 and validation["validator"]["commit"] == expected_validator_commit
                 and validation["producer"]["commit"] == expected_producer_commit
                 and validation["producer"]["file_sha256"] == _file_sha256(result_path)
                 and validation["producer"]["result_payload_sha256"]
                 == result["result_payload_sha256"]
                 and validation["result_payload_sha256"] == _payload_sha256(validation)
                 and all(validation["gates"].values()),
                 "active search aggregate validation differs")
        results.append(result)
        artifacts.append({
            "array_index": array_index,
            "job": expected_job,
            "result_file_sha256": _file_sha256(result_path),
            "result_payload_sha256": result["result_payload_sha256"],
            "validation_file_sha256": _file_sha256(validation_path),
            "validation_payload_sha256": validation["result_payload_sha256"],
        })
    rows = aggregate_search_rows(results, config["target_rows"])
    useful_rows = [item["row"] for item in rows if item["useful_boundary_observed"]]
    unresolved_rows = [item["row"] for item in rows if not item["useful_boundary_observed"]]
    result = {
        "schema_version": "vlsa_distal_active_boundary_search_audit.v1",
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation,
        "config": config,
        "producer_commit": expected_producer_commit,
        "validator_commit": expected_validator_commit,
        "artifact_records": artifacts,
        "row_summary": rows,
        "useful_boundary_rows": useful_rows,
        "unresolved_rows": unresolved_rows,
        "reserved_test_episode_count_accessed": 0,
        "MLP_training_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": (
            "bounded_active_search_supports_targeted_collection_for_all_rows"
            if not unresolved_rows
            else "bounded_active_search_leaves_rows_without_useful_boundary"
        ),
    }
    result["result_payload_sha256"] = _payload_sha256(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-validator-commit", required=True)
    parser.add_argument("--expected-auditor-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        repo_root=args.repo_root.resolve(),
        producer_root=args.producer_root.resolve(),
        search_config_path=args.search_config.resolve(),
        expected_producer_commit=args.expected_producer_commit,
        expected_validator_commit=args.expected_validator_commit,
        expected_auditor_commit=args.expected_auditor_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
