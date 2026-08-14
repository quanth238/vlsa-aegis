#!/usr/bin/env python3
"""Independently validate task-0 moka query-boundary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import _allocation_record
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


VALIDATION_SCHEMA = "vlsa_distal_l5_moka_noise_query_boundary_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    return _sha256(_canonical(payload))


def validate(
    *, repo_root: Path, producer_root: Path, discovery_root: Path,
    boundary_config: Path, boundary_manifest: Path,
    expected_producer_commit: str, expected_validation_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.moka_noise_boundary import (
        load_cases, load_config, validate_discovery_binding,
    )
    from main.multilink_ellipsoid.query_boundary_coverage import (
        RESULT_SCHEMA, initially_safe, load_config as load_coverage_config,
        nominal_prefix_unsafe, row_coverage,
    )

    config = load_config(boundary_config, repo_root=repo_root)
    cases = load_cases(boundary_manifest, config)
    _, discovery_validation = validate_discovery_binding(
        config=config, cases=cases, discovery_root=discovery_root
    )
    coverage_config = load_coverage_config(
        repo_root / config["method_bindings"]["query_coverage_config"]
    )
    records = []
    retained = []
    for case_index, expected_case in enumerate(cases):
        path = producer_root / ("case-%02d" % case_index) / "result.json"
        result = _load(path)
        _require(result.get("schema_version") == RESULT_SCHEMA,
                 "moka boundary result schema differs")
        _require(result.get("status") == "complete", "moka boundary result incomplete")
        _require(result.get("case_index") == case_index,
                 "moka boundary result index differs")
        _require(result.get("case") == expected_case,
                 "moka boundary result case differs")
        _require(result.get("result_payload_sha256") == _payload_sha256(result),
                 "moka boundary result payload differs")
        _require(
            result.get("source", {}).get("commit") == expected_producer_commit
            and result.get("source", {}).get("dirty") is False,
            "moka boundary producer source differs",
        )
        binding = result.get("targeted_extension_binding", {})
        _require(
            binding.get("boundary_config_file_sha256")
            == config["config_file_sha256"]
            and binding.get("trajectory_group_id")
            == expected_case["trajectory_group_id"],
            "moka boundary binding differs",
        )
        recomputed = []
        for item in result["boundary_records"]:
            current = item["current"]
            prefix = item["nominal_prefix"]
            current_ok = initially_safe(
                current["row_clearance_m"],
                protected_contact_count=current["protected_contact_count"],
                active_obstacle_l1_displacement_m=(
                    current["active_obstacle_l1_displacement_m"]
                ),
                config=coverage_config,
            )
            prefix_unsafe = nominal_prefix_unsafe(
                prefix["row_minimum_clearance_m"],
                protected_contact_count=prefix["protected_contact_count"],
                maximum_active_obstacle_l1_displacement_m=(
                    prefix["maximum_active_obstacle_l1_displacement_m"]
                ),
                config=coverage_config,
            )
            _require(current["strictly_safe"] == current_ok,
                     "moka boundary current status differs")
            _require(item["nominal_prefix_unsafe"] == prefix_unsafe,
                     "moka boundary prefix status differs")
            _require(item["retained"] == bool(current_ok and prefix_unsafe),
                     "moka boundary retention differs")
            if item["retained"]:
                recomputed.append(item)
        _require(result["retained_states"] == recomputed,
                 "moka boundary retained states differ")
        _require(result["row_coverage"] == row_coverage(recomputed, coverage_config),
                 "moka boundary row coverage differs")
        retained.extend({
            **item, "case_id": expected_case["case_id"],
            "split": expected_case["split"],
            "trajectory_group_id": expected_case["trajectory_group_id"],
        } for item in recomputed)
        records.append({
            "case_id": expected_case["case_id"],
            "split": expected_case["split"],
            "trajectory_group_id": expected_case["trajectory_group_id"],
            "source_result_file_sha256": _file_sha256(path),
            "source_result_payload_sha256": result["result_payload_sha256"],
            "query_boundary_count": result["query_boundary_count"],
            "retained_state_count": result["retained_state_count"],
        })
    aggregate_rows = row_coverage(retained, coverage_config)
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "source": _git_identity(repo_root, expected_validation_commit),
        "allocation": _allocation_record(),
        "producer": {
            "root": str(producer_root),
            "expected_commit": expected_producer_commit,
        },
        "discovery_validation_payload_sha256": discovery_validation["payload_sha256"],
        "case_count": len(cases),
        "retained_state_count": len(retained),
        "records": records,
        "row_coverage": aggregate_rows,
        "per_split": {
            split: {
                "case_count": sum(item["split"] == split for item in records),
                "retained_state_count": sum(
                    item["retained_state_count"] for item in records
                    if item["split"] == split
                ),
            }
            for split in ("train", "validation", "diagnostic")
        },
        "adaptive_collection_authorized": bool(retained),
        "MLP_training_authorized": False,
        "calibration_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": (
            "validated_warning_states_ready_for_frozen_adaptive_boundary_collection"
            if retained else
            "no_initially_safe_warning_state_found_do_not_collect_or_train"
        ),
    }
    output["result_payload_sha256"] = _sha256(_canonical(output))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--boundary-config", type=Path, required=True)
    parser.add_argument("--boundary-manifest", type=Path, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-validation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(), producer_root=args.producer_root.resolve(),
        discovery_root=args.discovery_root.resolve(),
        boundary_config=args.boundary_config.resolve(),
        boundary_manifest=args.boundary_manifest.resolve(),
        expected_producer_commit=args.expected_producer_commit,
        expected_validation_commit=args.expected_validation_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "retained_state_count": result["retained_state_count"],
        "row_coverage": result["row_coverage"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
