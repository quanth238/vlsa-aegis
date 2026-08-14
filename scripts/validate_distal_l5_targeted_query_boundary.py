#!/usr/bin/env python3
"""Independently validate one targeted L5 query-boundary artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def validate(
    *, repo_root: Path, result_path: Path, producer_commit: str,
    validator_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.query_boundary_coverage import (
        RESULT_SCHEMA, initially_safe, load_config as load_coverage_config,
        nominal_prefix_unsafe, row_coverage,
    )
    from main.multilink_ellipsoid.targeted_l5_boundary_extension import (
        load_cases as load_target_cases,
        load_config as load_target_config,
    )

    result = _load(result_path)
    _require(result["schema_version"] == RESULT_SCHEMA,
             "targeted coverage result schema differs")
    _require(result["status"] == "complete", "targeted coverage incomplete")
    _require(result["scientific_result"] is True,
             "targeted coverage is not scientific")
    _require(result["source"]["commit"] == producer_commit,
             "targeted coverage producer differs")
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(_sha256(canonical(payload)) == claimed,
             "targeted coverage payload hash differs")
    target_config = load_target_config(
        repo_root / "configs/vlsa_distal_l5_targeted_extension.v1.json",
        repo_root=repo_root,
    )
    target_cases = load_target_cases(
        repo_root / "manifests/vlsa_distal_l5_targeted_extension.v1.jsonl",
        target_config,
    )
    binding = result["targeted_extension_binding"]
    index = int(binding["target_case_index"])
    _require(
        binding["config"] == target_config
        and binding["target_case"] == target_cases[index]
        and result["case"] == target_cases[index]
        and binding["parent_configs_used_for_method_only"] is True,
        "targeted coverage binding differs",
    )
    coverage_config = load_coverage_config(
        repo_root / "configs/vlsa_distal_query_boundary_coverage.v1.json"
    )
    _require(result["config"] == coverage_config,
             "targeted coverage method config differs")
    _require(result["retained_state_count"] == 1,
             "targeted extension must retain exactly one state per episode")
    _require(len(result["retained_states"]) == 1,
             "targeted retained state list differs")
    for record in result["boundary_records"]:
        current = record["current"]
        nominal = record["nominal_prefix"]
        current_ok = initially_safe(
            current["row_clearance_m"],
            protected_contact_count=current["protected_contact_count"],
            active_obstacle_l1_displacement_m=current[
                "active_obstacle_l1_displacement_m"
            ],
            config=coverage_config,
        )
        prefix_unsafe = nominal_prefix_unsafe(
            nominal["row_minimum_clearance_m"],
            protected_contact_count=nominal["protected_contact_count"],
            maximum_active_obstacle_l1_displacement_m=nominal[
                "maximum_active_obstacle_l1_displacement_m"
            ],
            config=coverage_config,
        )
        _require(current_ok == current["strictly_safe"],
                 "targeted current safety differs")
        _require(prefix_unsafe == record["nominal_prefix_unsafe"],
                 "targeted prefix safety differs")
        _require(record["retained"] == (current_ok and prefix_unsafe),
                 "targeted retention differs")
    expected_retained = [
        record for record in result["boundary_records"] if record["retained"]
    ]
    _require(expected_retained == result["retained_states"],
             "targeted retained records differ")
    _require(row_coverage(expected_retained, coverage_config)
             == result["row_coverage"],
             "targeted row coverage differs")
    _require(result["candidate_execution_performed"] is False,
             "targeted coverage executed candidates")
    output = {
        "schema_version": "vlsa_distal_l5_targeted_query_boundary_validation.v1",
        "status": "complete",
        "scientific_result": True,
        "producer": {
            "path": str(result_path),
            "file_sha256": _file_sha256(result_path),
            "result_payload_sha256": claimed,
            "commit": producer_commit,
        },
        "validator_source": _git_identity(repo_root, validator_commit),
        "case_id": result["case"]["case_id"],
        "split": result["case"]["split"],
        "state_id": result["retained_states"][0]["state_id"],
        "checks": {
            "payload_self_hash": True,
            "target_manifest_binding": True,
            "static_case_binding": True,
            "initial_state_safe": True,
            "nominal_prefix_unsafe": True,
            "one_state_per_episode": True,
            "candidate_execution_absent": True,
        },
        "next_gate": "run_immutable_adaptive_v3_boundary_collection",
        "training_authorized": False,
    }
    output["validation_payload_sha256"] = _sha256(canonical(output))
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(), result_path=args.result.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case_id"],
        "state_id": result["state_id"],
        "validation_payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
