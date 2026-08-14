#!/usr/bin/env python3
"""Audit real query boundaries on validated task-0 moka discovery trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.audit_distal_query_boundary_coverage import audit
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write


def main(argv: Sequence[str] | None = None) -> int:
    from main.multilink_ellipsoid.moka_noise_boundary import (
        load_cases, load_config, validate_discovery_binding,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--boundary-config", type=Path, required=True)
    parser.add_argument("--boundary-manifest", type=Path, required=True)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.boundary_config.resolve(), repo_root=args.repo_root.resolve())
    cases = load_cases(args.boundary_manifest.resolve(), config)
    if not 0 <= int(args.case_index) < len(cases):
        raise ValueError("moka boundary case index differs")
    validate_discovery_binding(
        config=config, cases=cases, discovery_root=args.discovery_root.resolve()
    )
    selected = cases[int(args.case_index)]
    bindings = config["method_bindings"]
    result = audit(
        repo_root=args.repo_root.resolve(),
        population_manifest=args.population_manifest.resolve(),
        selection_manifest=args.boundary_manifest.resolve(),
        experiment_config=args.repo_root.resolve() / "configs/vlsa_distal_clean_action_risk.v1.json",
        coverage_config=args.repo_root.resolve() / bindings["query_coverage_config"],
        geometry_config=args.repo_root.resolve() / bindings["geometry_config"],
        table1_root=args.discovery_root.resolve(),
        case_index=int(args.case_index), expected_commit=args.expected_commit,
        selected_case_override=selected,
        targeted_extension_binding={
            "schema_version": config["schema_version"],
            "protocol_id": config["protocol_id"],
            "boundary_config_file_sha256": config["config_file_sha256"],
            "boundary_config_payload_sha256": config["config_payload_sha256"],
            "boundary_manifest": str(args.boundary_manifest.resolve()),
            "case_index": int(args.case_index),
            "trajectory_group_id": selected["trajectory_group_id"],
            "diagnostic_only": selected["split"] == "diagnostic",
        },
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": selected["case_id"],
        "split": selected["split"],
        "retained_state_count": result["retained_state_count"],
        "active_witness_counts": [
            row["nominal_active_witness_count"] for row in result["row_coverage"]
        ],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
