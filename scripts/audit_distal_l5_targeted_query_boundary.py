#!/usr/bin/env python3
"""Run the frozen query-boundary audit on one preregistered extension case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from main.multilink_ellipsoid.targeted_l5_boundary_extension import (
    load_cases, load_config,
)
from scripts.audit_distal_query_boundary_coverage import audit
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--coverage-config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    target_config = load_config(args.target_config.resolve(), repo_root=root)
    cases = load_cases(args.target_manifest.resolve(), target_config)
    if not 0 <= int(args.case_index) < len(cases):
        raise ValueError("targeted query-boundary case index differs")
    selected = cases[int(args.case_index)]
    binding = {
        "config": target_config,
        "target_manifest": str(args.target_manifest.resolve()),
        "target_manifest_file_sha256": target_config["target_population"][
            "manifest_file_sha256"
        ],
        "target_case_index": int(args.case_index),
        "target_case": selected,
        "parent_configs_used_for_method_only": True,
    }
    result = audit(
        repo_root=root,
        population_manifest=args.population_manifest.resolve(),
        selection_manifest=(root / "manifests/vlsa_distal_clean_action_risk.v1.jsonl"),
        experiment_config=args.experiment_config.resolve(),
        coverage_config=args.coverage_config.resolve(),
        geometry_config=args.geometry_config.resolve(),
        table1_root=args.table1_root.resolve(),
        case_index=int(args.case_index),
        expected_commit=args.expected_commit,
        selected_case_override=selected,
        targeted_extension_binding=binding,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": selected["case_id"],
        "split": selected["split"],
        "retained_state_count": result["retained_state_count"],
        "active_witness_counts": [
            item["nominal_active_witness_count"]
            for item in result["row_coverage"]
        ],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
