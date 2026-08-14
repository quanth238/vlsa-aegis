#!/usr/bin/env python3
"""Classify a fresh trajectory and collect its real query boundary when eligible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write


def main(argv: Sequence[str] | None = None) -> int:
    from main.multilink_ellipsoid.l5_factorized_boundary import (
        classify_evaluation, load_config, load_manifest, selected_case,
    )
    from scripts.audit_distal_query_boundary_coverage import audit

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--classification-output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config = load_config(args.config.resolve(), repo_root)
    rows = load_manifest(args.manifest.resolve(), config)
    if not 0 <= args.case_index < len(rows):
        raise ValueError("factorized boundary case index differs")
    row = rows[args.case_index]
    result_relative = Path("results") / "aegis" / row["case_id"] / "result.json"
    result_path = args.case_root.resolve() / result_relative
    classification = classify_evaluation(
        row=row, result_path=result_path,
        results_root=args.case_root.resolve() / "results",
    )
    _atomic_write(args.classification_output.resolve(), classification)
    if not classification["eligible"]:
        print(json.dumps({
            "case_id": row["case_id"], "classification": classification["classification"],
            "eligible": False,
        }, sort_keys=True), flush=True)
        return 0
    case = selected_case(
        row=row, classification=classification,
        archived_relative_path=str(result_relative),
    )
    bindings = config["method_bindings"]
    coverage = audit(
        repo_root=repo_root,
        population_manifest=repo_root / config["population"]["table1_manifest"],
        selection_manifest=args.manifest.resolve(),
        experiment_config=repo_root / bindings["clean_config"],
        coverage_config=repo_root / bindings["query_coverage_config"],
        geometry_config=repo_root / bindings["geometry_config"],
        table1_root=args.case_root.resolve(), case_index=args.case_index,
        expected_commit=args.expected_commit, selected_case_override=case,
        targeted_extension_binding={
            "schema_version": config["schema_version"],
            "protocol_id": config["protocol_id"],
            "config_file_sha256": config["config_file_sha256"],
            "config_payload_sha256": config["config_payload_sha256"],
            "manifest_file_sha256": config["population"]["manifest_file_sha256"],
            "case_index": args.case_index,
            "trajectory_group_id": row["factorized_trajectory_group_id"],
            "split": row["factorized_split"],
        },
    )
    _atomic_write(args.coverage_output.resolve(), coverage)
    print(json.dumps({
        "case_id": row["case_id"], "classification": classification["classification"],
        "eligible": True, "retained_state_count": coverage["retained_state_count"],
        "coverage_payload_sha256": coverage["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
