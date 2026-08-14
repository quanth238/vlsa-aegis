#!/usr/bin/env python3
"""Run unchanged adaptive-v3 labels for one factorized-boundary coverage result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.evaluate_distal_query_action_risk_e05 import evaluate
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def main(argv: Sequence[str] | None = None) -> int:
    from main.multilink_ellipsoid.adaptive_query_action_risk_v2 import (
        RESULT_SCHEMA_V3, coarse_candidate_definitions, load_config_v3,
    )
    from main.multilink_ellipsoid.grouped_query_action_risk import (
        load_config as load_grouped_config,
    )
    from main.multilink_ellipsoid.l5_factorized_boundary import (
        load_config as load_factorized_config, load_manifest,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coverage-result", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    factorized = load_factorized_config(args.config.resolve(), root)
    rows = load_manifest(args.manifest.resolve(), factorized)
    row = rows[args.case_index]
    bindings = factorized["method_bindings"]
    grouped = load_grouped_config(root / bindings["grouped_config"])
    adaptive = load_config_v3(root / bindings["adaptive_config"])
    _require(
        _file_sha256(root / bindings["base_config"])
        == grouped["method"]["base_config_file_sha256"],
        "factorized risk base method differs",
    )
    coverage = _load(args.coverage_result.resolve())
    _require(coverage["retained_state_count"] == 1,
             "factorized risk requires one retained state")
    retained = coverage["retained_states"][0]
    case = coverage["case"]
    _require(
        case["case_id"] == row["case_id"]
        and case["trajectory_group_id"] == row["factorized_trajectory_group_id"]
        and case["split"] == row["factorized_split"],
        "factorized risk case differs",
    )
    archived = args.table1_root.resolve() / case["archived_result_relative_path"]
    binding = {
        "adaptive_config": adaptive, "grouped_config": grouped,
        "coverage_result_path": str(args.coverage_result.resolve()),
        "coverage_result_file_sha256": _file_sha256(args.coverage_result.resolve()),
        "coverage_result_payload_sha256": coverage["result_payload_sha256"],
        "case_index": args.case_index, "case": case, "retained_state": retained,
        "factorized_boundary": {
            "config_file_sha256": factorized["config_file_sha256"],
            "config_payload_sha256": factorized["config_payload_sha256"],
            "manifest_file_sha256": factorized["population"]["manifest_file_sha256"],
            "trajectory_group_id": row["factorized_trajectory_group_id"],
            "split": row["factorized_split"],
        },
    }

    def definitions(nominal, frame, base):
        return coarse_candidate_definitions(nominal, frame, base, adaptive)

    result = evaluate(
        repo_root=root,
        population_manifest_path=root / factorized["population"]["table1_manifest"],
        archived_path=archived,
        geometry_config_path=root / bindings["geometry_config"],
        experiment_config_path=root / bindings["base_config"],
        expected_commit=args.expected_commit, output_path=args.output.resolve(),
        case_id_override=case["case_id"],
        policy_noise_seed_override=case["policy_noise_seed"],
        state_step_override=int(retained["step"]),
        query_index_override=int(retained["query_index"]),
        result_schema_override=RESULT_SCHEMA_V3,
        claim_scope_override=factorized["claim_scope"],
        population_binding=binding,
        candidate_definitions_override=definitions,
        candidate_protocol_binding={
            "adaptive_config_file_sha256": adaptive["config_file_sha256"],
            "adaptive_config_payload_sha256": adaptive["config_payload_sha256"],
            "factorized_config_file_sha256": factorized["config_file_sha256"],
        },
        apply_released_aegis_ee_to_all_proposed_actions=True,
        adaptive_boundary_config=adaptive,
    )
    result["training_authorized"] = False
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": case["case_id"], "state_id": retained["state_id"],
        "target_row": result["adaptive_boundary_sampling"]["target_row"],
        "candidate_count": result["candidate_count"],
        "summary": result["summary"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
