#!/usr/bin/env python3
"""Collect one per-row adaptive post-AEGIS L5 boundary state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import evaluate
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from main.multilink_ellipsoid.adaptive_query_action_risk_v2 import (
        CONFIG_SCHEMA_V3, RESULT_SCHEMA, RESULT_SCHEMA_V3,
        coarse_candidate_definitions, load_config, load_config_v3,
    )
    from main.multilink_ellipsoid.grouped_query_action_risk import (
        load_config as load_grouped_config,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--grouped-config", type=Path, required=True)
    parser.add_argument("--adaptive-config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    grouped = load_grouped_config(args.grouped_config.resolve())
    adaptive_raw = json.loads(args.adaptive_config.resolve().read_text())
    is_v3 = adaptive_raw.get("schema_version") == CONFIG_SCHEMA_V3
    adaptive = (
        load_config_v3(args.adaptive_config.resolve())
        if is_v3 else load_config(args.adaptive_config.resolve())
    )
    _require(
        _file_sha256(args.base_config.resolve())
        == grouped["method"]["base_config_file_sha256"],
        "adaptive v2 base method file differs",
    )
    coverage_path = args.coverage_root.resolve() / (
        "case-%02d" % int(args.case_index)
    ) / "result.json"
    coverage = _load(coverage_path)
    _require(coverage["retained_state_count"] == 1,
             "adaptive v2 collector requires one retained state")
    retained = coverage["retained_states"][0]
    case = coverage["case"]
    archived = args.table1_root.resolve() / case["archived_result_relative_path"]
    binding = {
        "adaptive_config": adaptive,
        "grouped_config": grouped,
        "coverage_result_path": str(coverage_path),
        "coverage_result_file_sha256": _file_sha256(coverage_path),
        "coverage_result_payload_sha256": coverage["result_payload_sha256"],
        "case_index": int(args.case_index),
        "case": case,
        "retained_state": retained,
    }

    def definitions(nominal, frame, base):
        return coarse_candidate_definitions(nominal, frame, base, adaptive)

    result = evaluate(
        repo_root=args.repo_root.resolve(),
        population_manifest_path=args.population_manifest.resolve(),
        archived_path=archived,
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.base_config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
        case_id_override=case["case_id"],
        state_step_override=int(retained["step"]),
        query_index_override=int(retained["query_index"]),
        result_schema_override=RESULT_SCHEMA_V3 if is_v3 else RESULT_SCHEMA,
        claim_scope_override=adaptive["claim_scope"],
        population_binding=binding,
        candidate_definitions_override=definitions,
        candidate_protocol_binding={
            "adaptive_config_file_sha256": adaptive["config_file_sha256"],
            "adaptive_config_payload_sha256": adaptive["config_payload_sha256"],
        },
        apply_released_aegis_ee_to_all_proposed_actions=True,
        adaptive_boundary_config=adaptive,
    )
    result["training_authorized"] = False
    _atomic_write(args.output.resolve(), result)
    sampling = result["adaptive_boundary_sampling"]
    print(json.dumps({
        "case_id": case["case_id"],
        "state_id": retained["state_id"],
        "target_row": sampling["target_row"],
        "bracket_found": sampling["bracket_found"],
        "coarse_candidates": len(sampling["coarse_screening_records"]),
        "authoritative_candidates": result["candidate_count"],
        "summary": result["summary"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
