#!/usr/bin/env python3
"""Run the frozen E05 query-risk method at one grouped retained state."""

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
    from main.multilink_ellipsoid.grouped_query_action_risk import (
        RESULT_SCHEMA, load_config,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--grouped-config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument(
        "--released-aegis-consistency",
        action="store_true",
        help="Transfer each L5 residual to the raw action, then apply original AEGIS once.",
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.grouped_config.resolve())
    _require(_file_sha256(args.base_config.resolve())
             == config["method"]["base_config_file_sha256"],
             "grouped query-risk base method file differs")
    coverage_path = args.coverage_root.resolve() / (
        "case-%02d" % int(args.case_index)
    ) / "result.json"
    coverage = _load(coverage_path)
    _require(coverage["retained_state_count"] == 1,
             "grouped query-risk requires one retained state")
    retained = coverage["retained_states"][0]
    case = coverage["case"]
    archived = args.table1_root.resolve() / case["archived_result_relative_path"]
    binding = {
        "grouped_config": config,
        "coverage_result_path": str(coverage_path),
        "coverage_result_file_sha256": _file_sha256(coverage_path),
        "coverage_result_payload_sha256": coverage["result_payload_sha256"],
        "case_index": int(args.case_index),
        "case": case,
        "retained_state": retained,
    }
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        population_manifest_path=args.population_manifest.resolve(),
        archived_path=archived,
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.base_config.resolve(),
        expected_commit=args.expected_commit, output_path=args.output.resolve(),
        case_id_override=case["case_id"],
        state_step_override=int(retained["step"]),
        query_index_override=int(retained["query_index"]),
        result_schema_override=RESULT_SCHEMA,
        claim_scope_override=config["claim_scope"],
        population_binding=binding,
        apply_released_aegis_ee_to_all_proposed_actions=bool(
            args.released_aegis_consistency
        ),
    )
    result["training_authorized"] = False
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": case["case_id"],
        "state_id": retained["state_id"],
        "interpretation": result["interpretation"],
        "summary": result["summary"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
