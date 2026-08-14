#!/usr/bin/env python3
"""Run one bounded active-boundary search job with the fixed complete backup."""

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
    from main.multilink_ellipsoid.active_boundary_search import (
        RESULT_SCHEMA, candidate_definitions, canonical, load_config,
        summarize_target_rows, target_job,
    )
    from scripts.replay_distal_three_ellipsoid_multicbf import _sha256

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.search_config.resolve())
    job = target_job(config, args.array_index)
    finite_audit_path = Path(config["source"]["finite_audit_result"])
    _require(_file_sha256(finite_audit_path)
             == config["source"]["finite_audit_file_sha256"],
             "active boundary finite audit file differs")
    finite_audit = _load(finite_audit_path)
    _require(finite_audit["result_payload_sha256"]
             == config["source"]["finite_audit_payload_sha256"],
             "active boundary finite audit payload differs")
    _require(finite_audit["next_active_search_rows"] == config["target_rows"],
             "active boundary target rows differ")

    coverage_path = args.coverage_root.resolve() / (
        "case-%02d" % int(job["case_index"])
    ) / "result.json"
    coverage = _load(coverage_path)
    retained = coverage["retained_states"][0]
    _require(coverage["case"]["case_id"] == job["case_id"],
             "active boundary case binding differs")
    _require(retained["state_id"] == job["state_id"],
             "active boundary state binding differs")
    archived = args.table1_root.resolve() / coverage["case"][
        "archived_result_relative_path"
    ]

    def definitions(nominal, frame, _base_config):
        return candidate_definitions(
            nominal, frame, config, job["temporal_profile"]
        )

    binding = {
        "search_config": config,
        "job": job,
        "coverage_result_path": str(coverage_path),
        "coverage_result_file_sha256": _file_sha256(coverage_path),
        "coverage_result_payload_sha256": coverage["result_payload_sha256"],
        "finite_audit_result_path": str(finite_audit_path),
        "finite_audit_result_file_sha256": _file_sha256(finite_audit_path),
        "finite_audit_result_payload_sha256": finite_audit["result_payload_sha256"],
    }
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        population_manifest_path=args.population_manifest.resolve(),
        archived_path=archived,
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.base_config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
        case_id_override=job["case_id"],
        state_step_override=int(retained["step"]),
        query_index_override=int(retained["query_index"]),
        result_schema_override=RESULT_SCHEMA,
        claim_scope_override=config["claim_scope"],
        population_binding=binding,
        candidate_definitions_override=definitions,
        candidate_protocol_binding={
            "config_file_sha256": config["config_file_sha256"],
            "config_payload_sha256": config["config_payload_sha256"],
            "finite_search": config["finite_search"],
            "job": job,
        },
    )
    result["active_boundary_search_config"] = config
    result["active_boundary_job"] = job
    result["target_row_summary"] = summarize_target_rows(
        result["candidates"], config["target_rows"],
        config["finite_search"]["robust_boundary_margin_m"],
    )
    source_states = [
        item for item in finite_audit["state_records"]
        if item["state_id"] == job["state_id"] and item["split"] == job["split"]
    ]
    _require(len(source_states) == 1, "active boundary source state differs")
    for row_summary in result["target_row_summary"]:
        source_row = source_states[0]["row_records"][int(row_summary["row"])]
        row_summary["source_finite_set_globally_safe_negative_side"] = bool(
            source_row["globally_safe_robust_negative_side_observed"]
        )
        row_summary["source_finite_set_robust_positive_side"] = bool(
            source_row["robust_positive_side_observed"]
        )
        row_summary["useful_boundary_observed_combined"] = bool(
            (
                row_summary["robust_positive_side_observed"]
                or row_summary["source_finite_set_robust_positive_side"]
            )
            and (
                row_summary["globally_safe_robust_negative_side_observed"]
                or row_summary["source_finite_set_globally_safe_negative_side"]
            )
        )
    result["MLP_training_authorized"] = False
    result["QP_authorized"] = False
    result["closed_loop_authorized"] = False
    result["interpretation"] = (
        "bounded_active_search_found_useful_boundary"
        if any(item["useful_boundary_observed_combined"]
               for item in result["target_row_summary"])
        else "bounded_active_search_no_useful_boundary_observed_in_job"
    )
    result["result_payload_sha256"] = _sha256(canonical(result))
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "job": job,
        "interpretation": result["interpretation"],
        "summary": result["summary"],
        "target_row_summary": result["target_row_summary"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
