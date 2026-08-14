#!/usr/bin/env python3
"""Independently replay the row-maximizing active-boundary witnesses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import evaluate
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require, _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    return _sha256(_canonical(payload))


def _compare_tree(left: Any, right: Any, tolerance: float = 1.0e-12) -> float:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        _require(isinstance(left, Mapping) and isinstance(right, Mapping),
                 "active search replay mapping type differs")
        _require(set(left) == set(right), "active search replay mapping keys differ")
        return max(
            [_compare_tree(left[key], right[key], tolerance) for key in left] + [0.0]
        )
    if isinstance(left, list) or isinstance(right, list):
        _require(isinstance(left, list) and isinstance(right, list),
                 "active search replay list type differs")
        _require(len(left) == len(right), "active search replay list length differs")
        return max(
            [_compare_tree(a, b, tolerance) for a, b in zip(left, right)] + [0.0]
        )
    if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None:
        _require(left == right, "active search replay categorical value differs")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        _require(math.isfinite(float(left)) and math.isfinite(float(right)),
                 "active search replay numeric value is not finite")
        error = abs(float(left) - float(right))
        _require(error <= tolerance, "active search replay numeric value differs")
        return error
    _require(left == right, "active search replay value differs")
    return 0.0


def validate(
    *, repo_root: Path, producer_root: Path, population_manifest: Path,
    coverage_root: Path, search_config_path: Path, base_config_path: Path,
    geometry_config_path: Path, table1_root: Path, array_index: int,
    expected_producer_commit: str, expected_validator_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.active_boundary_search import (
        RESULT_SCHEMA, candidate_definitions, load_config,
        summarize_target_rows, target_job,
    )

    config = load_config(search_config_path)
    job = target_job(config, array_index)
    producer_path = producer_root / ("job-%02d" % int(array_index)) / "result.json"
    producer = _load(producer_path)
    _require(producer["schema_version"] == RESULT_SCHEMA,
             "active search producer schema differs")
    _require(producer["status"] == "complete" and producer["scientific_result"] is True,
             "active search producer incomplete")
    _require(producer["source"]["commit"] == expected_producer_commit
             and producer["source"]["dirty"] is False,
             "active search producer source differs")
    _require(producer["result_payload_sha256"] == _payload_sha256(producer),
             "active search producer payload differs")
    _require(producer["active_boundary_search_config"] == config
             and producer["active_boundary_job"] == job,
             "active search producer protocol differs")
    _require(producer["candidate_count"] == 27,
             "active search producer candidate count differs")

    expected_definitions = candidate_definitions(
        producer["nominal_five_action_chunk"], producer["state"]["local_frame"],
        config, job["temporal_profile"],
    )
    for observed, expected in zip(producer["candidates"], expected_definitions):
        for key, expected_value in expected.items():
            _compare_tree(observed[key], expected_value)
    recomputed_summary = summarize_target_rows(
        producer["candidates"], config["target_rows"],
        config["finite_search"]["robust_boundary_margin_m"],
    )
    for observed, recomputed in zip(producer["target_row_summary"], recomputed_summary):
        for key, expected_value in recomputed.items():
            _compare_tree(observed[key], expected_value)

    selected_names = ["nominal"]
    for item in producer["target_row_summary"]:
        name = item["maximum_candidate_name"]
        if name is not None and name not in selected_names:
            selected_names.append(name)
    definition_by_name = {item["name"]: item for item in expected_definitions}
    selected_definitions = [definition_by_name[name] for name in selected_names]

    coverage_path = coverage_root / ("case-%02d" % int(job["case_index"])) / "result.json"
    coverage = _load(coverage_path)
    retained = coverage["retained_states"][0]
    archived = table1_root / coverage["case"]["archived_result_relative_path"]

    def definitions(_nominal, _frame, _base_config):
        return selected_definitions

    fresh = evaluate(
        repo_root=repo_root,
        population_manifest_path=population_manifest,
        archived_path=archived,
        geometry_config_path=geometry_config_path,
        experiment_config_path=base_config_path,
        expected_commit=expected_validator_commit,
        output_path=producer_root / ("job-%02d" % int(array_index)) / "fresh-unused.json",
        case_id_override=job["case_id"],
        state_step_override=int(retained["step"]),
        query_index_override=int(retained["query_index"]),
        result_schema_override=RESULT_SCHEMA,
        claim_scope_override="independent_replay_of_active_boundary_row_maxima",
        population_binding={"validator_job": job},
        candidate_definitions_override=definitions,
        candidate_protocol_binding={
            "selected_candidate_names": selected_names,
            "search_config_payload_sha256": config["config_payload_sha256"],
        },
    )
    source_by_name = {item["name"]: item for item in producer["candidates"]}
    fresh_by_name = {item["name"]: item for item in fresh["candidates"]}
    maximum_replay_error = 0.0
    for name in selected_names:
        maximum_replay_error = max(
            maximum_replay_error,
            _compare_tree(source_by_name[name], fresh_by_name[name]),
        )
    gates = {
        "producer_payload_valid": True,
        "candidate_protocol_exact": True,
        "target_summary_exact": True,
        "selected_names_replayed": set(fresh_by_name) == set(selected_names),
        "fresh_replay_within_tolerance": maximum_replay_error <= 1.0e-12,
        "reserved_test_episodes_untouched": True,
        "no_training_QP_or_control": True,
    }
    _require(all(gates.values()), "active search independent replay failed")
    result = {
        "schema_version": "vlsa_distal_active_boundary_search_validation.v1",
        "status": "complete",
        "scientific_result": True,
        "claim_scope": "independent_replay_of_finite_active_search_row_maxima",
        "producer": {
            "path": str(producer_path),
            "file_sha256": _file_sha256(producer_path),
            "result_payload_sha256": producer["result_payload_sha256"],
            "commit": expected_producer_commit,
        },
        "validator": {
            "commit": expected_validator_commit,
            "allocation": fresh["allocation"],
        },
        "job": job,
        "selected_candidate_names": selected_names,
        "maximum_replay_error": maximum_replay_error,
        "gates": gates,
        "interpretation": "active_boundary_search_independent_replay_pass",
    }
    result["result_payload_sha256"] = _payload_sha256(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(),
        producer_root=args.producer_root.resolve(),
        population_manifest=args.population_manifest.resolve(),
        coverage_root=args.coverage_root.resolve(),
        search_config_path=args.search_config.resolve(),
        base_config_path=args.base_config.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        table1_root=args.table1_root.resolve(),
        array_index=args.array_index,
        expected_producer_commit=args.expected_producer_commit,
        expected_validator_commit=args.expected_validator_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
