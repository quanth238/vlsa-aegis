#!/usr/bin/env python3
"""Fresh H100 replay validator for the pure-backup representative audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.evaluate_distal_pncbf_backup_oracle_e05 import _canonical
from scripts.evaluate_distal_pncbf_pure_backup_audit_e05 import evaluate
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


def _max_numeric_error(left: Any, right: Any) -> float:
    if isinstance(left, bool) or isinstance(right, bool):
        _require(left is right, "boolean evidence differs")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if isinstance(left, str) or left is None:
        _require(left == right, "scalar evidence differs")
        return 0.0
    if isinstance(left, list):
        _require(isinstance(right, list) and len(left) == len(right), "list evidence differs")
        return max((_max_numeric_error(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, dict):
        _require(isinstance(right, dict) and set(left) == set(right), "mapping evidence differs")
        return max((_max_numeric_error(left[key], right[key]) for key in left), default=0.0)
    _require(left == right, "evidence differs")
    return 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    accepted = _load(args.result.resolve())
    _require(accepted["schema_version"] == "vlsa_distal_pncbf_pure_backup_audit_e05_result.v1", "result schema differs")
    _require(accepted["source"]["commit"] == args.source_commit, "producer source differs")
    _git_identity(args.repo_root.resolve(), args.validator_commit)
    replay = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        source_result_path=args.source_result.resolve(),
        expected_commit=args.validator_commit,
    )
    error = _max_numeric_error(accepted["audit_states"], replay["audit_states"])
    _require(error <= 1.0e-12, "fresh audit replay differs")
    _require(accepted["gates"] == replay["gates"], "fresh gates differ")
    _require(accepted["interpretation"] == replay["interpretation"], "fresh interpretation differs")
    validation = {
        "schema_version": "vlsa_distal_pncbf_pure_backup_audit_e05_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "source_result_file_sha256": _file_sha256(args.result.resolve()),
        "source_result_payload_sha256": accepted["result_payload_sha256"],
        "producer_commit": args.source_commit,
        "validator_commit": args.validator_commit,
        "maximum_fresh_replay_error": error,
        "gates": replay["gates"],
        "interpretation": replay["interpretation"],
        "primary_problem_solved": replay["primary_problem_solved"],
    }
    validation["validation_payload_sha256"] = _sha256(_canonical(validation))
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
