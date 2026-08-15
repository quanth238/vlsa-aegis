#!/usr/bin/env python3
"""Independently reproduce the immutable four-task L5 availability audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.audit_distal_l5_four_task_availability import classify_case
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def validate(
    *, repo_root: Path, table1_root: Path, result_path: Path,
    config_path: Path, producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_four_task_availability import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, load_config, load_population,
        payload_sha256, summarize_records,
    )

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path, repo_root=repo_root)
    result = _load(result_path)
    _require(
        result["schema_version"] == RESULT_SCHEMA
        and result["source"]["commit"] == producer_commit,
        "availability result identity differs",
    )
    _require(
        result["result_payload_sha256"] == payload_sha256(result)
        and result["config"] == config,
        "availability result payload differs",
    )
    rows = load_population(
        repo_root / config["source"]["population_manifest"], config,
    )
    records = [classify_case(row, table1_root, config) for row in rows]
    summary = summarize_records(records, config)
    _require(
        records == result["records"] and summary == result["summary"],
        "availability independent replay differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "summary": summary,
        "maximum_case_mismatch_count": 0,
        "training_authorized": False,
        "boundary_collection_authorized": bool(
            summary["all_four_task_folds_feasible"]
        ),
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["validation_payload_sha256"] = payload_sha256(output)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root.resolve(), table1_root=args.table1_root.resolve(),
        result_path=args.result.resolve(), config_path=args.config.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "summary": output["summary"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
