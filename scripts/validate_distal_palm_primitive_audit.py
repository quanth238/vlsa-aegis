#!/usr/bin/env python3
"""Independently compare and aggregate the frozen palm replay cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def _scientific_view(record: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    for key in ("allocation", "source", "wall_seconds", "result_payload_sha256"):
        value.pop(key, None)
    return value


def validate(
    *, repo_root: Path, config_path: Path, producer_dir: Path,
    replay_dir: Path, producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.palm_primitive_audit import (
        RESULT_SCHEMA,
        RESULT_SCHEMA_V2,
        VALIDATION_SCHEMA,
        canonical,
        load_config,
        payload_sha256,
        summarize_case_records,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, validator_commit)
    config = load_config(config_path, repo_root=repo_root)
    case_ids = (
        list(config["cohort"]["contact_case_ids"])
        + list(config["cohort"]["control_case_ids"])
    )

    def load_directory(path: Path, expected_commit: str) -> list[dict[str, Any]]:
        output = []
        for case_id in case_ids:
            record_path = path / (str(case_id) + ".json")
            _require(record_path.is_file(), "palm audit case result is missing")
            record = _load(record_path)
            expected_schema = (
                RESULT_SCHEMA_V2 if "compiled_obstacle" in config else RESULT_SCHEMA
            )
            _require(record.get("schema_version") == expected_schema,
                     "palm audit case schema differs")
            _require(record.get("case_id") == case_id,
                     "palm audit case identity differs")
            _require(record.get("source", {}).get("commit") == expected_commit,
                     "palm audit case source commit differs")
            _require(record.get("result_payload_sha256") == payload_sha256(record),
                     "palm audit case payload differs")
            output.append(record)
        return output

    producer = load_directory(producer_dir, producer_commit)
    replay = load_directory(replay_dir, validator_commit)
    mismatches = []
    for left, right in zip(producer, replay):
        if canonical(_scientific_view(left)) != canonical(_scientific_view(right)):
            mismatches.append(str(left["case_id"]))
    producer_summary = summarize_case_records(producer, config)
    replay_summary = summarize_case_records(replay, config)
    _require(not mismatches, "independent palm replay differs")
    _require(
        canonical(producer_summary) == canonical(replay_summary),
        "independent palm summary differs",
    )
    geometry_pass = bool(producer_summary["geometry_gate_pass"])
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passed" if geometry_pass else "scientific_no_go",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": identity,
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "allocation": allocation_record(),
        "config": config,
        "case_ids": case_ids,
        "independent_replay": {
            "case_count": len(case_ids),
            "mismatch_case_ids": mismatches,
            "exact_scientific_reproduction": not mismatches,
        },
        "summary": producer_summary,
        "boundary_collection_authorized": geometry_pass,
        "authorized_boundary_groups": (
            list(config["next_if_pass"]["boundary_groups"])
            if geometry_pass else []
        ),
        "training_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": producer_summary["interpretation"],
    }
    validation["validation_payload_sha256"] = payload_sha256(validation)
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        producer_dir=args.producer_dir.resolve(),
        replay_dir=args.replay_dir.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "summary": result["summary"],
        "validation_payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
