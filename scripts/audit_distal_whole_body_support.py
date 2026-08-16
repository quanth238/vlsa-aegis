#!/usr/bin/env python3
"""Audit whole-body constraint and global support without new simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)
from scripts.validate_distal_exact_group_boundary import _scientific_view


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    *, repo_root: Path, audit_config_path: Path, cohort_config_path: Path,
    producer_dir: Path, replay_dir: Path, source_validation_path: Path,
    expected_audit_commit: str, expected_artifact_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, canonical as artifact_canonical, load_cases, load_config,
        payload_sha256 as artifact_payload_sha256,
    )
    from main.multilink_ellipsoid.whole_body_support_audit import (
        AUDIT_SCHEMA, audit_cases, load_audit_config, payload_sha256,
    )

    audit_config = load_audit_config(audit_config_path)
    cohort_config = load_config(cohort_config_path)
    cases = load_cases(repo_root / cohort_config["selection_manifest"], cohort_config)
    source_validation = _load(source_validation_path)
    _require(
        _file_sha256(source_validation_path)
        == audit_config["source_validation_file_sha256"],
        "source validation file hash differs",
    )
    _require(
        source_validation.get("validation_payload_sha256")
        == audit_config["source_validation_payload_sha256"],
        "source validation payload hash differs",
    )
    _require(
        source_validation.get("validation_payload_sha256")
        == artifact_payload_sha256(
            source_validation, key="validation_payload_sha256",
        ),
        "source validation payload is invalid",
    )

    def load_directory(path: Path) -> list[dict[str, Any]]:
        output = []
        for selection in cases:
            record = _load(path / (selection["case_id"] + ".json"))
            _require(record.get("schema_version") == CASE_SCHEMA,
                     "whole-body case schema differs")
            _require(record.get("case_id") == selection["case_id"],
                     "whole-body case identity differs")
            _require(record.get("source", {}).get("commit") == expected_artifact_commit,
                     "whole-body artifact commit differs")
            _require(
                record.get("result_payload_sha256")
                == artifact_payload_sha256(record),
                "whole-body artifact payload differs",
            )
            output.append(record)
        return output

    producer = load_directory(producer_dir)
    replay = load_directory(replay_dir)
    mismatches = [
        left["case_id"] for left, right in zip(producer, replay)
        if artifact_canonical(_scientific_view(left))
        != artifact_canonical(_scientific_view(right))
    ]
    _require(not mismatches, "whole-body support independent replay differs")
    producer_summary = audit_cases(producer, audit_config)
    replay_summary = audit_cases(replay, audit_config)
    _require(
        artifact_canonical(producer_summary) == artifact_canonical(replay_summary),
        "whole-body support summaries differ",
    )
    output = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed_read_only_whole_body_support_audit",
        "scientific_result": False,
        "claim_scope": audit_config["claim_scope"],
        "source": _git_identity(repo_root, expected_audit_commit),
        "artifact_source_commit": expected_artifact_commit,
        "audit_config": audit_config,
        "cohort_config_file_sha256": cohort_config["config_file_sha256"],
        "source_validation": {
            "path": str(source_validation_path),
            "file_sha256": _file_sha256(source_validation_path),
            "validation_payload_sha256": source_validation[
                "validation_payload_sha256"
            ],
        },
        "independent_replay": {
            "case_count": len(cases),
            "mismatch_case_ids": mismatches,
            "exact_scientific_reproduction": not mismatches,
        },
        "summary": producer_summary,
        "training_authorized": False,
        "correction_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["audit_payload_sha256"] = payload_sha256(
        output, key="audit_payload_sha256",
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--cohort-config", type=Path, required=True)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--source-validation", type=Path, required=True)
    parser.add_argument("--expected-audit-commit", required=True)
    parser.add_argument("--expected-artifact-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        repo_root=args.repo_root.resolve(),
        audit_config_path=args.audit_config.resolve(),
        cohort_config_path=args.cohort_config.resolve(),
        producer_dir=args.producer_dir.resolve(),
        replay_dir=args.replay_dir.resolve(),
        source_validation_path=args.source_validation.resolve(),
        expected_audit_commit=args.expected_audit_commit,
        expected_artifact_commit=args.expected_artifact_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "summary": result["summary"],
        "audit_payload_sha256": result["audit_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
