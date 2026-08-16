#!/usr/bin/env python3
"""Freeze a symmetric coverage-preserving bank from opened training cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def select(
    *, repo_root: Path, bank_config_path: Path, audit_config_path: Path,
    cohort_config_path: Path,
    producer_dir: Path, source_audit_path: Path, expected_commit: str,
    expected_artifact_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, load_cases, load_config, payload_sha256 as artifact_payload_sha256,
    )
    from main.multilink_ellipsoid.whole_body_candidate_bank import (
        load_bank_config, select_bank,
    )
    from main.multilink_ellipsoid.whole_body_support_audit import (
        load_audit_config, payload_sha256,
    )

    bank_config = load_bank_config(bank_config_path)
    audit_config = load_audit_config(audit_config_path)
    cohort_config = load_config(cohort_config_path)
    selections = load_cases(repo_root / cohort_config["selection_manifest"], cohort_config)
    source_audit = _load(source_audit_path)
    _require(
        source_audit.get("audit_payload_sha256")
        == bank_config["source_support_audit_payload_sha256"],
        "source support audit payload differs",
    )
    _require(
        hashlib.sha256(source_audit_path.read_bytes()).hexdigest()
        == bank_config["source_support_audit_file_sha256"],
        "source support audit file differs",
    )
    records = []
    for selection in selections:
        record = _load(producer_dir / (selection["case_id"] + ".json"))
        _require(record.get("schema_version") == CASE_SCHEMA,
                 "candidate-bank case schema differs")
        _require(record.get("source", {}).get("commit") == expected_artifact_commit,
                 "candidate-bank artifact commit differs")
        _require(
            record.get("result_payload_sha256") == artifact_payload_sha256(record),
            "candidate-bank artifact payload differs",
        )
        records.append(record)
    selection = select_bank(
        records, audit_config,
        bank_size=int(bank_config["selected_candidate_count"]),
        development_split=str(bank_config["development_split"]),
    )
    output = {
        "schema_version": "vlsa_distal_whole_body_candidate_bank_selection.v1",
        "status": selection["selection_status"],
        "scientific_result": False,
        "claim_scope": "Development-only candidate-bank reduction; no simulation, training, prediction, correction, QP, denoising, closed loop, or safety claim.",
        "source": _git_identity(repo_root, expected_commit),
        "artifact_source_commit": expected_artifact_commit,
        "bank_config": bank_config,
        "source_support_audit_payload_sha256": source_audit[
            "audit_payload_sha256"
        ],
        "selection": selection,
        "training_authorized": False,
        "control_authorized": False,
    }
    output["selection_payload_sha256"] = payload_sha256(
        output, key="selection_payload_sha256",
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bank-config", type=Path, required=True)
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--cohort-config", type=Path, required=True)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-artifact-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = select(
        repo_root=args.repo_root.resolve(),
        bank_config_path=args.bank_config.resolve(),
        audit_config_path=args.audit_config.resolve(),
        cohort_config_path=args.cohort_config.resolve(),
        producer_dir=args.producer_dir.resolve(),
        source_audit_path=args.source_audit.resolve(),
        expected_commit=args.expected_commit,
        expected_artifact_commit=args.expected_artifact_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "selected_candidate_names": result["selection"].get(
            "selected_candidate_names", []
        ),
        "selected_score": result["selection"].get("selected_score"),
        "feasible_bank_count": result["selection"].get("feasible_bank_count"),
        "selection_payload_sha256": result["selection_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
