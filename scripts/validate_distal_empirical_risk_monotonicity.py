#!/usr/bin/env python3
"""Independently replay the empirical-risk monotonicity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.audit_distal_empirical_risk_monotonicity import audit
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _load,
    _sha256,
)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def validate(
    *, repo_root: Path, run_root: Path, dataset_summary_path: Path,
    audit_path: Path, config_path: Path, expected_commit: str,
):
    from main.multilink_ellipsoid.empirical_risk_monotonicity import (
        RESULT_SCHEMA,
        VALIDATION_SCHEMA,
    )

    recorded = _load(audit_path)
    if recorded.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("empirical monotonicity result schema differs")
    copy = dict(recorded)
    payload = copy.pop("result_payload_sha256", None)
    if payload != _sha256(_canonical(copy)):
        raise ValueError("empirical monotonicity result payload differs")
    replay = audit(
        repo_root=repo_root,
        run_root=run_root,
        dataset_summary_path=dataset_summary_path,
        config_path=config_path,
        expected_commit=expected_commit,
    )
    if replay["case_reports"] != recorded["case_reports"]:
        raise ValueError("empirical monotonicity case replay differs")
    if replay["classification"] != recorded["classification"]:
        raise ValueError("empirical monotonicity classification replay differs")
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "audit_file_sha256": _file_sha256(audit_path),
        "audit_payload_sha256": payload,
        "dataset_summary_file_sha256": _file_sha256(dataset_summary_path),
        "classification": recorded["classification"],
        "monotonic_regularization_authorized": recorded[
            "monotonic_regularization_authorized"
        ],
        "MLP_training_authorized": bool(
            recorded["monotonic_regularization_authorized"]
        ),
        "interpretation": recorded["interpretation"],
    }
    value["validation_payload_sha256"] = _sha256(_canonical(value))
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        repo_root=args.repo_root.resolve(), run_root=args.run_root.resolve(),
        dataset_summary_path=args.dataset_summary.resolve(),
        audit_path=args.audit.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
