#!/usr/bin/env python3
"""Audit realized post-AEGIS monotonicity on a validated grouped dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
)


def audit(
    *, repo_root: Path, run_root: Path, dataset_summary_path: Path,
    config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.empirical_candidate_risk_dataset import (
        SUMMARY_SCHEMA as DATASET_SUMMARY_SCHEMA,
    )
    from main.multilink_ellipsoid.empirical_risk_monotonicity import (
        RESULT_SCHEMA,
        canonical,
        case_report,
        classify,
        load_config,
        sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    config = load_config(config_path)
    dataset_config_path = repo_root / config["dataset_config"]
    if _file_sha256(dataset_config_path) != config["dataset_config_file_sha256"]:
        raise ValueError("empirical monotonicity dataset config differs")
    summary = _load(dataset_summary_path)
    if summary.get("schema_version") != DATASET_SUMMARY_SCHEMA:
        raise ValueError("empirical monotonicity dataset summary schema differs")
    if summary.get("status") != "validated":
        raise ValueError("empirical monotonicity dataset is not validated")
    if summary.get("config", {}).get("config_file_sha256") != _file_sha256(
        dataset_config_path
    ):
        raise ValueError("empirical monotonicity dataset binding differs")
    reports = []
    for index, artifact in enumerate(summary["case_artifacts"]):
        result_path = run_root / ("case-%02d" % index) / "result.json"
        if _file_sha256(result_path) != artifact["file_sha256"]:
            raise ValueError("empirical monotonicity case artifact differs")
        result = _load(result_path)
        source_path = Path(result["source_curve"]["path"])
        if _file_sha256(source_path) != artifact["source_curve_file_sha256"]:
            raise ValueError("empirical monotonicity source curve differs")
        source = _load(source_path)
        reports.append(case_report(result, source, config))
    classification = classify(
        reports,
        config,
        dataset_strict_gate_pass=bool(summary["strict_gate_pass"]),
    )
    value = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": allocation_record(),
        "config": config,
        "dataset": {
            "summary_path": str(dataset_summary_path),
            "summary_file_sha256": _file_sha256(dataset_summary_path),
            "summary_payload_sha256": summary["validation_payload_sha256"],
            "producer_commit": summary["validator_commit"],
            "strict_gate_pass": bool(summary["strict_gate_pass"]),
        },
        "case_reports": reports,
        "classification": classification,
        "monotonic_regularization_authorized": classification[
            "monotonic_regularization_authorized"
        ],
        "MLP_training_authorized": False,
        "interpretation": (
            "realized_monotonicity_supports_matched_regularization_ablation"
            if classification["monotonic_regularization_authorized"]
            else "realized_monotonicity_does_not_authorize_regularization"
        ),
    }
    value["result_payload_sha256"] = sha256(canonical(value))
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = audit(
        repo_root=args.repo_root.resolve(),
        run_root=args.run_root.resolve(),
        dataset_summary_path=args.dataset_summary.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "classification": value["classification"],
        "interpretation": value["interpretation"],
        "result_payload_sha256": value["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
