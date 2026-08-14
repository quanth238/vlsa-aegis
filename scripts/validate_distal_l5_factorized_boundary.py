#!/usr/bin/env python3
"""Independently validate one grouped factorized-boundary producer case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity,
)


def main(argv: Sequence[str] | None = None) -> int:
    from main.multilink_ellipsoid.l5_factorized_boundary import (
        VALIDATION_SCHEMA, classify_evaluation, file_sha256, load_config,
        load_manifest, payload_sha256, phase_coverage,
    )
    from main.multilink_ellipsoid.query_boundary_coverage import (
        initially_safe, nominal_prefix_unsafe,
        load_config as load_coverage_config,
    )
    from scripts.validate_distal_l5_aegis_consistent_risk import validate as validate_risk

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    config = load_config(args.config.resolve(), root)
    rows = load_manifest(args.manifest.resolve(), config)
    row = rows[args.case_index]
    case_root = args.producer_root.resolve() / ("case-%02d" % args.case_index)
    result_path = case_root / "results" / "aegis" / row["case_id"] / "result.json"
    if not result_path.is_file():
        failure = case_root / "runtime-failure.json"
        output = {
            "schema_version": VALIDATION_SCHEMA, "status": "producer_failed",
            "scientific_result": False, "case_id": row["case_id"],
            "split": row["factorized_split"],
            "producer_failure_present": failure.is_file(),
            "producer_failure_file_sha256": file_sha256(failure) if failure.is_file() else None,
            "validator_source": _git_identity(root, args.validator_commit),
        }
        output["payload_sha256"] = payload_sha256(output)
        _atomic_write(args.output.resolve(), output)
        return 0
    recomputed = classify_evaluation(
        row=row, result_path=result_path, results_root=case_root / "results"
    )
    classification_path = case_root / "classification.json"
    observed = json.loads(classification_path.read_text(encoding="utf-8"))
    if recomputed != observed or observed.get("payload_sha256") != payload_sha256(observed):
        raise ValueError("factorized classification replay differs")
    coverage_path = case_root / "coverage" / "result.json"
    risk_path = case_root / "risk" / "result.json"
    checks = {"classification_reproduced": True}
    retained_count = 0
    risk_report = None
    risk_validation_summary = None
    if observed["eligible"]:
        if not coverage_path.is_file():
            raise ValueError("eligible factorized case lacks coverage")
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        payload = dict(coverage)
        claimed = payload.pop("result_payload_sha256")
        if claimed != __import__("hashlib").sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest():
            raise ValueError("factorized coverage payload differs")
        coverage_config = load_coverage_config(
            root / config["method_bindings"]["query_coverage_config"]
        )
        retained = []
        for item in coverage["boundary_records"]:
            current = item["current"]
            prefix = item["nominal_prefix"]
            keep = initially_safe(
                current["row_clearance_m"],
                protected_contact_count=current["protected_contact_count"],
                active_obstacle_l1_displacement_m=current[
                    "active_obstacle_l1_displacement_m"
                ], config=coverage_config,
            ) and nominal_prefix_unsafe(
                prefix["row_minimum_clearance_m"],
                protected_contact_count=prefix["protected_contact_count"],
                maximum_active_obstacle_l1_displacement_m=prefix[
                    "maximum_active_obstacle_l1_displacement_m"
                ], config=coverage_config,
            )
            if bool(item["retained"]) != bool(keep):
                raise ValueError("factorized coverage retention differs")
            if keep:
                retained.append(item)
        if retained != coverage["retained_states"]:
            raise ValueError("factorized retained states differ")
        retained_count = len(retained)
        checks["coverage_reproduced"] = True
        if retained_count == 1:
            if not risk_path.is_file():
                raise ValueError("factorized retained state lacks risk result")
            risk_validation = validate_risk(
                repo_root=root, result_path=risk_path,
                producer_commit=args.producer_commit,
                validator_commit=args.validator_commit,
                adaptive_collection_v3=True,
            )
            risk_report = phase_coverage(
                json.loads(risk_path.read_text(encoding="utf-8"))
            )
            risk_validation_summary = {
                "file_sha256": risk_validation["producer"]["file_sha256"],
                "result_payload_sha256": risk_validation["producer"][
                    "result_payload_sha256"
                ],
                "no_proxy_safe_physical_collision": risk_validation["checks"][
                    "no_proxy_safe_physical_collision"
                ],
                "mixed_safe_unsafe_support": risk_validation["checks"][
                    "mixed_safe_unsafe_support"
                ],
                "candidate_count": risk_validation["counts"]["candidates"],
                "timeout_count": risk_validation["counts"]["timeouts"],
            }
            checks["risk_protocol_reproduced"] = True
            risk_validation_path = case_root / "risk" / "validation.json"
            _atomic_write(risk_validation_path, risk_validation)
        elif risk_path.exists():
            raise ValueError("factorized case without one retained state has risk result")
    elif coverage_path.exists() or risk_path.exists():
        raise ValueError("ineligible factorized case has downstream artifacts")
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "validated",
        "scientific_result": True, "case_id": row["case_id"],
        "split": row["factorized_split"],
        "classification": observed["classification"],
        "eligible": observed["eligible"], "retained_state_count": retained_count,
        "risk_report": risk_report,
        "risk_validation_summary": risk_validation_summary,
        "checks": checks,
        "producer_commit": args.producer_commit,
        "validator_source": _git_identity(root, args.validator_commit),
    }
    output["payload_sha256"] = payload_sha256(output)
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
