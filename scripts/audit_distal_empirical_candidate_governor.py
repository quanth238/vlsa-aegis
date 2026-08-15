#!/usr/bin/env python3
"""Replay four fixed normal banks using the frozen empirical L6 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.audit_distal_compiled_box_risk_target import _evaluate_case
from scripts.evaluate_distal_query_action_risk_e05 import _allocation_record
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _require,
    _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def audit(*, repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.empirical_candidate_governor import (
        RESULT_SCHEMA,
        classify,
        load_config,
    )
    from main.multilink_ellipsoid.l6_proxy_scale_audit import (
        load_empirical_proxy_config,
    )

    config = load_config(config_path)
    population = repo_root / config["population_manifest"]
    geometry = repo_root / config["geometry_config"]
    proxy_path = repo_root / config["empirical_l6_proxy_config"]
    _require(
        _file_sha256(population) == config["population_manifest_file_sha256"],
        "empirical candidate-governor population differs",
    )
    _require(
        _file_sha256(geometry) == config["geometry_config_file_sha256"],
        "empirical candidate-governor geometry differs",
    )
    _require(
        _file_sha256(proxy_path) == config["empirical_l6_proxy_config_file_sha256"],
        "empirical candidate-governor proxy config differs",
    )
    proxy = load_empirical_proxy_config(proxy_path)
    audit_config = dict(config)
    audit_config["empirical_l6_proxy"] = proxy
    cases = [
        _evaluate_case(
            repo_root=repo_root,
            population_manifest=population,
            geometry_config_path=geometry,
            case_config=case,
            audit_config=audit_config,
        )
        for case in config["cases"]
    ]
    classification = classify(cases, config["gate"])
    value = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": _allocation_record(),
        "config": config,
        "empirical_l6_proxy": proxy,
        "cases": cases,
        "classification": classification,
        "strict_gate_pass": classification["strict_gate_pass"],
        "interpretation": (
            "empirical_candidate_governor_oracle_pass"
            if classification["strict_gate_pass"]
            else "empirical_candidate_governor_oracle_no_go"
        ),
        "training_authorized": False,
    }
    value["result_payload_sha256"] = _sha256(_canonical(value))
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = audit(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(
        json.dumps(
            {
                "strict_gate_pass": value["strict_gate_pass"],
                "interpretation": value["interpretation"],
                "classification": value["classification"],
                "result_payload_sha256": value["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
