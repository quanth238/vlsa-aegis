#!/usr/bin/env python3
"""Calibrate an empirical L6 proxy scale from immutable validated controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import _allocation_record
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def audit(*, repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.l6_proxy_scale_audit import (
        RESULT_SCHEMA,
        audit_source,
        load_config,
    )

    config = load_config(config_path)
    result_path = Path(config["source_result"])
    validation_path = Path(config["source_validation"])
    _require(
        _file_sha256(result_path) == config["source_result_file_sha256"],
        "L6 proxy-scale source result differs",
    )
    _require(
        _file_sha256(validation_path) == config["source_validation_file_sha256"],
        "L6 proxy-scale source validation differs",
    )
    source = _load(result_path)
    validation = _load(validation_path)
    _require(
        source["result_payload_sha256"] == config["source_result_payload_sha256"],
        "L6 proxy-scale source payload differs",
    )
    _require(
        validation["validation_payload_sha256"]
        == config["source_validation_payload_sha256"],
        "L6 proxy-scale validation payload differs",
    )
    _require(
        validation["source_result_payload_sha256"] == source["result_payload_sha256"],
        "L6 proxy-scale validation binding differs",
    )
    analysis = audit_source(source, config)
    value = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": _allocation_record(),
        "config": config,
        "source_result_binding": {
            "path": str(result_path),
            "file_sha256": config["source_result_file_sha256"],
            "payload_sha256": source["result_payload_sha256"],
        },
        "source_validation_binding": {
            "path": str(validation_path),
            "file_sha256": config["source_validation_file_sha256"],
            "payload_sha256": validation["validation_payload_sha256"],
        },
        "analysis": analysis,
        "strict_gate_pass": analysis["strict_gate_pass"],
        "interpretation": (
            "empirical_L6_proxy_scale_mechanism_pass"
            if analysis["strict_gate_pass"]
            else "empirical_L6_proxy_scale_mechanism_no_go"
        ),
        "certificate_status": config["calibration"]["certificate_status"],
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
                "selected_scale": value["analysis"]["selected_scale"],
                "gates": value["analysis"]["gates"],
                "result_payload_sha256": value["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
