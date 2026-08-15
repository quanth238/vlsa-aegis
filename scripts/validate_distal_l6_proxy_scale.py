#!/usr/bin/env python3
"""Independently validate the empirical L6 proxy-scale audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _load,
    _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def validate(
    *, result_path: Path, config_path: Path, expected_commit: str
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l6_proxy_scale_audit import (
        RESULT_SCHEMA,
        audit_source,
        load_config,
    )

    result = _load(result_path)
    config = load_config(config_path)
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("L6 proxy-scale result schema differs")
    if result.get("source", {}).get("commit") != expected_commit:
        raise ValueError("L6 proxy-scale producer commit differs")
    payload = result.get("result_payload_sha256")
    copy = dict(result)
    copy.pop("result_payload_sha256", None)
    if payload != _sha256(_canonical(copy)):
        raise ValueError("L6 proxy-scale result payload differs")
    if result.get("config") != config:
        raise ValueError("L6 proxy-scale embedded config differs")
    source = _load(Path(config["source_result"]))
    analysis = audit_source(source, config)
    if result["analysis"] != analysis:
        raise ValueError("L6 proxy-scale analysis differs")
    if bool(result["strict_gate_pass"]) != bool(analysis["strict_gate_pass"]):
        raise ValueError("L6 proxy-scale gate differs")
    return {
        "schema_version": "vlsa_distal_l6_proxy_scale_audit_validation.v1",
        "status": "complete",
        "source_result": str(result_path),
        "source_result_file_sha256": _file_sha256(result_path),
        "source_result_payload_sha256": payload,
        "validator_commit": expected_commit,
        "strict_gate_pass": bool(result["strict_gate_pass"]),
        "interpretation": result["interpretation"],
        "selected_scale": analysis["selected_scale"],
        "gates": analysis["gates"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        result_path=args.result.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    value["validation_payload_sha256"] = _sha256(_canonical(value))
    _atomic_write(args.output.resolve(), value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
