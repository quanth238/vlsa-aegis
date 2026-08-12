#!/usr/bin/env python3
"""Independent structural/scientific validation of the pilot aggregate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.aggregate_distal_repulsion_generalization_pilot import aggregate
from scripts.replay_distal_three_ellipsoid_multicbf import _load, _require, _sha256


VALIDATION_SCHEMA = "vlsa_distal_repulsion_generalization_validation.v1"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(
    *,
    summary_path: Path,
    config_path: Path,
    selection_manifest_path: Path,
    result_paths: Sequence[Path],
    expected_commit: str,
) -> dict[str, Any]:
    summary = _load(summary_path)
    recomputed = aggregate(
        config_path=config_path,
        selection_manifest_path=selection_manifest_path,
        result_paths=result_paths,
        expected_commit=expected_commit,
    )
    _require(summary == recomputed, "generalization summary differs from recomputation")
    _require(
        summary["schema_version"] == "vlsa_distal_repulsion_generalization_summary.v1",
        "generalization summary schema differs",
    )
    required_case_gate = []
    for path in result_paths:
        result = _load(path)
        raw = result["arms"]["raw_aegis"]
        _require(not raw["gates"]["buffer_0mm"], "generalization raw case is not unsafe")
        _require(result["gate"]["raw_reproduces_collision"], "historical contact did not reproduce")
        _require(
            all(count == 25 for count in raw["final"]["record"]["substep_counts"]),
            "generalization internal substep count differs",
        )
        required_case_gate.append(
            {
                "case_id": result["case_id"],
                "raw_collision_reproduced": True,
                "smooth_positive_clearance_gain": bool(
                    result["arms"]["smooth"]["clearance_gain_m"] > 0.0
                ),
                "smooth_buffer_0mm": bool(
                    result["arms"]["smooth"]["gates"]["buffer_0mm"]
                ),
                "smooth_buffer_1mm": bool(
                    result["arms"]["smooth"]["gates"]["buffer_1mm"]
                ),
            }
        )
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passed",
        "scientific_result": True,
        "producer_source_commit": expected_commit,
        "summary_payload_sha256": summary["result_payload_sha256"],
        "summary_interpretation": summary["interpretation"],
        "case_checks": required_case_gate,
        "aggregate": summary["aggregate"],
    }
    validation["validation_payload_sha256"] = _sha256(
        json.dumps(validation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        summary_path=args.summary.resolve(),
        config_path=args.config.resolve(),
        selection_manifest_path=args.selection_manifest.resolve(),
        result_paths=[path.resolve() for path in args.result],
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
