#!/usr/bin/env python3
"""Aggregate the three immutable generalization-pilot case results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _file_sha256,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_repulsion_generalization_summary.v1"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_case(path: Path, expected_commit: str) -> dict[str, Any]:
    result = _load(path)
    _require(
        result.get("schema_version")
        == "vlsa_distal_repulsion_generalization_case_result.v1",
        "generalization case schema differs",
    )
    _require(result.get("status") == "complete", "generalization case is incomplete")
    _require(
        result.get("source", {}).get("commit") == expected_commit,
        "generalization case source differs",
    )
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(
        _sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        == claimed,
        "generalization case self-hash differs",
    )
    return result


def aggregate(
    *, config_path: Path, selection_manifest_path: Path, result_paths: Sequence[Path], expected_commit: str
) -> dict[str, Any]:
    from main.multilink_ellipsoid.repulsion_generalization import (
        aggregate_results,
        load_cases,
        load_config,
    )

    config = load_config(config_path)
    cases = load_cases(selection_manifest_path, config)
    results = [_validate_case(path, expected_commit) for path in result_paths]
    _require(
        [result["case_id"] for result in results] == [row["case_id"] for row in cases],
        "generalization aggregate case order differs",
    )
    aggregate_record = aggregate_results(results, config)
    aggregate_record["passed"] = bool(
        aggregate_record["mechanism_gate_pass"]
        and aggregate_record["smooth_beats_or_matches_fixed_safe_rate"]
    )
    output = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source_commit": expected_commit,
        "config": config,
        "selection_manifest": {
            "path": str(selection_manifest_path),
            "file_sha256": _file_sha256(selection_manifest_path),
            "case_ids": [row["case_id"] for row in cases],
        },
        "case_results": [
            {
                "case_id": result["case_id"],
                "path": str(path),
                "file_sha256": _file_sha256(path),
                "payload_sha256": result["result_payload_sha256"],
                "interpretation": result["interpretation"],
                "nominal_internal_minimum_clearance_m": result[
                    "nominal_internal_minimum_clearance_m"
                ],
                "arms": {
                    name: {
                        "minimum_clearance_m": arm["final"]["record"][
                            "minimum_clearance_m"
                        ],
                        "clearance_gain_m": arm["clearance_gain_m"],
                        "correction_l2_action": arm["correction_l2_action"],
                        "gates": arm["gates"],
                    }
                    for name, arm in result["arms"].items()
                },
            }
            for path, result in zip(result_paths, results)
        ],
        "aggregate": aggregate_record,
        "interpretation": (
            "smooth_counterfactual_repulsion_mechanism_transfers_to_pilot_cases"
            if aggregate_record["passed"]
            else "smooth_counterfactual_repulsion_generalization_pilot_no_go"
        ),
    }
    output["result_payload_sha256"] = _sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate(
        config_path=args.config.resolve(),
        selection_manifest_path=args.selection_manifest.resolve(),
        result_paths=[path.resolve() for path in args.result],
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "interpretation": result["interpretation"],
                "passed": result["aggregate"]["passed"],
                "result_payload_sha256": result["result_payload_sha256"],
                "status": result["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
