#!/usr/bin/env python3
"""Independently validate the first-warning calibrated episode diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _file_sha256, _load, _require, _sha256,
)


SUMMARY_SCHEMA = "vlsa_distal_first_warning_calibrated_repulsion_summary.v1"
VALIDATION_SCHEMA = "vlsa_distal_first_warning_calibrated_repulsion_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


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
    *, config_path: Path, result_paths: Sequence[Path], expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from main.multilink_ellipsoid.first_warning_calibrated_repulsion import (
        RESULT_SCHEMA, aggregate, load_config,
    )

    config = load_config(config_path)
    _require(len(result_paths) == len(config["case_ids"]), "calibrated result count differs")
    rows = []
    checks = []
    for index, (case_id, radius, path) in enumerate(zip(
        config["case_ids"], config["requested_radius_by_case"], result_paths
    )):
        result = _load(path)
        _require(result["schema_version"] == RESULT_SCHEMA, "calibrated result schema differs")
        _require(result["source"]["commit"] == expected_commit, "calibrated source differs")
        _require(result["case_id"] == case_id, "calibrated case differs")
        payload = dict(result)
        claimed = payload.pop("result_payload_sha256")
        _require(_sha256(_canonical(payload)) == claimed, "calibrated result self-hash differs")
        binding = result["controller_binding"]
        _require(binding["calibrated_config"] == config, "calibrated embedded config differs")
        _require(binding["online_adaptation"] is False, "calibrated online claim differs")
        _require(float(binding["requested_radius"]) == float(radius), "calibrated radius differs")
        for intervention in result["interventions"]:
            _require(
                float(intervention["proposal"]["requested_correction_l2_action"])
                == float(radius),
                "calibrated intervention radius differs",
            )
        rows.append(result)
        checks.append({
            "case_index": index,
            "case_id": case_id,
            "requested_radius": float(radius),
            "result_file_sha256": _file_sha256(path),
            "result_payload_sha256": claimed,
            "intervention_count": int(result["intervention_count"]),
            "clipped_intervention_count": int(result["clipped_intervention_count"]),
            "raw_L5_L7_contact_pass": bool(result["raw_L5_L7_contact_pass"]),
            "paper_car_pass": bool(result["paper_car_pass"]),
            "native_task_success": bool(result["native_task_success"]),
            "timeout": bool(result["timeout"]),
            "primary_problem_solved": bool(result["primary_problem_solved"]),
        })
    combined = aggregate(rows, config)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source_commit": expected_commit,
        "config": config,
        "cases": checks,
        "aggregate": combined,
        "interpretation": (
            "first_warning_calibrated_magnitude_passes_safe_task_gate"
            if combined["strict_gate_pass"]
            else "first_warning_calibrated_magnitude_no_go"
        ),
    }
    summary["result_payload_sha256"] = _sha256(_canonical(summary))
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passed",
        "scientific_result": True,
        "source_commit": expected_commit,
        "summary_payload_sha256": summary["result_payload_sha256"],
        "checks": checks,
        "aggregate": combined,
    }
    validation["validation_payload_sha256"] = _sha256(_canonical(validation))
    return summary, validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args(argv)
    summary, validation = validate(
        config_path=args.config.resolve(),
        result_paths=[path.resolve() for path in args.result],
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.summary.resolve(), summary)
    _atomic_write(args.validation.resolve(), validation)
    print(json.dumps({
        "interpretation": summary["interpretation"],
        "strict_gate_pass": summary["aggregate"]["strict_gate_pass"],
        "result_payload_sha256": summary["result_payload_sha256"],
        "validation_payload_sha256": validation["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
