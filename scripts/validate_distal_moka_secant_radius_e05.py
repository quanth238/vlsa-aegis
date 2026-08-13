#!/usr/bin/env python3
"""Independent arithmetic validator for the matched secant-radius ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "vlsa_distal_moka_secant_radius_e05_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _gate(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, bool]:
    threshold = config["gate"]
    return {
        "mlp_fit_cosine": float(report["compact_mlp"]["fit"]["cosine"]) >= float(threshold["minimum_fit_response_cosine"]),
        "mlp_fit_sign": float(report["compact_mlp"]["fit"]["sign_accuracy"]) >= float(threshold["minimum_fit_sign_accuracy"]),
        "mlp_heldout_cosine": float(report["compact_mlp"]["heldout"]["cosine"]) >= float(threshold["minimum_heldout_response_cosine"]),
        "mlp_heldout_sign": float(report["compact_mlp"]["heldout"]["sign_accuracy"]) >= float(threshold["minimum_heldout_sign_accuracy"]),
        "near_active_row_cosine": float(report["compact_mlp"]["near_active_row_gradient_cosine_to_ridge"]["mean"]) >= float(threshold["minimum_near_active_row_cosine_to_ridge"]),
        "ridge_heldout_cosine": float(report["local_ridge"]["heldout"]["cosine"]) >= float(threshold["minimum_ridge_heldout_response_cosine"]),
        "ridge_heldout_sign": float(report["local_ridge"]["heldout"]["sign_accuracy"]) >= float(threshold["minimum_ridge_heldout_sign_accuracy"]),
    }


def validate(result_path: Path, reference_path: Path, expected_commit: str) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    reference = json.loads(reference_path.read_text())
    _require(result.get("schema_version") == "vlsa_distal_moka_secant_radius_e05_result.v1", "schema differs")
    _require(result.get("status") == "complete", "result incomplete")
    _require(result.get("source", {}).get("commit") == expected_commit, "commit differs")
    _require(result.get("rollout_count") == 129, "rollout count differs")
    _require(result.get("correction_or_qp_attempted") is False, "control attempted")
    binding = result.get("direction_binding", {})
    _require(binding.get("count") == 32 and binding.get("dimension") == 15, "directions differ")
    _require(binding.get("generation_radius_action") == 0.05, "generation radius differs")
    _require(binding.get("identical_for_all_test_radii") is True, "directions not shared")
    _require(len(str(binding.get("directions_float64_sha256"))) == 64, "direction hash differs")
    _require(result.get("reference", {}).get("result_file_sha256") == _file_sha256(reference_path), "reference file differs")
    _require(result.get("reference", {}).get("result_payload_sha256") == reference.get("result_payload_sha256"), "reference payload differs")
    reports = result.get("radius_reports", {})
    _require(set(reports) == {"0.025", "0.0125"}, "radius reports differ")
    passing = []
    for name in ("0.025", "0.0125"):
        report = reports[name]
        _require(float(report["radius_action"]) == float(name), "radius differs")
        _require(report["local_ridge"]["design_rank"] == 15, "design rank differs")
        for section in (report["compact_mlp"]["fit"], report["compact_mlp"]["heldout"], report["local_ridge"]["fit"], report["local_ridge"]["heldout"]):
            for key in ("cosine", "sign_accuracy", "rmse_m_per_action"):
                _require(math.isfinite(float(section[key])), "metric is nonfinite")
        checks = _gate(result["config"], report)
        _require(report["gate"]["checks"] == checks, "radius checks differ")
        _require(report["gate"]["pass"] is bool(all(checks.values())), "radius gate differs")
        if all(checks.values()):
            passing.append(name)
    _require(result.get("passing_radii") == passing, "passing radii differ")
    _require(result.get("overall_gate_pass") is bool(passing), "overall gate differs")
    for key in ("ridge_row_cosine", "mlp_row_cosine", "near_active_ridge_row_cosine", "near_active_mlp_row_cosine"):
        section = result["cross_radius"][key]
        _require(int(section["finite_count"]) > 0 and math.isfinite(float(section["mean"])), "cross-radius metric differs")
    payload = dict(result)
    recorded = payload.pop("result_payload_sha256", None)
    _require(hashlib.sha256(_canonical(payload)).hexdigest() == recorded, "payload hash differs")
    output = {
        "schema_version": SCHEMA,
        "status": "validated",
        "expected_commit": expected_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": recorded,
        "reference_file_sha256": _file_sha256(reference_path),
        "overall_gate_pass": result["overall_gate_pass"],
        "passing_radii": passing,
        "interpretation": result["interpretation"],
        "rollout_count": result["rollout_count"],
    }
    output["validation_payload_sha256"] = hashlib.sha256(_canonical(output)).hexdigest()
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(args.result.resolve(), args.reference_result.resolve(), args.expected_commit)
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
