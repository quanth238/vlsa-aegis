#!/usr/bin/env python3
"""Independent validator for the matched local nonlinear prediction gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "vlsa_distal_moka_local_action_value_e05_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


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


def _checks(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, bool]:
    gate = config["gate"]
    response = report["test_response"]
    branches = report["test_branches"]
    return {
        "test_response_cosine": float(response["all_witnesses"]["cosine"])
        >= float(gate["minimum_test_response_cosine"]),
        "test_response_sign": float(response["all_witnesses"]["sign_accuracy"])
        >= float(gate["minimum_test_response_sign_accuracy"]),
        "near_active_response_cosine": float(response["near_active_witnesses"]["cosine"])
        >= float(gate["minimum_near_active_response_cosine"]),
        "near_active_response_sign": float(response["near_active_witnesses"]["sign_accuracy"])
        >= float(gate["minimum_near_active_response_sign_accuracy"]),
        "safer_branch_accuracy": float(branches["safer_branch_accuracy"])
        >= float(gate["minimum_safer_branch_accuracy"]),
        "worst_margin_rmse": float(branches["worst_margin_rmse_m"])
        <= float(gate["maximum_test_worst_margin_rmse_m"]),
    }


def validate(result_path: Path, reference_path: Path, expected_commit: str) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    reference = json.loads(reference_path.read_text())
    _require(result.get("schema_version") == "vlsa_distal_moka_local_action_value_e05_result.v1", "schema differs")
    _require(result.get("status") == "complete", "result incomplete")
    _require(result.get("source", {}).get("commit") == expected_commit, "commit differs")
    _require(result.get("rollout_count") == 65, "rollout count differs")
    _require(result.get("correction_or_qp_attempted") is False, "control attempted")
    _require(result.get("reference", {}).get("result_file_sha256") == _file_sha256(reference_path), "reference file differs")
    _require(result.get("reference", {}).get("result_payload_sha256") == reference.get("result_payload_sha256"), "reference payload differs")
    binding = result.get("direction_binding", {})
    _require(binding.get("count") == 32 and binding.get("dimension") == 15, "direction binding differs")
    _require(binding.get("radius_action") == 0.0125, "radius differs")
    _require(binding.get("directions_float64_sha256") == "ebb3a64d63d20e8adffe45e40f2be45e5e9e7fcfde95163fe61bc201368ff5dd", "direction hash differs")
    _require(result.get("split") == {
        "train_direction_indexes": list(range(20)),
        "validation_direction_indexes": list(range(20, 24)),
        "test_direction_indexes": list(range(24, 32)),
    }, "split differs")
    reports = result.get("model_reports", {})
    _require(set(reports) == {"direction_conditioned_scalar", "nonlinear_local_action_value"}, "model arms differ")
    parameter_counts = set()
    passing = []
    for name in sorted(reports):
        report = reports[name]
        parameter_counts.add(int(report["model"]["parameter_count"]))
        for split_name in ("train_response", "validation_response", "test_response"):
            for group in ("all_witnesses", "near_active_witnesses"):
                for metric in ("cosine", "sign_accuracy", "rmse_m_per_action"):
                    _require(math.isfinite(float(report[split_name][group][metric])), "response metric nonfinite")
        _require(report["test_branches"]["direction_count"] == 8, "test direction count differs")
        checks = _checks(result["config"], report)
        _require(report["gate"]["checks"] == checks, "model gate checks differ")
        _require(report["gate"]["pass"] is bool(all(checks.values())), "model gate differs")
        if all(checks.values()):
            passing.append(name)
    _require(len(parameter_counts) == 1, "matched model parameter counts differ")
    _require(result.get("passing_models") == passing, "passing models differ")
    _require(result.get("overall_gate_pass") is bool(passing), "overall gate differs")
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
        "passing_models": passing,
        "overall_gate_pass": result["overall_gate_pass"],
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
