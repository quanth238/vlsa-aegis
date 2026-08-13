#!/usr/bin/env python3
"""Independent validator for the frozen 64-direction ranker audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "vlsa_distal_moka_frozen_ranker_audit_e05_validation.v1"


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


def _binomial(successes: int, trials: int) -> float:
    return float(
        sum(math.comb(trials, value) for value in range(successes, trials + 1))
        / float(2 ** trials)
    )


def validate(result_path: Path, frozen_path: Path, expected_commit: str) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    _require(result.get("schema_version") == "vlsa_distal_moka_frozen_ranker_audit_e05_result.v1", "schema differs")
    _require(result.get("status") == "complete", "result incomplete")
    _require(result.get("source", {}).get("commit") == expected_commit, "commit differs")
    _require(result.get("rollout_count") == 129, "rollout count differs")
    _require(result.get("training_or_control_attempted") is False, "training or control attempted")
    _require(result.get("frozen_model", {}).get("result_file_sha256") == _file_sha256(frozen_path), "frozen file differs")
    _require(result.get("frozen_model", {}).get("result_payload_sha256") == frozen.get("result_payload_sha256"), "frozen payload differs")
    _require(result.get("frozen_model", {}).get("model_sha256") == frozen["model_reports"]["direction_conditioned_scalar"]["model"]["model_sha256"], "frozen model differs")
    binding = result.get("direction_binding", {})
    _require(binding.get("count") == 64 and binding.get("dimension") == 15, "directions differ")
    _require(binding.get("direction_seed") == 2026081410, "direction seed differs")
    _require(binding.get("radius_action") == 0.0125, "radius differs")
    _require(binding.get("disjoint_from_32_development_directions_up_to_sign") is True, "directions overlap development")
    _require(len(str(binding.get("directions_float64_sha256"))) == 64, "direction hash differs")
    metrics = result.get("metrics", {})
    successes = int(metrics["successes"])
    count = int(metrics["direction_count"])
    _require(count == 64, "metric count differs")
    _require(abs(float(metrics["branch_accuracy"]) - successes / count) <= 1.0e-15, "branch accuracy differs")
    _require(abs(float(metrics["wrong_direction_rate"]) - (1.0 - successes / count)) <= 1.0e-15, "wrong direction rate differs")
    _require(abs(float(metrics["one_sided_binomial_p"]) - _binomial(successes, count)) <= 1.0e-15, "binomial p differs")
    thresholds = result["config"]["gate"]
    checks = {
        "branch_accuracy": float(metrics["branch_accuracy"]) >= float(thresholds["minimum_branch_accuracy"]),
        "binomial_significance": float(metrics["one_sided_binomial_p"]) < float(thresholds["maximum_one_sided_binomial_p"]),
        "positive_near_active_l5_gain": float(metrics["selected_near_active_l5_gain"]["mean_m"]) > float(thresholds["minimum_mean_selected_near_active_l5_gain_m"]),
    }
    _require(metrics["gate"]["checks"] == checks, "gate checks differ")
    _require(metrics["gate"]["pass"] is bool(all(checks.values())), "gate differs")
    _require(result.get("overall_gate_pass") is metrics["gate"]["pass"], "overall gate differs")
    reports = result.get("best_of_n", {})
    _require(set(reports) == {"4", "8", "16", "32", "64"}, "Best-of-N reports differ")
    for size in (4, 8, 16, 32, 64):
        report = reports[str(size)]
        _require(1 <= int(report["exact_rank"]) <= size, "Best-of-N rank differs")
        _require(float(report["regret_m"]) >= -1.0e-15, "Best-of-N regret differs")
        _require(0.0 <= float(report["exact_percentile"]) <= 1.0, "Best-of-N percentile differs")
    payload = dict(result)
    recorded = payload.pop("result_payload_sha256", None)
    _require(hashlib.sha256(_canonical(payload)).hexdigest() == recorded, "payload hash differs")
    output = {
        "schema_version": SCHEMA,
        "status": "validated",
        "expected_commit": expected_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": recorded,
        "frozen_file_sha256": _file_sha256(frozen_path),
        "branch_accuracy": metrics["branch_accuracy"],
        "one_sided_binomial_p": metrics["one_sided_binomial_p"],
        "overall_gate_pass": result["overall_gate_pass"],
        "interpretation": result["interpretation"],
        "rollout_count": result["rollout_count"],
    }
    output["validation_payload_sha256"] = hashlib.sha256(_canonical(output)).hexdigest()
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--frozen-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(args.result.resolve(), args.frozen_result.resolve(), args.expected_commit)
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
