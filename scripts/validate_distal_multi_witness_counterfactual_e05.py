#!/usr/bin/env python3
"""Independent validator for the E05 multi-witness field comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(path: Path, expected_commit: str, validator_commit: str) -> dict[str, Any]:
    result = json.loads(path.read_text())
    claimed = result["result_payload_sha256"]
    payload = dict(result)
    payload.pop("result_payload_sha256")
    _require(hashlib.sha256(_canonical(payload)).hexdigest() == claimed, "result hash differs")
    _require(result["source"]["commit"] == expected_commit, "source commit differs")
    _require(not result["source"]["dirty"], "source tree was dirty")
    _require(result["status"] == "complete" and result["scientific_result"], "result incomplete")
    _require(
        197 in {int(event["step"]) for event in result["nominal"]["rollout"]["protected_contacts"]},
        "nominal action-197 contact differs",
    )
    arms = result["arms"]
    _require(set(arms) == {"single_hard_min", "smooth_max", "multi_witness"}, "arms differ")
    maximum_path = float(result["config"]["action_space"]["maximum_total_path_length_action"])
    for name, arm in arms.items():
        _require(float(arm["path_length_action"]) <= maximum_path + 1.0e-9, "%s path differs" % name)
        for iteration in arm["iterations"]:
            selected = iteration["selected"]
            if selected is not None:
                _require(
                    float(selected["hard_margin_m"]) > float(iteration["center"]["hard_margin_m"]),
                    "%s accepted non-improving step" % name,
                )
            if name == "multi_witness":
                witnesses = iteration["witnesses"]
                _require(2 <= len(witnesses) <= 8, "multi-witness count differs")
                identities = {(int(item["action_offset"]), int(item["ellipsoid_row"])) for item in witnesses}
                _require(len(identities) == len(witnesses), "multi-witness identities collapsed")
    derivative = result["prior_derivative_free"]
    _require(derivative["avoidance_gate"], "derivative-free support differs")
    multi = arms["multi_witness"]["best"]
    gate = result["gate"]
    expected = {
        "multi_witness_verified_safe": bool(multi["avoidance_gate"]),
        "positive_clearance": bool(float(multi["hard_margin_m"]) >= 0.0),
        "zero_protected_contact": bool(not multi["rollout"]["protected_contacts"]),
        "paper_car_pass": bool(
            float(multi["rollout"]["maximum_active_obstacle_l1_displacement_m"])
            <= float(result["config"]["gate"]["paper_car_threshold_m"])
        ),
        "within_shared_path_budget": bool(
            float(arms["multi_witness"]["path_length_action"]) <= maximum_path + 1.0e-10
        ),
        "beats_single_hard_min_margin": bool(
            float(multi["hard_margin_m"])
            > float(arms["single_hard_min"]["best"]["hard_margin_m"]) + 1.0e-9
        ),
        "derivative_free_safe_support_exists": True,
    }
    for key, value in expected.items():
        _require(bool(gate[key]) == value, "gate differs: %s" % key)
    expected_pass = bool(all(expected.values()))
    _require(bool(gate["passed"]) == expected_pass, "aggregate gate differs")
    return {
        "schema_version": "vlsa_distal_multi_witness_counterfactual_e05_validation.v1",
        "status": "valid",
        "scientific_result": True,
        "source_result_path": str(path),
        "source_result_payload_sha256": claimed,
        "source_commit": expected_commit,
        "validator_commit": validator_commit,
        "rollout_count": int(result["rollout_count"]),
        "nominal_hard_margin_mm": 1000.0 * float(result["nominal"]["hard_margin_m"]),
        "single_hard_min_best_mm": 1000.0
        * float(arms["single_hard_min"]["best"]["hard_margin_m"]),
        "smooth_max_best_mm": 1000.0 * float(arms["smooth_max"]["best"]["hard_margin_m"]),
        "multi_witness_best_mm": 1000.0 * float(multi["hard_margin_m"]),
        "multi_witness_verified_safe": bool(multi["avoidance_gate"]),
        "gate_passed": expected_pass,
        "interpretation": result["interpretation"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(args.result.resolve(), args.expected_commit, args.validator_commit)
    receipt["validation_payload_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    _write_atomic(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
