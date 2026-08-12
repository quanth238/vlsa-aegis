#!/usr/bin/env python3
"""Independent validation for the E05 fixed-step avoidance field gate."""

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
    import numpy as np

    result = json.loads(path.read_text())
    claimed = result["result_payload_sha256"]
    payload = dict(result)
    payload.pop("result_payload_sha256")
    _require(hashlib.sha256(_canonical(payload)).hexdigest() == claimed, "result hash differs")
    _require(result["source"]["commit"] == expected_commit, "source commit differs")
    _require(not result["source"]["dirty"], "source tree was dirty")
    _require(result["status"] == "complete" and result["scientific_result"], "result incomplete")
    _require(
        result["config"]["claim_scope"].startswith("single_archived_e05_state_avoidance_only"),
        "claim scope differs",
    )
    nominal = result["nominal"]
    _require(
        197 in {int(event["step"]) for event in nominal["rollout"]["protected_contacts"]},
        "nominal action-197 contact differs",
    )
    step = float(result["config"]["field_estimation"]["fixed_normalized_step_action"])
    maximum = float(result["config"]["action_space"]["maximum_total_correction_l2_action"])
    recomputed = result["recomputed_field"]
    path_records = recomputed["path"]
    _require(path_records, "recomputed path is empty")
    for index, candidate in enumerate(path_records):
        correction = np.asarray(candidate["correction"], dtype=np.float64).reshape(15)
        _require(float(np.linalg.norm(correction)) <= maximum + 1.0e-9, "correction exceeds budget")
        if index:
            previous = np.asarray(path_records[index - 1]["correction"], dtype=np.float64).reshape(15)
            _require(
                abs(float(np.linalg.norm(correction - previous)) - step) <= 1.0e-9,
                "recomputed path did not take a full fixed step",
            )
            _require(
                abs(float(candidate["cumulative_registered_step_action"]) - index * step)
                <= 1.0e-9,
                "cumulative fixed-step budget differs",
            )
    fixed = result["fixed_initial_direction"]
    direction = np.asarray(recomputed["initial_unit_direction"], dtype=np.float64).reshape(15)
    for index, candidate in enumerate(fixed["path"]):
        correction = np.asarray(candidate["correction"], dtype=np.float64).reshape(15)
        _require(
            float(np.linalg.norm(correction - index * step * direction)) <= 1.0e-8,
            "fixed-direction comparator changed direction",
        )
    gate = result["gate"]
    recomputed_safe = any(bool(item["avoidance_gate"]) for item in path_records)
    _require(
        recomputed_safe == bool(gate["recomputed_field_verified_safe"]),
        "recomputed field gate differs",
    )
    analytical = result["prior_safe_baselines"]["iterative_analytical"]
    derivative_free = result["prior_safe_baselines"]["derivative_free"]
    _require(analytical["avoidance_gate"], "prior analytical safe support differs")
    _require(derivative_free["avoidance_gate"], "prior derivative-free support differs")
    expected_pass = bool(
        gate["recomputed_field_verified_safe"]
        and gate["action_197_contact_eliminated"]
        and gate["paper_car_pass"]
        and gate["within_shared_radius_one_budget"]
        and gate["recomputation_beats_fixed_initial_direction"]
        and gate["analytical_safe_support_exists"]
        and gate["derivative_free_safe_support_exists"]
    )
    _require(expected_pass == bool(gate["passed"]), "aggregate gate differs")
    return {
        "schema_version": "vlsa_distal_fixed_step_counterfactual_field_e05_validation.v1",
        "status": "valid",
        "scientific_result": True,
        "source_result_path": str(path),
        "source_result_payload_sha256": claimed,
        "source_commit": expected_commit,
        "validator_commit": validator_commit,
        "rollout_count": int(result["rollout_count"]),
        "nominal_hard_margin_mm": 1000.0 * float(nominal["hard_margin_m"]),
        "recomputed_best_hard_margin_mm": 1000.0
        * float(recomputed["best"]["hard_margin_m"]),
        "fixed_initial_best_hard_margin_mm": 1000.0
        * float(fixed["best"]["hard_margin_m"]),
        "recomputed_field_verified_safe": recomputed_safe,
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
    receipt = validate(
        args.result.resolve(), args.expected_commit, args.validator_commit
    )
    receipt["validation_payload_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    _write_atomic(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
