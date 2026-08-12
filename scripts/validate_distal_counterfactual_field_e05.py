#!/usr/bin/env python3
"""Independently validate the paired long-horizon E05 field result."""

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


def validate(result_path: Path, expected_commit: str, validator_commit: str) -> dict[str, Any]:
    import numpy as np

    result = json.loads(result_path.read_text())
    claimed = result["result_payload_sha256"]
    payload = dict(result)
    payload.pop("result_payload_sha256")
    _require(hashlib.sha256(_canonical(payload)).hexdigest() == claimed, "result hash differs")
    _require(result["source"]["commit"] == expected_commit, "source commit differs")
    _require(not result["source"]["dirty"], "source tree was dirty")
    _require(result["status"] == "complete" and result["scientific_result"], "result is incomplete")
    _require(
        result["determinism"] == {
            "identical_state_hashes": True,
            "identical_clearance_trace": True,
            "nominal_contact_steps": result["determinism"]["nominal_contact_steps"],
        },
        "determinism record differs",
    )
    _require(197 in result["determinism"]["nominal_contact_steps"], "action-197 collision missing")
    pairs = result["paired_rollouts"]
    _require(len(pairs) == 64, "paired rollout count differs")
    _require(sum(item["split"] == "fit" for item in pairs) == 48, "fit split differs")
    _require(sum(item["split"] == "heldout" for item in pairs) == 16, "heldout split differs")
    for pair in pairs:
        direction = np.asarray(pair["direction"], dtype=np.float64).reshape(5, 3)
        _require(abs(float(np.linalg.norm(direction)) - 1.0) <= 1.0e-10, "direction norm differs")
        _require(float(np.max(np.abs(np.sum(direction, axis=0)))) <= 1.0e-10, "endpoint preservation differs")
        _require(
            abs(float(pair["positive"]["score"]["correction_l2_action"]) - 0.1) <= 1.0e-10
            and abs(float(pair["negative"]["score"]["correction_l2_action"]) - 0.1) <= 1.0e-10,
            "paired correction norm differs",
        )
    comparison = result["comparison"]
    named = [
        comparison["learned_plus"],
        comparison["learned_minus"],
        comparison["fixed_plus"],
        comparison["fixed_minus"],
    ]
    _require(len(comparison["random_candidates"]) == 32, "random comparator count differs")
    for candidate in named + comparison["random_candidates"]:
        _require(
            abs(float(candidate["score"]["correction_l2_action"]) - 0.1) <= 1.0e-10,
            "matched comparator norm differs",
        )
        endpoint = np.asarray(candidate["endpoint_correction_sum"], dtype=np.float64)
        _require(float(np.max(np.abs(endpoint))) <= 1.0e-10, "candidate endpoint differs")
    fit = result["field_fit"]
    gate = result["config"]["gate"]
    prediction_recomputed = bool(
        float(fit["heldout_pearson"]) >= float(gate["heldout_minimum_pearson"])
        and float(fit["heldout_sign_accuracy"]) >= float(gate["heldout_minimum_sign_accuracy"])
    )
    learned = comparison["learned_plus"]
    mechanism_recomputed = bool(
        prediction_recomputed
        and learned["exact_safe"]
        and learned["task_preserving"]
        and float(comparison["learned_utility_gain_m"]) > 0.0
        and float(learned["score"]["utility_m"])
        > float(comparison["learned_minus"]["score"]["utility_m"])
        and float(learned["score"]["utility_m"])
        > float(comparison["fixed_plus"]["score"]["utility_m"])
        and float(comparison["matched_random_p_value"])
        <= float(gate["maximum_random_p_value"])
    )
    _require(prediction_recomputed == bool(result["prediction_gate_pass"]), "prediction gate differs")
    _require(mechanism_recomputed == bool(result["mechanism_gate_pass"]), "mechanism gate differs")
    return {
        "schema_version": "vlsa_distal_counterfactual_field_e05_validation.v1",
        "status": "valid",
        "scientific_result": True,
        "source_result_path": str(result_path),
        "source_result_payload_sha256": claimed,
        "source_commit": expected_commit,
        "validator_commit": validator_commit,
        "rollout_count": int(result["rollout_count"]),
        "nominal_minimum_clearance_mm": 1000.0 * float(result["nominal"]["score"]["clearance_softmin_m"]),
        "nominal_contact_steps": result["determinism"]["nominal_contact_steps"],
        "heldout_pearson": float(fit["heldout_pearson"]),
        "heldout_sign_accuracy": float(fit["heldout_sign_accuracy"]),
        "learned_minimum_clearance_mm": 1000.0 * float(learned["rollout"]["minimum_clearance_m"]),
        "learned_terminal_eef_error_mm": 1000.0 * float(learned["score"]["terminal_eef_error_m"]),
        "learned_exact_safe": bool(learned["exact_safe"]),
        "fixed_minimum_clearance_mm": 1000.0 * float(comparison["fixed_plus"]["rollout"]["minimum_clearance_m"]),
        "random_p_value": float(comparison["matched_random_p_value"]),
        "prediction_gate_pass": prediction_recomputed,
        "mechanism_gate_pass": mechanism_recomputed,
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
