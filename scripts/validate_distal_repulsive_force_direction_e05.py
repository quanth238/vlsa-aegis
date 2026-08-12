#!/usr/bin/env python3
"""Independently validate the immutable E05 repulsive-force result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


VALIDATION_SCHEMA = "vlsa_distal_repulsive_force_direction_e05_validation.v1"
RESULT_SCHEMA = "vlsa_distal_repulsive_force_direction_e05_result.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "input is missing or symlinked")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "input must contain one object")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(result_path: Path, expected_producer_commit: str) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.repulsive_force import (
        load_repulsive_force_config,
        matched_random_p_value,
    )

    result = _load(result_path)
    _require(result.get("schema_version") == RESULT_SCHEMA, "result schema differs")
    _require(result.get("status") == "complete", "result is incomplete")
    _require(result.get("scientific_result") is True, "result is not scientific")
    _require(result.get("case_id") == "vlsa-t1-goal-ii-t0-e05", "case differs")
    _require(result["source"]["commit"] == expected_producer_commit, "producer commit differs")
    _require(result["source"]["dirty"] is False, "producer source was dirty")
    _require(result["allocation"]["slurm_job_id"] is not None, "Slurm job is absent")
    _require("H100" in str(result["allocation"]["gpu"]), "producer GPU was not H100")
    _require(result["archived_table1"]["read_only"] is True, "Table 1 was not read-only")
    _require(
        result["archived_table1"]["file_sha256"]
        == "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b",
        "Table-1 file hash differs",
    )
    payload_hash = result.pop("result_payload_sha256")
    _require(_sha256(_canonical(result)) == payload_hash, "result payload hash differs")
    result["result_payload_sha256"] = payload_hash
    config_path = Path(__file__).resolve().parents[1] / "configs/vlsa_distal_repulsive_force_direction_e05.v1.json"
    config = load_repulsive_force_config(config_path)
    for key in ("config_file_sha256", "config_payload_sha256"):
        _require(result["config"][key] == config[key], "config hash differs: %s" % key)
    _require(result["model"]["frozen_before_test_step"] is True, "model was not frozen")
    _require(result["model"]["training_state_steps"] == [182, 183], "training split differs")
    _require(result["model"]["validation_state_steps"] == [184], "validation split differs")
    test = result["test"]
    _require(test["step"] == 185, "test step differs")
    _require(test["model_sha256_before_test"] == result["model"]["model_sha256"], "test model differs")
    learned_direction = np.asarray(test["learned_direction"], dtype=np.float64)
    fixed_direction = np.asarray(test["fixed_direction"], dtype=np.float64)
    _require(abs(float(np.linalg.norm(learned_direction)) - 1.0) <= 1.0e-10, "learned direction norm differs")
    _require(abs(float(np.linalg.norm(fixed_direction)) - 1.0) <= 1.0e-10, "fixed direction norm differs")
    _require(np.min(np.asarray(test["learned_clearance_weights"], dtype=np.float64)) >= -1.0e-12, "learned force is not monotone")
    radius_checks = []
    for radius_record, expected_radius in zip(test["radii"], [0.1, 0.25]):
        radius = float(radius_record["radius_action"])
        _require(radius == expected_radius, "test radius differs")
        learned = radius_record["learned"]
        fixed = radius_record["fixed"]
        random = radius_record["random"]
        _require(len(random) == 256, "random comparator count differs")
        for name, candidate in (("learned", learned), ("fixed", fixed)):
            correction = np.asarray(candidate["correction"], dtype=np.float64)
            _require(abs(float(np.linalg.norm(correction)) - radius) <= 1.0e-10, "%s correction norm differs" % name)
        random_corrections = np.asarray([item["correction"] for item in random], dtype=np.float64)
        _require(np.max(np.abs(np.linalg.norm(random_corrections, axis=1) - radius)) <= 1.0e-10, "random correction norms differ")
        _require(np.max(np.abs(random_corrections[0::2] + random_corrections[1::2])) <= 1.0e-10, "random directions are not antithetic")
        nominal = float(radius_record["nominal_softmin_margin_m"])
        learned_gain = float(learned["exact"]["softmin_margin_m"] - nominal)
        fixed_gain = float(fixed["exact"]["softmin_margin_m"] - nominal)
        random_gains = np.asarray(
            [item["exact"]["softmin_margin_m"] - nominal for item in random],
            dtype=np.float64,
        )
        p_value = matched_random_p_value(learned_gain, random_gains)
        _require(abs(learned_gain - float(radius_record["learned_exact_gain_m"])) <= 1.0e-12, "learned gain differs")
        _require(abs(fixed_gain - float(radius_record["fixed_exact_gain_m"])) <= 1.0e-12, "fixed gain differs")
        _require(abs((learned_gain - fixed_gain) - float(radius_record["gain_over_fixed_m"])) <= 1.0e-12, "gain over fixed differs")
        _require(abs(p_value - float(radius_record["matched_random_p_value"])) <= 1.0e-12, "random p-value differs")
        gate = config["gate"]
        expected_pass = bool(
            learned_gain >= float(gate["minimum_exact_clearance_gain_m"])
            and learned_gain - fixed_gain >= float(gate["minimum_gain_over_fixed_m"])
            and p_value <= float(gate["maximum_matched_random_p_value"])
        )
        _require(radius_record["direction_gate_pass"] is expected_pass, "radius verdict differs")
        radius_checks.append(
            {
                "radius_action": radius,
                "learned_gain_m": learned_gain,
                "fixed_gain_m": fixed_gain,
                "gain_over_fixed_m": learned_gain - fixed_gain,
                "matched_random_p_value": p_value,
                "pass": expected_pass,
            }
        )
    expected_direction_pass = bool(
        all(item["pass"] for item in radius_checks)
        and test["exact_safe_learned_radius_count"] > 0
    )
    _require(test["direction_gate_pass"] is expected_direction_pass, "test verdict differs")
    _require(result["direction_gate_pass"] is expected_direction_pass, "top-level direction verdict differs")
    continuation = result["continuation"]
    if expected_direction_pass:
        _require(continuation["attempted"] is True, "continuation was not attempted")
        for step in continuation["steps"]:
            _require(step["exact_clone_match"] is True, "continuation clone mismatch")
            _require(step["raw_protected_contact"]["nonpositive_protected_contact_count"] == 0, "continuation contact")
            _require(step["active_obstacle_l1_displacement_from_episode_start_m"] <= 0.001, "continuation CAR")
    else:
        _require(continuation["attempted"] is False, "failed direction gate executed continuation")
        _require(continuation["reason"] == "direction_gate_failed", "continuation refusal differs")
    _require(result["continuation_gate_pass"] is bool(continuation["passed"]), "continuation verdict differs")
    expected_overall = bool(expected_direction_pass and continuation["passed"])
    _require(result["primary_problem_solved"] is expected_overall, "overall verdict differs")
    interpretation = (
        "local_learned_repulsive_mechanism_pass"
        if expected_overall
        else ("learned_force_direction_no_go" if not expected_direction_pass else "learned_force_continuation_no_go")
    )
    _require(result["interpretation"] == interpretation, "interpretation differs")
    receipt = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": payload_hash,
        "producer_commit": expected_producer_commit,
        "config_file_sha256": config["config_file_sha256"],
        "model_sha256": result["model"]["model_sha256"],
        "radius_checks": radius_checks,
        "direction_gate_pass": expected_direction_pass,
        "continuation_gate_pass": bool(continuation["passed"]),
        "primary_problem_solved": expected_overall,
        "interpretation": interpretation,
    }
    receipt["receipt_payload_sha256"] = _sha256(_canonical(receipt))
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(args.result.resolve(), args.expected_producer_commit)
    _atomic_write(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
