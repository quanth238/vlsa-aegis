#!/usr/bin/env python3
"""Independently validate the immutable archived-E05 field-oracle result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


RESULT_SCHEMA = "vlsa_distal_field_mixture_oracle_e05_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_field_mixture_oracle_e05_validation.v1"
TABLE1_FILE_SHA256 = "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"


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


def _validate_candidate(
    candidate: Mapping[str, Any],
    nominal_actions: Any,
    *,
    minimum_progress: float,
    maximum_obstacle_displacement: float,
    maximum_correction_norm: float,
) -> dict[str, Any]:
    import numpy as np

    correction = np.asarray(candidate["correction"], dtype=np.float64)
    actions = np.asarray(candidate["actions"], dtype=np.float64)
    nominal = np.asarray(nominal_actions, dtype=np.float64)
    _require(correction.shape == (6,), "candidate correction shape differs")
    _require(actions.shape == nominal.shape == (2, 7), "candidate action shape differs")
    correction_norm = float(np.linalg.norm(correction))
    _require(
        abs(correction_norm - float(candidate["correction_l2"])) <= 1.0e-10,
        "candidate correction norm differs",
    )
    _require(correction_norm <= maximum_correction_norm + 1.0e-10, "candidate exceeds trust region")
    expected = nominal.copy()
    expected[:, :3] += correction.reshape(2, 3)
    _require(np.max(np.abs(actions - expected)) <= 1.0e-12, "candidate action/correction differs")
    _require(
        np.max(np.abs(actions[:, :3])) <= 1.0 + 1.0e-12,
        "candidate XYZ action bound differs",
    )
    exact = candidate["exact"]
    rows = np.asarray(exact["row_minimum_m"], dtype=np.float64)
    _require(rows.shape == (7,), "candidate row count differs")
    _require(
        abs(float(np.min(rows)) - float(exact["minimum_row_m"])) <= 1.0e-12,
        "candidate minimum margin differs",
    )
    no_contact = all(
        int(step["raw_protected_contact"]["nonpositive_protected_contact_count"]) == 0
        for step in exact["steps"]
    )
    obstacle_displacement = float(
        candidate["active_obstacle_l1_displacement_from_episode_start_m"]
    )
    exact_safe = bool(
        float(exact["minimum_row_m"]) >= 0.0
        and obstacle_displacement <= maximum_obstacle_displacement
        and no_contact
    )
    progressing = bool(float(candidate["task_progress_ratio"]) >= minimum_progress)
    _require(candidate["no_raw_l5_l6_l7_contact"] is no_contact, "candidate contact flag differs")
    _require(candidate["exact_safe"] is exact_safe, "candidate exact-safe flag differs")
    _require(candidate["task_progressing"] is progressing, "candidate progress flag differs")
    _require(
        candidate["safe_and_task_progressing"] is bool(exact_safe and progressing),
        "candidate combined flag differs",
    )
    return {
        "correction_l2": correction_norm,
        "minimum_row_m": float(exact["minimum_row_m"]),
        "task_progress_ratio": float(candidate["task_progress_ratio"]),
        "safe_and_task_progressing": bool(exact_safe and progressing),
    }


def validate(result_path: Path, expected_producer_commit: str) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.field_oracle import load_field_oracle_config

    result = _load(result_path)
    _require(result.get("schema_version") == RESULT_SCHEMA, "result schema differs")
    _require(result.get("status") == "complete", "result is incomplete")
    _require(result.get("scientific_result") is True, "result is not scientific")
    _require(result.get("case_id") == "vlsa-t1-goal-ii-t0-e05", "case differs")
    _require(result["source"]["commit"] == expected_producer_commit, "producer commit differs")
    _require(result["source"]["dirty"] is False, "producer source was dirty")
    _require(result["allocation"]["slurm_job_id"] is not None, "Slurm job is absent")
    _require("H100" in str(result["allocation"]["device"]["name"]), "producer GPU was not H100")
    _require(result["archived_table1"]["read_only"] is True, "Table 1 was not read-only")
    _require(result["archived_table1"]["file_sha256"] == TABLE1_FILE_SHA256, "Table-1 hash differs")
    payload_hash = result.pop("result_payload_sha256")
    _require(_sha256(_canonical(result)) == payload_hash, "result payload hash differs")
    result["result_payload_sha256"] = payload_hash

    config_path = Path(__file__).resolve().parents[1] / "configs/vlsa_distal_field_mixture_oracle_e05.v1.json"
    config = load_field_oracle_config(config_path)
    for key in ("config_file_sha256", "config_payload_sha256"):
        _require(result["config"][key] == config[key], "config hash differs: %s" % key)

    nominal = result["nominal"]
    nominal_actions = np.asarray(nominal["actions"], dtype=np.float64)
    nominal_rows = np.asarray(nominal["rollout"]["row_minimum_m"], dtype=np.float64)
    _require(nominal_actions.shape == (2, 7), "nominal action shape differs")
    _require(nominal_rows.shape == (7,), "nominal row count differs")
    nominal_minimum = float(nominal["rollout"]["minimum_row_m"])
    _require(abs(nominal_minimum - float(np.min(nominal_rows))) <= 1.0e-12, "nominal minimum differs")
    _require(
        abs(nominal_minimum - float(config["nominal"]["expected_minimum_margin_m"]))
        <= float(config["nominal"]["expected_minimum_tolerance_m"]),
        "archived dangerous margin differs",
    )

    minimum_progress = float(config["success_definition"]["minimum_nominal_eef_progress_ratio"])
    maximum_obstacle_displacement = float(
        config["success_definition"]["maximum_active_obstacle_l1_displacement_from_episode_start_m"]
    )
    maximum_correction_norm = float(config["search"]["maximum_total_correction_l2"])
    expected_arms = list(config["comparison"]["arms"])
    _require(set(result["arms"]) == set(expected_arms), "arm names differ")
    arm_checks: dict[str, Any] = {}
    for arm_name in expected_arms:
        arm = result["arms"][arm_name]
        _require(arm["arm_name"] == arm_name, "arm name differs")
        _require(len(arm["iterations"]) == int(config["search"]["iterations"]), "iteration count differs")
        exposed = [arm["best"]]
        for index, iteration in enumerate(arm["iterations"]):
            _require(int(iteration["iteration"]) == index, "iteration index differs")
            _require(int(iteration["candidate_count"]) > 0, "iteration has no candidates")
            exposed.extend([iteration["selected"], iteration["best_so_far"]])
        summaries = [
            _validate_candidate(
                candidate,
                nominal_actions,
                minimum_progress=minimum_progress,
                maximum_obstacle_displacement=maximum_obstacle_displacement,
                maximum_correction_norm=maximum_correction_norm,
            )
            for candidate in exposed
        ]
        matched_checks = []
        matched = arm["matched_final_correction_norms"]
        expected_radii = list(config["comparison"]["matched_final_correction_norms"])
        _require(len(matched) == len(expected_radii), "matched-radius count differs")
        for record, expected_radius in zip(matched, expected_radii):
            _require(float(record["radius"]) == float(expected_radius), "matched radius differs")
            _require(int(record["candidate_count"]) > 0, "matched radius has no candidates")
            summary = _validate_candidate(
                record["best"],
                nominal_actions,
                minimum_progress=minimum_progress,
                maximum_obstacle_displacement=maximum_obstacle_displacement,
                maximum_correction_norm=maximum_correction_norm,
            )
            _require(abs(summary["correction_l2"] - float(expected_radius)) <= 1.0e-10, "matched correction norm differs")
            matched_checks.append({"radius": float(expected_radius), **summary})
        best_summary = summaries[0]
        expected_gate = bool(best_summary["safe_and_task_progressing"])
        _require(arm["gate_pass"] is expected_gate, "arm gate differs")
        arm_checks[arm_name] = {
            "gate_pass": expected_gate,
            "best": best_summary,
            "matched_final_correction_norms": matched_checks,
            "rollout_count": int(arm["rollout_count"]),
        }

    unrestricted = bool(arm_checks[expected_arms[0]]["gate_pass"])
    structured = bool(
        arm_checks["nonnegative_normal_field_mixture"]["gate_pass"]
        or arm_checks["normal_plus_task_tangent_mixture"]["gate_pass"]
    )
    overall = bool(unrestricted and structured)
    _require(result["unrestricted_oracle_ceiling_pass"] is unrestricted, "unrestricted gate differs")
    _require(result["structured_field_gate_pass"] is structured, "structured gate differs")
    _require(result["primary_problem_solved"] is overall, "overall gate differs")
    interpretation = (
        "structured_field_oracle_supported"
        if structured
        else (
            "two_action_repair_exists_but_structured_fields_fail"
            if unrestricted
            else "bounded_two_action_oracle_no_safe_task_progressing_support"
        )
    )
    _require(result["interpretation"] == interpretation, "interpretation differs")
    receipt = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": payload_hash,
        "producer_commit": expected_producer_commit,
        "nominal_minimum_row_m": nominal_minimum,
        "arm_checks": arm_checks,
        "unrestricted_oracle_ceiling_pass": unrestricted,
        "structured_field_gate_pass": structured,
        "primary_problem_solved": overall,
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
