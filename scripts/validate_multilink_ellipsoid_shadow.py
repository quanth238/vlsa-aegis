#!/usr/bin/env python3
"""Validate one H100 AEGIS rollout with the read-only multi-link QP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


CASE_ID = "vlsa-t1-goal-ii-t0-e05"
ARCHIVED_FILE_SHA256 = "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
ARCHIVED_PAYLOAD_SHA256 = "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c"
EXPECTED_MANIFEST_ROW_SHA256 = "d7a11ccf75af4b9f9823cad820fe261a4a791fc9891fe1f605cbed23648c88df"
EXPECTED_INITIAL_STATE_SHA256 = "714334cdae0ad6cd8540115802e9bf9b3591449e29d712ccc4651d3e89c22cf6"
EXPECTED_NOISE_SHA256 = "b50949ebc3b6a774f3800922487c13b00ee76d1f26cec5d8d7cdd8d0197f63e5"
EXPECTED_ACTION_COUNT = 237
EXPECTED_FIRST_ROBOT_CONTACT_STEP = 187
EXPECTED_PAPER_COLLISION_STEP = 188
EXPECTED_TASK_SUCCESS_STEP = 236


class ValidationError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError("%s is missing or symlinked" % label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError("%s is invalid JSON" % label) from error
    if not isinstance(value, dict):
        raise ValidationError("%s must contain one JSON object" % label)
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _payload_hash(result: Mapping[str, Any]) -> str:
    return _sha256(
        _canonical(
            {
                key: value
                for key, value in result.items()
                if key != "result_payload_sha256"
            }
        )
    )


def _direct_robot_contact_geoms(result: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for pair in result["contact_telemetry"]["unique_contact_pairs"]:
        for geom_key in ("geom1", "geom2"):
            name = pair.get(geom_key)
            if isinstance(name, str) and name.startswith("robot0_link"):
                names.add(name)
    return names


def _eef_path_after_contact(result: Mapping[str, Any]) -> float:
    positions = [
        action["post_step_controller_proxy"]["eef_position"]
        for action in result["actions"][EXPECTED_FIRST_ROBOT_CONTACT_STEP:]
    ]
    total = 0.0
    for first, second in zip(positions, positions[1:]):
        total += math.sqrt(sum((float(b) - float(a)) ** 2 for a, b in zip(first, second)))
    return total


def _hand_check_bound(record: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(record["source_geom_kind"])
    size = [float(value) for value in record["source_geom_size_m"]]
    rbound = float(record["source_rbound_m"])
    observed = [float(value) for value in record["semiaxes_m"]]
    source = str(record["bound_source"])
    if source.startswith("mujoco_geom_rbound_sphere"):
        expected = [rbound, rbound, rbound]
    elif kind == "sphere" and source == "exact_mujoco_sphere":
        expected = [size[0], size[0], size[0]]
    elif kind == "ellipsoid" and source == "exact_mujoco_ellipsoid":
        expected = size
    elif kind == "capsule" and source == "closed_form_capsule_enclosing_ellipsoid":
        radial = math.sqrt(size[0] * (size[0] + size[1]))
        expected = [radial, radial, size[0] + size[1]]
    elif kind == "cylinder" and source == "loewner_cylinder_enclosing_ellipsoid":
        expected = [math.sqrt(1.5) * size[0], math.sqrt(1.5) * size[0], math.sqrt(3.0) * size[1]]
    elif kind == "box" and source == "loewner_box_enclosing_ellipsoid":
        expected = [math.sqrt(3.0) * value for value in size]
    else:
        raise ValidationError("unrecognized live arm-geom enclosure formula")
    _require(
        all(abs(first - second) <= 1.0e-12 for first, second in zip(observed, expected)),
        "live arm-geom ellipsoid semiaxes fail independent formula check",
    )
    certificate = record.get("enclosure_certificate")
    _require(
        isinstance(certificate, dict)
        and certificate.get("verified") is True
        and certificate.get("maximum_normalized_quadratic") == 1.0,
        "live arm-geom enclosure certificate is missing",
    )
    return {
        "body_name": record["body_name"],
        "geom_name": record["geom_name"],
        "geom_kind": kind,
        "geom_size_m": size,
        "geom_rbound_m": rbound,
        "ellipsoid_semiaxes_m": observed,
        "bound_source": source,
        "formula_check": "passed",
    }


def _git_identity(root: Path, expected_commit: str) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--short")
    _require(commit == expected_commit, "validator source commit differs")
    _require(not status, "validator source tree is dirty")
    return {"commit": commit, "dirty": False, "branch": run("branch", "--show-current")}


def validate(
    candidate_path: Path,
    archived_path: Path,
    config_path: Path,
    repo_root: Path,
    expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.shadow import load_shadow_config

    candidate = _load(candidate_path, "candidate result")
    archived = _load(archived_path, "archived Table-1 result")
    config = load_shadow_config(config_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "archived file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "archived payload identity differs")
    _require(_payload_hash(archived) == ARCHIVED_PAYLOAD_SHA256, "archived payload hash is invalid")
    _require(candidate.get("result_payload_sha256") == _payload_hash(candidate), "candidate payload hash is invalid")
    for value in (candidate, archived):
        _require(value.get("case_id") == CASE_ID, "case identity differs")
        _require(value.get("arm") == "pi05_plus_aegis_translational", "source arm differs")
        _require(value.get("status") == "complete", "AEGIS rollout is not complete")
        _require(value.get("scientific_result") is True, "AEGIS result is not scientific")
        _require(value.get("task_success") is True, "native SafeLIBERO task did not succeed")
        _require(value["metrics"]["executed_action_count"] == EXPECTED_ACTION_COUNT, "action count differs")
        _require(value["metrics"]["collision_first_step"] == EXPECTED_PAPER_COLLISION_STEP, "paper CAR crossing differs")
        _require(value["contact_telemetry"]["first_contact_step"] == EXPECTED_FIRST_ROBOT_CONTACT_STEP, "first robot contact differs")
        _require(value["goal_progress"]["summary"]["first_all_satisfied_step"] == EXPECTED_TASK_SUCCESS_STEP, "task success step differs")
        _require(value["pairing"]["manifest_row_sha256"] == EXPECTED_MANIFEST_ROW_SHA256, "manifest row differs")
        _require(value["pairing"]["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256, "initial state differs")
        _require(value["pairing"]["policy_noise_schedule_sha256"] == EXPECTED_NOISE_SHA256, "policy noise differs")

    ledger_keys = (
        "nominal_raw_sequence_sha256",
        "nominal_translational_sequence_sha256",
        "executed_sequence_sha256",
        "env_step_input_sequence_sha256",
        "control_path_sequence_sha256",
        "z_before_sequence_sha256",
        "z_after_sequence_sha256",
        "policy_query_returned_action_hash_sequence_sha256",
        "policy_query_rng_seed_sequence_sha256",
        "policy_query_schedule_sha256",
    )
    for key in ledger_keys:
        _require(
            candidate["action_invariance_ledger"].get(key)
            == archived["action_invariance_ledger"].get(key),
            "read-only observer changed action ledger field %s" % key,
        )
    _require(candidate["actions"][187]["robot_obstacle_contact"] is True, "raw contact is absent at action 187")
    _require(candidate["actions"][187]["qp"]["barrier_h"] > 0.0, "released AEGIS barrier is not positive at contact")
    _require(abs(candidate["actions"][187]["qp"]["constraint_lhs"]) <= 1e-10, "released AEGIS QP residual differs at contact")
    contact_geoms = _direct_robot_contact_geoms(candidate)
    _require("robot0_link5_collision" in contact_geoms, "candidate lacks direct physical link-5 contact")
    _require("robot0_link6_collision" in contact_geoms, "candidate lacks direct physical link-6 contact")
    post_contact_actions = EXPECTED_ACTION_COUNT - EXPECTED_PAPER_COLLISION_STEP
    _require(post_contact_actions == 49, "post-collision action count contract differs")
    eef_path = _eef_path_after_contact(candidate)
    _require(eef_path > 0.01, "robot did not continue useful end-effector motion after contact")

    shadow = candidate.get("multilink_ellipsoid_shadow")
    _require(isinstance(shadow, dict), "candidate lacks multi-link shadow output")
    _require(shadow.get("control_effect") == "read_only_no_executed_action_change", "shadow was not read-only")
    _require(shadow.get("step_count") == EXPECTED_ACTION_COUNT, "shadow step count differs")
    _require(shadow.get("status") in ("complete", "qp_failures_present"), "shadow did not terminate explicitly")
    steps = shadow.get("steps")
    _require(isinstance(steps, list) and len(steps) == EXPECTED_ACTION_COUNT, "shadow step records are incomplete")
    expected_bodies = {"robot0_link%d" % index for index in range(1, 8)}
    geometry_rows = shadow["geometry"]["link_ellipsoids"]
    hand_checked_bounds = [_hand_check_bound(item) for item in geometry_rows]
    ellipsoid_count = int(shadow["geometry"]["link_ellipsoid_count"])
    _require(ellipsoid_count == len(hand_checked_bounds), "ellipsoid count differs from live geometry")
    observed_bodies: set[str] = set()
    qp_reasons: dict[str, int] = {}
    for index, step in enumerate(steps):
        _require(step.get("step") == index, "shadow step indexes are not contiguous")
        _require(step.get("constraint_count") == ellipsoid_count, "QP did not receive every live ellipsoid constraint")
        _require(step["qp"]["diagnostics"].get("input_constraint_count") == ellipsoid_count, "solver diagnostics do not prove one simultaneous multi-constraint QP")
        _require(all(len(item["cbf_row_m_per_rad"]) == 7 for item in step["constraints"]), "joint-space constraint row is not seven-dimensional")
        _require(step["D_opt"]["semantics"] == "optimizer_support_gap_buffer", "D_opt semantics differ")
        _require(step["D_sim"]["available"] is True, "raw simulator verification is unavailable")
        _require(isinstance(step["D_sim"]["value"], dict), "raw simulator verification is malformed")
        _require(step["D_sim"]["source"].startswith("post_step_raw_simulator"), "D_sim source differs")
        observed_bodies.update(item["body_name"] for item in step["constraints"])
        reason = str(step["qp"]["reason"])
        qp_reasons[reason] = qp_reasons.get(reason, 0) + 1
        timing = step["qp"]["diagnostics"].get("timing", {})
        for field in ("setup_wall_seconds", "solve_wall_seconds", "total_wall_seconds"):
            value = timing.get(field)
            _require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.0, "QP timing is missing or invalid")
    first_contact_d_sim = steps[EXPECTED_FIRST_ROBOT_CONTACT_STEP]["D_sim"]["value"]
    _require(first_contact_d_sim["robot_contact_count"] > 0, "D_sim misses the first raw robot contact")
    _require(first_contact_d_sim["minimum_robot_contact_distance_m"] <= 0.0, "D_sim first-contact distance is not nonpositive")
    _require(expected_bodies.issubset(observed_bodies), "whole-arm link1-link7 coverage is incomplete")
    geometry_bodies = {item["body_name"] for item in geometry_rows}
    _require(expected_bodies.issubset(geometry_bodies), "initial geometry omits a protected arm link")
    _require(shadow["config"]["config_file_sha256"] == config["config_file_sha256"], "shadow config file identity differs")
    allocation = shadow.get("allocation")
    _require(isinstance(allocation, dict) and "H100" in allocation["device"]["name"], "simulation was not H100-backed")
    _require(allocation.get("slurm_job_id") == os.environ.get("SLURM_JOB_ID"), "producer Slurm identity differs")

    return {
        "schema_version": "vlsa_multilink_ellipsoid_shadow_validation.v1",
        "status": "validated",
        "claim_scope": config["claim_scope"],
        "case_id": CASE_ID,
        "source": _git_identity(repo_root, expected_commit),
        "producer_allocation": allocation,
        "validator_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "candidate": {
            "path": str(candidate_path),
            "file_sha256": _file_sha256(candidate_path),
            "payload_sha256": candidate["result_payload_sha256"],
        },
        "archived_table1": {
            "path": str(archived_path),
            "file_sha256": ARCHIVED_FILE_SHA256,
            "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
            "read_only": True,
        },
        "pairing": {
            "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
            "policy_noise_schedule_sha256": EXPECTED_NOISE_SHA256,
            "executed_sequence_sha256": candidate["action_invariance_ledger"]["executed_sequence_sha256"],
            "action_invariant_with_archived_aegis": True,
        },
        "raw_simulation_evidence": {
            "first_robot_contact_step": EXPECTED_FIRST_ROBOT_CONTACT_STEP,
            "paper_collision_step": EXPECTED_PAPER_COLLISION_STEP,
            "direct_robot_contact_geoms": sorted(contact_geoms),
            "released_aegis_barrier_h_at_first_contact": candidate["actions"][187]["qp"]["barrier_h"],
            "post_collision_executed_action_count": post_contact_actions,
            "post_contact_eef_path_m": eef_path,
            "native_task_success": True,
            "native_task_success_step": EXPECTED_TASK_SUCCESS_STEP,
        },
        "whole_arm_ellipsoid_qp": {
            "protected_body_names": sorted(expected_bodies),
            "ellipsoid_count": shadow["geometry"]["link_ellipsoid_count"],
            "hand_checked_live_bounds": hand_checked_bounds,
            "constraint_count_min": shadow["constraint_count_min"],
            "constraint_count_max": shadow["constraint_count_max"],
            "qp_reason_counts": dict(sorted(qp_reasons.items())),
            "all_qps_valid": shadow["all_qps_valid"],
            "first_nominal_violation": shadow["first_nominal_violation"],
            "minimum_h_opt_m": shadow["minimum_h_opt_m"],
            "qp_total_wall_timing": shadow["qp_total_wall_timing"],
            "qp_solve_wall_timing": shadow["qp_solve_wall_timing"],
            "D_opt": config["optimizer"]["optimizer_clearance_m"],
            "D_sim": config["simulator_verification"]["D_sim"],
            "D_sim_at_first_robot_contact": first_contact_d_sim,
        },
    }


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(
        args.candidate.resolve(),
        args.archived.resolve(),
        args.config.resolve(),
        args.repo_root.resolve(),
        args.expected_commit,
    )
    receipt["receipt_payload_sha256"] = _sha256(_canonical(receipt))
    _atomic_write(args.output.resolve(), receipt)
    print(json.dumps({"status": "validated", "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
