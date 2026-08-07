#!/usr/bin/env python3
"""Independently validate the AEGIS Cartesian joint-velocity bridge pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluate_pi05_droid_joint_velocity_pair import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    PAPER_CAR_THRESHOLD_M,
    _atomic_write,
    _canonical,
    _file_sha256,
    _git_identity,
    _load,
    _protected_events,
    _require,
    _sha256,
)
from scripts.replay_aegis_cartesian_joint_velocity_bridge_pair import (
    ARM_RESULT_SCHEMA,
    PAIR_RESULT_SCHEMA,
)


def _validate_arm(
    arm: Mapping[str, Any],
    *,
    arm_name: str,
    active: bool,
    archived: Mapping[str, Any],
    candidate_root: Path,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    _require(arm["schema_version"] == ARM_RESULT_SCHEMA, "bridge arm schema differs")
    _require(arm["arm"] == arm_name and arm["status"] == "complete", "bridge arm differs")
    actions = arm["actions"]
    _require(len(actions) == len(archived["actions"]) == 237, "bridge arm horizon differs")
    protected_steps = []
    robot_steps = []
    car_steps = []
    success_steps = []
    modified_steps = []
    maximum_displacement = 0.0
    minimum_residual = math.inf
    for index, (record, archived_action) in enumerate(zip(actions, archived["actions"])):
        _require(record["step"] == archived_action["step"] == index, "bridge action index differs")
        source = np.asarray(record["archived_aegis_action"], dtype=np.float64)
        _require(
            np.array_equal(source, np.asarray(archived_action["env_step_input"], dtype=np.float64)),
            "bridge archived action differs",
        )
        nominal = np.asarray(record["nominal_joint_velocity_rad_s"], dtype=np.float64)
        executed = np.asarray(record["executed_env_action"], dtype=np.float64)
        _require(nominal.shape == (7,) and executed.shape == (8,), "bridge action shape differs")
        _require(np.all(np.isfinite(nominal)) and np.all(np.isfinite(executed)), "bridge action nonfinite")
        _require(executed[7] == source[6] == record["executed_gripper_command"], "gripper changed")
        qp = record["multicbf"]
        _require(qp["constraint_count"] == 3 and qp["qp"]["valid"], "bridge QP differs")
        residual = qp["qp"]["diagnostics"].get("minimum_raw_residual_m_per_s")
        if residual is not None:
            residual = float(residual)
            minimum_residual = min(minimum_residual, residual)
            _require(residual >= -5.0e-7, "bridge QP residual fails")
        if qp["modified"]:
            modified_steps.append(index)
        if active:
            safe = np.asarray(qp["qp"]["executed_safe_joint_velocity_rad_s"], dtype=np.float64)
            _require(np.array_equal(executed[:7], safe), "active QP was not executed")
            _require(qp["applied"] is True, "active QP application flag differs")
        else:
            _require(np.array_equal(executed[:7], nominal), "bridge-only arm changed nominal qdot")
            _require(qp["applied"] is False, "bridge-only QP application flag differs")
        robot_events = record["robot_contact_events"]
        protected_events = record["protected_link_contact_events"]
        _require(protected_events == _protected_events(robot_events), "protected contact projection differs")
        if robot_events:
            robot_steps.append(index)
        if protected_events:
            protected_steps.append(index)
        displacement = float(record["active_obstacle_l1_displacement_m"])
        maximum_displacement = max(maximum_displacement, displacement)
        if displacement > PAPER_CAR_THRESHOLD_M:
            car_steps.append(index)
        if record["goal_progress"]["all_satisfied"]:
            success_steps.append(index)
        d_sim = qp["D_sim"]
        _require(d_sim["available"], "bridge D_sim unavailable")
        _require(
            d_sim["value"]["protected_link_contact_count"] == len(protected_events),
            "bridge D_sim contact count differs",
        )
        _require(
            math.isclose(
                float(d_sim["value"]["active_obstacle_l1_displacement_m"]),
                displacement,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "bridge D_sim displacement differs",
        )
    raw = arm["raw_simulation_evidence"]
    _require(raw["first_robot_contact_step"] == (robot_steps[0] if robot_steps else None), "robot contact summary differs")
    _require(raw["first_protected_link_contact_step"] == (protected_steps[0] if protected_steps else None), "protected contact summary differs")
    _require(raw["first_paper_car_step"] == (car_steps[0] if car_steps else None), "CAR summary differs")
    _require(raw["native_task_success_step"] == (success_steps[0] if success_steps else None), "task summary differs")
    _require(raw["native_task_success"] is bool(success_steps), "task flag differs")
    _require(raw["paper_car_pass"] is (not car_steps), "CAR flag differs")
    _require(raw["protected_link_contact_pass"] is (not protected_steps), "contact flag differs")
    _require(
        math.isclose(
            float(raw["maximum_active_obstacle_l1_displacement_m"]),
            maximum_displacement,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "maximum displacement differs",
    )
    summary = arm["multicbf_summary"]
    _require(summary["all_qps_valid"] and summary["valid_qp_count"] == 237, "QP summary differs")
    _require(summary["material_intervention_count"] == len(modified_steps), "QP modification summary differs")
    video_path = candidate_root / arm["video"]["path"]
    _require(video_path.is_file() and not video_path.is_symlink(), "bridge video missing")
    video_sha = _file_sha256(video_path)
    _require(video_sha == arm["video"]["sha256"], "bridge video hash differs")
    reader = imageio.get_reader(str(video_path))
    try:
        frames = int(reader.count_frames())
    finally:
        reader.close()
    _require(frames == 238 == arm["video"]["frames"], "bridge video frame count differs")
    return {
        "action_count": 237,
        "first_robot_contact_step": robot_steps[0] if robot_steps else None,
        "first_protected_contact_step": protected_steps[0] if protected_steps else None,
        "first_car_step": car_steps[0] if car_steps else None,
        "first_success_step": success_steps[0] if success_steps else None,
        "maximum_displacement_m": maximum_displacement,
        "intervention_count": len(modified_steps),
        "minimum_qp_residual_m_per_s": (
            None if math.isinf(minimum_residual) else minimum_residual
        ),
        "video_sha256": video_sha,
        "decoded_frames": frames,
    }


def validate(
    *,
    repo_root: Path,
    candidate_path: Path,
    archived_path: Path,
    config_path: Path,
    producer_commit: str,
    validator_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.cartesian_bridge import load_cartesian_bridge_config
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, validator_commit)
    allocation = allocation_record()
    candidate = _load(candidate_path)
    archived = _load(archived_path)
    _require(candidate["schema_version"] == PAIR_RESULT_SCHEMA, "bridge pair schema differs")
    _require(candidate["case_id"] == CASE_ID and candidate["status"] == "complete", "bridge pair differs")
    _require(candidate["source"]["commit"] == producer_commit, "bridge producer commit differs")
    payload_hash = candidate["result_payload_sha256"]
    payload = dict(candidate)
    payload.pop("result_payload_sha256")
    _require(_sha256(_canonical(payload)) == payload_hash, "bridge result payload hash differs")
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "archived file hash differs")
    _require(archived["result_payload_sha256"] == ARCHIVED_PAYLOAD_SHA256, "archived payload differs")
    config = load_cartesian_bridge_config(config_path)
    _require(candidate["config"] == config, "bridge config differs")
    baseline_name, active_name = config["arms"]
    arms = candidate["arms"]
    _require(set(arms) == {baseline_name, active_name}, "bridge arm set differs")
    baseline = _validate_arm(
        arms[baseline_name],
        arm_name=baseline_name,
        active=False,
        archived=archived,
        candidate_root=candidate_path.parent,
    )
    active = _validate_arm(
        arms[active_name],
        arm_name=active_name,
        active=True,
        archived=archived,
        candidate_root=candidate_path.parent,
    )
    first_intervention = candidate["pairing_validation"]["first_material_intervention_step"]
    for index in range(int(first_intervention) + 1):
        left = arms[baseline_name]["actions"][index]
        right = arms[active_name]["actions"][index]
        _require(
            left["pre_step_simulator_state_sha256"] == right["pre_step_simulator_state_sha256"],
            "independent bridge state-prefix check fails",
        )
        _require(
            np.array_equal(
                np.asarray(left["nominal_joint_velocity_rad_s"]),
                np.asarray(right["nominal_joint_velocity_rad_s"]),
            ),
            "independent bridge nominal-prefix check fails",
        )
    _require(candidate["baseline_competent"] is False, "bridge competence interpretation differs")
    _require(candidate["safe_problem_solved"] is False, "bridge solved interpretation differs")
    _require(candidate["interpretation"] == "bridge_incompetent_no_safety_efficacy_claim", "bridge interpretation differs")
    return {
        "schema_version": "vlsa_aegis_cartesian_joint_velocity_bridge_validation.v1",
        "status": "validated",
        "case_id": CASE_ID,
        "candidate": {
            "path": str(candidate_path),
            "file_sha256": _file_sha256(candidate_path),
            "payload_sha256": payload_hash,
            "producer_commit": producer_commit,
            "producer_allocation": candidate["allocation"],
        },
        "validator_source": source,
        "validator_allocation": allocation,
        "baseline": baseline,
        "active": active,
        "interpretation": candidate["interpretation"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(
        repo_root=args.repo_root.resolve(),
        candidate_path=args.candidate.resolve(),
        archived_path=args.archived.resolve(),
        config_path=args.config.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    receipt["receipt_payload_sha256"] = _sha256(_canonical(receipt))
    _atomic_write(args.output.resolve(), receipt)
    print(json.dumps({
        "status": receipt["status"],
        "interpretation": receipt["interpretation"],
        "output": str(args.output.resolve()),
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
