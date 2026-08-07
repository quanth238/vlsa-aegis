#!/usr/bin/env python3
"""Independently validate the paired π0.5-DROID joint-velocity pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


CASE_ID = "vlsa-t1-goal-ii-t0-e05"
ARCHIVED_FILE_SHA256 = "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
PAPER_CAR_THRESHOLD_M = 0.001
PROTECTED_BODIES = {"robot0_link5", "robot0_link6", "robot0_link7"}


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "validation input is missing or symlinked")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "validation input must contain one object")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists(), "validation output already exists")
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode(
        "utf-8"
    ) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git_identity(root: Path, expected_commit: str) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()

    commit = run("rev-parse", "HEAD")
    _require(commit == expected_commit, "validator source commit differs")
    _require(not run("status", "--short"), "validator source tree is dirty")
    return {"commit": commit, "dirty": False, "branch": run("branch", "--show-current")}


def _event_is_protected(event: Mapping[str, Any]) -> bool:
    return event.get("other", {}).get("body_name") in PROTECTED_BODIES


def _validate_arm(
    arm: Mapping[str, Any],
    *,
    arm_name: str,
    candidate_root: Path,
    expected_horizon: int,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    _require(arm["schema_version"] == "vlsa_pi05_droid_joint_velocity_arm.v1", "arm schema differs")
    _require(arm["arm"] == arm_name, "arm name differs")
    _require(arm["status"] == "complete", "arm is not complete")
    actions = arm["actions"]
    _require(isinstance(actions, list) and len(actions) == expected_horizon, "arm horizon differs")
    _require(arm["action_count"] == expected_horizon, "arm action count differs")
    protected_steps = []
    robot_steps = []
    car_steps = []
    success_steps = []
    maximum_displacement = 0.0
    modified_steps = []
    minimum_residual = math.inf
    for index, record in enumerate(actions):
        _require(record["step"] == index, "arm action indexes differ")
        nominal = np.asarray(record["nominal_joint_velocity_clipped_rad_s"], dtype=np.float64)
        executed = np.asarray(record["executed_env_action"], dtype=np.float64)
        _require(nominal.shape == (7,) and executed.shape == (8,), "arm action shape differs")
        _require(np.all(np.isfinite(nominal)) and np.all(np.isfinite(executed)), "arm action is nonfinite")
        _require(record["executed_gripper_command"] == executed[7], "gripper binding differs")
        _require(executed[7] in (-1.0, 1.0), "gripper command is not binary")
        qp = record["multicbf"]
        _require(qp["constraint_count"] == 3, "QP row count differs")
        _require(qp["qp"]["valid"], "QP is invalid")
        residual = qp["qp"]["diagnostics"].get("minimum_raw_residual_m_per_s")
        if residual is not None:
            minimum_residual = min(minimum_residual, float(residual))
            _require(float(residual) >= -5.0e-7, "QP residual check fails")
        if qp["modified"]:
            modified_steps.append(index)
        if arm_name.endswith("baseline"):
            _require(np.array_equal(executed[:7], nominal), "baseline changed the joint velocity")
            _require(qp["applied"] is False, "baseline QP was marked applied")
        else:
            safe = np.asarray(qp["qp"]["executed_safe_joint_velocity_rad_s"], dtype=np.float64)
            _require(np.array_equal(executed[:7], safe), "active QP execution binding differs")
            _require(qp["applied"] is True, "active QP was not marked applied")
        robot_events = record["robot_contact_events"]
        protected_events = record["protected_link_contact_events"]
        _require(
            protected_events == [event for event in robot_events if _event_is_protected(event)],
            "protected contact projection differs",
        )
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
        _require(d_sim["available"], "D_sim is unavailable")
        _require(
            d_sim["value"]["protected_link_contact_count"] == len(protected_events),
            "D_sim protected contact count differs",
        )
        _require(
            math.isclose(
                float(d_sim["value"]["active_obstacle_l1_displacement_m"]),
                displacement,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "D_sim displacement differs",
        )
    raw = arm["raw_simulation_evidence"]
    _require(raw["first_robot_contact_step"] == (robot_steps[0] if robot_steps else None), "first robot contact differs")
    _require(raw["first_protected_link_contact_step"] == (protected_steps[0] if protected_steps else None), "first protected contact differs")
    _require(raw["first_paper_car_step"] == (car_steps[0] if car_steps else None), "first CAR differs")
    _require(raw["native_task_success_step"] == (success_steps[0] if success_steps else None), "first success differs")
    _require(raw["native_task_success"] is bool(success_steps), "native success flag differs")
    _require(raw["paper_car_pass"] is (not car_steps), "paper CAR pass differs")
    _require(raw["protected_link_contact_pass"] is (not protected_steps), "contact pass differs")
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
    _require(summary["all_qps_valid"], "QP summary validity differs")
    _require(summary["material_intervention_count"] == len(modified_steps), "intervention count differs")
    _require(
        summary["first_material_intervention_step"]
        == (modified_steps[0] if modified_steps else None),
        "first intervention differs",
    )
    video_path = candidate_root / arm["video"]["path"]
    _require(video_path.is_file() and not video_path.is_symlink(), "arm video is missing")
    video_sha = _file_sha256(video_path)
    _require(video_sha == arm["video"]["sha256"], "arm video hash differs")
    reader = imageio.get_reader(str(video_path))
    try:
        decoded_frames = int(reader.count_frames())
    finally:
        reader.close()
    _require(decoded_frames == expected_horizon + 1, "arm decoded frame count differs")
    _require(arm["video"]["frames"] == decoded_frames, "arm recorded frame count differs")
    return {
        "action_count": expected_horizon,
        "first_contact_step": protected_steps[0] if protected_steps else None,
        "first_car_step": car_steps[0] if car_steps else None,
        "first_success_step": success_steps[0] if success_steps else None,
        "maximum_displacement_m": maximum_displacement,
        "intervention_count": len(modified_steps),
        "minimum_qp_residual_m_per_s": None if math.isinf(minimum_residual) else minimum_residual,
        "video_sha256": video_sha,
        "decoded_frames": decoded_frames,
    }


def validate(
    *,
    repo_root: Path,
    candidate_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    archived_path: Path,
    producer_commit: str,
    validator_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.joint_velocity import load_joint_velocity_pair_config
    from main.multilink_ellipsoid.shadow import allocation_record
    from scripts.evaluate_pi05_droid_joint_velocity_pair import _checkpoint_tree_record

    validator_source = _git_identity(repo_root, validator_commit)
    validator_allocation = allocation_record()
    candidate = _load(candidate_path)
    _require(candidate["schema_version"] == "vlsa_pi05_droid_joint_velocity_pair_result.v1", "pair schema differs")
    _require(candidate["case_id"] == CASE_ID, "pair case differs")
    _require(candidate["status"] == "complete", "pair result is not complete")
    _require(candidate["source"]["commit"] == producer_commit, "producer commit differs")
    _require(candidate["source"]["dirty"] is False, "producer source was dirty")
    payload_hash = candidate["result_payload_sha256"]
    payload = dict(candidate)
    payload.pop("result_payload_sha256")
    _require(_sha256(_canonical(payload)) == payload_hash, "pair payload hash differs")
    config = load_joint_velocity_pair_config(config_path)
    _require(candidate["config"] == config, "pair config differs")
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "archived Table-1 hash differs")
    _require(
        candidate["archived_table1_geometry_source"]["file_sha256"]
        == ARCHIVED_FILE_SHA256,
        "recorded archived hash differs",
    )
    checkpoint = _checkpoint_tree_record(checkpoint_path)
    _require(checkpoint == candidate["policy_checkpoint"], "checkpoint tree differs")
    expected_horizon = int(config["pairing"]["action_horizon"])
    baseline_name, active_name = config["arms"]
    arms = candidate["arms"]
    _require(set(arms) == {baseline_name, active_name}, "pair arm set differs")
    baseline = _validate_arm(
        arms[baseline_name],
        arm_name=baseline_name,
        candidate_root=candidate_path.parent,
        expected_horizon=expected_horizon,
    )
    active = _validate_arm(
        arms[active_name],
        arm_name=active_name,
        candidate_root=candidate_path.parent,
        expected_horizon=expected_horizon,
    )
    pairing = candidate["pairing_validation"]
    _require(pairing["first_policy_chunks_equal"], "first chunks were not equal")
    _require(pairing["state_equal_before_first_intervention"], "state prefix differs")
    _require(pairing["nominal_actions_equal_before_first_intervention"], "nominal prefix differs")
    first_intervention = arms[active_name]["multicbf_summary"]["first_material_intervention_step"]
    _require(pairing["first_material_intervention_step"] == first_intervention, "paired first intervention differs")
    for index in range(int(first_intervention or expected_horizon)):
        baseline_action = arms[baseline_name]["actions"][index]
        active_action = arms[active_name]["actions"][index]
        _require(
            baseline_action["pre_step_simulator_state_sha256"]
            == active_action["pre_step_simulator_state_sha256"],
            "independent paired state-prefix check fails",
        )
        _require(
            np.array_equal(
                np.asarray(baseline_action["nominal_joint_velocity_clipped_rad_s"]),
                np.asarray(active_action["nominal_joint_velocity_clipped_rad_s"]),
            ),
            "independent nominal-prefix check fails",
        )
    _require(candidate["baseline_competent"] is False, "baseline competence interpretation differs")
    _require(candidate["safe_problem_solved"] is False, "safe result interpretation differs")
    _require(
        candidate["interpretation"] == "baseline_incompetent_no_safety_efficacy_claim",
        "pair interpretation differs",
    )
    return {
        "schema_version": "vlsa_pi05_droid_joint_velocity_pair_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "case_id": CASE_ID,
        "candidate": {
            "path": str(candidate_path),
            "file_sha256": _file_sha256(candidate_path),
            "payload_sha256": payload_hash,
            "producer_commit": producer_commit,
            "producer_allocation": candidate["allocation"],
        },
        "validator_source": validator_source,
        "validator_allocation": validator_allocation,
        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
        "baseline": baseline,
        "active": active,
        "interpretation": candidate["interpretation"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(
        repo_root=args.repo_root.resolve(),
        candidate_path=args.candidate.resolve(),
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        archived_path=args.archived.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    receipt["receipt_payload_sha256"] = _sha256(_canonical(receipt))
    _atomic_write(args.output.resolve(), receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "interpretation": receipt["interpretation"],
                "output": str(args.output.resolve()),
                "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
