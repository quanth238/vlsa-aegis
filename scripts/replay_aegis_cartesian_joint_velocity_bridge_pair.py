#!/usr/bin/env python3
"""Replay released AEGIS Cartesian actions through paired joint controllers."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
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
    _joint_state,
    _load,
    _protected_events,
    _qp_summary,
    _require,
    _sha256,
)


PAIR_RESULT_SCHEMA = "vlsa_aegis_cartesian_joint_velocity_bridge_pair_result.v1"
ARM_RESULT_SCHEMA = "vlsa_aegis_cartesian_joint_velocity_bridge_arm.v1"


def _build_joint_environment(
    *,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    archived: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[Any, Any, Mapping[str, Any], Any, dict[str, Any]]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        _active_obstacle,
        _build_environment,
        _settle,
        array_sha256,
        pairing_record,
    )

    reference_env = None
    try:
        reference_env, reference_task, reference_observation, selected_initial_state = _build_environment(
            runtime,
            case,
            render_resolution=TABLE_RENDER_RESOLUTION,
        )
        reference_observation = _settle(
            reference_env,
            reference_observation,
            int(config["pairing"]["settle_actions"]),
        )
        reference_obstacle, _ = _active_obstacle(reference_env, reference_observation)
        reference_state = np.asarray(
            reference_env.sim.get_state().flatten(), dtype=np.float64
        ).copy()
        reference_pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=reference_observation,
            task_description=str(reference_task.language),
            active_obstacle_name=reference_obstacle,
            settled_simulator_state=reference_state,
        )
        for key in (
            "manifest_row_sha256",
            "initial_state_sha256",
            "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(
                reference_pairing[key] == archived["pairing"][key],
                "OSC reference pairing differs: %s" % key,
            )
    finally:
        if reference_env is not None:
            reference_env.close()

    joint_env, task, _, joint_initial_state = _build_environment(
        runtime,
        case,
        render_resolution=TABLE_RENDER_RESOLUTION,
        controller_configs=config["controller"],
        control_frequency_hz=int(config["pairing"]["control_frequency_hz"]),
        ignore_done=True,
    )
    _require(
        np.array_equal(np.asarray(joint_initial_state), np.asarray(selected_initial_state)),
        "joint-controller selected initial state differs",
    )
    observation = joint_env.regenerate_obs_from_state(reference_state)
    transplanted = np.asarray(joint_env.sim.get_state().flatten(), dtype=np.float64)
    _require(np.array_equal(transplanted, reference_state), "settled-state transplant differs")
    obstacle_name, _ = _active_obstacle(joint_env, observation)
    _require(obstacle_name == reference_obstacle, "joint-controller obstacle differs")
    joint_pairing = pairing_record(
        case=case,
        selected_initial_state=joint_initial_state,
        settled_observation=observation,
        task_description=str(task.language),
        active_obstacle_name=obstacle_name,
        settled_simulator_state=transplanted,
    )
    for key in (
        "manifest_row_sha256",
        "initial_state_sha256",
        "settled_simulator_state_sha256",
        "settled_active_obstacle_position_sha256",
        "policy_noise_schedule_sha256",
    ):
        _require(joint_pairing[key] == archived["pairing"][key], "joint pairing differs: %s" % key)
    _require(
        joint_pairing["initial_observation_contract"]["state_array_sha256"]
        == archived["pairing"]["initial_observation_contract"]["state_array_sha256"],
        "joint-controller proprioceptive state differs",
    )
    observation_contract_differences = sorted(
        key
        for key, value in joint_pairing["initial_observation_contract"].items()
        if value != archived["pairing"]["initial_observation_contract"].get(key)
    )
    action_relevant_pairing = {
        "manifest_row_sha256": joint_pairing["manifest_row_sha256"],
        "initial_state_sha256": joint_pairing["initial_state_sha256"],
        "settled_simulator_state_sha256": joint_pairing["settled_simulator_state_sha256"],
        "settled_active_obstacle_position_sha256": joint_pairing[
            "settled_active_obstacle_position_sha256"
        ],
        "state_array_sha256": joint_pairing["initial_observation_contract"][
            "state_array_sha256"
        ],
        "archived_executed_sequence_sha256": archived["action_invariance_ledger"][
            "executed_sequence_sha256"
        ],
        "controller_config_sha256": _sha256(_canonical(config["controller"])),
        "control_frequency_hz": int(config["pairing"]["control_frequency_hz"]),
        "action_horizon": int(config["pairing"]["action_horizon"]),
    }
    return joint_env, task, observation, joint_initial_state, {
        "schema_version": "vlsa_aegis_cartesian_joint_velocity_bridge_pairing.v1",
        "archived_pairing_fields_verified": True,
        "settled_simulator_state_sha256": array_sha256(transplanted),
        "settled_state_transplanted_from_original_osc": True,
        "controller_config_sha256": action_relevant_pairing["controller_config_sha256"],
        "control_frequency_hz": int(config["pairing"]["control_frequency_hz"]),
        "action_horizon": int(config["pairing"]["action_horizon"]),
        "archived_observation_contract_difference_fields": observation_contract_differences,
        "rerendered_images_are_not_policy_inputs": True,
        "pairing_payload_sha256": _sha256(_canonical(action_relevant_pairing)),
    }


def _run_arm(
    *,
    arm: str,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    archived: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        _active_obstacle,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _processed_image,
        array_sha256,
    )
    from main.multilink_ellipsoid.cartesian_bridge import (
        AegisCartesianJointVelocityBridge,
        build_joint_velocity_filter,
    )

    active = arm == config["arms"][1]
    _require(active or arm == config["arms"][0], "unknown Cartesian bridge arm")
    arm_root = output_root / "arms" / arm
    arm_root.mkdir(parents=True, exist_ok=False)
    partial_video = arm_root / "episode.partial.mp4"
    final_video = arm_root / "episode.mp4"
    env = None
    writer = None
    frames = 0
    started = time.perf_counter_ns()
    result: dict[str, Any]
    try:
        env, task, observation, _, pairing = _build_joint_environment(
            runtime=runtime,
            case=case,
            archived=archived,
            config=config,
        )
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        perception = archived["perception"]
        geometry_source = {
            "p2": perception["mvee_center"],
            "R2": perception["mvee_rotation"],
            "Q2_diag": perception["mvee_semiaxes"],
            "record": {"label": perception["obstacle_label"]},
        }
        bridge = AegisCartesianJointVelocityBridge(config)
        filter_controller = build_joint_velocity_filter(config, geometry_source)
        geometry = filter_controller.geometry_record(env)
        _require(geometry["link_ellipsoid_count"] == 3, "bridge geometry count differs")
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(env, goal_atoms, step=-1, previous_values=None)
        previous_goal_values = initial_goal["values"]
        contact_authority = _contact_model_authority(env, obstacle_name)
        writer = runtime["imageio"].get_writer(
            str(partial_video),
            fps=int(config["pairing"]["control_frequency_hz"]),
            codec="libx264",
            macro_block_size=None,
        )
        writer.append_data(_processed_image(observation, "agentview_image"))
        frames += 1

        action_records = []
        qp_records = []
        direct_robot_geoms: set[str] = set()
        direct_protected_geoms: set[str] = set()
        first_robot_contact_step = None
        first_protected_contact_step = None
        first_car_step = None
        first_success_step = None
        maximum_displacement = 0.0
        failure = None
        archived_actions = archived["actions"]
        horizon = int(config["pairing"]["action_horizon"])
        _require(len(archived_actions) == horizon, "archived bridge horizon differs")
        for step, archived_action in enumerate(archived_actions):
            _require(int(archived_action["step"]) == step, "archived action indexes differ")
            source_action = np.asarray(archived_action["env_step_input"], dtype=np.float64)
            _require(
                source_action.shape == (7,)
                and np.array_equal(source_action, np.asarray(archived_action["executed"])),
                "archived AEGIS execution binding differs",
            )
            pre_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            current_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            nominal_qdot, bridge_record = bridge.nominal(
                env,
                current_eef,
                source_action,
                step=step,
            )
            safe_qdot, qp_record = filter_controller.filter(env, nominal_qdot, step=step)
            qp_record["applied"] = active
            qp_records.append(qp_record)
            if active:
                if safe_qdot is None:
                    failure = {
                        "component": "aegis_cartesian_joint_velocity_l5_l7_multicbf",
                        "step": step,
                        "reason": qp_record["qp"]["reason"],
                    }
                    break
                executed_qdot = np.asarray(safe_qdot, dtype=np.float64)
            else:
                executed_qdot = np.asarray(nominal_qdot, dtype=np.float64)
            env_action = np.concatenate((executed_qdot, [source_action[6]]))
            pre_qpos, pre_qvel = _joint_state(env)
            step_started = time.perf_counter_ns()
            observation, reward, done, _ = env.step(env_action)
            step_wall = (time.perf_counter_ns() - step_started) * 1.0e-9
            post_qpos, post_qvel = _joint_state(env)
            goal = _goal_progress_snapshot(
                env,
                goal_atoms,
                step=step,
                previous_values=previous_goal_values,
            )
            previous_goal_values = goal["values"]
            if goal["all_satisfied"] and first_success_step is None:
                first_success_step = step
            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=step,
                contact_authority=contact_authority,
            )
            _require(contacts["status"] == "available", "raw contact evidence unavailable")
            robot_events = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            protected_events = _protected_events(robot_events)
            if robot_events and first_robot_contact_step is None:
                first_robot_contact_step = step
            if protected_events and first_protected_contact_step is None:
                first_protected_contact_step = step
            for event in robot_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_robot_geoms.add(name)
            for event in protected_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_protected_geoms.add(name)
            displacement = float(
                np.sum(
                    np.abs(
                        np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
                        - initial_obstacle_position
                    )
                )
            )
            maximum_displacement = max(maximum_displacement, displacement)
            if first_car_step is None and displacement > PAPER_CAR_THRESHOLD_M:
                first_car_step = step
            protected_distances = [float(event["distance"]) for event in protected_events]
            qp_record["D_sim"] = {
                "available": True,
                "value": {
                    "minimum_protected_link_contact_distance_m": (
                        min(protected_distances) if protected_distances else None
                    ),
                    "active_obstacle_l1_displacement_m": displacement,
                    "protected_link_contact_count": len(protected_events),
                },
                "semantics": config["simulator_verification"]["D_sim"],
                "source": "post_step_raw_simulator_cartesian_joint_velocity_bridge",
            }
            post_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            action_records.append(
                {
                    "step": step,
                    "archived_aegis_action_sha256": _sha256(_canonical(source_action.tolist())),
                    "archived_aegis_action": source_action.tolist(),
                    "pre_step_simulator_state_sha256": array_sha256(pre_state),
                    "post_step_simulator_state_sha256": array_sha256(post_state),
                    "bridge": bridge_record,
                    "nominal_joint_velocity_rad_s": list(nominal_qdot),
                    "executed_joint_velocity_rad_s": executed_qdot.tolist(),
                    "executed_gripper_command": float(source_action[6]),
                    "executed_env_action": env_action.tolist(),
                    "pre_joint_position_rad": pre_qpos,
                    "pre_joint_velocity_rad_s": pre_qvel,
                    "post_joint_position_rad": post_qpos,
                    "post_joint_velocity_rad_s": post_qvel,
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                    "active_obstacle_l1_displacement_m": displacement,
                    "robot_contact_events": robot_events,
                    "protected_link_contact_events": protected_events,
                    "multicbf": qp_record,
                    "env_step_wall_seconds": step_wall,
                }
            )
            writer.append_data(_processed_image(observation, "agentview_image"))
            frames += 1

        goal_summary = _goal_progress_summary(initial_goal, action_records)
        _require(
            goal_summary["first_all_satisfied_step"] == first_success_step,
            "bridge goal summary differs",
        )
        status = "complete" if failure is None and len(action_records) == horizon else "method_failure"
        result = {
            "schema_version": ARM_RESULT_SCHEMA,
            "status": status,
            "arm": arm,
            "case_id": CASE_ID,
            "control_effect": (
                "direct_joint_velocity_with_l5_l6_l7_multicbf"
                if active
                else "direct_joint_velocity_bridge_only"
            ),
            "pairing": pairing,
            "geometry": geometry,
            "action_count": len(action_records),
            "actions": action_records,
            "multicbf_summary": _qp_summary(qp_records),
            "goal_progress": {
                **goal_definition,
                "initial": initial_goal,
                "summary": goal_summary,
            },
            "raw_simulation_evidence": {
                "first_robot_contact_step": first_robot_contact_step,
                "first_protected_link_contact_step": first_protected_contact_step,
                "direct_robot_contact_geoms": sorted(direct_robot_geoms),
                "direct_protected_link_contact_geoms": sorted(direct_protected_geoms),
                "first_paper_car_step": first_car_step,
                "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
                "paper_car_pass": first_car_step is None,
                "protected_link_contact_pass": first_protected_contact_step is None,
                "native_task_success": first_success_step is not None,
                "native_task_success_step": first_success_step,
            },
            "failure": failure,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if env is not None:
                env.close()
    _require(partial_video.is_file() and partial_video.stat().st_size > 0, "bridge video missing")
    os.replace(partial_video, final_video)
    result["video"] = {
        "path": str(final_video.relative_to(output_root)),
        "sha256": _file_sha256(final_video),
        "frames": frames,
        "fps": int(config["pairing"]["control_frequency_hz"]),
        "complete_horizon": frames == int(config["pairing"]["action_horizon"]) + 1,
    }
    result["result_payload_sha256"] = _sha256(_canonical(result))
    return result


def evaluate_pair(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    config_path: Path,
    expected_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import _runtime_imports, read_jsonl, validate_case_row
    from main.multilink_ellipsoid.cartesian_bridge import load_cartesian_bridge_config
    from main.multilink_ellipsoid.shadow import allocation_record

    _require(not output_path.exists(), "bridge pair result already exists")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    config = load_cartesian_bridge_config(config_path)
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "archived Table-1 hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "archived payload differs")
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary bridge manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=False)
    output_root = output_path.parent
    baseline = _run_arm(
        arm=config["arms"][0],
        runtime=runtime,
        case=case,
        config=config,
        archived=archived,
        output_root=output_root,
    )
    active = _run_arm(
        arm=config["arms"][1],
        runtime=runtime,
        case=case,
        config=config,
        archived=archived,
        output_root=output_root,
    )
    _require(
        baseline["pairing"]["pairing_payload_sha256"]
        == active["pairing"]["pairing_payload_sha256"],
        "bridge settled inputs differ",
    )
    first_intervention = active["multicbf_summary"]["first_material_intervention_step"]
    prefix_count = (
        min(baseline["action_count"], active["action_count"])
        if first_intervention is None
        else int(first_intervention) + 1
    )
    state_prefix_equal = True
    nominal_prefix_equal = True
    for index in range(prefix_count):
        left = baseline["actions"][index]
        right = active["actions"][index]
        state_prefix_equal = state_prefix_equal and (
            left["pre_step_simulator_state_sha256"]
            == right["pre_step_simulator_state_sha256"]
        )
        nominal_prefix_equal = nominal_prefix_equal and np.array_equal(
            np.asarray(left["nominal_joint_velocity_rad_s"]),
            np.asarray(right["nominal_joint_velocity_rad_s"]),
        )
    _require(state_prefix_equal, "bridge state differs before first intervention")
    _require(nominal_prefix_equal, "bridge nominal differs before first intervention")
    baseline_raw = baseline["raw_simulation_evidence"]
    active_raw = active["raw_simulation_evidence"]
    baseline_competent = bool(
        baseline["status"] == "complete"
        and baseline_raw["native_task_success"]
        and not baseline_raw["protected_link_contact_pass"]
        and not baseline_raw["paper_car_pass"]
    )
    safe_problem_solved = bool(
        baseline_competent
        and active["status"] == "complete"
        and active_raw["native_task_success"]
        and active_raw["protected_link_contact_pass"]
        and active_raw["paper_car_pass"]
    )
    if not baseline_competent:
        interpretation = "bridge_incompetent_no_safety_efficacy_claim"
    elif safe_problem_solved:
        interpretation = "primary_direct_joint_velocity_problem_solved"
    elif active["status"] != "complete":
        interpretation = "active_method_failure"
    else:
        interpretation = "active_direct_joint_velocity_safety_or_task_failure"
    result = {
        "schema_version": PAIR_RESULT_SCHEMA,
        "status": (
            "complete"
            if baseline["status"] == "complete" and active["status"] == "complete"
            else "method_failure"
        ),
        "case_id": CASE_ID,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation,
        "config": config,
        "archived_table1": {
            "path": str(archived_path),
            "file_sha256": ARCHIVED_FILE_SHA256,
            "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
            "executed_sequence_sha256": archived["action_invariance_ledger"]["executed_sequence_sha256"],
            "read_only": True,
        },
        "pairing_validation": {
            "settled_pairing_payload_sha256": baseline["pairing"]["pairing_payload_sha256"],
            "first_material_intervention_step": first_intervention,
            "equal_pre_state_prefix_action_count": prefix_count,
            "state_equal_through_first_intervention_pre_state": state_prefix_equal,
            "nominal_joint_velocities_equal_through_first_intervention": nominal_prefix_equal,
            "common_action_horizon": int(config["pairing"]["action_horizon"]),
        },
        "arms": {baseline["arm"]: baseline, active["arm"]: active},
        "baseline_competent": baseline_competent,
        "safe_problem_solved": safe_problem_solved,
        "interpretation": interpretation,
    }
    result["result_payload_sha256"] = _sha256(_canonical(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_pair(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "baseline_competent": result["baseline_competent"],
                "safe_problem_solved": result["safe_problem_solved"],
                "interpretation": result["interpretation"],
                "output": str(args.output.resolve()),
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
