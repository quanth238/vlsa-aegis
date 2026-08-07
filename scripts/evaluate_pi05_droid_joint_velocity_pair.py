#!/usr/bin/env python3
"""Run paired π0.5-DROID joint-velocity baseline and L5-L7 multi-CBF arms."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence


CASE_ID = "vlsa-t1-goal-ii-t0-e05"
ARCHIVED_FILE_SHA256 = "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
ARCHIVED_PAYLOAD_SHA256 = "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c"
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
    _require(path.is_file() and not path.is_symlink(), "JSON input is missing or symlinked")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON input must contain one object")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists(), "result output already exists")
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
    _require(commit == expected_commit, "joint-velocity source commit differs")
    _require(not run("status", "--short"), "joint-velocity source tree is dirty")
    return {
        "commit": commit,
        "dirty": False,
        "branch": run("branch", "--show-current"),
    }


def _checkpoint_tree_record(root: Path) -> dict[str, Any]:
    _require(root.is_dir() and not root.is_symlink(), "DROID checkpoint is missing or symlinked")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    _require(files, "DROID checkpoint contains no files")
    records = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in files:
        _require(not path.is_symlink(), "DROID checkpoint contains a symlink")
        before = path.stat()
        digest = _file_sha256(path)
        after = path.stat()
        identity_before = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        _require(identity_before == identity_after, "DROID checkpoint changed while hashing")
        relative = path.relative_to(root).as_posix()
        size = int(after.st_size)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        total_bytes += size
        records.append({"relative_path": relative, "bytes": size, "sha256": digest})
    return {
        "path": str(root.resolve()),
        "format": "jax_orbax_ocdbt",
        "file_count": len(records),
        "total_bytes": total_bytes,
        "tree_sha256": aggregate.hexdigest(),
        "tree_hash_algorithm": "sha256(path_nul_size_nul_file_sha256_newline)",
        "files": records,
    }


def _protected_events(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        event
        for event in events
        if event.get("other", {}).get("body_name") in PROTECTED_BODIES
    ]


def _joint_state(env: Any) -> tuple[list[float], list[float]]:
    import numpy as np

    from main.multilink_ellipsoid.shadow import _arm_dof_indices, _raw_model_data

    model, data = _raw_model_data(env.sim)
    dofs = _arm_dof_indices(env)
    qpos = []
    for dof_id in dofs:
        joint_id = int(model.dof_jntid[int(dof_id)])
        qpos.append(float(data.qpos[int(model.jnt_qposadr[joint_id])]))
    qvel = np.asarray(data.qvel, dtype=np.float64)[list(dofs)].tolist()
    return qpos, qvel


def _gripper_closed_fraction(observation: Mapping[str, Any]) -> float:
    import numpy as np

    qpos = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64)
    _require(qpos.ndim == 1 and qpos.size == 2 and np.all(np.isfinite(qpos)), "gripper state differs")
    return float(np.clip(1.0 - np.mean(np.abs(qpos)) / 0.04, 0.0, 1.0))


def _policy_input(
    runtime: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    prompt: str,
    rng_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    from main.evaluate_safelibero_aegis import _processed_image, array_sha256

    image_tools = runtime["image_tools"]
    external = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            _processed_image(observation, "agentview_image"), 224, 224
        )
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            _processed_image(observation, "robot0_eye_in_hand_image"), 224, 224
        )
    )
    joint_position = np.asarray(observation["robot0_joint_pos"], dtype=np.float64)
    _require(
        joint_position.shape == (7,) and np.all(np.isfinite(joint_position)),
        "Panda joint-position observation differs",
    )
    gripper_position = np.asarray(
        [_gripper_closed_fraction(observation)], dtype=np.float64
    )
    policy_input = {
        "observation/exterior_image_1_left": external,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": str(prompt),
        "__crfs__": {"rng_seed": int(rng_seed)},
    }
    return policy_input, {
        "external_image_sha256": array_sha256(external),
        "wrist_image_sha256": array_sha256(wrist),
        "joint_position_sha256": array_sha256(joint_position),
        "gripper_position_sha256": array_sha256(gripper_position),
        "prompt": str(prompt),
    }


def _pairing_record(
    *,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    selected_initial_state: Any,
    observation: Mapping[str, Any],
    settled_state: Any,
    task_description: str,
    obstacle_name: str,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import _processed_image, array_sha256

    query_count = int(math.ceil(config["pairing"]["action_horizon"] / config["pairing"]["open_loop_execution_steps"]))
    base_seed = int(case["policy_noise_seed"])
    payload = {
        "schema_version": "vlsa_pi05_droid_joint_velocity_pairing.v1",
        "manifest_row_sha256": _sha256(_canonical(case)),
        "selected_initial_state_sha256": array_sha256(selected_initial_state),
        "settled_simulator_state_sha256": array_sha256(settled_state),
        "agentview_sha256": array_sha256(_processed_image(observation, "agentview_image")),
        "wrist_sha256": array_sha256(
            _processed_image(observation, "robot0_eye_in_hand_image")
        ),
        "joint_position_sha256": array_sha256(
            np.asarray(observation["robot0_joint_pos"], dtype=np.float64)
        ),
        "gripper_closed_fraction": _gripper_closed_fraction(observation),
        "active_obstacle_name": obstacle_name,
        "active_obstacle_position_sha256": array_sha256(
            np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
        ),
        "prompt": str(task_description),
        "policy_noise_schedule": {
            "base_seed": base_seed,
            "query_count": query_count,
            "query_seeds": [base_seed + index for index in range(query_count)],
        },
        "controller_config_sha256": _sha256(_canonical(config["controller"])),
        "action_horizon": int(config["pairing"]["action_horizon"]),
        "control_frequency_hz": int(config["pairing"]["control_frequency_hz"]),
    }
    payload["pairing_payload_sha256"] = _sha256(_canonical(payload))
    return payload


def _settle_joint_velocity(env: Any, observation: Mapping[str, Any], count: int) -> Mapping[str, Any]:
    import numpy as np

    action = np.zeros(8, dtype=np.float64)
    action[-1] = -1.0
    current = observation
    for _ in range(count):
        current, _, _, _ = env.step(action)
    return current


def _qp_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if not records:
        return {"step_count": 0, "all_qps_valid": False}
    valid = [bool(item["qp"]["valid"]) for item in records]
    modified = [item for item in records if item["modified"]]
    total = np.asarray(
        [item["timing"]["total_filter_wall_seconds"] for item in records],
        dtype=np.float64,
    )
    solver = np.asarray(
        [
            item["qp"]["diagnostics"]["timing"]["total_wall_seconds"]
            for item in records
        ],
        dtype=np.float64,
    )
    return {
        "step_count": len(records),
        "all_qps_valid": all(valid),
        "valid_qp_count": sum(valid),
        "material_intervention_count": len(modified),
        "first_material_intervention_step": (
            None if not modified else int(modified[0]["step"])
        ),
        "maximum_correction_l2_rad_s": max(
            float(item["active_correction_l2_rad_s"] or 0.0) for item in records
        ),
        "minimum_h_opt_m": min(float(item["minimum_h_opt_m"]) for item in records),
        "total_filter_timing": {
            "mean_seconds": float(np.mean(total)),
            "p95_seconds": float(np.quantile(total, 0.95)),
            "maximum_seconds": float(np.max(total)),
        },
        "qp_timing": {
            "mean_seconds": float(np.mean(solver)),
            "p95_seconds": float(np.quantile(solver, 0.95)),
            "maximum_seconds": float(np.max(solver)),
        },
    }


def _run_arm(
    *,
    arm: str,
    runtime: Mapping[str, Any],
    client: Any,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    archived: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        _active_obstacle,
        _build_environment,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _processed_image,
        array_sha256,
    )
    from main.multilink_ellipsoid.joint_velocity import DistalThreeJointVelocityCbf

    active = arm == "pi05_droid_joint_velocity_l5_l7_multicbf"
    _require(
        active or arm == "pi05_droid_joint_velocity_baseline",
        "unknown joint-velocity arm",
    )
    arm_root = output_root / "arms" / arm
    arm_root.mkdir(parents=True, exist_ok=False)
    partial_video = arm_root / "episode.partial.mp4"
    final_video = arm_root / "episode.mp4"
    env = None
    writer = None
    frames = 0
    started = time.perf_counter_ns()
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime,
            case,
            render_resolution=TABLE_RENDER_RESOLUTION,
            controller_configs=config["controller"],
            control_frequency_hz=int(config["pairing"]["control_frequency_hz"]),
            ignore_done=True,
        )
        observation = _settle_joint_velocity(
            env, observation, int(config["pairing"]["settle_actions"])
        )
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(obstacle_name == archived["obstacle"]["active_name"], "active obstacle differs")
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        pairing = _pairing_record(
            case=case,
            config=config,
            selected_initial_state=selected_initial_state,
            observation=observation,
            settled_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
            task_description=str(task.language),
            obstacle_name=obstacle_name,
        )
        perception = archived["perception"]
        filter_controller = DistalThreeJointVelocityCbf.from_aegis_geometry(
            config,
            {
                "p2": perception["mvee_center"],
                "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        geometry = filter_controller.geometry_record(env)
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env, goal_atoms, step=-1, previous_values=None
        )
        previous_goal_values = initial_goal["values"]
        contact_authority = _contact_model_authority(env, obstacle_name)
        writer = runtime["imageio"].get_writer(
            str(partial_video), fps=15, codec="libx264", macro_block_size=None
        )
        initial_frame = _processed_image(observation, "agentview_image")
        writer.append_data(initial_frame)
        frames += 1

        action_plan: collections.deque[tuple[int, int, Any]] = collections.deque()
        policy_queries = []
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
        horizon = int(config["pairing"]["action_horizon"])
        execute_count = int(config["pairing"]["open_loop_execution_steps"])
        expected_chunk_shape = (
            int(config["pairing"]["model_action_horizon"]),
            8,
        )
        for step in range(horizon):
            pre_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            if not action_plan:
                query_index = len(policy_queries)
                rng_seed = int(case["policy_noise_seed"]) + query_index
                policy_input, input_record = _policy_input(
                    runtime,
                    observation,
                    prompt=str(task.language),
                    rng_seed=rng_seed,
                )
                query_started = time.perf_counter_ns()
                response = client.infer(policy_input)
                query_wall = (time.perf_counter_ns() - query_started) * 1.0e-9
                _require("actions" in response, "π0.5-DROID response has no actions")
                chunk = np.asarray(response["actions"], dtype=np.float64)
                _require(
                    chunk.shape == expected_chunk_shape and np.all(np.isfinite(chunk)),
                    "π0.5-DROID action chunk differs",
                )
                for chunk_index in range(execute_count):
                    action_plan.append((query_index, chunk_index, chunk[chunk_index].copy()))
                policy_queries.append(
                    {
                        "query_index": query_index,
                        "rng_seed": rng_seed,
                        "input": input_record,
                        "returned_action_shape": list(chunk.shape),
                        "returned_actions_sha256": array_sha256(chunk),
                        "query_wall_seconds": query_wall,
                        "server_timing": response.get("server_timing"),
                        "policy_timing": response.get("policy_timing"),
                    }
                )
            query_index, chunk_index, model_action = action_plan.popleft()
            nominal_qdot_raw = np.asarray(model_action[:7], dtype=np.float64)
            nominal_qdot = np.clip(nominal_qdot_raw, -1.0, 1.0)
            gripper_model = float(model_action[7])
            _require(math.isfinite(gripper_model), "π0.5-DROID gripper action is nonfinite")
            gripper_env = 1.0 if gripper_model > 0.5 else -1.0
            safe_qdot, qp_record = filter_controller.filter(
                env, nominal_qdot, step=step
            )
            qp_record["applied"] = active
            qp_records.append(qp_record)
            if active:
                if safe_qdot is None:
                    failure = {
                        "component": "joint_velocity_l5_l7_multicbf",
                        "step": step,
                        "reason": qp_record["qp"]["reason"],
                    }
                    break
                executed_qdot = np.asarray(safe_qdot, dtype=np.float64)
            else:
                executed_qdot = nominal_qdot.copy()
            env_action = np.concatenate((executed_qdot, [gripper_env]))
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
            _require(goal["all_satisfied"] is bool(done), "native goal and done differ")
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
                        np.asarray(
                            observation["%s_pos" % obstacle_name], dtype=np.float64
                        )
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
                "source": "post_step_raw_simulator_joint_velocity_pair",
            }
            post_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            action_records.append(
                {
                    "step": step,
                    "policy_query_index": query_index,
                    "policy_chunk_index": chunk_index,
                    "pre_step_simulator_state_sha256": array_sha256(pre_state),
                    "post_step_simulator_state_sha256": array_sha256(post_state),
                    "model_action_raw": model_action.tolist(),
                    "nominal_joint_velocity_raw_rad_s": nominal_qdot_raw.tolist(),
                    "nominal_joint_velocity_clipped_rad_s": nominal_qdot.tolist(),
                    "nominal_saturated_joint_count": int(
                        np.count_nonzero(nominal_qdot_raw != nominal_qdot)
                    ),
                    "model_gripper_closed_fraction": gripper_model,
                    "executed_gripper_command": gripper_env,
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
            "goal summary first-success step differs",
        )
        status = "complete" if failure is None and len(action_records) == horizon else "method_failure"
        result = {
            "schema_version": "vlsa_pi05_droid_joint_velocity_arm.v1",
            "status": status,
            "arm": arm,
            "case_id": CASE_ID,
            "control_effect": (
                "direct_l5_l6_l7_joint_velocity_multicbf"
                if active
                else "unfiltered_pi05_droid_joint_velocity"
            ),
            "pairing": pairing,
            "geometry": geometry,
            "policy_queries": policy_queries,
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
    _require(partial_video.is_file() and partial_video.stat().st_size > 0, "arm video missing")
    os.replace(partial_video, final_video)
    result["video"] = {
        "path": str(final_video.relative_to(output_root)),
        "sha256": _file_sha256(final_video),
        "frames": frames,
        "fps": 15,
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
    checkpoint_path: Path,
    expected_commit: str,
    host: str,
    port: int,
    output_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        _runtime_imports,
        _server_identity,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.joint_velocity import load_joint_velocity_pair_config
    from main.multilink_ellipsoid.shadow import allocation_record

    _require(not output_path.exists(), "pair result already exists")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    config = load_joint_velocity_pair_config(config_path)
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "archived Table-1 hash differs")
    _require(
        archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
        "archived Table-1 payload differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary joint-velocity manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    checkpoint = _checkpoint_tree_record(checkpoint_path)
    runtime = _runtime_imports(include_aegis=False)
    client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
    server = _server_identity(client)
    output_root = output_path.parent
    baseline = _run_arm(
        arm=config["arms"][0],
        runtime=runtime,
        client=client,
        case=case,
        config=config,
        archived=archived,
        output_root=output_root,
    )
    active = _run_arm(
        arm=config["arms"][1],
        runtime=runtime,
        client=client,
        case=case,
        config=config,
        archived=archived,
        output_root=output_root,
    )
    _require(
        baseline["pairing"]["pairing_payload_sha256"]
        == active["pairing"]["pairing_payload_sha256"],
        "paired settled inputs differ",
    )
    first_chunks_equal = bool(
        baseline["policy_queries"]
        and active["policy_queries"]
        and baseline["policy_queries"][0]["returned_actions_sha256"]
        == active["policy_queries"][0]["returned_actions_sha256"]
    )
    _require(first_chunks_equal, "paired first π0.5-DROID chunks differ")
    first_intervention = active["multicbf_summary"]["first_material_intervention_step"]
    prefix_count = (
        min(baseline["action_count"], active["action_count"])
        if first_intervention is None
        else int(first_intervention)
    )
    prefix_state_equal = True
    prefix_nominal_equal = True
    for index in range(prefix_count):
        baseline_action = baseline["actions"][index]
        active_action = active["actions"][index]
        prefix_state_equal = prefix_state_equal and (
            baseline_action["pre_step_simulator_state_sha256"]
            == active_action["pre_step_simulator_state_sha256"]
        )
        prefix_nominal_equal = prefix_nominal_equal and np.array_equal(
            np.asarray(baseline_action["nominal_joint_velocity_clipped_rad_s"]),
            np.asarray(active_action["nominal_joint_velocity_clipped_rad_s"]),
        )
    _require(prefix_state_equal, "paired state differs before first intervention")
    _require(prefix_nominal_equal, "paired nominal actions differ before first intervention")
    baseline_competent = baseline["raw_simulation_evidence"]["native_task_success"]
    active_complete = active["status"] == "complete"
    safe_problem_solved = bool(
        baseline_competent
        and active_complete
        and active["raw_simulation_evidence"]["protected_link_contact_pass"]
        and active["raw_simulation_evidence"]["paper_car_pass"]
        and active["raw_simulation_evidence"]["native_task_success"]
    )
    if not baseline_competent:
        interpretation = "baseline_incompetent_no_safety_efficacy_claim"
    elif safe_problem_solved:
        interpretation = "primary_joint_velocity_problem_solved"
    elif not active_complete:
        interpretation = "active_method_failure"
    else:
        interpretation = "active_joint_velocity_safety_or_task_failure"
    result = {
        "schema_version": "vlsa_pi05_droid_joint_velocity_pair_result.v1",
        "status": "complete" if baseline["status"] == "complete" and active_complete else "method_failure",
        "case_id": CASE_ID,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation,
        "config": config,
        "policy_checkpoint": checkpoint,
        "policy_server": server,
        "archived_table1_geometry_source": {
            "path": str(archived_path),
            "file_sha256": ARCHIVED_FILE_SHA256,
            "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
            "read_only": True,
        },
        "pairing_validation": {
            "settled_pairing_payload_sha256": baseline["pairing"]["pairing_payload_sha256"],
            "first_policy_chunks_equal": first_chunks_equal,
            "first_material_intervention_step": first_intervention,
            "equal_state_prefix_action_count": prefix_count,
            "state_equal_before_first_intervention": prefix_state_equal,
            "nominal_actions_equal_before_first_intervention": prefix_nominal_equal,
            "common_requested_action_horizon": int(config["pairing"]["action_horizon"]),
        },
        "arms": {baseline["arm"]: baseline, active["arm"]: active},
        "baseline_competent": bool(baseline_competent),
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_pair(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        expected_commit=args.expected_commit,
        host=args.host,
        port=args.port,
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
