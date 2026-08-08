#!/usr/bin/env python3
"""Paired barrier-free EE and EmbodiSteer-Joint baselines on primary E05.

This is a paper-derived pi0.5/LIBERO adaptation, not a claim of exact
reproduction of the authors' unreleased implementation.  Both arms use the
same frozen pi05_libero checkpoint and the same policy-noise seed schedule.
No collision geometry, ellipsoid, CBF, or QP is constructed or queried.
"""

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
    _require(commit == expected_commit, "EmbodiSteer baseline source commit differs")
    _require(not run("status", "--short"), "EmbodiSteer baseline source is dirty")
    return {
        "commit": commit,
        "dirty": False,
        "branch": run("branch", "--show-current"),
    }


def _checkpoint_tree_record(root: Path) -> dict[str, Any]:
    _require(root.is_dir() and not root.is_symlink(), "pi05_libero checkpoint missing")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    _require(files, "pi05_libero checkpoint empty")
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in files:
        _require(not path.is_symlink(), "pi05_libero checkpoint contains symlink")
        relative = path.relative_to(root).as_posix()
        size = int(path.stat().st_size)
        digest = _file_sha256(path)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        total_bytes += size
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": aggregate.hexdigest(),
        "tree_hash_algorithm": "sha256(path_nul_size_nul_file_sha256_newline)",
    }


def _response_actions(response: Mapping[str, Any]) -> Any:
    import numpy as np

    _require("actions" in response, "pi0.5 response has no actions")
    actions = np.asarray(response["actions"], dtype=np.float64)
    _require(
        actions.shape == (10, 7) and np.all(np.isfinite(actions)),
        "pi0.5 action chunk differs",
    )
    return actions


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


def _arm_joint_contract(env: Any, margin: float) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.shadow import _arm_dof_indices, _raw_model_data

    model, _ = _raw_model_data(env.sim)
    dofs = _arm_dof_indices(env)
    joint_ids = [int(model.dof_jntid[index]) for index in dofs]
    lower = np.asarray([model.jnt_range[index][0] for index in joint_ids]) + margin
    upper = np.asarray([model.jnt_range[index][1] for index in joint_ids]) - margin
    _require(np.all(lower < upper), "Panda joint limits are invalid")
    return {
        "dof_indices": dofs,
        "joint_ids": joint_ids,
        "qpos_addresses": [int(model.jnt_qposadr[index]) for index in joint_ids],
        "lower": lower,
        "upper": upper,
    }


def _kinematics_callback(probe_env: Any, base_state: Any):
    import numpy as np

    from main.multilink_ellipsoid.shadow import (
        _arm_dof_indices,
        _eef_jacobian,
        _eef_site_id,
        _raw_model_data,
    )

    probe_env.sim.set_state_from_flattened(np.asarray(base_state, dtype=np.float64))
    probe_env.sim.forward()
    model, data = _raw_model_data(probe_env.sim)
    dofs = _arm_dof_indices(probe_env)
    addresses = [
        int(model.jnt_qposadr[int(model.dof_jntid[index])]) for index in dofs
    ]
    site_id = _eef_site_id(probe_env)

    def evaluate(configuration: Any):
        value = np.asarray(configuration, dtype=np.float64)
        _require(
            value.shape == (7,) and np.all(np.isfinite(value)),
            "kinematic configuration differs",
        )
        data.qpos[addresses] = value
        data.qvel[list(dofs)] = 0.0
        probe_env.sim.forward()
        position = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
        rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
        jacobian = np.asarray(_eef_jacobian(probe_env, dofs), dtype=np.float64).copy()
        return position, rotation, jacobian

    return evaluate


def _reference_state(runtime: Mapping[str, Any], case: Mapping[str, Any], settle: int):
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        _active_obstacle,
        _build_environment,
        _settle,
        pairing_record,
    )

    env = None
    try:
        env, task, observation, initial = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, settle)
        obstacle, _ = _active_obstacle(env, observation)
        state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
        pairing = pairing_record(
            case=case,
            selected_initial_state=initial,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle,
            settled_simulator_state=state,
        )
        return {
            "state": state,
            "selected_initial_state": np.asarray(initial).copy(),
            "task_language": str(task.language),
            "obstacle_name": obstacle,
            "pairing": pairing,
        }
    finally:
        if env is not None:
            env.close()


def _build_arm(
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    joint: bool,
):
    import numpy as np

    from main.evaluate_safelibero_aegis import TABLE_RENDER_RESOLUTION, _build_environment

    kwargs = {
        "render_resolution": TABLE_RENDER_RESOLUTION,
        "control_frequency_hz": int(config["action_protocol"]["control_frequency_hz"]),
        "ignore_done": True,
    }
    if joint:
        kwargs["controller_configs"] = config["action_protocol"]["joint_controller"]
    env, task, _, initial = _build_environment(runtime, case, **kwargs)
    _require(
        np.array_equal(np.asarray(initial), reference["selected_initial_state"]),
        "arm initial state selection differs",
    )
    observation = env.regenerate_obs_from_state(reference["state"])
    _require(
        np.array_equal(
            np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
            reference["state"],
        ),
        "arm settled-state transplant differs",
    )
    _require(str(task.language) == reference["task_language"], "arm task differs")
    return env, task, observation


def _joint_chunk(
    *,
    runtime: Mapping[str, Any],
    client: Any,
    observation: Mapping[str, Any],
    task_language: str,
    seed: int,
    env: Any,
    probe_env: Any,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    from main.evaluate_safelibero_aegis import _policy_observation, array_sha256
    from main.multilink_ellipsoid.embodisteer_joint_baseline import (
        CONTROL_SCHEMA,
        apply_joint_denoising_residual,
        initialize_joint_trajectory,
        joint_trajectory_to_pose_actions,
    )

    parameters = config["joint_denoising"]
    policy_input = _policy_observation(
        runtime,
        observation,
        task_description=task_language,
        resize_size=224,
        rng_seed=seed,
    )
    policy_input["__crfs__"]["embodisteer_joint_denoising"] = {
        "schema_version": CONTROL_SCHEMA,
        "mode": "initialize",
        "action_horizon": 10,
        "num_steps": int(parameters["flow_euler_steps"]),
    }
    started = time.perf_counter_ns()
    response = client.infer(policy_input)
    initialize_wall = (time.perf_counter_ns() - started) * 1.0e-9
    initial_actions = _response_actions(response)
    primitive = response.get("embodisteer_joint_denoising")
    _require(
        isinstance(primitive, Mapping) and primitive.get("mode") == "initialize",
        "joint initialization primitive missing",
    )
    model_actions = np.asarray(primitive["model_actions"], dtype=np.float32)
    _require(model_actions.shape == (10, 32), "joint initialization model shape differs")
    start_configuration = np.asarray(_joint_state(env)[0], dtype=np.float64)
    base_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
    kinematics = _kinematics_callback(probe_env, base_state)
    contract = _arm_joint_contract(env, float(parameters["joint_limit_margin_rad"]))
    trajectory, initialization = initialize_joint_trajectory(
        initial_actions[:, :6],
        start_configuration,
        kinematics,
        alpha=float(parameters["noise_initialization_scale_alpha"]),
        damping=float(parameters["jacobian_damping_lambda_pinv"]),
        update_clip_rad=float(parameters["joint_update_clip_rad"]),
        translation_scale_m=float(parameters["cartesian_translation_scale_m_per_action_unit"]),
        rotation_scale_rad=float(parameters["cartesian_rotation_scale_rad_per_action_unit"]),
        lower=contract["lower"],
        upper=contract["upper"],
    )
    flow_records = []
    final_actions = initial_actions
    for reverse_index in range(int(parameters["flow_euler_steps"])):
        reverse_time = 1.0 - reverse_index / float(parameters["flow_euler_steps"])
        pose_actions = joint_trajectory_to_pose_actions(
            trajectory,
            start_configuration,
            kinematics,
            translation_scale_m=float(parameters["cartesian_translation_scale_m_per_action_unit"]),
            rotation_scale_rad=float(parameters["cartesian_rotation_scale_rad_per_action_unit"]),
        )
        step_input = _policy_observation(
            runtime,
            observation,
            task_description=task_language,
            resize_size=224,
            rng_seed=seed,
        )
        step_input["__crfs__"]["embodisteer_joint_denoising"] = {
            "schema_version": CONTROL_SCHEMA,
            "mode": "step",
            "action_horizon": 10,
            "num_steps": int(parameters["flow_euler_steps"]),
            "time": reverse_time,
            "model_actions": model_actions.tolist(),
            "physical_pose_actions": pose_actions.tolist(),
        }
        step_started = time.perf_counter_ns()
        step_response = client.infer(step_input)
        step_wall = (time.perf_counter_ns() - step_started) * 1.0e-9
        final_actions = _response_actions(step_response)
        step_primitive = step_response.get("embodisteer_joint_denoising")
        _require(
            isinstance(step_primitive, Mapping) and step_primitive.get("mode") == "step",
            "joint flow-step primitive missing",
        )
        input_physical = np.asarray(
            step_primitive["input_physical_actions"], dtype=np.float64
        )
        _require(
            np.allclose(input_physical[:, :6], pose_actions, atol=1.0e-7, rtol=0.0),
            "server joint flow input differs from FK pose actions",
        )
        next_physical = np.asarray(step_primitive["physical_actions"], dtype=np.float64)
        trajectory, residual = apply_joint_denoising_residual(
            trajectory,
            start_configuration,
            input_physical[:, :6],
            next_physical[:, :6],
            kinematics,
            damping=float(parameters["jacobian_damping_lambda_pinv"]),
            update_clip_rad=float(parameters["joint_update_clip_rad"]),
            translation_scale_m=float(parameters["cartesian_translation_scale_m_per_action_unit"]),
            rotation_scale_rad=float(parameters["cartesian_rotation_scale_rad_per_action_unit"]),
            lower=contract["lower"],
            upper=contract["upper"],
        )
        model_actions = np.asarray(step_primitive["model_actions"], dtype=np.float32)
        flow_records.append(
            {
                "reverse_index": reverse_index,
                "time": reverse_time,
                "fk_pose_actions_sha256": array_sha256(pose_actions),
                "model_actions_sha256": array_sha256(model_actions),
                "joint_trajectory_sha256": array_sha256(trajectory),
                "residual": residual,
                "wall_seconds": step_wall,
            }
        )
    return trajectory, {
        "rng_seed": seed,
        "initial_model_noise_sha256": array_sha256(
            np.asarray(primitive["model_actions"], dtype=np.float32)
        ),
        "initial_decoded_actions_sha256": array_sha256(initial_actions),
        "initialization": initialization,
        "initialize_wall_seconds": initialize_wall,
        "flow_steps": flow_records,
        "final_gripper_actions": final_actions[:, 6].tolist(),
        "final_joint_trajectory_sha256": array_sha256(trajectory),
        "collision_geometry_queried": False,
        "barrier_qp_solved": False,
    }


def _flow_step_equivalence_preflight(
    *,
    runtime: Mapping[str, Any],
    client: Any,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that ten exposed one-step Euler calls equal ordinary sampling."""

    import numpy as np

    from main.evaluate_safelibero_aegis import _policy_observation, array_sha256
    from main.multilink_ellipsoid.embodisteer_joint_baseline import CONTROL_SCHEMA

    env = None
    try:
        env, task, observation = _build_arm(
            runtime, case, config, reference, joint=False
        )
        seed = int(case["policy_noise_seed"])
        ordinary_input = _policy_observation(
            runtime,
            observation,
            task_description=str(task.language),
            resize_size=224,
            rng_seed=seed,
        )
        ordinary = _response_actions(client.infer(ordinary_input))
        initialize_input = _policy_observation(
            runtime,
            observation,
            task_description=str(task.language),
            resize_size=224,
            rng_seed=seed,
        )
        initialize_input["__crfs__"]["embodisteer_joint_denoising"] = {
            "schema_version": CONTROL_SCHEMA,
            "mode": "initialize",
            "action_horizon": 10,
            "num_steps": 10,
        }
        initial_response = client.infer(initialize_input)
        primitive = initial_response["embodisteer_joint_denoising"]
        model_actions = np.asarray(primitive["model_actions"], dtype=np.float32)
        physical = np.asarray(primitive["physical_actions"], dtype=np.float64)
        for reverse_index in range(10):
            step_input = _policy_observation(
                runtime,
                observation,
                task_description=str(task.language),
                resize_size=224,
                rng_seed=seed,
            )
            step_input["__crfs__"]["embodisteer_joint_denoising"] = {
                "schema_version": CONTROL_SCHEMA,
                "mode": "step",
                "action_horizon": 10,
                "num_steps": 10,
                "time": 1.0 - reverse_index / 10.0,
                "model_actions": model_actions.tolist(),
                "physical_pose_actions": physical[:, :6].tolist(),
            }
            response = client.infer(step_input)
            primitive = response["embodisteer_joint_denoising"]
            model_actions = np.asarray(primitive["model_actions"], dtype=np.float32)
            physical = np.asarray(primitive["physical_actions"], dtype=np.float64)
        maximum_error = float(np.max(np.abs(ordinary - physical)))
        diagnostic = {
            "ordinary_actions_sha256": array_sha256(ordinary),
            "chained_flow_step_actions_sha256": array_sha256(physical),
            "maximum_absolute_physical_action_error": maximum_error,
            "mean_absolute_physical_action_error": float(
                np.mean(np.abs(ordinary - physical))
            ),
            "maximum_absolute_error_by_action_dimension": np.max(
                np.abs(ordinary - physical), axis=0
            ).tolist(),
            "maximum_translation_discrepancy_m": float(
                0.05 * np.max(np.abs(ordinary[:, :3] - physical[:, :3]))
            ),
            "maximum_rotation_discrepancy_rad": float(
                0.5 * np.max(np.abs(ordinary[:, 3:6] - physical[:, 3:6]))
            ),
            "executed_first_five_gripper_signs_equal": bool(
                np.array_equal(
                    np.sign(ordinary[:5, 6]), np.sign(physical[:5, 6])
                )
            ),
            "maximum_error_index": [
                int(value)
                for value in np.unravel_index(
                    int(np.argmax(np.abs(ordinary - physical))), ordinary.shape
                )
            ],
            "ordinary_value_at_maximum_error": float(
                ordinary[
                    np.unravel_index(
                        int(np.argmax(np.abs(ordinary - physical))), ordinary.shape
                    )
                ]
            ),
            "chained_value_at_maximum_error": float(
                physical[
                    np.unravel_index(
                        int(np.argmax(np.abs(ordinary - physical))), ordinary.shape
                    )
                ]
            ),
        }
        print("FLOW_STEP_EQUIVALENCE " + json.dumps(diagnostic, sort_keys=True))
        _require(
            maximum_error <= 5.0e-5,
            "exposed flow-step primitive differs from ordinary pi0.5 sampling: "
            + json.dumps(diagnostic, sort_keys=True),
        )
        return {
            "status": "passing",
            "seed": seed,
            **diagnostic,
            "tolerance": 5.0e-5,
        }
    finally:
        if env is not None:
            env.close()


def _run_arm(
    *,
    arm: str,
    runtime: Mapping[str, Any],
    client: Any,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    reference: Mapping[str, Any],
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
        _policy_observation,
        _processed_image,
        array_sha256,
        pairing_record,
        query_seed,
    )

    joint = arm == "joint_denoising_no_guidance"
    _require(joint or arm == "cartesian_ee_no_guidance", "unknown baseline arm")
    arm_root = output_root / "arms" / arm
    arm_root.mkdir(parents=True, exist_ok=False)
    partial_video = arm_root / "episode.partial.mp4"
    final_video = arm_root / "episode.mp4"
    final_jpg = arm_root / "final.jpg"
    env = None
    probe_env = None
    writer = None
    frames = 0
    started = time.perf_counter_ns()
    try:
        env, task, observation = _build_arm(
            runtime, case, config, reference, joint=joint
        )
        if joint:
            probe_env, _, _, probe_initial = _build_environment(
                runtime,
                case,
                render_resolution=32,
                controller_configs=config["action_protocol"]["joint_controller"],
                control_frequency_hz=int(config["action_protocol"]["control_frequency_hz"]),
                ignore_done=True,
            )
            _require(
                np.array_equal(np.asarray(probe_initial), reference["selected_initial_state"]),
                "joint kinematics probe initial state differs",
            )
            probe_env.sim.set_state_from_flattened(reference["state"])
            probe_env.sim.forward()
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(obstacle_name == reference["obstacle_name"], "arm obstacle differs")
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        pairing = pairing_record(
            case=case,
            selected_initial_state=reference["selected_initial_state"],
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(env.sim.get_state().flatten()),
        )
        _require(
            pairing["settled_simulator_state_sha256"]
            == reference["pairing"]["settled_simulator_state_sha256"],
            "paired arm settled simulator state differs",
        )
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env, goal_atoms, step=-1, previous_values=None
        )
        previous_goal_values = initial_goal["values"]
        contact_authority = _contact_model_authority(env, obstacle_name)
        writer = runtime["imageio"].get_writer(
            str(partial_video),
            fps=int(config["action_protocol"]["control_frequency_hz"]),
            codec="libx264",
            macro_block_size=None,
            output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "18"],
        )
        frame = _processed_image(observation, "agentview_image")
        writer.append_data(frame)
        frames += 1

        action_plan: collections.deque[dict[str, Any]] = collections.deque()
        policy_queries = []
        actions = []
        robot_contact_geoms: set[str] = set()
        protected_contact_geoms: set[str] = set()
        first_robot_contact = None
        first_protected_contact = None
        first_car = None
        first_success = None
        maximum_displacement = 0.0
        max_actions = int(config["action_protocol"]["max_actions"])
        execute_count = int(config["action_protocol"]["execute_actions_per_query"])
        for step in range(max_actions):
            if not action_plan:
                query_index = len(policy_queries)
                seed = query_seed(int(case["policy_noise_seed"]), query_index)
                query_started = time.perf_counter_ns()
                if joint:
                    trajectory, query_record = _joint_chunk(
                        runtime=runtime,
                        client=client,
                        observation=observation,
                        task_language=str(task.language),
                        seed=seed,
                        env=env,
                        probe_env=probe_env,
                        config=config,
                    )
                    gripper_actions = query_record.pop("final_gripper_actions")
                    for chunk_index in range(execute_count):
                        action_plan.append(
                            {
                                "query_index": query_index,
                                "chunk_index": chunk_index,
                                "joint_target": np.asarray(trajectory[chunk_index]).copy(),
                                "gripper": float(gripper_actions[chunk_index]),
                            }
                        )
                else:
                    policy_input = _policy_observation(
                        runtime,
                        observation,
                        task_description=str(task.language),
                        resize_size=224,
                        rng_seed=seed,
                    )
                    response = client.infer(policy_input)
                    chunk = _response_actions(response)
                    for chunk_index in range(execute_count):
                        action_plan.append(
                            {
                                "query_index": query_index,
                                "chunk_index": chunk_index,
                                "cartesian_action": chunk[chunk_index].copy(),
                            }
                        )
                    query_record = {
                        "rng_seed": seed,
                        "returned_actions_sha256": array_sha256(chunk),
                        "server_timing": response.get("server_timing"),
                        "policy_timing": response.get("policy_timing"),
                        "collision_geometry_queried": False,
                        "barrier_qp_solved": False,
                    }
                query_record.update(
                    {
                        "query_index": query_index,
                        "total_wall_seconds": (
                            time.perf_counter_ns() - query_started
                        ) * 1.0e-9,
                    }
                )
                policy_queries.append(query_record)

            planned = action_plan.popleft()
            pre_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            pre_qpos, pre_qvel = _joint_state(env)
            if joint:
                target = np.asarray(planned["joint_target"], dtype=np.float64)
                normalized_delta = (target - np.asarray(pre_qpos)) / 0.05
                arm_action = np.clip(normalized_delta, -1.0, 1.0)
                gripper = float(planned["gripper"])
                env_action = np.concatenate((arm_action, [gripper]))
                action_source = {
                    "joint_target_rad": target.tolist(),
                    "unclipped_normalized_joint_delta": normalized_delta.tolist(),
                    "saturated_joint_count": int(
                        np.count_nonzero(normalized_delta != arm_action)
                    ),
                }
            else:
                env_action = np.asarray(planned["cartesian_action"], dtype=np.float64)
                action_source = {"cartesian_policy_action": env_action.tolist()}
            _require(np.all(np.isfinite(env_action)), "environment action is nonfinite")
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
            if goal["all_satisfied"] and first_success is None:
                first_success = step
            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=step,
                contact_authority=contact_authority,
            )
            _require(contacts["status"] == "available", "raw contacts unavailable")
            robot_events = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            protected_events = _protected_events(robot_events)
            if robot_events and first_robot_contact is None:
                first_robot_contact = step
            if protected_events and first_protected_contact is None:
                first_protected_contact = step
            for event in robot_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    robot_contact_geoms.add(name)
            for event in protected_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    protected_contact_geoms.add(name)
            displacement = float(
                np.sum(
                    np.abs(
                        np.asarray(observation["%s_pos" % obstacle_name])
                        - initial_obstacle_position
                    )
                )
            )
            maximum_displacement = max(maximum_displacement, displacement)
            if first_car is None and displacement > PAPER_CAR_THRESHOLD_M:
                first_car = step
            post_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            actions.append(
                {
                    "step": step,
                    "query_index": int(planned["query_index"]),
                    "chunk_index": int(planned["chunk_index"]),
                    "pre_state_sha256": array_sha256(pre_state),
                    "post_state_sha256": array_sha256(post_state),
                    "executed_env_action": env_action.tolist(),
                    "action_source": action_source,
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
                    "env_step_wall_seconds": step_wall,
                }
            )
            frame = _processed_image(observation, "agentview_image")
            writer.append_data(frame)
            frames += 1
            if first_success is not None:
                break

        goal_summary = _goal_progress_summary(initial_goal, actions)
        evidence = {
            "native_task_success": first_success is not None,
            "native_task_success_step": first_success,
            "first_robot_contact_step": first_robot_contact,
            "first_protected_link_contact_step": first_protected_contact,
            "direct_robot_contact_geoms": sorted(robot_contact_geoms),
            "direct_protected_link_contact_geoms": sorted(protected_contact_geoms),
            "first_paper_car_step": first_car,
            "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
            "paper_car_pass": first_car is None,
            "protected_link_contact_pass": first_protected_contact is None,
        }
        result = {
            "schema_version": "vlsa_embodisteer_joint_baseline_arm.v1",
            "status": "complete",
            "arm": arm,
            "case_id": CASE_ID,
            "controller": (
                config["action_protocol"]["joint_controller"]
                if joint
                else {"type": "OSC_POSE", "source": "SafeLIBERO default"}
            ),
            "control_effect": (
                "paper_derived_joint_space_denoising_without_guidance"
                if joint
                else "ordinary_cartesian_denoising_without_guidance"
            ),
            "barrier_projection_enabled": False,
            "ellipsoid_constraints_enabled": False,
            "qp_enabled": False,
            "pairing": pairing,
            "policy_queries": policy_queries,
            "action_count": len(actions),
            "actions": actions,
            "goal_progress": {
                **goal_definition,
                "initial": initial_goal,
                "summary": goal_summary,
            },
            "raw_simulation_evidence": evidence,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        runtime["imageio"].imwrite(str(final_jpg), frame)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    _require(partial_video.is_file() and partial_video.stat().st_size > 0, "arm video missing")
    os.replace(partial_video, final_video)
    result["video"] = {
        "path": str(final_video.relative_to(output_root)),
        "sha256": _file_sha256(final_video),
        "frames": frames,
        "fps": int(config["action_protocol"]["control_frequency_hz"]),
    }
    result["final_jpg"] = {
        "path": str(final_jpg.relative_to(output_root)),
        "sha256": _file_sha256(final_jpg),
    }
    result["result_payload_sha256"] = _sha256(_canonical(result))
    return result


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    expected_commit: str,
    host: str,
    port: int,
    output_path: Path,
) -> dict[str, Any]:
    from main.evaluate_safelibero_aegis import (
        _runtime_imports,
        _server_identity,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.embodisteer_joint_baseline import (
        load_joint_baseline_config,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    _require(not output_path.exists(), "baseline result already exists")
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    config = load_joint_baseline_config(config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    checkpoint = _checkpoint_tree_record(checkpoint_path)
    runtime = _runtime_imports(include_aegis=False)
    client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
    server = _server_identity(client)
    reference = _reference_state(
        runtime, case, int(config["pairing"]["settle_actions"])
    )
    sampler_preflight = _flow_step_equivalence_preflight(
        runtime=runtime,
        client=client,
        case=case,
        config=config,
        reference=reference,
    )
    output_root = output_path.parent
    cartesian = _run_arm(
        arm=config["arms"][0],
        runtime=runtime,
        client=client,
        case=case,
        config=config,
        reference=reference,
        output_root=output_root,
    )
    joint = _run_arm(
        arm=config["arms"][1],
        runtime=runtime,
        client=client,
        case=case,
        config=config,
        reference=reference,
        output_root=output_root,
    )
    _require(
        cartesian["pairing"]["settled_simulator_state_sha256"]
        == joint["pairing"]["settled_simulator_state_sha256"],
        "paired baseline settled states differ",
    )
    cartesian_success = cartesian["raw_simulation_evidence"]["native_task_success"]
    joint_success = joint["raw_simulation_evidence"]["native_task_success"]
    if cartesian_success and joint_success:
        interpretation = "joint_denoising_preserves_single_case_task_competence"
    elif cartesian_success and not joint_success:
        interpretation = "joint_denoising_adaptation_fails_competence_gate"
    elif not cartesian_success and joint_success:
        interpretation = "joint_denoising_improves_this_single_case_but_not_paper_comparability"
    else:
        interpretation = "both_barrier_free_arms_fail_single_case_no_paper_like_competence_claim"
    result = {
        "schema_version": "vlsa_embodisteer_joint_baseline_pair_result.v1",
        "status": "complete",
        "case_id": CASE_ID,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation,
        "config": config,
        "checkpoint": checkpoint,
        "policy_server": server,
        "sampler_regression_preflight": sampler_preflight,
        "paper_fidelity": {
            "same_frozen_cartesian_checkpoint": True,
            "joint_space_sampling_variable": True,
            "fk_before_each_denoising_step": True,
            "damped_jacobian_residual_after_each_denoising_step": True,
            "collision_guidance_disabled": True,
            "libero_incremental_action_adaptation": True,
            "exact_author_code_reproduction": False,
            "reason_not_exact": "authors_project_page_reported_code_coming_soon_and_pi05_libero_uses_incremental_7D_OSC_flow_actions_instead_of_chunk_start_relative_10D_DDPM_actions",
        },
        "geometry_isolation": {
            "l5_l6_l7_ellipsoids_constructed": False,
            "collision_sdf_queried": False,
            "barrier_constraints_built": False,
            "qp_solved": False,
            "physical_obstacle_remains_in_scene": True,
        },
        "arms": {config["arms"][0]: cartesian, config["arms"][1]: joint},
        "comparison": {
            "interpretation": interpretation,
            "cartesian_native_task_success": cartesian_success,
            "joint_native_task_success": joint_success,
            "cartesian_action_count": cartesian["action_count"],
            "joint_action_count": joint["action_count"],
            "cartesian_protected_contact_step": cartesian["raw_simulation_evidence"][
                "first_protected_link_contact_step"
            ],
            "joint_protected_contact_step": joint["raw_simulation_evidence"][
                "first_protected_link_contact_step"
            ],
        },
    }
    result["result_payload_sha256"] = _sha256(_canonical(result))
    _atomic_write(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        expected_commit=args.expected_commit,
        host=args.host,
        port=args.port,
        output_path=args.output.resolve(),
    )
    print(json.dumps(result["comparison"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
