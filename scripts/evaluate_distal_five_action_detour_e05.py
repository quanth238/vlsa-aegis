#!/usr/bin/env python3
"""Run an early five-action, endpoint-preserving E05 detour and recovery."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    PAPER_CAR_THRESHOLD_M,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _is_protected_event,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_five_action_detour_e05_result.v1"


def _archived_action(records: list[dict[str, Any]], step: int) -> Any:
    import numpy as np

    record = records[step]
    _require(int(record["step"]) == step, "archived action step differs")
    action = np.asarray(record["env_step_input"], dtype=np.float64)
    _require(
        action.shape == (7,)
        and np.array_equal(action, np.asarray(record["executed"], dtype=np.float64)),
        "archived action binding differs",
    )
    return action


def _disable_images(env: Any) -> int:
    disabled = 0
    for observable in env.env._observables.values():
        if str(getattr(observable, "modality", "")) == "image":
            observable.set_enabled(False)
            disabled += 1
    return disabled


def _public_rollout(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "actions": record["actions"].tolist(),
        "clearance_trace_m": record["clearance_trace_m"].tolist(),
        "minimum_clearance_m": float(record["minimum_clearance_m"]),
        "eef_position_trace_m": record["eef_position_trace_m"].tolist(),
        "state_sha256": list(record["state_sha256"]),
        "synchronization": dict(record["synchronization"]),
        "protected_contacts": list(record["protected_contacts"]),
        "robot_contacts": list(record["robot_contacts"]),
        "active_obstacle_l1_displacement_m": list(
            record["active_obstacle_l1_displacement_m"]
        ),
        "done_steps": list(record["done_steps"]),
        "env_step_wall_seconds": float(record["env_step_wall_seconds"]),
    }


class FiveActionProbe:
    def __init__(self, one_step_probe: Any, contact_reader: Any) -> None:
        self.one_step_probe = one_step_probe
        self.contact_reader = contact_reader

    @property
    def env(self) -> Any:
        return self.one_step_probe.probe_env

    def rollout(self, main_env: Any, actions: Any) -> dict[str, Any]:
        import numpy as np

        from main.multilink_ellipsoid.rollout import _dynamic_state_vector
        from main.multilink_ellipsoid.sitl_candidate import _obstacle_root_body_id

        commands = np.asarray(actions, dtype=np.float64)
        _require(
            commands.shape == (5, 7) and np.all(np.isfinite(commands)),
            "five-action rollout shape differs",
        )
        synchronization = self.one_step_probe.synchronize(main_env)
        obstacle_id = _obstacle_root_body_id(
            self.env.sim.model, self.one_step_probe.active_obstacle_name
        )
        obstacle_before = np.asarray(
            self.env.sim.data.xpos[obstacle_id], dtype=np.float64
        ).copy()
        clearances = []
        eef_positions = []
        state_vectors = []
        protected_contacts = []
        robot_contacts = []
        displacements = []
        done_steps = []
        started = time.perf_counter_ns()
        for offset, command in enumerate(commands):
            observation, _, done, _ = self.env.step(command.tolist())
            clearances.append(self.one_step_probe.clearances(self.env)[:7])
            eef_positions.append(
                np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            )
            state_vectors.append(_dynamic_state_vector(self.env))
            contacts = self.contact_reader(self.env, offset)
            robot = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            protected = [event for event in robot if _is_protected_event(event)]
            robot_contacts.extend(robot)
            protected_contacts.extend(protected)
            obstacle_now = np.asarray(
                self.env.sim.data.xpos[obstacle_id], dtype=np.float64
            )
            displacements.append(float(np.sum(np.abs(obstacle_now - obstacle_before))))
            if done:
                done_steps.append(offset)
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        clearance = np.asarray(clearances, dtype=np.float64)
        eef = np.asarray(eef_positions, dtype=np.float64)
        return {
            "actions": commands,
            "clearance_trace_m": clearance,
            "minimum_clearance_m": float(np.min(clearance)),
            "eef_position_trace_m": eef,
            "state_vectors": state_vectors,
            "state_sha256": [
                hashlib.sha256(vector.tobytes()).hexdigest()
                for vector in state_vectors
            ],
            "synchronization": synchronization,
            "protected_contacts": protected_contacts,
            "robot_contacts": robot_contacts,
            "active_obstacle_l1_displacement_m": displacements,
            "done_steps": done_steps,
            "env_step_wall_seconds": elapsed,
        }


def _corrected_actions(nominal: Any, correction: Any, limit: float) -> Any:
    import numpy as np

    actions = np.asarray(nominal, dtype=np.float64).copy()
    delta = np.asarray(correction, dtype=np.float64).reshape(5, 3)
    actions[:, :3] += delta
    if float(np.max(np.abs(actions[:, :3]))) > float(limit) + 1.0e-10:
        raise ValueError("five-action detour exceeds action bounds")
    return actions


def _linearization(
    probe: FiveActionProbe,
    env: Any,
    nominal: Any,
    correction: Any,
    target_terminal_eef: Any,
    *,
    epsilon: float,
    action_limit: float,
) -> dict[str, Any]:
    import numpy as np

    center_actions = _corrected_actions(nominal, correction, action_limit)
    center = probe.rollout(env, center_actions)
    h = np.asarray(center["clearance_trace_m"], dtype=np.float64).reshape(35)
    terminal = np.asarray(center["eef_position_trace_m"][-1], dtype=np.float64)
    target = np.asarray(target_terminal_eef, dtype=np.float64)
    rows = np.zeros((35, 15), dtype=np.float64)
    terminal_rows = np.zeros((3, 15), dtype=np.float64)
    rollout_count = 1
    wall = float(center["env_step_wall_seconds"])
    for variable in range(15):
        slot, dimension = divmod(variable, 3)
        plus = center_actions.copy()
        minus = center_actions.copy()
        plus[slot, dimension] = min(action_limit, plus[slot, dimension] + epsilon)
        minus[slot, dimension] = max(-action_limit, minus[slot, dimension] - epsilon)
        denominator = float(plus[slot, dimension] - minus[slot, dimension])
        _require(denominator > 0.0, "five-action finite-difference denominator differs")
        positive = probe.rollout(env, plus)
        negative = probe.rollout(env, minus)
        rows[:, variable] = (
            np.asarray(positive["clearance_trace_m"], dtype=np.float64).reshape(35)
            - np.asarray(negative["clearance_trace_m"], dtype=np.float64).reshape(35)
        ) / denominator
        terminal_rows[:, variable] = (
            np.asarray(positive["eef_position_trace_m"][-1], dtype=np.float64)
            - np.asarray(negative["eef_position_trace_m"][-1], dtype=np.float64)
        ) / denominator
        rollout_count += 2
        wall += float(positive["env_step_wall_seconds"])
        wall += float(negative["env_step_wall_seconds"])
    return {
        "center": center,
        "clearance": h,
        "clearance_rows": rows,
        "terminal_error": terminal - target,
        "terminal_rows": terminal_rows,
        "rollout_count": rollout_count,
        "probe_env_step_wall_seconds": wall,
    }


def _search_detour(
    probe: FiveActionProbe,
    env: Any,
    nominal: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import cvxpy as cp
    import numpy as np

    from main.multilink_ellipsoid.five_action_detour import solve_detour_qp

    qp_config = config["sequential_qp"]
    action_limit = float(qp_config["action_limit"])
    site_id = env.sim.model.site_name2id("robot0_grip_site")
    start_eef = np.asarray(
        env.sim.data.site_xpos[site_id], dtype=np.float64
    ).copy()
    nominal_rollout = probe.rollout(env, nominal)
    target_terminal = np.asarray(
        nominal_rollout["eef_position_trace_m"][-1], dtype=np.float64
    )
    nominal_delta = target_terminal - start_eef
    denominator = float(np.dot(nominal_delta, nominal_delta))
    correction = np.zeros(15, dtype=np.float64)
    history = []
    total_rollouts = 1
    total_probe_wall = float(nominal_rollout["env_step_wall_seconds"])
    best = None
    for iteration in range(int(qp_config["iterations"])):
        linear = _linearization(
            probe,
            env,
            nominal,
            correction,
            target_terminal,
            epsilon=float(config["finite_difference"]["perturbation_action"]),
            action_limit=action_limit,
        )
        total_rollouts += int(linear["rollout_count"])
        total_probe_wall += float(linear["probe_env_step_wall_seconds"])
        qp = solve_detour_qp(
            cp=cp,
            nominal_actions=nominal,
            center_correction=correction,
            clearance_trace_m=linear["clearance"],
            clearance_jacobian_m_per_action=linear["clearance_rows"],
            terminal_eef_error_m=linear["terminal_error"],
            terminal_eef_jacobian_m_per_action=linear["terminal_rows"],
            config=config,
        )
        if not qp["valid"]:
            history.append({"iteration": iteration, "qp": qp, "exact": None})
            break
        correction = np.asarray(qp["total_correction"], dtype=np.float64)
        actions = _corrected_actions(nominal, correction, action_limit)
        exact = probe.rollout(env, actions)
        total_rollouts += 1
        total_probe_wall += float(exact["env_step_wall_seconds"])
        terminal = np.asarray(exact["eef_position_trace_m"][-1], dtype=np.float64)
        terminal_error = float(np.linalg.norm(terminal - target_terminal))
        progress = (
            0.0
            if denominator <= 1.0e-12
            else float(np.dot(terminal - start_eef, nominal_delta) / denominator)
        )
        candidate = {
            "actions": actions.tolist(),
            "correction": correction.tolist(),
            "correction_l2": float(np.linalg.norm(correction)),
            "maximum_abs_correction": float(np.max(np.abs(correction))),
            "endpoint_correction_sum": np.sum(correction.reshape(5, 3), axis=0).tolist(),
            "terminal_eef_error_m": terminal_error,
            "task_progress_ratio": progress,
            "exact": _public_rollout(exact),
            "exact_safe": bool(
                float(exact["minimum_clearance_m"])
                >= float(qp_config["clearance_buffer_m"])
                and not exact["protected_contacts"]
                and not exact["robot_contacts"]
                and max(exact["active_obstacle_l1_displacement_m"])
                <= PAPER_CAR_THRESHOLD_M
            ),
        }
        candidate["gate_pass"] = bool(
            candidate["exact_safe"]
            and terminal_error <= float(qp_config["maximum_terminal_eef_error_m"])
            and progress >= float(qp_config["minimum_progress_ratio"])
        )
        history.append({"iteration": iteration, "qp": qp, "exact": candidate})
        if best is None or (
            candidate["gate_pass"],
            -candidate["terminal_eef_error_m"],
            -candidate["correction_l2"],
            candidate["exact"]["minimum_clearance_m"],
        ) > (
            best["gate_pass"],
            -best["terminal_eef_error_m"],
            -best["correction_l2"],
            best["exact"]["minimum_clearance_m"],
        ):
            best = candidate
        if candidate["gate_pass"]:
            break
    return {
        "nominal": _public_rollout(nominal_rollout),
        "history": history,
        "best": best,
        "gate_pass": bool(best is not None and best["gate_pass"]),
        "rollout_count": total_rollouts,
        "probe_env_step_wall_seconds": total_probe_wall,
    }


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    expected_commit: str,
    output_path: Path,
    host: str,
    port: int,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        TABLE_VIDEO_FPS,
        _active_obstacle,
        _aegis_action,
        _build_environment,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _eef_proxy,
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _policy_observation,
        _processed_image,
        _runtime_imports,
        _server_identity,
        _settle,
        array_sha256,
        max_steps_for_case,
        pairing_record,
        query_seed,
        read_jsonl,
        translational_action,
        validate_case_row,
    )
    from main.multilink_ellipsoid.five_action_detour import (
        load_five_action_detour_config,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list) and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "Table-1 action horizon differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    config = load_five_action_detour_config(experiment_config_path)
    geometry_config = load_shadow_config(geometry_config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=True)
    env = None
    probe_env = None
    video_writer = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_partial = output_path.with_name("episode.partial.mp4")
    video_final = output_path.with_name("episode.mp4")
    final_jpg = output_path.with_name("final.jpg")
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        stale_proxy = _eef_proxy(runtime, observation)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, probe_initial_state = _build_environment(
            runtime, case, render_resolution=32
        )
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        _require(str(probe_task.language) == str(task.language), "probe task differs")
        _require(
            np.array_equal(np.asarray(probe_initial_state), np.asarray(selected_initial_state)),
            "probe state differs",
        )
        _require(
            np.array_equal(
                np.asarray(probe_env.sim.get_state().flatten()),
                np.asarray(env.sim.get_state().flatten()),
            ),
            "probe settled simulator state differs",
        )
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(env.sim.get_state().flatten()),
        )
        for key in (
            "manifest_row_sha256",
            "initial_state_sha256",
            "initial_observation_sha256",
            "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == archived["pairing"][key], "pairing differs: %s" % key)
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {
                "p2": perception["mvee_center"],
                "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        one_step_probe = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        probe_authority = _contact_model_authority(probe_env, obstacle_name)

        def contact_reader(probe: Any, offset: int) -> dict[str, Any]:
            contacts = _detailed_active_obstacle_contacts(
                probe,
                obstacle_name,
                step=182 + int(offset),
                contact_authority=probe_authority,
            )
            _require(contacts["status"] == "available", "probe contact evidence differs")
            return contacts

        probe = FiveActionProbe(one_step_probe, contact_reader)
        disabled_probe_images = _disable_images(probe_env)
        contact_authority = _contact_model_authority(env, obstacle_name)
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(env, goal_atoms, step=-1, previous_values=None)
        previous_goal_values = initial_goal["values"]
        terminal_frame = _processed_image(observation, "agentview_image")
        video_writer = runtime["imageio"].get_writer(
            str(video_partial),
            fps=TABLE_VIDEO_FPS,
            codec="libx264",
            macro_block_size=None,
            pixelformat="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        video_writer.append_data(terminal_frame)
        action_records = []
        first_robot_contact_step = None
        first_protected_contact_step = None
        first_car_step = None
        maximum_displacement = 0.0
        for step in range(182):
            action = _archived_action(archived_actions, step)
            observation, reward, done, _ = env.step(action.tolist())
            _require(not bool(done), "archived prefix completed before detour")
            goal = _goal_progress_snapshot(env, goal_atoms, step=step, previous_values=previous_goal_values)
            previous_goal_values = goal["values"]
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            action_records.append(
                {
                    "step": step,
                    "source": "immutable_archived_aegis_prefix",
                    "action": action.tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                }
            )
        nominal_detour = np.asarray(
            [_archived_action(archived_actions, step) for step in range(182, 187)],
            dtype=np.float64,
        )
        detour = _search_detour(probe, env, nominal_detour, config)
        failure = None
        if not detour["gate_pass"]:
            failure = {"component": "five_action_detour_oracle", "step": 182, "reason": "no_verified_task_rejoining_detour"}
        else:
            selected_actions = np.asarray(detour["best"]["actions"], dtype=np.float64)
            fresh = probe.rollout(env, selected_actions)
            _require(
                float(fresh["minimum_clearance_m"])
                >= float(config["sequential_qp"]["clearance_buffer_m"])
                and not fresh["protected_contacts"]
                and not fresh["robot_contacts"]
                and max(fresh["active_obstacle_l1_displacement_m"]) <= PAPER_CAR_THRESHOLD_M,
                "fresh selected detour verification failed",
            )
            expected_states = fresh["state_vectors"]
            for offset, action in enumerate(selected_actions):
                step = 182 + offset
                observation, reward, done, _ = env.step(action.tolist())
                state_error = float(
                    np.max(
                        np.abs(
                            _dynamic_state_vector(env)
                            - np.asarray(expected_states[offset], dtype=np.float64)
                        )
                    )
                )
                _require(state_error <= 1.0e-10, "executed detour differs from clone")
                goal = _goal_progress_snapshot(env, goal_atoms, step=step, previous_values=previous_goal_values)
                previous_goal_values = goal["values"]
                contacts = _detailed_active_obstacle_contacts(
                    env, obstacle_name, step=step, contact_authority=contact_authority
                )
                robot_events = [
                    event for event in contacts["events"]
                    if event.get("other", {}).get("classification") == "robot"
                ]
                protected_events = [event for event in robot_events if _is_protected_event(event)]
                if robot_events and first_robot_contact_step is None:
                    first_robot_contact_step = step
                if protected_events and first_protected_contact_step is None:
                    first_protected_contact_step = step
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
                terminal_frame = _processed_image(observation, "agentview_image")
                video_writer.append_data(terminal_frame)
                action_records.append(
                    {
                        "step": step,
                        "source": "verified_five_action_task_rejoining_detour",
                        "action": action.tolist(),
                        "reward": float(reward),
                        "done": bool(done),
                        "goal_progress": goal,
                        "clone_state_max_abs_error": state_error,
                        "robot_contact_events": robot_events,
                        "protected_contact_events": protected_events,
                        "active_obstacle_l1_displacement_m": displacement,
                    }
                )
        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
        server_identity = _server_identity(client)
        policy_queries = []
        action_plan = collections.deque()
        proxy = _eef_proxy(runtime, observation)
        p2 = np.asarray(perception["mvee_center"], dtype=np.float64)
        z_fixed = p2 - np.asarray(stale_proxy["p1"], dtype=np.float64)
        z_fixed /= np.linalg.norm(z_fixed)
        released_geometry = {
            "p2": p2,
            "R2": np.asarray(perception["mvee_rotation"], dtype=np.float64),
            "Q2_diag": np.asarray(perception["mvee_semiaxes"], dtype=np.float64),
            "z_fixed": z_fixed,
        }
        released_geometry["z_fixed"] = np.asarray(
            archived_actions[186]["qp"]["z_after"], dtype=np.float64
        )
        q1_diag = np.asarray([0.06, 0.12, 0.11], dtype=np.float64)
        native_success = False
        if failure is None:
            for step in range(187, max_steps_for_case(case)):
                if not action_plan:
                    query_index = (
                        int(policy_queries[-1]["query_index"]) + 1
                        if policy_queries else 37
                    )
                    seed = query_seed(int(case["policy_noise_seed"]), query_index)
                    policy_input = _policy_observation(
                        runtime,
                        observation,
                        task_description=str(task.language),
                        resize_size=224,
                        rng_seed=seed,
                    )
                    query_started = time.perf_counter_ns()
                    response = client.infer(policy_input)
                    query_wall = (time.perf_counter_ns() - query_started) * 1.0e-9
                    returned = np.asarray(response["actions"], dtype=np.float64)
                    _require(returned.shape == (int(case["model_action_horizon"]), 7), "live chunk differs")
                    action_plan.extend(returned[index].copy() for index in range(int(case["replan_steps"])))
                    policy_queries.append(
                        {
                            "query_index": query_index,
                            "step": step,
                            "rng_seed": seed,
                            "returned_actions_sha256": array_sha256(returned),
                            "wall_seconds": query_wall,
                            "server_timing": response.get("server_timing"),
                        }
                    )
                raw = np.asarray(action_plan.popleft(), dtype=np.float64)
                nominal, qp_record = _aegis_action(
                    runtime,
                    nominal_translational=translational_action(raw),
                    proxy=proxy,
                    geometry=released_geometry,
                    q1_diag=q1_diag,
                    diagnostics_enabled=True,
                )
                observation, reward, done, _ = env.step(nominal)
                proxy = _eef_proxy(runtime, observation)
                goal = _goal_progress_snapshot(env, goal_atoms, step=step, previous_values=previous_goal_values)
                previous_goal_values = goal["values"]
                contacts = _detailed_active_obstacle_contacts(
                    env, obstacle_name, step=step, contact_authority=contact_authority
                )
                robot_events = [
                    event for event in contacts["events"]
                    if event.get("other", {}).get("classification") == "robot"
                ]
                protected_events = [event for event in robot_events if _is_protected_event(event)]
                if robot_events and first_robot_contact_step is None:
                    first_robot_contact_step = step
                if protected_events and first_protected_contact_step is None:
                    first_protected_contact_step = step
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
                terminal_frame = _processed_image(observation, "agentview_image")
                video_writer.append_data(terminal_frame)
                action_records.append(
                    {
                        "step": step,
                        "source": "live_pi05_replan_five_steps_then_released_aegis",
                        "raw": raw.tolist(),
                        "action": list(nominal),
                        "released_aegis_qp": qp_record,
                        "reward": float(reward),
                        "done": bool(done),
                        "goal_progress": goal,
                        "robot_contact_events": robot_events,
                        "protected_contact_events": protected_events,
                        "active_obstacle_l1_displacement_m": displacement,
                    }
                )
                if done:
                    native_success = True
                    break
        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        native_success = bool(goal_summary["first_all_satisfied_step"] is not None)
        problem_solved = bool(
            failure is None
            and detour["gate_pass"]
            and first_robot_contact_step is None
            and first_protected_contact_step is None
            and first_car_step is None
            and native_success
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete" if failure is None else "method_failure",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": ARCHIVED_FILE_SHA256,
                "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
                "read_only": True,
            },
            "config": config,
            "geometry_config": geometry_config,
            "geometry": geometry.geometry_record(env),
            "pairing": pairing,
            "probe_environment": {
                "disabled_image_observable_count": disabled_probe_images,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
            },
            "detour": detour,
            "policy_server": server_identity,
            "policy_query_count": len(policy_queries),
            "policy_queries": policy_queries,
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {**goal_definition, "initial": initial_goal, "summary": goal_summary},
            "raw_simulation_evidence": {
                "first_robot_contact_step": first_robot_contact_step,
                "first_protected_link_contact_step": first_protected_contact_step,
                "first_paper_car_step": first_car_step,
                "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
                "native_task_success": native_success,
                "native_task_success_step": goal_summary["first_all_satisfied_step"],
            },
            "video": {
                "path": str(video_final),
                "file_sha256": _file_sha256(video_final),
                "frames_written": len(action_records) + 1,
                "fps": TABLE_VIDEO_FPS,
            },
            "final_jpg": {"path": str(final_jpg), "file_sha256": _file_sha256(final_jpg)},
            "primary_problem_solved": problem_solved,
            "failure": failure,
            "interpretation": (
                "early_five_action_detour_safe_task_success"
                if problem_solved else (
                    "early_five_action_detour_no_safe_rejoining_support"
                    if not detour["gate_pass"] else (
                        "early_five_action_detour_collision_or_car"
                        if first_robot_contact_step is not None or first_car_step is not None
                        else "early_five_action_detour_safe_but_task_failed"
                    )
                )
            ),
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        return result
    finally:
        if video_writer is not None:
            video_writer.close()
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
        host=args.host,
        port=args.port,
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_problem_solved": result["primary_problem_solved"],
                "interpretation": result["interpretation"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
