#!/usr/bin/env python3
"""Run the exact receding five-action task-rejoining oracle on E05."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Optional, Sequence

from scripts.evaluate_distal_five_action_detour_e05 import (
    FiveActionProbe,
    _archived_action,
    _disable_images,
    _public_rollout,
    _search_detour,
)
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


RESULT_SCHEMA = "vlsa_distal_receding_five_action_e05_result.v1"


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
        _build_environment,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
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
    from main.multilink_ellipsoid.receding_five_action import (
        load_receding_five_action_config,
        receding_query_index,
        rollout_is_safe,
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
    _require(
        _file_sha256(archived_path) == ARCHIVED_FILE_SHA256,
        "Table-1 file hash differs",
    )
    _require(
        archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
        "Table-1 payload differs",
    )
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list)
        and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "Table-1 action horizon differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    config = load_receding_five_action_config(experiment_config_path)
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

        def contact_reader(probe: Any, evidence_step: int) -> dict[str, Any]:
            contacts = _detailed_active_obstacle_contacts(
                probe,
                obstacle_name,
                step=int(evidence_step),
                contact_authority=probe_authority,
            )
            _require(contacts["status"] == "available", "probe contact evidence differs")
            return contacts

        probe = FiveActionProbe(one_step_probe, contact_reader)
        probe.obstacle_reference_position_m = initial_obstacle_position.copy()
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
            _require(not bool(done), "archived prefix completed before receding oracle")
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=step, previous_values=previous_goal_values
            )
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

        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
        server_identity = _server_identity(client)
        policy_queries = []
        window_records = []
        failure = None
        native_success = False
        qp_config = config["sequential_qp"]
        for step in range(182, max_steps_for_case(case)):
            query_record = None
            if step == 182:
                archived_window = np.asarray(
                    [_archived_action(archived_actions, item) for item in range(182, 187)],
                    dtype=np.float64,
                )
                nominal = np.asarray(
                    [translational_action(action) for action in archived_window],
                    dtype=np.float64,
                )
                _require(
                    np.array_equal(nominal, archived_window),
                    "archived initial window is not translational",
                )
                nominal_source = "immutable_archived_aegis_window_182_186"
            else:
                query_index = receding_query_index(step)
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
                _require(
                    returned.shape == (int(case["model_action_horizon"]), 7),
                    "live receding chunk differs",
                )
                nominal = np.asarray(
                    [translational_action(returned[index]) for index in range(5)],
                    dtype=np.float64,
                )
                query_record = {
                    "query_index": query_index,
                    "step": step,
                    "rng_seed": seed,
                    "returned_actions_sha256": array_sha256(returned),
                    "nominal_five_translation_actions_sha256": array_sha256(nominal),
                    "wall_seconds": query_wall,
                    "server_timing": response.get("server_timing"),
                }
                policy_queries.append(query_record)
                nominal_source = "fresh_frozen_pi05_translation_window"

            nominal_rollout = probe.rollout(env, nominal, step_base=step)
            nominal_safe = rollout_is_safe(
                nominal_rollout,
                clearance_buffer_m=float(qp_config["clearance_buffer_m"]),
                paper_car_threshold_m=PAPER_CAR_THRESHOLD_M,
            )
            search = None
            if nominal_safe:
                selected = nominal
                selected_rollout = nominal_rollout
                selected_source = "fresh_exact_safe_nominal_window"
            else:
                search = _search_detour(
                    probe,
                    env,
                    nominal,
                    config,
                    step_base=step,
                )
                if not search["gate_pass"]:
                    failure = {
                        "component": "receding_five_action_oracle",
                        "step": step,
                        "reason": "no_verified_task_rejoining_five_action_candidate",
                    }
                    window_records.append(
                        {
                            "step": step,
                            "nominal_source": nominal_source,
                            "policy_query": query_record,
                            "nominal": _public_rollout(nominal_rollout),
                            "nominal_safe": False,
                            "search": search,
                            "selected_source": None,
                            "executed_prefix_actions": 0,
                        }
                    )
                    break
                selected = np.asarray(search["best"]["actions"], dtype=np.float64)
                selected_rollout = probe.rollout(env, selected, step_base=step)
                _require(
                    rollout_is_safe(
                        selected_rollout,
                        clearance_buffer_m=float(qp_config["clearance_buffer_m"]),
                        paper_car_threshold_m=PAPER_CAR_THRESHOLD_M,
                    ),
                    "fresh selected receding detour verification failed",
                )
                selected_source = "fresh_exact_task_rejoining_detour_window"

            expected_first_state = np.asarray(
                selected_rollout["state_vectors"][0], dtype=np.float64
            )
            executed = np.asarray(selected[0], dtype=np.float64)
            observation, reward, done, _ = env.step(executed.tolist())
            state_error = float(
                np.max(np.abs(_dynamic_state_vector(env) - expected_first_state))
            )
            _require(state_error <= 1.0e-10, "executed receding prefix differs from clone")
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=step, previous_values=previous_goal_values
            )
            previous_goal_values = goal["values"]
            contacts = _detailed_active_obstacle_contacts(
                env, obstacle_name, step=step, contact_authority=contact_authority
            )
            robot_events = [
                event
                for event in contacts["events"]
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
            _require(not robot_events, "verified receding prefix produced robot contact")
            _require(displacement <= PAPER_CAR_THRESHOLD_M, "verified receding prefix failed CAR")
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            window_record = {
                "step": step,
                "nominal_source": nominal_source,
                "policy_query": query_record,
                "nominal": _public_rollout(nominal_rollout),
                "nominal_safe": nominal_safe,
                "search": search,
                "selected_source": selected_source,
                "selected": _public_rollout(selected_rollout),
                "selected_actions": selected.tolist(),
                "executed_prefix_actions": 1,
                "executed_first_action": executed.tolist(),
                "clone_state_max_abs_error": state_error,
            }
            window_records.append(window_record)
            action_records.append(
                {
                    "step": step,
                    "source": selected_source,
                    "action": executed.tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                    "clone_state_max_abs_error": state_error,
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
            and first_robot_contact_step is None
            and first_protected_contact_step is None
            and first_car_step is None
            and native_success
        )
        unsafe_windows = [record for record in window_records if not record["nominal_safe"]]
        corrected_windows = [
            record
            for record in window_records
            if record["selected_source"] == "fresh_exact_task_rejoining_detour_window"
        ]
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
            "policy_server": server_identity,
            "policy_query_count": len(policy_queries),
            "policy_queries": policy_queries,
            "receding_summary": {
                "window_count": len(window_records),
                "unsafe_nominal_window_count": len(unsafe_windows),
                "corrected_window_count": len(corrected_windows),
                "first_unsafe_nominal_step": (
                    None if not unsafe_windows else int(unsafe_windows[0]["step"])
                ),
                "first_corrected_step": (
                    None if not corrected_windows else int(corrected_windows[0]["step"])
                ),
                "total_search_rollouts": int(
                    sum(
                        int(record["search"]["rollout_count"])
                        for record in window_records
                        if record["search"] is not None
                    )
                ),
                "execute_prefix_actions": 1,
            },
            "windows": window_records,
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {
                **goal_definition,
                "initial": initial_goal,
                "summary": goal_summary,
            },
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
            "final_jpg": {
                "path": str(final_jpg),
                "file_sha256": _file_sha256(final_jpg),
            },
            "primary_problem_solved": problem_solved,
            "failure": failure,
            "interpretation": (
                "receding_five_action_oracle_safe_task_success"
                if problem_solved
                else (
                    "receding_five_action_oracle_deadlock"
                    if failure is not None
                    else "receding_five_action_oracle_safe_but_task_failed"
                )
            ),
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _sha256(
            json.dumps(
                result, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
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
