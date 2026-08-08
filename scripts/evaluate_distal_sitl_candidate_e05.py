#!/usr/bin/env python3
"""Run the exact cloned-OSC L5--L7 candidate heuristic on primary E05."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

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


def _disable_probe_images(env: Any) -> int:
    disabled = 0
    for observable in env.env._observables.values():
        if str(getattr(observable, "modality", "")) == "image":
            observable.set_enabled(False)
            disabled += 1
    return disabled


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    geometry_config_path: Path,
    heuristic_config_path: Path,
    expected_commit: str,
    output_path: Path,
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
        _processed_image,
        _runtime_imports,
        _settle,
        array_sha256,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import (
        DistalSitlCandidateFilter,
        load_sitl_candidate_config,
        summarize_sitl_steps,
    )

    started = time.perf_counter_ns()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "archived Table-1 file hash differs")
    _require(
        archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
        "archived Table-1 payload identity differs",
    )
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list)
        and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "archived successful AEGIS action horizon differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary SITL manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    geometry_config = load_shadow_config(geometry_config_path)
    heuristic_config = load_sitl_candidate_config(heuristic_config_path)
    _require(geometry_config["case_ids"] == [CASE_ID], "geometry config case differs")
    _require(heuristic_config["case_ids"] == [CASE_ID], "heuristic config case differs")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
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
            "probe initial state differs",
        )
        _require(
            np.array_equal(
                np.asarray(probe_env.sim.get_state().flatten()),
                np.asarray(env.sim.get_state().flatten()),
            ),
            "settled probe simulator state differs",
        )
        disabled_probe_observables = _disable_probe_images(probe_env)
        obstacle_name, _ = _active_obstacle(env, observation)
        probe_obstacle_name, _ = _active_obstacle(probe_env, probe_observation)
        _require(probe_obstacle_name == obstacle_name, "probe active obstacle differs")
        _require(obstacle_name == archived["obstacle"]["active_name"], "active obstacle differs from Table 1")
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        settled_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=settled_state,
        )
        for key in (
            "manifest_row_sha256",
            "initial_state_sha256",
            "initial_observation_sha256",
            "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == archived["pairing"][key], "SITL pairing field differs: %s" % key)

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
        geometry_record = geometry.geometry_record(env)
        _require(geometry_record["distal_ellipsoid_count"] == 7, "SITL distal geometry count differs")
        _require(geometry_record["total_constraint_geometry_count"] == 8, "SITL total geometry count differs")
        controller = DistalSitlCandidateFilter(heuristic_config, geometry, probe_env)

        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(env, goal_atoms, step=-1, previous_values=None)
        previous_goal_values = initial_goal["values"]
        contact_authority = _contact_model_authority(env, obstacle_name)
        action_records = []
        filter_records = []
        direct_robot_contact_geoms = set()
        direct_protected_contact_geoms = set()
        first_robot_contact_step = None
        first_protected_contact_step = None
        first_car_step = None
        maximum_displacement = 0.0
        failure = None
        terminal_frame = _processed_image(observation, "agentview_image")
        frames_written = 0
        if video_partial.exists():
            video_partial.unlink()
        video_writer = runtime["imageio"].get_writer(
            str(video_partial),
            fps=TABLE_VIDEO_FPS,
            codec="libx264",
            macro_block_size=None,
            pixelformat="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        video_writer.append_data(terminal_frame)
        frames_written += 1

        for index, archived_action in enumerate(archived_actions):
            _require(int(archived_action["step"]) == index, "archived action indexes differ")
            nominal = np.asarray(archived_action["env_step_input"], dtype=np.float64)
            _require(
                nominal.shape == (7,) and np.array_equal(nominal, np.asarray(archived_action["executed"])),
                "archived env.step input binding differs",
            )
            executed, filter_step = controller.filter(env, nominal, step=index)
            filter_records.append(filter_step)
            if executed is None:
                failure = {
                    "component": "distal_sitl_candidate_filter",
                    "step": index,
                    "reason": "no_exactly_verified_candidate",
                }
                break
            observation, reward, done, _ = env.step(executed)
            try:
                controller.verify_executed_transition(env, filter_step)
            except ValueError as error:
                failure = {
                    "component": "cloned_osc_fidelity",
                    "step": index,
                    "reason": str(error),
                }
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=index, previous_values=previous_goal_values
            )
            previous_goal_values = goal["values"]
            _require(goal["all_satisfied"] is bool(done), "native goal vector differs")
            contacts = _detailed_active_obstacle_contacts(
                env, obstacle_name, step=index, contact_authority=contact_authority
            )
            _require(contacts["status"] == "available", "active contact evidence unavailable")
            robot_events = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            protected_events = [event for event in robot_events if _is_protected_event(event)]
            if robot_events and first_robot_contact_step is None:
                first_robot_contact_step = index
            if protected_events and first_protected_contact_step is None:
                first_protected_contact_step = index
            for event in robot_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_robot_contact_geoms.add(name)
            for event in protected_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_protected_contact_geoms.add(name)
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
                first_car_step = index
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            frames_written += 1
            action_records.append(
                {
                    "step": index,
                    "archived_aegis_action_sha256": _sha256(
                        json.dumps(nominal.tolist(), separators=(",", ":"), allow_nan=False).encode("utf-8")
                    ),
                    "nominal_archived_aegis_action": nominal.tolist(),
                    "executed_sitl_action": list(executed),
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                    "eef_position_m": np.asarray(observation["robot0_eef_pos"], dtype=np.float64).tolist(),
                    "active_obstacle_l1_displacement_m": displacement,
                    "robot_contact_events": robot_events,
                    "protected_link_contact_events": protected_events,
                    "filter": filter_step,
                }
            )
            if failure is not None or done:
                break

        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        summary = summarize_sitl_steps(filter_records)
        native_success_step = goal_summary["first_all_satisfied_step"]
        native_success = native_success_step is not None
        paper_car_pass = first_car_step is None
        protected_contact_pass = first_protected_contact_step is None
        problem_solved = bool(
            failure is None
            and summary["status"] == "complete"
            and summary["all_executed_transitions_match_clone"]
            and protected_contact_pass
            and paper_car_pass
            and native_success
        )
        result = {
            "schema_version": "vlsa_distal_sitl_candidate_e05_result.v1",
            "status": "complete" if failure is None else "method_failure",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": heuristic_config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": heuristic_config,
            "geometry_config": geometry_config,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": ARCHIVED_FILE_SHA256,
                "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
                "read_only": True,
                "original_outcome": {
                    "first_link5_contact_step": 187,
                    "first_paper_car_step": 188,
                    "native_task_success_step": 236,
                },
            },
            "pairing": pairing,
            "probe_environment": {
                "same_bddl_task_episode_and_settled_state": True,
                "disabled_image_observable_count": disabled_probe_observables,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
            },
            "geometry": geometry_record,
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {**goal_definition, "initial": initial_goal, "summary": goal_summary},
            "raw_simulation_evidence": {
                "first_robot_contact_step": first_robot_contact_step,
                "first_protected_link_contact_step": first_protected_contact_step,
                "direct_robot_contact_geoms": sorted(direct_robot_contact_geoms),
                "direct_protected_link_contact_geoms": sorted(direct_protected_contact_geoms),
                "first_paper_car_step": first_car_step,
                "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
                "paper_car_pass": paper_car_pass,
                "protected_link_contact_pass": protected_contact_pass,
                "native_task_success": native_success,
                "native_task_success_step": native_success_step,
            },
            "filter_summary": summary,
            "video": {
                "path": str(video_final),
                "file_sha256": _file_sha256(video_final),
                "frames_written": frames_written,
                "fps": TABLE_VIDEO_FPS,
            },
            "final_jpg": {
                "path": str(final_jpg),
                "file_sha256": _file_sha256(final_jpg),
                "frame_sha256": array_sha256(terminal_frame),
            },
            "primary_problem_solved": problem_solved,
            "failure": failure,
            "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
        }
        result["result_payload_sha256"] = _sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        )
        return result
    finally:
        if video_writer is not None:
            video_writer.close()
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--heuristic-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        heuristic_config_path=args.heuristic_config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_problem_solved": result["primary_problem_solved"],
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
