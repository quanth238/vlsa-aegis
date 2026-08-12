#!/usr/bin/env python3
"""Audit live pi0.5 feedback after the near-miss E05 compound scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _disable_images,
    _public,
    _validate_registered,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    PAPER_CAR_THRESHOLD_M,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_scale_feedback_audit_e05_result.v1"


def _goal_xy_margin(goal: Mapping[str, Any]) -> dict[str, float]:
    import numpy as np

    arguments = goal["argument_poses"][0]["arguments"]
    first = np.asarray(arguments[0]["position"], dtype=np.float64)
    second = np.asarray(arguments[1]["position"], dtype=np.float64)
    distance = float(np.linalg.norm(first[:2] - second[:2]))
    return {
        "horizontal_center_distance_m": distance,
        "native_on_horizontal_threshold_m": 0.03,
        "horizontal_threshold_margin_m": 0.03 - distance,
        "vertical_offset_m": float(first[2] - second[2]),
    }


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    scale_result_path: Path,
    scale_label: str,
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
    from main.multilink_ellipsoid.scale_feedback_audit import (
        load_scale_feedback_audit_config,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import (
        SlabbedEightConstraintProbe,
        _obstacle_root_body_id,
        _protected_contact_evidence,
    )

    started = time.perf_counter_ns()
    config = load_scale_feedback_audit_config(experiment_config_path)
    _require(scale_label in config["registered_inputs"], "scale arm differs")
    expected_scale = config["registered_inputs"][scale_label]
    registered = _validate_registered(
        scale_result_path,
        expected_scale,
        "vlsa_distal_full_compound_replay_e05_result.v1",
    )
    _require(
        float(registered["compound"]["correction_scale"])
        == float(expected_scale["correction_scale"]),
        "registered correction scale differs",
    )
    _require(
        registered["raw_simulation_evidence"]["native_task_success"] is False
        and registered["raw_simulation_evidence"]["first_protected_link_contact_step"]
        == 232,
        "registered scale outcome differs",
    )
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    matches = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    geometry_config = load_shadow_config(geometry_config_path)
    runtime = _runtime_imports(include_aegis=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_partial = output_path.with_name("episode.partial.mp4")
    video_final = output_path.with_name("episode.mp4")
    final_jpg = output_path.with_name("final.jpg")
    env = None
    probe_env = None
    video_writer = None
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
        _require(np.array_equal(probe_initial_state, selected_initial_state), "probe initial state differs")
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
            _require(pairing[key] == registered["pairing"][key], "pairing differs: %s" % key)
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
        primary_clearance = SlabbedEightConstraintProbe(
            env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        disabled_images = _disable_images(probe_env)
        instrumented = InstrumentedContinuationProbe(
            one_step, obstacle_name, initial_obstacle_position
        )
        obstacle_id = _obstacle_root_body_id(env.sim.model, obstacle_name)
        contact_authority = _contact_model_authority(env, obstacle_name)
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(env, goal_atoms, step=-1, previous_values=None)
        previous_goal_values = initial_goal["values"]

        clearances = []
        sample_identities = []
        protected_contacts = []
        displacement_trace = []
        substep_counts = []
        boundary_errors = []

        def measure(step: int, substep: int) -> None:
            values = np.asarray(primary_clearance.clearances(env)[:7], dtype=np.float64)
            clearances.append(values)
            sample_identities.append({"step": int(step), "substep": int(substep)})
            evidence = _protected_contact_evidence(env, obstacle_name)
            for event in evidence["events"]:
                protected_contacts.append({"step": int(step), "substep": int(substep), **event})
            obstacle = np.asarray(env.sim.data.xpos[obstacle_id], dtype=np.float64)
            displacement_trace.append(
                float(np.sum(np.abs(obstacle - initial_obstacle_position)))
            )

        env.sim.forward()
        measure(-1, -1)
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
        expected_substeps = int(
            config["internal_verification"]["expected_mujoco_substeps_per_action"]
        )
        boundary_tolerance = float(
            config["internal_verification"][
                "ordinary_env_step_boundary_equivalence_tolerance"
            ]
        )
        live_execution_tolerance = float(
            config["internal_verification"][
                "live_execution_clearance_equivalence_tolerance"
            ]
        )

        def execute(command: Any, step: int, source_name: str) -> tuple[bool, dict[str, Any]]:
            nonlocal observation, previous_goal_values, terminal_frame
            count_before = len(clearances)
            original_step = env.sim.step

            def instrumented_step(*args: Any, **kwargs: Any) -> Any:
                output = original_step(*args, **kwargs)
                measure(step, len(clearances) - count_before)
                return output

            env.sim.step = instrumented_step
            try:
                observation, reward, done, _ = env.step(np.asarray(command).tolist())
            finally:
                env.sim.step = original_step
            count = len(clearances) - count_before
            _require(count == expected_substeps, "MuJoCo substep count differs")
            substep_counts.append(count)
            boundary = np.asarray(primary_clearance.clearances(env)[:7], dtype=np.float64)
            error = float(np.max(np.abs(boundary - clearances[-1])))
            _require(error <= boundary_tolerance, "internal/boundary clearance differs")
            boundary_errors.append(error)
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=step, previous_values=previous_goal_values
            )
            previous_goal_values = goal["values"]
            detailed = _detailed_active_obstacle_contacts(
                env, obstacle_name, step=step, contact_authority=contact_authority
            )
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            record = {
                "step": int(step),
                "source": source_name,
                "action": np.asarray(command, dtype=np.float64).tolist(),
                "reward": float(reward),
                "done": bool(done),
                "goal_progress": goal,
                "boundary_active_obstacle_contacts": detailed["events"],
            }
            action_records.append(record)
            return bool(done), record

        prelude_records = {
            int(item["step"]): item
            for item in registered["actions"]
            if int(item["step"]) <= int(config["state_protocol"]["prelude_last_step"])
        }
        _require(set(prelude_records) == set(range(206)), "registered prelude action range differs")
        for step in range(206):
            done, _ = execute(
                prelude_records[step]["action"], step, "registered_compound_scale_prelude"
            )
            _require(not done, "registered scale prelude unexpectedly completed task")
        registered_trace = np.asarray(
            registered["trace"]["clearance_trace_m"], dtype=np.float64
        )[: len(clearances)]
        prelude_trace = np.asarray(clearances, dtype=np.float64)
        prelude_trace_error = float(np.max(np.abs(prelude_trace - registered_trace)))
        _require(
            prelude_trace_error
            <= float(
                config["internal_verification"]["registered_prelude_trace_tolerance"]
            ),
            "registered scale prelude trace differs",
        )
        goal_205 = action_records[-1]["goal_progress"]
        _require(goal_205["all_satisfied"] is False, "registered near-miss unexpectedly satisfies task")
        goal_205_geometry = _goal_xy_margin(goal_205)

        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, int(port))
        server_identity = _server_identity(client)
        p2 = np.asarray(perception["mvee_center"], dtype=np.float64)
        released_geometry = {
            "p2": p2,
            "R2": np.asarray(perception["mvee_rotation"], dtype=np.float64),
            "Q2_diag": np.asarray(perception["mvee_semiaxes"], dtype=np.float64),
            "z_fixed": np.asarray(archived["actions"][186]["qp"]["z_after"], dtype=np.float64),
        }
        q1_diag = np.asarray([0.06, 0.12, 0.11], dtype=np.float64)
        policy_queries = []
        windows = []
        failure = None
        step = int(config["state_protocol"]["live_query_start_step"])
        query_index = int(config["state_protocol"]["live_query_first_index"])
        horizon = int(config["state_protocol"]["live_query_action_horizon"])
        buffer_m = float(config["gate"]["internal_substep_clearance_buffer_m"])
        while step < max_steps_for_case(case):
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
                returned.shape == (int(case["model_action_horizon"]), 7)
                and np.all(np.isfinite(returned)),
                "live pi0.5 chunk differs",
            )
            one_step.synchronize(env)
            virtual_observation = instrumented.env.env._get_observations()
            proxy = _eef_proxy(runtime, virtual_observation)
            nominal_actions = []
            aegis_records = []
            for raw in returned[:horizon]:
                nominal, qp_record = _aegis_action(
                    runtime,
                    nominal_translational=translational_action(raw),
                    proxy=proxy,
                    geometry=released_geometry,
                    q1_diag=q1_diag,
                    diagnostics_enabled=True,
                )
                nominal_actions.append(nominal)
                virtual_observation, _, _, _ = instrumented.env.step(nominal)
                proxy = _eef_proxy(runtime, virtual_observation)
                aegis_records.append(qp_record)
            nominal_actions = np.asarray(nominal_actions, dtype=np.float64)
            lookahead = instrumented.rollout_internal(
                env,
                nominal_actions,
                expected_substeps=expected_substeps,
                boundary_tolerance=boundary_tolerance,
                step_base=step,
            )
            lookahead_trace = np.asarray(lookahead["clearance_trace_m"], dtype=np.float64)
            future_minimum = float(np.min(lookahead_trace[1:]))
            predicted_contact = bool(lookahead["protected_contacts"])
            predicted_car = bool(
                float(lookahead["maximum_active_obstacle_l1_displacement_m"])
                > PAPER_CAR_THRESHOLD_M
            )
            query_record = {
                "query_index": query_index,
                "step": step,
                "rng_seed": seed,
                "returned_actions_sha256": array_sha256(returned),
                "nominal_five_actions_sha256": array_sha256(nominal_actions),
                "wall_seconds": query_wall,
                "server_timing": response.get("server_timing"),
            }
            policy_queries.append(query_record)
            window = {
                "step": step,
                "policy_query": query_record,
                "released_aegis_qp_records": aegis_records,
                "actions": nominal_actions.tolist(),
                "lookahead": _public(lookahead),
                "future_minimum_clearance_m": future_minimum,
                "buffer_warning": future_minimum < buffer_m,
                "protected_contact_warning": predicted_contact,
                "paper_car_warning": predicted_car,
                "executed_actions": 0,
            }
            windows.append(window)
            if predicted_contact or predicted_car:
                failure = {
                    "component": "exact_five_action_lookahead",
                    "step": step,
                    "reason": (
                        "predicted_protected_contact"
                        if predicted_contact
                        else "predicted_paper_car"
                    ),
                }
                break
            clone_trace = lookahead_trace
            for offset, command in enumerate(nominal_actions):
                count_before = len(clearances)
                done, record = execute(command, step + offset, "live_pi05_released_aegis")
                actual = np.asarray(clearances[count_before:], dtype=np.float64)
                expected = clone_trace[
                    1 + offset * expected_substeps : 1 + (offset + 1) * expected_substeps
                ]
                clone_error = float(np.max(np.abs(actual - expected)))
                _require(
                    clone_error <= live_execution_tolerance,
                    "live execution differs from exact lookahead: %.17g > %.17g"
                    % (clone_error, live_execution_tolerance),
                )
                record["lookahead_clearance_max_abs_error_m"] = clone_error
                window["executed_actions"] += 1
                if done:
                    break
            if action_records[-1]["done"]:
                break
            step += horizon
            query_index += 1

        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        trace = np.asarray(clearances, dtype=np.float64)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        task_success = goal_summary["first_all_satisfied_step"] is not None
        first_contact = min((item["step"] for item in protected_contacts), default=None)
        first_car = next(
            (
                sample_identities[index]["step"]
                for index, value in enumerate(displacement_trace)
                if value > PAPER_CAR_THRESHOLD_M
            ),
            None,
        )
        minimum_clearance = float(np.min(trace))
        collision_free_task_success = bool(
            task_success and first_contact is None and first_car is None
        )
        strict_pass = bool(
            collision_free_task_success
            and minimum_clearance >= buffer_m
            and failure is None
        )
        first_buffer_warning = next(
            (item["step"] for item in windows if item["buffer_warning"]), None
        )
        first_contact_warning = next(
            (item["step"] for item in windows if item["protected_contact_warning"]), None
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete" if failure is None else "method_failure",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": ARCHIVED_FILE_SHA256,
                "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
                "read_only": True,
            },
            "registered_input": {
                "scale_label": scale_label,
                "correction_scale": float(expected_scale["correction_scale"]),
                "path": str(scale_result_path),
                "result_payload_sha256": registered["result_payload_sha256"],
                "prelude_trace_max_abs_error_m": prelude_trace_error,
            },
            "pairing": pairing,
            "geometry_config": geometry_config,
            "geometry": geometry.geometry_record(env),
            "probe_environment": {
                "disabled_image_observable_count": disabled_images,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
                "model_timestep_s": float(env.env.model_timestep),
                "control_timestep_s": float(env.env.control_timestep),
            },
            "action_205_task_threshold_audit": {
                "native_goal": goal_205,
                **goal_205_geometry,
            },
            "policy_server": server_identity,
            "policy_query_count": len(policy_queries),
            "policy_queries": policy_queries,
            "windows": windows,
            "lookahead_summary": {
                "window_count": len(windows),
                "first_buffer_warning_step": first_buffer_warning,
                "first_protected_contact_warning_step": first_contact_warning,
                "lookahead_prevented_known_contact_execution": failure is not None,
            },
            "trace": {
                "clearance_trace_m": _public(trace),
                "minimum_clearance_m": minimum_clearance,
                "row_minimum_clearance_m": _public(np.min(trace, axis=0)),
                "protected_contacts": protected_contacts,
                "active_obstacle_l1_displacement_trace_m": displacement_trace,
                "maximum_active_obstacle_l1_displacement_m": float(max(displacement_trace)),
                "sample_count": int(trace.shape[0]),
                "sample_identities": sample_identities,
                "substep_counts": substep_counts,
                "maximum_boundary_equivalence_error_m": float(max(boundary_errors)),
            },
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {**goal_definition, "initial": initial_goal, "summary": goal_summary},
            "raw_simulation_evidence": {
                "first_protected_link_contact_step": first_contact,
                "first_paper_car_step": first_car,
                "maximum_active_obstacle_l1_displacement_m": float(max(displacement_trace)),
                "native_task_success": task_success,
                "native_task_success_step": goal_summary["first_all_satisfied_step"],
            },
            "collision_free_task_success": collision_free_task_success,
            "primary_problem_solved": strict_pass,
            "failure": failure,
            "interpretation": (
                "live_feedback_collision_free_task_success_but_buffer_no_go"
                if collision_free_task_success and not strict_pass
                else (
                    "live_feedback_buffered_safe_task_success"
                    if strict_pass
                    else (
                        "five_action_lookahead_warned_and_failed_closed"
                        if failure is not None
                        else "live_feedback_safe_but_task_failed"
                    )
                )
            ),
            "video": {
                "path": str(video_final),
                "file_sha256": _file_sha256(video_final),
                "frames_written": len(action_records) + 1,
                "fps": TABLE_VIDEO_FPS,
            },
            "final_jpg": {"path": str(final_jpg), "file_sha256": _file_sha256(final_jpg)},
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
    parser.add_argument("--scale-result", type=Path, required=True)
    parser.add_argument("--scale-label", required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        scale_result_path=args.scale_result.resolve(),
        scale_label=args.scale_label,
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
        host=args.host,
        port=args.port,
    )
    _atomic_write(args.output.resolve(), value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "scale": value["registered_input"]["correction_scale"],
                "task_success": value["raw_simulation_evidence"]["native_task_success"],
                "primary_problem_solved": value["primary_problem_solved"],
                "interpretation": value["interpretation"],
                "result_payload_sha256": value["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
