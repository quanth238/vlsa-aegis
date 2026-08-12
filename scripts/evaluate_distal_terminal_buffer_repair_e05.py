#!/usr/bin/env python3
"""Search an exact task-preserving micro-repair for E05 action 205."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_receding_route_oracle_e05 import _internal_summary
from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _disable_images,
    _result_actions,
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


RESULT_SCHEMA = "vlsa_distal_terminal_buffer_repair_e05_result.v1"


def _safe(record: Mapping[str, Any], goal: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    return bool(
        float(record["minimum_clearance_m"])
        >= float(gate["internal_substep_clearance_buffer_m"])
        and len(record["protected_contacts"]) == int(gate["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
        and (not bool(gate["require_native_goal_after_candidate"]) or bool(goal["all_satisfied"]))
    )


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    full_compound_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
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
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _processed_image,
        _runtime_imports,
        _settle,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = json.loads(experiment_config_path.read_text())
    full = _validate_registered(
        full_compound_path,
        config["registered_inputs"]["full_compound_result"],
        "vlsa_distal_full_compound_replay_e05_result.v1",
    )
    _require(full["raw_simulation_evidence"]["native_task_success"] is True, "registered task did not succeed")
    _require(not full["trace"]["protected_contacts"], "registered compound has contact")
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    matches = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(geometry_config_path)
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
        initial_obstacle_position = np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64).copy()
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
            _require(pairing[key] == full["pairing"][key], "pairing differs: %s" % key)
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
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        disabled_images = _disable_images(probe_env)
        instrumented = InstrumentedContinuationProbe(one_step, obstacle_name, initial_obstacle_position)
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(env, goal_atoms, step=-1, previous_values=None)
        previous_goal_values = initial_goal["values"]
        complete_actions = _result_actions(full, 0, 205)
        video_writer = runtime["imageio"].get_writer(
            str(video_partial), fps=TABLE_VIDEO_FPS, codec="libx264", macro_block_size=None,
            pixelformat="yuv420p", output_params=["-crf", "18", "-movflags", "+faststart"]
        )
        terminal_frame = _processed_image(observation, "agentview_image")
        video_writer.append_data(terminal_frame)
        action_records = []
        for step, command in enumerate(complete_actions[:205]):
            observation, reward, done, _ = env.step(command.tolist())
            _require(not done, "task completed before registered repair step")
            goal = _goal_progress_snapshot(env, goal_atoms, step=step, previous_values=previous_goal_values)
            previous_goal_values = goal["values"]
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            action_records.append({"step": step, "action": command.tolist(), "reward": float(reward), "done": bool(done), "goal_progress": goal})
        expected_state_hash = full["actions"][204]["goal_progress"]["simulator_state_sha256_after"]
        _require(action_records[-1]["goal_progress"]["simulator_state_sha256_after"] == expected_state_hash, "pre-repair replay state differs")
        nominal = complete_actions[205].copy()
        limit = float(config["candidate_search"]["action_limit"])
        directions = []
        for raw in itertools.product((-1.0, 0.0, 1.0), repeat=3):
            vector = np.asarray(raw, dtype=np.float64)
            if np.linalg.norm(vector) == 0.0:
                continue
            directions.append(vector / np.linalg.norm(vector))
        candidates = []

        def evaluate_candidate(action: Any, source_name: str, radius: float, direction_index: Optional[int]) -> dict[str, Any]:
            record = instrumented.rollout_internal(
                env,
                np.asarray(action, dtype=np.float64).reshape(1, 7),
                expected_substeps=int(config["internal_verification"]["expected_mujoco_substeps_per_action"]),
                boundary_tolerance=float(config["internal_verification"]["ordinary_env_step_boundary_equivalence_tolerance"]),
                step_base=205,
            )
            goal = _goal_progress_snapshot(
                instrumented.env, goal_atoms, step=205, previous_values=previous_goal_values
            )
            applied = np.asarray(action, dtype=np.float64)[:3] - nominal[:3]
            return {
                "source": source_name,
                "radius_action": float(radius),
                "direction_index": direction_index,
                "action": np.asarray(action, dtype=np.float64).tolist(),
                "applied_correction": applied.tolist(),
                "applied_correction_l2_action": float(np.linalg.norm(applied)),
                "record": _internal_summary(record),
                "goal": goal,
                "verification_gate": _safe(record, goal, config["gate"]),
            }

        candidates.append(evaluate_candidate(nominal, "registered_nominal", 0.0, None))
        seen = {tuple(nominal.tolist())}
        for radius in config["candidate_search"]["radii_action"]:
            for index, direction in enumerate(directions):
                action = nominal.copy()
                action[:3] = np.clip(action[:3] + float(radius) * direction, -limit, limit)
                key = tuple(action.tolist())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(evaluate_candidate(action, "local_cartesian_grid", float(radius), index))
        safe = [item for item in candidates if item["verification_gate"]]
        selected = (
            None
            if not safe
            else min(
                safe,
                key=lambda item: (
                    float(item["applied_correction_l2_action"]),
                    -float(item["record"]["minimum_clearance_m"]),
                ),
            )
        )
        executed_goal = None
        clone_state_error = None
        if selected is not None:
            expected = one_step.transition(env, selected["action"])
            observation, reward, done, _ = env.step(selected["action"])
            executed_goal = _goal_progress_snapshot(env, goal_atoms, step=205, previous_values=previous_goal_values)
            actual = _dynamic_state_vector(env)
            clone_state_error = float(np.max(np.abs(actual - np.asarray(expected["next_state_vector"]))))
            _require(clone_state_error <= 1.0e-10, "executed repair differs from clone")
            _require(executed_goal["all_satisfied"] and done, "verified repair did not complete task")
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            action_records.append({"step": 205, "action": selected["action"], "reward": float(reward), "done": bool(done), "goal_progress": executed_goal})
        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        solved = bool(selected is not None and goal_summary["first_all_satisfied_step"] == 205)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "archived_table1": {"path": str(archived_path), "file_sha256": ARCHIVED_FILE_SHA256, "payload_sha256": ARCHIVED_PAYLOAD_SHA256, "read_only": True},
            "registered_inputs": {"full_compound_payload_sha256": full["result_payload_sha256"]},
            "pairing": pairing,
            "geometry_config": geometry_config,
            "geometry": geometry.geometry_record(env),
            "probe_environment": {"disabled_image_observable_count": disabled_images, "osc_controller": "OSC_POSE", "control_frequency_hz": 20},
            "repair_step": 205,
            "nominal_action": nominal.tolist(),
            "candidate_count": len(candidates),
            "safe_candidate_count": len(safe),
            "candidates": candidates,
            "selected": selected,
            "executed_clone_state_max_abs_error": clone_state_error,
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {**goal_definition, "initial": initial_goal, "summary": goal_summary},
            "video": {"path": str(video_final), "file_sha256": _file_sha256(video_final), "frames_written": len(action_records) + 1, "fps": TABLE_VIDEO_FPS},
            "final_jpg": {"path": str(final_jpg), "file_sha256": _file_sha256(final_jpg)},
            "primary_problem_solved": solved,
            "interpretation": "terminal_micro_repair_safe_task_success" if solved else "terminal_micro_repair_no_go",
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
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
    parser.add_argument("--full-compound-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = evaluate(
        repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(), full_compound_path=args.full_compound_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(), experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit, output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({"primary_problem_solved": value["primary_problem_solved"], "safe_candidate_count": value["safe_candidate_count"], "result_payload_sha256": value["result_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
