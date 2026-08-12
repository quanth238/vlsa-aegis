#!/usr/bin/env python3
"""Replay the registered complete E05 compound trajectory with internal checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    _archived_actions,
    _public,
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


RESULT_SCHEMA = "vlsa_distal_full_compound_replay_e05_result.v1"


def _trace_gate(record: Mapping[str, Any], gate: Mapping[str, Any], task_success: bool) -> bool:
    return bool(
        float(record["minimum_clearance_m"])
        >= float(gate["internal_substep_clearance_buffer_m"])
        and len(record["protected_contacts"]) == int(gate["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
        and (not bool(gate["require_native_task_success"]) or bool(task_success))
    )


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    detour_path: Path,
    smooth_path: Path,
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
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
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
    config = json.loads(experiment_config_path.read_text())
    registered = config["registered_inputs"]
    detour = _validate_registered(
        detour_path,
        registered["detour_result"],
        "vlsa_distal_five_action_detour_e05_result.v1",
    )
    smooth = _validate_registered(
        smooth_path,
        registered["smooth_result"],
        "vlsa_distal_multi_witness_counterfactual_e05_result.v1",
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
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(geometry_config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_partial = output_path.with_name("episode.partial.mp4")
    video_final = output_path.with_name("episode.mp4")
    final_jpg = output_path.with_name("final.jpg")
    env = None
    video_writer = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
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
        clearance_probe = SlabbedEightConstraintProbe(
            env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        obstacle_id = _obstacle_root_body_id(env.sim.model, obstacle_name)
        contact_authority = _contact_model_authority(env, obstacle_name)
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(env, goal_atoms, step=-1, previous_values=None)
        previous_goal_values = initial_goal["values"]

        expected_count = int(config["compound_definition"]["expected_action_count"])
        actions = _result_actions(detour, 0, expected_count - 1)
        archived_prefix = _archived_actions(archived, 0, 181)
        _require(np.array_equal(actions[:182], archived_prefix), "detour prefix differs from Table 1")
        correction = np.asarray(
            smooth["arms"]["smooth_max"]["best"]["correction"], dtype=np.float64
        ).reshape(5, 3)
        base_actions = actions.copy()
        actions[182:187, :3] += correction
        limit = float(config["compound_definition"]["action_limit"])
        _require(float(np.max(np.abs(actions[182:187, :3]))) <= limit, "compound action clips")

        clearances = []
        displacement_trace = []
        contacts = []
        sample_identities = []
        row_boundary_errors = []
        substep_counts = []
        action_records = []

        def measure(step: int, substep: int) -> None:
            values = np.asarray(clearance_probe.clearances(env)[:7], dtype=np.float64)
            clearances.append(values)
            sample_identities.append({"step": int(step), "substep": int(substep)})
            evidence = _protected_contact_evidence(env, obstacle_name)
            for event in evidence["events"]:
                contacts.append({"step": int(step), "substep": int(substep), **event})
            obstacle_position = np.asarray(env.sim.data.xpos[obstacle_id], dtype=np.float64)
            displacement_trace.append(float(np.sum(np.abs(obstacle_position - initial_obstacle_position))))

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
        done = False
        expected_substeps = int(config["internal_verification"]["expected_mujoco_substeps_per_action"])
        boundary_tolerance = float(
            config["internal_verification"]["ordinary_env_step_boundary_equivalence_tolerance"]
        )
        for step, command in enumerate(actions):
            count_before = len(clearances)
            original_step = env.sim.step

            def instrumented_step(*args: Any, **kwargs: Any) -> Any:
                output = original_step(*args, **kwargs)
                measure(step, len(clearances) - count_before)
                return output

            env.sim.step = instrumented_step
            try:
                observation, reward, done, _ = env.step(command.tolist())
            finally:
                env.sim.step = original_step
            count = len(clearances) - count_before
            _require(count == expected_substeps, "MuJoCo substep count differs")
            substep_counts.append(count)
            boundary = np.asarray(clearance_probe.clearances(env)[:7], dtype=np.float64)
            error = float(np.max(np.abs(boundary - clearances[-1])))
            _require(error <= boundary_tolerance, "internal/boundary clearance differs")
            row_boundary_errors.append(error)
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=step, previous_values=previous_goal_values
            )
            previous_goal_values = goal["values"]
            detailed = _detailed_active_obstacle_contacts(
                env, obstacle_name, step=step, contact_authority=contact_authority
            )
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            action_records.append(
                {
                    "step": step,
                    "source": (
                        "registered_compound_correction"
                        if 182 <= step <= 186
                        else (
                            "immutable_archived_aegis_prefix"
                            if step < 182
                            else "registered_task_successful_detour_continuation"
                        )
                    ),
                    "action": command.tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                    "boundary_active_obstacle_contacts": detailed["events"],
                }
            )
            if done:
                break
        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        trace = np.asarray(clearances, dtype=np.float64)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        task_success = goal_summary["first_all_satisfied_step"] is not None
        trace_record = {
            "clearance_trace_m": _public(trace),
            "minimum_clearance_m": float(np.min(trace)),
            "row_minimum_clearance_m": _public(np.min(trace, axis=0)),
            "minimum_identity": sample_identities[int(np.argmin(trace) // trace.shape[1])],
            "protected_contacts": contacts,
            "active_obstacle_l1_displacement_trace_m": displacement_trace,
            "maximum_active_obstacle_l1_displacement_m": float(max(displacement_trace)),
            "sample_count": int(trace.shape[0]),
            "sample_identities": sample_identities,
            "substep_counts": substep_counts,
            "maximum_boundary_equivalence_error_m": float(max(row_boundary_errors)),
        }
        passed = _trace_gate(trace_record, config["gate"], task_success)
        first_contact = min((item["step"] for item in contacts), default=None)
        first_car = next(
            (
                sample_identities[index]["step"]
                for index, value in enumerate(displacement_trace)
                if value > PAPER_CAR_THRESHOLD_M
            ),
            None,
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
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
            "registered_inputs": {
                "detour_payload_sha256": detour["result_payload_sha256"],
                "smooth_payload_sha256": smooth["result_payload_sha256"],
            },
            "pairing": pairing,
            "geometry_config": geometry_config,
            "geometry": geometry.geometry_record(env),
            "probe_environment": {
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
                "model_timestep_s": float(env.env.model_timestep),
                "control_timestep_s": float(env.env.control_timestep),
            },
            "compound": {
                "base_action_count": int(base_actions.shape[0]),
                "correction": correction.tolist(),
                "correction_l2_action": float(np.linalg.norm(correction)),
                "clipped_coordinate_count": 0,
                "base_actions_182_186": base_actions[182:187].tolist(),
                "compound_actions_182_186": actions[182:187].tolist(),
            },
            "trace": trace_record,
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {**goal_definition, "initial": initial_goal, "summary": goal_summary},
            "raw_simulation_evidence": {
                "first_protected_link_contact_step": first_contact,
                "first_paper_car_step": first_car,
                "maximum_active_obstacle_l1_displacement_m": trace_record[
                    "maximum_active_obstacle_l1_displacement_m"
                ],
                "native_task_success": task_success,
                "native_task_success_step": goal_summary["first_all_satisfied_step"],
            },
            "video": {
                "path": str(video_final),
                "file_sha256": _file_sha256(video_final),
                "frames_written": len(action_records) + 1,
                "fps": TABLE_VIDEO_FPS,
            },
            "final_jpg": {"path": str(final_jpg), "file_sha256": _file_sha256(final_jpg)},
            "primary_problem_solved": passed,
            "interpretation": (
                "complete_compound_positive_control_safe_task_success"
                if passed
                else "complete_compound_positive_control_no_go"
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
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--detour-result", type=Path, required=True)
    parser.add_argument("--smooth-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        detour_path=args.detour_result.resolve(),
        smooth_path=args.smooth_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({"status": value["status"], "primary_problem_solved": value["primary_problem_solved"], "result_payload_sha256": value["result_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
