#!/usr/bin/env python3
"""Run the task-valid E38 repulsion field through archived task completion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_counterfactual_field_e05 import (
    FixedContinuationProbe,
    _archived_action,
    _disable_images,
)
from scripts.evaluate_distal_repulsion_generalization_pilot import (
    _archived_actions,
    _internal_verify,
    _public,
    _run_smooth,
)
from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _is_protected_event,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_repulsion_task_valid_e38_receding_result.v1"


def _load_config(path: Path) -> dict[str, Any]:
    value = _load(path)
    _require(
        value.get("schema_version")
        == "vlsa_distal_repulsion_task_valid_e38_receding.v1",
        "E38 receding config schema differs",
    )
    _require(
        value.get("protocol_id")
        == "vlsa-distal-repulsion-task-valid-e38-receding-v1",
        "E38 receding protocol differs",
    )
    _require(value["state_protocol"]["start_action"] == 108, "E38 start differs")
    _require(
        value["state_protocol"]["lookahead_actions"] == 20,
        "E38 lookahead differs",
    )
    _require(
        value["state_protocol"]["execute_prefix_max_actions"] == 5,
        "E38 execution prefix differs",
    )
    return value


def _field_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the frozen field settings to the existing implementation contract."""

    return {
        "action_space": config["action_space"],
        "field_estimation": config["field_estimation"],
        "gate": {
            "internal_substep_clearance_buffers_m": list(
                config["gate"]["proxy_certification_buffers_m"]
            ),
            "minimum_heldout_direction_cosine": float(
                config["gate"]["minimum_heldout_direction_cosine"]
            ),
            "minimum_heldout_sign_accuracy": float(
                config["gate"]["minimum_heldout_sign_accuracy"]
            ),
            "paper_car_threshold_m": float(config["gate"]["paper_car_threshold_m"]),
            "protected_raw_contact_count": 0,
        },
        "internal_verification": config["internal_verification"],
    }


def _heldout_pass(field: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    required_cosine = float(config["gate"]["minimum_heldout_direction_cosine"])
    required_sign = float(config["gate"]["minimum_heldout_sign_accuracy"])
    records = [
        item["heldout"]
        for item in field["iterations"]
        if item.get("selected") is not None
    ]
    return bool(
        records
        and all(
            item.get("cosine") is not None
            and float(item["cosine"]) >= required_cosine
            and float(item["sign_accuracy"]) >= required_sign
            for item in records
        )
    )


def evaluate(
    *,
    repo_root: Path,
    table1_manifest_path: Path,
    selection_manifest_path: Path,
    archived_root: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    focused_result_path: Path,
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
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = _load_config(experiment_config_path)
    field_config = _field_config(config)
    selected_rows = [
        json.loads(line)
        for line in selection_manifest_path.read_text().splitlines()
        if line.strip()
    ]
    _require(len(selected_rows) == 1, "E38 selection count differs")
    selected = selected_rows[0]
    _require(selected.get("case_id") == "vlsa-t1-goal-ii-t3-e38", "E38 case differs")
    rows = [
        row
        for row in read_jsonl(table1_manifest_path)
        if row.get("case_id") == selected["case_id"]
    ]
    _require(len(rows) == 1, "E38 Table-1 case differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    archived_path = archived_root / selected["aegis_result_relative_path"]
    archived = _load(archived_path)
    _require(
        _file_sha256(archived_path) == selected["aegis_result_file_sha256"],
        "E38 archived file differs",
    )
    _require(
        archived.get("result_payload_sha256")
        == selected["aegis_result_payload_sha256"],
        "E38 archived payload differs",
    )
    _require(bool(archived.get("task_success")), "E38 archived task did not succeed")
    focused = _load(focused_result_path)
    registered = config["registered_inputs"]
    _require(
        _file_sha256(focused_result_path) == registered["focused_result_file_sha256"],
        "E38 focused result file differs",
    )
    _require(
        focused.get("result_payload_sha256")
        == registered["focused_result_payload_sha256"],
        "E38 focused result payload differs",
    )
    _require(focused.get("eligibility", {}).get("aegis_native_task_success"), "E38 eligibility differs")
    geometry_config = load_shadow_config(geometry_config_path)
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
        _require(str(probe_task.language) == str(task.language), "E38 probe task differs")
        _require(
            np.array_equal(np.asarray(probe_initial_state), np.asarray(selected_initial_state)),
            "E38 probe initial state differs",
        )
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(obstacle_name == selected["active_obstacle_name"], "E38 obstacle differs")
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
            _require(pairing[key] == archived["pairing"][key], "E38 pairing differs: %s" % key)
        perception = archived["perception"]
        obstacle_rotation = np.asarray(perception["mvee_rotation"], dtype=np.float64)
        rotation_canonicalization = "none"
        if float(np.linalg.det(obstacle_rotation)) < 0.0:
            obstacle_rotation = obstacle_rotation.copy()
            obstacle_rotation[:, -1] *= -1.0
            rotation_canonicalization = "flip_last_eigenvector_preserves_centered_ellipsoid"
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {
                "p2": perception["mvee_center"],
                "R2": obstacle_rotation,
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        disabled_images = _disable_images(probe_env)
        boundary = FixedContinuationProbe(one_step, obstacle_name, initial_obstacle_position)
        instrumented = InstrumentedContinuationProbe(
            one_step, obstacle_name, initial_obstacle_position
        )
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
        maximum_main_displacement = 0.0
        first_boundary_robot_contact_step = None
        first_boundary_protected_contact_step = None

        def execute(command: Any, step: int, source_name: str) -> tuple[bool, dict[str, Any]]:
            nonlocal observation, previous_goal_values, terminal_frame
            nonlocal maximum_main_displacement
            nonlocal first_boundary_robot_contact_step, first_boundary_protected_contact_step
            action = np.asarray(command, dtype=np.float64)
            observation, reward, done, _ = env.step(action.tolist())
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
            if robot_events and first_boundary_robot_contact_step is None:
                first_boundary_robot_contact_step = step
            if protected_events and first_boundary_protected_contact_step is None:
                first_boundary_protected_contact_step = step
            displacement = float(
                np.sum(
                    np.abs(
                        np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
                        - initial_obstacle_position
                    )
                )
            )
            maximum_main_displacement = max(maximum_main_displacement, displacement)
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            record = {
                "step": int(step),
                "source": source_name,
                "action": action.tolist(),
                "reward": float(reward),
                "done": bool(done),
                "goal_progress": goal,
                "robot_contact_events": robot_events,
                "protected_contact_events": protected_events,
                "active_obstacle_l1_displacement_m": displacement,
            }
            action_records.append(record)
            return bool(done), record

        start_action = int(config["state_protocol"]["start_action"])
        for step in range(start_action):
            done, _ = execute(
                _archived_action(archived["actions"], step),
                step,
                "immutable_archived_aegis_prefix",
            )
            _require(not done, "E38 completed before registered warning state")

        windows = []
        executed_internal_contacts = []
        executed_proxy_minima = []
        executed_internal_maximum_displacement = 0.0
        failure = None
        done = False
        action_count = len(archived["actions"])
        step = start_action
        while step < action_count and not done:
            lookahead_count = min(
                int(config["state_protocol"]["lookahead_actions"]), action_count - step
            )
            base_actions = _archived_actions(archived, step, lookahead_count)
            nominal = _internal_verify(
                instrumented, env, base_actions, field_config, step_base=step
            )
            nominal_margin = float(nominal["record"]["minimum_clearance_m"])
            warning = nominal_margin < float(config["gate"]["ellipsoid_warning_clearance_m"])
            field = None
            selected_actions = np.asarray(base_actions, dtype=np.float64)
            selected_source = "archived_aegis_nominal_no_warning"
            if warning and lookahead_count >= 5:
                field = _run_smooth(
                    boundary,
                    instrumented,
                    env,
                    base_actions,
                    field_config,
                    len(windows),
                    step_base=step,
                )
                selected_actions = np.asarray(
                    field["final"]["record"].get("actions", base_actions),
                    dtype=np.float64,
                )
                # The internal record intentionally excludes commands. Reconstruct them
                # from the registered correction and immutable nominal actions.
                selected_actions = np.asarray(base_actions, dtype=np.float64).copy()
                selected_actions[:5, :3] += np.asarray(
                    field["correction"], dtype=np.float64
                ).reshape(5, 3)
                selected_source = "recomputed_smooth_warning_repulsion"
            # Execute up to five actions, but preserve a complete correctable
            # five-action terminal window rather than stranding 1--4 actions.
            remaining = action_count - step
            execute_count = min(
                int(config["state_protocol"]["execute_prefix_max_actions"]),
                max(1, remaining - 5) if remaining > 5 else remaining,
            )
            prefix_actions = selected_actions[:execute_count]
            prefix = _internal_verify(
                instrumented, env, prefix_actions, field_config, step_base=step
            )
            prefix_record = prefix["record"]
            physical_prefix_safe = bool(
                len(prefix_record["protected_contacts"]) == 0
                and float(prefix_record["maximum_active_obstacle_l1_displacement_m"])
                <= float(config["gate"]["paper_car_threshold_m"])
            )
            heldout_valid = True if field is None else _heldout_pass(field, config)
            window = {
                "step": int(step),
                "lookahead_action_count": int(lookahead_count),
                "execute_action_count": int(execute_count),
                "warning": bool(warning),
                "nominal": _public(nominal),
                "field": None if field is None else _public(field),
                "field_heldout_gate": bool(heldout_valid),
                "selected_source": selected_source,
                "selected_actions": selected_actions.tolist(),
                "selected_prefix": _public(prefix),
                "physical_prefix_safe": physical_prefix_safe,
            }
            windows.append(window)
            if not physical_prefix_safe or not heldout_valid:
                failure = {
                    "step": int(step),
                    "reason": (
                        "selected_prefix_failed_physical_verification"
                        if not physical_prefix_safe
                        else "smooth_field_failed_heldout_direction_gate"
                    ),
                }
                break
            expected_state = np.asarray(
                _dynamic_state_vector(instrumented.env), dtype=np.float64
            )
            executed_internal_contacts.extend(prefix_record["protected_contacts"])
            executed_proxy_minima.append(float(prefix_record["minimum_clearance_m"]))
            executed_internal_maximum_displacement = max(
                executed_internal_maximum_displacement,
                float(prefix_record["maximum_active_obstacle_l1_displacement_m"]),
            )
            for offset, command in enumerate(prefix_actions):
                done, _ = execute(command, step + offset, selected_source)
                if done:
                    execute_count = offset + 1
                    break
            actual_state = np.asarray(_dynamic_state_vector(env), dtype=np.float64)
            state_error = float(np.max(np.abs(actual_state - expected_state)))
            _require(state_error <= 1.0e-10, "E38 executed prefix differs from clone")
            window["executed_action_count"] = int(execute_count)
            window["clone_state_max_abs_error"] = state_error
            step += execute_count

        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        native_success = goal_summary["first_all_satisfied_step"] is not None
        physical_safe = bool(
            len(executed_internal_contacts) == 0
            and first_boundary_protected_contact_step is None
            and executed_internal_maximum_displacement
            <= float(config["gate"]["paper_car_threshold_m"])
        )
        minimum_proxy = (
            min(executed_proxy_minima) if executed_proxy_minima else float("inf")
        )
        proxy_zero = bool(minimum_proxy >= 0.0)
        proxy_one = bool(minimum_proxy >= 0.001)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete" if failure is None else "method_failure",
            "scientific_result": True,
            "case_id": selected["case_id"],
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "selection_case": selected,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": selected["aegis_result_file_sha256"],
                "payload_sha256": selected["aegis_result_payload_sha256"],
                "read_only": True,
            },
            "registered_focused_result": {
                "path": str(focused_result_path),
                "file_sha256": registered["focused_result_file_sha256"],
                "payload_sha256": registered["focused_result_payload_sha256"],
                "read_only": True,
            },
            "geometry_config": geometry_config,
            "obstacle_rotation_canonicalization": rotation_canonicalization,
            "geometry": geometry.geometry_record(env),
            "pairing": pairing,
            "probe_environment": {
                "disabled_image_observable_count": disabled_images,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
            },
            "windows": windows,
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {**goal_definition, "initial": initial_goal, "summary": goal_summary},
            "physical_mujoco_authority": {
                "protected_internal_contact_count_after_intervention": len(
                    executed_internal_contacts
                ),
                "first_boundary_robot_contact_step": first_boundary_robot_contact_step,
                "first_boundary_protected_contact_step": first_boundary_protected_contact_step,
                "maximum_active_obstacle_l1_displacement_m": maximum_main_displacement,
                "maximum_internal_active_obstacle_l1_displacement_m": (
                    executed_internal_maximum_displacement
                ),
                "car_pass": executed_internal_maximum_displacement
                <= float(config["gate"]["paper_car_threshold_m"]),
                "physical_safe": physical_safe,
            },
            "ellipsoid_certification": {
                "minimum_executed_internal_clearance_m": minimum_proxy,
                "zero_margin_pass": proxy_zero,
                "one_mm_buffer_pass": proxy_one,
                "semantics": "optimizer_proxy_reported_separately_from_mujoco_contact",
            },
            "task_authority": {
                "native_task_success": native_success,
                "native_task_success_step": goal_summary["first_all_satisfied_step"],
                "archived_action_ledger_exhausted": bool(step >= action_count and not native_success),
            },
            "summary": {
                "window_count": len(windows),
                "warning_window_count": sum(bool(item["warning"]) for item in windows),
                "repulsion_window_count": sum(item["field"] is not None for item in windows),
                "physical_collision_free_task_completion": bool(physical_safe and native_success),
                "ellipsoid_certified_task_completion_zero_margin": bool(
                    physical_safe and native_success and proxy_zero
                ),
                "ellipsoid_certified_task_completion_one_mm": bool(
                    physical_safe and native_success and proxy_one
                ),
            },
            "failure": failure,
            "video": {
                "path": str(video_final),
                "file_sha256": _file_sha256(video_final),
                "frames_written": len(action_records) + 1,
                "fps": TABLE_VIDEO_FPS,
            },
            "final_jpg": {"path": str(final_jpg), "file_sha256": _file_sha256(final_jpg)},
            "interpretation": (
                "repeated_repulsion_physical_safe_task_success_but_proxy_uncertified"
                if physical_safe and native_success and not proxy_zero
                else "repeated_repulsion_physical_and_proxy_safe_task_success"
                if physical_safe and native_success and proxy_zero
                else "repeated_repulsion_physical_safe_but_task_failed"
                if physical_safe and not native_success
                else "repeated_repulsion_did_not_prevent_physical_collision"
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
    parser.add_argument("--table1-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--focused-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        table1_manifest_path=args.table1_manifest.resolve(),
        selection_manifest_path=args.selection_manifest.resolve(),
        archived_root=args.archived_root.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        focused_result_path=args.focused_result.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "interpretation": result["interpretation"],
                "summary": result["summary"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
