#!/usr/bin/env python3
"""Run the fixed warning-triggered normal-repulsion backup policy on E05."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _archived_actions,
    _disable_images,
    _public,
    _validate_registered,
)
from scripts.evaluate_distal_counterfactual_field_e05 import FixedContinuationProbe
from scripts.evaluate_distal_receding_route_oracle_e05 import _internal_summary
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256, ARCHIVED_PAYLOAD_SHA256, CASE_ID,
    EXPECTED_ACTION_HORIZON, _atomic_write, _file_sha256, _git_identity,
    _is_protected_event, _load, _require, _sha256,
)


RESULT_SCHEMA = "vlsa_distal_pncbf_backup_oracle_e05_result.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _backup_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    output = _internal_summary(record)
    for key in (
        "current_minimum_clearance_m",
        "future_minimum_clearance_m",
        "future_row_minimum_clearance_m",
    ):
        if key in record:
            output[key] = _public(record[key])
    return output


def evaluate(
    *, repo_root: Path, manifest_path: Path, archived_path: Path,
    controllability_path: Path, geometry_config_path: Path,
    experiment_config_path: Path, expected_commit: str, host: str,
    port: int, output_path: Path, proposal_result_path: Optional[Path] = None,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, TABLE_VIDEO_FPS,
        _active_obstacle, _aegis_action, _build_environment,
        _contact_model_authority, _detailed_active_obstacle_contacts,
        _eef_proxy, _goal_progress_definition, _goal_progress_snapshot,
        _goal_progress_summary, _policy_observation, _processed_image,
        _runtime_imports, _server_identity, _settle, array_sha256,
        max_steps_for_case, pairing_record, query_seed, read_jsonl,
        translational_action, validate_case_row,
    )
    from main.multilink_ellipsoid.pncbf_backup import (
        future_clearance, load_config, physical_safe, select_repulsive_candidate, update_latch,
    )
    from main.multilink_ellipsoid.pncbf_policy_value import exact_suffix_policy_values
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.rollout import (
        _auxiliary_sim_snapshot,
        _controller_snapshot,
        _restore_auxiliary_sim_snapshot,
        _restore_controller_snapshot,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow, allocation_record, load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_config(experiment_config_path)
    frozen_proposal = None
    if proposal_result_path is not None:
        proposal_registration = config["registered_inputs"].get("task_successful_proposal_result")
        _require(proposal_registration is not None, "proposal result is not registered")
        _require(_file_sha256(proposal_result_path) == proposal_registration["file_sha256"], "proposal file differs")
        frozen_proposal = _load(proposal_result_path)
        _require(frozen_proposal["result_payload_sha256"] == proposal_registration["result_payload_sha256"], "proposal payload differs")
        _require(frozen_proposal["source"]["commit"] == proposal_registration["source_commit"], "proposal source differs")
        _require(frozen_proposal["native_task_success"] is True, "proposal source did not complete task")
        _require(frozen_proposal["raw_contact_pass"] is False, "proposal source did not contain protected collision")
    registered = config["registered_inputs"]["controllability_result"]
    _require(_file_sha256(controllability_path) == registered["file_sha256"], "controllability file differs")
    control = _validate_registered(controllability_path, registered, "vlsa_distal_controllability_manifold_e05_result.v1")
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    _require(len(archived["actions"]) == EXPECTED_ACTION_HORIZON, "Table-1 horizon differs")
    rows = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "manifest case differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=True)
    geometry_config = load_shadow_config(geometry_config_path)
    target = np.asarray(config["registered_inputs"]["initial_prefix_coefficients"], dtype=np.float64)
    finalists = control["search"]["continuous"][config["registered_inputs"]["initial_prefix_arm"]]["internal_finalists"]
    selected = [row for row in finalists if np.array_equal(np.asarray(row["coefficients"]), target)]
    _require(len(selected) == 1 and selected[0]["protected_contact_count"] == 0, "initial prefix differs")
    initial_prefix = np.asarray(selected[0]["actions"][:5], dtype=np.float64)

    env = probe_env = video_writer = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_partial = output_path.with_name("episode.partial.mp4")
    video_final = output_path.with_name("episode.mp4")
    final_jpg = output_path.with_name("final.jpg")
    try:
        env, task, observation, initial_state = _build_environment(runtime, case, render_resolution=TABLE_RENDER_RESOLUTION)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, probe_initial = _build_environment(runtime, case, render_resolution=32)
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        _require(str(task.language) == str(probe_task.language), "probe task differs")
        _require(np.array_equal(initial_state, probe_initial), "probe state differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle = np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64).copy()
        pairing = pairing_record(case=case, selected_initial_state=initial_state, settled_observation=observation, task_description=str(task.language), active_obstacle_name=obstacle_name, settled_simulator_state=np.asarray(env.sim.get_state().flatten()))
        for key in ("manifest_row_sha256", "initial_state_sha256", "initial_observation_sha256", "settled_simulator_state_sha256", "settled_active_obstacle_position_sha256", "policy_noise_schedule_sha256"):
            _require(pairing[key] == archived["pairing"][key], "pairing differs: %s" % key)
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(geometry_config, {"p2": perception["mvee_center"], "R2": perception["mvee_rotation"], "Q2_diag": perception["mvee_semiaxes"], "record": {"label": perception["obstacle_label"]}})
        one_step = SlabbedEightConstraintProbe(probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name)
        disabled_images = _disable_images(probe_env)
        boundary = FixedContinuationProbe(one_step, obstacle_name, initial_obstacle)
        instrumented = InstrumentedContinuationProbe(one_step, obstacle_name, initial_obstacle)
        expected_substeps = int(config["measurement"]["expected_mujoco_substeps_per_action"])
        tolerance = float(config["measurement"]["boundary_equivalence_tolerance"])
        car_limit = float(config["measurement"]["paper_car_threshold_m"])
        activation = float(config["warning"]["activation_clearance_m"])
        release = float(config["warning"]["release_clearance_m"])
        contact_authority = _contact_model_authority(env, obstacle_name)
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(env, goal_atoms, step=-1, previous_values=None)
        previous_goal = initial_goal["values"]
        terminal_frame = _processed_image(observation, "agentview_image")
        video_writer = runtime["imageio"].get_writer(str(video_partial), fps=TABLE_VIDEO_FPS, codec="libx264", macro_block_size=None, pixelformat="yuv420p", output_params=["-crf", "18", "-movflags", "+faststart"])
        video_writer.append_data(terminal_frame)
        actions = []
        first_contact = None
        first_car = None
        maximum_car = 0.0

        def execute(command: Any, step: int, source_name: str) -> bool:
            nonlocal observation, previous_goal, terminal_frame, first_contact, first_car, maximum_car
            command = np.asarray(command, dtype=np.float64)
            observation, reward, done, _ = env.step(command.tolist())
            goal = _goal_progress_snapshot(env, goal_atoms, step=step, previous_values=previous_goal)
            previous_goal = goal["values"]
            contacts = _detailed_active_obstacle_contacts(env, obstacle_name, step=step, contact_authority=contact_authority)
            protected = [event for event in contacts["events"] if event.get("other", {}).get("classification") == "robot" and _is_protected_event(event)]
            if protected and first_contact is None:
                first_contact = step
            displacement = float(np.sum(np.abs(np.asarray(observation["%s_pos" % obstacle_name]) - initial_obstacle)))
            maximum_car = max(maximum_car, displacement)
            if displacement > car_limit and first_car is None:
                first_car = step
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            actions.append({"step": step, "source": source_name, "action": command.tolist(), "reward": float(reward), "done": bool(done), "goal_progress": goal, "protected_contact_events": _public(protected), "active_obstacle_l1_displacement_m": displacement, "next_state_sha256": array_sha256(_dynamic_state_vector(env))})
            return bool(done)

        for step in range(182):
            done = execute(_archived_actions(archived, step, step)[0], step, "immutable_archived_aegis_prefix")
            _require(not done, "task completed before activation")
        initial_record = instrumented.rollout_internal(env, initial_prefix, expected_substeps=expected_substeps, boundary_tolerance=tolerance, step_base=182)
        _require(physical_safe(initial_record, car_limit), "registered initial prefix is not physically safe")
        initial_policy_value_window = None
        if config.get("policy_value", {}).get("enabled") is True:
            initial_policy_value_window = {
                "step": 182,
                "action_count": len(initial_prefix),
                "executed_action_count": len(initial_prefix),
                "latch_entering": False,
                "latch_after_nominal": False,
                "nominal_physical_safe": True,
                "physically_unsafe_nominal_forced_backup": False,
                "frozen_source_nominal_max_abs_error_before_first_intervention": None,
                "nominal": _backup_summary(initial_record),
                "candidates": [],
                "selected_source": "registered_contact_free_initial_repulsion",
                "selected": _public(_backup_summary(initial_record)),
                "selected_clearance_trace_m": _public(initial_record["clearance_trace_m"]),
                "released_aegis_qp_records": [],
                "clone_state_max_abs_error": 0.0,
            }
        initial_clone_errors = []
        for offset, command in enumerate(initial_prefix):
            expected = one_step.transition(env, command)
            done = execute(command, 182 + offset, "registered_contact_free_initial_repulsion")
            error = float(np.max(np.abs(np.asarray(_dynamic_state_vector(env)) - np.asarray(expected["next_state_vector"]))))
            _require(error <= 1.0e-10, "initial execution differs from clone")
            initial_clone_errors.append(error)
            _require(not done, "task completed inside initial prefix")
        if initial_policy_value_window is not None:
            initial_policy_value_window["clone_state_max_abs_error"] = max(initial_clone_errors)

        client = None
        if frozen_proposal is None:
            client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, int(port))
            server_identity = _server_identity(client)
        else:
            server_identity = frozen_proposal["policy_server"]
            frozen_raw_by_step = {
                int(item["step"]): np.asarray(item["nominal_raw"], dtype=np.float64)
                for item in frozen_proposal["live_aegis_records"]
            }
            frozen_executed_by_step = {
                int(item["step"]): np.asarray(item["executed"], dtype=np.float64)
                for item in frozen_proposal["live_aegis_records"]
            }
        proxy = _eef_proxy(runtime, observation)
        p2 = np.asarray(perception["mvee_center"], dtype=np.float64)
        z = p2 - np.asarray(proxy["p1"], dtype=np.float64)
        z /= np.linalg.norm(z)
        released_geometry = {"p2": p2, "R2": np.asarray(perception["mvee_rotation"]), "Q2_diag": np.asarray(perception["mvee_semiaxes"]), "z_fixed": z}
        q1_diag = np.asarray([0.06, 0.12, 0.11])
        query_index = int(config["state_protocol"]["first_live_policy_query_index"])
        policy_queries = []
        windows = [] if initial_policy_value_window is None else [initial_policy_value_window]
        latch = False
        failure = None
        terminal_trigger = None
        done = False
        repulsion_has_changed_state = False
        for step in range(187, max_steps_for_case(case), 5):
            execute_count = min(5, max_steps_for_case(case) - step)
            if frozen_proposal is not None:
                execute_count = 0
                for offset in range(5):
                    if step + offset not in frozen_raw_by_step:
                        break
                    execute_count += 1
                if execute_count == 0:
                    if config.get("policy_value", {}).get("enabled") is True:
                        terminal_trigger = {
                            "step": len(actions),
                            "loop_step": step,
                            "reason": "task_successful_proposal_ledger_exhausted",
                        }
                    else:
                        failure = {"step": step, "reason": "task_successful_proposal_ledger_exhausted"}
                    break
            seed = query_seed(int(case["policy_noise_seed"]), query_index)
            if frozen_proposal is None:
                policy_input = _policy_observation(runtime, observation, task_description=str(task.language), resize_size=224, rng_seed=seed)
                query_started = time.perf_counter_ns()
                response = client.infer(policy_input)
                returned = np.asarray(response["actions"], dtype=np.float64)
                _require(returned.shape == (int(case["model_action_horizon"]), 7), "policy chunk differs")
                policy_queries.append({"query_index": query_index, "rng_seed": seed, "returned_actions_sha256": array_sha256(returned), "returned_actions": returned.tolist(), "wall_seconds": (time.perf_counter_ns() - query_started) * 1e-9, "server_timing": response.get("server_timing")})
            else:
                available = [frozen_raw_by_step.get(step + offset) for offset in range(execute_count)]
                _require(not any(value is None for value in available), "task-successful proposal ledger is noncontiguous")
                returned = np.asarray(available, dtype=np.float64)
                policy_queries.append({"query_index": query_index, "rng_seed": seed, "returned_actions_sha256": array_sha256(returned), "returned_actions": returned.tolist(), "wall_seconds": 0.0, "server_timing": None, "source": "registered_task_successful_job_39354_action_ledger"})
            query_index += 1
            one_step.synchronize(env)
            virtual_observation = probe_env.env._get_observations()
            virtual_proxy = _eef_proxy(runtime, virtual_observation)
            virtual_geometry = {"p2": p2.copy(), "R2": np.asarray(perception["mvee_rotation"]), "Q2_diag": np.asarray(perception["mvee_semiaxes"]), "z_fixed": np.asarray(released_geometry["z_fixed"]).copy()}
            nominal_actions = []
            qp_records = []
            for raw in returned[:execute_count]:
                nominal, qp = _aegis_action(runtime, nominal_translational=translational_action(raw), proxy=virtual_proxy, geometry=virtual_geometry, q1_diag=q1_diag, diagnostics_enabled=True)
                nominal_actions.append(nominal)
                virtual_observation, _, _, _ = probe_env.step(nominal)
                virtual_proxy = _eef_proxy(runtime, virtual_observation)
                qp_records.append(qp)
            nominal_actions = np.asarray(nominal_actions, dtype=np.float64)
            source_nominal_max_abs_error = None
            if frozen_proposal is not None and not repulsion_has_changed_state:
                source_actions = np.asarray(
                    [frozen_executed_by_step[step + offset] for offset in range(execute_count)],
                    dtype=np.float64,
                )
                source_nominal_max_abs_error = float(np.max(np.abs(nominal_actions - source_actions)))
                _require(source_nominal_max_abs_error <= 1.0e-10, "frozen proposal binding differs before intervention")
            nominal_record = instrumented.rollout_internal(env, nominal_actions, expected_substeps=expected_substeps, boundary_tolerance=tolerance, step_base=step)
            nominal_trace = np.asarray(nominal_record["clearance_trace_m"])
            nominal_record["current_minimum_clearance_m"] = float(np.min(nominal_trace[0]))
            nominal_record["future_minimum_clearance_m"] = future_clearance(nominal_record)
            nominal_record["future_row_minimum_clearance_m"] = np.min(nominal_trace[1:], axis=0)
            nominal_margin = float(nominal_record["future_minimum_clearance_m"])
            nominal_physical_safe = physical_safe(nominal_record, car_limit)
            latch_entering = latch
            latch = update_latch(latch, nominal_margin, activation, release)
            physically_forced = not nominal_physical_safe
            if physically_forced:
                latch = True
            candidates = []
            selected_actions = nominal_actions
            selected_record = nominal_record
            selected_summary = _backup_summary(nominal_record)
            selected_source = "fresh_pi05_original_aegis_no_warning"
            if latch:
                active_row = int(np.argmin(np.asarray(nominal_record["future_row_minimum_clearance_m"])))
                links = geometry._slabbed_links(env)
                direction = np.asarray(links[active_row].center) - np.asarray(geometry.obstacle.center)
                direction /= np.linalg.norm(direction)
                for requested in config["backup_policy"]["candidate_correction_l2_action"]:
                    correction = np.tile(direction * float(requested) / np.sqrt(float(execute_count)), (execute_count, 1))
                    candidate_actions = nominal_actions.copy()
                    candidate_actions[:, :3] = np.clip(candidate_actions[:, :3] + correction, -1.0, 1.0)
                    applied = candidate_actions[:, :3] - nominal_actions[:, :3]
                    record = instrumented.rollout_internal(env, candidate_actions, expected_substeps=expected_substeps, boundary_tolerance=tolerance, step_base=step)
                    candidate_trace = np.asarray(record["clearance_trace_m"])
                    record["current_minimum_clearance_m"] = float(np.min(candidate_trace[0]))
                    record["future_minimum_clearance_m"] = future_clearance(record)
                    record["future_row_minimum_clearance_m"] = np.min(candidate_trace[1:], axis=0)
                    candidates.append({"requested_correction_l2_action": float(requested), "correction_l2_action": float(np.linalg.norm(applied)), "direction": direction.tolist(), "actions": candidate_actions.tolist(), "record": _backup_summary(record)})
                chosen = select_repulsive_candidate(candidates, nominal_clearance_m=nominal_margin, activation_clearance_m=activation, paper_car_threshold_m=car_limit)
                if chosen is None:
                    if nominal_physical_safe and nominal_margin >= activation:
                        selected_source = "verified_safe_nominal_inside_hysteresis_deadband"
                    else:
                        failure = {"step": step, "reason": "no_physically_safe_clearance_improving_normal_repulsion"}
                else:
                    selected_actions = np.asarray(chosen["actions"], dtype=np.float64)
                    selected_record = chosen["record"]
                    selected_summary = chosen["record"]
                    selected_source = "hysteretic_normal_repulsion"
            window = {"step": step, "action_count": execute_count, "latch_entering": latch_entering, "latch_after_nominal": latch, "nominal_physical_safe": nominal_physical_safe, "physically_unsafe_nominal_forced_backup": physically_forced, "frozen_source_nominal_max_abs_error_before_first_intervention": source_nominal_max_abs_error, "nominal": _backup_summary(nominal_record), "candidates": candidates, "selected_source": None if failure else selected_source, "selected": None if failure else _public(selected_summary), "released_aegis_qp_records": qp_records}
            if config.get("policy_value", {}).get("enabled") is True and failure is None:
                selected_trace_record = instrumented.rollout_internal(
                    env,
                    selected_actions,
                    expected_substeps=expected_substeps,
                    boundary_tolerance=tolerance,
                    step_base=step,
                )
                window["selected_clearance_trace_m"] = _public(
                    selected_trace_record["clearance_trace_m"]
                )
            windows.append(window)
            if failure:
                break
            clone_errors = []
            for offset, command in enumerate(selected_actions):
                expected = one_step.transition(env, command)
                done = execute(command, step + offset, selected_source)
                error = float(np.max(np.abs(np.asarray(_dynamic_state_vector(env)) - np.asarray(expected["next_state_vector"]))))
                _require(error <= 1.0e-10, "backup execution differs from clone")
                clone_errors.append(error)
                if first_contact is not None or first_car is not None:
                    failure = {"step": step + offset, "reason": "executed_verified_prefix_produced_contact_or_car"}
                    break
                if done:
                    break
            window["executed_action_count"] = len(clone_errors)
            if "selected_clearance_trace_m" in window:
                window["selected_clearance_trace_m"] = window["selected_clearance_trace_m"][
                    : 1 + expected_substeps * len(clone_errors)
                ]
            window["clone_state_max_abs_error"] = max(clone_errors)
            released_geometry["z_fixed"] = np.asarray(qp_records[len(clone_errors) - 1]["z_after"], dtype=np.float64)
            if selected_source == "hysteretic_normal_repulsion":
                repulsion_has_changed_state = True
            if failure or done:
                break

        policy_value = None
        if config.get("policy_value", {}).get("enabled") is True:
            terminal = config["terminal_backup"]
            tail_action_count = int(terminal["tail_verification_actions"])
            primary_state = np.array(env.sim.get_state().flatten(), copy=True)
            primary_auxiliary = _auxiliary_sim_snapshot(env)
            primary_controller = _controller_snapshot(env)
            primary_dynamic = np.array(_dynamic_state_vector(env), copy=True)

            def restore_primary() -> None:
                env.sim.set_state_from_flattened(primary_state)
                env.sim.forward()
                _restore_auxiliary_sim_snapshot(env, primary_auxiliary)
                _restore_controller_snapshot(env, primary_controller)
                _require(
                    np.array_equal(_dynamic_state_vector(env), primary_dynamic),
                    "terminal backup restore differs",
                )

            hold_actions = np.zeros((tail_action_count, 7), dtype=np.float64)
            tail_record = instrumented.rollout_internal(
                env, hold_actions, expected_substeps=expected_substeps,
                boundary_tolerance=tolerance, step_base=len(actions),
            )
            tail_record["current_minimum_clearance_m"] = float(
                np.min(np.asarray(tail_record["clearance_trace_m"])[0])
            )
            tail_record["future_minimum_clearance_m"] = future_clearance(tail_record)
            tail_record["future_row_minimum_clearance_m"] = np.min(
                np.asarray(tail_record["clearance_trace_m"])[1:], axis=0
            )
            terminal_safe = physical_safe(tail_record, car_limit) and bool(
                tail_record["future_minimum_clearance_m"]
                >= float(config["policy_value"]["safety_buffer_m"])
            )
            terminal_selected = "verified_zero_motion_hold"
            terminal_fallback = "normal_retreat_not_needed_in_this_rollout"
            terminal_actions = hold_actions
            if not terminal_safe:
                restore_primary()
                links = geometry._slabbed_links(env)
                active_row = int(np.argmin(np.asarray(tail_record["future_row_minimum_clearance_m"])))
                direction = np.asarray(links[active_row].center) - np.asarray(geometry.obstacle.center)
                direction /= np.linalg.norm(direction)
                retreat_actions = np.zeros((tail_action_count, 7), dtype=np.float64)
                retreat_actions[:, :3] = direction * float(terminal["retreat_action_l2_per_action"])
                tail_record = instrumented.rollout_internal(
                    env, retreat_actions, expected_substeps=expected_substeps,
                    boundary_tolerance=tolerance, step_base=len(actions),
                )
                tail_trace = np.asarray(tail_record["clearance_trace_m"])
                tail_record["current_minimum_clearance_m"] = float(np.min(tail_trace[0]))
                tail_record["future_minimum_clearance_m"] = future_clearance(tail_record)
                tail_record["future_row_minimum_clearance_m"] = np.min(tail_trace[1:], axis=0)
                terminal_safe = physical_safe(tail_record, car_limit) and bool(
                    tail_record["future_minimum_clearance_m"]
                    >= float(config["policy_value"]["safety_buffer_m"])
                )
                terminal_selected = "registered_normal_retreat"
                terminal_fallback = "normal_retreat_executed_after_unsafe_hold"
                terminal_actions = retreat_actions
            policy_value = exact_suffix_policy_values(
                windows,
                terminal_tail_clearance_trace_m=tail_record["clearance_trace_m"],
                safety_buffer_m=float(config["policy_value"]["safety_buffer_m"]),
                expected_substeps_per_action=expected_substeps,
                boundary_tolerance_m=tolerance,
            )
            policy_value["terminal_backup"] = {
                "selected": terminal_selected,
                "fallback_if_hold_unsafe": terminal_fallback,
                "action_count": tail_action_count,
                "actions": terminal_actions.tolist(),
                "physically_safe": physical_safe(tail_record, car_limit),
                "buffer_safe": terminal_safe,
                "rollout": _backup_summary(tail_record),
            }
            policy_value["terminal_trigger"] = terminal_trigger
            policy_value["terminal_clearance_trace_m"] = _public(
                tail_record["clearance_trace_m"]
            )
            restore_primary()

        video_writer.close(); video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, actions)
        success = goal_summary["first_all_satisfied_step"] is not None
        verified_selected = all(
            item["selected"] is not None and physical_safe(item["selected"], car_limit)
            for item in windows
        )
        solved = bool(success and first_contact is None and first_car is None and failure is None and verified_selected)
        result = {
            "schema_version": RESULT_SCHEMA, "status": "complete" if failure is None else "method_failure", "scientific_result": True,
            "case_id": CASE_ID, "claim_scope": config["claim_scope"], "source": source, "allocation": allocation, "config": config,
            "archived_table1": {"path": str(archived_path), "file_sha256": ARCHIVED_FILE_SHA256, "payload_sha256": ARCHIVED_PAYLOAD_SHA256, "read_only": True},
            "registered_controllability": {"path": str(controllability_path), "file_sha256": _file_sha256(controllability_path), "payload_sha256": control["result_payload_sha256"]},
            "registered_task_successful_proposal": None if frozen_proposal is None else {"path": str(proposal_result_path), "file_sha256": _file_sha256(proposal_result_path), "payload_sha256": frozen_proposal["result_payload_sha256"], "source_slurm_job_id": frozen_proposal["allocation"]["slurm_job_id"], "native_task_success_step": frozen_proposal["native_task_success_step"], "first_protected_contact": frozen_proposal["physical_safety_after_activation"]["first_protected_contact"]},
            "pairing": pairing, "probe_environment": {"disabled_image_observable_count": disabled_images, "osc_controller": "OSC_POSE", "control_frequency_hz": 20},
            "policy_server": server_identity, "policy_queries": policy_queries, "windows": windows, "actions": actions,
            "goal_progress": {**goal_definition, "initial": initial_goal, "summary": goal_summary},
            "summary": {"window_count": len(windows), "repulsion_window_count": sum(item["selected_source"] == "hysteretic_normal_repulsion" for item in windows), "hysteresis_deadband_nominal_window_count": sum(item["selected_source"] == "verified_safe_nominal_inside_hysteresis_deadband" for item in windows), "all_executed_windows_freshly_physically_verified": verified_selected, "native_task_success": success, "native_task_success_step": goal_summary["first_all_satisfied_step"], "first_protected_contact_step": first_contact, "first_paper_car_step": first_car, "maximum_active_obstacle_l1_displacement_m": maximum_car, "safe_task_success": solved},
            "policy_value": policy_value,
            "failure": failure, "primary_problem_solved": solved,
            "interpretation": (
                "finite_horizon_exact_backup_policy_value_pass"
                if policy_value is not None
                and policy_value["all_decision_states_safe"]
                and policy_value["nonincreasing_along_backup"]
                and policy_value["successor_boundaries_consistent"]
                and policy_value["maximum_bellman_residual"] <= tolerance
                and policy_value["terminal_backup"]["buffer_safe"]
                else (
                    "fixed_backup_policy_safe_task_success"
                    if solved else "fixed_backup_policy_oracle_no_go"
                )
            ),
            "video": {"path": str(video_final), "file_sha256": _file_sha256(video_final), "frames_written": len(actions) + 1, "fps": TABLE_VIDEO_FPS},
            "final_jpg": {"path": str(final_jpg), "file_sha256": _file_sha256(final_jpg)}, "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
        }
        result["result_payload_sha256"] = _sha256(_canonical(result))
        return result
    finally:
        if video_writer is not None: video_writer.close()
        if probe_env is not None: probe_env.close()
        if env is not None: env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--controllability-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposal-result", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(), archived_path=args.archived.resolve(), controllability_path=args.controllability_result.resolve(), geometry_config_path=args.geometry_config.resolve(), experiment_config_path=args.experiment_config.resolve(), expected_commit=args.expected_commit, host=args.host, port=args.port, output_path=args.output.resolve(), proposal_result_path=None if args.proposal_result is None else args.proposal_result.resolve())
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({"interpretation": result["interpretation"], "primary_problem_solved": result["primary_problem_solved"], "summary": result["summary"], "failure": result["failure"], "result_payload_sha256": result["result_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
