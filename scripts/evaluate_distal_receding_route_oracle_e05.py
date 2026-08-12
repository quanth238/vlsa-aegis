#!/usr/bin/env python3
"""Run the exact receding persistent-route oracle through the E05 episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_counterfactual_field_e05 import FixedContinuationProbe
from scripts.evaluate_distal_post_detour_live_gate_e05 import _heldout_gate
from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _archived_actions,
    _direct_smooth_field,
    _disable_images,
    _public,
    _validate_registered,
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


RESULT_SCHEMA = "vlsa_distal_receding_route_oracle_e05_result.v2"


def _internal_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "minimum_clearance_m": float(record["minimum_clearance_m"]),
        "row_minimum_clearance_m": _public(record["row_minimum_clearance_m"]),
        "protected_contacts": list(record["protected_contacts"]),
        "maximum_active_obstacle_l1_displacement_m": float(
            record["maximum_active_obstacle_l1_displacement_m"]
        ),
        "sample_count": int(record["sample_count"]),
        "substep_counts": list(record["substep_counts"]),
        "maximum_boundary_equivalence_error_m": float(
            record["maximum_boundary_equivalence_error_m"]
        ),
        "env_step_wall_seconds": float(record["env_step_wall_seconds"]),
    }


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    route_oracle_path: Path,
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
    from main.multilink_ellipsoid.post_detour_live_gate import (
        load_receding_route_config,
        persistent_route_correction,
        persistent_route_directions,
        verified_internal,
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
    _require(len(archived["actions"]) == EXPECTED_ACTION_HORIZON, "Table-1 horizon differs")
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    config = load_receding_route_config(experiment_config_path)
    registered = config["registered_inputs"]
    route_oracle = _validate_registered(
        route_oracle_path,
        registered["route_oracle_result"],
        "vlsa_distal_post_detour_route_oracle_e05_result.v1",
    )
    _require(route_oracle["gate"]["passed"] is True, "registered route oracle did not pass")
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
        _require(np.array_equal(probe_initial_state, selected_initial_state), "probe initial state differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle_position = np.asarray(observation["%s_pos" % obstacle_name]).copy()
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
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        disabled_images = _disable_images(probe_env)
        boundary = FixedContinuationProbe(one_step, obstacle_name, initial_obstacle_position)
        instrumented = InstrumentedContinuationProbe(one_step, obstacle_name, initial_obstacle_position)
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

        def execute(command: Any, step: int, source_name: str) -> tuple[Any, bool]:
            nonlocal observation, previous_goal_values, terminal_frame
            nonlocal first_robot_contact_step, first_protected_contact_step, first_car_step
            nonlocal maximum_displacement
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
                    "source": source_name,
                    "action": action.tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                    "robot_contact_events": robot_events,
                    "protected_contact_events": protected_events,
                    "active_obstacle_l1_displacement_m": displacement,
                }
            )
            return goal, bool(done)

        for step in range(182):
            _, done = execute(_archived_actions(archived, step, step)[0], step, "immutable_archived_aegis_prefix")
            _require(not done, "archived prefix completed before receding route activation")

        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
        server_identity = _server_identity(client)
        q1_diag = np.asarray([0.06, 0.12, 0.11], dtype=np.float64)
        actual_z = np.asarray(archived["actions"][181]["qp"]["z_after"], dtype=np.float64)
        policy_queries = []
        windows = []
        active_mode = None
        failure = None
        total_search_rollouts = 0
        query_base = int(config["state_protocol"]["first_live_policy_query_index"])

        def verify(actions: Any, step: int) -> tuple[dict[str, Any], Mapping[str, Any]]:
            record = instrumented.rollout_internal(
                env,
                actions,
                expected_substeps=int(config["internal_verification"]["expected_mujoco_substeps_per_action"]),
                boundary_tolerance=float(
                    config["internal_verification"]["ordinary_env_step_boundary_equivalence_tolerance"]
                ),
                step_base=step,
            )
            return _internal_summary(record), record

        for step in range(182, max_steps_for_case(case), 5):
            query_record = None
            if step == 182:
                nominal_actions = _archived_actions(archived, 182, 186)
                qp_records = [archived["actions"][index]["qp"] for index in range(182, 187)]
            else:
                query_index = query_base + (step - 187) // 5
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
                _require(returned.shape == (int(case["model_action_horizon"]), 7), "live action chunk differs")
                query_record = {
                    "step": step,
                    "query_index": query_index,
                    "rng_seed": seed,
                    "returned_actions_sha256": array_sha256(returned),
                    "wall_seconds": query_wall,
                    "server_timing": response.get("server_timing"),
                }
                policy_queries.append(query_record)
                one_step.synchronize(env)
                virtual_observation = instrumented.env.env._get_observations()
                virtual_proxy = _eef_proxy(runtime, virtual_observation)
                virtual_geometry = {
                    "p2": np.asarray(perception["mvee_center"], dtype=np.float64),
                    "R2": np.asarray(perception["mvee_rotation"], dtype=np.float64),
                    "Q2_diag": np.asarray(perception["mvee_semiaxes"], dtype=np.float64),
                    "z_fixed": actual_z.copy(),
                }
                nominal_actions = []
                qp_records = []
                for raw in returned[:5]:
                    nominal, qp_record = _aegis_action(
                        runtime,
                        nominal_translational=translational_action(raw),
                        proxy=virtual_proxy,
                        geometry=virtual_geometry,
                        q1_diag=q1_diag,
                        diagnostics_enabled=True,
                    )
                    nominal_actions.append(nominal)
                    virtual_observation, _, _, _ = instrumented.env.step(nominal)
                    virtual_proxy = _eef_proxy(runtime, virtual_observation)
                    qp_records.append(qp_record)
                nominal_actions = np.asarray(nominal_actions, dtype=np.float64)
            nominal_summary, nominal_record = verify(nominal_actions, step)
            total_search_rollouts += 1
            nominal_safe = verified_internal(nominal_record, config["gate"])
            selected_actions = nominal_actions
            selected_summary = nominal_summary
            selected_source = "fresh_exact_safe_nominal_window"
            route_evidence = []
            field_evidence = None
            derivative_evidence = None
            active_mode_entering = active_mode
            nominal_clearance = float(nominal_record["minimum_clearance_m"])
            if nominal_safe and nominal_clearance >= float(
                config["route_search"]["release_clearance_m"]
            ):
                active_mode = None
            route_search_required = bool(
                not nominal_safe
                or (
                    active_mode is not None
                    and nominal_clearance
                    < float(config["route_search"]["release_clearance_m"])
                )
            )
            if route_search_required:
                active_row = int(np.argmin(np.asarray(nominal_record["row_minimum_clearance_m"])[:7]))
                links = geometry._slabbed_links(env)
                route_directions = persistent_route_directions(
                    links[active_row].center, geometry.obstacle.center
                )
                verified_routes = []
                candidate_modes = (
                    [active_mode]
                    if nominal_safe and active_mode is not None
                    else config["route_search"]["modes"]
                )
                for mode_index, mode in enumerate(candidate_modes):
                    for norm in config["route_search"]["coarse_norms_action"]:
                        correction = persistent_route_correction(route_directions[mode], float(norm))
                        actions = nominal_actions.copy()
                        actions[:, :3] = np.clip(
                            actions[:, :3] + correction.reshape(5, 3),
                            -float(config["route_search"]["action_limit"]),
                            float(config["route_search"]["action_limit"]),
                        )
                        applied = (actions[:, :3] - nominal_actions[:, :3]).reshape(15)
                        summary, record = verify(actions, step)
                        total_search_rollouts += 1
                        safe = verified_internal(record, config["gate"])
                        item = {
                            "mode": mode,
                            "mode_index": mode_index,
                            "requested_norm_action": float(norm),
                            "applied_correction_l2_action": float(np.linalg.norm(applied)),
                            "actions": actions.tolist(),
                            "record": summary,
                            "verification_gate": safe,
                        }
                        route_evidence.append(item)
                        if safe:
                            verified_routes.append(item)
                if verified_routes:
                    selected_route = min(
                        verified_routes,
                        key=lambda item: (
                            float(item["applied_correction_l2_action"]),
                            0 if item["mode"] == active_mode else 1,
                            int(item["mode_index"]),
                        ),
                    )
                    selected_actions = np.asarray(selected_route["actions"], dtype=np.float64)
                    selected_summary = selected_route["record"]
                    selected_source = "fresh_exact_persistent_%s_route" % selected_route["mode"]
                    active_mode = selected_route["mode"]
                elif nominal_safe:
                    selected_source = "fresh_exact_safe_nominal_window_persistent_route_unavailable"
                else:
                    best_route = max(
                        route_evidence,
                        key=lambda item: float(item["record"]["minimum_clearance_m"]),
                    )
                    field_config = dict(config)
                    field_config["action_space"] = dict(config["field_refinement"])
                    field = _direct_smooth_field(
                        boundary,
                        instrumented,
                        env,
                        np.asarray(best_route["actions"], dtype=np.float64),
                        field_config,
                        step_base=step,
                    )
                    total_search_rollouts += int(field["rollout_count"])
                    field_actions = np.asarray(field["best_boundary"]["actions"], dtype=np.float64)
                    field_total_correction = (
                        field_actions[:, :3] - nominal_actions[:, :3]
                    ).reshape(15)
                    field_summary, field_record = verify(field_actions, step)
                    total_search_rollouts += 1
                    field_safe = bool(
                        verified_internal(field_record, config["gate"])
                        and _heldout_gate(field, config["gate"])
                        and float(np.linalg.norm(field_total_correction))
                        <= float(config["derivative_free"]["maximum_correction_l2_action"])
                        + 1.0e-10
                    )
                    field_evidence = {
                        "base_route": best_route,
                        "optimization": _public(field),
                        "total_correction_l2_action": float(
                            np.linalg.norm(field_total_correction)
                        ),
                        "record": field_summary,
                        "verification_gate": field_safe,
                    }
                    if field_safe:
                        selected_actions = field_actions
                        selected_summary = field_summary
                        selected_source = "fresh_exact_route_plus_smooth_refinement"
                        active_mode = best_route["mode"]
                    else:
                        search = config["derivative_free"]
                        rng = np.random.RandomState(int(search["seed"]) + step)
                        pool = []
                        elites = []
                        for generation in range(int(search["generations"])):
                            for candidate_index in range(int(search["candidate_count_per_generation"])):
                                raw = rng.normal(size=15)
                                raw /= np.linalg.norm(raw)
                                if generation == 0 or not elites:
                                    correction = raw * rng.uniform(
                                        0.25, float(search["maximum_correction_l2_action"])
                                    )
                                else:
                                    parent = np.asarray(elites[candidate_index % len(elites)]["correction"])
                                    correction = parent + float(
                                        search["maximum_correction_l2_action"]
                                    ) * (0.5 ** generation) * raw
                                    correction_norm = float(np.linalg.norm(correction))
                                    if correction_norm > float(search["maximum_correction_l2_action"]):
                                        correction *= float(search["maximum_correction_l2_action"]) / correction_norm
                                actions = nominal_actions.copy()
                                actions[:, :3] = np.clip(
                                    actions[:, :3] + correction.reshape(5, 3), -1.0, 1.0
                                )
                                applied = (actions[:, :3] - nominal_actions[:, :3]).reshape(15)
                                summary, record = verify(actions, step)
                                total_search_rollouts += 1
                                pool.append(
                                    {
                                        "correction": applied.tolist(),
                                        "correction_l2_action": float(np.linalg.norm(applied)),
                                        "actions": actions.tolist(),
                                        "record": summary,
                                        "verification_gate": verified_internal(record, config["gate"]),
                                    }
                                )
                            elites = sorted(
                                pool,
                                key=lambda item: (
                                    bool(item["verification_gate"]),
                                    float(item["record"]["minimum_clearance_m"]),
                                ),
                                reverse=True,
                            )[: int(search["elite_count"])]
                        safe_pool = [item for item in pool if item["verification_gate"]]
                        derivative_evidence = {
                            "candidate_count": len(pool),
                            "safe_count": len(safe_pool),
                            "best": elites[0],
                        }
                        if safe_pool:
                            selected_derivative = min(
                                safe_pool, key=lambda item: float(item["correction_l2_action"])
                            )
                            selected_actions = np.asarray(selected_derivative["actions"], dtype=np.float64)
                            selected_summary = selected_derivative["record"]
                            selected_source = "fresh_exact_derivative_free_feasibility_control"
                            active_mode = None
                        else:
                            failure = {
                                "component": "receding_persistent_route_oracle",
                                "step": step,
                                "reason": "no_verified_nominal_route_refinement_or_derivative_free_candidate",
                            }
            window = {
                "step": step,
                "policy_query": query_record,
                "released_aegis_qp_records": qp_records,
                "nominal_actions": nominal_actions.tolist(),
                "nominal": nominal_summary,
                "nominal_safe": nominal_safe,
                "active_mode_entering": active_mode_entering,
                "active_mode_selected": active_mode,
                "persistent_routes": route_evidence,
                "field_refinement": field_evidence,
                "derivative_free": derivative_evidence,
                "selected_source": None if failure is not None else selected_source,
                "selected_actions": None if failure is not None else selected_actions.tolist(),
                "selected": None if failure is not None else selected_summary,
            }
            windows.append(window)
            if failure is not None:
                break
            clone_errors = []
            done = False
            for offset, selected_action in enumerate(selected_actions):
                expected = one_step.transition(env, selected_action)
                _, done = execute(selected_action, step + offset, selected_source)
                state_error = float(
                    np.max(
                        np.abs(
                            np.asarray(_dynamic_state_vector(env), dtype=np.float64)
                            - np.asarray(expected["next_state_vector"], dtype=np.float64)
                        )
                    )
                )
                _require(state_error <= 1.0e-10, "executed receding route action differs from clone")
                clone_errors.append(state_error)
                if first_robot_contact_step is not None or first_car_step is not None:
                    failure = {
                        "component": "main_environment_safety_authority",
                        "step": step + offset,
                        "reason": "verified_action_produced_contact_or_car",
                    }
                    break
                if done:
                    break
            window["executed_prefix_actions"] = len(clone_errors)
            window["clone_state_max_abs_error"] = max(clone_errors)
            actual_z = np.asarray(qp_records[len(clone_errors) - 1]["z_after"], dtype=np.float64)
            if failure is not None or done:
                break

        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        native_success = goal_summary["first_all_satisfied_step"] is not None
        problem_solved = bool(
            failure is None
            and first_robot_contact_step is None
            and first_protected_contact_step is None
            and first_car_step is None
            and native_success
        )
        corrected_windows = [
            item
            for item in windows
            if item["selected_source"] is not None
            and not item["selected_source"].startswith("fresh_exact_safe_nominal_window")
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
            "registered_inputs": {
                "route_oracle_payload_sha256": route_oracle["result_payload_sha256"],
            },
            "config": config,
            "geometry_config": geometry_config,
            "geometry": geometry.geometry_record(env),
            "pairing": pairing,
            "probe_environment": {
                "disabled_image_observable_count": disabled_images,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
            },
            "policy_server": server_identity,
            "policy_query_count": len(policy_queries),
            "policy_queries": policy_queries,
            "receding_summary": {
                "window_count": len(windows),
                "unsafe_nominal_window_count": sum(not item["nominal_safe"] for item in windows),
                "corrected_window_count": len(corrected_windows),
                "total_search_rollouts": total_search_rollouts,
                "execute_prefix_actions": int(config["state_protocol"]["execute_prefix_actions"]),
                "route_mode_sequence": [item["active_mode_selected"] for item in windows],
            },
            "windows": windows,
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
                "receding_persistent_route_oracle_safe_task_success"
                if problem_solved
                else (
                    "receding_persistent_route_oracle_deadlock"
                    if failure is not None
                    else "receding_persistent_route_oracle_safe_but_task_failed"
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
    parser.add_argument("--route-oracle-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        route_oracle_path=args.route_oracle_result.resolve(),
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
