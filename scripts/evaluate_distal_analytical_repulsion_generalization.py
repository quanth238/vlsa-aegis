#!/usr/bin/env python3
"""Run one sealed full-episode analytical-repulsion generalization case."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import _allocation_record, _public
from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _disable_images,
)
from scripts.evaluate_distal_soft_prefix_live_replan_e05 import InternalSafetyMonitor
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _archived_actions(result: Mapping[str, Any], start: int, count: int) -> Any:
    import numpy as np

    by_step = {int(row["step"]): row for row in result["actions"]}
    _require(
        all(step in by_step for step in range(start, start + count)),
        "analytical-repulsion archived horizon differs",
    )
    values = np.asarray(
        [by_step[step]["executed"] for step in range(start, start + count)],
        dtype=np.float64,
    )
    _require(values.shape == (count, 7), "analytical-repulsion archived actions differ")
    return values


def _contact_link_name(model: Any, protected_geom_id: int) -> str:
    """Return the closest protected ancestor of one contacted robot geom.

    A geom attached below L7 has L7, L6, and L5 in its ancestry.  Classification
    must use the first protected ancestor while walking toward the root; requiring
    exactly one protected ancestor incorrectly rejects genuine distal contact.
    """

    targets = ("robot0_link5", "robot0_link6", "robot0_link7")
    body = int(model.geom_bodyid[int(protected_geom_id)])
    while body >= 0:
        name = model.body_id2name(body)
        if name in targets:
            return str(name)
        if body == 0:
            break
        body = int(model.body_parentid[body])
    raise ValueError("protected contact has no L5--L7 ancestor")


def _classify_contact_links(env: Any, events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    model = env.sim.model
    targets = ("robot0_link5", "robot0_link6", "robot0_link7")
    counts = {name: 0 for name in targets}
    for event in events:
        counts[_contact_link_name(model, int(event["protected_geom_id"]))] += 1
    return counts


def evaluate(
    *,
    repo_root: Path,
    population_manifest_path: Path,
    selection_manifest_path: Path,
    table1_root: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    case_index: int,
    expected_commit: str,
    host: str,
    port: int,
    output_path: Path,
    requested_radius_override: Optional[float] = None,
    result_schema_override: Optional[str] = None,
    claim_scope_override: Optional[str] = None,
    controller_binding: Optional[Mapping[str, Any]] = None,
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
    from main.multilink_ellipsoid.analytical_repulsion_generalization import (
        RESULT_SCHEMA,
        corrected_proposal,
        frame_integrity_metrics,
        load_cases,
        load_config,
        warning_trigger,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        _eef_site_id,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe
    from scripts.replay_distal_three_ellipsoid_multicbf import _is_protected_event

    started = time.perf_counter_ns()
    config = load_config(experiment_config_path)
    selected_cases = load_cases(selection_manifest_path)
    _require(0 <= int(case_index) < len(selected_cases), "case index differs")
    selected = selected_cases[int(case_index)]
    source = _git_identity(repo_root, expected_commit)
    allocation = _allocation_record()
    archive_path = table1_root / selected["archived_result_relative_path"]
    _require(_file_sha256(archive_path) == selected["archived_result_file_sha256"],
             "sealed AEGIS file differs")
    archived = _load(archive_path)
    _require(archived["result_payload_sha256"] == selected["archived_result_payload_sha256"],
             "sealed AEGIS payload differs")
    _require(archived.get("task_success") is True, "sealed raw AEGIS task did not succeed")
    rows = [row for row in read_jsonl(population_manifest_path)
            if row.get("case_id") == selected["case_id"]]
    _require(len(rows) == 1, "population case differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    _require(int(case["replan_steps"]) == 5, "policy replan stride differs")

    runtime = _runtime_imports(include_aegis=True)
    geometry_config = load_shadow_config(geometry_config_path)
    env = probe_env = video_writer = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_partial = output_path.with_name("episode.partial.mp4")
    video_final = output_path.with_name("episode.mp4")
    final_jpg = output_path.with_name("final.jpg")
    try:
        probe_env, probe_task, probe_observation, probe_initial_state = _build_environment(
            runtime,
            case,
            render_resolution=int(config["execution"]["probe_render_resolution"]),
        )
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        disabled_probe_images = _disable_images(probe_env)
        # OSMesa's offscreen context is process-global in this LIBERO stack.
        # Construct the image-disabled probe first and the 1024px main renderer
        # last, so the small probe buffer cannot corrupt later policy images.
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        _require(str(task.language) == str(probe_task.language), "probe task differs")
        _require(np.array_equal(np.asarray(selected_initial_state), np.asarray(probe_initial_state)),
                 "probe initial state differs")
        _require(np.array_equal(np.asarray(env.sim.get_state().flatten()),
                                np.asarray(probe_env.sim.get_state().flatten())),
                 "probe settled state differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(obstacle_name == selected["active_obstacle_name"], "active obstacle differs")
        obstacle_reference = np.asarray(observation[obstacle_name + "_pos"], dtype=np.float64).copy()
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(env.sim.get_state().flatten()),
        )
        for key in (
            "manifest_row_sha256", "initial_state_sha256",
            "initial_observation_sha256", "settled_simulator_state_sha256",
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
            probe_env, geometry, clearance_m=0.0,
            active_obstacle_name=obstacle_name,
        )
        instrumented = InstrumentedContinuationProbe(
            one_step, obstacle_name, obstacle_reference
        )
        monitor = InternalSafetyMonitor(env, geometry, obstacle_name, obstacle_reference)
        contact_authority = _contact_model_authority(env, obstacle_name)
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(env, goal_atoms, step=-1, previous_values=None)
        previous_goal_values = initial_goal["values"]
        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, int(port))
        server_identity = _server_identity(client)
        q1_diag = np.asarray([0.06, 0.12, 0.11], dtype=np.float64)
        released_geometry_base = {
            "p2": np.asarray(perception["mvee_center"], dtype=np.float64),
            "R2": np.asarray(perception["mvee_rotation"], dtype=np.float64),
            "Q2_diag": np.asarray(perception["mvee_semiaxes"], dtype=np.float64),
        }

        video_writer = runtime["imageio"].get_writer(
            str(video_partial), fps=TABLE_VIDEO_FPS, codec="libx264",
            macro_block_size=None, pixelformat="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        terminal_frame = _processed_image(observation, "agentview_image")
        frame_integrity_records = [
            {"step": -1, **frame_integrity_metrics(terminal_frame)}
        ]
        frame_integrity_limit = float(
            config["execution"][
                "camera_integrity_maximum_normalized_neighbor_difference"
            ]
        )
        _require(
            frame_integrity_records[-1]["maximum_neighbor_difference"]
            <= frame_integrity_limit,
            "initial camera frame integrity failed",
        )
        video_writer.append_data(terminal_frame)
        action_records = []
        query_records = []
        interventions = []
        clone_execution_errors_m = []
        clone_execution_records = []
        terminal_eef_positions = [np.asarray(observation["robot0_eef_pos"], dtype=np.float64)]
        first_intervention_step = None
        live_mode = False
        live_query_index = None
        live_z = None

        def project_through_aegis(main_env: Any, proposed_actions: Any, initial_z: Any) -> tuple[Any, Any, list[Any]]:
            proposed = np.asarray(proposed_actions, dtype=np.float64)
            _require(proposed.shape == (5, 7), "AEGIS projection proposal differs")
            z = np.asarray(initial_z, dtype=np.float64).copy()
            _require(z.shape == (3,) and np.all(np.isfinite(z)), "AEGIS z differs")
            sync = one_step.synchronize(main_env)
            _require(sync["maximum_absolute_error"] == 0.0, "AEGIS clone state differs")
            projected_geometry = {**released_geometry_base, "z_fixed": z}
            executed = []
            qp_records = []
            for proposed_action in proposed:
                site_position = np.asarray(
                    probe_env.sim.data.site_xpos[_eef_site_id(probe_env)], dtype=np.float64
                )
                eef_name = probe_env.robots[0].robot_model.eef_name
                quaternion_wxyz = np.asarray(
                    probe_env.sim.data.get_body_xquat(eef_name), dtype=np.float64
                )
                rotation = runtime["Rotation"].from_quat(
                    quaternion_wxyz[[1, 2, 3, 0]]
                ).as_matrix()
                proxy = {
                    "p1": site_position + rotation @ np.asarray([0.0, 0.0, -0.08]),
                    "R1": rotation,
                }
                command, qp = _aegis_action(
                    runtime,
                    nominal_translational=proposed_action,
                    proxy=proxy,
                    geometry=projected_geometry,
                    q1_diag=q1_diag,
                    diagnostics_enabled=True,
                )
                command = np.asarray(command, dtype=np.float64)
                probe_env.step(command.tolist())
                executed.append(command)
                qp_records.append(qp)
            return (
                np.asarray(executed, dtype=np.float64),
                np.asarray(projected_geometry["z_fixed"], dtype=np.float64).copy(),
                qp_records,
            )

        def predict(actions: Any, step: int) -> dict[str, Any]:
            return instrumented.rollout_internal(
                env, actions,
                expected_substeps=int(config["execution"]["expected_mujoco_substeps_per_action"]),
                boundary_tolerance=float(config["execution"]["boundary_equivalence_tolerance_m"]),
                step_base=step,
            )

        step = 0
        max_steps = max_steps_for_case(case)
        while step < max_steps:
            remaining = min(5, max_steps - step)
            if remaining < 5:
                break
            if not live_mode:
                nominal = _archived_actions(archived, step, 5)
                nominal_source = "immutable_raw_AEGIS_until_first_intervention"
                query_index = step // 5
                archived_by_step = {
                    int(row["step"]): row for row in archived["actions"]
                }
                if step > 0:
                    initial_z = np.asarray(
                        archived_by_step[step - 1]["qp"]["z_after"],
                        dtype=np.float64,
                    )
                else:
                    initial_z = np.asarray(
                        archived_by_step[0]["qp"]["z_before"],
                        dtype=np.float64,
                    )
            else:
                query_index = int(live_query_index)
                seed = query_seed(int(case["policy_noise_seed"]), query_index)
                policy_input = _policy_observation(
                    runtime, observation, task_description=str(task.language),
                    resize_size=224, rng_seed=seed,
                )
                query_started = time.perf_counter_ns()
                response = client.infer(policy_input)
                query_wall = (time.perf_counter_ns() - query_started) * 1.0e-9
                returned = np.asarray(response["actions"], dtype=np.float64)
                _require(returned.shape == (int(case["model_action_horizon"]), 7),
                         "live policy chunk differs")
                proposed = np.asarray(
                    [translational_action(row) for row in returned[:5]], dtype=np.float64
                )
                nominal, nominal_z_after, nominal_qp = project_through_aegis(
                    env, proposed, live_z
                )
                initial_z = np.asarray(live_z, dtype=np.float64).copy()
                nominal_source = "fresh_frozen_pi05_plus_original_AEGIS"
                query_records.append({
                    "query_index": query_index,
                    "rng_seed": seed,
                    "returned_actions_sha256": array_sha256(returned),
                    "wall_seconds": query_wall,
                    "server_timing": response.get("server_timing"),
                    "nominal_qp_records": _public(nominal_qp),
                })
                live_query_index += 1

            nominal_prediction = predict(nominal, step)
            warning = warning_trigger(
                nominal_prediction,
                buffer_m=float(config["warning_oracle"]["buffer_m"]),
                car_threshold_m=float(config["warning_oracle"]["paper_car_threshold_m"]),
            )
            selected_actions = nominal
            selected_prediction = nominal_prediction
            selected_z_after = None if not live_mode else nominal_z_after
            source_name = nominal_source
            intervention = None
            if warning:
                _require(initial_z is not None,
                         "warning lacks registered AEGIS memory")
                current = np.asarray(one_step.clearances(env)[:7], dtype=np.float64)
                links = geometry._slabbed_links(env)
                active_row = int(np.argmin(current))
                outward = (
                    np.asarray(links[active_row].center, dtype=np.float64)
                    - np.asarray(geometry.obstacle.center, dtype=np.float64)
                )
                proposal = corrected_proposal(
                    nominal, outward,
                    radius=float(
                        config["fixed_repulsion"]["requested_correction_l2_action"]
                        if requested_radius_override is None
                        else requested_radius_override
                    ),
                    action_limit=float(config["fixed_repulsion"]["action_limit"]),
                )
                selected_actions, selected_z_after, correction_qp = project_through_aegis(
                    env, proposal["actions"], initial_z
                )
                selected_prediction = predict(selected_actions, step)
                intervention = {
                    "step": step,
                    "query_index": query_index,
                    "current_row_clearance_m": current.tolist(),
                    "active_row": active_row,
                    "nominal_prediction": _public(nominal_prediction),
                    "proposal": proposal,
                    "post_AEGIS_actions": selected_actions.tolist(),
                    "post_AEGIS_correction_l2_action": float(
                        np.linalg.norm(selected_actions[:, :3] - nominal[:, :3])
                    ),
                    "corrected_prediction": _public(selected_prediction),
                    "corrected_prediction_safe": not warning_trigger(
                        selected_prediction,
                        buffer_m=float(config["warning_oracle"]["buffer_m"]),
                        car_threshold_m=float(config["warning_oracle"]["paper_car_threshold_m"]),
                    ),
                    "correction_qp_records": _public(correction_qp),
                }
                interventions.append(intervention)
                source_name = (
                    "fixed_front_loaded_radius2_outward_normal_plus_original_AEGIS"
                    if requested_radius_override is None
                    else "first_warning_calibrated_outward_normal_plus_original_AEGIS"
                )
                if first_intervention_step is None:
                    first_intervention_step = step
                    live_mode = True
                    live_query_index = query_index + 1

            predicted_final = np.asarray(_dynamic_state_vector(probe_env), dtype=np.float64).copy()
            done_in_chunk = False
            chunk_internal_records = []
            for offset, command in enumerate(selected_actions):
                current_step = step + offset
                observation, reward, done, _, internal = monitor.execute(
                    command, step=current_step
                )
                goal = _goal_progress_snapshot(
                    env, goal_atoms, step=current_step,
                    previous_values=previous_goal_values,
                )
                previous_goal_values = goal["values"]
                _require(goal["all_satisfied"] is bool(done), "goal and done differ")
                contacts = _detailed_active_obstacle_contacts(
                    env, obstacle_name, step=current_step,
                    contact_authority=contact_authority,
                )
                protected = [
                    event for event in contacts["events"]
                    if event.get("other", {}).get("classification") == "robot"
                    and _is_protected_event(event)
                ]
                terminal_frame = _processed_image(observation, "agentview_image")
                frame_integrity = {
                    "step": int(current_step),
                    **frame_integrity_metrics(terminal_frame),
                }
                frame_integrity_records.append(frame_integrity)
                _require(
                    frame_integrity["maximum_neighbor_difference"]
                    <= frame_integrity_limit,
                    "camera frame integrity failed at step %d: %.9f" % (
                        current_step,
                        frame_integrity["maximum_neighbor_difference"],
                    ),
                )
                video_writer.append_data(terminal_frame)
                terminal_eef_positions.append(
                    np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
                )
                action_records.append({
                    "step": current_step,
                    "source": source_name,
                    "action": np.asarray(command, dtype=np.float64).tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                    "boundary_protected_contact_events": _public(protected),
                    "internal": _public(internal),
                })
                chunk_internal_records.append(internal)
                if done:
                    done_in_chunk = True
                    break
            if not done_in_chunk:
                observed_final = np.asarray(_dynamic_state_vector(env), dtype=np.float64)
                absolute_error = np.abs(observed_final - predicted_final)
                clone_error = float(np.max(absolute_error))
                maximum_index = int(np.argmax(absolute_error))
                observed_simulator = np.asarray(
                    env.sim.get_state().flatten(), dtype=np.float64
                )
                predicted_simulator = np.asarray(
                    probe_env.sim.get_state().flatten(), dtype=np.float64
                )
                simulator_error = float(
                    np.max(np.abs(observed_simulator - predicted_simulator))
                )
                clone_execution_errors_m.append(clone_error)
                observed_boundary = np.asarray(
                    one_step.clearances(env)[:7], dtype=np.float64
                )
                predicted_boundary = np.asarray(
                    selected_prediction["boundary_clearance_trace_m"][-1],
                    dtype=np.float64,
                )
                clone_execution_records.append({
                    "step": int(step),
                    "source": source_name,
                    "full_dynamic_state_maximum_absolute_error": clone_error,
                    "full_dynamic_state_maximum_error_index": maximum_index,
                    "simulator_state_maximum_absolute_error": simulator_error,
                    "boundary_clearance_maximum_absolute_error_m": float(
                        np.max(np.abs(observed_boundary - predicted_boundary))
                    ),
                    "predicted_boundary_clearance_m": predicted_boundary.tolist(),
                    "observed_boundary_clearance_m": observed_boundary.tolist(),
                    "predicted_minimum_clearance_m": float(
                        selected_prediction["minimum_clearance_m"]
                    ),
                    "predicted_protected_contact_count": len(
                        selected_prediction["protected_contacts"]
                    ),
                    "predicted_car_m": float(
                        selected_prediction[
                            "maximum_active_obstacle_l1_displacement_m"
                        ]
                    ),
                    "observed_protected_contact_samples": int(sum(
                        int(row["protected_contact_samples"])
                        for row in chunk_internal_records
                    )),
                    "observed_car_m": float(max(
                        float(row["maximum_active_obstacle_l1_displacement_m"])
                        for row in chunk_internal_records
                    )),
                })
            if selected_z_after is not None:
                live_z = np.asarray(selected_z_after, dtype=np.float64).copy()
            step += len(selected_actions) if not done_in_chunk else offset + 1
            if done_in_chunk:
                break

        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        native_success = goal_summary["first_all_satisfied_step"] is not None
        timeout = bool(not native_success and step >= max_steps)
        safety = monitor.summary()
        contact_by_link = _classify_contact_links(env, safety["protected_contact_events"])
        contact_pass = sum(contact_by_link.values()) == 0
        car_pass = bool(
            safety["maximum_active_obstacle_l1_displacement_m"]
            <= float(config["warning_oracle"]["paper_car_threshold_m"])
        )
        terminal_window = np.asarray(terminal_eef_positions[-26:], dtype=np.float64)
        terminal_motion = float(
            np.max(np.linalg.norm(terminal_window - terminal_window[0], axis=1))
        ) if len(terminal_window) > 1 else 0.0
        deadlock_diagnostic = bool(
            not native_success and contact_pass and car_pass and terminal_motion < 0.002
        )
        primary = bool(native_success and contact_pass and car_pass and not timeout)
        result = {
            "schema_version": (
                RESULT_SCHEMA if result_schema_override is None
                else str(result_schema_override)
            ),
            "status": "complete",
            "scientific_result": True,
            "case_id": selected["case_id"],
            "case_index": int(case_index),
            "claim_scope": (
                config["claim_scope"] if claim_scope_override is None
                else str(claim_scope_override)
            ),
            "controller_binding": (
                None if controller_binding is None else _public(controller_binding)
            ),
            "source": source,
            "allocation": allocation,
            "config": config,
            "selection": selected,
            "sealed_raw_AEGIS": {
                "path": str(archive_path),
                "file_sha256": _file_sha256(archive_path),
                "payload_sha256": archived["result_payload_sha256"],
                "native_task_success": True,
                "raw_L5_L7_contact_pass": False,
                "paper_car_pass": False,
                "first_relevant_contact_step": selected["first_relevant_contact_step"],
                "protected_contact_body": selected["protected_contact_body"],
            },
            "pairing": pairing,
            "policy_server": server_identity,
            "policy_queries": query_records,
            "first_intervention_step": first_intervention_step,
            "warning_count": len(interventions),
            "intervention_count": len(interventions),
            "clipped_intervention_count": sum(
                bool(row["proposal"]["clipped"]) for row in interventions
            ),
            "interventions": interventions,
            "render_context": {
                "probe_constructed_before_main_renderer": True,
                "disabled_probe_image_observables": int(disabled_probe_images),
                "probe_render_resolution": int(
                    config["execution"]["probe_render_resolution"]
                ),
                "main_render_resolution": int(TABLE_RENDER_RESOLUTION),
            },
            "frame_integrity_limit": frame_integrity_limit,
            "frame_integrity_maximum_neighbor_difference": max(
                float(row["maximum_neighbor_difference"])
                for row in frame_integrity_records
            ),
            "frame_integrity_records": frame_integrity_records,
            "clone_execution_maximum_absolute_error": (
                None if not clone_execution_errors_m
                else max(clone_execution_errors_m)
            ),
            "clone_execution_errors": clone_execution_errors_m,
            "clone_execution_records": clone_execution_records,
            "clone_execution_mismatch_count": sum(
                float(value)
                > float(config["execution"]["boundary_equivalence_tolerance_m"])
                for value in clone_execution_errors_m
            ),
            "clone_execution_maximum_boundary_clearance_error_m": (
                None if not clone_execution_records else max(
                    float(row["boundary_clearance_maximum_absolute_error_m"])
                    for row in clone_execution_records
                )
            ),
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {**goal_definition, "initial": initial_goal, "summary": goal_summary},
            "physical_safety": safety,
            "contact_samples_by_link": contact_by_link,
            "raw_L5_L7_contact_pass": contact_pass,
            "paper_car_pass": car_pass,
            "native_task_success": native_success,
            "native_task_success_step": goal_summary["first_all_satisfied_step"],
            "timeout": timeout,
            "terminal_25_action_EE_motion_m": terminal_motion,
            "deadlock_diagnostic": deadlock_diagnostic,
            "primary_problem_solved": primary,
            "interpretation": (
                "safe_task_success" if primary
                else "collision_or_CAR_failure" if not (contact_pass and car_pass)
                else "contact_free_task_timeout_or_failure"
            ),
            "video": {
                "path": str(video_final),
                "file_sha256": _file_sha256(video_final),
                "frames_written": len(action_records) + 1,
                "fps": int(config["execution"]["video_fps"]),
            },
            "final_jpg": {"path": str(final_jpg), "file_sha256": _file_sha256(final_jpg)},
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _sha256(_canonical(result))
        return result
    finally:
        if video_writer is not None:
            video_writer.close()
        if env is not None:
            env.close()
        if probe_env is not None:
            probe_env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        population_manifest_path=args.population_manifest.resolve(),
        selection_manifest_path=args.selection_manifest.resolve(),
        table1_root=args.table1_root.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        case_index=args.case_index,
        expected_commit=args.expected_commit,
        host=args.host,
        port=args.port,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case_id"],
        "interpretation": result["interpretation"],
        "primary_problem_solved": result["primary_problem_solved"],
        "warning_count": result["warning_count"],
        "native_task_success": result["native_task_success"],
        "raw_L5_L7_contact_pass": result["raw_L5_L7_contact_pass"],
        "paper_car_pass": result["paper_car_pass"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
