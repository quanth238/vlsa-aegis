#!/usr/bin/env python3
"""Run predictive L5--L7 plus EE flow guidance on the primary case."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    PAPER_CAR_THRESHOLD_M,
    _atomic_write,
    _canonical,
    _file_sha256,
    _git_identity,
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


def _response_actions(response: Mapping[str, Any], *, expected_horizon: int) -> Any:
    import numpy as np

    if "actions" not in response:
        raise ValueError("pi0.5 response has no actions")
    actions = np.asarray(response["actions"], dtype=np.float64)
    if actions.shape != (expected_horizon, 7) or not np.all(np.isfinite(actions)):
        raise ValueError("pi0.5 action chunk shape or values differ")
    return actions


def _rollout_public_record(rollout: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "h_opt_m": rollout["h_opt_m"].tolist(),
        "minimum_h_opt_m": float(rollout["minimum_h_opt_m"]),
        "next_state_sha256": list(rollout["next_state_sha256"]),
        "synchronization": dict(rollout["synchronization"]),
        "step_wall_seconds": list(rollout["step_wall_seconds"]),
        "total_env_step_wall_seconds": float(
            rollout["total_env_step_wall_seconds"]
        ),
        "done_steps": list(rollout["done_steps"]),
    }


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    config_path: Path,
    expected_commit: str,
    host: str,
    port: int,
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
        validate_case_row,
    )
    from main.multilink_ellipsoid.predictive_flow import (
        ClonedOscTrajectoryProbe,
        PredictiveFullBodyGeometry,
        build_flow_guidance_envelope,
        exact_trajectory_verification,
        identify_osc_clearance_model,
        load_predictive_flow_config,
        translational_chunk,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record

    started = time.perf_counter_ns()
    archived = _load(archived_path)
    _require(
        _file_sha256(archived_path) == ARCHIVED_FILE_SHA256,
        "archived Table-1 file hash differs",
    )
    _require(
        archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
        "archived Table-1 payload identity differs",
    )
    config = load_predictive_flow_config(config_path)
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary predictive-flow manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    _require(config["case_ids"] == [CASE_ID], "predictive-flow config case differs")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    env = None
    probe_env = None
    video_writer = None
    video_partial = output_path.with_name("episode.partial.mp4")
    video_final = output_path.with_name("episode.mp4")
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime,
            case,
            render_resolution=TABLE_RENDER_RESOLUTION,
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, probe_initial_state = _build_environment(
            runtime,
            case,
            render_resolution=32,
        )
        probe_observation = _settle(
            probe_env, probe_observation, TABLE_SETTLE_ACTIONS
        )
        _require(str(probe_task.language) == str(task.language), "probe task differs")
        _require(
            np.array_equal(
                np.asarray(probe_initial_state), np.asarray(selected_initial_state)
            ),
            "probe initial state differs",
        )
        _require(
            np.array_equal(
                np.asarray(probe_env.sim.get_state().flatten()),
                np.asarray(env.sim.get_state().flatten()),
            ),
            "settled predictive probe simulator state differs",
        )
        disabled_probe_observables = _disable_probe_images(probe_env)
        obstacle_name, _ = _active_obstacle(env, observation)
        probe_obstacle_name, _ = _active_obstacle(probe_env, probe_observation)
        _require(probe_obstacle_name == obstacle_name, "probe active obstacle differs")
        _require(
            obstacle_name == archived["obstacle"]["active_name"],
            "predictive-flow active obstacle differs from Table 1",
        )
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
            _require(
                pairing[key] == archived["pairing"][key],
                "predictive-flow pairing field differs: %s" % key,
            )

        perception = archived["perception"]
        geometry = PredictiveFullBodyGeometry.from_aegis_geometry(
            config,
            {
                "p2": perception["mvee_center"],
                "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        geometry_record = geometry.geometry_record(env)
        _require(
            geometry_record["protected_body_count"] == 4,
            "predictive-flow geometry count differs",
        )
        probe = ClonedOscTrajectoryProbe(probe_env, geometry)
        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
        server_identity = _server_identity(client)

        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env, goal_atoms, step=-1, previous_values=None
        )
        previous_goal_values = initial_goal["values"]
        contact_authority = _contact_model_authority(env, obstacle_name)
        action_plan: collections.deque[Any] = collections.deque()
        expected_states: collections.deque[Any] = collections.deque()
        expected_hashes: collections.deque[str] = collections.deque()
        action_records = []
        query_records = []
        first_robot_contact_step = None
        first_car_step = None
        robot_contact_geoms = set()
        maximum_displacement = 0.0
        failure = None
        task_success = False
        success_step = None
        frames_written = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if video_partial.exists():
            video_partial.unlink()
        video_writer = runtime["imageio"].get_writer(
            str(video_partial),
            fps=TABLE_VIDEO_FPS,
            codec="libx264",
            macro_block_size=None,
        )
        trajectory = config["trajectory_model"]
        flow = config["flow_guidance"]
        clone_tolerance = float(config["verification"]["clone_state_tolerance"])
        gamma = float(flow["barrier_decay_gamma"])
        clearance_tolerance = float(trajectory["clearance_tolerance_m"])
        action_limit = float(flow["action_limit"])
        max_steps = max_steps_for_case(case)

        for step in range(max_steps):
            video_writer.append_data(
                _processed_image(observation, "agentview_image")
            )
            frames_written += 1
            if not action_plan:
                query_index = len(query_records)
                seed = query_seed(int(case["policy_noise_seed"]), query_index)
                policy_input = _policy_observation(
                    runtime,
                    observation,
                    task_description=str(task.language),
                    resize_size=224,
                    rng_seed=seed,
                )
                nominal_started = time.perf_counter_ns()
                nominal_response = client.infer(policy_input)
                nominal_infer_seconds = (
                    time.perf_counter_ns() - nominal_started
                ) * 1.0e-9
                nominal_raw = _response_actions(
                    nominal_response,
                    expected_horizon=int(
                        config["action_protocol"]["model_action_horizon"]
                    ),
                )
                nominal_hash = array_sha256(nominal_raw)
                if query_index == 0:
                    pairing["initial_policy_action_chunk_sha256"] = nominal_hash
                    _require(
                        nominal_hash
                        == archived["pairing"][
                            "initial_policy_action_chunk_sha256"
                        ],
                        "initial live pi0.5 action chunk differs from Table 1",
                    )
                current_h = geometry.clearances(env)
                nominal_actions = translational_chunk(
                    nominal_raw, action_limit=action_limit
                )
                nominal_rollout = probe.rollout(env, nominal_actions)
                if (
                    nominal_rollout["synchronization"]["maximum_absolute_error"]
                    > clone_tolerance
                ):
                    raise ValueError("nominal trajectory clone synchronization failed")
                nominal_verification = exact_trajectory_verification(
                    current_h,
                    nominal_rollout,
                    gamma=gamma,
                    tolerance_m=clearance_tolerance,
                )
                activation = bool(
                    nominal_rollout["minimum_h_opt_m"]
                    <= float(trajectory["activation_h_opt_m"])
                    or not nominal_verification["safe"]
                )
                attempts = []
                accepted_raw = nominal_raw
                accepted_actions = nominal_actions
                accepted_rollout = nominal_rollout
                accepted_verification = nominal_verification
                if activation:
                    center_raw = nominal_raw
                    accepted_raw = None
                    accepted_actions = None
                    accepted_rollout = None
                    accepted_verification = None
                    for attempt_index in range(
                        int(flow["relinearization_attempts"]) + 1
                    ):
                        model = identify_osc_clearance_model(
                            probe,
                            env,
                            center_raw,
                            perturbation_action=float(
                                trajectory["finite_difference_action"]
                            ),
                            action_limit=action_limit,
                            clone_state_tolerance=clone_tolerance,
                        )
                        envelope, constraint_record = build_flow_guidance_envelope(
                            model,
                            center_raw,
                            current_h,
                            gamma=gamma,
                            action_limit=action_limit,
                            projection_tolerance=float(
                                flow["projection_residual_tolerance"]
                            ),
                        )
                        guided_input = _policy_observation(
                            runtime,
                            observation,
                            task_description=str(task.language),
                            resize_size=224,
                            rng_seed=seed,
                        )
                        guided_input["__crfs__"]["flow_guidance"] = envelope
                        guided_started = time.perf_counter_ns()
                        guided_response = client.infer(guided_input)
                        guided_infer_seconds = (
                            time.perf_counter_ns() - guided_started
                        ) * 1.0e-9
                        guided_raw = _response_actions(
                            guided_response,
                            expected_horizon=10,
                        )
                        sampler_record = guided_response.get("flow_guidance")
                        if not isinstance(sampler_record, Mapping):
                            raise ValueError("guided policy response lacks diagnostics")
                        guided_actions = translational_chunk(
                            guided_raw, action_limit=action_limit
                        )
                        guided_rollout = probe.rollout(env, guided_actions)
                        guided_verification = exact_trajectory_verification(
                            current_h,
                            guided_rollout,
                            gamma=gamma,
                            tolerance_m=clearance_tolerance,
                        )
                        attempt = {
                            "attempt_index": attempt_index,
                            "linearization_center_action_sha256": array_sha256(
                                center_raw
                            ),
                            "osc_model": model["record"],
                            "constraints": constraint_record,
                            "sampler": dict(sampler_record),
                            "guided_policy_infer_seconds": guided_infer_seconds,
                            "guided_action_sha256": array_sha256(guided_raw),
                            "exact_verification": guided_verification,
                        }
                        attempts.append(attempt)
                        if (
                            sampler_record.get("output_constraints_satisfied")
                            is True
                            and guided_verification["safe"]
                        ):
                            accepted_raw = guided_raw
                            accepted_actions = guided_actions
                            accepted_rollout = guided_rollout
                            accepted_verification = guided_verification
                            break
                        center_raw = guided_raw
                    if accepted_actions is None:
                        failure = {
                            "component": "predictive_flow_guidance",
                            "step": step,
                            "query_index": query_index,
                            "reason": "no_exactly_verified_safe_guided_chunk",
                        }
                query_record = {
                    "query_index": query_index,
                    "step": step,
                    "rng_seed": seed,
                    "nominal_action_sha256": nominal_hash,
                    "nominal_policy_infer_seconds": nominal_infer_seconds,
                    "nominal_server_timing": nominal_response.get("server_timing"),
                    "current_h_opt_m": current_h.tolist(),
                    "nominal_rollout": _rollout_public_record(nominal_rollout),
                    "nominal_exact_verification": nominal_verification,
                    "activation": activation,
                    "guidance_attempts": attempts,
                    "accepted": accepted_actions is not None,
                    "accepted_action_sha256": (
                        None
                        if accepted_raw is None
                        else array_sha256(accepted_raw)
                    ),
                    "accepted_exact_verification": accepted_verification,
                }
                query_records.append(query_record)
                if failure is not None:
                    break
                execute_count = int(
                    config["action_protocol"]["execute_actions_per_query"]
                )
                for index in range(execute_count):
                    action_plan.append(accepted_actions[index].copy())
                    expected_states.append(
                        np.asarray(
                            accepted_rollout["next_state_vectors"][index],
                            dtype=np.float64,
                        )
                    )
                    expected_hashes.append(
                        accepted_rollout["next_state_sha256"][index]
                    )

            executed = np.asarray(action_plan.popleft(), dtype=np.float64)
            expected_state = expected_states.popleft()
            expected_state_hash = expected_hashes.popleft()
            observation, reward, done, _ = env.step(executed.tolist())
            actual_state = _dynamic_state_vector(env)
            state_error = float(np.max(np.abs(actual_state - expected_state)))
            if state_error > clone_tolerance:
                failure = {
                    "component": "cloned_osc_trajectory_fidelity",
                    "step": step,
                    "reason": "executed state differs from verified trajectory",
                    "maximum_absolute_error": state_error,
                }
            goal = _goal_progress_snapshot(
                env,
                goal_atoms,
                step=step,
                previous_values=previous_goal_values,
            )
            previous_goal_values = goal["values"]
            _require(goal["all_satisfied"] is bool(done), "native goal vector differs")
            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=step,
                contact_authority=contact_authority,
            )
            _require(contacts["status"] == "available", "contact evidence unavailable")
            robot_events = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            if robot_events and first_robot_contact_step is None:
                first_robot_contact_step = step
            for event in robot_events:
                geom_name = event["other"].get("geom_name")
                if isinstance(geom_name, str) and geom_name:
                    robot_contact_geoms.add(geom_name)
            displacement = float(
                np.sum(
                    np.abs(
                        np.asarray(
                            observation["%s_pos" % obstacle_name],
                            dtype=np.float64,
                        )
                        - initial_obstacle_position
                    )
                )
            )
            maximum_displacement = max(maximum_displacement, displacement)
            if first_car_step is None and displacement > PAPER_CAR_THRESHOLD_M:
                first_car_step = step
            action_records.append(
                {
                    "step": step,
                    "executed_action": executed.tolist(),
                    "executed_action_sha256": _sha256(
                        _canonical(executed.tolist())
                    ),
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                    "eef_position_m": np.asarray(
                        observation["robot0_eef_pos"], dtype=np.float64
                    ).tolist(),
                    "active_obstacle_l1_displacement_m": displacement,
                    "robot_contact_events": robot_events,
                    "verified_clone_next_state_sha256": expected_state_hash,
                    "main_next_state_sha256": hashlib.sha256(
                        actual_state.tobytes()
                    ).hexdigest(),
                    "main_vs_verified_clone_max_abs_error": state_error,
                }
            )
            if failure is not None:
                break
            if done:
                task_success = True
                success_step = step
                break

        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        protected_contact_pass = first_robot_contact_step is None
        paper_car_pass = first_car_step is None
        problem_solved = bool(
            failure is None
            and protected_contact_pass
            and paper_car_pass
            and task_success
            and all(
                query["accepted"]
                and query["accepted_exact_verification"]["safe"]
                for query in query_records
            )
        )
        guidance_attempts = [
            attempt
            for query in query_records
            for attempt in query["guidance_attempts"]
        ]

        def _stats(values: Sequence[float]) -> dict[str, Any] | None:
            if not values:
                return None
            array = np.asarray(values, dtype=np.float64)
            return {
                "count": int(array.size),
                "mean_seconds": float(np.mean(array)),
                "median_seconds": float(np.median(array)),
                "p95_seconds": float(np.quantile(array, 0.95)),
                "maximum_seconds": float(np.max(array)),
            }

        result = {
            "schema_version": "vlsa_predictive_flow_guidance_e05_result.v1",
            "status": "complete" if failure is None else "method_failure",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "policy_server": server_identity,
            "pairing": pairing,
            "probe_environment": {
                "same_bddl_task_episode_and_settled_state": True,
                "disabled_image_observable_count": disabled_probe_observables,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
            },
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
            "geometry": geometry_record,
            "policy_query_count": len(query_records),
            "policy_queries": query_records,
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {
                **goal_definition,
                "initial": initial_goal,
                "summary": goal_summary,
            },
            "raw_simulation_evidence": {
                "first_robot_contact_step": first_robot_contact_step,
                "direct_robot_contact_geoms": sorted(robot_contact_geoms),
                "first_paper_car_step": first_car_step,
                "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
                "protected_contact_pass": protected_contact_pass,
                "paper_car_pass": paper_car_pass,
                "native_task_success": task_success,
                "native_task_success_step": success_step,
            },
            "timing": {
                "nominal_policy_inference": _stats(
                    [
                        query["nominal_policy_infer_seconds"]
                        for query in query_records
                    ]
                ),
                "guided_policy_inference": _stats(
                    [
                        attempt["guided_policy_infer_seconds"]
                        for attempt in guidance_attempts
                    ]
                ),
                "osc_model_identification": _stats(
                    [
                        attempt["osc_model"]["model_wall_seconds"]
                        for attempt in guidance_attempts
                    ]
                ),
                "total_wall_seconds": (
                    time.perf_counter_ns() - started
                )
                * 1.0e-9,
            },
            "video": {
                "path": str(video_final),
                "file_sha256": _file_sha256(video_final),
                "frames_written": frames_written,
                "fps": TABLE_VIDEO_FPS,
            },
            "primary_problem_solved": problem_solved,
            "failure": failure,
        }
        result["result_payload_sha256"] = _sha256(_canonical(result))
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        host=args.host,
        port=args.port,
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
