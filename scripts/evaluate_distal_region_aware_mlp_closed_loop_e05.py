#!/usr/bin/env python3
"""Conditionally run learned regional receding QPs on primary E05."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.region_aware_mlp import (
    CLOSED_LOOP_RESULT_SCHEMA, RESULT_SCHEMA, VALIDATION_SCHEMA,
    load_config, load_model,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import (
    _disable_probe_images, _geometry,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _is_protected_event,
    _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    import numpy as np

    if not values:
        return {"count": 0, "mean": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)), "maximum": float(np.max(array)),
    }


def main() -> int:
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
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.region_aware_closed_loop import (
        RegionAwareRecedingFilter,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config
    from main.multilink_ellipsoid.two_step_margin import feature_context, feature_vectors

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--learned-result", type=Path, required=True)
    parser.add_argument("--learned-validation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    repo = args.repo_root.resolve()
    manifest_path = args.manifest.resolve()
    archived_path = args.archived.resolve()
    geometry_path = args.geometry_config.resolve()
    exact_box_path = args.exact_box_config.resolve()
    config_path = args.config.resolve()
    learned_result_path = args.learned_result.resolve()
    learned_validation_path = args.learned_validation.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    config = load_config(config_path)
    learned = _load(learned_result_path)
    learned_validation = _load(learned_validation_path)
    archived = _load(archived_path)
    source = config["immutable_source"]
    _require(
        _file_sha256(manifest_path) == config["source_population_manifest_sha256"]
        and _file_sha256(geometry_path) == config["geometry_config_file_sha256"]
        and _file_sha256(exact_box_path) == config["exact_box_config_file_sha256"]
        and _file_sha256(archived_path) == source["archived_e05_file_sha256"]
        and archived.get("result_payload_sha256")
        == source["archived_e05_payload_sha256"]
        and learned.get("schema_version") == RESULT_SCHEMA
        and learned.get("result_payload_sha256")
        == _hash_without(learned, "result_payload_sha256")
        and learned.get("model_artifact", {}).get("file_sha256")
        == _file_sha256(model_path)
        and learned_validation.get("schema_version") == VALIDATION_SCHEMA
        and learned_validation.get("status") == "valid"
        and learned_validation.get("result_file_sha256")
        == _file_sha256(learned_result_path)
        and learned_validation.get("closed_loop_e05_authorized") is True
        and learned.get("decision", {}).get("closed_loop_e05_authorized") is True,
        "region-aware closed-loop authorization differs",
    )
    rows = read_jsonl(manifest_path)
    case_id = config["closed_loop"]["case_id"]
    matches = [item for item in rows if item.get("case_id") == case_id]
    _require(len(matches) == 1, "region-aware E05 manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo)
    archived_actions = archived["actions"]
    _require(len(archived_actions) == 237, "region-aware E05 action ledger differs")
    runtime = _runtime_imports(include_aegis=True)
    geometry_config = load_shadow_config(geometry_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_path)
    models, model_state = load_model(model_path)
    env = probe_env = None
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
            np.array_equal(np.asarray(probe_initial_state), np.asarray(selected_initial_state))
            and np.array_equal(
                np.asarray(probe_env.sim.get_state().flatten()),
                np.asarray(env.sim.get_state().flatten()),
            ),
            "region-aware settled probe state differs",
        )
        disabled_probe_images = _disable_probe_images(probe_env)
        obstacle_name, _ = _active_obstacle(env, observation)
        probe_obstacle, _ = _active_obstacle(probe_env, probe_observation)
        _require(
            probe_obstacle == obstacle_name == archived["obstacle"]["active_name"],
            "region-aware active obstacle differs",
        )
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        pairing = pairing_record(
            case=case, selected_initial_state=selected_initial_state,
            settled_observation=observation, task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64
            ),
        )
        for key in (
            "manifest_row_sha256", "initial_state_sha256",
            "initial_observation_sha256", "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == archived["pairing"][key],
                     "region-aware pairing differs: %s" % key)
        pairing["initial_policy_action_chunk_sha256"] = archived["pairing"][
            "initial_policy_action_chunk_sha256"
        ]
        pairing["nominal_prefix_source"] = (
            "immutable_successful_released_aegis_env_step_input"
        )
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config, exact_box_config=exact_box_config,
            archived=archived, env=env, obstacle_name=obstacle_name,
        )
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=obstacle_name,
            quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        controller = RegionAwareRecedingFilter(
            models=models, model_state=model_state, config=config, probe=probe
        )
        perception = archived["perception"]
        proxy = _eef_proxy(runtime, observation)
        p2 = np.asarray(perception["mvee_center"], dtype=np.float64)
        z_fixed = p2 - np.asarray(proxy["p1"], dtype=np.float64)
        _require(float(np.linalg.norm(z_fixed)) > 1.0e-12,
                 "region-aware released AEGIS direction is degenerate")
        z_fixed /= np.linalg.norm(z_fixed)
        released_aegis_geometry = {
            "p2": p2,
            "R2": np.asarray(perception["mvee_rotation"], dtype=np.float64),
            "Q2_diag": np.asarray(perception["mvee_semiaxes"], dtype=np.float64),
            "z_fixed": z_fixed,
        }
        q1_diag = np.asarray([0.06, 0.12, 0.11], dtype=np.float64) ** 2
        client = runtime["websocket_client_policy"].WebsocketClientPolicy(
            args.host, args.port
        )
        server_identity = _server_identity(client)
        action_plan = collections.deque()
        policy_queries = []
        recovery_active = False
        recovery_activation_step = None
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env, goal_atoms, step=-1, previous_values=None
        )
        previous_goal_values = initial_goal["values"]
        contact_authority = _contact_model_authority(env, obstacle_name)
        action_records = []
        first_robot_contact_step = None
        first_protected_contact_step = None
        first_car_step = None
        maximum_displacement = 0.0
        direct_robot_contact_geoms = set()
        direct_protected_contact_geoms = set()
        failure = None
        terminal_frame = _processed_image(observation, "agentview_image")
        if video_partial.exists():
            raise ValueError("region-aware video partial already exists")
        video_writer = runtime["imageio"].get_writer(
            str(video_partial), fps=TABLE_VIDEO_FPS, codec="libx264",
            macro_block_size=None, pixelformat="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        video_writer.append_data(terminal_frame)
        frames_written = 1
        maximum_steps = min(
            int(config["closed_loop"]["maximum_steps"]), max_steps_for_case(case)
        )
        for step in range(maximum_steps):
            nominal_raw = None
            released_qp = None
            second_released_qp = None
            policy_query_index = None
            if not recovery_active:
                _require(step < len(archived_actions),
                         "region-aware immutable prefix ended before intervention")
                source_action = archived_actions[step]
                _require(int(source_action["step"]) == step,
                         "region-aware archived action index differs")
                nominal = np.asarray(source_action["env_step_input"], dtype=np.float64)
                if step + 1 < len(archived_actions):
                    second = np.asarray(
                        archived_actions[step + 1]["env_step_input"], dtype=np.float64
                    )
                else:
                    second = nominal.copy()
                nominal_source = "immutable_released_aegis_prefix"
            else:
                if not action_plan:
                    query_index = (
                        step // int(case.get("replan_steps", 5))
                        if not policy_queries
                        else int(policy_queries[-1]["query_index"]) + 1
                    )
                    seed = query_seed(int(case["policy_noise_seed"]), query_index)
                    policy_input = _policy_observation(
                        runtime, observation, task_description=str(task.language),
                        resize_size=224, rng_seed=seed,
                    )
                    query_started = time.perf_counter_ns()
                    response = client.infer(policy_input)
                    query_seconds = (time.perf_counter_ns() - query_started) * 1.0e-9
                    returned = np.asarray(response.get("actions"), dtype=np.float64)
                    _require(
                        returned.shape == (int(case["model_action_horizon"]), 7)
                        and np.all(np.isfinite(returned)),
                        "region-aware live pi0.5 chunk differs",
                    )
                    replan_steps = int(case.get("replan_steps", 5))
                    action_plan.extend(returned[index].copy() for index in range(replan_steps))
                    policy_queries.append({
                        "query_index": query_index, "step": step, "rng_seed": seed,
                        "returned_actions_sha256": array_sha256(returned),
                        "wall_seconds": query_seconds,
                        "server_timing": response.get("server_timing"),
                    })
                policy_query_index = int(policy_queries[-1]["query_index"])
                nominal_raw = np.asarray(action_plan.popleft(), dtype=np.float64)
                nominal_list, released_qp = _aegis_action(
                    runtime, nominal_translational=translational_action(nominal_raw),
                    proxy=proxy, geometry=released_aegis_geometry,
                    q1_diag=q1_diag, diagnostics_enabled=True,
                )
                nominal = np.asarray(nominal_list, dtype=np.float64)
                if action_plan:
                    second_raw = np.asarray(action_plan[0], dtype=np.float64)
                    second_list, second_released_qp = _aegis_action(
                        runtime, nominal_translational=translational_action(second_raw),
                        proxy=proxy, geometry=released_aegis_geometry,
                        q1_diag=q1_diag, diagnostics_enabled=True,
                    )
                    second = np.asarray(second_list, dtype=np.float64)
                else:
                    second = nominal.copy()
                nominal_source = "live_pi05_libero_then_released_aegis"
            context = feature_context(env, probe)
            _, pair_features = feature_vectors(
                context, nominal[:3], nominal[:3], second[:3]
            )
            executed, filter_record = controller.propose(
                env, nominal, second, pair_features,
                context["current_clearance_m"], step=step
            )
            if executed is None:
                failure = {
                    "component": "learned_regional_QP", "step": step,
                    "reason": filter_record["reason"],
                }
                action_records.append({
                    "step": step, "nominal_source": nominal_source,
                    "policy_query_index": policy_query_index,
                    "filter": filter_record, "executed": False,
                })
                break
            observation, reward, done, _ = env.step(executed.tolist())
            actual_hash = hashlib.sha256(
                np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
            ).hexdigest()
            clone_hash = filter_record["exact_measurement"][
                "first_transition_next_state_sha256"
            ]
            clone_match = actual_hash == clone_hash
            _require(clone_match, "region-aware clone/execution state differs")
            proxy = _eef_proxy(runtime, observation)
            if filter_record["modified"] and not recovery_active:
                recovery_active = True
                recovery_activation_step = step
                action_plan.clear()
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=step, previous_values=previous_goal_values
            )
            previous_goal_values = goal["values"]
            _require(goal["all_satisfied"] is bool(done),
                     "region-aware native goal vector differs")
            contacts = _detailed_active_obstacle_contacts(
                env, obstacle_name, step=step, contact_authority=contact_authority
            )
            _require(contacts["status"] == "available",
                     "region-aware contact evidence unavailable")
            robot_events = [
                event for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            protected_events = [
                event for event in robot_events if _is_protected_event(event)
            ]
            if robot_events and first_robot_contact_step is None:
                first_robot_contact_step = step
            if protected_events and first_protected_contact_step is None:
                first_protected_contact_step = step
            for event in robot_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_robot_contact_geoms.add(name)
            for event in protected_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_protected_contact_geoms.add(name)
            displacement = float(np.sum(np.abs(
                np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
                - initial_obstacle_position
            )))
            maximum_displacement = max(maximum_displacement, displacement)
            if first_car_step is None and displacement > 0.001:
                first_car_step = step
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            frames_written += 1
            action_records.append({
                "step": step, "nominal_source": nominal_source,
                "policy_query_index": policy_query_index,
                "nominal_raw_policy_action": (
                    None if nominal_raw is None else nominal_raw.tolist()
                ),
                "released_AEGIS_QP": released_qp,
                "second_released_AEGIS_QP": second_released_qp,
                "nominal_action": nominal.tolist(),
                "second_prediction_action": second.tolist(),
                "executed_action": executed.tolist(), "filter": filter_record,
                "clone_execution_state_match": clone_match,
                "actual_next_state_sha256": actual_hash,
                "reward": float(reward), "done": bool(done),
                "goal_progress": goal,
                "eef_position_m": np.asarray(
                    observation["robot0_eef_pos"], dtype=np.float64
                ).tolist(),
                "active_obstacle_l1_displacement_m": displacement,
                "robot_contact_events": robot_events,
                "protected_link_contact_events": protected_events,
            })
            if done:
                break
        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        executed_records = [
            item for item in action_records if item.get("executed") is not False
        ]
        goal_summary = _goal_progress_summary(initial_goal, executed_records)
        success_step = goal_summary["first_all_satisfied_step"]
        exact_records = [
            item["filter"]["exact_measurement"] for item in executed_records
            and item.get("filter", {}).get("exact_measurement") is not None
        ]
        minimum_distal = (
            None if not exact_records else float(min(
                min(item["minimum_distal_margin_m"]) for item in exact_records
            ))
        )
        exact_safe_count = sum(item["true_distal_safe"] for item in exact_records)
        modified_count = sum(
            bool(item.get("filter", {}).get("modified")) for item in action_records
        )
        protected_contact_pass = first_protected_contact_step is None
        paper_car_pass = first_car_step is None
        native_success = success_step is not None
        primary_solved = bool(
            failure is None and protected_contact_pass and paper_car_pass
            and native_success and len(exact_records) == len(executed_records)
            and exact_safe_count == len(exact_records)
            and minimum_distal is not None and minimum_distal >= 0.0
        )
        filter_times = [
            float(item["filter"]["total_filter_wall_seconds"])
            for item in action_records if item.get("filter", {}).get(
                "total_filter_wall_seconds"
            ) is not None
        ]
        qp_times = [
            float(item["filter"]["QP_wall_seconds"])
            for item in action_records if item.get("filter", {}).get(
                "QP_wall_seconds"
            ) is not None
        ]
        result = {
            "schema_version": CLOSED_LOOP_RESULT_SCHEMA,
            "status": "complete" if failure is None else "method_failure",
            "scientific_result": True, "case_id": case_id,
            "claim_scope": config["claim_scope"],
            "source": _git_identity(repo, args.expected_commit),
            "allocation": allocation_record(), "config": config,
            "learned_authorization": {
                "result_path": str(learned_result_path),
                "result_file_sha256": _file_sha256(learned_result_path),
                "result_payload_sha256": learned["result_payload_sha256"],
                "validation_path": str(learned_validation_path),
                "validation_file_sha256": _file_sha256(learned_validation_path),
                "model_file_sha256": _file_sha256(model_path),
            },
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": _file_sha256(archived_path),
                "payload_sha256": archived["result_payload_sha256"],
                "read_only": True,
            },
            "pairing": pairing, "policy_server": server_identity,
            "policy_query_count": len(policy_queries),
            "policy_queries": policy_queries,
            "hybrid_recovery": {
                "enabled": True, "activation_step": recovery_activation_step,
                "live_recovery_started": recovery_active,
            },
            "probe_environment": {
                "disabled_image_observable_count": disabled_probe_images,
                "osc_controller": "OSC_POSE", "control_frequency_hz": 20,
                "exact_rollout_role": config["closed_loop"][
                    "exact_cloned_OSC_rollout_role"
                ],
            },
            "action_count": len(action_records), "actions": action_records,
            "goal_progress": {
                **goal_definition, "initial": initial_goal, "summary": goal_summary,
            },
            "raw_simulation_evidence": {
                "first_robot_contact_step": first_robot_contact_step,
                "first_protected_link_contact_step": first_protected_contact_step,
                "direct_robot_contact_geoms": sorted(direct_robot_contact_geoms),
                "direct_protected_link_contact_geoms": sorted(
                    direct_protected_contact_geoms
                ),
                "first_paper_car_step": first_car_step,
                "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
                "paper_car_pass": paper_car_pass,
                "protected_link_contact_pass": protected_contact_pass,
                "native_task_success": native_success,
                "native_task_success_step": success_step,
                "minimum_exact_two_step_distal_margin_m": minimum_distal,
                "exact_safe_measurement_count": exact_safe_count,
            },
            "filter_summary": {
                "modified_action_count": modified_count,
                "valid_action_count": len(exact_records),
                "filter_wall_seconds": _stats(filter_times),
                "all_27_QP_wall_seconds": _stats(qp_times),
            },
            "video": {
                "path": str(video_final), "file_sha256": _file_sha256(video_final),
                "frames_written": frames_written, "fps": TABLE_VIDEO_FPS,
            },
            "final_jpg": {
                "path": str(final_jpg), "file_sha256": _file_sha256(final_jpg),
                "frame_sha256": array_sha256(terminal_frame),
            },
            "primary_problem_solved": primary_solved, "failure": failure,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _hash_without(
            result, "result_payload_sha256"
        )
        _atomic_write(output_path, result)
        print(json.dumps({
            "status": result["status"],
            "primary_problem_solved": primary_solved,
            "raw_simulation_evidence": result["raw_simulation_evidence"],
            "output": str(output_path),
        }, sort_keys=True), flush=True)
        return 0
    finally:
        if video_writer is not None:
            video_writer.close()
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
