#!/usr/bin/env python3
"""Live pi0.5-LIBERO joint-velocity competence and distal safety demo."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from scripts.evaluate_embodisteer_joint_baselines_e05 import (
    CASE_ID,
    PAPER_CAR_THRESHOLD_M,
    _atomic_write,
    _camera_state,
    _checkpoint_tree_record,
    _file_sha256,
    _frame_orientation,
    _frame_quality,
    _git_identity,
    _joint_state,
    _protected_events,
    _reference_state,
    _require,
    _response_actions,
    _sha256,
)


PAIR_SCHEMA = "vlsa_distal_joint_velocity_demo_result.v1"
WORKER_SCHEMA = "vlsa_distal_joint_velocity_demo_worker.v1"
ARM_SCHEMA = "vlsa_distal_joint_velocity_demo_arm.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _build_joint_environment(
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[Any, Any, Mapping[str, Any]]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        _build_environment,
    )

    env, task, _, initial = _build_environment(
        runtime,
        case,
        render_resolution=TABLE_RENDER_RESOLUTION,
        controller_configs=config["controller"],
        control_frequency_hz=int(
            config["action_protocol"]["control_frequency_hz"]
        ),
        ignore_done=True,
    )
    _require(
        np.array_equal(np.asarray(initial), reference["selected_initial_state"]),
        "joint-velocity selected initial state differs",
    )
    observation = env.regenerate_obs_from_state(reference["state"])
    _require(
        np.array_equal(
            np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
            reference["state"],
        ),
        "joint-velocity settled-state transplant differs",
    )
    _require(str(task.language) == reference["task_language"], "task differs")
    return env, task, observation


def _timing_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if not records:
        return {"step_count": 0}

    def stats(values: Any) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_seconds": float(np.mean(array)),
            "p95_seconds": float(np.quantile(array, 0.95)),
            "maximum_seconds": float(np.max(array)),
        }

    return {
        "step_count": len(records),
        "total_filter": stats(
            [item["timing"]["total_filter_wall_seconds"] for item in records]
        ),
        "geometry_and_jacobian": stats(
            [
                item["timing"]["geometry_and_jacobian_wall_seconds"]
                for item in records
            ]
        ),
        "qp": stats(
            [
                item["qp"]["diagnostics"]["timing"]["total_wall_seconds"]
                for item in records
            ]
        ),
        "material_intervention_count": sum(
            bool(item["qp"]["material_intervention"]) for item in records
        ),
    }


def _run_arm(
    *,
    arm: str,
    runtime: Mapping[str, Any],
    client: Any,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    reference: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        _active_obstacle,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _policy_observation,
        _processed_image,
        array_sha256,
        pairing_record,
        query_seed,
        translational_action,
    )
    from main.multilink_ellipsoid.distal_joint_velocity_demo import (
        DistalJointVelocityDemoFilter,
    )

    constraints_enabled = arm == config["arms"][1]
    _require(
        constraints_enabled or arm == config["arms"][0],
        "unknown joint-velocity demo arm",
    )
    arm_root = output_root / "arms" / arm
    arm_root.mkdir(parents=True, exist_ok=False)
    partial_video = arm_root / "episode.partial.mp4"
    final_video = arm_root / "episode.mp4"
    final_jpg = arm_root / "final.jpg"
    env = None
    writer = None
    frames = 0
    started = time.perf_counter_ns()
    result: dict[str, Any]
    try:
        env, task, observation = _build_joint_environment(
            runtime, case, config, reference
        )
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(obstacle_name == reference["obstacle_name"], "obstacle differs")
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        pairing = pairing_record(
            case=case,
            selected_initial_state=reference["selected_initial_state"],
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(env.sim.get_state().flatten()),
        )
        _require(
            pairing["settled_simulator_state_sha256"]
            == reference["pairing"]["settled_simulator_state_sha256"],
            "joint-velocity settled pairing differs",
        )
        controller = DistalJointVelocityDemoFilter.from_config(config)
        geometry = (
            controller.geometry_record(env) if constraints_enabled else None
        )
        if geometry is not None:
            _require(
                geometry["distal_ellipsoid_count"] == 7
                and geometry["total_constraint_geometry_count"] == 8,
                "active geometry count differs",
            )
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env, goal_atoms, step=-1, previous_values=None
        )
        previous_goal_values = initial_goal["values"]
        contact_authority = _contact_model_authority(env, obstacle_name)
        writer = runtime["imageio"].get_writer(
            str(partial_video),
            fps=int(config["action_protocol"]["control_frequency_hz"]),
            codec="libx264",
            macro_block_size=None,
            output_params=[
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                "18",
            ],
        )
        frame = _processed_image(observation, "agentview_image")
        upright_reference = frame.copy()
        initial_quality = _frame_quality(frame)
        initial_orientation = _frame_orientation(frame, upright_reference)
        _require(
            initial_quality["passing"] and initial_orientation["passing"],
            "initial frame failed visual integrity",
        )
        initial_camera = _camera_state(env, "agentview")
        writer.append_data(frame)
        frames += 1

        action_plan: collections.deque[dict[str, Any]] = collections.deque()
        policy_queries = []
        actions = []
        filter_records = []
        robot_contact_geoms: set[str] = set()
        protected_contact_geoms: set[str] = set()
        first_robot_contact = None
        first_protected_contact = None
        first_car = None
        first_success = None
        maximum_displacement = 0.0
        failure = None
        max_actions = int(config["action_protocol"]["max_actions"])
        execute_count = int(
            config["action_protocol"]["execute_actions_per_query"]
        )
        for step in range(max_actions):
            if not action_plan:
                query_index = len(policy_queries)
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
                chunk = _response_actions(response)
                query_wall = (time.perf_counter_ns() - query_started) * 1.0e-9
                for chunk_index in range(execute_count):
                    action_plan.append(
                        {
                            "query_index": query_index,
                            "chunk_index": chunk_index,
                            "action": np.asarray(
                                translational_action(chunk[chunk_index]),
                                dtype=np.float64,
                            ),
                        }
                    )
                policy_queries.append(
                    {
                        "query_index": query_index,
                        "rng_seed": seed,
                        "returned_actions_sha256": array_sha256(chunk),
                        "executed_translational_prefix_sha256": array_sha256(
                            np.asarray(
                                [
                                    translational_action(chunk[index])
                                    for index in range(execute_count)
                                ],
                                dtype=np.float64,
                            )
                        ),
                        "wall_seconds": query_wall,
                        "server_timing": response.get("server_timing"),
                    }
                )

            planned = action_plan.popleft()
            source_action = np.asarray(planned["action"], dtype=np.float64)
            pre_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            pre_qpos, pre_qvel = _joint_state(env)
            safe_qdot, filter_record = controller.solve(
                env,
                source_action,
                constraints_enabled=constraints_enabled,
                step=step,
            )
            filter_records.append(filter_record)
            if safe_qdot is None:
                failure = {
                    "component": "joint_velocity_multi_constraint_qp",
                    "step": step,
                    "reason": filter_record["qp"]["reason"],
                }
                break
            env_action = np.concatenate(
                (np.asarray(safe_qdot, dtype=np.float64), [source_action[6]])
            )
            step_started = time.perf_counter_ns()
            observation, reward, done, _ = env.step(env_action)
            step_wall = (time.perf_counter_ns() - step_started) * 1.0e-9
            post_qpos, post_qvel = _joint_state(env)
            goal = _goal_progress_snapshot(
                env,
                goal_atoms,
                step=step,
                previous_values=previous_goal_values,
            )
            previous_goal_values = goal["values"]
            _require(goal["all_satisfied"] is bool(done), "goal and done differ")
            if goal["all_satisfied"] and first_success is None:
                first_success = step
            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=step,
                contact_authority=contact_authority,
            )
            _require(contacts["status"] == "available", "raw contacts unavailable")
            robot_events = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            protected_events = _protected_events(robot_events)
            if robot_events and first_robot_contact is None:
                first_robot_contact = step
            if protected_events and first_protected_contact is None:
                first_protected_contact = step
            for event in robot_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    robot_contact_geoms.add(name)
            for event in protected_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    protected_contact_geoms.add(name)
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
            if first_car is None and displacement > PAPER_CAR_THRESHOLD_M:
                first_car = step
            post_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            qdot_tracking_error = float(
                np.linalg.norm(
                    np.asarray(post_qvel, dtype=np.float64)
                    - np.asarray(safe_qdot, dtype=np.float64)
                )
            )
            actions.append(
                {
                    "step": step,
                    "query_index": int(planned["query_index"]),
                    "chunk_index": int(planned["chunk_index"]),
                    "pre_state_sha256": array_sha256(pre_state),
                    "post_state_sha256": array_sha256(post_state),
                    "source_cartesian_action": source_action.tolist(),
                    "executed_env_action": env_action.tolist(),
                    "pre_joint_position_rad": pre_qpos,
                    "pre_joint_velocity_rad_s": pre_qvel,
                    "post_joint_position_rad": post_qpos,
                    "post_joint_velocity_rad_s": post_qvel,
                    "joint_velocity_tracking_error_l2_rad_s": qdot_tracking_error,
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                    "active_obstacle_l1_displacement_m": displacement,
                    "robot_contact_events": robot_events,
                    "protected_link_contact_events": protected_events,
                    "filter": filter_record,
                    "env_step_wall_seconds": step_wall,
                }
            )
            frame = _processed_image(observation, "agentview_image")
            quality = _frame_quality(frame)
            orientation = _frame_orientation(frame, upright_reference)
            _require(
                quality["passing"] and orientation["passing"],
                "agent-view frame failed integrity at step %d" % step,
            )
            actions[-1]["agentview_frame_quality"] = quality
            actions[-1]["agentview_frame_orientation"] = orientation
            writer.append_data(frame)
            frames += 1
            if first_success is not None:
                break

        goal_summary = _goal_progress_summary(initial_goal, actions)
        tracking_errors = [
            item["joint_velocity_tracking_error_l2_rad_s"] for item in actions
        ]
        evidence = {
            "native_task_success": first_success is not None,
            "native_task_success_step": first_success,
            "first_robot_contact_step": first_robot_contact,
            "first_protected_link_contact_step": first_protected_contact,
            "direct_robot_contact_geoms": sorted(robot_contact_geoms),
            "direct_protected_link_contact_geoms": sorted(
                protected_contact_geoms
            ),
            "first_paper_car_step": first_car,
            "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
            "paper_car_pass": first_car is None,
            "protected_link_contact_pass": first_protected_contact is None,
        }
        result = {
            "schema_version": ARM_SCHEMA,
            "status": (
                "complete"
                if failure is None
                else "method_failure"
            ),
            "arm": arm,
            "case_id": CASE_ID,
            "constraints_enabled": constraints_enabled,
            "constraint_count_per_step": 8 if constraints_enabled else 0,
            "geometry": geometry,
            "pairing": pairing,
            "policy_queries": policy_queries,
            "action_count": len(actions),
            "actions": actions,
            "goal_progress": {
                **goal_definition,
                "initial": initial_goal,
                "summary": goal_summary,
            },
            "raw_simulation_evidence": evidence,
            "joint_velocity_execution": {
                "tracking_error_l2_rad_s_mean": (
                    None if not tracking_errors else float(np.mean(tracking_errors))
                ),
                "tracking_error_l2_rad_s_maximum": (
                    None if not tracking_errors else float(np.max(tracking_errors))
                ),
            },
            "filter_timing": _timing_summary(filter_records),
            "failure": failure,
            "visual_integrity": {
                "initial_frame": initial_quality,
                "initial_orientation": initial_orientation,
                "initial_camera": initial_camera,
                "all_frames_passing": all(
                    item["agentview_frame_quality"]["passing"]
                    and item["agentview_frame_orientation"]["passing"]
                    for item in actions
                ),
            },
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        runtime["imageio"].imwrite(str(final_jpg), frame)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if env is not None:
                env.close()
    _require(
        partial_video.is_file() and partial_video.stat().st_size > 0,
        "joint-velocity demo video missing",
    )
    os.replace(partial_video, final_video)
    result["video"] = {
        "path": str(final_video.relative_to(output_root)),
        "sha256": _file_sha256(final_video),
        "frames": frames,
        "fps": int(config["action_protocol"]["control_frequency_hz"]),
    }
    result["final_jpg"] = {
        "path": str(final_jpg.relative_to(output_root)),
        "sha256": _file_sha256(final_jpg),
    }
    result["result_payload_sha256"] = _sha256(_canonical(result))
    return result


def evaluate_worker(
    *,
    arm: str,
    repo_root: Path,
    manifest_path: Path,
    config_path: Path,
    expected_commit: str,
    host: str,
    port: int,
    artifact_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    from main.evaluate_safelibero_aegis import (
        _runtime_imports,
        _server_identity,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.distal_joint_velocity_demo import (
        load_demo_config,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    _require(not output_path.exists(), "worker result already exists")
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary demo manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    config = load_demo_config(config_path)
    _require(arm in config["arms"], "worker arm differs")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
    server = _server_identity(client)
    reference = _reference_state(
        runtime, case, int(config["pairing"]["settle_actions"])
    )
    arm_result = _run_arm(
        arm=arm,
        runtime=runtime,
        client=client,
        case=case,
        config=config,
        reference=reference,
        output_root=artifact_root,
    )
    wrapper = {
        "schema_version": WORKER_SCHEMA,
        "status": "complete",
        "arm": arm,
        "source": source,
        "allocation": allocation,
        "config_payload_sha256": config["config_payload_sha256"],
        "policy_server": server,
        "reference_settled_state_sha256": reference["pairing"][
            "settled_simulator_state_sha256"
        ],
        "arm_result": arm_result,
    }
    wrapper["result_payload_sha256"] = _sha256(_canonical(wrapper))
    _atomic_write(output_path, wrapper)
    return wrapper


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    expected_commit: str,
    host: str,
    port: int,
    output_path: Path,
) -> dict[str, Any]:
    from main.evaluate_safelibero_aegis import read_jsonl, validate_case_row
    from main.multilink_ellipsoid.distal_joint_velocity_demo import (
        load_demo_config,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    _require(not output_path.exists(), "demo result already exists")
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary demo manifest row is not unique")
    validate_case_row(matches[0], repo_root)
    config = load_demo_config(config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    checkpoint = _checkpoint_tree_record(checkpoint_path)
    output_root = output_path.parent
    worker_root = output_root / "workers"
    script_path = Path(__file__).resolve()

    def run_worker(arm: str) -> dict[str, Any]:
        worker_output = worker_root / (arm + ".json")
        command = [
            sys.executable,
            str(script_path),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest_path),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--expected-commit",
            expected_commit,
            "--host",
            host,
            "--port",
            str(port),
            "--output",
            str(worker_output),
            "--artifact-root",
            str(output_root),
            "--worker-arm",
            arm,
        ]
        subprocess.run(command, check=True)
        return json.loads(worker_output.read_text(encoding="utf-8"))

    baseline_worker = run_worker(config["arms"][0])
    baseline = baseline_worker["arm_result"]
    baseline_competent = bool(
        baseline["status"] == "complete"
        and baseline["raw_simulation_evidence"]["native_task_success"]
    )
    active_worker = None
    active = None
    if baseline_competent:
        active_worker = run_worker(config["arms"][1])
        active = active_worker["arm_result"]
    ability_demonstrated = bool(
        active is not None
        and active["status"] == "complete"
        and active["raw_simulation_evidence"]["native_task_success"]
        and active["raw_simulation_evidence"]["protected_link_contact_pass"]
        and active["raw_simulation_evidence"]["paper_car_pass"]
    )
    if not baseline_competent:
        interpretation = "live_joint_velocity_baseline_failed_competence_gate"
    elif ability_demonstrated:
        interpretation = "primary_distal_joint_velocity_ability_demonstrated"
    elif active is not None and active["status"] != "complete":
        interpretation = "active_joint_velocity_method_failure"
    else:
        interpretation = "active_joint_velocity_safety_or_task_failure"
    workers = [baseline_worker] + (
        [] if active_worker is None else [active_worker]
    )
    for worker in workers:
        _require(worker["source"] == source, "worker source differs")
        _require(
            worker["allocation"]["slurm_job_id"] == allocation["slurm_job_id"],
            "worker allocation differs",
        )
    initial_action_chunks_equal = None
    if active is not None:
        _require(
            baseline["pairing"]["settled_simulator_state_sha256"]
            == active["pairing"]["settled_simulator_state_sha256"],
            "paired settled state differs",
        )
        initial_action_chunks_equal = bool(
            baseline["policy_queries"][0]["returned_actions_sha256"]
            == active["policy_queries"][0]["returned_actions_sha256"]
        )
        _require(initial_action_chunks_equal, "initial live policy chunk differs")
    result = {
        "schema_version": PAIR_SCHEMA,
        "status": "complete",
        "case_id": CASE_ID,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation,
        "config": config,
        "checkpoint": checkpoint,
        "policy_server": baseline_worker["policy_server"],
        "baseline": baseline,
        "active": active,
        "baseline_competent": baseline_competent,
        "active_skipped_due_to_competence_gate": not baseline_competent,
        "initial_live_policy_action_chunks_equal": initial_action_chunks_equal,
        "ability_demonstrated": ability_demonstrated,
        "interpretation": interpretation,
        "worker_results": [
            str((worker_root / (worker["arm"] + ".json")).relative_to(output_root))
            for worker in workers
        ],
    }
    result["result_payload_sha256"] = _sha256(_canonical(result))
    _atomic_write(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--worker-arm")
    args = parser.parse_args(argv)
    if args.worker_arm:
        _require(args.artifact_root is not None, "worker artifact root is required")
        result = evaluate_worker(
            arm=args.worker_arm,
            repo_root=args.repo_root.resolve(),
            manifest_path=args.manifest.resolve(),
            config_path=args.config.resolve(),
            expected_commit=args.expected_commit,
            host=args.host,
            port=args.port,
            artifact_root=args.artifact_root.resolve(),
            output_path=args.output.resolve(),
        )
    else:
        result = evaluate(
            repo_root=args.repo_root.resolve(),
            manifest_path=args.manifest.resolve(),
            config_path=args.config.resolve(),
            checkpoint_path=args.checkpoint.resolve(),
            expected_commit=args.expected_commit,
            host=args.host,
            port=args.port,
            output_path=args.output.resolve(),
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "result_payload_sha256": result["result_payload_sha256"],
                **(
                    {
                        "baseline_competent": result["baseline_competent"],
                        "ability_demonstrated": result["ability_demonstrated"],
                        "interpretation": result["interpretation"],
                    }
                    if not args.worker_arm
                    else {"arm": args.worker_arm}
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
