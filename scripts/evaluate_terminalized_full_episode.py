#!/usr/bin/env python3
"""Run the frozen terminalized risk selector from the first policy query."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.terminalized_full_episode import (
    RESULT_SCHEMA,
    canonical,
    load_config,
    payload_sha256,
    scientific_view,
)
from scripts.evaluate_terminalized_late_flow_task_success import (
    LiveTerminalizedSelector,
    RawRobotContactMonitor,
    _allocation_record,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


def _selector_config(config: Mapping[str, Any]) -> dict[str, Any]:
    method = config["method"]
    return {
        "registered_method": {
            "config_path": method["terminal_method_config"],
            "config_file_sha256": method["terminal_method_config_file_sha256"],
            "source_context_path": method["source_context_path"],
            "source_context_file_sha256": method["source_context_file_sha256"],
            "source_context_payload_sha256": method[
                "source_context_payload_sha256"
            ],
            "source_geometry_config": method["source_geometry_config"],
            "source_geometry_config_file_sha256": method[
                "source_geometry_config_file_sha256"
            ],
        },
    }


def _run_episode(
    *, repo_root: Path, runtime: Mapping[str, Any], case: Mapping[str, Any],
    archived: Mapping[str, Any], config: Mapping[str, Any], client: Any,
    frozen_episode: Optional[Mapping[str, Any]], output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        TABLE_VIDEO_FPS,
        _active_obstacle,
        _build_environment,
        _contact_model_authority,
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _processed_image,
        _settle,
        max_steps_for_case,
        pairing_record,
        query_seed,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from scripts.replay_distal_three_ellipsoid_multicbf import (
        PAPER_CAR_THRESHOLD_M,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    video_partial = output_root / "episode.partial.mp4"
    video_final = output_root / "episode.mp4"
    final_jpg = output_root / "final.jpg"
    env = None
    writer = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION,
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle = np.asarray(
            observation[obstacle_name + "_pos"], dtype=np.float64,
        ).copy()
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64,
            ),
        )
        for key in (
            "manifest_row_sha256", "initial_state_sha256",
            "initial_observation_sha256", "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == archived["pairing"][key],
                     "full-episode pairing differs: %s" % key)
        monitor = RawRobotContactMonitor(
            env, initial_obstacle,
            _contact_model_authority(env, obstacle_name),
            config["physical_contact_groups"],
        )
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env, goal_atoms, step=-1, previous_values=None,
        )
        previous_goal = initial_goal["values"]
        producer = frozen_episode is None
        if producer:
            writer = runtime["imageio"].get_writer(
                str(video_partial), fps=TABLE_VIDEO_FPS, codec="libx264",
                macro_block_size=None, pixelformat="yuv420p",
                output_params=["-crf", "18", "-movflags", "+faststart"],
            )
        terminal_frame = _processed_image(observation, "agentview_image")
        if writer is not None:
            writer.append_data(terminal_frame)

        selector = None
        frozen_actions = None
        selections: list[dict[str, Any]] = []
        if producer:
            selector = LiveTerminalizedSelector(
                repo_root=repo_root, runtime=runtime, env=env,
                obstacle_name=obstacle_name, archived=archived,
                config=_selector_config(config), client=client,
            )
        else:
            frozen_actions = np.asarray(
                [row["executed_action"] for row in frozen_episode["actions"]],
                dtype=np.float64,
            ).reshape((-1, 7))
            _require(
                frozen_actions.ndim == 2 and frozen_actions.shape[1] == 7,
                "full-episode frozen action ledger differs",
            )
            selections = list(frozen_episode["selection_history"])

        plan: collections.deque[Any] = collections.deque()
        action_records = []
        step = 0
        query_index = 0
        done = False
        physical_failure = False
        abstained = False
        max_steps = max_steps_for_case(case)
        while step < max_steps and not done and not physical_failure:
            if frozen_actions is not None:
                if step >= len(frozen_actions):
                    break
                action = frozen_actions[step].copy()
                source_name = str(frozen_episode["actions"][step]["source"])
            else:
                if not plan:
                    seed = query_seed(
                        int(case["policy_noise_seed"]), query_index,
                    )
                    selected, selection = selector.select_safe_or_abstain(
                        observation=observation,
                        task_description=str(task.language),
                        rng_seed=seed, query_index=query_index,
                    )
                    selections.append(selection)
                    query_index += 1
                    if selected is None:
                        abstained = True
                        break
                    plan.extend(selected[index].copy() for index in range(5))
                action = np.asarray(plan.popleft(), dtype=np.float64)
                source_name = "full_episode_late_flow_safe_selector_no_QP"

            observation, reward, done, _, internal = monitor.execute(
                action, step=step,
            )
            _require(int(internal["substep_count"]) == 25,
                     "full-episode MuJoCo substep count differs")
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=step, previous_values=previous_goal,
            )
            previous_goal = goal["values"]
            _require(goal["all_satisfied"] is bool(done),
                     "full-episode goal/done differs")
            terminal_frame = _processed_image(observation, "agentview_image")
            if writer is not None:
                writer.append_data(terminal_frame)
            action_records.append({
                "step": int(step), "source": source_name,
                "executed_action": np.asarray(action, dtype=np.float64).tolist(),
                "reward": float(reward), "done": bool(done),
                "goal_progress": goal,
                "internal_physical_measurement": internal,
            })
            step += 1
            physical_failure = bool(
                monitor.first_contact is not None or monitor.first_car is not None
            )

        if frozen_actions is not None:
            _require(step == len(frozen_actions),
                     "full-episode replay action horizon differs")
            abstained = bool(
                frozen_episode["terminal_reason"]
                == "all_predicted_unsafe_abstention"
            )
        if writer is not None:
            writer.close()
            writer = None
            video_partial.replace(video_final)
            runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        task_success = goal_summary["first_all_satisfied_step"] is not None
        timeout = bool(
            not task_success and not physical_failure and not abstained
            and step >= max_steps
        )
        terminal_reason = (
            "native_task_success" if task_success and not physical_failure
            else "raw_robot_contact" if monitor.first_contact is not None
            else "paper_CAR" if monitor.first_car is not None
            else "all_predicted_unsafe_abstention" if abstained
            else "timeout" if timeout else "invalid"
        )
        _require(terminal_reason != "invalid",
                 "full-episode stopped without registered terminal")
        physical = monitor.summary()
        contact_pass = int(physical["robot_contact_sample_count"]) == 0
        car_pass = (
            float(physical["maximum_active_obstacle_l1_displacement_m"])
            <= PAPER_CAR_THRESHOLD_M
        )
        complete_actions = [row["executed_action"] for row in action_records]
        episode = {
            "actions": action_records,
            "complete_executed_actions_sha256": hashlib.sha256(
                canonical(complete_actions)
            ).hexdigest(),
            "action_count": len(action_records),
            "selection_history": selections,
            "selection_count": len(selections),
            "selection_history_sha256": hashlib.sha256(
                canonical(selections)
            ).hexdigest(),
            "abstention_count": sum(
                bool(row["abstained"]) for row in selections
            ),
            "goal_progress": {
                **goal_definition, "initial": initial_goal,
                "summary": goal_summary,
            },
            "terminal_reason": terminal_reason,
            "native_task_success": bool(task_success),
            "native_task_success_step": goal_summary["first_all_satisfied_step"],
            "timeout": timeout,
            "physical_safety": physical,
            "raw_robot_contact_pass": contact_pass,
            "first_raw_robot_contact": physical["first_robot_contact"],
            "robot_contact_sample_count": int(
                physical["robot_contact_sample_count"]
            ),
            "robot_contact_sample_count_by_group": dict(
                physical["robot_contact_sample_count_by_group"]
            ),
            "robot_contact_events_sha256": physical[
                "robot_contact_events_sha256"
            ],
            "paper_CAR_pass": car_pass,
            "first_paper_CAR_step": (
                None if physical["first_paper_CAR"] is None
                else int(physical["first_paper_CAR"]["step"])
            ),
            "maximum_active_obstacle_l1_displacement_m": float(
                physical["maximum_active_obstacle_l1_displacement_m"]
            ),
            "collision_free_task_success": bool(
                task_success and contact_pass and car_pass
            ),
            "terminal_dynamic_state_sha256": hashlib.sha256(
                np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
            ).hexdigest(),
            "goal_progress_summary_sha256": hashlib.sha256(
                canonical(goal_summary)
            ).hexdigest(),
            "video": (
                None if not producer else {
                    "path": str(video_final),
                    "file_sha256": _file_sha256(video_final),
                    "frames_written": len(action_records) + 1,
                    "fps": TABLE_VIDEO_FPS,
                }
            ),
            "final_jpg": (
                None if not producer else {
                    "path": str(final_jpg),
                    "file_sha256": _file_sha256(final_jpg),
                }
            ),
        }
        if frozen_episode is not None:
            expected = {
                key: frozen_episode[key] for key in (
                    "complete_executed_actions_sha256", "action_count",
                    "selection_count", "selection_history_sha256",
                    "abstention_count", "terminal_reason",
                    "native_task_success", "native_task_success_step", "timeout",
                    "raw_robot_contact_pass", "first_raw_robot_contact",
                    "robot_contact_sample_count",
                    "robot_contact_sample_count_by_group",
                    "robot_contact_events_sha256", "paper_CAR_pass",
                    "first_paper_CAR_step",
                    "maximum_active_obstacle_l1_displacement_m",
                    "collision_free_task_success",
                    "terminal_dynamic_state_sha256",
                    "goal_progress_summary_sha256",
                )
            }
            actual = {key: episode[key] for key in expected}
            _require(actual == expected,
                     "full-episode exact action replay differs")
        return pairing, episode
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


def evaluate(
    *, repo_root: Path, manifest_path: Path, archived_path: Path,
    config_path: Path, expected_commit: str, replica: str,
    host: str, port: int, output_path: Path,
    frozen_producer_result_path: Optional[Path] = None,
) -> dict[str, Any]:
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, read_jsonl, validate_case_row,
    )

    started = time.perf_counter_ns()
    config = load_config(config_path)
    _require(replica in ("producer", "replay"),
             "full-episode replica differs")
    _require(
        (replica == "replay") is (frozen_producer_result_path is not None),
        "full-episode producer binding differs",
    )
    source = _git_identity(repo_root, expected_commit)
    allocation = _allocation_record(require_h100=replica == "producer")
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256,
             "full-episode archived file differs")
    archived = _load(archived_path)
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
             "full-episode archived payload differs")
    matches = [
        row for row in read_jsonl(manifest_path)
        if row.get("case_id") == CASE_ID
    ]
    _require(len(matches) == 1, "full-episode manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=False)
    client = None
    server_identity = None
    frozen_episode = None
    producer_binding = None
    if replica == "producer":
        client = runtime["websocket_client_policy"].WebsocketClientPolicy(
            host, int(port),
        )
        from main.evaluate_safelibero_aegis import _server_identity
        server_identity = _server_identity(client)
    else:
        producer = _load(frozen_producer_result_path)
        _require(
            producer.get("schema_version") == RESULT_SCHEMA
            and producer.get("status") == "complete"
            and producer.get("replica") == "producer"
            and producer.get("result_payload_sha256")
            == payload_sha256(producer, "result_payload_sha256"),
            "full-episode frozen producer differs",
        )
        frozen_episode = producer["episode"]
        producer_binding = {
            "path": str(frozen_producer_result_path),
            "file_sha256": _file_sha256(frozen_producer_result_path),
            "payload_sha256": producer["result_payload_sha256"],
            "source_commit": producer["source"]["commit"],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairing, episode = _run_episode(
        repo_root=repo_root, runtime=runtime, case=case, archived=archived,
        config=config, client=client, frozen_episode=frozen_episode,
        output_root=output_path.parent / "episode",
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "replica": replica,
        "source": source,
        "allocation": allocation,
        "config": config,
        "case_id": CASE_ID,
        "pairing": pairing,
        "pairing_sha256": hashlib.sha256(canonical(pairing)).hexdigest(),
        "policy_server": server_identity,
        "producer_binding": producer_binding,
        "episode": episode,
        "population_claim_authorized": False,
        "formal_safety_claim": False,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["scientific_view"] = scientific_view(result)
    if frozen_producer_result_path is not None:
        producer = _load(frozen_producer_result_path)
        _require(result["scientific_view"] == producer["scientific_view"],
                 "full-episode scientific replay differs")
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--replica", choices=("producer", "replay"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-producer-result", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        replica=args.replica, host=args.host, port=args.port,
        output_path=args.output.resolve(),
        frozen_producer_result_path=(
            None if args.frozen_producer_result is None
            else args.frozen_producer_result.resolve()
        ),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"], "replica": result["replica"],
        "terminal_reason": result["episode"]["terminal_reason"],
        "task_success": result["episode"]["native_task_success"],
        "collision_free_task_success": result["episode"][
            "collision_free_task_success"
        ],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
