#!/usr/bin/env python3
"""Execute the active three-link multi-CBF on immutable AEGIS actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence


CASE_ID = "vlsa-t1-goal-ii-t0-e05"
ARCHIVED_FILE_SHA256 = "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
ARCHIVED_PAYLOAD_SHA256 = "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c"
EXPECTED_ACTION_HORIZON = 237
PAPER_CAR_THRESHOLD_M = 0.001
PROTECTED_BODY_NAMES = {"robot0_link5", "robot0_link6", "robot0_link7"}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("JSON input is missing or symlinked")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON input must be one object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_identity(root: Path, expected_commit: str) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--short")
    _require(commit == expected_commit, "active replay source commit differs")
    _require(not status, "active replay source tree is dirty")
    return {
        "commit": commit,
        "dirty": False,
        "branch": run("branch", "--show-current"),
    }


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode(
        "utf-8"
    ) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _is_protected_event(event: Mapping[str, Any]) -> bool:
    other = event.get("other", {})
    return other.get("body_name") in PROTECTED_BODY_NAMES


def replay(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    config_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _runtime_imports,
        _settle,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.active import (
        DistalThreeEllipsoidMultiCbf,
        load_active_config,
        summarize_active_records,
    )
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
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list)
        and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "archived AEGIS action horizon differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary active-replay manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    config = load_active_config(config_path)
    _require(config["case_ids"] == [CASE_ID], "active multi-CBF config case differs")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    env = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime,
            case,
            render_resolution=TABLE_RENDER_RESOLUTION,
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(
            obstacle_name == archived["obstacle"]["active_name"],
            "active replay obstacle differs",
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
                "active replay pairing field differs: %s" % key,
            )

        perception = archived["perception"]
        controller = DistalThreeEllipsoidMultiCbf.from_aegis_geometry(
            config,
            {
                "p2": perception["mvee_center"],
                "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        geometry = controller.geometry_record(env)
        _require(geometry["link_ellipsoid_count"] == 3, "active geometry count differs")
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env,
            goal_atoms,
            step=-1,
            previous_values=None,
        )
        contact_authority = _contact_model_authority(env, obstacle_name)
        action_records: list[dict[str, Any]] = []
        filter_records: list[dict[str, Any]] = []
        direct_robot_contact_geoms: set[str] = set()
        direct_protected_contact_geoms: set[str] = set()
        first_robot_contact_step: int | None = None
        first_protected_contact_step: int | None = None
        first_car_step: int | None = None
        maximum_displacement = 0.0
        previous_goal_values = initial_goal["values"]
        failure: dict[str, Any] | None = None

        for index, archived_action in enumerate(archived_actions):
            _require(int(archived_action["step"]) == index, "archived action indexes differ")
            nominal = np.asarray(archived_action["env_step_input"], dtype=np.float64)
            _require(
                nominal.shape == (7,)
                and np.array_equal(nominal, np.asarray(archived_action["executed"])),
                "archived AEGIS env.step input binding differs",
            )
            executed, filter_step = controller.filter(env, nominal, step=index)
            filter_records.append(filter_step)
            if executed is None:
                failure = {
                    "component": "distal_three_ellipsoid_multicbf_qp",
                    "step": index,
                    "reason": filter_step["qp"]["reason"],
                }
                break
            executed_array = np.asarray(executed, dtype=np.float64)
            _require(executed_array.shape == (7,), "filtered action shape differs")
            _require(
                np.array_equal(executed_array[3:6], np.zeros(3)),
                "filtered action changed translational protocol rotation",
            )
            _require(
                float(executed_array[6]) == float(nominal[6]),
                "filtered action changed the gripper command",
            )
            observation, reward, done, _ = env.step(executed)
            goal = _goal_progress_snapshot(
                env,
                goal_atoms,
                step=index,
                previous_values=previous_goal_values,
            )
            previous_goal_values = goal["values"]
            _require(
                goal["all_satisfied"] is bool(done),
                "native goal vector differs from env.done",
            )
            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=index,
                contact_authority=contact_authority,
            )
            _require(contacts["status"] == "available", "active contact evidence unavailable")
            robot_events = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            protected_events = [event for event in robot_events if _is_protected_event(event)]
            if robot_events and first_robot_contact_step is None:
                first_robot_contact_step = index
            if protected_events and first_protected_contact_step is None:
                first_protected_contact_step = index
            for event in robot_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_robot_contact_geoms.add(name)
            for event in protected_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_protected_contact_geoms.add(name)
            displacement = float(
                np.sum(
                    np.abs(
                        np.asarray(
                            observation["%s_pos" % obstacle_name], dtype=np.float64
                        )
                        - initial_obstacle_position
                    )
                )
            )
            maximum_displacement = max(maximum_displacement, displacement)
            if first_car_step is None and displacement > PAPER_CAR_THRESHOLD_M:
                first_car_step = index
            protected_distances = [float(event["distance"]) for event in protected_events]
            filter_step["D_sim"] = {
                "available": True,
                "value": {
                    "minimum_protected_link_contact_distance_m": (
                        min(protected_distances) if protected_distances else None
                    ),
                    "active_obstacle_l1_displacement_m": displacement,
                    "protected_link_contact_count": len(protected_events),
                },
                "semantics": config["simulator_verification"]["D_sim"],
                "source": "post_step_raw_simulator_active_multicbf_replay",
            }
            action_records.append(
                {
                    "step": index,
                    "archived_aegis_action_sha256": _sha256(
                        _canonical(archived_action["env_step_input"])
                    ),
                    "nominal_archived_aegis_action": nominal.tolist(),
                    "executed_multicbf_action": executed,
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                    "eef_position_m": np.asarray(
                        observation["robot0_eef_pos"], dtype=np.float64
                    ).tolist(),
                    "active_obstacle_l1_displacement_m": displacement,
                    "robot_contact_events": robot_events,
                    "protected_link_contact_events": protected_events,
                    "multicbf": filter_step,
                }
            )
            if done:
                break

        goal_summary = _goal_progress_summary(initial_goal, action_records)
        active_summary = summarize_active_records(filter_records)
        native_success_step = goal_summary["first_all_satisfied_step"]
        native_success = native_success_step is not None
        paper_car_pass = first_car_step is None
        protected_contact_pass = first_protected_contact_step is None
        problem_solved = bool(
            failure is None
            and active_summary["all_qps_valid"]
            and protected_contact_pass
            and paper_car_pass
            and native_success
        )
        result: dict[str, Any] = {
            "schema_version": "vlsa_distal_three_ellipsoid_multicbf_replay.v1",
            "status": "complete" if failure is None else "method_failure",
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
                "executed_sequence_sha256": archived["action_invariance_ledger"][
                    "executed_sequence_sha256"
                ],
                "original_outcome": {
                    "first_link5_contact_step": 187,
                    "first_paper_car_step": 188,
                    "native_task_success_step": 236,
                },
            },
            "pairing": pairing,
            "geometry": geometry,
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {
                **goal_definition,
                "initial": initial_goal,
                "summary": goal_summary,
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
                "native_task_success_step": native_success_step,
            },
            "multicbf_summary": active_summary,
            "primary_problem_solved": problem_solved,
            "failure": failure,
            "replay_wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _sha256(_canonical(result))
        return result
    finally:
        if env is not None:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = replay(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
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
