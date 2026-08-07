#!/usr/bin/env python3
"""Replay the immutable primary AEGIS actions through the three-link shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence


CASE_ID = "vlsa-t1-goal-ii-t0-e05"
ARCHIVED_FILE_SHA256 = "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
ARCHIVED_PAYLOAD_SHA256 = "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c"


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON input must be one object")
    return value


def _git_identity(root: Path, expected_commit: str) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--short")
    _require(commit == expected_commit, "replay source commit differs")
    _require(not status, "replay source tree is dirty")
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
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
        summarize_shadow_records,
    )

    started = time.perf_counter_ns()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "archived file hash differs")
    _require(
        archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
        "archived payload identity differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary replay manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    config = load_shadow_config(config_path)
    _require(config["case_ids"] == [CASE_ID], "replay config case differs")
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
            "replay active obstacle differs",
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
                "replay pairing field differs: %s" % key,
            )

        perception = archived["perception"]
        shadow = MultilinkEllipsoidShadow.from_aegis_geometry(
            config,
            {
                "p2": perception["mvee_center"],
                "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        geometry = shadow.geometry_record(env)
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env,
            goal_atoms,
            step=-1,
            previous_values=None,
        )
        contact_authority = _contact_model_authority(env, obstacle_name)
        action_records: list[dict[str, Any]] = []
        shadow_records: list[dict[str, Any]] = []
        first_robot_contact_step: int | None = None
        direct_robot_contact_geoms: set[str] = set()
        previous_goal_values = initial_goal["values"]
        for index, archived_action in enumerate(archived["actions"]):
            _require(int(archived_action["step"]) == index, "archived action indexes differ")
            executed = np.asarray(archived_action["env_step_input"], dtype=np.float64)
            _require(
                executed.shape == (7,)
                and np.array_equal(executed, np.asarray(archived_action["executed"])),
                "archived executed action binding differs",
            )
            shadow_step = shadow.evaluate(env, executed, step=index)
            observation, reward, done, _ = env.step(executed)
            goal = _goal_progress_snapshot(
                env,
                goal_atoms,
                step=index,
                previous_values=previous_goal_values,
            )
            previous_goal_values = goal["values"]
            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=index,
                contact_authority=contact_authority,
            )
            _require(contacts["status"] == "available", "replay contact evidence is unavailable")
            robot_events = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            if robot_events and first_robot_contact_step is None:
                first_robot_contact_step = index
            for event in robot_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_robot_contact_geoms.add(name)
            displacement = float(
                np.sum(
                    np.abs(
                        np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
                        - initial_obstacle_position
                    )
                )
            )
            robot_distances = [float(event["distance"]) for event in robot_events]
            shadow_step["D_sim"] = {
                "available": True,
                "value": {
                    "minimum_robot_contact_distance_m": (
                        min(robot_distances) if robot_distances else None
                    ),
                    "active_obstacle_l1_displacement_m": displacement,
                    "robot_contact_count": len(robot_distances),
                },
                "semantics": config["simulator_verification"]["D_sim"],
                "source": "post_step_raw_simulator_exact_archived_action_replay",
            }
            _require(bool(done) is bool(archived_action["done"]), "replay done flag differs")
            _require(
                math.isclose(float(reward), float(archived_action["reward"]), rel_tol=0.0, abs_tol=1.0e-12),
                "replay reward differs",
            )
            _require(
                math.isclose(
                    displacement,
                    float(archived_action["obstacle_l1_displacement_m"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ),
                "replay obstacle displacement differs",
            )
            replay_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            archived_eef = np.asarray(
                archived_action["post_step_controller_proxy"]["eef_position"],
                dtype=np.float64,
            )
            _require(
                np.allclose(replay_eef, archived_eef, rtol=0.0, atol=1.0e-12),
                "replay end-effector state differs",
            )
            _require(
                bool(robot_events) is bool(archived_action["robot_obstacle_contact"]),
                "replay robot-contact flag differs",
            )
            shadow_records.append(shadow_step)
            action_records.append(
                {
                    "step": index,
                    "archived_env_step_input_sha256": _sha256(
                        _canonical(archived_action["env_step_input"])
                    ),
                    "done": bool(done),
                    "goal_progress": goal,
                    "eef_position_m": replay_eef.tolist(),
                    "robot_contact_events": robot_events,
                    "shadow": shadow_step,
                }
            )

        goal_summary = _goal_progress_summary(initial_goal, action_records)
        _require(len(action_records) == 237, "replay action horizon differs")
        _require(goal_summary["first_all_satisfied_step"] == 236, "replay task-success step differs")
        _require(first_robot_contact_step == 187, "replay first robot-contact step differs")
        _require("robot0_link5_collision" in direct_robot_contact_geoms, "replay lacks link-5 contact")
        _require("robot0_link6_collision" in direct_robot_contact_geoms, "replay lacks link-6 contact")
        shadow_summary = summarize_shadow_records(shadow_records)
        result: dict[str, Any] = {
            "schema_version": "vlsa_distal_three_ellipsoid_replay.v1",
            "status": "validated",
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
                "direct_robot_contact_geoms": sorted(direct_robot_contact_geoms),
                "native_task_success": True,
                "native_task_success_step": goal_summary[
                    "first_all_satisfied_step"
                ],
            },
            "shadow_summary": shadow_summary,
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
