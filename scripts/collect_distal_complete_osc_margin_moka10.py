#!/usr/bin/env python3
"""Recollect every grouped state/action pair with complete OSC inputs twice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.complete_osc_margin import (
    COLLECTION_RESULT_SCHEMA, DATASET_SCHEMA, align_named_complete_inputs,
    flatten_numeric_tree, jsonable, load_config, payload_sha256,
)
from main.multilink_ellipsoid.targeted_boundary_expansion import (
    DATASET_SCHEMA as SOURCE_DATASET_SCHEMA,
)
from main.multilink_ellipsoid.two_step_margin import feature_context, summarize_chunk
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action, _snapshot_env
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_native_geom_inventory_moka10 import _read_manifest, _split_counts
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


_GEOMETRY_PLACEHOLDER_CASE_ID = "vlsa-t1-goal-ii-t0-e05"


def _geometry_placeholder_row(rows: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return the established compatibility-only released-AEGIS MVEE source."""

    if _GEOMETRY_PLACEHOLDER_CASE_ID not in rows:
        raise ValueError("complete-OSC geometry placeholder case is missing")
    return rows[_GEOMETRY_PLACEHOLDER_CASE_ID]


def _transform_record(item: Any) -> dict[str, Any]:
    import numpy as np

    size = getattr(item, "semiaxes_m", getattr(item, "half_size_m", None))
    return {
        "body_name": getattr(item, "body_name", None),
        "geom_id": None if getattr(item, "geom_id", None) is None else int(item.geom_id),
        "geom_name": getattr(item, "geom_name", None),
        "center_m": np.asarray(item.center, dtype=np.float64).tolist(),
        "rotation_world_from_local": np.asarray(item.rotation, dtype=np.float64).tolist(),
        "semiaxes_or_half_size_m": np.asarray(size, dtype=np.float64).tolist(),
    }


def _numeric_transform(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "center_m": record["center_m"],
        "rotation_world_from_local": record["rotation_world_from_local"],
        "semiaxes_or_half_size_m": record["semiaxes_or_half_size_m"],
    }


def _state_input(env: Any, probe: Any, obstacle_name: str) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.oracle_affine import _obstacle_root_body_id
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector

    snapshot = jsonable(_snapshot_env(env))
    robot = env.robots[0]
    controller = robot.controller
    position_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    velocity_indexes = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
    obstacle_id = _obstacle_root_body_id(env.sim.model, obstacle_name)
    links = [_transform_record(item) for item in probe._ellipsoids(env)[:7]]
    obstacles = [_transform_record(item) for item in probe._obstacles(env)]
    current_orientation = np.asarray(
        getattr(controller, "ee_ori_mat"), dtype=np.float64
    )
    semantic = {
        "q_rad": np.asarray(env.sim.data.qpos[position_indexes], dtype=np.float64).tolist(),
        "qdot_rad_s": np.asarray(env.sim.data.qvel[velocity_indexes], dtype=np.float64).tolist(),
        "current_EE_position_m": np.asarray(controller.ee_pos, dtype=np.float64).tolist(),
        "current_EE_orientation_world_from_EE": current_orientation.tolist(),
        "goal_EE_position_m": np.asarray(controller.goal_pos, dtype=np.float64).tolist(),
        "goal_EE_orientation_world_from_EE": np.asarray(
            controller.goal_ori, dtype=np.float64
        ).tolist(),
        "obstacle_root_body_name": str(obstacle_name),
        "obstacle_root_position_m": np.asarray(
            env.sim.data.xpos[obstacle_id], dtype=np.float64
        ).tolist(),
        "obstacle_root_orientation_world_from_body": np.asarray(
            env.sim.data.xmat[obstacle_id], dtype=np.float64
        ).reshape(3, 3).tolist(),
        "seven_robot_ellipsoid_transforms": links,
        "exact_obstacle_primitive_transforms": obstacles,
    }
    numeric_semantic = {
        key: value for key, value in semantic.items()
        if key not in {
            "obstacle_root_body_name", "seven_robot_ellipsoid_transforms",
            "exact_obstacle_primitive_transforms",
        }
    }
    numeric_semantic["seven_robot_ellipsoid_transforms"] = [
        _numeric_transform(item) for item in links
    ]
    numeric_semantic["exact_obstacle_primitive_transforms"] = [
        _numeric_transform(item) for item in obstacles
    ]
    names, vector = flatten_numeric_tree({
        "complete_snapshot": snapshot,
        "semantic_state": numeric_semantic,
    })
    dynamic = _dynamic_state_vector(env)
    return {
        "complete_snapshot": snapshot,
        "complete_snapshot_sha256": hashlib.sha256(json.dumps(
            snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest(),
        "dynamic_state_sha256": hashlib.sha256(dynamic.tobytes()).hexdigest(),
        "dynamic_state_dimension": int(dynamic.size),
        "semantic_state": semantic,
        "complete_input_feature_names": names,
        "complete_input_vector": vector.tolist(),
    }


def _contact_receipt(chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipt = []
    for action_index, transition in enumerate(chunk["transitions"]):
        for event in transition["raw_protected_contact_events"]:
            receipt.append({
                "action_index": int(action_index),
                "substep_index": int(event["substep_index"]),
                "contact_index": int(event["contact_index"]),
                "distance_m": float(event["distance_m"]),
                "position_m": [float(value) for value in event["position_m"]],
                "protected_body_name": str(event["protected_body_name"]),
                "protected_geom_id": int(event["protected_geom_id"]),
                "protected_geom_name": str(event["protected_geom_name"]),
                "obstacle_geom_id": int(event["obstacle_geom_id"]),
                "obstacle_geom_name": str(event["obstacle_geom_name"]),
            })
    return receipt


def _rollout_receipt(probe: Any, env: Any, actions: Any) -> dict[str, Any]:
    import numpy as np

    commands = np.asarray(actions, dtype=np.float64)
    if commands.shape != (2, 7) or not np.all(np.isfinite(commands)):
        raise ValueError("complete-OSC action chunk differs")
    # ``rollout_chunk`` predates ndarray callers and intentionally checks the
    # Python sequence's truth value. Keep that compatibility at this adapter.
    chunk = probe.rollout_chunk(env, commands.tolist())
    summary = summarize_chunk(chunk)
    contacts = _contact_receipt(chunk)
    return {
        "margin": summary["minimum_substep_clearance_m"],
        "witnesses": summary["minimum_substep_witnesses"],
        "raw_contacts": contacts,
        "raw_contact_count": len(contacts),
        "raw_safe": bool(summary["D_sim_raw_safe"]),
        "maximum_obstacle_motion_m": float(
            summary["maximum_within_step_obstacle_l1_displacement_m"]
        ),
        "next_state_sha256_per_action": [
            str(item["next_state_sha256"]) for item in chunk["transitions"]
        ],
        "initial_synchronization": chunk["initial_synchronization"],
    }


def main() -> int:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--same-task-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "population": args.population_manifest, "selected": args.selected_manifest,
        "same_task": args.same_task_manifest, "targeted": args.targeted_manifest,
        "source_dataset": args.expanded_dataset, "archived": args.archived_root,
        "geometry": args.geometry_config, "exact_box": args.exact_box_config,
        "dataset": args.dataset, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, key, label in (
        (paths["population"], "source_population_manifest_sha256", "population"),
        (paths["selected"], "selected_manifest_file_sha256", "selected"),
        (paths["same_task"], "same_task_manifest_file_sha256", "same-task"),
        (paths["targeted"], "targeted_manifest_file_sha256", "targeted"),
        (paths["source_dataset"], "expanded_dataset_file_sha256", "expanded-dataset"),
        (paths["geometry"], "geometry_config_file_sha256", "geometry"),
        (paths["exact_box"], "exact_box_config_file_sha256", "exact-box"),
    ):
        _require(_file_sha256(path) == source[key], "complete-OSC %s differs" % label)
    source_dataset = _load(paths["source_dataset"])
    _require(
        source_dataset.get("schema_version") == SOURCE_DATASET_SCHEMA
        and source_dataset.get("dataset_payload_sha256")
        == source["expanded_dataset_payload_sha256"]
        == payload_sha256(source_dataset, "dataset_payload_sha256"),
        "complete-OSC expanded dataset identity differs",
    )
    source_states = source_dataset["state_records"]
    _require(
        len(source_states) == int(source["expected_state_count"])
        and _split_counts(source_states) == config["split"]["expected_state_counts"],
        "complete-OSC state population differs",
    )
    selected = []
    for path in (paths["selected"], paths["same_task"], paths["targeted"]):
        selected.extend(_read_manifest(path, _file_sha256(path)))
    row_by_case = {str(item["case_id"]): item for item in selected}
    _require(
        len(selected) == len(row_by_case) == int(source["expected_episode_count"])
        and {str(item["case_id"]) for item in source_states} == set(row_by_case),
        "complete-OSC episode population differs",
    )
    placeholder_row = _geometry_placeholder_row(row_by_case)
    placeholder_path = paths["archived"] / placeholder_row["archived_relative_path"]
    geometry_placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == placeholder_row["archived_file_sha256"]
        and geometry_placeholder.get("result_payload_sha256")
        == placeholder_row["archived_payload_sha256"]
        and geometry_placeholder.get("case_id") == _GEOMETRY_PLACEHOLDER_CASE_ID,
        "complete-OSC geometry placeholder differs",
    )
    test_cases = sorted({
        str(item["case_id"]) for item in source_states if item["split"] == "test"
    })
    _require(test_cases == config["split"]["test_case_ids"], "complete-OSC test split differs")
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    state_records = []
    episode_results = []
    pairings = {}
    maximum_margin_difference = 0.0
    nondeterministic_pairs = []
    total_raw_contacts = 0
    for case_id in sorted(row_by_case):
        selected_row = row_by_case[case_id]
        archived_path = paths["archived"] / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256") == selected_row["archived_payload_sha256"]
            and archived.get("case_id") == case_id,
            "complete-OSC archive differs: %s" % case_id,
        )
        case = population[case_id]
        validate_case_row(case, paths["repo"])
        env = probe_env = None
        try:
            env, probe_env, task, observation, setup = _build_pair(runtime, case)
            pairing = pairing_record(
                case=case, selected_initial_state=setup["selected_initial_state"],
                settled_observation=observation, task_description=str(task.language),
                active_obstacle_name=setup["obstacle_name"],
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
                _require(pairing[key] == archived["pairing"][key], "complete-OSC pairing differs: %s:%s" % (case_id, key))
            pairings[case_id] = pairing
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                # The released AEGIS obstacle MVEE is required only to build
                # the compatibility shadow object. The seven learned rows use
                # the live exact-box union below, so freeze the same valid E05
                # placeholder used by the established grouped collectors.
                archived=geometry_placeholder, env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            states = sorted(
                (item for item in source_states if item["case_id"] == case_id),
                key=lambda item: int(item["state_step"]),
            )
            _require(len(states) == 5, "complete-OSC states per episode differ")
            by_step = {int(item["state_step"]): item for item in states}
            maximum_step = max(by_step)
            actions = archived["actions"]
            case_nondeterministic = 0
            for step in range(maximum_step + 1):
                if step in by_step:
                    source_state = by_step[step]
                    first = _canonical_action(actions[step], step)
                    second = _canonical_action(actions[step + 1], step + 1)
                    _require(
                        np.array_equal(first, np.asarray(source_state["nominal_first_action"], dtype=np.float64))
                        and np.array_equal(second, np.asarray(source_state["nominal_second_action"], dtype=np.float64)),
                        "complete-OSC nominal action differs: %s:%d" % (case_id, step),
                    )
                    state_input = _state_input(env, probe, setup["obstacle_name"])
                    names = state_input.pop("complete_input_feature_names")
                    values = state_input.pop("complete_input_vector")
                    _require(
                        len(names) == len(values) and len(names) == len(set(names)),
                        "complete-OSC per-state named feature receipt differs",
                    )
                    state_input["complete_input_named_values"] = {
                        name: float(value) for name, value in zip(names, values)
                    }
                    context = feature_context(env, probe)
                    current = np.asarray(context["current_clearance_m"], dtype=np.float64)
                    candidates = []
                    xyz_values = np.asarray(source_state["candidate_first_xyz"], dtype=np.float64)
                    _require(xyz_values.shape == (125, 3), "complete-OSC candidate grid differs")
                    for candidate_index, xyz in enumerate(xyz_values):
                        candidate_first = first.copy()
                        candidate_first[:3] = xyz
                        action_pair = np.asarray([candidate_first, second], dtype=np.float64)
                        first_receipt = _rollout_receipt(probe, env, action_pair)
                        second_receipt = _rollout_receipt(probe, env, action_pair)
                        first_margin = np.asarray(first_receipt["margin"], dtype=np.float64)
                        second_margin = np.asarray(second_receipt["margin"], dtype=np.float64)
                        difference = float(np.max(np.abs(first_margin - second_margin)))
                        maximum_margin_difference = max(maximum_margin_difference, difference)
                        margin_equal = bool(np.array_equal(first_margin, second_margin))
                        contact_equal = bool(
                            json.dumps(first_receipt["raw_contacts"], sort_keys=True, separators=(",", ":"))
                            == json.dumps(second_receipt["raw_contacts"], sort_keys=True, separators=(",", ":"))
                        )
                        next_equal = bool(
                            first_receipt["next_state_sha256_per_action"]
                            == second_receipt["next_state_sha256_per_action"]
                        )
                        deterministic = bool(margin_equal and contact_equal and next_equal)
                        if not deterministic:
                            case_nondeterministic += 1
                            nondeterministic_pairs.append({
                                "state_index": int(source_state["state_index"]),
                                "candidate_index": int(candidate_index),
                                "margin_bitwise_equal": margin_equal,
                                "contact_receipt_equal": contact_equal,
                                "next_state_hashes_equal": next_equal,
                                "maximum_margin_difference_m": difference,
                            })
                        total_raw_contacts += int(first_receipt["raw_contact_count"])
                        candidates.append({
                            "candidate_index": int(candidate_index),
                            "full_two_action_commands": action_pair.tolist(),
                            "action_pair_sha256": hashlib.sha256(action_pair.tobytes()).hexdigest(),
                            "rollout_minimum_ellipsoid_margin_m": first_margin.tolist(),
                            "minimum_margin_witnesses": first_receipt["witnesses"],
                            "raw_mujoco_protected_contacts": first_receipt["raw_contacts"],
                            "raw_mujoco_protected_contact_count": int(first_receipt["raw_contact_count"]),
                            "raw_mujoco_protected_contact_free": bool(
                                first_receipt["raw_contact_count"] == 0
                            ),
                            "D_sim_raw_contact_and_static_obstacle_safe": bool(
                                first_receipt["raw_safe"]
                            ),
                            "maximum_obstacle_motion_m": float(first_receipt["maximum_obstacle_motion_m"]),
                            "next_state_sha256_per_action": first_receipt["next_state_sha256_per_action"],
                            "replay_deterministic": deterministic,
                            "repeat_receipt": {
                                "margin_bitwise_equal": margin_equal,
                                "contact_receipt_equal": contact_equal,
                                "next_state_hashes_equal": next_equal,
                                "maximum_margin_difference_m": difference,
                            },
                        })
                    record = {
                        "state_index": int(source_state["state_index"]),
                        "case_id": case_id, "split": str(source_state["split"]),
                        "task_level_group_id": str(source_state["task_level_group_id"]),
                        "state_step": int(step),
                        **state_input,
                        "current_ellipsoid_margin_m": current.tolist(),
                        "nominal_first_action": first.tolist(),
                        "nominal_second_action": second.tolist(),
                        "candidate_count": len(candidates), "candidates": candidates,
                    }
                    state_records.append(record)
                    print(json.dumps({
                        "case_id": case_id, "state_step": step,
                        "state_index": record["state_index"],
                        "pair_count_complete": len(state_records) * 125,
                        "nondeterministic_pair_count": len(nondeterministic_pairs),
                    }, sort_keys=True), flush=True)
                if step < maximum_step:
                    env.step(_canonical_action(actions[step], step).tolist())
            episode_results.append({
                "case_id": case_id, "split": str(states[0]["split"]),
                "state_count": len(states),
                "nondeterministic_pair_count": case_nondeterministic,
            })
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    state_records.sort(key=lambda item: int(item["state_index"]))
    aligned_names, aligned_vectors = align_named_complete_inputs([
        item["complete_input_named_values"] for item in state_records
    ])
    for record, vector in zip(state_records, aligned_vectors):
        record.pop("complete_input_named_values")
        record["complete_input_vector"] = vector.tolist()
    pair_count = sum(int(item["candidate_count"]) for item in state_records)
    split_counts = _split_counts(state_records)
    input_sufficiency_pass = bool(
        len(state_records) == int(source["expected_state_count"])
        and pair_count == int(source["expected_pair_count"])
        and split_counts == config["split"]["expected_state_counts"]
        and not nondeterministic_pairs
        and all(len(item["complete_input_vector"]) == len(aligned_names) for item in state_records)
    )
    dataset = {
        "schema_version": DATASET_SCHEMA,
        "source_commit": args.expected_commit,
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "source_expanded_dataset_file_sha256": _file_sha256(paths["source_dataset"]),
        "complete_input_feature_names": aligned_names,
        "complete_input_dimension": len(aligned_names),
        "state_records": state_records,
        "summary": {
            "episode_count": len(episode_results), "state_count": len(state_records),
            "state_split_counts": split_counts, "pair_count": pair_count,
            "replay_count": pair_count * 2,
            "nondeterministic_pair_count": len(nondeterministic_pairs),
            "maximum_repeat_margin_absolute_difference_m": maximum_margin_difference,
            "raw_mujoco_protected_contact_observation_count": total_raw_contacts,
            "input_sufficiency_pass": input_sufficiency_pass,
        },
    }
    dataset["dataset_payload_sha256"] = payload_sha256(dataset, "dataset_payload_sha256")
    _atomic_write(paths["dataset"], dataset)
    result = {
        "schema_version": COLLECTION_RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "archived_table1_root": {"path": str(paths["archived"]), "read_only": True},
        "pairings": pairings, "episode_results": episode_results,
        "dataset": {
            "path": str(paths["dataset"]),
            "file_sha256": _file_sha256(paths["dataset"]),
            "payload_sha256": dataset["dataset_payload_sha256"],
        },
        "input_sufficiency": {
            **dataset["summary"],
            "nondeterministic_pairs": nondeterministic_pairs,
        },
        "decision": {
            "input_sufficiency_pass": input_sufficiency_pass,
            "MLP_training_authorized": input_sufficiency_pass,
            "stop_reason": None if input_sufficiency_pass else "identical_snapshot_action_replay_not_deterministic",
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(result, "result_payload_sha256")
    _atomic_write(paths["output"], result)
    print(json.dumps({"decision": result["decision"], "dataset": result["dataset"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
