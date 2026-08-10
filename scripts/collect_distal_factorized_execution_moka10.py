#!/usr/bin/env python3
"""Collect complete q/substep traces for the factorized OSC pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.complete_osc_margin import (
    COLLECTION_RESULT_SCHEMA as COMPLETE_COLLECTION_SCHEMA,
    DATASET_SCHEMA as COMPLETE_DATASET_SCHEMA,
    payload_sha256 as complete_payload_sha256,
)
from main.multilink_ellipsoid.factorized_execution_pilot import (
    COLLECTION_SCHEMA, DATASET_SCHEMA, candidate_chunks, load_config,
    payload_sha256, trace_arrays,
)
from scripts.collect_distal_complete_osc_margin_moka10 import (
    _GEOMETRY_PLACEHOLDER_CASE_ID, _contact_receipt, _geometry_placeholder_row,
)
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_native_geom_inventory_moka10 import _read_manifest
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _source_code(source: str) -> int:
    if source == "nominal":
        return 0
    if source.startswith("finite_difference"):
        return 1
    if source == "random_antithetic":
        return 2
    raise ValueError("factorized-execution candidate source differs")


def _write_npz_atomic(path: Path, arrays: dict[str, object]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(path)


def _split_code(name: str) -> int:
    return {"train": 0, "validation": 1, "test": 2}[str(name)]


def main() -> int:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--complete-dataset", type=Path, required=True)
    parser.add_argument("--complete-collection", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--same-task-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--array-dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "complete_dataset": args.complete_dataset,
        "complete_collection": args.complete_collection,
        "population": args.population_manifest,
        "selected": args.selected_manifest, "same_task": args.same_task_manifest,
        "targeted": args.targeted_manifest, "archived": args.archived_root,
        "geometry": args.geometry_config, "exact_box": args.exact_box_config,
        "array_dataset": args.array_dataset, "metadata": args.metadata,
        "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    complete_dataset = _load(paths["complete_dataset"])
    complete_collection = _load(paths["complete_collection"])
    _require(
        _file_sha256(paths["complete_dataset"])
        == source["complete_dataset_file_sha256"]
        and complete_dataset.get("schema_version") == COMPLETE_DATASET_SCHEMA
        and complete_dataset.get("dataset_payload_sha256")
        == source["complete_dataset_payload_sha256"]
        == complete_payload_sha256(complete_dataset, "dataset_payload_sha256")
        and _file_sha256(paths["complete_collection"])
        == source["complete_collection_file_sha256"]
        and complete_collection.get("schema_version") == COMPLETE_COLLECTION_SCHEMA
        and complete_collection.get("result_payload_sha256")
        == source["complete_collection_payload_sha256"]
        == complete_payload_sha256(complete_collection, "result_payload_sha256"),
        "factorized-execution immutable complete dataset differs",
    )
    complete_source = complete_collection["config"]["immutable_source"]
    for path, key, label in (
        (paths["population"], "source_population_manifest_sha256", "population"),
        (paths["selected"], "selected_manifest_file_sha256", "selected"),
        (paths["same_task"], "same_task_manifest_file_sha256", "same-task"),
        (paths["targeted"], "targeted_manifest_file_sha256", "targeted"),
        (paths["geometry"], "geometry_config_file_sha256", "geometry"),
        (paths["exact_box"], "exact_box_config_file_sha256", "exact-box"),
    ):
        _require(_file_sha256(path) == complete_source[key],
                 "factorized-execution %s source differs" % label)
    state_records = sorted(
        complete_dataset["state_records"], key=lambda item: int(item["state_index"])
    )
    _require(
        len(state_records) == int(source["expected_state_count"])
        and int(complete_dataset["complete_input_dimension"])
        == int(source["complete_state_input_dimension"]),
        "factorized-execution state population differs",
    )
    selected_rows = []
    for path in (paths["selected"], paths["same_task"], paths["targeted"]):
        selected_rows.extend(_read_manifest(path, _file_sha256(path)))
    row_by_case = {str(item["case_id"]): item for item in selected_rows}
    _require(
        len(row_by_case) == int(source["expected_episode_count"])
        and {str(item["case_id"]) for item in state_records} == set(row_by_case),
        "factorized-execution episode population differs",
    )
    placeholder_row = _geometry_placeholder_row(row_by_case)
    placeholder_path = paths["archived"] / placeholder_row["archived_relative_path"]
    geometry_placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == placeholder_row["archived_file_sha256"]
        and geometry_placeholder.get("result_payload_sha256")
        == placeholder_row["archived_payload_sha256"]
        and geometry_placeholder.get("case_id") == _GEOMETRY_PLACEHOLDER_CASE_ID,
        "factorized-execution compatibility geometry differs",
    )
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    arrays: dict[str, list[object]] = {
        "state_input_vector": [], "action_chunk": [], "joint_position_rad": [],
        "ellipsoid_clearance_m": [], "state_index": [], "candidate_index": [],
        "split_code": [], "source_code": [], "raw_contact_count": [],
        "maximum_obstacle_motion_m": [], "next_state_sha256_per_action": [],
    }
    state_metadata = []
    repeat_mismatches = []
    nominal_pairing_mismatches = []
    pairings = {}
    raw_contact_total = 0
    maximum_repeat_q_difference = 0.0
    maximum_repeat_margin_difference = 0.0
    for case_id in sorted(row_by_case):
        selected_row = row_by_case[case_id]
        archived_path = paths["archived"] / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256")
            == selected_row["archived_payload_sha256"],
            "factorized-execution archived episode differs: %s" % case_id,
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
                _require(
                    pairing[key] == complete_collection["pairings"][case_id][key],
                    "factorized-execution pairing differs: %s:%s" % (case_id, key),
                )
            pairings[case_id] = pairing
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                archived=geometry_placeholder, env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            states = sorted(
                (item for item in state_records if item["case_id"] == case_id),
                key=lambda item: int(item["state_step"]),
            )
            by_step = {int(item["state_step"]): item for item in states}
            maximum_step = max(by_step)
            actions = archived["actions"]
            for step in range(maximum_step + 1):
                if step in by_step:
                    state = by_step[step]
                    dynamic_sha256 = hashlib.sha256(
                        _dynamic_state_vector(env).tobytes()
                    ).hexdigest()
                    _require(
                        dynamic_sha256 == state["dynamic_state_sha256"],
                        "factorized-execution complete state differs",
                    )
                    first = _canonical_action(actions[step], step)
                    second = _canonical_action(actions[step + 1], step + 1)
                    _require(
                        np.array_equal(first, np.asarray(state["nominal_first_action"], dtype=np.float64))
                        and np.array_equal(second, np.asarray(state["nominal_second_action"], dtype=np.float64)),
                        "factorized-execution nominal action differs",
                    )
                    candidates = candidate_chunks(
                        np.asarray([first, second], dtype=np.float64),
                        int(state["state_index"]), config,
                    )
                    metadata_candidates = []
                    for candidate in candidates:
                        candidate_index = int(candidate["candidate_index"])
                        action_chunk = np.asarray(
                            candidate["full_two_action_commands"], dtype=np.float64
                        )
                        chunk = probe.rollout_chunk(env, action_chunk.tolist())
                        receipt = trace_arrays(chunk, config)
                        contacts = _contact_receipt(chunk)
                        _require(
                            len(contacts) == int(receipt["raw_protected_contact_count"]),
                            "factorized-execution contact receipt differs",
                        )
                        if candidate_index == 0:
                            old_candidates = state["candidates"]
                            matches = [item for item in old_candidates if np.array_equal(
                                np.asarray(item["full_two_action_commands"], dtype=np.float64),
                                action_chunk,
                            )]
                            # The historical clipped XYZ grid does not always
                            # contain the exact nominal point.  When it does,
                            # require byte-identical labels; otherwise the
                            # complete dynamic-state hash plus registered
                            # duplicate rollout is the pairing authority.
                            if len(matches) == 1:
                                old = matches[0]
                                if (
                                    not np.array_equal(
                                        np.asarray(old["rollout_minimum_ellipsoid_margin_m"], dtype=np.float64),
                                        receipt["minimum_ellipsoid_margin_m"],
                                    )
                                    or old["next_state_sha256_per_action"]
                                    != receipt["next_state_sha256_per_action"]
                                    or int(old["raw_mujoco_protected_contact_count"])
                                    != int(receipt["raw_protected_contact_count"])
                                ):
                                    nominal_pairing_mismatches.append({
                                        "state_index": int(state["state_index"]),
                                        "reason": "immutable_nominal_rollout_differs",
                                    })
                            elif len(matches) > 1:
                                nominal_pairing_mismatches.append({
                                    "state_index": int(state["state_index"]),
                                    "reason": "immutable_nominal_candidate_duplicated",
                                })
                        if candidate_index in config["candidate_design"]["repeat_candidate_indexes"]:
                            repeated_chunk = probe.rollout_chunk(env, action_chunk.tolist())
                            repeated = trace_arrays(repeated_chunk, config)
                            q_difference = float(np.max(np.abs(
                                receipt["joint_position_rad"]
                                - repeated["joint_position_rad"]
                            )))
                            margin_difference = float(np.max(np.abs(
                                receipt["ellipsoid_clearance_m"]
                                - repeated["ellipsoid_clearance_m"]
                            )))
                            maximum_repeat_q_difference = max(
                                maximum_repeat_q_difference, q_difference
                            )
                            maximum_repeat_margin_difference = max(
                                maximum_repeat_margin_difference, margin_difference
                            )
                            repeated_contacts = _contact_receipt(repeated_chunk)
                            if not (
                                np.array_equal(
                                    receipt["joint_position_rad"],
                                    repeated["joint_position_rad"],
                                )
                                and np.array_equal(
                                    receipt["ellipsoid_clearance_m"],
                                    repeated["ellipsoid_clearance_m"],
                                )
                                and contacts == repeated_contacts
                                and receipt["next_state_sha256_per_action"]
                                == repeated["next_state_sha256_per_action"]
                            ):
                                repeat_mismatches.append({
                                    "state_index": int(state["state_index"]),
                                    "candidate_index": candidate_index,
                                    "maximum_q_difference_rad": q_difference,
                                    "maximum_margin_difference_m": margin_difference,
                                })
                        arrays["state_input_vector"].append(
                            np.asarray(state["complete_input_vector"], dtype=np.float64)
                        )
                        arrays["action_chunk"].append(action_chunk)
                        arrays["joint_position_rad"].append(receipt["joint_position_rad"])
                        arrays["ellipsoid_clearance_m"].append(receipt["ellipsoid_clearance_m"])
                        arrays["state_index"].append(int(state["state_index"]))
                        arrays["candidate_index"].append(candidate_index)
                        arrays["split_code"].append(_split_code(state["split"]))
                        arrays["source_code"].append(_source_code(str(candidate["source"])))
                        arrays["raw_contact_count"].append(
                            int(receipt["raw_protected_contact_count"])
                        )
                        arrays["maximum_obstacle_motion_m"].append(
                            float(receipt["maximum_obstacle_motion_m"])
                        )
                        arrays["next_state_sha256_per_action"].append(
                            receipt["next_state_sha256_per_action"]
                        )
                        raw_contact_total += int(receipt["raw_protected_contact_count"])
                        metadata_candidates.append({
                            key: candidate[key] for key in candidate
                            if key != "full_two_action_commands"
                        } | {
                            "action_chunk_sha256": hashlib.sha256(
                                action_chunk.tobytes()
                            ).hexdigest(),
                            "joint_trace_sha256": hashlib.sha256(
                                receipt["joint_position_rad"].tobytes()
                            ).hexdigest(),
                            "clearance_trace_sha256": hashlib.sha256(
                                receipt["ellipsoid_clearance_m"].tobytes()
                            ).hexdigest(),
                        })
                    state_metadata.append({
                        "state_index": int(state["state_index"]),
                        "case_id": case_id, "state_step": int(step),
                        "split": str(state["split"]),
                        "candidate_count": len(metadata_candidates),
                        "candidates": metadata_candidates,
                    })
                    print(json.dumps({
                        "case_id": case_id, "state_step": step,
                        "state_index": int(state["state_index"]),
                        "completed_rollout_count": len(arrays["state_index"]),
                        "repeat_mismatch_count": len(repeat_mismatches),
                    }, sort_keys=True), flush=True)
                if step < maximum_step:
                    env.step(_canonical_action(actions[step], step).tolist())
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    numpy_arrays = {
        "state_input_vector": np.asarray(arrays["state_input_vector"], dtype=np.float64),
        "action_chunk": np.asarray(arrays["action_chunk"], dtype=np.float64),
        "joint_position_rad": np.asarray(arrays["joint_position_rad"], dtype=np.float64),
        "ellipsoid_clearance_m": np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64),
        "state_index": np.asarray(arrays["state_index"], dtype=np.int64),
        "candidate_index": np.asarray(arrays["candidate_index"], dtype=np.int64),
        "split_code": np.asarray(arrays["split_code"], dtype=np.int8),
        "source_code": np.asarray(arrays["source_code"], dtype=np.int8),
        "raw_contact_count": np.asarray(arrays["raw_contact_count"], dtype=np.int64),
        "maximum_obstacle_motion_m": np.asarray(
            arrays["maximum_obstacle_motion_m"], dtype=np.float64
        ),
        "next_state_sha256_per_action": np.asarray(
            arrays["next_state_sha256_per_action"], dtype="U64"
        ),
    }
    expected_rollouts = int(config["candidate_design"]["expected_rollout_count"])
    collection_pass = bool(
        len(state_metadata) == int(source["expected_state_count"])
        and len(numpy_arrays["state_index"]) == expected_rollouts
        and not repeat_mismatches and not nominal_pairing_mismatches
    )
    _write_npz_atomic(paths["array_dataset"], numpy_arrays)
    metadata = {
        "schema_version": DATASET_SCHEMA,
        "source_commit": args.expected_commit,
        "config_file_sha256": config["config_file_sha256"],
        "complete_dataset_file_sha256": _file_sha256(paths["complete_dataset"]),
        "array_dataset": {
            "path": str(paths["array_dataset"]),
            "file_sha256": _file_sha256(paths["array_dataset"]),
        },
        "state_records": state_metadata,
        "summary": {
            "episode_count": len(row_by_case), "state_count": len(state_metadata),
            "rollout_count": len(numpy_arrays["state_index"]),
            "repeat_rollout_count": int(source["expected_state_count"])
            * len(config["candidate_design"]["repeat_candidate_indexes"]),
            "joint_trace_state_count": int(
                config["rollout_target"]["joint_trace_state_count"]
            ),
            "complete_state_input_dimension": int(
                source["complete_state_input_dimension"]
            ),
            "raw_contact_observation_count": int(raw_contact_total),
            "repeat_mismatch_count": len(repeat_mismatches),
            "nominal_pairing_mismatch_count": len(nominal_pairing_mismatches),
            "collection_gate_pass": collection_pass,
        },
    }
    metadata["dataset_payload_sha256"] = payload_sha256(
        metadata, "dataset_payload_sha256"
    )
    _atomic_write(paths["metadata"], metadata)
    result = {
        "schema_version": COLLECTION_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "pairings": pairings,
        "complete_dataset": {
            "path": str(paths["complete_dataset"]),
            "file_sha256": _file_sha256(paths["complete_dataset"]),
            "payload_sha256": complete_dataset["dataset_payload_sha256"],
        },
        "trajectory_dataset": {
            "array_path": str(paths["array_dataset"]),
            "array_file_sha256": _file_sha256(paths["array_dataset"]),
            "metadata_path": str(paths["metadata"]),
            "metadata_file_sha256": _file_sha256(paths["metadata"]),
            "metadata_payload_sha256": metadata["dataset_payload_sha256"],
        },
        "determinism": {
            "repeat_mismatches": repeat_mismatches,
            "maximum_repeat_joint_difference_rad": maximum_repeat_q_difference,
            "maximum_repeat_margin_difference_m": maximum_repeat_margin_difference,
        },
        "pairing": {"nominal_mismatches": nominal_pairing_mismatches},
        "decision": {
            "collection_gate_pass": collection_pass,
            "paired_model_training_authorized": collection_pass,
            "stop_reason": None if collection_pass else
            "trajectory_determinism_or_immutable_pairing_failed",
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "decision": result["decision"], "trajectory_dataset": result["trajectory_dataset"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
