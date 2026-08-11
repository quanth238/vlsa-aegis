#!/usr/bin/env python3
"""Collect the preregistered reserved direct-horizon OSC population."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    COLLECTION_SCHEMA, DATASET_SCHEMA, load_direct_horizon_config,
)
from main.multilink_ellipsoid.factorized_execution_pilot import (
    candidate_chunks, load_config as load_factorized_config, payload_sha256,
    trace_arrays,
)
from main.multilink_ellipsoid.two_step_margin import feature_context
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.collect_distal_complete_osc_margin_moka10 import (
    _GEOMETRY_PLACEHOLDER_CASE_ID, _contact_receipt, _geometry_placeholder_row,
    _state_input,
)
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
    raise ValueError("reserved direct-horizon candidate source differs")


def _write_npz_atomic(path: Path, arrays: Mapping[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(path)


def _aligned_state_vector(
    state_input: Mapping[str, Any], aligned_names: Sequence[str],
) -> Any:
    import numpy as np

    raw_names = [str(value) for value in state_input["complete_input_feature_names"]]
    raw_values = np.asarray(
        state_input["complete_input_vector"], dtype=np.float64
    )
    if len(raw_names) != len(raw_values) or len(set(raw_names)) != len(raw_names):
        raise ValueError("reserved direct-horizon raw state schema differs")
    values = {name: float(value) for name, value in zip(raw_names, raw_values)}
    aligned = []
    for name in aligned_names:
        if name.endswith(".value"):
            base = name[:-6]
            aligned.append(values.get(base, 0.0))
        elif name.endswith(".present"):
            base = name[:-8]
            aligned.append(float(base in values))
        else:
            raise ValueError("reserved direct-horizon aligned feature differs")
    output = np.asarray(aligned, dtype=np.float64)
    if output.shape != (len(aligned_names),) or not np.all(np.isfinite(output)):
        raise ValueError("reserved direct-horizon aligned vector differs")
    return output


def select_reserved_steps(
    scan: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    settings = config["reserved_state_selection"]
    count = int(settings["selected_state_count_per_episode"])
    spacing = int(settings["minimum_step_spacing"])
    eligible = [
        dict(item) for item in scan
        if int(item["state_step"]) >= int(settings["scan_start_step"])
        and bool(item["two_action_horizon_available"])
        and (
            not bool(settings["require_all_seven_current_clearances_nonnegative"])
            or float(item["minimum_current_clearance_m"]) >= 0.0
        )
    ]
    eligible.sort(key=lambda item: (
        float(item["minimum_current_clearance_m"]), int(item["state_step"])
    ))
    chosen: list[dict[str, Any]] = []
    for item in eligible:
        step = int(item["state_step"])
        if all(abs(step - int(other["state_step"])) >= spacing for other in chosen):
            chosen.append(item)
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise RuntimeError("reserved episode lacks five spaced nonnegative states")
    for rank, item in enumerate(chosen):
        item["selection_rank"] = int(rank)
    return sorted(chosen, key=lambda item: int(item["state_step"]))


def _scan_episode(
    *, runtime: Mapping[str, Any], case: Mapping[str, Any], archived: Mapping[str, Any],
    geometry_config: Any, exact_box_config: Any, geometry_placeholder: Mapping[str, Any],
    config: Mapping[str, Any], expected_obstacle_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    from main.evaluate_safelibero_aegis import pairing_record
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe

    env = probe_env = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        _require(
            setup["obstacle_name"] == expected_obstacle_name,
            "reserved direct-horizon active obstacle differs",
        )
        pairing = pairing_record(
            case=case, selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation, task_description=str(task.language),
            active_obstacle_name=setup["obstacle_name"],
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64
            ),
        )
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
        actions = archived["actions"]
        scan = []
        for step in range(len(actions) - 1):
            if step >= int(config["reserved_state_selection"]["scan_start_step"]):
                current = np.asarray(
                    feature_context(env, probe)["current_clearance_m"],
                    dtype=np.float64,
                )
                scan.append({
                    "state_step": int(step),
                    "current_ellipsoid_margin_m": current.tolist(),
                    "minimum_current_clearance_m": float(np.min(current)),
                    "two_action_horizon_available": bool(step + 1 < len(actions)),
                })
            if step < len(actions) - 2:
                env.step(_canonical_action(actions[step], step).tolist())
        return select_reserved_steps(scan, config), pairing
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main() -> int:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factorized-config", type=Path, required=True)
    parser.add_argument("--complete-dataset", type=Path, required=True)
    parser.add_argument("--complete-collection", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--reserved-manifest", type=Path, required=True)
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
        "factorized_config": args.factorized_config,
        "complete_dataset": args.complete_dataset,
        "complete_collection": args.complete_collection,
        "population": args.population_manifest,
        "reserved": args.reserved_manifest, "selected": args.selected_manifest,
        "same_task": args.same_task_manifest, "targeted": args.targeted_manifest,
        "archived": args.archived_root, "geometry": args.geometry_config,
        "exact_box": args.exact_box_config, "array_dataset": args.array_dataset,
        "metadata": args.metadata, "output": args.output,
    }.items()}
    config = load_direct_horizon_config(paths["config"])
    factorized_config = load_factorized_config(paths["factorized_config"])
    source = config["immutable_source"]
    for path, key in (
        (paths["population"], "population_manifest_file_sha256"),
        (paths["reserved"], "reserved_prediction_manifest_file_sha256"),
        (paths["factorized_config"], "factorized_config_file_sha256"),
        (paths["geometry"], "geometry_config_file_sha256"),
        (paths["exact_box"], "exact_box_config_file_sha256"),
        (paths["complete_dataset"], "complete_dataset_file_sha256"),
        (paths["complete_collection"], "complete_collection_file_sha256"),
    ):
        _require(_file_sha256(path) == source[key],
                 "reserved direct-horizon immutable source differs")
    complete_dataset = _load(paths["complete_dataset"])
    complete_collection = _load(paths["complete_collection"])
    aligned_names = [str(value) for value in complete_dataset[
        "complete_input_feature_names"
    ]]
    _require(
        len(aligned_names) == 2110
        and complete_collection.get("decision", {}).get(
            "input_sufficiency_pass"
        ) is True,
        "reserved direct-horizon complete-input source differs",
    )
    reserved_rows = _read_manifest(paths["reserved"], _file_sha256(paths["reserved"]))
    expected_cases = config["population"]["reserved_prediction_episode_ids"]
    _require(
        [row["case_id"] for row in reserved_rows] == expected_cases
        and all(row.get("selection_role") == "new_reserved_direct_horizon_prediction"
                for row in reserved_rows),
        "reserved direct-horizon manifest population differs",
    )
    source_rows = []
    for path in (paths["selected"], paths["same_task"], paths["targeted"]):
        source_rows.extend(_read_manifest(path, _file_sha256(path)))
    source_by_case = {str(row["case_id"]): row for row in source_rows}
    placeholder_row = _geometry_placeholder_row(source_by_case)
    placeholder_path = paths["archived"] / placeholder_row["archived_relative_path"]
    geometry_placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == placeholder_row["archived_file_sha256"]
        and geometry_placeholder.get("result_payload_sha256")
        == placeholder_row["archived_payload_sha256"]
        and geometry_placeholder.get("case_id") == _GEOMETRY_PLACEHOLDER_CASE_ID,
        "reserved direct-horizon geometry placeholder differs",
    )
    row_by_case = {str(row["case_id"]): row for row in reserved_rows}
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    selections: dict[str, list[dict[str, Any]]] = {}
    pairings = {}
    archives = {}
    for case_id in expected_cases:
        row = row_by_case[case_id]
        archived_path = paths["archived"] / row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == row["archived_file_sha256"]
            and archived.get("result_payload_sha256")
            == row["archived_payload_sha256"],
            "reserved direct-horizon archived episode differs",
        )
        case = population[case_id]
        validate_case_row(case, paths["repo"])
        selected, pairing = _scan_episode(
            runtime=runtime, case=case, archived=archived,
            geometry_config=geometry_config, exact_box_config=exact_box_config,
            geometry_placeholder=geometry_placeholder, config=config,
            expected_obstacle_name=str(row["active_obstacle_name"]),
        )
        selections[case_id] = selected
        pairings[case_id] = pairing
        archives[case_id] = archived
        print(json.dumps({
            "case_id": case_id,
            "selected_steps": [item["state_step"] for item in selected],
            "selected_current_minima_m": [
                item["minimum_current_clearance_m"] for item in selected
            ],
        }, sort_keys=True), flush=True)
    arrays: dict[str, list[Any]] = {
        "state_input_vector": [], "action_chunk": [],
        "joint_position_rad": [], "ellipsoid_clearance_m": [],
        "state_index": [], "candidate_index": [], "split_code": [],
        "source_code": [], "raw_contact_count": [],
        "maximum_obstacle_motion_m": [], "next_state_sha256_per_action": [],
    }
    state_records = []
    repeat_mismatches = []
    nominal_execution_mismatches = []
    maximum_repeat_q_difference = 0.0
    maximum_repeat_margin_difference = 0.0
    total_raw_contacts = 0
    state_offset = int(config["reserved_state_selection"]["state_index_offset"])
    for episode_index, case_id in enumerate(expected_cases):
        case = population[case_id]
        archived = archives[case_id]
        selected_by_step = {
            int(item["state_step"]): item for item in selections[case_id]
        }
        env = probe_env = None
        try:
            env, probe_env, _, _, setup = _build_pair(runtime, case)
            _require(
                setup["obstacle_name"] == row_by_case[case_id]["active_obstacle_name"],
                "reserved direct-horizon collection obstacle differs",
            )
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config,
                exact_box_config=exact_box_config,
                archived=geometry_placeholder, env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            actions = archived["actions"]
            maximum_step = max(selected_by_step)
            expected_next_hash: dict[int, str] = {}
            for step in range(maximum_step + 1):
                if step in selected_by_step:
                    selected = selected_by_step[step]
                    local_index = sorted(selected_by_step).index(step)
                    state_index = state_offset + 5 * episode_index + local_index
                    first = _canonical_action(actions[step], step)
                    second = _canonical_action(actions[step + 1], step + 1)
                    state_input = _state_input(env, probe, setup["obstacle_name"])
                    aligned = _aligned_state_vector(state_input, aligned_names)
                    current = np.asarray(
                        feature_context(env, probe)["current_clearance_m"],
                        dtype=np.float64,
                    )
                    _require(
                        np.array_equal(
                            current,
                            np.asarray(selected["current_ellipsoid_margin_m"],
                                       dtype=np.float64),
                        ),
                        "reserved direct-horizon scan/collection state differs",
                    )
                    candidates = candidate_chunks(
                        np.asarray([first, second], dtype=np.float64),
                        state_index, factorized_config,
                    )
                    candidate_receipts = []
                    for candidate in candidates:
                        candidate_index = int(candidate["candidate_index"])
                        action_chunk = np.asarray(
                            candidate["full_two_action_commands"], dtype=np.float64
                        )
                        chunk = probe.rollout_chunk(env, action_chunk.tolist())
                        receipt = trace_arrays(chunk, factorized_config)
                        contacts = _contact_receipt(chunk)
                        _require(
                            len(contacts) == int(receipt["raw_protected_contact_count"]),
                            "reserved direct-horizon contact receipt differs",
                        )
                        if candidate_index in config["candidate_design"][
                            "repeat_candidate_indexes"
                        ]:
                            repeated_chunk = probe.rollout_chunk(
                                env, action_chunk.tolist()
                            )
                            repeated = trace_arrays(
                                repeated_chunk, factorized_config
                            )
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
                            if not (
                                np.array_equal(
                                    receipt["joint_position_rad"],
                                    repeated["joint_position_rad"],
                                )
                                and np.array_equal(
                                    receipt["ellipsoid_clearance_m"],
                                    repeated["ellipsoid_clearance_m"],
                                )
                                and receipt["next_state_sha256_per_action"]
                                == repeated["next_state_sha256_per_action"]
                                and contacts == _contact_receipt(repeated_chunk)
                            ):
                                repeat_mismatches.append({
                                    "state_index": state_index,
                                    "candidate_index": candidate_index,
                                    "maximum_q_difference_rad": q_difference,
                                    "maximum_margin_difference_m": margin_difference,
                                })
                        if candidate_index == 0:
                            expected_next_hash[step] = str(
                                receipt["next_state_sha256_per_action"][0]
                            )
                        arrays["state_input_vector"].append(aligned)
                        arrays["action_chunk"].append(action_chunk)
                        arrays["joint_position_rad"].append(
                            receipt["joint_position_rad"]
                        )
                        arrays["ellipsoid_clearance_m"].append(
                            receipt["ellipsoid_clearance_m"]
                        )
                        arrays["state_index"].append(state_index)
                        arrays["candidate_index"].append(candidate_index)
                        arrays["split_code"].append(2)
                        arrays["source_code"].append(
                            _source_code(str(candidate["source"]))
                        )
                        arrays["raw_contact_count"].append(
                            int(receipt["raw_protected_contact_count"])
                        )
                        arrays["maximum_obstacle_motion_m"].append(
                            float(receipt["maximum_obstacle_motion_m"])
                        )
                        arrays["next_state_sha256_per_action"].append(
                            receipt["next_state_sha256_per_action"]
                        )
                        total_raw_contacts += int(
                            receipt["raw_protected_contact_count"]
                        )
                        candidate_receipts.append({
                            "candidate_index": candidate_index,
                            "action_chunk_sha256": hashlib.sha256(
                                action_chunk.tobytes()
                            ).hexdigest(),
                            "minimum_margin_m": np.min(
                                receipt["ellipsoid_clearance_m"], axis=0
                            ).tolist(),
                            "raw_contact_count": int(
                                receipt["raw_protected_contact_count"]
                            ),
                        })
                    state_records.append({
                        "state_index": state_index, "case_id": case_id,
                        "split": "test", "state_step": int(step),
                        "selection_rank": int(selected["selection_rank"]),
                        "minimum_current_clearance_m": float(
                            selected["minimum_current_clearance_m"]
                        ),
                        "current_ellipsoid_margin_m": current.tolist(),
                        "nominal_first_action": first.tolist(),
                        "nominal_second_action": second.tolist(),
                        "dynamic_state_sha256": state_input["dynamic_state_sha256"],
                        "complete_snapshot": state_input["complete_snapshot"],
                        "complete_snapshot_sha256": state_input[
                            "complete_snapshot_sha256"
                        ],
                        "complete_input_vector_sha256": hashlib.sha256(
                            aligned.tobytes()
                        ).hexdigest(),
                        "candidate_count": len(candidates),
                        "candidate_receipts": candidate_receipts,
                    })
                    print(json.dumps({
                        "case_id": case_id, "state_step": int(step),
                        "state_index": state_index,
                        "rollout_count_complete": len(arrays["state_index"]),
                    }, sort_keys=True), flush=True)
                if step < maximum_step:
                    env.step(_canonical_action(actions[step], step).tolist())
                    if step in expected_next_hash:
                        actual = hashlib.sha256(
                            _dynamic_state_vector(env).tobytes()
                        ).hexdigest()
                        if actual != expected_next_hash[step]:
                            nominal_execution_mismatches.append({
                                "case_id": case_id, "state_step": int(step),
                                "expected": expected_next_hash[step],
                                "actual": actual,
                            })
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    count = int(config["population"]["reserved_rollout_count"])
    numeric = {
        "state_input_vector": np.asarray(
            arrays["state_input_vector"], dtype=np.float64
        ),
        "action_chunk": np.asarray(arrays["action_chunk"], dtype=np.float64),
        "joint_position_rad": np.asarray(
            arrays["joint_position_rad"], dtype=np.float64
        ),
        "ellipsoid_clearance_m": np.asarray(
            arrays["ellipsoid_clearance_m"], dtype=np.float64
        ),
        "state_index": np.asarray(arrays["state_index"], dtype=np.int64),
        "candidate_index": np.asarray(arrays["candidate_index"], dtype=np.int64),
        "split_code": np.asarray(arrays["split_code"], dtype=np.int8),
        "source_code": np.asarray(arrays["source_code"], dtype=np.int8),
        "raw_contact_count": np.asarray(
            arrays["raw_contact_count"], dtype=np.int64
        ),
        "maximum_obstacle_motion_m": np.asarray(
            arrays["maximum_obstacle_motion_m"], dtype=np.float64
        ),
        "next_state_sha256_per_action": np.asarray(
            arrays["next_state_sha256_per_action"], dtype="U64"
        ),
    }
    collection_pass = bool(
        len(state_records) == int(config["population"]["reserved_state_count"])
        and numeric["state_input_vector"].shape == (count, 2110)
        and numeric["action_chunk"].shape == (count, 2, 7)
        and numeric["joint_position_rad"].shape == (count, 51, 7)
        and numeric["ellipsoid_clearance_m"].shape == (count, 51, 7)
        and not repeat_mismatches and not nominal_execution_mismatches
        and maximum_repeat_q_difference == 0.0
        and maximum_repeat_margin_difference == 0.0
    )
    _write_npz_atomic(paths["array_dataset"], numeric)
    metadata = {
        "schema_version": DATASET_SCHEMA, "status": "complete",
        "source_commit": args.expected_commit,
        "config_file_sha256": _file_sha256(paths["config"]),
        "complete_input_feature_names": aligned_names,
        "state_records": sorted(
            state_records, key=lambda item: int(item["state_index"])
        ),
        "pairings": pairings,
        "selection": {
            case_id: selections[case_id] for case_id in expected_cases
        },
        "summary": {
            "episode_count": len(expected_cases),
            "state_count": len(state_records),
            "rollout_count": int(len(numeric["state_index"])),
            "random_action_count": int(np.count_nonzero(
                numeric["source_code"] == 2
            )),
            "joint_trace_state_count": 51,
            "complete_state_input_dimension": 2110,
            "repeat_rollout_count": int(
                len(state_records)
                * len(config["candidate_design"]["repeat_candidate_indexes"])
            ),
            "maximum_repeat_q_difference_rad": maximum_repeat_q_difference,
            "maximum_repeat_margin_difference_m": maximum_repeat_margin_difference,
            "repeat_mismatch_count": len(repeat_mismatches),
            "nominal_execution_mismatch_count": len(
                nominal_execution_mismatches
            ),
            "raw_contact_observation_count": int(total_raw_contacts),
            "collection_gate_pass": collection_pass,
        },
        "array_dataset": {
            "path": str(paths["array_dataset"]),
            "file_sha256": _file_sha256(paths["array_dataset"]),
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
        "archived_table1_root": {"path": str(paths["archived"]), "read_only": True},
        "selection": metadata["selection"], "pairings": pairings,
        "dataset": {
            "metadata_path": str(paths["metadata"]),
            "metadata_file_sha256": _file_sha256(paths["metadata"]),
            "metadata_payload_sha256": metadata["dataset_payload_sha256"],
            "array_path": str(paths["array_dataset"]),
            "array_file_sha256": _file_sha256(paths["array_dataset"]),
        },
        "audit": {
            **metadata["summary"],
            "repeat_mismatches": repeat_mismatches,
            "nominal_execution_mismatches": nominal_execution_mismatches,
        },
        "decision": {
            "collection_gate_pass": collection_pass,
            "prediction_training_and_reserved_evaluation_authorized": collection_pass,
            "residual_bound_QP_or_closed_loop_authorized": False,
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "decision": result["decision"], "audit": result["audit"],
        "dataset": result["dataset"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
