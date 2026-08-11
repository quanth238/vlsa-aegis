#!/usr/bin/env python3
"""Independently validate and replay the reserved direct-horizon population."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    COLLECTION_SCHEMA, COLLECTION_VALIDATION_SCHEMA, DATASET_SCHEMA,
    load_direct_horizon_config,
)
from main.multilink_ellipsoid.factorized_execution_pilot import (
    candidate_chunks, load_config as load_factorized_config, payload_sha256,
    trace_arrays,
)
from scripts.collect_distal_complete_osc_margin_moka10 import (
    _GEOMETRY_PLACEHOLDER_CASE_ID, _contact_receipt, _geometry_placeholder_row,
    _state_input,
)
from scripts.collect_distal_direct_horizon_reserved_moka10 import (
    _aligned_state_vector, restore_json_snapshot,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_native_geom_inventory_moka10 import _read_manifest
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def main() -> int:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factorized-config", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--reserved-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--same-task-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--array-dataset", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-collection-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "factorized_config": args.factorized_config,
        "population": args.population_manifest,
        "reserved": args.reserved_manifest, "selected": args.selected_manifest,
        "same_task": args.same_task_manifest, "targeted": args.targeted_manifest,
        "archived": args.archived_root, "geometry": args.geometry_config,
        "exact_box": args.exact_box_config, "collection": args.collection,
        "metadata": args.metadata, "array": args.array_dataset,
        "output": args.output,
    }.items()}
    config = load_direct_horizon_config(paths["config"])
    factorized_config = load_factorized_config(paths["factorized_config"])
    collection = _load(paths["collection"])
    metadata = _load(paths["metadata"])
    _require(
        collection.get("schema_version") == COLLECTION_SCHEMA
        and collection.get("result_payload_sha256")
        == payload_sha256(collection, "result_payload_sha256")
        and collection.get("decision", {}).get("collection_gate_pass") is True
        and collection["source"]["commit"] == args.expected_collection_commit
        and collection["dataset"]["metadata_file_sha256"]
        == _file_sha256(paths["metadata"])
        and collection["dataset"]["array_file_sha256"]
        == _file_sha256(paths["array"])
        and metadata.get("schema_version") == DATASET_SCHEMA
        and metadata.get("dataset_payload_sha256")
        == payload_sha256(metadata, "dataset_payload_sha256")
        and metadata["array_dataset"]["file_sha256"]
        == _file_sha256(paths["array"]),
        "reserved direct-horizon collection identity differs",
    )
    archive = np.load(paths["array"], allow_pickle=False)
    count = int(config["population"]["reserved_rollout_count"])
    state_count = int(config["population"]["reserved_state_count"])
    arrays = {
        "state_input_vector": archive["state_input_vector"].astype(np.float64),
        "action_chunk": archive["action_chunk"].astype(np.float64),
        "joint_position_rad": archive["joint_position_rad"].astype(np.float64),
        "ellipsoid_clearance_m": archive["ellipsoid_clearance_m"].astype(np.float64),
        "state_index": archive["state_index"].astype(np.int64),
        "candidate_index": archive["candidate_index"].astype(np.int64),
        "split_code": archive["split_code"].astype(np.int8),
        "source_code": archive["source_code"].astype(np.int8),
        "raw_contact_count": archive["raw_contact_count"].astype(np.int64),
        "maximum_obstacle_motion_m": archive["maximum_obstacle_motion_m"].astype(np.float64),
        "next_state_sha256_per_action": archive[
            "next_state_sha256_per_action"
        ].astype(str),
    }
    shape_gate = bool(
        arrays["state_input_vector"].shape == (count, 2110)
        and arrays["action_chunk"].shape == (count, 2, 7)
        and arrays["joint_position_rad"].shape == (count, 51, 7)
        and arrays["ellipsoid_clearance_m"].shape == (count, 51, 7)
        and arrays["next_state_sha256_per_action"].shape == (count, 2)
        and len(set(arrays["state_index"].tolist())) == state_count
        and np.all(arrays["split_code"] == 2)
        and np.count_nonzero(arrays["source_code"] == 2)
        == int(config["population"]["reserved_random_action_count"])
        and all(np.all(np.isfinite(value)) for value in (
            arrays["state_input_vector"], arrays["action_chunk"],
            arrays["joint_position_rad"], arrays["ellipsoid_clearance_m"],
            arrays["maximum_obstacle_motion_m"],
        ))
    )
    candidate_mismatches = []
    initial_condition_mismatches = []
    row_by_identity = {}
    records = sorted(metadata["state_records"], key=lambda item: int(item["state_index"]))
    for state in records:
        state_index = int(state["state_index"])
        rows = np.flatnonzero(arrays["state_index"] == state_index)
        by_candidate = {
            int(arrays["candidate_index"][row]): int(row) for row in rows
        }
        row_by_identity.update({(state_index, key): value for key, value in by_candidate.items()})
        expected = candidate_chunks(
            np.asarray([
                state["nominal_first_action"], state["nominal_second_action"],
            ], dtype=np.float64),
            state_index, factorized_config,
        )
        if sorted(by_candidate) != list(range(len(expected))):
            candidate_mismatches.append({"state_index": state_index, "reason": "indexes"})
            continue
        q0 = arrays["joint_position_rad"][rows, 0]
        if not np.all(q0 == q0[0]):
            initial_condition_mismatches.append(state_index)
        for item in expected:
            row = by_candidate[int(item["candidate_index"])]
            expected_action = np.asarray(item["full_two_action_commands"], dtype=np.float64)
            if not np.array_equal(arrays["action_chunk"][row], expected_action):
                candidate_mismatches.append({
                    "state_index": state_index,
                    "candidate_index": int(item["candidate_index"]),
                    "reason": "action",
                })
    # Freshly replay one nominal chunk from every stored complete snapshot.
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
        "reserved direct-horizon validation geometry placeholder differs",
    )
    reserved_rows = _read_manifest(paths["reserved"], _file_sha256(paths["reserved"]))
    reserved_by_case = {str(row["case_id"]): row for row in reserved_rows}
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    replay_mismatches = []
    maximum_q_difference = 0.0
    maximum_margin_difference = 0.0
    names = [str(value) for value in metadata["complete_input_feature_names"]]
    for case_id in config["population"]["reserved_prediction_episode_ids"]:
        _require(case_id in reserved_by_case, "reserved validation case differs")
        case = population[case_id]
        validate_case_row(case, paths["repo"])
        env = probe_env = None
        try:
            env, probe_env, _, _, setup = _build_pair(runtime, case)
            _require(
                setup["obstacle_name"]
                == reserved_by_case[case_id]["active_obstacle_name"],
                "reserved direct-horizon validation obstacle differs",
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
            for state in (item for item in records if item["case_id"] == case_id):
                state_index = int(state["state_index"])
                row = row_by_identity[(state_index, 0)]
                restore_json_snapshot(env, state["complete_snapshot"])
                replay_input = _aligned_state_vector(
                    _state_input(env, probe, setup["obstacle_name"]), names,
                )
                replay = trace_arrays(
                    probe.rollout_chunk(env, arrays["action_chunk"][row].tolist()),
                    factorized_config,
                )
                q_difference = float(np.max(np.abs(
                    replay["joint_position_rad"] - arrays["joint_position_rad"][row]
                )))
                margin_difference = float(np.max(np.abs(
                    replay["ellipsoid_clearance_m"]
                    - arrays["ellipsoid_clearance_m"][row]
                )))
                maximum_q_difference = max(maximum_q_difference, q_difference)
                maximum_margin_difference = max(
                    maximum_margin_difference, margin_difference
                )
                if not (
                    np.array_equal(replay_input, arrays["state_input_vector"][row])
                    and np.array_equal(
                        replay["joint_position_rad"], arrays["joint_position_rad"][row]
                    )
                    and np.array_equal(
                        replay["ellipsoid_clearance_m"],
                        arrays["ellipsoid_clearance_m"][row],
                    )
                    and replay["next_state_sha256_per_action"]
                    == arrays["next_state_sha256_per_action"][row].tolist()
                    and int(replay["raw_protected_contact_count"])
                    == int(arrays["raw_contact_count"][row])
                ):
                    replay_mismatches.append({
                        "state_index": state_index,
                        "maximum_q_difference_rad": q_difference,
                        "maximum_margin_difference_m": margin_difference,
                    })
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    valid = bool(
        shape_gate and not candidate_mismatches
        and not initial_condition_mismatches and not replay_mismatches
        and maximum_q_difference == 0.0 and maximum_margin_difference == 0.0
    )
    output = {
        "schema_version": COLLECTION_VALIDATION_SCHEMA,
        "status": "complete", "valid": valid,
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(),
        "collection_file_sha256": _file_sha256(paths["collection"]),
        "collection_payload_sha256": collection["result_payload_sha256"],
        "metadata_file_sha256": _file_sha256(paths["metadata"]),
        "array_file_sha256": _file_sha256(paths["array"]),
        "audit": {
            "shape_gate_pass": shape_gate,
            "state_count": len(records), "rollout_count": count,
            "fresh_nominal_replay_count": len(records),
            "candidate_mismatches": candidate_mismatches,
            "initial_condition_mismatches": initial_condition_mismatches,
            "replay_mismatches": replay_mismatches,
            "maximum_replay_q_difference_rad": maximum_q_difference,
            "maximum_replay_margin_difference_m": maximum_margin_difference,
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
