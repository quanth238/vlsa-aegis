#!/usr/bin/env python3
"""Evaluate zero/small-cutoff native distance consistency on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.adaptive_native_distance import (
    AdaptiveNativeSubstepProbe, RESULT_SCHEMA, load_config,
    measure_adaptive_groups, result_payload,
)
from main.multilink_ellipsoid.native_geom_margin import compiled_group_inventory
from main.multilink_ellipsoid.targeted_boundary_expansion import DATASET_SCHEMA
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _manifest(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    _require(_file_sha256(path) == expected_sha256,
             "adaptive native-distance manifest differs")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _chunk_summary(chunk: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    summaries = [item["adaptive_native"] for item in chunk["transitions"]]
    margins = np.asarray([
        item["minimum_substep_group_margins_m"] for item in summaries
    ], dtype=np.float64)
    count_keys = (
        "raw_protected_contact_count", "negative_without_raw_pair_contact_count",
        "raw_pair_contact_without_negative_count", "cutoff_induced_negative_count",
        "repeated_uncensored_inconsistency_count", "unregistered_raw_pair_count",
    )
    output = {
        "action_count": int(chunk["action_count"]),
        "initial_group_margins_m": summaries[0]["initial_group_margins_m"],
        "minimum_substep_group_margins_m": np.min(margins, axis=0).tolist(),
        "minimum_substep_global_margin_m": float(np.min(margins)),
        "all_queries_finite": all(item["all_queries_finite"] for item in summaries),
    }
    for key in count_keys:
        output[key] = sum(int(item[key]) for item in summaries)
    return output


def main() -> int:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--prior-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "population": args.population_manifest, "test_manifest": args.test_manifest,
        "dataset": args.expanded_dataset, "archived": args.archived_root,
        "geometry": args.geometry_config, "exact_box": args.exact_box_config,
        "prior_result": args.prior_result,
        "prior_validation": args.prior_validation, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, key, label in (
        (paths["population"], "source_population_manifest_sha256", "population"),
        (paths["dataset"], "expanded_dataset_file_sha256", "dataset"),
        (paths["geometry"], "geometry_config_file_sha256", "geometry"),
        (paths["exact_box"], "exact_box_config_file_sha256", "exact-box"),
        (paths["prior_result"], "prior_inventory_result_file_sha256", "prior-result"),
        (paths["prior_validation"], "prior_inventory_validation_file_sha256", "prior-validation"),
    ):
        _require(_file_sha256(path) == source[key],
                 "adaptive native-distance %s differs" % label)
    prior = _load(paths["prior_result"])
    validation = _load(paths["prior_validation"])
    _require(
        prior.get("result_payload_sha256")
        == source["prior_inventory_result_payload_sha256"]
        == _hash_without(prior, "result_payload_sha256")
        and prior.get("decision", {}).get("native_inventory_gate_pass") is False
        and validation.get("status") == "valid"
        and validation.get("native_inventory_gate_pass") is False,
        "adaptive native-distance authorization differs",
    )
    rows = _manifest(paths["test_manifest"], source["test_manifest_sha256"])
    row_by_case = {
        str(item["case_id"]): item for item in rows
        if str(item["case_id"]) in config["population"]["case_ids"]
    }
    _require(list(sorted(row_by_case)) == list(sorted(config["population"]["case_ids"])),
             "adaptive native-distance cases differ")
    dataset = _load(paths["dataset"])
    _require(
        dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source["expanded_dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256"),
        "adaptive native-distance dataset differs",
    )
    states = [
        item for item in dataset["state_records"]
        if item["case_id"] in row_by_case
    ]
    _require(
        len(states) == int(config["population"]["expected_state_count"])
        and all(item["split"] == "test" for item in states),
        "adaptive native-distance state population differs",
    )
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    placeholder_row = row_by_case["vlsa-t1-goal-ii-t0-e05"]
    placeholder_path = paths["archived"] / placeholder_row["archived_relative_path"]
    placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == placeholder_row["archived_file_sha256"]
        and placeholder.get("result_payload_sha256")
        == placeholder_row["archived_payload_sha256"],
        "adaptive native-distance placeholder differs",
    )
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    state_results = []
    witnesses = []
    episode_results = []
    semantic_hashes = set()
    for case_id in config["population"]["case_ids"]:
        row = row_by_case[case_id]
        archived_path = paths["archived"] / row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == row["archived_file_sha256"]
            and archived.get("result_payload_sha256") == row["archived_payload_sha256"],
            "adaptive native-distance archive differs: %s" % case_id,
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
                _require(pairing[key] == archived["pairing"][key],
                         "adaptive native-distance pairing differs: %s" % key)
            inventory = compiled_group_inventory(env, setup["obstacle_name"])
            probe_inventory = compiled_group_inventory(probe_env, setup["obstacle_name"])
            _require(
                inventory["semantic_protected_group_sha256"]
                == probe_inventory["semantic_protected_group_sha256"],
                "adaptive native-distance clone inventory differs",
            )
            semantic_hashes.add(inventory["semantic_protected_group_sha256"])
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                archived=placeholder, env=env, obstacle_name=setup["obstacle_name"],
            )
            probe = AdaptiveNativeSubstepProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
                native_inventory=probe_inventory,
                adaptive_measurement=config["measurement"],
            )
            case_states = sorted(
                (item for item in states if item["case_id"] == case_id),
                key=lambda item: int(item["state_step"]),
            )
            by_step = {int(item["state_step"]): item for item in case_states}
            witness_steps = set(
                config["population"]["primary_contact_witness_steps"]
                if case_id == "vlsa-t1-goal-ii-t0-e05" else []
            )
            maximum_step = max(set(by_step) | witness_steps)
            actions = archived["actions"]
            for step in range(maximum_step + 1):
                if step in by_step:
                    current = measure_adaptive_groups(
                        env, inventory, config["measurement"]
                    )
                    first = _canonical_action(actions[step], step)
                    second = _canonical_action(actions[step + 1], step + 1)
                    rollout = _chunk_summary(
                        probe.rollout_chunk(env, [first, second])
                    )
                    state_results.append({
                        "source_state_index": int(by_step[step]["state_index"]),
                        "case_id": case_id, "state_step": step,
                        "initial_group_margins_m": current["group_margins_m"],
                        "initial_global_minimum_margin_m": current[
                            "global_minimum_margin_m"
                        ],
                        "initially_safe": bool(
                            current["global_minimum_margin_m"] >= 0.0
                            and current["negative_without_raw_pair_contact_count"] == 0
                            and current["raw_pair_contact_without_negative_count"] == 0
                        ),
                        "current_measurement": {
                            key: current[key] for key in (
                                "raw_protected_contact_count",
                                "unregistered_raw_pair_count",
                                "negative_without_raw_pair_contact_count",
                                "raw_pair_contact_without_negative_count",
                                "cutoff_induced_negative_count",
                                "repeated_uncensored_inconsistency_count",
                                "all_queries_finite",
                            )
                        },
                        "nominal_two_action_rollout": rollout,
                    })
                if step in witness_steps:
                    action = _canonical_action(actions[step], step)
                    witnesses.append({
                        "case_id": case_id, "state_step": step,
                        **_chunk_summary(probe.rollout_chunk(env, [action])),
                    })
                if step < maximum_step:
                    env.step(_canonical_action(actions[step], step).tolist())
            episode_results.append({
                "case_id": case_id,
                "semantic_protected_groups": inventory[
                    "semantic_protected_groups"
                ],
                "semantic_protected_group_sha256": inventory[
                    "semantic_protected_group_sha256"
                ],
                "protected_group_count": len(inventory["groups"]),
                "obstacle_collision_geom_count": inventory[
                    "obstacle_collision_geom_count"
                ],
            })
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    all_measurements = [item["current_measurement"] for item in state_results]
    all_rollouts = [item["nominal_two_action_rollout"] for item in state_results]
    all_records = all_measurements + all_rollouts + witnesses
    aggregates = {
        "state_count": len(state_results), "episode_count": len(episode_results),
        "semantic_protected_group_inventory_hash_count": len(semantic_hashes),
        "protected_group_count": episode_results[0]["protected_group_count"],
        "initially_safe_test_state_count": sum(
            item["initially_safe"] for item in state_results
        ),
        "registered_raw_contact_count": sum(
            int(item["raw_protected_contact_count"]) for item in witnesses
        ),
        "negative_without_raw_pair_contact_count": sum(
            int(item["negative_without_raw_pair_contact_count"])
            for item in all_records
        ),
        "raw_pair_contact_without_negative_count": sum(
            int(item["raw_pair_contact_without_negative_count"])
            for item in all_records
        ),
        "cutoff_induced_negative_count": sum(
            int(item["cutoff_induced_negative_count"]) for item in all_records
        ),
        "repeated_uncensored_inconsistency_count": sum(
            int(item["repeated_uncensored_inconsistency_count"])
            for item in all_records
        ),
        "unregistered_raw_pair_count": sum(
            int(item["unregistered_raw_pair_count"]) for item in all_records
        ),
        "all_queries_finite": all(item["all_queries_finite"] for item in all_records),
    }
    gate = config["gate"]
    passed = bool(
        aggregates["semantic_protected_group_inventory_hash_count"]
        == int(gate["semantic_protected_group_inventory_hash_count"])
        and aggregates["negative_without_raw_pair_contact_count"]
        <= int(gate["maximum_negative_without_raw_pair_contact_count"])
        and aggregates["raw_pair_contact_without_negative_count"]
        <= int(gate["maximum_raw_pair_contact_without_negative_count"])
        and aggregates["cutoff_induced_negative_count"]
        <= int(gate["maximum_cutoff_induced_negative_count"])
        and aggregates["repeated_uncensored_inconsistency_count"]
        <= int(gate["maximum_repeated_uncensored_inconsistency_count"])
        and aggregates["unregistered_raw_pair_count"] == 0
        and aggregates["all_queries_finite"]
        and aggregates["registered_raw_contact_count"]
        >= int(gate["required_registered_raw_contact_count"])
        and aggregates["initially_safe_test_state_count"]
        == int(gate["required_initially_safe_test_state_count"])
    )
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "episode_results": episode_results, "state_results": state_results,
        "primary_contact_witnesses": witnesses, "aggregates": aggregates,
        "decision": {
            "adaptive_native_distance_gate_pass": passed,
            "full_physical_target_collection_preregistration_authorized": passed,
            "training_authorized": False, "QP_authorized": False,
            "closed_loop_E05_authorized": False,
            "stop_reason": None if passed else "adaptive_native_distance_gate_failed",
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["result_payload_sha256"] = result_payload(output)
    _atomic_write(paths["output"], output)
    print(json.dumps({
        "aggregates": aggregates, "decision": output["decision"],
        "state_results": [{
            "case_id": item["case_id"], "state_step": item["state_step"],
            "initial_margin_m": item["initial_global_minimum_margin_m"],
            "initially_safe": item["initially_safe"],
        } for item in state_results], "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
