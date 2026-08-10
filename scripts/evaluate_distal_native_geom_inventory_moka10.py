#!/usr/bin/env python3
"""Discover and validate native MuJoCo L5--L7 distance groups on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.native_geom_margin import (
    NativeGeomSubstepProbe, RESULT_SCHEMA, compiled_group_inventory,
    load_config, measure_native_groups, result_payload,
)
from main.multilink_ellipsoid.targeted_boundary_expansion import DATASET_SCHEMA
from main.multilink_ellipsoid.two_step_margin import (
    PAIR_FEATURE_NAMES, feature_context, feature_vectors,
)
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


def _read_manifest(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    _require(_file_sha256(path) == expected_sha256, "native inventory manifest differs")
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def _split_counts(states: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        split: sum(item["split"] == split for item in states)
        for split in ("train", "validation", "test")
    }


def _measurement_kwargs(config: Mapping[str, Any]) -> dict[str, float]:
    measurement = config["measurement"]
    return {
        "distance_max_m": float(measurement["distance_max_m"]),
        "distance_censor_tolerance_m": float(
            measurement["distance_censor_tolerance_m"]
        ),
        "safe_distance_m": float(measurement["safe_distance_m"]),
        "contact_distance_threshold_m": float(
            measurement["contact_distance_threshold_m"]
        ),
        "contact_distance_consistency_tolerance_m": float(
            measurement["contact_distance_consistency_tolerance_m"]
        ),
    }


def _chunk_summary(chunk: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    native = [item["native_geometry"] for item in chunk["transitions"]]
    physical = np.asarray(
        [item["minimum_substep_group_margins_m"] for item in native],
        dtype=np.float64,
    )
    ellipsoid = np.asarray([
        item["minimum_substep_clearance_m"][:7]
        for item in chunk["transitions"]
    ], dtype=np.float64)
    native_contacts = []
    for transition_index, transition in enumerate(chunk["transitions"]):
        for substep in transition["substeps"]:
            for contact in substep["native_geometry"][
                "raw_protected_contact_records"
            ]:
                native_contacts.append({
                    **contact, "transition_index": int(transition_index),
                    "substep_index": int(substep["substep_index"]),
                })
    return {
        "action_count": int(chunk["action_count"]),
        "initial_native_group_margins_m": native[0][
            "initial_group_margins_m"
        ],
        "minimum_native_group_margins_m": np.min(physical, axis=0).tolist(),
        "minimum_native_global_margin_m": float(np.min(physical)),
        "minimum_ellipsoid_D_opt_margins_m": np.min(ellipsoid, axis=0).tolist(),
        "minimum_ellipsoid_D_opt_global_margin_m": float(np.min(ellipsoid)),
        "raw_protected_contact_count": len(native_contacts),
        "raw_protected_contact_records": native_contacts,
        "all_queries_finite_and_uncensored": all(
            item["all_queries_finite_and_uncensored"] for item in native
        ),
        "every_raw_contact_pair_registered": all(
            item["every_raw_contact_pair_registered"] for item in native
        ),
        "every_raw_contact_distance_consistent": all(
            item["every_raw_contact_distance_consistent"] for item in native
        ),
    }


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
    parser.add_argument("--boundary-manifest", type=Path, required=True)
    parser.add_argument("--same-task-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "population": args.population_manifest,
        "boundary": args.boundary_manifest, "same_task": args.same_task_manifest,
        "targeted": args.targeted_manifest, "dataset": args.expanded_dataset,
        "archived": args.archived_root, "geometry": args.geometry_config,
        "exact_box": args.exact_box_config, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, key, label in (
        (paths["population"], "source_population_manifest_sha256", "population"),
        (paths["dataset"], "expanded_dataset_file_sha256", "dataset"),
        (paths["geometry"], "geometry_config_file_sha256", "geometry"),
        (paths["exact_box"], "exact_box_config_file_sha256", "exact-box"),
    ):
        _require(_file_sha256(path) == source[key], "native inventory %s differs" % label)
    selected = []
    geometry_placeholder_row_by_case = {}
    for key, path in (
        ("boundary_generalization_manifest_sha256", paths["boundary"]),
        ("same_task_expansion_manifest_sha256", paths["same_task"]),
        ("targeted_expansion_manifest_sha256", paths["targeted"]),
    ):
        cohort_rows = _read_manifest(path, source[key])
        cohort_placeholder = next(
            (item for item in cohort_rows if str(item["case_id"]).endswith("e05")),
            cohort_rows[0],
        )
        selected.extend(cohort_rows)
        for item in cohort_rows:
            geometry_placeholder_row_by_case[str(item["case_id"])] = (
                cohort_placeholder
            )
    row_by_case = {str(item["case_id"]): item for item in selected}
    _require(
        len(selected) == len(row_by_case)
        == int(config["population"]["expected_complete_episode_count"]),
        "native inventory selected episode population differs",
    )
    dataset = _load(paths["dataset"])
    _require(
        dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source["expanded_dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256"),
        "native inventory dataset identity differs",
    )
    states = dataset["state_records"]
    _require(
        len(states) == int(config["population"]["expected_state_count"])
        and _split_counts(states) == config["population"]["expected_split_counts"]
        and {str(item["case_id"]) for item in states} == set(row_by_case),
        "native inventory state population differs",
    )
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    runtime = _runtime_imports(include_aegis=False)
    state_results = []
    episode_results = []
    semantic_hashes = set()
    contact_witnesses = []
    all_pairings = {}
    measurement_kwargs = _measurement_kwargs(config)
    for case_id in sorted(row_by_case):
        selected_row = row_by_case[case_id]
        archived_path = paths["archived"] / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256")
            == selected_row["archived_payload_sha256"]
            and archived.get("case_id") == case_id,
            "native inventory archived episode differs: %s" % case_id,
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
                         "native inventory pairing differs for %s: %s" % (case_id, key))
            all_pairings[case_id] = pairing
            inventory = compiled_group_inventory(env, setup["obstacle_name"])
            probe_inventory = compiled_group_inventory(probe_env, setup["obstacle_name"])
            _require(
                inventory["semantic_protected_group_sha256"]
                == probe_inventory["semantic_protected_group_sha256"],
                "native inventory main/clone protected groups differ",
            )
            semantic_hashes.add(inventory["semantic_protected_group_sha256"])
            geometry_placeholder_row = geometry_placeholder_row_by_case[case_id]
            geometry_placeholder_path = paths["archived"] / geometry_placeholder_row[
                "archived_relative_path"
            ]
            geometry_placeholder = _load(geometry_placeholder_path)
            _require(
                _file_sha256(geometry_placeholder_path)
                == geometry_placeholder_row["archived_file_sha256"]
                and geometry_placeholder.get("result_payload_sha256")
                == geometry_placeholder_row["archived_payload_sha256"],
                "native inventory cohort geometry placeholder differs",
            )
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                archived=geometry_placeholder, env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = NativeGeomSubstepProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
                native_inventory=probe_inventory,
                native_measurement=config["measurement"],
            )
            case_states = sorted(
                (item for item in states if item["case_id"] == case_id),
                key=lambda item: int(item["state_step"]),
            )
            by_step = {int(item["state_step"]): item for item in case_states}
            witness_steps = set()
            if case_id == "vlsa-t1-goal-ii-t0-e05":
                witness_steps = set(config["measurement"]["primary_contact_witness_steps"])
            maximum_step = max(set(by_step) | witness_steps)
            actions = archived["actions"]
            _require(maximum_step + 1 < len(actions), "native inventory action horizon missing")
            episode_start = len(state_results)
            for step in range(maximum_step + 1):
                if step in by_step:
                    state = by_step[step]
                    current = measure_native_groups(
                        env, inventory, **measurement_kwargs
                    )
                    first = _canonical_action(actions[step], step)
                    second = _canonical_action(actions[step + 1], step + 1)
                    context = feature_context(env, probe)
                    _, live_pair_features = feature_vectors(
                        context, first[:3], first[:3], second[:3]
                    )
                    expected_pair_features = np.asarray(
                        state["pair_state_feature_vectors"], dtype=np.float64
                    )
                    receipt_difference = np.abs(
                        live_pair_features - expected_pair_features
                    )
                    receipt_error = float(np.max(receipt_difference))
                    if receipt_error > 1.0e-8:
                        row, column = np.unravel_index(
                            int(np.argmax(receipt_difference)),
                            receipt_difference.shape,
                        )
                        print(json.dumps({
                            "event": "native_inventory_state_receipt_mismatch",
                            "case_id": case_id, "state_step": step,
                            "maximum_absolute_error": receipt_error,
                            "constraint_row": int(row),
                            "feature_index": int(column),
                            "feature_name": PAIR_FEATURE_NAMES[int(column)],
                            "expected": float(expected_pair_features[row, column]),
                            "observed": float(live_pair_features[row, column]),
                            "current_clearance_error": float(np.max(np.abs(
                                np.asarray(context["current_clearance_m"], dtype=np.float64)
                                - np.asarray(state["current_clearance_m"], dtype=np.float64)
                            ))),
                        }, sort_keys=True), flush=True)
                    _require(receipt_error <= 1.0e-8,
                             "native inventory state feature receipt differs")
                    summary = _chunk_summary(probe.rollout_chunk(env, [first, second]))
                    initial_safe = bool(current["global_minimum_margin_m"] >= 0.0)
                    state_results.append({
                        "state_index": int(state["state_index"]),
                        "case_id": case_id, "split": str(state["split"]),
                        "state_step": step, "state_receipt_maximum_error": receipt_error,
                        "semantic_protected_groups": inventory[
                            "semantic_protected_groups"
                        ],
                        "semantic_protected_group_sha256": inventory[
                            "semantic_protected_group_sha256"
                        ],
                        "obstacle_collision_geom_count": inventory[
                            "obstacle_collision_geom_count"
                        ],
                        "initial_native_group_margins_m": current["group_margins_m"],
                        "initial_native_link_minimum_margins_m": current[
                            "link_minimum_margins_m"
                        ],
                        "initial_native_global_margin_m": current[
                            "global_minimum_margin_m"
                        ],
                        "cohort": "prevention" if initial_safe else "recovery",
                        "nominal_two_action_rollout": summary,
                    })
                if step in witness_steps:
                    action = _canonical_action(actions[step], step)
                    witness = _chunk_summary(probe.rollout_chunk(env, [action]))
                    contact_witnesses.append({
                        "case_id": case_id, "state_step": step, **witness,
                    })
                if step < maximum_step:
                    env.step(_canonical_action(actions[step], step).tolist())
            episode_results.append({
                "case_id": case_id,
                "comparison_geometry_placeholder_case_id": str(
                    geometry_placeholder_row["case_id"]
                ),
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
                "state_count": len(state_results) - episode_start,
            })
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    state_results.sort(key=lambda item: int(item["state_index"]))
    _require(
        [int(item["state_index"]) for item in state_results]
        == list(range(len(states))),
        "native inventory state output order differs",
    )
    all_rollouts = [item["nominal_two_action_rollout"] for item in state_results]
    test_prevention = sum(
        item["split"] == "test" and item["cohort"] == "prevention"
        for item in state_results
    )
    contact_count = sum(item["raw_protected_contact_count"] for item in contact_witnesses)
    aggregates = {
        "state_count": len(state_results),
        "episode_count": len(episode_results),
        "protected_group_count": len(state_results[0]["semantic_protected_groups"]),
        "semantic_inventory_hash_count": len(semantic_hashes),
        "prevention_state_count": sum(item["cohort"] == "prevention" for item in state_results),
        "recovery_state_count": sum(item["cohort"] == "recovery" for item in state_results),
        "test_initially_native_safe_state_count": int(test_prevention),
        "primary_contact_witness_count": int(contact_count),
        "all_queries_finite_and_uncensored": bool(
            all(item["all_queries_finite_and_uncensored"] for item in all_rollouts)
            and all(item["all_queries_finite_and_uncensored"] for item in contact_witnesses)
        ),
        "every_raw_contact_pair_registered": bool(
            all(item["every_raw_contact_pair_registered"] for item in all_rollouts)
            and all(item["every_raw_contact_pair_registered"] for item in contact_witnesses)
        ),
        "every_raw_contact_distance_consistent": bool(
            all(item["every_raw_contact_distance_consistent"] for item in all_rollouts)
            and all(item["every_raw_contact_distance_consistent"] for item in contact_witnesses)
        ),
    }
    gate = config["gate"]
    passed = bool(
        aggregates["semantic_inventory_hash_count"] == 1
        and aggregates["all_queries_finite_and_uncensored"]
        and aggregates["every_raw_contact_pair_registered"]
        and aggregates["every_raw_contact_distance_consistent"]
        and aggregates["primary_contact_witness_count"]
        >= int(gate["required_primary_contact_witness_count"])
        and aggregates["test_initially_native_safe_state_count"]
        == int(gate["required_test_initially_native_safe_state_count"])
    )
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "pairings": all_pairings, "episode_results": episode_results,
        "state_results": state_results, "primary_contact_witnesses": contact_witnesses,
        "aggregates": aggregates,
        "decision": {
            "native_inventory_gate_pass": passed,
            "physical_target_collection_authorized": passed,
            "training_authorized": False, "QP_authorized": False,
            "closed_loop_E05_authorized": False,
            "stop_reason": None if passed else "native_geometry_inventory_gate_failed",
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["result_payload_sha256"] = result_payload(output)
    _atomic_write(paths["output"], output)
    print(json.dumps({
        "aggregates": aggregates, "decision": output["decision"],
        "episode_inventories": episode_results, "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
