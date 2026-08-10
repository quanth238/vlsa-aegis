#!/usr/bin/env python3
"""Replay all paired states and classify their initial protected contact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.initial_contact_audit import (
    RESULT_SCHEMA, load_config, measure_initial_contact, summarize,
)
from main.multilink_ellipsoid.native_geom_margin import compiled_group_inventory
from main.multilink_ellipsoid.targeted_boundary_expansion import DATASET_SCHEMA
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair
from scripts.evaluate_distal_native_geom_inventory_moka10 import (
    _read_manifest, _split_counts,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def main() -> int:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--boundary-manifest", type=Path, required=True)
    parser.add_argument("--same-task-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--proxy-contact-result", type=Path, required=True)
    parser.add_argument("--proxy-contact-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "population": args.population_manifest, "boundary": args.boundary_manifest,
        "same_task": args.same_task_manifest, "targeted": args.targeted_manifest,
        "dataset": args.expanded_dataset, "archived": args.archived_root,
        "proxy_result": args.proxy_contact_result,
        "proxy_validation": args.proxy_contact_validation,
        "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, key, label in (
        (paths["population"], "source_population_manifest_sha256", "population"),
        (paths["boundary"], "boundary_generalization_manifest_sha256", "boundary"),
        (paths["same_task"], "same_task_expansion_manifest_sha256", "same-task"),
        (paths["targeted"], "targeted_expansion_manifest_sha256", "targeted"),
        (paths["dataset"], "expanded_dataset_file_sha256", "dataset"),
        (paths["proxy_result"], "proxy_contact_result_file_sha256", "proxy-result"),
        (paths["proxy_validation"], "proxy_contact_validation_file_sha256", "proxy-validation"),
    ):
        _require(_file_sha256(path) == source[key], f"initial-contact {label} differs")
    prior = _load(paths["proxy_result"])
    prior_validation = _load(paths["proxy_validation"])
    _require(
        prior.get("result_payload_sha256")
        == source["proxy_contact_result_payload_sha256"]
        == _hash_without(prior, "result_payload_sha256")
        and prior.get("decision", {}).get("initial_state_audit_authorized") is True
        and prior_validation.get("status") == "valid"
        and prior_validation.get("initial_state_audit_authorized") is True,
        "initial-contact authorization differs",
    )
    selected = []
    for path in (paths["boundary"], paths["same_task"], paths["targeted"]):
        selected.extend(_read_manifest(path, _file_sha256(path)))
    row_by_case = {str(item["case_id"]): item for item in selected}
    _require(
        len(selected) == len(row_by_case)
        == int(config["population"]["expected_complete_episode_count"]),
        "initial-contact episode population differs",
    )
    dataset = _load(paths["dataset"])
    _require(
        dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source["expanded_dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256"),
        "initial-contact dataset differs",
    )
    states = dataset["state_records"]
    _require(
        len(states) == int(config["population"]["expected_state_count"])
        and _split_counts(states) == config["population"]["expected_split_counts"]
        and {str(item["case_id"]) for item in states} == set(row_by_case),
        "initial-contact state population differs",
    )
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    runtime = _runtime_imports(include_aegis=False)
    state_results = []
    semantic_hashes = set()
    episode_results = []
    for case_id in sorted(row_by_case):
        selected_row = row_by_case[case_id]
        archived_path = paths["archived"] / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256")
            == selected_row["archived_payload_sha256"]
            and archived.get("case_id") == case_id,
            f"initial-contact archive differs: {case_id}",
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
                    pairing[key] == archived["pairing"][key],
                    f"initial-contact pairing differs: {case_id}:{key}",
                )
            inventory = compiled_group_inventory(env, setup["obstacle_name"])
            semantic_hashes.add(inventory["semantic_protected_group_sha256"])
            case_states = sorted(
                (item for item in states if item["case_id"] == case_id),
                key=lambda item: int(item["state_step"]),
            )
            by_step = {int(item["state_step"]): item for item in case_states}
            maximum_step = max(by_step)
            actions = archived["actions"]
            _require(maximum_step < len(actions), "initial-contact prefix differs")
            for step in range(maximum_step + 1):
                if step in by_step:
                    state = by_step[step]
                    measurement = measure_initial_contact(
                        env, inventory, config["measurement"]
                    )
                    state_results.append({
                        "state_index": int(state["state_index"]),
                        "case_id": case_id, "split": str(state["split"]),
                        "state_step": step,
                        "semantic_protected_group_sha256": inventory[
                            "semantic_protected_group_sha256"
                        ],
                        **measurement,
                    })
                if step < maximum_step:
                    env.step(_canonical_action(actions[step], step).tolist())
            episode_results.append({
                "case_id": case_id,
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
    state_results.sort(key=lambda item: int(item["state_index"]))
    analysis = summarize(state_results, config, len(semantic_hashes))
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "episode_results": episode_results, "state_results": state_results,
        **analysis,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["result_payload_sha256"] = _hash_without(
        output, "result_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps({
        "aggregates": output["aggregates"], "decision": output["decision"],
        "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
