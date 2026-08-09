#!/usr/bin/env python3
"""Evaluate exact two- and five-step E05 recovery chunk families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.chunk_oracle import (
    RESULT_SCHEMA, distributed_chunks, load_chunk_oracle_config,
    one_shot_chunks,
)
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value); payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _summarize_rollout(
    rollout: Mapping[str, Any], nominal_chunk: Any, candidate: Mapping[str, Any],
    config: Mapping[str, Any], candidate_index: int,
) -> dict[str, Any]:
    import numpy as np

    transitions = rollout["transitions"]
    margins = np.asarray(
        [item["minimum_substep_clearance_m"][:7] for item in transitions],
        dtype=np.float64,
    )
    correction = np.asarray(candidate["correction"], dtype=np.float64)
    raw_contacts = sum(int(item["raw_protected_contact_count"]) for item in transitions)
    maximum_motion = max(
        float(item["maximum_within_step_obstacle_l1_displacement_m"])
        for item in transitions
    )
    safe = bool(
        np.all(margins >= float(config["geometry"]["clearance_target_m"]))
        and raw_contacts == 0
        and maximum_motion
        <= float(config["verification"]["maximum_per_step_obstacle_l1_displacement_m"])
    )
    active = np.unravel_index(int(np.argmin(margins)), margins.shape)
    return {
        "candidate_index": int(candidate_index),
        "source": str(candidate["source"]),
        "parameter": np.asarray(candidate["parameter"], dtype=np.float64).tolist(),
        "objective_chunk_correction_l2": float(np.linalg.norm(correction)),
        "correction_sum_xyz": np.sum(correction, axis=0).tolist(),
        "minimum_substep_clearance_m": np.min(margins, axis=0).tolist(),
        "minimum_distal_substep_clearance_m": float(np.min(margins)),
        "active_chunk_step": int(active[0]),
        "active_constraint_index": int(active[1]),
        "raw_protected_contact_count": int(raw_contacts),
        "maximum_per_step_obstacle_l1_displacement_m": maximum_motion,
        "verified_safe": safe,
        "chunk": np.asarray(candidate["chunk"], dtype=np.float64).tolist(),
        "steps": [
            {
                "chunk_step": index,
                "minimum_substep_clearance_m": margins[index].tolist(),
                "minimum_distal_substep_clearance_m": float(np.min(margins[index])),
                "raw_protected_contact_count": int(item["raw_protected_contact_count"]),
                "maximum_within_step_obstacle_l1_displacement_m": float(
                    item["maximum_within_step_obstacle_l1_displacement_m"]
                ),
                "next_state_sha256": str(item["next_state_sha256"]),
            }
            for index, item in enumerate(transitions)
        ],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_chunk_oracle_config(args.config.resolve())
    settings = config["source"]
    _require(
        _file_sha256(args.population_manifest.resolve())
        == settings["table1_population_manifest_sha256"],
        "chunk-oracle population manifest differs",
    )
    _require(
        _file_sha256(args.prior_result.resolve())
        == settings["prior_full_bound_result_sha256"]
        and _file_sha256(args.prior_validation.resolve())
        == settings["prior_full_bound_validation_sha256"],
        "prior full-bound receipt differs",
    )
    prior = _load(args.prior_result.resolve())
    prior_validation = _load(args.prior_validation.resolve())
    _require(
        prior.get("result_payload_sha256")
        == settings["prior_full_bound_result_payload_sha256"]
        and prior["decision"]["physical_full_bound_recovery_exists"] is False
        and prior["decision"]["multi_step_required_before_larger_region_collection"] is True
        and prior_validation["larger_region_collection_authorized"] is False,
        "prior full-bound NO-GO trigger differs",
    )
    _require(
        _file_sha256(args.geometry_config.resolve())
        == config["geometry"]["distal_geometry_config_sha256"]
        and _file_sha256(args.exact_box_config.resolve())
        == config["geometry"]["exact_box_config_sha256"],
        "chunk-oracle geometry inputs differ",
    )
    archived_path = args.archived_root.resolve() / settings["archived_relative_path"]
    archived = _load(archived_path)
    _require(
        archived_path.is_file() and not archived_path.is_symlink()
        and _file_sha256(archived_path) == settings["archived_file_sha256"]
        and archived.get("result_payload_sha256") == settings["archived_payload_sha256"]
        and len(archived.get("actions", [])) == settings["action_count"]
        and archived.get("action_invariance_ledger", {}).get("executed_sequence_sha256")
        == settings["executed_sequence_sha256"],
        "chunk-oracle E05 archive differs",
    )
    population = {
        item["case_id"]: item
        for item in read_jsonl(args.population_manifest.resolve())
    }
    case = population[config["case_id"]]
    validate_case_row(case, args.repo_root.resolve())
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(args.geometry_config.resolve())
    exact_box_config = load_obstacle_primitive_config(args.exact_box_config.resolve())
    env = probe_env = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        _require(setup["obstacle_name"] == "moka_pot_obstacle_1", "chunk-oracle obstacle differs")
        pairing = pairing_record(
            case=case,
            selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=setup["obstacle_name"],
            settled_simulator_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
        )
        for key in (
            "manifest_row_sha256", "initial_state_sha256",
            "initial_observation_sha256", "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256", "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == archived["pairing"][key], "chunk-oracle pairing differs: %s" % key)
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config,
            exact_box_config=exact_box_config,
            archived=archived,
            env=env,
            obstacle_name=setup["obstacle_name"],
        )
        probe = SubstepEightConstraintProbe(
            probe_env, geometry,
            active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6,
            contact_distance_threshold_m=float(
                config["verification"]["raw_contact_distance_threshold_m"]
            ),
            obstacle_primitive_union=exact_boxes,
        )
        start = int(settings["start_action_step"])
        for index in range(start):
            env.step(_canonical_action(archived["actions"][index], index).tolist())
        nominal_by_horizon = {
            horizon: np.asarray([
                _canonical_action(archived["actions"][start + offset], start + offset)
                for offset in range(horizon)
            ], dtype=np.float64)
            for horizon in (2, 5)
        }
        baselines = {}
        for horizon, nominal_chunk in nominal_by_horizon.items():
            candidate = {
                "source": "immutable_aegis_chunk",
                "parameter": nominal_chunk[0, :3],
                "chunk": nominal_chunk,
                "correction": np.zeros((horizon, 3), dtype=np.float64),
            }
            baselines[str(horizon)] = _summarize_rollout(
                probe.rollout_chunk(env, nominal_chunk.tolist()),
                nominal_chunk, candidate, config, -1,
            )
        _require(
            all(not item["verified_safe"] for item in baselines.values()),
            "immutable chunk does not reproduce the proxy crossing",
        )

        arms = {}
        globally_best = None
        for arm in config["arms"]:
            horizon = int(arm["horizon"])
            nominal_chunk = nominal_by_horizon[horizon]
            candidates = (
                one_shot_chunks(nominal_chunk, config)
                if arm["family"] == "one_shot"
                else distributed_chunks(nominal_chunk, config)
            )
            records = []
            best = None
            arm_started = time.perf_counter_ns()
            for candidate_index, candidate in enumerate(candidates):
                record = _summarize_rollout(
                    probe.rollout_chunk(env, candidate["chunk"].tolist()),
                    nominal_chunk, candidate, config, candidate_index,
                )
                records.append(record)
                if record["verified_safe"] and (
                    best is None
                    or (
                        record["objective_chunk_correction_l2"], candidate_index
                    ) < (
                        best["objective_chunk_correction_l2"], best["candidate_index"]
                    )
                ):
                    best = record
            arms[arm["arm_id"]] = {
                "horizon": horizon,
                "family": arm["family"],
                "candidate_count": len(records),
                "verified_safe_candidate_count": sum(
                    item["verified_safe"] for item in records
                ),
                "smallest_verified_safe_chunk": best,
                "best_minimum_clearance_candidate": max(
                    records, key=lambda item: item["minimum_distal_substep_clearance_m"]
                ),
                "candidates": records,
                "wall_seconds": (time.perf_counter_ns() - arm_started) * 1.0e-9,
            }
            if best is not None:
                candidate_best = {**best, "arm_id": arm["arm_id"]}
                if globally_best is None or (
                    candidate_best["objective_chunk_correction_l2"],
                    candidate_best["arm_id"],
                ) < (
                    globally_best["objective_chunk_correction_l2"],
                    globally_best["arm_id"],
                ):
                    globally_best = candidate_best

        execution = None
        execution_pass = False
        if globally_best is not None:
            clone_hashes = []
            executed_hashes = []
            per_step_matches = []
            for action in globally_best["chunk"]:
                transition = probe.transition(env, action)
                clone_hashes.append(transition["next_state_sha256"])
                env.step(action)
                executed_hash = hashlib.sha256(
                    np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
                ).hexdigest()
                executed_hashes.append(executed_hash)
                per_step_matches.append(executed_hash == transition["next_state_sha256"])
            execution_pass = bool(all(per_step_matches))
            execution = {
                "arm_id": globally_best["arm_id"],
                "candidate_index": globally_best["candidate_index"],
                "clone_next_state_sha256": clone_hashes,
                "executed_next_state_sha256": executed_hashes,
                "per_step_clone_execution_match": per_step_matches,
                "all_steps_match": execution_pass,
            }

        any_safe = bool(globally_best is not None)
        decision = {
            "verified_safe_chunk_exists": any_safe,
            "passing_arm_ids": sorted(
                key for key, value in arms.items()
                if value["verified_safe_candidate_count"] > 0
            ),
            "execution_fidelity_pass": execution_pass,
            "larger_chunk_region_collection_authorized": bool(any_safe and execution_pass),
            "neural_training_authorized": False,
            "closed_loop_e05_authorized": False,
            "registered_chunk_families_exhausted_without_recovery": bool(not any_safe),
        }
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "table1": {
                "path": str(archived_path),
                "file_sha256": _file_sha256(archived_path),
                "payload_sha256": archived["result_payload_sha256"],
                "modified": False,
            },
            "prior_full_bound_gate": {
                "job_id": settings["prior_full_bound_job_id"],
                "result_file_sha256": _file_sha256(args.prior_result.resolve()),
                "validation_file_sha256": _file_sha256(args.prior_validation.resolve()),
                "larger_region_collection_authorized": False,
            },
            "pairing": pairing,
            "start_action_step": start,
            "baselines": baselines,
            "arms": arms,
            "selected_safe_chunk": globally_best,
            "executed_selected_chunk": execution,
            "decision": decision,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
        return result
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--prior-validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
