#!/usr/bin/env python3
"""Audit ellipsoid-safe candidate support before training an E05 MLP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _source_action
from scripts.evaluate_distal_contact_ranker_e05 import (
    _matched_source_next_hash, _successful_query_action,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID, _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


SCHEMA = "vlsa_distal_ellipsoid_candidate_support_e05.v1"
RESULT_SCHEMA = "vlsa_distal_ellipsoid_candidate_support_e05_result.v1"


def _hash_without(value, key):
    payload = dict(value); payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(config.get("schema_version") == SCHEMA, "ellipsoid-support config schema differs")
    _require(config.get("case_id") == CASE_ID, "ellipsoid-support case differs")
    _require(config.get("collect_steps") == list(range(184, 193)), "ellipsoid-support states differ")
    _require(config.get("test_steps") == [187, 190], "ellipsoid-support test states differ")
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--successful-sitl", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); started = time.perf_counter_ns()

    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.oracle_affine import (
        SubstepEightConstraintProbe, oracle_candidate_xyz,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    config = _load_config(args.config.resolve()); ids = config["immutable_sources"]
    _require(_file_sha256(args.archived.resolve()) == ids["archived_table1_file_sha256"], "Table-1 source differs")
    _require(_file_sha256(args.successful_sitl.resolve()) == ids["successful_sitl_file_sha256"], "successful SITL source differs")
    _require(_file_sha256(args.geometry_config.resolve()) == ids["geometry_config_file_sha256"], "geometry source differs")
    successful = _load(args.successful_sitl.resolve()); archived = _load(args.archived.resolve())
    _require(successful.get("result_payload_sha256") == ids["successful_sitl_payload_sha256"], "successful SITL payload differs")
    rows = [row for row in read_jsonl(args.manifest.resolve()) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "E05 manifest row is not unique"); case = rows[0]
    validate_case_row(case, args.repo_root.resolve())
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    allocation = allocation_record(); runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(args.geometry_config.resolve())
    env = probe_env = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        pairing = pairing_record(
            case=case, selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation, task_description=str(task.language),
            active_obstacle_name=setup["obstacle_name"],
            settled_simulator_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
        )
        for key in (
            "manifest_row_sha256", "initial_state_sha256", "initial_observation_sha256",
            "settled_simulator_state_sha256", "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == successful["pairing"][key], "pairing differs: %s" % key)
        from main.multilink_ellipsoid.shadow import MultilinkEllipsoidShadow
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {
                "p2": perception["mvee_center"],
                "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        _require(geometry.geometry_record(env)["distal_ellipsoid_count"] == 7, "distal geometry count differs")
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=None,
        )
        actions = successful["actions"]
        for step in range(config["collect_steps"][0]):
            env.step(_source_action(actions[step], step).tolist())
        state_results = []; records = []
        candidate_config = {"candidate_set": config["candidate_set"]}
        for step in config["collect_steps"]:
            query = _successful_query_action(actions[step], step)
            executed = _source_action(actions[step], step)
            candidates = oracle_candidate_xyz(query[:3], candidate_config)
            if not any(np.array_equal(item["xyz"], executed[:3]) for item in candidates):
                candidates.append({"source": "successful_oracle_action", "xyz": executed[:3].copy()})
            else:
                for item in candidates:
                    if np.array_equal(item["xyz"], executed[:3]): item["source"] = "successful_oracle_action"
            local = []
            for candidate_index, candidate in enumerate(candidates):
                action = query.copy(); action[:3] = candidate["xyz"]
                transition = probe.transition(env, action)
                margins = np.asarray(transition["minimum_substep_clearance_m"][:7], dtype=np.float64)
                record = {
                    "record_index": len(records), "state_step": int(step),
                    "candidate_index": int(candidate_index), "candidate_source": str(candidate["source"]),
                    "nominal_xyz": query[:3].tolist(), "candidate_xyz": np.asarray(candidate["xyz"]).tolist(),
                    "minimum_substep_ellipsoid_margin_m": margins.tolist(),
                    "minimum_all_constraint_margin_m": float(np.min(margins)),
                    "D_opt_all_seven_safe": bool(np.all(margins >= 0.0)),
                    "D_sim_raw_contact_count_diagnostic": int(transition["raw_protected_contact_count"]),
                    "next_state_sha256": str(transition["next_state_sha256"]),
                    "env_step_wall_seconds": float(transition["env_step_wall_seconds"]),
                }
                records.append(record); local.append(record)
            oracle = [item for item in local if item["candidate_source"] == "successful_oracle_action"]
            _require(len(oracle) == 1, "state lacks one successful-oracle action")
            safe = [item for item in local if item["D_opt_all_seven_safe"]]
            env.step(executed.tolist())
            actual_hash = _sha256(np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes())
            _require(actual_hash == _matched_source_next_hash(actions[step], executed[:3]) == oracle[0]["next_state_sha256"], "successful transition differs")
            state_results.append({
                "state_step": int(step), "candidate_count": len(local),
                "ellipsoid_safe_candidate_count": len(safe),
                "best_minimum_ellipsoid_margin_m": max(item["minimum_all_constraint_margin_m"] for item in local),
                "successful_oracle_ellipsoid_safe": oracle[0]["D_opt_all_seven_safe"],
                "successful_oracle_minimum_ellipsoid_margin_m": oracle[0]["minimum_all_constraint_margin_m"],
                "successful_oracle_raw_contact_count_diagnostic": oracle[0]["D_sim_raw_contact_count_diagnostic"],
            })
        test = [item for item in state_results if item["state_step"] in config["test_steps"]]
        support = bool(len(test) == 2 and all(item["ellipsoid_safe_candidate_count"] > 0 for item in test))
        result = {
            "schema_version": RESULT_SCHEMA, "status": "complete", "scientific_result": True,
            "claim_scope": config["claim_scope"], "source": source, "allocation": allocation,
            "config": config, "pairing": pairing, "state_results": state_results,
            "records": records,
            "D_opt": "seven_robot_ellipsoids_vs_released_AEGIS_obstacle_MVEE",
            "D_sim": "raw_contact_count_diagnostic_only",
            "decision": {
                "ellipsoid_candidate_support_pass": support,
                "ellipsoid_margin_mlp_training_authorized": support,
                "stop_reason": None if support else "one_or_more_primary_test_states_have_empty_all-seven-ellipsoid-safe_candidate_set",
            },
            "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
            "total_clone_env_step_wall_seconds": sum(item["env_step_wall_seconds"] for item in records),
        }
        result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
        _atomic_write(args.output.resolve(), result)
        print(json.dumps(result["decision"], sort_keys=True))
    finally:
        if probe_env is not None: probe_env.close()
        if env is not None: env.close()
    return 0


if __name__ == "__main__": raise SystemExit(main())
