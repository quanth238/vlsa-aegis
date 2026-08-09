#!/usr/bin/env python3
"""Paired random-direction control for the learned E05 contact-risk gradient."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.gradient_random_control import (
    empirical_equal_or_earlier_p,
    first_safe_radius,
    sample_feasible_unit_directions,
)
from scripts.collect_distal_boundary_generalization_moka10 import (
    _restore_env, _snapshot_env,
)
from scripts.evaluate_distal_contact_ranker_e05 import (
    _build_model, _predict_risk_and_gradient, _successful_query_action,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _source_action
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID, _atomic_write, _file_sha256, _git_identity, _load, _require,
)


SCHEMA = "vlsa_distal_gradient_random_control_e05.v1"
RESULT_SCHEMA = "vlsa_distal_gradient_random_control_e05_result.v1"


def _hash_without(value, key):
    payload = dict(value); payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(config.get("schema_version") == SCHEMA, "random-control config schema differs")
    _require(config.get("case_id") == CASE_ID, "random-control case differs")
    _require(config.get("test_steps") == [187, 190], "random-control states differ")
    _require(config.get("correction_radii_action") == [0.1, 0.25, 0.5, 1.0], "random-control radii differ")
    return config


def _load_model(path):
    import torch

    artifact = torch.load(path, map_location="cpu")
    _require(artifact.get("schema_version") == "vlsa_distal_contact_ranker_e05_model.v1", "model schema differs")
    model = _build_model(torch, 33, artifact["hidden_widths"]).to(dtype=torch.float64)
    model.load_state_dict(artifact["model_state_dict"]); model.eval()
    return model, artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--successful-sitl", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--contact-dataset", type=Path, required=True)
    parser.add_argument("--contact-model", type=Path, required=True)
    parser.add_argument("--contact-result", type=Path, required=True)
    parser.add_argument("--contact-validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); started = time.perf_counter_ns()

    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.execution_margin_nn import feature_vector
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow, allocation_record, load_shadow_config,
    )

    config = load_config(args.config.resolve()); identities = config["immutable_sources"]
    sources = {
        "archived_table1": args.archived.resolve(),
        "successful_sitl": args.successful_sitl.resolve(),
        "geometry_config": args.geometry_config.resolve(),
        "contact_dataset": args.contact_dataset.resolve(),
        "contact_model": args.contact_model.resolve(),
        "contact_result": args.contact_result.resolve(),
        "contact_validation": args.contact_validation.resolve(),
    }
    for name, path in sources.items():
        _require(path.is_file() and not path.is_symlink(), "random-control source missing: %s" % name)
    expected_files = {
        "archived_table1": identities["archived_table1_file_sha256"],
        "successful_sitl": identities["successful_sitl_file_sha256"],
        "geometry_config": identities["geometry_config_file_sha256"],
        "contact_dataset": identities["contact_dataset_file_sha256"],
        "contact_model": identities["contact_model_file_sha256"],
        "contact_result": identities["contact_result_file_sha256"],
        "contact_validation": identities["contact_validation_file_sha256"],
    }
    for name, expected in expected_files.items():
        _require(_file_sha256(sources[name]) == expected, "random-control source hash differs: %s" % name)
    archived = _load(sources["archived_table1"]); successful = _load(sources["successful_sitl"])
    dataset = _load(sources["contact_dataset"]); prior = _load(sources["contact_result"])
    validation = _load(sources["contact_validation"])
    _require(successful.get("result_payload_sha256") == identities["successful_sitl_payload_sha256"], "successful SITL payload differs")
    _require(dataset.get("dataset_payload_sha256") == identities["contact_dataset_payload_sha256"], "contact dataset payload differs")
    _require(prior.get("result_payload_sha256") == identities["contact_result_payload_sha256"], "contact result payload differs")
    _require(dataset.get("source", {}).get("commit") == identities["contact_source_commit"], "contact dataset commit differs")
    _require(prior.get("source", {}).get("commit") == identities["contact_source_commit"], "contact result commit differs")
    _require(prior.get("decision", {}).get("gradient_steering_go") is True and prior.get("decision", {}).get("ranking_feasibility_go") is False, "prior decision differs")
    _require(validation.get("status") == "passed" and validation.get("gradient_steering_go") is True, "prior validation differs")
    model, artifact = _load_model(sources["contact_model"])
    rows = [row for row in read_jsonl(args.manifest.resolve()) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "E05 manifest row is not unique"); case = rows[0]
    validate_case_row(case, args.repo_root.resolve())
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    allocation = allocation_record(); runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(sources["geometry_config"])
    env = probe_env = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        pairing = pairing_record(
            case=case, selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation, task_description=str(task.language),
            active_obstacle_name=setup["obstacle_name"],
            settled_simulator_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
        )
        _require(pairing == dataset["pairing"], "random-control pairing differs")
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {
                "p2": perception["mvee_center"], "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=None,
        )
        actions = successful["actions"]; snapshots = {}
        for step in range(max(config["test_steps"]) + 1):
            if step in config["test_steps"]: snapshots[step] = _snapshot_env(env)
            env.step(_source_action(actions[step], step).tolist())
        radii = [float(value) for value in config["correction_radii_action"]]
        random_settings = config["random_control"]
        learned_records = []; random_records = []; state_results = []
        for step in config["test_steps"]:
            _restore_env(env, snapshots[step]); query = _successful_query_action(actions[step], step)
            nominal_rows = [item for item in dataset["records"] if item["state_step"] == step and item["candidate_source"] == "nominal"]
            reference_rows = [item for item in dataset["records"] if item["state_step"] == step and item["candidate_source"] == "successful_oracle_action"]
            _require(len(nominal_rows) == 1 and len(reference_rows) == 1, "random-control dataset state rows differ")
            nominal_transition = probe.transition(env, query)
            recomputed_feature = feature_vector(
                nominal_transition["substeps"][0], query[:3], query[:3]
            )
            stored_feature = np.asarray(nominal_rows[0]["feature_vector"], dtype=np.float64)
            feature_error = float(np.max(np.abs(recomputed_feature - stored_feature)))
            _require(feature_error <= 1e-10, "random-control nominal feature exceeds clone tolerance")
            risk, gradient = _predict_risk_and_gradient(
                model, stored_feature, artifact["feature_mean"],
                artifact["feature_standard_deviation"],
            )
            prior_gradient = next(item for item in prior["gradient_audit"] if item["state_step"] == step)
            _require(abs(risk - float(prior_gradient["nominal_risk"])) <= 1e-12, "random-control nominal risk differs")
            _require(np.allclose(gradient, prior_gradient["risk_gradient"], rtol=0.0, atol=1e-12), "random-control gradient differs")
            norm = float(np.linalg.norm(gradient)); _require(norm > 1e-12, "random-control learned gradient is zero")
            learned_direction = -gradient / norm
            _require(np.all(np.abs(query[:3] + max(radii) * learned_direction) <= 1.0 + 1e-12), "learned direction violates full-radius bounds")
            reference_eef = np.asarray(reference_rows[0]["next_eef_position_m"], dtype=np.float64)

            def evaluate(direction, radius, arm, direction_index):
                xyz = query[:3] + float(radius) * np.asarray(direction, dtype=np.float64)
                _require(np.all(np.abs(xyz) <= 1.0 + 1e-12), "random-control action bound differs")
                _require(abs(float(np.linalg.norm(xyz - query[:3])) - float(radius)) <= 1e-10, "random-control correction norm differs")
                action = query.copy(); action[:3] = xyz
                transition = probe.transition(env, action)
                return {
                    "state_step": int(step), "arm": arm,
                    "direction_index": direction_index, "radius_action": float(radius),
                    "direction": np.asarray(direction, dtype=np.float64).tolist(),
                    "candidate_xyz": xyz.tolist(),
                    "D_sim_raw_safe": bool(transition["raw_protected_contact_count"] == 0),
                    "raw_protected_contact_count": int(transition["raw_protected_contact_count"]),
                    "maximum_within_step_obstacle_l1_displacement_m": float(transition["maximum_within_step_obstacle_l1_displacement_m"]),
                    "next_eef_reference_error_m": float(np.linalg.norm(np.asarray(transition["next_eef_position_m"]) - reference_eef)),
                    "next_state_sha256": transition["next_state_sha256"],
                    "env_step_wall_seconds": float(transition["env_step_wall_seconds"]),
                }

            state_learned = [evaluate(learned_direction, radius, "learned_negative_gradient", None) for radius in radii]
            learned_records.extend(state_learned)
            directions = sample_feasible_unit_directions(
                query[:3], radius=max(radii),
                count=int(random_settings["direction_count_per_state"]),
                seed=int(random_settings["seeds_by_state"][str(step)]),
                action_limit=float(random_settings["action_limit"]),
                maximum_attempts=int(random_settings["maximum_sampling_attempts"]),
            )
            state_random = []
            for direction_index, direction in enumerate(directions):
                for radius in radii:
                    state_random.append(evaluate(direction, radius, "matched_random", direction_index))
            random_records.extend(state_random)
            learned_first = first_safe_radius(radii, [item["D_sim_raw_safe"] for item in state_learned])
            random_first = []
            for direction_index in range(len(directions)):
                selected = [item for item in state_random if item["direction_index"] == direction_index]
                random_first.append(first_safe_radius(radii, [item["D_sim_raw_safe"] for item in selected]))
            empirical_p = empirical_equal_or_earlier_p(learned_first, random_first)
            by_radius = []
            for radius in radii:
                selected = [item for item in state_random if item["radius_action"] == radius]
                safe_errors = [item["next_eef_reference_error_m"] for item in selected if item["D_sim_raw_safe"]]
                by_radius.append({
                    "radius_action": radius, "random_count": len(selected),
                    "random_safe_count": sum(item["D_sim_raw_safe"] for item in selected),
                    "random_safe_rate": sum(item["D_sim_raw_safe"] for item in selected) / len(selected),
                    "safe_reference_error_mean_m": None if not safe_errors else float(np.mean(safe_errors)),
                })
            threshold = float(config["randomization_gate"]["empirical_equal_or_earlier_safe_p_maximum"])
            state_results.append({
                "state_step": int(step), "nominal_risk": risk,
                "maximum_recomputed_feature_error": feature_error,
                "learned_gradient": gradient.tolist(),
                "learned_first_safe_radius_action": learned_first,
                "random_first_safe_radius_counts": {
                    "none": sum(value is None for value in random_first),
                    **{str(radius): sum(value == radius for value in random_first) for radius in radii},
                },
                "random_equal_or_earlier_count": sum(value is not None and learned_first is not None and value <= learned_first + 1e-15 for value in random_first),
                "empirical_equal_or_earlier_safe_p": empirical_p,
                "random_safe_by_radius": by_radius,
                "directional_advantage_pass": bool(learned_first is not None and empirical_p <= threshold),
            })
        advantage = bool(all(item["directional_advantage_pass"] for item in state_results))
        result = {
            "schema_version": RESULT_SCHEMA, "status": "complete", "scientific_result": True,
            "claim_scope": config["claim_scope"], "source": source,
            "allocation": allocation, "config": config, "pairing": pairing,
            "immutable_contact_artifacts": {
                name: {"path": str(path), "file_sha256": expected_files[name]}
                for name, path in sources.items() if name.startswith("contact_")
            },
            "state_results": state_results, "learned_records": learned_records,
            "random_records": random_records,
            "decision": {
                "learned_gradient_beats_matched_random": advantage,
                "boundary_focused_multi_state_refinement_authorized": advantage,
                "supervision_or_model_rethink_required": not advantage,
                "closed_loop_authorized": False,
                "stop_reason": None if advantage else "learned_gradient_did_not_pass_both_matched_randomization_gates",
            },
            "total_fresh_rollout_count": len(learned_records) + len(random_records),
            "total_clone_env_step_wall_seconds": float(sum(item["env_step_wall_seconds"] for item in learned_records + random_records)),
            "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
        }
        result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
        _atomic_write(args.output.resolve(), result)
        print(json.dumps(result["decision"], sort_keys=True))
    finally:
        if probe_env is not None: probe_env.close()
        if env is not None: env.close()
    return 0


if __name__ == "__main__": raise SystemExit(main())
