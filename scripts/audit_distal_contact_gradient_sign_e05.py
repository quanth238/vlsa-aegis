#!/usr/bin/env python3
"""Audit contact-risk gradient sign, XYZ indexing, and cloned-OSC alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.contact_gradient_sign_audit import (
    compare_contact_burden,
    contact_burden,
    paired_gradient_candidates,
)
from scripts.collect_distal_boundary_generalization_moka10 import (
    _restore_env,
    _snapshot_env,
)
from scripts.evaluate_distal_contact_ranker_e05 import (
    _build_model,
    _successful_query_action,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _source_action
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


SCHEMA = "vlsa_distal_contact_gradient_sign_audit_e05.v1"
RESULT_SCHEMA = "vlsa_distal_contact_gradient_sign_audit_e05_result.v1"


def _hash_without(value, key):
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(config.get("schema_version") == SCHEMA, "gradient-sign config schema differs")
    _require(config.get("case_id") == CASE_ID, "gradient-sign case differs")
    _require(config.get("test_steps") == [187, 190], "gradient-sign states differ")
    _require(
        config.get("epsilon_action") == [0.0001, 0.001, 0.01, 0.05, 0.1],
        "gradient-sign epsilons differ",
    )
    _require(config.get("action_dimensions") == [0, 1, 2], "gradient-sign dimensions differ")
    return config


def _load_model(path):
    import torch

    artifact = torch.load(path, map_location="cpu")
    _require(
        artifact.get("schema_version") == "vlsa_distal_contact_ranker_e05_model.v1",
        "gradient-sign model schema differs",
    )
    model = _build_model(torch, 33, artifact["hidden_widths"]).to(dtype=torch.float64)
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    return model, artifact


def _predict(model, feature, mean, deviation):
    import numpy as np
    import torch

    device = next(model.parameters()).device
    raw = torch.as_tensor(feature, dtype=torch.float64, device=device).requires_grad_(True)
    normalized = (
        raw - torch.as_tensor(mean, dtype=torch.float64, device=device)
    ) / torch.as_tensor(deviation, dtype=torch.float64, device=device)
    probabilities = torch.sigmoid(model(normalized[None, :]))[0]
    risk = torch.max(probabilities)
    gradient = torch.autograd.grad(risk, raw)[0][-3:]
    return {
        "risk": float(risk.detach().cpu()),
        "link_probabilities": probabilities.detach().cpu().numpy().astype(np.float64),
        "gradient_xyz": gradient.detach().cpu().numpy().astype(np.float64),
        "active_link_index": int(torch.argmax(probabilities).detach().cpu()),
    }


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
    parser.add_argument("--random-control-result", type=Path, required=True)
    parser.add_argument("--random-control-validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()

    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.execution_margin_nn import FEATURE_NAMES, feature_vector
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )

    config = load_config(args.config.resolve())
    identities = config["immutable_sources"]
    sources = {
        "archived_table1": args.archived.resolve(),
        "successful_sitl": args.successful_sitl.resolve(),
        "geometry_config": args.geometry_config.resolve(),
        "contact_dataset": args.contact_dataset.resolve(),
        "contact_model": args.contact_model.resolve(),
        "contact_result": args.contact_result.resolve(),
        "contact_validation": args.contact_validation.resolve(),
        "random_control_result": args.random_control_result.resolve(),
        "random_control_validation": args.random_control_validation.resolve(),
    }
    expected_files = {
        "archived_table1": identities["archived_table1_file_sha256"],
        "successful_sitl": identities["successful_sitl_file_sha256"],
        "geometry_config": identities["geometry_config_file_sha256"],
        "contact_dataset": identities["contact_dataset_file_sha256"],
        "contact_model": identities["contact_model_file_sha256"],
        "contact_result": identities["contact_result_file_sha256"],
        "contact_validation": identities["contact_validation_file_sha256"],
        "random_control_result": identities["random_control_result_file_sha256"],
        "random_control_validation": identities["random_control_validation_file_sha256"],
    }
    for name, path in sources.items():
        _require(path.is_file() and not path.is_symlink(), "gradient-sign source missing: %s" % name)
        _require(_file_sha256(path) == expected_files[name], "gradient-sign source hash differs: %s" % name)

    archived = _load(sources["archived_table1"])
    successful = _load(sources["successful_sitl"])
    dataset = _load(sources["contact_dataset"])
    prior = _load(sources["contact_result"])
    prior_validation = _load(sources["contact_validation"])
    random_result = _load(sources["random_control_result"])
    random_validation = _load(sources["random_control_validation"])
    _require(successful.get("result_payload_sha256") == identities["successful_sitl_payload_sha256"], "successful SITL payload differs")
    _require(dataset.get("dataset_payload_sha256") == identities["contact_dataset_payload_sha256"], "contact dataset payload differs")
    _require(prior.get("result_payload_sha256") == identities["contact_result_payload_sha256"], "contact result payload differs")
    _require(random_result.get("result_payload_sha256") == identities["random_control_result_payload_sha256"], "random-control payload differs")
    _require(prior.get("source", {}).get("commit") == identities["contact_source_commit"], "contact source commit differs")
    _require(prior_validation.get("status") == "passed", "contact validation differs")
    _require(random_validation.get("status") == "passed", "random-control validation differs")
    _require(random_result.get("decision", {}).get("learned_gradient_beats_matched_random") is False, "random-control decision differs")
    _require(random_result.get("decision", {}).get("closed_loop_authorized") is False, "random-control closed-loop decision differs")
    _require(
        list(FEATURE_NAMES[-3:]) == config["model_consistency_gate"]["required_feature_suffix"],
        "candidate XYZ feature suffix differs",
    )
    model, artifact = _load_model(sources["contact_model"])
    rows = [row for row in read_jsonl(args.manifest.resolve()) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "E05 manifest row is not unique")
    case = rows[0]
    validate_case_row(case, args.repo_root.resolve())
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(sources["geometry_config"])

    env = probe_env = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        pairing = pairing_record(
            case=case,
            selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=setup["obstacle_name"],
            settled_simulator_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
        )
        _require(pairing == dataset["pairing"], "gradient-sign pairing differs")
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
        probe = SubstepEightConstraintProbe(
            probe_env,
            geometry,
            active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1e-6,
            contact_distance_threshold_m=0.0,
            obstacle_primitive_union=None,
        )
        actions = successful["actions"]
        snapshots = {}
        for step in range(max(config["test_steps"]) + 1):
            if step in config["test_steps"]:
                snapshots[step] = _snapshot_env(env)
            env.step(_source_action(actions[step], step).tolist())

        records = []
        state_results = []
        epsilons = [float(value) for value in config["epsilon_action"]]
        risk_tolerance = float(config["model_consistency_gate"]["strict_risk_ordering_tolerance"])
        relative_limit = float(config["model_consistency_gate"]["smallest_epsilon_central_difference_relative_error_maximum"])
        penetration_tolerance = float(config["simulator_ordering"]["penetration_tie_tolerance_m"])
        for step in config["test_steps"]:
            _restore_env(env, snapshots[step])
            query = _successful_query_action(actions[step], step)
            nominal_rows = [
                item for item in dataset["records"]
                if item["state_step"] == step and item["candidate_source"] == "nominal"
            ]
            _require(len(nominal_rows) == 1, "gradient-sign nominal record differs")
            nominal_transition = probe.transition(env, query)
            recomputed = feature_vector(nominal_transition["substeps"][0], query[:3], query[:3])
            stored = np.asarray(nominal_rows[0]["feature_vector"], dtype=np.float64)
            feature_error = float(np.max(np.abs(recomputed - stored)))
            _require(feature_error <= 1e-10, "gradient-sign nominal feature exceeds clone tolerance")
            nominal_prediction = _predict(
                model, stored, artifact["feature_mean"], artifact["feature_standard_deviation"]
            )
            prior_gradient = next(item for item in prior["gradient_audit"] if item["state_step"] == step)
            _require(abs(nominal_prediction["risk"] - float(prior_gradient["nominal_risk"])) <= 1e-12, "gradient-sign nominal risk differs")
            _require(np.allclose(nominal_prediction["gradient_xyz"], prior_gradient["risk_gradient"], rtol=0.0, atol=1e-12), "gradient-sign nominal gradient differs")
            gradient = nominal_prediction["gradient_xyz"]
            gradient_norm = float(np.linalg.norm(gradient))
            state_records = []
            for epsilon in epsilons:
                minus_xyz, plus_xyz, direction = paired_gradient_candidates(
                    query[:3], gradient, epsilon, float(config["action_limit"])
                )
                pair = {}
                for sign, xyz in (("minus", minus_xyz), ("plus", plus_xyz)):
                    candidate_feature = stored.copy()
                    candidate_feature[-3:] = xyz
                    indexing_pass = bool(
                        np.array_equal(candidate_feature[:-3], stored[:-3])
                        and np.array_equal(candidate_feature[-3:], xyz)
                    )
                    prediction = _predict(
                        model,
                        candidate_feature,
                        artifact["feature_mean"],
                        artifact["feature_standard_deviation"],
                    )
                    action = query.copy()
                    action[:3] = xyz
                    action_tail_pass = bool(np.array_equal(action[3:], query[3:]))
                    transition = probe.transition(env, action)
                    pair[sign] = {
                        "candidate_xyz": xyz.tolist(),
                        "full_action": action.tolist(),
                        "risk": prediction["risk"],
                        "link_probabilities": prediction["link_probabilities"].tolist(),
                        "active_link_index": prediction["active_link_index"],
                        "feature_indexing_pass": indexing_pass,
                        "action_tail_bitwise_pass": action_tail_pass,
                        "contact_burden": contact_burden(transition),
                        "minimum_distal_substep_clearance_m": float(transition["minimum_distal_substep_clearance_m"]),
                        "maximum_within_step_obstacle_l1_displacement_m": float(transition["maximum_within_step_obstacle_l1_displacement_m"]),
                        "next_state_sha256": transition["next_state_sha256"],
                        "env_step_wall_seconds": float(transition["env_step_wall_seconds"]),
                    }
                finite_difference = (pair["plus"]["risk"] - pair["minus"]["risk"]) / (2.0 * epsilon)
                relative_error = abs(finite_difference - gradient_norm) / max(gradient_norm, 1e-15)
                model_ordering_pass = bool(pair["minus"]["risk"] + risk_tolerance < pair["plus"]["risk"])
                indexing_pass = bool(all(
                    pair[sign]["feature_indexing_pass"] and pair[sign]["action_tail_bitwise_pass"]
                    for sign in ("minus", "plus")
                ))
                simulator_preference = compare_contact_burden(
                    pair["minus"]["contact_burden"],
                    pair["plus"]["contact_burden"],
                    penetration_tolerance,
                )
                record = {
                    "state_step": int(step),
                    "epsilon_action": epsilon,
                    "nominal_xyz": query[:3].tolist(),
                    "normalized_gradient_direction": direction.tolist(),
                    "gradient_norm": gradient_norm,
                    "finite_difference_directional_derivative": float(finite_difference),
                    "finite_difference_relative_error": float(relative_error),
                    "model_ordering_pass": model_ordering_pass,
                    "indexing_pass": indexing_pass,
                    "simulator_preference": simulator_preference,
                    "minus": pair["minus"],
                    "plus": pair["plus"],
                }
                records.append(record)
                state_records.append(record)
            smallest = state_records[0]
            finite_difference_pass = bool(
                smallest["finite_difference_relative_error"] <= relative_limit
            )
            state_results.append({
                "state_step": int(step),
                "nominal_risk": nominal_prediction["risk"],
                "nominal_link_probabilities": nominal_prediction["link_probabilities"].tolist(),
                "nominal_active_link_index": nominal_prediction["active_link_index"],
                "gradient_xyz": gradient.tolist(),
                "gradient_norm": gradient_norm,
                "maximum_recomputed_feature_error": feature_error,
                "all_model_orderings_pass": bool(all(item["model_ordering_pass"] for item in state_records)),
                "all_indexing_checks_pass": bool(all(item["indexing_pass"] for item in state_records)),
                "smallest_epsilon_finite_difference_pass": finite_difference_pass,
                "simulator_minus_preference_count": sum(item["simulator_preference"] == "minus" for item in state_records),
                "simulator_plus_preference_count": sum(item["simulator_preference"] == "plus" for item in state_records),
                "simulator_tie_count": sum(item["simulator_preference"] == "tie" for item in state_records),
            })

        implementation_pass = bool(all(
            item["all_model_orderings_pass"]
            and item["all_indexing_checks_pass"]
            and item["smallest_epsilon_finite_difference_pass"]
            for item in state_results
        ))
        simulator_plus_count = sum(item["simulator_plus_preference_count"] for item in state_results)
        simulator_minus_count = sum(item["simulator_minus_preference_count"] for item in state_results)
        simulator_tie_count = sum(item["simulator_tie_count"] for item in state_results)
        misalignment = bool(implementation_pass and simulator_plus_count > 0)
        if not implementation_pass:
            interpretation = "implementation_fault_detected"
        elif misalignment:
            interpretation = "implementation_consistent_simulator_misalignment_observed"
        else:
            interpretation = "implementation_consistent_no_sign_normalization_or_indexing_fault_found"
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "pairing": pairing,
            "immutable_sources": {
                name: {"path": str(path), "file_sha256": expected_files[name]}
                for name, path in sources.items()
            },
            "protected_links": list(artifact["protected_links"]),
            "feature_names": list(FEATURE_NAMES),
            "records": records,
            "state_results": state_results,
            "decision": {
                "implementation_consistency_pass": implementation_pass,
                "simulator_misalignment_observed": misalignment,
                "simulator_minus_preference_count": simulator_minus_count,
                "simulator_plus_preference_count": simulator_plus_count,
                "simulator_tie_count": simulator_tie_count,
                "interpretation": interpretation,
                "paired_directional_supervision_is_next_model_test": implementation_pass,
                "closed_loop_authorized": False,
            },
            "total_fresh_rollout_count": 2 * len(records),
            "total_clone_env_step_wall_seconds": float(sum(
                item[sign]["env_step_wall_seconds"]
                for item in records for sign in ("minus", "plus")
            )),
            "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
        }
        result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
        _atomic_write(args.output.resolve(), result)
        print(json.dumps(result["decision"], sort_keys=True))
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

