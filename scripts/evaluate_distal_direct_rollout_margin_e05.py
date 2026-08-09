#!/usr/bin/env python3
"""Evaluate direct rollout-margin steering and QP at E05 action 185."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.direct_rollout_margin import (
    DIRECT_MARGIN_RESULT_SCHEMA,
    DIRECT_MARGIN_TRAINING_SCHEMA,
    load_direct_margin_config,
    load_direct_margin_model,
    predict_direct_margins_and_jacobian,
    project_direct_margin_action,
)
from main.multilink_ellipsoid.gradient_random_control import (
    sample_feasible_unit_directions,
)
from main.multilink_ellipsoid.two_step_margin import (
    TWO_STEP_DATASET_SCHEMA,
    feature_context,
    summarize_chunk,
)
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _summary_record(summary: Mapping[str, Any]) -> dict[str, Any]:
    margins = [float(value) for value in summary["minimum_substep_clearance_m"]]
    return {
        "minimum_substep_clearance_m": margins,
        "rho_roll_m": float(min(margins)),
        "active_constraint_index": int(min(range(7), key=lambda index: margins[index])),
        "D_opt_proxy_safe": bool(summary["D_opt_proxy_safe"]),
        "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
        "raw_protected_contact_count": int(summary["raw_protected_contact_count"]),
        "maximum_within_step_obstacle_l1_displacement_m": float(
            summary["maximum_within_step_obstacle_l1_displacement_m"]
        ),
        "next_state_sha256": str(summary["next_state_sha256"]),
        "env_step_wall_seconds": float(summary["env_step_wall_seconds"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()

    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )

    config = load_direct_margin_config(args.config.resolve())
    identities = config["immutable_sources"]
    immutable = {
        "population_manifest": (args.population_manifest, identities["population_manifest_file_sha256"]),
        "archived_table1": (args.archived, identities["archived_table1_file_sha256"]),
        "geometry_config": (args.geometry_config, identities["geometry_config_file_sha256"]),
        "exact_box_config": (args.exact_box_config, identities["exact_box_config_file_sha256"]),
        "dataset": (args.dataset, identities["dataset_file_sha256"]),
        "dataset_result": (args.dataset_result, identities["dataset_result_file_sha256"]),
        "dataset_validation": (args.dataset_validation, identities["dataset_validation_file_sha256"]),
    }
    for name, (path, expected) in immutable.items():
        _require(
            path.is_file() and not path.is_symlink()
            and _file_sha256(path.resolve()) == expected,
            "direct-margin immutable source differs: %s" % name,
        )
    dataset = _load(args.dataset.resolve())
    dataset_validation = _load(args.dataset_validation.resolve())
    training = _load(args.training.resolve())
    archived = _load(args.archived.resolve())
    _require(
        dataset.get("schema_version") == TWO_STEP_DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256") == identities["dataset_payload_sha256"]
        and dataset.get("source_commit") == identities["dataset_source_commit"],
        "direct-margin dataset payload differs",
    )
    _require(
        dataset_validation.get("status") == "valid"
        and dataset_validation.get("neural_training_authorized") is True,
        "direct-margin dataset validation differs",
    )
    _require(
        training.get("schema_version") == DIRECT_MARGIN_TRAINING_SCHEMA
        and training.get("status") == "complete"
        and training.get("training_payload_sha256")
        == _hash_without(training, "training_payload_sha256")
        and training.get("source", {}).get("commit") == args.expected_commit,
        "direct-margin training artifact differs",
    )
    _require(
        _file_sha256(args.model.resolve())
        == training["model_artifact"]["file_sha256"],
        "direct-margin model file differs",
    )
    model, model_state = load_direct_margin_model(args.model.resolve())
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    rows = [
        row for row in read_jsonl(args.population_manifest.resolve())
        if row.get("case_id") == config["case_id"]
    ]
    _require(len(rows) == 1, "direct-margin E05 manifest row is not unique")
    case = rows[0]
    validate_case_row(case, args.repo_root.resolve())
    episode = next(
        item for item in dataset["episode_results"]
        if item["case_id"] == config["case_id"]
    )
    _require(
        episode["split"] == "test"
        and episode["selected_step"] == config["state_step"]
        and episode["eligible"] is True,
        "direct-margin E05 dataset state differs",
    )

    geometry_config = load_shadow_config(args.geometry_config.resolve())
    exact_box_config = load_obstacle_primitive_config(
        args.exact_box_config.resolve()
    )
    env = probe_env = None
    try:
        env, probe_env, _, _, setup = _build_pair(runtime, case)
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config,
            exact_box_config=exact_box_config,
            archived=archived,
            env=env,
            obstacle_name=setup["obstacle_name"],
        )
        for step in range(int(config["state_step"])):
            env.step(_canonical_action(archived["actions"][step], step).tolist())
        probe = SubstepEightConstraintProbe(
            probe_env, geometry,
            active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6,
            contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        step = int(config["state_step"])
        first = _canonical_action(archived["actions"][step], step)
        second = _canonical_action(archived["actions"][step + 1], step + 1)
        _require(
            np.array_equal(first, np.asarray(episode["nominal_first_action"], dtype=np.float64))
            and np.array_equal(second, np.asarray(episode["nominal_second_action"], dtype=np.float64)),
            "direct-margin E05 nominal chunk differs",
        )
        context = feature_context(env, probe)
        nominal = _summary_record(summarize_chunk(probe.rollout_chunk(env, [first, second])))
        expected_nominal = next(
            item["minimum_two_step_m"] for item in episode["search"]
            if item["step"] == step
        )
        _require(
            abs(nominal["rho_roll_m"] - float(expected_nominal)) <= 1.0e-10,
            "direct-margin E05 nominal rollout differs",
        )
        predicted, jacobian, inference = predict_direct_margins_and_jacobian(
            model, model_state, context, first[:3], first[:3], second[:3]
        )
        predicted_active = int(np.argmin(predicted))
        gradient = np.asarray(jacobian[predicted_active], dtype=np.float64)
        gradient_norm = float(np.linalg.norm(gradient))
        _require(gradient_norm > 1.0e-12, "direct-margin nominal gradient is zero")
        learned_direction = gradient / gradient_norm
        random_config = config["matched_random"]
        radius = float(random_config["radius_action"])
        _require(
            np.all(np.abs(first[:3] + radius * learned_direction) <= 1.0 + 1.0e-12),
            "direct-margin learned action violates bounds",
        )

        def evaluate(direction: Any, arm: str, index: Any) -> dict[str, Any]:
            candidate = first.copy()
            candidate[:3] = first[:3] + radius * np.asarray(direction, dtype=np.float64)
            _require(
                np.all(np.abs(candidate[:3]) <= 1.0 + 1.0e-12)
                and abs(float(np.linalg.norm(candidate[:3] - first[:3])) - radius) <= 1.0e-10,
                "direct-margin matched action differs",
            )
            summary = _summary_record(
                summarize_chunk(probe.rollout_chunk(env, [candidate, second]))
            )
            summary.update({
                "arm": arm, "direction_index": index,
                "direction": np.asarray(direction, dtype=np.float64).tolist(),
                "radius_action": radius, "candidate_xyz": candidate[:3].tolist(),
                "rho_improvement_m": summary["rho_roll_m"] - nominal["rho_roll_m"],
            })
            return summary

        learned_record = evaluate(learned_direction, "learned_margin_gradient", None)
        directions = sample_feasible_unit_directions(
            first[:3], radius=radius,
            count=int(random_config["direction_count"]),
            seed=int(random_config["seed"]), action_limit=1.0,
            maximum_attempts=int(random_config["maximum_sampling_attempts"]),
        )
        random_records = [
            evaluate(direction, "matched_random", index)
            for index, direction in enumerate(directions)
        ]
        equal_or_better = sum(
            item["rho_improvement_m"]
            >= learned_record["rho_improvement_m"] - 1.0e-15
            for item in random_records
        )
        empirical_p = (1.0 + equal_or_better) / (1.0 + len(random_records))
        random_gate = bool(
            learned_record["rho_improvement_m"] > 0.0
            and empirical_p
            <= float(random_config["empirical_equal_or_better_p_maximum"])
        )

        projections = {}
        for calibrated in (False, True):
            name = "calibrated" if calibrated else "uncalibrated"
            projection = project_direct_margin_action(
                model, model_state, context, first[:3], second[:3], config,
                calibrated=calibrated,
            )
            exact = None
            if projection["valid"]:
                candidate = first.copy()
                candidate[:3] = projection["projected_xyz"]
                exact = _summary_record(
                    summarize_chunk(probe.rollout_chunk(env, [candidate, second]))
                )
            qps = [
                item["qp"] for item in projection["iterations"]
                if item.get("qp") is not None
            ]
            valid_seven_row_qp = bool(
                qps and all(item["valid"] and len(item["lower"]) == 7 for item in qps)
            )
            gate = bool(
                projection["valid"] and valid_seven_row_qp
                and exact is not None
                and exact["D_opt_proxy_safe"] and exact["D_sim_raw_safe"]
            )
            projections[name] = {
                "projection": projection,
                "valid_seven_row_qp": valid_seven_row_qp,
                "fresh_exact_two_step": exact,
                "projection_gate_pass": gate,
            }

        training_audit = training["training"]
        model_gate = bool(training_audit["e05_model_gate_pass"])
        false_safe_gate = bool(training_audit["test_false_safe_gate_pass"])
        projection_gate = bool(projections["calibrated"]["projection_gate_pass"])
        research_go = bool(
            model_gate and false_safe_gate and random_gate and projection_gate
        )
        result = {
            "schema_version": DIRECT_MARGIN_RESULT_SCHEMA,
            "status": "complete", "scientific_result": True,
            "claim_scope": config["claim_scope"], "source": source,
            "allocation": allocation, "config": config,
            "immutable_artifacts": {
                name: {"path": str(path.resolve()), "file_sha256": expected}
                for name, (path, expected) in immutable.items()
            },
            "training": {
                "path": str(args.training.resolve()),
                "file_sha256": _file_sha256(args.training.resolve()),
                "payload_sha256": training["training_payload_sha256"],
                "model_path": str(args.model.resolve()),
                "model_file_sha256": _file_sha256(args.model.resolve()),
                "audit": training_audit,
            },
            "state": {
                "case_id": config["case_id"], "state_step": step,
                "nominal_first_action": first.tolist(),
                "nominal_second_action": second.tolist(),
                "nominal_exact_two_step": nominal,
            },
            "nominal_prediction": {
                "predicted_margin_m": predicted.tolist(),
                "predicted_active_constraint_index": predicted_active,
                "jacobian_m_per_action": jacobian.tolist(),
                "active_gradient": gradient.tolist(),
                "active_gradient_norm": gradient_norm,
                "inference_and_jacobian_wall_seconds": inference,
            },
            "matched_random": {
                "learned": learned_record,
                "random_records": random_records,
                "random_improvement_mean_m": float(np.mean([
                    item["rho_improvement_m"] for item in random_records
                ])),
                "random_improvement_maximum_m": float(max(
                    item["rho_improvement_m"] for item in random_records
                )),
                "random_equal_or_better_count": int(equal_or_better),
                "empirical_equal_or_better_p": float(empirical_p),
                "matched_random_gate_pass": random_gate,
            },
            "projections": projections,
            "decision": {
                "e05_model_gate_pass": model_gate,
                "all_test_false_safe_gate_pass": false_safe_gate,
                "learned_gradient_beats_matched_random": random_gate,
                "calibrated_seven_row_qp_exact_gate_pass": projection_gate,
                "research_direction_go": research_go,
                "closed_loop_authorized": False,
                "stop_reason": None if research_go else "one_or_more_direct_margin_mechanism_gates_failed",
            },
            "fresh_exact_two_step_rollout_count": 1 + len(random_records) + sum(
                value["fresh_exact_two_step"] is not None
                for value in projections.values()
            ) + 1,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _hash_without(
            result, "result_payload_sha256"
        )
        _atomic_write(args.output.resolve(), result)
        print(json.dumps({
            "decision": result["decision"],
            "learned_improvement_mm": 1000.0 * learned_record["rho_improvement_m"],
            "matched_random_p": empirical_p,
            "calibrated_projection": projections["calibrated"]["projection_gate_pass"],
        }, sort_keys=True), flush=True)
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
