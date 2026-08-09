#!/usr/bin/env python3
"""Audit grouped held-out task generalization and exact test projections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.boundary_generalization import (
    GENERALIZATION_DATASET_SCHEMA, GENERALIZATION_RESULT_SCHEMA,
    GENERALIZATION_TRAINING_SCHEMA, load_generalization_config,
    load_selected_manifest,
)
from main.multilink_ellipsoid.execution_margin_nn import load_model_artifact
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _file_sha256, _git_identity, _load, _require


def _hash_without(value, key):
    payload = dict(value); payload.pop(key, None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _model_metrics(model, state, records, config):
    import numpy as np
    import torch
    from main.multilink_ellipsoid.boundary_capacity import _predict_all_gradients

    features = np.asarray([item["feature_vector"] for item in records], dtype=np.float64)
    current = np.asarray([item["current_clearance_m"] for item in records], dtype=np.float64)
    actual = np.asarray([item["minimum_substep_clearance_m"] for item in records], dtype=np.float64)
    normalized = (features - np.asarray(state["feature_mean"])) / np.asarray(state["feature_standard_deviation"])
    device = next(model.parameters()).device
    tensor = torch.as_tensor(normalized, dtype=torch.float64, device=device).requires_grad_(True)
    scale = torch.as_tensor(np.asarray(state["feature_standard_deviation"])[-3:], dtype=torch.float64, device=device)
    with torch.enable_grad():
        loss_mm, gradient_mm = _predict_all_gradients(model, tensor, scale, create_graph=False)
    predicted = current - loss_mm.detach().cpu().numpy() / 1000.0
    predicted_gradient = gradient_mm.detach().cpu().numpy() / 1000.0
    conservative = predicted - np.asarray(state["calibration_m"])[None, :]
    active = np.argmin(actual, axis=1)
    indexes = np.arange(len(records))
    active_actual = actual[indexes, active]
    boundary = np.abs(active_actual) <= float(config["sampling"]["boundary_band_m"])
    _require(np.any(boundary), "held-out episode has no boundary records")
    bidx = indexes[boundary]; brows = active[boundary]
    error = predicted[bidx, brows] - actual[bidx, brows]
    baseline_error = current[bidx, brows] - actual[bidx, brows]
    false_safe = np.logical_and(conservative >= 0.0, actual < 0.0)
    cosines = []
    for index, item in enumerate(records):
        gradient = item.get("gradient_m_per_action")
        valid = item.get("gradient_valid_rows")
        row = int(active[index])
        if gradient is None or valid is None or not bool(valid[row]) or not boundary[index]:
            continue
        target = np.asarray(gradient, dtype=np.float64)[row]
        estimate = predicted_gradient[index, row]
        denominator = float(np.linalg.norm(target) * np.linalg.norm(estimate))
        if denominator > 1e-12:
            cosines.append(float(np.dot(target, estimate) / denominator))
    return {
        "boundary_record_count": int(np.count_nonzero(boundary)),
        "active_boundary_rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "active_boundary_current_clearance_baseline_rmse_m": float(np.sqrt(np.mean(baseline_error ** 2))),
        "conservative_false_safe_candidate_count": int(np.count_nonzero(np.any(false_safe, axis=1))),
        "active_gradient_cosine_count": len(cosines),
        "active_gradient_cosine_mean": None if not cosines else float(np.mean(cosines)),
        "active_gradient_cosine_minimum": None if not cosines else float(np.min(cosines)),
    }


def _projection(
    *, runtime, case, archived, step, geometry_config, exact_box_config,
    config, model, state,
):
    import numpy as np
    from main.multilink_ellipsoid.execution_margin_nn import predict_margin_and_jacobian, project_action_with_model
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe

    env = probe_env = None
    try:
        env, probe_env, _, _, setup = _build_pair(runtime, case)
        geometry, boxes = _geometry(
            geometry_config=geometry_config, exact_box_config=exact_box_config,
            archived=archived, env=env, obstacle_name=setup["obstacle_name"],
        )
        for index in range(step):
            env.step(_canonical_action(archived["actions"][index], index).tolist())
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=boxes,
        )
        nominal = _canonical_action(archived["actions"][step], step)
        nominal_exact = probe.transition(env, nominal)
        start = nominal_exact["substeps"][0]
        margin, _, _ = predict_margin_and_jacobian(model, state, start, nominal[:3], nominal[:3])
        conservative = margin - np.asarray(state["calibration_m"])
        projected = project_action_with_model(model, state, start, nominal[:3], config)
        exact = None
        if projected["valid"]:
            action = nominal.copy(); action[:3] = projected["projected_xyz"]
            exact = probe.transition(env, action)
        qp_iterations = [item["qp"] for item in projected["iterations"] if item.get("qp") is not None]
        seven_row = bool(qp_iterations and all(len(item["lower"]) == 7 and item["valid"] for item in qp_iterations))
        exact_proxy_safe = bool(exact is not None and np.all(np.asarray(exact["minimum_substep_clearance_m"][:7]) >= 0.0))
        exact_raw_safe = bool(exact is not None and exact["raw_protected_contact_count"] == 0 and exact["maximum_within_step_obstacle_l1_displacement_m"] <= 1e-4)
        return {
            "state_step": step,
            "nominal_minimum_exact_margin_m": float(np.min(nominal_exact["minimum_substep_clearance_m"][:7])),
            "nominal_minimum_conservative_neural_margin_m": float(np.min(conservative)),
            "activated": bool(np.min(conservative) < float(config["projection"]["activation_warning_m"])),
            "projection": projected,
            "valid_seven_row_qp": seven_row,
            "projected_exact_minimum_margin_m": None if exact is None else float(np.min(exact["minimum_substep_clearance_m"][:7])),
            "projected_exact_proxy_safe": exact_proxy_safe,
            "projected_exact_raw_safe": exact_raw_safe,
            "projection_gate_pass": bool(projected["valid"] and seven_row and exact_proxy_safe and exact_raw_safe),
        }
    finally:
        if probe_env is not None: probe_env.close()
        if env is not None: env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); started = time.perf_counter_ns()
    from main.evaluate_safelibero_aegis import _runtime_imports, read_jsonl
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    config = load_generalization_config(args.config.resolve())
    selected = load_selected_manifest(args.selected_manifest.resolve(), config)
    dataset = _load(args.dataset.resolve()); training = _load(args.training.resolve()); validation = _load(args.dataset_validation.resolve())
    _require(dataset.get("schema_version") == GENERALIZATION_DATASET_SCHEMA and dataset.get("source_commit") == args.expected_commit, "evaluation dataset differs")
    _require(training.get("schema_version") == GENERALIZATION_TRAINING_SCHEMA and training.get("source", {}).get("commit") == args.expected_commit, "training artifact differs")
    _require(training.get("training_payload_sha256") == _hash_without(training, "training_payload_sha256"), "training payload differs")
    _require(validation.get("neural_training_authorized") is True and validation.get("dataset_file_sha256") == _file_sha256(args.dataset.resolve()), "dataset gate differs")
    geometry_config = load_shadow_config(args.geometry_config.resolve()); exact_box_config = load_obstacle_primitive_config(args.exact_box_config.resolve())
    population = {row["case_id"]: row for row in read_jsonl(args.population_manifest.resolve())}
    episode_state = {item["case_id"]: item for item in dataset["episode_results"]}
    test_rows = [row for row in selected if row["split"] == "test"]
    runtime = _runtime_imports(include_aegis=False)
    arm_results = {}
    for arm in config["training"]["arms"]:
        identity = training["arms"][arm]["model_artifact"]
        model_path = Path(identity["path"]).resolve()
        _require(model_path.is_file() and _file_sha256(model_path) == identity["file_sha256"] and model_path.parent == args.training.resolve().parent, "model artifact differs")
        model, state = load_model_artifact(model_path, device="cpu")
        episode_metrics = {}; projections = {}
        for row in test_rows:
            case_id = row["case_id"]
            records = [item for item in dataset["records"] if item["case_id"] == case_id and item["split"] == "test"]
            episode_metrics[case_id] = _model_metrics(model, state, records, config)
            archived_path = args.archived_root.resolve() / row["archived_relative_path"]
            archived = _load(archived_path)
            _require(_file_sha256(archived_path) == row["archived_file_sha256"] and archived.get("result_payload_sha256") == row["archived_payload_sha256"], "test archive differs")
            projections[case_id] = _projection(
                runtime=runtime, case=population[case_id], archived=archived,
                step=int(episode_state[case_id]["selected_step"]), geometry_config=geometry_config,
                exact_box_config=exact_box_config, config=config, model=model, state=state,
            )
        threshold = float(config["decision_gate"]["active_gradient_cosine_similarity_minimum"])
        model_gate = bool(all(
            value["active_boundary_rmse_m"] < value["active_boundary_current_clearance_baseline_rmse_m"]
            and value["conservative_false_safe_candidate_count"] == 0
            and value["active_gradient_cosine_count"] > 0
            and value["active_gradient_cosine_mean"] >= threshold
            for value in episode_metrics.values()
        ))
        projection_gate = bool(all(value["projection_gate_pass"] for value in projections.values()))
        arm_results[arm] = {
            "model_artifact": identity, "training": training["arms"][arm]["training"],
            "held_out_episode_metrics": episode_metrics, "held_out_exact_projections": projections,
            "decision": {"every_test_episode_model_gate_pass": model_gate, "every_test_episode_projection_gate_pass": projection_gate, "arm_generalization_gate_pass": bool(model_gate and projection_gate)},
        }
    go = bool(arm_results["boundary_margin_gradient"]["decision"]["arm_generalization_gate_pass"])
    result = {
        "schema_version": GENERALIZATION_RESULT_SCHEMA, "status": "complete", "scientific_result": True,
        "claim_scope": config["claim_scope"], "source": _git_identity(args.repo_root.resolve(), args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "grouped_split": config["split"], "test_case_ids": [row["case_id"] for row in test_rows],
        "dataset": {"path": str(args.dataset.resolve()), "file_sha256": _file_sha256(args.dataset.resolve()), "payload_sha256": dataset["dataset_payload_sha256"]},
        "training": {"path": str(args.training.resolve()), "file_sha256": _file_sha256(args.training.resolve()), "payload_sha256": training["training_payload_sha256"]},
        "arms": arm_results,
        "decision": {"grouped_generalization_go": go, "closed_loop_e05_authorized": go, "stop_reason": None if go else "gradient_arm_failed_one_or_more_held_out_episode_gates"},
        "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
    }
    result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
