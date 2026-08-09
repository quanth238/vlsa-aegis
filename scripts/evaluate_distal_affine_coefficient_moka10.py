#!/usr/bin/env python3
"""Evaluate learned affine QP rows with fresh exact held-out OSC rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_coefficient_model import (
    AFFINE_COEFFICIENT_DATASET_SCHEMA,
    AFFINE_COEFFICIENT_RESULT_SCHEMA,
    AFFINE_COEFFICIENT_TRAINING_SCHEMA,
    load_affine_coefficient_config,
    load_affine_coefficient_model,
    predict_affine_coefficients,
)
from main.multilink_ellipsoid.oracle_affine_safe_set import (
    solve_affine_certificate_qp,
)
from main.multilink_ellipsoid.two_step_margin import (
    feature_context, feature_vectors, load_selected_manifest,
)
from scripts.collect_distal_affine_coefficient_moka10 import _chunk_values
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action, _restore_env, _snapshot_env,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    import numpy as np
    from main.evaluate_safelibero_aegis import _runtime_imports, read_jsonl
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    config = load_affine_coefficient_config(args.config.resolve())
    dataset = _load(args.dataset.resolve())
    dataset_result = _load(args.dataset_result.resolve())
    dataset_validation = _load(args.dataset_validation.resolve())
    training = _load(args.training.resolve())
    _require(
        dataset.get("schema_version") == AFFINE_COEFFICIENT_DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == _hash_without(dataset, "dataset_payload_sha256")
        and dataset_result.get("decision", {}).get("neural_training_authorized") is True
        and dataset_validation.get("neural_training_authorized") is True
        and training.get("schema_version") == AFFINE_COEFFICIENT_TRAINING_SCHEMA
        and training.get("training_payload_sha256")
        == _hash_without(training, "training_payload_sha256")
        and training.get("model_artifact", {}).get("file_sha256")
        == _file_sha256(args.model.resolve()),
        "affine-coefficient learned inputs differ",
    )
    evaluation_authorized = bool(
        training["test_metrics"]["learned_evaluation_authorized"]
    )
    state_results = []
    case_results = {}
    if evaluation_authorized:
        selected = load_selected_manifest(args.selected_manifest.resolve(), config)
        selected_by_case = {item["case_id"]: item for item in selected}
        population = {
            item["case_id"]: item for item in read_jsonl(args.population_manifest.resolve())
        }
        geometry_config = load_shadow_config(args.geometry_config.resolve())
        exact_box_config = load_obstacle_primitive_config(args.exact_box_config.resolve())
        primary = selected_by_case["vlsa-t1-goal-ii-t0-e05"]
        placeholder_path = args.archived_root.resolve() / primary["archived_relative_path"]
        placeholder = _load(placeholder_path)
        runtime = _runtime_imports(include_aegis=False)
        model, model_state = load_affine_coefficient_model(args.model.resolve())
        test_states = [item for item in dataset["state_records"] if item["split"] == "test"]
        for case_id in sorted({item["case_id"] for item in test_states}):
            row = selected_by_case[case_id]
            archived_path = args.archived_root.resolve() / row["archived_relative_path"]
            archived = _load(archived_path)
            _require(
                _file_sha256(archived_path) == row["archived_file_sha256"]
                and archived.get("result_payload_sha256") == row["archived_payload_sha256"],
                "affine-coefficient held-out archive differs: %s" % case_id,
            )
            env = probe_env = None
            local_results = []
            try:
                env, probe_env, _, _, setup = _build_pair(runtime, population[case_id])
                geometry, exact_boxes = _geometry(
                    geometry_config=geometry_config, exact_box_config=exact_box_config,
                    archived=placeholder, env=env, obstacle_name=setup["obstacle_name"],
                )
                probe = SubstepEightConstraintProbe(
                    probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                    quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                    obstacle_primitive_union=exact_boxes,
                )
                states = sorted(
                    (item for item in test_states if item["case_id"] == case_id),
                    key=lambda item: int(item["state_step"]),
                )
                needed = {int(item["state_step"]) for item in states}
                snapshots = {}
                actions = archived["actions"]
                for step in range(max(needed) + 1):
                    if step in needed:
                        snapshots[step] = _snapshot_env(env)
                    if step < max(needed):
                        env.step(_canonical_action(actions[step], step).tolist())
                for state in states:
                    step = int(state["state_step"])
                    _restore_env(env, snapshots[step])
                    first = _canonical_action(actions[step], step)
                    second = _canonical_action(actions[step + 1], step + 1)
                    context = feature_context(env, probe)
                    _, live_features = feature_vectors(
                        context, first[:3], first[:3], second[:3]
                    )
                    feature_error = float(np.max(np.abs(
                        live_features
                        - np.asarray(state["pair_state_feature_vectors"], dtype=np.float64)
                    )))
                    clearance_error = float(np.max(np.abs(
                        np.asarray(context["current_clearance_m"], dtype=np.float64)
                        - np.asarray(state["current_clearance_m"], dtype=np.float64)
                    )))
                    _require(
                        feature_error <= 1.0e-8 and clearance_error <= 1.0e-8,
                        "affine-coefficient held-out state receipt differs",
                    )
                    b, a, error = predict_affine_coefficients(
                        model, model_state, live_features
                    )
                    intercept = b - error - model_state["calibration_m"]
                    certificate = {
                        "valid": True, "intercept_at_nominal_m": intercept.tolist(),
                        "gradients_m_per_action": a.tolist(),
                    }
                    qp = solve_affine_certificate_qp(
                        first[:3], state["action_lower"], state["action_upper"],
                        certificate, config["projection"],
                    )
                    exact = None
                    if qp["valid"]:
                        candidate = first.copy()
                        candidate[:3] = np.asarray(qp["candidate_xyz"], dtype=np.float64)
                        summary, ee = _chunk_values(
                            probe.rollout_chunk(env, [candidate, second])
                        )
                        exact = {
                            "minimum_distal_margin_m": summary[
                                "minimum_substep_clearance_m"
                            ],
                            "minimum_released_AEGIS_EE_margin_m": float(ee),
                            "D_opt_seven_distal_safe": bool(summary["D_opt_proxy_safe"]),
                            "released_AEGIS_EE_proxy_safe": bool(ee >= 0.0),
                            "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
                            "raw_protected_contact_count": int(
                                summary["raw_protected_contact_count"]
                            ),
                            "maximum_within_step_obstacle_l1_displacement_m": float(
                                summary["maximum_within_step_obstacle_l1_displacement_m"]
                            ),
                            "next_state_sha256": summary["next_state_sha256"],
                        }
                    gate = bool(
                        qp["valid"]
                        and qp.get("diagnostics", {}).get("input_constraint_count") == 7
                        and exact is not None and exact["D_opt_seven_distal_safe"]
                        and exact["released_AEGIS_EE_proxy_safe"]
                        and exact["D_sim_raw_safe"]
                        and exact["maximum_within_step_obstacle_l1_displacement_m"]
                        <= 1.0e-4
                    )
                    item = {
                        "case_id": case_id, "state_step": step,
                        "state_offset_from_crossing": state[
                            "state_offset_from_crossing"
                        ],
                        "feature_max_abs_error": feature_error,
                        "current_clearance_max_abs_error_m": clearance_error,
                        "predicted_nominal_margin_m": b.tolist(),
                        "predicted_gradient_m_per_action": a.tolist(),
                        "predicted_state_conditioned_error_m": error.tolist(),
                        "calibration_m": model_state["calibration_m"].tolist(),
                        "calibrated_lower_intercept_m": intercept.tolist(),
                        "qp": qp, "exact_two_step_verification": exact,
                        "state_gate_pass": gate,
                    }
                    state_results.append(item)
                    local_results.append(item)
            finally:
                if probe_env is not None:
                    probe_env.close()
                if env is not None:
                    env.close()
            case_results[case_id] = {
                "state_count": len(local_results),
                "case_gate_pass": bool(
                    len(local_results)
                    == int(config["state_sampling"]["expected_states_per_episode"])
                    and all(item["state_gate_pass"] for item in local_results)
                ),
            }
    projection_gate = bool(
        evaluation_authorized and len(case_results) == 3
        and all(item["case_gate_pass"] for item in case_results.values())
    )
    research_go = bool(evaluation_authorized and projection_gate)
    output = {
        "schema_version": AFFINE_COEFFICIENT_RESULT_SCHEMA,
        "status": "complete", "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(args.repo_root.resolve(), args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "dataset": {
            "path": str(args.dataset.resolve()),
            "file_sha256": _file_sha256(args.dataset.resolve()),
            "payload_sha256": dataset["dataset_payload_sha256"],
        },
        "training": {
            "path": str(args.training.resolve()),
            "file_sha256": _file_sha256(args.training.resolve()),
            "payload_sha256": training["training_payload_sha256"],
            "test_metrics": training["test_metrics"],
        },
        "model": {
            "path": str(args.model.resolve()),
            "file_sha256": _file_sha256(args.model.resolve()),
        },
        "state_results": state_results, "case_results": case_results,
        "decision": {
            "learned_evaluation_authorized": evaluation_authorized,
            "test_projection_gate_pass": projection_gate,
            "research_direction_go": research_go,
            "closed_loop_e05_authorized": False,
            "stop_reason": None if research_go else (
                "coefficient_model_metrics_failed"
                if not evaluation_authorized else "one_or_more_exact_test_QPs_failed"
            ),
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["result_payload_sha256"] = _hash_without(output, "result_payload_sha256")
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
