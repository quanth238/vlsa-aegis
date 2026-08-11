#!/usr/bin/env python3
"""Run the preregistered explicit local-affine execution-Jacobian pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Tuple

from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    normalized_secant_fitted_gate,
)
from main.multilink_ellipsoid.factorized_execution_pilot import (
    payload_sha256,
)
from main.multilink_ellipsoid.factorized_explicit_execution_jacobian import (
    explicit_jacobian_consistency, load_explicit_jacobian_config,
    predict_explicit_jacobian, save_explicit_jacobian_weights,
    train_explicit_jacobian_ensemble,
)
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    _hash_array, add_direct_horizon_arguments, direct_paths,
    validate_direct_sources,
)
from scripts.evaluate_distal_direct_horizon_normalized_secant_moka10 import (
    compute_metrics, validate_matched_sources,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


RESULT_SCHEMA = "vlsa_distal_factorized_explicit_execution_jacobian_result.v1"
VALIDATION_SCHEMA = (
    "vlsa_distal_factorized_explicit_execution_jacobian_validation.v1"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_direct_horizon_arguments(parser)
    parser.add_argument("--matched-direct-run", type=Path, required=True)
    parser.add_argument("--matched-direct-validation", type=Path, required=True)
    parser.add_argument("--root-cause-run", type=Path, required=True)
    parser.add_argument("--matched-normalized-run", type=Path, required=True)
    parser.add_argument("--matched-normalized-config", type=Path, required=True)
    parser.add_argument(
        "--matched-normalized-validation", type=Path, required=True,
    )
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def extra_paths(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "matched_direct_run": args.matched_direct_run.resolve(),
        "matched_direct_validation": args.matched_direct_validation.resolve(),
        "root_cause_run": args.root_cause_run.resolve(),
        "matched_normalized_run": args.matched_normalized_run.resolve(),
        "matched_normalized_config": args.matched_normalized_config.resolve(),
        "matched_normalized_validation": (
            args.matched_normalized_validation.resolve()
        ),
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "output": args.output.resolve(),
    }


def validate_normalized_source(
    paths: Mapping[str, Path], config: Mapping[str, Any],
) -> Dict[str, Any]:
    source = config["immutable_source"]
    run = paths["matched_normalized_run"]
    _require(run.is_dir() and not run.is_symlink(),
             "explicit-J normalized source directory differs")
    for path, key in (
        (paths["matched_normalized_config"],
         "matched_normalized_config_file_sha256"),
        (run / "model.npz", "matched_normalized_model_file_sha256"),
        (run / "predictions.npz",
         "matched_normalized_predictions_file_sha256"),
        (run / "result.json", "matched_normalized_result_file_sha256"),
        (paths["matched_normalized_validation"],
         "matched_normalized_validation_file_sha256"),
    ):
        _require(_file_sha256(path) == source[key],
                 "explicit-J normalized source hash differs")
    result = _load(run / "result.json")
    validation = _load(paths["matched_normalized_validation"])
    _require(
        result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and validation.get("valid") is True
        and validation.get("result_payload_sha256")
        == result["result_payload_sha256"],
        "explicit-J normalized source receipt differs",
    )
    return result


def explicit_decision(
    *, fitted: Mapping[str, Any], consistency: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    gate = config["prediction_gate"]
    structural = {
        "predicted_secants_equal_explicit_J": bool(
            consistency[
                "predicted_secant_vs_explicit_J_maximum_absolute_rad_per_action"
            ] <= float(gate[
                "maximum_predicted_secant_vs_explicit_J_absolute_error_rad_per_action"
            ])
        ),
        "J_at_initial_state_is_zero": bool(
            consistency["maximum_absolute_predicted_J_at_substep_zero"]
            <= float(gate[
                "maximum_architecturally_forbidden_J_absolute_rad_per_action"
            ])
        ),
        "second_action_has_no_pre_causal_effect": bool(
            consistency[
                "maximum_absolute_predicted_second_action_J_through_substep_25"
            ] <= float(gate[
                "maximum_architecturally_forbidden_J_absolute_rad_per_action"
            ])
        ),
    }
    passed = bool(
        fitted["fitted_sensitivity_gate_pass"] and all(structural.values())
    )
    return {
        "matched_fitted_gate": fitted,
        "structural_tests": structural,
        "explicit_execution_jacobian_prediction_gate_pass": passed,
        "authorized_next_action": (
            "preregister_new_grouped_unseen_episode_prediction" if passed
            else "rethink_decoder_or_optimization_without_calibration_or_QP"
        ),
        "residual_bound_calibration_QP_closed_loop_authorized": False,
    }


def run_prediction(
    *, paths: Mapping[str, Path], factorized_config: Mapping[str, Any],
    complete_dataset: Mapping[str, Any],
    complete_collection: Mapping[str, Any], arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any], local_geometry: Mapping[str, Any],
    config: Mapping[str, Any], models: Any, state: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    predicted_q, prediction = predict_explicit_jacobian(models, state, arrays)
    geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        arrays, predicted_q, ("train", "validation", "test"),
        return_trace=True,
    )
    predicted_trace = geometry.pop("predicted_clearance_trace_m")
    exact_static_trace = geometry.pop("exact_q_static_clearance_trace_m")
    predicted_minimum = geometry.pop("predicted_minimum_margin_m")
    exact_static_minimum = geometry.pop("exact_q_static_minimum_margin_m")
    metrics = compute_metrics(
        arrays=arrays, predicted_q=predicted_q,
        predicted_trace=predicted_trace, sensitivities=sensitivities,
        factorized_config=factorized_config,
    )
    consistency = explicit_jacobian_consistency(
        arrays=arrays, sensitivities=sensitivities,
        predicted_q=predicted_q, prediction=prediction,
    )
    stored = {
        "joint_position_rad": predicted_q,
        "predicted_clearance_trace_m": predicted_trace,
        "predicted_minimum_margin_m": predicted_minimum,
        "exact_q_static_clearance_trace_m": exact_static_trace,
        "exact_q_static_minimum_margin_m": exact_static_minimum,
        "state_index": prediction["state_index"],
        "nominal_displacement_rad": prediction["nominal_displacement_rad"],
        "execution_jacobian_rad_per_action": prediction[
            "execution_jacobian_rad_per_action"
        ],
    }
    return predicted_q, geometry, metrics, consistency, stored


def main() -> int:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, _, _, factorized_config, complete_dataset,
        complete_collection, arrays, sensitivities, normalization,
        representation, local_geometry,
    ) = prepare(args)
    paths.update(direct_paths(args))
    paths.update(extra_paths(args))
    config = load_explicit_jacobian_config(paths["direct_config"])
    validate_direct_sources(paths, config)
    _, root_result = validate_matched_sources(paths, config)
    normalized_result = validate_normalized_source(paths, config)
    _require(
        int(representation["structured_input_dimension"])
        == int(arrays["features"].shape[1]),
        "explicit-J structured representation differs",
    )
    torch.set_num_threads(8)
    models, state, training = train_explicit_jacobian_ensemble(
        arrays, sensitivities, local_geometry, config, normalization,
    )
    predicted_q, geometry, metrics, consistency, stored = run_prediction(
        paths=paths, factorized_config=factorized_config,
        complete_dataset=complete_dataset,
        complete_collection=complete_collection, arrays=arrays,
        sensitivities=sensitivities, local_geometry=local_geometry,
        config=config, models=models, state=state,
    )
    baseline_validation_rmse = float(
        root_result["audit"]["populations"]["validation"]
        ["joint_error_overall"]["RMSE"]
    )
    fitted = normalized_secant_fitted_gate(
        sensitivity=metrics["sensitivity"],
        validation_temporal=metrics["temporal_joint"]["validation"],
        baseline_validation_joint_RMSE_rad=baseline_validation_rmse,
        config=config,
    )
    decision = explicit_decision(
        fitted=fitted, consistency=consistency, config=config,
    )
    model_receipt = save_explicit_jacobian_weights(
        paths["experimental_model"], state,
    )
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["predictions"], **stored)
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "matched_change": {
            "changed": (
                "implicit_candidate_conditioned_trajectory_decoder_to_"
                "nominal_intercept_plus_explicit_local_execution_J_decoder"
            ),
            "unchanged": [
                "data", "episode_splits", "ensemble_seeds", "optimizer",
                "schedule", "joint_trajectory_loss",
                "symmetric_safety_normal_loss",
                "training_RMS_normalized_paired_secant_loss",
            ],
            "new_unseen_episode_evaluation_executed": False,
        },
        "matched_normalized_baseline": {
            "result_file_sha256": _file_sha256(
                paths["matched_normalized_run"] / "result.json"
            ),
            "result_payload_sha256": normalized_result[
                "result_payload_sha256"
            ],
            "decision": normalized_result["decision"],
            "metrics": normalized_result["metrics"],
        },
        "training_population_representation": representation,
        "local_geometry_jacobian_audit": local_geometry["audit"],
        "training": training, "model": model_receipt,
        "predictions": {
            "path": str(paths["predictions"]),
            "file_sha256": _file_sha256(paths["predictions"]),
            "joint_sha256": _hash_array(stored["joint_position_rad"]),
            "clearance_trace_sha256": _hash_array(
                stored["predicted_clearance_trace_m"]
            ),
            "execution_jacobian_sha256": _hash_array(
                stored["execution_jacobian_rad_per_action"]
            ),
        },
        "geometry": geometry, "metrics": metrics,
        "explicit_jacobian_consistency": consistency,
        "decision": decision,
        "forbidden_action_receipt": {
            "new_unseen_episode_evaluation_executed": False,
            "new_rollout_labels_collected": False,
            "residual_bound_calibration_executed": False,
            "QP_executed": False, "closed_loop_executed": False,
            "poisson_or_SDF_executed": False,
            "binary_classifier_executed": False,
        },
        "future_untouched_intervention_manifest": {
            "file_sha256": config["immutable_source"]
            ["future_untouched_manifest_file_sha256"],
            "opened_or_evaluated": False,
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "decision": decision, "consistency": consistency,
        "train_sensitivity": metrics["sensitivity"]["train"]
        ["all_horizon_trace"]["joint"],
        "validation_sensitivity": metrics["sensitivity"]["validation"]
        ["all_horizon_trace"]["joint"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
