#!/usr/bin/env python3
"""Run the matched direct-horizon normalized paired-secant loss ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    load_direct_horizon_config, normalized_secant_fitted_gate,
    predict_direct_horizon, save_direct_horizon_weights,
    temporal_error_metrics, train_direct_horizon_ensemble,
)
from main.multilink_ellipsoid.factorized_direct_horizon_root_cause_audit import (
    sensitivity_audit,
)
from main.multilink_ellipsoid.factorized_execution_pilot import (
    payload_sha256, safety_metrics,
)
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    _hash_array, add_direct_horizon_arguments, direct_paths,
    validate_direct_sources,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


RESULT_SCHEMA = "vlsa_distal_factorized_direct_horizon_normalized_secant_result.v1"
VALIDATION_SCHEMA = (
    "vlsa_distal_factorized_direct_horizon_normalized_secant_validation.v1"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_direct_horizon_arguments(parser)
    parser.add_argument("--matched-direct-run", type=Path, required=True)
    parser.add_argument("--root-cause-run", type=Path, required=True)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def extra_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "matched_direct_run": args.matched_direct_run.resolve(),
        "root_cause_run": args.root_cause_run.resolve(),
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "output": args.output.resolve(),
    }


def validate_matched_sources(
    paths: Mapping[str, Path], config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = config["immutable_source"]
    matched = paths["matched_direct_run"]
    root = paths["root_cause_run"]
    for directory in (matched, root):
        _require(directory.is_dir() and not directory.is_symlink(),
                 "normalized-secant matched source directory differs")
    for path, key in (
        (matched / "model.npz", "matched_direct_model_file_sha256"),
        (matched / "predictions.npz", "matched_direct_predictions_file_sha256"),
        (matched / "result.json", "matched_direct_result_file_sha256"),
        (matched / "validation.json", "matched_direct_validation_file_sha256"),
        (root / "result.json", "root_cause_result_file_sha256"),
        (root / "validation.json", "root_cause_validation_file_sha256"),
    ):
        _require(_file_sha256(path) == source[key],
                 "normalized-secant matched source hash differs")
    baseline = _load(matched / "result.json")
    root_result = _load(root / "result.json")
    root_validation = _load(root / "validation.json")
    _require(
        baseline.get("result_payload_sha256")
        == payload_sha256(baseline, "result_payload_sha256")
        and root_result.get("result_payload_sha256")
        == payload_sha256(root_result, "result_payload_sha256")
        and root_validation.get("valid") is True
        and root_validation.get("result_payload_sha256")
        == root_result["result_payload_sha256"],
        "normalized-secant matched source receipt differs",
    )
    return baseline, root_result


def split_temporal(arrays: Mapping[str, Any], predicted_q: Any) -> dict[str, Any]:
    import numpy as np

    split = np.asarray(arrays["split"], dtype=object)
    exact = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
    predicted = np.asarray(predicted_q, dtype=np.float64)
    return {
        name: temporal_error_metrics(predicted[split == name], exact[split == name])
        for name in ("train", "validation", "test")
    }


def split_safety(
    arrays: Mapping[str, Any], predicted_margin: Any,
    factorized_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        name: safety_metrics(
            arrays["minimum_margin_m"], predicted_margin, arrays,
            factorized_config, split_name=name, evaluation_only=True,
        ) for name in ("train", "validation", "test")
    }


def compute_metrics(
    *, arrays: Mapping[str, Any], predicted_q: Any, predicted_trace: Any,
    sensitivities: Mapping[str, Any], factorized_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    state_to_split = {
        int(state): str(split) for state, split in zip(
            np.asarray(arrays["state_index"], dtype=np.int64),
            np.asarray(arrays["split"], dtype=object),
        )
    }
    predicted_minimum = np.min(np.asarray(predicted_trace), axis=1)
    return {
        "temporal_joint": split_temporal(arrays, predicted_q),
        "safety": split_safety(arrays, predicted_minimum, factorized_config),
        "sensitivity": sensitivity_audit(
            arrays=arrays, predicted_q=predicted_q,
            predicted_h=predicted_trace, sensitivities=sensitivities,
            state_to_split=state_to_split,
        ),
    }


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
    config = load_direct_horizon_config(paths["direct_config"])
    _require(
        config["protocol_id"]
        == "vlsa-distal-factorized-direct-horizon-normalized-secant-moka10-v1",
        "normalized-secant protocol differs",
    )
    validate_direct_sources(paths, config)
    baseline, root_result = validate_matched_sources(paths, config)
    _require(
        int(representation["structured_input_dimension"])
        == int(arrays["features"].shape[1]),
        "normalized-secant structured representation differs",
    )
    torch.set_num_threads(8)
    models, state, training = train_direct_horizon_ensemble(
        arrays, sensitivities, local_geometry, config, normalization,
    )
    predicted_q = predict_direct_horizon(models, state, arrays)
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
    baseline_validation_RMSE = float(
        root_result["audit"]["populations"]["validation"]
        ["joint_error_overall"]["RMSE"]
    )
    decision = normalized_secant_fitted_gate(
        sensitivity=metrics["sensitivity"],
        validation_temporal=metrics["temporal_joint"]["validation"],
        baseline_validation_joint_RMSE_rad=baseline_validation_RMSE,
        config=config,
    )
    model_receipt = save_direct_horizon_weights(
        paths["experimental_model"], state,
    )
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["predictions"],
        joint_position_rad=predicted_q,
        predicted_clearance_trace_m=predicted_trace,
        predicted_minimum_margin_m=predicted_minimum,
        exact_q_static_clearance_trace_m=exact_static_trace,
        exact_q_static_minimum_margin_m=exact_static_minimum,
    )
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "matched_change": {
            "changed": "raw_secant_Huber_to_training_RMS_normalized_paired_secant_vector_MSE",
            "unchanged": [
                "direct_horizon_architecture", "joint_trajectory_loss",
                "symmetric_safety_normal_loss", "data", "episode_splits",
                "ensemble_seeds", "optimizer", "schedule",
            ],
            "one_sided_geometry_loss_executed": False,
        },
        "matched_baseline": {
            "result_file_sha256": _file_sha256(
                paths["matched_direct_run"] / "result.json"
            ),
            "result_payload_sha256": baseline["result_payload_sha256"],
            "root_cause_result_file_sha256": _file_sha256(
                paths["root_cause_run"] / "result.json"
            ),
            "root_cause_result_payload_sha256": root_result[
                "result_payload_sha256"
            ],
            "sensitivity": root_result["audit"]["sensitivity"]["source"],
        },
        "training_population_representation": representation,
        "local_geometry_jacobian_audit": local_geometry["audit"],
        "training": training, "model": model_receipt,
        "predictions": {
            "path": str(paths["predictions"]),
            "file_sha256": _file_sha256(paths["predictions"]),
            "joint_sha256": _hash_array(predicted_q),
            "clearance_trace_sha256": _hash_array(predicted_trace),
            "exact_static_trace_sha256": _hash_array(exact_static_trace),
        },
        "geometry": geometry, "metrics": metrics, "decision": decision,
        "forbidden_action_receipt": {
            "new_unseen_episode_evaluation_executed": False,
            "new_rollout_labels_collected": False,
            "one_sided_geometry_loss_executed": False,
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
        "decision": decision, "training": training,
        "train_sensitivity": metrics["sensitivity"]["train"]
        ["all_horizon_trace"]["joint"],
        "validation_sensitivity": metrics["sensitivity"]["validation"]
        ["all_horizon_trace"]["joint"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
