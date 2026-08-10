#!/usr/bin/env python3
"""Run the frozen one-sided geometry-loss ablation on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_execution_pilot import (
    COLLECTION_SCHEMA, DATASET_SCHEMA, cosine_summary, dataset_arrays,
    load_config, load_weights, payload_sha256, predict, safety_metrics,
    save_weights, secant_predictions, sensitivity_arrays,
)
from main.multilink_ellipsoid.factorized_one_sided_geometry import (
    RESULT_SCHEMA, fit_local_geometry_jacobians, load_one_sided_config,
    one_sided_decision, train_one_sided_geometry_ensemble,
)
from scripts.evaluate_distal_factorized_execution_moka10 import (
    _test_sensitivity_mask, evaluate_predicted_geometry,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_array(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factorized-config", type=Path, required=True)
    parser.add_argument("--complete-dataset", type=Path, required=True)
    parser.add_argument("--complete-collection", type=Path, required=True)
    parser.add_argument("--trajectory-metadata", type=Path, required=True)
    parser.add_argument("--trajectory-array", type=Path, required=True)
    parser.add_argument("--trajectory-collection", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--factorized-result", type=Path, required=True)
    parser.add_argument("--factorized-validation", type=Path, required=True)
    parser.add_argument("--surface-result", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--same-task-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)


def resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "factorized_config": args.factorized_config,
        "complete_dataset": args.complete_dataset,
        "complete_collection": args.complete_collection,
        "metadata": args.trajectory_metadata, "array": args.trajectory_array,
        "collection": args.trajectory_collection,
        "baseline_model": args.baseline_model,
        "baseline_predictions": args.baseline_predictions,
        "factorized_result": args.factorized_result,
        "factorized_validation": args.factorized_validation,
        "surface_result": args.surface_result,
        "population": args.population_manifest,
        "selected": args.selected_manifest,
        "same_task": args.same_task_manifest,
        "targeted": args.targeted_manifest,
        "archived": args.archived_root,
        "geometry": args.geometry_config, "exact_box": args.exact_box_config,
    }.items()}


def load_inputs(paths: Mapping[str, Path]) -> tuple[Any, ...]:
    import numpy as np

    config = load_one_sided_config(paths["config"])
    source = config["immutable_source"]
    for source_key, path_key in (
        ("factorized_config_file_sha256", "factorized_config"),
        ("trajectory_array_file_sha256", "array"),
        ("trajectory_metadata_file_sha256", "metadata"),
        ("trajectory_collection_file_sha256", "collection"),
        ("baseline_model_file_sha256", "baseline_model"),
        ("baseline_predictions_file_sha256", "baseline_predictions"),
        ("factorized_result_file_sha256", "factorized_result"),
        ("factorized_validation_file_sha256", "factorized_validation"),
        ("surface_result_file_sha256", "surface_result"),
    ):
        _require(
            _file_sha256(paths[path_key]) == source[source_key],
            "one-sided geometry immutable source differs",
        )
    factorized_config = load_config(paths["factorized_config"])
    complete_dataset = _load(paths["complete_dataset"])
    complete_collection = _load(paths["complete_collection"])
    metadata = _load(paths["metadata"])
    collection = _load(paths["collection"])
    result = _load(paths["factorized_result"])
    validation = _load(paths["factorized_validation"])
    surface = _load(paths["surface_result"])
    _require(
        metadata.get("schema_version") == DATASET_SCHEMA
        and metadata.get("dataset_payload_sha256")
        == payload_sha256(metadata, "dataset_payload_sha256")
        and metadata["array_dataset"]["file_sha256"] == _file_sha256(paths["array"])
        and collection.get("schema_version") == COLLECTION_SCHEMA
        and collection.get("result_payload_sha256")
        == payload_sha256(collection, "result_payload_sha256")
        and collection.get("decision", {}).get("collection_gate_pass") is True
        and result.get("result_payload_sha256")
        == source["factorized_result_payload_sha256"]
        and result.get("decision", {}).get("factorization_GO") is False
        and result["metrics"]["factorized_margin"]["false_safe_action_count"] == 44
        and validation.get("validation_payload_sha256")
        == source["factorized_validation_payload_sha256"]
        and validation.get("valid") is True
        and surface.get("result_payload_sha256")
        == source["surface_result_payload_sha256"]
        and surface.get("decision", {}).get("surface_loss_pilot_supported") is False,
        "one-sided geometry validated source differs",
    )
    archive = np.load(paths["array"], allow_pickle=False)
    arrays = dataset_arrays(metadata, archive)
    sensitivities = sensitivity_arrays(arrays)
    models, state = load_weights(paths["baseline_model"])
    baseline_q = predict(models, state, arrays)
    stored = np.load(paths["baseline_predictions"], allow_pickle=False)[
        "factorized_joint_position_rad"
    ].astype(np.float64)
    _require(
        _hash_array(baseline_q) == source["baseline_joint_prediction_sha256"]
        and np.array_equal(baseline_q, stored),
        "one-sided geometry baseline prediction differs",
    )
    return (
        config, factorized_config, complete_dataset, complete_collection,
        metadata, collection, result, validation, surface, arrays,
        sensitivities, baseline_q,
    )


def geometry_kwargs(
    paths: Mapping[str, Path], factorized_config: Mapping[str, Any],
    complete_dataset: Mapping[str, Any], complete_collection: Mapping[str, Any],
    arrays: Mapping[str, Any], predicted_q: Any, splits: tuple[str, ...],
) -> dict[str, Any]:
    return evaluate_predicted_geometry(
        repo_root=paths["repo"], config=factorized_config,
        complete_dataset=complete_dataset, arrays=arrays,
        predicted_q=predicted_q, population_manifest=paths["population"],
        selected_manifest=paths["selected"],
        same_task_manifest=paths["same_task"],
        targeted_manifest=paths["targeted"], archived_root=paths["archived"],
        geometry_config_path=paths["geometry"],
        exact_box_config_path=paths["exact_box"],
        complete_collection=complete_collection, evaluation_splits=splits,
    )


def validation_pattern(
    predicted_margin: Any, arrays: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    predicted = np.asarray(predicted_margin, dtype=np.float64)
    exact_trace = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    exact_minimum = np.min(exact_trace, axis=1)
    selected = (
        (np.asarray(arrays["split"], dtype=object) == "validation")
        & (np.asarray(arrays["source_code"], dtype=np.int8) == 2)
    )
    predicted_safe = np.all(predicted >= 0.0, axis=1)
    exact_safe = np.all(exact_minimum >= 0.0, axis=1)
    rows = np.flatnonzero(selected & predicted_safe & ~exact_safe)
    terminal = int(config["validation_pattern_gate"]["terminal_substep_index"])
    l5_rows = set(config["validation_pattern_gate"]["L5_constraint_rows"])
    terminal_l5 = 0
    localization = []
    for row in rows:
        substep, constraint = np.unravel_index(
            int(np.argmin(exact_trace[row])), (51, 7),
        )
        terminal_l5 += int(substep == terminal and constraint in l5_rows)
        localization.append({
            "row_index": int(row),
            "state_index": int(np.asarray(arrays["state_index"])[row]),
            "candidate_index": int(np.asarray(arrays["candidate_index"])[row]),
            "worst_substep_index": int(substep),
            "worst_constraint_row": int(constraint),
            "exact_worst_margin_m": float(exact_trace[row, substep, constraint]),
        })
    fraction = 0.0 if not len(rows) else float(terminal_l5 / len(rows))
    return {
        "random_action_count": int(np.count_nonzero(selected)),
        "false_safe_action_count": int(len(rows)),
        "terminal_L5_false_safe_count": int(terminal_l5),
        "terminal_L5_false_safe_fraction": fraction,
        "false_safe_localization": localization,
    }


def prediction_metrics(
    predicted_q: Any, predicted_margin: Any, arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any], factorized_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    test_random = (
        (np.asarray(arrays["split"], dtype=object) == "test")
        & (np.asarray(arrays["source_code"], dtype=np.int8) == 2)
    )
    error = np.asarray(predicted_q)[test_random] - np.asarray(
        arrays["joint_position_rad"]
    )[test_random]
    q_secant = secant_predictions(predicted_q, sensitivities, arrays)
    margin_secant = secant_predictions(predicted_margin, sensitivities, arrays)
    sensitivity_test = _test_sensitivity_mask(sensitivities, arrays)
    joint_sensitivity = cosine_summary(
        np.asarray(sensitivities["joint_sensitivity_rad_per_action"])[sensitivity_test],
        q_secant[sensitivity_test],
    )
    margin_sensitivity = cosine_summary(
        np.asarray(sensitivities["margin_sensitivity_m_per_action"])[sensitivity_test],
        margin_secant[sensitivity_test],
    )
    return {
        "test_joint_trajectory_RMSE_rad": float(np.sqrt(np.mean(error ** 2))),
        "test_endpoint_RMSE_rad": float(np.sqrt(np.mean(error[:, -1] ** 2))),
        "joint_sensitivity": joint_sensitivity,
        "margin_sensitivity": margin_sensitivity,
        "test_safety": safety_metrics(
            arrays["minimum_margin_m"], predicted_margin, arrays,
            factorized_config, split_name="test", evaluation_only=True,
        ),
        "validation_safety": safety_metrics(
            arrays["minimum_margin_m"], predicted_margin, arrays,
            factorized_config, split_name="validation", evaluation_only=True,
        ),
    }


def main() -> int:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = resolved_paths(args)
    paths.update({
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "output": args.output.resolve(),
    })
    (
        config, factorized_config, complete_dataset, complete_collection,
        _, collection, baseline_result, _, _, arrays, sensitivities, baseline_q,
    ) = load_inputs(paths)
    torch.set_num_threads(8)
    local_geometry = fit_local_geometry_jacobians(arrays, config)
    baseline_validation_geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        arrays, baseline_q, ("validation",),
    )
    baseline_validation_margin = baseline_validation_geometry.pop(
        "predicted_minimum_margin_m"
    )
    baseline_validation_geometry.pop("exact_q_static_minimum_margin_m")
    pattern = validation_pattern(baseline_validation_margin, arrays, config)
    pattern_pass = (
        pattern["false_safe_action_count"]
        >= config["validation_pattern_gate"]["minimum_validation_false_safe_action_count"]
        and pattern["terminal_L5_false_safe_fraction"]
        >= config["validation_pattern_gate"]["minimum_terminal_L5_fraction"]
    )
    linearization_pass = (
        local_geometry["audit"]["validation"]["linearization_RMSE_m"]
        <= config["local_geometry_jacobian"][
            "maximum_validation_random_linearization_RMSE_m"
        ]
    )
    if not (pattern_pass and linearization_pass):
        decision = {
            "gate_tests": {
                "validation_has_false_safe": pattern[
                    "false_safe_action_count"
                ] >= config["validation_pattern_gate"][
                    "minimum_validation_false_safe_action_count"
                ],
                "validation_terminal_L5_pattern": pattern[
                    "terminal_L5_false_safe_fraction"
                ] >= config["validation_pattern_gate"][
                    "minimum_terminal_L5_fraction"
                ],
                "validation_linearization": linearization_pass,
            },
            "one_sided_geometry_GO": False,
            "conclusion": "conditional_validation_gate_blocks_training",
            "uncertainty_QP_or_closed_loop_authorized": False,
        }
        result = {
            "schema_version": RESULT_SCHEMA, "status": "complete",
            "scientific_result": True, "claim_scope": config["claim_scope"],
            "source": _git_identity(paths["repo"], args.expected_commit),
            "allocation": allocation_record(), "config": config,
            "immutable_baseline_result": {
                "path": str(paths["factorized_result"]),
                "file_sha256": _file_sha256(paths["factorized_result"]),
                "test_metrics": baseline_result["metrics"]["factorized_margin"],
            },
            "validation_pattern_audit": pattern,
            "baseline_validation_geometry": baseline_validation_geometry,
            "local_geometry_jacobian_audit": local_geometry["audit"],
            "training": {"executed": False}, "model": None,
            "predictions": None, "metrics": None,
            "forbidden_action_receipt": {
                "new_simulation_labels_collected": False,
                "surface_position_loss_executed": False,
                "binary_classifier_executed": False,
                "uncertainty_calibration_executed": False,
                "poisson_or_SDF_executed": False,
                "QP_executed": False, "closed_loop_executed": False,
            },
            "decision": decision,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = payload_sha256(
            result, "result_payload_sha256"
        )
        _atomic_write(paths["output"], result)
        print(json.dumps({
            "validation_pattern": pattern,
            "local_geometry": local_geometry["audit"],
            "decision": decision,
        }, sort_keys=True), flush=True)
        return 0
    models, state, training = train_one_sided_geometry_ensemble(
        arrays, sensitivities, local_geometry, factorized_config, config,
    )
    experimental_q = predict(models, state, arrays)
    model_artifact = save_weights(paths["experimental_model"], state)
    experimental_geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        arrays, experimental_q, ("validation", "test"),
    )
    experimental_margin = experimental_geometry.pop("predicted_minimum_margin_m")
    exact_q_static_margin = experimental_geometry.pop(
        "exact_q_static_minimum_margin_m"
    )
    metrics = prediction_metrics(
        experimental_q, experimental_margin, arrays, sensitivities,
        factorized_config,
    )
    baseline_test = baseline_result["metrics"]["factorized_margin"]
    decision = one_sided_decision(
        validation_audit=pattern,
        jacobian_audit=local_geometry["audit"], baseline_test=baseline_test,
        experimental_test=metrics["test_safety"],
        joint_sensitivity_cosine=metrics["joint_sensitivity"]["mean_cosine"],
        margin_sensitivity_cosine=metrics["margin_sensitivity"]["mean_cosine"],
        config=config,
    )
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["predictions"],
        experimental_joint_position_rad=experimental_q,
        experimental_minimum_margin_m=experimental_margin,
        exact_q_static_minimum_margin_m=exact_q_static_margin,
        baseline_validation_minimum_margin_m=baseline_validation_margin,
    )
    prediction_receipt = {
        "path": str(paths["predictions"]),
        "file_sha256": _file_sha256(paths["predictions"]),
        "experimental_joint_sha256": _hash_array(experimental_q),
        "experimental_margin_sha256": _hash_array(experimental_margin),
        "exact_q_static_margin_sha256": _hash_array(exact_q_static_margin),
        "baseline_validation_margin_sha256": _hash_array(
            baseline_validation_margin
        ),
    }
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_baseline_result": {
            "path": str(paths["factorized_result"]),
            "file_sha256": _file_sha256(paths["factorized_result"]),
            "test_metrics": baseline_test,
        },
        "validation_pattern_audit": pattern,
        "baseline_validation_geometry": baseline_validation_geometry,
        "local_geometry_jacobian_audit": local_geometry["audit"],
        "training": {"executed": True, **training}, "model": model_artifact,
        "predictions": prediction_receipt,
        "metrics": {
            **metrics,
            "experimental_geometry": experimental_geometry,
        },
        "forbidden_action_receipt": {
            "new_simulation_labels_collected": False,
            "surface_position_loss_executed": False,
            "binary_classifier_executed": False,
            "uncertainty_calibration_executed": False,
            "poisson_or_SDF_executed": False,
            "QP_executed": False, "closed_loop_executed": False,
        },
        "decision": decision,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "validation_pattern": pattern,
        "local_geometry": local_geometry["audit"],
        "test": metrics["test_safety"],
        "validation": metrics["validation_safety"],
        "joint_sensitivity": metrics["joint_sensitivity"],
        "margin_sensitivity": metrics["margin_sensitivity"],
        "decision": decision,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
