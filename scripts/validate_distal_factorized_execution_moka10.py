#!/usr/bin/env python3
"""Independently replay model and geometry metrics for the factorized pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from main.multilink_ellipsoid.factorized_execution_pilot import (
    COLLECTION_SCHEMA, DATASET_SCHEMA, RESULT_SCHEMA, VALIDATION_SCHEMA,
    cosine_summary, dataset_arrays, factorized_decision, load_config,
    load_weights, payload_sha256, predict, safety_metrics,
    secant_predictions, sensitivity_arrays,
)
from scripts.evaluate_distal_factorized_execution_moka10 import (
    _test_sensitivity_mask, evaluate_predicted_geometry,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_array(value: object) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--complete-dataset", type=Path, required=True)
    parser.add_argument("--complete-collection", type=Path, required=True)
    parser.add_argument("--trajectory-metadata", type=Path, required=True)
    parser.add_argument("--trajectory-array", type=Path, required=True)
    parser.add_argument("--trajectory-collection", type=Path, required=True)
    parser.add_argument("--direct-model", type=Path, required=True)
    parser.add_argument("--factorized-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--same-task-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "complete_dataset": args.complete_dataset,
        "complete_collection": args.complete_collection,
        "metadata": args.trajectory_metadata, "array": args.trajectory_array,
        "collection": args.trajectory_collection,
        "direct_model": args.direct_model,
        "factorized_model": args.factorized_model,
        "predictions": args.predictions,
        "population": args.population_manifest, "selected": args.selected_manifest,
        "same_task": args.same_task_manifest, "targeted": args.targeted_manifest,
        "archived": args.archived_root, "geometry": args.geometry_config,
        "exact_box": args.exact_box_config, "result": args.result,
        "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    complete_dataset = _load(paths["complete_dataset"])
    complete_collection = _load(paths["complete_collection"])
    metadata = _load(paths["metadata"])
    collection = _load(paths["collection"])
    result = _load(paths["result"])
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit
        and metadata.get("schema_version") == DATASET_SCHEMA
        and metadata.get("dataset_payload_sha256")
        == payload_sha256(metadata, "dataset_payload_sha256")
        and collection.get("schema_version") == COLLECTION_SCHEMA
        and collection.get("result_payload_sha256")
        == payload_sha256(collection, "result_payload_sha256")
        and result["models"]["direct_margin"]["file_sha256"]
        == _file_sha256(paths["direct_model"])
        and result["models"]["factorized_execution"]["file_sha256"]
        == _file_sha256(paths["factorized_model"])
        and result["predictions"]["file_sha256"]
        == _file_sha256(paths["predictions"]),
        "factorized-execution result identity differs",
    )
    dataset_archive = np.load(paths["array"], allow_pickle=False)
    arrays = dataset_arrays(metadata, dataset_archive)
    sensitivities = sensitivity_arrays(arrays)
    direct_models, direct_state = load_weights(paths["direct_model"])
    factor_models, factor_state = load_weights(paths["factorized_model"])
    direct_prediction = predict(direct_models, direct_state, arrays)
    factor_q_prediction = predict(factor_models, factor_state, arrays)
    prediction_archive = np.load(paths["predictions"], allow_pickle=False)
    stored_direct = prediction_archive["direct_margin_m"].astype(np.float64)
    stored_q = prediction_archive["factorized_joint_position_rad"].astype(np.float64)
    stored_margin = prediction_archive["factorized_minimum_margin_m"].astype(np.float64)
    stored_exact_q_static_margin = prediction_archive[
        "exact_q_static_minimum_margin_m"
    ].astype(np.float64)
    prediction_match = bool(
        np.array_equal(direct_prediction, stored_direct)
        and np.array_equal(factor_q_prediction, stored_q)
        and _hash_array(direct_prediction)
        == result["predictions"]["direct_margin_sha256"]
        and _hash_array(factor_q_prediction)
        == result["predictions"]["factorized_joint_sha256"]
        and _hash_array(stored_margin)
        == result["predictions"]["factorized_margin_sha256"]
        and _hash_array(stored_exact_q_static_margin)
        == result["predictions"]["exact_q_static_margin_sha256"]
    )
    direct_metrics = {
        split: safety_metrics(
            arrays["minimum_margin_m"], direct_prediction, arrays, config,
            split_name=split, evaluation_only=(split == "test"),
        ) for split in ("train", "validation", "test")
    }
    test_random = (
        (np.asarray(arrays["split"], dtype=object) == "test")
        & (np.asarray(arrays["source_code"], dtype=np.int8) == 2)
    )
    joint_error = factor_q_prediction[test_random] - np.asarray(
        arrays["joint_position_rad"]
    )[test_random]
    joint_metrics = {
        "test_random_row_count": int(np.count_nonzero(test_random)),
        "test_random_joint_trajectory_RMSE_rad": float(np.sqrt(np.mean(joint_error ** 2))),
        "test_random_joint_absolute_error_p95_rad": float(np.quantile(
            np.abs(joint_error), 0.95
        )),
        "test_random_joint_absolute_error_maximum_rad": float(np.max(
            np.abs(joint_error)
        )),
        "test_random_endpoint_RMSE_rad": float(np.sqrt(np.mean(
            joint_error[:, -1] ** 2
        ))),
    }
    sensitivity_test = _test_sensitivity_mask(sensitivities, arrays)
    q_secant = secant_predictions(factor_q_prediction, sensitivities, arrays)
    direct_margin_secant = secant_predictions(
        direct_prediction, sensitivities, arrays
    )
    joint_sensitivity = cosine_summary(
        np.asarray(sensitivities["joint_sensitivity_rad_per_action"])[sensitivity_test],
        q_secant[sensitivity_test],
    )
    direct_margin_sensitivity = cosine_summary(
        np.asarray(sensitivities["margin_sensitivity_m_per_action"])[sensitivity_test],
        direct_margin_secant[sensitivity_test],
    )
    geometry = evaluate_predicted_geometry(
        repo_root=paths["repo"], config=config,
        complete_dataset=complete_dataset, arrays=arrays,
        predicted_q=factor_q_prediction,
        population_manifest=paths["population"],
        selected_manifest=paths["selected"],
        same_task_manifest=paths["same_task"],
        targeted_manifest=paths["targeted"], archived_root=paths["archived"],
        geometry_config_path=paths["geometry"],
        exact_box_config_path=paths["exact_box"],
        complete_collection=complete_collection,
    )
    recomputed_margin = geometry.pop("predicted_minimum_margin_m")
    recomputed_exact_q_static_margin = geometry.pop(
        "exact_q_static_minimum_margin_m"
    )
    geometry_match = bool(
        np.array_equal(recomputed_margin, stored_margin)
        and np.array_equal(
            recomputed_exact_q_static_margin, stored_exact_q_static_margin
        )
    )
    exact_geometry_metrics = safety_metrics(
        arrays["minimum_margin_m"], recomputed_exact_q_static_margin,
        arrays, config, split_name="test", evaluation_only=True,
    )
    factor_metrics = safety_metrics(
        arrays["minimum_margin_m"], recomputed_margin, arrays, config,
        split_name="test", evaluation_only=True,
    )
    factor_margin_secant = secant_predictions(
        recomputed_margin, sensitivities, arrays
    )
    factor_margin_sensitivity = cosine_summary(
        np.asarray(sensitivities["margin_sensitivity_m_per_action"])[sensitivity_test],
        factor_margin_secant[sensitivity_test],
    )
    test_raw_contact = np.asarray(arrays["raw_contact_count"])[test_random] > 0
    factor_safe = np.all(recomputed_margin[test_random] >= 0.0, axis=1)
    physical_false_safe = int(np.count_nonzero(factor_safe & test_raw_contact))
    decision = factorized_decision(
        collection_pass=bool(collection["decision"]["collection_gate_pass"]),
        direct_metrics=direct_metrics["test"],
        exact_geometry_metrics=exact_geometry_metrics,
        factorized_metrics=factor_metrics,
        joint_rmse_rad=joint_metrics["test_random_joint_trajectory_RMSE_rad"],
        link_center_rmse_m=geometry["test_random_link_center_RMSE_m"],
        joint_sensitivity_cosine=float(joint_sensitivity["mean_cosine"]),
        margin_sensitivity_cosine=float(factor_margin_sensitivity["mean_cosine"]),
        config=config,
    )
    metrics_match = bool(
        direct_metrics == result["metrics"]["direct_margin"]
        and joint_metrics == result["metrics"]["factorized_joint"]
        and geometry == result["metrics"]["factorized_geometry"]
        and exact_geometry_metrics
        == result["metrics"]["exact_q_static_geometry_margin"]
        and factor_metrics == result["metrics"]["factorized_margin"]
        and joint_sensitivity == result["metrics"]["joint_sensitivity"]
        and direct_margin_sensitivity
        == result["metrics"]["direct_margin_sensitivity"]
        and factor_margin_sensitivity
        == result["metrics"]["factorized_margin_sensitivity"]
        and physical_false_safe
        == result["metrics"]["factorized_raw_contact_false_safe_action_count"]
        and decision == result["decision"]
    )
    forbidden = result.get("forbidden_action_receipt", {})
    forbidden_clean = bool(forbidden and not any(forbidden.values()))
    valid = bool(
        prediction_match and geometry_match and metrics_match and forbidden_clean
    )
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "complete",
        "valid": valid,
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "audit": {
            "model_predictions_exactly_reproduced": prediction_match,
            "MuJoCo_FK_ellipsoid_margins_exactly_reproduced": geometry_match,
            "all_metrics_and_decision_exactly_reproduced": metrics_match,
            "forbidden_actions_absent": forbidden_clean,
            "decision": decision,
        },
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
