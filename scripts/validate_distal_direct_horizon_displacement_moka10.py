#!/usr/bin/env python3
"""Independently reload and validate direct-horizon prediction evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, apply_structured_representation,
    direct_horizon_decision, load_direct_horizon_config,
    load_direct_horizon_weights, predict_direct_horizon,
)
from main.multilink_ellipsoid.factorized_execution_pilot import (
    payload_sha256, safety_metrics, sensitivity_arrays,
)
from main.multilink_ellipsoid.factorized_structured_orientation import (
    load_structured_config,
)
from main.multilink_ellipsoid.factorized_time_conditioned_decoder import (
    eligible_state_support,
)
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    _hash_array, add_direct_horizon_arguments, direct_paths,
    evaluate_reserved_geometry, load_reserved, model_metrics,
    validate_direct_sources,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def main() -> int:
    import numpy as np
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_direct_horizon_arguments(parser)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, _, _, factorized_config, complete_dataset,
        complete_collection, training_arrays, _, _, representation, _,
    ) = prepare(args)
    paths.update(direct_paths(args))
    paths.update({
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "result": args.result.resolve(), "output": args.output.resolve(),
    })
    config = load_direct_horizon_config(paths["direct_config"])
    structured_config = load_structured_config(paths["structured_config"])
    validate_direct_sources(paths, config)
    result = _load(paths["result"])
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result["source"]["commit"] == args.expected_commit
        and result["model"]["file_sha256"]
        == _file_sha256(paths["experimental_model"])
        and result["predictions"]["file_sha256"]
        == _file_sha256(paths["predictions"]),
        "direct-horizon result identity differs",
    )
    _, reserved_metadata, reserved_raw = load_reserved(paths, config)
    reserved_arrays, reserved_representation = apply_structured_representation(
        complete_dataset, reserved_raw, structured_config,
    )
    _require(
        reserved_representation["removed_indexes_sha256"]
        == representation["removed_feature_indexes_sha256"]
        and reserved_representation["retained_indexes_sha256"]
        == representation["retained_feature_indexes_sha256"],
        "direct-horizon validation representation differs",
    )
    models, state = load_direct_horizon_weights(paths["experimental_model"])
    reserved_normalized = (
        np.asarray(reserved_arrays["features"], dtype=np.float64)
        - np.asarray(state["feature_mean"], dtype=np.float64)
    ) / np.asarray(state["feature_std"], dtype=np.float64)
    _require(
        _hash_array(reserved_normalized)
        == result["reserved_population_representation"][
            "training_normalized_features_sha256"
        ],
        "direct-horizon normalized reserved representation differs",
    )
    reserved_q = predict_direct_horizon(models, state, reserved_arrays)
    training_q = predict_direct_horizon(models, state, training_arrays)
    stored = np.load(paths["predictions"], allow_pickle=False)
    _require(
        np.array_equal(
            reserved_q, stored["direct_horizon_joint_position_rad"]
        )
        and _hash_array(reserved_q)
        == result["predictions"]["direct_horizon_joint_sha256"],
        "direct-horizon joint prediction does not reproduce",
    )
    validation_geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        training_arrays, training_q, ("validation",),
    )
    validation_margin = validation_geometry.pop("predicted_minimum_margin_m")
    validation_exact_static = validation_geometry.pop(
        "exact_q_static_minimum_margin_m"
    )
    reserved_geometry = evaluate_reserved_geometry(
        paths=paths, factorized_config=factorized_config,
        complete_collection=complete_collection, metadata=reserved_metadata,
        arrays=reserved_raw,
        predictions={"direct_horizon_displacement": reserved_q},
    )
    margins = reserved_geometry.pop("minimum_margin_m")
    reserved_margin = margins["direct_horizon_displacement"]
    exact_static = margins["exact_q_static"]
    stored_equal = bool(
        np.array_equal(
            validation_margin,
            stored["validation_direct_horizon_minimum_margin_m"],
        )
        and np.array_equal(
            validation_exact_static,
            stored["validation_exact_q_static_minimum_margin_m"],
        )
        and np.array_equal(
            reserved_margin, stored["direct_horizon_minimum_margin_m"],
        )
        and np.array_equal(
            exact_static, stored["exact_q_static_minimum_margin_m"],
        )
    )
    validation_metrics = safety_metrics(
        training_arrays["minimum_margin_m"], validation_margin,
        training_arrays, factorized_config, split_name="validation",
        evaluation_only=True,
    )
    validation_support = eligible_state_support(
        exact_margin=training_arrays["minimum_margin_m"],
        predicted_margin=validation_margin, arrays=training_arrays,
        split_name="validation",
    )
    reserved_metrics = model_metrics(
        predicted_q=reserved_q, predicted_margin=reserved_margin,
        arrays=reserved_raw, sensitivities=sensitivity_arrays(reserved_raw),
        factorized_config=factorized_config,
    )
    exact_static_metrics = safety_metrics(
        reserved_raw["minimum_margin_m"], exact_static, reserved_raw,
        factorized_config, split_name="test", evaluation_only=True,
    )
    source_comparison = result["metrics"]["reserved_prediction_comparison"]
    decision = direct_horizon_decision(
        validation_metrics=validation_metrics,
        reserved_metrics=reserved_metrics["safety"],
        reserved_exact_static_metrics=exact_static_metrics,
        validation_support=validation_support,
        reserved_support=reserved_metrics["eligible_support"],
        reserved_temporal=reserved_metrics["random_temporal_joint"],
        joint_sensitivity=reserved_metrics["joint_sensitivity_cosine"],
        safety_sensitivity=reserved_metrics["safety_sensitivity_cosine"],
        matched_cumulative=source_comparison[
            "previous_cumulative_increments"
        ]["random_temporal_joint"],
        config=config,
    )
    metric_equal = bool(
        validation_metrics
        == result["metrics"]["validation_direct_horizon_safety"]
        and validation_support
        == result["metrics"]["validation_direct_horizon_support"]
        and reserved_metrics
        == source_comparison["direct_horizon_displacement"]
        and decision == result["decision"]
        and reserved_geometry["geometry_metrics"][
            "direct_horizon_displacement"
        ] == result["metrics"]["reserved_geometry"]["geometry_metrics"][
            "direct_horizon_displacement"
        ]
    )
    valid = bool(stored_equal and metric_equal)
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "complete",
        "valid": valid,
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(),
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "model_file_sha256": _file_sha256(paths["experimental_model"]),
        "predictions_file_sha256": _file_sha256(paths["predictions"]),
        "audit": {
            "stored_prediction_arrays_exactly_reproduced": stored_equal,
            "metrics_and_decision_exactly_reproduced": metric_equal,
            "reserved_action_count_recomputed": int(len(reserved_q)),
            "validation_prediction_sha256": _hash_array(validation_margin),
            "reserved_prediction_sha256": _hash_array(reserved_margin),
            "decision": decision,
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
