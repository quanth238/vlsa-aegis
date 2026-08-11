#!/usr/bin/env python3
"""Independently validate the matched normalized paired-secant ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    load_direct_horizon_config, load_direct_horizon_weights,
    normalized_secant_fitted_gate, predict_direct_horizon,
)
from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    _hash_array, add_direct_horizon_arguments, direct_paths,
    validate_direct_sources,
)
from scripts.evaluate_distal_direct_horizon_normalized_secant_moka10 import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, compute_metrics, validate_matched_sources,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.validate_distal_direct_horizon_displacement_moka10 import (
    _array_equal_with_nan,
)


def main() -> int:
    import numpy as np
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_direct_horizon_arguments(parser)
    parser.add_argument("--matched-direct-run", type=Path, required=True)
    parser.add_argument("--matched-direct-validation", type=Path, required=True)
    parser.add_argument("--root-cause-run", type=Path, required=True)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, _, _, factorized_config, complete_dataset,
        complete_collection, arrays, sensitivities, _, representation,
        local_geometry,
    ) = prepare(args)
    paths.update(direct_paths(args))
    paths.update({
        "matched_direct_run": args.matched_direct_run.resolve(),
        "matched_direct_validation": args.matched_direct_validation.resolve(),
        "root_cause_run": args.root_cause_run.resolve(),
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "result": args.result.resolve(), "output": args.output.resolve(),
    })
    config = load_direct_horizon_config(paths["direct_config"])
    _require(
        config["protocol_id"]
        == "vlsa-distal-factorized-direct-horizon-normalized-secant-moka10-v1",
        "normalized-secant validation protocol differs",
    )
    validate_direct_sources(paths, config)
    _, root_result = validate_matched_sources(paths, config)
    result = _load(paths["result"])
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit")
        == args.expected_result_commit
        and result.get("model", {}).get("file_sha256")
        == _file_sha256(paths["experimental_model"])
        and result.get("predictions", {}).get("file_sha256")
        == _file_sha256(paths["predictions"]),
        "normalized-secant result identity differs",
    )
    _require(
        int(representation["structured_input_dimension"])
        == int(arrays["features"].shape[1])
        and result["local_geometry_jacobian_audit"]
        == local_geometry["audit"],
        "normalized-secant validation representation differs",
    )
    models, state = load_direct_horizon_weights(paths["experimental_model"])
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
    stored = np.load(paths["predictions"], allow_pickle=False)
    pairs = {
        "joint_position_rad": (predicted_q, stored["joint_position_rad"]),
        "predicted_clearance_trace_m": (
            predicted_trace, stored["predicted_clearance_trace_m"]
        ),
        "predicted_minimum_margin_m": (
            predicted_minimum, stored["predicted_minimum_margin_m"]
        ),
        "exact_q_static_clearance_trace_m": (
            exact_static_trace, stored["exact_q_static_clearance_trace_m"]
        ),
        "exact_q_static_minimum_margin_m": (
            exact_static_minimum, stored["exact_q_static_minimum_margin_m"]
        ),
    }
    array_audit = {
        name: {
            "equal_with_matching_nan_mask": bool(equal),
            "maximum_finite_difference": float(maximum),
        }
        for name, (equal, maximum) in (
            (name, _array_equal_with_nan(*pair))
            for name, pair in pairs.items()
        )
    }
    arrays_equal = bool(all(
        item["equal_with_matching_nan_mask"]
        for item in array_audit.values()
    ))
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
    metrics_equal = bool(
        metrics == result["metrics"]
        and decision == result["decision"]
        and geometry == result["geometry"]
        and _hash_array(predicted_q)
        == result["predictions"]["joint_sha256"]
        and _hash_array(predicted_trace)
        == result["predictions"]["clearance_trace_sha256"]
    )
    valid = bool(arrays_equal and metrics_equal)
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
            "stored_arrays_exactly_reproduced": arrays_equal,
            "stored_array_audit": array_audit,
            "metrics_geometry_and_decision_exactly_reproduced": metrics_equal,
            "action_count_recomputed": int(len(predicted_q)),
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
