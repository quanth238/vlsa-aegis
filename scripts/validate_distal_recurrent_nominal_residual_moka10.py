#!/usr/bin/env python3
"""Independently replay the recurrent nominal-plus-residual fitted gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.factorized_recurrent_nominal_residual import (
    fitted_prediction_gate, load_recurrent_config, load_recurrent_weights,
    predict_recurrent,
)
from scripts.evaluate_distal_direct_horizon_normalized_secant_moka10 import (
    compute_metrics,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs,
)
from scripts.evaluate_distal_recurrent_nominal_residual_moka10 import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, add_arguments, fresh_direct_baseline,
    recurrent_paths, support_metrics, validate_recurrent_sources,
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
    add_arguments(parser)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-commit", required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, _, _, factorized_config, complete_dataset,
        complete_collection, arrays, sensitivities, _, representation,
        local_geometry,
    ) = prepare(args)
    paths.update(recurrent_paths(args))
    paths["result"] = args.result.resolve()
    config = load_recurrent_config(paths["recurrent_config"])
    direct_result = validate_recurrent_sources(paths, config)
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
        "recurrent result identity differs",
    )
    models, state = load_recurrent_weights(paths["experimental_model"])
    predicted_q, _ = predict_recurrent(models, state, arrays)
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
    array_audit = {}
    for name, pair in pairs.items():
        equal, maximum = _array_equal_with_nan(*pair)
        array_audit[name] = {
            "equal_with_matching_nan_mask": bool(equal),
            "maximum_finite_difference": float(maximum),
        }
    arrays_equal = bool(all(
        item["equal_with_matching_nan_mask"] for item in array_audit.values()
    ))
    metrics = compute_metrics(
        arrays=arrays, predicted_q=predicted_q,
        predicted_trace=predicted_trace, sensitivities=sensitivities,
        factorized_config=factorized_config,
    )
    support = support_metrics(arrays, predicted_minimum)
    decision = fitted_prediction_gate(
        metrics=metrics, support=support, config=config,
    )
    direct_metrics, direct_support = fresh_direct_baseline(
        paths=paths, arrays=arrays, sensitivities=sensitivities,
        factorized_config=factorized_config,
    )
    reproduced = bool(
        metrics == result["metrics"]
        and support == result["support"]
        and decision == result["decision"]
        and geometry == result["geometry"]
        and direct_metrics == direct_result["metrics"]
        and direct_metrics
        == result["matched_comparison"]["direct_metrics"]
        and direct_support
        == result["matched_comparison"]["direct_support"]
        and representation == result["representation"]
        and local_geometry["audit"]
        == result["local_geometry_jacobian_audit"]
    )
    valid = bool(arrays_equal and reproduced)
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
            "metrics_geometry_support_decision_exactly_reproduced": reproduced,
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
