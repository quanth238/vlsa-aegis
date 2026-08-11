#!/usr/bin/env python3
"""Independently replay the explicit execution-Jacobian prediction result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    normalized_secant_fitted_gate,
)
from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.factorized_explicit_execution_jacobian import (
    load_explicit_jacobian_config, load_explicit_jacobian_weights,
)
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    _hash_array, direct_paths, validate_direct_sources,
)
from scripts.evaluate_distal_direct_horizon_normalized_secant_moka10 import (
    validate_matched_sources,
)
from scripts.evaluate_distal_explicit_execution_jacobian_moka10 import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, add_arguments, explicit_decision,
    extra_paths, run_prediction, validate_normalized_source,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
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
    paths.update(direct_paths(args))
    paths.update(extra_paths(args))
    paths["result"] = args.result.resolve()
    config = load_explicit_jacobian_config(paths["direct_config"])
    validate_direct_sources(paths, config)
    _, root_result = validate_matched_sources(paths, config)
    validate_normalized_source(paths, config)
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
        "explicit-J result identity differs",
    )
    _require(
        int(representation["structured_input_dimension"])
        == int(arrays["features"].shape[1])
        and result["local_geometry_jacobian_audit"]
        == local_geometry["audit"],
        "explicit-J validation representation differs",
    )
    models, state = load_explicit_jacobian_weights(
        paths["experimental_model"]
    )
    predicted_q, geometry, metrics, consistency, recomputed = run_prediction(
        paths=paths, factorized_config=factorized_config,
        complete_dataset=complete_dataset,
        complete_collection=complete_collection, arrays=arrays,
        sensitivities=sensitivities, local_geometry=local_geometry,
        config=config, models=models, state=state,
    )
    stored = np.load(paths["predictions"], allow_pickle=False)
    array_audit = {}
    for name, value in recomputed.items():
        equal, maximum = _array_equal_with_nan(value, stored[name])
        array_audit[name] = {
            "equal_with_matching_nan_mask": bool(equal),
            "maximum_finite_difference": float(maximum),
        }
    arrays_equal = bool(all(
        item["equal_with_matching_nan_mask"]
        for item in array_audit.values()
    ))
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
    metrics_equal = bool(
        metrics == result["metrics"]
        and decision == result["decision"]
        and consistency == result["explicit_jacobian_consistency"]
        and geometry == result["geometry"]
        and _hash_array(predicted_q)
        == result["predictions"]["joint_sha256"]
        and _hash_array(recomputed["predicted_clearance_trace_m"])
        == result["predictions"]["clearance_trace_sha256"]
        and _hash_array(recomputed["execution_jacobian_rad_per_action"])
        == result["predictions"]["execution_jacobian_sha256"]
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
            "metrics_geometry_consistency_and_decision_exactly_reproduced": (
                metrics_equal
            ),
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
