#!/usr/bin/env python3
"""Independently validate the one-sided geometry-loss ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_execution_pilot import (
    load_weights, payload_sha256, predict,
)
from main.multilink_ellipsoid.factorized_one_sided_geometry import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, fit_local_geometry_jacobians,
    one_sided_decision,
)
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    _hash_array, add_common_arguments, geometry_kwargs, load_inputs,
    prediction_metrics, resolved_paths, validation_pattern,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _array_equal_with_nan(left, right) -> tuple[bool, float]:
    import numpy as np

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.array_equal(np.isnan(left), np.isnan(right)):
        return False, float("inf")
    finite = np.isfinite(left) & np.isfinite(right)
    maximum = 0.0 if not np.any(finite) else float(np.max(np.abs(left[finite] - right[finite])))
    return bool(np.array_equal(left[finite], right[finite])), maximum


def main() -> int:
    import numpy as np
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = resolved_paths(args)
    paths.update({
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "result": args.result.resolve(), "output": args.output.resolve(),
    })
    (
        config, factorized_config, complete_dataset, complete_collection,
        _, _, baseline_result, _, _, arrays, sensitivities, baseline_q,
    ) = load_inputs(paths)
    result = _load(paths["result"])
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit,
        "one-sided geometry result source differs",
    )
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
    if result.get("training", {}).get("executed") is False:
        expected_decision = {
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
                "validation_linearization": local_geometry["audit"][
                    "validation"
                ]["linearization_RMSE_m"] <= config[
                    "local_geometry_jacobian"
                ]["maximum_validation_random_linearization_RMSE_m"],
            },
            "one_sided_geometry_GO": False,
            "conclusion": "conditional_validation_gate_blocks_training",
            "uncertainty_QP_or_closed_loop_authorized": False,
        }
        checks = {
            "validation_pattern_exact": pattern
            == result["validation_pattern_audit"],
            "jacobian_audit_exact": local_geometry["audit"]
            == result["local_geometry_jacobian_audit"],
            "decision_exact": expected_decision == result["decision"],
            "training_correctly_skipped": not all(
                expected_decision["gate_tests"].values()
            ),
            "forbidden_actions_respected": all(
                value is False
                for value in result["forbidden_action_receipt"].values()
            ),
        }
        validation = {
            "schema_version": VALIDATION_SCHEMA,
            "status": "complete", "valid": bool(all(checks.values())),
            "scientific_result": False,
            "source": _git_identity(paths["repo"], args.expected_commit),
            "allocation": allocation_record(),
            "result_file_sha256": _file_sha256(paths["result"]),
            "result_payload_sha256": result["result_payload_sha256"],
            "checks": checks,
            "maximum_absolute_difference": {
                "baseline_validation_margin_m": _array_equal_with_nan(
                    baseline_validation_margin,
                    baseline_validation_margin,
                )[1],
            },
            "recomputed_decision": expected_decision,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        validation["validation_payload_sha256"] = payload_sha256(
            validation, "validation_payload_sha256"
        )
        _atomic_write(paths["output"], validation)
        print(json.dumps({
            "valid": validation["valid"], "checks": checks,
            "decision": expected_decision,
        }, sort_keys=True), flush=True)
        return 0 if validation["valid"] else 1
    _require(
        result.get("predictions", {}).get("file_sha256")
        == _file_sha256(paths["predictions"])
        and result.get("model", {}).get("file_sha256")
        == _file_sha256(paths["experimental_model"]),
        "one-sided geometry trained artifact source differs",
    )
    models, state = load_weights(paths["experimental_model"])
    experimental_q = predict(models, state, arrays)
    stored = np.load(paths["predictions"], allow_pickle=False)
    equal_q, max_q = _array_equal_with_nan(
        experimental_q, stored["experimental_joint_position_rad"],
    )
    experimental_geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        arrays, experimental_q, ("validation", "test"),
    )
    experimental_margin = experimental_geometry.pop("predicted_minimum_margin_m")
    exact_q_static_margin = experimental_geometry.pop(
        "exact_q_static_minimum_margin_m"
    )
    equal_baseline, max_baseline = _array_equal_with_nan(
        baseline_validation_margin,
        stored["baseline_validation_minimum_margin_m"],
    )
    equal_margin, max_margin = _array_equal_with_nan(
        experimental_margin, stored["experimental_minimum_margin_m"],
    )
    equal_exact, max_exact = _array_equal_with_nan(
        exact_q_static_margin, stored["exact_q_static_minimum_margin_m"],
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
    checks = {
        "model_prediction_exact": equal_q,
        "baseline_validation_geometry_exact": equal_baseline,
        "experimental_geometry_exact": equal_margin,
        "exact_q_static_geometry_exact": equal_exact,
        "validation_pattern_exact": pattern == result["validation_pattern_audit"],
        "jacobian_audit_exact": local_geometry["audit"]
        == result["local_geometry_jacobian_audit"],
        "metrics_exact": metrics["test_safety"] == result["metrics"]["test_safety"]
        and metrics["validation_safety"] == result["metrics"]["validation_safety"]
        and metrics["joint_sensitivity"] == result["metrics"]["joint_sensitivity"]
        and metrics["margin_sensitivity"] == result["metrics"]["margin_sensitivity"],
        "decision_exact": decision == result["decision"],
        "forbidden_actions_respected": all(
            value is False for value in result["forbidden_action_receipt"].values()
        ),
    }
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "complete", "valid": bool(all(checks.values())),
        "scientific_result": False,
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(),
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "prediction_file_sha256": _file_sha256(paths["predictions"]),
        "model_file_sha256": _file_sha256(paths["experimental_model"]),
        "checks": checks,
        "maximum_absolute_difference": {
            "joint_position_rad": max_q,
            "baseline_validation_margin_m": max_baseline,
            "experimental_margin_m": max_margin,
            "exact_q_static_margin_m": max_exact,
        },
        "recomputed_decision": decision,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], validation)
    print(json.dumps({
        "valid": validation["valid"], "checks": checks,
        "maximum_absolute_difference": validation["maximum_absolute_difference"],
        "decision": decision,
    }, sort_keys=True), flush=True)
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
