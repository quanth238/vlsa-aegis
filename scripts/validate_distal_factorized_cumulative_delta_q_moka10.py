#!/usr/bin/env python3
"""Independently validate cumulative delta-q prediction and geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_cumulative_delta_q import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_cumulative_delta_weights,
    predict_cumulative_delta,
)
from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import (
    add_cumulative_arguments, evaluate_prediction, prepare,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _array_equal_with_nan(left, right):
    import numpy as np

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.array_equal(np.isnan(left), np.isnan(right)):
        return False, float("inf")
    finite = np.isfinite(left) & np.isfinite(right)
    maximum = 0.0 if not np.any(finite) else float(
        np.max(np.abs(left[finite] - right[finite]))
    )
    return bool(np.array_equal(left[finite], right[finite])), maximum


def main() -> int:
    import numpy as np
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_cumulative_arguments(parser)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, config, _, factorized_config, complete_dataset,
        complete_collection, arrays, sensitivities, _, _, _,
    ) = prepare(args)
    paths.update({
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "result": args.result.resolve(), "output": args.output.resolve(),
    })
    result = _load(paths["result"])
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("model", {}).get("file_sha256")
        == _file_sha256(paths["experimental_model"])
        and result.get("predictions", {}).get("file_sha256")
        == _file_sha256(paths["predictions"]),
        "cumulative-delta-q result identity differs",
    )
    models, state = load_cumulative_delta_weights(paths["experimental_model"])
    predicted_q = predict_cumulative_delta(models, state, arrays)
    archive = np.load(paths["predictions"], allow_pickle=False)
    stored_q = archive["experimental_joint_position_rad"]
    q_exact, q_difference = _array_equal_with_nan(predicted_q, stored_q)
    evaluation = evaluate_prediction(
        paths=paths, config=config, factorized_config=factorized_config,
        complete_dataset=complete_dataset, complete_collection=complete_collection,
        arrays=arrays, sensitivities=sensitivities, predicted_q=predicted_q,
    )
    margin_exact, margin_difference = _array_equal_with_nan(
        evaluation["predicted_margin"],
        archive["experimental_minimum_margin_m"],
    )
    exact_static, exact_static_difference = _array_equal_with_nan(
        evaluation["exact_static"], archive["exact_q_static_minimum_margin_m"],
    )
    metrics = {
        **evaluation["metrics"], "temporal_joint": evaluation["temporal"],
        "eligible_support": evaluation["support"],
        "experimental_geometry": evaluation["geometry"],
    }
    checks = {
        "model_prediction_exact": q_exact,
        "exact_initial_condition": float(np.max(np.abs(
            predicted_q[:, 0] - np.asarray(arrays["joint_position_rad"])[:, 0]
        ))) == 0.0,
        "geometry_margin_exact": margin_exact,
        "exact_q_static_margin_exact": exact_static,
        "metrics_exact": metrics == result.get("metrics"),
        "decision_exact": evaluation["decision"] == result.get("decision"),
        "reserved_collection_receipt_clean": result.get(
            "reserved_episode_receipt", {}
        ).get("collection_submitted") is False,
        "forbidden_actions_absent": all(
            value is False for value in result.get(
                "forbidden_action_receipt", {}
            ).values()
        ),
    }
    valid = bool(all(checks.values()))
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "complete",
        "valid": valid, "scientific_result": False,
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(),
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "checks": checks,
        "maximum_absolute_difference": {
            "joint_rad": q_difference, "margin_m": margin_difference,
            "exact_static_margin_m": exact_static_difference,
        },
        "recomputed_decision": evaluation["decision"],
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
