#!/usr/bin/env python3
"""Independently replay the time-conditioned decoder result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.factorized_one_sided_geometry import (
    fit_local_geometry_jacobians,
)
from main.multilink_ellipsoid.factorized_time_conditioned_decoder import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, eligible_state_support,
    load_time_decoder_config, load_time_weights, predict_time_conditioned,
    temporal_joint_metrics, time_decoder_decision,
)
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs, load_inputs,
)
from scripts.evaluate_distal_factorized_time_conditioned_decoder_moka10 import (
    _array_equal_with_nan, add_time_arguments, experiment_metrics, time_paths,
    validate_time_sources,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def main() -> int:
    import numpy as np
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_time_arguments(parser)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = time_paths(args)
    paths.update({
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "result": args.result.resolve(), "output": args.output.resolve(),
    })
    config = load_time_decoder_config(paths["time_config"])
    (
        _, factorized_config, complete_dataset, complete_collection,
        _, _, _, _, _, arrays, sensitivities, _,
    ) = load_inputs(paths)
    source_result, source_q, _ = validate_time_sources(paths, config, arrays)
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
        "time-conditioned decoder result identity differs",
    )
    local_geometry = fit_local_geometry_jacobians(arrays, source_result["config"])
    models, state = load_time_weights(paths["experimental_model"])
    predicted_q = predict_time_conditioned(models, state, arrays)
    stored = np.load(paths["predictions"], allow_pickle=False)
    equal_q, max_q = _array_equal_with_nan(
        predicted_q, stored["experimental_joint_position_rad"],
    )
    geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        arrays, predicted_q, ("validation", "test"),
    )
    predicted_margin = geometry.pop("predicted_minimum_margin_m")
    exact_static = geometry.pop("exact_q_static_minimum_margin_m")
    equal_margin, max_margin = _array_equal_with_nan(
        predicted_margin, stored["experimental_minimum_margin_m"],
    )
    equal_exact, max_exact = _array_equal_with_nan(
        exact_static, stored["exact_q_static_minimum_margin_m"],
    )
    metrics = experiment_metrics(
        predicted_q=predicted_q, predicted_margin=predicted_margin,
        arrays=arrays, sensitivities=sensitivities,
        factorized_config=factorized_config,
    )
    source_temporal = {
        split: temporal_joint_metrics(
            source_q, arrays["joint_position_rad"], arrays, split,
        ) for split in ("validation", "test")
    }
    temporal = {
        split: temporal_joint_metrics(
            predicted_q, arrays["joint_position_rad"], arrays, split,
        ) for split in ("validation", "test")
    }
    support = {
        split: eligible_state_support(
            exact_margin=arrays["minimum_margin_m"],
            predicted_margin=predicted_margin, arrays=arrays, split_name=split,
        ) for split in ("validation", "test")
    }
    source_metrics = result["immutable_source_model"]["metrics"]
    decision = time_decoder_decision(
        source_metrics=source_metrics, source_temporal=source_temporal,
        experimental_metrics=metrics, experimental_temporal=temporal,
        validation_support=support["validation"], test_support=support["test"],
        config=config,
    )
    checks = {
        "model_prediction_exact": equal_q,
        "experimental_geometry_exact": equal_margin,
        "exact_q_static_geometry_exact": equal_exact,
        "local_geometry_audit_exact": local_geometry["audit"]
        == result["local_geometry_jacobian_audit"],
        "metrics_exact": (
            metrics["test_safety"] == result["metrics"]["test_safety"]
            and metrics["validation_safety"]
            == result["metrics"]["validation_safety"]
            and metrics["joint_sensitivity"]
            == result["metrics"]["joint_sensitivity"]
            and metrics["margin_sensitivity"]
            == result["metrics"]["margin_sensitivity"]
            and temporal == result["metrics"]["temporal_joint"]
            and support == result["metrics"]["eligible_support"]
        ),
        "decision_exact": decision == result["decision"],
        "forbidden_actions_respected": all(
            value is False for value in result["forbidden_action_receipt"].values()
        ),
    }
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "complete",
        "valid": bool(all(checks.values())), "scientific_result": False,
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(),
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "prediction_file_sha256": _file_sha256(paths["predictions"]),
        "model_file_sha256": _file_sha256(paths["experimental_model"]),
        "checks": checks,
        "maximum_absolute_difference": {
            "joint_position_rad": max_q,
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
