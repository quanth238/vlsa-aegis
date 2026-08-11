#!/usr/bin/env python3
"""Independently replay the frozen structured-orientation result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_execution_pilot import (
    load_weights, payload_sha256, predict,
)
from main.multilink_ellipsoid.factorized_one_sided_geometry import (
    fit_local_geometry_jacobians,
)
from main.multilink_ellipsoid.factorized_structured_orientation import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_structured_config,
    structured_decision, structured_orientation_arrays,
)
from main.multilink_ellipsoid.factorized_time_conditioned_decoder import (
    eligible_state_support, load_time_decoder_config, load_time_weights,
    predict_time_conditioned, temporal_joint_metrics,
)
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs, load_inputs, prediction_metrics,
)
from scripts.evaluate_distal_factorized_structured_orientation_moka10 import (
    add_structured_arguments, structured_paths, validate_structured_sources,
)
from scripts.evaluate_distal_factorized_time_conditioned_decoder_moka10 import (
    _array_equal_with_nan, validate_time_sources,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def main() -> int:
    import numpy as np
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_structured_arguments(parser)
    parser.add_argument("--flat-model", type=Path, required=True)
    parser.add_argument("--time-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = structured_paths(args)
    paths.update({
        "flat_model": args.flat_model.resolve(),
        "time_model": args.time_model.resolve(),
        "predictions": args.predictions.resolve(),
        "result": args.result.resolve(), "output": args.output.resolve(),
    })
    config = load_structured_config(paths["structured_config"])
    time_config = load_time_decoder_config(paths["time_config"])
    validate_structured_sources(paths, config)
    (
        one_sided_config, factorized_config, complete_dataset,
        complete_collection, _, _, _, _, _, arrays, sensitivities, _,
    ) = load_inputs(paths)
    source_flat, _, _ = validate_time_sources(paths, time_config, arrays)
    source_time = _load(paths["source_time_result"])
    result = _load(paths["result"])
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("models", {}).get("flat", {}).get("file_sha256")
        == _file_sha256(paths["flat_model"])
        and result.get("models", {}).get("time", {}).get("file_sha256")
        == _file_sha256(paths["time_model"])
        and result.get("predictions", {}).get("file_sha256")
        == _file_sha256(paths["predictions"]),
        "structured-orientation result identity differs",
    )
    structured_arrays, normalization, representation = structured_orientation_arrays(
        complete_dataset, arrays, config,
    )
    local_geometry = fit_local_geometry_jacobians(arrays, one_sided_config)
    flat_models, flat_state = load_weights(paths["flat_model"])
    time_models, time_state = load_time_weights(paths["time_model"])
    normalization_exact = all(
        np.array_equal(state[key], normalization[key])
        for state in (flat_state, time_state)
        for key in ("feature_mean", "feature_std")
    )
    flat_q = predict(flat_models, flat_state, structured_arrays)
    time_q = predict_time_conditioned(time_models, time_state, structured_arrays)
    stored = np.load(paths["predictions"], allow_pickle=False)
    differences = {}
    equalities = {}
    for name, computed, archived in (
        ("flat_joint", flat_q, stored["flat_joint_position_rad"]),
        ("time_joint", time_q, stored["time_joint_position_rad"]),
    ):
        equalities[name], differences[name] = _array_equal_with_nan(
            computed, archived,
        )
    recomputed = {}
    exact_static_reference = None
    for name, predicted_q in (("flat", flat_q), ("time", time_q)):
        geometry = geometry_kwargs(
            paths, factorized_config, complete_dataset, complete_collection,
            structured_arrays, predicted_q, ("validation", "test"),
        )
        margin = geometry.pop("predicted_minimum_margin_m")
        exact_static = geometry.pop("exact_q_static_minimum_margin_m")
        if exact_static_reference is None:
            exact_static_reference = exact_static
        equalities[name + "_margin"], differences[name + "_margin"] = (
            _array_equal_with_nan(margin, stored[name + "_minimum_margin_m"])
        )
        recomputed[name] = {
            "metrics": prediction_metrics(
                predicted_q, margin, structured_arrays, sensitivities,
                factorized_config,
            ),
            "temporal": {
                split: temporal_joint_metrics(
                    predicted_q, structured_arrays["joint_position_rad"],
                    structured_arrays, split,
                ) for split in ("validation", "test")
            },
            "support": {
                split: eligible_state_support(
                    exact_margin=structured_arrays["minimum_margin_m"],
                    predicted_margin=margin, arrays=structured_arrays,
                    split_name=split,
                ) for split in ("validation", "test")
            },
            "geometry": geometry,
        }
    equalities["exact_static"], differences["exact_static"] = (
        _array_equal_with_nan(
            exact_static_reference, stored["exact_q_static_minimum_margin_m"],
        )
    )
    source_time_metrics = source_time["metrics"]
    decision = structured_decision(
        representation=representation,
        flat_metrics=recomputed["flat"]["metrics"],
        time_metrics=recomputed["time"]["metrics"],
        flat_temporal=recomputed["flat"]["temporal"],
        time_temporal=recomputed["time"]["temporal"],
        flat_support=recomputed["flat"]["support"]["test"],
        time_support=recomputed["time"]["support"]["test"],
        source_time_metrics=source_time_metrics,
        source_time_temporal=source_time_metrics["temporal_joint"],
        source_time_support_count=source_time_metrics["eligible_support"]["test"][
            "supported_state_count"
        ], config=config,
    )
    metric_exact = all(
        {
            **recomputed[name]["metrics"],
            "temporal_joint": recomputed[name]["temporal"],
            "eligible_support": recomputed[name]["support"],
            "experimental_geometry": recomputed[name]["geometry"],
        } == result["metrics"][name]
        for name in ("flat", "time")
    )
    checks = {
        "structured_representation_exact": representation == result["representation"],
        "normalization_exact": normalization_exact,
        "stored_predictions_exact": all(equalities.values()),
        "local_geometry_audit_exact": local_geometry["audit"]
        == result["local_geometry_jacobian_audit"],
        "immutable_flat_baseline_exact": source_flat["metrics"]
        == result["immutable_baselines"]["flat"]["metrics"],
        "immutable_time_baseline_exact": source_time_metrics
        == result["immutable_baselines"]["time"]["metrics"],
        "metrics_exact": metric_exact,
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
        "model_file_sha256": {
            "flat": _file_sha256(paths["flat_model"]),
            "time": _file_sha256(paths["time_model"]),
        },
        "prediction_file_sha256": _file_sha256(paths["predictions"]),
        "checks": checks, "maximum_absolute_difference": differences,
        "recomputed_decision": decision,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], validation)
    print(json.dumps({
        "valid": validation["valid"], "checks": checks,
        "maximum_absolute_difference": differences, "decision": decision,
    }, sort_keys=True), flush=True)
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
