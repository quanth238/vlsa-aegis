#!/usr/bin/env python3
"""Evaluate the matched shared time-conditioned execution decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_execution_pilot import (
    payload_sha256, sensitivity_arrays,
)
from main.multilink_ellipsoid.factorized_one_sided_geometry import (
    fit_local_geometry_jacobians,
)
from main.multilink_ellipsoid.factorized_time_conditioned_decoder import (
    RESULT_SCHEMA, eligible_state_support, load_time_decoder_config,
    predict_time_conditioned, save_time_weights, temporal_joint_metrics,
    time_decoder_decision, train_time_conditioned_ensemble,
)
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    add_common_arguments, geometry_kwargs, load_inputs, prediction_metrics,
    resolved_paths,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def add_time_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--time-config", type=Path, required=True)
    parser.add_argument("--source-one-sided-model", type=Path, required=True)
    parser.add_argument("--source-one-sided-predictions", type=Path, required=True)
    parser.add_argument("--source-one-sided-result", type=Path, required=True)
    parser.add_argument("--source-one-sided-validation", type=Path, required=True)
    parser.add_argument("--unsupported-audit-result", type=Path, required=True)
    parser.add_argument("--unsupported-audit-validation", type=Path, required=True)


def time_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = resolved_paths(args)
    paths.update({
        "time_config": args.time_config.resolve(),
        "source_one_sided_model": args.source_one_sided_model.resolve(),
        "source_one_sided_predictions": args.source_one_sided_predictions.resolve(),
        "source_one_sided_result": args.source_one_sided_result.resolve(),
        "source_one_sided_validation": args.source_one_sided_validation.resolve(),
        "unsupported_audit_result": args.unsupported_audit_result.resolve(),
        "unsupported_audit_validation": args.unsupported_audit_validation.resolve(),
    })
    return paths


def _hash_array(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _array_equal_with_nan(left: Any, right: Any) -> tuple[bool, float]:
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


def validate_time_sources(
    paths: Mapping[str, Path], config: Mapping[str, Any], arrays: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Any, Any]:
    import numpy as np
    from main.multilink_ellipsoid.factorized_execution_pilot import load_weights, predict

    source = config["immutable_source"]
    for source_key, path_key in (
        ("one_sided_config_file_sha256", "config"),
        ("one_sided_model_file_sha256", "source_one_sided_model"),
        ("one_sided_predictions_file_sha256", "source_one_sided_predictions"),
        ("one_sided_result_file_sha256", "source_one_sided_result"),
        ("one_sided_validation_file_sha256", "source_one_sided_validation"),
        ("unsupported_audit_result_file_sha256", "unsupported_audit_result"),
        ("unsupported_audit_validation_file_sha256", "unsupported_audit_validation"),
        ("trajectory_array_file_sha256", "array"),
        ("trajectory_metadata_file_sha256", "metadata"),
    ):
        _require(_file_sha256(paths[path_key]) == source[source_key],
                 "time-conditioned decoder immutable source differs")
    result = _load(paths["source_one_sided_result"])
    validation = _load(paths["source_one_sided_validation"])
    audit = _load(paths["unsupported_audit_result"])
    audit_validation = _load(paths["unsupported_audit_validation"])
    _require(
        result.get("result_payload_sha256")
        == source["one_sided_result_payload_sha256"]
        and result.get("decision", {}).get("one_sided_geometry_GO") is False
        and result["metrics"]["test_safety"]["false_safe_action_count"]
        == source["source_test_false_safe_action_count"]
        and result["metrics"]["test_safety"]["state_safe_support_count"]
        == source["source_test_state_safe_support_count"]
        and result["metrics"]["validation_safety"]["false_safe_action_count"]
        == source["source_validation_false_safe_action_count"]
        and result["metrics"]["validation_safety"]["state_safe_support_count"]
        == source["source_validation_state_safe_support_count"]
        and validation.get("validation_payload_sha256")
        == source["one_sided_validation_payload_sha256"]
        and validation.get("valid") is True
        and audit.get("result_payload_sha256")
        == source["unsupported_audit_result_payload_sha256"]
        and audit.get("decision", {}).get("verdict") == "strict_NO_GO_unchanged"
        and audit_validation.get("validation_payload_sha256")
        == source["unsupported_audit_validation_payload_sha256"]
        and audit_validation.get("valid") is True,
        "time-conditioned decoder validated source differs",
    )
    models, state = load_weights(paths["source_one_sided_model"])
    source_q = predict(models, state, arrays)
    archive = np.load(paths["source_one_sided_predictions"], allow_pickle=False)
    _require(np.array_equal(
        source_q, archive["experimental_joint_position_rad"],
    ), "time-conditioned decoder source joint prediction differs")
    return result, source_q, archive["experimental_minimum_margin_m"].astype(np.float64)


def experiment_metrics(
    *, predicted_q: Any, predicted_margin: Any, arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any], factorized_config: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = prediction_metrics(
        predicted_q, predicted_margin, arrays, sensitivities, factorized_config,
    )
    return metrics


def main() -> int:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_time_arguments(parser)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = time_paths(args)
    paths.update({
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "output": args.output.resolve(),
    })
    config = load_time_decoder_config(paths["time_config"])
    (
        _, factorized_config, complete_dataset, complete_collection,
        _, _, _, _, _, arrays, sensitivities, _,
    ) = load_inputs(paths)
    source_result, source_q, source_margin = validate_time_sources(
        paths, config, arrays,
    )
    local_geometry = fit_local_geometry_jacobians(arrays, source_result["config"])
    _require(
        local_geometry["audit"]["validation"]["linearization_RMSE_m"]
        <= config["local_geometry_jacobian"][
            "maximum_validation_random_linearization_RMSE_m"
        ], "time-conditioned decoder local geometry gate differs",
    )
    torch.set_num_threads(8)
    models, state, training = train_time_conditioned_ensemble(
        arrays, sensitivities, local_geometry, config,
    )
    experimental_q = predict_time_conditioned(models, state, arrays)
    model_receipt = save_time_weights(paths["experimental_model"], state)
    geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        arrays, experimental_q, ("validation", "test"),
    )
    experimental_margin = geometry.pop("predicted_minimum_margin_m")
    exact_q_static_margin = geometry.pop("exact_q_static_minimum_margin_m")
    metrics = experiment_metrics(
        predicted_q=experimental_q, predicted_margin=experimental_margin,
        arrays=arrays, sensitivities=sensitivities,
        factorized_config=factorized_config,
    )
    source_temporal = {
        split: temporal_joint_metrics(
            source_q, arrays["joint_position_rad"], arrays, split,
        ) for split in ("validation", "test")
    }
    experimental_temporal = {
        split: temporal_joint_metrics(
            experimental_q, arrays["joint_position_rad"], arrays, split,
        ) for split in ("validation", "test")
    }
    validation_support = eligible_state_support(
        exact_margin=arrays["minimum_margin_m"],
        predicted_margin=experimental_margin, arrays=arrays,
        split_name="validation",
    )
    test_support = eligible_state_support(
        exact_margin=arrays["minimum_margin_m"],
        predicted_margin=experimental_margin, arrays=arrays, split_name="test",
    )
    source_metrics = {
        key: source_result["metrics"][key] for key in (
            "test_safety", "validation_safety", "joint_sensitivity",
            "margin_sensitivity",
        )
    }
    decision = time_decoder_decision(
        source_metrics=source_metrics, source_temporal=source_temporal,
        experimental_metrics=metrics,
        experimental_temporal={"validation": experimental_temporal["validation"],
                               "test": experimental_temporal["test"]},
        validation_support=validation_support, test_support=test_support,
        config=config,
    )
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["predictions"],
        experimental_joint_position_rad=experimental_q,
        experimental_minimum_margin_m=experimental_margin,
        exact_q_static_minimum_margin_m=exact_q_static_margin,
    )
    prediction_receipt = {
        "path": str(paths["predictions"]),
        "file_sha256": _file_sha256(paths["predictions"]),
        "experimental_joint_sha256": _hash_array(experimental_q),
        "experimental_margin_sha256": _hash_array(experimental_margin),
        "exact_q_static_margin_sha256": _hash_array(exact_q_static_margin),
    }
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_source_model": {
            "result_path": str(paths["source_one_sided_result"]),
            "result_file_sha256": _file_sha256(paths["source_one_sided_result"]),
            "metrics": source_metrics, "temporal_joint_metrics": source_temporal,
        },
        "local_geometry_jacobian_audit": local_geometry["audit"],
        "training": training, "model": model_receipt,
        "predictions": prediction_receipt,
        "metrics": {
            **metrics, "temporal_joint": experimental_temporal,
            "eligible_support": {
                "validation": validation_support, "test": test_support,
            },
            "experimental_geometry": geometry,
        },
        "decision": decision,
        "forbidden_action_receipt": {
            key: False for key in config["forbidden_actions"]
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "source": source_metrics, "experimental": metrics,
        "temporal": experimental_temporal,
        "support": {"validation": validation_support, "test": test_support},
        "decision": decision,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
