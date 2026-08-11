#!/usr/bin/env python3
"""Run the frozen structured-orientation matched execution-model ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_execution_pilot import (
    load_weights, payload_sha256, predict, save_weights,
)
from main.multilink_ellipsoid.factorized_one_sided_geometry import (
    fit_local_geometry_jacobians, train_one_sided_geometry_ensemble,
)
from main.multilink_ellipsoid.factorized_structured_orientation import (
    RESULT_SCHEMA, load_structured_config, structured_decision,
    structured_orientation_arrays,
)
from main.multilink_ellipsoid.factorized_time_conditioned_decoder import (
    eligible_state_support, load_time_decoder_config, predict_time_conditioned,
    save_time_weights, temporal_joint_metrics,
    train_time_conditioned_ensemble,
)
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs, load_inputs, prediction_metrics,
)
from scripts.evaluate_distal_factorized_time_conditioned_decoder_moka10 import (
    add_time_arguments, time_paths, validate_time_sources,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_array(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def add_structured_arguments(parser: argparse.ArgumentParser) -> None:
    add_time_arguments(parser)
    parser.add_argument("--structured-config", type=Path, required=True)
    parser.add_argument("--orientation-config", type=Path, required=True)
    parser.add_argument("--orientation-result", type=Path, required=True)
    parser.add_argument("--orientation-validation", type=Path, required=True)
    parser.add_argument("--source-time-model", type=Path, required=True)
    parser.add_argument("--source-time-result", type=Path, required=True)
    parser.add_argument("--source-time-validation", type=Path, required=True)


def structured_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = time_paths(args)
    paths.update({
        "structured_config": args.structured_config.resolve(),
        "orientation_config": args.orientation_config.resolve(),
        "orientation_result": args.orientation_result.resolve(),
        "orientation_validation": args.orientation_validation.resolve(),
        "source_time_model": args.source_time_model.resolve(),
        "source_time_result": args.source_time_result.resolve(),
        "source_time_validation": args.source_time_validation.resolve(),
    })
    return paths


def validate_structured_sources(
    paths: Mapping[str, Path], config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = config["immutable_source"]
    for source_key, path_key in (
        ("orientation_confirmation_config_file_sha256", "orientation_config"),
        ("orientation_confirmation_result_file_sha256", "orientation_result"),
        ("orientation_confirmation_validation_file_sha256", "orientation_validation"),
        ("one_sided_config_file_sha256", "config"),
        ("time_config_file_sha256", "time_config"),
        ("trajectory_array_file_sha256", "array"),
        ("trajectory_metadata_file_sha256", "metadata"),
        ("flat_source_model_file_sha256", "source_one_sided_model"),
        ("flat_source_result_file_sha256", "source_one_sided_result"),
        ("time_source_model_file_sha256", "source_time_model"),
        ("time_source_result_file_sha256", "source_time_result"),
    ):
        _require(
            _file_sha256(paths[path_key]) == source[source_key],
            "structured-orientation immutable source differs",
        )
    orientation = _load(paths["orientation_result"])
    orientation_validation = _load(paths["orientation_validation"])
    source_time = _load(paths["source_time_result"])
    source_time_validation = _load(paths["source_time_validation"])
    _require(
        orientation.get("result_payload_sha256")
        == source["orientation_confirmation_result_payload_sha256"]
        and orientation.get("audit", {}).get("decision", {}).get(
            "orientation_representation_causal"
        ) is True
        and orientation_validation.get("validation_payload_sha256")
        == source["orientation_confirmation_validation_payload_sha256"]
        and orientation_validation.get("valid") is True
        and source_time.get("status") == "complete"
        and source_time_validation.get("valid") is True,
        "structured-orientation validated source differs",
    )
    return orientation, source_time


def run_models(
    *, paths: Mapping[str, Path], config: Mapping[str, Any],
    one_sided_config: Mapping[str, Any], time_config: Mapping[str, Any],
    factorized_config: Mapping[str, Any], complete_dataset: Mapping[str, Any],
    complete_collection: Mapping[str, Any], arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any], local_geometry: Mapping[str, Any],
    normalization: Mapping[str, Any], flat_model_path: Path,
    time_model_path: Path,
) -> dict[str, Any]:
    structured_arrays, _, representation = structured_orientation_arrays(
        complete_dataset, arrays, config,
    )
    flat_models, flat_state, flat_training = train_one_sided_geometry_ensemble(
        structured_arrays, sensitivities, local_geometry, factorized_config,
        one_sided_config, normalization_override=normalization,
    )
    flat_q = predict(flat_models, flat_state, structured_arrays)
    flat_model = save_weights(flat_model_path, flat_state)
    time_models, time_state, time_training = train_time_conditioned_ensemble(
        structured_arrays, sensitivities, local_geometry, time_config,
        normalization_override=normalization,
    )
    time_q = predict_time_conditioned(time_models, time_state, structured_arrays)
    time_model = save_time_weights(time_model_path, time_state)
    outputs: dict[str, Any] = {
        "arrays": structured_arrays, "representation": representation,
        "flat_q": flat_q, "time_q": time_q,
        "training": {"flat": flat_training, "time": time_training},
        "models": {"flat": flat_model, "time": time_model},
    }
    for name, predicted_q in (("flat", flat_q), ("time", time_q)):
        geometry = geometry_kwargs(
            paths, factorized_config, complete_dataset, complete_collection,
            structured_arrays, predicted_q, ("validation", "test"),
        )
        margin = geometry.pop("predicted_minimum_margin_m")
        exact_static = geometry.pop("exact_q_static_minimum_margin_m")
        outputs[name] = {
            "margin": margin, "exact_static": exact_static,
            "geometry": geometry,
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
        }
    return outputs


def main() -> int:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_structured_arguments(parser)
    parser.add_argument("--flat-model", type=Path, required=True)
    parser.add_argument("--time-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = structured_paths(args)
    paths.update({
        "flat_model": args.flat_model.resolve(),
        "time_model": args.time_model.resolve(),
        "predictions": args.predictions.resolve(),
        "output": args.output.resolve(),
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
    local_geometry = fit_local_geometry_jacobians(arrays, one_sided_config)
    structured_arrays, normalization, representation = structured_orientation_arrays(
        complete_dataset, arrays, config,
    )
    _require(
        local_geometry["audit"]["validation"]["linearization_RMSE_m"]
        <= time_config["local_geometry_jacobian"][
            "maximum_validation_random_linearization_RMSE_m"
        ], "structured-orientation local geometry gate differs",
    )
    torch.set_num_threads(8)
    outputs = run_models(
        paths=paths, config=config, one_sided_config=one_sided_config,
        time_config=time_config, factorized_config=factorized_config,
        complete_dataset=complete_dataset, complete_collection=complete_collection,
        arrays=arrays, sensitivities=sensitivities,
        local_geometry=local_geometry, normalization=normalization,
        flat_model_path=paths["flat_model"], time_model_path=paths["time_model"],
    )
    _require(
        representation == outputs["representation"]
        and np.array_equal(structured_arrays["features"], outputs["arrays"]["features"]),
        "structured-orientation representation replay differs",
    )
    source_time_metrics = source_time["metrics"]
    decision = structured_decision(
        representation=representation,
        flat_metrics=outputs["flat"]["metrics"],
        time_metrics=outputs["time"]["metrics"],
        flat_temporal=outputs["flat"]["temporal"],
        time_temporal=outputs["time"]["temporal"],
        flat_support=outputs["flat"]["support"]["test"],
        time_support=outputs["time"]["support"]["test"],
        source_time_metrics=source_time_metrics,
        source_time_temporal=source_time_metrics["temporal_joint"],
        source_time_support_count=source_time_metrics["eligible_support"]["test"][
            "supported_state_count"
        ], config=config,
    )
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["predictions"],
        flat_joint_position_rad=outputs["flat_q"],
        flat_minimum_margin_m=outputs["flat"]["margin"],
        time_joint_position_rad=outputs["time_q"],
        time_minimum_margin_m=outputs["time"]["margin"],
        exact_q_static_minimum_margin_m=outputs["flat"]["exact_static"],
    )
    prediction_receipt = {
        "path": str(paths["predictions"]),
        "file_sha256": _file_sha256(paths["predictions"]),
        "flat_joint_sha256": _hash_array(outputs["flat_q"]),
        "flat_margin_sha256": _hash_array(outputs["flat"]["margin"]),
        "time_joint_sha256": _hash_array(outputs["time_q"]),
        "time_margin_sha256": _hash_array(outputs["time"]["margin"]),
        "exact_static_sha256": _hash_array(outputs["flat"]["exact_static"]),
    }
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "representation": representation,
        "immutable_baselines": {
            "flat": {
                "result_file_sha256": _file_sha256(paths["source_one_sided_result"]),
                "metrics": source_flat["metrics"],
            },
            "time": {
                "result_file_sha256": _file_sha256(paths["source_time_result"]),
                "metrics": source_time_metrics,
            },
        },
        "local_geometry_jacobian_audit": local_geometry["audit"],
        "training": outputs["training"], "models": outputs["models"],
        "predictions": prediction_receipt,
        "metrics": {
            name: {
                **outputs[name]["metrics"],
                "temporal_joint": outputs[name]["temporal"],
                "eligible_support": outputs[name]["support"],
                "experimental_geometry": outputs[name]["geometry"],
            } for name in ("flat", "time")
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
        "representation": representation,
        "flat": result["metrics"]["flat"],
        "time": result["metrics"]["time"],
        "decision": decision,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
