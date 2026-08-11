#!/usr/bin/env python3
"""Train and evaluate exact-q0 cumulative OSC joint displacement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_cumulative_delta_q import (
    RESULT_SCHEMA, cumulative_delta_decision, load_cumulative_delta_config,
    predict_cumulative_delta, save_cumulative_delta_weights,
    train_cumulative_delta_ensemble,
)
from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.factorized_one_sided_geometry import (
    fit_local_geometry_jacobians,
)
from main.multilink_ellipsoid.factorized_structured_orientation import (
    load_structured_config, structured_orientation_arrays,
)
from main.multilink_ellipsoid.factorized_time_conditioned_decoder import (
    eligible_state_support, load_time_decoder_config, temporal_joint_metrics,
)
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs, load_inputs, prediction_metrics,
)
from scripts.evaluate_distal_factorized_structured_orientation_moka10 import (
    add_structured_arguments, structured_paths, validate_structured_sources,
)
from scripts.evaluate_distal_factorized_time_conditioned_decoder_moka10 import (
    validate_time_sources,
)
from scripts.evaluate_distal_native_geom_inventory_moka10 import _read_manifest
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_array(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def add_cumulative_arguments(parser: argparse.ArgumentParser) -> None:
    add_structured_arguments(parser)
    parser.add_argument("--cumulative-config", type=Path, required=True)
    parser.add_argument("--reserved-manifest", type=Path, required=True)
    parser.add_argument("--source-structured-flat-model", type=Path, required=True)
    parser.add_argument("--source-structured-time-model", type=Path, required=True)
    parser.add_argument("--source-structured-predictions", type=Path, required=True)
    parser.add_argument("--source-structured-result", type=Path, required=True)
    parser.add_argument("--source-structured-validation", type=Path, required=True)


def cumulative_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = structured_paths(args)
    paths.update({
        "cumulative_config": args.cumulative_config.resolve(),
        "reserved_manifest": args.reserved_manifest.resolve(),
        "source_structured_flat_model": args.source_structured_flat_model.resolve(),
        "source_structured_time_model": args.source_structured_time_model.resolve(),
        "source_structured_predictions": args.source_structured_predictions.resolve(),
        "source_structured_result": args.source_structured_result.resolve(),
        "source_structured_validation": args.source_structured_validation.resolve(),
    })
    return paths


def validate_cumulative_sources(
    paths: Mapping[str, Path], config: Mapping[str, Any],
) -> Mapping[str, Any]:
    source = config["immutable_source"]
    for source_key, path_key in (
        ("structured_config_file_sha256", "structured_config"),
        ("structured_result_file_sha256", "source_structured_result"),
        ("structured_validation_file_sha256", "source_structured_validation"),
        ("structured_flat_model_file_sha256", "source_structured_flat_model"),
        ("structured_time_model_file_sha256", "source_structured_time_model"),
        ("structured_predictions_file_sha256", "source_structured_predictions"),
        ("trajectory_array_file_sha256", "array"),
        ("trajectory_metadata_file_sha256", "metadata"),
        ("reserved_episode_manifest_file_sha256", "reserved_manifest"),
    ):
        _require(
            _file_sha256(paths[path_key]) == source[source_key],
            "cumulative-delta-q immutable source differs",
        )
    result = _load(paths["source_structured_result"])
    validation = _load(paths["source_structured_validation"])
    rows = _read_manifest(
        paths["reserved_manifest"], _file_sha256(paths["reserved_manifest"])
    )
    _require(
        result.get("result_payload_sha256")
        == source["structured_result_payload_sha256"]
        and result.get("decision", {}).get("structured_representation_GO") is False
        and validation.get("validation_payload_sha256")
        == source["structured_validation_payload_sha256"]
        and validation.get("valid") is True
        and [row["case_id"] for row in rows]
        == config["population"]["reserved_final_episode_ids"]
        and all(row.get("selection_role") == "new_reserved_final_evaluation"
                for row in rows),
        "cumulative-delta-q validated source differs",
    )
    return result


def prepare(
    args: argparse.Namespace,
) -> tuple[dict[str, Path], Mapping[str, Any], Mapping[str, Any], Any, ...]:
    paths = cumulative_paths(args)
    config = load_cumulative_delta_config(paths["cumulative_config"])
    structured_config = load_structured_config(paths["structured_config"])
    time_config = load_time_decoder_config(paths["time_config"])
    source_structured = validate_cumulative_sources(paths, config)
    validate_structured_sources(paths, structured_config)
    (
        one_sided_config, factorized_config, complete_dataset,
        complete_collection, _, _, _, _, _, arrays, sensitivities, _,
    ) = load_inputs(paths)
    validate_time_sources(paths, time_config, arrays)
    structured_arrays, normalization, representation = structured_orientation_arrays(
        complete_dataset, arrays, structured_config,
    )
    local_geometry = fit_local_geometry_jacobians(
        structured_arrays, one_sided_config,
    )
    return (
        paths, config, source_structured, factorized_config, complete_dataset,
        complete_collection, structured_arrays, sensitivities, normalization,
        representation, local_geometry,
    )


def evaluate_prediction(
    *, paths: Mapping[str, Path], config: Mapping[str, Any],
    factorized_config: Mapping[str, Any], complete_dataset: Mapping[str, Any],
    complete_collection: Mapping[str, Any], arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any], predicted_q: Any,
) -> dict[str, Any]:
    geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        arrays, predicted_q, ("validation", "test"),
    )
    predicted_margin = geometry.pop("predicted_minimum_margin_m")
    exact_static = geometry.pop("exact_q_static_minimum_margin_m")
    metrics = prediction_metrics(
        predicted_q, predicted_margin, arrays, sensitivities, factorized_config,
    )
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
    decision = cumulative_delta_decision(
        metrics=metrics, validation_support=support["validation"],
        test_support=support["test"], temporal=temporal, config=config,
    )
    return {
        "predicted_margin": predicted_margin, "exact_static": exact_static,
        "metrics": metrics, "temporal": temporal, "support": support,
        "geometry": geometry, "decision": decision,
    }


def main() -> int:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_cumulative_arguments(parser)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, config, source_structured, factorized_config, complete_dataset,
        complete_collection, arrays, sensitivities, normalization,
        representation, local_geometry,
    ) = prepare(args)
    paths.update({
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(), "output": args.output.resolve(),
    })
    torch.set_num_threads(8)
    models, state, training = train_cumulative_delta_ensemble(
        arrays, sensitivities, local_geometry, config, normalization,
    )
    predicted_q = predict_cumulative_delta(models, state, arrays)
    model_receipt = save_cumulative_delta_weights(
        paths["experimental_model"], state,
    )
    evaluation = evaluate_prediction(
        paths=paths, config=config, factorized_config=factorized_config,
        complete_dataset=complete_dataset, complete_collection=complete_collection,
        arrays=arrays, sensitivities=sensitivities, predicted_q=predicted_q,
    )
    np.savez_compressed(
        paths["predictions"], experimental_joint_position_rad=predicted_q,
        experimental_minimum_margin_m=evaluation["predicted_margin"],
        exact_q_static_minimum_margin_m=evaluation["exact_static"],
    )
    prediction_receipt = {
        "path": str(paths["predictions"]),
        "file_sha256": _file_sha256(paths["predictions"]),
        "joint_sha256": _hash_array(predicted_q),
        "margin_sha256": _hash_array(evaluation["predicted_margin"]),
        "exact_static_sha256": _hash_array(evaluation["exact_static"]),
    }
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "representation": representation,
        "immutable_structured_baseline": {
            "result_file_sha256": _file_sha256(paths["source_structured_result"]),
            "metrics": source_structured["metrics"],
            "decision": source_structured["decision"],
        },
        "local_geometry_jacobian_audit": local_geometry["audit"],
        "training": training, "model": model_receipt,
        "predictions": prediction_receipt,
        "metrics": {
            **evaluation["metrics"], "temporal_joint": evaluation["temporal"],
            "eligible_support": evaluation["support"],
            "experimental_geometry": evaluation["geometry"],
        },
        "decision": evaluation["decision"],
        "reserved_episode_receipt": {
            "manifest_file_sha256": _file_sha256(paths["reserved_manifest"]),
            "collection_submitted": False,
        },
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
        "metrics": result["metrics"], "decision": result["decision"],
        "training": training,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
