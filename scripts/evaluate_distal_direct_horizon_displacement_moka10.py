#!/usr/bin/env python3
"""Train direct-horizon displacement and evaluate frozen reserved episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.factorized_cumulative_delta_q import (
    load_cumulative_delta_weights, predict_cumulative_delta,
)
from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    COLLECTION_SCHEMA, COLLECTION_VALIDATION_SCHEMA, DATASET_SCHEMA, RESULT_SCHEMA,
    apply_structured_representation, direct_horizon_decision,
    load_direct_horizon_config, predict_direct_horizon,
    save_direct_horizon_weights, sensitivity_magnitude_metrics,
    temporal_error_metrics, train_direct_horizon_ensemble,
)
from main.multilink_ellipsoid.factorized_execution_pilot import (
    cosine_summary, dataset_arrays, load_weights, payload_sha256, predict,
    safety_metrics, secant_predictions, sensitivity_arrays,
)
from main.multilink_ellipsoid.factorized_structured_orientation import (
    load_structured_config,
)
from main.multilink_ellipsoid.factorized_time_conditioned_decoder import (
    eligible_state_support,
)
from scripts.collect_distal_complete_osc_margin_moka10 import (
    _GEOMETRY_PLACEHOLDER_CASE_ID, _geometry_placeholder_row,
)
from scripts.collect_distal_direct_horizon_reserved_moka10 import (
    restore_json_snapshot,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import (
    add_cumulative_arguments, prepare,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_factorized_execution_moka10 import (
    _test_sensitivity_mask,
)
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs,
)
from scripts.evaluate_distal_native_geom_inventory_moka10 import _read_manifest
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_array(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def add_direct_horizon_arguments(parser: argparse.ArgumentParser) -> None:
    add_cumulative_arguments(parser)
    parser.add_argument("--direct-horizon-config", type=Path, required=True)
    parser.add_argument("--reserved-collection", type=Path, required=True)
    parser.add_argument("--reserved-metadata", type=Path, required=True)
    parser.add_argument("--reserved-array", type=Path, required=True)
    parser.add_argument("--reserved-validation", type=Path, required=True)
    parser.add_argument("--direct-reserved-manifest", type=Path, required=True)
    parser.add_argument("--previous-factorized-model", type=Path, required=True)
    parser.add_argument("--previous-cumulative-model", type=Path, required=True)


def direct_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {key: value.resolve() for key, value in {
        "direct_config": args.direct_horizon_config,
        "reserved_collection": args.reserved_collection,
        "reserved_metadata": args.reserved_metadata,
        "reserved_array": args.reserved_array,
        "reserved_validation": args.reserved_validation,
        "direct_reserved_manifest": args.direct_reserved_manifest,
        "previous_factorized_model": args.previous_factorized_model,
        "previous_cumulative_model": args.previous_cumulative_model,
    }.items()}


def load_reserved(
    paths: Mapping[str, Path], config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    import numpy as np

    collection = _load(paths["reserved_collection"])
    metadata = _load(paths["reserved_metadata"])
    validation = _load(paths["reserved_validation"])
    _require(
        collection.get("schema_version") == COLLECTION_SCHEMA
        and collection.get("result_payload_sha256")
        == payload_sha256(collection, "result_payload_sha256")
        and collection.get("decision", {}).get("collection_gate_pass") is True
        and collection["dataset"]["metadata_file_sha256"]
        == _file_sha256(paths["reserved_metadata"])
        and collection["dataset"]["array_file_sha256"]
        == _file_sha256(paths["reserved_array"])
        and metadata.get("schema_version") == DATASET_SCHEMA
        and metadata.get("dataset_payload_sha256")
        == payload_sha256(metadata, "dataset_payload_sha256")
        and metadata["array_dataset"]["file_sha256"]
        == _file_sha256(paths["reserved_array"])
        and validation.get("schema_version") == COLLECTION_VALIDATION_SCHEMA
        and validation.get("validation_payload_sha256")
        == payload_sha256(validation, "validation_payload_sha256")
        and validation.get("valid") is True
        and validation["collection_file_sha256"]
        == _file_sha256(paths["reserved_collection"])
        and validation["metadata_file_sha256"]
        == _file_sha256(paths["reserved_metadata"])
        and validation["array_file_sha256"]
        == _file_sha256(paths["reserved_array"])
        and metadata["summary"]["state_count"]
        == config["population"]["reserved_state_count"]
        and metadata["summary"]["rollout_count"]
        == config["population"]["reserved_rollout_count"],
        "direct-horizon reserved population differs",
    )
    archive = np.load(paths["reserved_array"], allow_pickle=False)
    arrays = dataset_arrays(metadata, archive)
    return collection, metadata, arrays


def validate_direct_sources(
    paths: Mapping[str, Path], config: Mapping[str, Any],
) -> None:
    source = config["immutable_source"]
    for source_key, path_key in (
        ("population_manifest_file_sha256", "population"),
        ("reserved_prediction_manifest_file_sha256", "direct_reserved_manifest"),
        ("factorized_config_file_sha256", "factorized_config"),
        ("structured_config_file_sha256", "structured_config"),
        ("geometry_config_file_sha256", "geometry"),
        ("exact_box_config_file_sha256", "exact_box"),
        ("complete_dataset_file_sha256", "complete_dataset"),
        ("complete_collection_file_sha256", "complete_collection"),
        ("trajectory_array_file_sha256", "array"),
        ("trajectory_metadata_file_sha256", "metadata"),
        ("trajectory_collection_file_sha256", "collection"),
        ("previous_factorized_model_file_sha256", "previous_factorized_model"),
        ("previous_cumulative_model_file_sha256", "previous_cumulative_model"),
    ):
        _require(
            _file_sha256(paths[path_key]) == source[source_key],
            "direct-horizon immutable source differs",
        )


def _source_geometry(
    *, paths: Mapping[str, Path], complete_collection: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source_rows = []
    for path in (paths["selected"], paths["same_task"], paths["targeted"]):
        source_rows.extend(_read_manifest(path, _file_sha256(path)))
    source_by_case = {str(row["case_id"]): row for row in source_rows}
    placeholder_row = _geometry_placeholder_row(source_by_case)
    placeholder_path = paths["archived"] / placeholder_row["archived_relative_path"]
    placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == placeholder_row["archived_file_sha256"]
        and placeholder.get("result_payload_sha256")
        == placeholder_row["archived_payload_sha256"]
        and placeholder.get("case_id") == _GEOMETRY_PLACEHOLDER_CASE_ID,
        "direct-horizon geometry placeholder differs",
    )
    return placeholder, complete_collection


def evaluate_reserved_geometry(
    *, paths: Mapping[str, Path], factorized_config: Mapping[str, Any],
    complete_collection: Mapping[str, Any], metadata: Mapping[str, Any],
    arrays: Mapping[str, Any], predictions: Mapping[str, Any],
    return_trace: bool = False,
) -> dict[str, Any]:
    """Evaluate all model joint traces through the same known FK/ellipsoids."""

    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config, minimum_union_support_gap_witnesses,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import load_shadow_config

    placeholder, _ = _source_geometry(
        paths=paths, complete_collection=complete_collection,
    )
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    reserved_rows = _read_manifest(
        paths["direct_reserved_manifest"],
        _file_sha256(paths["direct_reserved_manifest"])
    )
    row_by_case = {str(row["case_id"]): row for row in reserved_rows}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    exact_q = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
    predicted = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in predictions.items()
    }
    if any(value.shape != exact_q.shape for value in predicted.values()):
        raise ValueError("direct-horizon reserved prediction shape differs")
    margins = {
        name: np.full((len(exact_q), 7), np.nan, dtype=np.float64)
        for name in (*predicted.keys(), "exact_q_static")
    }
    traces = {
        name: np.full((len(exact_q), 51, 7), np.nan, dtype=np.float64)
        for name in (*predicted.keys(), "exact_q_static")
    } if return_trace else None
    center_error_sum = {name: 0.0 for name in predicted}
    center_error_count = {name: 0 for name in predicted}
    center_error_values = {name: [] for name in predicted}
    rows_by_identity = {
        (int(state), int(candidate)): int(row)
        for row, (state, candidate) in enumerate(zip(
            np.asarray(arrays["state_index"], dtype=np.int64),
            np.asarray(arrays["candidate_index"], dtype=np.int64),
        ))
    }
    state_records = list(metadata["state_records"])
    evaluated = 0
    for case_id in sorted(row_by_case):
        case = population[case_id]
        validate_case_row(case, paths["repo"])
        env = probe_env = None
        try:
            env, probe_env, _, _, setup = _build_pair(runtime, case)
            _require(
                setup["obstacle_name"] == row_by_case[case_id]["active_obstacle_name"],
                "direct-horizon reserved geometry obstacle differs",
            )
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                archived=placeholder, env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            robot = env.robots[0]
            indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
            for state in sorted(
                (item for item in state_records if item["case_id"] == case_id),
                key=lambda item: int(item["state_index"]),
            ):
                restore_json_snapshot(env, state["complete_snapshot"])
                obstacles = probe._obstacles(env)
                state_index = int(state["state_index"])
                for candidate_index in range(
                    int(factorized_config["candidate_design"][
                        "expected_candidate_count_per_state"
                    ])
                ):
                    row = rows_by_identity[(state_index, candidate_index)]
                    traces = {"exact_q_static": exact_q[row]}
                    traces.update({name: value[row] for name, value in predicted.items()})
                    trace_margins = {name: [] for name in traces}
                    random_candidate = int(arrays["source_code"][row]) == 2
                    exact_centers_by_k = []
                    for k in range(51):
                        for name, trace in traces.items():
                            env.sim.data.qpos[indexes] = trace[k]
                            env.sim.forward()
                            links = probe._ellipsoids(env)[:7]
                            gap, _ = minimum_union_support_gap_witnesses(
                                links, obstacles
                            )
                            trace_margins[name].append(gap)
                            if random_candidate:
                                centers = np.asarray(
                                    [item.center for item in links], dtype=np.float64
                                )
                                if name == "exact_q_static":
                                    exact_centers_by_k.append(centers)
                                else:
                                    error = np.linalg.norm(
                                        centers - exact_centers_by_k[k], axis=1
                                    )
                                    center_error_sum[name] += float(np.sum(error ** 2))
                                    center_error_count[name] += int(error.size)
                                    center_error_values[name].extend(error.tolist())
                    for name, values in trace_margins.items():
                        trace = np.asarray(values, dtype=np.float64)
                        margins[name][row] = np.min(trace, axis=0)
                        if return_trace:
                            traces[name][row] = trace
                    evaluated += 1
                restore_json_snapshot(env, state["complete_snapshot"])
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    if evaluated != len(exact_q) or any(
        not np.all(np.isfinite(value)) for value in margins.values()
    ):
        raise ValueError("direct-horizon reserved geometry count differs")
    geometry_metrics = {}
    for name in predicted:
        values = np.asarray(center_error_values[name], dtype=np.float64)
        geometry_metrics[name] = {
            "random_link_center_error_count": int(center_error_count[name]),
            "random_link_center_RMSE_m": float(math.sqrt(
                center_error_sum[name] / center_error_count[name]
            )),
            "random_link_center_p95_m": float(np.quantile(values, 0.95)),
            "random_link_center_maximum_m": float(np.max(values)),
        }
    output = {
        "minimum_margin_m": margins,
        "geometry_metrics": geometry_metrics,
        "evaluated_action_count": int(evaluated),
        "obstacle_geometry_mode": "fixed_k0_exact_box_union",
    }
    if return_trace:
        output["clearance_trace_m"] = traces
    return output


def model_metrics(
    *, predicted_q: Any, predicted_margin: Any, arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any], factorized_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    random = np.asarray(arrays["source_code"], dtype=np.int8) == 2
    random_temporal = temporal_error_metrics(
        np.asarray(predicted_q)[random],
        np.asarray(arrays["joint_position_rad"])[random],
    )
    all_temporal = temporal_error_metrics(
        predicted_q, arrays["joint_position_rad"]
    )
    q_secant = secant_predictions(predicted_q, sensitivities, arrays)
    margin_secant = secant_predictions(predicted_margin, sensitivities, arrays)
    exact_q_sensitivity = np.asarray(
        sensitivities["joint_sensitivity_rad_per_action"], dtype=np.float64
    )
    exact_margin_sensitivity = np.asarray(
        sensitivities["margin_sensitivity_m_per_action"], dtype=np.float64
    )
    joint_cosine = cosine_summary(exact_q_sensitivity, q_secant)
    margin_cosine = cosine_summary(exact_margin_sensitivity, margin_secant)
    return {
        "random_temporal_joint": random_temporal,
        "all_candidate_temporal_joint": all_temporal,
        "safety": safety_metrics(
            arrays["minimum_margin_m"], predicted_margin, arrays,
            factorized_config, split_name="test", evaluation_only=True,
        ),
        "eligible_support": eligible_state_support(
            exact_margin=arrays["minimum_margin_m"],
            predicted_margin=predicted_margin, arrays=arrays, split_name="test",
        ),
        "joint_sensitivity_cosine": joint_cosine,
        "joint_sensitivity_magnitude": sensitivity_magnitude_metrics(
            exact_q_sensitivity, q_secant,
        ),
        "safety_sensitivity_cosine": margin_cosine,
        "safety_sensitivity_magnitude": sensitivity_magnitude_metrics(
            exact_margin_sensitivity, margin_secant,
        ),
    }


def main() -> int:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_direct_horizon_arguments(parser)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, _, _, factorized_config, complete_dataset,
        complete_collection, training_arrays, training_sensitivities,
        normalization, representation, local_geometry,
    ) = prepare(args)
    paths.update(direct_paths(args))
    paths.update({
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "output": args.output.resolve(),
    })
    config = load_direct_horizon_config(paths["direct_config"])
    structured_config = load_structured_config(paths["structured_config"])
    validate_direct_sources(paths, config)
    collection, reserved_metadata, reserved_raw = load_reserved(paths, config)
    reserved_arrays, reserved_representation = apply_structured_representation(
        complete_dataset, reserved_raw, structured_config,
    )
    reserved_normalized = (
        np.asarray(reserved_arrays["features"], dtype=np.float64)
        - np.asarray(normalization["feature_mean"], dtype=np.float64)
    ) / np.asarray(normalization["feature_std"], dtype=np.float64)
    reserved_representation["maximum_absolute_training_normalized_value"] = float(
        np.max(np.abs(reserved_normalized))
    )
    reserved_representation[
        "training_normalized_features_sha256"
    ] = _hash_array(reserved_normalized)
    if (
        reserved_representation["removed_indexes_sha256"]
        != representation["removed_feature_indexes_sha256"]
        or reserved_representation["retained_indexes_sha256"]
        != representation["retained_feature_indexes_sha256"]
    ):
        raise ValueError("direct-horizon reserved representation differs")
    torch.set_num_threads(8)
    models, state, training = train_direct_horizon_ensemble(
        training_arrays, training_sensitivities, local_geometry, config,
        normalization,
    )
    new_training_q = predict_direct_horizon(models, state, training_arrays)
    new_reserved_q = predict_direct_horizon(models, state, reserved_arrays)
    previous_factorized_models, previous_factorized_state = load_weights(
        paths["previous_factorized_model"]
    )
    previous_factorized_q = predict(
        previous_factorized_models, previous_factorized_state, reserved_raw,
    )
    previous_cumulative_models, previous_cumulative_state = (
        load_cumulative_delta_weights(paths["previous_cumulative_model"])
    )
    previous_cumulative_q = predict_cumulative_delta(
        previous_cumulative_models, previous_cumulative_state, reserved_arrays,
    )
    # Existing validation is part of the gate; old diagnostic test episodes are
    # not used to tune or decide this new experiment.
    validation_geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        training_arrays, new_training_q, ("validation",),
    )
    validation_margin = validation_geometry.pop("predicted_minimum_margin_m")
    validation_exact_static = validation_geometry.pop(
        "exact_q_static_minimum_margin_m"
    )
    validation_metrics = safety_metrics(
        training_arrays["minimum_margin_m"], validation_margin,
        training_arrays, factorized_config, split_name="validation",
        evaluation_only=True,
    )
    validation_support = eligible_state_support(
        exact_margin=training_arrays["minimum_margin_m"],
        predicted_margin=validation_margin, arrays=training_arrays,
        split_name="validation",
    )
    reserved_geometry = evaluate_reserved_geometry(
        paths=paths, factorized_config=factorized_config,
        complete_collection=complete_collection, metadata=reserved_metadata,
        arrays=reserved_raw, predictions={
            "direct_horizon_displacement": new_reserved_q,
            "previous_factorized_one_sided": previous_factorized_q,
            "previous_cumulative_increments": previous_cumulative_q,
        },
    )
    margins = reserved_geometry.pop("minimum_margin_m")
    exact_static = margins.pop("exact_q_static")
    model_predictions = {
        "direct_horizon_displacement": new_reserved_q,
        "previous_factorized_one_sided": previous_factorized_q,
        "previous_cumulative_increments": previous_cumulative_q,
    }
    reserved_sensitivities = sensitivity_arrays(reserved_raw)
    comparisons = {
        name: model_metrics(
            predicted_q=model_predictions[name], predicted_margin=margins[name],
            arrays=reserved_raw, sensitivities=reserved_sensitivities,
            factorized_config=factorized_config,
        ) for name in model_predictions
    }
    exact_static_audit = safety_metrics(
        reserved_raw["minimum_margin_m"], exact_static, reserved_raw,
        factorized_config, split_name="test", evaluation_only=True,
    )
    experimental = comparisons["direct_horizon_displacement"]
    decision = direct_horizon_decision(
        validation_metrics=validation_metrics,
        reserved_metrics=experimental["safety"],
        reserved_exact_static_metrics=exact_static_audit,
        validation_support=validation_support,
        reserved_support=experimental["eligible_support"],
        reserved_temporal=experimental["random_temporal_joint"],
        joint_sensitivity=experimental["joint_sensitivity_cosine"],
        safety_sensitivity=experimental["safety_sensitivity_cosine"],
        matched_cumulative=comparisons[
            "previous_cumulative_increments"
        ]["random_temporal_joint"],
        config=config,
    )
    model_receipt = save_direct_horizon_weights(
        paths["experimental_model"], state,
    )
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["predictions"],
        direct_horizon_joint_position_rad=new_reserved_q,
        previous_factorized_joint_position_rad=previous_factorized_q,
        previous_cumulative_joint_position_rad=previous_cumulative_q,
        direct_horizon_minimum_margin_m=margins[
            "direct_horizon_displacement"
        ],
        previous_factorized_minimum_margin_m=margins[
            "previous_factorized_one_sided"
        ],
        previous_cumulative_minimum_margin_m=margins[
            "previous_cumulative_increments"
        ],
        exact_q_static_minimum_margin_m=exact_static,
        validation_direct_horizon_minimum_margin_m=validation_margin,
        validation_exact_q_static_minimum_margin_m=validation_exact_static,
    )
    prediction_receipt = {
        "path": str(paths["predictions"]),
        "file_sha256": _file_sha256(paths["predictions"]),
        "direct_horizon_joint_sha256": _hash_array(new_reserved_q),
        "direct_horizon_margin_sha256": _hash_array(
            margins["direct_horizon_displacement"]
        ),
        "previous_factorized_joint_sha256": _hash_array(previous_factorized_q),
        "previous_factorized_margin_sha256": _hash_array(
            margins["previous_factorized_one_sided"]
        ),
        "previous_cumulative_joint_sha256": _hash_array(previous_cumulative_q),
        "previous_cumulative_margin_sha256": _hash_array(
            margins["previous_cumulative_increments"]
        ),
        "exact_static_margin_sha256": _hash_array(exact_static),
        "validation_margin_sha256": _hash_array(validation_margin),
    }
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "training_population_representation": representation,
        "reserved_population_representation": reserved_representation,
        "local_geometry_jacobian_audit": local_geometry["audit"],
        "reserved_collection": {
            "path": str(paths["reserved_collection"]),
            "file_sha256": _file_sha256(paths["reserved_collection"]),
            "payload_sha256": collection["result_payload_sha256"],
            "validation_file_sha256": _file_sha256(paths["reserved_validation"]),
        },
        "training": training, "model": model_receipt,
        "predictions": prediction_receipt,
        "metrics": {
            "validation_direct_horizon_safety": validation_metrics,
            "validation_direct_horizon_support": validation_support,
            "validation_geometry": validation_geometry,
            "reserved_exact_q_static_audit": exact_static_audit,
            "reserved_prediction_comparison": comparisons,
            "reserved_geometry": reserved_geometry,
        },
        "decision": decision,
        "forbidden_action_receipt": {
            key + "_executed": False
            for key in config["forbidden_before_prediction_pass"]
        },
        "future_untouched_intervention_manifest": {
            "file_sha256": config["immutable_source"][
                "future_untouched_manifest_file_sha256"
            ],
            "opened_or_evaluated": False,
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "metrics": result["metrics"], "decision": decision,
        "training": training,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
