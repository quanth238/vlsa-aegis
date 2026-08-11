#!/usr/bin/env python3
"""Train and evaluate the paired factorized-execution pilot on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.factorized_execution_pilot import (
    COLLECTION_SCHEMA, DATASET_SCHEMA, DATASET_VALIDATION_SCHEMA, RESULT_SCHEMA, cosine_summary,
    dataset_arrays, factorized_decision, load_config, load_weights, payload_sha256,
    predict, safety_metrics, save_weights, secant_predictions,
    sensitivity_arrays, train_ensemble,
)
from scripts.collect_distal_complete_osc_margin_moka10 import (
    _GEOMETRY_PLACEHOLDER_CASE_ID, _geometry_placeholder_row,
)
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action, _restore_env, _snapshot_env,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_native_geom_inventory_moka10 import _read_manifest
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_array(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _test_sensitivity_mask(
    sensitivities: Mapping[str, Any], arrays: Mapping[str, Any],
) -> Any:
    import numpy as np

    split_by_state = {
        int(state): str(split) for state, split in zip(
            np.asarray(arrays["state_index"], dtype=np.int64),
            np.asarray(arrays["split"], dtype=object),
        )
    }
    return np.asarray([
        split_by_state[int(state)] == "test"
        for state in np.asarray(sensitivities["state_index"], dtype=np.int64)
    ], dtype=bool)


def evaluate_predicted_geometry(
    *, repo_root: Path, config: Mapping[str, Any], complete_dataset: Mapping[str, Any],
    arrays: Mapping[str, Any], predicted_q: Any, population_manifest: Path,
    selected_manifest: Path, same_task_manifest: Path, targeted_manifest: Path,
    archived_root: Path, geometry_config_path: Path, exact_box_config_path: Path,
    complete_collection: Mapping[str, Any], return_margin: bool = True,
    return_trace: bool = False,
    evaluation_splits: Sequence[str] = ("test",),
) -> dict[str, Any]:
    """Evaluate predicted q with fixed-k0 exact-box ellipsoid geometry."""

    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config, minimum_union_support_gap_witnesses,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import load_shadow_config

    source = complete_collection["config"]["immutable_source"]
    for path, key in (
        (population_manifest, "source_population_manifest_sha256"),
        (selected_manifest, "selected_manifest_file_sha256"),
        (same_task_manifest, "same_task_manifest_file_sha256"),
        (targeted_manifest, "targeted_manifest_file_sha256"),
        (geometry_config_path, "geometry_config_file_sha256"),
        (exact_box_config_path, "exact_box_config_file_sha256"),
    ):
        _require(_file_sha256(path) == source[key],
                 "factorized-execution geometry source differs")
    selected_rows = []
    for path in (selected_manifest, same_task_manifest, targeted_manifest):
        selected_rows.extend(_read_manifest(path, _file_sha256(path)))
    row_by_case = {str(item["case_id"]): item for item in selected_rows}
    placeholder_row = _geometry_placeholder_row(row_by_case)
    placeholder_path = archived_root / placeholder_row["archived_relative_path"]
    geometry_placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == placeholder_row["archived_file_sha256"]
        and geometry_placeholder.get("result_payload_sha256")
        == placeholder_row["archived_payload_sha256"]
        and geometry_placeholder.get("case_id") == _GEOMETRY_PLACEHOLDER_CASE_ID,
        "factorized-execution geometry placeholder differs",
    )
    population = {item["case_id"]: item for item in read_jsonl(population_manifest)}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    q_prediction = np.asarray(predicted_q, dtype=np.float64)
    if q_prediction.shape != np.asarray(arrays["joint_position_rad"]).shape:
        raise ValueError("factorized-execution predicted q shape differs")
    factor_margin = np.full(
        np.asarray(arrays["minimum_margin_m"]).shape, np.nan, dtype=np.float64
    )
    exact_q_static_margin = np.full_like(factor_margin, np.nan)
    factor_trace = (
        np.full((len(q_prediction), 51, 7), np.nan, dtype=np.float64)
        if return_trace else None
    )
    exact_q_static_trace = (
        np.full((len(q_prediction), 51, 7), np.nan, dtype=np.float64)
        if return_trace else None
    )
    row_by_identity = {
        (int(state), int(candidate)): int(row)
        for row, (state, candidate) in enumerate(zip(
            np.asarray(arrays["state_index"], dtype=np.int64),
            np.asarray(arrays["candidate_index"], dtype=np.int64),
        ))
    }
    center_squared_error_sum = 0.0
    center_error_count = 0
    center_errors = []
    evaluated_action_count = 0
    requested_splits = tuple(str(item) for item in evaluation_splits)
    if (
        not requested_splits
        or len(set(requested_splits)) != len(requested_splits)
        or any(item not in {"train", "validation", "test"} for item in requested_splits)
    ):
        raise ValueError("factorized-execution geometry split selection differs")
    evaluation_states = [
        item for item in complete_dataset["state_records"]
        if str(item["split"]) in requested_splits
    ]
    case_ids = sorted(set(str(item["case_id"]) for item in evaluation_states))
    for case_id in case_ids:
        selected_row = row_by_case[case_id]
        archived_path = archived_root / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256")
            == selected_row["archived_payload_sha256"],
            "factorized-execution geometry archive differs",
        )
        case = population[case_id]
        validate_case_row(case, repo_root)
        env = probe_env = None
        try:
            env, probe_env, _, _, setup = _build_pair(runtime, case)
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                archived=geometry_placeholder, env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            states = sorted(
                (item for item in evaluation_states if item["case_id"] == case_id),
                key=lambda item: int(item["state_step"]),
            )
            by_step = {int(item["state_step"]): item for item in states}
            maximum_step = max(by_step)
            archived_actions = archived["actions"]
            robot = env.robots[0]
            position_indexes = np.asarray(
                robot._ref_joint_pos_indexes, dtype=np.int64
            )
            for step in range(maximum_step + 1):
                if step in by_step:
                    state = by_step[step]
                    _require(
                        hashlib.sha256(_dynamic_state_vector(env).tobytes()).hexdigest()
                        == state["dynamic_state_sha256"],
                        "factorized-execution geometry state differs",
                    )
                    snapshot = _snapshot_env(env)
                    obstacles = probe._obstacles(env)
                    for candidate_index in range(
                        int(config["candidate_design"]["expected_candidate_count_per_state"])
                    ):
                        row = row_by_identity[(int(state["state_index"]), candidate_index)]
                        predicted_trace = q_prediction[row]
                        exact_trace = np.asarray(
                            arrays["joint_position_rad"][row], dtype=np.float64
                        )
                        trace_margin = []
                        exact_trace_margin = []
                        evaluate_error = bool(
                            int(np.asarray(arrays["source_code"])[row]) == 2
                        )
                        for predicted_joint, exact_joint in zip(
                            predicted_trace, exact_trace
                        ):
                            env.sim.data.qpos[position_indexes] = predicted_joint
                            env.sim.forward()
                            predicted_links = probe._ellipsoids(env)[:7]
                            predicted_gap, _ = minimum_union_support_gap_witnesses(
                                predicted_links, obstacles
                            )
                            trace_margin.append(predicted_gap)
                            if evaluate_error:
                                predicted_centers = np.asarray(
                                    [item.center for item in predicted_links],
                                    dtype=np.float64,
                                )
                            env.sim.data.qpos[position_indexes] = exact_joint
                            env.sim.forward()
                            exact_links = probe._ellipsoids(env)[:7]
                            exact_gap, _ = minimum_union_support_gap_witnesses(
                                exact_links, obstacles
                            )
                            exact_trace_margin.append(exact_gap)
                            if evaluate_error:
                                exact_centers = np.asarray(
                                    [item.center for item in exact_links], dtype=np.float64
                                )
                                errors = np.linalg.norm(
                                    predicted_centers - exact_centers, axis=1
                                )
                                center_squared_error_sum += float(np.sum(errors ** 2))
                                center_error_count += int(errors.size)
                                center_errors.extend(errors.tolist())
                        predicted_trace_margin = np.asarray(
                            trace_margin, dtype=np.float64
                        )
                        exact_static_trace_margin = np.asarray(
                            exact_trace_margin, dtype=np.float64
                        )
                        factor_margin[row] = np.min(
                            predicted_trace_margin, axis=0
                        )
                        exact_q_static_margin[row] = np.min(
                            exact_static_trace_margin, axis=0
                        )
                        if return_trace:
                            factor_trace[row] = predicted_trace_margin
                            exact_q_static_trace[row] = exact_static_trace_margin
                        evaluated_action_count += 1
                    _restore_env(env, snapshot)
                if step < maximum_step:
                    env.step(_canonical_action(
                        archived_actions[step], step
                    ).tolist())
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    expected = len(evaluation_states) * int(
        config["candidate_design"]["expected_candidate_count_per_state"]
    )
    if (
        evaluated_action_count != expected
        or np.count_nonzero(np.all(np.isfinite(factor_margin), axis=1)) != expected
        or np.count_nonzero(np.all(
            np.isfinite(exact_q_static_margin), axis=1
        )) != expected
    ):
        raise ValueError("factorized-execution geometry evaluation count differs")
    center = np.asarray(center_errors, dtype=np.float64)
    prefix = "test" if requested_splits == ("test",) else "selected"
    output = {
        "predicted_minimum_margin_m": factor_margin if return_margin else None,
        "exact_q_static_minimum_margin_m": (
            exact_q_static_margin if return_margin else None
        ),
        "%s_evaluated_action_count" % prefix: int(evaluated_action_count),
        "%s_random_link_center_error_count" % prefix: int(center_error_count),
        "%s_random_link_center_RMSE_m" % prefix: float(math.sqrt(
            center_squared_error_sum / center_error_count
        )),
        "%s_random_link_center_p95_m" % prefix: float(np.quantile(center, 0.95)),
        "%s_random_link_center_maximum_m" % prefix: float(np.max(center)),
        "obstacle_geometry_mode": "fixed_k0_exact_box_union",
    }
    if requested_splits != ("test",):
        output["evaluation_splits"] = list(requested_splits)
    if return_trace:
        output["predicted_clearance_trace_m"] = factor_trace
        output["exact_q_static_clearance_trace_m"] = exact_q_static_trace
    return output


def main() -> int:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--complete-dataset", type=Path, required=True)
    parser.add_argument("--complete-collection", type=Path, required=True)
    parser.add_argument("--trajectory-metadata", type=Path, required=True)
    parser.add_argument("--trajectory-array", type=Path, required=True)
    parser.add_argument("--trajectory-collection", type=Path, required=True)
    parser.add_argument("--trajectory-validation", type=Path, required=True)
    parser.add_argument("--matched-direct-result", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--same-task-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--direct-model", type=Path, required=True)
    parser.add_argument("--factorized-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "complete_dataset": args.complete_dataset,
        "complete_collection": args.complete_collection,
        "metadata": args.trajectory_metadata, "array": args.trajectory_array,
        "collection": args.trajectory_collection,
        "dataset_validation": args.trajectory_validation,
        "matched": args.matched_direct_result,
        "population": args.population_manifest, "selected": args.selected_manifest,
        "same_task": args.same_task_manifest, "targeted": args.targeted_manifest,
        "archived": args.archived_root, "geometry": args.geometry_config,
        "exact_box": args.exact_box_config, "direct_model": args.direct_model,
        "factorized_model": args.factorized_model,
        "predictions": args.predictions, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    complete_dataset = _load(paths["complete_dataset"])
    complete_collection = _load(paths["complete_collection"])
    metadata = _load(paths["metadata"])
    collection = _load(paths["collection"])
    dataset_validation = _load(paths["dataset_validation"])
    matched = _load(paths["matched"])
    _require(
        metadata.get("schema_version") == DATASET_SCHEMA
        and metadata.get("dataset_payload_sha256")
        == payload_sha256(metadata, "dataset_payload_sha256")
        and metadata["array_dataset"]["file_sha256"] == _file_sha256(paths["array"])
        and collection.get("schema_version") == COLLECTION_SCHEMA
        and collection.get("result_payload_sha256")
        == payload_sha256(collection, "result_payload_sha256")
        and collection["trajectory_dataset"]["metadata_file_sha256"]
        == _file_sha256(paths["metadata"])
        and collection["trajectory_dataset"]["array_file_sha256"]
        == _file_sha256(paths["array"])
        and bool(collection["decision"]["collection_gate_pass"])
        and dataset_validation.get("schema_version") == DATASET_VALIDATION_SCHEMA
        and dataset_validation.get("validation_payload_sha256")
        == payload_sha256(dataset_validation, "validation_payload_sha256")
        and dataset_validation.get("valid") is True
        and dataset_validation.get("metadata_file_sha256")
        == _file_sha256(paths["metadata"])
        and dataset_validation.get("array_file_sha256")
        == _file_sha256(paths["array"])
        and _file_sha256(paths["matched"])
        == source["matched_direct_result_file_sha256"]
        and matched.get("result_payload_sha256")
        == source["matched_direct_result_payload_sha256"],
        "factorized-execution evaluation input differs",
    )
    archive = np.load(paths["array"], allow_pickle=False)
    arrays = dataset_arrays(metadata, archive)
    sensitivities = sensitivity_arrays(arrays)
    torch.set_num_threads(8)
    direct_models, direct_state, direct_training = train_ensemble(
        arrays, sensitivities, config, arm="direct_margin"
    )
    factor_models, factor_state, factor_training = train_ensemble(
        arrays, sensitivities, config, arm="factorized_execution"
    )
    direct_prediction = predict(direct_models, direct_state, arrays)
    factor_q_prediction = predict(factor_models, factor_state, arrays)
    direct_artifact = save_weights(paths["direct_model"], direct_state)
    factor_artifact = save_weights(paths["factorized_model"], factor_state)
    direct_metrics = {
        split: safety_metrics(
            arrays["minimum_margin_m"], direct_prediction, arrays, config,
            split_name=split, evaluation_only=(split == "test"),
        ) for split in ("train", "validation", "test")
    }
    test_random = (
        (np.asarray(arrays["split"], dtype=object) == "test")
        & (np.asarray(arrays["source_code"], dtype=np.int8) == 2)
    )
    joint_error = factor_q_prediction[test_random] - np.asarray(
        arrays["joint_position_rad"]
    )[test_random]
    joint_metrics = {
        "test_random_row_count": int(np.count_nonzero(test_random)),
        "test_random_joint_trajectory_RMSE_rad": float(np.sqrt(np.mean(joint_error ** 2))),
        "test_random_joint_absolute_error_p95_rad": float(np.quantile(
            np.abs(joint_error), 0.95
        )),
        "test_random_joint_absolute_error_maximum_rad": float(np.max(
            np.abs(joint_error)
        )),
        "test_random_endpoint_RMSE_rad": float(np.sqrt(np.mean(
            joint_error[:, -1] ** 2
        ))),
    }
    sensitivity_test = _test_sensitivity_mask(sensitivities, arrays)
    factor_q_secant = secant_predictions(
        factor_q_prediction, sensitivities, arrays
    )
    direct_margin_secant = secant_predictions(
        direct_prediction, sensitivities, arrays
    )
    joint_sensitivity = cosine_summary(
        np.asarray(sensitivities["joint_sensitivity_rad_per_action"])[sensitivity_test],
        factor_q_secant[sensitivity_test],
    )
    direct_margin_sensitivity = cosine_summary(
        np.asarray(sensitivities["margin_sensitivity_m_per_action"])[sensitivity_test],
        direct_margin_secant[sensitivity_test],
    )
    geometry = evaluate_predicted_geometry(
        repo_root=paths["repo"], config=config,
        complete_dataset=complete_dataset, arrays=arrays,
        predicted_q=factor_q_prediction,
        population_manifest=paths["population"],
        selected_manifest=paths["selected"],
        same_task_manifest=paths["same_task"],
        targeted_manifest=paths["targeted"], archived_root=paths["archived"],
        geometry_config_path=paths["geometry"],
        exact_box_config_path=paths["exact_box"],
        complete_collection=complete_collection,
    )
    factor_margin = geometry.pop("predicted_minimum_margin_m")
    exact_q_static_margin = geometry.pop("exact_q_static_minimum_margin_m")
    exact_geometry_metrics = safety_metrics(
        arrays["minimum_margin_m"], exact_q_static_margin, arrays, config,
        split_name="test", evaluation_only=True,
    )
    factor_metrics = safety_metrics(
        arrays["minimum_margin_m"], factor_margin, arrays, config,
        split_name="test", evaluation_only=True,
    )
    factor_margin_secant = secant_predictions(
        factor_margin, sensitivities, arrays
    )
    factor_margin_sensitivity = cosine_summary(
        np.asarray(sensitivities["margin_sensitivity_m_per_action"])[sensitivity_test],
        factor_margin_secant[sensitivity_test],
    )
    test_raw_contact = np.asarray(arrays["raw_contact_count"])[test_random] > 0
    factor_safe = np.all(factor_margin[test_random] >= 0.0, axis=1)
    physical_false_safe = int(np.count_nonzero(factor_safe & test_raw_contact))
    decision = factorized_decision(
        collection_pass=bool(collection["decision"]["collection_gate_pass"]),
        direct_metrics=direct_metrics["test"],
        exact_geometry_metrics=exact_geometry_metrics,
        factorized_metrics=factor_metrics,
        joint_rmse_rad=joint_metrics["test_random_joint_trajectory_RMSE_rad"],
        link_center_rmse_m=geometry["test_random_link_center_RMSE_m"],
        joint_sensitivity_cosine=float(joint_sensitivity["mean_cosine"]),
        margin_sensitivity_cosine=float(factor_margin_sensitivity["mean_cosine"]),
        config=config,
    )
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["predictions"], direct_margin_m=direct_prediction,
        factorized_joint_position_rad=factor_q_prediction,
        factorized_minimum_margin_m=factor_margin,
        exact_q_static_minimum_margin_m=exact_q_static_margin,
    )
    prediction_receipt = {
        "path": str(paths["predictions"]),
        "file_sha256": _file_sha256(paths["predictions"]),
        "direct_margin_sha256": _hash_array(direct_prediction),
        "factorized_joint_sha256": _hash_array(factor_q_prediction),
        "factorized_margin_sha256": _hash_array(factor_margin),
        "exact_q_static_margin_sha256": _hash_array(exact_q_static_margin),
    }
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "collection": {
            "path": str(paths["collection"]),
            "file_sha256": _file_sha256(paths["collection"]),
            "payload_sha256": collection["result_payload_sha256"],
        },
        "dataset": {
            "metadata_file_sha256": _file_sha256(paths["metadata"]),
            "metadata_payload_sha256": metadata["dataset_payload_sha256"],
            "array_file_sha256": _file_sha256(paths["array"]),
        },
        "training": {
            "direct_margin": direct_training,
            "factorized_execution": factor_training,
        },
        "models": {
            "direct_margin": direct_artifact,
            "factorized_execution": factor_artifact,
        },
        "predictions": prediction_receipt,
        "metrics": {
            "direct_margin": direct_metrics,
            "factorized_joint": joint_metrics,
            "factorized_geometry": geometry,
            "exact_q_static_geometry_margin": exact_geometry_metrics,
            "factorized_margin": factor_metrics,
            "joint_sensitivity": joint_sensitivity,
            "direct_margin_sensitivity": direct_margin_sensitivity,
            "factorized_margin_sensitivity": factor_margin_sensitivity,
            "factorized_raw_contact_false_safe_action_count": physical_false_safe,
            "prior_matched_direct_reference": {
                "old56_test_false_safe_actions": 72,
                "old56_test_near_boundary_RMSE_m": 0.003058,
                "completeOSC_test_false_safe_actions": 271,
                "completeOSC_test_near_boundary_RMSE_m": 0.008497,
            },
        },
        "forbidden_action_receipt": {
            "poisson_or_SDF_executed": False, "calibration_executed": False,
            "QP_executed": False, "closed_loop_E05_executed": False,
            "VLA_policy_inference_executed": False,
        },
        "decision": decision,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "decision": decision,
        "direct_test": direct_metrics["test"],
        "factorized_test": factor_metrics,
        "exact_q_static_geometry": exact_geometry_metrics,
        "joint": joint_metrics,
        "geometry": geometry,
        "joint_sensitivity": joint_sensitivity,
        "margin_sensitivity": factor_margin_sensitivity,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
