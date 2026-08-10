#!/usr/bin/env python3
"""Audit immutable factorized-execution terminal bias without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_execution_pilot import (
    _arm_targets, dataset_arrays, load_config, load_weights, payload_sha256,
    predict,
)
from main.multilink_ellipsoid.factorized_terminal_bias_audit import (
    RESULT_SCHEMA, audit_decision, ensemble_disagreement_audit,
    geometry_signal_decision, load_terminal_bias_config, residual_audit,
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


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factorized-config", type=Path, required=True)
    parser.add_argument("--complete-dataset", type=Path, required=True)
    parser.add_argument("--complete-collection", type=Path, required=True)
    parser.add_argument("--trajectory-metadata", type=Path, required=True)
    parser.add_argument("--trajectory-array", type=Path, required=True)
    parser.add_argument("--trajectory-collection", type=Path, required=True)
    parser.add_argument("--factorized-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--factorized-result", type=Path, required=True)
    parser.add_argument("--factorized-validation", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--same-task-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)


def resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "factorized_config": args.factorized_config,
        "complete_dataset": args.complete_dataset,
        "complete_collection": args.complete_collection,
        "metadata": args.trajectory_metadata, "array": args.trajectory_array,
        "collection": args.trajectory_collection, "model": args.factorized_model,
        "predictions": args.predictions, "result": args.factorized_result,
        "validation": args.factorized_validation,
        "population": args.population_manifest, "selected": args.selected_manifest,
        "same_task": args.same_task_manifest, "targeted": args.targeted_manifest,
        "archived": args.archived_root, "geometry": args.geometry_config,
        "exact_box": args.exact_box_config,
    }.items()}


def _hash_array(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _save_npz_atomic(path: Path, arrays: Mapping[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _validate_sources(paths: Mapping[str, Path], config: Mapping[str, Any]) -> tuple[Any, ...]:
    source = config["immutable_source"]
    for source_key, path_key in (
        ("factorized_config_file_sha256", "factorized_config"),
        ("trajectory_array_file_sha256", "array"),
        ("trajectory_metadata_file_sha256", "metadata"),
        ("trajectory_collection_file_sha256", "collection"),
        ("factorized_model_file_sha256", "model"),
        ("predictions_file_sha256", "predictions"),
        ("factorized_result_file_sha256", "result"),
        ("factorized_validation_file_sha256", "validation"),
    ):
        _require(_file_sha256(paths[path_key]) == source[source_key],
                 "terminal-bias immutable source differs")
    factorized_config = load_config(paths["factorized_config"])
    complete_dataset = _load(paths["complete_dataset"])
    complete_collection = _load(paths["complete_collection"])
    metadata = _load(paths["metadata"])
    collection = _load(paths["collection"])
    result = _load(paths["result"])
    validation = _load(paths["validation"])
    _require(
        result.get("result_payload_sha256")
        == source["factorized_result_payload_sha256"]
        and result.get("source", {}).get("commit")
        == source["factorized_result_commit"]
        and result.get("decision", {}).get("factorization_GO") is False
        and validation.get("valid") is True
        and validation.get("validation_payload_sha256")
        == source["factorized_validation_payload_sha256"]
        and collection.get("decision", {}).get("collection_gate_pass") is True,
        "terminal-bias validated source differs",
    )
    return (
        factorized_config, complete_dataset, complete_collection, metadata,
        collection, result, validation,
    )


def _member_predictions(models: Any, state: Mapping[str, Any], arrays: Mapping[str, Any]) -> Any:
    import numpy as np
    import torch

    normalized = (
        np.asarray(arrays["features"], dtype=np.float64) - state["feature_mean"]
    ) / state["feature_std"]
    outputs = []
    with torch.no_grad():
        for model in models:
            parts = []
            for start in range(0, len(normalized), 2048):
                parts.append(model(torch.as_tensor(
                    normalized[start:start + 2048], dtype=torch.float32,
                )).cpu().numpy())
            outputs.append(np.concatenate(parts, axis=0))
    residual = np.asarray(outputs).reshape(
        (len(models), len(normalized)) + tuple(state["output_shape"])
    )
    _, base, _ = _arm_targets(arrays, str(state["arm"]))
    return base[None, ...] + residual


def _ridge(x: Any, y: Any, regularization: float) -> Any:
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.linalg.solve(
        x.T @ x + float(regularization) * np.eye(x.shape[1]), x.T @ y,
    )


def _cosine_rows(first: Any, second: Any) -> Any:
    import numpy as np

    a = np.asarray(first, dtype=np.float64).reshape(-1, first.shape[-1])
    b = np.asarray(second, dtype=np.float64).reshape(-1, second.shape[-1])
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denominator > 1e-12
    return np.sum(a[valid] * b[valid], axis=1) / denominator[valid]


def analyze_records(records: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    exact_h = np.asarray(records["exact_static_clearance_m"], dtype=np.float64)
    predicted_h = np.asarray(records["predicted_static_clearance_m"], dtype=np.float64)
    split_names = np.asarray(["train", "validation", "test"], dtype=object)
    splits = split_names[np.asarray(records["split_code"], dtype=np.int8)]
    state_index = np.asarray(records["state_index"], dtype=np.int64)
    exact_q = np.asarray(records["exact_joint_position_rad"], dtype=np.float64)
    predicted_q = np.asarray(records["predicted_joint_position_rad"], dtype=np.float64)
    nominal_q = np.asarray(records["nominal_joint_position_rad"], dtype=np.float64)
    nominal_h = np.asarray(records["nominal_static_clearance_m"], dtype=np.float64)
    jacobian = np.asarray(records["exact_dh_dq_m_per_rad"], dtype=np.float64)
    action_delta = np.asarray(records["action_delta"], dtype=np.float64)
    boundary = float(config["residual_audit"]["near_boundary_absolute_margin_m"])
    j_rows = jacobian[state_index]
    q0_rows = nominal_q[state_index]
    h0_rows = nominal_h[state_index]
    exact_delta = exact_h - h0_rows
    linear_delta = np.einsum("bkrj,bkj->bkr", j_rows, exact_q - q0_rows)
    prediction_error = predicted_h - exact_h
    linear_prediction_error = np.einsum(
        "bkrj,bkj->bkr", j_rows, predicted_q - exact_q,
    )
    near = np.abs(exact_h) <= boundary
    nontrivial = np.abs(exact_delta) >= float(
        config["geometry_signal_audit"]["minimum_nontrivial_clearance_delta_m"]
    )
    denominator = float(
        np.linalg.norm(exact_delta[nontrivial])
        * np.linalg.norm(linear_delta[nontrivial])
    )
    signed_cosine = float(
        exact_delta[nontrivial] @ linear_delta[nontrivial] / denominator
    )
    signed_agreement = float(np.mean(
        np.sign(exact_delta[nontrivial]) == np.sign(linear_delta[nontrivial])
    ))
    ridge = float(config["geometry_signal_audit"]["random_action_ridge"])
    generator = np.random.default_rng(int(
        config["geometry_signal_audit"]["candidate_resample_seed"]
    ))
    resample_count = int(config["geometry_signal_audit"]["candidate_resample_count"])
    resample_size = int(config["geometry_signal_audit"]["candidate_resample_size"])
    composed_direct_cosines = []
    resampled_cosines = []
    for state in range(int(config["population"]["state_count"])):
        rows = np.flatnonzero(state_index == state)
        if len(rows) != 64:
            raise ValueError("terminal-bias state random population differs")
        u = action_delta[rows]
        q_delta = exact_q[rows] - nominal_q[state][None, ...]
        h_delta = exact_h[rows] - nominal_h[state][None, ...]
        q_coefficient = np.empty((51, 14, 7), dtype=np.float64)
        h_coefficient = np.empty((51, 14, 7), dtype=np.float64)
        for substep in range(51):
            q_coefficient[substep] = _ridge(u, q_delta[:, substep], ridge)
            h_coefficient[substep] = _ridge(u, h_delta[:, substep], ridge)
        composed = np.einsum(
            "krj,kdj->krd", jacobian[state], q_coefficient,
        )
        direct = np.transpose(h_coefficient, (0, 2, 1))
        active = np.min(np.abs(exact_h[rows]), axis=0) <= boundary
        if np.any(active):
            composed_direct_cosines.extend(
                _cosine_rows(composed[active], direct[active]).tolist()
            )
        for _ in range(resample_count):
            sample = generator.choice(len(rows), size=resample_size, replace=False)
            q_resampled = np.empty_like(q_coefficient)
            for substep in range(51):
                q_resampled[substep] = _ridge(
                    u[sample], q_delta[sample, substep], ridge,
                )
            resampled = np.einsum(
                "krj,kdj->krd", jacobian[state], q_resampled,
            )
            if np.any(active):
                resampled_cosines.extend(
                    _cosine_rows(composed[active], resampled[active]).tolist()
                )
    geometry_metrics = {
        "near_boundary_scalar_count": int(np.count_nonzero(near)),
        "near_boundary_linearization_RMSE_m": float(np.sqrt(np.mean(
            (linear_delta[near] - exact_delta[near]) ** 2
        ))),
        "near_boundary_prediction_error_linearization_RMSE_m": float(np.sqrt(
            np.mean((linear_prediction_error[near] - prediction_error[near]) ** 2)
        )),
        "signed_delta_cosine": signed_cosine,
        "signed_delta_sign_agreement": signed_agreement,
        "composed_vs_direct_action_gradient_median_cosine": float(np.median(
            composed_direct_cosines
        )),
        "composed_vs_direct_action_gradient_p05_cosine": float(np.quantile(
            composed_direct_cosines, 0.05
        )),
        "resampled_action_gradient_count": int(len(resampled_cosines)),
        "resampled_action_gradient_median_cosine": float(np.median(
            resampled_cosines
        )),
        "resampled_action_gradient_p05_cosine": float(np.quantile(
            resampled_cosines, 0.05
        )),
    }
    residual = residual_audit(
        predicted_h=predicted_h, exact_h=exact_h, splits=splits, config=config,
    )
    geometry = geometry_signal_decision(geometry_metrics, config)
    ensemble = ensemble_disagreement_audit(
        member_minimum_m=np.asarray(records["ensemble_member_minimum_m"]),
        class_code=np.asarray(records["ensemble_class_code"]), config=config,
    )
    decision = audit_decision(
        residual=residual, geometry=geometry, ensemble=ensemble,
    )
    return {
        "residual": residual,
        "geometry_signal_metrics": geometry_metrics,
        "geometry_signal_decision": geometry,
        "ensemble_disagreement": ensemble,
        "decision": decision,
    }


def collect_records(paths: Mapping[str, Path], config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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

    (
        factorized_config, complete_dataset, complete_collection, metadata,
        _, _, _,
    ) = _validate_sources(paths, config)
    _require(
        _file_sha256(paths["complete_dataset"])
        == factorized_config["immutable_source"]["complete_dataset_file_sha256"]
        and _file_sha256(paths["complete_collection"])
        == factorized_config["immutable_source"]["complete_collection_file_sha256"],
        "terminal-bias complete source differs",
    )
    dataset_archive = np.load(paths["array"], allow_pickle=False)
    arrays = dataset_arrays(metadata, dataset_archive)
    prediction_archive = np.load(paths["predictions"], allow_pickle=False)
    stored_prediction = prediction_archive[
        "factorized_joint_position_rad"
    ].astype(np.float64)
    models, model_state = load_weights(paths["model"])
    member_q = _member_predictions(models, model_state, arrays)
    # Preserve job 37980's arithmetic order: average float32 residuals first,
    # then add the float64 q0 baseline. Averaging baseline-plus-member outputs
    # changes the receipt at float32 rounding scale.
    recomputed_mean_q = predict(models, model_state, arrays)
    _require(
        np.array_equal(recomputed_mean_q, stored_prediction),
        "terminal-bias ensemble mean did not reproduce",
    )
    # Use the immutable saved mean for the exact same arithmetic receipt as
    # job 37980; member values remain available for disagreement evaluation.
    mean_q = stored_prediction
    random_mask = np.asarray(arrays["source_code"], dtype=np.int8) == 2
    random_rows = np.flatnonzero(random_mask)
    _require(len(random_rows) == config["population"]["expected_action_count"],
             "terminal-bias random population differs")
    compact_by_row = {int(row): index for index, row in enumerate(random_rows)}
    count = len(random_rows)
    records = {
        "row_index": random_rows.astype(np.int64),
        "state_index": np.asarray(arrays["state_index"], dtype=np.int64)[random_rows],
        "candidate_index": np.asarray(arrays["candidate_index"], dtype=np.int64)[random_rows],
        "split_code": np.asarray(dataset_archive["split_code"], dtype=np.int8)[random_rows],
        "action_delta": np.full((count, 14), np.nan, dtype=np.float64),
        "exact_joint_position_rad": np.asarray(
            arrays["joint_position_rad"], dtype=np.float64
        )[random_rows],
        "predicted_joint_position_rad": mean_q[random_rows],
        "exact_static_clearance_m": np.full((count, 51, 7), np.nan, dtype=np.float64),
        "predicted_static_clearance_m": np.full((count, 51, 7), np.nan, dtype=np.float64),
        "nominal_joint_position_rad": np.full((85, 51, 7), np.nan, dtype=np.float64),
        "nominal_static_clearance_m": np.full((85, 51, 7), np.nan, dtype=np.float64),
        "exact_dh_dq_m_per_rad": np.full((85, 51, 7, 7), np.nan, dtype=np.float64),
        "ensemble_member_minimum_m": np.full((count, 5), np.nan, dtype=np.float64),
        "ensemble_class_code": np.zeros(count, dtype=np.int8),
    }
    source = complete_collection["config"]["immutable_source"]
    for path_key, source_key in (
        ("population", "source_population_manifest_sha256"),
        ("selected", "selected_manifest_file_sha256"),
        ("same_task", "same_task_manifest_file_sha256"),
        ("targeted", "targeted_manifest_file_sha256"),
        ("geometry", "geometry_config_file_sha256"),
        ("exact_box", "exact_box_config_file_sha256"),
    ):
        _require(_file_sha256(paths[path_key]) == source[source_key],
                 "terminal-bias geometry source differs")
    selected_rows = []
    for path_key in ("selected", "same_task", "targeted"):
        selected_rows.extend(_read_manifest(
            paths[path_key], _file_sha256(paths[path_key])
        ))
    row_by_case = {str(item["case_id"]): item for item in selected_rows}
    placeholder_row = _geometry_placeholder_row(row_by_case)
    placeholder_path = paths["archived"] / placeholder_row["archived_relative_path"]
    geometry_placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == placeholder_row["archived_file_sha256"]
        and geometry_placeholder.get("result_payload_sha256")
        == placeholder_row["archived_payload_sha256"]
        and geometry_placeholder.get("case_id") == _GEOMETRY_PLACEHOLDER_CASE_ID,
        "terminal-bias geometry placeholder differs",
    )
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    state_records = complete_dataset["state_records"]
    epsilon = float(config["geometry_signal_audit"]["joint_step_rad"])
    action_flat = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    for case_id in sorted(set(item["case_id"] for item in state_records)):
        selected_row = row_by_case[case_id]
        archived_path = paths["archived"] / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256")
            == selected_row["archived_payload_sha256"],
            "terminal-bias case archive differs",
        )
        case = population[case_id]
        validate_case_row(case, paths["repo"])
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
                quadratic_tolerance=1e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            states = sorted(
                (item for item in state_records if item["case_id"] == case_id),
                key=lambda item: int(item["state_step"]),
            )
            by_step = {int(item["state_step"]): item for item in states}
            maximum_step = max(by_step)
            robot = env.robots[0]
            position_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
            for step in range(maximum_step + 1):
                if step in by_step:
                    state = by_step[step]
                    state_id = int(state["state_index"])
                    _require(
                        hashlib.sha256(_dynamic_state_vector(env).tobytes()).hexdigest()
                        == state["dynamic_state_sha256"],
                        "terminal-bias state identity differs",
                    )
                    snapshot = _snapshot_env(env)
                    obstacles = probe._obstacles(env)
                    state_rows = np.flatnonzero(
                        np.asarray(arrays["state_index"], dtype=np.int64) == state_id
                    )
                    by_candidate = {
                        int(np.asarray(arrays["candidate_index"])[row]): int(row)
                        for row in state_rows
                    }
                    nominal_row = by_candidate[0]
                    nominal_action = action_flat[nominal_row]
                    nominal_q = np.asarray(
                        arrays["joint_position_rad"][nominal_row], dtype=np.float64
                    )
                    records["nominal_joint_position_rad"][state_id] = nominal_q

                    def gap(joint: Any) -> Any:
                        env.sim.data.qpos[position_indexes] = joint
                        env.sim.forward()
                        links = probe._ellipsoids(env)[:7]
                        value, _ = minimum_union_support_gap_witnesses(links, obstacles)
                        return value

                    for substep in range(51):
                        center = nominal_q[substep].copy()
                        records["nominal_static_clearance_m"][state_id, substep] = gap(center)
                        for joint_index in range(7):
                            positive = center.copy()
                            negative = center.copy()
                            positive[joint_index] += epsilon
                            negative[joint_index] -= epsilon
                            records["exact_dh_dq_m_per_rad"][
                                state_id, substep, :, joint_index
                            ] = (gap(positive) - gap(negative)) / (2.0 * epsilon)
                    compact_indexes = []
                    for row in state_rows:
                        if not random_mask[row]:
                            continue
                        compact = compact_by_row[int(row)]
                        compact_indexes.append(compact)
                        records["action_delta"][compact] = action_flat[row] - nominal_action
                        for substep in range(51):
                            records["exact_static_clearance_m"][compact, substep] = gap(
                                arrays["joint_position_rad"][row, substep]
                            )
                            records["predicted_static_clearance_m"][compact, substep] = gap(
                                mean_q[row, substep]
                            )
                    compact_indexes = np.asarray(compact_indexes, dtype=np.int64)
                    exact_min = np.min(
                        records["exact_static_clearance_m"][compact_indexes], axis=(1, 2)
                    )
                    predicted_min = np.min(
                        records["predicted_static_clearance_m"][compact_indexes], axis=(1, 2)
                    )
                    exact_safe = exact_min >= 0.0
                    predicted_safe = predicted_min >= 0.0
                    false_local = np.flatnonzero(predicted_safe & ~exact_safe)
                    controls = np.flatnonzero(predicted_safe == exact_safe)
                    selected_controls = []
                    unused = set(controls.tolist())
                    for false_index in false_local:
                        candidates = sorted(unused) if unused else controls.tolist()
                        if not candidates:
                            continue
                        chosen = min(candidates, key=lambda item: (
                            abs(exact_min[item] - exact_min[false_index]), item
                        ))
                        selected_controls.append(chosen)
                        unused.discard(chosen)
                    false_compact = compact_indexes[false_local]
                    control_compact = compact_indexes[np.asarray(
                        sorted(set(selected_controls)), dtype=np.int64
                    )] if selected_controls else np.asarray([], dtype=np.int64)
                    records["ensemble_class_code"][false_compact] = 1
                    records["ensemble_class_code"][control_compact] = 2
                    for compact in np.concatenate((false_compact, control_compact)):
                        row = int(random_rows[compact])
                        for member_index in range(member_q.shape[0]):
                            member_values = [
                                gap(member_q[member_index, row, substep])
                                for substep in range(51)
                            ]
                            records["ensemble_member_minimum_m"][
                                compact, member_index
                            ] = float(np.min(member_values))
                    _restore_env(env, snapshot)
                if step < maximum_step:
                    env.step(_canonical_action(archived["actions"][step], step).tolist())
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    required = [
        "action_delta", "exact_joint_position_rad", "predicted_joint_position_rad",
        "exact_static_clearance_m", "predicted_static_clearance_m",
        "nominal_joint_position_rad", "nominal_static_clearance_m",
        "exact_dh_dq_m_per_rad",
    ]
    _require(all(np.all(np.isfinite(records[key])) for key in required),
             "terminal-bias records incomplete")
    test = records["split_code"] == 2
    stored_predicted_minimum = prediction_archive[
        "factorized_minimum_margin_m"
    ][random_rows[test]]
    stored_exact_minimum = prediction_archive[
        "exact_q_static_minimum_margin_m"
    ][random_rows[test]]
    reproduced_predicted = np.min(
        records["predicted_static_clearance_m"][test], axis=1
    )
    reproduced_exact = np.min(records["exact_static_clearance_m"][test], axis=1)
    _require(
        np.array_equal(reproduced_predicted, stored_predicted_minimum)
        and np.array_equal(reproduced_exact, stored_exact_minimum),
        "terminal-bias prior test geometry did not reproduce",
    )
    analysis = analyze_records(records, config)
    analysis["source_reproduction"] = {
        "ensemble_mean_joint_prediction_exact": True,
        "test_predicted_static_minimum_exact": True,
        "test_exact_q_static_minimum_exact": True,
    }
    return records, analysis


def main() -> int:
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = resolved_paths(args)
    config = load_terminal_bias_config(paths["config"])
    records, analysis = collect_records(paths, config)
    _save_npz_atomic(args.records.resolve(), records)
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "records": {
            "path": str(args.records.resolve()),
            "file_sha256": _file_sha256(args.records.resolve()),
            "row_index_sha256": _hash_array(records["row_index"]),
            "exact_clearance_sha256": _hash_array(
                records["exact_static_clearance_m"]
            ),
            "predicted_clearance_sha256": _hash_array(
                records["predicted_static_clearance_m"]
            ),
        },
        "metrics": {key: value for key, value in analysis.items() if key != "decision"},
        "decision": analysis["decision"],
        "forbidden_action_receipt": {
            key: False for key in config["forbidden_actions"]
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "decision": result["decision"],
        "geometry": result["metrics"]["geometry_signal_metrics"],
        "ensemble": result["metrics"]["ensemble_disagreement"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
