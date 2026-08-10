#!/usr/bin/env python3
"""Diagnose whether factorized false-safes have larger surface-position error."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_execution_pilot import (
    dataset_arrays, load_config, payload_sha256,
)
from main.multilink_ellipsoid.surface_error_diagnostic import (
    SURFACE_RESULT_SCHEMA, ellipsoid_support_point, load_surface_config,
    matched_control_indexes, spearman_correlation, surface_loss_decision,
    vector_summary,
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
        "collection": args.trajectory_collection, "predictions": args.predictions,
        "factorized_result": args.factorized_result,
        "factorized_validation": args.factorized_validation,
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


def _validate_inputs(paths: Mapping[str, Path], config: Mapping[str, Any]) -> tuple[Any, ...]:
    source = config["immutable_source"]
    for key, path_key in (
        ("factorized_config_file_sha256", "factorized_config"),
        ("trajectory_array_file_sha256", "array"),
        ("trajectory_metadata_file_sha256", "metadata"),
        ("trajectory_collection_file_sha256", "collection"),
        ("predictions_file_sha256", "predictions"),
        ("factorized_result_file_sha256", "factorized_result"),
        ("factorized_validation_file_sha256", "factorized_validation"),
    ):
        _require(_file_sha256(paths[path_key]) == source[key],
                 "surface-error immutable source differs")
    factorized_config = load_config(paths["factorized_config"])
    complete_dataset = _load(paths["complete_dataset"])
    complete_collection = _load(paths["complete_collection"])
    metadata = _load(paths["metadata"])
    collection = _load(paths["collection"])
    factorized_result = _load(paths["factorized_result"])
    factorized_validation = _load(paths["factorized_validation"])
    _require(
        factorized_result.get("result_payload_sha256")
        == source["factorized_result_payload_sha256"]
        and factorized_result.get("source", {}).get("commit")
        == source["factorized_result_commit"]
        and factorized_result.get("decision", {}).get("factorization_GO") is False
        and factorized_result["metrics"]["factorized_margin"][
            "false_safe_action_count"
        ] == config["population"]["expected_false_safe_action_count"]
        and factorized_validation.get("valid") is True
        and factorized_validation.get("validation_payload_sha256")
        == source["factorized_validation_payload_sha256"]
        and collection.get("decision", {}).get("collection_gate_pass") is True,
        "surface-error validated factorized source differs",
    )
    return (
        factorized_config, complete_dataset, complete_collection, metadata,
        collection, factorized_result, factorized_validation,
    )


def compute_surface_error_records(
    paths: Mapping[str, Path], config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the safety-critical surface-position errors on test random rows."""

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
        _, factorized_result, _,
    ) = _validate_inputs(paths, config)
    _require(
        _file_sha256(paths["complete_dataset"])
        == factorized_config["immutable_source"]["complete_dataset_file_sha256"]
        and _file_sha256(paths["complete_collection"])
        == factorized_config["immutable_source"]["complete_collection_file_sha256"],
        "surface-error complete source differs",
    )
    dataset_archive = np.load(paths["array"], allow_pickle=False)
    arrays = dataset_arrays(metadata, dataset_archive)
    prediction_archive = np.load(paths["predictions"], allow_pickle=False)
    predicted_q = prediction_archive["factorized_joint_position_rad"].astype(np.float64)
    stored_predicted_margin = prediction_archive[
        "factorized_minimum_margin_m"
    ].astype(np.float64)
    stored_exact_static_margin = prediction_archive[
        "exact_q_static_minimum_margin_m"
    ].astype(np.float64)
    exact_dynamic_margin = np.asarray(arrays["minimum_margin_m"], dtype=np.float64)
    test_random = (
        (np.asarray(arrays["split"], dtype=object) == "test")
        & (np.asarray(arrays["source_code"], dtype=np.int8) == 2)
    )
    test_rows = np.flatnonzero(test_random)
    _require(len(test_rows) == config["population"]["expected_action_count"],
             "surface-error test population differs")
    exact_safe = np.all(exact_dynamic_margin >= 0.0, axis=1)
    predicted_safe = np.all(stored_predicted_margin >= 0.0, axis=1)
    false_safe_mask = test_random & predicted_safe & ~exact_safe
    correct_unsafe_mask = test_random & ~predicted_safe & ~exact_safe
    correct_safe_mask = test_random & predicted_safe & exact_safe
    false_unsafe_mask = test_random & ~predicted_safe & exact_safe
    _require(
        np.count_nonzero(false_safe_mask)
        == config["population"]["expected_false_safe_action_count"],
        "surface-error false-safe population differs",
    )

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
                 "surface-error geometry source differs")
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
        "surface-error geometry placeholder differs",
    )
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    compact_by_row = {int(row): index for index, row in enumerate(test_rows)}
    count = len(test_rows)
    record_arrays = {
        "row_index": test_rows.astype(np.int64),
        "state_index": np.asarray(arrays["state_index"], dtype=np.int64)[test_rows],
        "candidate_index": np.asarray(arrays["candidate_index"], dtype=np.int64)[test_rows],
        "class_code": np.full(count, -1, dtype=np.int8),
        "exact_dynamic_minimum_margin_m": np.min(exact_dynamic_margin[test_rows], axis=1),
        "exact_static_minimum_margin_m": np.full(count, np.nan, dtype=np.float64),
        "predicted_static_minimum_margin_m": np.full(count, np.nan, dtype=np.float64),
        "static_margin_overestimate_m": np.full(count, np.nan, dtype=np.float64),
        "worst_surface_position_error_m": np.full(count, np.nan, dtype=np.float64),
        "worst_center_position_error_m": np.full(count, np.nan, dtype=np.float64),
        "worst_surface_clearance_overestimate_m": np.full(count, np.nan, dtype=np.float64),
        "worst_center_clearance_overestimate_m": np.full(count, np.nan, dtype=np.float64),
        "worst_support_clearance_overestimate_m": np.full(count, np.nan, dtype=np.float64),
        "worst_substep_index": np.full(count, -1, dtype=np.int16),
        "worst_ellipsoid_row": np.full(count, -1, dtype=np.int8),
        "worst_obstacle_index": np.full(count, -1, dtype=np.int16),
        "worst_predicted_witness_matches_exact": np.zeros(count, dtype=np.int8),
        "worst_link_body_name": np.full(count, "", dtype="<U64"),
    }
    class_by_row = np.full(len(exact_safe), -1, dtype=np.int8)
    class_by_row[correct_safe_mask] = 0
    class_by_row[correct_unsafe_mask] = 1
    class_by_row[false_safe_mask] = 2
    class_by_row[false_unsafe_mask] = 3
    record_arrays["class_code"][:] = class_by_row[test_rows]
    recomputed_predicted = np.full((count, 7), np.nan, dtype=np.float64)
    recomputed_exact = np.full((count, 7), np.nan, dtype=np.float64)
    test_states = [
        item for item in complete_dataset["state_records"] if item["split"] == "test"
    ]

    for case_id in config["population"]["test_case_ids"]:
        selected_row = row_by_case[case_id]
        archived_path = paths["archived"] / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256")
            == selected_row["archived_payload_sha256"],
            "surface-error case archive differs",
        )
        case = population[case_id]
        validate_case_row(case, paths["repo"])
        env = probe_env = None
        try:
            env, probe_env, _, _, setup = _build_pair(runtime, case)
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config,
                exact_box_config=exact_box_config,
                archived=geometry_placeholder, env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            states = sorted(
                (item for item in test_states if item["case_id"] == case_id),
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
                        "surface-error state identity differs",
                    )
                    snapshot = _snapshot_env(env)
                    obstacles = probe._obstacles(env)
                    state_rows = np.flatnonzero(
                        (np.asarray(arrays["state_index"]) == int(state["state_index"]))
                        & test_random
                    )
                    for row in state_rows:
                        compact = compact_by_row[int(row)]
                        predicted_trace = predicted_q[row]
                        exact_trace = np.asarray(
                            arrays["joint_position_rad"][row], dtype=np.float64
                        )
                        exact_gaps = []
                        predicted_gaps = []
                        surface_errors = []
                        center_errors = []
                        surface_retractions = []
                        center_retractions = []
                        support_retractions = []
                        exact_witness_rows = []
                        predicted_witness_rows = []
                        body_names = []
                        for predicted_joint, exact_joint in zip(
                            predicted_trace, exact_trace
                        ):
                            env.sim.data.qpos[position_indexes] = predicted_joint
                            env.sim.forward()
                            predicted_links = probe._ellipsoids(env)[:7]
                            predicted_gap, predicted_witness = (
                                minimum_union_support_gap_witnesses(
                                    predicted_links, obstacles
                                )
                            )
                            env.sim.data.qpos[position_indexes] = exact_joint
                            env.sim.forward()
                            exact_links = probe._ellipsoids(env)[:7]
                            exact_gap, exact_witness = minimum_union_support_gap_witnesses(
                                exact_links, obstacles
                            )
                            exact_gaps.append(exact_gap)
                            predicted_gaps.append(predicted_gap)
                            one_surface = []
                            one_center = []
                            one_surface_retraction = []
                            one_center_retraction = []
                            one_support_retraction = []
                            one_names = []
                            for ellipsoid_row, (predicted_link, exact_link) in enumerate(
                                zip(predicted_links, exact_links)
                            ):
                                obstacle = obstacles[int(exact_witness[ellipsoid_row])]
                                direction = np.asarray(obstacle.center) - np.asarray(
                                    exact_link.center
                                )
                                direction = direction / np.linalg.norm(direction)
                                exact_surface = ellipsoid_support_point(
                                    exact_link, direction
                                )
                                predicted_surface = ellipsoid_support_point(
                                    predicted_link, direction
                                )
                                surface_delta = predicted_surface - exact_surface
                                center_delta = (
                                    np.asarray(predicted_link.center)
                                    - np.asarray(exact_link.center)
                                )
                                surface_retraction = -float(direction @ surface_delta)
                                center_retraction = -float(direction @ center_delta)
                                one_surface.append(float(np.linalg.norm(surface_delta)))
                                one_center.append(float(np.linalg.norm(center_delta)))
                                one_surface_retraction.append(surface_retraction)
                                one_center_retraction.append(center_retraction)
                                one_support_retraction.append(
                                    surface_retraction - center_retraction
                                )
                                one_names.append(str(exact_link.body_name))
                            surface_errors.append(one_surface)
                            center_errors.append(one_center)
                            surface_retractions.append(one_surface_retraction)
                            center_retractions.append(one_center_retraction)
                            support_retractions.append(one_support_retraction)
                            exact_witness_rows.append(exact_witness)
                            predicted_witness_rows.append(predicted_witness)
                            body_names.append(one_names)
                        exact_gaps = np.asarray(exact_gaps, dtype=np.float64)
                        predicted_gaps = np.asarray(predicted_gaps, dtype=np.float64)
                        recomputed_exact[compact] = np.min(exact_gaps, axis=0)
                        recomputed_predicted[compact] = np.min(predicted_gaps, axis=0)
                        flat_worst = int(np.argmin(exact_gaps))
                        substep_index, ellipsoid_row = np.unravel_index(
                            flat_worst, exact_gaps.shape
                        )
                        record_arrays["exact_static_minimum_margin_m"][compact] = float(
                            np.min(exact_gaps)
                        )
                        record_arrays["predicted_static_minimum_margin_m"][compact] = float(
                            np.min(predicted_gaps)
                        )
                        record_arrays["static_margin_overestimate_m"][compact] = float(
                            np.min(predicted_gaps) - np.min(exact_gaps)
                        )
                        record_arrays["worst_surface_position_error_m"][compact] = (
                            surface_errors[substep_index][ellipsoid_row]
                        )
                        record_arrays["worst_center_position_error_m"][compact] = (
                            center_errors[substep_index][ellipsoid_row]
                        )
                        record_arrays["worst_surface_clearance_overestimate_m"][compact] = (
                            surface_retractions[substep_index][ellipsoid_row]
                        )
                        record_arrays["worst_center_clearance_overestimate_m"][compact] = (
                            center_retractions[substep_index][ellipsoid_row]
                        )
                        record_arrays["worst_support_clearance_overestimate_m"][compact] = (
                            support_retractions[substep_index][ellipsoid_row]
                        )
                        record_arrays["worst_substep_index"][compact] = substep_index
                        record_arrays["worst_ellipsoid_row"][compact] = ellipsoid_row
                        record_arrays["worst_obstacle_index"][compact] = int(
                            exact_witness_rows[substep_index][ellipsoid_row]
                        )
                        record_arrays[
                            "worst_predicted_witness_matches_exact"
                        ][compact] = int(
                            predicted_witness_rows[substep_index][ellipsoid_row]
                            == exact_witness_rows[substep_index][ellipsoid_row]
                        )
                        record_arrays["worst_link_body_name"][compact] = (
                            body_names[substep_index][ellipsoid_row]
                        )
                    _restore_env(env, snapshot)
                if step < maximum_step:
                    env.step(_canonical_action(archived_actions[step], step).tolist())
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()

    finite_required = [
        key for key, value in record_arrays.items()
        if np.asarray(value).dtype.kind == "f"
    ]
    _require(
        all(np.all(np.isfinite(record_arrays[key])) for key in finite_required)
        and np.array_equal(
            recomputed_predicted, stored_predicted_margin[test_rows]
        )
        and np.array_equal(
            recomputed_exact, stored_exact_static_margin[test_rows]
        ),
        "surface-error FK geometry did not reproduce",
    )
    false_rows = np.flatnonzero(false_safe_mask)
    control_rows = np.flatnonzero(correct_unsafe_mask)
    matched_rows = matched_control_indexes(
        false_safe_indexes=false_rows, control_indexes=control_rows,
        state_index=np.asarray(arrays["state_index"], dtype=np.int64),
        exact_margin_m=np.min(exact_dynamic_margin, axis=1),
    )
    false_compact = np.asarray(
        [compact_by_row[int(row)] for row in false_rows], dtype=np.int64
    )
    matched_compact = np.asarray(
        [compact_by_row[int(row)] for row in matched_rows], dtype=np.int64
    )
    correct_unsafe_compact = np.flatnonzero(record_arrays["class_code"] == 1)
    correct_safe_compact = np.flatnonzero(record_arrays["class_code"] == 0)
    unsafe_compact = np.flatnonzero(np.isin(record_arrays["class_code"], [1, 2]))

    def group_summary(indexes: Any) -> dict[str, Any]:
        return {
            "surface_position_error_m": vector_summary(
                record_arrays["worst_surface_position_error_m"][indexes]
            ),
            "center_position_error_m": vector_summary(
                record_arrays["worst_center_position_error_m"][indexes]
            ),
            "surface_clearance_overestimate_m": vector_summary(
                record_arrays["worst_surface_clearance_overestimate_m"][indexes]
            ),
            "center_clearance_overestimate_m": vector_summary(
                record_arrays["worst_center_clearance_overestimate_m"][indexes]
            ),
            "support_clearance_overestimate_m": vector_summary(
                record_arrays["worst_support_clearance_overestimate_m"][indexes]
            ),
            "static_margin_overestimate_m": vector_summary(
                record_arrays["static_margin_overestimate_m"][indexes]
            ),
            "predicted_witness_match_fraction": float(np.mean(
                record_arrays["worst_predicted_witness_matches_exact"][indexes]
            )),
        }

    false_surface = record_arrays["worst_surface_position_error_m"][false_compact]
    matched_surface = record_arrays[
        "worst_surface_position_error_m"
    ][matched_compact]
    correlation = spearman_correlation(
        record_arrays["worst_surface_clearance_overestimate_m"][unsafe_compact],
        record_arrays["static_margin_overestimate_m"][unsafe_compact],
    )
    false_link_counts = {
        str(name): int(np.count_nonzero(
            record_arrays["worst_link_body_name"][false_compact] == name
        ))
        for name in sorted(set(
            record_arrays["worst_link_body_name"][false_compact].tolist()
        ))
    }
    metrics = {
        "test_random_action_count": int(count),
        "false_safe_action_count": int(len(false_compact)),
        "correct_unsafe_action_count": int(len(correct_unsafe_compact)),
        "correct_safe_action_count": int(len(correct_safe_compact)),
        "false_unsafe_action_count": int(np.count_nonzero(
            record_arrays["class_code"] == 3
        )),
        "recomputed_geometry_exact": True,
        "false_safe": group_summary(false_compact),
        "matched_correct_unsafe": group_summary(matched_compact),
        "all_correct_unsafe": group_summary(correct_unsafe_compact),
        "accepted_correct_safe": group_summary(correct_safe_compact),
        "false_safe_surface_error_median_excess_m": float(
            np.median(false_surface) - np.median(matched_surface)
        ),
        "paired_surface_error_win_fraction": float(np.mean(
            false_surface > matched_surface
        )),
        "matched_control_unique_action_count": int(len(set(matched_rows.tolist()))),
        "unsafe_surface_retraction_margin_overestimate_spearman": float(correlation),
        "false_safe_worst_link_counts": false_link_counts,
        "false_safe_worst_substep": vector_summary(
            record_arrays["worst_substep_index"][false_compact].astype(np.float64)
        ),
        "source_factorized_metrics": {
            "joint_trajectory_RMSE_rad": factorized_result["metrics"][
                "factorized_joint"
            ]["test_random_joint_trajectory_RMSE_rad"],
            "link_center_RMSE_m": factorized_result["metrics"][
                "factorized_geometry"
            ]["test_random_link_center_RMSE_m"],
            "joint_sensitivity_cosine": factorized_result["metrics"][
                "joint_sensitivity"
            ]["mean_cosine"],
            "margin_sensitivity_cosine": factorized_result["metrics"][
                "factorized_margin_sensitivity"
            ]["mean_cosine"],
        },
    }
    decision = surface_loss_decision(metrics, config)
    record_arrays["matched_control_row_index_for_false_safe"] = matched_rows
    record_arrays["false_safe_row_index"] = false_rows
    return record_arrays, {"metrics": metrics, "decision": decision}


def main() -> int:
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = resolved_paths(args)
    config = load_surface_config(paths["config"])
    records, analysis = compute_surface_error_records(paths, config)
    _save_npz_atomic(args.records.resolve(), records)
    records_receipt = {
        "path": str(args.records.resolve()),
        "file_sha256": _file_sha256(args.records.resolve()),
        "row_index_sha256": _hash_array(records["row_index"]),
        "surface_error_sha256": _hash_array(
            records["worst_surface_position_error_m"]
        ),
    }
    result = {
        "schema_version": SURFACE_RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "records": records_receipt, "metrics": analysis["metrics"],
        "decision": analysis["decision"],
        "forbidden_action_receipt": {
            key: False for key in config["forbidden_actions"]
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "decision": result["decision"], "metrics": result["metrics"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
