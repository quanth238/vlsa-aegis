#!/usr/bin/env python3
"""Explain the three unsupported states of the validated one-sided model."""

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
    _arm_targets, load_weights, payload_sha256, predict,
)
from main.multilink_ellipsoid.factorized_unsupported_state_audit import (
    RESULT_SCHEMA, interpret_state, load_unsupported_state_config,
    standardized_state_support, unsupported_state_indexes,
)
from scripts.collect_distal_complete_osc_margin_moka10 import (
    _GEOMETRY_PLACEHOLDER_CASE_ID, _geometry_placeholder_row,
)
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action, _restore_env, _snapshot_env,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    add_common_arguments, load_inputs, resolved_paths,
)
from scripts.evaluate_distal_native_geom_inventory_moka10 import _read_manifest
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def add_audit_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--experimental-predictions", type=Path, required=True)
    parser.add_argument("--one-sided-result", type=Path, required=True)
    parser.add_argument("--one-sided-validation", type=Path, required=True)


def audit_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = resolved_paths(args)
    output.update({
        "audit_config": args.audit_config.resolve(),
        "experimental_model": args.experimental_model.resolve(),
        "experimental_predictions": args.experimental_predictions.resolve(),
        "one_sided_result": args.one_sided_result.resolve(),
        "one_sided_validation": args.one_sided_validation.resolve(),
    })
    return output


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


def _validate_experimental_sources(
    paths: Mapping[str, Path], config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = config["immutable_source"]
    for source_key, path_key in (
        ("one_sided_config_file_sha256", "config"),
        ("one_sided_model_file_sha256", "experimental_model"),
        ("one_sided_predictions_file_sha256", "experimental_predictions"),
        ("one_sided_result_file_sha256", "one_sided_result"),
        ("one_sided_validation_file_sha256", "one_sided_validation"),
        ("baseline_model_file_sha256", "baseline_model"),
        ("baseline_predictions_file_sha256", "baseline_predictions"),
        ("trajectory_array_file_sha256", "array"),
        ("trajectory_metadata_file_sha256", "metadata"),
    ):
        _require(_file_sha256(paths[path_key]) == source[source_key],
                 "unsupported-state immutable source differs")
    result = _load(paths["one_sided_result"])
    validation = _load(paths["one_sided_validation"])
    _require(
        result.get("result_payload_sha256")
        == source["one_sided_result_payload_sha256"]
        and result.get("decision", {}).get("one_sided_geometry_GO") is False
        and result.get("metrics", {}).get("test_safety", {}).get(
            "false_safe_action_count"
        ) == 0
        and result.get("metrics", {}).get("test_safety", {}).get(
            "state_safe_support_count"
        ) == 12
        and validation.get("validation_payload_sha256")
        == source["one_sided_validation_payload_sha256"]
        and validation.get("valid") is True,
        "unsupported-state validated one-sided source differs",
    )
    return result, validation


def _state_trajectory_metrics(
    *, exact_q: Any, baseline_q: Any, experimental_q: Any,
    exact_margin: Any, state_index: Any, selected: Any,
) -> dict[int, dict[str, Any]]:
    import numpy as np

    output = {}
    states = np.asarray(state_index, dtype=np.int64)
    mask = np.asarray(selected, dtype=bool)
    for state in sorted(set(states[mask].tolist())):
        rows = mask & (states == state)
        exact_safe = np.all(np.asarray(exact_margin)[rows] >= 0.0, axis=1)
        base_error = np.asarray(baseline_q)[rows] - np.asarray(exact_q)[rows]
        new_error = np.asarray(experimental_q)[rows] - np.asarray(exact_q)[rows]
        baseline_safe_rmse = (
            None if not np.any(exact_safe) else
            float(np.sqrt(np.mean(base_error[exact_safe] ** 2)))
        )
        one_sided_safe_rmse = (
            None if not np.any(exact_safe) else
            float(np.sqrt(np.mean(new_error[exact_safe] ** 2)))
        )
        output[int(state)] = {
            "exact_safe_candidate_count": int(np.count_nonzero(exact_safe)),
            "baseline_all_joint_RMSE_rad": float(np.sqrt(np.mean(base_error ** 2))),
            "one_sided_all_joint_RMSE_rad": float(np.sqrt(np.mean(new_error ** 2))),
            "baseline_terminal_joint_RMSE_rad": float(np.sqrt(np.mean(
                base_error[:, -1] ** 2
            ))),
            "one_sided_terminal_joint_RMSE_rad": float(np.sqrt(np.mean(
                new_error[:, -1] ** 2
            ))),
            "baseline_exact_safe_joint_RMSE_rad": baseline_safe_rmse,
            "one_sided_exact_safe_joint_RMSE_rad": one_sided_safe_rmse,
        }
    return output


def analyze_records(
    *, records: Mapping[str, Any], arrays: Mapping[str, Any],
    baseline_q: Any, experimental_q: Any, complete_dataset: Mapping[str, Any],
    trajectory_archive: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    split = np.asarray(arrays["split"], dtype=object)
    source_code = np.asarray(arrays["source_code"], dtype=np.int8)
    state_ids = np.asarray(arrays["state_index"], dtype=np.int64)
    test_random = (split == "test") & (source_code == 2)
    exact_margin = np.asarray(arrays["minimum_margin_m"], dtype=np.float64)
    stored_experimental = np.asarray(
        records["stored_experimental_minimum_margin_m"], dtype=np.float64
    )
    unsupported = np.asarray(records["unsupported_state_index"], dtype=np.int64)
    trajectory = _state_trajectory_metrics(
        exact_q=arrays["joint_position_rad"], baseline_q=baseline_q,
        experimental_q=experimental_q, exact_margin=exact_margin,
        state_index=state_ids, selected=test_random,
    )
    supported_test = [
        state for state in sorted(trajectory)
        if state not in set(unsupported.tolist())
    ]
    p95_error = float(np.quantile([
        trajectory[state]["one_sided_all_joint_RMSE_rad"]
        for state in supported_test
    ], 0.95))
    state_records = sorted(
        complete_dataset["state_records"], key=lambda item: int(item["state_index"])
    )
    metadata_by_state = {int(item["state_index"]): item for item in state_records}
    unique_state_indexes = []
    unique_features = []
    unique_splits = []
    unique_cases = []
    exact_safe_support = []
    state_input = np.asarray(trajectory_archive["state_input_vector"], dtype=np.float64)
    for state in sorted(set(state_ids.tolist())):
        rows = np.flatnonzero(state_ids == state)
        random_rows = rows[source_code[rows] == 2]
        unique_state_indexes.append(int(state))
        unique_features.append(state_input[rows[0]])
        unique_splits.append(str(split[rows[0]]))
        unique_cases.append(str(metadata_by_state[state]["case_id"]))
        exact_safe_support.append(bool(np.any(np.all(
            exact_margin[random_rows] >= 0.0, axis=1
        ))))
    support = standardized_state_support(
        state_features=np.asarray(unique_features),
        state_indexes=unique_state_indexes, splits=unique_splits,
        case_ids=unique_cases, exact_safe_support=exact_safe_support,
        query_indexes=unsupported.tolist(),
        maximum_abs_z=float(config["measurements"]["maximum_single_feature_abs_z"]),
    )
    record_rows = np.asarray(records["row_index"], dtype=np.int64)
    compact_by_row = {int(row): index for index, row in enumerate(record_rows)}
    baseline_trace = np.asarray(records["baseline_static_clearance_m"])
    experimental_trace = np.asarray(records["experimental_static_clearance_m"])
    member_minimum = np.asarray(records["member_minimum_margin_m"])
    output_states = []
    for state in unsupported:
        rows = np.flatnonzero(test_random & (state_ids == state))
        compact = np.asarray([compact_by_row[int(row)] for row in rows], dtype=np.int64)
        exact_safe = np.all(exact_margin[rows] >= 0.0, axis=1)
        predicted_worst = np.min(stored_experimental[compact], axis=1)
        safe_local = np.flatnonzero(exact_safe)
        if len(safe_local):
            best_local = int(safe_local[np.argmax(predicted_worst[exact_safe])])
            representative_basis = "exact_safe_with_closest_predicted_margin_to_zero"
            closest_predicted_margin = float(predicted_worst[best_local])
        else:
            exact_worst = np.min(exact_margin[rows], axis=1)
            best_local = int(np.argmax(exact_worst))
            representative_basis = "least_unsafe_exact_candidate"
            closest_predicted_margin = None
        best_row = int(rows[best_local])
        best_compact = int(compact[best_local])
        experimental_active = np.unravel_index(
            int(np.argmin(experimental_trace[best_compact])), (51, 7)
        )
        baseline_active = np.unravel_index(
            int(np.argmin(baseline_trace[best_compact])), (51, 7)
        )
        exact_active = np.unravel_index(
            int(np.argmin(arrays["ellipsoid_clearance_m"][best_row])), (51, 7)
        )
        member_counts = []
        member_best = []
        for member in range(member_minimum.shape[1]):
            member_score = np.min(member_minimum[compact, member], axis=1)
            member_counts.append(int(np.count_nonzero(
                exact_safe & (member_score >= 0.0)
            )))
            member_best.append(
                None if not np.any(exact_safe) else
                float(np.max(member_score[exact_safe]))
            )
        state_support = support["queries"][str(int(state))]
        large = bool(
            trajectory[int(state)]["one_sided_all_joint_RMSE_rad"]
            > p95_error + 1e-15
        )
        interpretation = interpret_state(
            closest_predicted_margin_m=closest_predicted_margin,
            support_pass=bool(state_support["support_pass"]),
            member_accepted_counts=member_counts,
            candidate_exact_safe_count=int(np.count_nonzero(exact_safe)),
            large_trajectory_error=large, config=config,
        )
        metadata = metadata_by_state[int(state)]
        baseline_row_margin = np.min(baseline_trace[best_compact], axis=0)
        experimental_row_margin = np.min(experimental_trace[best_compact], axis=0)
        output_states.append({
            "state_index": int(state), "case_id": str(metadata["case_id"]),
            "state_step": int(metadata["state_step"]),
            "exact_safe_candidate_count": int(np.count_nonzero(exact_safe)),
            "representative_candidate_index": int(
                np.asarray(arrays["candidate_index"])[best_row]
            ),
            "representative_candidate_basis": representative_basis,
            "closest_exact_safe_one_sided_predicted_worst_margin_m": (
                closest_predicted_margin
            ),
            "representative_candidate_one_sided_predicted_worst_margin_m": float(
                predicted_worst[best_local]
            ),
            "representative_candidate_exact_worst_margin_m": float(np.min(
                exact_margin[best_row]
            )),
            "baseline_predicted_worst_margin_m": float(np.min(baseline_row_margin)),
            "baseline_accepts_best_candidate": bool(np.all(baseline_row_margin >= 0.0)),
            "one_sided_predicted_margin_by_row_m": experimental_row_margin.tolist(),
            "baseline_predicted_margin_by_row_m": baseline_row_margin.tolist(),
            "one_sided_active": {
                "substep": int(experimental_active[0]), "row": int(experimental_active[1])
            },
            "baseline_active": {
                "substep": int(baseline_active[0]), "row": int(baseline_active[1])
            },
            "exact_active": {
                "substep": int(exact_active[0]), "row": int(exact_active[1])
            },
            "member_accepted_exact_safe_candidate_counts": member_counts,
            "member_best_exact_safe_worst_margin_m": member_best,
            "state_support": state_support,
            "trajectory_error": {
                **trajectory[int(state)],
                "supported_test_state_p95_one_sided_joint_RMSE_rad": p95_error,
                "large_trajectory_error": large,
            },
            "exact_safe_action_inside_registered_region": bool(np.any(exact_safe)),
            "interpretation": interpretation,
        })
    primary_counts = {}
    for item in output_states:
        key = item["interpretation"]["primary_explanation"]
        primary_counts[key] = primary_counts.get(key, 0) + 1
    return {
        "unsupported_state_count": int(len(output_states)),
        "supported_test_state_count": int(len(supported_test)),
        "supported_test_state_p95_one_sided_joint_RMSE_rad": p95_error,
        "training_state_support": support,
        "states": output_states, "primary_explanation_counts": primary_counts,
        "verdict_unchanged": "strict_NO_GO",
        "correction_fit_authorized": False,
    }


def collect_records(
    paths: Mapping[str, Path], audit_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config, minimum_union_support_gap_witnesses,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import load_shadow_config

    _validate_experimental_sources(paths, audit_config)
    (
        _, factorized_config, complete_dataset, complete_collection,
        metadata, _, _, _, _, arrays, _, baseline_q,
    ) = load_inputs(paths)
    torch.set_num_threads(8)
    experimental_models, experimental_state = load_weights(paths["experimental_model"])
    experimental_q = predict(experimental_models, experimental_state, arrays)
    member_q = _member_predictions(experimental_models, experimental_state, arrays)
    stored_experimental_archive = np.load(
        paths["experimental_predictions"], allow_pickle=False
    )
    _require(np.array_equal(
        experimental_q,
        stored_experimental_archive["experimental_joint_position_rad"],
    ), "unsupported-state one-sided joint prediction differs")
    baseline_archive = np.load(paths["baseline_predictions"], allow_pickle=False)
    baseline_margin = baseline_archive["factorized_minimum_margin_m"].astype(np.float64)
    experimental_margin = stored_experimental_archive[
        "experimental_minimum_margin_m"
    ].astype(np.float64)
    split = np.asarray(arrays["split"], dtype=object)
    source_code = np.asarray(arrays["source_code"], dtype=np.int8)
    test_random = (split == "test") & (source_code == 2)
    unsupported = unsupported_state_indexes(
        exact_margin=arrays["minimum_margin_m"],
        predicted_margin=experimental_margin,
        state_index=arrays["state_index"], selected=test_random,
        expected_count=int(audit_config["population"]["expected_unsupported_state_count"]),
    )
    selected_rows = np.flatnonzero(
        test_random & np.isin(np.asarray(arrays["state_index"]), unsupported)
    )
    count = len(selected_rows)
    _require(count == len(unsupported) * 64,
             "unsupported-state selected population differs")
    compact_by_row = {int(row): index for index, row in enumerate(selected_rows)}
    records = {
        "row_index": selected_rows.astype(np.int64),
        "unsupported_state_index": np.asarray(unsupported, dtype=np.int64),
        "baseline_static_clearance_m": np.full((count, 51, 7), np.nan),
        "experimental_static_clearance_m": np.full((count, 51, 7), np.nan),
        "member_minimum_margin_m": np.full((count, 5, 7), np.nan),
        "stored_experimental_minimum_margin_m": experimental_margin[selected_rows],
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
                 "unsupported-state geometry source differs")
    manifest_rows = []
    for key in ("selected", "same_task", "targeted"):
        manifest_rows.extend(_read_manifest(paths[key], _file_sha256(paths[key])))
    row_by_case = {str(item["case_id"]): item for item in manifest_rows}
    placeholder_row = _geometry_placeholder_row(row_by_case)
    placeholder_path = paths["archived"] / placeholder_row["archived_relative_path"]
    placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == placeholder_row["archived_file_sha256"]
        and placeholder.get("result_payload_sha256")
        == placeholder_row["archived_payload_sha256"]
        and placeholder.get("case_id") == _GEOMETRY_PLACEHOLDER_CASE_ID,
        "unsupported-state geometry placeholder differs",
    )
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    state_metadata = {
        int(item["state_index"]): item for item in complete_dataset["state_records"]
    }
    unsupported_cases = sorted(set(
        state_metadata[state]["case_id"] for state in unsupported
    ))
    for case_id in unsupported_cases:
        selected_row = row_by_case[case_id]
        archived_path = paths["archived"] / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        case = population[case_id]
        validate_case_row(case, paths["repo"])
        env = probe_env = None
        try:
            env, probe_env, _, _, setup = _build_pair(runtime, case)
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                archived=placeholder, env=env, obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            case_states = sorted(
                (state_metadata[state] for state in unsupported
                 if state_metadata[state]["case_id"] == case_id),
                key=lambda item: int(item["state_step"]),
            )
            by_step = {int(item["state_step"]): item for item in case_states}
            maximum_step = max(by_step)
            position_indexes = np.asarray(
                env.robots[0]._ref_joint_pos_indexes, dtype=np.int64
            )
            for step in range(maximum_step + 1):
                if step in by_step:
                    state = by_step[step]
                    state_id = int(state["state_index"])
                    _require(
                        hashlib.sha256(_dynamic_state_vector(env).tobytes()).hexdigest()
                        == state["dynamic_state_sha256"],
                        "unsupported-state identity differs",
                    )
                    snapshot = _snapshot_env(env)
                    obstacles = probe._obstacles(env)

                    def gap(joint: Any) -> Any:
                        env.sim.data.qpos[position_indexes] = joint
                        env.sim.forward()
                        value, _ = minimum_union_support_gap_witnesses(
                            probe._ellipsoids(env)[:7], obstacles
                        )
                        return value

                    rows = np.flatnonzero(
                        test_random
                        & (np.asarray(arrays["state_index"], dtype=np.int64) == state_id)
                    )
                    for row in rows:
                        compact = compact_by_row[int(row)]
                        for substep in range(51):
                            records["baseline_static_clearance_m"][compact, substep] = gap(
                                baseline_q[row, substep]
                            )
                            records["experimental_static_clearance_m"][compact, substep] = gap(
                                experimental_q[row, substep]
                            )
                        for member in range(5):
                            trace = [
                                gap(member_q[member, row, substep])
                                for substep in range(51)
                            ]
                            records["member_minimum_margin_m"][compact, member] = np.min(
                                np.asarray(trace), axis=0
                            )
                    _restore_env(env, snapshot)
                if step < maximum_step:
                    env.step(_canonical_action(archived["actions"][step], step).tolist())
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    _require(
        all(np.all(np.isfinite(records[key])) for key in (
            "baseline_static_clearance_m", "experimental_static_clearance_m",
            "member_minimum_margin_m",
        )), "unsupported-state geometry records incomplete",
    )
    recomputed_baseline = np.min(records["baseline_static_clearance_m"], axis=1)
    recomputed_experimental = np.min(
        records["experimental_static_clearance_m"], axis=1
    )
    _require(
        np.array_equal(recomputed_baseline, baseline_margin[selected_rows])
        and np.array_equal(
            recomputed_experimental, experimental_margin[selected_rows]
        ), "unsupported-state stored geometry did not reproduce",
    )
    trajectory_archive = np.load(paths["array"], allow_pickle=False)
    analysis = analyze_records(
        records=records, arrays=arrays, baseline_q=baseline_q,
        experimental_q=experimental_q, complete_dataset=complete_dataset,
        trajectory_archive=trajectory_archive, config=audit_config,
    )
    analysis["source_reproduction"] = {
        "independent_one_sided_validation_already_passed": True,
        "experimental_joint_prediction_exact": True,
        "baseline_static_margin_exact": True,
        "experimental_static_margin_exact": True,
    }
    return records, analysis


def main() -> int:
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_audit_arguments(parser)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = audit_paths(args)
    config = load_unsupported_state_config(paths["audit_config"])
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
        },
        "audit": analysis,
        "decision": {
            "verdict": "strict_NO_GO_unchanged",
            "correction_fit_authorized": False,
            "QP_or_closed_loop_authorized": False,
        },
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
        "states": result["audit"]["states"],
        "primary_explanation_counts": result["audit"]["primary_explanation_counts"],
        "decision": result["decision"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
