#!/usr/bin/env python3
"""Compare direct label-fitted and finite-difference affine OSC constraints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_coefficient_model import (
    AFFINE_COEFFICIENT_DATASET_SCHEMA,
    AFFINE_COEFFICIENT_DATASET_RESULT_SCHEMA,
)
from main.multilink_ellipsoid.affine_oracle_comparison import (
    RESULT_SCHEMA, affine_values, fit_direct_halfspaces,
    fit_finite_difference_lower_bounds, load_config, summarize_predictions,
)
from main.multilink_ellipsoid.two_step_margin import (
    feature_context, feature_vectors, load_selected_manifest,
)
from scripts.collect_distal_affine_coefficient_moka10 import _chunk_values
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action, _restore_env, _snapshot_env,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _state_seed(global_seed: int, case_id: str, state_step: int) -> int:
    payload = "%d::%s::%d" % (global_seed, case_id, state_step)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _method_aggregate(
    state_results: list[dict[str, Any]], method: str,
    decision_gate: Mapping[str, Any],
) -> dict[str, Any]:
    records = [
        {
            "true_safe": candidate["true_safe"],
            "predicted_safe": candidate["predictions"][method]["all_rows_safe"],
        }
        for state in state_results for candidate in state["fresh_actions"]
    ]
    summary = summarize_predictions(records)
    supported = [
        state for state in state_results
        if state["true_safe_fresh_action_count"] > 0
    ]
    accepted = [
        state for state in supported
        if state["method_summaries"][method]["accepted_true_safe_action_count"] > 0
    ]
    coverage = len(accepted) / len(supported) if supported else None
    recall = summary["safe_action_recall"]
    passed = bool(
        summary["false_safe_action_count"]
        == int(decision_gate["false_safe_action_count"])
        and recall is not None
        and recall >= float(decision_gate["minimum_aggregate_safe_action_recall"])
        and coverage is not None
        and coverage >= float(decision_gate["minimum_safe_support_state_coverage"])
    )
    summary.update({
        "fresh_safe_support_state_count": len(supported),
        "accepted_safe_support_state_count": len(accepted),
        "safe_support_state_coverage": coverage,
        "method_gate_pass": passed,
    })
    return summary


def main() -> int:
    import numpy as np
    from main.evaluate_safelibero_aegis import _runtime_imports, read_jsonl
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {
        "repo": args.repo_root.resolve(),
        "population": args.population_manifest.resolve(),
        "selected": args.selected_manifest.resolve(),
        "archived": args.archived_root.resolve(),
        "geometry": args.geometry_config.resolve(),
        "exact_box": args.exact_box_config.resolve(),
        "config": args.config.resolve(),
        "dataset": args.dataset.resolve(),
        "dataset_result": args.dataset_result.resolve(),
        "dataset_validation": args.dataset_validation.resolve(),
        "output": args.output.resolve(),
    }
    config = load_config(paths["config"])
    source_settings = config["immutable_source"]
    for path, expected, label in (
        (paths["population"], config["source_population_manifest_sha256"], "population"),
        (paths["selected"], config["selected_manifest_sha256"], "selected"),
        (paths["geometry"], config["geometry_config_file_sha256"], "geometry"),
        (paths["exact_box"], config["exact_box_config_file_sha256"], "exact-box"),
        (paths["dataset"], source_settings["dataset_file_sha256"], "dataset"),
        (paths["dataset_result"], source_settings["dataset_result_file_sha256"], "dataset-result"),
        (paths["dataset_validation"], source_settings["dataset_validation_file_sha256"], "dataset-validation"),
    ):
        _require(_file_sha256(path) == expected, "oracle comparison %s differs" % label)
    dataset = _load(paths["dataset"])
    dataset_result = _load(paths["dataset_result"])
    dataset_validation = _load(paths["dataset_validation"])
    _require(
        dataset.get("schema_version") == AFFINE_COEFFICIENT_DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source_settings["dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256")
        and dataset.get("source_commit") == source_settings["dataset_source_commit"]
        and dataset_result.get("schema_version")
        == AFFINE_COEFFICIENT_DATASET_RESULT_SCHEMA
        and dataset_result.get("result_payload_sha256")
        == source_settings["dataset_result_payload_sha256"]
        == _hash_without(dataset_result, "result_payload_sha256")
        and dataset_validation.get("status") == "valid",
        "oracle comparison immutable dataset identity differs",
    )
    states = dataset["state_records"]
    _require(
        len(states) == int(source_settings["expected_state_count"])
        and [item["state_index"] for item in states] == list(range(len(states)))
        and all(
            len(item["candidate_first_xyz"])
            == int(source_settings["expected_fit_actions_per_state"])
            for item in states
        ),
        "oracle comparison immutable state records differ",
    )
    selected = load_selected_manifest(paths["selected"], config)
    selected_by_case = {item["case_id"]: item for item in selected}
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    primary = selected_by_case["vlsa-t1-goal-ii-t0-e05"]
    placeholder_path = paths["archived"] / primary["archived_relative_path"]
    placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == primary["archived_file_sha256"]
        and placeholder.get("result_payload_sha256")
        == primary["archived_payload_sha256"],
        "oracle comparison geometry placeholder differs",
    )
    runtime = _runtime_imports(include_aegis=False)
    state_results: list[dict[str, Any]] = []
    finite_epsilon = float(config["finite_difference"]["central_epsilon_action"])
    maximum_motion = float(
        config["exact_verification"][
            "maximum_per_step_obstacle_l1_displacement_m"
        ]
    )
    count_per_state = int(config["fresh_actions"]["count_per_state"])

    for case_id in sorted(selected_by_case):
        row = selected_by_case[case_id]
        archived_path = paths["archived"] / row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == row["archived_file_sha256"]
            and archived.get("result_payload_sha256") == row["archived_payload_sha256"],
            "oracle comparison archived case differs: %s" % case_id,
        )
        case_states = sorted(
            (item for item in states if item["case_id"] == case_id),
            key=lambda item: int(item["state_step"]),
        )
        _require(len(case_states) == 5, "oracle comparison per-case state count differs")
        env = probe_env = None
        try:
            env, probe_env, _, _, setup = _build_pair(runtime, population[case_id])
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                archived=placeholder, env=env, obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            needed = {int(item["state_step"]) for item in case_states}
            snapshots = {}
            actions = archived["actions"]
            for step in range(max(needed) + 1):
                if step in needed:
                    snapshots[step] = _snapshot_env(env)
                if step < max(needed):
                    env.step(_canonical_action(actions[step], step).tolist())
            for state in case_states:
                step = int(state["state_step"])
                _restore_env(env, snapshots[step])
                first = _canonical_action(actions[step], step)
                second = _canonical_action(actions[step + 1], step + 1)
                _require(
                    float(np.max(np.abs(
                        first - np.asarray(state["nominal_first_action"], dtype=np.float64)
                    ))) <= 1.0e-12
                    and float(np.max(np.abs(
                        second - np.asarray(state["nominal_second_action"], dtype=np.float64)
                    ))) <= 1.0e-12,
                    "oracle comparison immutable action differs",
                )
                context = feature_context(env, probe)
                _, live_features = feature_vectors(
                    context, first[:3], first[:3], second[:3]
                )
                feature_error = float(np.max(np.abs(
                    live_features
                    - np.asarray(state["pair_state_feature_vectors"], dtype=np.float64)
                )))
                clearance_error = float(np.max(np.abs(
                    np.asarray(context["current_clearance_m"], dtype=np.float64)
                    - np.asarray(state["current_clearance_m"], dtype=np.float64)
                )))
                _require(
                    feature_error <= 1.0e-8 and clearance_error <= 1.0e-8,
                    "oracle comparison live state receipt differs",
                )
                fit_xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
                fit_margins = np.asarray(
                    state["candidate_minimum_distal_margin_m"], dtype=np.float64
                )
                lower = np.asarray(state["action_lower"], dtype=np.float64)
                upper = np.asarray(state["action_upper"], dtype=np.float64)
                direct = fit_direct_halfspaces(
                    fit_xyz, fit_margins, first[:3], lower, upper,
                    config["direct_halfspace"],
                )
                plus_margins = []
                minus_margins = []
                finite_probe_records = []
                for dimension in range(3):
                    _require(
                        first[dimension] - finite_epsilon >= -1.0
                        and first[dimension] + finite_epsilon <= 1.0,
                        "oracle comparison central finite difference exceeds action bounds",
                    )
                    pair = []
                    for sign in (1.0, -1.0):
                        action = first.copy()
                        action[dimension] += sign * finite_epsilon
                        summary, ee = _chunk_values(
                            probe.rollout_chunk(env, [action, second])
                        )
                        pair.append(summary["minimum_substep_clearance_m"])
                        finite_probe_records.append({
                            "dimension": dimension, "sign": int(sign),
                            "candidate_xyz": action[:3].tolist(),
                            "minimum_distal_margin_m": summary[
                                "minimum_substep_clearance_m"
                            ],
                            "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
                            "minimum_released_AEGIS_EE_margin_m": float(ee),
                        })
                    plus_margins.append(pair[0])
                    minus_margins.append(pair[1])
                finite = fit_finite_difference_lower_bounds(
                    state["nominal_minimum_distal_margin_m"], plus_margins,
                    minus_margins, finite_epsilon, fit_xyz, fit_margins,
                    first[:3], float(config["finite_difference"]["one_sided_padding_m"]),
                )
                seed = _state_seed(
                    int(config["fresh_actions"]["seed"]), case_id, step
                )
                rng = np.random.RandomState(seed)
                fresh_xyz = rng.uniform(lower, upper, size=(count_per_state, 3))
                fresh_records = []
                direct_records = []
                finite_records = []
                direct_row_false_safe = np.zeros(7, dtype=np.int64)
                finite_row_false_safe = np.zeros(7, dtype=np.int64)
                for fresh_index, xyz in enumerate(fresh_xyz):
                    action = first.copy()
                    action[:3] = xyz
                    summary, ee = _chunk_values(
                        probe.rollout_chunk(env, [action, second])
                    )
                    margins = np.asarray(
                        summary["minimum_substep_clearance_m"], dtype=np.float64
                    )
                    proxy_safe = bool(np.all(margins >= 0.0))
                    raw_safe = bool(
                        summary["D_sim_raw_safe"]
                        and summary["maximum_within_step_obstacle_l1_displacement_m"]
                        <= maximum_motion
                    )
                    true_safe = bool(proxy_safe and raw_safe)
                    direct_values = affine_values(
                        direct["intercept_at_nominal"],
                        direct["gradients_per_action"], xyz, first[:3],
                    )
                    finite_values = affine_values(
                        finite["intercept_at_nominal_m"],
                        finite["gradients_m_per_action"], xyz, first[:3],
                    )
                    direct_row_false_safe += np.logical_and(
                        direct_values >= 0.0, margins < 0.0
                    ).astype(np.int64)
                    finite_row_false_safe += np.logical_and(
                        finite_values >= 0.0, margins < 0.0
                    ).astype(np.int64)
                    direct_safe = bool(np.all(direct_values >= 0.0))
                    finite_safe = bool(np.all(finite_values >= 0.0))
                    direct_records.append({
                        "true_safe": true_safe, "predicted_safe": direct_safe
                    })
                    finite_records.append({
                        "true_safe": true_safe, "predicted_safe": finite_safe
                    })
                    fresh_records.append({
                        "fresh_index": fresh_index, "candidate_xyz": xyz.tolist(),
                        "minimum_distal_margin_m": margins.tolist(),
                        "D_opt_seven_distal_safe": proxy_safe,
                        "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
                        "raw_protected_contact_count": int(
                            summary["raw_protected_contact_count"]
                        ),
                        "maximum_within_step_obstacle_l1_displacement_m": float(
                            summary["maximum_within_step_obstacle_l1_displacement_m"]
                        ),
                        "minimum_released_AEGIS_EE_margin_m": float(ee),
                        "true_safe": true_safe,
                        "predictions": {
                            "direct_halfspace": {
                                "row_values": direct_values.tolist(),
                                "all_rows_safe": direct_safe,
                            },
                            "finite_difference": {
                                "row_values_m": finite_values.tolist(),
                                "all_rows_safe": finite_safe,
                            },
                        },
                    })
                direct_summary = summarize_predictions(direct_records)
                finite_summary = summarize_predictions(finite_records)
                direct_summary["per_row_false_safe_action_count"] = (
                    direct_row_false_safe.tolist()
                )
                finite_summary["per_row_false_safe_action_count"] = (
                    finite_row_false_safe.tolist()
                )
                state_results.append({
                    "state_index": int(state["state_index"]), "case_id": case_id,
                    "split": state["split"], "state_step": step,
                    "state_offset_from_crossing": int(
                        state["state_offset_from_crossing"]
                    ),
                    "feature_max_abs_error": feature_error,
                    "clearance_max_abs_error_m": clearance_error,
                    "nominal_first_xyz": first[:3].tolist(),
                    "action_lower": lower.tolist(), "action_upper": upper.tolist(),
                    "fresh_seed": seed, "fit_action_count": len(fit_xyz),
                    "direct_halfspace": direct,
                    "finite_difference": finite,
                    "finite_difference_probe_records": finite_probe_records,
                    "fresh_actions": fresh_records,
                    "true_safe_fresh_action_count": sum(
                        item["true_safe"] for item in fresh_records
                    ),
                    "method_summaries": {
                        "direct_halfspace": direct_summary,
                        "finite_difference": finite_summary,
                    },
                })
                print(json.dumps({
                    "state": len(state_results), "case_id": case_id,
                    "state_step": step,
                    "fresh_true_safe": state_results[-1][
                        "true_safe_fresh_action_count"
                    ],
                    "direct_false_safe": direct_summary["false_safe_action_count"],
                    "fd_false_safe": finite_summary["false_safe_action_count"],
                }, sort_keys=True), flush=True)
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()

    _require(
        len(state_results) == int(source_settings["expected_state_count"])
        and sorted(item["state_index"] for item in state_results)
        == list(range(len(state_results))),
        "oracle comparison completed state set differs",
    )
    state_results.sort(key=lambda item: item["state_index"])
    methods = {
        name: _method_aggregate(state_results, name, config["decision_gate"])
        for name in ("direct_halfspace", "finite_difference")
    }
    assumption_supported = bool(any(
        item["method_gate_pass"] for item in methods.values()
    ))
    decision = {
        "direct_halfspace_gate_pass": methods["direct_halfspace"]["method_gate_pass"],
        "finite_difference_gate_pass": methods["finite_difference"]["method_gate_pass"],
        "oracle_affine_assumption_supported": assumption_supported,
        "future_direct_halfspace_training_authorized": assumption_supported,
        "next_model_action": (
            "train_direct_halfspace_one_sided_loss_with_held_out_tightening"
            if assumption_supported else
            "reduce_trust_region_or_use_multiple_local_affine_regions"
        ),
        "closed_loop_e05_authorized": False,
    }
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_dataset": {
            "path": str(paths["dataset"]),
            "file_sha256": _file_sha256(paths["dataset"]),
            "payload_sha256": dataset["dataset_payload_sha256"],
            "read_only": True,
        },
        "state_results": state_results, "method_aggregates": methods,
        "decision": decision,
        "rollout_counts": {
            "finite_difference": len(state_results) * 6,
            "fresh_evaluation": len(state_results) * count_per_state,
            "total_new_cloned_OSC": len(state_results) * (6 + count_per_state),
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = _hash_without(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "method_aggregates": methods, "decision": decision,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
