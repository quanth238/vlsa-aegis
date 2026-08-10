#!/usr/bin/env python3
"""Test whether regional coefficient resampling changes safety decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.multi_region_affine_oracle import (
    certificate_at_nominal, containing_region_indexes, region_affine_values,
)
from main.multilink_ellipsoid.multi_region_decision_stability import (
    RESULT_SCHEMA, fit_resampled_region_target, jaccard, load_config,
    shared_region_resamples,
)
from main.multilink_ellipsoid.oracle_affine_safe_set import (
    solve_affine_certificate_qp,
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


def _seed(global_seed: int, case_id: str, step: int, region: int) -> int:
    payload = "%d::%s::%d::%d::decision" % (
        global_seed, case_id, step, region,
    )
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def _exact_record(
    summary: Mapping[str, Any], ee: float, maximum_motion_m: float,
) -> dict[str, Any]:
    import numpy as np

    margins = np.asarray(summary["minimum_substep_clearance_m"], dtype=np.float64)[:7]
    true_safe = bool(
        np.all(margins >= 0.0)
        and summary["D_sim_raw_safe"]
        and int(summary["raw_protected_contact_count"]) == 0
        and float(summary["maximum_within_step_obstacle_l1_displacement_m"])
        <= maximum_motion_m
    )
    return {
        "minimum_distal_margin_m": margins.tolist(),
        "minimum_released_AEGIS_EE_margin_m": float(ee),
        "D_opt_seven_distal_safe": bool(np.all(margins >= 0.0)),
        "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
        "raw_protected_contact_count": int(summary["raw_protected_contact_count"]),
        "maximum_within_step_obstacle_l1_displacement_m": float(
            summary["maximum_within_step_obstacle_l1_displacement_m"]
        ),
        "true_safe": true_safe, "next_state_sha256": summary["next_state_sha256"],
    }


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
    parser.add_argument("--off-grid-result", type=Path, required=True)
    parser.add_argument("--multi-region-result", type=Path, required=True)
    parser.add_argument("--multi-region-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {name: path.resolve() for name, path in {
        "repo": args.repo_root, "population": args.population_manifest,
        "selected": args.selected_manifest, "archived": args.archived_root,
        "geometry": args.geometry_config, "exact_box": args.exact_box_config,
        "config": args.config, "dataset": args.dataset,
        "off_grid": args.off_grid_result, "multi": args.multi_region_result,
        "multi_validation": args.multi_region_validation, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, expected, label in (
        (paths["population"], config["source_population_manifest_sha256"], "population"),
        (paths["selected"], config["selected_manifest_sha256"], "selected"),
        (paths["geometry"], config["geometry_config_file_sha256"], "geometry"),
        (paths["exact_box"], config["exact_box_config_file_sha256"], "exact-box"),
        (paths["dataset"], source["dataset_file_sha256"], "dataset"),
        (paths["off_grid"], source["off_grid_result_file_sha256"], "off-grid"),
        (paths["multi"], source["multi_region_result_file_sha256"], "multi-region"),
        (paths["multi_validation"], source["multi_region_validation_file_sha256"], "multi-validation"),
    ):
        _require(_file_sha256(path) == expected, "decision-stability %s differs" % label)
    dataset = _load(paths["dataset"])
    off_grid = _load(paths["off_grid"])
    baseline = _load(paths["multi"])
    baseline_validation = _load(paths["multi_validation"])
    _require(
        dataset.get("dataset_payload_sha256")
        == source["dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256")
        and off_grid.get("result_payload_sha256")
        == source["off_grid_result_payload_sha256"]
        == _hash_without(off_grid, "result_payload_sha256")
        and baseline.get("result_payload_sha256")
        == source["multi_region_result_payload_sha256"]
        == _hash_without(baseline, "result_payload_sha256")
        and baseline_validation.get("status") == "valid",
        "decision-stability immutable payload differs",
    )
    source_states = {int(item["state_index"]): item for item in dataset["state_records"]}
    fresh_states = {int(item["state_index"]): item for item in off_grid["state_results"]}
    baseline_states = {int(item["state_index"]): item for item in baseline["state_results"]}
    expected_count = int(source["expected_state_count"])
    _require(
        set(source_states) == set(fresh_states) == set(baseline_states)
        == set(range(expected_count)),
        "decision-stability state set differs",
    )
    replicate_count = int(config["resampling"]["replicate_count"])
    selected_count = int(config["resampling"]["selected_actions_per_region"])
    tolerance = float(baseline["config"]["partition"][
        "inclusive_membership_tolerance"
    ])
    state_results = []
    for state_index in range(expected_count):
        source_state = source_states[state_index]
        fresh_state = fresh_states[state_index]
        base_state = baseline_states[state_index]
        xyz = np.asarray(source_state["candidate_first_xyz"], dtype=np.float64)
        margins = np.asarray(
            source_state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        regions = base_state["regions"]
        region_subsets = []
        for region in regions:
            region_subsets.append(shared_region_resamples(
                region["fit_candidate_indexes"], replicate_count=replicate_count,
                selected_count=selected_count,
                seed=_seed(
                    int(config["resampling"]["seed"]), source_state["case_id"],
                    int(source_state["state_step"]), int(region["region_index"]),
                ),
            ))
        base_flags = np.asarray([
            bool(item["arms"]["0mm"]["multi_region_predicted_safe"])
            for item in base_state["fresh_actions"]
        ], dtype=bool)
        true_flags = np.asarray([
            bool(item["true_safe"]) for item in fresh_state["fresh_actions"]
        ], dtype=bool)
        replicates = []
        for replicate_index in range(replicate_count):
            targets = [
                fit_resampled_region_target(
                    xyz, margins, region, region_subsets[region_index][replicate_index],
                    config["ridge_huber"],
                ) for region_index, region in enumerate(regions)
            ]
            flags = []
            accepted_indexes = []
            false_safe_indexes = []
            for fresh_index, fresh in enumerate(fresh_state["fresh_actions"]):
                action = np.asarray(fresh["candidate_xyz"], dtype=np.float64)
                containing = containing_region_indexes(action, regions, tolerance)
                accepted = bool(any(
                    np.all(region_affine_values(targets[index], action) >= 0.0)
                    for index in containing
                ))
                flags.append(accepted)
                if accepted:
                    accepted_indexes.append(fresh_index)
                    if not true_flags[fresh_index]:
                        false_safe_indexes.append(fresh_index)
            flags_array = np.asarray(flags, dtype=bool)
            accepted_true = int(np.count_nonzero(np.logical_and(flags_array, true_flags)))
            replicates.append({
                "replicate_index": replicate_index, "regional_targets": targets,
                "accepted_fresh_indexes": accepted_indexes,
                "false_safe_fresh_indexes": false_safe_indexes,
                "accepted_true_safe_action_count": accepted_true,
                "accepted_set_jaccard_to_full_fit": jaccard(base_flags, flags_array),
                "QP": None,
            })
        state_results.append({
            "state_index": state_index, "case_id": source_state["case_id"],
            "split": source_state["split"], "state_step": int(source_state["state_step"]),
            "nominal_first_xyz": source_state["nominal_first_action"][:3],
            "regions": regions, "baseline_accepted_fresh_indexes": np.flatnonzero(
                base_flags
            ).tolist(),
            "baseline_accepted_true_safe_action_count": int(np.count_nonzero(
                np.logical_and(base_flags, true_flags)
            )),
            "baseline_selected_region_index": base_state["QP"]["0mm"][
                "selected_region_index"
            ],
            "baseline_selected_xyz": base_state["QP"]["0mm"][
                "selected_solution"
            ]["candidate_xyz"],
            "replicates": replicates,
            "feature_max_abs_error": None, "clearance_max_abs_error_m": None,
        })

    selected = load_selected_manifest(paths["selected"], {
        "selected_manifest_sha256": config["selected_manifest_sha256"],
        "split": {
            "train_task_groups": ["vlsa-t1-goal-ii-t2", "vlsa-t1-spatial-i-t1"],
            "validation_task_groups": ["vlsa-t1-goal-ii-t3"],
            "test_task_groups": ["vlsa-t1-goal-ii-t0"],
        },
    })
    selected_by_case = {item["case_id"]: item for item in selected}
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    primary = selected_by_case["vlsa-t1-goal-ii-t0-e05"]
    placeholder_path = paths["archived"] / primary["archived_relative_path"]
    placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == primary["archived_file_sha256"]
        and placeholder.get("result_payload_sha256") == primary["archived_payload_sha256"],
        "decision-stability geometry placeholder differs",
    )
    runtime = _runtime_imports(include_aegis=False)
    state_by_index = {item["state_index"]: item for item in state_results}
    maximum_motion = float(config["exact_verification"][
        "maximum_per_step_obstacle_l1_displacement_m"
    ])
    rollout_count = 0
    for case_id in sorted(selected_by_case):
        row = selected_by_case[case_id]
        archived_path = paths["archived"] / row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == row["archived_file_sha256"]
            and archived.get("result_payload_sha256") == row["archived_payload_sha256"],
            "decision-stability archived case differs: %s" % case_id,
        )
        case_states = sorted(
            (item for item in source_states.values() if item["case_id"] == case_id),
            key=lambda item: int(item["state_step"]),
        )
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
            for source_state in case_states:
                state = state_by_index[int(source_state["state_index"])]
                step = int(source_state["state_step"])
                _restore_env(env, snapshots[step])
                first = _canonical_action(actions[step], step)
                second = _canonical_action(actions[step + 1], step + 1)
                context = feature_context(env, probe)
                _, live_features = feature_vectors(context, first[:3], first[:3], second[:3])
                state["feature_max_abs_error"] = float(np.max(np.abs(
                    live_features - np.asarray(
                        source_state["pair_state_feature_vectors"], dtype=np.float64
                    )
                )))
                state["clearance_max_abs_error_m"] = float(np.max(np.abs(
                    np.asarray(context["current_clearance_m"], dtype=np.float64)
                    - np.asarray(source_state["current_clearance_m"], dtype=np.float64)
                )))
                _require(
                    state["feature_max_abs_error"] <= 1.0e-8
                    and state["clearance_max_abs_error_m"] <= 1.0e-8,
                    "decision-stability live state differs",
                )
                baseline_xyz = np.asarray(state["baseline_selected_xyz"], dtype=np.float64)
                for replicate in state["replicates"]:
                    proposals = []
                    for region, target in zip(state["regions"], replicate["regional_targets"]):
                        qp = solve_affine_certificate_qp(
                            first[:3], region["action_lower"], region["action_upper"],
                            certificate_at_nominal(target, first[:3], 0.0),
                            config["projection"],
                        )
                        proposals.append({
                            "region_index": int(region["region_index"]),
                            "valid": bool(qp["valid"]), "reason": qp["reason"],
                            "candidate_xyz": qp["candidate_xyz"],
                            "correction_l2": qp["correction_l2"],
                        })
                    valid = [item for item in proposals if item["valid"]]
                    valid.sort(key=lambda item: (
                        float(item["correction_l2"]), int(item["region_index"])
                    ))
                    chosen = None if not valid else valid[0]
                    exact = None
                    shift = None
                    if chosen is not None:
                        candidate = first.copy()
                        candidate[:3] = np.asarray(chosen["candidate_xyz"], dtype=np.float64)
                        chunk, ee = _chunk_values(probe.rollout_chunk(env, [candidate, second]))
                        rollout_count += 1
                        exact = _exact_record(chunk, ee, maximum_motion)
                        shift = float(np.linalg.norm(candidate[:3] - baseline_xyz))
                    replicate["QP"] = {
                        "regional_proposals": proposals,
                        "selected_region_index": (
                            None if chosen is None else int(chosen["region_index"])
                        ),
                        "selected_xyz": None if chosen is None else chosen["candidate_xyz"],
                        "selected_action_shift_l2_from_full_fit": shift,
                        "selected_region_matches_full_fit": bool(
                            chosen is not None
                            and int(chosen["region_index"])
                            == int(state["baseline_selected_region_index"])
                        ),
                        "fresh_exact_two_step": exact,
                        "gate_pass": bool(chosen is not None and exact["true_safe"]),
                    }
                state_shifts = [
                    item["QP"]["selected_action_shift_l2_from_full_fit"]
                    for item in state["replicates"]
                    if item["QP"]["selected_action_shift_l2_from_full_fit"] is not None
                ]
                print(json.dumps({
                    "state_index": state["state_index"], "case_id": case_id,
                    "state_step": step,
                    "minimum_jaccard": min(
                        item["accepted_set_jaccard_to_full_fit"]
                        for item in state["replicates"]
                    ),
                    "false_safe": sum(len(item["false_safe_fresh_indexes"])
                                      for item in state["replicates"]),
                    "safe_qp": sum(item["QP"]["gate_pass"]
                                   for item in state["replicates"]),
                    "maximum_action_shift": (
                        None if not state_shifts else max(state_shifts)
                    ),
                }, sort_keys=True), flush=True)
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()

    global_jaccards = []
    replicate_summaries = []
    for replicate_index in range(replicate_count):
        base = []
        sampled = []
        false_safe = 0
        retained_support = 0
        baseline_support = 0
        for state in state_results:
            baseline_indexes = set(state["baseline_accepted_fresh_indexes"])
            sampled_indexes = set(state["replicates"][replicate_index][
                "accepted_fresh_indexes"
            ])
            base.extend(index in baseline_indexes for index in range(96))
            sampled.extend(index in sampled_indexes for index in range(96))
            false_safe += len(state["replicates"][replicate_index][
                "false_safe_fresh_indexes"
            ])
            if state["baseline_accepted_true_safe_action_count"] > 0:
                baseline_support += 1
                retained_support += int(state["replicates"][replicate_index][
                    "accepted_true_safe_action_count"
                ] > 0)
        value = jaccard(base, sampled)
        global_jaccards.append(value)
        replicate_summaries.append({
            "replicate_index": replicate_index,
            "global_accepted_set_jaccard": value,
            "false_safe_action_count": false_safe,
            "baseline_supported_state_count": baseline_support,
            "retained_supported_state_count": retained_support,
        })
    all_replicates = [item for state in state_results for item in state["replicates"]]
    shifts = np.asarray([
        item["QP"]["selected_action_shift_l2_from_full_fit"]
        for item in all_replicates
        if item["QP"]["selected_action_shift_l2_from_full_fit"] is not None
    ], dtype=np.float64)
    false_safe_total = sum(
        len(item["false_safe_fresh_indexes"]) for item in all_replicates
    )
    minimum_state_jaccard = min(
        item["accepted_set_jaccard_to_full_fit"] for item in all_replicates
    )
    exact_safe_count = sum(item["QP"]["gate_pass"] for item in all_replicates)
    region_match_count = sum(
        item["QP"]["selected_region_matches_full_fit"] for item in all_replicates
    )
    p95 = float(np.percentile(shifts, 95)) if len(shifts) else None
    maximum = float(np.max(shifts)) if len(shifts) else None
    gate = config["decision_gate"]
    decisions = {
        "zero_replicated_false_safes": false_safe_total
        == int(gate["replicated_off_grid_false_safe_action_count"]),
        "global_jaccard_gate": min(global_jaccards)
        >= float(gate["minimum_global_accepted_set_jaccard_per_replicate"]),
        "state_jaccard_gate": minimum_state_jaccard
        >= float(gate["minimum_state_accepted_set_jaccard"]),
        "support_retention_gate": all(
            item["baseline_supported_state_count"]
            == int(gate["expected_baseline_supported_state_count"])
            and item["retained_supported_state_count"]
            == item["baseline_supported_state_count"]
            for item in replicate_summaries
        ),
        "fresh_exact_selected_QP_gate": bool(
            rollout_count == int(gate["expected_selected_QP_rollout_count"])
            and exact_safe_count == rollout_count
        ),
        "selected_action_p95_gate": bool(
            p95 is not None
            and p95 <= float(gate["selected_action_shift_l2_p95_maximum"])
        ),
        "selected_action_maximum_gate": bool(
            maximum is not None
            and maximum <= float(gate["selected_action_shift_l2_maximum"])
        ),
    }
    overall = bool(all(decisions.values()))
    decision = {
        "component_gates": decisions,
        "decision_stability_gate_pass": overall,
        "coefficient_nonuniqueness_decision_harmless": overall,
        "mlp_training_authorized": overall,
        "mlp_training_executed": False,
        "closed_loop_e05_authorized": False,
        "closed_loop_e05_executed": False,
        "stop_reason": None if overall else "one_or_more_decision_stability_gates_failed",
    }
    aggregates = {
        "state_count": len(state_results), "replicate_count": replicate_count,
        "replicated_off_grid_decision_count": expected_count * replicate_count * 96,
        "replicated_off_grid_false_safe_action_count": false_safe_total,
        "minimum_global_accepted_set_jaccard": min(global_jaccards),
        "minimum_state_accepted_set_jaccard": minimum_state_jaccard,
        "selected_QP_rollout_count": rollout_count,
        "fresh_exact_safe_selected_QP_count": exact_safe_count,
        "selected_region_match_count": region_match_count,
        "selected_action_shift_l2_mean": float(np.mean(shifts)),
        "selected_action_shift_l2_p95": p95,
        "selected_action_shift_l2_maximum": maximum,
    }
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_inputs": {
            name: {"path": str(paths[name]), "file_sha256": _file_sha256(paths[name])}
            for name in ("dataset", "off_grid", "multi", "multi_validation")
        },
        "replicate_summaries": replicate_summaries,
        "state_results": state_results, "aggregates": aggregates,
        "decision": decision,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
    _atomic_write(paths["output"], result)
    print(json.dumps({"aggregates": aggregates, "decision": decision},
                     sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
