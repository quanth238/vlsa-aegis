#!/usr/bin/env python3
"""Evaluate fixed overlapping regional affine oracles without learning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_coefficient_model import (
    AFFINE_COEFFICIENT_DATASET_SCHEMA,
)
from main.multilink_ellipsoid.multi_region_affine_oracle import (
    RESULT_SCHEMA, audit_region_resampling, certificate_at_nominal,
    containing_region_indexes, fit_region_target, fixed_regions, load_config,
    region_affine_values,
)
from main.multilink_ellipsoid.oracle_affine_safe_set import (
    solve_affine_certificate_qp,
)
from main.multilink_ellipsoid.ridge_huber_oracle import affine_values
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


METHODS = ("multi_region", "single_affine")


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _state_region_seed(
    global_seed: int, case_id: str, state_step: int, region_index: int,
) -> int:
    payload = "%d::%s::%d::%d::multi-region" % (
        global_seed, case_id, state_step, region_index,
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _arm_key(clearance_m: float) -> str:
    return "%dmm" % round(float(clearance_m) * 1000.0)


def _summary(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    true_count = sum(true for true, _ in pairs)
    predicted_count = sum(predicted for _, predicted in pairs)
    accepted = sum(true and predicted for true, predicted in pairs)
    false_safe = sum((not true) and predicted for true, predicted in pairs)
    return {
        "action_count": len(pairs), "true_safe_action_count": true_count,
        "predicted_safe_action_count": predicted_count,
        "accepted_true_safe_action_count": accepted,
        "false_safe_action_count": false_safe,
        "safe_action_recall": accepted / true_count if true_count else None,
    }


def _immutable_true_safe(
    record: Mapping[str, Any], clearance_m: float, maximum_motion_m: float,
) -> bool:
    import numpy as np

    return bool(
        np.all(np.asarray(record["minimum_distal_margin_m"], dtype=np.float64)
               >= float(clearance_m))
        and record["D_sim_raw_safe"]
        and int(record["raw_protected_contact_count"]) == 0
        and float(record["maximum_within_step_obstacle_l1_displacement_m"])
        <= maximum_motion_m
    )


def _exact_record(
    summary: Mapping[str, Any], ee: float, clearance_m: float,
    maximum_motion_m: float,
) -> dict[str, Any]:
    import numpy as np

    margins = np.asarray(summary["minimum_substep_clearance_m"], dtype=np.float64)
    distal = margins[:7]
    raw_safe = bool(
        summary["D_sim_raw_safe"]
        and int(summary["raw_protected_contact_count"]) == 0
        and float(summary["maximum_within_step_obstacle_l1_displacement_m"])
        <= maximum_motion_m
    )
    return {
        "minimum_distal_margin_m": distal.tolist(),
        "minimum_released_AEGIS_EE_margin_m": float(ee),
        "clearance_target_m": float(clearance_m),
        "D_opt_seven_distal_at_target": bool(np.all(distal >= float(clearance_m))),
        "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
        "raw_protected_contact_count": int(summary["raw_protected_contact_count"]),
        "maximum_within_step_obstacle_l1_displacement_m": float(
            summary["maximum_within_step_obstacle_l1_displacement_m"]
        ),
        "true_safe": bool(np.all(distal >= float(clearance_m)) and raw_safe),
        "next_state_sha256": summary["next_state_sha256"],
    }


def _single_certificate(
    target: Mapping[str, Any], nominal_xyz: Any, clearance_m: float,
) -> dict[str, Any]:
    import numpy as np

    return {
        "valid": True,
        "intercept_at_nominal_m": (
            np.asarray(target["nominal_margin_m"], dtype=np.float64)
            - np.asarray(target["one_sided_error_m"], dtype=np.float64)
            - float(clearance_m)
        ).tolist(),
        "gradients_m_per_action": target["gradient_m_per_action"],
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
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--off-grid-result", type=Path, required=True)
    parser.add_argument("--off-grid-validation", type=Path, required=True)
    parser.add_argument("--single-affine-result", type=Path, required=True)
    parser.add_argument("--single-affine-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {name: value.resolve() for name, value in {
        "repo": args.repo_root, "population": args.population_manifest,
        "selected": args.selected_manifest, "archived": args.archived_root,
        "geometry": args.geometry_config, "exact_box": args.exact_box_config,
        "config": args.config, "dataset": args.dataset,
        "dataset_result": args.dataset_result,
        "dataset_validation": args.dataset_validation,
        "off_grid_result": args.off_grid_result,
        "off_grid_validation": args.off_grid_validation,
        "single_affine_result": args.single_affine_result,
        "single_affine_validation": args.single_affine_validation,
        "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, expected, label in (
        (paths["population"], config["source_population_manifest_sha256"], "population"),
        (paths["selected"], config["selected_manifest_sha256"], "selected"),
        (paths["geometry"], config["geometry_config_file_sha256"], "geometry"),
        (paths["exact_box"], config["exact_box_config_file_sha256"], "exact-box"),
        (paths["dataset"], source["dataset_file_sha256"], "dataset"),
        (paths["dataset_result"], source["dataset_result_file_sha256"], "dataset-result"),
        (paths["dataset_validation"], source["dataset_validation_file_sha256"], "dataset-validation"),
        (paths["off_grid_result"], source["off_grid_result_file_sha256"], "off-grid-result"),
        (paths["off_grid_validation"], source["off_grid_validation_file_sha256"], "off-grid-validation"),
        (paths["single_affine_result"], source["single_affine_result_file_sha256"], "single-affine-result"),
        (paths["single_affine_validation"], source["single_affine_validation_file_sha256"], "single-affine-validation"),
    ):
        _require(_file_sha256(path) == expected, "multi-region %s differs" % label)
    dataset = _load(paths["dataset"])
    dataset_result = _load(paths["dataset_result"])
    dataset_validation = _load(paths["dataset_validation"])
    off_grid = _load(paths["off_grid_result"])
    off_grid_validation = _load(paths["off_grid_validation"])
    single_result = _load(paths["single_affine_result"])
    single_validation = _load(paths["single_affine_validation"])
    _require(
        dataset.get("schema_version") == AFFINE_COEFFICIENT_DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source["dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256")
        and dataset.get("source_commit") == source["dataset_source_commit"]
        and dataset_result.get("status") == "complete"
        and dataset_validation.get("status") == "valid"
        and off_grid.get("result_payload_sha256")
        == source["off_grid_result_payload_sha256"]
        == _hash_without(off_grid, "result_payload_sha256")
        and off_grid_validation.get("status") == "valid"
        and single_result.get("result_payload_sha256")
        == source["single_affine_result_payload_sha256"]
        == _hash_without(single_result, "result_payload_sha256")
        and single_validation.get("status") == "valid",
        "multi-region immutable source identity differs",
    )
    states = dataset["state_records"]
    off_grid_by_index = {
        int(item["state_index"]): item for item in off_grid["state_results"]
    }
    single_by_index = {
        int(item["state_index"]): item for item in single_result["state_results"]
    }
    _require(
        len(states) == int(source["expected_state_count"])
        and set(off_grid_by_index) == set(single_by_index) == set(range(len(states))),
        "multi-region state population differs",
    )
    maximum_motion = float(config["exact_verification"][
        "maximum_per_step_obstacle_l1_displacement_m"
    ])
    tolerance = float(config["partition"]["inclusive_membership_tolerance"])
    state_results = []
    for state in states:
        index = int(state["state_index"])
        fresh_source = off_grid_by_index[index]
        single_source = single_by_index[index]
        _require(
            fresh_source["case_id"] == single_source["case_id"] == state["case_id"]
            and int(fresh_source["state_step"]) == int(state["state_step"])
            and len(fresh_source["fresh_actions"])
            == int(source["off_grid_actions_per_state"]),
            "multi-region off-grid pairing differs",
        )
        fit_xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
        fit_margins = np.asarray(
            state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        nominal_xyz = np.asarray(state["nominal_first_action"][:3], dtype=np.float64)
        regions = fixed_regions(
            fit_xyz, state["action_lower"], state["action_upper"],
            config["partition"],
        )
        targets = []
        stability = []
        for region in regions:
            target = fit_region_target(
                fit_xyz, fit_margins, region, config["ridge_huber"]
            )
            targets.append(target)
            stability.append(audit_region_resampling(
                fit_xyz, fit_margins, region, target,
                config["ridge_huber"], config["resampling"],
                _state_region_seed(
                    int(config["resampling"]["seed"]), state["case_id"],
                    int(state["state_step"]), int(region["region_index"]),
                ),
            ))
        single_target = single_source["ridge_huber_target"]
        fresh_records = []
        summaries = {
            _arm_key(clearance): {method: [] for method in METHODS}
            for clearance in config["clearance_arms_m"]
        }
        for fresh in fresh_source["fresh_actions"]:
            xyz = np.asarray(fresh["candidate_xyz"], dtype=np.float64)
            containing = containing_region_indexes(xyz, regions, tolerance)
            region_minima = {
                str(region_index): float(np.min(region_affine_values(
                    targets[region_index], xyz
                ))) for region_index in containing
            }
            single_values = affine_values(
                single_target["nominal_margin_m"],
                single_target["gradient_m_per_action"],
                single_target["one_sided_error_m"], xyz, nominal_xyz,
            )
            arms = {}
            for clearance in config["clearance_arms_m"]:
                key = _arm_key(clearance)
                true_safe = _immutable_true_safe(fresh, clearance, maximum_motion)
                predictions = {
                    "multi_region": any(
                        region_minima[str(region_index)] >= float(clearance)
                        for region_index in containing
                    ),
                    "single_affine": bool(np.all(single_values >= float(clearance))),
                }
                for method in METHODS:
                    summaries[key][method].append((true_safe, predictions[method]))
                arms[key] = {
                    "clearance_target_m": float(clearance), "true_safe": true_safe,
                    "multi_region_predicted_safe": predictions["multi_region"],
                    "single_affine_predicted_safe": predictions["single_affine"],
                }
            fresh_records.append({
                "fresh_index": int(fresh["fresh_index"]),
                "candidate_xyz": fresh["candidate_xyz"],
                "minimum_distal_margin_m": fresh["minimum_distal_margin_m"],
                "D_sim_raw_safe": fresh["D_sim_raw_safe"],
                "raw_protected_contact_count": fresh["raw_protected_contact_count"],
                "maximum_within_step_obstacle_l1_displacement_m": fresh[
                    "maximum_within_step_obstacle_l1_displacement_m"
                ],
                "containing_region_indexes": containing,
                "regional_minimum_lower_bounds_m": region_minima,
                "single_affine_values_m": np.asarray(single_values).tolist(),
                "arms": arms,
            })
        state_results.append({
            "state_index": index, "case_id": state["case_id"],
            "split": state["split"], "state_step": int(state["state_step"]),
            "state_offset_from_crossing": int(state["state_offset_from_crossing"]),
            "nominal_first_xyz": nominal_xyz.tolist(),
            "action_lower": state["action_lower"], "action_upper": state["action_upper"],
            "regions": regions, "regional_targets": targets,
            "regional_resampling_stability": stability,
            "all_active_region_rows_stable": bool(all(
                item["all_active_rows_stable"] for item in stability
            )),
            "single_affine_target": single_target,
            "fresh_actions": fresh_records,
            "off_grid_summaries": {
                key: {method: _summary(values) for method, values in methods.items()}
                for key, methods in summaries.items()
            },
            "QP": {},
        })
    split_counts = {
        name: sum(item["split"] == name for item in state_results)
        for name in ("train", "validation", "test")
    }
    _require(split_counts == config["split"]["expected_state_counts"],
             "multi-region grouped split differs")

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
        and placeholder.get("result_payload_sha256")
        == primary["archived_payload_sha256"],
        "multi-region geometry placeholder differs",
    )
    runtime = _runtime_imports(include_aegis=False)
    state_by_index = {item["state_index"]: item for item in state_results}
    rollout_count = 0
    for case_id in sorted(selected_by_case):
        row = selected_by_case[case_id]
        archived_path = paths["archived"] / row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == row["archived_file_sha256"]
            and archived.get("result_payload_sha256") == row["archived_payload_sha256"],
            "multi-region archived case differs: %s" % case_id,
        )
        case_states = sorted(
            (item for item in states if item["case_id"] == case_id),
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
                index = int(source_state["state_index"])
                result_state = state_by_index[index]
                step = int(source_state["state_step"])
                _restore_env(env, snapshots[step])
                first = _canonical_action(actions[step], step)
                second = _canonical_action(actions[step + 1], step + 1)
                context = feature_context(env, probe)
                _, live_features = feature_vectors(context, first[:3], first[:3], second[:3])
                feature_error = float(np.max(np.abs(
                    live_features - np.asarray(
                        source_state["pair_state_feature_vectors"], dtype=np.float64
                    )
                )))
                clearance_error = float(np.max(np.abs(
                    np.asarray(context["current_clearance_m"], dtype=np.float64)
                    - np.asarray(source_state["current_clearance_m"], dtype=np.float64)
                )))
                _require(feature_error <= 1.0e-8 and clearance_error <= 1.0e-8,
                         "multi-region live state receipt differs")
                result_state["feature_max_abs_error"] = feature_error
                result_state["clearance_max_abs_error_m"] = clearance_error
                for clearance in config["clearance_arms_m"]:
                    arm = _arm_key(clearance)
                    proposals = []
                    for region, target in zip(
                        result_state["regions"], result_state["regional_targets"]
                    ):
                        certificate = certificate_at_nominal(target, first[:3], clearance)
                        qp = solve_affine_certificate_qp(
                            first[:3], region["action_lower"], region["action_upper"],
                            certificate, config["projection"],
                        )
                        exact = None
                        if qp["valid"]:
                            candidate = first.copy()
                            candidate[:3] = np.asarray(qp["candidate_xyz"], dtype=np.float64)
                            chunk, ee = _chunk_values(
                                probe.rollout_chunk(env, [candidate, second])
                            )
                            rollout_count += 1
                            exact = _exact_record(
                                chunk, ee, float(clearance), maximum_motion
                            )
                        proposals.append({
                            "region_index": int(region["region_index"]),
                            "solution": qp, "fresh_exact_two_step": exact,
                            "false_safe": bool(qp["valid"] and not exact["true_safe"]),
                            "verified_safe": bool(qp["valid"] and exact["true_safe"]),
                        })
                    safe_proposals = [item for item in proposals if item["verified_safe"]]
                    safe_proposals.sort(key=lambda item: (
                        float(item["solution"]["correction_l2"]), item["region_index"]
                    ))
                    selected_proposal = None if not safe_proposals else safe_proposals[0]
                    single_certificate = _single_certificate(
                        result_state["single_affine_target"], first[:3], clearance
                    )
                    single_qp = solve_affine_certificate_qp(
                        first[:3], source_state["action_lower"],
                        source_state["action_upper"], single_certificate,
                        config["projection"],
                    )
                    single_exact = None
                    if single_qp["valid"]:
                        candidate = first.copy()
                        candidate[:3] = np.asarray(single_qp["candidate_xyz"], dtype=np.float64)
                        chunk, ee = _chunk_values(probe.rollout_chunk(env, [candidate, second]))
                        rollout_count += 1
                        single_exact = _exact_record(
                            chunk, ee, float(clearance), maximum_motion
                        )
                    result_state["QP"][arm] = {
                        "clearance_target_m": float(clearance),
                        "multi_region_proposals": proposals,
                        "valid_proposal_count": sum(item["solution"]["valid"] for item in proposals),
                        "false_safe_proposal_count": sum(item["false_safe"] for item in proposals),
                        "verified_safe_proposal_count": len(safe_proposals),
                        "selected_region_index": (
                            None if selected_proposal is None
                            else int(selected_proposal["region_index"])
                        ),
                        "selected_solution": (
                            None if selected_proposal is None
                            else selected_proposal["solution"]
                        ),
                        "selected_fresh_exact_two_step": (
                            None if selected_proposal is None
                            else selected_proposal["fresh_exact_two_step"]
                        ),
                        "selected_gate_pass": selected_proposal is not None,
                        "single_affine": {
                            "solution": single_qp, "fresh_exact_two_step": single_exact,
                            "false_safe": bool(
                                single_qp["valid"] and not single_exact["true_safe"]
                            ),
                            "gate_pass": bool(
                                single_qp["valid"] and single_exact["true_safe"]
                            ),
                        },
                    }
                print(json.dumps({
                    "state": index + 1, "case_id": case_id, "state_step": step,
                    "stable": result_state["all_active_region_rows_stable"],
                    "arms": {
                        arm: {
                            "off_grid_false_safe": result_state[
                                "off_grid_summaries"
                            ][arm]["multi_region"]["false_safe_action_count"],
                            "accepted_safe": result_state[
                                "off_grid_summaries"
                            ][arm]["multi_region"]["accepted_true_safe_action_count"],
                            "qp_false_safe": result_state["QP"][arm][
                                "false_safe_proposal_count"
                            ],
                            "selected_safe": result_state["QP"][arm][
                                "selected_gate_pass"
                            ],
                        } for arm in result_state["QP"]
                    },
                }, sort_keys=True), flush=True)
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()

    stability_gate = bool(all(
        state["all_active_region_rows_stable"] for state in state_results
    ))
    arm_aggregates = {}
    arm_decisions = {}
    required_recovery = set(config["oracle_gate"][
        "required_zero_margin_recovery_state_indexes"
    ])
    for clearance in config["clearance_arms_m"]:
        arm = _arm_key(clearance)
        method_aggregates = {}
        for method in METHODS:
            pairs = [
                (
                    fresh["arms"][arm]["true_safe"],
                    fresh["arms"][arm][method + "_predicted_safe"],
                ) for state in state_results for fresh in state["fresh_actions"]
            ]
            aggregate = _summary(pairs)
            aggregate.update({
                "states_with_exact_off_grid_safe_support": sum(
                    state["off_grid_summaries"][arm][method][
                        "true_safe_action_count"
                    ] > 0 for state in state_results
                ),
                "supported_states_with_accepted_safe_action": sum(
                    state["off_grid_summaries"][arm][method]["true_safe_action_count"] > 0
                    and state["off_grid_summaries"][arm][method][
                        "accepted_true_safe_action_count"
                    ] > 0 for state in state_results
                ),
                "states_with_selected_exact_safe_QP": sum(
                    (
                        state["QP"][arm]["selected_gate_pass"]
                        if method == "multi_region"
                        else state["QP"][arm]["single_affine"]["gate_pass"]
                    ) for state in state_results
                ),
                "QP_false_safe_proposal_count": sum(
                    (
                        state["QP"][arm]["false_safe_proposal_count"]
                        if method == "multi_region"
                        else int(state["QP"][arm]["single_affine"]["false_safe"])
                    ) for state in state_results
                ),
            })
            method_aggregates[method] = aggregate
        recovered = sorted(
            index for index in required_recovery
            if state_results[index]["off_grid_summaries"][arm]["multi_region"][
                "accepted_true_safe_action_count"
            ] > 0 and state_results[index]["QP"][arm]["selected_gate_pass"]
        )
        multi = method_aggregates["multi_region"]
        gate = bool(
            multi["false_safe_action_count"] == 0
            and multi["QP_false_safe_proposal_count"] == 0
            and multi["supported_states_with_accepted_safe_action"]
            == multi["states_with_exact_off_grid_safe_support"]
            and multi["states_with_selected_exact_safe_QP"] == len(state_results)
            and stability_gate
            and (float(clearance) != 0.0 or set(recovered) == required_recovery)
        )
        arm_aggregates[arm] = method_aggregates
        arm_decisions[arm] = {
            "clearance_target_m": float(clearance), "gate_pass": gate,
            "required_recovery_state_indexes": sorted(required_recovery),
            "recovered_required_state_indexes": recovered,
        }
    overall = bool(all(item["gate_pass"] for item in arm_decisions.values()))
    decision = {
        "clearance_arm_decisions": arm_decisions,
        "all_active_region_rows_resampling_stable": stability_gate,
        "multi_region_oracle_gate_pass": overall,
        "mlp_training_authorized": overall,
        "mlp_training_executed": False,
        "closed_loop_e05_authorized": False,
        "closed_loop_e05_executed": False,
        "stop_reason": None if overall else "one_or_more_multi_region_oracle_gates_failed",
    }
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_inputs": {
            name: {"path": str(paths[name]), "file_sha256": _file_sha256(paths[name])}
            for name in (
                "dataset", "dataset_result", "dataset_validation",
                "off_grid_result", "off_grid_validation",
                "single_affine_result", "single_affine_validation",
            )
        },
        "split_state_counts": split_counts, "state_results": state_results,
        "arm_aggregates": arm_aggregates, "decision": decision,
        "new_cloned_OSC_rollout_count": rollout_count,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "arm_aggregates": arm_aggregates, "decision": decision,
        "new_cloned_OSC_rollout_count": rollout_count,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
