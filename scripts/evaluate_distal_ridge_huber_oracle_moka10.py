#!/usr/bin/env python3
"""Evaluate ridge-Huber and minimum-L1 affine targets before learning."""

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
from main.multilink_ellipsoid.oracle_affine_safe_set import (
    fit_candidate_conditioned_affine_certificate, solve_affine_certificate_qp,
)
from main.multilink_ellipsoid.ridge_huber_oracle import (
    RESULT_SCHEMA, affine_values, audit_resampling, fit_state_targets, load_config,
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


METHODS = ("ridge_huber", "minimum_L1")


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _state_seed(global_seed: int, case_id: str, state_step: int) -> int:
    payload = "%d::%s::%d::ridge-huber" % (global_seed, case_id, state_step)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _summarize(records: list[tuple[bool, bool]]) -> dict[str, Any]:
    true_count = sum(true for true, _ in records)
    predicted_count = sum(predicted for _, predicted in records)
    accepted = sum(true and predicted for true, predicted in records)
    false_safe = sum((not true) and predicted for true, predicted in records)
    return {
        "action_count": len(records), "true_safe_action_count": true_count,
        "predicted_safe_action_count": predicted_count,
        "accepted_true_safe_action_count": accepted,
        "false_safe_action_count": false_safe,
        "safe_action_recall": accepted / true_count if true_count else None,
    }


def _exact_record(summary: Mapping[str, Any], ee: float, maximum_motion: float) -> dict[str, Any]:
    proxy_safe = bool(summary["D_opt_proxy_safe"])
    raw_safe = bool(
        summary["D_sim_raw_safe"]
        and summary["maximum_within_step_obstacle_l1_displacement_m"]
        <= maximum_motion
    )
    return {
        "minimum_distal_margin_m": summary["minimum_substep_clearance_m"],
        "minimum_released_AEGIS_EE_margin_m": float(ee),
        "D_opt_seven_distal_safe": proxy_safe,
        "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
        "raw_protected_contact_count": int(summary["raw_protected_contact_count"]),
        "maximum_within_step_obstacle_l1_displacement_m": float(
            summary["maximum_within_step_obstacle_l1_displacement_m"]
        ),
        "true_safe": bool(proxy_safe and raw_safe),
        "next_state_sha256": summary["next_state_sha256"],
    }


def _minimum_l1_resampling_audit(
    *, state: Mapping[str, Any], ridge_stability: Mapping[str, Any],
    resampling: Mapping[str, Any],
) -> dict[str, Any]:
    """Refit the minimum-L1 target on the exact ridge-Huber resamples."""

    import numpy as np

    xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
    margins = np.asarray(
        state["candidate_minimum_distal_margin_m"], dtype=np.float64
    )
    raw_safe = np.asarray(state["candidate_raw_safe"], dtype=bool)
    nominal = np.asarray(state["nominal_first_action"][:3], dtype=np.float64)
    nominal_margin = np.asarray(
        state["nominal_minimum_distal_margin_m"], dtype=np.float64
    )
    full = np.asarray(
        state["coefficient_target"]["gradient_m_per_action"], dtype=np.float64
    )
    gradients = []
    valid = []
    for indexes in ridge_stability["replicate_candidate_indexes"]:
        selected = np.asarray(indexes, dtype=np.int64)
        certificate = fit_candidate_conditioned_affine_certificate(
            np.vstack((xyz[selected], nominal[None, :])),
            np.vstack((margins[selected], nominal_margin[None, :])),
            nominal,
            np.concatenate((raw_safe[selected], np.asarray([False], dtype=bool))),
            np.concatenate((selected, np.asarray([len(xyz)], dtype=np.int64))),
            one_sided_padding_m=1.0e-6, target_clearance_m=0.0,
            postcheck_tolerance_m=1.0e-8,
        )
        valid.append(bool(certificate["valid"]))
        gradients.append(
            certificate["gradients_m_per_action"]
            if certificate["valid"] else np.zeros((7, 3), dtype=np.float64).tolist()
        )
    replicate = np.asarray(gradients, dtype=np.float64)
    threshold = float(resampling["near_zero_gradient_m_per_action"])
    row_audits = []
    for row in range(7):
        active = bool(
            np.any(margins[:, row] < 0.0)
            or float(np.min(np.abs(margins[:, row]))) <= 0.005
        )
        norm = float(np.linalg.norm(full[row]))
        replicate_norm = np.linalg.norm(replicate[:, row, :], axis=1)
        differences = np.linalg.norm(replicate[:, row, :] - full[row], axis=1)
        if norm <= threshold:
            cosine_minimum = None
            relative_norm_difference_maximum = None
            stable = bool(
                all(valid) and float(np.max(replicate_norm)) <= threshold
                and float(np.max(differences)) <= threshold
            )
        else:
            cosine = (
                replicate[:, row, :] @ full[row]
                / np.maximum(replicate_norm * norm, 1.0e-30)
            )
            cosine_minimum = float(np.min(cosine))
            relative_norm_difference_maximum = float(np.max(
                np.abs(replicate_norm - norm) / norm
            ))
            stable = bool(
                all(valid)
                and cosine_minimum >= float(resampling["minimum_cosine"])
                and relative_norm_difference_maximum
                <= float(resampling["maximum_relative_norm_difference"])
            )
        row_audits.append({
            "constraint_index": row, "active": active,
            "full_gradient_norm_m_per_action": norm,
            "cosine_minimum": cosine_minimum,
            "relative_norm_difference_maximum": relative_norm_difference_maximum,
            "stable": stable, "gate_applies": active,
            "gate_pass": bool((not active) or stable),
        })
    return {
        "replicate_count": len(replicate),
        "valid_replicate_count": int(sum(valid)),
        "row_audits": row_audits,
        "all_active_rows_stable": bool(all(item["gate_pass"] for item in row_audits)),
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
        "off_grid_validation": args.off_grid_validation, "output": args.output,
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
    ):
        _require(_file_sha256(path) == expected, "ridge-Huber %s differs" % label)
    dataset = _load(paths["dataset"])
    dataset_result = _load(paths["dataset_result"])
    dataset_validation = _load(paths["dataset_validation"])
    off_grid = _load(paths["off_grid_result"])
    off_grid_validation = _load(paths["off_grid_validation"])
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
        and off_grid_validation.get("result_file_sha256")
        == source["off_grid_result_file_sha256"],
        "ridge-Huber immutable source identity differs",
    )
    states = dataset["state_records"]
    off_grid_by_index = {
        int(item["state_index"]): item for item in off_grid["state_results"]
    }
    _require(
        len(states) == int(source["expected_state_count"])
        and len(off_grid_by_index) == len(states)
        and [item["state_index"] for item in states] == list(range(len(states))),
        "ridge-Huber state population differs",
    )

    state_results = []
    for state in states:
        index = int(state["state_index"])
        fresh_source = off_grid_by_index[index]
        _require(
            fresh_source["case_id"] == state["case_id"]
            and fresh_source["state_step"] == state["state_step"]
            and fresh_source["split"] == state["split"]
            and len(fresh_source["fresh_actions"])
            == int(source["off_grid_actions_per_state"]),
            "ridge-Huber off-grid state pairing differs",
        )
        fit_xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
        fit_margins = np.asarray(
            state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        nominal_xyz = np.asarray(state["nominal_first_action"][:3], dtype=np.float64)
        nominal_margin = np.asarray(
            state["nominal_minimum_distal_margin_m"], dtype=np.float64
        )
        ridge = fit_state_targets(
            fit_xyz, fit_margins, nominal_xyz, nominal_margin,
            config["ridge_huber"],
        )
        stability = audit_resampling(
            fit_xyz, fit_margins, nominal_xyz, nominal_margin, ridge,
            config["ridge_huber"], config["resampling"],
            _state_seed(
                int(config["resampling"]["seed"]), state["case_id"],
                int(state["state_step"]),
            ),
        )
        minimum_l1_stability = _minimum_l1_resampling_audit(
            state=state, ridge_stability=stability,
            resampling=config["resampling"],
        )
        minimum_l1 = state["coefficient_target"]
        records = {method: [] for method in METHODS}
        fresh_records = []
        for fresh in fresh_source["fresh_actions"]:
            xyz = np.asarray(fresh["candidate_xyz"], dtype=np.float64)
            true_safe = bool(fresh["true_safe"])
            ridge_values = affine_values(
                ridge["nominal_margin_m"], ridge["gradient_m_per_action"],
                ridge["one_sided_error_m"], xyz, nominal_xyz,
            )
            l1_values = affine_values(
                minimum_l1["nominal_margin_m"],
                minimum_l1["gradient_m_per_action"],
                minimum_l1["state_conditioned_error_m"], xyz, nominal_xyz,
            )
            predictions = {
                "ridge_huber": bool(np.all(ridge_values >= 0.0)),
                "minimum_L1": bool(np.all(l1_values >= 0.0)),
            }
            for method in METHODS:
                records[method].append((true_safe, predictions[method]))
            fresh_records.append({
                "fresh_index": int(fresh["fresh_index"]),
                "candidate_xyz": fresh["candidate_xyz"],
                "minimum_distal_margin_m": fresh["minimum_distal_margin_m"],
                "D_sim_raw_safe": fresh["D_sim_raw_safe"],
                "raw_protected_contact_count": fresh[
                    "raw_protected_contact_count"
                ],
                "maximum_within_step_obstacle_l1_displacement_m": fresh[
                    "maximum_within_step_obstacle_l1_displacement_m"
                ],
                "true_safe": true_safe,
                "ridge_huber_values_m": ridge_values.tolist(),
                "minimum_L1_values_m": l1_values.tolist(),
                "ridge_huber_predicted_safe": predictions["ridge_huber"],
                "minimum_L1_predicted_safe": predictions["minimum_L1"],
            })
        state_results.append({
            "state_index": index, "case_id": state["case_id"],
            "split": state["split"], "state_step": int(state["state_step"]),
            "state_offset_from_crossing": int(state["state_offset_from_crossing"]),
            "nominal_first_xyz": nominal_xyz.tolist(),
            "action_lower": state["action_lower"],
            "action_upper": state["action_upper"],
            "ridge_huber_target": ridge,
            "minimum_L1_target": minimum_l1,
            "resampling_stability": stability,
            "minimum_L1_resampling_stability": minimum_l1_stability,
            "fresh_actions": fresh_records,
            "off_grid_summaries": {
                method: _summarize(records[method]) for method in METHODS
            },
            "QP": {},
        })
    split_counts = {
        name: sum(item["split"] == name for item in state_results)
        for name in ("train", "validation", "test")
    }
    _require(
        split_counts == config["split"]["expected_state_counts"],
        "ridge-Huber grouped split differs",
    )

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
        "ridge-Huber geometry placeholder differs",
    )
    runtime = _runtime_imports(include_aegis=False)
    maximum_motion = float(config["exact_verification"][
        "maximum_per_step_obstacle_l1_displacement_m"
    ])
    state_by_index = {item["state_index"]: item for item in state_results}
    for case_id in sorted(selected_by_case):
        row = selected_by_case[case_id]
        archived_path = paths["archived"] / row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == row["archived_file_sha256"]
            and archived.get("result_payload_sha256") == row["archived_payload_sha256"],
            "ridge-Huber archived case differs: %s" % case_id,
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
            for state in case_states:
                index = int(state["state_index"])
                result_state = state_by_index[index]
                step = int(state["state_step"])
                _restore_env(env, snapshots[step])
                first = _canonical_action(actions[step], step)
                second = _canonical_action(actions[step + 1], step + 1)
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
                    "ridge-Huber live state receipt differs",
                )
                result_state["feature_max_abs_error"] = feature_error
                result_state["clearance_max_abs_error_m"] = clearance_error
                targets = {
                    "ridge_huber": result_state["ridge_huber_target"],
                    "minimum_L1": result_state["minimum_L1_target"],
                }
                for method, target in targets.items():
                    error_key = (
                        "one_sided_error_m" if method == "ridge_huber"
                        else "state_conditioned_error_m"
                    )
                    certificate = {
                        "valid": True,
                        "intercept_at_nominal_m": (
                            np.asarray(target["nominal_margin_m"], dtype=np.float64)
                            - np.asarray(target[error_key], dtype=np.float64)
                        ).tolist(),
                        "gradients_m_per_action": target[
                            "gradient_m_per_action"
                        ],
                    }
                    qp = solve_affine_certificate_qp(
                        first[:3], state["action_lower"], state["action_upper"],
                        certificate, config["projection"],
                    )
                    exact = None
                    if qp["valid"]:
                        candidate = first.copy()
                        candidate[:3] = np.asarray(qp["candidate_xyz"], dtype=np.float64)
                        summary, ee = _chunk_values(
                            probe.rollout_chunk(env, [candidate, second])
                        )
                        exact = _exact_record(summary, ee, maximum_motion)
                    gate = bool(
                        qp["valid"]
                        and qp.get("diagnostics", {}).get("input_constraint_count") == 7
                        and exact is not None and exact["true_safe"]
                    )
                    result_state["QP"][method] = {
                        "solution": qp, "fresh_exact_two_step": exact,
                        "gate_pass": gate,
                    }
                print(json.dumps({
                    "state": index + 1, "case_id": case_id, "state_step": step,
                    "ridge_false_safe": result_state["off_grid_summaries"][
                        "ridge_huber"
                    ]["false_safe_action_count"],
                    "ridge_accepted_safe": result_state["off_grid_summaries"][
                        "ridge_huber"
                    ]["accepted_true_safe_action_count"],
                    "ridge_stable": result_state["resampling_stability"][
                        "all_active_rows_stable"
                    ],
                    "ridge_qp_safe": result_state["QP"]["ridge_huber"]["gate_pass"],
                    "minimum_L1_qp_safe": result_state["QP"]["minimum_L1"]["gate_pass"],
                }, sort_keys=True), flush=True)
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()

    aggregates = {}
    for method in METHODS:
        pairs = [
            (fresh["true_safe"], fresh[method + "_predicted_safe"])
            for state in state_results for fresh in state["fresh_actions"]
        ]
        summary = _summarize(pairs)
        summary.update({
            "states_with_accepted_safe_action": sum(
                state["off_grid_summaries"][method][
                    "accepted_true_safe_action_count"
                ] >= int(config["oracle_gate"][
                    "minimum_accepted_safe_action_count_per_state"
                ]) for state in state_results
            ),
            "states_with_valid_fresh_safe_QP": sum(
                state["QP"][method]["gate_pass"] for state in state_results
            ),
        })
        aggregates[method] = summary
    stability_gate = bool(all(
        state["resampling_stability"]["all_active_rows_stable"]
        for state in state_results
    ))
    minimum_l1_stability_gate = bool(all(
        state["minimum_L1_resampling_stability"]["all_active_rows_stable"]
        for state in state_results
    ))
    ridge_gate = bool(
        aggregates["ridge_huber"]["false_safe_action_count"] == 0
        and aggregates["ridge_huber"]["states_with_accepted_safe_action"]
        == len(state_results)
        and stability_gate
        and aggregates["ridge_huber"]["states_with_valid_fresh_safe_QP"]
        == len(state_results)
    )
    minimum_l1_gate = bool(
        aggregates["minimum_L1"]["false_safe_action_count"] == 0
        and aggregates["minimum_L1"]["states_with_accepted_safe_action"]
        == len(state_results)
        and aggregates["minimum_L1"]["states_with_valid_fresh_safe_QP"]
        == len(state_results)
    )
    decision = {
        "ridge_huber_oracle_gate_pass": ridge_gate,
        "minimum_L1_comparator_gate_pass": minimum_l1_gate,
        "all_active_rows_resampling_stable": stability_gate,
        "minimum_L1_all_active_rows_resampling_stable": (
            minimum_l1_stability_gate
        ),
        "learned_training_authorized": ridge_gate,
        "learned_training_executed": False,
        "closed_loop_e05_authorized": False,
        "stop_reason": None if ridge_gate else "one_or_more_ridge_huber_oracle_gates_failed",
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
            )
        },
        "split_state_counts": split_counts, "state_results": state_results,
        "method_aggregates": aggregates, "decision": decision,
        "new_cloned_OSC_rollout_count": len(state_results) * len(METHODS),
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = _hash_without(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "method_aggregates": aggregates, "decision": decision,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
