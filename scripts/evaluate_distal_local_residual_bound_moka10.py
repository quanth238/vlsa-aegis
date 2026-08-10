#!/usr/bin/env python3
"""Evaluate LOO nearest-neighbor conservative execution-margin bounds."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.action_conditioned_margin import (
    load_config as load_action_config, load_model, predict_member_margins,
    train_ensemble, value_training_arrays,
)
from main.multilink_ellipsoid.local_residual_bound import (
    OOF_SCHEMA, RESULT_SCHEMA, build_pool, learned_regional_targets,
    load_config, local_feature_matrix, local_lower_values, save_oof_artifact,
    solve_regional_qps, target_values,
)
from main.multilink_ellipsoid.supported_region_aware_mlp import fresh_payload
from main.multilink_ellipsoid.targeted_boundary_expansion import (
    DATASET_SCHEMA, ORACLE_SCHEMA,
)
from main.multilink_ellipsoid.two_step_margin import feature_context, feature_vectors
from scripts.collect_distal_affine_coefficient_moka10 import _chunk_values
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action, _restore_env, _snapshot_env,
)
from scripts.evaluate_distal_supported_region_aware_mlp_moka10 import _manifest
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_region_aware_mlp_moka10 import _exact_safe
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _combine_oracle(
    oracle_states: Sequence[Mapping[str, Any]], fresh: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = copy.deepcopy(list(oracle_states))
    by_index = {int(item["state_index"]): item for item in output}
    for item in fresh["state_results"]:
        by_index[int(item["state_index"])]["fresh_actions"] = copy.deepcopy(
            item["fresh_actions"]
        )
    return output


def _recorded_safety_flags(action: Mapping[str, Any]) -> tuple[Any, bool]:
    """Read both registered off-grid artifact schemas without changing labels."""

    zero_arm = action.get("arms", {}).get("0mm")
    oracle_safe = (
        None if zero_arm is None
        else bool(zero_arm["multi_region_predicted_safe"])
    )
    true_safe = bool(
        action["true_safe"] if "true_safe" in action
        else zero_arm["true_safe"]
    )
    return oracle_safe, true_safe


def _generate_oof(
    *, states: Sequence[Mapping[str, Any]], arrays: Mapping[str, Any],
    action_config: Mapping[str, Any], expected_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train one unchanged action-margin ensemble per held training episode."""

    import numpy as np

    train_cases = sorted({
        str(state["case_id"]) for state in states if state["split"] == "train"
    })
    if len(train_cases) != 12:
        raise ValueError("local-residual training episode count differs")
    state_by_index = {int(state["state_index"]): state for state in states}
    features = []
    residuals = []
    state_indexes = []
    action_indexes = []
    constraint_indexes = []
    case_indexes = []
    fold_audits = []
    base_split = np.asarray(arrays["split"], dtype=object)
    row_state_index = np.asarray(arrays["state_index"], dtype=np.int64)
    for case_index, case_id in enumerate(train_cases):
        held_states = sorted(
            (state for state in states if state["case_id"] == case_id),
            key=lambda state: int(state["state_index"]),
        )
        if len(held_states) != 5 or any(
            state["split"] != "train" for state in held_states
        ):
            raise ValueError("local-residual held episode differs")
        held_indexes = {int(state["state_index"]) for state in held_states}
        fold_split = base_split.copy()
        fold_split[np.isin(row_state_index, list(held_indexes))] = "test"
        fold_arrays = dict(arrays)
        fold_arrays["split"] = fold_split
        models, model_state, training = train_ensemble(
            fold_arrays, action_config
        )
        fold_residual_count = 0
        for state in held_states:
            xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
            exact = np.asarray(
                state["candidate_minimum_distal_margin_m"], dtype=np.float64
            )
            members = predict_member_margins(
                models, model_state, state, xyz
            )
            predicted = np.mean(members, axis=0)
            local = local_feature_matrix(state, xyz)
            features.append(local.reshape(125 * 7, -1))
            residuals.append((predicted - exact).reshape(-1))
            state_indexes.append(np.repeat(int(state["state_index"]), 125 * 7))
            action_indexes.append(np.repeat(np.arange(125), 7))
            constraint_indexes.append(np.tile(np.arange(7), 125))
            case_indexes.append(np.repeat(case_index, 125 * 7))
            fold_residual_count += 125 * 7
        fold_audits.append({
            "fold_index": case_index, "held_case_id": case_id,
            "held_state_indexes": sorted(held_indexes),
            "residual_count": fold_residual_count,
            "training": training,
        })
    output = {
        "features": np.concatenate(features, axis=0),
        "residual_m": np.concatenate(residuals, axis=0),
        "state_index": np.concatenate(state_indexes, axis=0),
        "action_index": np.concatenate(action_indexes, axis=0),
        "constraint_index": np.concatenate(constraint_indexes, axis=0),
        "case_index": np.concatenate(case_indexes, axis=0),
    }
    if len(output["residual_m"]) != int(expected_count):
        raise ValueError("local-residual OOF count differs")
    pool = build_pool(
        output["features"], output["residual_m"], output["constraint_index"]
    )
    audit = {
        "fold_count": len(fold_audits), "folds": fold_audits,
        "residual_count": len(output["residual_m"]),
        "dangerous_residual_maximum_m": float(np.max(output["residual_m"])),
        "dangerous_residual_mean_m": float(np.mean(output["residual_m"])),
        "feature_mean": pool["feature_mean"].tolist(),
        "feature_standard_deviation": pool[
            "feature_standard_deviation"
        ].tolist(),
        "train_case_ids": train_cases,
    }
    return {**output, "pool": pool}, audit


def _evaluate_population(
    *, split: str, states: Sequence[Mapping[str, Any]],
    oracle_by_index: Mapping[int, Mapping[str, Any]], models: Sequence[Any],
    model_state: Mapping[str, Any], pool: Mapping[str, Any],
    config: Mapping[str, Any], ridge_huber: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np

    results = []
    for state in sorted(
        (item for item in states if item["split"] == split),
        key=lambda item: int(item["state_index"]),
    ):
        state_index = int(state["state_index"])
        oracle_state = oracle_by_index[state_index]
        targets, grid = learned_regional_targets(
            models, model_state, state, oracle_state["regions"], pool, config,
            ridge_huber,
        )
        fresh = oracle_state.get("fresh_actions", [])
        if len(fresh) != 96:
            raise ValueError("local-residual off-grid population differs")
        fresh_xyz = np.asarray(
            [item["candidate_xyz"] for item in fresh], dtype=np.float64
        )
        fresh_values = local_lower_values(
            models, model_state, state, fresh_xyz, pool, config["local_bound"]
        )
        exact_margin = np.asarray(
            [item["minimum_distal_margin_m"] for item in fresh],
            dtype=np.float64,
        )
        learned_flags = []
        oracle_flags = []
        true_flags = []
        false_indexes = []
        for fresh_index, action in enumerate(fresh):
            learned = bool(any(
                np.all(target_values(
                    targets[int(region_index)], action["candidate_xyz"]
                ) >= 0.0)
                for region_index in action["containing_region_indexes"]
            ))
            oracle_safe, true_safe = _recorded_safety_flags(action)
            learned_flags.append(learned)
            oracle_flags.append(oracle_safe)
            true_flags.append(true_safe)
            if learned and not true_safe:
                false_indexes.append(fresh_index)
        results.append({
            "state_index": state_index, "case_id": state["case_id"],
            "state_step": int(state["state_step"]), "split": split,
            "oracle_accepted_flags": oracle_flags,
            "learned_accepted_flags": learned_flags,
            "true_safe_flags": true_flags,
            "false_safe_fresh_indexes": false_indexes,
            "accepted_true_safe_action_count": int(sum(
                learned and safe
                for learned, safe in zip(learned_flags, true_flags)
            )),
            "direct_mean_error_m": (
                np.asarray(fresh_values["mean_m"]) - exact_margin
            ).tolist(),
            "local_upper_error_m": np.asarray(
                fresh_values["upper_error_m"]
            ).tolist(),
            "predicted_targets": targets,
            "grid_bound_summary": {
                "mean_minimum_m": float(np.min(grid["mean_m"])),
                "upper_error_minimum_m": float(np.min(grid["upper_error_m"])),
                "upper_error_maximum_m": float(np.max(grid["upper_error_m"])),
                "lower_minimum_m": float(np.min(grid["lower_m"])),
                "neighbor_distance_maximum_rms_z": float(np.max(
                    grid["maximum_neighbor_distance_rms_z"]
                )),
            },
            "QP": {"selected": None}, "fresh_exact_two_step": None,
        })
    return results


def _aggregates(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    errors = np.concatenate([
        np.asarray(item["direct_mean_error_m"], dtype=np.float64).reshape(-1)
        for item in results
    ])
    return {
        "state_count": len(results),
        "off_grid_action_count": sum(
            len(item["learned_accepted_flags"]) for item in results
        ),
        "off_grid_false_safe_action_count": sum(
            len(item["false_safe_fresh_indexes"]) for item in results
        ),
        "safe_support_state_count": sum(
            item["accepted_true_safe_action_count"] > 0 for item in results
        ),
        "accepted_true_safe_action_count": sum(
            int(item["accepted_true_safe_action_count"]) for item in results
        ),
        "direct_mean_RMSE_m": float(np.sqrt(np.mean(errors * errors))),
        "valid_selected_QP_count": sum(
            item["QP"]["selected"] is not None for item in results
        ),
        "fresh_exact_safe_selected_QP_count": sum(
            item["fresh_exact_two_step"] is not None
            and item["fresh_exact_two_step"]["true_safe"] for item in results
        ),
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
    parser.add_argument("--test-selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--action-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--validation-fresh", type=Path, required=True)
    parser.add_argument("--action-result", type=Path, required=True)
    parser.add_argument("--action-validation", type=Path, required=True)
    parser.add_argument("--action-model", type=Path, required=True)
    parser.add_argument("--target-result", type=Path, required=True)
    parser.add_argument("--target-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--oof-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "population": args.population_manifest,
        "test_selected": args.test_selected_manifest,
        "archived": args.archived_root, "geometry": args.geometry_config,
        "exact_box": args.exact_box_config, "action_config": args.action_config,
        "config": args.config, "dataset": args.expanded_dataset,
        "oracle": args.expanded_oracle, "validation_fresh": args.validation_fresh,
        "action_result": args.action_result,
        "action_validation": args.action_validation,
        "action_model": args.action_model, "target_result": args.target_result,
        "target_validation": args.target_validation,
        "oof": args.oof_artifact, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    action_config = load_action_config(paths["action_config"])
    source = config["immutable_source"]
    for path, expected, label in (
        (paths["population"], config["source_population_manifest_sha256"], "population"),
        (paths["test_selected"], config["test_selected_manifest_sha256"], "test-selected"),
        (paths["geometry"], config["geometry_config_file_sha256"], "geometry"),
        (paths["exact_box"], config["exact_box_config_file_sha256"], "exact-box"),
        (paths["action_config"], source["action_conditioned_config_file_sha256"], "action-config"),
        (paths["dataset"], source["expanded_dataset_file_sha256"], "dataset"),
        (paths["oracle"], source["expanded_oracle_file_sha256"], "oracle"),
        (paths["validation_fresh"], source["validation_fresh_file_sha256"], "validation-fresh"),
        (paths["action_result"], source["action_conditioned_result_file_sha256"], "action-result"),
        (paths["action_validation"], source["action_conditioned_validation_file_sha256"], "action-validation"),
        (paths["action_model"], source["action_conditioned_model_file_sha256"], "action-model"),
        (paths["target_result"], source["target_authority_result_file_sha256"], "target-result"),
        (paths["target_validation"], source["target_authority_validation_file_sha256"], "target-validation"),
    ):
        _require(
            _file_sha256(path) == expected,
            "local-residual immutable %s differs" % label,
        )
    dataset = _load(paths["dataset"])
    oracle = _load(paths["oracle"])
    fresh = _load(paths["validation_fresh"])
    action_result = _load(paths["action_result"])
    action_validation = _load(paths["action_validation"])
    target_result = _load(paths["target_result"])
    target_validation = _load(paths["target_validation"])
    _require(
        dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source["expanded_dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256")
        and oracle.get("schema_version") == ORACLE_SCHEMA
        and oracle.get("oracle_payload_sha256")
        == source["expanded_oracle_payload_sha256"]
        == _hash_without(oracle, "oracle_payload_sha256")
        and fresh.get("validation_fresh_payload_sha256")
        == source["validation_fresh_payload_sha256"] == fresh_payload(fresh)
        and action_result.get("result_payload_sha256")
        == source["action_conditioned_result_payload_sha256"]
        == _hash_without(action_result, "result_payload_sha256")
        and action_validation.get("status") == "valid"
        and target_result.get("result_payload_sha256")
        == source["target_authority_result_payload_sha256"]
        == _hash_without(target_result, "result_payload_sha256")
        and target_validation.get("status") == "valid"
        and target_validation.get("target_authority_gate_pass") is True
        and target_validation.get(
            "local_residual_calibration_preregistration_authorized"
        ) is True,
        "local-residual authorization differs",
    )
    states = dataset["state_records"]
    combined_oracle = _combine_oracle(oracle["state_results"], fresh)
    oracle_by_index = {
        int(item["state_index"]): item for item in combined_oracle
    }
    counts = {
        split: sum(item["split"] == split for item in states)
        for split in ("train", "validation", "test")
    }
    _require(
        len(states) == int(source["expected_state_count"])
        and counts == config["split"]["expected_state_counts"]
        and sorted({
            item["case_id"] for item in states if item["split"] == "test"
        }) == config["split"]["test_case_ids"],
        "local-residual split differs",
    )
    arrays = value_training_arrays(states)
    oof, oof_audit = _generate_oof(
        states=states, arrays=arrays, action_config=action_config,
        expected_count=int(config["leave_one_episode_out"][
            "expected_residual_count"
        ]),
    )
    oof_metadata = {
        "schema_version": OOF_SCHEMA,
        "source_commit": args.expected_commit,
        "action_config_file_sha256": action_config["config_file_sha256"],
        "fold_case_ids": oof_audit["train_case_ids"],
        "residual_count": oof_audit["residual_count"],
    }
    oof_artifact = save_oof_artifact(
        paths["oof"], features=oof["features"],
        residual_m=oof["residual_m"], state_index=oof["state_index"],
        action_index=oof["action_index"],
        constraint_index=oof["constraint_index"],
        case_index=oof["case_index"], metadata=oof_metadata,
    )
    pool = oof["pool"]
    models, model_state = load_model(paths["action_model"])
    validation_results = _evaluate_population(
        split="validation", states=states, oracle_by_index=oracle_by_index,
        models=models, model_state=model_state, pool=pool, config=config,
        ridge_huber=action_config["ridge_huber"],
    )
    test_results = _evaluate_population(
        split="test", states=states, oracle_by_index=oracle_by_index,
        models=models, model_state=model_state, pool=pool, config=config,
        ridge_huber=action_config["ridge_huber"],
    )
    validation_aggregates = _aggregates(validation_results)
    test_aggregates = _aggregates(test_results)
    gates = config["learned_gate"]
    preliminary = bool(
        test_aggregates["off_grid_false_safe_action_count"]
        == int(gates["test_off_grid_false_safe_action_count"])
        and test_aggregates["safe_support_state_count"]
        == int(gates["required_test_state_safe_support_count"])
    )

    if preliminary:
        test_rows = _manifest(
            paths["test_selected"], config["test_selected_manifest_sha256"]
        )
        row_by_case = {str(item["case_id"]): item for item in test_rows}
        population = {
            item["case_id"]: item for item in read_jsonl(paths["population"])
        }
        placeholder_row = row_by_case["vlsa-t1-goal-ii-t0-e05"]
        placeholder_path = paths["archived"] / placeholder_row[
            "archived_relative_path"
        ]
        placeholder = _load(placeholder_path)
        _require(
            _file_sha256(placeholder_path) == source["archived_e05_file_sha256"]
            and placeholder.get("result_payload_sha256")
            == source["archived_e05_payload_sha256"],
            "local-residual E05 geometry source differs",
        )
        geometry_config = load_shadow_config(paths["geometry"])
        exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
        runtime = _runtime_imports(include_aegis=False)
        result_by_index = {
            int(item["state_index"]): item for item in test_results
        }
        maximum_motion = float(config["exact_verification"][
            "maximum_per_step_obstacle_l1_displacement_m"
        ])
        for case_id in config["split"]["test_case_ids"]:
            row = row_by_case[case_id]
            archived_path = paths["archived"] / row["archived_relative_path"]
            archived = _load(archived_path)
            _require(
                _file_sha256(archived_path) == row["archived_file_sha256"]
                and archived.get("result_payload_sha256")
                == row["archived_payload_sha256"],
                "local-residual test archive differs: %s" % case_id,
            )
            env = probe_env = None
            try:
                env, probe_env, _, _, setup = _build_pair(
                    runtime, population[case_id]
                )
                geometry, exact_boxes = _geometry(
                    geometry_config=geometry_config,
                    exact_box_config=exact_box_config, archived=placeholder,
                    env=env, obstacle_name=setup["obstacle_name"],
                )
                probe = SubstepEightConstraintProbe(
                    probe_env, geometry,
                    active_obstacle_name=setup["obstacle_name"],
                    quadratic_tolerance=1.0e-6,
                    contact_distance_threshold_m=0.0,
                    obstacle_primitive_union=exact_boxes,
                )
                case_states = sorted((
                    state for state in states if state["split"] == "test"
                    and state["case_id"] == case_id
                ), key=lambda state: int(state["state_step"]))
                needed = {int(state["state_step"]) for state in case_states}
                snapshots = {}
                actions = archived["actions"]
                for step in range(max(needed) + 1):
                    if step in needed:
                        snapshots[step] = _snapshot_env(env)
                    if step < max(needed):
                        env.step(_canonical_action(actions[step], step).tolist())
                for state in case_states:
                    state_index = int(state["state_index"])
                    step = int(state["state_step"])
                    item = result_by_index[state_index]
                    _restore_env(env, snapshots[step])
                    first = _canonical_action(actions[step], step)
                    second = _canonical_action(actions[step + 1], step + 1)
                    context = feature_context(env, probe)
                    _, live_features = feature_vectors(
                        context, first[:3], first[:3], second[:3]
                    )
                    feature_error = float(np.max(np.abs(
                        live_features - np.asarray(
                            state["pair_state_feature_vectors"], dtype=np.float64
                        )
                    )))
                    _require(
                        feature_error <= 1.0e-8,
                        "local-residual test feature receipt differs",
                    )
                    oracle_state = oracle_by_index[state_index]
                    targets, _ = learned_regional_targets(
                        models, model_state, state, oracle_state["regions"],
                        pool, config, action_config["ridge_huber"],
                    )
                    qp = solve_regional_qps(
                        first[:3], oracle_state["regions"], targets,
                        config["projection"],
                    )
                    chosen = qp["selected"]
                    exact = None
                    if chosen is not None:
                        candidate = first.copy()
                        candidate[:3] = np.asarray(
                            chosen["candidate_xyz"], dtype=np.float64
                        )
                        summary, ee = _chunk_values(
                            probe.rollout_chunk(env, [candidate, second])
                        )
                        exact = {
                            "minimum_distal_margin_m": summary[
                                "minimum_substep_clearance_m"
                            ],
                            "minimum_released_AEGIS_EE_margin_m": float(ee),
                            "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
                            "raw_protected_contact_count": int(
                                summary["raw_protected_contact_count"]
                            ),
                            "maximum_within_step_obstacle_l1_displacement_m": float(
                                summary[
                                    "maximum_within_step_obstacle_l1_displacement_m"
                                ]
                            ),
                            "next_state_sha256": summary["next_state_sha256"],
                        }
                        exact["true_safe"] = _exact_safe(exact, maximum_motion)
                    item["feature_max_abs_error"] = feature_error
                    item["QP"] = qp
                    item["fresh_exact_two_step"] = exact
            finally:
                if probe_env is not None:
                    probe_env.close()
                if env is not None:
                    env.close()
    test_aggregates = _aggregates(test_results)
    learned_pass = bool(
        preliminary
        and test_aggregates["valid_selected_QP_count"]
        == int(gates["required_valid_selected_QP_count"])
        and test_aggregates["fresh_exact_safe_selected_QP_count"]
        == int(gates["required_fresh_exact_safe_selected_QP_count"])
    )
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_inputs": {
            "expanded_dataset_file_sha256": _file_sha256(paths["dataset"]),
            "expanded_oracle_file_sha256": _file_sha256(paths["oracle"]),
            "action_model_file_sha256": _file_sha256(paths["action_model"]),
            "target_authority_result_file_sha256": _file_sha256(paths["target_result"]),
        },
        "oof_artifact": oof_artifact, "oof_audit": oof_audit,
        "local_pool": {
            "feature_mean": pool["feature_mean"].tolist(),
            "feature_standard_deviation": pool[
                "feature_standard_deviation"
            ].tolist(),
        },
        "validation_aggregates": validation_aggregates,
        "validation_state_results": validation_results,
        "test_aggregates": test_aggregates,
        "test_state_results": test_results,
        "decision": {
            "off_grid_preliminary_gate_pass": preliminary,
            "learned_gate_pass": learned_pass,
            "receding_closed_loop_E05_preregistration_authorized": learned_pass,
            "closed_loop_e05_executed": False,
            "stop_reason": (
                None if learned_pass else "local_residual_bound_gate_failed"
            ),
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["result_payload_sha256"] = _hash_without(
        output, "result_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps({
        "decision": output["decision"],
        "validation_aggregates": validation_aggregates,
        "test_aggregates": test_aggregates, "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
