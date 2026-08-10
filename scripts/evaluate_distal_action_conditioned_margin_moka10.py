#!/usr/bin/env python3
"""Evaluate action-conditioned two-step margins and regional QPs on H100."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.action_conditioned_margin import (
    CURRENT_CLEARANCE_INDEX, RESULT_SCHEMA, calibrate_validation,
    guarded_margin_values, jaccard,
    learned_regional_targets, load_config, save_model, solve_regional_qps,
    target_values, train_ensemble, value_training_arrays,
)
from main.multilink_ellipsoid.supported_region_aware_mlp import (
    VALIDATION_FRESH_SCHEMA, fresh_payload,
)
from main.multilink_ellipsoid.targeted_boundary_expansion import (
    DATASET_SCHEMA, ORACLE_SCHEMA,
)
from main.multilink_ellipsoid.two_step_margin import feature_context, feature_vectors
from scripts.collect_distal_affine_coefficient_moka10 import _chunk_values
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action, _restore_env, _snapshot_env,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_region_aware_mlp_moka10 import _exact_safe
from scripts.evaluate_distal_supported_region_aware_mlp_moka10 import _manifest
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _aggregates(test_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    learned = [
        flag for item in test_results for flag in item["learned_accepted_flags"]
    ]
    oracle = [
        flag for item in test_results for flag in item["oracle_accepted_flags"]
    ]
    shifts = [
        float(item["selected_action_shift_from_oracle_l2"])
        for item in test_results
        if item["selected_action_shift_from_oracle_l2"] is not None
    ]
    prediction_error = np.concatenate([
        np.asarray(item["direct_value_error_m"], dtype=np.float64).reshape(-1)
        for item in test_results
    ])
    baseline_error = np.concatenate([
        np.asarray(item["current_clearance_baseline_error_m"], dtype=np.float64).reshape(-1)
        for item in test_results
    ])
    return {
        "state_count": len(test_results),
        "off_grid_action_count": len(learned),
        "off_grid_false_safe_action_count": sum(
            len(item["false_safe_fresh_indexes"]) for item in test_results
        ),
        "global_accepted_set_jaccard_to_oracle": jaccard(oracle, learned),
        "minimum_state_accepted_set_jaccard_to_oracle": min(
            float(item["accepted_set_jaccard_to_oracle"])
            for item in test_results
        ),
        "safe_support_state_count": sum(
            int(item["accepted_true_safe_action_count"] > 0)
            for item in test_results
        ),
        "direct_value_RMSE_m": float(np.sqrt(np.mean(prediction_error ** 2))),
        "current_clearance_baseline_RMSE_m": float(
            np.sqrt(np.mean(baseline_error ** 2))
        ),
        "valid_selected_QP_count": sum(
            item["QP"]["selected"] is not None for item in test_results
        ),
        "fresh_exact_safe_selected_QP_count": sum(
            item["fresh_exact_two_step"] is not None
            and item["fresh_exact_two_step"]["true_safe"]
            for item in test_results
        ),
        "released_AEGIS_EE_compatible_selected_QP_count": sum(
            item["fresh_exact_two_step"] is not None
            and item["fresh_exact_two_step"][
                "minimum_released_AEGIS_EE_margin_m"
            ] >= 0.0
            for item in test_results
        ),
        "selected_action_shift_from_oracle_l2_p95": (
            None if not shifts else float(np.quantile(shifts, 0.95))
        ),
        "selected_action_shift_from_oracle_l2_maximum": (
            None if not shifts else float(np.max(shifts))
        ),
    }


def _combine_oracle(
    oracle_states: Sequence[Mapping[str, Any]],
    validation_fresh: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = copy.deepcopy(list(oracle_states))
    by_index = {int(item["state_index"]): item for item in output}
    for item in validation_fresh["state_results"]:
        by_index[int(item["state_index"])]["fresh_actions"] = copy.deepcopy(
            item["fresh_actions"]
        )
    return output


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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--support-result", type=Path, required=True)
    parser.add_argument("--support-validation", type=Path, required=True)
    parser.add_argument("--supported-mlp-result", type=Path, required=True)
    parser.add_argument("--supported-mlp-validation", type=Path, required=True)
    parser.add_argument("--validation-fresh", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "population": args.population_manifest,
        "test_selected": args.test_selected_manifest,
        "archived": args.archived_root, "geometry": args.geometry_config,
        "exact_box": args.exact_box_config, "config": args.config,
        "dataset": args.expanded_dataset, "oracle": args.expanded_oracle,
        "support": args.support_result,
        "support_validation": args.support_validation,
        "supported": args.supported_mlp_result,
        "supported_validation": args.supported_mlp_validation,
        "validation_fresh": args.validation_fresh,
        "model": args.model, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, expected, label in (
        (paths["population"], config["source_population_manifest_sha256"], "population"),
        (paths["test_selected"], config["test_selected_manifest_sha256"], "test-selected"),
        (paths["geometry"], config["geometry_config_file_sha256"], "geometry"),
        (paths["exact_box"], config["exact_box_config_file_sha256"], "exact-box"),
        (paths["dataset"], source["expanded_dataset_file_sha256"], "dataset"),
        (paths["oracle"], source["expanded_oracle_file_sha256"], "oracle"),
        (paths["support"], source["support_result_file_sha256"], "support"),
        (paths["support_validation"], source["support_validation_file_sha256"], "support-validation"),
        (paths["supported"], source["supported_mlp_result_file_sha256"], "supported-MLP-result"),
        (paths["supported_validation"], source["supported_mlp_validation_file_sha256"], "supported-MLP-validation"),
        (paths["validation_fresh"], source["validation_fresh_file_sha256"], "validation-fresh"),
    ):
        _require(
            _file_sha256(path) == expected,
            "action-conditioned immutable %s differs" % label,
        )
    dataset = _load(paths["dataset"])
    oracle = _load(paths["oracle"])
    support = _load(paths["support"])
    support_validation = _load(paths["support_validation"])
    supported = _load(paths["supported"])
    supported_validation = _load(paths["supported_validation"])
    validation_fresh = _load(paths["validation_fresh"])
    _require(
        dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source["expanded_dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256")
        and oracle.get("schema_version") == ORACLE_SCHEMA
        and oracle.get("oracle_payload_sha256")
        == source["expanded_oracle_payload_sha256"]
        == _hash_without(oracle, "oracle_payload_sha256")
        and support.get("result_payload_sha256")
        == source["support_result_payload_sha256"]
        == _hash_without(support, "result_payload_sha256")
        and support_validation.get("status") == "valid"
        and support.get("aggregates", {}).get("supported_test_state_count") == 15
        and support.get("aggregates", {}).get("smooth_test_state_count") == 15
        and supported.get("result_payload_sha256")
        == source["supported_mlp_result_payload_sha256"]
        == _hash_without(supported, "result_payload_sha256")
        and supported.get("decision", {}).get("learned_gate_pass") is False
        and supported.get("decision", {}).get(
            "action_conditioned_model_preregistration_authorized"
        ) is True
        and supported_validation.get("status") == "valid"
        and supported_validation.get(
            "action_conditioned_model_preregistration_authorized"
        ) is True
        and validation_fresh.get("schema_version") == VALIDATION_FRESH_SCHEMA
        and validation_fresh.get("validation_fresh_payload_sha256")
        == source["validation_fresh_payload_sha256"]
        == fresh_payload(validation_fresh),
        "action-conditioned authorization differs",
    )
    states = dataset["state_records"]
    combined_oracle = _combine_oracle(oracle["state_results"], validation_fresh)
    expected = int(source["expected_state_count"])
    _require(
        len(states) == len(combined_oracle) == expected,
        "action-conditioned state count differs",
    )
    counts = {
        split: sum(item["split"] == split for item in states)
        for split in ("train", "validation", "test")
    }
    _require(
        counts == config["split"]["expected_state_counts"]
        and sorted({
            item["case_id"] for item in states if item["split"] == "test"
        }) == config["split"]["test_case_ids"]
        and all(
            len({
                int(item["state_index"]): item for item in combined_oracle
            }[int(state["state_index"])].get("fresh_actions", [])) == 96
            for state in states if state["split"] in ("validation", "test")
        ),
        "action-conditioned grouped population differs",
    )
    arrays = value_training_arrays(states)
    models, model_state, training_audit = train_ensemble(arrays, config)
    calibration = calibrate_validation(
        models, model_state, states, combined_oracle, config["uncertainty"]
    )
    model_artifact = save_model(paths["model"], model_state)

    state_by_index = {int(item["state_index"]): item for item in states}
    oracle_by_index = {
        int(item["state_index"]): item for item in combined_oracle
    }
    test_results = []
    learned_global = []
    oracle_global = []
    for state_index in sorted(
        index for index, state in state_by_index.items()
        if state["split"] == "test"
    ):
        state = state_by_index[state_index]
        oracle_state = oracle_by_index[state_index]
        targets, grid = learned_regional_targets(
            models, model_state, state, oracle_state["regions"], config
        )
        fresh = oracle_state["fresh_actions"]
        fresh_xyz = np.asarray(
            [item["candidate_xyz"] for item in fresh], dtype=np.float64
        )
        exact_margin = np.asarray(
            [item["minimum_distal_margin_m"] for item in fresh],
            dtype=np.float64,
        )
        direct = guarded_margin_values(
            models, model_state, state, fresh_xyz, config["uncertainty"],
            calibrated=False,
        )
        current = np.asarray(
            state["pair_state_feature_vectors"], dtype=np.float64
        )[:, CURRENT_CLEARANCE_INDEX]
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
            oracle_safe = bool(
                action["arms"]["0mm"]["multi_region_predicted_safe"]
            )
            true_safe = bool(action["arms"]["0mm"]["true_safe"])
            learned_flags.append(learned)
            oracle_flags.append(oracle_safe)
            true_flags.append(true_safe)
            if learned and not true_safe:
                false_indexes.append(fresh_index)
        learned_global.extend(learned_flags)
        oracle_global.extend(oracle_flags)
        test_results.append({
            "state_index": state_index, "case_id": state["case_id"],
            "state_step": int(state["state_step"]),
            "oracle_accepted_flags": oracle_flags,
            "learned_accepted_flags": learned_flags,
            "true_safe_flags": true_flags,
            "false_safe_fresh_indexes": false_indexes,
            "accepted_true_safe_action_count": int(sum(
                learned and safe
                for learned, safe in zip(learned_flags, true_flags)
            )),
            "accepted_set_jaccard_to_oracle": jaccard(
                oracle_flags, learned_flags
            ),
            "direct_value_error_m": (
                np.asarray(direct["mean_m"]) - exact_margin
            ).tolist(),
            "current_clearance_baseline_error_m": (
                current[None, :] - exact_margin
            ).tolist(),
            "predicted_targets": targets,
            "grid_prediction_summary": {
                "mean_minimum_m": float(np.min(grid["grid_mean_m"])),
                "guard_maximum_m": float(np.max(grid["grid_guard_m"])),
                "lower_minimum_m": float(np.min(grid["grid_lower_m"])),
            },
            "QP": {"selected": None}, "fresh_exact_two_step": None,
            "selected_action_shift_from_oracle_l2": None,
        })
    preliminary_aggregates = _aggregates(test_results)
    gates = config["learned_gate"]
    preliminary = bool(
        preliminary_aggregates["off_grid_false_safe_action_count"]
        == int(gates["test_off_grid_false_safe_action_count"])
        and preliminary_aggregates["global_accepted_set_jaccard_to_oracle"]
        >= float(gates["minimum_global_accepted_set_jaccard_to_oracle"])
        and preliminary_aggregates["minimum_state_accepted_set_jaccard_to_oracle"]
        >= float(gates["minimum_state_accepted_set_jaccard_to_oracle"])
        and preliminary_aggregates["safe_support_state_count"]
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
            "action-conditioned E05 geometry source differs",
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
                "action-conditioned test archive differs: %s" % case_id,
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
                    state for state in states
                    if state["split"] == "test"
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
                            state["pair_state_feature_vectors"],
                            dtype=np.float64,
                        )
                    )))
                    _require(
                        feature_error <= 1.0e-8,
                        "action-conditioned test feature receipt differs",
                    )
                    oracle_state = oracle_by_index[state_index]
                    predicted, _ = learned_regional_targets(
                        models, model_state, state, oracle_state["regions"],
                        config,
                    )
                    qp = solve_regional_qps(
                        first[:3], oracle_state["regions"], predicted,
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
                        oracle_selected = oracle_state["QP"]["0mm"][
                            "selected_solution"
                        ]
                        _require(
                            oracle_selected is not None,
                            "action-conditioned oracle QP support differs",
                        )
                        item["selected_action_shift_from_oracle_l2"] = float(
                            np.linalg.norm(
                                candidate[:3] - np.asarray(
                                    oracle_selected["candidate_xyz"],
                                    dtype=np.float64,
                                )
                            )
                        )
                    item["feature_max_abs_error"] = feature_error
                    item["QP"] = qp
                    item["fresh_exact_two_step"] = exact
            finally:
                if probe_env is not None:
                    probe_env.close()
                if env is not None:
                    env.close()

    aggregates = _aggregates(test_results)
    p95 = aggregates["selected_action_shift_from_oracle_l2_p95"]
    maximum = aggregates["selected_action_shift_from_oracle_l2_maximum"]
    learned_pass = bool(
        preliminary
        and aggregates["valid_selected_QP_count"]
        == int(gates["required_valid_selected_QP_count"])
        and aggregates["fresh_exact_safe_selected_QP_count"]
        == int(gates["required_fresh_exact_safe_selected_QP_count"])
        and aggregates["released_AEGIS_EE_compatible_selected_QP_count"]
        == int(gates["required_released_AEGIS_EE_compatible_selected_QP_count"])
        and p95 is not None
        and p95 <= float(gates["selected_action_shift_from_oracle_l2_p95_maximum"])
        and maximum is not None
        and maximum <= float(gates["selected_action_shift_from_oracle_l2_maximum"])
    )
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_inputs": {
            "expanded_dataset_file_sha256": _file_sha256(paths["dataset"]),
            "expanded_oracle_file_sha256": _file_sha256(paths["oracle"]),
            "support_result_file_sha256": _file_sha256(paths["support"]),
            "supported_mlp_result_file_sha256": _file_sha256(paths["supported"]),
            "validation_fresh_file_sha256": _file_sha256(paths["validation_fresh"]),
        },
        "training": training_audit, "calibration": calibration,
        "model_artifact": model_artifact,
        "test_aggregates": aggregates, "test_state_results": test_results,
        "decision": {
            "off_grid_preliminary_gate_pass": preliminary,
            "learned_gate_pass": learned_pass,
            "receding_closed_loop_E05_preregistration_authorized": learned_pass,
            "plain_action_conditioned_MLP_rejected": not learned_pass,
            "closed_loop_e05_executed": False,
            "stop_reason": (
                None if learned_pass
                else "supported_action_conditioned_margin_MLP_gate_failed"
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
        "test_aggregates": output["test_aggregates"],
        "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
