#!/usr/bin/env python3
"""Retrain the unchanged regional MLP after the 15/15 support gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.region_aware_mlp import (
    guarded_targets, jaccard, load_config as load_reference_config,
    regional_training_arrays, save_model, solve_regional_qps, target_values,
    train_ensemble,
)
from main.multilink_ellipsoid.supported_region_aware_mlp import (
    RESULT_SCHEMA, VALIDATION_FRESH_SCHEMA, fresh_payload, load_config,
    require_unchanged_model_sections,
)
from main.multilink_ellipsoid.targeted_boundary_expansion import (
    DATASET_SCHEMA, ORACLE_SCHEMA,
)
from main.multilink_ellipsoid.two_step_margin import (
    feature_context, feature_vectors,
)
from scripts.collect_distal_affine_coefficient_moka10 import _chunk_values
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action, _restore_env, _snapshot_env,
)
from scripts.evaluate_distal_affine_oracle_comparison_moka10 import _state_seed
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_region_aware_mlp_moka10 import _calibrate, _exact_safe
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _manifest(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    _require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        "supported region-aware manifest differs",
    )
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def _collect_validation_fresh(
    *, repo_root: Path, population_path: Path, validation_rows: Sequence[Mapping[str, Any]],
    archived_root: Path, geometry_path: Path, exact_box_path: Path,
    states: Sequence[Mapping[str, Any]], oracle_states: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any], expected_commit: str,
) -> dict[str, Any]:
    """Measure deterministic off-grid calibration actions for E30 only."""

    import numpy as np
    from main.evaluate_safelibero_aegis import _runtime_imports, read_jsonl
    from main.multilink_ellipsoid.multi_region_affine_oracle import (
        containing_region_indexes,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import load_shadow_config

    row_by_case = {str(item["case_id"]): item for item in validation_rows}
    target_case = config["split"]["new_validation_case_id"]
    _require(
        target_case in row_by_case and "vlsa-t1-goal-ii-t0-e00" in row_by_case,
        "supported region-aware validation manifest population differs",
    )
    target_row = row_by_case[target_case]
    placeholder_row = row_by_case["vlsa-t1-goal-ii-t0-e00"]
    target_path = archived_root / target_row["archived_relative_path"]
    placeholder_path = archived_root / placeholder_row["archived_relative_path"]
    archived = _load(target_path)
    placeholder = _load(placeholder_path)
    _require(
        _file_sha256(target_path) == target_row["archived_file_sha256"]
        and archived.get("result_payload_sha256")
        == target_row["archived_payload_sha256"]
        and _file_sha256(placeholder_path)
        == placeholder_row["archived_file_sha256"]
        and placeholder.get("result_payload_sha256")
        == placeholder_row["archived_payload_sha256"],
        "supported region-aware validation archive differs",
    )
    population = {item["case_id"]: item for item in read_jsonl(population_path)}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(geometry_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_path)
    selected_states = sorted(
        (item for item in states if item["case_id"] == target_case),
        key=lambda item: int(item["state_step"]),
    )
    expected_indexes = config["split"]["new_validation_state_indexes"]
    _require(
        [int(item["state_index"]) for item in selected_states]
        == expected_indexes
        and all(item["split"] == "validation" for item in selected_states),
        "supported region-aware validation state identities differ",
    )
    oracle_by_index = {int(item["state_index"]): item for item in oracle_states}
    env = probe_env = None
    output_states = []
    try:
        env, probe_env, _, _, setup = _build_pair(runtime, population[target_case])
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config, exact_box_config=exact_box_config,
            archived=placeholder, env=env, obstacle_name=setup["obstacle_name"],
        )
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        needed = {int(item["state_step"]) for item in selected_states}
        snapshots = {}
        actions = archived["actions"]
        for step in range(max(needed) + 1):
            if step in needed:
                snapshots[step] = _snapshot_env(env)
            if step < max(needed):
                env.step(_canonical_action(actions[step], step).tolist())
        count = int(config["validation_fresh_actions"]["count_per_state"])
        global_seed = int(config["validation_fresh_actions"]["seed"])
        tolerance = float(config["partition"]["inclusive_membership_tolerance"])
        maximum_motion = float(config["exact_verification"][
            "maximum_per_step_obstacle_l1_displacement_m"
        ])
        for state in selected_states:
            state_index = int(state["state_index"])
            step = int(state["state_step"])
            _restore_env(env, snapshots[step])
            first = _canonical_action(actions[step], step)
            second = _canonical_action(actions[step + 1], step + 1)
            context = feature_context(env, probe)
            _, live_features = feature_vectors(
                context, first[:3], first[:3], second[:3],
            )
            feature_error = float(np.max(np.abs(
                live_features - np.asarray(
                    state["pair_state_feature_vectors"], dtype=np.float64,
                )
            )))
            _require(
                feature_error <= 1.0e-8
                and np.max(np.abs(
                    first - np.asarray(state["nominal_first_action"])
                )) <= 1.0e-12
                and np.max(np.abs(
                    second - np.asarray(state["nominal_second_action"])
                )) <= 1.0e-12,
                "supported region-aware validation state receipt differs",
            )
            lower = np.asarray(state["action_lower"], dtype=np.float64)
            upper = np.asarray(state["action_upper"], dtype=np.float64)
            seed = _state_seed(global_seed, target_case, step)
            rng = np.random.RandomState(seed)
            fresh_xyz = rng.uniform(lower, upper, size=(count, 3))
            regions = oracle_by_index[state_index]["regions"]
            records = []
            for fresh_index, xyz in enumerate(fresh_xyz):
                action = first.copy()
                action[:3] = xyz
                summary, ee = _chunk_values(
                    probe.rollout_chunk(env, [action, second])
                )
                margins = np.asarray(
                    summary["minimum_substep_clearance_m"], dtype=np.float64,
                )
                true_safe = bool(
                    np.all(margins >= 0.0) and summary["D_sim_raw_safe"]
                    and int(summary["raw_protected_contact_count"]) == 0
                    and float(summary[
                        "maximum_within_step_obstacle_l1_displacement_m"
                    ]) <= maximum_motion
                )
                records.append({
                    "fresh_index": fresh_index, "candidate_xyz": xyz.tolist(),
                    "minimum_distal_margin_m": margins.tolist(),
                    "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
                    "raw_protected_contact_count": int(
                        summary["raw_protected_contact_count"]
                    ),
                    "maximum_within_step_obstacle_l1_displacement_m": float(
                        summary["maximum_within_step_obstacle_l1_displacement_m"]
                    ),
                    "minimum_released_AEGIS_EE_margin_m": float(ee),
                    "true_safe": true_safe,
                    "containing_region_indexes": containing_region_indexes(
                        xyz, regions, tolerance,
                    ),
                })
            output_states.append({
                "state_index": state_index, "case_id": target_case,
                "split": "validation", "state_step": step,
                "feature_max_abs_error": feature_error, "fresh_seed": seed,
                "fresh_actions": records,
                "true_safe_fresh_action_count": sum(
                    item["true_safe"] for item in records
                ),
            })
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()
    output = {
        "schema_version": VALIDATION_FRESH_SCHEMA,
        "source": _git_identity(repo_root, expected_commit),
        "test_episode_features_or_labels_used": False,
        "state_results": output_states,
        "summary": {
            "case_id": target_case, "state_count": len(output_states),
            "fresh_action_count": sum(
                len(item["fresh_actions"]) for item in output_states
            ),
            "purpose": "validation_only_one_sided_calibration",
        },
    }
    output["validation_fresh_payload_sha256"] = fresh_payload(output)
    return output


def _test_aggregates(
    test_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    learned = [
        flag for item in test_results for flag in item["learned_accepted_flags"]
    ]
    oracle = [
        flag for item in test_results for flag in item["oracle_accepted_flags"]
    ]
    false_safe = sum(
        len(item["false_safe_fresh_indexes"]) for item in test_results
    )
    state_jaccards = [
        float(item["accepted_set_jaccard_to_oracle"]) for item in test_results
    ]
    shifts = [
        float(item["selected_action_shift_from_oracle_l2"])
        for item in test_results
        if item["selected_action_shift_from_oracle_l2"] is not None
    ]
    return {
        "state_count": len(test_results),
        "off_grid_action_count": len(learned),
        "off_grid_false_safe_action_count": false_safe,
        "global_accepted_set_jaccard_to_oracle": jaccard(oracle, learned),
        "minimum_state_accepted_set_jaccard_to_oracle": min(state_jaccards),
        "safe_support_state_count": sum(
            int(item["accepted_true_safe_action_count"] > 0)
            for item in test_results
        ),
        "valid_selected_QP_count": sum(item["QP"]["selected"] is not None for item in test_results),
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
    parser.add_argument("--validation-selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--support-result", type=Path, required=True)
    parser.add_argument("--support-validation", type=Path, required=True)
    parser.add_argument("--decision-stability-result", type=Path, required=True)
    parser.add_argument("--decision-stability-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--validation-fresh", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "population": args.population_manifest,
        "test_selected": args.test_selected_manifest,
        "validation_selected": args.validation_selected_manifest,
        "archived": args.archived_root, "geometry": args.geometry_config,
        "exact_box": args.exact_box_config,
        "reference_config": args.reference_config, "config": args.config,
        "dataset": args.expanded_dataset, "oracle": args.expanded_oracle,
        "support": args.support_result,
        "support_validation": args.support_validation,
        "decision": args.decision_stability_result,
        "decision_validation": args.decision_stability_validation,
        "validation_fresh": args.validation_fresh,
        "model": args.model, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, expected, label in (
        (paths["population"], config["source_population_manifest_sha256"], "population"),
        (paths["test_selected"], config["test_selected_manifest_sha256"], "test-selected"),
        (paths["validation_selected"], config["validation_selected_manifest_sha256"], "validation-selected"),
        (paths["geometry"], config["geometry_config_file_sha256"], "geometry"),
        (paths["exact_box"], config["exact_box_config_file_sha256"], "exact-box"),
        (paths["reference_config"], source["reference_region_aware_config_file_sha256"], "reference-config"),
        (paths["dataset"], source["expanded_dataset_file_sha256"], "dataset"),
        (paths["oracle"], source["expanded_oracle_file_sha256"], "oracle"),
        (paths["support"], source["support_result_file_sha256"], "support"),
        (paths["support_validation"], source["support_validation_file_sha256"], "support-validation"),
        (paths["decision"], source["decision_stability_result_file_sha256"], "decision-stability"),
        (paths["decision_validation"], source["decision_stability_validation_file_sha256"], "decision-validation"),
    ):
        _require(
            _file_sha256(path) == expected,
            "supported region-aware immutable %s differs" % label,
        )
    reference = load_reference_config(paths["reference_config"])
    require_unchanged_model_sections(config, reference)
    dataset = _load(paths["dataset"])
    oracle = _load(paths["oracle"])
    support = _load(paths["support"])
    support_validation = _load(paths["support_validation"])
    decision = _load(paths["decision"])
    decision_validation = _load(paths["decision_validation"])
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
        and support.get("aggregates", {}).get(
            "current_region_aware_MLP_retraining_preregistration_authorized"
        ) is True
        and decision.get("result_payload_sha256")
        == source["decision_stability_result_payload_sha256"]
        == _hash_without(decision, "result_payload_sha256")
        and decision_validation.get("status") == "valid"
        and decision_validation.get("mlp_training_authorized") is True,
        "supported region-aware authorization differs",
    )
    states = dataset["state_records"]
    oracle_states = oracle["state_results"]
    expected = int(source["expected_state_count"])
    _require(
        len(states) == len(oracle_states) == expected,
        "supported region-aware state count differs",
    )
    counts = {
        name: sum(item["split"] == name for item in states)
        for name in ("train", "validation", "test")
    }
    _require(
        counts == config["split"]["expected_state_counts"]
        and sorted({
            item["case_id"] for item in states if item["split"] == "test"
        }) == config["split"]["test_case_ids"],
        "supported region-aware split differs",
    )
    validation_rows = _manifest(
        paths["validation_selected"],
        config["validation_selected_manifest_sha256"],
    )
    fresh = _collect_validation_fresh(
        repo_root=paths["repo"], population_path=paths["population"],
        validation_rows=validation_rows, archived_root=paths["archived"],
        geometry_path=paths["geometry"], exact_box_path=paths["exact_box"],
        states=states, oracle_states=oracle_states, config=config,
        expected_commit=args.expected_commit,
    )
    _atomic_write(paths["validation_fresh"], fresh)
    combined_oracle_states = copy.deepcopy(oracle_states)
    combined_by_index = {
        int(item["state_index"]): item for item in combined_oracle_states
    }
    for item in fresh["state_results"]:
        combined_by_index[int(item["state_index"])]["fresh_actions"] = copy.deepcopy(
            item["fresh_actions"]
        )
    _require(
        all(
            len(combined_by_index[int(state["state_index"])].get("fresh_actions", []))
            == int(source["off_grid_actions_per_state"])
            for state in states if state["split"] in ("validation", "test")
        ),
        "supported region-aware validation/test off-grid coverage differs",
    )
    arrays = regional_training_arrays(states, combined_oracle_states)
    models, model_state, training_audit = train_ensemble(arrays, config)
    calibration = _calibrate(
        models=models, model_state=model_state, arrays=arrays,
        states=states, multi_states=combined_oracle_states,
        off_grid_states=combined_oracle_states, config=config,
    )
    model_state.pop("predictions_m", None)
    model_artifact = save_model(paths["model"], model_state)

    state_by_index = {int(item["state_index"]): item for item in states}
    oracle_by_index = {
        int(item["state_index"]): item for item in combined_oracle_states
    }
    test_results = []
    false_safe_total = 0
    learned_global = []
    oracle_global = []
    minimum_state_jaccard = 1.0
    support_count = 0
    for state_index in sorted(
        index for index, state in state_by_index.items()
        if state["split"] == "test"
    ):
        state = state_by_index[state_index]
        oracle_state = oracle_by_index[state_index]
        targets = guarded_targets(
            models, model_state, state["pair_state_feature_vectors"],
            oracle_state["regions"], config["uncertainty"],
        )
        learned_flags = []
        oracle_flags = []
        true_flags = []
        false_indexes = []
        for fresh_index, action in enumerate(oracle_state["fresh_actions"]):
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
        state_jaccard = jaccard(oracle_flags, learned_flags)
        accepted_true = sum(
            learned and true
            for learned, true in zip(learned_flags, true_flags)
        )
        false_safe_total += len(false_indexes)
        support_count += int(accepted_true > 0)
        minimum_state_jaccard = min(minimum_state_jaccard, state_jaccard)
        learned_global.extend(learned_flags)
        oracle_global.extend(oracle_flags)
        test_results.append({
            "state_index": state_index, "case_id": state["case_id"],
            "state_step": int(state["state_step"]),
            "oracle_accepted_flags": oracle_flags,
            "learned_accepted_flags": learned_flags,
            "true_safe_flags": true_flags,
            "false_safe_fresh_indexes": false_indexes,
            "accepted_true_safe_action_count": int(accepted_true),
            "accepted_set_jaccard_to_oracle": state_jaccard,
            "predicted_targets": targets, "QP": {"selected": None},
            "fresh_exact_two_step": None,
            "selected_action_shift_from_oracle_l2": None,
        })
    gates = config["learned_gate"]
    preliminary = bool(
        false_safe_total == int(gates["test_off_grid_false_safe_action_count"])
        and jaccard(oracle_global, learned_global)
        >= float(gates["minimum_global_accepted_set_jaccard_to_oracle"])
        and minimum_state_jaccard
        >= float(gates["minimum_state_accepted_set_jaccard_to_oracle"])
        and support_count == int(gates["required_test_state_safe_support_count"])
    )
    if preliminary:
        test_rows = _manifest(
            paths["test_selected"], config["test_selected_manifest_sha256"]
        )
        row_by_case = {str(item["case_id"]): item for item in test_rows}
        test_cases = config["split"]["test_case_ids"]
        _require(
            all(case_id in row_by_case for case_id in test_cases),
            "supported region-aware test manifest population differs",
        )
        population = {
            item["case_id"]: item for item in read_jsonl(paths["population"])
        }
        geometry_config = load_shadow_config(paths["geometry"])
        exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
        placeholder_row = row_by_case["vlsa-t1-goal-ii-t0-e05"]
        placeholder_path = paths["archived"] / placeholder_row[
            "archived_relative_path"
        ]
        placeholder = _load(placeholder_path)
        _require(
            _file_sha256(placeholder_path) == source["archived_e05_file_sha256"]
            and placeholder.get("result_payload_sha256")
            == source["archived_e05_payload_sha256"],
            "supported region-aware E05 geometry source differs",
        )
        runtime = _runtime_imports(include_aegis=False)
        result_by_index = {
            int(item["state_index"]): item for item in test_results
        }
        maximum_motion = float(config["exact_verification"][
            "maximum_per_step_obstacle_l1_displacement_m"
        ])
        for case_id in test_cases:
            row = row_by_case[case_id]
            archived_path = paths["archived"] / row["archived_relative_path"]
            archived = _load(archived_path)
            _require(
                _file_sha256(archived_path) == row["archived_file_sha256"]
                and archived.get("result_payload_sha256")
                == row["archived_payload_sha256"],
                "supported region-aware test archive differs: %s" % case_id,
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
                case_states = sorted(
                    (
                        state for state in states
                        if state["split"] == "test"
                        and state["case_id"] == case_id
                    ),
                    key=lambda state: int(state["state_step"]),
                )
                needed = {int(state["state_step"]) for state in case_states}
                snapshots = {}
                archived_actions = archived["actions"]
                for step in range(max(needed) + 1):
                    if step in needed:
                        snapshots[step] = _snapshot_env(env)
                    if step < max(needed):
                        env.step(
                            _canonical_action(
                                archived_actions[step], step
                            ).tolist()
                        )
                for state in case_states:
                    state_index = int(state["state_index"])
                    step = int(state["state_step"])
                    item = result_by_index[state_index]
                    _restore_env(env, snapshots[step])
                    first = _canonical_action(archived_actions[step], step)
                    second = _canonical_action(
                        archived_actions[step + 1], step + 1
                    )
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
                        "supported region-aware test feature receipt differs",
                    )
                    oracle_state = oracle_by_index[state_index]
                    predicted = guarded_targets(
                        models, model_state, live_features,
                        oracle_state["regions"], config["uncertainty"],
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
                            chosen["candidate_xyz"], dtype=np.float64,
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
                        exact["true_safe"] = _exact_safe(
                            exact, maximum_motion
                        )
                        oracle_selected = oracle_state["QP"]["0mm"][
                            "selected_solution"
                        ]
                        _require(
                            oracle_selected is not None,
                            "supported region-aware oracle QP support differs",
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
    aggregates = _test_aggregates(test_results)
    p95 = aggregates["selected_action_shift_from_oracle_l2_p95"]
    maximum = aggregates["selected_action_shift_from_oracle_l2_maximum"]
    learned_pass = bool(
        preliminary
        and aggregates["valid_selected_QP_count"]
        == int(gates["required_valid_selected_QP_count"])
        and aggregates["fresh_exact_safe_selected_QP_count"]
        == int(gates["required_fresh_exact_safe_selected_QP_count"])
        and aggregates["released_AEGIS_EE_compatible_selected_QP_count"]
        == int(gates[
            "required_released_AEGIS_EE_compatible_selected_QP_count"
        ])
        and p95 is not None
        and p95 <= float(
            gates["selected_action_shift_from_oracle_l2_p95_maximum"]
        )
        and maximum is not None
        and maximum <= float(
            gates["selected_action_shift_from_oracle_l2_maximum"]
        )
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
            "decision_stability_result_file_sha256": _file_sha256(
                paths["decision"]
            ),
        },
        "validation_fresh": {
            "file_sha256": _file_sha256(paths["validation_fresh"]),
            "payload_sha256": fresh["validation_fresh_payload_sha256"],
            "summary": fresh["summary"],
        },
        "training": training_audit, "calibration": calibration,
        "model_artifact": model_artifact,
        "test_aggregates": aggregates, "test_state_results": test_results,
        "decision": {
            "off_grid_preliminary_gate_pass": preliminary,
            "learned_gate_pass": learned_pass,
            "receding_closed_loop_E05_preregistration_authorized": learned_pass,
            "action_conditioned_model_preregistration_authorized": not learned_pass,
            "closed_loop_e05_executed": False,
            "stop_reason": (
                None if learned_pass
                else "supported_state_coefficient_output_MLP_gate_failed"
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
