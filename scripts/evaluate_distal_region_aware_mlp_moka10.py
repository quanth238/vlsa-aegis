#!/usr/bin/env python3
"""Train and evaluate the gated region-aware safety MLP on H100."""

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
    RESULT_SCHEMA as MULTI_RESULT_SCHEMA,
)
from main.multilink_ellipsoid.region_aware_mlp import (
    RESULT_SCHEMA, guarded_targets, jaccard, load_config,
    regional_training_arrays, save_model, solve_regional_qps, target_values,
    train_ensemble,
)
from main.multilink_ellipsoid.two_step_margin import feature_context, feature_vectors
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


def _exact_safe(record: Mapping[str, Any], maximum_motion_m: float) -> bool:
    import numpy as np

    return bool(
        np.all(np.asarray(record["minimum_distal_margin_m"], dtype=np.float64) >= 0.0)
        and record["D_sim_raw_safe"]
        and int(record["raw_protected_contact_count"]) == 0
        and float(record["maximum_within_step_obstacle_l1_displacement_m"])
        <= maximum_motion_m
    )


def _selected_rows(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    _require(
        hashlib.sha256(raw).hexdigest() == config["selected_manifest_sha256"],
        "region-aware selected manifest hash differs",
    )
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    _require(
        len(rows) == 10 and len({item["case_id"] for item in rows}) == 10
        and sorted(item["case_id"] for item in rows if item["split"] == "test")
        == config["split"]["test_case_ids"],
        "region-aware selected manifest population differs",
    )
    return rows


def _calibrate(
    *, models: list[Any], model_state: dict[str, Any], arrays: Mapping[str, Any],
    states: list[Mapping[str, Any]], multi_states: list[Mapping[str, Any]],
    off_grid_states: list[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validation-only per-region/row one-sided calibration."""

    import numpy as np

    validation = np.asarray(arrays["split"] == "validation", dtype=bool)
    members = np.asarray(model_state["predictions_m"], dtype=np.float64)
    multiplier = float(config["uncertainty"]["standard_deviation_multiplier"])
    over = np.full((27, 7), -np.inf, dtype=np.float64)
    fit_counts = np.zeros((27, 7), dtype=np.int64)
    for sample_index in np.flatnonzero(validation):
        region_index = int(arrays["region_index"][sample_index])
        row = int(arrays["constraint_index"][sample_index])
        delta = np.asarray(arrays["deltas"][sample_index], dtype=np.float64)
        member_values = (
            members[:, sample_index, 0, None]
            + np.einsum("mi,vi->mv", members[:, sample_index, 1:4], delta)
        )
        guard = multiplier * float(np.max(np.std(member_values, axis=0)))
        mean = np.mean(members[:, sample_index], axis=0)
        predicted = mean[0] - guard + delta @ mean[1:4]
        exact = np.asarray(arrays["exact_margin_m"][sample_index], dtype=np.float64)
        over[region_index, row] = max(
            over[region_index, row], float(np.max(predicted - exact))
        )
        fit_counts[region_index, row] += len(exact)

    state_by_index = {int(item["state_index"]): item for item in states}
    multi_by_index = {int(item["state_index"]): item for item in multi_states}
    off_by_index = {int(item["state_index"]): item for item in off_grid_states}
    off_counts = np.zeros((27, 7), dtype=np.int64)
    model_state["calibration_m"] = np.zeros((27, 7), dtype=np.float64)
    for state_index, state in state_by_index.items():
        if state["split"] != "validation":
            continue
        multi = multi_by_index[state_index]
        fresh = off_by_index[state_index]
        targets = guarded_targets(
            models, model_state, state["pair_state_feature_vectors"],
            multi["regions"], config["uncertainty"],
        )
        for action in fresh["fresh_actions"]:
            xyz = action["candidate_xyz"]
            exact = np.asarray(action["minimum_distal_margin_m"], dtype=np.float64)
            for region_index in action["containing_region_indexes"]:
                predicted = target_values(targets[int(region_index)], xyz)
                over[int(region_index)] = np.maximum(
                    over[int(region_index)], predicted - exact
                )
                off_counts[int(region_index)] += 1
    _require(
        np.all(np.isfinite(over)) and np.all(fit_counts > 0)
        and np.all(off_counts > 0),
        "region-aware validation calibration lacks support",
    )
    padding = float(config["uncertainty"]["fixed_padding_m"])
    calibration = np.maximum(over, 0.0) + padding
    model_state["calibration_m"] = calibration
    return {
        "calibration_m": calibration.tolist(),
        "maximum_m": float(np.max(calibration)),
        "mean_m": float(np.mean(calibration)),
        "fit_value_count": int(np.sum(fit_counts)),
        "off_grid_region_value_count": int(np.sum(off_counts)),
        "validation_state_indexes": sorted(
            int(index) for index, state in state_by_index.items()
            if state["split"] == "validation"
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
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--off-grid-result", type=Path, required=True)
    parser.add_argument("--multi-region-result", type=Path, required=True)
    parser.add_argument("--multi-region-validation", type=Path, required=True)
    parser.add_argument("--decision-stability-result", type=Path, required=True)
    parser.add_argument("--decision-stability-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {name: value.resolve() for name, value in {
        "repo": args.repo_root, "population": args.population_manifest,
        "selected": args.selected_manifest, "archived": args.archived_root,
        "geometry": args.geometry_config, "exact_box": args.exact_box_config,
        "config": args.config, "dataset": args.dataset,
        "off_grid": args.off_grid_result, "multi": args.multi_region_result,
        "multi_validation": args.multi_region_validation,
        "decision": args.decision_stability_result,
        "decision_validation": args.decision_stability_validation,
        "model": args.model, "output": args.output,
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
        (paths["decision"], source["decision_stability_result_file_sha256"], "decision-stability"),
        (paths["decision_validation"], source["decision_stability_validation_file_sha256"], "decision-validation"),
    ):
        _require(_file_sha256(path) == expected, "region-aware %s differs" % label)
    dataset = _load(paths["dataset"])
    off_grid = _load(paths["off_grid"])
    multi = _load(paths["multi"])
    multi_validation = _load(paths["multi_validation"])
    decision = _load(paths["decision"])
    decision_validation = _load(paths["decision_validation"])
    _require(
        dataset.get("schema_version") == AFFINE_COEFFICIENT_DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source["dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256")
        and off_grid.get("result_payload_sha256")
        == source["off_grid_result_payload_sha256"]
        == _hash_without(off_grid, "result_payload_sha256")
        and multi.get("schema_version") == MULTI_RESULT_SCHEMA
        and multi.get("result_payload_sha256")
        == source["multi_region_result_payload_sha256"]
        == _hash_without(multi, "result_payload_sha256")
        and multi_validation.get("status") == "valid"
        and decision.get("result_payload_sha256")
        == source["decision_stability_result_payload_sha256"]
        == _hash_without(decision, "result_payload_sha256")
        and decision_validation.get("status") == "valid"
        and decision_validation.get("mlp_training_authorized") is True,
        "region-aware immutable authorization differs",
    )
    states = dataset["state_records"]
    multi_states = multi["state_results"]
    off_grid_states = off_grid["state_results"]
    expected = int(source["expected_state_count"])
    _require(
        len(states) == len(multi_states) == len(off_grid_states) == expected,
        "region-aware state count differs",
    )
    counts = {
        name: sum(item["split"] == name for item in states)
        for name in ("train", "validation", "test")
    }
    _require(counts == config["split"]["expected_state_counts"],
             "region-aware split counts differ")
    test_cases = sorted({item["case_id"] for item in states if item["split"] == "test"})
    _require(test_cases == config["split"]["test_case_ids"],
             "region-aware test cases differ")

    arrays = regional_training_arrays(states, multi_states)
    models, model_state, training_audit = train_ensemble(arrays, config)
    calibration = _calibrate(
        models=models, model_state=model_state, arrays=arrays, states=states,
        multi_states=multi_states, off_grid_states=multi_states, config=config,
    )
    model_state.pop("predictions_m", None)
    model_artifact = save_model(paths["model"], model_state)

    state_by_index = {int(item["state_index"]): item for item in states}
    multi_by_index = {int(item["state_index"]): item for item in multi_states}
    off_by_index = {int(item["state_index"]): item for item in multi_states}
    test_results = []
    learned_global = []
    oracle_global = []
    false_safe_total = 0
    support_count = 0
    minimum_state_jaccard = 1.0
    for state_index in sorted(
        index for index, item in state_by_index.items() if item["split"] == "test"
    ):
        state = state_by_index[state_index]
        multi_state = multi_by_index[state_index]
        off_state = off_by_index[state_index]
        targets = guarded_targets(
            models, model_state, state["pair_state_feature_vectors"],
            multi_state["regions"], config["uncertainty"],
        )
        learned_flags = []
        oracle_flags = []
        true_flags = []
        false_indexes = []
        for fresh_index, fresh in enumerate(off_state["fresh_actions"]):
            learned = bool(any(
                np.all(target_values(targets[int(region_index)], fresh["candidate_xyz"]) >= 0.0)
                for region_index in fresh["containing_region_indexes"]
            ))
            oracle = bool(fresh["arms"]["0mm"]["multi_region_predicted_safe"])
            true = bool(fresh["arms"]["0mm"]["true_safe"])
            learned_flags.append(learned)
            oracle_flags.append(oracle)
            true_flags.append(true)
            if learned and not true:
                false_indexes.append(fresh_index)
        state_jaccard = jaccard(oracle_flags, learned_flags)
        accepted_true = sum(
            learned and true for learned, true in zip(learned_flags, true_flags)
        )
        support_count += int(accepted_true > 0)
        false_safe_total += len(false_indexes)
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
            "predicted_targets": targets,
            "QP": None, "fresh_exact_two_step": None,
            "selected_action_shift_from_oracle_l2": None,
        })
    global_jaccard = jaccard(oracle_global, learned_global)
    gates = config["learned_gate"]
    preliminary = bool(
        false_safe_total == int(gates["test_off_grid_false_safe_action_count"])
        and global_jaccard
        >= float(gates["minimum_global_accepted_set_jaccard_to_oracle"])
        and minimum_state_jaccard
        >= float(gates["minimum_state_accepted_set_jaccard_to_oracle"])
        and support_count == int(gates["required_test_state_safe_support_count"])
    )

    valid_qp = 0
    exact_safe_qp = 0
    ee_compatible_qp = 0
    action_shifts = []
    if preliminary:
        selected = _selected_rows(paths["selected"], config)
        selected_by_case = {item["case_id"]: item for item in selected}
        population = {
            item["case_id"]: item for item in read_jsonl(paths["population"])
        }
        geometry_config = load_shadow_config(paths["geometry"])
        exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
        placeholder_row = selected_by_case[config["closed_loop"]["case_id"]]
        placeholder = _load(
            paths["archived"] / placeholder_row["archived_relative_path"]
        )
        runtime = _runtime_imports(include_aegis=False)
        result_by_index = {item["state_index"]: item for item in test_results}
        for case_id in test_cases:
            row = selected_by_case[case_id]
            archived_path = paths["archived"] / row["archived_relative_path"]
            archived = _load(archived_path)
            _require(
                _file_sha256(archived_path) == row["archived_file_sha256"]
                and archived.get("result_payload_sha256")
                == row["archived_payload_sha256"],
                "region-aware held-out archive differs: %s" % case_id,
            )
            env = probe_env = None
            try:
                env, probe_env, _, _, setup = _build_pair(runtime, population[case_id])
                geometry, exact_boxes = _geometry(
                    geometry_config=geometry_config,
                    exact_box_config=exact_box_config, archived=placeholder, env=env,
                    obstacle_name=setup["obstacle_name"],
                )
                probe = SubstepEightConstraintProbe(
                    probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                    quadratic_tolerance=1.0e-6,
                    contact_distance_threshold_m=0.0,
                    obstacle_primitive_union=exact_boxes,
                )
                case_states = sorted(
                    (item for item in states if item["split"] == "test"
                     and item["case_id"] == case_id),
                    key=lambda item: int(item["state_step"]),
                )
                needed = {int(item["state_step"]) for item in case_states}
                snapshots = {}
                archived_actions = archived["actions"]
                for step in range(max(needed) + 1):
                    if step in needed:
                        snapshots[step] = _snapshot_env(env)
                    if step < max(needed):
                        env.step(_canonical_action(archived_actions[step], step).tolist())
                for state in case_states:
                    state_index = int(state["state_index"])
                    item = result_by_index[state_index]
                    step = int(state["state_step"])
                    _restore_env(env, snapshots[step])
                    first = _canonical_action(archived_actions[step], step)
                    second = _canonical_action(archived_actions[step + 1], step + 1)
                    context = feature_context(env, probe)
                    _, live_features = feature_vectors(
                        context, first[:3], first[:3], second[:3]
                    )
                    feature_error = float(np.max(np.abs(
                        live_features - np.asarray(
                            state["pair_state_feature_vectors"], dtype=np.float64
                        )
                    )))
                    _require(feature_error <= 1.0e-8,
                             "region-aware held-out feature receipt differs")
                    predicted = guarded_targets(
                        models, model_state, live_features,
                        multi_by_index[state_index]["regions"], config["uncertainty"],
                    )
                    qp = solve_regional_qps(
                        first[:3], multi_by_index[state_index]["regions"], predicted,
                        config["projection"],
                    )
                    chosen = qp["selected"]
                    exact = None
                    if chosen is not None:
                        valid_qp += 1
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
                                summary["maximum_within_step_obstacle_l1_displacement_m"]
                            ),
                            "next_state_sha256": summary["next_state_sha256"],
                        }
                        exact["true_safe"] = _exact_safe(
                            exact,
                            float(config["closed_loop"][
                                "maximum_per_step_obstacle_l1_displacement_m"
                            ]),
                        )
                        exact_safe_qp += int(exact["true_safe"])
                        ee_compatible_qp += int(
                            exact["minimum_released_AEGIS_EE_margin_m"] >= 0.0
                        )
                        oracle_xyz = np.asarray(
                            multi_by_index[state_index]["QP"]["0mm"]
                            ["selected_solution"]["candidate_xyz"], dtype=np.float64,
                        )
                        shift = float(np.linalg.norm(candidate[:3] - oracle_xyz))
                        action_shifts.append(shift)
                        item["selected_action_shift_from_oracle_l2"] = shift
                    item["feature_max_abs_error"] = feature_error
                    item["QP"] = qp
                    item["fresh_exact_two_step"] = exact
            finally:
                if probe_env is not None:
                    probe_env.close()
                if env is not None:
                    env.close()

    p95_shift = None if not action_shifts else float(np.quantile(action_shifts, 0.95))
    max_shift = None if not action_shifts else float(np.max(action_shifts))
    learned_pass = bool(
        preliminary
        and valid_qp == int(gates["required_valid_selected_QP_count"])
        and exact_safe_qp
        == int(gates["required_fresh_exact_safe_selected_QP_count"])
        and ee_compatible_qp
        == int(gates["required_released_AEGIS_EE_compatible_selected_QP_count"])
        and p95_shift is not None
        and p95_shift
        <= float(gates["selected_action_shift_from_oracle_l2_p95_maximum"])
        and max_shift is not None
        and max_shift <= float(gates["selected_action_shift_from_oracle_l2_maximum"])
    )
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_inputs": {
            "dataset_file_sha256": _file_sha256(paths["dataset"]),
            "off_grid_result_file_sha256": _file_sha256(paths["off_grid"]),
            "multi_region_result_file_sha256": _file_sha256(paths["multi"]),
            "decision_stability_result_file_sha256": _file_sha256(paths["decision"]),
        },
        "training": training_audit, "calibration": calibration,
        "model_artifact": model_artifact,
        "test_aggregates": {
            "state_count": len(test_results),
            "off_grid_action_count": len(learned_global),
            "off_grid_false_safe_action_count": false_safe_total,
            "global_accepted_set_jaccard_to_oracle": global_jaccard,
            "minimum_state_accepted_set_jaccard_to_oracle": minimum_state_jaccard,
            "safe_support_state_count": support_count,
            "valid_selected_QP_count": valid_qp,
            "fresh_exact_safe_selected_QP_count": exact_safe_qp,
            "released_AEGIS_EE_compatible_selected_QP_count": ee_compatible_qp,
            "selected_action_shift_from_oracle_l2_p95": p95_shift,
            "selected_action_shift_from_oracle_l2_maximum": max_shift,
        },
        "test_state_results": test_results,
        "decision": {
            "off_grid_preliminary_gate_pass": preliminary,
            "learned_gate_pass": learned_pass,
            "closed_loop_e05_authorized": learned_pass,
            "closed_loop_e05_executed": False,
            "stop_reason": None if learned_pass else "learned_unseen_state_gate_failed",
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["result_payload_sha256"] = _hash_without(
        output, "result_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps({
        "decision": output["decision"], "test_aggregates": output["test_aggregates"],
        "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
