#!/usr/bin/env python3
"""Run controlled omitted-variable counterfactuals with fixed old56 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.complete_osc_margin import (
    COLLECTION_RESULT_SCHEMA, DATASET_SCHEMA, payload_sha256,
)
from main.multilink_ellipsoid.matched_input_ablation import (
    RESULT_SCHEMA as MATCHED_RESULT_SCHEMA,
    VALIDATION_SCHEMA as MATCHED_VALIDATION_SCHEMA,
)
from main.multilink_ellipsoid.omitted_variable_audit import (
    RESULT_SCHEMA, analyze_records, array_sha256, axis_rotation,
    complete_model_gate, final_decision, intervention_specs, load_config,
    old56_rows, select_candidate_indexes,
)
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action, _restore_env, _snapshot_env,
)
from scripts.collect_distal_complete_osc_margin_moka10 import (
    _geometry_placeholder_row, _rollout_receipt, _state_input,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_native_geom_inventory_moka10 import _read_manifest, _split_counts
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _combined_complete_hash(env: Any, actions: Any) -> str:
    import numpy as np
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector

    state = np.ascontiguousarray(_dynamic_state_vector(env), dtype=np.float64)
    command = np.ascontiguousarray(actions, dtype=np.float64)
    return hashlib.sha256(state.tobytes() + command.tobytes()).hexdigest()


def _apply_intervention(env: Any, actions: Any, spec: Mapping[str, Any]) -> None:
    import numpy as np

    family = str(spec["family"])
    robot = env.robots[0]
    controller = robot.controller
    if family == "baseline":
        return
    if family == "goal_orientation":
        goal = np.asarray(controller.goal_ori, dtype=np.float64)
        if goal.shape != (3, 3):
            raise ValueError("goal orientation matrix differs")
        controller.goal_ori = goal @ axis_rotation(
            int(spec["axis"]), float(spec["angle_degrees"])
        )
    elif family == "rotation_action":
        actions[0, 3 + int(spec["axis"])] = float(spec["value"])
    elif family == "gripper_command":
        actions[:, 6] = float(spec["value"])
    elif family == "controller_memory":
        name = str(spec["name"])
        if name == "toggle_new_update":
            controller.new_update = not bool(controller.new_update)
        elif name == "zero_relative_ori":
            if controller.relative_ori is not None:
                controller.relative_ori = np.zeros_like(controller.relative_ori)
        elif name == "current_orientation_as_ori_ref":
            controller.ori_ref = np.asarray(
                controller.ee_ori_mat, dtype=np.float64
            ).copy()
        elif name == "zero_previous_torques":
            if controller.torques is not None:
                controller.torques = np.zeros_like(controller.torques)
            if robot.torques is not None:
                robot.torques = np.zeros_like(robot.torques)
        else:
            raise ValueError("controller-memory intervention differs")
    else:
        raise ValueError("omitted-variable intervention family differs")


def _canonical_contacts(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main() -> int:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--same-task-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-result", type=Path, required=True)
    parser.add_argument("--matched-result", type=Path, required=True)
    parser.add_argument("--matched-validation", type=Path, required=True)
    parser.add_argument("--complete-model", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "population": args.population_manifest, "selected": args.selected_manifest,
        "same_task": args.same_task_manifest, "targeted": args.targeted_manifest,
        "dataset": args.dataset, "collection": args.collection_result,
        "matched_result": args.matched_result,
        "matched_validation": args.matched_validation,
        "complete_model": args.complete_model, "archived": args.archived_root,
        "geometry": args.geometry_config, "exact_box": args.exact_box_config,
        "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source_config = config["immutable_source"]
    file_receipts = (
        (paths["dataset"], "dataset_file_sha256"),
        (paths["collection"], "collection_result_file_sha256"),
        (paths["matched_result"], "matched_result_file_sha256"),
        (paths["matched_validation"], "matched_validation_file_sha256"),
        (paths["complete_model"], "complete_model_file_sha256"),
        (paths["population"], "population_manifest_sha256"),
        (paths["selected"], "selected_manifest_sha256"),
        (paths["same_task"], "same_task_manifest_sha256"),
        (paths["targeted"], "targeted_manifest_sha256"),
        (paths["geometry"], "geometry_config_sha256"),
        (paths["exact_box"], "exact_box_config_sha256"),
    )
    for path, key in file_receipts:
        _require(_file_sha256(path) == source_config[key], "omitted-variable source differs: %s" % key)
    dataset = _load(paths["dataset"])
    collection = _load(paths["collection"])
    matched = _load(paths["matched_result"])
    matched_validation = _load(paths["matched_validation"])
    _require(
        dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source_config["dataset_payload_sha256"]
        == payload_sha256(dataset, "dataset_payload_sha256")
        and collection.get("schema_version") == COLLECTION_RESULT_SCHEMA
        and collection.get("result_payload_sha256")
        == source_config["collection_result_payload_sha256"]
        == payload_sha256(collection, "result_payload_sha256")
        and matched.get("schema_version") == MATCHED_RESULT_SCHEMA
        and matched.get("result_payload_sha256")
        == source_config["matched_result_payload_sha256"]
        == payload_sha256(matched, "result_payload_sha256")
        and matched_validation.get("schema_version") == MATCHED_VALIDATION_SCHEMA
        and matched_validation.get("validation_payload_sha256")
        == source_config["matched_validation_payload_sha256"]
        == payload_sha256(matched_validation, "validation_payload_sha256"),
        "omitted-variable payload identity differs",
    )
    deterministic = bool(
        int(dataset["summary"]["pair_count"])
        == int(source_config["expected_candidate_pair_count"])
        and int(dataset["summary"]["replay_count"])
        == int(source_config["expected_duplicate_replay_count"])
        and int(dataset["summary"]["nondeterministic_pair_count"]) == 0
        and float(dataset["summary"]["maximum_repeat_margin_absolute_difference_m"]) == 0.0
        and all(
            candidate["replay_deterministic"]
            and candidate["repeat_receipt"]["margin_bitwise_equal"]
            and candidate["repeat_receipt"]["contact_receipt_equal"]
            and candidate["repeat_receipt"]["next_state_hashes_equal"]
            for state in dataset["state_records"]
            for candidate in state["candidates"]
        )
    )
    _require(deterministic, "omitted-variable determinism test failed")
    states = dataset["state_records"]
    _require(
        len(states) == int(source_config["expected_state_count"])
        and _split_counts(states) == config["split"]["expected_state_counts"],
        "omitted-variable state population differs",
    )
    selected = []
    for path in (paths["selected"], paths["same_task"], paths["targeted"]):
        selected.extend(_read_manifest(path, _file_sha256(path)))
    row_by_case = {str(item["case_id"]): item for item in selected}
    _require(
        len(row_by_case) == 17
        and {str(item["case_id"]) for item in states} == set(row_by_case),
        "omitted-variable episode population differs",
    )
    placeholder_row = _geometry_placeholder_row(row_by_case)
    placeholder_path = paths["archived"] / placeholder_row["archived_relative_path"]
    geometry_placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == placeholder_row["archived_file_sha256"]
        and geometry_placeholder.get("result_payload_sha256")
        == placeholder_row["archived_payload_sha256"],
        "omitted-variable geometry placeholder differs",
    )
    population = {item["case_id"]: item for item in read_jsonl(paths["population"])}
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    specs = intervention_specs(config)
    records = []
    audited_states = []
    for case_id in sorted(row_by_case):
        selected_row = row_by_case[case_id]
        archived_path = paths["archived"] / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256") == selected_row["archived_payload_sha256"],
            "omitted-variable archive differs: %s" % case_id,
        )
        case = population[case_id]
        validate_case_row(case, paths["repo"])
        env = probe_env = None
        try:
            env, probe_env, task, observation, setup = _build_pair(runtime, case)
            pairing = pairing_record(
                case=case, selected_initial_state=setup["selected_initial_state"],
                settled_observation=observation, task_description=str(task.language),
                active_obstacle_name=setup["obstacle_name"],
                settled_simulator_state=np.asarray(
                    env.sim.get_state().flatten(), dtype=np.float64
                ),
            )
            for key in (
                "manifest_row_sha256", "initial_state_sha256",
                "initial_observation_sha256", "settled_simulator_state_sha256",
                "settled_active_obstacle_position_sha256",
                "policy_noise_schedule_sha256",
            ):
                _require(
                    pairing[key] == collection["pairings"][case_id][key],
                    "omitted-variable pairing differs: %s:%s" % (case_id, key),
                )
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                archived=geometry_placeholder, env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            case_states = sorted(
                (item for item in states if item["case_id"] == case_id),
                key=lambda item: int(item["state_step"]),
            )
            by_step = {int(item["state_step"]): item for item in case_states}
            maximum_step = max(by_step)
            for step in range(maximum_step + 1):
                if step in by_step:
                    state = by_step[step]
                    live = _state_input(env, probe, setup["obstacle_name"])
                    _require(
                        live["complete_snapshot_sha256"]
                        == state["complete_snapshot_sha256"],
                        "omitted-variable complete snapshot differs: %s:%d" % (case_id, step),
                    )
                    first = _canonical_action(archived["actions"][step], step)
                    second = _canonical_action(archived["actions"][step + 1], step + 1)
                    _require(
                        np.array_equal(first, np.asarray(state["nominal_first_action"], dtype=np.float64))
                        and np.array_equal(second, np.asarray(state["nominal_second_action"], dtype=np.float64)),
                        "omitted-variable nominal action differs",
                    )
                    xyz = np.asarray([
                        item["full_two_action_commands"][0][:3]
                        for item in state["candidates"]
                    ], dtype=np.float64)
                    lower, upper = np.min(xyz, axis=0), np.max(xyz, axis=0)
                    baseline_snapshot = _snapshot_env(env)
                    selections = select_candidate_indexes(state)
                    for selection in selections:
                        candidate_index = int(selection["candidate_index"])
                        source_candidate = state["candidates"][candidate_index]
                        _require(
                            int(source_candidate["candidate_index"]) == candidate_index,
                            "omitted-variable candidate ordering differs",
                        )
                        base_actions = np.asarray(
                            source_candidate["full_two_action_commands"], dtype=np.float64
                        )
                        _restore_env(env, baseline_snapshot)
                        baseline_old = old56_rows(
                            env, probe, first[:3], base_actions[0, :3], second[:3],
                            lower, upper,
                        )
                        baseline_old_hash = array_sha256(baseline_old)
                        baseline_complete_hash = _combined_complete_hash(env, base_actions)
                        for spec in specs:
                            _restore_env(env, baseline_snapshot)
                            variant_actions = base_actions.copy()
                            _apply_intervention(env, variant_actions, spec)
                            variant_old = old56_rows(
                                env, probe, first[:3], base_actions[0, :3], second[:3],
                                lower, upper,
                            )
                            old_hash = array_sha256(variant_old)
                            complete_hash = _combined_complete_hash(env, variant_actions)
                            effective = bool(
                                spec["family"] == "baseline"
                                or complete_hash != baseline_complete_hash
                            )
                            receipt = _rollout_receipt(probe, env, variant_actions)
                            margin = np.asarray(receipt["margin"][:7], dtype=np.float64)
                            if spec["family"] == "baseline":
                                source_margin = np.asarray(
                                    source_candidate["rollout_minimum_ellipsoid_margin_m"],
                                    dtype=np.float64,
                                )
                                _require(
                                    np.array_equal(margin, source_margin)
                                    and int(receipt["raw_contact_count"])
                                    == int(source_candidate["raw_mujoco_protected_contact_count"])
                                    and receipt["next_state_sha256_per_action"]
                                    == source_candidate["next_state_sha256_per_action"],
                                    "omitted-variable baseline replay differs",
                                )
                            records.append({
                                "state_index": int(state["state_index"]),
                                "case_id": case_id, "split": str(state["split"]),
                                "state_step": int(step),
                                "candidate_index": candidate_index,
                                "selection_source": selection["selection_source"],
                                "variant_family": str(spec["family"]),
                                "variant_name": str(spec["name"]),
                                "effective_intervention": effective,
                                "rollout_executed": True,
                                "old56_sha256": old_hash,
                                "baseline_old56_sha256": baseline_old_hash,
                                "old56_hash_equal_to_baseline": bool(
                                    old_hash == baseline_old_hash
                                    and np.array_equal(variant_old, baseline_old)
                                ),
                                "complete_snapshot_action_sha256": complete_hash,
                                "baseline_complete_snapshot_action_sha256": baseline_complete_hash,
                                "rollout_minimum_ellipsoid_margin_m": margin.tolist(),
                                "worst_margin_m": float(np.min(margin)),
                                "proxy_safe": bool(np.all(margin >= 0.0)),
                                "raw_contact_free": bool(receipt["raw_contact_count"] == 0),
                                "raw_contact_count": int(receipt["raw_contact_count"]),
                                "raw_contact_receipt_sha256": hashlib.sha256(
                                    _canonical_contacts(receipt["raw_contacts"]).encode("utf-8")
                                ).hexdigest(),
                                "next_state_sha256_per_action": receipt["next_state_sha256_per_action"],
                                "maximum_obstacle_motion_m": float(receipt["maximum_obstacle_motion_m"]),
                            })
                    _restore_env(env, baseline_snapshot)
                    audited_states.append(int(state["state_index"]))
                    print(json.dumps({
                        "case_id": case_id, "state_step": step,
                        "state_index": int(state["state_index"]),
                        "record_count": len(records),
                    }, sort_keys=True), flush=True)
                if step < maximum_step:
                    env.step(_canonical_action(archived["actions"][step], step).tolist())
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    _require(
        sorted(audited_states) == list(range(85)),
        "omitted-variable audit dropped a state",
    )
    audit = analyze_records(records)
    _require(
        audit["all_effective_interventions_preserve_old56_bytes"],
        "omitted-variable intervention changed old56 input",
    )
    complete = complete_model_gate(matched)
    decision = final_decision(audit, complete)
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "determinism_test": {
            "pair_count": int(dataset["summary"]["pair_count"]),
            "duplicate_replay_count": int(dataset["summary"]["replay_count"]),
            "mismatch_count": int(dataset["summary"]["nondeterministic_pair_count"]),
            "maximum_margin_absolute_difference_m": float(
                dataset["summary"]["maximum_repeat_margin_absolute_difference_m"]
            ),
            "margins_contacts_and_next_state_exact": deterministic,
        },
        "state_count": len(audited_states),
        "selected_candidate_count": len({
            (item["state_index"], item["candidate_index"]) for item in records
        }),
        "variant_records": records, "omitted_variable_audit": audit,
        "complete_model_test": complete, "decision": decision,
        "forbidden_action_receipt": {
            "QP_executed": False, "calibration_executed": False,
            "closed_loop_E05_executed": False,
            "model_retraining_executed": False,
            "new_policy_inference_executed": False,
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "determinism_test": result["determinism_test"],
        "omitted_variable_audit": audit,
        "complete_model_test": complete, "decision": decision,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
