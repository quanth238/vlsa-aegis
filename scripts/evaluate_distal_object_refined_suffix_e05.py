#!/usr/bin/env python3
"""Run the privileged object-aware continuous-refinement suffix oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.object_refined_suffix import (
    RESULT_SCHEMA, candidate_rank, eligible_indices, load_config,
    refinement_chunks, select_candidate,
)
from main.multilink_ellipsoid.waypoint_candidate_closed_loop import (
    WAYPOINT_RESULT_SCHEMA, waypoint_chunks,
)
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_affine_oracle_closed_loop_e05 import _measured_env_step
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_waypoint_closed_loop_e05 import _summarize_rollout
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _verify_prerequisite(
    config: Mapping[str, Any], result_path: Path, validation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prerequisite = config["prerequisite"]
    result = _load(result_path)
    validation = _load(validation_path)
    prefix_end = int(config["primary_case"]["suffix_start_step"])
    actions = result.get("actions", [])
    _require(
        _file_sha256(result_path) == prerequisite["waypoint_result_file_sha256"]
        and _file_sha256(validation_path)
        == prerequisite["waypoint_validation_file_sha256"]
        and result.get("result_payload_sha256")
        == prerequisite["waypoint_result_payload_sha256"]
        and result.get("schema_version") == WAYPOINT_RESULT_SCHEMA
        and result.get("status") == "complete"
        and validation.get("status") == "validated"
        and validation.get("all_executed_substeps_all_eight_safe") is True
        and validation.get("all_executed_next_states_match_clone") is True
        and validation.get("zero_protected_contact") is True
        and validation.get("zero_paper_CAR") is True
        and isinstance(actions, list) and len(actions) == 237
        and all(actions[index].get("executed") is True for index in range(prefix_end)),
        "validated waypoint prerequisite differs",
    )
    receipt = {
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "validation_path": str(validation_path),
        "validation_file_sha256": _file_sha256(validation_path),
        "validated_exact_safe_prefix_end_exclusive": prefix_end,
    }
    return result, receipt


def _relative_object_vector(snapshot: Mapping[str, Any]) -> list[float]:
    atoms = snapshot.get("argument_poses", [])
    _require(len(atoms) == 1, "goal atom count differs")
    arguments = atoms[0].get("arguments", [])
    _require(
        len(arguments) == 2
        and arguments[0].get("name") == "akita_black_bowl_1"
        and arguments[1].get("name") == "plate_1",
        "goal object identities differ",
    )
    bowl = arguments[0]["position"]
    plate = arguments[1]["position"]
    return [float(bowl[index]) - float(plate[index]) for index in range(3)]


def _reference_trajectory(
    runtime: Mapping[str, Any], case: Mapping[str, Any], actions: Sequence[Any],
    archived: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _goal_progress_definition, _goal_progress_snapshot, pairing_record,
    )

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
            _require(pairing[key] == archived["pairing"][key],
                     "reference pairing differs: %s" % key)
        _, atoms = _goal_progress_definition(env)
        eef_positions = []
        relative_vectors = []
        first_done = None
        for step, source in enumerate(actions):
            observation, _, done, _ = env.step(_canonical_action(source, step).tolist())
            snapshot = _goal_progress_snapshot(
                env, atoms, step=step, previous_values=None
            )
            eef_positions.append(np.asarray(
                observation["robot0_eef_pos"], dtype=np.float64
            ))
            relative_vectors.append(np.asarray(
                _relative_object_vector(snapshot), dtype=np.float64
            ))
            if first_done is None and done:
                first_done = step
        return np.asarray(eef_positions), np.asarray(relative_vectors), {
            "source": "parallel_immutable_successful_AEGIS_replay",
            "count": len(eef_positions),
            "native_success_step": first_done,
            "read_only": True,
        }
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def evaluate(
    *, repo_root: Path, population_manifest: Path, selected_manifest: Path,
    archived_root: Path, geometry_config_path: Path, exact_box_config_path: Path,
    source_config_path: Path, waypoint_config_path: Path, config_path: Path,
    prerequisite_result_path: Path, prerequisite_validation_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _goal_progress_definition, _goal_progress_snapshot, _runtime_imports,
        array_sha256, pairing_record, read_jsonl,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config
    from main.multilink_ellipsoid.two_step_margin import (
        load_selected_manifest, load_two_step_config,
    )
    from main.multilink_ellipsoid.waypoint_candidate_closed_loop import (
        load_waypoint_config,
    )

    started = time.perf_counter_ns()
    config = load_config(config_path)
    waypoint_config = load_waypoint_config(waypoint_config_path)
    prerequisite_result, prerequisite = _verify_prerequisite(
        config, prerequisite_result_path, prerequisite_validation_path
    )
    source_config = load_two_step_config(source_config_path)
    selected = load_selected_manifest(selected_manifest, source_config)
    case_id = config["primary_case"]["case_id"]
    selected_rows = [row for row in selected if row["case_id"] == case_id]
    _require(len(selected_rows) == 1, "primary selected row differs")
    population = {row["case_id"]: row for row in read_jsonl(population_manifest)}
    _require(case_id in population, "primary population row missing")
    case = population[case_id]
    archived_path = archived_root / selected_rows[0]["archived_relative_path"]
    archived = _load(archived_path)
    _require(
        _file_sha256(archived_path) == selected_rows[0]["archived_file_sha256"]
        and archived.get("result_payload_sha256")
        == selected_rows[0]["archived_payload_sha256"],
        "immutable Table-1 result differs",
    )
    actions = archived.get("actions", [])
    _require(len(actions) == config["primary_case"]["expected_action_count"],
             "immutable action count differs")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    reference_eef, reference_object, reference = _reference_trajectory(
        runtime, case, actions, archived
    )
    _require(
        reference["native_success_step"]
        == config["primary_case"]["original_native_task_success_step"],
        "reference success step differs",
    )
    reference["eef_array_sha256"] = array_sha256(reference_eef)
    reference["object_relative_array_sha256"] = array_sha256(reference_object)
    prefix_end = int(config["primary_case"]["suffix_start_step"])
    minimum_margin = float(
        config["activation"]["intervene_if_any_eight_margin_below_m"]
    )
    maximum_motion = float(
        config["activation"]["maximum_per_step_obstacle_l1_displacement_m"]
    )
    env = probe_env = None
    suffix_records: list[dict[str, Any]] = []
    failure = None
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
            _require(pairing[key] == archived["pairing"][key],
                     "pairing differs: %s" % key)
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config, exact_box_config=exact_box_config,
            archived=archived, env=env, obstacle_name=setup["obstacle_name"],
        )
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        _, goal_atoms = _goal_progress_definition(env)
        _, probe_goal_atoms = _goal_progress_definition(probe_env)
        initial_obstacle = np.asarray(
            observation["%s_pos" % setup["obstacle_name"]], dtype=np.float64
        ).copy()
        prefix_started = time.perf_counter_ns()
        prefix_last_goal = None
        for step in range(prefix_end):
            prior = prerequisite_result["actions"][step]
            action = np.asarray(prior["executed_action"], dtype=np.float64)
            observation, _, done, _ = env.step(action.tolist())
            state_hash = hashlib.sha256(
                _dynamic_state_vector(env).tobytes()
            ).hexdigest()
            expected_hash = prior["executed_measurement"]["next_state_sha256"]
            if state_hash != expected_hash:
                failure = {
                    "component": "prefix_state_hash_mismatch", "step": step,
                    "expected": expected_hash, "observed": state_hash,
                }
                break
            _require(not done, "validated prefix unexpectedly completed task")
            if step == prefix_end - 1:
                prefix_last_goal = _goal_progress_snapshot(
                    env, goal_atoms, step=step, previous_values=None
                )
        _require(failure is None, "prefix_state_hash_mismatch")
        prefix_receipt = {
            "executed_action_count": prefix_end,
            "all_next_state_hashes_match_validated_artifact": True,
            "terminal_next_state_sha256": prerequisite_result["actions"][
                prefix_end - 1
            ]["executed_measurement"]["next_state_sha256"],
            "terminal_object_relative_to_plate_m": _relative_object_vector(
                prefix_last_goal
            ),
            "wall_seconds": (time.perf_counter_ns() - prefix_started) * 1.0e-9,
        }
        first_contact_step = None
        first_car_step = None
        native_success_step = None
        total_oracle_seconds = 0.0
        maximum_oracle_seconds = 0.0
        candidate_total = 0
        for step in range(prefix_end, len(actions)):
            activation_count = min(2, len(actions) - step)
            nominal_actions = np.asarray([
                _canonical_action(actions[step + offset], step + offset)
                for offset in range(activation_count)
            ], dtype=np.float64)
            nominal_rollout = probe.rollout_chunk(env, nominal_actions.tolist())
            nominal_summary = _summarize_rollout(
                nominal_rollout,
                maximum_obstacle_displacement_m=maximum_motion,
            )
            nominal_safe = bool(
                nominal_summary["D_sim_raw_safe"]
                and min(nominal_summary["minimum_all_eight_substep_clearance_m"])
                >= minimum_margin
            )
            intervention = not nominal_safe
            executed = nominal_actions[0].copy()
            search_record = None
            selector = None
            selected_summary = nominal_summary
            if intervention:
                search_started = time.perf_counter_ns()
                horizon = min(4, len(actions) - step)
                nominal_chunk = np.asarray([
                    _canonical_action(actions[step + offset], step + offset)
                    for offset in range(horizon)
                ], dtype=np.float64)
                terminal_index = step + horizon - 1
                target_eef = reference_eef[terminal_index]
                target_object = reference_object[terminal_index]
                records: list[dict[str, Any]] = []
                chunks: list[Any] = []
                summaries: list[dict[str, Any]] = []
                seen = set()

                def add_candidate(chunk: Any, source_name: str) -> None:
                    array = np.asarray(chunk, dtype=np.float64)
                    key = array[:, :3].tobytes()
                    if key in seen:
                        return
                    seen.add(key)
                    rollout = probe.rollout_chunk(env, array.tolist())
                    summary = _summarize_rollout(
                        rollout, maximum_obstacle_displacement_m=maximum_motion
                    )
                    snapshot = _goal_progress_snapshot(
                        probe_env, probe_goal_atoms, step=terminal_index,
                        previous_values=None,
                    )
                    terminal_object = np.asarray(
                        _relative_object_vector(snapshot), dtype=np.float64
                    )
                    terminal_eef = np.asarray(
                        summary["terminal_eef_position_m"], dtype=np.float64
                    )
                    records.append({
                        "source": source_name,
                        "chunk": array.tolist(),
                        "chunk_correction_l2": float(np.linalg.norm(
                            array[:, :3] - nominal_chunk[:, :3]
                        )),
                        "minimum_all_eight_margin_m": summary[
                            "minimum_all_eight_substep_clearance_m"
                        ],
                        "raw_safe": bool(summary["D_sim_raw_safe"]),
                        "task_success_in_chunk": bool(
                            summary["task_success_in_chunk"]
                        ),
                        "terminal_object_relative_to_plate_m": (
                            terminal_object.tolist()
                        ),
                        "terminal_object_reference_error_m": float(
                            np.linalg.norm(terminal_object - target_object)
                        ),
                        "terminal_eef_reference_error_m": float(
                            np.linalg.norm(terminal_eef - target_eef)
                        ),
                    })
                    chunks.append(array)
                    summaries.append(summary)

                for candidate in waypoint_chunks(nominal_chunk, waypoint_config):
                    add_candidate(candidate["chunk"], "coarse/%s" % candidate["source"])
                stage_records = []
                top_k = int(config["search"]["top_k_safe_anchors"])
                for magnitude in config["search"]["refinement_magnitudes"]:
                    eligible = eligible_indices(records, minimum_margin)
                    eligible.sort(key=lambda index: candidate_rank(records[index], index))
                    anchors = [chunks[index] for index in eligible[:top_k]]
                    before = len(records)
                    for chunk in refinement_chunks(anchors, float(magnitude)):
                        add_candidate(chunk, "refine_%s" % magnitude)
                    stage_records.append({
                        "magnitude": float(magnitude),
                        "anchor_count": len(anchors),
                        "new_candidate_count": len(records) - before,
                        "eligible_before": len(eligible),
                    })
                selector = select_candidate(records, minimum_margin)
                search_seconds = (time.perf_counter_ns() - search_started) * 1.0e-9
                total_oracle_seconds += search_seconds
                maximum_oracle_seconds = max(maximum_oracle_seconds, search_seconds)
                candidate_total += len(records)
                search_record = {
                    "horizon_actions": horizon,
                    "candidate_count": len(records),
                    "eligible_candidate_count": selector["eligible_candidate_count"],
                    "refinement_stages": stage_records,
                    "candidate_chunk_array_sha256": array_sha256(np.asarray(chunks)),
                    "constraint_margin_array_sha256": array_sha256(np.asarray([
                        record["minimum_all_eight_margin_m"] for record in records
                    ], dtype=np.float64)),
                    "object_error_array_sha256": array_sha256(np.asarray([
                        record["terminal_object_reference_error_m"]
                        for record in records
                    ], dtype=np.float64)),
                    "future_reference_object_relative_to_plate_m": target_object.tolist(),
                    "future_reference_eef_position_m": target_eef.tolist(),
                    "oracle_wall_seconds": search_seconds,
                    "QP_used": False,
                    "learned_model_used": False,
                }
                if selector.get("valid") is not True:
                    failure = {
                        "component": "object_aware_candidate_selector",
                        "step": step, "reason": selector.get("reason"),
                    }
                    suffix_records.append({
                        "step": step, "intervention": True,
                        "nominal_exact_summary": nominal_summary,
                        "search": search_record, "selector": selector,
                        "executed": False,
                    })
                    break
                selected_index = int(selector["selected_candidate_index"])
                selected_chunk = chunks[selected_index]
                executed = selected_chunk[0].copy()
                fresh_rollout = probe.rollout_chunk(env, selected_chunk.tolist())
                selected_summary = _summarize_rollout(
                    fresh_rollout, maximum_obstacle_displacement_m=maximum_motion
                )
                if not (
                    selected_summary["D_sim_raw_safe"]
                    and min(selected_summary[
                        "minimum_all_eight_substep_clearance_m"
                    ]) >= minimum_margin
                ):
                    failure = {
                        "component": "fresh_exact_object_refined_verification",
                        "step": step,
                        "reason": "selected_chunk_not_exactly_safe",
                    }
                    suffix_records.append({
                        "step": step, "intervention": True,
                        "nominal_exact_summary": nominal_summary,
                        "search": search_record, "selector": selector,
                        "selected_exact_summary": selected_summary,
                        "executed": False,
                    })
                    break
            measured = _measured_env_step(env, executed, probe)
            observation = measured.pop("observation")
            clone_match = bool(
                measured["next_state_sha256"]
                == selected_summary["first_transition_next_state_sha256"]
            )
            displacement = float(np.sum(np.abs(
                np.asarray(observation["%s_pos" % setup["obstacle_name"]])
                - initial_obstacle
            )))
            if first_contact_step is None and measured["raw_protected_contact_count"]:
                first_contact_step = step
            if first_car_step is None and displacement > 1.0e-3:
                first_car_step = step
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=step, previous_values=None
            )
            _require(goal["all_satisfied"] is measured["done"],
                     "native goal vector differs")
            safe = bool(
                measured["D_opt_seven_distal_safe"]
                and measured["released_AEGIS_EE_proxy_safe"]
                and measured["raw_protected_contact_count"] == 0
                and measured["maximum_within_step_obstacle_l1_displacement_m"]
                <= maximum_motion
            )
            suffix_records.append({
                "step": step,
                "nominal_action": nominal_actions[0].tolist(),
                "executed_action": executed.tolist(),
                "modified": bool(not np.array_equal(
                    executed[:3], nominal_actions[0, :3]
                )),
                "intervention": intervention,
                "nominal_exact_summary": nominal_summary,
                "search": search_record, "selector": selector,
                "selected_exact_summary": selected_summary,
                "executed_measurement": measured,
                "executed_next_state_matches_exact_clone": clone_match,
                "active_obstacle_l1_displacement_m": displacement,
                "goal_progress": goal,
                "object_relative_to_plate_m": _relative_object_vector(goal),
                "QP_used": False, "learned_model_used": False,
                "executed": True,
            })
            if not safe or not clone_match:
                failure = {
                    "component": "executed_suffix_transition",
                    "step": step,
                    "reason": "unsafe" if not safe else "clone_hash_mismatch",
                }
                break
            if measured["done"]:
                native_success_step = step
                break
        executed_suffix = [row for row in suffix_records if row.get("executed")]
        suffix_minimum = None if not executed_suffix else np.min(np.asarray([
            row["executed_measurement"]["minimum_all_eight_substep_clearance_m"]
            for row in executed_suffix
        ], dtype=np.float64), axis=0)
        prefix_minimum = np.asarray(
            prerequisite_result["closed_loop"][
                "minimum_all_eight_executed_substep_clearance_m"
            ], dtype=np.float64,
        )
        combined_minimum = (
            prefix_minimum if suffix_minimum is None
            else np.minimum(prefix_minimum, suffix_minimum)
        )
        problem_solved = bool(
            failure is None and native_success_step is not None
            and first_contact_step is None and first_car_step is None
            and np.all(combined_minimum >= 0.0)
            and all(row["executed_next_state_matches_exact_clone"]
                    for row in executed_suffix)
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete" if failure is None else "method_failure",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "source": source, "allocation": allocation, "config": config,
            "prerequisite": prerequisite,
            "archived_table1": {
                "path": str(archived_path), "file_sha256": _file_sha256(archived_path),
                "payload_sha256": archived["result_payload_sha256"], "read_only": True,
            },
            "pairing": pairing, "reference": reference,
            "prefix_replay": prefix_receipt,
            "suffix": {
                "start_step": prefix_end,
                "executed_action_count": len(executed_suffix),
                "intervention_count": sum(
                    bool(row.get("intervention")) for row in suffix_records
                ),
                "candidate_count_total": candidate_total,
                "minimum_all_eight_executed_substep_clearance_m": (
                    None if suffix_minimum is None else suffix_minimum.tolist()
                ),
                "native_task_success": native_success_step is not None,
                "native_task_success_step": native_success_step,
                "first_protected_contact_step": first_contact_step,
                "first_paper_CAR_step": first_car_step,
                "total_oracle_wall_seconds": total_oracle_seconds,
                "maximum_oracle_wall_seconds": maximum_oracle_seconds,
                "failure": failure,
            },
            "combined": {
                "minimum_all_eight_executed_substep_clearance_m": (
                    combined_minimum.tolist()
                ),
                "primary_problem_solved": problem_solved,
            },
            "actions": suffix_records,
            "decision": {
                "privileged_object_refined_suffix_e05_go": problem_solved,
                "neural_training_authorized": False,
                "deployable_method_demonstrated": False,
                "QP_success_demonstrated": False,
                "stop_reason": None if problem_solved else (
                    failure["reason"] if failure else
                    "native_task_not_completed_by_registered_suffix"
                ),
            },
            "visual_replay": {
                "status": "pending_separate_state_hash_replay"
            },
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _hash_without(
            result, "result_payload_sha256"
        )
        return result
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--waypoint-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prerequisite-result", type=Path, required=True)
    parser.add_argument("--prerequisite-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        population_manifest=args.population_manifest.resolve(),
        selected_manifest=args.selected_manifest.resolve(),
        archived_root=args.archived_root.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        exact_box_config_path=args.exact_box_config.resolve(),
        source_config_path=args.source_config.resolve(),
        waypoint_config_path=args.waypoint_config.resolve(),
        config_path=args.config.resolve(),
        prerequisite_result_path=args.prerequisite_result.resolve(),
        prerequisite_validation_path=args.prerequisite_validation.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
