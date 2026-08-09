#!/usr/bin/env python3
"""Run the privileged exact-candidate receding oracle on primary E05.

This opt-in diagnostic deliberately bypasses the affine QP.  When the
registered two-step OSC rollout of the nominal action is unsafe, it evaluates
the registered 512-action XYZ grid, chooses the closest exactly safe action,
executes only its first transition, and repeats from the resulting state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.exact_candidate_closed_loop import (
    EXACT_CANDIDATE_RESULT_SCHEMA,
    load_exact_candidate_config,
    select_exact_safe_candidate,
)
from main.multilink_ellipsoid.oracle_affine_closed_loop import (
    ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA_V2,
    summarize_exact_chunk_all_eight,
)
from main.multilink_ellipsoid.two_step_margin import (
    grid_actions,
    load_selected_manifest,
    load_two_step_config,
)
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_affine_oracle_closed_loop_e05 import (
    _chunk_actions,
    _hash_without,
    _measured_env_step,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


PAPER_CAR_THRESHOLD_M = 1.0e-3


def _verify_prerequisite(
    config: Mapping[str, Any], result_path: Path, validation_path: Path
) -> dict[str, Any]:
    prerequisite = config["prerequisite"]
    result = _load(result_path)
    validation = _load(validation_path)
    records = result.get("actions", [])
    final = records[-1] if isinstance(records, list) and records else {}
    _require(
        _file_sha256(result_path)
        == prerequisite["eight_row_result_file_sha256"]
        and _file_sha256(validation_path)
        == prerequisite["eight_row_validation_file_sha256"]
        and result.get("result_payload_sha256")
        == prerequisite["eight_row_result_payload_sha256"]
        and result.get("schema_version")
        == ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA_V2
        and result.get("status") == "method_failure"
        and result.get("closed_loop", {}).get("failure", {}).get("step") == 186
        and final.get("executed") is False
        and final.get("grid", {}).get("candidate_count") == 512
        and final.get("grid", {}).get("all_eight_raw_safe_candidate_count") == 1
        and final.get("certificate", {}).get("reason")
        == "no_8_row_affine_certificate"
        and validation.get("status") == "validated",
        "eight-row affine prerequisite differs",
    )
    return {
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "validation_path": str(validation_path),
        "validation_file_sha256": _file_sha256(validation_path),
        "observed_safe_grid_candidate_at_step_186": True,
        "affine_certificate_failed_at_step_186": True,
    }


def evaluate(
    *, repo_root: Path, population_manifest: Path, selected_manifest: Path,
    archived_root: Path, geometry_config_path: Path, exact_box_config_path: Path,
    source_config_path: Path, config_path: Path, prerequisite_result_path: Path,
    prerequisite_validation_path: Path, expected_commit: str, output_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _runtime_imports,
        array_sha256,
        pairing_record,
        read_jsonl,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_exact_candidate_config(config_path)
    source_config = load_two_step_config(source_config_path)
    for key in (
        "action_limit", "trust_region_linf_action", "grid_points_per_dimension",
        "expected_grid_action_count",
    ):
        _require(
            source_config["sampling"][key] == config["sampling"][key],
            "source sampling differs: %s" % key,
        )
    prerequisite = _verify_prerequisite(
        config, prerequisite_result_path, prerequisite_validation_path
    )
    selected = load_selected_manifest(selected_manifest, source_config)
    case_id = config["primary_case"]["case_id"]
    selected_rows = [item for item in selected if item["case_id"] == case_id]
    _require(len(selected_rows) == 1, "primary selected row differs")
    selected_row = selected_rows[0]
    population_rows = {
        item["case_id"]: item for item in read_jsonl(population_manifest)
    }
    _require(case_id in population_rows, "primary population row missing")
    case = population_rows[case_id]
    archived_path = archived_root / selected_row["archived_relative_path"]
    archived = _load(archived_path)
    _require(
        _file_sha256(archived_path) == selected_row["archived_file_sha256"]
        and archived.get("result_payload_sha256")
        == selected_row["archived_payload_sha256"],
        "immutable Table-1 E05 result differs",
    )
    actions = archived.get("actions")
    _require(
        isinstance(actions, list)
        and len(actions) == config["primary_case"]["expected_action_count"],
        "immutable Table-1 E05 action count differs",
    )
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    env = probe_env = None
    records: list[dict[str, Any]] = []
    failure: Optional[dict[str, Any]] = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        _require(
            setup["obstacle_name"] == archived["obstacle"]["active_name"],
            "active obstacle differs from Table 1",
        )
        pairing = pairing_record(
            case=case,
            selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation,
            task_description=str(task.language),
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
                pairing[key] == archived["pairing"][key],
                "pairing differs: %s" % key,
            )
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config,
            exact_box_config=exact_box_config,
            archived=archived,
            env=env,
            obstacle_name=setup["obstacle_name"],
        )
        probe = SubstepEightConstraintProbe(
            probe_env,
            geometry,
            active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6,
            contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % setup["obstacle_name"]], dtype=np.float64
        ).copy()
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env, goal_atoms, step=-1, previous_values=None
        )
        previous_goal_values = initial_goal["values"]
        maximum_step_motion = float(
            config["activation"]["maximum_per_step_obstacle_l1_displacement_m"]
        )
        minimum_candidate_margin = float(
            config["candidate_selector"]["minimum_all_eight_margin_m"]
        )
        first_car_step = None
        first_contact_step = None
        maximum_episode_obstacle_displacement = 0.0
        total_grid_oracle_seconds = 0.0
        maximum_grid_oracle_seconds = 0.0
        for step in range(len(actions)):
            first = _canonical_action(actions[step], step)
            second = (
                _canonical_action(actions[step + 1], step + 1)
                if step + 1 < len(actions) else None
            )
            nominal_chunk = probe.rollout_chunk(env, _chunk_actions(first, second))
            nominal_summary = summarize_exact_chunk_all_eight(
                nominal_chunk,
                maximum_obstacle_displacement_m=maximum_step_motion,
            )
            intervention = not nominal_summary["safe_for_execution"]
            executed = first.copy()
            grid_record = None
            selector = None
            selected_exact_summary = nominal_summary
            if intervention:
                grid_started = time.perf_counter_ns()
                _, _, candidates = grid_actions(first[:3], source_config)
                candidate_xyz = []
                margins = []
                raw_safe = []
                grid_indexes = []
                rollout_seconds = 0.0
                for grid_index, xyz in enumerate(candidates):
                    candidate = first.copy()
                    candidate[:3] = xyz
                    candidate_chunk = probe.rollout_chunk(
                        env, _chunk_actions(candidate, second)
                    )
                    summary = summarize_exact_chunk_all_eight(
                        candidate_chunk,
                        maximum_obstacle_displacement_m=maximum_step_motion,
                    )
                    candidate_xyz.append(np.asarray(xyz, dtype=np.float64))
                    margins.append(np.asarray(
                        summary["minimum_all_eight_substep_clearance_m"],
                        dtype=np.float64,
                    ))
                    raw_safe.append(bool(summary["D_sim_raw_safe"]))
                    grid_indexes.append(grid_index)
                    rollout_seconds += float(summary["env_step_wall_seconds"])
                xyz_array = np.asarray(candidate_xyz, dtype=np.float64)
                margin_array = np.asarray(margins, dtype=np.float64)
                selector = select_exact_safe_candidate(
                    xyz_array,
                    margin_array,
                    raw_safe,
                    grid_indexes,
                    first[:3],
                    minimum_margin_m=minimum_candidate_margin,
                )
                grid_seconds = (time.perf_counter_ns() - grid_started) * 1.0e-9
                total_grid_oracle_seconds += grid_seconds
                maximum_grid_oracle_seconds = max(
                    maximum_grid_oracle_seconds, grid_seconds
                )
                grid_record = {
                    "candidate_count": len(candidates),
                    "candidate_xyz_array_sha256": array_sha256(xyz_array),
                    "constraint_margin_array_sha256": array_sha256(margin_array),
                    "raw_safe_array_sha256": hashlib.sha256(
                        np.asarray(raw_safe, dtype=np.uint8).tobytes()
                    ).hexdigest(),
                    "eligible_candidate_count": int(
                        selector["eligible_candidate_count"]
                    ),
                    "candidate_rollout_env_step_seconds": rollout_seconds,
                    "grid_oracle_wall_seconds": grid_seconds,
                    "affine_QP_used": False,
                }
                if selector.get("valid") is not True:
                    failure = {
                        "component": "exact_grid_candidate_selector",
                        "step": step,
                        "reason": selector.get("reason"),
                    }
                    records.append({
                        "step": step,
                        "nominal_action": first.tolist(),
                        "nominal_second_action": (
                            None if second is None else second.tolist()
                        ),
                        "nominal_exact_summary": nominal_summary,
                        "intervention": True,
                        "grid": grid_record,
                        "selector": selector,
                        "affine_QP_used": False,
                        "executed": False,
                    })
                    break
                executed[:3] = np.asarray(
                    selector["selected_candidate_xyz"], dtype=np.float64
                )
                selected_chunk = probe.rollout_chunk(
                    env, _chunk_actions(executed, second)
                )
                selected_exact_summary = summarize_exact_chunk_all_eight(
                    selected_chunk,
                    maximum_obstacle_displacement_m=maximum_step_motion,
                )
                fresh_margin = min(
                    selected_exact_summary[
                        "minimum_all_eight_substep_clearance_m"
                    ]
                )
                if not (
                    selected_exact_summary["safe_for_execution"]
                    and fresh_margin >= minimum_candidate_margin
                ):
                    failure = {
                        "component": "fresh_exact_two_step_verification",
                        "step": step,
                        "reason": "selected_grid_candidate_not_exactly_safe",
                    }
                    records.append({
                        "step": step,
                        "nominal_action": first.tolist(),
                        "nominal_second_action": (
                            None if second is None else second.tolist()
                        ),
                        "nominal_exact_summary": nominal_summary,
                        "intervention": True,
                        "grid": grid_record,
                        "selector": selector,
                        "selected_exact_summary": selected_exact_summary,
                        "affine_QP_used": False,
                        "executed": False,
                    })
                    break
            measured = _measured_env_step(env, executed, probe)
            observation = measured.pop("observation")
            clone_match = bool(
                measured["next_state_sha256"]
                == selected_exact_summary["first_transition_next_state_sha256"]
            )
            displacement = float(np.sum(np.abs(
                np.asarray(
                    observation["%s_pos" % setup["obstacle_name"]],
                    dtype=np.float64,
                ) - initial_obstacle_position
            )))
            maximum_episode_obstacle_displacement = max(
                maximum_episode_obstacle_displacement, displacement
            )
            if first_car_step is None and displacement > PAPER_CAR_THRESHOLD_M:
                first_car_step = step
            if first_contact_step is None and measured["raw_protected_contact_count"]:
                first_contact_step = step
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=step, previous_values=previous_goal_values
            )
            previous_goal_values = goal["values"]
            _require(
                goal["all_satisfied"] is measured["done"],
                "native goal vector differs",
            )
            executed_safe = bool(
                measured["D_opt_seven_distal_safe"]
                and measured["released_AEGIS_EE_proxy_safe"]
                and measured["raw_protected_contact_count"] == 0
                and measured["maximum_within_step_obstacle_l1_displacement_m"]
                <= maximum_step_motion
            )
            record = {
                "step": step,
                "nominal_action": first.tolist(),
                "nominal_second_action": None if second is None else second.tolist(),
                "executed_action": executed.tolist(),
                "modified": bool(not np.array_equal(executed[:3], first[:3])),
                "nominal_exact_summary": nominal_summary,
                "intervention": intervention,
                "grid": grid_record,
                "selector": selector,
                "selected_exact_summary": selected_exact_summary,
                "executed_measurement": measured,
                "executed_next_state_matches_exact_clone": clone_match,
                "active_obstacle_l1_displacement_m": displacement,
                "goal_progress": goal,
                "affine_QP_used": False,
                "executed": True,
            }
            records.append(record)
            if not executed_safe or not clone_match:
                failure = {
                    "component": (
                        "executed_transition_safety"
                        if not executed_safe else "cloned_OSC_fidelity"
                    ),
                    "step": step,
                    "reason": (
                        "executed_action_violated_registered_safety_gate"
                        if not executed_safe else "executed_next_state_hash_differs"
                    ),
                }
                break
            if measured["done"]:
                break
        executed_records = [item for item in records if item.get("executed")]
        goal_summary = _goal_progress_summary(initial_goal, executed_records)
        native_success_step = goal_summary["first_all_satisfied_step"]
        native_success = native_success_step is not None
        minimum_all_eight = (
            None if not executed_records else np.min(np.asarray([
                item["executed_measurement"][
                    "minimum_all_eight_substep_clearance_m"
                ] for item in executed_records
            ], dtype=np.float64), axis=0)
        )
        problem_solved = bool(
            failure is None
            and native_success
            and first_contact_step is None
            and first_car_step is None
            and minimum_all_eight is not None
            and np.all(minimum_all_eight >= 0.0)
            and all(
                item["executed_next_state_matches_exact_clone"]
                for item in executed_records
            )
        )
        intervention_count = sum(
            bool(item.get("intervention")) for item in records
        )
        result = {
            "schema_version": EXACT_CANDIDATE_RESULT_SCHEMA,
            "status": "complete" if failure is None else "method_failure",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "prerequisite": prerequisite,
            "case_id": case_id,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": _file_sha256(archived_path),
                "payload_sha256": archived["result_payload_sha256"],
                "read_only": True,
            },
            "pairing": pairing,
            "probe_environment": {
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
                "disabled_image_observables": setup["disabled_images"],
                "scientific_rendering": False,
            },
            "goal_progress_definition": goal_definition,
            "goal_progress_summary": goal_summary,
            "closed_loop": {
                "action_count": len(executed_records),
                "intervention_count": intervention_count,
                "modified_action_count": sum(
                    bool(item.get("modified")) for item in executed_records
                ),
                "first_intervention_step": next((
                    item["step"] for item in records if item.get("intervention")
                ), None),
                "first_protected_contact_step": first_contact_step,
                "first_paper_CAR_step": first_car_step,
                "maximum_active_obstacle_l1_displacement_m": (
                    maximum_episode_obstacle_displacement
                ),
                "minimum_all_eight_executed_substep_clearance_m": (
                    None if minimum_all_eight is None
                    else minimum_all_eight.tolist()
                ),
                "minimum_distal_executed_substep_clearance_m": (
                    None if minimum_all_eight is None
                    else float(np.min(minimum_all_eight[:7]))
                ),
                "minimum_released_AEGIS_EE_proxy_executed_substep_clearance_m": (
                    None if minimum_all_eight is None
                    else float(minimum_all_eight[7])
                ),
                "native_task_success": native_success,
                "native_task_success_step": native_success_step,
                "all_executed_next_states_match_clone": bool(
                    executed_records and all(
                        item["executed_next_state_matches_exact_clone"]
                        for item in executed_records
                    )
                ),
                "total_grid_oracle_wall_seconds": total_grid_oracle_seconds,
                "maximum_grid_oracle_wall_seconds": maximum_grid_oracle_seconds,
                "mean_grid_oracle_wall_seconds_per_intervention": (
                    None if not intervention_count
                    else total_grid_oracle_seconds / intervention_count
                ),
                "affine_QP_used": False,
                "primary_problem_solved": problem_solved,
                "failure": failure,
            },
            "visual_replay": {
                "status": "pending_separate_single_context_state_hash_replay",
                "reason": "avoid_multi_context_OSMesa_corruption_in_scientific_run",
            },
            "actions": records,
            "decision": {
                "privileged_exact_candidate_closed_loop_e05_go": problem_solved,
                "neural_training_authorized": False,
                "deployable_method_demonstrated": False,
                "QP_success_demonstrated": False,
                "stop_reason": None if problem_solved else (
                    failure["reason"] if failure is not None
                    else "native_task_not_completed_collision_free"
                ),
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
        config_path=args.config.resolve(),
        prerequisite_result_path=args.prerequisite_result.resolve(),
        prerequisite_validation_path=args.prerequisite_validation.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
