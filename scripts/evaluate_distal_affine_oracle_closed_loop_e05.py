#!/usr/bin/env python3
"""Run the privileged receding affine-oracle filter through E05 completion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.oracle_affine_closed_loop import (
    ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA,
    load_oracle_affine_closed_loop_config,
    summarize_exact_chunk_all_eight,
)
from main.multilink_ellipsoid.oracle_affine_safe_set import (
    fit_candidate_conditioned_affine_certificate,
    solve_affine_certificate_qp,
)
from main.multilink_ellipsoid.two_step_margin import (
    grid_actions,
    load_selected_manifest,
    load_two_step_config,
)
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_execution_margin_nn_e05 import _geometry
from scripts.evaluate_distal_sitl_candidate_e05 import _disable_probe_images
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


PAPER_CAR_THRESHOLD_M = 0.001


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _build_video_pair(
    runtime: Mapping[str, Any], case: Mapping[str, Any]
) -> tuple[Any, Any, Any, Mapping[str, Any], dict[str, Any]]:
    """Build paired environments while keeping the executed camera enabled."""

    import numpy as np
    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _settle,
    )

    env, task, observation, selected_initial_state = _build_environment(
        runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
    )
    observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
    probe_env, probe_task, probe_observation, probe_initial_state = _build_environment(
        runtime, case, render_resolution=32
    )
    probe_observation = _settle(
        probe_env, probe_observation, TABLE_SETTLE_ACTIONS
    )
    _require(str(probe_task.language) == str(task.language), "probe task differs")
    _require(
        np.array_equal(np.asarray(probe_initial_state), np.asarray(selected_initial_state)),
        "probe initial state differs",
    )
    _require(
        np.array_equal(
            np.asarray(probe_env.sim.get_state().flatten()),
            np.asarray(env.sim.get_state().flatten()),
        ),
        "settled probe simulator state differs",
    )
    obstacle_name, _ = _active_obstacle(env, observation)
    probe_obstacle_name, _ = _active_obstacle(probe_env, probe_observation)
    _require(probe_obstacle_name == obstacle_name, "probe active obstacle differs")
    disabled_probe_images = _disable_probe_images(probe_env)
    return env, probe_env, task, observation, {
        "selected_initial_state": selected_initial_state,
        "obstacle_name": obstacle_name,
        "disabled_probe_image_observables": disabled_probe_images,
    }


def _measured_env_step(env: Any, action: Any, probe: Any) -> dict[str, Any]:
    """Execute the accepted action and measure all eight rows at every substep."""

    import numpy as np
    from main.multilink_ellipsoid.rollout import _base_env, _dynamic_state_vector

    base = _base_env(env)
    trace = [probe._capture(env, 0, "interval_start")]
    original_update = base._update_observables

    def traced_update(*args: Any, **kwargs: Any) -> Any:
        value = original_update(*args, **kwargs)
        trace.append(probe._capture(
            env, len(trace), "after_internal_mujoco_step"
        ))
        return value

    base._update_observables = traced_update
    started = time.perf_counter_ns()
    try:
        observation, reward, done, _ = env.step(action.tolist())
    finally:
        base._update_observables = original_update
    elapsed = (time.perf_counter_ns() - started) * 1.0e-9
    expected = int(base.control_timestep / base.model_timestep)
    _require(len(trace) == expected + 1, "executed substep count differs")
    clearance = np.asarray(
        [item["clearance_m"] for item in trace], dtype=np.float64
    )
    _require(clearance.shape == (expected + 1, 8), "executed row count differs")
    contacts = [
        {**event, "substep_index": int(substep["substep_index"])}
        for substep in trace for event in substep["contact_events"]
    ]
    obstacle_positions = np.asarray(
        [item["obstacle_position_m"] for item in trace], dtype=np.float64
    )
    displacement = np.sum(
        np.abs(obstacle_positions - obstacle_positions[0]), axis=1
    )
    minimum = np.min(clearance, axis=0)
    vector = _dynamic_state_vector(env)
    return {
        "observation": observation,
        "reward": float(reward),
        "done": bool(done),
        "minimum_all_eight_substep_clearance_m": minimum.tolist(),
        "minimum_distal_substep_clearance_m": float(np.min(minimum[:7])),
        "minimum_released_AEGIS_EE_proxy_substep_clearance_m": float(minimum[7]),
        "D_opt_seven_distal_safe": bool(np.all(minimum[:7] >= 0.0)),
        "released_AEGIS_EE_proxy_safe": bool(minimum[7] >= 0.0),
        "raw_protected_contact_count": len(contacts),
        "raw_protected_contact_events": contacts,
        "maximum_within_step_obstacle_l1_displacement_m": float(
            np.max(displacement)
        ),
        "next_state_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
        "env_step_wall_seconds": elapsed,
    }


def _verify_prerequisite(
    config: Mapping[str, Any], result_path: Path, validation_path: Path
) -> dict[str, Any]:
    prerequisite = config["prerequisite"]
    result = _load(result_path)
    validation = _load(validation_path)
    _require(
        _file_sha256(result_path)
        == prerequisite["representation_result_file_sha256"]
        and _file_sha256(validation_path)
        == prerequisite["representation_validation_file_sha256"]
        and result.get("result_payload_sha256")
        == prerequisite["representation_result_payload_sha256"]
        and result.get("decision", {}).get("representation_go") is True
        and validation.get("representation_go") is True,
        "affine representation prerequisite differs",
    )
    return {
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "validation_path": str(validation_path),
        "validation_file_sha256": _file_sha256(validation_path),
        "representation_go": True,
    }


def _chunk_actions(first: Any, second: Optional[Any]) -> list[Any]:
    return [first] if second is None else [first, second]


def evaluate(
    *, repo_root: Path, population_manifest: Path, selected_manifest: Path,
    archived_root: Path, geometry_config_path: Path, exact_box_config_path: Path,
    source_config_path: Path, config_path: Path,
    representation_result_path: Path, representation_validation_path: Path,
    expected_commit: str, output_path: Path, video_path: Path, final_jpg_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        TABLE_VIDEO_FPS,
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _processed_image,
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
    config = load_oracle_affine_closed_loop_config(config_path)
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
        config, representation_result_path, representation_validation_path
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
    env = probe_env = video_writer = None
    video_partial = video_path.with_name(video_path.stem + ".partial" + video_path.suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    final_jpg_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    failure: Optional[dict[str, Any]] = None
    terminal_frame = None
    try:
        env, probe_env, task, observation, setup = _build_video_pair(runtime, case)
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
            _require(pairing[key] == archived["pairing"][key], "pairing differs: %s" % key)
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config, exact_box_config=exact_box_config,
            archived=archived, env=env, obstacle_name=setup["obstacle_name"],
        )
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
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
        if video_partial.exists():
            video_partial.unlink()
        terminal_frame = _processed_image(observation, "agentview_image")
        video_writer = runtime["imageio"].get_writer(
            str(video_partial), fps=TABLE_VIDEO_FPS, codec="libx264",
            macro_block_size=None, pixelformat="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        video_writer.append_data(terminal_frame)
        first_car_step = None
        first_contact_step = None
        maximum_episode_obstacle_displacement = 0.0
        maximum_grid_oracle_seconds = 0.0
        total_grid_oracle_seconds = 0.0
        total_qp_seconds = 0.0
        certificate_settings = config["affine_certificate"]
        maximum_step_motion = float(
            config["activation"]["maximum_per_step_obstacle_l1_displacement_m"]
        )
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
            certificate = None
            qp = None
            grid_record = None
            selected_exact_chunk = nominal_chunk
            selected_exact_summary = nominal_summary
            if intervention:
                grid_started = time.perf_counter_ns()
                lower, upper, candidates = grid_actions(first[:3], source_config)
                grid_margins = []
                grid_raw_ee_safe = []
                grid_indexes = []
                candidate_xyz = []
                candidate_rollout_seconds = 0.0
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
                    grid_margins.append(np.asarray(
                        summary["minimum_all_eight_substep_clearance_m"][:7],
                        dtype=np.float64,
                    ))
                    grid_raw_ee_safe.append(bool(
                        summary["D_sim_raw_safe"]
                        and summary["released_AEGIS_EE_proxy_safe"]
                    ))
                    grid_indexes.append(grid_index)
                    candidate_rollout_seconds += float(summary["env_step_wall_seconds"])
                xyz_array = np.asarray(candidate_xyz, dtype=np.float64)
                margin_array = np.asarray(grid_margins, dtype=np.float64)
                certificate_started = time.perf_counter_ns()
                certificate = fit_candidate_conditioned_affine_certificate(
                    xyz_array, margin_array, first[:3], grid_raw_ee_safe,
                    grid_indexes,
                    one_sided_padding_m=float(
                        certificate_settings["one_sided_padding_m"]
                    ),
                    target_clearance_m=float(
                        certificate_settings["target_clearance_m"]
                    ),
                    postcheck_tolerance_m=float(
                        certificate_settings["coefficient_postcheck_tolerance_m"]
                    ),
                )
                certificate_seconds = (
                    time.perf_counter_ns() - certificate_started
                ) * 1.0e-9
                qp = solve_affine_certificate_qp(
                    first[:3], lower, upper, certificate, config["optimizer"]
                )
                qp_seconds = float(
                    qp.get("diagnostics", {}).get("timing", {}).get(
                        "total_wall_seconds", 0.0
                    )
                )
                total_qp_seconds += qp_seconds
                grid_seconds = (time.perf_counter_ns() - grid_started) * 1.0e-9
                total_grid_oracle_seconds += grid_seconds
                maximum_grid_oracle_seconds = max(
                    maximum_grid_oracle_seconds, grid_seconds
                )
                grid_record = {
                    "candidate_count": len(candidates),
                    "candidate_xyz_array_sha256": array_sha256(xyz_array),
                    "seven_margin_array_sha256": array_sha256(margin_array),
                    "all_eight_raw_safe_candidate_count": int(sum(
                        bool(raw) and bool(np.all(margin_array[index] >= 0.0))
                        for index, raw in enumerate(grid_raw_ee_safe)
                    )),
                    "candidate_rollout_env_step_seconds": candidate_rollout_seconds,
                    "certificate_wall_seconds": certificate_seconds,
                    "grid_oracle_wall_seconds": grid_seconds,
                    "qp_wall_seconds": qp_seconds,
                }
                if not (
                    certificate.get("valid") is True
                    and qp.get("valid") is True
                    and qp.get("diagnostics", {}).get("input_constraint_count") == 7
                ):
                    failure = {
                        "component": "seven_row_affine_oracle_qp",
                        "step": step,
                        "reason": (
                            certificate.get("reason")
                            if not certificate.get("valid") else qp.get("reason")
                        ),
                    }
                    records.append({
                        "step": step, "nominal_action": first.tolist(),
                        "nominal_second_action": None if second is None else second.tolist(),
                        "nominal_exact_summary": nominal_summary,
                        "intervention": True, "grid": grid_record,
                        "certificate": certificate, "qp": qp,
                        "executed": False,
                    })
                    break
                executed[:3] = np.asarray(qp["candidate_xyz"], dtype=np.float64)
                selected_exact_chunk = probe.rollout_chunk(
                    env, _chunk_actions(executed, second)
                )
                selected_exact_summary = summarize_exact_chunk_all_eight(
                    selected_exact_chunk,
                    maximum_obstacle_displacement_m=maximum_step_motion,
                )
                if not selected_exact_summary["safe_for_execution"]:
                    failure = {
                        "component": "fresh_exact_two_step_verification",
                        "step": step,
                        "reason": "affine_qp_candidate_not_exactly_safe",
                    }
                    records.append({
                        "step": step, "nominal_action": first.tolist(),
                        "nominal_second_action": None if second is None else second.tolist(),
                        "nominal_exact_summary": nominal_summary,
                        "intervention": True, "grid": grid_record,
                        "certificate": certificate, "qp": qp,
                        "selected_exact_summary": selected_exact_summary,
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
            _require(goal["all_satisfied"] is measured["done"], "native goal vector differs")
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
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
                "certificate": certificate,
                "qp": qp,
                "selected_exact_summary": selected_exact_summary,
                "executed_measurement": measured,
                "executed_next_state_matches_exact_clone": clone_match,
                "active_obstacle_l1_displacement_m": displacement,
                "goal_progress": goal,
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
        video_writer.close()
        video_writer = None
        video_partial.replace(video_path)
        _require(terminal_frame is not None, "terminal frame missing")
        runtime["imageio"].imwrite(str(final_jpg_path), terminal_frame)
        all_executed_records = [item for item in records if item.get("executed")]
        goal_summary = _goal_progress_summary(initial_goal, all_executed_records)
        native_success_step = goal_summary["first_all_satisfied_step"]
        native_success = native_success_step is not None
        minimum_all_eight = (
            None if not all_executed_records else np.min(np.asarray([
                item["executed_measurement"][
                    "minimum_all_eight_substep_clearance_m"
                ] for item in all_executed_records
            ], dtype=np.float64), axis=0)
        )
        problem_solved = bool(
            failure is None and native_success and first_contact_step is None
            and first_car_step is None and minimum_all_eight is not None
            and np.all(minimum_all_eight >= 0.0)
            and all(
                item["executed_next_state_matches_exact_clone"]
                for item in all_executed_records
            )
        )
        result = {
            "schema_version": ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA,
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
                "disabled_probe_image_observable_count": setup[
                    "disabled_probe_image_observables"
                ],
            },
            "goal_progress_definition": goal_definition,
            "goal_progress_summary": goal_summary,
            "closed_loop": {
                "action_count": len(all_executed_records),
                "intervention_count": sum(
                    bool(item.get("intervention")) for item in records
                ),
                "modified_action_count": sum(
                    bool(item.get("modified")) for item in all_executed_records
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
                    None if minimum_all_eight is None else minimum_all_eight.tolist()
                ),
                "minimum_distal_executed_substep_clearance_m": (
                    None if minimum_all_eight is None else float(np.min(minimum_all_eight[:7]))
                ),
                "minimum_released_AEGIS_EE_proxy_executed_substep_clearance_m": (
                    None if minimum_all_eight is None else float(minimum_all_eight[7])
                ),
                "native_task_success": native_success,
                "native_task_success_step": native_success_step,
                "all_executed_next_states_match_clone": bool(
                    all_executed_records and all(
                        item["executed_next_state_matches_exact_clone"]
                        for item in all_executed_records
                    )
                ),
                "total_grid_oracle_wall_seconds": total_grid_oracle_seconds,
                "maximum_grid_oracle_wall_seconds": maximum_grid_oracle_seconds,
                "total_qp_wall_seconds": total_qp_seconds,
                "mean_qp_wall_seconds": (
                    None if not sum(bool(item.get("intervention")) for item in records)
                    else total_qp_seconds / sum(
                        bool(item.get("intervention")) for item in records
                    )
                ),
                "primary_problem_solved": problem_solved,
                "failure": failure,
            },
            "video": {
                "path": str(video_path),
                "file_sha256": _file_sha256(video_path),
                "frame_count": len(all_executed_records) + 1,
                "fps": TABLE_VIDEO_FPS,
            },
            "final_image": {
                "path": str(final_jpg_path),
                "file_sha256": _file_sha256(final_jpg_path),
                "frame_array_sha256": array_sha256(terminal_frame),
            },
            "actions": records,
            "decision": {
                "privileged_closed_loop_e05_go": problem_solved,
                "neural_training_authorized": False,
                "deployable_method_demonstrated": False,
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
        if video_writer is not None:
            video_writer.close()
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
    parser.add_argument("--representation-result", type=Path, required=True)
    parser.add_argument("--representation-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--final-jpg", type=Path, required=True)
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
        representation_result_path=args.representation_result.resolve(),
        representation_validation_path=args.representation_validation.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
        video_path=args.video.resolve(),
        final_jpg_path=args.final_jpg.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
