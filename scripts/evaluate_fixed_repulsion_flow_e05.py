#!/usr/bin/env python3
"""Compare fixed physical repulsion inside versus after pi0.5 flow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    PAPER_CAR_THRESHOLD_M,
    _atomic_write,
    _canonical,
    _file_sha256,
    _git_identity,
    _is_protected_event,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_fixed_repulsion_flow_e05_result.v1"
LATE_RAMPED_RESULT_SCHEMA = "vlsa_late_ramped_repulsion_flow_e05_result.v1"


def _disable_images(env: Any) -> int:
    disabled = 0
    for observable in env.env._observables.values():
        if str(getattr(observable, "modality", "")) == "image":
            observable.set_enabled(False)
            disabled += 1
    return disabled


def _archived_action(records: list[dict[str, Any]], step: int) -> Any:
    import numpy as np

    record = records[step]
    _require(int(record["step"]) == step, "archived action index differs")
    action = np.asarray(record["env_step_input"], dtype=np.float64)
    _require(action.shape == (7,), "archived action shape differs")
    return action


def _public_rollout(rollout: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "h_opt_m": rollout["h_opt_m"].tolist(),
        "minimum_h_opt_m": float(rollout["minimum_h_opt_m"]),
        "eef_position_m": rollout["eef_position_m"].tolist(),
        "next_state_sha256": list(rollout["next_state_sha256"]),
        "synchronization": dict(rollout["synchronization"]),
        "step_wall_seconds": list(rollout["step_wall_seconds"]),
        "total_env_step_wall_seconds": float(rollout["total_env_step_wall_seconds"]),
        "done_steps": list(rollout["done_steps"]),
    }


def _identify_five_action_model(
    probe: Any,
    env: Any,
    center_actions: Any,
    *,
    perturbation: float,
    action_limit: float,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.predictive_flow import translational_chunk

    center = translational_chunk(center_actions, action_limit=action_limit)
    base = probe.rollout(env, center)
    base_h = np.asarray(base["h_opt_m"][:5, :7], dtype=np.float64)
    jacobian = np.zeros((5, 7, 15), dtype=np.float64)
    probe_wall = float(base["total_env_step_wall_seconds"])
    for variable in range(15):
        slot, dimension = divmod(variable, 3)
        plus = center.copy()
        minus = center.copy()
        plus[slot, dimension] = min(
            action_limit, plus[slot, dimension] + perturbation
        )
        minus[slot, dimension] = max(
            -action_limit, minus[slot, dimension] - perturbation
        )
        denominator = float(plus[slot, dimension] - minus[slot, dimension])
        _require(denominator > 0.0, "repulsive finite-difference denominator differs")
        plus_result = probe.rollout(env, plus)
        minus_result = probe.rollout(env, minus)
        jacobian[:, :, variable] = (
            np.asarray(plus_result["h_opt_m"][:5, :7], dtype=np.float64)
            - np.asarray(minus_result["h_opt_m"][:5, :7], dtype=np.float64)
        ) / denominator
        probe_wall += float(plus_result["total_env_step_wall_seconds"])
        probe_wall += float(minus_result["total_env_step_wall_seconds"])
    return {
        "center_actions": center,
        "base": base,
        "base_h": base_h,
        "jacobian": jacobian,
        "record": {
            "rollout_count": 31,
            "simulated_env_step_count": 310,
            "jacobian_shape": [5, 7, 15],
            "jacobian_sha256": hashlib.sha256(
                np.ascontiguousarray(jacobian, dtype="<f8").tobytes()
            ).hexdigest(),
            "probe_env_step_wall_seconds": probe_wall,
        },
    }


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    expected_commit: str,
    host: str,
    port: int,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _policy_observation,
        _runtime_imports,
        _server_identity,
        _settle,
        array_sha256,
        pairing_record,
        query_seed,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.predictive_flow import (
        _eef_site_id,
        _raw_model_data,
        translational_chunk,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.repulsive_flow import (
        build_repulsive_flow_envelope,
        five_action_row_model,
        fixed_chunk_direction,
        load_repulsive_flow_config,
        posthoc_chunk,
    )
    from main.multilink_ellipsoid.late_ramped_flow import (
        build_schedule_envelope,
        load_late_ramped_flow_config,
        norm_matched_posthoc_chunk,
        surviving_output_correction,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list)
        and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "Table-1 action horizon differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary manifest row differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    raw_experiment_config = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    late_ramped = bool(
        raw_experiment_config.get("schema_version")
        == "vlsa_late_ramped_repulsion_flow_e05.v1"
    )
    config = (
        load_late_ramped_flow_config(experiment_config_path)
        if late_ramped
        else load_repulsive_flow_config(experiment_config_path)
    )
    geometry_config = load_shadow_config(geometry_config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    env = None
    probe_env = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, probe_initial_state = _build_environment(
            runtime, case, render_resolution=32
        )
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
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
            "probe settled state differs",
        )
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        settled_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=settled_state,
        )
        for key in (
            "manifest_row_sha256",
            "initial_state_sha256",
            "initial_observation_sha256",
            "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == archived["pairing"][key], "pairing differs: %s" % key)
        disabled_images = {
            # The live pi0.5 query still requires the main camera observation.
            "main": 0,
            "probe": _disable_images(probe_env),
        }
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {
                "p2": perception["mvee_center"],
                "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        geometry_record = geometry.geometry_record(env)
        one_step_probe = SlabbedEightConstraintProbe(
            probe_env,
            geometry,
            clearance_m=0.0,
            active_obstacle_name=obstacle_name,
        )

        class SevenSlabTrajectoryProbe:
            def __init__(self) -> None:
                self.probe_env = probe_env

            def synchronize(self, main_env: Any) -> dict[str, Any]:
                return one_step_probe.synchronize(main_env)

            def rollout(self, main_env: Any, action_chunk: Any) -> dict[str, Any]:
                actions = np.asarray(action_chunk, dtype=np.float64)
                _require(
                    actions.shape == (10, 7) and np.all(np.isfinite(actions)),
                    "seven-slab probe requires one finite 10x7 chunk",
                )
                synchronization = self.synchronize(main_env)
                clearances = []
                eef_positions = []
                next_state_vectors = []
                step_wall_seconds = []
                done_steps = []
                for index, action in enumerate(actions):
                    step_started = time.perf_counter_ns()
                    _, _, done, _ = self.probe_env.step(action.tolist())
                    step_wall_seconds.append(
                        (time.perf_counter_ns() - step_started) * 1.0e-9
                    )
                    clearances.append(one_step_probe.clearances(self.probe_env)[:7])
                    _, data = _raw_model_data(self.probe_env.sim)
                    eef_positions.append(
                        np.asarray(
                            data.site_xpos[_eef_site_id(self.probe_env)],
                            dtype=np.float64,
                        ).copy()
                    )
                    next_state_vectors.append(_dynamic_state_vector(self.probe_env))
                    if done:
                        done_steps.append(index)
                clearance_array = np.asarray(clearances, dtype=np.float64)
                eef_array = np.asarray(eef_positions, dtype=np.float64)
                _require(clearance_array.shape == (10, 7), "seven-slab clearance trace differs")
                _require(
                    eef_array.shape == (10, 3) and np.all(np.isfinite(eef_array)),
                    "seven-slab end-effector trace differs",
                )
                return {
                    "actions": actions,
                    "h_opt_m": clearance_array,
                    "eef_position_m": eef_array,
                    "minimum_h_opt_m": float(np.min(clearance_array)),
                    "next_state_vectors": next_state_vectors,
                    "next_state_sha256": [
                        hashlib.sha256(value.tobytes()).hexdigest()
                        for value in next_state_vectors
                    ],
                    "synchronization": synchronization,
                    "step_wall_seconds": step_wall_seconds,
                    "total_env_step_wall_seconds": float(sum(step_wall_seconds)),
                    "done_steps": done_steps,
                }

        probe = SevenSlabTrajectoryProbe()
        for step in range(180):
            observation, _, done, _ = env.step(
                _archived_action(archived_actions, step).tolist()
            )
            _require(not bool(done), "archived prefix completed before step 180")

        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
        server_identity = _server_identity(client)
        seed = query_seed(int(case["policy_noise_seed"]), 36)
        policy_input = _policy_observation(
            runtime,
            observation,
            task_description=str(task.language),
            resize_size=224,
            rng_seed=seed,
        )
        nominal_started = time.perf_counter_ns()
        nominal_response = client.infer(policy_input)
        nominal_infer_seconds = (time.perf_counter_ns() - nominal_started) * 1.0e-9
        nominal_raw = np.asarray(nominal_response["actions"], dtype=np.float64)
        _require(nominal_raw.shape == (10, 7), "live nominal chunk shape differs")
        archived_raw_prefix = np.asarray(
            [archived_actions[step]["nominal_raw"] for step in range(180, 185)],
            dtype=np.float64,
        )
        live_difference = np.abs(nominal_raw[:5] - archived_raw_prefix)
        archived_gripper_signs_equal = bool(
            np.array_equal(
                np.sign(nominal_raw[:5, 6]),
                np.sign(archived_raw_prefix[:, 6]),
            )
        )
        nominal_actions = translational_chunk(nominal_raw, action_limit=1.0)
        model = _identify_five_action_model(
            probe,
            env,
            nominal_actions,
            perturbation=float(config["repulsive_direction"]["finite_difference_action"]),
            action_limit=1.0,
        )
        row_minima, rows_7x15 = five_action_row_model(
            model["jacobian"], model["base_h"]
        )
        direction, weights = fixed_chunk_direction(
            row_minima,
            rows_7x15,
            temperature_m=float(config["repulsive_direction"]["softmin_temperature_m"]),
            guided_slots=list(config["flow_guidance"]["guided_action_slots"]),
        )
        if late_ramped:
            guided_slots = list(config["flow_guidance"]["guided_action_slots"])

            def five_metrics(rollout: Mapping[str, Any]) -> dict[str, Any]:
                trace = np.asarray(rollout["h_opt_m"][:5, :7], dtype=np.float64)
                return {
                    "hard_minimum_m": float(np.min(trace)),
                    "row_minimum_m": np.min(trace, axis=0).tolist(),
                    "eef_terminal_position_m": np.asarray(
                        rollout["eef_position_m"][4], dtype=np.float64
                    ).tolist(),
                    "rollout": _public_rollout(rollout),
                }

            nominal_rollout = probe.rollout(env, nominal_actions)
            start_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            ordinary_terminal = np.asarray(
                nominal_rollout["eef_position_m"][4], dtype=np.float64
            )
            ordinary_delta = ordinary_terminal - start_eef
            progress_denominator = float(np.dot(ordinary_delta, ordinary_delta))
            arms: dict[str, Any] = {
                "ordinary_pi05_chunk": {
                    **five_metrics(nominal_rollout),
                    "surviving_output_correction_l2": 0.0,
                    "task_progress_ratio": 1.0,
                }
            }
            scheduled_queries: dict[str, Any] = {}
            for schedule_name in (
                "uniform_final_five",
                "late_linear_last_two",
                "final_step_only",
            ):
                envelope = build_schedule_envelope(
                    nominal_raw,
                    direction,
                    guided_slots=guided_slots,
                    schedule=list(config["flow_guidance"]["schedules"][schedule_name]),
                    action_limit=float(config["flow_guidance"]["action_limit"]),
                )
                guided_input = _policy_observation(
                    runtime,
                    observation,
                    task_description=str(task.language),
                    resize_size=224,
                    rng_seed=seed,
                )
                guided_input["__crfs__"]["scheduled_repulsive_flow_guidance"] = envelope
                infer_started = time.perf_counter_ns()
                response = client.infer(guided_input)
                infer_seconds = (time.perf_counter_ns() - infer_started) * 1.0e-9
                guided_raw = np.asarray(response["actions"], dtype=np.float64)
                _require(guided_raw.shape == (10, 7), "scheduled guided chunk shape differs")
                correction, surviving_norm = surviving_output_correction(
                    nominal_raw, guided_raw, guided_slots
                )
                reference = np.tile(np.asarray(direction, dtype=np.float64), 3)
                flattened = correction.reshape(-1)
                alignment = (
                    0.0
                    if surviving_norm <= 1.0e-12
                    else float(np.dot(flattened, reference) / (surviving_norm * np.linalg.norm(reference)))
                )
                guided_actions = translational_chunk(guided_raw, action_limit=1.0)
                guided_rollout = probe.rollout(env, guided_actions)
                guided_terminal = np.asarray(
                    guided_rollout["eef_position_m"][4], dtype=np.float64
                )
                progress = (
                    0.0
                    if progress_denominator <= 1.0e-12
                    else float(
                        np.dot(guided_terminal - start_eef, ordinary_delta)
                        / progress_denominator
                    )
                )
                matched_raw, matched_norm = norm_matched_posthoc_chunk(
                    nominal_raw,
                    direction,
                    guided_slots=guided_slots,
                    target_correction_l2=surviving_norm,
                    action_limit=1.0,
                )
                _require(
                    abs(matched_norm - surviving_norm)
                    <= float(config["success_definition"]["matching_tolerance_action_l2"]),
                    "norm-matched posthoc correction differs",
                )
                matched_actions = translational_chunk(matched_raw, action_limit=1.0)
                matched_rollout = probe.rollout(env, matched_actions)
                matched_terminal = np.asarray(
                    matched_rollout["eef_position_m"][4], dtype=np.float64
                )
                matched_progress = (
                    0.0
                    if progress_denominator <= 1.0e-12
                    else float(
                        np.dot(matched_terminal - start_eef, ordinary_delta)
                        / progress_denominator
                    )
                )
                arms[schedule_name] = {
                    **five_metrics(guided_rollout),
                    "schedule_action": list(
                        config["flow_guidance"]["schedules"][schedule_name]
                    ),
                    "injected_total_per_slot_action": float(
                        config["flow_guidance"]["injected_total_per_slot_action"]
                    ),
                    "surviving_output_correction": correction.tolist(),
                    "surviving_output_correction_l2": surviving_norm,
                    "survival_fraction_of_unclipped_injected_chunk_l2": (
                        surviving_norm
                        / (
                            np.sqrt(len(guided_slots))
                            * float(config["flow_guidance"]["injected_total_per_slot_action"])
                        )
                    ),
                    "surviving_direction_cosine": alignment,
                    "task_progress_ratio": progress,
                    "inference_seconds": infer_seconds,
                    "policy_diagnostic": response.get(
                        "scheduled_repulsive_flow_guidance"
                    ),
                }
                arms["norm_matched_posthoc_%s" % schedule_name] = {
                    **five_metrics(matched_rollout),
                    "matched_to": schedule_name,
                    "surviving_output_correction_l2": matched_norm,
                    "task_progress_ratio": matched_progress,
                }
                scheduled_queries[schedule_name] = {
                    "guided_action_sha256": array_sha256(guided_raw),
                    "server_timing": response.get("server_timing"),
                }

            uniform_norm = float(
                arms["uniform_final_five"]["surviving_output_correction_l2"]
            )
            late_norm = float(
                arms["late_linear_last_two"]["surviving_output_correction_l2"]
            )
            final_norm = float(
                arms["final_step_only"]["surviving_output_correction_l2"]
            )
            timing_gate_pass = bool(late_norm > uniform_norm and final_norm > uniform_norm)
            result = {
                "schema_version": LATE_RAMPED_RESULT_SCHEMA,
                "status": "complete",
                "scientific_result": True,
                "case_id": CASE_ID,
                "claim_scope": config["claim_scope"],
                "source": source,
                "allocation": allocation,
                "archived_table1": {
                    "path": str(archived_path),
                    "file_sha256": ARCHIVED_FILE_SHA256,
                    "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
                    "read_only": True,
                },
                "config": config,
                "geometry_config": geometry_config,
                "geometry": geometry_record,
                "pairing": pairing,
                "disabled_image_observables": disabled_images,
                "policy_server": server_identity,
                "query": {
                    "query_index": 36,
                    "step": 180,
                    "rng_seed": seed,
                    "nominal_action_sha256": array_sha256(nominal_raw),
                    "maximum_live_vs_archived_prefix_difference": float(
                        np.max(live_difference)
                    ),
                    "per_dimension_live_vs_archived_prefix_maximum": np.max(
                        live_difference, axis=0
                    ).tolist(),
                    "live_vs_archived_gripper_signs_equal": archived_gripper_signs_equal,
                    "live_vs_archived_status": "diagnostic_only_no_late_query_equivalence_claim",
                    "scientific_arm_pairing": "same_live_state_observation_rng_seed_physical_direction_and_horizon",
                    "nominal_infer_seconds": nominal_infer_seconds,
                    "server_nominal_timing": nominal_response.get("server_timing"),
                    "scheduled_queries": scheduled_queries,
                },
                "repulsive_model": {
                    **model["record"],
                    "row_minimum_m": row_minima.tolist(),
                    "fixed_softmin_weights": weights.tolist(),
                    "physical_output_direction": direction.tolist(),
                },
                "arms": arms,
                "timing_gate_pass": timing_gate_pass,
                "execution": {
                    "attempted": False,
                    "reason": config["verification"]["reason"],
                },
                "primary_problem_solved": False,
                "interpretation": (
                    "late_guidance_reduces_denoiser_cancellation"
                    if timing_gate_pass
                    else "late_guidance_does_not_reduce_denoiser_cancellation"
                ),
                "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
            }
            result["result_payload_sha256"] = _sha256(_canonical(result))
            return result
        posthoc_raw = posthoc_chunk(
            nominal_raw,
            direction,
            guided_slots=list(config["flow_guidance"]["guided_action_slots"]),
            total_budget_action=float(config["flow_guidance"]["total_nominal_guidance_budget_action"]),
            action_limit=1.0,
        )
        envelope = build_repulsive_flow_envelope(nominal_raw, direction, config)
        guided_input = _policy_observation(
            runtime,
            observation,
            task_description=str(task.language),
            resize_size=224,
            rng_seed=seed,
        )
        guided_input["__crfs__"]["repulsive_flow_guidance"] = envelope
        guided_started = time.perf_counter_ns()
        guided_response = client.infer(guided_input)
        guided_infer_seconds = (time.perf_counter_ns() - guided_started) * 1.0e-9
        guided_raw = np.asarray(guided_response["actions"], dtype=np.float64)
        _require(guided_raw.shape == (10, 7), "guided flow chunk shape differs")
        nominal_rollout = probe.rollout(env, nominal_actions)
        posthoc_rollout = probe.rollout(
            env, translational_chunk(posthoc_raw, action_limit=1.0)
        )
        guided_actions = translational_chunk(guided_raw, action_limit=1.0)
        guided_rollout = probe.rollout(env, guided_actions)

        def five_metrics(rollout: Mapping[str, Any]) -> dict[str, Any]:
            trace = np.asarray(rollout["h_opt_m"][:5, :7], dtype=np.float64)
            return {
                "hard_minimum_m": float(np.min(trace)),
                "row_minimum_m": np.min(trace, axis=0).tolist(),
                "eef_terminal_position_m": np.asarray(
                    rollout["eef_position_m"][4], dtype=np.float64
                ).tolist(),
                "rollout": _public_rollout(rollout),
            }

        arms = {
            "ordinary_pi05_chunk": five_metrics(nominal_rollout),
            "posthoc_fixed_repulsion": five_metrics(posthoc_rollout),
            "fixed_repulsion_inside_final_flow_steps": five_metrics(guided_rollout),
        }
        ordinary_min = arms["ordinary_pi05_chunk"]["hard_minimum_m"]
        posthoc_min = arms["posthoc_fixed_repulsion"]["hard_minimum_m"]
        guided_min = arms["fixed_repulsion_inside_final_flow_steps"]["hard_minimum_m"]
        direction_gate_pass = bool(
            guided_min >= 0.0 and guided_min > ordinary_min and guided_min > posthoc_min
        )
        execution = {
            "attempted": False,
            "passed": False,
            "reason": "guided_flow_comparative_gate_failed",
        }
        if direction_gate_pass:
            contact_authority = _contact_model_authority(env, obstacle_name)
            start_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            expected_hashes = guided_rollout["next_state_sha256"][:5]
            action_records = []
            maximum_displacement = 0.0
            protected_contact = False
            for offset in range(5):
                observation, _, done, _ = env.step(guided_actions[offset].tolist())
                actual_hash = hashlib.sha256(_dynamic_state_vector(env).tobytes()).hexdigest()
                contacts = _detailed_active_obstacle_contacts(
                    env,
                    obstacle_name,
                    step=180 + offset,
                    contact_authority=contact_authority,
                )
                robot_events = [
                    event
                    for event in contacts["events"]
                    if event.get("other", {}).get("classification") == "robot"
                ]
                protected_events = [event for event in robot_events if _is_protected_event(event)]
                protected_contact = protected_contact or bool(protected_events)
                displacement = float(
                    np.sum(
                        np.abs(
                            np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
                            - initial_obstacle_position
                        )
                    )
                )
                maximum_displacement = max(maximum_displacement, displacement)
                action_records.append(
                    {
                        "step": 180 + offset,
                        "action": guided_actions[offset].tolist(),
                        "actual_state_sha256": actual_hash,
                        "accepted_clone_state_sha256": expected_hashes[offset],
                        "exact_clone_match": actual_hash == expected_hashes[offset],
                        "protected_contact_events": protected_events,
                        "active_obstacle_l1_displacement_m": displacement,
                        "done": bool(done),
                    }
                )
            ordinary_terminal = np.asarray(
                arms["ordinary_pi05_chunk"]["eef_terminal_position_m"], dtype=np.float64
            )
            guided_terminal = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            ordinary_delta = ordinary_terminal - start_eef
            guided_delta = guided_terminal - start_eef
            denominator = float(np.dot(ordinary_delta, ordinary_delta))
            progress_ratio = 0.0 if denominator <= 1e-12 else float(
                np.dot(guided_delta, ordinary_delta) / denominator
            )
            execution = {
                "attempted": True,
                "actions": action_records,
                "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
                "protected_contact": protected_contact,
                "ordinary_eef_progress_ratio": progress_ratio,
                "passed": bool(
                    all(record["exact_clone_match"] for record in action_records)
                    and not protected_contact
                    and maximum_displacement <= PAPER_CAR_THRESHOLD_M
                    and progress_ratio >= 0.5
                ),
                "reason": None,
            }
            if not execution["passed"]:
                execution["reason"] = "executed_guided_prefix_gate_failed"
        overall = bool(direction_gate_pass and execution["passed"])
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "geometry_config": geometry_config,
            "geometry": geometry_record,
            "pairing": pairing,
            "disabled_image_observables": disabled_images,
            "policy_server": server_identity,
            "query": {
                "query_index": 36,
                "step": 180,
                "rng_seed": seed,
                "nominal_action_sha256": array_sha256(nominal_raw),
                "guided_action_sha256": array_sha256(guided_raw),
                "maximum_live_vs_archived_prefix_difference": float(np.max(live_difference)),
                "per_dimension_live_vs_archived_prefix_maximum": np.max(
                    live_difference, axis=0
                ).tolist(),
                "live_vs_archived_gripper_signs_equal": archived_gripper_signs_equal,
                "live_vs_archived_status": "diagnostic_only_no_late_query_equivalence_claim",
                "scientific_arm_pairing": "same_live_state_observation_rng_seed_and_horizon",
                "nominal_infer_seconds": nominal_infer_seconds,
                "guided_infer_seconds": guided_infer_seconds,
                "server_nominal_timing": nominal_response.get("server_timing"),
                "server_guided_timing": guided_response.get("server_timing"),
                "guided_sampler": guided_response.get("repulsive_flow_guidance"),
            },
            "repulsive_model": {
                **model["record"],
                "row_minimum_m": row_minima.tolist(),
                "fixed_softmin_weights": weights.tolist(),
                "physical_output_direction": direction.tolist(),
            },
            "arms": arms,
            "guided_minus_ordinary_minimum_m": guided_min - ordinary_min,
            "guided_minus_posthoc_minimum_m": guided_min - posthoc_min,
            "direction_gate_pass": direction_gate_pass,
            "execution": execution,
            "primary_problem_solved": overall,
            "interpretation": (
                "fixed_repulsion_inside_flow_local_pass"
                if overall
                else (
                    "fixed_repulsion_inside_flow_no_safe_support"
                    if guided_min < 0.0
                    else (
                        "fixed_repulsion_inside_flow_no_clearance_gain"
                        if guided_min <= ordinary_min
                        else (
                            "fixed_repulsion_inside_flow_worse_than_posthoc"
                            if guided_min <= posthoc_min
                            else "fixed_repulsion_inside_flow_execution_no_go"
                        )
                    )
                )
            ),
            "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
        }
        result["result_payload_sha256"] = _sha256(_canonical(result))
        return result
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
        host=args.host,
        port=args.port,
    )
    _atomic_write(args.output.resolve(), result)
    summary = {
        "primary_problem_solved": result["primary_problem_solved"],
        "interpretation": result["interpretation"],
        "result_payload_sha256": result["result_payload_sha256"],
    }
    if "timing_gate_pass" in result:
        summary["timing_gate_pass"] = result["timing_gate_pass"]
    else:
        summary["direction_gate_pass"] = result["direction_gate_pass"]
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
