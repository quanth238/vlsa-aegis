#!/usr/bin/env python3
"""Run the exact cloned-OSC L5--L7 candidate heuristic on primary E05."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    PAPER_CAR_THRESHOLD_M,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _is_protected_event,
    _load,
    _require,
    _sha256,
)


def _disable_probe_images(env: Any) -> int:
    disabled = 0
    for observable in env.env._observables.values():
        if str(getattr(observable, "modality", "")) == "image":
            observable.set_enabled(False)
            disabled += 1
    return disabled


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    geometry_config_path: Path,
    heuristic_config_path: Path,
    expected_commit: str,
    output_path: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    initial_field_arm: str | None = None,
    field_config_path: Path | None = None,
    replan_prefix_steps: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        TABLE_VIDEO_FPS,
        _active_obstacle,
        _aegis_action,
        _build_environment,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _eef_proxy,
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _policy_observation,
        _processed_image,
        _runtime_imports,
        _server_identity,
        _settle,
        array_sha256,
        max_steps_for_case,
        pairing_record,
        query_seed,
        read_jsonl,
        translational_action,
        validate_case_row,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import (
        DistalSitlCandidateFilter,
        load_sitl_candidate_config,
        summarize_sitl_steps,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from scripts.evaluate_distal_field_mixture_oracle_e05 import (
        _local_basis as _field_local_basis,
        _public_basis as _field_public_basis,
        _public_rollout as _field_public_rollout,
        _run_arm as _run_field_arm,
    )
    from scripts.evaluate_distal_repulsive_force_direction_e05 import (
        TwoActionSlabProbe,
        _json_action,
    )

    started = time.perf_counter_ns()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "archived Table-1 file hash differs")
    _require(
        archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
        "archived Table-1 payload identity differs",
    )
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list)
        and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "archived successful AEGIS action horizon differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary SITL manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    geometry_config = load_shadow_config(geometry_config_path)
    heuristic_config = load_sitl_candidate_config(heuristic_config_path)
    field_config = None
    field_enabled = initial_field_arm is not None
    allowed_field_arms = {
        "fixed_analytical_softmin_repulsion",
        "nonnegative_normal_field_mixture",
        "normal_plus_task_tangent_mixture",
        "unrestricted_six_dimensional_correction",
    }
    if field_enabled:
        from main.multilink_ellipsoid.field_oracle import load_field_oracle_config

        _require(initial_field_arm in allowed_field_arms, "initial field arm differs")
        _require(field_config_path is not None, "field config is required")
        field_config = load_field_oracle_config(field_config_path)
        _require(
            replan_prefix_steps is not None and 1 <= int(replan_prefix_steps) <= 5,
            "field recovery replan prefix must be between one and five",
        )
    _require(geometry_config["case_ids"] == [CASE_ID], "geometry config case differs")
    _require(heuristic_config["case_ids"] == [CASE_ID], "heuristic config case differs")
    nominal_source = heuristic_config["nominal_action_source"]
    hybrid_recovery = nominal_source.startswith(
        "immutable_released_aegis_until_first_sitl_intervention"
    )
    live_policy = nominal_source.startswith("live_pi05_libero") or hybrid_recovery
    if field_enabled:
        _require(hybrid_recovery, "field intervention requires hybrid recovery")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=live_policy)
    env = None
    probe_env = None
    reference_env = None
    video_writer = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_partial = output_path.with_name("episode.partial.mp4")
    video_final = output_path.with_name("episode.mp4")
    final_jpg = output_path.with_name("final.jpg")
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        stale_proxy = _eef_proxy(runtime, observation) if live_policy else None
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
            "settled probe simulator state differs",
        )
        disabled_probe_observables = _disable_probe_images(probe_env)
        obstacle_name, _ = _active_obstacle(env, observation)
        probe_obstacle_name, _ = _active_obstacle(probe_env, probe_observation)
        _require(probe_obstacle_name == obstacle_name, "probe active obstacle differs")
        _require(obstacle_name == archived["obstacle"]["active_name"], "active obstacle differs from Table 1")
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
            _require(pairing[key] == archived["pairing"][key], "SITL pairing field differs: %s" % key)
        if hybrid_recovery:
            pairing["initial_policy_action_chunk_sha256"] = archived["pairing"][
                "initial_policy_action_chunk_sha256"
            ]
            pairing["nominal_prefix_source"] = (
                "immutable_successful_released_aegis_env_step_input"
            )

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
        _require(geometry_record["distal_ellipsoid_count"] == 7, "SITL distal geometry count differs")
        _require(geometry_record["total_constraint_geometry_count"] == 8, "SITL total geometry count differs")
        reference_tracking = heuristic_config["candidate_search"][
            "selection_objective"
        ].startswith("lexicographic_minimum_reference_eef_error")
        reference_observation = None
        if reference_tracking:
            (
                reference_env,
                reference_task,
                reference_observation,
                reference_initial_state,
            ) = _build_environment(runtime, case, render_resolution=32)
            reference_observation = _settle(
                reference_env, reference_observation, TABLE_SETTLE_ACTIONS
            )
            _require(
                str(reference_task.language) == str(task.language),
                "reference task differs",
            )
            _require(
                np.array_equal(
                    np.asarray(reference_initial_state),
                    np.asarray(selected_initial_state),
                ),
                "reference initial state differs",
            )
            _require(
                np.array_equal(
                    np.asarray(reference_env.sim.get_state().flatten()),
                    np.asarray(env.sim.get_state().flatten()),
                ),
                "settled reference simulator state differs",
            )
            _disable_probe_images(reference_env)
        controller = DistalSitlCandidateFilter(
            heuristic_config,
            geometry,
            probe_env,
            active_obstacle_name=obstacle_name,
        )
        two_action_probe = TwoActionSlabProbe(controller.probe)

        client = None
        server_identity = None
        policy_queries = []
        action_plan = collections.deque()
        recovery_active = False
        recovery_activation_step = None
        proxy = stale_proxy
        released_aegis_geometry = None
        q1_diag = None
        if live_policy:
            _require(stale_proxy is not None, "live AEGIS stale EE proxy is unavailable")
            p2 = np.asarray(perception["mvee_center"], dtype=np.float64)
            z_fixed = p2 - np.asarray(stale_proxy["p1"], dtype=np.float64)
            z_norm = float(np.linalg.norm(z_fixed))
            _require(z_norm > 1e-12, "released AEGIS direction is degenerate")
            z_fixed /= z_norm
            released_aegis_geometry = {
                "p2": p2,
                "R2": np.asarray(perception["mvee_rotation"], dtype=np.float64),
                "Q2_diag": np.asarray(perception["mvee_semiaxes"], dtype=np.float64),
                "z_fixed": z_fixed,
            }
            q1_diag = (
                np.asarray([0.06, 0.12, 0.2], dtype=np.float64)
                if any(
                    token in str(task.language)
                    for token in ("orange juice", "milk", "alphabet soup")
                )
                else np.asarray([0.06, 0.12, 0.11], dtype=np.float64)
            )
            client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
            server_identity = _server_identity(client)

        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(env, goal_atoms, step=-1, previous_values=None)
        previous_goal_values = initial_goal["values"]
        contact_authority = _contact_model_authority(env, obstacle_name)
        action_records = []
        filter_records = []
        direct_robot_contact_geoms = set()
        direct_protected_contact_geoms = set()
        first_robot_contact_step = None
        first_protected_contact_step = None
        first_car_step = None
        maximum_displacement = 0.0
        failure = None
        initial_field_record = None
        terminal_frame = _processed_image(observation, "agentview_image")
        frames_written = 0
        if video_partial.exists():
            video_partial.unlink()
        video_writer = runtime["imageio"].get_writer(
            str(video_partial),
            fps=TABLE_VIDEO_FPS,
            codec="libx264",
            macro_block_size=None,
            pixelformat="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        video_writer.append_data(terminal_frame)
        frames_written += 1

        action_limit = max_steps_for_case(case) if live_policy else len(archived_actions)
        for index in range(action_limit):
            nominal_raw = None
            aegis_qp_record = None
            policy_query_index = None
            applied_nominal_source = None
            use_live_action = live_policy and (
                not hybrid_recovery or recovery_active
            )
            if use_live_action:
                if not action_plan:
                    _require(client is not None, "live pi0.5 client is unavailable")
                    query_index = (
                        int(policy_queries[-1]["query_index"]) + 1
                        if policy_queries
                        else (index // int(case.get("replan_steps", 5)) if hybrid_recovery else 0)
                    )
                    seed = query_seed(int(case["policy_noise_seed"]), query_index)
                    policy_input = _policy_observation(
                        runtime,
                        observation,
                        task_description=str(task.language),
                        resize_size=224,
                        rng_seed=seed,
                    )
                    query_started = time.perf_counter_ns()
                    response = client.infer(policy_input)
                    query_wall_seconds = (time.perf_counter_ns() - query_started) * 1e-9
                    returned = np.asarray(response.get("actions"), dtype=np.float64)
                    _require(
                        returned.shape == (int(case["model_action_horizon"]), 7)
                        and np.all(np.isfinite(returned)),
                        "live pi0.5 action chunk differs",
                    )
                    returned_hash = array_sha256(returned)
                    if query_index == 0 and not hybrid_recovery:
                        paired_prefix_length = int(case.get("replan_steps", 5))
                        archived_initial_prefix = np.asarray(
                            [
                                archived_actions[item]["nominal_raw"]
                                for item in range(paired_prefix_length)
                            ],
                            dtype=np.float64,
                        )
                        difference = np.abs(
                            returned[:paired_prefix_length]
                            - archived_initial_prefix
                        )
                        per_dimension_maximum = np.max(difference, axis=0)
                        maximum_raw_difference = float(np.max(difference))
                        first_five_gripper_signs_equal = bool(
                            np.array_equal(
                                np.sign(returned[:5, 6]),
                                np.sign(archived_initial_prefix[:5, 6]),
                            )
                        )
                        pairing["initial_policy_action_chunk_sha256"] = returned_hash
                        pairing["initial_policy_action_equivalence"] = {
                            "exact_hash_match": bool(
                                returned_hash
                                == archived["pairing"][
                                    "initial_policy_action_chunk_sha256"
                                ]
                            ),
                            "maximum_absolute_raw_action_difference": maximum_raw_difference,
                            "per_dimension_maximum_absolute_difference": per_dimension_maximum.tolist(),
                            "compared_executed_prefix_length": paired_prefix_length,
                            "unexecuted_suffix_elementwise_comparison": "unavailable_archived_raw_values",
                            "calibrated_raw_tolerance": 0.005,
                            "first_five_gripper_signs_equal": first_five_gripper_signs_equal,
                            "accepted": bool(
                                maximum_raw_difference <= 0.005
                                and first_five_gripper_signs_equal
                            ),
                            "calibration_source": (
                                "allocation_job_37054_split_jit_equivalence_gate"
                            ),
                        }
                        _require(
                            maximum_raw_difference <= 0.005
                            and first_five_gripper_signs_equal,
                            (
                                "initial live pi0.5 action chunk exceeds calibrated "
                                "Table-1 equivalence: max=%g signs=%s"
                                % (
                                    maximum_raw_difference,
                                    first_five_gripper_signs_equal,
                                )
                            ),
                        )
                    replan_steps = (
                        int(replan_prefix_steps)
                        if field_enabled and recovery_active
                        else int(case.get("replan_steps", 5))
                    )
                    action_plan.extend(returned[item].copy() for item in range(replan_steps))
                    policy_queries.append(
                        {
                            "query_index": query_index,
                            "step": index,
                            "rng_seed": seed,
                            "returned_actions_sha256": returned_hash,
                            "wall_seconds": query_wall_seconds,
                            "server_timing": response.get("server_timing"),
                        }
                    )
                policy_query_index = int(policy_queries[-1]["query_index"])
                applied_nominal_source = (
                    "live_pi05_libero_recovery_then_released_aegis_ee_qp"
                    if hybrid_recovery
                    else "live_pi05_libero_then_released_aegis_ee_qp"
                )
                nominal_raw = np.asarray(action_plan.popleft(), dtype=np.float64)
                translational = translational_action(nominal_raw)
                _require(
                    proxy is not None
                    and released_aegis_geometry is not None
                    and q1_diag is not None,
                    "released AEGIS live state is unavailable",
                )
                nominal_list, aegis_qp_record = _aegis_action(
                    runtime,
                    nominal_translational=translational,
                    proxy=proxy,
                    geometry=released_aegis_geometry,
                    q1_diag=q1_diag,
                    diagnostics_enabled=True,
                )
                nominal = np.asarray(nominal_list, dtype=np.float64)
            else:
                _require(
                    index < len(archived_actions),
                    "immutable AEGIS prefix ended before a distal intervention",
                )
                archived_action = archived_actions[index]
                _require(int(archived_action["step"]) == index, "archived action indexes differ")
                nominal = np.asarray(archived_action["env_step_input"], dtype=np.float64)
                _require(
                    nominal.shape == (7,)
                    and np.array_equal(nominal, np.asarray(archived_action["executed"])),
                    "archived env.step input binding differs",
                )
                applied_nominal_source = (
                    "immutable_successful_released_aegis_env_step_input"
                )
            reference_next_eef = None
            if reference_tracking:
                _require(
                    reference_env is not None and reference_observation is not None,
                    "reference trajectory is unavailable",
                )
                reference_observation, _, _, _ = reference_env.step(nominal.tolist())
                reference_next_eef = np.asarray(
                    reference_observation["robot0_eef_pos"], dtype=np.float64
                )
            field_transition = None
            if field_enabled and index == 185 and not recovery_active:
                _require(field_config is not None, "field configuration is unavailable")
                nominal_second = np.asarray(
                    _json_action(archived_actions[186], 186), dtype=np.float64
                )
                nominal_actions = np.asarray([nominal, nominal_second], dtype=np.float64)
                eef_start = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
                nominal_two = two_action_probe.transition(
                    env, nominal_actions[0], nominal_actions[1]
                )
                nominal_eef_terminal = np.asarray(
                    nominal_two["steps"][-1]["eef_position_m"], dtype=np.float64
                )
                nominal_eef_delta = nominal_eef_terminal - eef_start
                _require(
                    float(np.linalg.norm(nominal_eef_delta)) > 1.0e-8,
                    "field recovery nominal task direction is degenerate",
                )
                initial_basis = _field_local_basis(
                    two_action_probe,
                    env,
                    nominal_actions,
                    epsilon=float(field_config["finite_difference"]["perturbation_action"]),
                    action_limit=float(field_config["search"]["action_limit"]),
                    nominal_eef_start=eef_start,
                    nominal_eef_direction=nominal_eef_delta
                    / np.linalg.norm(nominal_eef_delta),
                )
                field_arm_result = _run_field_arm(
                    str(initial_field_arm),
                    two_action_probe,
                    env,
                    nominal_actions,
                    config=field_config,
                    initial_obstacle_position=initial_obstacle_position,
                    nominal_eef_start=eef_start,
                    nominal_eef_delta=nominal_eef_delta,
                    initial_basis=initial_basis,
                )
                best = field_arm_result["best"]
                if not bool(best["safe_and_task_progressing"]):
                    failure = {
                        "component": "initial_two_action_field_oracle",
                        "step": index,
                        "reason": "no_exact_safe_task_progressing_candidate",
                    }
                    break
                executed = list(best["actions"][0])
                # Never execute a cached search trace. Re-run the selected
                # two-action candidate from the untouched main state, inspect
                # all active-obstacle contacts, and bind execution to this
                # fresh clone.
                fresh_two = two_action_probe.transition(
                    env, best["actions"][0], best["actions"][1]
                )
                fresh_no_contact = all(
                    int(step_record["raw_protected_contact"][
                        "nonpositive_protected_contact_count"
                    ]) == 0
                    for step_record in fresh_two["steps"]
                )
                fresh_displacement = max(
                    float(step_record["active_obstacle_l1_displacement_m"])
                    for step_record in fresh_two["steps"]
                )
                probe_contact_authority = _contact_model_authority(probe_env, obstacle_name)
                verified_first = controller.probe.transition(
                    env, best["actions"][0]
                )
                first_probe_contacts = _detailed_active_obstacle_contacts(
                    probe_env,
                    obstacle_name,
                    step=index,
                    contact_authority=probe_contact_authority,
                )
                _require(
                    first_probe_contacts["status"] == "available",
                    "fresh field contact evidence is unavailable",
                )
                probe_env.step(list(best["actions"][1]))
                second_probe_contacts = _detailed_active_obstacle_contacts(
                    probe_env,
                    obstacle_name,
                    step=index + 1,
                    contact_authority=probe_contact_authority,
                )
                _require(
                    second_probe_contacts["status"] == "available",
                    "fresh field second-step contact evidence is unavailable",
                )
                fresh_robot_contacts = [
                    event
                    for contact_record in (
                        first_probe_contacts,
                        second_probe_contacts,
                    )
                    for event in contact_record["events"]
                    if event.get("other", {}).get("classification") == "robot"
                ]
                _require(
                    float(fresh_two["minimum_row_m"]) >= 0.0
                    and fresh_no_contact
                    and not fresh_robot_contacts
                    and fresh_displacement <= PAPER_CAR_THRESHOLD_M,
                    "fresh selected field rollout is unsafe",
                )
                field_transition = {
                    "state_vector": verified_first["next_state_vector"],
                    "clearances_m": verified_first["next_h_opt_m"],
                    "raw_protected_contact": verified_first[
                        "raw_protected_contact"
                    ],
                }
                # The post-hoc field changes only the physical command.  The
                # released AEGIS virtual direction still advances according
                # to its archived nominal action-185 QP, exactly as it would
                # without the opt-in field correction.
                _require(
                    released_aegis_geometry is not None,
                    "released AEGIS virtual state is unavailable",
                )
                released_aegis_geometry["z_fixed"] = np.asarray(
                    archived_actions[185]["qp"]["z_after"], dtype=np.float64
                )
                filter_step = {
                    "schema_version": "vlsa_distal_field_recovery_step_e05.v1",
                    "step": int(index),
                    "constraint_count": 7,
                    "nominal_released_aegis_action": nominal.tolist(),
                    "nominal_next_clearance_m": nominal_two["steps"][0]["clearances_m"][:7],
                    "nominal_safe": bool(float(nominal_two["minimum_row_m"]) >= 0.0),
                    "nominal_preferred": False,
                    "nominal_raw_protected_contact": nominal_two["steps"][0][
                        "raw_protected_contact"
                    ],
                    "activation": True,
                    "finite_difference": {
                        "used": True,
                        "probe_count": int(initial_basis["rollout_count"]),
                    },
                    "qp": {"used": False, "diagnostics": None},
                    "verification": {
                        "accepted": True,
                        "accepted_source": str(initial_field_arm),
                        "attempts": [
                            {
                                "raw_protected_contact": field_transition[
                                    "raw_protected_contact"
                                ]
                            }
                        ],
                        "first_step_nominal_repeatability": None,
                        "main_env_post_step_checked": False,
                        "main_vs_probe_next_state_max_abs_error": None,
                        "main_vs_probe_next_clearance_max_abs_error_m": None,
                    },
                    "executed_action": executed,
                    "modified": True,
                    "correction_l2": float(
                        np.linalg.norm(
                            np.asarray(executed[:3], dtype=np.float64) - nominal[:3]
                        )
                    ),
                    "timing": {
                        "candidate_env_step_wall_seconds": float(
                            field_arm_result["probe_env_step_wall_seconds"]
                        ),
                        "total_filter_wall_seconds": float(
                            field_arm_result["probe_env_step_wall_seconds"]
                        ),
                    },
                }
                initial_field_record = {
                    "arm": str(initial_field_arm),
                    "trigger_step": int(index),
                    "replan_prefix_steps_after_intervention": int(replan_prefix_steps),
                    "nominal_two_action_rollout": _field_public_rollout(
                        nominal_two, 0.002
                    ),
                    "initial_basis": _field_public_basis(initial_basis, 0.002),
                    "search": field_arm_result,
                    "fresh_selected_verification": _field_public_rollout(
                        fresh_two, 0.002
                    ),
                    "fresh_selected_terminal_robot_contact_events": fresh_robot_contacts,
                    "executed_only_first_verified_action": True,
                }
            else:
                executed, filter_step = controller.filter(
                    env,
                    nominal,
                    step=index,
                    reference_next_eef_position_m=reference_next_eef,
                )
            filter_records.append(filter_step)
            if executed is None:
                failure = {
                    "component": "distal_sitl_candidate_filter",
                    "step": index,
                    "reason": "no_exactly_verified_candidate",
                }
                break
            observation, reward, done, _ = env.step(executed)
            if live_policy:
                proxy = _eef_proxy(runtime, observation)
            if hybrid_recovery and not recovery_active and filter_step["modified"]:
                recovery_active = True
                recovery_activation_step = index
                action_plan.clear()
            try:
                if field_transition is None:
                    controller.verify_executed_transition(env, filter_step)
                else:
                    actual_state = _dynamic_state_vector(env)
                    expected_state = np.asarray(
                        field_transition["state_vector"], dtype=np.float64
                    )
                    state_error = float(
                        np.max(np.abs(actual_state - expected_state))
                    )
                    actual_clearance = controller.probe.clearances(env)[:7]
                    expected_clearance = np.asarray(
                        field_transition["clearances_m"][:7], dtype=np.float64
                    )
                    clearance_error = float(
                        np.max(np.abs(actual_clearance - expected_clearance))
                    )
                    filter_step["verification"]["main_env_post_step_checked"] = True
                    filter_step["verification"][
                        "main_vs_probe_next_state_max_abs_error"
                    ] = state_error
                    filter_step["verification"][
                        "main_vs_probe_next_clearance_max_abs_error_m"
                    ] = clearance_error
                    _require(state_error <= 1.0e-10, "field clone state differs")
                    _require(clearance_error <= 1.0e-10, "field clone clearance differs")
            except ValueError as error:
                failure = {
                    "component": "cloned_osc_fidelity",
                    "step": index,
                    "reason": str(error),
                }
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=index, previous_values=previous_goal_values
            )
            previous_goal_values = goal["values"]
            _require(goal["all_satisfied"] is bool(done), "native goal vector differs")
            contacts = _detailed_active_obstacle_contacts(
                env, obstacle_name, step=index, contact_authority=contact_authority
            )
            _require(contacts["status"] == "available", "active contact evidence unavailable")
            robot_events = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            protected_events = [event for event in robot_events if _is_protected_event(event)]
            if robot_events and first_robot_contact_step is None:
                first_robot_contact_step = index
            if protected_events and first_protected_contact_step is None:
                first_protected_contact_step = index
            for event in robot_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_robot_contact_geoms.add(name)
            for event in protected_events:
                name = event["other"].get("geom_name")
                if isinstance(name, str) and name:
                    direct_protected_contact_geoms.add(name)
            displacement = float(
                np.sum(
                    np.abs(
                        np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
                        - initial_obstacle_position
                    )
                )
            )
            maximum_displacement = max(maximum_displacement, displacement)
            if first_car_step is None and displacement > PAPER_CAR_THRESHOLD_M:
                first_car_step = index
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            frames_written += 1
            action_records.append(
                {
                    "step": index,
                    "nominal_aegis_action_sha256": _sha256(
                        json.dumps(nominal.tolist(), separators=(",", ":"), allow_nan=False).encode("utf-8")
                    ),
                    "nominal_action_source": heuristic_config["nominal_action_source"],
                    "applied_nominal_source": applied_nominal_source,
                    "nominal_raw_policy_action": (
                        None if nominal_raw is None else nominal_raw.tolist()
                    ),
                    "policy_query_index": policy_query_index,
                    "released_aegis_qp": aegis_qp_record,
                    "nominal_released_aegis_action": nominal.tolist(),
                    "executed_sitl_action": list(executed),
                    "reward": float(reward),
                    "done": bool(done),
                    "goal_progress": goal,
                    "eef_position_m": np.asarray(observation["robot0_eef_pos"], dtype=np.float64).tolist(),
                    "active_obstacle_l1_displacement_m": displacement,
                    "robot_contact_events": robot_events,
                    "protected_link_contact_events": protected_events,
                    "filter": filter_step,
                }
            )
            if failure is not None or done:
                break

        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        summary = summarize_sitl_steps(filter_records)
        native_success_step = goal_summary["first_all_satisfied_step"]
        native_success = native_success_step is not None
        paper_car_pass = first_car_step is None
        protected_contact_pass = first_protected_contact_step is None
        robot_contact_pass = first_robot_contact_step is None
        problem_solved = bool(
            failure is None
            and summary["status"] == "complete"
            and summary["all_executed_transitions_match_clone"]
            and protected_contact_pass
            and (robot_contact_pass if field_enabled else True)
            and paper_car_pass
            and native_success
        )
        result = {
            "schema_version": (
                "vlsa_distal_sitl_hybrid_recovery_e05_result.v1"
                if hybrid_recovery
                else (
                    "vlsa_distal_sitl_live_e05_result.v1"
                    if live_policy
                    else "vlsa_distal_sitl_candidate_e05_result.v1"
                )
            ),
            "status": "complete" if failure is None else "method_failure",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": heuristic_config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": heuristic_config,
            "geometry_config": geometry_config,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": ARCHIVED_FILE_SHA256,
                "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
                "read_only": True,
                "original_outcome": {
                    "first_link5_contact_step": 187,
                    "first_paper_car_step": 188,
                    "native_task_success_step": 236,
                },
            },
            "pairing": pairing,
            "policy_server": server_identity,
            "policy_query_count": len(policy_queries),
            "policy_queries": policy_queries,
            "hybrid_recovery": {
                "enabled": hybrid_recovery,
                "activation_step": recovery_activation_step,
                "live_recovery_started": bool(hybrid_recovery and recovery_active),
                "first_live_query_index": (
                    None
                    if not policy_queries
                    else int(policy_queries[0]["query_index"])
                ),
                "immutable_prefix_action_count": (
                    recovery_activation_step + 1
                    if hybrid_recovery and recovery_activation_step is not None
                    else (len(action_records) if hybrid_recovery else 0)
                ),
            },
            "initial_field_recovery": {
                "enabled": field_enabled,
                "record": initial_field_record,
            },
            "probe_environment": {
                "same_bddl_task_episode_and_settled_state": True,
                "disabled_image_observable_count": disabled_probe_observables,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
            },
            "oracle_reference_tracking": {
                "enabled": reference_tracking,
                "source": (
                    "parallel_exact_replay_of_immutable_successful_released_aegis_actions"
                    if reference_tracking
                    else None
                ),
                "selection_metric": (
                    "next_eef_position_l2_m_then_nominal_action_l2"
                    if reference_tracking
                    else None
                ),
            },
            "geometry": geometry_record,
            "action_count": len(action_records),
            "actions": action_records,
            "goal_progress": {**goal_definition, "initial": initial_goal, "summary": goal_summary},
            "raw_simulation_evidence": {
                "first_robot_contact_step": first_robot_contact_step,
                "first_protected_link_contact_step": first_protected_contact_step,
                "direct_robot_contact_geoms": sorted(direct_robot_contact_geoms),
                "direct_protected_link_contact_geoms": sorted(direct_protected_contact_geoms),
                "first_paper_car_step": first_car_step,
                "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
                "paper_car_pass": paper_car_pass,
                "protected_link_contact_pass": protected_contact_pass,
                "all_robot_active_obstacle_contact_pass": robot_contact_pass,
                "native_task_success": native_success,
                "native_task_success_step": native_success_step,
            },
            "filter_summary": summary,
            "terminal_filter_record": (
                None if not filter_records else filter_records[-1]
            ),
            "video": {
                "path": str(video_final),
                "file_sha256": _file_sha256(video_final),
                "frames_written": frames_written,
                "fps": TABLE_VIDEO_FPS,
            },
            "final_jpg": {
                "path": str(final_jpg),
                "file_sha256": _file_sha256(final_jpg),
                "frame_sha256": array_sha256(terminal_frame),
            },
            "primary_problem_solved": problem_solved,
            "failure": failure,
            "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
        }
        result["result_payload_sha256"] = _sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        )
        return result
    finally:
        if video_writer is not None:
            video_writer.close()
        if probe_env is not None:
            probe_env.close()
        if reference_env is not None:
            reference_env.close()
        if env is not None:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--heuristic-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-field-arm")
    parser.add_argument("--field-config", type=Path)
    parser.add_argument("--replan-prefix-steps", type=int)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        heuristic_config_path=args.heuristic_config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
        host=args.host,
        port=args.port,
        initial_field_arm=args.initial_field_arm,
        field_config_path=(
            None if args.field_config is None else args.field_config.resolve()
        ),
        replan_prefix_steps=args.replan_prefix_steps,
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_problem_solved": result["primary_problem_solved"],
                "output": str(args.output.resolve()),
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
