#!/usr/bin/env python3
"""Run the archived E05 unrestricted/normal/tangent post-hoc oracle gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_repulsive_force_direction_e05 import (
    TwoActionSlabProbe,
    _disable_images,
    _json_action,
    _public_rollout,
)
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
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_field_mixture_oracle_e05_result.v1"


def _actions_from_correction(nominal: Any, correction: Any, action_limit: float) -> Any:
    import numpy as np

    actions = np.asarray(nominal, dtype=np.float64).copy()
    delta = np.asarray(correction, dtype=np.float64).reshape(2, 3)
    _require(actions.shape == (2, 7), "oracle nominal action shape differs")
    actions[:, :3] += delta
    _require(
        float(np.max(np.abs(actions[:, :3]))) <= float(action_limit) + 1.0e-12,
        "oracle action bound differs",
    )
    return actions


def _candidate(
    probe: TwoActionSlabProbe,
    env: Any,
    nominal_actions: Any,
    correction: Any,
    *,
    action_limit: float,
    initial_obstacle_position: Any,
    nominal_eef_start: Any,
    nominal_eef_delta: Any,
    minimum_progress_ratio: float,
    temperature_m: float,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.sitl_candidate import _obstacle_root_body_id

    delta = np.asarray(correction, dtype=np.float64)
    actions = _actions_from_correction(nominal_actions, delta, action_limit)
    rollout = probe.transition(env, actions[0], actions[1])
    public = _public_rollout(rollout, temperature_m)
    terminal_eef = np.asarray(rollout["steps"][-1]["eef_position_m"], dtype=np.float64)
    nominal_delta = np.asarray(nominal_eef_delta, dtype=np.float64)
    denominator = float(np.dot(nominal_delta, nominal_delta))
    progress_ratio = (
        0.0
        if denominator <= 1.0e-12
        else float(
            np.dot(
                terminal_eef - np.asarray(nominal_eef_start, dtype=np.float64),
                nominal_delta,
            )
            / denominator
        )
    )
    obstacle_id = _obstacle_root_body_id(
        probe.env.sim.model, probe.one_step_probe.active_obstacle_name
    )
    episode_displacement = float(
        np.sum(
            np.abs(
                np.asarray(probe.env.sim.data.xpos[obstacle_id], dtype=np.float64)
                - np.asarray(initial_obstacle_position, dtype=np.float64)
            )
        )
    )
    no_contact = all(
        int(step["raw_protected_contact"]["nonpositive_protected_contact_count"])
        == 0
        for step in rollout["steps"]
    )
    exact_safe = bool(
        float(rollout["minimum_row_m"]) >= 0.0
        and episode_displacement <= PAPER_CAR_THRESHOLD_M
        and no_contact
    )
    task_progressing = bool(progress_ratio >= float(minimum_progress_ratio))
    return {
        "correction": delta.tolist(),
        "correction_l2": float(np.linalg.norm(delta)),
        "actions": actions.tolist(),
        "exact": public,
        "exact_safe": exact_safe,
        "task_progress_ratio": progress_ratio,
        "task_progressing": task_progressing,
        "safe_and_task_progressing": bool(exact_safe and task_progressing),
        "active_obstacle_l1_displacement_from_episode_start_m": episode_displacement,
        "no_raw_l5_l6_l7_contact": no_contact,
    }


def _local_basis(
    probe: TwoActionSlabProbe,
    env: Any,
    actions: Any,
    *,
    epsilon: float,
    action_limit: float,
    nominal_eef_start: Any,
    nominal_eef_direction: Any,
) -> dict[str, Any]:
    import numpy as np

    center = np.asarray(actions, dtype=np.float64)
    base = probe.transition(env, center[0], center[1])
    plus_rows = []
    minus_rows = []
    plus_progress = []
    minus_progress = []
    denominators = []
    rollout_wall = float(base["env_step_wall_seconds"])
    for variable in range(6):
        slot, dimension = divmod(variable, 3)
        plus = center.copy()
        minus = center.copy()
        plus[slot, dimension] = min(
            float(action_limit), plus[slot, dimension] + float(epsilon)
        )
        minus[slot, dimension] = max(
            -float(action_limit), minus[slot, dimension] - float(epsilon)
        )
        denominator = float(plus[slot, dimension] - minus[slot, dimension])
        _require(denominator > 0.0, "oracle finite-difference denominator differs")
        plus_result = probe.transition(env, plus[0], plus[1])
        minus_result = probe.transition(env, minus[0], minus[1])
        rollout_wall += float(plus_result["env_step_wall_seconds"])
        rollout_wall += float(minus_result["env_step_wall_seconds"])
        plus_rows.append(plus_result["row_minimum_m"])
        minus_rows.append(minus_result["row_minimum_m"])
        start = np.asarray(nominal_eef_start, dtype=np.float64)
        direction = np.asarray(nominal_eef_direction, dtype=np.float64)
        plus_progress.append(
            float(
                np.dot(
                    np.asarray(plus_result["steps"][-1]["eef_position_m"]) - start,
                    direction,
                )
            )
        )
        minus_progress.append(
            float(
                np.dot(
                    np.asarray(minus_result["steps"][-1]["eef_position_m"]) - start,
                    direction,
                )
            )
        )
        denominators.append(denominator)
    plus_rows_array = np.asarray(plus_rows, dtype=np.float64)
    minus_rows_array = np.asarray(minus_rows, dtype=np.float64)
    denominator_array = np.asarray(denominators, dtype=np.float64)
    rows = ((plus_rows_array - minus_rows_array) / denominator_array[:, None]).T
    task_gradient = (
        np.asarray(plus_progress, dtype=np.float64)
        - np.asarray(minus_progress, dtype=np.float64)
    ) / denominator_array
    _require(rows.shape == (7, 6), "oracle clearance basis shape differs")
    _require(task_gradient.shape == (6,), "oracle task basis shape differs")
    return {
        "base": base,
        "clearance_rows_m_per_action": rows,
        "task_progress_gradient_m_per_action": task_gradient,
        "rollout_count": 13,
        "simulated_env_step_count": 26,
        "probe_env_step_wall_seconds": rollout_wall,
    }


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    feasible = bool(candidate["safe_and_task_progressing"])
    if feasible:
        return (
            2,
            -float(candidate["correction_l2"]),
            float(candidate["task_progress_ratio"]),
            float(candidate["exact"]["minimum_row_m"]),
        )
    progress_eligible = bool(candidate["task_progressing"])
    return (
        1 if progress_eligible else 0,
        float(candidate["exact"]["minimum_row_m"]),
        float(candidate["task_progress_ratio"]),
        -float(candidate["correction_l2"]),
    )


def _public_basis(basis: Mapping[str, Any], temperature_m: float) -> dict[str, Any]:
    return {
        "base": _public_rollout(basis["base"], temperature_m),
        "clearance_rows_m_per_action": basis["clearance_rows_m_per_action"].tolist(),
        "task_progress_gradient_m_per_action": basis[
            "task_progress_gradient_m_per_action"
        ].tolist(),
        "rollout_count": int(basis["rollout_count"]),
        "simulated_env_step_count": int(basis["simulated_env_step_count"]),
        "probe_env_step_wall_seconds": float(basis["probe_env_step_wall_seconds"]),
    }


def _arm_directions(
    arm_name: str,
    basis: Mapping[str, Any],
    config: Mapping[str, Any],
    iteration: int,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    from main.multilink_ellipsoid.field_oracle import (
        normal_mixture_directions,
        normal_plus_tangent_directions,
        task_tangent_direction,
        unrestricted_directions,
    )

    rows = np.asarray(basis["clearance_rows_m_per_action"], dtype=np.float64)
    margins = np.asarray(basis["base"]["row_minimum_m"], dtype=np.float64)
    task = np.asarray(basis["task_progress_gradient_m_per_action"], dtype=np.float64)
    search = config["search"]
    seed = int(search["random_seed"]) + 1009 * int(iteration)
    normals = normal_mixture_directions(
        rows,
        margins,
        count=int(search["normal_mixture_direction_count"]),
        seed=seed + 1,
    )
    metadata: dict[str, Any] = {}
    if arm_name == "fixed_analytical_softmin_repulsion":
        # ``normal_mixture_directions`` always puts the deterministic
        # soft-min weighted physical normal first.  Keeping just that row is
        # the strongest non-learned analytical baseline: it may change
        # magnitude through exact relinearization, but may not choose a new
        # mixture or tangent direction.
        directions = normals[:1]
        metadata = {
            "direction_definition": (
                "normalized_transpose_clearance_jacobian_times_"
                "seven_row_softmin_weights"
            )
        }
    elif arm_name == "unrestricted_six_dimensional_correction":
        preferred = [rows.T.dot(np.ones(7)), task, rows.T.dot(np.ones(7)) + task]
        directions = unrestricted_directions(
            int(search["unrestricted_direction_count"]), seed + 2, preferred
        )
    elif arm_name == "nonnegative_normal_field_mixture":
        directions = normals
    elif arm_name == "normal_plus_task_tangent_mixture":
        tangent, active = task_tangent_direction(
            rows,
            margins,
            task,
            active_band_m=float(search["active_row_band_m"]),
        )
        # Use the 16 normals most aligned with the exact local soft-min ascent.
        reference = rows.T.dot(np.exp(-(margins - np.min(margins)) / 0.002))
        reference /= np.linalg.norm(reference)
        order = np.argsort(-(normals @ reference))[:16]
        directions = normal_plus_tangent_directions(
            normals[order], tangent, list(search["normal_tangent_ratios"])
        )
        metadata = {"active_row_indexes": active, "task_tangent_direction": tangent.tolist()}
    else:
        raise ValueError("unknown field-oracle arm")
    return directions, metadata


def _run_arm(
    arm_name: str,
    probe: TwoActionSlabProbe,
    env: Any,
    nominal_actions: Any,
    *,
    config: Mapping[str, Any],
    initial_obstacle_position: Any,
    nominal_eef_start: Any,
    nominal_eef_delta: Any,
    initial_basis: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    search = config["search"]
    success = config["success_definition"]
    temperature_m = 0.002
    action_limit = float(search["action_limit"])
    maximum_norm = float(search["maximum_total_correction_l2"])
    minimum_progress = float(success["minimum_nominal_eef_progress_ratio"])
    correction = np.zeros(6, dtype=np.float64)
    best = _candidate(
        probe,
        env,
        nominal_actions,
        correction,
        action_limit=action_limit,
        initial_obstacle_position=initial_obstacle_position,
        nominal_eef_start=nominal_eef_start,
        nominal_eef_delta=nominal_eef_delta,
        minimum_progress_ratio=minimum_progress,
        temperature_m=temperature_m,
    )
    history = []
    total_rollouts = 1
    total_simulated_steps = 2
    total_probe_wall = float(best["exact"]["env_step_wall_seconds"])
    matched_directions = None
    for iteration in range(int(search["iterations"])):
        actions = _actions_from_correction(nominal_actions, correction, action_limit)
        basis = initial_basis if iteration == 0 else _local_basis(
            probe,
            env,
            actions,
            epsilon=float(config["finite_difference"]["perturbation_action"]),
            action_limit=action_limit,
            nominal_eef_start=nominal_eef_start,
            nominal_eef_direction=np.asarray(nominal_eef_delta, dtype=np.float64)
            / np.linalg.norm(nominal_eef_delta),
        )
        total_rollouts += 0 if iteration == 0 else int(basis["rollout_count"])
        total_simulated_steps += 0 if iteration == 0 else int(basis["simulated_env_step_count"])
        total_probe_wall += 0.0 if iteration == 0 else float(basis["probe_env_step_wall_seconds"])
        directions, direction_metadata = _arm_directions(
            arm_name, basis, config, iteration
        )
        if iteration == 0:
            matched_directions = directions
        candidates = []
        for direction in directions:
            for step_size in search["step_sizes"]:
                proposal = correction + float(step_size) * direction
                if float(np.linalg.norm(proposal)) > maximum_norm + 1.0e-12:
                    continue
                try:
                    candidate = _candidate(
                        probe,
                        env,
                        nominal_actions,
                        proposal,
                        action_limit=action_limit,
                        initial_obstacle_position=initial_obstacle_position,
                        nominal_eef_start=nominal_eef_start,
                        nominal_eef_delta=nominal_eef_delta,
                        minimum_progress_ratio=minimum_progress,
                        temperature_m=temperature_m,
                    )
                except ValueError:
                    continue
                candidate["direction_index"] = len(candidates)
                candidate["step_size"] = float(step_size)
                candidates.append(candidate)
                total_rollouts += 1
                total_simulated_steps += 2
                total_probe_wall += float(candidate["exact"]["env_step_wall_seconds"])
                if _candidate_key(candidate) > _candidate_key(best):
                    best = candidate
        _require(candidates, "field-oracle iteration has no feasible bounded candidates")
        selected = max(candidates, key=_candidate_key)
        correction = np.asarray(selected["correction"], dtype=np.float64)
        history.append(
            {
                "iteration": iteration,
                "basis": _public_basis(basis, temperature_m),
                "direction_count": int(directions.shape[0]),
                "direction_metadata": direction_metadata,
                "candidate_count": len(candidates),
                "selected": selected,
                "best_so_far": best,
            }
        )
    _require(matched_directions is not None, "field-oracle directions are unavailable")
    matched = []
    for radius in config["comparison"]["matched_final_correction_norms"]:
        radius_candidates = []
        # Match arms at identical total correction norms using directions
        # defined at the common nominal state. Later local directions are not
        # transported back to the nominal action coordinates.
        for direction in matched_directions:
            proposal = float(radius) * direction
            try:
                candidate = _candidate(
                    probe,
                    env,
                    nominal_actions,
                    proposal,
                    action_limit=action_limit,
                    initial_obstacle_position=initial_obstacle_position,
                    nominal_eef_start=nominal_eef_start,
                    nominal_eef_delta=nominal_eef_delta,
                    minimum_progress_ratio=minimum_progress,
                    temperature_m=temperature_m,
                )
            except ValueError:
                continue
            radius_candidates.append(candidate)
            total_rollouts += 1
            total_simulated_steps += 2
            total_probe_wall += float(candidate["exact"]["env_step_wall_seconds"])
        matched.append(
            {
                "radius": float(radius),
                "candidate_count": len(radius_candidates),
                "best": None if not radius_candidates else max(radius_candidates, key=_candidate_key),
            }
        )
    return {
        "arm_name": arm_name,
        "iterations": history,
        "best": best,
        "matched_final_correction_norms": matched,
        "rollout_count": total_rollouts,
        "simulated_env_step_count": total_simulated_steps,
        "probe_env_step_wall_seconds": total_probe_wall,
        "gate_pass": bool(best["safe_and_task_progressing"]),
    }


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _runtime_imports,
        _settle,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.field_oracle import load_field_oracle_config
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import (
        SlabbedEightConstraintProbe,
        _obstacle_root_body_id,
    )

    started = time.perf_counter_ns()
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list) and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "Table-1 action horizon differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary manifest row differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    config = load_field_oracle_config(experiment_config_path)
    geometry_config = load_shadow_config(geometry_config_path)
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
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle_id = _obstacle_root_body_id(env.sim.model, obstacle_name)
        initial_obstacle_position = np.asarray(
            env.sim.data.xpos[initial_obstacle_id], dtype=np.float64
        ).copy()
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(env.sim.get_state().flatten()),
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
        disabled_images = {"main": _disable_images(env), "probe": _disable_images(probe_env)}
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
        one_step_probe = SlabbedEightConstraintProbe(
            probe_env,
            geometry,
            clearance_m=0.0,
            active_obstacle_name=obstacle_name,
        )
        probe = TwoActionSlabProbe(one_step_probe)
        for step in range(185):
            observation, _, done, _ = env.step(_json_action(archived_actions[step], step))
            _require(not bool(done), "archived replay completed before step 185")
        nominal_actions = np.asarray(
            [
                _json_action(archived_actions[185], 185),
                _json_action(archived_actions[186], 186),
            ],
            dtype=np.float64,
        )
        eef_start = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        nominal_rollout = probe.transition(env, nominal_actions[0], nominal_actions[1])
        nominal_eef_terminal = np.asarray(
            nominal_rollout["steps"][-1]["eef_position_m"], dtype=np.float64
        )
        nominal_eef_delta = nominal_eef_terminal - eef_start
        _require(float(np.linalg.norm(nominal_eef_delta)) > 1.0e-8, "nominal task direction is degenerate")
        expected = float(config["nominal"]["expected_minimum_margin_m"])
        tolerance = float(config["nominal"]["expected_minimum_tolerance_m"])
        _require(
            abs(float(nominal_rollout["minimum_row_m"]) - expected) <= tolerance,
            "archived dangerous margin differs",
        )
        initial_basis = _local_basis(
            probe,
            env,
            nominal_actions,
            epsilon=float(config["finite_difference"]["perturbation_action"]),
            action_limit=float(config["search"]["action_limit"]),
            nominal_eef_start=eef_start,
            nominal_eef_direction=nominal_eef_delta / np.linalg.norm(nominal_eef_delta),
        )
        arms = {}
        for arm_name in config["comparison"]["arms"]:
            arms[arm_name] = _run_arm(
                arm_name,
                probe,
                env,
                nominal_actions,
                config=config,
                initial_obstacle_position=initial_obstacle_position,
                nominal_eef_start=eef_start,
                nominal_eef_delta=nominal_eef_delta,
                initial_basis=initial_basis,
            )
        unrestricted_pass = bool(arms["unrestricted_six_dimensional_correction"]["gate_pass"])
        structured_pass = bool(
            arms["nonnegative_normal_field_mixture"]["gate_pass"]
            or arms["normal_plus_task_tangent_mixture"]["gate_pass"]
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "case_id": CASE_ID,
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
            "geometry": geometry.geometry_record(env),
            "pairing": pairing,
            "disabled_image_observables": disabled_images,
            "nominal": {
                "actions": nominal_actions.tolist(),
                "rollout": _public_rollout(nominal_rollout, 0.002),
                "eef_start_m": eef_start.tolist(),
                "eef_terminal_m": nominal_eef_terminal.tolist(),
                "eef_delta_m": nominal_eef_delta.tolist(),
                "initial_basis": _public_basis(initial_basis, 0.002),
            },
            "arms": arms,
            "unrestricted_oracle_ceiling_pass": unrestricted_pass,
            "structured_field_gate_pass": structured_pass,
            "primary_problem_solved": bool(unrestricted_pass and structured_pass),
            "interpretation": (
                "structured_field_oracle_supported"
                if structured_pass
                else (
                    "two_action_repair_exists_but_structured_fields_fail"
                    if unrestricted_pass
                    else "bounded_two_action_oracle_no_safe_task_progressing_support"
                )
            ),
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "unrestricted_oracle_ceiling_pass": result["unrestricted_oracle_ceiling_pass"],
                "structured_field_gate_pass": result["structured_field_gate_pass"],
                "primary_problem_solved": result["primary_problem_solved"],
                "interpretation": result["interpretation"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
