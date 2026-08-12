#!/usr/bin/env python3
"""Run one immutable case of the cross-case repulsion generalization pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_counterfactual_field_e05 import (
    FixedContinuationProbe,
    _actions_with_correction,
    _archived_action,
    _disable_images,
)
from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_repulsion_generalization_case_result.v1"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _public(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def _archived_actions(result: Mapping[str, Any], start: int, count: int) -> Any:
    import numpy as np

    values = np.asarray(
        [_archived_action(result["actions"], step) for step in range(start, start + count)],
        dtype=np.float64,
    )
    _require(values.shape == (count, 7), "generalization action horizon differs")
    return values


def _candidate(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    correction: Any,
    config: Mapping[str, Any],
    *,
    step_base: int,
    include_trace: bool,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.smooth_field_attribution import smooth_min_value

    delta = np.asarray(correction, dtype=np.float64).reshape(15)
    actions = _actions_with_correction(
        base_actions, delta, float(config["action_space"]["action_limit"])
    )
    rollout = probe.rollout(env, actions, step_base=step_base)
    trace = np.asarray(rollout["clearance_trace_m"], dtype=np.float64)
    return {
        "actions": actions,
        "correction": delta,
        "correction_l2_action": float(np.linalg.norm(delta)),
        "hard_margin_m": float(np.min(trace)),
        "smooth_margin_m": smooth_min_value(
            trace, float(config["field_estimation"]["temperature_m"])
        ),
        "rollout": rollout,
        "_trace": trace,
    }


def _internal_verify(
    instrumented: InstrumentedContinuationProbe,
    env: Any,
    actions: Any,
    config: Mapping[str, Any],
    *,
    step_base: int,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.repulsion_generalization import acceptance

    record = instrumented.rollout_internal(
        env,
        actions,
        expected_substeps=int(
            config["internal_verification"]["expected_mujoco_substeps_per_action"]
        ),
        boundary_tolerance=float(
            config["internal_verification"][
                "ordinary_env_step_boundary_equivalence_tolerance"
            ]
        ),
        step_base=step_base,
    )
    gates = {
        "buffer_%dmm" % int(round(1000.0 * float(buffer))): acceptance(
            record, config, float(buffer)
        )
        for buffer in config["gate"]["internal_substep_clearance_buffers_m"]
    }
    return {"record": _public(record), "gates": gates}


def _fit_smooth(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    correction: Any,
    config: Mapping[str, Any],
    case_index: int,
    iteration: int,
    *,
    step_base: int,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.multi_witness_counterfactual import (
        bounded_paired_directions,
        fit_clearance_rows,
        smooth_min_direction,
    )
    from main.multilink_ellipsoid.smooth_field_attribution import smooth_min_value

    settings = config["field_estimation"]
    radius = float(settings["paired_perturbation_action"])
    current_xyz = np.asarray(base_actions[:5, :3]) + np.asarray(correction).reshape(5, 3)

    def paired(seed: int, count: int) -> tuple[Any, Any, Any, float]:
        directions = bounded_paired_directions(
            count,
            seed + 10000 * case_index,
            current_xyz,
            radius,
            float(config["action_space"]["action_limit"]),
        )
        plus = []
        minus = []
        wall = 0.0
        for direction in directions:
            positive = _candidate(
                probe,
                env,
                base_actions,
                np.asarray(correction) + radius * direction,
                config,
                step_base=step_base,
                include_trace=True,
            )
            negative = _candidate(
                probe,
                env,
                base_actions,
                np.asarray(correction) - radius * direction,
                config,
                step_base=step_base,
                include_trace=True,
            )
            plus.append(positive["_trace"])
            minus.append(negative["_trace"])
            wall += float(positive["rollout"]["env_step_wall_seconds"])
            wall += float(negative["rollout"]["env_step_wall_seconds"])
        return directions, np.asarray(plus), np.asarray(minus), wall

    directions, plus, minus, wall = paired(
        int(settings["direction_seed"]) + iteration,
        int(settings["paired_direction_count_per_iteration"]),
    )
    fit = fit_clearance_rows(directions, plus, minus, radius, float(settings["ridge"]))
    center = _candidate(
        probe,
        env,
        base_actions,
        correction,
        config,
        step_base=step_base,
        include_trace=True,
    )
    smooth = smooth_min_direction(
        center["_trace"].reshape(-1),
        fit["gradients"],
        float(settings["temperature_m"]),
    )
    held_directions, held_plus, held_minus, held_wall = paired(
        int(settings["heldout_direction_seed"]) + iteration,
        int(settings["heldout_direction_count_per_iteration"]),
    )
    targets = []
    predictions = []
    for index, direction in enumerate(held_directions):
        targets.append(
            (
                smooth_min_value(held_plus[index], float(settings["temperature_m"]))
                - smooth_min_value(held_minus[index], float(settings["temperature_m"]))
            )
            / (2.0 * radius)
        )
        predictions.append(float(np.dot(smooth["gradient"], direction)))
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    denominator = float(np.linalg.norm(targets) * np.linalg.norm(predictions))
    return {
        "center": center,
        "gradient": np.asarray(smooth["gradient"], dtype=np.float64),
        "weights": np.asarray(smooth["weights"], dtype=np.float64),
        "heldout": {
            "targets_m_per_action": targets,
            "predictions_m_per_action": predictions,
            "cosine": None
            if denominator <= 1.0e-15
            else float(np.dot(targets, predictions) / denominator),
            "sign_accuracy": float(np.mean(np.sign(targets) == np.sign(predictions))),
            "rmse_m_per_action": float(np.sqrt(np.mean((targets - predictions) ** 2))),
        },
        "rollout_count": int(2 * (len(directions) + len(held_directions)) + 1),
        "probe_wall_seconds": float(
            wall + held_wall + center["rollout"]["env_step_wall_seconds"]
        ),
    }


def _run_smooth(
    probe: FixedContinuationProbe,
    instrumented: InstrumentedContinuationProbe,
    env: Any,
    base_actions: Any,
    config: Mapping[str, Any],
    case_index: int,
    *,
    step_base: int,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.iterative_counterfactual_risk import (
        feasible_correction,
        project_mode_direction,
    )

    correction = np.zeros(15, dtype=np.float64)
    center = _candidate(
        probe, env, base_actions, correction, config, step_base=step_base, include_trace=True
    )
    initial_margin = float(center["hard_margin_m"])
    path = [_public(center)]
    iterations = []
    rollout_count = 1
    probe_wall = float(center["rollout"]["env_step_wall_seconds"])
    path_length = 0.0
    verified = None
    stop_reason = "maximum_iterations"
    for iteration in range(int(config["action_space"]["maximum_iterations"])):
        local = _fit_smooth(
            probe,
            env,
            base_actions,
            correction,
            config,
            case_index,
            iteration,
            step_base=step_base,
        )
        rollout_count += int(local["rollout_count"])
        probe_wall += float(local["probe_wall_seconds"])
        current_xyz = np.asarray(base_actions[:5, :3]) + correction.reshape(5, 3)
        direction = project_mode_direction(
            local["gradient"],
            current_xyz,
            action_limit=float(config["action_space"]["action_limit"]),
            preserve_endpoint=False,
        )
        raw_delta = float(config["action_space"]["trust_radius_action"]) * direction
        candidates = []
        for fraction in config["action_space"]["line_search_fractions"]:
            delta = float(fraction) * raw_delta
            proposed = correction + delta
            step_norm = float(np.linalg.norm(delta))
            if path_length + step_norm > float(
                config["action_space"]["maximum_total_path_length_action"]
            ) + 1.0e-10:
                continue
            if not feasible_correction(
                base_actions[:5, :3],
                proposed,
                radius=float(
                    config["action_space"]["maximum_total_correction_l2_action"]
                ),
                action_limit=float(config["action_space"]["action_limit"]),
                preserve_endpoint=False,
            ):
                continue
            candidate = _candidate(
                probe,
                env,
                base_actions,
                proposed,
                config,
                step_base=step_base,
                include_trace=True,
            )
            candidate["fraction"] = float(fraction)
            candidate["step_norm_action"] = step_norm
            candidates.append(candidate)
            rollout_count += 1
            probe_wall += float(candidate["rollout"]["env_step_wall_seconds"])
        improving = [
            item
            for item in candidates
            if float(item["hard_margin_m"]) > float(center["hard_margin_m"]) + 1.0e-9
        ]
        selected = max(improving, key=lambda item: float(item["hard_margin_m"])) if improving else None
        heldout = local["heldout"]
        iterations.append(
            {
                "iteration": iteration,
                "center_hard_margin_m": float(center["hard_margin_m"]),
                "heldout": _public(heldout),
                "effective_witness_count": float(1.0 / np.sum(local["weights"] ** 2)),
                "selected": None if selected is None else _public(selected),
            }
        )
        if selected is None:
            stop_reason = "no_exact_improving_line_search_step"
            break
        correction = np.asarray(selected["correction"], dtype=np.float64)
        center = selected
        path_length += float(selected["step_norm_action"])
        path.append(_public(center))
        internal = _internal_verify(
            instrumented, env, selected["actions"], config, step_base=step_base
        )
        rollout_count += 1
        if internal["gates"]["buffer_1mm"]:
            verified = internal
            stop_reason = "verified_buffer_1mm_safe"
            break
    final_actions = _actions_with_correction(
        base_actions, correction, float(config["action_space"]["action_limit"])
    )
    final = _internal_verify(instrumented, env, final_actions, config, step_base=step_base)
    rollout_count += 1
    return {
        "correction": correction.tolist(),
        "correction_l2_action": float(np.linalg.norm(correction)),
        "path_length_action": path_length,
        "initial_boundary_margin_m": initial_margin,
        "clearance_gain_m": float(final["record"]["minimum_clearance_m"] - initial_margin),
        "gates": final["gates"],
        "final": final,
        "iterations": iterations,
        "path": path,
        "stop_reason": stop_reason,
        "verified_during_search": verified is not None,
        "rollout_count": rollout_count,
        "probe_env_step_wall_seconds": probe_wall,
    }


def _analytical_direction(
    probe: FixedContinuationProbe,
    env: Any,
    current_actions: Any,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    from main.multilink_ellipsoid.repulsive_flow import five_action_row_model
    from main.multilink_ellipsoid.repulsive_force import softmin_weights
    from main.multilink_ellipsoid.iterative_counterfactual_risk import project_mode_direction

    epsilon = float(config["analytical_field"]["finite_difference_action"])
    base_five = probe.rollout(env, current_actions[:5])
    trace = np.asarray(base_five["clearance_trace_m"], dtype=np.float64)
    jacobian = np.zeros((5, 7, 15), dtype=np.float64)
    wall = float(base_five["env_step_wall_seconds"])
    for variable in range(15):
        slot, dimension = divmod(variable, 3)
        positive = np.asarray(current_actions[:5], dtype=np.float64).copy()
        negative = np.asarray(current_actions[:5], dtype=np.float64).copy()
        positive[slot, dimension] = min(1.0, positive[slot, dimension] + epsilon)
        negative[slot, dimension] = max(-1.0, negative[slot, dimension] - epsilon)
        denominator = float(positive[slot, dimension] - negative[slot, dimension])
        _require(denominator > 0.0, "analytical denominator differs")
        plus = probe.rollout(env, positive)
        minus = probe.rollout(env, negative)
        jacobian[:, :, variable] = (
            np.asarray(plus["clearance_trace_m"]) - np.asarray(minus["clearance_trace_m"])
        ) / denominator
        wall += float(plus["env_step_wall_seconds"] + minus["env_step_wall_seconds"])
    minima, rows = five_action_row_model(jacobian, trace)
    raw = rows.T.dot(
        softmin_weights(minima, float(config["analytical_field"]["temperature_m"]))
    )
    return project_mode_direction(
        raw, current_actions[:5, :3], action_limit=1.0, preserve_endpoint=False
    ), {"local_row_minima_m": minima, "rollout_count": 31, "probe_wall_seconds": wall}


def _run_analytical(
    probe: FixedContinuationProbe,
    instrumented: InstrumentedContinuationProbe,
    env: Any,
    base_actions: Any,
    config: Mapping[str, Any],
    *,
    step_base: int,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.iterative_counterfactual_risk import feasible_correction

    correction = np.zeros(15, dtype=np.float64)
    center = _candidate(
        probe, env, base_actions, correction, config, step_base=step_base, include_trace=True
    )
    initial_margin = float(center["hard_margin_m"])
    path = [_public(center)]
    iterations = []
    path_length = 0.0
    rollout_count = 1
    probe_wall = float(center["rollout"]["env_step_wall_seconds"])
    stop_reason = "maximum_iterations"
    for iteration in range(int(config["action_space"]["maximum_iterations"])):
        current_actions = _actions_with_correction(base_actions, correction, 1.0)
        direction, evidence = _analytical_direction(probe, env, current_actions, config)
        rollout_count += int(evidence["rollout_count"])
        probe_wall += float(evidence["probe_wall_seconds"])
        candidates = []
        for fraction in config["action_space"]["line_search_fractions"]:
            delta = (
                float(config["action_space"]["trust_radius_action"])
                * float(fraction)
                * direction
            )
            proposed = correction + delta
            step_norm = float(np.linalg.norm(delta))
            if path_length + step_norm > 1.0 + 1.0e-10:
                continue
            if not feasible_correction(
                base_actions[:5, :3],
                proposed,
                radius=1.0,
                action_limit=1.0,
                preserve_endpoint=False,
            ):
                continue
            candidate = _candidate(
                probe,
                env,
                base_actions,
                proposed,
                config,
                step_base=step_base,
                include_trace=True,
            )
            candidate["fraction"] = float(fraction)
            candidate["step_norm_action"] = step_norm
            candidates.append(candidate)
            rollout_count += 1
            probe_wall += float(candidate["rollout"]["env_step_wall_seconds"])
        improving = [
            item
            for item in candidates
            if float(item["hard_margin_m"]) > float(center["hard_margin_m"]) + 1.0e-9
        ]
        selected = max(improving, key=lambda item: float(item["hard_margin_m"])) if improving else None
        iterations.append(
            {
                "iteration": iteration,
                "local_row_minima_m": _public(evidence["local_row_minima_m"]),
                "direction": direction.tolist(),
                "selected": None if selected is None else _public(selected),
            }
        )
        if selected is None:
            stop_reason = "no_exact_improving_line_search_step"
            break
        correction = np.asarray(selected["correction"], dtype=np.float64)
        center = selected
        path_length += float(selected["step_norm_action"])
        path.append(_public(center))
    final_actions = _actions_with_correction(base_actions, correction, 1.0)
    final = _internal_verify(instrumented, env, final_actions, config, step_base=step_base)
    rollout_count += 1
    return {
        "correction": correction.tolist(),
        "correction_l2_action": float(np.linalg.norm(correction)),
        "path_length_action": path_length,
        "initial_boundary_margin_m": initial_margin,
        "clearance_gain_m": float(final["record"]["minimum_clearance_m"] - initial_margin),
        "gates": final["gates"],
        "final": final,
        "iterations": iterations,
        "path": path,
        "stop_reason": stop_reason,
        "rollout_count": rollout_count,
        "probe_env_step_wall_seconds": probe_wall,
    }


def _run_exact_search(
    probe: FixedContinuationProbe,
    instrumented: InstrumentedContinuationProbe,
    env: Any,
    base_actions: Any,
    config: Mapping[str, Any],
    case_index: int,
    *,
    step_base: int,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.iterative_counterfactual_risk import feasible_correction

    settings = config["exact_search"]
    rng = np.random.RandomState(int(settings["random_seed"]) + 10000 * case_index)
    nominal = _candidate(
        probe,
        env,
        base_actions,
        np.zeros(15),
        config,
        step_base=step_base,
        include_trace=True,
    )
    initial_margin = float(nominal["hard_margin_m"])
    pool = []
    elite = []
    rollout_count = 1
    probe_wall = float(nominal["rollout"]["env_step_wall_seconds"])
    for generation in range(int(settings["generations"])):
        generated = []
        attempts = 0
        while len(generated) < int(settings["candidate_count_per_generation"]):
            attempts += 1
            if attempts > 100000:
                raise ValueError("exact search candidate generation exhausted")
            raw = rng.normal(size=15)
            norm = float(np.linalg.norm(raw))
            if norm <= 1.0e-12:
                continue
            direction = raw / norm
            if generation == 0 or not elite:
                correction = direction
            else:
                parent = np.asarray(
                    elite[len(generated) % len(elite)]["correction"], dtype=np.float64
                )
                correction = parent + (0.5 ** generation) * direction
                correction_norm = float(np.linalg.norm(correction))
                if correction_norm > 1.0:
                    correction *= 1.0 / correction_norm
            if not feasible_correction(
                base_actions[:5, :3],
                correction,
                radius=1.0,
                action_limit=1.0,
                preserve_endpoint=False,
            ):
                continue
            candidate = _candidate(
                probe,
                env,
                base_actions,
                correction,
                config,
                step_base=step_base,
                include_trace=False,
            )
            generated.append(candidate)
            rollout_count += 1
            probe_wall += float(candidate["rollout"]["env_step_wall_seconds"])
        pool.extend(generated)
        elite = sorted(pool, key=lambda item: float(item["hard_margin_m"]), reverse=True)[
            : int(settings["elite_count"])
        ]
    best = max(pool, key=lambda item: float(item["hard_margin_m"]))
    actions = _actions_with_correction(base_actions, best["correction"], 1.0)
    final = _internal_verify(instrumented, env, actions, config, step_base=step_base)
    rollout_count += 1
    return {
        "correction": _public(best["correction"]),
        "correction_l2_action": float(best["correction_l2_action"]),
        "initial_boundary_margin_m": initial_margin,
        "clearance_gain_m": float(final["record"]["minimum_clearance_m"] - initial_margin),
        "gates": final["gates"],
        "final": final,
        "candidate_count": len(pool),
        "rollout_count": rollout_count,
        "probe_env_step_wall_seconds": probe_wall,
    }


def evaluate(
    *,
    repo_root: Path,
    table1_manifest_path: Path,
    selection_manifest_path: Path,
    archived_root: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    case_index: int,
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
    from main.multilink_ellipsoid.repulsion_generalization import load_cases, load_config
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_config(experiment_config_path)
    selected_cases = load_cases(selection_manifest_path, config)
    _require(0 <= int(case_index) < len(selected_cases), "generalization case index differs")
    selected = selected_cases[int(case_index)]
    matches = [row for row in read_jsonl(table1_manifest_path) if row.get("case_id") == selected["case_id"]]
    _require(len(matches) == 1, "generalization Table-1 case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    _require(
        int(case["case_ordinal"]) == int(selected["table1_case_ordinal"]),
        "generalization Table-1 ordinal differs",
    )
    archived_path = archived_root / selected["archived_result_relative_path"]
    archived = _load(archived_path)
    _require(
        _file_sha256(archived_path) == selected["archived_result_file_sha256"],
        "generalization Table-1 file differs",
    )
    _require(
        archived.get("result_payload_sha256") == selected["archived_result_payload_sha256"],
        "generalization Table-1 payload differs",
    )
    _require(len(archived.get("actions", [])) == 300, "generalization action ledger differs")
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
        _require(str(probe_task.language) == str(task.language), "generalization task differs")
        _require(
            np.array_equal(np.asarray(probe_initial_state), np.asarray(selected_initial_state)),
            "generalization initial state differs",
        )
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(obstacle_name == selected["active_obstacle_name"], "generalization obstacle differs")
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
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
            _require(pairing[key] == archived["pairing"][key], "generalization pairing differs: %s" % key)
        _require(
            pairing["policy_noise_schedule_sha256"]
            == selected["policy_noise_schedule_sha256"],
            "generalization selected policy schedule differs",
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
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        disabled_images = _disable_images(probe_env)
        boundary = FixedContinuationProbe(one_step, obstacle_name, initial_obstacle_position)
        instrumented = InstrumentedContinuationProbe(one_step, obstacle_name, initial_obstacle_position)
        intervention_step = int(selected["intervention_step"])
        for step in range(intervention_step):
            _, _, done, _ = env.step(_archived_action(archived["actions"], step).tolist())
            _require(not bool(done), "generalization episode completed before intervention")
        base_actions = _archived_actions(
            archived,
            intervention_step,
            int(config["state_protocol"]["evaluation_horizon_actions"]),
        )
        raw = _internal_verify(
            instrumented, env, base_actions, config, step_base=intervention_step
        )
        protected_steps = {
            int(event["step"])
            for event in raw["record"]["protected_contacts"]
            if any(
                str(body_name).replace("robot0_", "")
                in str(event.get("protected_geom_name", ""))
                for body_name in selected["protected_contact_bodies"]
            )
        }
        _require(
            int(selected["first_relevant_contact_step"]) in protected_steps,
            "generalization historical protected contact did not reproduce",
        )
        analytical = _run_analytical(
            boundary,
            instrumented,
            env,
            base_actions,
            config,
            step_base=intervention_step,
        )
        smooth = _run_smooth(
            boundary,
            instrumented,
            env,
            base_actions,
            config,
            int(case_index),
            step_base=intervention_step,
        )
        exact = _run_exact_search(
            boundary,
            instrumented,
            env,
            base_actions,
            config,
            int(case_index),
            step_base=intervention_step,
        )
        raw_margin = float(raw["record"]["minimum_clearance_m"])
        for arm in (analytical, smooth, exact):
            arm["clearance_gain_m"] = float(
                arm["final"]["record"]["minimum_clearance_m"] - raw_margin
            )
        arms = {
            "raw_aegis": {
                "correction_l2_action": 0.0,
                "clearance_gain_m": 0.0,
                "gates": raw["gates"],
                "final": raw,
            },
            "analytical": analytical,
            "smooth": smooth,
            "exact_search": exact,
        }
        gate = {
            "raw_reproduces_collision": bool(
                not raw["gates"]["buffer_0mm"]
                and int(selected["first_relevant_contact_step"]) in protected_steps
            ),
            "smooth_improves_clearance": bool(float(smooth["clearance_gain_m"]) > 0.0),
            "smooth_zero_margin_safe": bool(smooth["gates"]["buffer_0mm"]),
            "smooth_one_mm_safe": bool(smooth["gates"]["buffer_1mm"]),
            "exact_search_zero_margin_safe": bool(exact["gates"]["buffer_0mm"]),
            "smooth_beats_or_matches_analytical_margin": bool(
                float(smooth["final"]["record"]["minimum_clearance_m"])
                >= float(analytical["final"]["record"]["minimum_clearance_m"]) - 1.0e-9
            ),
        }
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "case_id": selected["case_id"],
            "case_index": int(case_index),
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "selection_case": selected,
            "selection_manifest": {
                "path": str(selection_manifest_path),
                "file_sha256": _file_sha256(selection_manifest_path),
                "read_only": True,
            },
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": selected["archived_result_file_sha256"],
                "payload_sha256": selected["archived_result_payload_sha256"],
                "read_only": True,
            },
            "geometry_config": geometry_config,
            "geometry": geometry.geometry_record(env),
            "pairing": pairing,
            "probe_environment": {
                "disabled_image_observable_count": disabled_images,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
                "model_timestep_s": float(probe_env.env.model_timestep),
                "control_timestep_s": float(probe_env.env.control_timestep),
            },
            "intervention_step": intervention_step,
            "evaluation_last_step": int(selected["evaluation_last_step"]),
            "nominal_internal_minimum_clearance_m": raw_margin,
            "arms": arms,
            "gate": gate,
            "interpretation": (
                "smooth_repulsion_transfers_with_one_mm_buffer"
                if gate["smooth_one_mm_safe"]
                else "smooth_repulsion_transfers_at_zero_margin"
                if gate["smooth_zero_margin_safe"]
                else "smooth_repulsion_improves_but_does_not_prevent_collision"
                if gate["smooth_improves_clearance"]
                else "smooth_repulsion_does_not_transfer"
            ),
            "rollout_count": int(
                1
                + analytical["rollout_count"]
                + smooth["rollout_count"]
                + exact["rollout_count"]
            ),
            "probe_env_step_wall_seconds": float(
                raw["record"]["env_step_wall_seconds"]
                + analytical["probe_env_step_wall_seconds"]
                + smooth["probe_env_step_wall_seconds"]
                + exact["probe_env_step_wall_seconds"]
            ),
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
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
    parser.add_argument("--table1-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        table1_manifest_path=args.table1_manifest.resolve(),
        selection_manifest_path=args.selection_manifest.resolve(),
        archived_root=args.archived_root.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        case_index=args.case_index,
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "case_id": result["case_id"],
                "interpretation": result["interpretation"],
                "result_payload_sha256": result["result_payload_sha256"],
                "status": result["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
