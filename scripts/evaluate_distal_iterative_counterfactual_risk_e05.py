#!/usr/bin/env python3
"""Decisive pure-risk/constraint/radius oracle diagnostic for E05."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_counterfactual_field_e05 import (
    FixedContinuationProbe,
    _actions_with_correction,
    _archived_action,
    _disable_images,
    _public_rollout,
    _reference_actions,
    _strip_private,
    _validate_reference_result,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_iterative_counterfactual_risk_e05_result.v1"


def _validate_prior(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    result = _load(path)
    expected = config["prior_counterfactual_result"]
    _require(
        result.get("schema_version") == "vlsa_distal_counterfactual_field_e05_result.v1",
        "prior counterfactual schema differs",
    )
    _require(
        result.get("result_payload_sha256") == expected["result_payload_sha256"],
        "prior counterfactual payload differs",
    )
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(
        _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        == claimed,
        "prior counterfactual self-hash differs",
    )
    _require(result["source"]["commit"] == expected["source_commit"], "prior source differs")
    return result


def _candidate(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    correction: Any,
    target_terminal_eef: Any,
    config: Mapping[str, Any],
    *,
    include_trace: bool,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.iterative_counterfactual_risk import (
        active_witness,
        correction_selection_score,
    )

    delta = np.asarray(correction, dtype=np.float64).reshape(15)
    actions = _actions_with_correction(
        base_actions, delta, float(config["action_space"]["action_limit"])
    )
    rollout = probe.rollout(env, actions)
    hard = float(rollout["minimum_clearance_m"])
    terminal_error = float(
        np.linalg.norm(
            np.asarray(rollout["eef_position_trace_m"][-1], dtype=np.float64)
            - np.asarray(target_terminal_eef, dtype=np.float64)
        )
    )
    gates = {}
    for buffer_m in config["gate"]["clearance_buffers_m"]:
        key = "buffer_%dmm" % int(round(1000.0 * float(buffer_m)))
        gates[key] = bool(
            hard >= float(buffer_m)
            and len(rollout["protected_contacts"])
            == int(config["gate"]["protected_raw_contact_count"])
            and float(rollout["maximum_active_obstacle_l1_displacement_m"])
            <= float(config["gate"]["paper_car_threshold_m"])
            and terminal_error <= float(config["gate"]["maximum_terminal_eef_error_m"])
        )
    return {
        "correction": delta.tolist(),
        "correction_l2_action": float(np.linalg.norm(delta)),
        "endpoint_correction_sum": np.sum(delta.reshape(5, 3), axis=0).tolist(),
        "hard_margin_m": hard,
        "pure_risk_m": -hard,
        "terminal_eef_error_m": terminal_error,
        "selection_score_m": correction_selection_score(
            hard, terminal_error, delta, config
        ),
        "active_witness": active_witness(rollout["clearance_trace_m"]),
        "gates": gates,
        "rollout": _public_rollout(rollout, include_trace=include_trace),
        "_record": rollout,
    }


def _best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no candidates are available")
    return max(
        candidates,
        key=lambda item: (
            bool(item["gates"]["buffer_1mm"]),
            bool(item["gates"]["buffer_0mm"]),
            float(item["selection_score_m"]),
            float(item["hard_margin_m"]),
            -float(item["correction_l2_action"]),
        ),
    )


def _field_path(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    target_terminal_eef: Any,
    config: Mapping[str, Any],
    *,
    preserve_endpoint: bool,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.counterfactual_field import paired_chunk_directions
    from main.multilink_ellipsoid.iterative_counterfactual_risk import (
        feasible_correction,
        fit_safety_direction,
        project_mode_direction,
    )

    estimation = config["field_estimation"]
    maximum_radius = max(float(value) for value in config["action_space"]["total_trust_radii_action"])
    correction = np.zeros(15, dtype=np.float64)
    center = _candidate(
        probe, env, base_actions, correction, target_terminal_eef, config, include_trace=True
    )
    accepted = [center]
    iterations = []
    rollout_count = 1
    probe_wall = float(center["rollout"]["env_step_wall_seconds"])
    for iteration in range(int(estimation["maximum_iterations"])):
        current_xyz = np.asarray(base_actions[:5, :3], dtype=np.float64) + correction.reshape(5, 3)
        directions = paired_chunk_directions(
            int(estimation["paired_direction_count_per_radius"]),
            int(estimation["direction_seed"]) + 1000 * int(not preserve_endpoint) + iteration,
            current_xyz,
            float(estimation["paired_perturbation_action"]),
            float(config["action_space"]["action_limit"]),
            preserve_endpoint=preserve_endpoint,
        )
        positive_margins = []
        negative_margins = []
        witness_switches = 0
        pairs = []
        for direction in directions:
            positive = _candidate(
                probe,
                env,
                base_actions,
                correction + float(estimation["paired_perturbation_action"]) * direction,
                target_terminal_eef,
                config,
                include_trace=False,
            )
            negative = _candidate(
                probe,
                env,
                base_actions,
                correction - float(estimation["paired_perturbation_action"]) * direction,
                target_terminal_eef,
                config,
                include_trace=False,
            )
            positive_margins.append(float(positive["hard_margin_m"]))
            negative_margins.append(float(negative["hard_margin_m"]))
            switched = positive["active_witness"] != negative["active_witness"]
            witness_switches += int(switched)
            pairs.append(
                {
                    "direction": direction.tolist(),
                    "positive_hard_margin_m": float(positive["hard_margin_m"]),
                    "negative_hard_margin_m": float(negative["hard_margin_m"]),
                    "positive_witness": positive["active_witness"],
                    "negative_witness": negative["active_witness"],
                    "witness_switched": switched,
                }
            )
            rollout_count += 2
            probe_wall += float(positive["rollout"]["env_step_wall_seconds"])
            probe_wall += float(negative["rollout"]["env_step_wall_seconds"])
        fit = fit_safety_direction(
            directions,
            positive_margins,
            negative_margins,
            float(estimation["paired_perturbation_action"]),
            float(estimation["ridge"]),
        )
        direction = project_mode_direction(
            fit["gradient"],
            current_xyz,
            action_limit=float(config["action_space"]["action_limit"]),
            preserve_endpoint=preserve_endpoint,
        )
        proposals = []
        for fraction in estimation["line_search_fractions"]:
            step = float(estimation["inner_step_action"]) * float(fraction)
            proposed = correction + step * direction
            if not feasible_correction(
                base_actions[:5, :3],
                proposed,
                radius=maximum_radius,
                action_limit=float(config["action_space"]["action_limit"]),
                preserve_endpoint=preserve_endpoint,
            ):
                continue
            candidate = _candidate(
                probe, env, base_actions, proposed, target_terminal_eef, config, include_trace=True
            )
            proposals.append(candidate)
            rollout_count += 1
            probe_wall += float(candidate["rollout"]["env_step_wall_seconds"])
        improving = [
            item for item in proposals if float(item["hard_margin_m"]) > float(center["hard_margin_m"]) + 1e-9
        ]
        selected = max(improving, key=lambda item: float(item["hard_margin_m"])) if improving else None
        iterations.append(
            {
                "iteration": iteration,
                "center": _strip_private(center),
                "pairs": pairs,
                "witness_switch_count": witness_switches,
                "fit": {
                    "gradient": fit["gradient"].tolist(),
                    "unit_direction": direction.tolist(),
                    "fit_rmse": float(fit["fit_rmse"]),
                    "directional_targets": fit["directional_targets"].tolist(),
                    "directional_predictions": fit["directional_predictions"].tolist(),
                },
                "proposals": [_strip_private(item) for item in proposals],
                "selected": None if selected is None else _strip_private(selected),
            }
        )
        if selected is None:
            break
        correction = np.asarray(selected["correction"], dtype=np.float64)
        center = selected
        accepted.append(center)
        if center["gates"]["buffer_1mm"]:
            break
    by_radius = {}
    for radius in config["action_space"]["total_trust_radii_action"]:
        feasible = [
            item for item in accepted if float(item["correction_l2_action"]) <= float(radius) + 1e-10
        ]
        by_radius[str(radius)] = _strip_private(_best(feasible))
    return {
        "preserve_endpoint": preserve_endpoint,
        "accepted_path": [_strip_private(item) for item in accepted],
        "iterations": iterations,
        "best_by_radius": by_radius,
        "best": _strip_private(_best(accepted)),
        "rollout_count": rollout_count,
        "probe_env_step_wall_seconds": probe_wall,
    }


def _analytical_path(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    target_terminal_eef: Any,
    config: Mapping[str, Any],
    *,
    preserve_endpoint: bool,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.iterative_counterfactual_risk import (
        feasible_correction,
        project_mode_direction,
    )
    from main.multilink_ellipsoid.repulsive_flow import five_action_row_model
    from main.multilink_ellipsoid.repulsive_force import softmin_weights

    estimation = config["field_estimation"]
    epsilon = float(config["comparators"]["analytical_finite_difference_action"])
    maximum_radius = max(float(value) for value in config["action_space"]["total_trust_radii_action"])
    correction = np.zeros(15, dtype=np.float64)
    center = _candidate(
        probe, env, base_actions, correction, target_terminal_eef, config, include_trace=True
    )
    accepted = [center]
    iterations = []
    rollout_count = 1
    probe_wall = float(center["rollout"]["env_step_wall_seconds"])
    for iteration in range(int(estimation["maximum_iterations"])):
        current_actions = _actions_with_correction(
            base_actions, correction, float(config["action_space"]["action_limit"])
        )
        base_five = probe.rollout(env, current_actions[:5])
        trace = base_five["clearance_trace_m"]
        jacobian = np.zeros((5, 7, 15), dtype=np.float64)
        rollout_count += 1
        probe_wall += float(base_five["env_step_wall_seconds"])
        for variable in range(15):
            slot, dimension = divmod(variable, 3)
            positive_actions = current_actions[:5].copy()
            negative_actions = current_actions[:5].copy()
            positive_actions[slot, dimension] = min(
                float(config["action_space"]["action_limit"]),
                positive_actions[slot, dimension] + epsilon,
            )
            negative_actions[slot, dimension] = max(
                -float(config["action_space"]["action_limit"]),
                negative_actions[slot, dimension] - epsilon,
            )
            denominator = float(
                positive_actions[slot, dimension] - negative_actions[slot, dimension]
            )
            _require(denominator > 0.0, "analytical denominator differs")
            positive = probe.rollout(env, positive_actions)
            negative = probe.rollout(env, negative_actions)
            jacobian[:, :, variable] = (
                np.asarray(positive["clearance_trace_m"], dtype=np.float64)
                - np.asarray(negative["clearance_trace_m"], dtype=np.float64)
            ) / denominator
            rollout_count += 2
            probe_wall += float(positive["env_step_wall_seconds"])
            probe_wall += float(negative["env_step_wall_seconds"])
        minima, rows = five_action_row_model(jacobian, trace)
        raw = rows.T.dot(softmin_weights(minima, 0.002))
        direction = project_mode_direction(
            raw,
            current_actions[:5, :3],
            action_limit=float(config["action_space"]["action_limit"]),
            preserve_endpoint=preserve_endpoint,
        )
        proposals = []
        for fraction in estimation["line_search_fractions"]:
            proposed = correction + float(estimation["inner_step_action"]) * float(fraction) * direction
            if not feasible_correction(
                base_actions[:5, :3],
                proposed,
                radius=maximum_radius,
                action_limit=float(config["action_space"]["action_limit"]),
                preserve_endpoint=preserve_endpoint,
            ):
                continue
            candidate = _candidate(
                probe, env, base_actions, proposed, target_terminal_eef, config, include_trace=True
            )
            proposals.append(candidate)
            rollout_count += 1
            probe_wall += float(candidate["rollout"]["env_step_wall_seconds"])
        improving = [
            item for item in proposals if float(item["hard_margin_m"]) > float(center["hard_margin_m"]) + 1e-9
        ]
        selected = max(improving, key=lambda item: float(item["hard_margin_m"])) if improving else None
        iterations.append(
            {
                "iteration": iteration,
                "center": _strip_private(center),
                "local_row_minima_m": np.asarray(minima).tolist(),
                "direction": direction.tolist(),
                "proposals": [_strip_private(item) for item in proposals],
                "selected": None if selected is None else _strip_private(selected),
            }
        )
        if selected is None:
            break
        correction = np.asarray(selected["correction"], dtype=np.float64)
        center = selected
        accepted.append(center)
        if center["gates"]["buffer_1mm"]:
            break
    by_radius = {}
    for radius in config["action_space"]["total_trust_radii_action"]:
        feasible = [item for item in accepted if item["correction_l2_action"] <= float(radius) + 1e-10]
        by_radius[str(radius)] = _strip_private(_best(feasible))
    return {
        "preserve_endpoint": preserve_endpoint,
        "accepted_path": [_strip_private(item) for item in accepted],
        "iterations": iterations,
        "best_by_radius": by_radius,
        "best": _strip_private(_best(accepted)),
        "rollout_count": rollout_count,
        "probe_env_step_wall_seconds": probe_wall,
    }


def _random_and_derivative_free(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    target_terminal_eef: Any,
    config: Mapping[str, Any],
    *,
    preserve_endpoint: bool,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.counterfactual_field import (
        endpoint_projection_matrix,
        paired_chunk_directions,
    )
    from main.multilink_ellipsoid.iterative_counterfactual_risk import feasible_correction

    comparators = config["comparators"]
    base_xyz = np.asarray(base_actions[:5, :3], dtype=np.float64)
    headroom = float(config["action_space"]["action_limit"]) - np.abs(base_xyz.reshape(15))
    fixed = headroom <= 1e-10
    projection = (
        endpoint_projection_matrix(fixed)
        if preserve_endpoint
        else np.diag((~fixed).astype(np.float64))
    )
    output = {}
    total_rollouts = 0
    total_wall = 0.0
    for radius_index, radius_value in enumerate(config["action_space"]["total_trust_radii_action"]):
        radius = float(radius_value)
        directions = paired_chunk_directions(
            int(comparators["matched_random_candidates_per_radius"]),
            int(comparators["matched_random_seed"]) + 1000 * int(not preserve_endpoint) + radius_index,
            base_xyz,
            radius,
            float(config["action_space"]["action_limit"]),
            preserve_endpoint=preserve_endpoint,
        )
        random_candidates = []
        for direction in directions:
            candidate = _candidate(
                probe, env, base_actions, radius * direction, target_terminal_eef, config, include_trace=False
            )
            random_candidates.append(candidate)
            total_rollouts += 1
            total_wall += float(candidate["rollout"]["env_step_wall_seconds"])
        rng = np.random.RandomState(
            int(comparators["derivative_free_seed"]) + 1000 * int(not preserve_endpoint) + radius_index
        )
        pool = []
        elite = []
        for generation in range(int(comparators["derivative_free_generations"])):
            generation_candidates = []
            attempts = 0
            while len(generation_candidates) < int(comparators["derivative_free_candidates_per_radius"]):
                attempts += 1
                if attempts > 100000:
                    raise ValueError("derivative-free candidate generation exhausted")
                raw = rng.normal(size=15)
                raw = projection.dot(raw)
                norm = float(np.linalg.norm(raw))
                if norm <= 1e-12:
                    continue
                direction = raw / norm
                if generation == 0 or not elite:
                    correction = radius * direction
                else:
                    parent = np.asarray(elite[len(generation_candidates) % len(elite)]["correction"], dtype=np.float64)
                    correction = projection.dot(
                        parent + radius * (0.5 ** generation) * direction
                    )
                    correction_norm = float(np.linalg.norm(correction))
                    if correction_norm > radius:
                        correction *= radius / correction_norm
                if not feasible_correction(
                    base_xyz,
                    correction,
                    radius=radius,
                    action_limit=float(config["action_space"]["action_limit"]),
                    preserve_endpoint=preserve_endpoint,
                ):
                    continue
                candidate = _candidate(
                    probe, env, base_actions, correction, target_terminal_eef, config, include_trace=False
                )
                generation_candidates.append(candidate)
                total_rollouts += 1
                total_wall += float(candidate["rollout"]["env_step_wall_seconds"])
            pool.extend(generation_candidates)
            elite = sorted(
                pool,
                key=lambda item: (
                    bool(item["gates"]["buffer_1mm"]),
                    bool(item["gates"]["buffer_0mm"]),
                    float(item["selection_score_m"]),
                ),
                reverse=True,
            )[: int(comparators["derivative_free_elite_count"])]
        output[str(radius_value)] = {
            "random_best": _strip_private(_best(random_candidates)),
            "random_safe_0mm_count": sum(item["gates"]["buffer_0mm"] for item in random_candidates),
            "random_safe_1mm_count": sum(item["gates"]["buffer_1mm"] for item in random_candidates),
            "derivative_free_best": _strip_private(_best(pool)),
            "derivative_free_safe_0mm_count": sum(item["gates"]["buffer_0mm"] for item in pool),
            "derivative_free_safe_1mm_count": sum(item["gates"]["buffer_1mm"] for item in pool),
            "derivative_free_candidate_count": len(pool),
        }
    return {
        "preserve_endpoint": preserve_endpoint,
        "by_radius": output,
        "rollout_count": total_rollouts,
        "probe_env_step_wall_seconds": total_wall,
    }


def _any_safe(value: Any, buffer_key: str = "buffer_0mm") -> bool:
    if isinstance(value, dict):
        if value.get("gates", {}).get(buffer_key) is True:
            return True
        return any(_any_safe(item, buffer_key) for item in value.values())
    if isinstance(value, list):
        return any(_any_safe(item, buffer_key) for item in value)
    return False


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    reference_result_path: Path,
    prior_result_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    expected_commit: str,
    output_path: Path,
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
    from main.multilink_ellipsoid.iterative_counterfactual_risk import (
        active_witness,
        fit_safety_direction,
        load_iterative_risk_config,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_iterative_risk_config(experiment_config_path)
    reference = _validate_reference_result(reference_result_path, config)
    prior = _validate_prior(prior_result_path, config)
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list) and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "Table-1 action horizon differs",
    )
    matches = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
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
        obstacle_name, _ = _active_obstacle(env, observation)
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
            _require(pairing[key] == archived["pairing"][key], "pairing differs: %s" % key)
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
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        disabled_images = _disable_images(probe_env)
        probe = FixedContinuationProbe(one_step_probe, obstacle_name, initial_obstacle_position)
        for step in range(182):
            observation, _, done, _ = env.step(_archived_action(archived_actions, step).tolist())
            _require(not bool(done), "archived prefix completed before iterative-risk state")
        base_actions = _reference_actions(reference, 182, 201)
        zero = np.zeros(15, dtype=np.float64)
        nominal_initial = _candidate(
            probe, env, base_actions, zero, np.zeros(3), config, include_trace=True
        )
        target_terminal_eef = np.asarray(
            nominal_initial["_record"]["eef_position_trace_m"][-1], dtype=np.float64
        )
        nominal = _candidate(
            probe, env, base_actions, zero, target_terminal_eef, config, include_trace=True
        )
        _require(
            nominal_initial["rollout"]["state_sha256"] == nominal["rollout"]["state_sha256"],
            "iterative-risk nominal replay differs",
        )
        _require(
            197 in {int(event["step"]) for event in nominal["rollout"]["protected_contacts"]},
            "known action-197 collision did not reproduce",
        )

        prior_directions = np.asarray(
            [item["direction"] for item in prior["paired_rollouts"]], dtype=np.float64
        )
        prior_plus = np.asarray(
            [item["positive"]["rollout"]["minimum_clearance_m"] for item in prior["paired_rollouts"]]
        )
        prior_minus = np.asarray(
            [item["negative"]["rollout"]["minimum_clearance_m"] for item in prior["paired_rollouts"]]
        )
        pure_refit = fit_safety_direction(
            prior_directions,
            prior_plus,
            prior_minus,
            float(prior["config"]["sampling"]["paired_perturbation_action"]),
            float(config["field_estimation"]["ridge"]),
        )
        previous_direction = np.asarray(prior["field_fit"]["learned_unit_direction"], dtype=np.float64)
        refit_cosine = float(np.dot(pure_refit["unit_direction"], previous_direction))
        fixed_plus_trace = np.asarray(
            prior["comparison"]["fixed_plus"]["rollout"]["clearance_trace_m"], dtype=np.float64
        )
        fixed_minus_trace = np.asarray(
            prior["comparison"]["fixed_minus"]["rollout"]["clearance_trace_m"], dtype=np.float64
        )
        denominator = 2.0 * float(prior["comparison"]["correction_l2_action"])
        fixed_d5 = float((np.min(fixed_plus_trace[:5]) - np.min(fixed_minus_trace[:5])) / denominator)
        fixed_d20 = float((np.min(fixed_plus_trace) - np.min(fixed_minus_trace)) / denominator)
        analytical_sign = (
            "short_horizon_improves_but_long_horizon_worsens"
            if fixed_d5 > 0.0 and fixed_d20 < 0.0
            else "wrong_sign_at_both_horizons"
            if fixed_d5 < 0.0 and fixed_d20 < 0.0
            else "same_improving_sign_at_both_horizons"
            if fixed_d5 > 0.0 and fixed_d20 > 0.0
            else "mixed_or_degenerate_sign"
        )
        existing_audit = {
            "pure_risk_refit_unit_direction": pure_refit["unit_direction"].tolist(),
            "pure_risk_fit_rmse": float(pure_refit["fit_rmse"]),
            "pure_risk_vs_previous_task_penalized_direction_cosine": refit_cosine,
            "analytical_clearance_derivative_5_action_m_per_action": fixed_d5,
            "analytical_clearance_derivative_20_action_m_per_action": fixed_d20,
            "analytical_sign_interpretation": analytical_sign,
            "fixed_plus_5_action_witness": active_witness(fixed_plus_trace[:5]),
            "fixed_minus_5_action_witness": active_witness(fixed_minus_trace[:5]),
            "fixed_plus_20_action_witness": active_witness(fixed_plus_trace),
            "fixed_minus_20_action_witness": active_witness(fixed_minus_trace),
        }

        primary = {
            "field": _field_path(
                probe, env, base_actions, target_terminal_eef, config, preserve_endpoint=True
            ),
            "analytical": _analytical_path(
                probe, env, base_actions, target_terminal_eef, config, preserve_endpoint=True
            ),
            "search": _random_and_derivative_free(
                probe, env, base_actions, target_terminal_eef, config, preserve_endpoint=True
            ),
        }
        primary_safe = _any_safe(primary)
        primary_search_safe = _any_safe(primary["search"])
        primary_field_safe = _any_safe(primary["field"])
        secondary = None
        if not primary_search_safe:
            secondary = {
                "field": _field_path(
                    probe, env, base_actions, target_terminal_eef, config, preserve_endpoint=False
                ),
                "analytical": _analytical_path(
                    probe, env, base_actions, target_terminal_eef, config, preserve_endpoint=False
                ),
                "search": _random_and_derivative_free(
                    probe, env, base_actions, target_terminal_eef, config, preserve_endpoint=False
                ),
            }
        secondary_safe = bool(secondary is not None and _any_safe(secondary))
        learned_safe = _any_safe(primary["field"]) or bool(
            secondary is not None and _any_safe(secondary["field"])
        )
        empirical_oracle_safe = _any_safe(primary["search"]) or bool(
            secondary is not None and _any_safe(secondary["search"])
        )
        if primary_field_safe:
            interpretation = "iterative_pure_risk_field_finds_verified_safe_support"
        elif primary_search_safe:
            interpretation = "exact_endpoint_safe_support_exists_but_iterative_field_fails"
        elif secondary_safe and learned_safe:
            interpretation = "soft_endpoint_iterative_field_finds_verified_safe_support"
        elif empirical_oracle_safe:
            interpretation = "soft_endpoint_safe_support_exists_but_iterative_field_fails"
        else:
            interpretation = "no_empirical_safe_support_in_registered_five_action_family"
        result = {
            "schema_version": RESULT_SCHEMA,
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
            "reference_continuation": {
                "path": str(reference_result_path),
                "file_sha256": _file_sha256(reference_result_path),
                "payload_sha256": reference["result_payload_sha256"],
                "read_only": True,
            },
            "prior_counterfactual": {
                "path": str(prior_result_path),
                "file_sha256": _file_sha256(prior_result_path),
                "payload_sha256": prior["result_payload_sha256"],
                "read_only": True,
            },
            "config": config,
            "geometry_config": geometry_config,
            "geometry": geometry.geometry_record(env),
            "pairing": pairing,
            "probe_environment": {
                "disabled_image_observable_count": disabled_images,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
            },
            "nominal": _strip_private(nominal),
            "existing_data_audit": existing_audit,
            "exact_endpoint_primary": primary,
            "soft_endpoint_secondary": secondary,
            "primary_safe_support": primary_safe,
            "secondary_safe_support": secondary_safe,
            "learned_field_safe_support": learned_safe,
            "empirical_search_safe_support": empirical_oracle_safe,
            "interpretation": interpretation,
            "rollout_count": (
                2
                + sum(int(value["rollout_count"]) for value in primary.values())
                + (0 if secondary is None else sum(int(value["rollout_count"]) for value in secondary.values()))
            ),
            "probe_env_step_wall_seconds": (
                float(nominal_initial["rollout"]["env_step_wall_seconds"])
                + float(nominal["rollout"]["env_step_wall_seconds"])
                + sum(float(value["probe_env_step_wall_seconds"]) for value in primary.values())
                + (0.0 if secondary is None else sum(float(value["probe_env_step_wall_seconds"]) for value in secondary.values()))
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        reference_result_path=args.reference_result.resolve(),
        prior_result_path=args.prior_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "interpretation": result["interpretation"],
                "learned_field_safe_support": result["learned_field_safe_support"],
                "empirical_search_safe_support": result["empirical_search_safe_support"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
