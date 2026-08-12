#!/usr/bin/env python3
"""Avoidance-only fixed-step counterfactual field gate for archived E05."""

from __future__ import annotations

import argparse
import hashlib
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
    _public_rollout,
    _reference_actions,
    _validate_reference_result,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_fixed_step_counterfactual_field_e05_result.v1"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_prior(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    result = _load(path)
    expected = config["prior_iterative_result"]
    _require(
        result.get("schema_version") == "vlsa_distal_iterative_counterfactual_risk_e05_result.v1",
        "prior iterative schema differs",
    )
    _require(
        result.get("result_payload_sha256") == expected["result_payload_sha256"],
        "prior iterative payload differs",
    )
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(
        _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        == claimed,
        "prior iterative self-hash differs",
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

    from main.multilink_ellipsoid.iterative_counterfactual_risk import active_witness

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
    safe = bool(
        hard >= float(config["gate"]["minimum_exact_clearance_m"])
        and len(rollout["protected_contacts"])
        == int(config["gate"]["protected_raw_contact_count"])
        and float(rollout["maximum_active_obstacle_l1_displacement_m"])
        <= float(config["gate"]["paper_car_threshold_m"])
    )
    return {
        "correction": delta.tolist(),
        "correction_l2_action": float(np.linalg.norm(delta)),
        "cumulative_registered_step_action": None,
        "endpoint_correction_sum": np.sum(delta.reshape(5, 3), axis=0).tolist(),
        "hard_margin_m": hard,
        "pure_risk_m": -hard,
        "terminal_eef_error_m_diagnostic_only": terminal_error,
        "active_witness": active_witness(rollout["clearance_trace_m"]),
        "avoidance_gate": safe,
        "rollout": _public_rollout(rollout, include_trace=include_trace),
    }


def _best(path: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(max(path, key=lambda item: (bool(item["avoidance_gate"]), float(item["hard_margin_m"]))))


def _recomputed_path(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    target_terminal_eef: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.counterfactual_field import paired_chunk_directions
    from main.multilink_ellipsoid.fixed_step_counterfactual import fixed_step_update
    from main.multilink_ellipsoid.iterative_counterfactual_risk import (
        feasible_correction,
        fit_safety_direction,
        project_mode_direction,
    )

    estimation = config["field_estimation"]
    maximum_norm = float(config["action_space"]["maximum_total_correction_l2_action"])
    correction = np.zeros(15, dtype=np.float64)
    center = _candidate(
        probe, env, base_actions, correction, target_terminal_eef, config, include_trace=True
    )
    center["cumulative_registered_step_action"] = 0.0
    path = [center]
    iterations = []
    rollout_count = 1
    probe_wall = float(center["rollout"]["env_step_wall_seconds"])
    initial_direction = None
    stop_reason = "maximum_iterations"
    for iteration in range(int(estimation["maximum_iterations"])):
        current_xyz = np.asarray(base_actions[:5, :3], dtype=np.float64) + correction.reshape(5, 3)
        directions = paired_chunk_directions(
            int(estimation["paired_direction_count_per_iteration"]),
            int(estimation["direction_seed"]) + iteration,
            current_xyz,
            float(estimation["paired_perturbation_action"]),
            float(config["action_space"]["action_limit"]),
            preserve_endpoint=False,
        )
        plus = []
        minus = []
        pairs = []
        witness_switches = 0
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
            plus.append(float(positive["hard_margin_m"]))
            minus.append(float(negative["hard_margin_m"]))
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
            plus,
            minus,
            float(estimation["paired_perturbation_action"]),
            float(estimation["ridge"]),
        )
        direction = project_mode_direction(
            fit["gradient"],
            current_xyz,
            action_limit=float(config["action_space"]["action_limit"]),
            preserve_endpoint=False,
        )
        if initial_direction is None:
            initial_direction = np.asarray(direction, dtype=np.float64)
        proposed = None
        candidate = None
        reason = None
        try:
            proposed = fixed_step_update(
                correction,
                direction,
                step=float(estimation["fixed_normalized_step_action"]),
                maximum_norm=maximum_norm,
            )
        except ValueError as error:
            reason = str(error)
        if proposed is not None and not feasible_correction(
            base_actions[:5, :3],
            proposed,
            radius=maximum_norm,
            action_limit=float(config["action_space"]["action_limit"]),
            preserve_endpoint=False,
        ):
            reason = "fixed full step violates action bounds or total correction budget"
            proposed = None
        if proposed is not None:
            candidate = _candidate(
                probe, env, base_actions, proposed, target_terminal_eef, config, include_trace=True
            )
            candidate["cumulative_registered_step_action"] = float(
                (iteration + 1) * float(estimation["fixed_normalized_step_action"])
            )
            rollout_count += 1
            probe_wall += float(candidate["rollout"]["env_step_wall_seconds"])
        iterations.append(
            {
                "iteration": iteration,
                "center": center,
                "pairs": pairs,
                "witness_switch_count": witness_switches,
                "fit": {
                    "gradient": fit["gradient"].tolist(),
                    "unit_direction": direction.tolist(),
                    "fit_rmse": float(fit["fit_rmse"]),
                    "directional_targets": fit["directional_targets"].tolist(),
                    "directional_predictions": fit["directional_predictions"].tolist(),
                },
                "full_step_candidate": candidate,
                "refusal_reason": reason,
            }
        )
        if candidate is None:
            stop_reason = reason or "full_step_unavailable"
            break
        correction = np.asarray(candidate["correction"], dtype=np.float64)
        center = candidate
        path.append(center)
        if bool(center["avoidance_gate"]):
            stop_reason = "verified_safe"
            break
    _require(initial_direction is not None, "initial counterfactual direction missing")
    return {
        "path": path,
        "iterations": iterations,
        "best": _best(path),
        "final": path[-1],
        "initial_unit_direction": initial_direction.tolist(),
        "stop_reason": stop_reason,
        "rollout_count": rollout_count,
        "probe_env_step_wall_seconds": probe_wall,
    }


def _fixed_initial_path(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    target_terminal_eef: Any,
    initial_direction: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.fixed_step_counterfactual import fixed_step_update
    from main.multilink_ellipsoid.iterative_counterfactual_risk import feasible_correction

    direction = np.asarray(initial_direction, dtype=np.float64).reshape(15)
    maximum_norm = float(config["action_space"]["maximum_total_correction_l2_action"])
    step = float(config["field_estimation"]["fixed_normalized_step_action"])
    correction = np.zeros(15, dtype=np.float64)
    center = _candidate(
        probe, env, base_actions, correction, target_terminal_eef, config, include_trace=True
    )
    center["cumulative_registered_step_action"] = 0.0
    path = [center]
    rollout_count = 1
    probe_wall = float(center["rollout"]["env_step_wall_seconds"])
    stop_reason = "maximum_iterations"
    for iteration in range(int(config["field_estimation"]["maximum_iterations"])):
        try:
            proposed = fixed_step_update(
                correction, direction, step=step, maximum_norm=maximum_norm
            )
        except ValueError as error:
            stop_reason = str(error)
            break
        if not feasible_correction(
            base_actions[:5, :3],
            proposed,
            radius=maximum_norm,
            action_limit=float(config["action_space"]["action_limit"]),
            preserve_endpoint=False,
        ):
            stop_reason = "fixed full step violates action bounds or total correction budget"
            break
        candidate = _candidate(
            probe, env, base_actions, proposed, target_terminal_eef, config, include_trace=True
        )
        candidate["cumulative_registered_step_action"] = float((iteration + 1) * step)
        path.append(candidate)
        rollout_count += 1
        probe_wall += float(candidate["rollout"]["env_step_wall_seconds"])
        correction = proposed
        if bool(candidate["avoidance_gate"]):
            stop_reason = "verified_safe"
            break
    return {
        "path": path,
        "best": _best(path),
        "final": path[-1],
        "stop_reason": stop_reason,
        "rollout_count": rollout_count,
        "probe_env_step_wall_seconds": probe_wall,
    }


def _prior_avoidance_record(candidate: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rollout = candidate["rollout"]
    safe = bool(
        float(candidate["hard_margin_m"]) >= float(config["gate"]["minimum_exact_clearance_m"])
        and len(rollout["protected_contacts"])
        == int(config["gate"]["protected_raw_contact_count"])
        and float(rollout["maximum_active_obstacle_l1_displacement_m"])
        <= float(config["gate"]["paper_car_threshold_m"])
    )
    return {
        "correction_l2_action": float(candidate["correction_l2_action"]),
        "hard_margin_m": float(candidate["hard_margin_m"]),
        "protected_contact_count": len(rollout["protected_contacts"]),
        "maximum_active_obstacle_l1_displacement_m": float(
            rollout["maximum_active_obstacle_l1_displacement_m"]
        ),
        "avoidance_gate": safe,
        "terminal_eef_error_m_diagnostic_only": float(candidate["terminal_eef_error_m"]),
    }


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
    from main.multilink_ellipsoid.fixed_step_counterfactual import load_fixed_step_config
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_fixed_step_config(experiment_config_path)
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
        for step_index in range(182):
            observation, _, done, _ = env.step(
                _archived_action(archived_actions, step_index).tolist()
            )
            _require(not bool(done), "archived prefix completed before fixed-step state")
        base_actions = _reference_actions(reference, 182, 201)
        zero = np.zeros(15, dtype=np.float64)
        nominal_initial = _candidate(
            probe, env, base_actions, zero, np.zeros(3), config, include_trace=True
        )
        target_terminal_eef = np.asarray(
            nominal_initial["rollout"]["terminal_eef_position_m"], dtype=np.float64
        )
        nominal = _candidate(
            probe, env, base_actions, zero, target_terminal_eef, config, include_trace=True
        )
        _require(
            nominal_initial["rollout"]["state_sha256"] == nominal["rollout"]["state_sha256"],
            "fixed-step nominal replay differs",
        )
        _require(
            int(config["state_protocol"]["known_reference_collision_step"])
            in {int(event["step"]) for event in nominal["rollout"]["protected_contacts"]},
            "known action-197 collision did not reproduce",
        )
        recomputed = _recomputed_path(
            probe, env, base_actions, target_terminal_eef, config
        )
        fixed = _fixed_initial_path(
            probe,
            env,
            base_actions,
            target_terminal_eef,
            recomputed["initial_unit_direction"],
            config,
        )
        prior_soft = prior["soft_endpoint_secondary"]
        analytical = _prior_avoidance_record(prior_soft["analytical"]["best"], config)
        derivative_free = _prior_avoidance_record(
            prior_soft["search"]["by_radius"]["1.0"]["derivative_free_best"], config
        )
        random_best = _prior_avoidance_record(
            prior_soft["search"]["by_radius"]["1.0"]["random_best"], config
        )
        advantage_m = float(recomputed["best"]["hard_margin_m"]) - float(
            fixed["best"]["hard_margin_m"]
        )
        gate = {
            "recomputed_field_verified_safe": bool(recomputed["best"]["avoidance_gate"]),
            "action_197_contact_eliminated": bool(
                recomputed["best"]["avoidance_gate"]
                and not recomputed["best"]["rollout"]["protected_contacts"]
            ),
            "paper_car_pass": bool(
                float(
                    recomputed["best"]["rollout"][
                        "maximum_active_obstacle_l1_displacement_m"
                    ]
                )
                <= float(config["gate"]["paper_car_threshold_m"])
            ),
            "within_shared_radius_one_budget": bool(
                float(recomputed["best"]["correction_l2_action"])
                <= float(config["action_space"]["maximum_total_correction_l2_action"])
                + 1.0e-10
            ),
            "recomputation_beats_fixed_initial_direction": bool(advantage_m > 1.0e-9),
            "analytical_safe_support_exists": bool(analytical["avoidance_gate"]),
            "derivative_free_safe_support_exists": bool(derivative_free["avoidance_gate"]),
        }
        gate["passed"] = bool(all(gate.values()))
        if gate["passed"]:
            interpretation = "iteratively_recomputed_long_horizon_field_finds_known_safe_detour"
        elif analytical["avoidance_gate"] or derivative_free["avoidance_gate"]:
            interpretation = "safe_detour_exists_but_iterative_single_gradient_field_does_not_find_it"
        else:
            interpretation = "registered_avoidance_family_has_no_verified_safe_support"
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
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
            "prior_iterative_result": {
                "path": str(prior_result_path),
                "file_sha256": _file_sha256(prior_result_path),
                "payload_sha256": prior["result_payload_sha256"],
                "read_only": True,
            },
            "geometry_config": geometry_config,
            "geometry": geometry.geometry_record(env),
            "pairing": pairing,
            "probe_environment": {
                "disabled_image_observable_count": disabled_images,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
            },
            "nominal": nominal,
            "recomputed_field": recomputed,
            "fixed_initial_direction": fixed,
            "prior_safe_baselines": {
                "iterative_analytical": analytical,
                "derivative_free": derivative_free,
                "random_best": random_best,
            },
            "recomputed_minus_fixed_best_margin_m": advantage_m,
            "gate": gate,
            "interpretation": interpretation,
            "rollout_count": int(
                2 + recomputed["rollout_count"] + fixed["rollout_count"]
            ),
            "probe_env_step_wall_seconds": float(
                nominal_initial["rollout"]["env_step_wall_seconds"]
                + nominal["rollout"]["env_step_wall_seconds"]
                + recomputed["probe_env_step_wall_seconds"]
                + fixed["probe_env_step_wall_seconds"]
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
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "interpretation": result["interpretation"],
                "gate_passed": result["gate"]["passed"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
