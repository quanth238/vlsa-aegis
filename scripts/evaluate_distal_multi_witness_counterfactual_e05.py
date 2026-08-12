#!/usr/bin/env python3
"""Compare single, smooth, and coordinated multi-witness fields on E05."""

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


RESULT_SCHEMA = "vlsa_distal_multi_witness_counterfactual_e05_result.v1"


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
        "hard_margin_m": hard,
        "pure_risk_m": -hard,
        "terminal_eef_error_m_diagnostic_only": terminal_error,
        "active_witness": active_witness(rollout["clearance_trace_m"]),
        "avoidance_gate": safe,
        "rollout": _public_rollout(rollout, include_trace=include_trace),
        "_trace": np.asarray(rollout["clearance_trace_m"], dtype=np.float64),
    }


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


def _fit_local_rows(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    correction: Any,
    target_terminal_eef: Any,
    config: Mapping[str, Any],
    arm_index: int,
    iteration: int,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.multi_witness_counterfactual import (
        bounded_paired_directions,
        fit_clearance_rows,
    )

    estimation = config["field_estimation"]
    current_xyz = np.asarray(base_actions[:5, :3], dtype=np.float64) + np.asarray(
        correction, dtype=np.float64
    ).reshape(5, 3)
    directions = bounded_paired_directions(
        int(estimation["paired_direction_count_per_iteration"]),
        int(estimation["direction_seed"]) + iteration,
        current_xyz,
        float(estimation["paired_perturbation_action"]),
        float(config["action_space"]["action_limit"]),
    )
    plus_traces = []
    minus_traces = []
    pairs = []
    probe_wall = 0.0
    for direction in directions:
        positive = _candidate(
            probe,
            env,
            base_actions,
            np.asarray(correction) + float(estimation["paired_perturbation_action"]) * direction,
            target_terminal_eef,
            config,
            include_trace=True,
        )
        negative = _candidate(
            probe,
            env,
            base_actions,
            np.asarray(correction) - float(estimation["paired_perturbation_action"]) * direction,
            target_terminal_eef,
            config,
            include_trace=True,
        )
        plus_traces.append(positive["_trace"])
        minus_traces.append(negative["_trace"])
        probe_wall += float(positive["rollout"]["env_step_wall_seconds"])
        probe_wall += float(negative["rollout"]["env_step_wall_seconds"])
        pairs.append(
            {
                "direction": direction.tolist(),
                "positive_witness": positive["active_witness"],
                "negative_witness": negative["active_witness"],
                "witness_switched": positive["active_witness"] != negative["active_witness"],
            }
        )
    fit = fit_clearance_rows(
        directions,
        plus_traces,
        minus_traces,
        float(estimation["paired_perturbation_action"]),
        float(estimation["ridge"]),
    )
    return {
        "directions": directions,
        "fit": fit,
        "pairs": pairs,
        "rollout_count": 2 * len(directions),
        "probe_env_step_wall_seconds": probe_wall,
    }


def _direction_for_arm(
    arm: str,
    center: Mapping[str, Any],
    local: Mapping[str, Any],
    current_xyz: Any,
    correction: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.iterative_counterfactual_risk import project_mode_direction
    from main.multilink_ellipsoid.multi_witness_counterfactual import (
        select_near_active_witnesses,
        smooth_min_direction,
        solve_multi_witness_epigraph,
    )

    trace = np.asarray(center["_trace"], dtype=np.float64)
    flattened = trace.reshape(-1)
    gradients = np.asarray(local["fit"]["gradients"], dtype=np.float64)
    if arm == "single_hard_min":
        index = int(np.argmin(flattened))
        raw = gradients[index]
        return {
            "delta": float(config["action_space"]["trust_radius_action"])
            * project_mode_direction(
                raw,
                current_xyz,
                action_limit=float(config["action_space"]["action_limit"]),
                preserve_endpoint=False,
            ),
            "witnesses": [
                {
                    "flat_index": index,
                    "action_offset": int(index // 7),
                    "ellipsoid_row": int(index % 7),
                    "margin_m": float(flattened[index]),
                }
            ],
            "solver": "normalized_hard_witness_gradient",
        }
    if arm == "smooth_max":
        smooth = smooth_min_direction(
            flattened, gradients, float(config["smooth_max"]["temperature_m"])
        )
        return {
            "delta": float(config["action_space"]["trust_radius_action"])
            * project_mode_direction(
                smooth["gradient"],
                current_xyz,
                action_limit=float(config["action_space"]["action_limit"]),
                preserve_endpoint=False,
            ),
            "witnesses": [],
            "weights": smooth["weights"].tolist(),
            "solver": "normalized_smooth_min_gradient",
        }
    if arm == "multi_witness":
        settings = config["multi_witness"]
        witnesses = select_near_active_witnesses(
            trace,
            maximum_count=int(settings["maximum_witness_count"]),
            threshold_m=float(settings["near_active_threshold_m"]),
        )
        indices = np.asarray([item["flat_index"] for item in witnesses], dtype=np.int64)
        solved = solve_multi_witness_epigraph(
            flattened[indices],
            gradients[indices],
            current_xyz,
            correction,
            action_limit=float(config["action_space"]["action_limit"]),
            trust_radius=float(config["action_space"]["trust_radius_action"]),
            maximum_total_correction=float(
                config["action_space"]["maximum_total_correction_l2_action"]
            ),
            regularization=float(settings["epigraph_regularization_m_per_action2"]),
            solver=str(settings["solver"]),
        )
        return {
            "delta": solved["delta"],
            "witnesses": witnesses,
            "solver": solved["status"],
            "objective": solved["objective"],
            "predicted_margins_m": solved["predicted_margins_m"].tolist(),
            "predicted_worst_margin_m": solved["predicted_worst_margin_m"],
        }
    raise ValueError("unknown multi-witness arm")


def _run_arm(
    arm: str,
    arm_index: int,
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    target_terminal_eef: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.iterative_counterfactual_risk import feasible_correction

    correction = np.zeros(15, dtype=np.float64)
    center = _candidate(
        probe, env, base_actions, correction, target_terminal_eef, config, include_trace=True
    )
    path = [center]
    iterations = []
    path_length = 0.0
    rollout_count = 1
    probe_wall = float(center["rollout"]["env_step_wall_seconds"])
    stop_reason = "maximum_iterations"
    for iteration in range(int(config["action_space"]["maximum_iterations"])):
        local = _fit_local_rows(
            probe,
            env,
            base_actions,
            correction,
            target_terminal_eef,
            config,
            arm_index,
            iteration,
        )
        rollout_count += int(local["rollout_count"])
        probe_wall += float(local["probe_env_step_wall_seconds"])
        current_xyz = np.asarray(base_actions[:5, :3], dtype=np.float64) + correction.reshape(5, 3)
        proposal = _direction_for_arm(
            arm, center, local, current_xyz, correction, config
        )
        raw_delta = np.asarray(proposal["delta"], dtype=np.float64).reshape(15)
        candidates = []
        for fraction in config["action_space"]["line_search_fractions"]:
            delta = float(fraction) * raw_delta
            step_norm = float(np.linalg.norm(delta))
            proposed = correction + delta
            if path_length + step_norm > float(
                config["action_space"]["maximum_total_path_length_action"]
            ) + 1.0e-10:
                continue
            if not feasible_correction(
                base_actions[:5, :3],
                proposed,
                radius=float(config["action_space"]["maximum_total_correction_l2_action"]),
                action_limit=float(config["action_space"]["action_limit"]),
                preserve_endpoint=False,
            ):
                continue
            candidate = _candidate(
                probe, env, base_actions, proposed, target_terminal_eef, config, include_trace=True
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
                "center": _public(center),
                "witnesses": proposal.get("witnesses", []),
                "proposal": _public(proposal),
                "paired_witness_switch_count": sum(
                    int(item["witness_switched"]) for item in local["pairs"]
                ),
                "selected_row_fit_rmse_m_per_action": [
                    float(local["fit"]["row_fit_rmse"][item["flat_index"]])
                    for item in proposal.get("witnesses", [])
                ],
                "candidates": [_public(item) for item in candidates],
                "selected": None if selected is None else _public(selected),
            }
        )
        if selected is None:
            stop_reason = "no_exact_improving_line_search_step"
            break
        correction = np.asarray(selected["correction"], dtype=np.float64)
        center = selected
        path_length += float(selected["step_norm_action"])
        path.append(center)
        if bool(center["avoidance_gate"]):
            stop_reason = "verified_safe"
            break
    best = max(path, key=lambda item: (bool(item["avoidance_gate"]), float(item["hard_margin_m"])))
    return {
        "arm": arm,
        "path": [_public(item) for item in path],
        "iterations": iterations,
        "best": _public(best),
        "final": _public(path[-1]),
        "path_length_action": path_length,
        "stop_reason": stop_reason,
        "rollout_count": rollout_count,
        "probe_env_step_wall_seconds": probe_wall,
    }


def _prior_derivative_free(prior: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    candidate = prior["soft_endpoint_secondary"]["search"]["by_radius"]["1.0"][
        "derivative_free_best"
    ]
    rollout = candidate["rollout"]
    return {
        "hard_margin_m": float(candidate["hard_margin_m"]),
        "correction_l2_action": float(candidate["correction_l2_action"]),
        "terminal_eef_error_m_diagnostic_only": float(candidate["terminal_eef_error_m"]),
        "protected_contact_count": len(rollout["protected_contacts"]),
        "maximum_active_obstacle_l1_displacement_m": float(
            rollout["maximum_active_obstacle_l1_displacement_m"]
        ),
        "avoidance_gate": bool(
            float(candidate["hard_margin_m"]) >= float(config["gate"]["minimum_exact_clearance_m"])
            and not rollout["protected_contacts"]
            and float(rollout["maximum_active_obstacle_l1_displacement_m"])
            <= float(config["gate"]["paper_car_threshold_m"])
        ),
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
    from main.multilink_ellipsoid.multi_witness_counterfactual import load_multi_witness_config
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_multi_witness_config(experiment_config_path)
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
            _require(not bool(done), "archived prefix completed before multi-witness state")
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
            "multi-witness nominal replay differs",
        )
        _require(
            int(config["state_protocol"]["known_reference_collision_step"])
            in {int(event["step"]) for event in nominal["rollout"]["protected_contacts"]},
            "known action-197 collision did not reproduce",
        )
        arms = {}
        for arm_index, arm in enumerate(("single_hard_min", "smooth_max", "multi_witness")):
            arms[arm] = _run_arm(
                arm,
                arm_index,
                probe,
                env,
                base_actions,
                target_terminal_eef,
                config,
            )
        derivative_free = _prior_derivative_free(prior, config)
        multi_safe = bool(arms["multi_witness"]["best"]["avoidance_gate"])
        gate = {
            "multi_witness_verified_safe": multi_safe,
            "positive_clearance": bool(arms["multi_witness"]["best"]["hard_margin_m"] >= 0.0),
            "zero_protected_contact": bool(
                not arms["multi_witness"]["best"]["rollout"]["protected_contacts"]
            ),
            "paper_car_pass": bool(
                float(
                    arms["multi_witness"]["best"]["rollout"][
                        "maximum_active_obstacle_l1_displacement_m"
                    ]
                )
                <= float(config["gate"]["paper_car_threshold_m"])
            ),
            "within_shared_path_budget": bool(
                arms["multi_witness"]["path_length_action"]
                <= float(config["action_space"]["maximum_total_path_length_action"]) + 1.0e-10
            ),
            "beats_single_hard_min_margin": bool(
                float(arms["multi_witness"]["best"]["hard_margin_m"])
                > float(arms["single_hard_min"]["best"]["hard_margin_m"]) + 1.0e-9
            ),
            "derivative_free_safe_support_exists": bool(derivative_free["avoidance_gate"]),
        }
        gate["passed"] = bool(all(gate.values()))
        interpretation = (
            "multi_witness_counterfactual_field_recovers_verified_safe_detour"
            if gate["passed"]
            else "known_safe_detour_exists_but_multi_witness_local_field_does_not_recover_it"
            if derivative_free["avoidance_gate"]
            else "registered_avoidance_family_has_no_verified_safe_support"
        )
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
            "nominal": _public(nominal),
            "arms": arms,
            "prior_derivative_free": derivative_free,
            "gate": gate,
            "interpretation": interpretation,
            "rollout_count": int(2 + sum(value["rollout_count"] for value in arms.values())),
            "probe_env_step_wall_seconds": float(
                nominal_initial["rollout"]["env_step_wall_seconds"]
                + nominal["rollout"]["env_step_wall_seconds"]
                + sum(value["probe_env_step_wall_seconds"] for value in arms.values())
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
