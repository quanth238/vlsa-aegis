#!/usr/bin/env python3
"""Test a paired-rollout ellipsoid field on the later E05 collision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_five_action_detour_e05 import (
    _archived_action,
    _disable_images,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    PAPER_CAR_THRESHOLD_M,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_counterfactual_field_e05_result.v1"


def _validate_reference_result(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    result = _load(path)
    expected = config["reference_continuation"]
    _require(
        result.get("schema_version") == "vlsa_distal_five_action_detour_e05_result.v1",
        "reference continuation schema differs",
    )
    _require(
        result.get("result_payload_sha256") == expected["result_payload_sha256"],
        "reference continuation payload differs",
    )
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    recomputed = _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    _require(recomputed == claimed, "reference continuation self-hash differs")
    _require(
        result.get("source", {}).get("commit") == expected["source_commit"],
        "reference continuation source differs",
    )
    records = result.get("actions")
    _require(isinstance(records, list), "reference continuation actions missing")
    by_step = {int(record["step"]): record for record in records}
    start, stop = expected["use_executed_action_steps"]
    _require(all(step in by_step for step in range(start, stop + 1)), "reference action range differs")
    return result


def _reference_actions(reference: Mapping[str, Any], start: int, stop: int) -> Any:
    import numpy as np

    by_step = {int(record["step"]): record for record in reference["actions"]}
    actions = np.asarray([by_step[step]["action"] for step in range(start, stop + 1)], dtype=np.float64)
    _require(actions.shape == (stop - start + 1, 7), "reference action shape differs")
    _require(np.all(np.isfinite(actions)), "reference actions are nonfinite")
    return actions


class FixedContinuationProbe:
    def __init__(
        self,
        one_step_probe: Any,
        active_obstacle_name: str,
        obstacle_reference_position_m: Any,
    ) -> None:
        self.one_step_probe = one_step_probe
        self.active_obstacle_name = str(active_obstacle_name)
        self.obstacle_reference_position_m = obstacle_reference_position_m

    @property
    def env(self) -> Any:
        return self.one_step_probe.probe_env

    def rollout(self, main_env: Any, actions: Any, *, step_base: int = 182) -> dict[str, Any]:
        import numpy as np

        from main.multilink_ellipsoid.rollout import _dynamic_state_vector
        from main.multilink_ellipsoid.sitl_candidate import (
            _obstacle_root_body_id,
            _protected_contact_evidence,
        )

        commands = np.asarray(actions, dtype=np.float64)
        _require(
            commands.ndim == 2
            and commands.shape[1] == 7
            and 1 <= commands.shape[0] <= 20
            and np.all(np.isfinite(commands)),
            "fixed-continuation action shape differs",
        )
        synchronization = self.one_step_probe.synchronize(main_env)
        obstacle_id = _obstacle_root_body_id(self.env.sim.model, self.active_obstacle_name)
        obstacle_reference = np.asarray(self.obstacle_reference_position_m, dtype=np.float64).reshape(3)
        clearances = []
        eef_positions = []
        state_hashes = []
        contacts = []
        displacements = []
        done_steps = []
        started = time.perf_counter_ns()
        for offset, command in enumerate(commands):
            observation, _, done, _ = self.env.step(command.tolist())
            clearances.append(self.one_step_probe.clearances(self.env)[:7])
            eef_positions.append(np.asarray(observation["robot0_eef_pos"], dtype=np.float64))
            state = _dynamic_state_vector(self.env)
            state_hashes.append(hashlib.sha256(state.tobytes()).hexdigest())
            evidence = _protected_contact_evidence(self.env, self.active_obstacle_name)
            for event in evidence["events"]:
                contacts.append({"step": int(step_base) + offset, **event})
            obstacle = np.asarray(self.env.sim.data.xpos[obstacle_id], dtype=np.float64)
            displacements.append(float(np.sum(np.abs(obstacle - obstacle_reference))))
            if done:
                done_steps.append(int(step_base) + offset)
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        trace = np.asarray(clearances, dtype=np.float64)
        return {
            "actions": commands,
            "clearance_trace_m": trace,
            "row_minimum_clearance_m": np.min(trace, axis=0),
            "minimum_clearance_m": float(np.min(trace)),
            "eef_position_trace_m": np.asarray(eef_positions, dtype=np.float64),
            "state_sha256": state_hashes,
            "protected_contacts": contacts,
            "maximum_active_obstacle_l1_displacement_m": float(max(displacements)),
            "active_obstacle_l1_displacement_m": displacements,
            "done_steps": done_steps,
            "synchronization": synchronization,
            "env_step_wall_seconds": elapsed,
        }


def _actions_with_correction(base: Any, correction: Any, action_limit: float) -> Any:
    import numpy as np

    actions = np.asarray(base, dtype=np.float64).copy()
    delta = np.asarray(correction, dtype=np.float64).reshape(5, 3)
    actions[:5, :3] += delta
    _require(
        float(np.max(np.abs(actions[:5, :3]))) <= float(action_limit) + 1.0e-10,
        "counterfactual correction exceeds action bounds",
    )
    return actions


def _public_rollout(record: Mapping[str, Any], *, include_trace: bool = True) -> dict[str, Any]:
    output = {
        "row_minimum_clearance_m": record["row_minimum_clearance_m"].tolist(),
        "minimum_clearance_m": float(record["minimum_clearance_m"]),
        "terminal_eef_position_m": record["eef_position_trace_m"][-1].tolist(),
        "state_sha256": list(record["state_sha256"]),
        "protected_contacts": list(record["protected_contacts"]),
        "maximum_active_obstacle_l1_displacement_m": float(
            record["maximum_active_obstacle_l1_displacement_m"]
        ),
        "active_obstacle_l1_displacement_m": list(record["active_obstacle_l1_displacement_m"]),
        "done_steps": list(record["done_steps"]),
        "synchronization": dict(record["synchronization"]),
        "env_step_wall_seconds": float(record["env_step_wall_seconds"]),
    }
    if include_trace:
        output["clearance_trace_m"] = record["clearance_trace_m"].tolist()
    return output


def _score(
    record: Mapping[str, Any],
    correction: Any,
    target_terminal_eef: Any,
    config: Mapping[str, Any],
) -> dict[str, float]:
    import numpy as np

    from main.multilink_ellipsoid.repulsive_force import softmin

    score = config["score"]
    row_minima = np.asarray(record["row_minimum_clearance_m"], dtype=np.float64)
    clearance = softmin(row_minima, float(score["softmin_temperature_m"]))
    terminal_error = float(
        np.linalg.norm(
            np.asarray(record["eef_position_trace_m"][-1], dtype=np.float64)
            - np.asarray(target_terminal_eef, dtype=np.float64)
        )
    )
    correction_l2 = float(np.linalg.norm(np.asarray(correction, dtype=np.float64)))
    terminal_penalty = float(score["task_terminal_eef_weight_per_m"]) * terminal_error**2
    deviation_penalty = float(score["deviation_weight_m_per_action2"]) * correction_l2**2
    return {
        "clearance_softmin_m": clearance,
        "terminal_eef_error_m": terminal_error,
        "correction_l2_action": correction_l2,
        "terminal_penalty_m": terminal_penalty,
        "deviation_penalty_m": deviation_penalty,
        "utility_m": clearance - terminal_penalty - deviation_penalty,
    }


def _candidate(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    correction: Any,
    target_terminal_eef: Any,
    config: Mapping[str, Any],
    *,
    include_trace: bool = True,
) -> dict[str, Any]:
    import numpy as np

    limit = float(config["action_space"]["action_limit"])
    delta = np.asarray(correction, dtype=np.float64).reshape(15)
    actions = _actions_with_correction(base_actions, delta, limit)
    record = probe.rollout(env, actions)
    score = _score(record, delta, target_terminal_eef, config)
    gate = config["gate"]
    exact_safe = bool(
        float(record["minimum_clearance_m"]) >= float(gate["minimum_exact_clearance_m"])
        and len(record["protected_contacts"]) == int(gate["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
    )
    return {
        "correction": delta.tolist(),
        "endpoint_correction_sum": np.sum(delta.reshape(5, 3), axis=0).tolist(),
        "score": score,
        "exact_safe": exact_safe,
        "task_preserving": bool(
            score["terminal_eef_error_m"] <= float(gate["maximum_terminal_eef_error_m"])
        ),
        "rollout": _public_rollout(record, include_trace=include_trace),
        "_record": record,
    }


def _strip_private(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_private(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    reference_result_path: Path,
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
    from main.multilink_ellipsoid.counterfactual_field import (
        fit_paired_field,
        load_counterfactual_field_config,
        matched_random_p_value,
        paired_chunk_directions,
        project_direction,
        unit_direction,
    )
    from main.multilink_ellipsoid.repulsive_flow import five_action_row_model
    from main.multilink_ellipsoid.repulsive_force import softmin_weights
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_counterfactual_field_config(experiment_config_path)
    reference = _validate_reference_result(reference_result_path, config)
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
    _require(len(matches) == 1, "manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    geometry_config = load_shadow_config(geometry_config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    env = None
    probe_env = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            action = _archived_action(archived_actions, step)
            observation, _, done, _ = env.step(action.tolist())
            _require(not bool(done), "archived prefix completed before counterfactual state")

        base_actions = _reference_actions(reference, 182, 201)
        zero = np.zeros(15, dtype=np.float64)
        nominal = _candidate(probe, env, base_actions, zero, np.zeros(3), config)
        target_terminal_eef = np.asarray(
            nominal["_record"]["eef_position_trace_m"][-1], dtype=np.float64
        )
        nominal["score"] = _score(nominal["_record"], zero, target_terminal_eef, config)
        nominal_repeat = _candidate(probe, env, base_actions, zero, target_terminal_eef, config)
        _require(
            nominal["rollout"]["state_sha256"] == nominal_repeat["rollout"]["state_sha256"],
            "counterfactual nominal replay state differs",
        )
        _require(
            np.array_equal(
                nominal["_record"]["clearance_trace_m"],
                nominal_repeat["_record"]["clearance_trace_m"],
            ),
            "counterfactual nominal replay margin differs",
        )
        nominal_contact_steps = sorted(
            {int(event["step"]) for event in nominal["rollout"]["protected_contacts"]}
        )
        _require(
            int(config["state_protocol"]["known_reference_collision_step"])
            in nominal_contact_steps,
            "known action-197 reference collision did not reproduce",
        )

        sampling = config["sampling"]
        epsilon = float(sampling["paired_perturbation_action"])
        directions = paired_chunk_directions(
            int(sampling["paired_direction_count"]),
            int(sampling["direction_seed"]),
            base_actions[:5, :3],
            epsilon,
            float(config["action_space"]["action_limit"]),
        )
        pair_records = []
        plus_scores = []
        minus_scores = []
        total_rollouts = 2
        total_probe_wall = float(nominal["rollout"]["env_step_wall_seconds"]) + float(
            nominal_repeat["rollout"]["env_step_wall_seconds"]
        )
        for index, direction in enumerate(directions):
            positive = _candidate(
                probe, env, base_actions, epsilon * direction, target_terminal_eef, config, include_trace=False
            )
            negative = _candidate(
                probe, env, base_actions, -epsilon * direction, target_terminal_eef, config, include_trace=False
            )
            plus_scores.append(float(positive["score"]["utility_m"]))
            minus_scores.append(float(negative["score"]["utility_m"]))
            pair_records.append(
                {
                    "index": index,
                    "split": "fit" if index < int(sampling["fit_direction_count"]) else "heldout",
                    "direction": direction.tolist(),
                    "positive": _strip_private(positive),
                    "negative": _strip_private(negative),
                }
            )
            total_rollouts += 2
            total_probe_wall += float(positive["rollout"]["env_step_wall_seconds"])
            total_probe_wall += float(negative["rollout"]["env_step_wall_seconds"])
        fit = fit_paired_field(
            directions,
            plus_scores,
            minus_scores,
            epsilon,
            int(sampling["fit_direction_count"]),
            float(sampling["ridge"]),
        )
        learned_direction = unit_direction(fit["gradient"])

        analytical_epsilon = float(config["comparators"]["analytical_field_finite_difference_action"])
        base_five = probe.rollout(env, base_actions[:5])
        trace = np.asarray(base_five["clearance_trace_m"], dtype=np.float64)
        jacobian = np.zeros((5, 7, 15), dtype=np.float64)
        total_rollouts += 1
        total_probe_wall += float(base_five["env_step_wall_seconds"])
        for variable in range(15):
            slot, dimension = divmod(variable, 3)
            positive_actions = base_actions[:5].copy()
            negative_actions = base_actions[:5].copy()
            positive_actions[slot, dimension] = min(
                float(config["action_space"]["action_limit"]),
                positive_actions[slot, dimension] + analytical_epsilon,
            )
            negative_actions[slot, dimension] = max(
                -float(config["action_space"]["action_limit"]),
                negative_actions[slot, dimension] - analytical_epsilon,
            )
            denominator = float(
                positive_actions[slot, dimension] - negative_actions[slot, dimension]
            )
            _require(denominator > 0.0, "analytical field denominator differs")
            positive = probe.rollout(env, positive_actions)
            negative = probe.rollout(env, negative_actions)
            jacobian[:, :, variable] = (
                np.asarray(positive["clearance_trace_m"], dtype=np.float64)
                - np.asarray(negative["clearance_trace_m"], dtype=np.float64)
            ) / denominator
            total_rollouts += 2
            total_probe_wall += float(positive["env_step_wall_seconds"])
            total_probe_wall += float(negative["env_step_wall_seconds"])
        local_minima, local_rows = five_action_row_model(jacobian, trace)
        local_weights = softmin_weights(
            local_minima, float(config["score"]["softmin_temperature_m"])
        )
        fixed_raw = local_rows.T.dot(local_weights)
        fixed_direction = project_direction(fixed_raw, directions)

        radius = float(sampling["comparison_correction_l2_action"])
        learned_plus = _candidate(
            probe, env, base_actions, radius * learned_direction, target_terminal_eef, config
        )
        learned_minus = _candidate(
            probe, env, base_actions, -radius * learned_direction, target_terminal_eef, config
        )
        fixed_plus = _candidate(
            probe, env, base_actions, radius * fixed_direction, target_terminal_eef, config
        )
        fixed_minus = _candidate(
            probe, env, base_actions, -radius * fixed_direction, target_terminal_eef, config
        )
        named = [learned_plus, learned_minus, fixed_plus, fixed_minus]
        total_rollouts += len(named)
        total_probe_wall += sum(float(item["rollout"]["env_step_wall_seconds"]) for item in named)
        random_directions = paired_chunk_directions(
            int(config["comparators"]["matched_random_direction_count"]),
            int(config["comparators"]["random_seed"]),
            base_actions[:5, :3],
            radius,
            float(config["action_space"]["action_limit"]),
        )
        random_candidates = []
        for index, direction in enumerate(random_directions):
            candidate = _candidate(
                probe, env, base_actions, radius * direction, target_terminal_eef, config, include_trace=False
            )
            random_candidates.append({"index": index, "direction": direction.tolist(), **_strip_private(candidate)})
            total_rollouts += 1
            total_probe_wall += float(candidate["rollout"]["env_step_wall_seconds"])

        nominal_utility = float(nominal["score"]["utility_m"])
        learned_gain = float(learned_plus["score"]["utility_m"] - nominal_utility)
        random_gains = np.asarray(
            [candidate["score"]["utility_m"] - nominal_utility for candidate in random_candidates],
            dtype=np.float64,
        )
        random_p = matched_random_p_value(learned_gain, random_gains)
        gate = config["gate"]
        prediction_gate = bool(
            float(fit["heldout_pearson"]) >= float(gate["heldout_minimum_pearson"])
            and float(fit["heldout_sign_accuracy"]) >= float(gate["heldout_minimum_sign_accuracy"])
        )
        mechanism_gate = bool(
            prediction_gate
            and learned_plus["exact_safe"]
            and learned_plus["task_preserving"]
            and learned_gain > 0.0
            and learned_plus["score"]["utility_m"] > learned_minus["score"]["utility_m"]
            and (
                not bool(gate["learned_must_beat_fixed_utility"])
                or learned_plus["score"]["utility_m"] > fixed_plus["score"]["utility_m"]
            )
            and random_p <= float(gate["maximum_random_p_value"])
        )
        fit_public = {
            "gradient": fit["gradient"].tolist(),
            "gradient_l2": float(np.linalg.norm(fit["gradient"])),
            "learned_unit_direction": learned_direction.tolist(),
            "heldout_predictions": fit["heldout_predictions"].tolist(),
            "heldout_targets": fit["heldout_targets"].tolist(),
            "heldout_rmse": float(fit["heldout_rmse"]),
            "heldout_r2": float(fit["heldout_r2"]),
            "heldout_pearson": float(fit["heldout_pearson"]),
            "heldout_sign_accuracy": float(fit["heldout_sign_accuracy"]),
        }
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
                "source_commit": reference["source"]["commit"],
                "action_steps": [182, 201],
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
            "determinism": {
                "identical_state_hashes": True,
                "identical_clearance_trace": True,
                "nominal_contact_steps": nominal_contact_steps,
            },
            "base_actions_182_201": base_actions.tolist(),
            "nominal": _strip_private(nominal),
            "paired_rollouts": pair_records,
            "field_fit": fit_public,
            "fixed_analytical_field": {
                "local_row_minima_m": local_minima.tolist(),
                "local_row_weights": local_weights.tolist(),
                "raw_gradient": fixed_raw.tolist(),
                "projected_unit_direction": fixed_direction.tolist(),
            },
            "comparison": {
                "correction_l2_action": radius,
                "learned_plus": _strip_private(learned_plus),
                "learned_minus": _strip_private(learned_minus),
                "fixed_plus": _strip_private(fixed_plus),
                "fixed_minus": _strip_private(fixed_minus),
                "random_candidates": random_candidates,
                "learned_utility_gain_m": learned_gain,
                "fixed_utility_gain_m": float(fixed_plus["score"]["utility_m"] - nominal_utility),
                "random_utility_gains_m": random_gains.tolist(),
                "matched_random_p_value": random_p,
            },
            "prediction_gate_pass": prediction_gate,
            "mechanism_gate_pass": mechanism_gate,
            "interpretation": (
                "paired_long_horizon_ellipsoid_field_mechanism_supported"
                if mechanism_gate
                else "paired_long_horizon_ellipsoid_field_strict_no_go"
            ),
            "rollout_count": total_rollouts,
            "probe_env_step_wall_seconds": total_probe_wall,
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
                "mechanism_gate_pass": result["mechanism_gate_pass"],
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
