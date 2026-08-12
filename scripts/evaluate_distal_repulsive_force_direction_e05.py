#!/usr/bin/env python3
"""Run the preregistered local learned-repulsive-direction gate on E05."""

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
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_repulsive_force_direction_e05_result.v1"


def _disable_images(env: Any) -> int:
    disabled = 0
    for observable in env.env._observables.values():
        if str(getattr(observable, "modality", "")) == "image":
            observable.set_enabled(False)
            disabled += 1
    return disabled


def _json_action(record: Mapping[str, Any], step: int) -> list[float]:
    _require(int(record["step"]) == step, "archived action index differs")
    action = record["env_step_input"]
    _require(action == record["executed"], "archived executed action differs")
    _require(isinstance(action, list) and len(action) == 7, "archived action shape differs")
    return [float(value) for value in action]


class TwoActionSlabProbe:
    """Reset a cloned OSC environment and measure a complete two-action prefix."""

    def __init__(self, one_step_probe: Any) -> None:
        self.one_step_probe = one_step_probe

    @property
    def env(self) -> Any:
        return self.one_step_probe.probe_env

    def transition(
        self,
        main_env: Any,
        first_action: Sequence[float],
        second_action: Sequence[float],
    ) -> dict[str, Any]:
        import numpy as np

        from main.multilink_ellipsoid.rollout import _dynamic_state_vector
        from main.multilink_ellipsoid.sitl_candidate import (
            _obstacle_root_body_id,
            _protected_contact_evidence,
        )

        synchronization = self.one_step_probe.synchronize(main_env)
        commands = [
            np.asarray(first_action, dtype=np.float64),
            np.asarray(second_action, dtype=np.float64),
        ]
        if any(command.shape != (7,) or not np.all(np.isfinite(command)) for command in commands):
            raise ValueError("two-action probe command is invalid")
        obstacle_id = _obstacle_root_body_id(
            self.env.sim.model, self.one_step_probe.active_obstacle_name
        )
        obstacle_before = np.asarray(
            self.env.sim.data.xpos[obstacle_id], dtype=np.float64
        ).copy()
        steps = []
        started = time.perf_counter_ns()
        for command in commands:
            observation, _, done, _ = self.env.step(command.tolist())
            vector = _dynamic_state_vector(self.env)
            clearance = self.one_step_probe.clearances(self.env)
            contact = _protected_contact_evidence(
                self.env, self.one_step_probe.active_obstacle_name
            )
            obstacle_now = np.asarray(
                self.env.sim.data.xpos[obstacle_id], dtype=np.float64
            )
            steps.append(
                {
                    "action": command.tolist(),
                    "clearances_m": clearance.tolist(),
                    "state_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                    "state_vector": vector,
                    "eef_position_m": np.asarray(
                        observation["robot0_eef_pos"], dtype=np.float64
                    ).tolist(),
                    "raw_protected_contact": contact,
                    "active_obstacle_l1_displacement_m": float(
                        np.sum(np.abs(obstacle_now - obstacle_before))
                    ),
                    "done": bool(done),
                }
            )
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        row_trace = np.asarray(
            [step["clearances_m"][:7] for step in steps], dtype=np.float64
        )
        row_minimum = np.min(row_trace, axis=0)
        return {
            "synchronization": synchronization,
            "steps": steps,
            "row_minimum_m": row_minimum.tolist(),
            "minimum_row_m": float(np.min(row_minimum)),
            "env_step_wall_seconds": elapsed,
        }


def _public_rollout(record: Mapping[str, Any], temperature: float) -> dict[str, Any]:
    from main.multilink_ellipsoid.repulsive_force import softmin

    steps = []
    for item in record["steps"]:
        steps.append(
            {
                key: value
                for key, value in item.items()
                if key != "state_vector"
            }
        )
    row_minimum = [float(value) for value in record["row_minimum_m"]]
    return {
        "synchronization": record["synchronization"],
        "steps": steps,
        "row_minimum_m": row_minimum,
        "minimum_row_m": float(record["minimum_row_m"]),
        "softmin_margin_m": softmin(row_minimum, temperature),
        "env_step_wall_seconds": float(record["env_step_wall_seconds"]),
    }


def _safe_rollout(record: Mapping[str, Any], displacement_limit: float) -> bool:
    return bool(
        float(record["minimum_row_m"]) >= 0.0
        and max(
            float(step["active_obstacle_l1_displacement_m"])
            for step in record["steps"]
        )
        <= float(displacement_limit)
        and all(
            int(step["raw_protected_contact"]["nonpositive_protected_contact_count"])
            == 0
            for step in record["steps"]
        )
    )


def _basis(
    probe: TwoActionSlabProbe,
    env: Any,
    first_action: Any,
    second_action: Any,
    epsilon: float,
) -> dict[str, Any]:
    import numpy as np

    base = probe.transition(env, first_action, second_action)
    action = np.asarray(first_action, dtype=np.float64)
    plus_rows = []
    minus_rows = []
    denominators = []
    finite_difference_records = []
    for dimension in range(3):
        plus = action.copy()
        minus = action.copy()
        plus[dimension] = min(1.0, plus[dimension] + float(epsilon))
        minus[dimension] = max(-1.0, minus[dimension] - float(epsilon))
        denominator = float(plus[dimension] - minus[dimension])
        if denominator <= 0.0:
            raise ValueError("finite-difference denominator is nonpositive")
        plus_result = probe.transition(env, plus, second_action)
        minus_result = probe.transition(env, minus, second_action)
        plus_rows.append(plus_result["row_minimum_m"])
        minus_rows.append(minus_result["row_minimum_m"])
        denominators.append(denominator)
        finite_difference_records.append(
            {
                "dimension": dimension,
                "plus_action": plus.tolist(),
                "minus_action": minus.tolist(),
                "plus": plus_result,
                "minus": minus_result,
            }
        )
    plus_array = np.asarray(plus_rows, dtype=np.float64)
    minus_array = np.asarray(minus_rows, dtype=np.float64)
    rows = np.empty((7, 3), dtype=np.float64)
    for dimension in range(3):
        rows[:, dimension] = (
            plus_array[dimension] - minus_array[dimension]
        ) / denominators[dimension]
    if not np.all(np.isfinite(rows)):
        raise ValueError("finite-difference rows are nonfinite")
    return {
        "base": base,
        "rows_m_per_action": rows,
        "finite_difference": finite_difference_records,
    }


def _candidate_record(
    probe: TwoActionSlabProbe,
    env: Any,
    nominal_first: Any,
    nominal_second: Any,
    base_rows: Any,
    rows: Any,
    direction: Any,
    radius: float,
    temperature: float,
) -> dict[str, Any]:
    import numpy as np

    unit = np.asarray(direction, dtype=np.float64)
    correction = float(radius) * unit
    action = np.asarray(nominal_first, dtype=np.float64).copy()
    action[:3] += correction
    if np.max(np.abs(action[:3])) > 1.0 + 1.0e-12:
        raise ValueError("candidate action exceeds registered bounds")
    rollout = probe.transition(env, action, nominal_second)
    public = _public_rollout(rollout, temperature)
    affine_rows = np.asarray(base_rows, dtype=np.float64) + np.asarray(
        rows, dtype=np.float64
    ).dot(correction)
    return {
        "radius_action": float(radius),
        "direction": unit.tolist(),
        "correction": correction.tolist(),
        "action": action.tolist(),
        "affine_row_minimum_m": affine_rows.tolist(),
        "exact": public,
        "exact_safe": _safe_rollout(rollout, PAPER_CAR_THRESHOLD_M),
    }


def _state_dataset(
    probe: TwoActionSlabProbe,
    env: Any,
    step: int,
    first_action: Any,
    second_action: Any,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.repulsive_force import (
        paired_feasible_directions,
        softmin,
    )

    temperature = float(config["score"]["softmin_temperature_m"])
    basis = _basis(
        probe,
        env,
        first_action,
        second_action,
        float(config["finite_difference"]["perturbation_action"]),
    )
    base_rows = np.asarray(basis["base"]["row_minimum_m"], dtype=np.float64)
    rows = np.asarray(basis["rows_m_per_action"], dtype=np.float64)
    directions = paired_feasible_directions(
        int(config["sampling"]["paired_direction_count_per_state"]),
        int(seed),
        np.asarray(first_action, dtype=np.float64)[:3],
        max(float(value) for value in config["sampling"]["train_radii_action"]),
        float(config["sampling"]["action_limit"]),
    )
    candidates = []
    inputs = [base_rows.copy()]
    targets = [softmin(base_rows, temperature)]
    for radius in config["sampling"]["train_radii_action"]:
        for direction_index, direction in enumerate(directions):
            candidate = _candidate_record(
                probe,
                env,
                first_action,
                second_action,
                base_rows,
                rows,
                direction,
                float(radius),
                temperature,
            )
            candidate["direction_index"] = direction_index
            candidates.append(candidate)
            inputs.append(np.asarray(candidate["affine_row_minimum_m"], dtype=np.float64))
            targets.append(float(candidate["exact"]["softmin_margin_m"]))
    return {
        "step": int(step),
        "seed": int(seed),
        "nominal_first_action": np.asarray(first_action, dtype=np.float64).tolist(),
        "nominal_second_action": np.asarray(second_action, dtype=np.float64).tolist(),
        "basis": {
            "base": _public_rollout(basis["base"], temperature),
            "rows_m_per_action": rows.tolist(),
            "finite_difference": [
                {
                    "dimension": item["dimension"],
                    "plus_action": item["plus_action"],
                    "minus_action": item["minus_action"],
                    "plus": _public_rollout(item["plus"], temperature),
                    "minus": _public_rollout(item["minus"], temperature),
                }
                for item in basis["finite_difference"]
            ],
        },
        "candidates": candidates,
        "inputs": np.asarray(inputs, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.float64),
    }


def _public_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dataset.items()
        if key not in {"inputs", "targets"}
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
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _runtime_imports,
        _settle,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.repulsive_force import (
        fixed_repulsive_direction,
        learned_repulsive_direction,
        load_repulsive_force_config,
        matched_random_p_value,
        paired_feasible_directions,
        train_monotone_potential,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import (
        SlabbedEightConstraintProbe,
        _obstacle_root_body_id,
        _protected_contact_evidence,
    )

    started = time.perf_counter_ns()
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "archived Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "archived Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list)
        and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "archived action horizon differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary manifest row differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    geometry_config = load_shadow_config(geometry_config_path)
    config = load_repulsive_force_config(experiment_config_path)
    runtime = _runtime_imports(include_aegis=False)
    env = None
    probe_env = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=32
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
            "settled simulator states differ",
        )
        disabled_images = {
            "main": _disable_images(env),
            "probe": _disable_images(probe_env),
        }
        obstacle_name, _ = _active_obstacle(env, observation)
        probe_obstacle_name, _ = _active_obstacle(probe_env, probe_observation)
        _require(obstacle_name == probe_obstacle_name, "probe obstacle differs")
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
            _require(pairing[key] == archived["pairing"][key], "pairing field differs: %s" % key)
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
        _require(geometry_record["distal_ellipsoid_count"] == 7, "geometry row count differs")
        one_step_probe = SlabbedEightConstraintProbe(
            probe_env,
            geometry,
            clearance_m=0.0,
            active_obstacle_name=obstacle_name,
        )
        probe = TwoActionSlabProbe(one_step_probe)
        split = config["state_split"]
        target_steps = set(split["train_steps"] + split["validation_steps"] + split["test_steps"])
        state_records = {}
        train_datasets = []
        validation_datasets = []
        model_bundle = None
        model_record = None
        test_record = None
        continuation = {"attempted": False, "passed": False, "reason": "direction_gate_not_evaluated"}
        temperature = float(config["score"]["softmin_temperature_m"])
        for step in range(max(target_steps) + 1):
            first_action = np.asarray(_json_action(archived_actions[step], step), dtype=np.float64)
            second_action = np.asarray(_json_action(archived_actions[step + 1], step + 1), dtype=np.float64)
            if step in split["train_steps"]:
                dataset = _state_dataset(
                    probe,
                    env,
                    step,
                    first_action,
                    second_action,
                    config,
                    int(config["sampling"]["train_seed"]) + step,
                )
                train_datasets.append(dataset)
                state_records[str(step)] = _public_dataset(dataset)
            elif step in split["validation_steps"]:
                dataset = _state_dataset(
                    probe,
                    env,
                    step,
                    first_action,
                    second_action,
                    config,
                    int(config["sampling"]["validation_seed"]) + step,
                )
                validation_datasets.append(dataset)
                state_records[str(step)] = _public_dataset(dataset)
                train_x = np.concatenate([item["inputs"] for item in train_datasets], axis=0)
                train_y = np.concatenate([item["targets"] for item in train_datasets], axis=0)
                validation_x = np.concatenate([item["inputs"] for item in validation_datasets], axis=0)
                validation_y = np.concatenate([item["targets"] for item in validation_datasets], axis=0)
                model_bundle = train_monotone_potential(
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                    config["model"],
                )
                model_record = {
                    key: value
                    for key, value in model_bundle.items()
                    if key != "model"
                }
                model_record["frozen_before_test_step"] = True
                model_record["training_state_steps"] = split["train_steps"]
                model_record["validation_state_steps"] = split["validation_steps"]
            elif step in split["test_steps"]:
                _require(model_bundle is not None and model_record is not None, "model was not frozen before test")
                basis = _basis(
                    probe,
                    env,
                    first_action,
                    second_action,
                    float(config["finite_difference"]["perturbation_action"]),
                )
                base_rows = np.asarray(basis["base"]["row_minimum_m"], dtype=np.float64)
                local_rows = np.asarray(basis["rows_m_per_action"], dtype=np.float64)
                learned_direction, learned_weights = learned_repulsive_direction(
                    model_bundle, base_rows, local_rows
                )
                fixed_direction = fixed_repulsive_direction(
                    base_rows, local_rows, temperature
                )
                random_directions = paired_feasible_directions(
                    int(config["sampling"]["random_test_direction_count"]),
                    int(config["sampling"]["test_seed"]),
                    first_action[:3],
                    max(float(value) for value in config["sampling"]["test_radii_action"]),
                    float(config["sampling"]["action_limit"]),
                )
                base_public = _public_rollout(basis["base"], temperature)
                radius_records = []
                for radius in config["sampling"]["test_radii_action"]:
                    learned = _candidate_record(
                        probe, env, first_action, second_action, base_rows, local_rows,
                        learned_direction, float(radius), temperature,
                    )
                    fixed = _candidate_record(
                        probe, env, first_action, second_action, base_rows, local_rows,
                        fixed_direction, float(radius), temperature,
                    )
                    random_records = []
                    for direction_index, direction in enumerate(random_directions):
                        candidate = _candidate_record(
                            probe, env, first_action, second_action, base_rows, local_rows,
                            direction, float(radius), temperature,
                        )
                        candidate["direction_index"] = direction_index
                        random_records.append(candidate)
                    nominal_score = float(base_public["softmin_margin_m"])
                    learned_gain = float(learned["exact"]["softmin_margin_m"] - nominal_score)
                    fixed_gain = float(fixed["exact"]["softmin_margin_m"] - nominal_score)
                    random_gains = np.asarray(
                        [item["exact"]["softmin_margin_m"] - nominal_score for item in random_records],
                        dtype=np.float64,
                    )
                    p_value = matched_random_p_value(learned_gain, random_gains)
                    gate = config["gate"]
                    radius_pass = bool(
                        learned_gain >= float(gate["minimum_exact_clearance_gain_m"])
                        and learned_gain - fixed_gain >= float(gate["minimum_gain_over_fixed_m"])
                        and p_value <= float(gate["maximum_matched_random_p_value"])
                    )
                    radius_records.append(
                        {
                            "radius_action": float(radius),
                            "nominal_softmin_margin_m": nominal_score,
                            "learned": learned,
                            "fixed": fixed,
                            "random": random_records,
                            "learned_exact_gain_m": learned_gain,
                            "fixed_exact_gain_m": fixed_gain,
                            "gain_over_fixed_m": learned_gain - fixed_gain,
                            "matched_random_equally_or_better_count": int(np.sum(random_gains >= learned_gain)),
                            "matched_random_p_value": p_value,
                            "random_gain_median_m": float(np.median(random_gains)),
                            "random_gain_maximum_m": float(np.max(random_gains)),
                            "direction_gate_pass": radius_pass,
                        }
                    )
                safe_learned = [
                    item for item in radius_records if item["learned"]["exact_safe"]
                ]
                direction_pass = bool(
                    all(item["direction_gate_pass"] for item in radius_records)
                    and (safe_learned if config["gate"]["require_positive_exact_safe_support"] else True)
                )
                test_record = {
                    "step": step,
                    "model_sha256_before_test": model_record["model_sha256"],
                    "nominal_first_action": first_action.tolist(),
                    "nominal_second_action": second_action.tolist(),
                    "basis": {
                        "base": base_public,
                        "rows_m_per_action": local_rows.tolist(),
                        "finite_difference": [
                            {
                                "dimension": item["dimension"],
                                "plus_action": item["plus_action"],
                                "minus_action": item["minus_action"],
                                "plus": _public_rollout(item["plus"], temperature),
                                "minus": _public_rollout(item["minus"], temperature),
                            }
                            for item in basis["finite_difference"]
                        ],
                    },
                    "learned_direction": learned_direction.tolist(),
                    "learned_clearance_weights": learned_weights.tolist(),
                    "fixed_direction": fixed_direction.tolist(),
                    "direction_cosine_learned_fixed": float(np.dot(learned_direction, fixed_direction)),
                    "radii": radius_records,
                    "exact_safe_learned_radius_count": len(safe_learned),
                    "direction_gate_pass": direction_pass,
                }

                if direction_pass:
                    continuation = {
                        "attempted": True,
                        "passed": False,
                        "reason": None,
                        "steps": [],
                    }
                    start_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
                    nominal_terminal_eef = np.asarray(
                        base_public["steps"][-1]["eef_position_m"], dtype=np.float64
                    )
                    for continuation_offset in range(int(config["continuation"]["actions_to_execute"])):
                        continuation_step = step + continuation_offset
                        nominal_first = np.asarray(
                            _json_action(archived_actions[continuation_step], continuation_step), dtype=np.float64
                        )
                        nominal_second = np.asarray(
                            _json_action(archived_actions[continuation_step + 1], continuation_step + 1), dtype=np.float64
                        )
                        continuation_basis = _basis(
                            probe,
                            env,
                            nominal_first,
                            nominal_second,
                            float(config["finite_difference"]["perturbation_action"]),
                        )
                        continuation_rows = np.asarray(
                            continuation_basis["rows_m_per_action"], dtype=np.float64
                        )
                        continuation_base = np.asarray(
                            continuation_basis["base"]["row_minimum_m"], dtype=np.float64
                        )
                        continuation_direction, continuation_weights = learned_repulsive_direction(
                            model_bundle, continuation_base, continuation_rows
                        )
                        verified = []
                        for radius in config["sampling"]["test_radii_action"]:
                            candidate = _candidate_record(
                                probe, env, nominal_first, nominal_second,
                                continuation_base, continuation_rows,
                                continuation_direction, float(radius), temperature,
                            )
                            if candidate["exact_safe"]:
                                verified.append(candidate)
                        if not verified:
                            continuation["reason"] = "no_exact_safe_learned_candidate_at_step_%d" % continuation_step
                            break
                        selected = min(verified, key=lambda item: float(item["radius_action"]))
                        observation, _, done, _ = env.step(selected["action"])
                        actual_vector = _dynamic_state_vector(env)
                        expected_hash = selected["exact"]["steps"][0]["state_sha256"]
                        actual_hash = hashlib.sha256(actual_vector.tobytes()).hexdigest()
                        contact = _protected_contact_evidence(env, obstacle_name)
                        obstacle_id = _obstacle_root_body_id(env.sim.model, obstacle_name)
                        obstacle_displacement = float(
                            np.sum(
                                np.abs(
                                    np.asarray(env.sim.data.xpos[obstacle_id], dtype=np.float64)
                                    - initial_obstacle_position
                                )
                            )
                        )
                        step_pass = bool(
                            expected_hash == actual_hash
                            and int(contact["nonpositive_protected_contact_count"]) == 0
                            and obstacle_displacement <= PAPER_CAR_THRESHOLD_M
                            and not bool(done)
                        )
                        continuation["steps"].append(
                            {
                                "step": continuation_step,
                                "selected": selected,
                                "learned_direction": continuation_direction.tolist(),
                                "learned_clearance_weights": continuation_weights.tolist(),
                                "actual_state_sha256": actual_hash,
                                "accepted_clone_state_sha256": expected_hash,
                                "exact_clone_match": expected_hash == actual_hash,
                                "raw_protected_contact": contact,
                                "active_obstacle_l1_displacement_from_episode_start_m": obstacle_displacement,
                                "step_pass": step_pass,
                            }
                        )
                        if not step_pass:
                            continuation["reason"] = "executed_transition_gate_failed_at_step_%d" % continuation_step
                            break
                    if len(continuation["steps"]) == int(config["continuation"]["actions_to_execute"]):
                        corrected_terminal_eef = np.asarray(
                            observation["robot0_eef_pos"], dtype=np.float64
                        )
                        nominal_displacement = nominal_terminal_eef - start_eef
                        corrected_displacement = corrected_terminal_eef - start_eef
                        denominator = float(np.dot(nominal_displacement, nominal_displacement))
                        progress_ratio = (
                            0.0
                            if denominator <= 1.0e-12
                            else float(np.dot(corrected_displacement, nominal_displacement) / denominator)
                        )
                        continuation["nominal_terminal_eef_position_m"] = nominal_terminal_eef.tolist()
                        continuation["corrected_terminal_eef_position_m"] = corrected_terminal_eef.tolist()
                        continuation["nominal_eef_displacement_m"] = float(np.linalg.norm(nominal_displacement))
                        continuation["corrected_eef_displacement_m"] = float(np.linalg.norm(corrected_displacement))
                        continuation["nominal_eef_progress_ratio"] = progress_ratio
                        continuation["passed"] = bool(
                            all(item["step_pass"] for item in continuation["steps"])
                            and progress_ratio >= float(config["continuation"]["minimum_nominal_eef_progress_ratio"])
                        )
                        continuation["reason"] = None if continuation["passed"] else "useful_motion_gate_failed"
                else:
                    continuation = {
                        "attempted": False,
                        "passed": False,
                        "reason": "direction_gate_failed",
                    }
                break

            observation, _, done, _ = env.step(first_action.tolist())
            _require(not bool(done), "archived replay completed before test state")

        _require(test_record is not None and model_record is not None, "test result is unavailable")
        overall_pass = bool(test_record["direction_gate_pass"] and continuation["passed"])
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
            "config": config,
            "geometry_config": geometry_config,
            "geometry": geometry_record,
            "pairing": pairing,
            "disabled_image_observables": disabled_images,
            "training_states": [state_records[str(step)] for step in split["train_steps"]],
            "validation_states": [state_records[str(step)] for step in split["validation_steps"]],
            "model": model_record,
            "test": test_record,
            "continuation": continuation,
            "direction_gate_pass": bool(test_record["direction_gate_pass"]),
            "continuation_gate_pass": bool(continuation["passed"]),
            "primary_problem_solved": overall_pass,
            "interpretation": (
                "local_learned_repulsive_mechanism_pass"
                if overall_pass
                else (
                    "learned_force_direction_no_go"
                    if not test_record["direction_gate_pass"]
                    else "learned_force_continuation_no_go"
                )
            ),
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
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
                "direction_gate_pass": result["direction_gate_pass"],
                "continuation_gate_pass": result["continuation_gate_pass"],
                "interpretation": result["interpretation"],
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
