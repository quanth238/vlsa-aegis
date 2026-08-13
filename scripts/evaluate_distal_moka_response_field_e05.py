#!/usr/bin/env python3
"""Collect, train, and test the preregistered E05 Moka response field."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_counterfactual_field_e05 import (
    _actions_with_correction,
    _archived_action,
    _disable_images,
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


RESULT_SCHEMA = "vlsa_distal_moka_response_field_e05_result.v1"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _state_context(env: Any, actions: Any, obstacle_boxes: Sequence[Any]) -> Any:
    import numpy as np

    from main.multilink_ellipsoid.rollout import _controller_snapshot, _dynamic_state_vector

    commands = np.asarray(actions, dtype=np.float64)
    _require(commands.shape == (20, 7), "Moka response context actions differ")
    dynamic = _dynamic_state_vector(env)
    controller = _controller_snapshot(env)[0]
    controller_values = []
    for name in (
        "goal_pos",
        "goal_ori",
        "relative_ori",
        "ori_ref",
        "torques",
        "robot_torques",
        "gripper_current_action",
    ):
        value = controller[name]
        if value is not None:
            controller_values.extend(np.asarray(value, dtype=np.float64).reshape(-1).tolist())
    controller_values.append(float(controller["new_update"]))
    obstacle = []
    for box in obstacle_boxes:
        obstacle.extend(np.asarray(box.center, dtype=np.float64).tolist())
        obstacle.extend(np.asarray(box.rotation, dtype=np.float64).reshape(-1).tolist())
        obstacle.extend(np.asarray(box.half_extents_m, dtype=np.float64).tolist())
    context = np.concatenate(
        [
            dynamic,
            np.asarray(controller_values, dtype=np.float64),
            commands.reshape(-1),
            np.asarray(obstacle, dtype=np.float64),
        ]
    )
    _require(np.all(np.isfinite(context)), "Moka response context is nonfinite")
    return context


class MokaContinuationProbe:
    def __init__(self, one_step_probe: Any, obstacle_name: str) -> None:
        self.one_step_probe = one_step_probe
        self.obstacle_name = str(obstacle_name)

    @property
    def env(self) -> Any:
        return self.one_step_probe.probe_env

    def rollout(self, main_env: Any, actions: Any, *, step_base: int) -> dict[str, Any]:
        import numpy as np

        from main.multilink_ellipsoid.moka_response_field import (
            compiled_box_ellipsoids,
            multi_primitive_link_margins,
        )
        from main.multilink_ellipsoid.obstacle_proxy_audit import (
            compiled_obstacle_boxes,
            evaluate_obstacle_representations,
        )
        from main.multilink_ellipsoid.sitl_candidate import _protected_contact_evidence

        commands = np.asarray(actions, dtype=np.float64)
        _require(commands.shape == (20, 7), "Moka response rollout actions differ")
        synchronization = self.one_step_probe.synchronize(main_env)
        traces = []
        exact_overlap = []
        contacts = []
        wall_started = time.perf_counter_ns()
        for offset, command in enumerate(commands):
            self.env.step(command.tolist())
            links = self.one_step_probe.geometry._slabbed_links(self.env)
            boxes = compiled_obstacle_boxes(self.env, self.obstacle_name)
            obstacles = compiled_box_ellipsoids(boxes)
            traces.append(multi_primitive_link_margins(links, obstacles))
            exact = evaluate_obstacle_representations(links, self.one_step_probe.geometry.obstacle, boxes)
            exact_overlap.append(bool(exact["compiled_box_union_any_exact_solid_overlap"]))
            evidence = _protected_contact_evidence(self.env, self.obstacle_name)
            for event in evidence["events"]:
                contacts.append({"step": int(step_base) + offset, **event})
        elapsed = (time.perf_counter_ns() - wall_started) * 1.0e-9
        trace = np.asarray(traces, dtype=np.float64)
        _require(trace.shape == (20, 7), "Moka response trace differs")
        return {
            "trace_m": trace,
            "minimum_m": float(np.min(trace)),
            "exact_box_overlap": exact_overlap,
            "protected_contacts": contacts,
            "synchronization": synchronization,
            "env_step_wall_seconds": elapsed,
        }


def _public_rollout(value: Mapping[str, Any], *, trace: bool) -> dict[str, Any]:
    output = {
        "minimum_m": float(value["minimum_m"]),
        "exact_box_overlap": list(value["exact_box_overlap"]),
        "exact_box_overlap_count": int(sum(value["exact_box_overlap"])),
        "protected_contacts": list(value["protected_contacts"]),
        "protected_contact_count": len(value["protected_contacts"]),
        "synchronization": dict(value["synchronization"]),
        "env_step_wall_seconds": float(value["env_step_wall_seconds"]),
    }
    if trace:
        output["trace_m"] = value["trace_m"].tolist()
    return output


def _fit_rows(directions: Any, plus: Any, minus: Any, epsilon: float, ridge: float) -> dict[str, Any]:
    import numpy as np

    design = np.asarray(directions, dtype=np.float64)
    positive = np.asarray(plus, dtype=np.float64).reshape(design.shape[0], -1)
    negative = np.asarray(minus, dtype=np.float64).reshape(design.shape[0], -1)
    targets = (positive - negative) / (2.0 * float(epsilon))
    matrix = design.T.dot(design) + float(ridge) * np.eye(15)
    gradients = np.linalg.solve(matrix, design.T.dot(targets)).T
    return {"targets": targets, "gradients": gradients}


def _soft_direction(values: Any, rows: Any, temperature: float) -> Any:
    import numpy as np

    from main.multilink_ellipsoid.repulsive_force import softmin_weights

    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    gradients = np.asarray(rows, dtype=np.float64).reshape(flattened.size, 15)
    raw = softmin_weights(flattened, float(temperature)).dot(gradients)
    norm = float(np.linalg.norm(raw))
    _require(norm > 1.0e-12 and np.all(np.isfinite(raw)), "Moka response direction is degenerate")
    return raw / norm


def _direction_metrics(prediction: Any, target: Any) -> dict[str, float]:
    import numpy as np

    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    exact = np.asarray(target, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(predicted) * np.linalg.norm(exact))
    cosine = float("nan") if denominator <= 1.0e-20 else float(np.dot(predicted, exact) / denominator)
    mask = np.abs(exact) > 1.0e-8
    sign = float("nan") if not np.any(mask) else float(np.mean(np.sign(predicted[mask]) == np.sign(exact[mask])))
    return {
        "cosine": cosine,
        "sign_accuracy": sign,
        "rmse_m_per_action": float(np.sqrt(np.mean((predicted - exact) ** 2))),
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
    from main.multilink_ellipsoid.counterfactual_field import matched_random_p_value
    from main.multilink_ellipsoid.moka_response_field import (
        load_moka_response_config,
        predict_response,
        train_response_model,
    )
    from main.multilink_ellipsoid.multi_witness_counterfactual import bounded_paired_directions
    from main.multilink_ellipsoid.obstacle_proxy_audit import compiled_obstacle_boxes
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_moka_response_config(experiment_config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(isinstance(archived_actions, list) and len(archived_actions) == EXPECTED_ACTION_HORIZON, "Table-1 horizon differs")
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "Moka manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
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
        _require(str(task.language) == str(probe_task.language), "Moka probe task differs")
        _require(np.array_equal(np.asarray(selected_initial_state), np.asarray(probe_initial_state)), "Moka initial state differs")
        _require(np.array_equal(np.asarray(env.sim.get_state().flatten()), np.asarray(probe_env.sim.get_state().flatten())), "Moka settled state differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        _require("moka" in obstacle_name.lower(), "E05 active obstacle is not Moka")
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
            _require(pairing[key] == archived["pairing"][key], "Moka pairing differs: %s" % key)
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
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        probe = MokaContinuationProbe(one_step, obstacle_name)
        split = config["state_split"]
        state_steps = sorted(split["train_steps"] + split["validation_steps"] + split["test_steps"])
        state_records = []
        train_states = []
        validation_states = []
        test_states = []
        model_bundle = None
        model_record = None
        total_rollouts = 0
        total_probe_wall = 0.0
        epsilon = float(config["sampling"]["paired_perturbation_action"])
        fit_count = int(config["sampling"]["fit_direction_count"])
        for step in range(max(state_steps) + 1):
            if step in state_steps:
                base_actions = np.asarray(
                    [_archived_action(archived_actions, index) for index in range(step, step + 20)],
                    dtype=np.float64,
                )
                boxes = compiled_obstacle_boxes(env, obstacle_name)
                _require(len(boxes) == int(config["geometry"]["expected_compiled_moka_box_count"]), "Moka compiled box count differs")
                context = _state_context(env, base_actions, boxes)
                directions = bounded_paired_directions(
                    int(config["sampling"]["paired_direction_count_per_state"]),
                    int(config["sampling"]["direction_seed"]) + step,
                    base_actions[:5, :3],
                    epsilon,
                    float(config["action_space"]["action_limit"]),
                )
                base = probe.rollout(env, base_actions, step_base=step)
                total_rollouts += 1
                total_probe_wall += float(base["env_step_wall_seconds"])
                plus = []
                minus = []
                pairs = []
                for index, direction in enumerate(directions):
                    positive_actions = _actions_with_correction(
                        base_actions, epsilon * direction, float(config["action_space"]["action_limit"])
                    )
                    negative_actions = _actions_with_correction(
                        base_actions, -epsilon * direction, float(config["action_space"]["action_limit"])
                    )
                    positive = probe.rollout(env, positive_actions, step_base=step)
                    negative = probe.rollout(env, negative_actions, step_base=step)
                    plus.append(positive["trace_m"])
                    minus.append(negative["trace_m"])
                    total_rollouts += 2
                    total_probe_wall += float(positive["env_step_wall_seconds"]) + float(negative["env_step_wall_seconds"])
                    pairs.append(
                        {
                            "index": index,
                            "direction": direction.tolist(),
                            "positive_minimum_m": float(positive["minimum_m"]),
                            "negative_minimum_m": float(negative["minimum_m"]),
                            "positive_contact_count": len(positive["protected_contacts"]),
                            "negative_contact_count": len(negative["protected_contacts"]),
                        }
                    )
                fit = _fit_rows(
                    directions[:fit_count],
                    np.asarray(plus)[:fit_count],
                    np.asarray(minus)[:fit_count],
                    epsilon,
                    float(config["sampling"]["ridge"]),
                )
                heldout_targets = (
                    np.asarray(plus)[fit_count:] - np.asarray(minus)[fit_count:]
                ) / (2.0 * epsilon)
                state = {
                    "step": step,
                    "context": context,
                    "base_values_m": base["trace_m"],
                    "fit_directions": directions[:fit_count],
                    "fit_targets_m_per_action": fit["targets"],
                    "fit_gradients_m_per_action": fit["gradients"],
                    "heldout_directions": directions[fit_count:],
                    "heldout_targets_m_per_action": heldout_targets.reshape(len(directions) - fit_count, 140),
                    "pairs": pairs,
                    "base": base,
                    "base_actions": base_actions,
                }
                if step in split["train_steps"]:
                    train_states.append(state)
                    split_name = "train"
                elif step in split["validation_steps"]:
                    validation_states.append(state)
                    split_name = "validation"
                else:
                    _require(model_bundle is not None, "Moka response model was not frozen before test")
                    state["snapshot"] = {
                        "simulator": np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy(),
                        "auxiliary": None,
                        "controllers": None,
                    }
                    from main.multilink_ellipsoid.rollout import (
                        _auxiliary_sim_snapshot,
                        _controller_snapshot,
                    )
                    state["snapshot"]["auxiliary"] = _auxiliary_sim_snapshot(env)
                    state["snapshot"]["controllers"] = _controller_snapshot(env)
                    test_states.append(state)
                    split_name = "test"
                state_records.append(
                    {
                        "step": step,
                        "split": split_name,
                        "context_sha256": hashlib.sha256(context.tobytes()).hexdigest(),
                        "base": _public_rollout(base, trace=True),
                        "paired_rollouts": pairs,
                        "fit_direction_count": fit_count,
                        "heldout_direction_count": len(directions) - fit_count,
                    }
                )
                if step == max(split["validation_steps"]):
                    model_bundle = train_response_model(train_states, validation_states, config["model"])
                    model_record = {
                        key: value
                        for key, value in model_bundle.items()
                        if key not in {"model", "device", "feature_mean", "feature_scale"}
                    }
                    model_record["feature_mean"] = model_bundle["feature_mean"].tolist()
                    model_record["feature_scale"] = model_bundle["feature_scale"].tolist()
                    model_record["frozen_before_test"] = True
            action = np.asarray(_archived_action(archived_actions, step), dtype=np.float64)
            observation, _, done, _ = env.step(action.tolist())
            _require(not bool(done) or step >= max(state_steps), "Moka episode ended before response states")

        _require(model_bundle is not None and model_record is not None, "Moka response model missing")
        temperature = float(config["score"]["softmin_temperature_m"])
        test_results = []
        passed_states = 0
        for state in test_states:
            from main.multilink_ellipsoid.rollout import (
                _restore_auxiliary_sim_snapshot,
                _restore_controller_snapshot,
            )
            snapshot = state["snapshot"]
            env.sim.set_state_from_flattened(snapshot["simulator"])
            _restore_auxiliary_sim_snapshot(env, snapshot["auxiliary"])
            _restore_controller_snapshot(env, snapshot["controllers"])
            env.sim.forward()
            _restore_auxiliary_sim_snapshot(env, snapshot["auxiliary"])
            predicted_values, predicted_rows = predict_response(model_bundle, state["context"])
            exact_rows = np.asarray(state["fit_gradients_m_per_action"], dtype=np.float64).reshape(20, 7, 15)
            learned_direction = _soft_direction(predicted_values, predicted_rows, temperature)
            exact_direction = _soft_direction(state["base_values_m"], exact_rows, temperature)
            fixed_direction = _soft_direction(
                np.asarray(state["base_values_m"])[:5], exact_rows[:5], temperature
            )
            directional_prediction = np.asarray(state["heldout_directions"]).dot(
                predicted_rows.reshape(140, 15).T
            )
            directional_metrics = _direction_metrics(
                directional_prediction, state["heldout_targets_m_per_action"]
            )
            direction_cosine = float(np.dot(learned_direction, exact_direction))
            nominal_score = float(np.min(state["base_values_m"]))
            random_directions = bounded_paired_directions(
                int(config["sampling"]["matched_random_direction_count"]),
                int(config["sampling"]["random_seed"]) + int(state["step"]),
                state["base_actions"][:5, :3],
                max(config["sampling"]["test_radii_action"]),
                float(config["action_space"]["action_limit"]),
            )
            radii = []
            for radius in config["sampling"]["test_radii_action"]:
                named = {}
                for name, direction in (
                    ("learned", learned_direction),
                    ("fixed", fixed_direction),
                    ("exact_local_secant", exact_direction),
                ):
                    actions = _actions_with_correction(
                        state["base_actions"],
                        float(radius) * direction,
                        float(config["action_space"]["action_limit"]),
                    )
                    candidate = probe.rollout(env, actions, step_base=int(state["step"]))
                    total_rollouts += 1
                    total_probe_wall += float(candidate["env_step_wall_seconds"])
                    named[name] = _public_rollout(candidate, trace=False)
                random_candidates = []
                for direction in random_directions:
                    actions = _actions_with_correction(
                        state["base_actions"],
                        float(radius) * direction,
                        float(config["action_space"]["action_limit"]),
                    )
                    candidate = probe.rollout(env, actions, step_base=int(state["step"]))
                    total_rollouts += 1
                    total_probe_wall += float(candidate["env_step_wall_seconds"])
                    random_candidates.append(float(candidate["minimum_m"] - nominal_score))
                learned_gain = float(named["learned"]["minimum_m"] - nominal_score)
                fixed_gain = float(named["fixed"]["minimum_m"] - nominal_score)
                p_value = matched_random_p_value(learned_gain, random_candidates)
                radius_pass = bool(
                    learned_gain >= float(config["gate"]["minimum_exact_gain_m"])
                    and learned_gain - fixed_gain >= float(config["gate"]["minimum_learned_fixed_gain_m"])
                    and p_value <= float(config["gate"]["maximum_matched_random_p_value"])
                    and (
                        not config["gate"]["require_zero_new_raw_protected_contacts"]
                        or named["learned"]["protected_contact_count"]
                        <= len(state["base"]["protected_contacts"])
                    )
                )
                radii.append(
                    {
                        "radius_action": float(radius),
                        "nominal_minimum_m": nominal_score,
                        "learned": named["learned"],
                        "fixed": named["fixed"],
                        "exact_local_secant": named["exact_local_secant"],
                        "learned_gain_m": learned_gain,
                        "fixed_gain_m": fixed_gain,
                        "learned_minus_fixed_gain_m": learned_gain - fixed_gain,
                        "matched_random_gain_median_m": float(np.median(random_candidates)),
                        "matched_random_gain_maximum_m": float(np.max(random_candidates)),
                        "matched_random_p_value": p_value,
                        "gate_pass": radius_pass,
                    }
                )
            best = max(radii, key=lambda item: item["learned_gain_m"])
            state_pass = bool(
                direction_cosine >= float(config["gate"]["minimum_learned_exact_oracle_cosine"])
                and directional_metrics["sign_accuracy"] >= float(config["gate"]["minimum_directional_sign_accuracy"])
                and best["gate_pass"]
            )
            passed_states += int(state_pass)
            test_results.append(
                {
                    "step": int(state["step"]),
                    "learned_direction": learned_direction.tolist(),
                    "fixed_direction": fixed_direction.tolist(),
                    "exact_local_secant_direction": exact_direction.tolist(),
                    "learned_exact_direction_cosine": direction_cosine,
                    "heldout_directional_metrics": directional_metrics,
                    "radii": radii,
                    "state_gate_pass": state_pass,
                }
            )
        gate_pass = bool(passed_states >= int(config["gate"]["minimum_test_state_pass_count"]))
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "pairing": pairing,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": ARCHIVED_FILE_SHA256,
                "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
                "read_only": True,
            },
            "geometry_config": geometry_config,
            "geometry": geometry.geometry_record(env),
            "compiled_moka_box_count": len(compiled_obstacle_boxes(env, obstacle_name)),
            "disabled_images": disabled_images,
            "state_records": state_records,
            "model": model_record,
            "test": test_results,
            "test_state_pass_count": passed_states,
            "learning_gate_pass": gate_pass,
            "corrected_execution_attempted": False,
            "interpretation": (
                "one_task_moka_controller_conditioned_response_supported"
                if gate_pass
                else "one_task_moka_controller_conditioned_response_strict_no_go"
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
                "learning_gate_pass": result["learning_gate_pass"],
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
