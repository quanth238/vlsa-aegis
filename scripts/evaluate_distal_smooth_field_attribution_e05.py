#!/usr/bin/env python3
"""Directly attribute the E05 avoidance result to the smooth secant field."""

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


RESULT_SCHEMA = "vlsa_distal_smooth_field_attribution_e05_result.v1"


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


def _validate_registered(path: Path, expected: Mapping[str, Any], schema: str) -> dict[str, Any]:
    result = _load(path)
    _require(result.get("schema_version") == schema, "registered result schema differs")
    _require(
        result.get("result_payload_sha256") == expected["result_payload_sha256"],
        "registered result payload differs",
    )
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(
        _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        == claimed,
        "registered result self-hash differs",
    )
    _require(result["source"]["commit"] == expected["source_commit"], "registered source differs")
    return result


def _result_actions(result: Mapping[str, Any], start: int, stop: int) -> Any:
    import numpy as np

    by_step = {int(item["step"]): item for item in result["actions"]}
    actions = np.asarray([by_step[step]["action"] for step in range(start, stop + 1)], dtype=np.float64)
    _require(actions.shape == (stop - start + 1, 7), "registered action shape differs")
    return actions


def _archived_actions(result: Mapping[str, Any], start: int, stop: int) -> Any:
    import numpy as np

    return np.asarray([_archived_action(result["actions"], step) for step in range(start, stop + 1)])


class InstrumentedContinuationProbe(FixedContinuationProbe):
    """Record L5--L7 geometry and contacts after every MuJoCo model step."""

    def rollout_internal(
        self,
        main_env: Any,
        actions: Any,
        *,
        expected_substeps: int,
        boundary_tolerance: float,
        step_base: int = 182,
    ) -> dict[str, Any]:
        import numpy as np

        from main.multilink_ellipsoid.sitl_candidate import (
            _obstacle_root_body_id,
            _protected_contact_evidence,
        )

        commands = np.asarray(actions, dtype=np.float64)
        _require(
            commands.ndim == 2
            and commands.shape[1] == 7
            and 1 <= commands.shape[0] <= 25
            and np.all(np.isfinite(commands)),
            "instrumented continuation action shape differs",
        )
        synchronization = self.one_step_probe.synchronize(main_env)
        obstacle_id = _obstacle_root_body_id(self.env.sim.model, self.active_obstacle_name)
        obstacle_reference = np.asarray(self.obstacle_reference_position_m, dtype=np.float64).reshape(3)
        clearances = []
        contacts = []
        displacements = []
        sample_identities = []
        boundary_clearances = []
        boundary_errors = []
        substep_counts = []

        def measure(action_offset: int, substep: int) -> None:
            values = np.asarray(self.one_step_probe.clearances(self.env)[:7], dtype=np.float64)
            clearances.append(values)
            sample_identities.append(
                {"action_offset": int(action_offset), "substep": int(substep)}
            )
            evidence = _protected_contact_evidence(self.env, self.active_obstacle_name)
            for event in evidence["events"]:
                contacts.append(
                    {
                        "step": int(step_base) + max(0, int(action_offset)),
                        "action_offset": int(action_offset),
                        "substep": int(substep),
                        **event,
                    }
                )
            obstacle = np.asarray(self.env.sim.data.xpos[obstacle_id], dtype=np.float64)
            displacements.append(float(np.sum(np.abs(obstacle - obstacle_reference))))

        self.env.sim.forward()
        measure(-1, -1)
        started = time.perf_counter_ns()
        original_step = self.env.sim.step
        for action_offset, command in enumerate(commands):
            count_before = len(clearances)

            def instrumented_step(*args: Any, **kwargs: Any) -> Any:
                output = original_step(*args, **kwargs)
                measure(action_offset, len(clearances) - count_before)
                return output

            self.env.sim.step = instrumented_step
            try:
                self.env.step(command.tolist())
            finally:
                self.env.sim.step = original_step
            count = len(clearances) - count_before
            substep_counts.append(count)
            _require(count == int(expected_substeps), "MuJoCo substep count differs")
            boundary = np.asarray(self.one_step_probe.clearances(self.env)[:7], dtype=np.float64)
            boundary_clearances.append(boundary)
            error = float(np.max(np.abs(boundary - clearances[-1])))
            boundary_errors.append(error)
            _require(error <= float(boundary_tolerance), "internal/boundary clearance differs")
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        trace = np.asarray(clearances, dtype=np.float64)
        return {
            "clearance_trace_m": trace,
            "minimum_clearance_m": float(np.min(trace)),
            "row_minimum_clearance_m": np.min(trace, axis=0),
            "protected_contacts": contacts,
            "active_obstacle_l1_displacement_trace_m": displacements,
            "maximum_active_obstacle_l1_displacement_m": float(max(displacements)),
            "sample_count": int(trace.shape[0]),
            "sample_identities": sample_identities,
            "substep_counts": substep_counts,
            "boundary_clearance_trace_m": np.asarray(boundary_clearances),
            "maximum_boundary_equivalence_error_m": float(max(boundary_errors)),
            "synchronization": synchronization,
            "env_step_wall_seconds": elapsed,
        }


def _boundary_candidate(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    correction: Any,
    config: Mapping[str, Any],
    *,
    step_base: int = 182,
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


def _fit_rows(
    probe: FixedContinuationProbe,
    env: Any,
    base_actions: Any,
    correction: Any,
    config: Mapping[str, Any],
    iteration: int,
    *,
    step_base: int = 182,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.multi_witness_counterfactual import (
        bounded_paired_directions,
        fit_clearance_rows,
        smooth_min_direction,
    )
    from main.multilink_ellipsoid.smooth_field_attribution import smooth_min_value

    settings = config["field_estimation"]
    current_xyz = np.asarray(base_actions[:5, :3]) + np.asarray(correction).reshape(5, 3)
    radius = float(settings["paired_perturbation_action"])

    def paired(seed: int, count: int) -> tuple[Any, Any, Any, float]:
        directions = bounded_paired_directions(
            count, seed, current_xyz, radius, float(config["action_space"]["action_limit"])
        )
        plus = []
        minus = []
        wall = 0.0
        for direction in directions:
            positive = _boundary_candidate(
                probe,
                env,
                base_actions,
                np.asarray(correction) + radius * direction,
                config,
                step_base=step_base,
            )
            negative = _boundary_candidate(
                probe,
                env,
                base_actions,
                np.asarray(correction) - radius * direction,
                config,
                step_base=step_base,
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
    center = _boundary_candidate(
        probe, env, base_actions, correction, config, step_base=step_base
    )
    smooth = smooth_min_direction(
        center["_trace"].reshape(-1), fit["gradients"], float(settings["temperature_m"])
    )
    held_directions, held_plus, held_minus, held_wall = paired(
        int(settings["heldout_direction_seed"]) + iteration,
        int(settings["heldout_direction_count_per_iteration"]),
    )
    targets = []
    predictions = []
    for index, direction in enumerate(held_directions):
        positive_value = smooth_min_value(held_plus[index], float(settings["temperature_m"]))
        negative_value = smooth_min_value(held_minus[index], float(settings["temperature_m"]))
        targets.append((positive_value - negative_value) / (2.0 * radius))
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
            "cosine": None if denominator <= 1.0e-15 else float(np.dot(targets, predictions) / denominator),
            "sign_accuracy": float(np.mean(np.sign(targets) == np.sign(predictions))),
            "rmse_m_per_action": float(np.sqrt(np.mean((targets - predictions) ** 2))),
        },
        "rollout_count": int(2 * (len(directions) + len(held_directions)) + 1),
        "probe_wall_seconds": float(wall + held_wall + center["rollout"]["env_step_wall_seconds"]),
    }


def _direct_smooth_field(
    probe: FixedContinuationProbe,
    instrumented: InstrumentedContinuationProbe,
    env: Any,
    raw_actions: Any,
    config: Mapping[str, Any],
    *,
    step_base: int = 182,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.iterative_counterfactual_risk import (
        feasible_correction,
        project_mode_direction,
    )
    from main.multilink_ellipsoid.smooth_field_attribution import internal_verification_gate

    correction = np.zeros(15, dtype=np.float64)
    center = _boundary_candidate(
        probe, env, raw_actions, correction, config, step_base=step_base
    )
    path = [_public(center)]
    iterations = []
    path_length = 0.0
    rollout_count = 1
    stop_reason = "maximum_iterations"
    verified = None
    for iteration in range(int(config["action_space"]["maximum_iterations"])):
        local = _fit_rows(
            probe,
            env,
            raw_actions,
            correction,
            config,
            iteration,
            step_base=step_base,
        )
        rollout_count += int(local["rollout_count"])
        current_xyz = np.asarray(raw_actions[:5, :3]) + correction.reshape(5, 3)
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
                raw_actions[:5, :3],
                proposed,
                radius=float(config["action_space"]["maximum_total_correction_l2_action"]),
                action_limit=float(config["action_space"]["action_limit"]),
                preserve_endpoint=False,
            ):
                continue
            candidate = _boundary_candidate(
                probe, env, raw_actions, proposed, config, step_base=step_base
            )
            candidate["fraction"] = float(fraction)
            candidate["step_norm_action"] = step_norm
            candidates.append(candidate)
            rollout_count += 1
        improving = [
            item
            for item in candidates
            if float(item["hard_margin_m"]) > float(center["hard_margin_m"]) + 1.0e-9
        ]
        selected = max(improving, key=lambda item: float(item["hard_margin_m"])) if improving else None
        record = {
            "iteration": iteration,
            "center": _public(center),
            "heldout": _public(local["heldout"]),
            "maximum_weight": float(np.max(local["weights"])),
            "effective_witness_count": float(1.0 / np.sum(local["weights"] ** 2)),
            "candidates": [_public(item) for item in candidates],
            "selected": None if selected is None else _public(selected),
        }
        iterations.append(record)
        if selected is None:
            stop_reason = "no_exact_improving_line_search_step"
            break
        correction = np.asarray(selected["correction"], dtype=np.float64)
        center = selected
        path_length += float(selected["step_norm_action"])
        path.append(_public(center))
        internal = instrumented.rollout_internal(
            env,
            selected["actions"],
            expected_substeps=int(config["internal_verification"]["expected_mujoco_substeps_per_action"]),
            boundary_tolerance=float(
                config["internal_verification"]["ordinary_env_step_boundary_equivalence_tolerance"]
            ),
            step_base=step_base,
        )
        rollout_count += 1
        selected["internal_verification"] = internal
        if internal_verification_gate(internal, config["gate"]):
            verified = selected
            stop_reason = "verified_internal_substep_safe"
            break
    return {
        "correction": correction,
        "best_boundary": _public(center),
        "verified_candidate": None if verified is None else _public(verified),
        "path": path,
        "iterations": iterations,
        "path_length_action": path_length,
        "rollout_count": rollout_count,
        "stop_reason": stop_reason,
    }


def _verify_actions(
    instrumented: InstrumentedContinuationProbe,
    env: Any,
    actions: Any,
    config: Mapping[str, Any],
    *,
    step_base: int = 182,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.smooth_field_attribution import internal_verification_gate

    record = instrumented.rollout_internal(
        env,
        actions,
        expected_substeps=int(config["internal_verification"]["expected_mujoco_substeps_per_action"]),
        boundary_tolerance=float(
            config["internal_verification"]["ordinary_env_step_boundary_equivalence_tolerance"]
        ),
        step_base=step_base,
    )
    return {"record": _public(record), "verification_gate": internal_verification_gate(record, config["gate"])}


def _scale_correction(
    instrumented: InstrumentedContinuationProbe,
    env: Any,
    base_actions: Any,
    correction: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.smooth_field_attribution import (
        scale_grid,
        select_smallest_verified_scale,
    )

    records = []
    for scale in scale_grid(float(config["internal_verification"]["scale_grid_step"])):
        actions = _actions_with_correction(
            base_actions,
            float(scale) * np.asarray(correction),
            float(config["action_space"]["action_limit"]),
        )
        verified = _verify_actions(instrumented, env, actions, config)
        records.append({"scale": scale, **verified})
    selected = select_smallest_verified_scale(records)
    return {
        "grid": records,
        "smallest_verified": selected,
        "claim": "smallest_registered_scale_on_discovered_correction_ray_not_global_minimum_norm",
    }


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    detour_path: Path,
    smooth_path: Path,
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
    from main.multilink_ellipsoid.shadow import MultilinkEllipsoidShadow, allocation_record, load_shadow_config
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe
    from main.multilink_ellipsoid.smooth_field_attribution import load_smooth_field_attribution_config

    started = time.perf_counter_ns()
    config = load_smooth_field_attribution_config(experiment_config_path)
    inputs = config["registered_inputs"]
    detour = _validate_registered(
        detour_path, inputs["detour_result"], "vlsa_distal_five_action_detour_e05_result.v1"
    )
    smooth = _validate_registered(
        smooth_path, inputs["smooth_result"], "vlsa_distal_multi_witness_counterfactual_e05_result.v1"
    )
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    _require(len(archived["actions"]) == EXPECTED_ACTION_HORIZON, "Table-1 action horizon differs")
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
        _require(np.array_equal(probe_initial_state, selected_initial_state), "probe initial state differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle_position = np.asarray(observation["%s_pos" % obstacle_name]).copy()
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
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        disabled_images = _disable_images(probe_env)
        boundary = FixedContinuationProbe(one_step, obstacle_name, initial_obstacle_position)
        instrumented = InstrumentedContinuationProbe(one_step, obstacle_name, initial_obstacle_position)
        for step_index in range(182):
            observation, _, done, _ = env.step(_archived_action(archived["actions"], step_index).tolist())
            _require(not done, "archived prefix completed before attribution state")
        raw_actions = _archived_actions(archived, 182, 201)
        detour_actions = _result_actions(detour, 182, 201)
        detour_correction = (detour_actions[:5, :3] - raw_actions[:5, :3]).reshape(15)
        prior_smooth_correction = np.asarray(
            smooth["arms"]["smooth_max"]["best"]["correction"], dtype=np.float64
        )
        compound_actions = _actions_with_correction(
            detour_actions, prior_smooth_correction, float(config["action_space"]["action_limit"])
        )
        direct = _direct_smooth_field(boundary, instrumented, env, raw_actions, config)
        direct_correction = np.asarray(direct["correction"], dtype=np.float64)
        arms = {
            "raw_aegis": _verify_actions(instrumented, env, raw_actions, config),
            "earlier_five_action_detour": _verify_actions(instrumented, env, detour_actions, config),
            "detour_plus_smooth_compound": {
                **_verify_actions(instrumented, env, compound_actions, config),
                "total_correction_from_raw_l2_action": float(
                    np.linalg.norm(detour_correction + prior_smooth_correction)
                ),
                "scale_search_on_smooth_second_stage_ray": _scale_correction(
                    instrumented, env, detour_actions, prior_smooth_correction, config
                ),
            },
            "direct_raw_aegis_smooth_field": {
                "optimization": _public(direct),
                "final": _verify_actions(
                    instrumented,
                    env,
                    _actions_with_correction(
                        raw_actions, direct_correction, float(config["action_space"]["action_limit"])
                    ),
                    config,
                ),
                "scale_search_on_discovered_ray": _scale_correction(
                    instrumented, env, raw_actions, direct_correction, config
                ),
            },
        }
        direct_scale = arms["direct_raw_aegis_smooth_field"]["scale_search_on_discovered_ray"][
            "smallest_verified"
        ]
        gate = {
            "raw_aegis_unsafe": not bool(arms["raw_aegis"]["verification_gate"]),
            "direct_smooth_verified_safe": direct_scale is not None,
            "compound_comparator_verified_safe_diagnostic": bool(
                arms["detour_plus_smooth_compound"]["verification_gate"]
            ),
            "heldout_direction_gate": bool(
                direct["iterations"]
                and all(
                    item["heldout"]["cosine"] is not None
                    and float(item["heldout"]["cosine"])
                    >= float(config["gate"]["minimum_heldout_direction_cosine"])
                    and float(item["heldout"]["sign_accuracy"])
                    >= float(config["gate"]["minimum_heldout_sign_accuracy"])
                    for item in direct["iterations"]
                )
            ),
            "internal_substep_count_exact": all(
                count == int(config["internal_verification"]["expected_mujoco_substeps_per_action"])
                for arm in (
                    arms["raw_aegis"],
                    arms["earlier_five_action_detour"],
                    arms["detour_plus_smooth_compound"],
                )
                for count in arm["record"]["substep_counts"]
            ),
        }
        gate["passed"] = bool(
            gate["raw_aegis_unsafe"]
            and gate["direct_smooth_verified_safe"]
            and gate["heldout_direction_gate"]
            and gate["internal_substep_count_exact"]
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
            "registered_inputs": {
                "detour": {"path": str(detour_path), "payload_sha256": detour["result_payload_sha256"]},
                "smooth": {"path": str(smooth_path), "payload_sha256": smooth["result_payload_sha256"]},
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
            "arms": arms,
            "gate": gate,
            "interpretation": (
                "direct_smooth_counterfactual_field_attributed_on_raw_aegis"
                if gate["passed"]
                else "smooth_success_not_attributed_directly_to_raw_aegis"
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
    parser.add_argument("--detour-result", type=Path, required=True)
    parser.add_argument("--smooth-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        detour_path=args.detour_result.resolve(),
        smooth_path=args.smooth_result.resolve(),
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
