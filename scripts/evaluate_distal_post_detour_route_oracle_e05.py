#!/usr/bin/env python3
"""Test generic persistent route modes at the measured E05 action-187 state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_counterfactual_field_e05 import FixedContinuationProbe
from scripts.evaluate_distal_post_detour_live_gate_e05 import _heldout_gate
from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _archived_actions,
    _direct_smooth_field,
    _disable_images,
    _public,
    _result_actions,
    _validate_registered,
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


RESULT_SCHEMA = "vlsa_distal_post_detour_route_oracle_e05_result.v1"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "minimum_clearance_m": float(record["minimum_clearance_m"]),
        "row_minimum_clearance_m": _public(record["row_minimum_clearance_m"]),
        "protected_contacts": list(record["protected_contacts"]),
        "maximum_active_obstacle_l1_displacement_m": float(
            record["maximum_active_obstacle_l1_displacement_m"]
        ),
        "sample_count": int(record["sample_count"]),
        "substep_counts": list(record["substep_counts"]),
        "maximum_boundary_equivalence_error_m": float(
            record["maximum_boundary_equivalence_error_m"]
        ),
        "env_step_wall_seconds": float(record["env_step_wall_seconds"]),
    }


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    detour_path: Path,
    smooth_path: Path,
    live_gate_path: Path,
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
    from main.multilink_ellipsoid.post_detour_live_gate import (
        load_post_detour_route_config,
        persistent_route_correction,
        persistent_route_directions,
        verified_internal,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_post_detour_route_config(experiment_config_path)
    registered = config["registered_inputs"]
    detour = _validate_registered(
        detour_path,
        registered["detour_result"],
        "vlsa_distal_five_action_detour_e05_result.v1",
    )
    smooth = _validate_registered(
        smooth_path,
        registered["smooth_result"],
        "vlsa_distal_multi_witness_counterfactual_e05_result.v1",
    )
    live = _validate_registered(
        live_gate_path,
        registered["live_gate_result"],
        "vlsa_distal_post_detour_live_gate_e05_result.v1",
    )
    _require(live["interpretation"] == "post_detour_live_local_continuation_gate_no_go", "live gate verdict differs")
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
            observation, _, done, _ = env.step(
                _archived_actions(archived, step_index, step_index)[0].tolist()
            )
            _require(not done, "archived prefix completed before action 187")
        detour_actions = _result_actions(detour, 182, 201)
        second_stage = np.asarray(
            smooth["arms"]["smooth_max"]["best"]["correction"], dtype=np.float64
        ).reshape(5, 3)
        full_compound = detour_actions.copy()
        full_compound[:5, :3] += second_stage
        for command in full_compound[:5]:
            observation, _, done, _ = env.step(command.tolist())
            _require(not done, "compound prefix completed task before action 187")
        state_hash = hashlib.sha256(
            np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
        ).hexdigest()
        _require(state_hash == live["post_prefix_state"]["dynamic_state_sha256"], "action-187 state binding differs")

        expected_substeps = int(config["internal_verification"]["expected_mujoco_substeps_per_action"])
        tolerance = float(config["internal_verification"]["ordinary_env_step_boundary_equivalence_tolerance"])
        action_limit = float(config["route_search"]["action_limit"])
        nominal_actions = np.asarray(live["released_aegis_chunk"]["actions"], dtype=np.float64)
        positive_actions = full_compound[5:10]

        def candidate(actions: Any, requested: Any, source_name: str) -> dict[str, Any]:
            commands = np.asarray(actions, dtype=np.float64).copy()
            commands[:, :3] = np.clip(commands[:, :3], -action_limit, action_limit)
            applied = (commands[:, :3] - nominal_actions[:, :3]).reshape(15)
            record = instrumented.rollout_internal(
                env,
                commands,
                expected_substeps=expected_substeps,
                boundary_tolerance=tolerance,
                step_base=187,
            )
            return {
                "source": source_name,
                "actions": commands.tolist(),
                "requested_correction": np.asarray(requested, dtype=np.float64).reshape(15).tolist(),
                "applied_correction": applied.tolist(),
                "applied_correction_l2_action": float(np.linalg.norm(applied)),
                "clipped_coordinate_count": int(
                    np.count_nonzero(np.abs(np.asarray(actions)[:, :3] - commands[:, :3]) > 1.0e-12)
                ),
                "record": _summary(record),
                "verification_gate": verified_internal(record, config["gate"]),
            }

        nominal = candidate(nominal_actions, np.zeros(15), "fixed_live_pi05_released_aegis")
        positive_delta = (positive_actions[:, :3] - nominal_actions[:, :3]).reshape(15)
        positive = candidate(positive_actions, positive_delta, "registered_safe_continuation")
        _require(not nominal["verification_gate"], "registered live nominal unexpectedly safe")
        _require(positive["verification_gate"], "registered continuation positive control failed")

        nominal_boundary = boundary.rollout(env, nominal_actions, step_base=187)
        active_row = int(np.argmin(np.asarray(nominal_boundary["row_minimum_clearance_m"])[:7]))
        links = geometry._slabbed_links(env)
        routes = persistent_route_directions(links[active_row].center, geometry.obstacle.center)
        route_records: dict[str, Any] = {}
        for mode in config["route_search"]["modes"]:
            records = []
            for norm in config["route_search"]["coarse_norms_action"]:
                correction = persistent_route_correction(routes[mode], float(norm))
                actions = nominal_actions.copy()
                actions[:, :3] += correction.reshape(5, 3)
                records.append(candidate(actions, correction, "persistent_%s" % mode))
            route_records[mode] = records

        all_coarse = [item for records in route_records.values() for item in records]
        best_refine_base = max(
            [item for item in all_coarse if item["applied_correction_l2_action"] <= 1.0 + 1.0e-10],
            key=lambda item: float(item["record"]["minimum_clearance_m"]),
        )
        field_config = dict(config)
        field_config["action_space"] = dict(config["field_refinement"])
        field = _direct_smooth_field(
            boundary,
            instrumented,
            env,
            np.asarray(best_refine_base["actions"], dtype=np.float64),
            field_config,
            step_base=187,
        )
        field_actions = np.asarray(field["best_boundary"]["actions"], dtype=np.float64)
        field_requested = (field_actions[:, :3] - nominal_actions[:, :3]).reshape(15)
        field_final = candidate(field_actions, field_requested, "best_route_plus_smooth_refinement")
        field_heldout = _heldout_gate(field, config["gate"])
        field_final["heldout_direction_gate"] = field_heldout
        field_final["within_total_radius"] = bool(
            field_final["applied_correction_l2_action"]
            <= float(config["derivative_free"]["maximum_correction_l2_action"]) + 1.0e-10
        )

        search = config["derivative_free"]
        rng = np.random.RandomState(int(search["seed"]))
        pool: list[dict[str, Any]] = []
        elites: list[dict[str, Any]] = []
        generation_records = []
        maximum_radius = float(search["maximum_correction_l2_action"])
        for generation in range(int(search["generations"])):
            current = []
            for candidate_index in range(int(search["candidate_count_per_generation"])):
                raw = rng.normal(size=15)
                raw /= np.linalg.norm(raw)
                if generation == 0 or not elites:
                    correction = raw * rng.uniform(0.25, maximum_radius)
                else:
                    parent = np.asarray(
                        elites[candidate_index % len(elites)]["applied_correction"], dtype=np.float64
                    )
                    correction = parent + maximum_radius * (0.5 ** generation) * raw
                    correction_norm = float(np.linalg.norm(correction))
                    if correction_norm > maximum_radius:
                        correction *= maximum_radius / correction_norm
                actions = nominal_actions.copy()
                actions[:, :3] += correction.reshape(5, 3)
                current.append(candidate(actions, correction, "derivative_free"))
            pool.extend(current)
            elites = sorted(
                pool,
                key=lambda item: (
                    bool(item["verification_gate"]),
                    float(item["record"]["minimum_clearance_m"]),
                    -float(item["applied_correction_l2_action"]),
                ),
                reverse=True,
            )[: int(search["elite_count"])]
            generation_records.append(
                {
                    "generation": generation,
                    "candidate_count": len(current),
                    "safe_count": sum(bool(item["verification_gate"]) for item in current),
                    "best": elites[0],
                }
            )
        derivative_best = elites[0]
        structured_safe = [item for item in all_coarse if item["verification_gate"]]
        route_refinement_safe = bool(
            field_final["verification_gate"]
            and field_final["heldout_direction_gate"]
            and field_final["within_total_radius"]
        )
        derivative_safe = bool(derivative_best["verification_gate"])
        passed = bool(structured_safe or route_refinement_safe)
        if passed:
            interpretation = "generic_persistent_route_oracle_rediscovered_safe_action187_continuation"
        elif derivative_safe:
            interpretation = "full_action_search_succeeds_but_generic_persistent_route_family_fails"
        else:
            interpretation = "action187_route_oracle_no_go_within_registered_family_and_budget"
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
                "detour_payload_sha256": detour["result_payload_sha256"],
                "smooth_payload_sha256": smooth["result_payload_sha256"],
                "live_gate_payload_sha256": live["result_payload_sha256"],
            },
            "pairing": pairing,
            "geometry_config": geometry_config,
            "geometry": geometry.geometry_record(env),
            "probe_environment": {
                "disabled_image_observable_count": disabled_images,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
                "model_timestep_s": float(probe_env.env.model_timestep),
                "control_timestep_s": float(probe_env.env.control_timestep),
            },
            "post_prefix_state": {"step": 187, "dynamic_state_sha256": state_hash},
            "route_basis": {
                "active_nominal_row": active_row,
                "active_body_name": links[active_row].body_name,
                "robot_center_m": links[active_row].center.tolist(),
                "obstacle_center_m": geometry.obstacle.center.tolist(),
                "directions": {key: value.tolist() for key, value in routes.items()},
                "positive_control_l2_action": float(np.linalg.norm(positive_delta)),
                "positive_control_cosines": {
                    key: float(
                        np.dot(persistent_route_correction(value, 1.0), positive_delta)
                        / np.linalg.norm(positive_delta)
                    )
                    for key, value in routes.items()
                },
            },
            "arms": {
                "fixed_live_nominal": nominal,
                "registered_positive_control": positive,
                "persistent_routes": route_records,
                "best_route_plus_smooth_refinement": {
                    "base": best_refine_base,
                    "optimization": _public(field),
                    "final": field_final,
                },
                "full_action_derivative_free": {
                    "generations": generation_records,
                    "candidate_count": len(pool),
                    "safe_count": sum(bool(item["verification_gate"]) for item in pool),
                    "best": derivative_best,
                },
            },
            "gate": {
                "nominal_unsafe": not bool(nominal["verification_gate"]),
                "positive_control_verified_safe": bool(positive["verification_gate"]),
                "persistent_route_safe_count": len(structured_safe),
                "route_plus_smooth_verified_safe": route_refinement_safe,
                "derivative_free_verified_safe": derivative_safe,
                "passed": passed,
            },
            "interpretation": interpretation,
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
    parser.add_argument("--live-gate-result", type=Path, required=True)
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
        live_gate_path=args.live_gate_result.resolve(),
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
                "passed": result["gate"]["passed"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
