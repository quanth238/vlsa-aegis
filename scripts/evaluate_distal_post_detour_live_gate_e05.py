#!/usr/bin/env python3
"""Evaluate one focused live-policy continuation gate after the safe E05 detour."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_counterfactual_field_e05 import FixedContinuationProbe
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


RESULT_SCHEMA = "vlsa_distal_post_detour_live_gate_e05_result.v1"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _heldout_gate(field: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    iterations = field.get("iterations", [])
    return bool(
        iterations
        and all(
            item["heldout"]["cosine"] is not None
            and float(item["heldout"]["cosine"])
            >= float(gate["minimum_heldout_direction_cosine"])
            and float(item["heldout"]["sign_accuracy"])
            >= float(gate["minimum_heldout_sign_accuracy"])
            for item in iterations
        )
    )


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    detour_path: Path,
    smooth_path: Path,
    gate0_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    expected_commit: str,
    host: str,
    port: int,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _aegis_action,
        _build_environment,
        _eef_proxy,
        _policy_observation,
        _runtime_imports,
        _server_identity,
        _settle,
        array_sha256,
        pairing_record,
        query_seed,
        read_jsonl,
        translational_action,
        validate_case_row,
    )
    from main.multilink_ellipsoid.post_detour_live_gate import (
        load_post_detour_live_config,
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
    config = load_post_detour_live_config(experiment_config_path)
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
    gate0 = _validate_registered(
        gate0_path,
        registered["gate0_result"],
        "vlsa_distal_raw_multistart_gate0_e05_result.v1",
    )
    _require(
        gate0["interpretation"]
        == "five_action_prefix_insufficient_without_adaptive_or_modified_continuation_search_blocked",
        "registered Gate 0 verdict differs",
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
    runtime = _runtime_imports(include_aegis=True)
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
            _require(not done, "archived prefix completed before post-detour state")

        detour_actions = _result_actions(detour, 182, 201)
        second_stage = np.asarray(
            smooth["arms"]["smooth_max"]["best"]["correction"], dtype=np.float64
        ).reshape(5, 3)
        full_compound = detour_actions.copy()
        full_compound[:5, :3] += second_stage
        _require(float(np.max(np.abs(full_compound[:5, :3]))) <= 1.0 + 1.0e-10, "compound prefix exceeds bounds")
        expected_substeps = int(config["internal_verification"]["expected_mujoco_substeps_per_action"])
        tolerance = float(
            config["internal_verification"]["ordinary_env_step_boundary_equivalence_tolerance"]
        )
        prefix_record = instrumented.rollout_internal(
            env,
            full_compound[:5],
            expected_substeps=expected_substeps,
            boundary_tolerance=tolerance,
            step_base=182,
        )
        prefix_safe = verified_internal(prefix_record, config["gate"])
        _require(prefix_safe, "registered compound prefix is not internally safe")
        expected_post_prefix_state = np.asarray(
            _dynamic_state_vector(instrumented.env), dtype=np.float64
        ).copy()
        for command in full_compound[:5]:
            observation, _, done, _ = env.step(command.tolist())
            _require(not done, "compound prefix completed task before live gate")
        state_error = float(
            np.max(np.abs(_dynamic_state_vector(env) - expected_post_prefix_state))
        )
        _require(state_error <= 1.0e-10, "executed compound prefix differs from clone")

        def verify(actions: Any) -> dict[str, Any]:
            record = instrumented.rollout_internal(
                env,
                actions,
                expected_substeps=expected_substeps,
                boundary_tolerance=tolerance,
                step_base=187,
            )
            return {
                "record": _public(record),
                "verification_gate": verified_internal(record, config["gate"]),
            }

        registered_continuation = verify(full_compound[5:10])
        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, int(port))
        server_identity = _server_identity(client)
        query_index = int(config["state_protocol"]["policy_query_index"])
        seed = query_seed(int(case["policy_noise_seed"]), query_index)
        policy_input = _policy_observation(
            runtime,
            observation,
            task_description=str(task.language),
            resize_size=224,
            rng_seed=seed,
        )
        query_started = time.perf_counter_ns()
        response = client.infer(policy_input)
        query_wall = (time.perf_counter_ns() - query_started) * 1.0e-9
        returned = np.asarray(response["actions"], dtype=np.float64)
        _require(
            returned.shape == (int(case["model_action_horizon"]), 7)
            and np.all(np.isfinite(returned)),
            "fresh live action chunk differs",
        )

        one_step.synchronize(env)
        nominal_observation = instrumented.env.env._get_observations()
        proxy = _eef_proxy(runtime, nominal_observation)
        p2 = np.asarray(perception["mvee_center"], dtype=np.float64)
        released_geometry = {
            "p2": p2,
            "R2": np.asarray(perception["mvee_rotation"], dtype=np.float64),
            "Q2_diag": np.asarray(perception["mvee_semiaxes"], dtype=np.float64),
            "z_fixed": np.asarray(archived["actions"][186]["qp"]["z_after"], dtype=np.float64),
        }
        q1_diag = np.asarray([0.06, 0.12, 0.11], dtype=np.float64)
        nominal_actions = []
        aegis_records = []
        for raw in returned[: int(config["state_protocol"]["policy_execute_prefix"])]:
            nominal, qp_record = _aegis_action(
                runtime,
                nominal_translational=translational_action(raw),
                proxy=proxy,
                geometry=released_geometry,
                q1_diag=q1_diag,
                diagnostics_enabled=True,
            )
            nominal_actions.append(nominal)
            nominal_observation, _, _, _ = instrumented.env.step(nominal)
            proxy = _eef_proxy(runtime, nominal_observation)
            aegis_records.append(qp_record)
        nominal_actions = np.asarray(nominal_actions, dtype=np.float64)
        _require(nominal_actions.shape == (5, 7), "nominal released AEGIS chunk differs")
        nominal = verify(nominal_actions)
        field = None
        field_final = None
        field_heldout = None
        if not nominal["verification_gate"]:
            field = _direct_smooth_field(
                boundary,
                instrumented,
                env,
                nominal_actions,
                config,
                step_base=187,
            )
            field_heldout = _heldout_gate(field, config["gate"])
            field_final = verify(
                np.asarray(field["best_boundary"]["actions"], dtype=np.float64)
            )
        live_safe = bool(nominal["verification_gate"])
        field_safe = bool(field_final and field_final["verification_gate"] and field_heldout)
        positive_controls = bool(prefix_safe and registered_continuation["verification_gate"])
        passed = bool(positive_controls and (live_safe or field_safe))
        if not positive_controls:
            interpretation = "registered_safe_compound_binding_failed"
        elif live_safe:
            interpretation = "fresh_pi05_continuation_locally_safe_without_additional_field"
        elif field_safe:
            interpretation = "smooth_field_recovers_fresh_pi05_local_continuation"
        else:
            interpretation = "post_detour_live_local_continuation_gate_no_go"
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
                "gate0_payload_sha256": gate0["result_payload_sha256"],
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
            "post_prefix_state": {
                "step": 187,
                "clone_state_max_abs_error": state_error,
                "dynamic_state_sha256": hashlib.sha256(
                    np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
                ).hexdigest(),
            },
            "policy_server": server_identity,
            "policy_query": {
                "query_index": query_index,
                "rng_seed": seed,
                "returned_actions_sha256": array_sha256(returned),
                "wall_seconds": query_wall,
                "server_timing": response.get("server_timing"),
            },
            "released_aegis_chunk": {
                "construction": config["controls"]["nominal_chunk_definition"],
                "actions": nominal_actions.tolist(),
                "actions_sha256": array_sha256(nominal_actions),
                "qp_records": aegis_records,
            },
            "arms": {
                "registered_compound_prefix": {
                    "record": _public(prefix_record),
                    "verification_gate": prefix_safe,
                },
                "registered_compound_continuation": registered_continuation,
                "fresh_pi05_released_aegis": nominal,
                "fresh_pi05_released_aegis_plus_smooth_field_if_needed": {
                    "status": "not_needed_nominal_safe" if field is None else "evaluated",
                    "optimization": None if field is None else _public(field),
                    "final": field_final,
                    "heldout_direction_gate": field_heldout,
                },
            },
            "gate": {
                "registered_compound_prefix_verified_safe": prefix_safe,
                "registered_compound_continuation_verified_safe": bool(
                    registered_continuation["verification_gate"]
                ),
                "fresh_pi05_continuation_verified_safe": live_safe,
                "smooth_field_continuation_verified_safe": field_safe,
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
    parser.add_argument("--gate0-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        detour_path=args.detour_result.resolve(),
        smooth_path=args.smooth_result.resolve(),
        gate0_path=args.gate0_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
        host=args.host,
        port=args.port,
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
