#!/usr/bin/env python3
"""Evaluate Gate 0 before any raw-AEGIS multi-start search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _archived_actions,
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


RESULT_SCHEMA = "vlsa_distal_raw_multistart_gate0_e05_result.v1"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    detour_path: Path,
    smooth_path: Path,
    attribution_path: Path,
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
    from main.multilink_ellipsoid.raw_multistart_gate0 import (
        internal_gate,
        load_raw_multistart_gate0_config,
        transplant_compound_prefix,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_raw_multistart_gate0_config(experiment_config_path)
    registered = config["registered_inputs"]
    detour = _validate_registered(
        detour_path, registered["detour_result"], "vlsa_distal_five_action_detour_e05_result.v1"
    )
    smooth = _validate_registered(
        smooth_path,
        registered["smooth_result"],
        "vlsa_distal_multi_witness_counterfactual_e05_result.v1",
    )
    attribution = _validate_registered(
        attribution_path,
        registered["attribution_result"],
        "vlsa_distal_smooth_field_attribution_e05_result.v1",
    )
    _require(attribution["interpretation"] == "smooth_success_not_attributed_directly_to_raw_aegis", "attribution verdict differs")
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
        instrumented = InstrumentedContinuationProbe(one_step, obstacle_name, initial_obstacle_position)
        for step_index in range(182):
            observation, _, done, _ = env.step(_archived_actions(archived, step_index, step_index)[0].tolist())
            _require(not done, "archived prefix completed before Gate 0")

        raw_actions = _archived_actions(archived, 182, 201)
        detour_actions = _result_actions(detour, 182, 201)
        second_stage = np.asarray(smooth["arms"]["smooth_max"]["best"]["correction"], dtype=np.float64).reshape(5, 3)
        full_compound = detour_actions.copy()
        full_compound[:5, :3] += second_stage
        _require(float(np.max(np.abs(full_compound[:5, :3]))) <= 1.0 + 1.0e-10, "full compound exceeds action bounds")
        transplant = transplant_compound_prefix(
            raw_actions, full_compound, float(config["normalization_contract"]["action_limit"])
        )
        _require(np.array_equal(transplant["actions"][5:], raw_actions[5:]), "raw suffix changed")
        _require(np.array_equal(transplant["actions"][:5], full_compound[:5]), "compound prefix changed")

        expected_substeps = int(config["internal_verification"]["expected_mujoco_substeps_per_action"])
        tolerance = float(config["internal_verification"]["ordinary_env_step_boundary_equivalence_tolerance"])

        def verify(actions: Any) -> dict[str, Any]:
            record = instrumented.rollout_internal(
                env,
                actions,
                expected_substeps=expected_substeps,
                boundary_tolerance=tolerance,
            )
            return {"record": _public(record), "verification_gate": internal_gate(record, config["gate"])}

        arms = {
            "raw_aegis": verify(raw_actions),
            "full_compound_trajectory": verify(full_compound),
            "transplanted_compound_prefix_raw_suffix": {
                **verify(transplant["actions"]),
                "requested_correction": _public(transplant["requested_correction"]),
                "applied_correction": _public(transplant["applied_correction"]),
                "clipping_delta": _public(transplant["clipping_delta"]),
                "clipped_coordinate_count": transplant["clipped_coordinate_count"],
                "requested_l2_action": transplant["requested_l2_action"],
                "applied_l2_action": transplant["applied_l2_action"],
                "continuation_identity": {
                    "raw_suffix_sha256": hashlib.sha256(raw_actions[5:].tobytes()).hexdigest(),
                    "transplanted_suffix_sha256": hashlib.sha256(transplant["actions"][5:].tobytes()).hexdigest(),
                    "compound_prefix_sha256": hashlib.sha256(full_compound[:5].tobytes()).hexdigest(),
                    "transplanted_prefix_sha256": hashlib.sha256(transplant["actions"][:5].tobytes()).hexdigest(),
                },
            },
        }
        raw_unsafe = not bool(arms["raw_aegis"]["verification_gate"])
        full_safe = bool(arms["full_compound_trajectory"]["verification_gate"])
        transplant_safe = bool(arms["transplanted_compound_prefix_raw_suffix"]["verification_gate"])
        if full_safe and transplant_safe:
            interpretation = "raw_fixed_suffix_contains_verified_safe_positive_control_search_authorized"
        elif full_safe and not transplant_safe:
            interpretation = "five_action_prefix_insufficient_without_adaptive_or_modified_continuation_search_blocked"
        elif not full_safe:
            interpretation = "full_compound_failed_internal_substep_authority_binding_or_transient_audit_required"
        else:
            interpretation = "gate0_unclassified"
        gate = {
            "raw_aegis_unsafe": raw_unsafe,
            "full_compound_verified_safe": full_safe,
            "transplanted_prefix_verified_safe": transplant_safe,
            "no_action_bound_clipping": transplant["clipped_coordinate_count"] == 0,
            "raw_suffix_bitwise_identical": arms["transplanted_compound_prefix_raw_suffix"]["continuation_identity"]["raw_suffix_sha256"] == arms["transplanted_compound_prefix_raw_suffix"]["continuation_identity"]["transplanted_suffix_sha256"],
            "compound_prefix_bitwise_identical": arms["transplanted_compound_prefix_raw_suffix"]["continuation_identity"]["compound_prefix_sha256"] == arms["transplanted_compound_prefix_raw_suffix"]["continuation_identity"]["transplanted_prefix_sha256"],
        }
        gate["search_authorized"] = bool(
            gate["raw_aegis_unsafe"]
            and gate["full_compound_verified_safe"]
            and gate["transplanted_prefix_verified_safe"]
            and gate["no_action_bound_clipping"]
            and gate["raw_suffix_bitwise_identical"]
            and gate["compound_prefix_bitwise_identical"]
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
                "detour_payload_sha256": detour["result_payload_sha256"],
                "smooth_payload_sha256": smooth["result_payload_sha256"],
                "attribution_payload_sha256": attribution["result_payload_sha256"],
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
    parser.add_argument("--attribution-result", type=Path, required=True)
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
        attribution_path=args.attribution_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "interpretation": result["interpretation"], "search_authorized": result["gate"]["search_authorized"], "result_payload_sha256": result["result_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
