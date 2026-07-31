#!/usr/bin/env python3
"""Allocation-only SafeLIBERO static-Poisson shadow identification.

This stage is deliberately non-interventional.  It replays the first frozen
canary's 237 historical AEGIS Cartesian actions through the unchanged released
OSC environment.  A post-integration callback measures authoritative MuJoCo
contact and evaluates the settled static link-5/6 Poisson field, but it never
changes an action, control, model, or live simulator state.

An already-passed exact shadow-parity artifact from the same clean commit is a
required input.  The final result is useful only for identifying whether the
registered field signal precedes the known link contact; it is not evidence of
active collision avoidance or task-preserving correction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "vlsa_poisson_shadow_identification.v2"
DEFAULT_CASE_ID = "vlsa-t1-goal-ii-t0-e05"
EXPECTED_PARITY_SCHEMA = "vlsa_poisson_shadow_parity.v1"
CONTACT_DEFINITION = "mujoco_contact_dist_le_0"


class ShadowIdentificationRunnerError(RuntimeError):
    """A required allocation, pairing, or exact-replay invariant failed."""


def _package_versions(names: Sequence[str]) -> Dict[str, Any]:
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ShadowIdentificationRunnerError("%s is missing or symlinked" % label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShadowIdentificationRunnerError("%s is invalid JSON" % label) from error
    if not isinstance(payload, dict):
        raise ShadowIdentificationRunnerError("%s must contain one JSON object" % label)
    return payload


def _model_body_id(model: Any, name: str) -> int:
    method = getattr(model, "body_name2id", None)
    if method is not None:
        try:
            value = int(method(name))
        except Exception as error:
            raise ShadowIdentificationRunnerError(
                "required MuJoCo body %s is absent" % name
            ) from error
    else:
        mujoco = importlib.import_module("mujoco")
        raw_model = getattr(model, "_model", model)
        value = int(
            mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, name)
        )
    if value < 0:
        raise ShadowIdentificationRunnerError(
            "required MuJoCo body %s is absent" % name
        )
    return value


def _official_integration_state(sim: Any, np: Any) -> Any:
    """Read MuJoCo's complete official integration state without forwarding."""

    mujoco = importlib.import_module("mujoco")
    model_candidate = getattr(sim, "model", None)
    data_candidate = getattr(sim, "data", None)
    model = getattr(model_candidate, "_model", model_candidate)
    data = getattr(data_candidate, "_data", data_candidate)
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise ShadowIdentificationRunnerError(
            "simulator does not expose official MuJoCo model/data"
        )
    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(
        int(mujoco.mj_stateSize(model, specification)), dtype=np.float64
    )
    mujoco.mj_getState(model, data, state, specification)
    if state.ndim != 1 or state.size <= 0 or not np.all(np.isfinite(state)):
        raise ShadowIdentificationRunnerError(
            "official MuJoCo integration state is empty or non-finite"
        )
    return state


def _load_bound_runtime_protocol(
    *,
    root: Path,
    case: Mapping[str, Any],
    selection_config_path: Path,
) -> Tuple[Mapping[str, Any], Any, Mapping[str, Any], Path]:
    from main.poisson_fullbody.feasibility_protocol import load_feasibility_protocol
    from scripts.run_poisson_shadow_parity import _file_sha256

    selection = _load_json_object(selection_config_path, "selection protocol")
    if _file_sha256(selection_config_path) != case.get("protocol_config_sha256"):
        raise ShadowIdentificationRunnerError(
            "case selection-protocol SHA-256 binding differs"
        )
    runtime_record = selection.get("runtime_protocol")
    if not isinstance(runtime_record, dict):
        raise ShadowIdentificationRunnerError(
            "selection protocol lacks runtime_protocol binding"
        )
    relative = runtime_record.get("relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ShadowIdentificationRunnerError("runtime protocol path is invalid")
    runtime_path = root / relative
    if _file_sha256(runtime_path) != runtime_record.get("raw_file_sha256"):
        raise ShadowIdentificationRunnerError("runtime protocol raw SHA-256 differs")
    protocol, hashes = load_feasibility_protocol(
        runtime_path,
        expected_protocol_sha256=runtime_record.get(
            "semantic_protocol_sha256"
        ),
    )
    if (
        hashes.parameter_block_sha256
        != runtime_record.get("parameter_block_sha256")
    ):
        raise ShadowIdentificationRunnerError(
            "runtime protocol parameter-block binding differs"
        )
    return protocol, hashes, selection, runtime_path


def _require_upstream_parity(
    parity_path: Path,
    *,
    case_id: str,
    source_commit: str,
    historical_payload_sha256: str,
    manifest_sha256: str,
    manifest_row_sha256: str,
    expected_callback_count: int,
) -> Mapping[str, Any]:
    from main.poisson_fullbody.contracts import load_hashed_json

    parity = load_hashed_json(parity_path)
    if (
        parity.get("schema_version") != EXPECTED_PARITY_SCHEMA
        or parity.get("status") != "passed"
        or parity.get("case_id") != case_id
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity artifact has wrong schema, status, or case"
        )
    provenance = parity.get("provenance")
    callback = parity.get("callback_replay")
    acceptance = parity.get("acceptance")
    if not isinstance(provenance, dict) or not isinstance(callback, dict):
        raise ShadowIdentificationRunnerError("upstream exact-parity provenance is incomplete")
    if not isinstance(acceptance, dict) or not all(
        acceptance.get(field) is True
        for field in (
            "all_historical_post_step_states_exact",
            "ordinary_and_callback_states_exact",
            "ordinary_and_callback_observations_exact",
            "reward_done_goal_exact",
            "full_callback_exposure",
            "ordinary_step_path_unmodified",
        )
    ):
        raise ShadowIdentificationRunnerError("upstream exact-parity acceptance is incomplete")
    source = provenance.get("source")
    if not isinstance(source, dict) or source.get("commit") != source_commit:
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity commit differs from this clean source"
        )
    if (
        provenance.get("historical_result_payload_sha256")
        != historical_payload_sha256
        or provenance.get("manifest_sha256") != manifest_sha256
        or provenance.get("manifest_row_sha256") != manifest_row_sha256
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity evidence binding differs"
        )
    if (
        callback.get("callback_count") != expected_callback_count
        or callback.get("expected_callback_count") != expected_callback_count
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity callback exposure differs"
        )
    return parity


def _first_link56_contact(
    measurement: Any, link56_geom_ids: Sequence[int]
) -> Optional[Dict[str, Any]]:
    link_geoms = set(int(value) for value in link56_geom_ids)
    records = list(measurement.settled_state.physical_contact_point_records)
    records.extend(
        record
        for record in measurement.live_solver_phase_contact_point_records
        if record.is_physical_nonpositive_distance_contact
    )
    records.extend(measurement.post_state_physical_contact_point_records)
    records = [record for record in records if record.robot_geom_id in link_geoms]
    if not records:
        return None

    def order(record: Any) -> Tuple[int, int, int]:
        observation = (
            -1 if record.observation_index is None else int(record.observation_index)
        )
        phase = {
            "settled_post_integration_recomputed": 0,
            "live_solver_phase_preintegration_geometry": 1,
            "post_integration_recomputed": 2,
        }.get(str(record.source_phase), 3)
        return observation, phase, int(record.mujoco_contact_index)

    return min(records, key=order).to_dict()


def _prepare_shadow_runtime(
    *,
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    replay: Any,
    protocol: Mapping[str, Any],
    protocol_hashes: Any,
) -> Tuple[Any, Any, Mapping[str, Any], Sequence[Sequence[str]], Sequence[bool], Any, Any, Any]:
    from main.poisson_fullbody.field_bundle import build_static_field_bundle
    from main.poisson_fullbody.measurement import (
        FullRobotObstacleMonitor,
        clone_forwarded_state,
        resolve_collision_geom_sets,
    )
    from main.poisson_fullbody.robot_samples import validate_rigid_roundtrip
    from main.poisson_fullbody.shadow_identification import (
        StaticDriftThresholds,
        StaticPoissonShadowObserver,
    )
    from main.poisson_fullbody.surface_sampling import (
        build_robot_collision_samples,
        validate_robot_sample_evidence,
    )
    from scripts.run_poisson_shadow_parity import _prepare_environment

    env, task, observation, goal_atoms, previous_goal = _prepare_environment(
        evaluator, runtime, case, replay
    )
    construction_state_before = _official_integration_state(
        env.sim, runtime["np"]
    )
    obstacle_name, _ = evaluator._active_obstacle(env, observation)
    if obstacle_name != case.get("active_obstacle_name"):
        env.close()
        raise ShadowIdentificationRunnerError("active selected obstacle differs from manifest")
    authority = evaluator._contact_model_authority(env, obstacle_name)
    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    if len(env.robots) != 1:
        env.close()
        raise ShadowIdentificationRunnerError(
            "shadow identification requires exactly one robot"
        )
    robot_root_name = getattr(env.robots[0].robot_model, "root_body", None)
    if not isinstance(robot_root_name, str) or not robot_root_name:
        env.close()
        raise ShadowIdentificationRunnerError(
            "robosuite robot model does not expose its authoritative root body"
        )
    robot_roots = (_model_body_id(env.sim.model, robot_root_name),)
    if robot_roots[0] not in set(int(value) for value in authority["robot_body_ids"]):
        env.close()
        raise ShadowIdentificationRunnerError(
            "robosuite robot root is absent from frozen contact-model authority"
        )
    link_ids = tuple(
        _model_body_id(env.sim.model, name)
        for name in protocol["claim_scope"]["protected_robot_bodies"]
    )
    resolved = resolve_collision_geom_sets(
        raw_model,
        robot_root_body_ids=robot_roots,
        obstacle_root_body_ids=(int(authority["active_obstacle_root_body_id"]),),
        link56_body_ids=link_ids,
    )
    bundle = build_static_field_bundle(
        raw_model,
        raw_data,
        resolved=resolved,
        protocol=protocol,
        protocol_hashes=protocol_hashes,
    )
    forwarded = clone_forwarded_state(raw_model, raw_data)
    full_samples = build_robot_collision_samples(
        env.sim.model,
        forwarded,
        geom_ids=resolved.robot_geom_ids,
        epsilon_m=float(protocol["coverage"]["epsilon_m"]),
    )
    sampled_geom_ids = tuple(
        int(record["geom_id"]) for record in full_samples.geom_records
    )
    if sampled_geom_ids != tuple(int(value) for value in resolved.robot_geom_ids):
        env.close()
        raise ShadowIdentificationRunnerError(
            "full-robot measurement samples differ from authoritative resolved geoms"
        )
    roundtrip = validate_rigid_roundtrip(full_samples.samples, forwarded)
    if not roundtrip["passed"]:
        env.close()
        raise ShadowIdentificationRunnerError("full-robot sample transform audit failed")
    full_sampling_evidence = {
        "sample_count": len(full_samples.samples),
        "sample_ledger_sha256": full_samples.sample_ledger_sha256,
        "geom_records": list(full_samples.geom_records),
        "epsilon_m": full_samples.epsilon_m,
        "maximum_surface_cover_radius_m": (
            full_samples.maximum_surface_cover_radius_m
        ),
        "coverage_semantics": full_samples.coverage_semantics,
        "rigid_roundtrip": roundtrip,
    }
    try:
        type_counts = validate_robot_sample_evidence(
            full_sampling_evidence,
            resolved_geom_ids=resolved.robot_geom_ids,
            resolved_geom_names=resolved.robot_geom_names,
            resolved_body_ids=resolved.robot_body_ids,
            roundtrip_field="rigid_roundtrip",
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        env.close()
        raise ShadowIdentificationRunnerError(
            "full-robot surface-sampling evidence is invalid: %s" % error
        ) from error
    if type_counts != {"mesh": 11, "box": 4, "cylinder": 1}:
        env.close()
        raise ShadowIdentificationRunnerError(
            "first-canary robot collision geometry type counts changed"
        )
    cylinder_record = next(
        record
        for record in full_samples.geom_records
        if record["geom_type_name"] == "cylinder"
    )
    if (
        cylinder_record["geom_id"] != 84
        or cylinder_record["geom_name"] != "mount0_pedestal_col"
        or cylinder_record["geom_size"] != [0.18, 0.31, 0.0]
    ):
        env.close()
        raise ShadowIdentificationRunnerError(
            "first-canary pedestal-cylinder identity or dimensions changed"
        )
    admissibility = protocol["admissibility"]
    safety = protocol["safety"]
    monitor = FullRobotObstacleMonitor(
        env.sim,
        resolved,
        full_samples.samples,
        certified_coverage_radius_m=(
            full_samples.maximum_surface_cover_radius_m
        ),
        max_selected_geom_surface_drift_m=float(
            admissibility["max_selected_geom_surface_drift_m"]
        ),
        max_selected_geom_translation_drift_m=float(
            admissibility["max_selected_geom_translation_drift_m"]
        ),
        max_selected_geom_rotation_drift_rad=float(
            admissibility["max_selected_geom_rotation_drift_rad"]
        ),
        max_settled_obstacle_linear_speed_m_per_s=float(
            admissibility["max_selected_body_linear_speed_m_s"]
        ),
        max_settled_obstacle_angular_speed_rad_per_s=float(
            admissibility["max_selected_body_angular_speed_rad_s"]
        ),
        require_settled_static_motion=True,
        terminate_on_static_drift=False,
        near_contact_tolerance_m=float(safety["contact_margin_m"]),
        inner_updates_per_high_level_action=5,
        physics_substeps_per_inner_update=5,
    )
    monitor.require_settled_obstacle_motion_admissible()
    arm_dof_indices = tuple(int(value) for value in env.robots[0]._ref_joint_vel_indexes)
    observer = StaticPoissonShadowObserver(
        field=bundle.field,
        samples=bundle.protected_samples.samples,
        settled_obstacle_boxes=bundle.obstacle_boxes,
        arm_dof_indices=arm_dof_indices,
        alpha_gain_per_s=float(protocol["cbf"]["alpha_gain_per_s"]),
        physics_timestep_s=float(protocol["cadence"]["physics_timestep_s"]),
        drift_thresholds=StaticDriftThresholds(
            translation_m=float(
                admissibility["max_selected_geom_translation_drift_m"]
            ),
            rotation_rad=float(
                admissibility["max_selected_geom_rotation_drift_rad"]
            ),
            surface_m=float(admissibility["max_selected_geom_surface_drift_m"]),
        ),
        physics_substeps_per_high_level_action=25,
    )
    construction_state_after = _official_integration_state(
        env.sim, runtime["np"]
    )
    construction_state_before_hash = evaluator.array_sha256(
        construction_state_before
    )
    construction_state_after_hash = evaluator.array_sha256(
        construction_state_after
    )
    if not runtime["np"].array_equal(
        construction_state_after, construction_state_before
    ) or construction_state_after_hash != construction_state_before_hash:
        env.close()
        raise ShadowIdentificationRunnerError(
            "field and monitor construction changed complete MuJoCo integration state"
        )
    construction = {
        "active_obstacle_name": obstacle_name,
        "contact_model_authority_sha256": authority["authority_sha256"],
        "robot_root_body_name": robot_root_name,
        "robot_root_body_ids": list(robot_roots),
        "arm_dof_indices": list(arm_dof_indices),
        "resolved_geometry": resolved.to_dict(),
        "field_bundle": {
            "protocol_id": bundle.protocol_id,
            "protected_body_ids": list(bundle.protected_body_ids),
            "protected_body_names": list(bundle.protected_body_names),
            "diagnostics": asdict(bundle.diagnostics),
            "hashes": asdict(bundle.hashes),
            "surface_components": [
                asdict(value) for value in bundle.protected_samples.components
            ],
            "protected_sample_count": len(bundle.protected_samples.samples),
        },
        "full_robot_measurement_sampling": full_sampling_evidence,
        "settled_measurement": monitor.settled_state.to_dict(),
        "measurement_drift_mode": (
            "diagnostic_continue_contact_measurement_after_static_field_invalidation"
        ),
        "complete_integration_state_read_only_audit": {
            "mujoco_state_specification": "mjSTATE_INTEGRATION",
            "state_vector_length": int(construction_state_before.size),
            "before_sha256": construction_state_before_hash,
            "after_sha256": construction_state_after_hash,
            "exact_array_equal": True,
            "semantics": (
                "the complete official MuJoCo integration state was bitwise "
                "unchanged across field, sampling, and monitor construction"
            ),
        },
    }
    return (
        env,
        task,
        observation,
        goal_atoms,
        previous_goal,
        monitor,
        observer,
        construction,
    )


def _run_shadow(
    *,
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    replay: Any,
    protocol: Mapping[str, Any],
    protocol_hashes: Any,
    upstream_parity: Mapping[str, Any],
) -> Dict[str, Any]:
    from main.poisson_fullbody.measurement import StaticObstacleDriftInadmissible
    from scripts.run_poisson_shadow_parity import _canonical, _check_step, _sha256

    env = None
    try:
        (
            env,
            _,
            observation,
            goal_atoms,
            previous_goal,
            monitor,
            observer,
            construction,
        ) = _prepare_shadow_runtime(
            evaluator=evaluator,
            runtime=runtime,
            case=case,
            replay=replay,
            protocol=protocol,
            protocol_hashes=protocol_hashes,
        )
        state_hashes = []
        observation_hashes = []
        callback_state_hashes = []
        monitor_drift_exceptions = []
        for expected_step in replay.steps:
            current_callback_hashes = []

            def callback(sim: Any, substep_index: int) -> None:
                before_state = _official_integration_state(sim, runtime["np"])
                before_state_hash = evaluator.array_sha256(before_state)
                inner = int(substep_index) // 5
                physics = int(substep_index) % 5
                try:
                    monitor.observe_post_integration(
                        sim,
                        high_level_index=int(expected_step.step),
                        inner_control_index=inner,
                        physics_substep_index=physics,
                    )
                except StaticObstacleDriftInadmissible as error:
                    # The monitor has already recorded this substep.  Static
                    # field invalidation is handled independently by observer;
                    # exact OSC replay and contact measurement must continue.
                    monitor_drift_exceptions.append(
                        {
                            "observation_index": (
                                int(expected_step.step) * 25 + int(substep_index)
                            ),
                            "message": str(error),
                        }
                    )
                observer.observe(
                    sim,
                    high_level_index=int(expected_step.step),
                    physics_substep_index=int(substep_index),
                )
                after_state = _official_integration_state(sim, runtime["np"])
                after_state_hash = evaluator.array_sha256(after_state)
                if not runtime["np"].array_equal(
                    after_state, before_state
                ) or after_state_hash != before_state_hash:
                    raise ShadowIdentificationRunnerError(
                        "read-only shadow callback mutated complete MuJoCo integration state"
                    )
                current_callback_hashes.append(after_state_hash)

            observation, reward, done, _ = env.step_with_substep_callback(
                expected_step.action,
                callback,
                expected_substeps=25,
            )
            if len(current_callback_hashes) != 25:
                raise ShadowIdentificationRunnerError(
                    "shadow callback cadence differs at step %d" % expected_step.step
                )
            state_hash, observation_hash, previous_goal = _check_step(
                evaluator=evaluator,
                env=env,
                observation=observation,
                reward=reward,
                done=done,
                expected_step=expected_step,
                goal_atoms=goal_atoms,
                previous_goal_values=previous_goal,
                np=runtime["np"],
            )
            state_hashes.append(state_hash)
            observation_hashes.append(observation_hash)
            callback_state_hashes.extend(current_callback_hashes)
            proxy = evaluator._eef_proxy(runtime, observation)
            evaluator._update_eef_marker(env, proxy)
            if done and expected_step.step != len(replay.steps) - 1:
                raise ShadowIdentificationRunnerError(
                    "shadow replay terminated before the registered horizon"
                )
        measurement = monitor.result()
        expected_callback_count = 25 * len(replay.steps)
        if (
            int(measurement.observed_physics_substeps) != expected_callback_count
            or int(measurement.first_index) != 0
            or int(measurement.last_index) != expected_callback_count - 1
        ):
            raise ShadowIdentificationRunnerError(
                "MuJoCo contact measurement did not cover every physics callback"
            )
        if measurement.physical_contact_distance_semantics != CONTACT_DEFINITION:
            raise ShadowIdentificationRunnerError(
                "MuJoCo physical-contact authority semantics changed"
            )
        resolved = construction["resolved_geometry"]
        first_link_contact = _first_link56_contact(
            measurement, resolved["link56_geom_ids"]
        )
        identification = observer.result(
            expected_callback_count=expected_callback_count,
            first_link56_contact=first_link_contact,
        )
        state_sequence_hash = _sha256(_canonical(state_hashes))
        observation_sequence_hash = _sha256(_canonical(observation_hashes))
        upstream_callback = upstream_parity["callback_replay"]
        if state_sequence_hash != upstream_callback["state_sequence_sha256"]:
            raise ShadowIdentificationRunnerError(
                "shadow state sequence differs from upstream exact parity"
            )
        if observation_sequence_hash != upstream_callback["observation_sequence_sha256"]:
            raise ShadowIdentificationRunnerError(
                "shadow observation sequence differs from upstream exact parity"
            )
        return {
            "executed_action_count": len(state_hashes),
            "callback_count": len(callback_state_hashes),
            "expected_callback_count": expected_callback_count,
            "state_sequence_sha256": state_sequence_hash,
            "observation_sequence_sha256": observation_sequence_hash,
            "callback_state_sequence_sha256": _sha256(
                _canonical(callback_state_hashes)
            ),
            "terminal_simulator_state_sha256": state_hashes[-1],
            "construction": construction,
            "measurement": measurement.to_dict(),
            "monitor_static_drift_exception_count": len(monitor_drift_exceptions),
            "monitor_static_drift_exceptions": monitor_drift_exceptions,
            "poisson_identification": identification,
        }
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"),
    )
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=Path("configs/vlsa_poisson_link56_feasibility.v1.json"),
    )
    parser.add_argument("--historical-result-root", type=Path, required=True)
    parser.add_argument("--parity-result", type=Path, required=True)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    manifest = arguments.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    selection_config = arguments.selection_config
    if not selection_config.is_absolute():
        selection_config = root / selection_config
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "safelibero"))
    sys.path.insert(0, str(root / "main"))

    from main.poisson_fullbody.contracts import publish_hashed_json
    from main.poisson_fullbody.shadow_replay import load_historical_action_replay
    from scripts.run_poisson_shadow_parity import (
        _file_sha256,
        _git_record,
        _gpu_inventory,
        _load_case,
    )

    started = time.time()
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "scientific_result": False,
        "evidence_tier": "allocation_backed_exact_replay_shadow_identification",
        "claim_limit": (
            "read-only temporal identification only; no action was corrected and "
            "no active safety or utility efficacy was tested"
        ),
        "case_id": arguments.case_id,
        "timing": {"started_unix": started},
    }
    try:
        if arguments.case_id != DEFAULT_CASE_ID:
            raise ShadowIdentificationRunnerError(
                "shadow identification is registered only for the first canary %s"
                % DEFAULT_CASE_ID
            )
        source = _git_record(root)
        if source["status_short"]:
            raise ShadowIdentificationRunnerError(
                "shadow identification requires a clean source tree"
            )
        if not os.environ.get("SLURM_JOB_ID"):
            raise ShadowIdentificationRunnerError(
                "shadow identification must run inside a Slurm allocation"
            )
        gpu = _gpu_inventory()
        line_number, case, row_hash = _load_case(manifest, arguments.case_id)
        historical_record = case.get("historical_aegis_result")
        if not isinstance(historical_record, dict):
            raise ShadowIdentificationRunnerError(
                "manifest lacks historical AEGIS binding"
            )
        relative = historical_record.get("source_relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ShadowIdentificationRunnerError(
                "historical result relative path is invalid"
            )
        historical_path = arguments.historical_result_root.resolve() / relative
        if _file_sha256(historical_path) != historical_record.get("raw_file_sha256"):
            raise ShadowIdentificationRunnerError("historical result raw SHA-256 differs")
        replay = load_historical_action_replay(
            historical_path, expected_case_id=arguments.case_id
        )
        if replay.result_payload_sha256 != historical_record.get(
            "result_payload_sha256"
        ):
            raise ShadowIdentificationRunnerError(
                "historical result payload binding differs"
            )
        if len(replay.steps) != 237:
            raise ShadowIdentificationRunnerError(
                "registered first canary must expose exactly 237 historical actions"
            )
        manifest_hash = _file_sha256(manifest)
        upstream_parity = _require_upstream_parity(
            arguments.parity_result.resolve(),
            case_id=arguments.case_id,
            source_commit=source["commit"],
            historical_payload_sha256=replay.result_payload_sha256,
            manifest_sha256=manifest_hash,
            manifest_row_sha256=row_hash,
            expected_callback_count=25 * len(replay.steps),
        )
        protocol, protocol_hashes, selection, runtime_path = (
            _load_bound_runtime_protocol(
                root=root,
                case=case,
                selection_config_path=selection_config,
            )
        )
        evaluator = importlib.import_module("evaluate_safelibero_aegis")
        runtime = evaluator._runtime_imports(include_aegis=False)
        shadow = _run_shadow(
            evaluator=evaluator,
            runtime=runtime,
            case=case,
            replay=replay,
            protocol=protocol,
            protocol_hashes=protocol_hashes,
            upstream_parity=upstream_parity,
        )
        expected_state_hash = upstream_parity["callback_replay"][
            "state_sequence_sha256"
        ]
        expected_observation_hash = upstream_parity["callback_replay"][
            "observation_sequence_sha256"
        ]
        identification = shadow["poisson_identification"]
        payload.update(
            {
                "status": "passed",
                "provenance": {
                    "source": source,
                    "manifest_path": str(manifest),
                    "manifest_sha256": manifest_hash,
                    "manifest_line_number": line_number,
                    "manifest_row_sha256": row_hash,
                    "selection_config_path": str(selection_config),
                    "selection_config_sha256": _file_sha256(selection_config),
                    "selection_protocol_id": selection["protocol_id"],
                    "runtime_protocol_path": str(runtime_path),
                    "runtime_protocol_raw_sha256": _file_sha256(runtime_path),
                    "runtime_protocol_semantic_sha256": (
                        protocol_hashes.protocol_sha256
                    ),
                    "runtime_parameter_block_sha256": (
                        protocol_hashes.parameter_block_sha256
                    ),
                    "historical_result_path": str(historical_path),
                    "historical_result_file_sha256": replay.result_file_sha256,
                    "historical_result_payload_sha256": (
                        replay.result_payload_sha256
                    ),
                    "upstream_parity_path": str(arguments.parity_result.resolve()),
                    "upstream_parity_payload_sha256": upstream_parity[
                        "result_payload_sha256"
                    ],
                    "host": socket.gethostname(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
                    "python_executable": sys.executable,
                    "python_version": platform.python_version(),
                    "packages": _package_versions(
                        ("numpy", "mujoco", "robosuite", "scipy")
                    ),
                    "gpu": gpu,
                },
                "historical": replay.provenance(),
                "shadow_replay": shadow,
                "acceptance": {
                    "upstream_exact_parity_same_clean_commit": True,
                    "all_237_historical_actions_executed": (
                        shadow["executed_action_count"] == 237
                    ),
                    "all_5925_callbacks_observed": (
                        shadow["callback_count"] == 5925
                        and identification["observed_callback_count"] == 5925
                        and shadow["measurement"]["observed_physics_substeps"]
                        == 5925
                    ),
                    "historical_state_reward_done_goal_exact": (
                        shadow["state_sequence_sha256"] == expected_state_hash
                    ),
                    "upstream_observation_sequence_exact": (
                        shadow["observation_sequence_sha256"]
                        == expected_observation_hash
                    ),
                    "complete_mujoco_integration_state_unchanged_by_construction": (
                        shadow["construction"][
                            "complete_integration_state_read_only_audit"
                        ]["exact_array_equal"]
                    ),
                    "complete_mujoco_integration_state_unchanged_by_callback": True,
                    "static_queries_stop_at_first_registered_drift": True,
                    "contact_authority_is_mujoco_nonpositive_distance": True,
                    "no_action_or_control_mutation": True,
                    "no_active_safety_efficacy_claim": True,
                },
            }
        )
        if not all(payload["acceptance"].values()):
            raise ShadowIdentificationRunnerError(
                "one or more final shadow acceptance checks failed"
            )
    except Exception as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    payload["timing"].update(
        {
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started,
        }
    )
    publish_hashed_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
