"""Allocation-backed R00 reach-progress calibration case runner.

This module is additive to the frozen H03--H05 runner.  It records nominal
five-action reach progress on calibration groups and never invokes a repair,
flow intervention, or learned model.
"""

from __future__ import annotations

import json
import os
import platform
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from crfs_harness.artifacts import atomic_write_json, content_hash, load_json, scientific_config

from .reach_progress import (
    EXECUTED_REACH_ACTIONS,
    TARGET_OBJECT_NAME,
    ReachSnapshot,
    annotate_reach_rollout,
    capture_reach_snapshot,
)
from .runner import (
    OracleConfig,
    SafeLiberoCase,
    _array_hash,
    _determinism_check,
    _git_state,
    _infer,
    policy_observation,
)


FINAL_CALIBRATION_STATUSES = {
    "eligible_positive",
    "nonpositive_progress",
    "unsafe",
    "target_moved",
    "obstacle_moved",
    "invalid_phase",
}


@dataclass(frozen=True)
class ReachCalibrationConfig:
    oracle: OracleConfig
    target_name: str
    simulator_safety_margin_m: float
    maximum_target_displacement_m: float
    maximum_obstacle_displacement_m: float
    scene_motion_measurement: str
    minimum_positive_examples: int
    quantile: float
    quantile_method: str


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def reach_calibration_config_from_mapping(value: Mapping[str, Any], oracle: OracleConfig) -> ReachCalibrationConfig:
    progress = value.get("progress_calibration")
    if not isinstance(progress, Mapping):
        raise ValueError("R00 config requires a progress_calibration object")
    target_name = str(progress.get("target_object", ""))
    if target_name != TARGET_OBJECT_NAME:
        raise ValueError(f"R00 target must be {TARGET_OBJECT_NAME!r}")
    if oracle.executed_prefix != EXECUTED_REACH_ACTIONS:
        raise ValueError("R00 is registered for the first five executed actions")
    if oracle.action_horizon != 10:
        raise ValueError("R00 must retain the baseline ten-action model horizon")
    if oracle.measurement_repeats < 2:
        raise ValueError("R00 requires at least two simulator replay measurements")
    if progress.get("phase") != "pregrasp_reach":
        raise ValueError("R00 is restricted to the pregrasp_reach phase")
    if progress.get("reference_position") != "frozen_branch_target_position":
        raise ValueError("R00 must freeze the branch-start target position")
    if progress.get("metric") != "start_eef_target_distance_minus_end_eef_target_distance":
        raise ValueError("R00 has an unexpected progress metric")
    safety_margin = float(progress["simulator_safety_margin_m"])
    target_tolerance = float(progress["maximum_target_displacement_m"])
    obstacle_tolerance = float(progress["maximum_obstacle_displacement_m"])
    scene_motion_measurement = str(progress.get("scene_motion_measurement", ""))
    minimum_positive_examples = int(progress.get("minimum_positive_examples", 0))
    quantile = float(progress.get("quantile", float("nan")))
    quantile_method = str(progress.get("quantile_method", ""))
    if safety_margin <= 0 or target_tolerance < 0 or obstacle_tolerance < 0:
        raise ValueError("R00 margins and displacement tolerances are invalid")
    if target_tolerance > 0.001 or obstacle_tolerance > 0.001:
        raise ValueError("R00 displacement tolerances cannot exceed 1 mm")
    if scene_motion_measurement != "maximum_substep_displacement":
        raise ValueError("R00 requires maximum-substep target and obstacle motion")
    if abs(safety_margin - 0.005) > 1e-12:
        raise ValueError("R00 simulator safety margin must remain fixed at 5 mm")
    if minimum_positive_examples != 50:
        raise ValueError("R00 requires exactly 50 eligible positive examples")
    if abs(quantile - 0.25) > 1e-12 or quantile_method != "inverted_cdf":
        raise ValueError("R00 must use the preregistered inverted_cdf lower quartile")
    return ReachCalibrationConfig(
        oracle=oracle,
        target_name=target_name,
        simulator_safety_margin_m=safety_margin,
        maximum_target_displacement_m=target_tolerance,
        maximum_obstacle_displacement_m=obstacle_tolerance,
        scene_motion_measurement=scene_motion_measurement,
        minimum_positive_examples=minimum_positive_examples,
        quantile=quantile,
        quantile_method=quantile_method,
    )


def validate_reach_calibration_result(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "gate",
        "case_id",
        "run_id",
        "status",
        "config_hash",
        "provenance",
        "nominal",
        "trial",
    }
    missing = required - set(value)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
    if value.get("schema_version") != "1.0" or value.get("gate") != "R00":
        errors.append("result must be an R00 schema_version 1.0 artifact")
    if value.get("status") not in FINAL_CALIBRATION_STATUSES:
        errors.append(f"invalid R00 final status: {value.get('status')!r}")
    for key in ("case_id", "run_id", "config_hash"):
        if not isinstance(value.get(key), str) or not value.get(key):
            errors.append(f"{key} must be a non-empty string")
    trial = value.get("trial")
    if not isinstance(trial, Mapping):
        errors.append("trial must be an object")
    else:
        reach = trial.get("reach")
        if not isinstance(reach, Mapping) or not isinstance(reach.get("reach_progress_m"), (int, float)):
            errors.append("trial.reach.reach_progress_m must be numeric")
        elif not all(
            isinstance(reach.get(key), (int, float))
            for key in (
                "maximum_target_displacement_m",
                "maximum_active_obstacle_displacement_m",
            )
        ):
            errors.append("trial.reach maximum scene displacements must be numeric")
        if not isinstance(trial.get("eligible_for_p_min"), bool):
            errors.append("trial.eligible_for_p_min must be boolean")
    nominal = value.get("nominal")
    if not isinstance(nominal, Mapping) or not isinstance(nominal.get("actions"), list):
        errors.append("nominal.actions must be an array")
    else:
        actions = nominal["actions"]
        try:
            action_array = np.asarray(actions, dtype=np.float64)
        except (TypeError, ValueError):
            action_array = np.asarray([])
        if action_array.shape != (EXECUTED_REACH_ACTIONS, 7):
            errors.append("nominal.actions must be a finite 5x7 array")
        elif not np.isfinite(action_array).all():
            errors.append("nominal.actions must be finite")
        elif nominal.get("actions_sha256") != _array_hash(action_array):
            errors.append("nominal.actions_sha256 does not match nominal.actions")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
    else:
        required_provenance = {
            "git_commit",
            "git_dirty",
            "baseline_commit",
            "task_suite",
            "safety_level",
            "task_index",
            "episode_index",
            "environment_seed",
            "policy_seed",
            "random_control_seed",
            "group_id",
            "case_record",
            "case_record_sha256",
            "input_manifest_sha256",
            "checkpoint_id",
            "checkpoint_sha256",
            "noise_sha256",
            "sampler_steps",
            "model_action_horizon",
            "executed_action_horizon",
            "action_frame",
            "scene_motion_measurement",
            "slurm_job_id",
            "slurm_array_task_id",
            "partition",
            "device",
            "policy_determinism",
            "simulator_replay_exact",
        }
        absent = required_provenance - set(provenance)
        if absent:
            errors.append(f"provenance missing fields: {sorted(absent)}")
        for key in (
            "case_record_sha256",
            "input_manifest_sha256",
            "checkpoint_sha256",
            "noise_sha256",
        ):
            if key in provenance and not _is_sha256(provenance[key]):
                errors.append(f"provenance.{key} must be a lowercase SHA-256 digest")
        case_record = provenance.get("case_record")
        if not isinstance(case_record, Mapping):
            errors.append("provenance.case_record must be an object")
        else:
            if provenance.get("case_record_sha256") != content_hash(dict(case_record)):
                errors.append("provenance.case_record_sha256 does not match case_record")
            if case_record.get("case_id") != value.get("case_id"):
                errors.append("provenance.case_record does not match result case_id")
        for key in ("slurm_job_id", "slurm_array_task_id", "partition", "device"):
            if not isinstance(provenance.get(key), str) or not provenance.get(key):
                errors.append(f"provenance.{key} must be a non-empty allocation value")
        determinism = provenance.get("policy_determinism")
        if not isinstance(determinism, Mapping) or determinism.get("passed") is not True:
            errors.append("provenance.policy_determinism must pass")
        if provenance.get("simulator_replay_exact") is not True:
            errors.append("provenance.simulator_replay_exact must be true")
        if provenance.get("scene_motion_measurement") != (
            "maximum direct MuJoCo body displacement over 125 substeps"
        ):
            errors.append("provenance.scene_motion_measurement must cover all 125 substeps")
    return errors


def valid_reach_calibration_completion(
    path: str | Path,
    *,
    case_id: str | None = None,
    run_id: str | None = None,
    config_hash: str | None = None,
    input_manifest_sha256: str | None = None,
) -> bool:
    try:
        value = load_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(value, Mapping) or validate_reach_calibration_result(value):
        return False
    if case_id is not None and value.get("case_id") != case_id:
        return False
    if run_id is not None and value.get("run_id") != run_id:
        return False
    if config_hash is not None and value.get("config_hash") != config_hash:
        return False
    if input_manifest_sha256 is not None and value.get("provenance", {}).get(
        "input_manifest_sha256"
    ) != input_manifest_sha256:
        return False
    return True


def _target_contact_at_branch(environment: SafeLiberoCase, target_name: str) -> bool:
    domain = environment.env.env
    target_geoms = tuple(str(name) for name in domain.objects_dict[target_name].contact_geoms)
    sim = environment.env.sim
    eef_ids = {int(sim.model.geom_name2id(name)) for name in environment.eef_geoms}
    target_ids = {int(sim.model.geom_name2id(name)) for name in target_geoms}
    for contact in sim.data.contact[: sim.data.ncon]:
        first, second = int(contact.geom1), int(contact.geom2)
        if (first in eef_ids and second in target_ids) or (second in eef_ids and first in target_ids):
            return True
    return False


def _exact_rollout_replay(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    scalar_paths = (
        ("clearance_m",),
        ("reach", "reach_progress_m"),
        ("reach", "target_displacement_m"),
        ("reach", "active_obstacle_displacement_m"),
        ("reach", "maximum_target_displacement_m"),
        ("reach", "maximum_active_obstacle_displacement_m"),
    )

    def get(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = value
        for key in path:
            current = current[key]
        return current

    return bool(
        all(float(get(first, path)) == float(get(second, path)) for path in scalar_paths)
        and np.array_equal(np.asarray(first["end_eef_m"]), np.asarray(second["end_eef_m"]))
    )


def _annotated_rollout(
    environment: SafeLiberoCase,
    actions: np.ndarray,
    initial: ReachSnapshot,
    target_name: str,
) -> dict[str, Any]:
    if environment.obstacle_name is None:
        raise RuntimeError("active obstacle was not resolved at the branch state")
    rollout, reach = annotate_reach_rollout(
        environment,
        actions,
        target_name=target_name,
        obstacle_name=environment.obstacle_name,
        initial_snapshot=initial,
    )
    return {**rollout, "reach": reach}


def run_reach_calibration_case(
    case: Mapping[str, Any],
    config: ReachCalibrationConfig,
    *,
    repo_root: str | Path,
    input_manifest_sha256: str,
    client=None,
    environment: SafeLiberoCase | None = None,
) -> tuple[Path, str]:
    """Run one immutable nominal calibration case and atomically finalize it."""
    from openpi_client import websocket_client_policy

    oracle = config.oracle
    root = Path(repo_root).resolve()
    normalized_config = {
        **scientific_config(oracle.__dict__),
        "target_name": config.target_name,
        "simulator_safety_margin_m": config.simulator_safety_margin_m,
        "maximum_target_displacement_m": config.maximum_target_displacement_m,
        "maximum_obstacle_displacement_m": config.maximum_obstacle_displacement_m,
        "scene_motion_measurement": config.scene_motion_measurement,
        "minimum_positive_examples": config.minimum_positive_examples,
        "quantile": config.quantile,
        "quantile_method": config.quantile_method,
    }
    config_hash = content_hash(normalized_config)
    output = Path(oracle.output_root) / oracle.run_id / str(case["case_id"]) / "reach-calibration.json"
    if valid_reach_calibration_completion(
        output,
        case_id=str(case["case_id"]),
        run_id=oracle.run_id,
        config_hash=config_hash,
        input_manifest_sha256=input_manifest_sha256,
    ):
        return output, "skipped_valid_completion"

    git_commit, git_dirty = _git_state(root)
    noise = np.random.default_rng(int(case["policy_seed"])).normal(
        size=(oracle.action_horizon, oracle.action_dim)
    ).astype(np.float32)
    if client is None:
        client = websocket_client_policy.WebsocketClientPolicy(oracle.host, oracle.port)
    owns_environment = environment is None
    if environment is None:
        environment = SafeLiberoCase(dict(case), oracle)
    else:
        environment.configure_case(dict(case))

    try:
        initial_observation = environment.reset_and_settle()
        obstacle_name = environment.obstacle_name
        if obstacle_name is None:
            raise RuntimeError("failed to resolve the active obstacle")
        initial_snapshot = capture_reach_snapshot(environment, config.target_name, obstacle_name)
        initial_target_contact = _target_contact_at_branch(environment, config.target_name)
        initial_task_success = bool(environment.env.check_success())
        phase_valid = bool(
            str(case["task_suite"]) == "safelibero_spatial"
            and str(case["safety_level"]) == "II"
            and int(case["task_index"]) == 0
            and not initial_target_contact
            and not initial_task_success
        )

        policy_input = policy_observation(initial_observation, environment.prompt, oracle.resize_size)
        first_reply = _infer(client, policy_input, noise, oracle, intervention_mode="none")
        second_reply = _infer(client, policy_input, noise, oracle, intervention_mode="none")
        policy_determinism = _determinism_check(first_reply, second_reply)
        if not policy_determinism["passed"]:
            raise RuntimeError(f"fixed observation/noise policy replay is not exact: {policy_determinism}")
        full_actions = np.asarray(first_reply["actions"], dtype=np.float64)
        actions = full_actions[: oracle.executed_prefix, :7]
        first_rollout = _annotated_rollout(
            environment, actions, initial_snapshot, config.target_name
        )
        second_rollout = _annotated_rollout(
            environment, actions, initial_snapshot, config.target_name
        )
        simulator_replay_exact = _exact_rollout_replay(first_rollout, second_rollout)
        if not simulator_replay_exact:
            raise RuntimeError("paired nominal reach rollout is not exact")
        if int(first_rollout["measurement_samples"]) != 126:
            raise RuntimeError("five-action reach rollout must contain 126 measurement samples")

        reach = first_rollout["reach"]
        safe = bool(
            float(first_rollout["clearance_m"]) >= config.simulator_safety_margin_m
            and not bool(first_rollout["contact"])
        )
        target_stationary = bool(
            float(reach["maximum_target_displacement_m"])
            <= config.maximum_target_displacement_m
        )
        obstacle_stationary = bool(
            float(reach["maximum_active_obstacle_displacement_m"])
            <= config.maximum_obstacle_displacement_m
        )
        positive = bool(float(reach["reach_progress_m"]) > 0.0)
        eligible = bool(phase_valid and safe and target_stationary and obstacle_stationary and positive)
        if not phase_valid:
            status = "invalid_phase"
        elif not safe:
            status = "unsafe"
        elif not target_stationary:
            status = "target_moved"
        elif not obstacle_stationary:
            status = "obstacle_moved"
        elif not positive:
            status = "nonpositive_progress"
        else:
            status = "eligible_positive"

        provenance = {
            "evidence_tier": "real_safelibero_reach_progress_calibration",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "baseline_commit": "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
            "python_version": platform.python_version(),
            "host": socket.gethostname(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_suite": case["task_suite"],
            "safety_level": case["safety_level"],
            "task_index": case["task_index"],
            "episode_index": case["episode_index"],
            "environment_seed": case["environment_seed"],
            "policy_seed": case["policy_seed"],
            "random_control_seed": case["random_control_seed"],
            "group_id": case["group_id"],
            "case_record": dict(case),
            "case_record_sha256": content_hash(dict(case)),
            "input_manifest_sha256": input_manifest_sha256,
            "checkpoint_id": oracle.checkpoint_id,
            "checkpoint_sha256": oracle.checkpoint_sha256,
            "noise_sha256": _array_hash(noise),
            "sampler_steps": oracle.sampler_steps,
            "model_action_horizon": oracle.action_horizon,
            "executed_action_horizon": oracle.executed_prefix,
            "action_frame": "world-frame OSC translation delta",
            "policy_action_space": "unnormalized LIBERO controller action",
            "scene_motion_measurement": (
                "maximum direct MuJoCo body displacement over 125 substeps"
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "policy_determinism": policy_determinism,
            "simulator_replay_exact": simulator_replay_exact,
        }
        result = {
            "schema_version": "1.0",
            "gate": "R00",
            "case_id": case["case_id"],
            "run_id": oracle.run_id,
            "status": status,
            "config_hash": config_hash,
            "provenance": provenance,
            "nominal": {
                "actions": actions.tolist(),
                "actions_sha256": _array_hash(actions),
                "full_model_actions_sha256": _array_hash(full_actions),
            },
            "trial": {
                **first_rollout,
                "phase": "pregrasp_reach",
                "phase_valid": phase_valid,
                "initial_target_contact": initial_target_contact,
                "initial_task_success": initial_task_success,
                "safe_at_registered_margin": safe,
                "target_stationary": target_stationary,
                "active_obstacle_stationary": obstacle_stationary,
                "positive_progress": positive,
                "eligible_for_p_min": eligible,
                "simulator_safety_margin_m": config.simulator_safety_margin_m,
                "maximum_target_displacement_m": config.maximum_target_displacement_m,
                "maximum_obstacle_displacement_m": config.maximum_obstacle_displacement_m,
            },
        }
        errors = validate_reach_calibration_result(result)
        if errors:
            raise RuntimeError("refusing invalid R00 final artifact: " + "; ".join(errors))
        atomic_write_json(output, result)
        return output, status
    finally:
        if owns_environment:
            environment.close()
