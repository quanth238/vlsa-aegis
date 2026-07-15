"""Allocation-only R05A inverse-flow teacher apparatus canary.

This runner restores one immutable SafeLIBERO branch and exercises the frozen
pi0.5 sampler.  It deliberately never calls ``environment.rollout`` or steps a
policy-generated action.  Its output is an intermediate payload; the Slurm
worker adds independently measured allocation memory and atomically writes the
schema-valid ``results.json`` only after all validation inputs exist.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import socket
import time
from typing import Any, Mapping, Optional

import numpy as np

from crfs_harness.artifacts import (
    atomic_write_json,
    content_hash,
    file_sha256,
    load_json,
)

from .progress_calibration import _target_contact_at_branch
from .r02_runner import (
    BASELINE_COMMIT,
    NORMALIZATION_ASSET_SHA256,
    _array_from_record,
    _array_record,
    _json_compatible,
    _observation_fingerprint,
    _trace_record,
    _validate_array_record,
    _validate_trace_record,
    validate_r02_result,
)
from .r03a_runner import _trace_from_record, _trace_pairing_diagnostics
from .reach_progress import TARGET_OBJECT_NAME, capture_reach_snapshot
from .runner import OracleConfig, SafeLiberoCase, _git_state, policy_observation


SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "r05a_inverse_flow_allocation_canary"
PAYLOAD_TYPE = "r05a_inverse_flow_allocation_canary_payload"
GATE = "R05A"
CASE_ID = "crfs-1069f29a8d76463a"
GROUP_ID = "safelibero_spatial:II:0:46"
SOURCE_HOST = "worker-1"
EXPERIMENT_ROOT = Path("/mnt/data/quanth/experiments/crfs-oracle")
MANIFEST_SHA256 = "bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633"
SOURCE_R02_SHA256 = "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593"
SOURCE_R02_CONFIG_SHA256 = "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e"
R03_SUMMARY_SHA256 = "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e"
R03_ORDERED_RESULT_SET_DIGEST = "fa79a132f2fecbedab5917111ef71f022e52c933d8e535aaca118add5f5b7895"
CHECKPOINT_SHA256 = "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
DECISION_ARTIFACT = "docs/decisions/0028-pivot-to-inverse-flow-transport.md"
DECISION_SHA256 = "d2a00b1b049e92bb1ec8f11d60fa447e5e8bd5109cb8cf3f3fbb08f89498656f"
SCHEMA_PATH = "schemas/r05a-inverse-flow-canary.schema.json"
SCHEMA_SHA256 = "e1681f865f81f2986945d10fb14073fe4cd9e78cbc5026f6749ab321b9cfb7a7"
CONFIG_PATH = "configs/experiments/r05a_inverse_flow_canary.json"
CONFIG_FILE_SHA256 = "c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb"
SCIENTIFIC_CONFIG_HASH = "9e2ff74cda8ac3b5d4098942a82cc3352cebea3f4b8e1fc1d326c2dc24d2887d"
REGISTERED_XYZ_SCALE = (0.8422505, 0.827813, 0.937313)
EXPECTED_RESULT_STATUSES = {
    "completed_converged",
    "completed_nonconverged",
    "completed_apparatus_failure",
}
SOLVER_CONFIG: Mapping[str, Any] = {
    "num_steps": 10,
    "intervention_step": 5,
    "dt": -0.1,
    "max_iterations": 128,
    "learning_rate": 0.02,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1.0e-8,
    "xyz_max_abs_tolerance": 0.010,
    "xyz_rms_tolerance": 0.005,
    "full_max_abs_tolerance": 0.050,
    "full_rms_tolerance": 0.015,
    "constraint_slack_ulps": 8,
    "stop_on_first_feasible": False,
}
POLICY_CALL_SEQUENCE = (
    "compiled_frozen_before",
    "eager_source_pairing_before",
    "eager_normalized_before",
    "zero_schedule_before",
    "inverse_flow_teacher_first",
    "inverse_flow_teacher_duplicate",
    "canonical_schedule_replay",
    "zero_schedule_after",
    "eager_normalized_after",
    "eager_source_pairing_after",
    "compiled_frozen_after",
)
ALLOCATION_TEST_COUNTS: Mapping[str, int] = {
    "test_inverse_flow_control.py": 17,
    "test_inverse_flow_sampler.py": 8,
    "test_inverse_flow_policy.py": 10,
    "test_r05a_canary.py": 10,
}


class R05ACanarySourceError(ValueError):
    """An immutable source or current/source pairing is invalid."""


class R05ACanaryPolicyError(RuntimeError):
    """The policy transport failed before returning auditable evidence."""


@dataclass(frozen=True)
class R05ACanaryConfig:
    oracle: OracleConfig
    source_r02_results_root: str
    config_file_sha256: str
    scientific_config_hash: str
    artifact_schema_path: str
    artifact_schema_sha256: str
    decision_artifact: str
    decision_sha256: str
    solver_config: Mapping[str, Any]
    policy_call_sequence: tuple[str, ...]


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve(path: str, root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (Path(root).resolve() / value).resolve()


def normalized_r05a_canary_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove submission-state fields while retaining every scientific choice."""

    normalized = json.loads(json.dumps(value))
    normalized.pop("ready_to_run", None)
    normalized.pop("blocked_on", None)
    return normalized


def r05a_canary_config_hash(value: Mapping[str, Any]) -> str:
    return content_hash(normalized_r05a_canary_config(value))


def r05a_canary_config_from_mapping(
    value: Mapping[str, Any],
    oracle: OracleConfig,
    *,
    repo_root: str | Path,
    source_r02_results_root: str,
    config_file_sha256: str,
) -> R05ACanaryConfig:
    """Load the exact preregistered one-case canary configuration."""

    if value.get("ready_to_run") is not True or value.get("blocked_on") != []:
        raise ValueError("R05A canary config is not released for one exact run")
    if config_file_sha256 != CONFIG_FILE_SHA256:
        raise ValueError("R05A canary config file hash changed")
    scientific_config_hash = r05a_canary_config_hash(value)
    if scientific_config_hash != SCIENTIFIC_CONFIG_HASH:
        raise ValueError("R05A canary scientific config hash changed")
    if value.get("manifest_sha256") != MANIFEST_SHA256:
        raise ValueError("R05A canary manifest binding changed")
    if (
        oracle.action_horizon,
        oracle.action_dim,
        oracle.sampler_steps,
        oracle.intervention_step,
        oracle.executed_prefix,
        oracle.optimizer_max_iterations,
    ) != (10, 32, 10, 5, 5, 128):
        raise ValueError("R05A canary requires the registered 10x32, ten-step sampler")
    if oracle.measurement_repeats != 0 or oracle.stop_after_measurement is not True:
        raise ValueError("R05A canary forbids simulator efficacy measurement")
    settings = value.get("r05a_canary")
    if not isinstance(settings, Mapping):
        raise ValueError("R05A canary config requires r05a_canary settings")
    exact = {
        "case_index": 0,
        "case_id": CASE_ID,
        "group_id": GROUP_ID,
        "target_object": TARGET_OBJECT_NAME,
        "source_host": SOURCE_HOST,
        "source_r02_result_sha256": SOURCE_R02_SHA256,
        "source_r02_config_sha256": SOURCE_R02_CONFIG_SHA256,
        "source_r03_summary_sha256": R03_SUMMARY_SHA256,
        "source_r03_ordered_result_set_digest": R03_ORDERED_RESULT_SET_DIGEST,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "baseline_commit": BASELINE_COMMIT,
        "normalization_asset_sha256": NORMALIZATION_ASSET_SHA256,
        "decision_artifact": DECISION_ARTIFACT,
        "decision_sha256": DECISION_SHA256,
        "artifact_schema": SCHEMA_PATH,
        "artifact_schema_sha256": SCHEMA_SHA256,
        "policy_generated_action_steps_executed": 0,
        "teacher_generated_action_steps_executed": 0,
        "efficacy_rollouts_executed": 0,
        "simulator_efficacy_evaluated": False,
        "training": False,
        "probe_training_authorized": False,
        "scientific_claim_allowed": False,
    }
    for key, expected in exact.items():
        if settings.get(key) != expected:
            raise ValueError(f"R05A canary frozen field changed: {key}")
    if tuple(float(item) for item in settings.get("registered_xyz_action_scale", ())) != REGISTERED_XYZ_SCALE:
        raise ValueError("R05A canary registered XYZ scale changed")
    if settings.get("solver") != dict(SOLVER_CONFIG):
        raise ValueError("R05A canary solver configuration changed")
    if tuple(settings.get("policy_call_sequence", ())) != POLICY_CALL_SEQUENCE:
        raise ValueError("R05A canary policy call sequence changed")
    resources = settings.get("resource_contract")
    if not isinstance(resources, Mapping) or resources != {
        "partition": "main",
        "source_host": SOURCE_HOST,
        "gpus": 1,
        "cpus_per_task": 8,
        "host_memory_mib": 65536,
        "array": "0-0%1",
        "requeue": False,
        "validator_partition": "main",
        "validator_cpus": 2,
        "validator_host_memory_mib": 8192,
        "validator_dependency": "afterany",
    }:
        raise ValueError("R05A canary resource contract changed")
    root = Path(repo_root).resolve()
    for path_key, hash_key in (
        ("decision_artifact", "decision_sha256"),
        ("artifact_schema", "artifact_schema_sha256"),
        ("source_r02_config", "source_r02_config_sha256"),
        ("source_r03_summary", "source_r03_summary_sha256"),
    ):
        path = _resolve(str(settings[path_key]), root)
        if file_sha256(path) != settings[hash_key]:
            raise ValueError(f"R05A canary content binding differs: {path_key}")
    if oracle.checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("R05A canary checkpoint differs from the frozen R02 source")
    return R05ACanaryConfig(
        oracle=oracle,
        source_r02_results_root=str(Path(source_r02_results_root).resolve()),
        config_file_sha256=config_file_sha256,
        scientific_config_hash=scientific_config_hash,
        artifact_schema_path=str(_resolve(SCHEMA_PATH, root)),
        artifact_schema_sha256=SCHEMA_SHA256,
        decision_artifact=DECISION_ARTIFACT,
        decision_sha256=DECISION_SHA256,
        solver_config=dict(SOLVER_CONFIG),
        policy_call_sequence=POLICY_CALL_SEQUENCE,
    )


def _array_exact(left: Any, right: Any) -> bool:
    first = np.ascontiguousarray(np.asarray(left))
    second = np.ascontiguousarray(np.asarray(right))
    return bool(
        first.dtype == second.dtype
        and first.shape == second.shape
        and first.tobytes() == second.tobytes()
    )


def _trace_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        set(left) == set(right)
        and all(_array_exact(left[key], right[key]) for key in left)
    )


def _scalar(value: Any, *, name: str) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    return array.item()


def _nested_mapping_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _finite_array(value: Any, *, name: str, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
    array = np.asarray(value)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite numeric array")
    return array


def _load_source_r02(config: R05ACanaryConfig) -> tuple[Path, Mapping[str, Any], str]:
    path = Path(config.source_r02_results_root) / CASE_ID / "r02-paired.json"
    digest = file_sha256(path)
    if digest != SOURCE_R02_SHA256:
        raise R05ACanarySourceError(
            f"raw R02 hash mismatch: expected {SOURCE_R02_SHA256}, got {digest}"
        )
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise R05ACanarySourceError("raw R02 source is not an object")
    errors = validate_r02_result(value)
    if errors:
        raise R05ACanarySourceError("raw R02 validation failed: " + "; ".join(errors))
    if value.get("case_id") != CASE_ID or value.get("status") != "completed":
        raise R05ACanarySourceError("raw R02 source is not the completed fixed canary case")
    outcome = value.get("outcome")
    if not isinstance(outcome, Mapping) or not (
        outcome.get("r01_feasible_conditioned") is True
        and outcome.get("nominal_collision_reproduced") is True
        and outcome.get("direct_witness_reconfirmed") is True
    ):
        raise R05ACanarySourceError("raw R02 source is outside the witness-confirmed population")
    return path, value, digest


def _source_delta(raw_r02: Mapping[str, Any]) -> tuple[np.ndarray, Mapping[str, Any], float]:
    record = raw_r02.get("directions", {}).get("arrays", {}).get("delta_star_model")
    delta, errors = _validate_array_record(
        record, name="directions.arrays.delta_star_model", shape=(10, 32)
    )
    if errors or delta is None or not isinstance(record, Mapping):
        raise R05ACanarySourceError("invalid source Delta*: " + "; ".join(errors))
    mask = np.zeros((10, 32), dtype=np.bool_)
    mask[:5, :3] = True
    if bool(np.any(np.asarray(delta)[~mask] != 0)):
        raise R05ACanarySourceError("source Delta* violates the first-five XYZ mask")
    budget = float(np.linalg.norm(np.asarray(delta, dtype=np.float64)[:5, :3]))
    reported = raw_r02.get("directions", {}).get("l2_norms", {}).get("delta_star_model")
    if not math.isfinite(budget) or budget <= 0 or not math.isclose(
        budget, float(reported), rel_tol=0.0, abs_tol=1e-12
    ):
        raise R05ACanarySourceError("source Delta* budget conflicts with its R02 record")
    return np.asarray(delta), dict(record), budget


def _request(
    client: Any,
    observation: Mapping[str, Any],
    controls: Mapping[str, Any],
    *,
    require_trace: bool,
) -> tuple[Mapping[str, Any], float]:
    request = copy.deepcopy(dict(observation))
    request["__crfs__"] = copy.deepcopy(dict(controls))
    started = time.perf_counter_ns()
    try:
        reply = client.infer(request)
    except Exception as error:
        raise R05ACanaryPolicyError(f"{type(error).__name__}: {error}") from error
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
    if not isinstance(reply, Mapping) or "actions" not in reply:
        raise R05ACanaryPolicyError("policy reply has no actions")
    if require_trace and not isinstance(reply.get("crfs_trace"), Mapping):
        raise R05ACanaryPolicyError("policy reply has no CRFS trace")
    return reply, float(elapsed)


def _frozen_controls(
    noise: np.ndarray, *, return_trace: bool, return_normalized_final: bool = False
) -> Mapping[str, Any]:
    controls: dict[str, Any] = {
        "noise": np.asarray(noise, dtype=np.float32).copy(),
        "intervention_step": 5,
        "intervention_mode": "none",
        "return_trace": bool(return_trace),
    }
    if return_normalized_final:
        controls["return_normalized_final"] = True
    return controls


def _teacher_controls(
    noise: np.ndarray, target: np.ndarray, source_budget_float32: np.float32
) -> Mapping[str, Any]:
    return {
        "noise": np.asarray(noise, dtype=np.float32).copy(),
        "intervention_mode": "inverse_flow_teacher",
        "intervention_step": 5,
        "return_trace": True,
        "return_normalized_final": True,
        "target": np.asarray(target, dtype=np.float32).copy(),
        "target_space": "model",
        "model_l2_path_budget": np.float32(source_budget_float32),
        "solver_config": dict(SOLVER_CONFIG),
    }


def _schedule_controls(
    noise: np.ndarray, schedule: np.ndarray, budget: float
) -> Mapping[str, Any]:
    return {
        "noise": np.asarray(noise, dtype=np.float32).copy(),
        "intervention_mode": "residual_schedule",
        "intervention_step": 5,
        "return_trace": True,
        "return_normalized_final": True,
        "schedule": np.asarray(schedule, dtype=np.float32).copy(),
        "schedule_space": "model",
        "model_l2_path_budget": np.float32(budget),
    }


def _expected_times() -> np.ndarray:
    value = np.asarray(1.0, dtype=np.float32)
    dt = np.asarray(-0.1, dtype=np.float32)
    values = []
    for _ in range(10):
        values.append(value.copy())
        value = np.asarray(value + dt, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def validate_flow_recurrence(trace: Mapping[str, Any]) -> list[str]:
    """Recompute every ordinary Euler step from the serialized float32 trace."""

    errors: list[str] = []
    required = {
        "step_index_steps",
        "time_steps",
        "active_steps",
        "x_t_steps",
        "v_base_steps",
        "control_velocity_steps",
        "total_velocity_steps",
        "control_increment_steps",
        "x_next_steps",
        "initial_noise",
        "final_normalized",
    }
    missing = required - set(trace)
    if missing:
        return [f"flow trace missing leaves: {sorted(missing)}"]
    try:
        step = _finite_array(trace["step_index_steps"], name="step_index_steps", shape=(10,))
        times = _finite_array(trace["time_steps"], name="time_steps", shape=(10,))
        active = np.asarray(trace["active_steps"])
        x_t = _finite_array(trace["x_t_steps"], name="x_t_steps", shape=(10, 10, 32))
        v_base = _finite_array(trace["v_base_steps"], name="v_base_steps", shape=(10, 10, 32))
        control = _finite_array(
            trace["control_velocity_steps"], name="control_velocity_steps", shape=(10, 10, 32)
        )
        total = _finite_array(
            trace["total_velocity_steps"], name="total_velocity_steps", shape=(10, 10, 32)
        )
        increment = _finite_array(
            trace["control_increment_steps"], name="control_increment_steps", shape=(10, 10, 32)
        )
        x_next = _finite_array(trace["x_next_steps"], name="x_next_steps", shape=(10, 10, 32))
        final = _finite_array(trace["final_normalized"], name="final_normalized", shape=(10, 32))
        initial_noise = _finite_array(trace["initial_noise"], name="initial_noise", shape=(10, 32))
    except ValueError as error:
        return [str(error)]
    if step.dtype != np.dtype(np.int64) or not _array_exact(step, np.arange(10, dtype=np.int64)):
        errors.append("flow step indices are not exact int64 0..9")
    if times.dtype != np.dtype(np.float32) or not _array_exact(times, _expected_times()):
        errors.append("flow times do not match iterative float32 sampler time")
    expected_active = np.asarray([False] * 5 + [True] * 5, dtype=np.bool_)
    if active.dtype != np.dtype(np.bool_) or not _array_exact(active, expected_active):
        errors.append("flow active mask is not exact steps 5..9")
    for name, value in (
        ("x_t", x_t),
        ("v_base", v_base),
        ("control", control),
        ("total", total),
        ("increment", increment),
        ("x_next", x_next),
        ("final", final),
        ("initial_noise", initial_noise),
    ):
        if value.dtype != np.dtype(np.float32):
            errors.append(f"flow {name} must preserve float32")
    expected_total = np.where(control == 0, v_base, v_base + control)
    expected_increment = np.asarray(-0.1, dtype=np.float32) * control
    expected_next = x_t + np.asarray(-0.1, dtype=np.float32) * total
    if not _array_exact(total, expected_total):
        errors.append("flow total velocity violates exact zero-preserving addition")
    if not _array_exact(increment, expected_increment):
        errors.append("flow increment is not exact dt*u")
    if not _array_exact(x_next, expected_next):
        errors.append("flow x_next violates exact recorded float32 recurrence")
    if not _array_exact(x_t[1:], x_next[:-1]):
        errors.append("flow state rows are disconnected")
    if not _array_exact(x_t[0], initial_noise):
        errors.append("flow first state differs from explicit paired initial_noise")
    if not _array_exact(final, x_next[-1]):
        errors.append("flow final_normalized differs from the last recurrence state")
    return errors


def _schedule_diagnostics(
    schedule: np.ndarray, *, source_budget_float32: np.float32
) -> tuple[Mapping[str, Any], list[str]]:
    errors: list[str] = []
    value = np.asarray(schedule)
    if value.shape != (10, 10, 32):
        return {"shape": list(value.shape)}, [f"schedule must have shape (10, 10, 32), got {value.shape}"]
    if value.dtype != np.dtype(np.float32) or not bool(np.isfinite(value).all()):
        errors.append("schedule must preserve finite float32 values")
    mask = np.zeros((10, 32), dtype=np.bool_)
    mask[:5, :3] = True
    if bool(np.count_nonzero(value[:5])):
        errors.append("schedule steps 0..4 are not exactly zero")
    expanded_mask = np.broadcast_to(mask, value.shape)
    if bool(np.count_nonzero(np.where(expanded_mask, 0, value))):
        errors.append("schedule is nonzero outside first-five XYZ")
    dt = np.asarray(-0.1, dtype=np.float32)
    increments = dt * value
    per_step = np.linalg.norm(
        np.asarray(increments[5:, :5, :3], dtype=np.float32).reshape(5, -1), axis=1
    ).astype(np.float32)
    path = np.asarray(np.sum(per_step, dtype=np.float32), dtype=np.float32)
    budget = np.asarray(source_budget_float32, dtype=np.float32)
    allowed_budget = budget.copy()
    allowed_cap = np.asarray(budget / np.float32(5.0), dtype=np.float32)
    for _ in range(8):
        allowed_budget = np.nextafter(allowed_budget, np.float32(np.inf), dtype=np.float32)
        allowed_cap = np.nextafter(allowed_cap, np.float32(np.inf), dtype=np.float32)
    if bool(np.any(per_step > allowed_cap)):
        errors.append("a schedule increment exceeds the source-bound B/5 cap")
    if bool(path > allowed_budget):
        errors.append("schedule path exceeds the source-bound B")
    return (
        {
            "schedule": _array_record(value, dtype=np.float32),
            "increments": _array_record(increments, dtype=np.float32),
            "per_step_increment_l2": _array_record(per_step, dtype=np.float32),
            "path_length": float(path),
            "source_budget_float32": float(budget),
            "per_step_cap_float32": float(np.asarray(budget / np.float32(5.0), dtype=np.float32)),
            "constraint_slack_ulps": 8,
            "passed": not errors,
        },
        errors,
    )


def _fidelity_diagnostics(
    final_normalized: np.ndarray,
    target_normalized: np.ndarray,
    model_to_physical_scale: np.ndarray,
) -> Mapping[str, Any]:
    final = np.asarray(final_normalized, dtype=np.float32)
    target = np.asarray(target_normalized, dtype=np.float32)
    scale = np.asarray(model_to_physical_scale, dtype=np.float32)
    if final.shape != (10, 32) or target.shape != (10, 32) or scale.shape != (10, 32):
        raise ValueError("fidelity arrays must all have shape (10, 32)")
    error = np.asarray((final - target) * scale, dtype=np.float32)
    xyz = error[:5, :3]
    full = error[:5, :7]
    metrics = {
        "xyz_max_abs": float(np.max(np.abs(xyz))),
        "xyz_rms": float(np.sqrt(np.mean(np.square(xyz), dtype=np.float32))),
        "full_max_abs": float(np.max(np.abs(full))),
        "full_rms": float(np.sqrt(np.mean(np.square(full), dtype=np.float32))),
    }
    passed = bool(
        metrics["xyz_max_abs"] <= SOLVER_CONFIG["xyz_max_abs_tolerance"]
        and metrics["xyz_rms"] <= SOLVER_CONFIG["xyz_rms_tolerance"]
        and metrics["full_max_abs"] <= SOLVER_CONFIG["full_max_abs_tolerance"]
        and metrics["full_rms"] <= SOLVER_CONFIG["full_rms_tolerance"]
    )
    return {
        **metrics,
        "physical_error": _array_record(error, dtype=np.float32),
        "thresholds": {
            key: SOLVER_CONFIG[key]
            for key in (
                "xyz_max_abs_tolerance",
                "xyz_rms_tolerance",
                "full_max_abs_tolerance",
                "full_rms_tolerance",
            )
        },
        "passed": passed,
    }


def _teacher_summary(
    reply: Mapping[str, Any],
    *,
    target: np.ndarray,
    frozen_baseline: np.ndarray,
    paired_noise: np.ndarray,
    expected_realized_delta: np.ndarray,
    source_budget_float32: np.float32,
    elapsed_seconds: float,
) -> Mapping[str, Any]:
    trace = reply.get("crfs_trace")
    if not isinstance(trace, Mapping):
        raise R05ACanaryPolicyError("teacher reply has no trace")
    recurrence_errors = validate_flow_recurrence(trace)
    required = {
        "control_source",
        "control_valid",
        "schedule_applied",
        "solver_status",
        "solver_converged",
        "solver_iterations",
        "solver_fields_available",
        "solver_nonfinite",
        "solver_baseline_valid",
        "solver_schedule_valid",
        "target_pairing_checked",
        "target_pairing_exact",
        "inverse_target",
        "model_to_physical_scale",
        "source_control_budget",
        "parameter_requires_grad_restored",
        "parameter_grads_none_before",
        "parameter_grads_none_after",
        "parameter_grad_check_performed",
        "cuda_memory_available",
        "initial_noise",
        "canonical_replay_final",
        "final_normalized_physical",
        "dt",
        "intervention_step",
        "num_steps",
    }
    missing = required - set(trace)
    if missing:
        raise R05ACanaryPolicyError(f"teacher trace missing required leaves: {sorted(missing)}")
    status = int(_scalar(trace["solver_status"], name="solver_status"))
    converged = bool(_scalar(trace["solver_converged"], name="solver_converged"))
    fields_available = bool(
        _scalar(trace["solver_fields_available"], name="solver_fields_available")
    )
    target_trace = _finite_array(trace["inverse_target"], name="inverse_target", shape=(10, 32))
    scale = _finite_array(
        trace["model_to_physical_scale"], name="model_to_physical_scale", shape=(10, 32)
    )
    if target_trace.dtype != np.dtype(np.float32) or not _array_exact(target_trace, target):
        raise R05ACanaryPolicyError("teacher trace target differs from the exact float32 client target")
    if scale.dtype != np.dtype(np.float32) or not bool(np.all(scale > 0)):
        raise R05ACanaryPolicyError("teacher trace checkpoint scale is not positive float32")
    observed_budget = np.asarray(
        _scalar(trace["source_control_budget"], name="source_control_budget"), dtype=np.float32
    )
    if observed_budget.tobytes() != np.asarray(source_budget_float32, dtype=np.float32).tobytes():
        raise R05ACanaryPolicyError("teacher source-control budget differs from the R02-bound float32 scalar")
    schedule = None
    schedule_diagnostics: Optional[Mapping[str, Any]] = None
    schedule_errors: list[str] = []
    budget_checks: dict[str, bool] = {
        "source_control_budget_exact": True,
        "schedule_budget_exact": False,
        "realized_target_delta_elements_exact": False,
        "solver_model_error_exact": False,
        "solver_baseline_exact_fresh_target_pair": False,
    }
    reported_realized_norm: Optional[np.float32] = None
    independent_realized_norm: Optional[np.float32] = None
    if fields_available:
        field_keys = {
            "solver_schedule",
            "solver_baseline_final",
            "solver_internal_replay_final",
            "schedule_budget",
            "realized_target_delta_norm",
            "schedule_path_length",
            "schedule_per_step_increment_l2",
            "solver_model_error",
        }
        missing_fields = field_keys - set(trace)
        if missing_fields:
            raise R05ACanaryPolicyError(
                f"teacher marks solver fields available but omits {sorted(missing_fields)}"
            )
        schedule = _finite_array(trace["solver_schedule"], name="solver_schedule", shape=(10, 10, 32))
        schedule_diagnostics, schedule_errors = _schedule_diagnostics(
            schedule, source_budget_float32=source_budget_float32
        )
        baseline_final = _finite_array(
            trace["solver_baseline_final"], name="solver_baseline_final", shape=(10, 32)
        )
        schedule_budget = np.asarray(
            _scalar(trace["schedule_budget"], name="schedule_budget"), dtype=np.float32
        )
        reported_realized_norm = np.asarray(
            _scalar(trace["realized_target_delta_norm"], name="realized_target_delta_norm"),
            dtype=np.float32,
        )
        realized = np.zeros((10, 32), dtype=np.float32)
        realized[:5, :3] = np.asarray(target - baseline_final, dtype=np.float32)[:5, :3]
        independent_realized_norm = np.asarray(
            np.linalg.norm(realized.reshape(-1)), dtype=np.float32
        )
        solver_model_error = _finite_array(
            trace["solver_model_error"], name="solver_model_error", shape=(10, 32)
        )
        internal_replay_final = _finite_array(
            trace["solver_internal_replay_final"],
            name="solver_internal_replay_final",
            shape=(10, 32),
        )
        expected_model_error = np.asarray(internal_replay_final - target, dtype=np.float32)
        budget_checks.update(
            {
                "schedule_budget_exact": schedule_budget.tobytes()
                == np.asarray(source_budget_float32, dtype=np.float32).tobytes(),
                "realized_target_delta_elements_exact": _array_exact(
                    realized, np.asarray(expected_realized_delta, dtype=np.float32)
                ),
                "solver_model_error_exact": _array_exact(solver_model_error, expected_model_error),
                "solver_baseline_exact_fresh_target_pair": _array_exact(
                    baseline_final, np.asarray(frozen_baseline, dtype=np.float32)
                ),
            }
        )
    elif "solver_schedule" in trace:
        raise R05ACanaryPolicyError("teacher serialized a schedule while fields_available is false")
    final = _finite_array(trace["final_normalized"], name="final_normalized", shape=(10, 32))
    fidelity_final = (
        _finite_array(
            trace["solver_internal_replay_final"],
            name="solver_internal_replay_final",
            shape=(10, 32),
        )
        if fields_available
        else final
    )
    fidelity = _fidelity_diagnostics(fidelity_final, target, scale)
    cuda_available = bool(_scalar(trace["cuda_memory_available"], name="cuda_memory_available"))
    cuda: dict[str, Any] = {"available": cuda_available}
    cuda_keys = (
        "cuda_memory_allocated_before_bytes",
        "cuda_memory_reserved_before_bytes",
        "cuda_memory_allocated_after_bytes",
        "cuda_memory_reserved_after_bytes",
        "cuda_process_peak_allocated_bytes",
        "cuda_process_peak_reserved_bytes",
    )
    if cuda_available:
        missing_cuda = set(cuda_keys) - set(trace)
        if missing_cuda:
            raise R05ACanaryPolicyError(f"CUDA telemetry marked available but missing {sorted(missing_cuda)}")
        for key in cuda_keys:
            value = int(_scalar(trace[key], name=key))
            if value < 0:
                raise R05ACanaryPolicyError(f"{key} is negative")
            cuda[key] = value
    invariants = {
        "control_source_is_teacher": int(_scalar(trace["control_source"], name="control_source")) == 0,
        "control_valid": bool(_scalar(trace["control_valid"], name="control_valid")),
        "schedule_applied": bool(_scalar(trace["schedule_applied"], name="schedule_applied")),
        "baseline_valid": bool(_scalar(trace["solver_baseline_valid"], name="solver_baseline_valid")),
        "schedule_valid": bool(_scalar(trace["solver_schedule_valid"], name="solver_schedule_valid")),
        "target_pairing_checked": bool(
            _scalar(trace["target_pairing_checked"], name="target_pairing_checked")
        ),
        "target_pairing_exact": bool(_scalar(trace["target_pairing_exact"], name="target_pairing_exact")),
        "parameter_requires_grad_restored": bool(
            _scalar(trace["parameter_requires_grad_restored"], name="parameter_requires_grad_restored")
        ),
        "parameter_grads_none_before": bool(
            _scalar(trace["parameter_grads_none_before"], name="parameter_grads_none_before")
        ),
        "parameter_grads_none_after": bool(
            _scalar(trace["parameter_grads_none_after"], name="parameter_grads_none_after")
        ),
        "parameter_grad_check_performed": bool(
            _scalar(trace["parameter_grad_check_performed"], name="parameter_grad_check_performed")
        ),
        "dt_exact": np.asarray(_scalar(trace["dt"], name="dt"), dtype=np.float32).tobytes()
        == np.asarray(-0.1, dtype=np.float32).tobytes(),
        "intervention_step_exact": int(
            _scalar(trace["intervention_step"], name="intervention_step")
        )
        == 5,
        "num_steps_exact": int(_scalar(trace["num_steps"], name="num_steps")) == 10,
        "fixed_128_updates_complete": int(
            _scalar(trace["solver_iterations"], name="solver_iterations")
        )
        == 128,
        "initial_noise_exact": _array_exact(
            trace["initial_noise"], np.asarray(paired_noise, dtype=np.float32)
        ),
        "recurrence_exact": not recurrence_errors,
        "schedule_constraints_passed": not schedule_errors if fields_available else False,
        "budget_binding_passed": all(budget_checks.values()),
        "fidelity_passed": bool(fidelity["passed"]),
        "final_physical_exact_returned_actions": _array_exact(
            trace["final_normalized_physical"], reply["actions"]
        ),
        "cuda_memory_available": cuda_available,
    }
    if converged and not all(invariants.values()):
        # Keep the evidence, but the caller will mark apparatus failure.
        pass
    return {
        "status_code": status,
        "converged": converged,
        "iterations": int(_scalar(trace["solver_iterations"], name="solver_iterations")),
        "fields_available": fields_available,
        "nonfinite": bool(_scalar(trace["solver_nonfinite"], name="solver_nonfinite")),
        "source_control_budget_float32": float(observed_budget),
        "budget_checks": budget_checks,
        "reported_realized_target_delta_norm": (
            None if reported_realized_norm is None else float(reported_realized_norm)
        ),
        "independent_realized_target_delta_norm": (
            None if independent_realized_norm is None else float(independent_realized_norm)
        ),
        "realized_target_delta_norm_ulp_difference": (
            None
            if reported_realized_norm is None or independent_realized_norm is None
            else abs(
                int(reported_realized_norm.view(np.uint32))
                - int(independent_realized_norm.view(np.uint32))
            )
        ),
        "schedule": None if schedule is None else _array_record(schedule, dtype=np.float32),
        "schedule_diagnostics": schedule_diagnostics,
        "fidelity": fidelity,
        "checkpoint_model_to_physical_scale": _array_record(scale, dtype=np.float32),
        "recurrence_errors": recurrence_errors,
        "schedule_errors": schedule_errors,
        "invariants": invariants,
        "cuda_memory": cuda,
        "actions": _array_record(reply["actions"]),
        "trace": _trace_record(trace),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _replay_summary(
    reply: Mapping[str, Any],
    *,
    requested_schedule: np.ndarray,
    source_budget_float32: np.float32,
    paired_noise: np.ndarray,
    elapsed_seconds: float,
) -> Mapping[str, Any]:
    trace = reply.get("crfs_trace")
    if not isinstance(trace, Mapping):
        raise R05ACanaryPolicyError("schedule replay reply has no trace")
    recurrence_errors = validate_flow_recurrence(trace)
    schedule = _finite_array(
        trace.get("control_velocity_steps"),
        name="control_velocity_steps",
        shape=(10, 10, 32),
    )
    diagnostics, schedule_errors = _schedule_diagnostics(
        schedule, source_budget_float32=source_budget_float32
    )
    checks = {
        "control_source_is_replay": int(_scalar(trace.get("control_source"), name="control_source")) == 1,
        "control_valid": bool(_scalar(trace.get("control_valid"), name="control_valid")),
        "schedule_applied": bool(_scalar(trace.get("schedule_applied"), name="schedule_applied")),
        "requested_schedule_exact": _array_exact(schedule, requested_schedule),
        "final_physical_exact_returned_actions": _array_exact(
            trace.get("final_normalized_physical"), reply["actions"]
        ),
        "recurrence_exact": not recurrence_errors,
        "constraints_passed": not schedule_errors,
        "dt_exact": np.asarray(_scalar(trace.get("dt"), name="dt"), dtype=np.float32).tobytes()
        == np.asarray(-0.1, dtype=np.float32).tobytes(),
        "intervention_step_exact": int(
            _scalar(trace.get("intervention_step"), name="intervention_step")
        )
        == 5,
        "num_steps_exact": int(_scalar(trace.get("num_steps"), name="num_steps")) == 10,
        "initial_noise_exact": _array_exact(
            trace.get("initial_noise"), np.asarray(paired_noise, dtype=np.float32)
        ),
    }
    observed_budget = np.asarray(
        _scalar(trace.get("schedule_budget"), name="schedule_budget"), dtype=np.float32
    )
    checks["source_budget_exact"] = (
        observed_budget.tobytes()
        == np.asarray(source_budget_float32, dtype=np.float32).tobytes()
    )
    return {
        "actions": _array_record(reply["actions"]),
        "final_normalized": _array_record(trace["final_normalized"], dtype=np.float32),
        "trace": _trace_record(trace),
        "schedule_diagnostics": diagnostics,
        "recurrence_errors": recurrence_errors,
        "schedule_errors": schedule_errors,
        "checks": checks,
        "passed": all(checks.values()),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _deterministic_trace(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    """Exclude only explicitly runtime-varying CUDA allocation counters."""

    return {
        str(key): value
        for key, value in trace.items()
        if not str(key).startswith("cuda_")
    }


def _baseline_reply_record(reply: Mapping[str, Any], elapsed: float) -> Mapping[str, Any]:
    trace = reply.get("crfs_trace")
    return {
        "actions": _array_record(reply["actions"]),
        "trace": _trace_record(trace) if isinstance(trace, Mapping) else None,
        "elapsed_seconds": float(elapsed),
    }


def _source_pairing_record(source: Any, current: Any) -> Mapping[str, Any]:
    source_hash = content_hash(source)
    current_hash = content_hash(current)
    return {
        "source": source,
        "current": current,
        "source_sha256": source_hash,
        "current_sha256": current_hash,
        "exact": source_hash == current_hash,
    }


def _provenance(
    case: Mapping[str, Any],
    config: R05ACanaryConfig,
    *,
    repo_root: Path,
    source_path: Path,
    source_sha256: str,
    noise: np.ndarray,
) -> Mapping[str, Any]:
    commit, dirty = _git_state(repo_root)
    return {
        "evidence_tier": "real_pi05_allocation_apparatus_canary_only",
        "scientific_role": "integration_and_resource_measurement_only",
        "training": False,
        "scientific_claim_allowed": False,
        "probe_training_authorized": False,
        "git_commit": commit,
        "git_dirty": dirty,
        "baseline_commit": BASELINE_COMMIT,
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "allocation_gpu_uuid": os.environ.get("CRFS_ALLOCATED_GPU_UUID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "case_record": dict(case),
        "case_record_sha256": content_hash(dict(case)),
        "manifest_sha256": MANIFEST_SHA256,
        "config_file_sha256": config.config_file_sha256,
        "checkpoint_id": config.oracle.checkpoint_id,
        "checkpoint_sha256": config.oracle.checkpoint_sha256,
        "source_r02_case_path": str(source_path),
        "source_r02_case_sha256": source_sha256,
        "noise": _array_record(noise, dtype=np.float32),
        "sampler_steps": 10,
        "sampler_dt": -0.1,
        "action_horizon": 10,
        "action_dim": 32,
        "batch_size": 1,
        "planned_policy_call_sequence": list(config.policy_call_sequence),
        "maximum_policy_call_count": len(config.policy_call_sequence),
        "action_frame": "world-frame OSC translation delta",
        "normalization_rule": "displacement_scale_only_never_mean",
        "policy_generated_action_steps_executed": 0,
        "teacher_generated_action_steps_executed": 0,
    }


def run_r05a_canary(
    case: Mapping[str, Any],
    config: R05ACanaryConfig,
    *,
    repo_root: str | Path,
    input_manifest_sha256: str,
    client: Any = None,
    environment: Optional[SafeLiberoCase] = None,
) -> tuple[Path, str]:
    """Run the one-case sampler canary without an efficacy rollout."""

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("R05A canary must execute inside a Slurm allocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible == "NoDevFiles":
        raise RuntimeError("R05A canary requires one allocation-visible GPU")
    if socket.gethostname().split(".", 1)[0] != SOURCE_HOST:
        raise RuntimeError(f"R05A canary must remain pinned to {SOURCE_HOST}")
    if input_manifest_sha256 != MANIFEST_SHA256:
        raise R05ACanarySourceError("R05A canary input manifest differs")
    if dict(case).get("case_id") != CASE_ID or dict(case).get("group_id") != GROUP_ID:
        raise R05ACanarySourceError("R05A canary case/group identity changed")
    root = Path(repo_root).resolve()
    output = (
        Path(config.oracle.output_root)
        / config.oracle.run_id
        / CASE_ID
        / "canary-payload.json"
    )
    source_path, raw_r02, source_sha = _load_source_r02(config)
    if raw_r02.get("provenance", {}).get("case_record") != dict(case):
        raise R05ACanarySourceError("manifest record differs from immutable R02 source")
    delta64, delta_record, source_budget64 = _source_delta(raw_r02)
    source_reported_budget64 = float(
        raw_r02["directions"]["l2_norms"]["delta_star_model"]
    )
    # Source B is the literal float32 cast of the immutable R02 *reported*
    # scalar.  It is intentionally not recomputed from the rounded target.
    source_budget_float32 = np.float32(source_reported_budget64)
    delta32 = np.asarray(delta64, dtype=np.float32)
    noise = np.random.default_rng(int(case["policy_seed"])).normal(
        size=(10, 32)
    ).astype(np.float32)
    source_noise, noise_errors = _validate_array_record(
        raw_r02.get("provenance", {}).get("noise"),
        name="source R02 noise",
        shape=(10, 32),
    )
    if noise_errors or source_noise is None or not _array_exact(
        noise, np.asarray(source_noise, dtype=np.float32)
    ):
        raise R05ACanarySourceError("reconstructed policy noise differs from R02")
    if client is None:
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(
            config.oracle.host, config.oracle.port
        )
    owns_environment = environment is None
    if environment is None:
        environment = SafeLiberoCase(dict(case), config.oracle)
    else:
        environment.configure_case(dict(case))
    provenance = _provenance(
        case,
        config,
        repo_root=root,
        source_path=source_path,
        source_sha256=source_sha,
        noise=noise,
    )
    try:
        initial_observation = environment.reset_and_settle()
        if environment.obstacle_name is None:
            raise RuntimeError("R05A canary failed to resolve the active obstacle")
        branch = capture_reach_snapshot(
            environment, TARGET_OBJECT_NAME, environment.obstacle_name
        )
        if _target_contact_at_branch(environment, TARGET_OBJECT_NAME) or environment.env.check_success():
            raise RuntimeError("R05A canary branch is no longer a pregrasp reach state")
        policy_input = policy_observation(
            initial_observation, environment.prompt, config.oracle.resize_size
        )
        observation_fingerprint = _observation_fingerprint(policy_input)
        branch_snapshot = _json_compatible(branch.to_dict())
        source_pairing = raw_r02.get("pairing")
        if not isinstance(source_pairing, Mapping):
            raise R05ACanarySourceError("R02 source has no exact pairing record")
        initial_checks = {
            "case_record_exact": raw_r02.get("provenance", {}).get("case_record") == dict(case),
            "observation_exact": observation_fingerprint == source_pairing.get("policy_observation"),
            "branch_snapshot_exact": branch_snapshot == source_pairing.get("branch_snapshot"),
            "noise_exact": True,
        }
        if not all(initial_checks.values()):
            raise R05ACanarySourceError(f"R05A current/source branch pairing failed: {initial_checks}")

        calls: dict[str, Mapping[str, Any]] = {}
        compiled_before, elapsed = _request(
            client,
            policy_input,
            _frozen_controls(noise, return_trace=False),
            require_trace=False,
        )
        calls["compiled_frozen_before"] = _baseline_reply_record(compiled_before, elapsed)

        source_before, elapsed = _request(
            client,
            policy_input,
            _frozen_controls(noise, return_trace=True),
            require_trace=True,
        )
        calls["eager_source_pairing_before"] = _baseline_reply_record(source_before, elapsed)
        source_actions, action_errors = _validate_array_record(
            source_pairing.get("eager_actions"), name="source eager actions", shape=(10, 7)
        )
        if action_errors or source_actions is None:
            raise R05ACanarySourceError("invalid source eager actions: " + "; ".join(action_errors))
        if not _array_exact(
            np.asarray(source_before["actions"], dtype=source_actions.dtype),
            source_actions,
        ):
            raise R05ACanarySourceError("current eager actions differ from immutable R02")
        source_trace = _trace_from_record(
            source_pairing.get("eager_trace"), name="source R02 eager trace"
        )
        source_before_trace = source_before["crfs_trace"]
        source_trace_diagnostics = _trace_pairing_diagnostics(
            source_trace, source_before_trace
        )
        if source_trace_diagnostics.get("exact_native_leaf_pairing") is not True:
            raise R05ACanarySourceError("current eager trace differs from immutable R02")

        eager_before, elapsed = _request(
            client,
            policy_input,
            _frozen_controls(noise, return_trace=True, return_normalized_final=True),
            require_trace=True,
        )
        calls["eager_normalized_before"] = _baseline_reply_record(eager_before, elapsed)
        frozen_final = _finite_array(
            eager_before["crfs_trace"].get("final_normalized"),
            name="fresh frozen final_normalized",
            shape=(10, 32),
        )
        if frozen_final.dtype != np.dtype(np.float32):
            raise R05ACanaryPolicyError("fresh frozen normalized output did not preserve float32")
        if not _array_exact(eager_before["actions"], source_before["actions"]):
            raise R05ACanaryPolicyError("requesting final_normalized changed frozen actions")

        # Addition is restricted with a zero-preserving branch.  This retains
        # every untouched byte, including signed zero, in the paired target.
        target = np.where(
            delta32 == np.asarray(0.0, dtype=np.float32),
            frozen_final,
            np.asarray(frozen_final + delta32, dtype=np.float32),
        ).astype(np.float32, copy=False)
        control_mask = np.zeros((10, 32), dtype=np.bool_)
        control_mask[:5, :3] = True
        target_outside_exact = _array_exact(target[~control_mask], frozen_final[~control_mask])
        if not target_outside_exact:
            raise RuntimeError("R05A target changed an out-of-mask byte")
        realized_delta32 = np.where(
            delta32 == np.asarray(0.0, dtype=np.float32),
            np.asarray(0.0, dtype=np.float32),
            np.asarray(target - frozen_final, dtype=np.float32),
        )

        zero_schedule = np.zeros((10, 10, 32), dtype=np.float32)
        zero_before, elapsed = _request(
            client,
            policy_input,
            _schedule_controls(noise, zero_schedule, np.float32(0.0)),
            require_trace=True,
        )
        zero_before_summary = _replay_summary(
            zero_before,
            requested_schedule=zero_schedule,
            source_budget_float32=np.float32(0.0),
            paired_noise=noise,
            elapsed_seconds=elapsed,
        )
        calls["zero_schedule_before"] = zero_before_summary

        teacher_first, elapsed_first = _request(
            client,
            policy_input,
            _teacher_controls(noise, target, source_budget_float32),
            require_trace=True,
        )
        first_summary = _teacher_summary(
            teacher_first,
            target=target,
            frozen_baseline=frozen_final,
            paired_noise=noise,
            expected_realized_delta=realized_delta32,
            source_budget_float32=source_budget_float32,
            elapsed_seconds=elapsed_first,
        )
        calls["inverse_flow_teacher_first"] = first_summary

        teacher_duplicate, elapsed_duplicate = _request(
            client,
            policy_input,
            _teacher_controls(noise, target, source_budget_float32),
            require_trace=True,
        )
        duplicate_summary = _teacher_summary(
            teacher_duplicate,
            target=target,
            frozen_baseline=frozen_final,
            paired_noise=noise,
            expected_realized_delta=realized_delta32,
            source_budget_float32=source_budget_float32,
            elapsed_seconds=elapsed_duplicate,
        )
        calls["inverse_flow_teacher_duplicate"] = duplicate_summary
        first_trace = teacher_first["crfs_trace"]
        duplicate_trace = teacher_duplicate["crfs_trace"]
        teacher_determinism = {
            "status_exact": first_summary["status_code"] == duplicate_summary["status_code"],
            "convergence_exact": first_summary["converged"] == duplicate_summary["converged"],
            "schedule_exact": (
                first_summary["schedule"] is None
                and duplicate_summary["schedule"] is None
            )
            or (
                first_summary["schedule"] is not None
                and duplicate_summary["schedule"] is not None
                and first_summary["schedule"]["sha256"] == duplicate_summary["schedule"]["sha256"]
            ),
            "actions_exact": _array_exact(teacher_first["actions"], teacher_duplicate["actions"]),
            "deterministic_trace_exact": _trace_exact(
                _deterministic_trace(first_trace), _deterministic_trace(duplicate_trace)
            ),
        }

        if bool(first_summary["converged"]):
            if first_summary["schedule"] is None:
                raise RuntimeError("converged teacher has no schedule")
            schedule = _array_from_record(first_summary["schedule"])
            replay, elapsed = _request(
                client,
                policy_input,
                _schedule_controls(noise, schedule, source_budget_float32),
                require_trace=True,
            )
            replay_summary = dict(
                _replay_summary(
                    replay,
                    requested_schedule=schedule,
                    source_budget_float32=source_budget_float32,
                    paired_noise=noise,
                    elapsed_seconds=elapsed,
                )
            )
            replay_checks = dict(replay_summary["checks"])
            replay_checks.update(
                {
                    "teacher_actions_exact": _array_exact(replay["actions"], teacher_first["actions"]),
                    "teacher_final_exact": _array_exact(
                        replay["crfs_trace"]["final_normalized"],
                        first_trace["final_normalized"],
                    ),
                    "teacher_internal_final_exact": _array_exact(
                        replay["crfs_trace"]["final_normalized"],
                        first_trace["solver_internal_replay_final"],
                    ),
                }
            )
            replay_summary["checks"] = replay_checks
            replay_summary["passed"] = all(replay_checks.values())
            canonical_replay: Mapping[str, Any] = {
                "applicable": True,
                **replay_summary,
            }
            calls["canonical_schedule_replay"] = replay_summary
        else:
            canonical_replay = {
                "applicable": False,
                "passed": False,
                "reason": "finite_teacher_search_did_not_converge",
            }
            calls["canonical_schedule_replay"] = canonical_replay

        zero_after, elapsed = _request(
            client,
            policy_input,
            _schedule_controls(noise, zero_schedule, np.float32(0.0)),
            require_trace=True,
        )
        zero_after_summary = _replay_summary(
            zero_after,
            requested_schedule=zero_schedule,
            source_budget_float32=np.float32(0.0),
            paired_noise=noise,
            elapsed_seconds=elapsed,
        )
        calls["zero_schedule_after"] = zero_after_summary

        eager_after, elapsed = _request(
            client,
            policy_input,
            _frozen_controls(noise, return_trace=True, return_normalized_final=True),
            require_trace=True,
        )
        calls["eager_normalized_after"] = _baseline_reply_record(eager_after, elapsed)
        source_after, elapsed = _request(
            client,
            policy_input,
            _frozen_controls(noise, return_trace=True),
            require_trace=True,
        )
        calls["eager_source_pairing_after"] = _baseline_reply_record(source_after, elapsed)
        compiled_after, elapsed = _request(
            client,
            policy_input,
            _frozen_controls(noise, return_trace=False),
            require_trace=False,
        )
        calls["compiled_frozen_after"] = _baseline_reply_record(compiled_after, elapsed)

        source_after_diagnostics = _trace_pairing_diagnostics(
            source_trace, source_after["crfs_trace"]
        )
        zero_frozen_checks = {
            "zero_before_actions_exact_frozen": _array_exact(zero_before["actions"], eager_before["actions"]),
            "zero_before_final_exact_frozen": _array_exact(
                zero_before["crfs_trace"]["final_normalized"], frozen_final
            ),
            "zero_after_actions_exact_frozen": _array_exact(zero_after["actions"], eager_before["actions"]),
            "zero_after_final_exact_frozen": _array_exact(
                zero_after["crfs_trace"]["final_normalized"], frozen_final
            ),
            "zero_before_after_trace_exact": _trace_exact(
                zero_before["crfs_trace"], zero_after["crfs_trace"]
            ),
            "compiled_before_after_actions_exact": _array_exact(
                compiled_before["actions"], compiled_after["actions"]
            ),
            "compiled_eager_before_actions_exact": _array_exact(
                compiled_before["actions"], eager_before["actions"]
            ),
            "compiled_source_before_actions_exact": _array_exact(
                compiled_before["actions"], source_before["actions"]
            ),
            "eager_before_after_actions_exact": _array_exact(
                eager_before["actions"], eager_after["actions"]
            ),
            "compiled_eager_after_actions_exact": _array_exact(
                compiled_after["actions"], eager_after["actions"]
            ),
            "compiled_source_after_actions_exact": _array_exact(
                compiled_after["actions"], source_after["actions"]
            ),
            "eager_before_after_trace_exact": _trace_exact(
                eager_before["crfs_trace"], eager_after["crfs_trace"]
            ),
            "source_after_actions_exact": _array_exact(
                np.asarray(source_after["actions"], dtype=source_actions.dtype),
                source_actions,
            ),
            "source_after_trace_exact": source_after_diagnostics.get("exact_native_leaf_pairing") is True,
            "zero_before_recurrence_exact": bool(zero_before_summary["passed"]),
            "zero_after_recurrence_exact": bool(zero_after_summary["passed"]),
        }
        policy_scale = _array_from_record(first_summary["checkpoint_model_to_physical_scale"])
        registered_scale = np.asarray(REGISTERED_XYZ_SCALE, dtype=np.float32)
        scale_checks = {
            "positive_float32_full_scale": policy_scale.dtype == np.dtype(np.float32)
            and policy_scale.shape == (10, 32)
            and bool(np.all(policy_scale > 0)),
            "registered_xyz_scale_exact": _array_exact(
                policy_scale[:, :3], np.broadcast_to(registered_scale, (10, 3)).copy()
            ),
            "padded_channels_are_one": _array_exact(
                policy_scale[:, 7:], np.ones((10, 25), dtype=np.float32)
            ),
        }
        direct_record = raw_r02.get("arms", {}).get("direct_witness", {}).get("executed_actions")
        direct_actions, direct_errors = _validate_array_record(
            direct_record, name="source direct witness", shape=(5, 7)
        )
        if direct_errors or direct_actions is None:
            raise R05ACanarySourceError("source direct witness is invalid: " + "; ".join(direct_errors))
        target_physical = np.asarray(eager_before["actions"], dtype=np.float64).copy()
        target_physical[:5, :3] += np.asarray(realized_delta32[:5, :3], dtype=np.float64) * np.asarray(
            policy_scale[:5, :3], dtype=np.float64
        )
        direct_error = target_physical[:5, :7] - np.asarray(direct_actions, dtype=np.float64)
        target_physical_checks = {
            "xyz_max_abs": float(np.max(np.abs(direct_error[:, :3]))),
            "xyz_rms": float(np.sqrt(np.mean(np.square(direct_error[:, :3])))),
            "full_max_abs": float(np.max(np.abs(direct_error))),
            "full_rms": float(np.sqrt(np.mean(np.square(direct_error)))),
        }
        target_physical_checks["within_registered_apparatus_tolerances"] = bool(
            target_physical_checks["xyz_max_abs"] <= 0.010
            and target_physical_checks["xyz_rms"] <= 0.005
            and target_physical_checks["full_max_abs"] <= 0.050
            and target_physical_checks["full_rms"] <= 0.015
        )
        pairing_checks = {
            **initial_checks,
            "source_actions_exact_before": True,
            "source_trace_exact_before": True,
            "source_actions_exact_after": zero_frozen_checks["source_after_actions_exact"],
            "source_trace_exact_after": zero_frozen_checks["source_after_trace_exact"],
            "target_outside_mask_bitwise_exact": target_outside_exact,
        }
        teacher_invariants_passed = all(first_summary["invariants"].values())
        deterministic_passed = all(teacher_determinism.values())
        zero_frozen_passed = all(zero_frozen_checks.values())
        scale_passed = all(scale_checks.values())
        nonconvergence_fail_closed_checks = {
            "first_returned_actions_exact_frozen": _array_exact(
                teacher_first["actions"], eager_before["actions"]
            ),
            "duplicate_returned_actions_exact_frozen": _array_exact(
                teacher_duplicate["actions"], eager_before["actions"]
            ),
            "first_returned_final_exact_frozen": _array_exact(
                first_trace["final_normalized"], frozen_final
            ),
            "duplicate_returned_final_exact_frozen": _array_exact(
                duplicate_trace["final_normalized"], frozen_final
            ),
            "first_applied_control_exact_zero": bool(
                np.count_nonzero(np.asarray(first_trace["control_velocity_steps"])) == 0
            ),
            "duplicate_applied_control_exact_zero": bool(
                np.count_nonzero(np.asarray(duplicate_trace["control_velocity_steps"])) == 0
            ),
            "first_canonical_final_exact_frozen": _array_exact(
                first_trace["canonical_replay_final"], frozen_final
            ),
            "duplicate_canonical_final_exact_frozen": _array_exact(
                duplicate_trace["canonical_replay_final"], frozen_final
            ),
        }
        apparatus_passed = bool(
            first_summary["status_code"] == 0
            and duplicate_summary["status_code"] == 0
            and first_summary["converged"]
            and duplicate_summary["converged"]
            and teacher_invariants_passed
            and deterministic_passed
            and canonical_replay.get("passed") is True
            and zero_frozen_passed
            and scale_passed
            and all(pairing_checks.values())
            and target_physical_checks["within_registered_apparatus_tolerances"]
        )
        clean_nonconvergence_keys = {
            "control_source_is_teacher",
            "baseline_valid",
            "schedule_valid",
            "target_pairing_checked",
            "target_pairing_exact",
            "parameter_requires_grad_restored",
            "parameter_grads_none_before",
            "parameter_grads_none_after",
            "parameter_grad_check_performed",
            "dt_exact",
            "intervention_step_exact",
            "num_steps_exact",
            "fixed_128_updates_complete",
            "recurrence_exact",
            "schedule_constraints_passed",
            "budget_binding_passed",
            "cuda_memory_available",
        }
        clean_nonconvergence = bool(
            first_summary["status_code"] == 3
            and duplicate_summary["status_code"] == 3
            and not first_summary["converged"]
            and not duplicate_summary["converged"]
            and first_summary["fields_available"]
            and duplicate_summary["fields_available"]
            and not first_summary["nonfinite"]
            and not duplicate_summary["nonfinite"]
            and first_summary["fidelity"]["passed"] is False
            and duplicate_summary["fidelity"]["passed"] is False
            and first_summary["invariants"]["control_valid"] is False
            and duplicate_summary["invariants"]["control_valid"] is False
            and first_summary["invariants"]["schedule_applied"] is False
            and duplicate_summary["invariants"]["schedule_applied"] is False
            and all(
                first_summary["invariants"].get(key) is True
                and duplicate_summary["invariants"].get(key) is True
                for key in clean_nonconvergence_keys
            )
            and all(nonconvergence_fail_closed_checks.values())
            and deterministic_passed
            and zero_frozen_passed
            and scale_passed
            and all(pairing_checks.values())
            and target_physical_checks["within_registered_apparatus_tolerances"]
        )
        if apparatus_passed:
            status = "completed_converged"
        elif clean_nonconvergence:
            status = "completed_nonconverged"
        else:
            status = "completed_apparatus_failure"
        result_without_memory = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "gate": GATE,
            "case_id": CASE_ID,
            "run_id": config.oracle.run_id,
            "status": status,
            "config_hash": config.scientific_config_hash,
            "scientific_claim_allowed": False,
            "probe_training_authorized": False,
            "source_evidence": {
                "manifest_sha256": MANIFEST_SHA256,
                "r02_case_path": str(source_path),
                "r02_case_sha256": source_sha,
                "r02_case_validator": "validate_r02_result:passed",
                "r02_config_sha256": SOURCE_R02_CONFIG_SHA256,
                "r03_summary_sha256": R03_SUMMARY_SHA256,
                "r03_ordered_result_set_digest": R03_ORDERED_RESULT_SET_DIGEST,
                "decision_artifact": config.decision_artifact,
                "decision_sha256": config.decision_sha256,
                "artifact_schema_path": config.artifact_schema_path,
                "artifact_schema_sha256": config.artifact_schema_sha256,
            },
            "provenance": provenance,
            "simulator_use": {
                "setup": "one_reset_plus_20_dummy_settle_control_steps",
                "policy_generated_action_steps_executed": 0,
                "teacher_generated_action_steps_executed": 0,
                "efficacy_rollouts_executed": 0,
                "simulator_efficacy_evaluated": False,
            },
            "pairing": {
                "checks": pairing_checks,
                "observation": _source_pairing_record(
                    source_pairing.get("policy_observation"), observation_fingerprint
                ),
                "branch_snapshot": _source_pairing_record(
                    source_pairing.get("branch_snapshot"), branch_snapshot
                ),
                "source_actions_before": {
                    "source_float64": dict(source_pairing["eager_actions"]),
                    "current_native": _array_record(source_before["actions"]),
                    "current_canonical_float64": _array_record(
                        source_before["actions"], dtype=np.float64
                    ),
                    "canonical_exact": True,
                },
                "source_actions_after": {
                    "source_float64": dict(source_pairing["eager_actions"]),
                    "current_native": _array_record(source_after["actions"]),
                    "current_canonical_float64": _array_record(
                        source_after["actions"], dtype=np.float64
                    ),
                    "canonical_exact": zero_frozen_checks["source_after_actions_exact"],
                },
                "source_trace_before_diagnostics": source_trace_diagnostics,
                "source_trace_after_diagnostics": source_after_diagnostics,
            },
            "target": {
                "definition": "fresh_frozen_normalized_final_plus_immutable_r02_delta_star_model",
                "source_delta_star_model_float64": delta_record,
                "source_reported_budget_float64": source_reported_budget64,
                "source_recomputed_budget_float64": source_budget64,
                "source_budget_float32": float(source_budget_float32),
                "delta_star_model_float32": _array_record(delta32, dtype=np.float32),
                "fresh_frozen_normalized": _array_record(frozen_final, dtype=np.float32),
                "target_normalized": _array_record(target, dtype=np.float32),
                "realized_target_minus_frozen_float32": _array_record(realized_delta32, dtype=np.float32),
                "delta_float32_l2": float(np.linalg.norm(delta32.reshape(-1)).astype(np.float32)),
                "realized_delta_float32_l2": float(
                    np.linalg.norm(realized_delta32.reshape(-1)).astype(np.float32)
                ),
                "budget_minus_delta32_l2": float(
                    np.float32(source_budget_float32 - np.linalg.norm(delta32.reshape(-1)).astype(np.float32))
                ),
                "budget_minus_realized_l2": float(
                    np.float32(
                        source_budget_float32
                        - np.linalg.norm(realized_delta32.reshape(-1)).astype(np.float32)
                    )
                ),
                "outside_mask_bitwise_exact": target_outside_exact,
                "checkpoint_scale_checks": scale_checks,
                "direct_witness_physical_target": _array_record(direct_actions, dtype=np.float64),
                "constructed_target_physical_first_five": _array_record(
                    target_physical[:5, :7], dtype=np.float64
                ),
                "direct_witness_target_diagnostics": target_physical_checks,
            },
            "solver": {
                "algorithm": "deterministic_constrained_direct_shooting",
                "config": dict(config.solver_config),
                "first": first_summary,
                "duplicate": duplicate_summary,
                "nonconvergence_is_infeasibility_certificate": False,
                "optimality_certificate": False,
            },
            "determinism": {
                "teacher": teacher_determinism,
                "zero_and_frozen_before_after": zero_frozen_checks,
                "nonconvergence_fail_closed": nonconvergence_fail_closed_checks,
                "passed": deterministic_passed and zero_frozen_passed,
            },
            "canonical_replay": canonical_replay,
            "apparatus": {
                "passed_before_memory_finalization": apparatus_passed,
                "teacher_invariants_passed": teacher_invariants_passed,
                "determinism_passed": deterministic_passed,
                "canonical_replay_passed": canonical_replay.get("passed") is True,
                "zero_and_frozen_passed": zero_frozen_passed,
                "pairing_passed": all(pairing_checks.values()),
                "checkpoint_scale_passed": scale_passed,
                "clean_nonconvergence": clean_nonconvergence,
                "memory_pending": True,
            },
            "outcome": {
                "interpretation": "apparatus_only_no_teacher_efficacy_or_safety_claim",
                "teacher_converged": bool(first_summary["converged"]),
                "simulator_teacher_outcome": None,
                "student_training_authorized": False,
                "next_gate_authorized": False,
            },
            "policy_calls": calls,
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "payload_type": PAYLOAD_TYPE,
            "complete_result_requires_memory_finalization": True,
            "result_without_memory": result_without_memory,
        }
        atomic_write_json(output, payload)
        return output, status
    finally:
        if owns_environment:
            environment.close()


def validate_r05a_canary_schema(
    value: Mapping[str, Any],
    *,
    schema_path: Optional[str | Path] = None,
    require_jsonschema: bool = False,
) -> list[str]:
    path = Path(schema_path) if schema_path is not None else Path(__file__).resolve().parents[2] / SCHEMA_PATH
    try:
        observed = file_sha256(path)
    except OSError as error:
        return [f"cannot read R05A canary schema: {error}"]
    if observed != SCHEMA_SHA256:
        return ["R05A canary schema content hash differs"]
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema is required"] if require_jsonschema else []
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as error:
        return [f"cannot load R05A canary schema: {error}"]
    return [
        "schema: " + error.message
        for error in sorted(
            validator.iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]


def _artifact_trace(
    value: Any, *, name: str, errors: list[str]
) -> Optional[Mapping[str, np.ndarray]]:
    try:
        return _trace_from_record(value, name=name)
    except (R05ACanarySourceError, ValueError, TypeError, KeyError) as error:
        errors.append(f"{name} cannot be reconstructed: {error}")
        return None


def _artifact_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    errors: list[str],
) -> Optional[np.ndarray]:
    array, item_errors = _validate_array_record(value, name=name, shape=shape)
    errors.extend(item_errors)
    return array


def _parse_allocation_test_log(path: str | Path) -> tuple[str, Mapping[str, int]]:
    """Verify exact non-skipped dependency-backed suite evidence."""

    test_path = Path(path)
    digest = file_sha256(test_path)
    lines = test_path.read_text(encoding="utf-8").splitlines()
    observed: dict[str, int] = {}
    for suite, expected in ALLOCATION_TEST_COUNTS.items():
        prefix = f"verified_test_suite={suite} "
        markers = [line for line in lines if line.startswith(prefix)]
        expected_marker = (
            f"verified_test_suite={suite} expected={expected} "
            f"observed={expected} skips=0 status=passed"
        )
        if markers != [expected_marker]:
            raise ValueError(f"focused-test marker changed for {suite}")
        ran_prefix = f"[{suite}] Ran {expected} test"
        if sum(line.startswith(ran_prefix) for line in lines) != 1:
            raise ValueError(f"focused-test unittest count evidence changed for {suite}")
        if lines.count(f"[{suite}] OK") != 1:
            raise ValueError(f"focused-test unittest status changed for {suite}")
        suite_lines = [line for line in lines if line.startswith(f"[{suite}] ")]
        if any(
            "skipped=" in line or line.startswith(f"[{suite}] FAILED")
            or line.startswith(f"[{suite}] ERROR")
            for line in suite_lines
        ):
            raise ValueError(f"focused-test suite skipped or failed: {suite}")
        observed[suite] = expected
    marker_count = sum(line.startswith("verified_test_suite=") for line in lines)
    if marker_count != len(ALLOCATION_TEST_COUNTS):
        raise ValueError("focused-test log contains an unregistered suite marker")
    return digest, observed


def validate_r05a_canary_result(value: Mapping[str, Any]) -> list[str]:
    """Validate scientific semantics independently of the allocation runner."""

    errors: list[str] = []
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version changed")
    if value.get("artifact_type") != ARTIFACT_TYPE or value.get("gate") != GATE:
        errors.append("artifact identity changed")
    if value.get("case_id") != CASE_ID:
        errors.append("canary case identity changed")
    if value.get("config_hash") != SCIENTIFIC_CONFIG_HASH:
        errors.append("top-level scientific config hash changed")
    frozen_config_path = Path(__file__).resolve().parents[2] / CONFIG_PATH
    try:
        if file_sha256(frozen_config_path) != CONFIG_FILE_SHA256:
            errors.append("frozen R05A config file hash changed")
        frozen_config_value = load_json(frozen_config_path)
        if not isinstance(frozen_config_value, Mapping):
            errors.append("frozen R05A config is not an object")
        elif r05a_canary_config_hash(frozen_config_value) != SCIENTIFIC_CONFIG_HASH:
            errors.append("frozen R05A scientific config hash changed")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"frozen R05A config cannot be validated: {error}")
    status = value.get("status")
    if status not in EXPECTED_RESULT_STATUSES:
        errors.append(f"invalid canary status: {status!r}")
    if value.get("scientific_claim_allowed") is not False:
        errors.append("canary cannot authorize a scientific claim")
    if value.get("probe_training_authorized") is not False:
        errors.append("canary cannot authorize probe training")
    simulator = value.get("simulator_use")
    expected_simulator = {
        "setup": "one_reset_plus_20_dummy_settle_control_steps",
        "policy_generated_action_steps_executed": 0,
        "teacher_generated_action_steps_executed": 0,
        "efficacy_rollouts_executed": 0,
        "simulator_efficacy_evaluated": False,
    }
    if simulator != expected_simulator:
        errors.append("canary simulator-use contract changed")
    source = value.get("source_evidence")
    raw_r02: Optional[Mapping[str, Any]] = None
    raw_source_pairing: Optional[Mapping[str, Any]] = None
    if not isinstance(source, Mapping):
        errors.append("source_evidence must be an object")
    else:
        expected_source = {
            "manifest_sha256": MANIFEST_SHA256,
            "r02_case_sha256": SOURCE_R02_SHA256,
            "r02_config_sha256": SOURCE_R02_CONFIG_SHA256,
            "r03_summary_sha256": R03_SUMMARY_SHA256,
            "r03_ordered_result_set_digest": R03_ORDERED_RESULT_SET_DIGEST,
            "decision_artifact": DECISION_ARTIFACT,
            "decision_sha256": DECISION_SHA256,
            "artifact_schema_sha256": SCHEMA_SHA256,
        }
        for key, expected in expected_source.items():
            if source.get(key) != expected:
                errors.append(f"source_evidence.{key} changed")
        source_path_value = source.get("r02_case_path")
        if not isinstance(source_path_value, str) or not source_path_value:
            errors.append("source_evidence.r02_case_path is missing")
        else:
            source_path = Path(source_path_value)
            try:
                actual_source_sha = file_sha256(source_path)
            except (OSError, ValueError) as error:
                errors.append(f"immutable R02 source cannot be read: {error}")
            else:
                if actual_source_sha != SOURCE_R02_SHA256:
                    errors.append("immutable R02 source file hash changed")
                try:
                    loaded_source = load_json(source_path)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    errors.append(f"immutable R02 source cannot be decoded: {error}")
                else:
                    if not isinstance(loaded_source, Mapping):
                        errors.append("immutable R02 source must be an object")
                    else:
                        raw_r02 = loaded_source
                        source_errors = validate_r02_result(raw_r02)
                        if source_errors:
                            errors.append(
                                "immutable R02 source fails validate_r02_result: "
                                + "; ".join(source_errors)
                            )
                        if raw_r02.get("case_id") != CASE_ID:
                            errors.append("immutable R02 source case identity changed")
                        candidate_pairing = raw_r02.get("pairing")
                        if not isinstance(candidate_pairing, Mapping):
                            errors.append("immutable R02 source pairing is missing")
                        else:
                            raw_source_pairing = candidate_pairing
    provenance_noise: Optional[np.ndarray] = None
    allocation_case_dir: Optional[Path] = None
    gpu_samples_path: Optional[Path] = None
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
    else:
        if str(provenance.get("host", "")).split(".", 1)[0] != SOURCE_HOST:
            errors.append("canary did not run on worker-1")
        if provenance.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            errors.append("checkpoint binding changed")
        if provenance.get("config_file_sha256") != CONFIG_FILE_SHA256:
            errors.append("provenance config file hash changed")
        if not str(provenance.get("allocation_gpu_uuid", "")).startswith("GPU-"):
            errors.append("allocation GPU UUID is missing")
        if provenance.get("git_dirty") is not False:
            errors.append("canary source tree was dirty")
        allocation_tests = provenance.get("allocation_runtime_tests")
        if not isinstance(allocation_tests, Mapping) or allocation_tests.get("exit_code") != 0:
            errors.append("allocation dependency-backed focused tests did not pass")
        else:
            expected_case_dir = EXPERIMENT_ROOT / str(value.get("run_id")) / CASE_ID
            log_path_value = allocation_tests.get("log_path")
            if not isinstance(log_path_value, str):
                errors.append("allocation test log path is missing")
            else:
                log_path = Path(log_path_value)
                try:
                    allocation_case_dir = log_path.resolve(strict=True).parent
                    if log_path.resolve(strict=True) != (
                        expected_case_dir / "allocation-focused-tests.log"
                    ):
                        errors.append("allocation test log escaped the immutable case directory")
                    log_sha, observed_counts = _parse_allocation_test_log(log_path)
                    if allocation_tests.get("log_sha256") != log_sha:
                        errors.append("allocation test log digest differs from live shared storage")
                    if allocation_tests.get("expected_counts") != dict(ALLOCATION_TEST_COUNTS):
                        errors.append("allocation expected test counts changed")
                    if allocation_tests.get("observed_counts") != dict(observed_counts):
                        errors.append("allocation observed test counts changed")
                    if allocation_tests.get("zero_skips") is not True:
                        errors.append("allocation tests did not record zero skips")
                except (OSError, ValueError) as error:
                    errors.append(f"allocation focused-test evidence is invalid: {error}")
            gpu_path_value = provenance.get("gpu_samples_path")
            if isinstance(gpu_path_value, str):
                gpu_samples_path = Path(gpu_path_value)
                try:
                    if gpu_samples_path.resolve(strict=True) != (
                        expected_case_dir / "gpu-memory-samples.csv"
                    ):
                        errors.append("GPU samples escaped the immutable case directory")
                except OSError as error:
                    errors.append(f"GPU samples cannot be resolved: {error}")
            else:
                errors.append("GPU samples path is missing")
        provenance_noise = _artifact_array(
            provenance.get("noise"),
            name="provenance.noise",
            shape=(10, 32),
            errors=errors,
        )
    pairing = value.get("pairing")
    if not isinstance(pairing, Mapping) or not isinstance(pairing.get("checks"), Mapping):
        errors.append("pairing checks are missing")
    elif not all(item is True for item in pairing["checks"].values()):
        errors.append("an exact current/source pairing check failed")
    fresh: Optional[np.ndarray] = None
    target_array: Optional[np.ndarray] = None
    realized_target_delta: Optional[np.ndarray] = None
    target = value.get("target")
    if not isinstance(target, Mapping):
        errors.append("target must be an object")
    else:
        arrays: dict[str, Optional[np.ndarray]] = {}
        for key, shape in (
            ("source_delta_star_model_float64", (10, 32)),
            ("delta_star_model_float32", (10, 32)),
            ("fresh_frozen_normalized", (10, 32)),
            ("target_normalized", (10, 32)),
            ("realized_target_minus_frozen_float32", (10, 32)),
        ):
            array, item_errors = _validate_array_record(
                target.get(key), name=f"target.{key}", shape=shape
            )
            errors.extend(item_errors)
            arrays[key] = array
        fresh = arrays.get("fresh_frozen_normalized")
        target_array = arrays.get("target_normalized")
        delta32 = arrays.get("delta_star_model_float32")
        delta64 = arrays.get("source_delta_star_model_float64")
        realized = arrays.get("realized_target_minus_frozen_float32")
        realized_target_delta = realized
        if all(item is not None for item in (fresh, target_array, delta32, realized)):
            assert fresh is not None and target_array is not None and delta32 is not None and realized is not None
            mask = np.zeros((10, 32), dtype=np.bool_)
            mask[:5, :3] = True
            expected_target = np.where(
                delta32 == np.asarray(0.0, dtype=np.float32),
                fresh,
                np.asarray(fresh + delta32, dtype=np.float32),
            )
            expected_realized = np.where(
                delta32 == np.asarray(0.0, dtype=np.float32),
                np.asarray(0.0, dtype=np.float32),
                np.asarray(expected_target - fresh, dtype=np.float32),
            )
            if not _array_exact(target_array, expected_target):
                errors.append("target does not equal zero-preserving float32 frozen+Delta*")
            if not _array_exact(realized, expected_realized):
                errors.append("recorded realized target delta is inconsistent")
            if not _array_exact(target_array[~mask], fresh[~mask]):
                errors.append("target changed an outside-mask byte")
        if delta64 is not None and delta32 is not None:
            if delta64.dtype != np.dtype(np.float64):
                errors.append("source Delta* did not preserve float64")
            if delta32.dtype != np.dtype(np.float32) or not _array_exact(
                delta32, np.asarray(delta64, dtype=np.float32)
            ):
                errors.append("float32 Delta* is not the exact cast of immutable float64 Delta*")
            recomputed_budget64 = float(np.linalg.norm(np.asarray(delta64, dtype=np.float64)[:5, :3]))
            reported_budget64 = target.get("source_reported_budget_float64")
            if not isinstance(reported_budget64, (int, float)):
                errors.append("source float64 Delta* norm scalar is invalid")
            elif not math.isclose(
                recomputed_budget64, float(reported_budget64), rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append("source float64 Delta* norm differs from its R02 scalar")
            if isinstance(reported_budget64, (int, float)):
                expected_budget32 = np.asarray(float(reported_budget64), dtype=np.float32)
                observed_budget32 = np.asarray(target.get("source_budget_float32"), dtype=np.float32)
                if expected_budget32.tobytes() != observed_budget32.tobytes():
                    errors.append("runtime B is not float32(source R02 reported B)")
            if isinstance(raw_r02, Mapping):
                raw_delta64 = _artifact_array(
                    _nested_mapping_value(
                        raw_r02, "directions", "arrays", "delta_star_model"
                    ),
                    name="immutable_r02.directions.arrays.delta_star_model",
                    shape=(10, 32),
                    errors=errors,
                )
                raw_budget64 = _nested_mapping_value(
                    raw_r02, "directions", "l2_norms", "delta_star_model"
                )
                if raw_delta64 is None or not _array_exact(delta64, raw_delta64):
                    errors.append("target source Delta* differs from immutable R02")
                if not isinstance(raw_budget64, (int, float)) or reported_budget64 != raw_budget64:
                    errors.append("target source budget scalar differs from immutable R02")
        if target.get("outside_mask_bitwise_exact") is not True:
            errors.append("target outside-mask exact check did not pass")
    solver = value.get("solver")
    first: Optional[Mapping[str, Any]] = None
    duplicate: Optional[Mapping[str, Any]] = None
    if not isinstance(solver, Mapping) or solver.get("config") != dict(SOLVER_CONFIG):
        errors.append("solver config changed")
    else:
        if solver.get("nonconvergence_is_infeasibility_certificate") is not False:
            errors.append("finite search cannot claim an infeasibility certificate")
        if solver.get("optimality_certificate") is not False:
            errors.append("finite search cannot claim an optimality certificate")
        first = solver.get("first") if isinstance(solver.get("first"), Mapping) else None
        duplicate = solver.get("duplicate") if isinstance(solver.get("duplicate"), Mapping) else None
        if first is None or duplicate is None:
            errors.append("teacher first/duplicate records are missing")
        else:
            for label, item in (("first", first), ("duplicate", duplicate)):
                errors.extend(_validate_trace_record(item.get("trace"), name=f"solver.{label}.trace"))
                if item.get("source_control_budget_float32") != target.get("source_budget_float32"):
                    errors.append(f"solver.{label} source budget differs from target binding")
                invariants = item.get("invariants")
                if not isinstance(invariants, Mapping):
                    errors.append(f"solver.{label}.invariants missing")
            if bool(first.get("converged")) != bool(duplicate.get("converged")):
                errors.append("duplicate teacher convergence differs")
    # Recompute the decision from serialized arrays/traces.  Stored runner
    # booleans are evidence labels, never the authority for the CPU afterany
    # trust path.
    recomputed_converged = False
    recomputed_clean_nonconvergence = False
    recomputed_determinism = False
    recomputed_zero_frozen = False
    recomputed_pairing = False
    recomputed_scale = False
    recomputed_canonical = False
    recomputed_direct_target = False
    first_trace_arrays: Optional[Mapping[str, np.ndarray]] = None
    duplicate_trace_arrays: Optional[Mapping[str, np.ndarray]] = None
    if first is not None and duplicate is not None and fresh is not None and target_array is not None:
        first_trace_arrays = _artifact_trace(
            first.get("trace"), name="solver.first.trace", errors=errors
        )
        duplicate_trace_arrays = _artifact_trace(
            duplicate.get("trace"), name="solver.duplicate.trace", errors=errors
        )
    calls = value.get("policy_calls")
    if not isinstance(calls, Mapping):
        errors.append("policy_calls must be an object")
        calls = {}
    elif set(calls) != set(POLICY_CALL_SEQUENCE):
        errors.append("policy_calls do not equal the exact registered canary call set")
    if isinstance(provenance, Mapping):
        if provenance.get("planned_policy_call_sequence") != list(POLICY_CALL_SEQUENCE):
            errors.append("planned policy call sequence changed")
        if provenance.get("maximum_policy_call_count") != len(POLICY_CALL_SEQUENCE):
            errors.append("maximum policy call count changed")
    if first is not None and calls.get("inverse_flow_teacher_first") != first:
        errors.append("policy_calls teacher-first record differs from solver.first")
    if duplicate is not None and calls.get("inverse_flow_teacher_duplicate") != duplicate:
        errors.append("policy_calls teacher-duplicate record differs from solver.duplicate")
    top_canonical = value.get("canonical_replay")
    if isinstance(top_canonical, Mapping) and calls.get("canonical_schedule_replay") != top_canonical:
        errors.append("policy_calls canonical replay differs from top-level canonical replay")

    def action_from_call(name: str) -> Optional[np.ndarray]:
        item = calls.get(name)
        if not isinstance(item, Mapping):
            errors.append(f"policy_calls.{name} is missing")
            return None
        return _artifact_array(
            item.get("actions"),
            name=f"policy_calls.{name}.actions",
            shape=(10, 7),
            errors=errors,
        )

    eager_before_actions = action_from_call("eager_normalized_before")
    eager_after_actions = action_from_call("eager_normalized_after")
    source_before_actions = action_from_call("eager_source_pairing_before")
    source_after_actions = action_from_call("eager_source_pairing_after")
    compiled_before_actions = action_from_call("compiled_frozen_before")
    compiled_after_actions = action_from_call("compiled_frozen_after")
    first_actions = (
        _artifact_array(
            first.get("actions"), name="solver.first.actions", shape=(10, 7), errors=errors
        )
        if first is not None
        else None
    )
    duplicate_actions = (
        _artifact_array(
            duplicate.get("actions"), name="solver.duplicate.actions", shape=(10, 7), errors=errors
        )
        if duplicate is not None
        else None
    )

    # Independently reconstruct the physical teacher target.  The stored
    # diagnostics are descriptive only: the trust path binds the direct
    # witness to the immutable R02 file and recomputes all four tolerances.
    if (
        isinstance(target, Mapping)
        and isinstance(raw_r02, Mapping)
        and first_trace_arrays is not None
        and realized_target_delta is not None
        and eager_before_actions is not None
    ):
        raw_direct = _artifact_array(
            _nested_mapping_value(
                raw_r02, "arms", "direct_witness", "executed_actions"
            ),
            name="immutable_r02.arms.direct_witness.executed_actions",
            shape=(5, 7),
            errors=errors,
        )
        recorded_direct = _artifact_array(
            target.get("direct_witness_physical_target"),
            name="target.direct_witness_physical_target",
            shape=(5, 7),
            errors=errors,
        )
        recorded_constructed = _artifact_array(
            target.get("constructed_target_physical_first_five"),
            name="target.constructed_target_physical_first_five",
            shape=(5, 7),
            errors=errors,
        )
        try:
            scale = np.asarray(first_trace_arrays["model_to_physical_scale"])
            if scale.shape != (10, 32) or scale.dtype != np.dtype(np.float32):
                raise ValueError("first teacher checkpoint scale is not float32[10,32]")
            constructed = np.asarray(eager_before_actions, dtype=np.float64).copy()
            constructed[:5, :3] += np.asarray(
                realized_target_delta[:5, :3], dtype=np.float64
            ) * np.asarray(scale[:5, :3], dtype=np.float64)
            if raw_direct is not None:
                difference = constructed[:5, :7] - np.asarray(raw_direct, dtype=np.float64)
                xyz_max = float(np.max(np.abs(difference[:, :3])))
                xyz_rms = float(np.sqrt(np.mean(np.square(difference[:, :3]))))
                full_max = float(np.max(np.abs(difference)))
                full_rms = float(np.sqrt(np.mean(np.square(difference))))
                within = bool(
                    xyz_max <= 0.010
                    and xyz_rms <= 0.005
                    and full_max <= 0.050
                    and full_rms <= 0.015
                )
                diagnostics = target.get("direct_witness_target_diagnostics")
                stored_diagnostics_exact = bool(
                    isinstance(diagnostics, Mapping)
                    and diagnostics.get("xyz_max_abs") == xyz_max
                    and diagnostics.get("xyz_rms") == xyz_rms
                    and diagnostics.get("full_max_abs") == full_max
                    and diagnostics.get("full_rms") == full_rms
                    and diagnostics.get("within_registered_apparatus_tolerances") is within
                )
                recomputed_direct_target = bool(
                    recorded_direct is not None
                    and recorded_constructed is not None
                    and _array_exact(recorded_direct, raw_direct)
                    and _array_exact(recorded_constructed, constructed[:5, :7])
                    and stored_diagnostics_exact
                    and within
                )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"cannot recompute direct-witness target semantics: {error}")

    if isinstance(pairing, Mapping):
        pair_checks: list[bool] = []
        for name in ("observation", "branch_snapshot"):
            record = pairing.get(name)
            if not isinstance(record, Mapping):
                errors.append(f"pairing.{name} is missing")
                continue
            source_value = record.get("source")
            current_value = record.get("current")
            raw_key = "policy_observation" if name == "observation" else "branch_snapshot"
            raw_value = (
                raw_source_pairing.get(raw_key)
                if isinstance(raw_source_pairing, Mapping)
                else None
            )
            source_hash = content_hash(source_value)
            current_hash = content_hash(current_value)
            exact = source_hash == current_hash
            pair_checks.append(
                exact
                and raw_source_pairing is not None
                and source_value == raw_value
                and current_value == raw_value
                and record.get("source_sha256") == source_hash
                and record.get("current_sha256") == current_hash
            )
        raw_source_actions: Optional[np.ndarray] = None
        if isinstance(raw_source_pairing, Mapping):
            raw_source_actions = _artifact_array(
                raw_source_pairing.get("eager_actions"),
                name="immutable_r02.pairing.eager_actions",
                shape=(10, 7),
                errors=errors,
            )
        for name in ("source_actions_before", "source_actions_after"):
            record = pairing.get(name)
            if not isinstance(record, Mapping):
                errors.append(f"pairing.{name} is missing")
                continue
            source_action = _artifact_array(
                record.get("source_float64"),
                name=f"pairing.{name}.source_float64",
                shape=(10, 7),
                errors=errors,
            )
            current_action = _artifact_array(
                record.get("current_canonical_float64"),
                name=f"pairing.{name}.current_canonical_float64",
                shape=(10, 7),
                errors=errors,
            )
            pair_checks.append(
                source_action is not None
                and current_action is not None
                and raw_source_actions is not None
                and _array_exact(source_action, raw_source_actions)
                and _array_exact(source_action, current_action)
            )
        raw_source_trace_sha: Optional[str] = None
        raw_source_trace: Optional[Mapping[str, np.ndarray]] = None
        if isinstance(raw_source_pairing, Mapping):
            raw_trace_record = raw_source_pairing.get("eager_trace")
            if isinstance(raw_trace_record, Mapping) and _is_sha256(raw_trace_record.get("sha256")):
                raw_source_trace_sha = str(raw_trace_record["sha256"])
                raw_source_trace = _artifact_trace(
                    raw_trace_record,
                    name="immutable_r02.pairing.eager_trace",
                    errors=errors,
                )
            else:
                errors.append("immutable R02 eager trace digest is missing")
        for call_name, call_actions in (
            ("eager_source_pairing_before", source_before_actions),
            ("eager_source_pairing_after", source_after_actions),
        ):
            call = calls.get(call_name)
            call_trace = (
                _artifact_trace(
                    call.get("trace"), name=f"policy_calls.{call_name}.trace", errors=errors
                )
                if isinstance(call, Mapping)
                else None
            )
            pair_checks.append(
                raw_source_actions is not None
                and call_actions is not None
                and _array_exact(
                    np.asarray(call_actions, dtype=raw_source_actions.dtype), raw_source_actions
                )
                and raw_source_trace is not None
                and call_trace is not None
                and _trace_exact(raw_source_trace, call_trace)
            )
        for name in ("source_trace_before_diagnostics", "source_trace_after_diagnostics"):
            record = pairing.get(name)
            pair_checks.append(
                isinstance(record, Mapping)
                and raw_source_trace_sha is not None
                and record.get("exact_native_leaf_pairing") is True
                and record.get("source_trace_record_sha256") == raw_source_trace_sha
                and record.get("fresh_trace_record_sha256") == raw_source_trace_sha
            )
        recomputed_pairing = bool(pair_checks and all(pair_checks))

    teacher_common: list[bool] = []
    teacher_converged_checks: list[bool] = []
    teacher_nonconverged_checks: list[bool] = []
    raw_first_converged: Optional[bool] = None
    if (
        first_trace_arrays is not None
        and duplicate_trace_arrays is not None
        and fresh is not None
        and target_array is not None
    ):
        source_budget = np.asarray(target.get("source_budget_float32"), dtype=np.float32)
        for label, trace in (("first", first_trace_arrays), ("duplicate", duplicate_trace_arrays)):
            summary = first if label == "first" else duplicate
            assert summary is not None
            recurrence_ok = not validate_flow_recurrence(trace)
            try:
                status_code = int(_scalar(trace["solver_status"], name=f"{label}.solver_status"))
                converged_flag = bool(_scalar(trace["solver_converged"], name=f"{label}.solver_converged"))
                fields = bool(_scalar(trace["solver_fields_available"], name=f"{label}.fields"))
                nonfinite = bool(_scalar(trace["solver_nonfinite"], name=f"{label}.nonfinite"))
                iterations = int(
                    _scalar(trace["solver_iterations"], name=f"{label}.solver_iterations")
                )
                if label == "first":
                    raw_first_converged = converged_flag
                summary_checks = {
                    "status_code": summary.get("status_code") == status_code,
                    "converged": summary.get("converged") is converged_flag,
                    "iterations": summary.get("iterations") == iterations,
                    "fields_available": summary.get("fields_available") is fields,
                    "nonfinite": summary.get("nonfinite") is nonfinite,
                }
                for summary_key, summary_ok in summary_checks.items():
                    if not summary_ok:
                        errors.append(
                            f"solver.{label}.{summary_key} differs from its raw trace"
                        )
                common = bool(
                    recurrence_ok
                    and int(_scalar(trace["control_source"], name="control_source")) == 0
                    and bool(_scalar(trace["solver_baseline_valid"], name="baseline_valid"))
                    and bool(_scalar(trace["solver_schedule_valid"], name="schedule_valid"))
                    and bool(_scalar(trace["target_pairing_checked"], name="target_pairing_checked"))
                    and bool(_scalar(trace["target_pairing_exact"], name="target_pairing_exact"))
                    and bool(_scalar(trace["parameter_requires_grad_restored"], name="parameter_restore"))
                    and bool(_scalar(trace["parameter_grads_none_before"], name="grads_before"))
                    and bool(_scalar(trace["parameter_grads_none_after"], name="grads_after"))
                    and bool(_scalar(trace["parameter_grad_check_performed"], name="grad_check"))
                    and bool(_scalar(trace["cuda_memory_available"], name="cuda_memory"))
                    and iterations == 128
                    and _array_exact(trace["inverse_target"], target_array)
                    and provenance_noise is not None
                    and _array_exact(trace["initial_noise"], provenance_noise)
                    and _array_exact(trace["solver_baseline_final"], fresh)
                    and np.asarray(_scalar(trace["source_control_budget"], name="source_budget"), dtype=np.float32).tobytes()
                    == source_budget.tobytes()
                    and np.asarray(_scalar(trace["schedule_budget"], name="schedule_budget"), dtype=np.float32).tobytes()
                    == source_budget.tobytes()
                )
                schedule_diag, schedule_errors = _schedule_diagnostics(
                    trace["solver_schedule"], source_budget_float32=source_budget
                )
                del schedule_diag
                model_error_exact = _array_exact(
                    trace["solver_model_error"],
                    np.asarray(trace["solver_internal_replay_final"] - target_array, dtype=np.float32),
                )
                independently_realized = np.zeros((10, 32), dtype=np.float32)
                independently_realized[:5, :3] = np.asarray(
                    target_array - trace["solver_baseline_final"], dtype=np.float32
                )[:5, :3]
                realized_elements_exact = bool(
                    realized is not None and _array_exact(independently_realized, realized)
                )
                scale = np.asarray(trace["model_to_physical_scale"])
                scale_exact = bool(
                    scale.dtype == np.dtype(np.float32)
                    and scale.shape == (10, 32)
                    and np.all(scale > 0)
                    and _array_exact(
                        scale[:, :3],
                        np.broadcast_to(np.asarray(REGISTERED_XYZ_SCALE, dtype=np.float32), (10, 3)).copy(),
                    )
                    and _array_exact(scale[:, 7:], np.ones((10, 25), dtype=np.float32))
                )
                recomputed_scale = scale_exact if label == "first" else recomputed_scale and scale_exact
                internal_final = np.asarray(trace["solver_internal_replay_final"])
                returned_actions = first_actions if label == "first" else duplicate_actions
                physical_binding_exact = bool(
                    returned_actions is not None
                    and "final_normalized_physical" in trace
                    and _array_exact(trace["final_normalized_physical"], returned_actions)
                )
                fidelity_ok = bool(
                    _fidelity_diagnostics(internal_final, target_array, scale)["passed"]
                )
                expected_summary_invariants = {
                    "control_source_is_teacher": int(
                        _scalar(trace["control_source"], name="control_source")
                    )
                    == 0,
                    "control_valid": bool(
                        _scalar(trace["control_valid"], name="control_valid")
                    ),
                    "schedule_applied": bool(
                        _scalar(trace["schedule_applied"], name="schedule_applied")
                    ),
                    "baseline_valid": bool(
                        _scalar(trace["solver_baseline_valid"], name="baseline_valid")
                    ),
                    "schedule_valid": bool(
                        _scalar(trace["solver_schedule_valid"], name="schedule_valid")
                    ),
                    "target_pairing_checked": bool(
                        _scalar(trace["target_pairing_checked"], name="target_pairing_checked")
                    ),
                    "target_pairing_exact": bool(
                        _scalar(trace["target_pairing_exact"], name="target_pairing_exact")
                    ),
                    "parameter_requires_grad_restored": bool(
                        _scalar(
                            trace["parameter_requires_grad_restored"],
                            name="parameter_requires_grad_restored",
                        )
                    ),
                    "parameter_grads_none_before": bool(
                        _scalar(
                            trace["parameter_grads_none_before"],
                            name="parameter_grads_none_before",
                        )
                    ),
                    "parameter_grads_none_after": bool(
                        _scalar(
                            trace["parameter_grads_none_after"],
                            name="parameter_grads_none_after",
                        )
                    ),
                    "parameter_grad_check_performed": bool(
                        _scalar(
                            trace["parameter_grad_check_performed"],
                            name="parameter_grad_check_performed",
                        )
                    ),
                    "dt_exact": np.asarray(
                        _scalar(trace["dt"], name="dt"), dtype=np.float32
                    ).tobytes()
                    == np.asarray(-0.1, dtype=np.float32).tobytes(),
                    "intervention_step_exact": int(
                        _scalar(trace["intervention_step"], name="intervention_step")
                    )
                    == 5,
                    "num_steps_exact": int(
                        _scalar(trace["num_steps"], name="num_steps")
                    )
                    == 10,
                    "fixed_128_updates_complete": iterations == 128,
                    "initial_noise_exact": provenance_noise is not None
                    and _array_exact(trace["initial_noise"], provenance_noise),
                    "recurrence_exact": recurrence_ok,
                    "schedule_constraints_passed": not schedule_errors,
                    "budget_binding_passed": bool(
                        np.asarray(
                            _scalar(trace["source_control_budget"], name="source_budget"),
                            dtype=np.float32,
                        ).tobytes()
                        == source_budget.tobytes()
                        and np.asarray(
                            _scalar(trace["schedule_budget"], name="schedule_budget"),
                            dtype=np.float32,
                        ).tobytes()
                        == source_budget.tobytes()
                        and realized_elements_exact
                        and model_error_exact
                        and _array_exact(trace["solver_baseline_final"], fresh)
                    ),
                    "fidelity_passed": fidelity_ok,
                    "final_physical_exact_returned_actions": physical_binding_exact,
                    "cuda_memory_available": bool(
                        _scalar(trace["cuda_memory_available"], name="cuda_memory_available")
                    ),
                }
                expected_summary_budget_checks = {
                    "source_control_budget_exact": np.asarray(
                        _scalar(trace["source_control_budget"], name="source_budget"),
                        dtype=np.float32,
                    ).tobytes()
                    == source_budget.tobytes(),
                    "schedule_budget_exact": np.asarray(
                        _scalar(trace["schedule_budget"], name="schedule_budget"),
                        dtype=np.float32,
                    ).tobytes()
                    == source_budget.tobytes(),
                    "realized_target_delta_elements_exact": realized_elements_exact,
                    "solver_model_error_exact": model_error_exact,
                    "solver_baseline_exact_fresh_target_pair": _array_exact(
                        trace["solver_baseline_final"], fresh
                    ),
                }
                if summary.get("invariants") != expected_summary_invariants:
                    errors.append(f"solver.{label}.invariants differ from its raw trace")
                if summary.get("budget_checks") != expected_summary_budget_checks:
                    errors.append(f"solver.{label}.budget_checks differ from its raw trace")
                summary_fidelity = summary.get("fidelity")
                if not isinstance(summary_fidelity, Mapping) or (
                    summary_fidelity.get("passed") is not fidelity_ok
                ):
                    errors.append(f"solver.{label}.fidelity differs from its raw trace")
                if fields:
                    summary_schedule = _artifact_array(
                        summary.get("schedule"),
                        name=f"solver.{label}.schedule",
                        shape=(10, 10, 32),
                        errors=errors,
                    )
                    if summary_schedule is None or not _array_exact(
                        summary_schedule, trace["solver_schedule"]
                    ):
                        errors.append(f"solver.{label}.schedule differs from its raw trace")
                elif summary.get("schedule") is not None:
                    errors.append(f"solver.{label}.schedule exists without raw solver fields")
                teacher_common.append(
                    common
                    and not schedule_errors
                    and model_error_exact
                    and realized_elements_exact
                    and scale_exact
                    and physical_binding_exact
                )
                teacher_converged_checks.append(
                    common
                    and not schedule_errors
                    and model_error_exact
                    and realized_elements_exact
                    and scale_exact
                    and physical_binding_exact
                    and status_code == 0
                    and converged_flag
                    and fields
                    and not nonfinite
                    and bool(_scalar(trace["control_valid"], name="control_valid"))
                    and bool(_scalar(trace["schedule_applied"], name="schedule_applied"))
                    and fidelity_ok
                    and _array_exact(trace["control_velocity_steps"], trace["solver_schedule"])
                    and _array_exact(trace["final_normalized"], internal_final)
                )
                teacher_nonconverged_checks.append(
                    common
                    and not schedule_errors
                    and model_error_exact
                    and realized_elements_exact
                    and scale_exact
                    and physical_binding_exact
                    and status_code == 3
                    and not converged_flag
                    and fields
                    and not nonfinite
                    and not bool(_scalar(trace["control_valid"], name="control_valid"))
                    and not bool(_scalar(trace["schedule_applied"], name="schedule_applied"))
                    and not fidelity_ok
                    and np.count_nonzero(trace["control_velocity_steps"]) == 0
                    and _array_exact(trace["final_normalized"], fresh)
                    and _array_exact(trace["canonical_replay_final"], fresh)
                    and returned_actions is not None
                    and eager_before_actions is not None
                    and _array_exact(returned_actions, eager_before_actions)
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"cannot recompute {label} teacher semantics: {error}")

        recomputed_determinism = bool(
            first_actions is not None
            and duplicate_actions is not None
            and _array_exact(first_actions, duplicate_actions)
            and _trace_exact(
                _deterministic_trace(first_trace_arrays),
                _deterministic_trace(duplicate_trace_arrays),
            )
        )

    zero_before = calls.get("zero_schedule_before")
    zero_after = calls.get("zero_schedule_after")
    zero_before_trace = (
        _artifact_trace(zero_before.get("trace"), name="zero_schedule_before.trace", errors=errors)
        if isinstance(zero_before, Mapping)
        else None
    )
    eager_before_record = calls.get("eager_normalized_before")
    eager_after_record = calls.get("eager_normalized_after")
    eager_before_trace = (
        _artifact_trace(
            eager_before_record.get("trace"), name="eager_normalized_before.trace", errors=errors
        )
        if isinstance(eager_before_record, Mapping)
        else None
    )
    eager_after_trace = (
        _artifact_trace(
            eager_after_record.get("trace"), name="eager_normalized_after.trace", errors=errors
        )
        if isinstance(eager_after_record, Mapping)
        else None
    )
    zero_after_trace = (
        _artifact_trace(zero_after.get("trace"), name="zero_schedule_after.trace", errors=errors)
        if isinstance(zero_after, Mapping)
        else None
    )
    zero_before_actions = action_from_call("zero_schedule_before")
    zero_after_actions = action_from_call("zero_schedule_after")
    if (
        zero_before_trace is not None
        and zero_after_trace is not None
        and eager_before_trace is not None
        and eager_after_trace is not None
        and fresh is not None
        and eager_before_actions is not None
        and eager_after_actions is not None
        and compiled_before_actions is not None
        and compiled_after_actions is not None
        and source_before_actions is not None
        and source_after_actions is not None
        and zero_before_actions is not None
        and zero_after_actions is not None
    ):
        recomputed_zero_frozen = bool(
            not validate_flow_recurrence(zero_before_trace)
            and not validate_flow_recurrence(zero_after_trace)
            and np.count_nonzero(zero_before_trace["control_velocity_steps"]) == 0
            and np.count_nonzero(zero_after_trace["control_velocity_steps"]) == 0
            and provenance_noise is not None
            and _array_exact(zero_before_trace["initial_noise"], provenance_noise)
            and _array_exact(zero_after_trace["initial_noise"], provenance_noise)
            and _array_exact(zero_before_trace["final_normalized"], fresh)
            and _array_exact(zero_after_trace["final_normalized"], fresh)
            and _array_exact(eager_before_trace["final_normalized"], fresh)
            and _array_exact(eager_after_trace["final_normalized"], fresh)
            and _trace_exact(eager_before_trace, eager_after_trace)
            and _array_exact(zero_before_trace["final_normalized_physical"], zero_before_actions)
            and _array_exact(zero_after_trace["final_normalized_physical"], zero_after_actions)
            and _trace_exact(zero_before_trace, zero_after_trace)
            and _array_exact(zero_before_actions, eager_before_actions)
            and _array_exact(zero_after_actions, eager_before_actions)
            and _array_exact(compiled_before_actions, eager_before_actions)
            and _array_exact(compiled_before_actions, source_before_actions)
            and _array_exact(compiled_after_actions, eager_after_actions)
            and _array_exact(compiled_after_actions, source_after_actions)
            and _array_exact(compiled_before_actions, compiled_after_actions)
            and _array_exact(eager_before_actions, eager_after_actions)
        )

    canonical = value.get("canonical_replay")
    if isinstance(canonical, Mapping) and first_trace_arrays is not None:
        if canonical.get("applicable") is True:
            canonical_trace = _artifact_trace(
                canonical.get("trace"), name="canonical_replay.trace", errors=errors
            )
            canonical_actions = _artifact_array(
                canonical.get("actions"),
                name="canonical_replay.actions",
                shape=(10, 7),
                errors=errors,
            )
            if canonical_trace is not None and canonical_actions is not None and first_actions is not None:
                recomputed_canonical = bool(
                    not validate_flow_recurrence(canonical_trace)
                    and provenance_noise is not None
                    and _array_exact(canonical_trace["initial_noise"], provenance_noise)
                    and _array_exact(
                        canonical_trace["control_velocity_steps"], first_trace_arrays["solver_schedule"]
                    )
                    and _array_exact(
                        canonical_trace["final_normalized"], first_trace_arrays["solver_internal_replay_final"]
                    )
                    and _array_exact(canonical_actions, first_actions)
                    and _array_exact(
                        canonical_trace["final_normalized_physical"], canonical_actions
                    )
                )

    recomputed_converged = bool(
        len(teacher_converged_checks) == 2
        and all(teacher_converged_checks)
        and recomputed_determinism
        and recomputed_zero_frozen
        and recomputed_pairing
        and recomputed_scale
        and recomputed_canonical
        and recomputed_direct_target
    )
    recomputed_clean_nonconvergence = bool(
        len(teacher_nonconverged_checks) == 2
        and all(teacher_nonconverged_checks)
        and recomputed_determinism
        and recomputed_zero_frozen
        and recomputed_pairing
        and recomputed_scale
        and recomputed_direct_target
        and isinstance(canonical, Mapping)
        and canonical.get("applicable") is False
    )
    determinism = value.get("determinism")
    if not isinstance(determinism, Mapping):
        errors.append("determinism record is missing")
    canonical = value.get("canonical_replay")
    if not isinstance(canonical, Mapping):
        errors.append("canonical replay record is missing")
    memory = value.get("memory")
    if not isinstance(memory, Mapping):
        errors.append("memory record is missing")
    else:
        for key in (
            "host_cgroup_peak_bytes",
            "gpu_process_peak_allocated_bytes",
            "gpu_process_peak_reserved_bytes",
            "gpu_nvidia_smi_samples",
        ):
            if not isinstance(memory.get(key), int) or int(memory[key]) <= 0:
                errors.append(f"memory.{key} must be a positive measured integer")
        if isinstance(memory.get("host_cgroup_peak_bytes"), int) and int(
            memory["host_cgroup_peak_bytes"]
        ) > 65536 * 1024 * 1024:
            errors.append("host cgroup peak exceeds the registered 64 GiB allocation")
        if first_trace_arrays is not None and duplicate_trace_arrays is not None:
            try:
                if not (
                    bool(_scalar(first_trace_arrays["cuda_memory_available"], name="first CUDA"))
                    and bool(_scalar(duplicate_trace_arrays["cuda_memory_available"], name="duplicate CUDA"))
                ):
                    errors.append("teacher CUDA process telemetry is unavailable")
                expected_allocated = max(
                    int(_scalar(first_trace_arrays["cuda_process_peak_allocated_bytes"], name="first peak")),
                    int(_scalar(duplicate_trace_arrays["cuda_process_peak_allocated_bytes"], name="duplicate peak")),
                )
                expected_reserved = max(
                    int(_scalar(first_trace_arrays["cuda_process_peak_reserved_bytes"], name="first reserved")),
                    int(_scalar(duplicate_trace_arrays["cuda_process_peak_reserved_bytes"], name="duplicate reserved")),
                )
                if memory.get("gpu_process_peak_allocated_bytes") != expected_allocated:
                    errors.append("GPU allocated peak differs from raw sampler trace")
                if memory.get("gpu_process_peak_reserved_bytes") != expected_reserved:
                    errors.append("GPU reserved peak differs from raw sampler trace")
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"cannot recompute CUDA memory from raw trace: {error}")
        if isinstance(provenance, Mapping) and memory.get("allocation_gpu_uuid") != provenance.get(
            "allocation_gpu_uuid"
        ):
            errors.append("memory GPU UUID differs from allocation provenance")
        if gpu_samples_path is not None:
            try:
                sample_count, sample_uuid, compute_peak, device_peak, sample_sha = _read_gpu_samples(
                    gpu_samples_path
                )
                if allocation_case_dir is None or gpu_samples_path.resolve(strict=True).parent != allocation_case_dir:
                    errors.append("GPU and focused-test evidence do not share the immutable case directory")
                if memory.get("gpu_nvidia_smi_samples") != sample_count:
                    errors.append("GPU sample count differs from live shared storage")
                if memory.get("allocation_gpu_uuid") != sample_uuid:
                    errors.append("GPU sample UUID differs from live shared storage")
                if memory.get("gpu_nvidia_smi_peak_compute_mib") != compute_peak:
                    errors.append("GPU compute peak differs from live shared storage")
                if memory.get("gpu_nvidia_smi_peak_device_mib") != device_peak:
                    errors.append("GPU device peak differs from live shared storage")
                if memory.get("gpu_nvidia_smi_samples_sha256") != sample_sha:
                    errors.append("GPU sample digest differs from live shared storage")
            except (OSError, ValueError) as error:
                errors.append(f"GPU sample evidence is invalid: {error}")
    apparatus = value.get("apparatus")
    expected_status = (
        "completed_converged"
        if recomputed_converged
        else "completed_nonconverged"
        if recomputed_clean_nonconvergence
        else "completed_apparatus_failure"
    )
    if status != expected_status:
        errors.append(
            f"recorded status {status!r} differs from independently recomputed {expected_status!r}"
        )
    if not isinstance(apparatus, Mapping):
        errors.append("apparatus record is missing")
    else:
        if apparatus.get("memory_pending") is not False or apparatus.get("memory_passed") is not True:
            errors.append("memory finalization did not pass")
        passed = apparatus.get("passed") is True
        if passed != recomputed_converged:
            errors.append("apparatus.passed differs from independent trace recomputation")
        if apparatus.get("clean_nonconvergence") is not recomputed_clean_nonconvergence:
            errors.append("clean_nonconvergence differs from independent trace recomputation")
    outcome = value.get("outcome")
    if not isinstance(outcome, Mapping) or not (
        outcome.get("interpretation")
        == "apparatus_only_no_teacher_efficacy_or_safety_claim"
        and raw_first_converged is not None
        and outcome.get("teacher_converged") is raw_first_converged
        and outcome.get("simulator_teacher_outcome") is None
        and outcome.get("student_training_authorized") is False
        and outcome.get("next_gate_authorized") is False
    ):
        errors.append("canary outcome overclaims scientific authorization")
    return errors


def _read_gpu_samples(path: str | Path) -> tuple[int, str, float, float, str]:
    sample_path = Path(path)
    digest = file_sha256(sample_path)
    compute_values: list[float] = []
    device_values: list[float] = []
    gpu_uuid: Optional[str] = None
    for line_number, line in enumerate(sample_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("timestamp_ns,"):
            continue
        fields = line.split(",")
        if len(fields) != 4:
            raise ValueError(f"invalid GPU sample line {line_number}")
        try:
            timestamp = int(fields[0])
            observed_uuid = fields[1]
            compute = float(fields[2])
            device = float(fields[3])
        except ValueError as error:
            raise ValueError(f"invalid GPU sample line {line_number}") from error
        if timestamp <= 0 or not math.isfinite(compute) or not math.isfinite(device) or compute < 0 or device < 0:
            raise ValueError(f"invalid GPU sample values on line {line_number}")
        if not observed_uuid.startswith("GPU-"):
            raise ValueError(f"invalid GPU UUID on line {line_number}")
        if gpu_uuid is None:
            gpu_uuid = observed_uuid
        elif observed_uuid != gpu_uuid:
            raise ValueError("GPU monitor mixed allocation device identities")
        compute_values.append(compute)
        device_values.append(device)
    if not compute_values:
        raise ValueError("GPU monitor recorded no usable allocation samples")
    assert gpu_uuid is not None
    return len(compute_values), gpu_uuid, max(compute_values), max(device_values), digest


def finalize_r05a_canary(
    payload_path: str | Path,
    output_path: str | Path,
    *,
    host_cgroup_peak_bytes: int,
    gpu_samples_path: str | Path,
    allocation_tests_log: str | Path,
    allocation_tests_exit_code: int,
) -> Path:
    """Add measured allocation evidence, validate, then atomically publish."""

    payload = load_json(payload_path)
    if not isinstance(payload, Mapping) or payload.get("payload_type") != PAYLOAD_TYPE:
        raise ValueError("R05A canary payload identity is invalid")
    result = copy.deepcopy(payload.get("result_without_memory"))
    if not isinstance(result, dict) or "memory" in result:
        raise ValueError("R05A canary payload is not awaiting memory finalization")
    if not isinstance(host_cgroup_peak_bytes, int) or host_cgroup_peak_bytes <= 0:
        raise ValueError("host cgroup peak must be an actual positive byte count")
    if host_cgroup_peak_bytes > 65536 * 1024 * 1024:
        raise ValueError("host cgroup peak exceeds the registered 64 GiB request")
    if allocation_tests_exit_code != 0:
        raise ValueError("allocation dependency-backed focused tests did not pass")
    test_log = Path(allocation_tests_log)
    if test_log.stat().st_size <= 0:
        raise ValueError("allocation focused-test log is empty")
    test_log_sha, observed_test_counts = _parse_allocation_test_log(test_log)
    count, gpu_uuid, compute_peak, device_peak, samples_sha = _read_gpu_samples(gpu_samples_path)
    if result.get("provenance", {}).get("allocation_gpu_uuid") != gpu_uuid:
        raise ValueError("GPU samples differ from the allocation-bound GPU UUID")
    first_trace = _trace_from_record(
        result["solver"]["first"]["trace"], name="solver.first.trace"
    )
    duplicate_trace = _trace_from_record(
        result["solver"]["duplicate"]["trace"], name="solver.duplicate.trace"
    )
    if not (
        bool(_scalar(first_trace["cuda_memory_available"], name="first CUDA memory"))
        and bool(_scalar(duplicate_trace["cuda_memory_available"], name="duplicate CUDA memory"))
    ):
        raise ValueError("sampler did not expose CUDA process memory")
    process_allocated = max(
        int(_scalar(first_trace["cuda_process_peak_allocated_bytes"], name="first allocated peak")),
        int(_scalar(duplicate_trace["cuda_process_peak_allocated_bytes"], name="duplicate allocated peak")),
    )
    process_reserved = max(
        int(_scalar(first_trace["cuda_process_peak_reserved_bytes"], name="first reserved peak")),
        int(_scalar(duplicate_trace["cuda_process_peak_reserved_bytes"], name="duplicate reserved peak")),
    )
    if process_allocated <= 0 or process_reserved <= 0:
        raise ValueError("sampler CUDA process peaks are not positive")
    result["memory"] = {
        "requested_host_memory_mib": 65536,
        "host_cgroup_peak_bytes": host_cgroup_peak_bytes,
        "allocation_gpu_uuid": gpu_uuid,
        "gpu_process_peak_allocated_bytes": process_allocated,
        "gpu_process_peak_reserved_bytes": process_reserved,
        "gpu_nvidia_smi_peak_compute_mib": compute_peak,
        "gpu_nvidia_smi_peak_device_mib": device_peak,
        "gpu_nvidia_smi_samples": count,
        "gpu_nvidia_smi_samples_sha256": samples_sha,
        "measurement_scope": (
            "host=live_slurm_cgroup_peak; cuda=PyTorch_process_lifetime_upper_bound_no_reset; "
            "nvidia_smi=allocation_side_periodic_samples_not_continuous_peak"
        ),
    }
    # The schema deliberately rejects unrecorded telemetry; add the monitor
    # digest to provenance rather than weakening the memory record contract.
    result["provenance"]["allocation_runtime_tests"] = {
        "command_role": "dependency_backed_focused_tests_before_policy_server",
        "exit_code": allocation_tests_exit_code,
        "log_path": str(test_log),
        "log_sha256": test_log_sha,
        "expected_counts": dict(ALLOCATION_TEST_COUNTS),
        "observed_counts": dict(observed_test_counts),
        "zero_skips": True,
    }
    result["provenance"]["gpu_samples_path"] = str(Path(gpu_samples_path))
    result["apparatus"]["memory_pending"] = False
    result["apparatus"]["memory_passed"] = True
    result["apparatus"]["passed"] = bool(
        result["apparatus"].get("passed_before_memory_finalization") is True
    )
    errors = validate_r05a_canary_result(result) + validate_r05a_canary_schema(
        result, require_jsonschema=True
    )
    if errors:
        raise RuntimeError("refusing invalid R05A canary result: " + "; ".join(errors))
    destination = Path(output_path)
    atomic_write_json(destination, result)
    return destination


__all__ = [
    "R05ACanaryConfig",
    "finalize_r05a_canary",
    "r05a_canary_config_from_mapping",
    "r05a_canary_config_hash",
    "run_r05a_canary",
    "validate_flow_recurrence",
    "validate_r05a_canary_result",
    "validate_r05a_canary_schema",
]
