"""Allocation-only TRL-00A optimizer-free reference-trajectory canary.

The transaction restores the exact R02/AF-00A source, constructs the frozen
six-knot reference trajectory, and compares a same-budget online lift with its
unprojected diagnostic counterpart.  Every generated finite field is replayed
through the ordinary ``residual_schedule`` route.  The GPU writes raw evidence
only; it never publishes ``results.json`` and never executes a returned action
in the simulator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any, Mapping, Optional

import numpy as np

from crfs_harness.artifacts import atomic_write_json, file_sha256

from .progress_calibration import _target_contact_at_branch
from .r02_runner import (
    _array_record,
    _json_compatible,
    _observation_fingerprint,
    _trace_record,
    _validate_array_record,
)
from .r03a_runner import _trace_from_record, _trace_pairing_diagnostics
from .r05a_actual_forward_search import (
    ACTION_SHAPE,
    COMPACT_SHAPE,
    DT_FLOAT32,
    ActualForwardSearchError,
    objective_and_gates,
    validate_executed_constraints,
)
from .r05a_canary import (
    CASE_ID,
    GROUP_ID,
    MANIFEST_SHA256,
    REGISTERED_XYZ_SCALE,
    SOURCE_HOST,
    R05ACanaryPolicyError,
    R05ACanarySourceError,
    _array_exact,
    _deterministic_trace,
    _finite_array,
    _frozen_controls,
    _load_source_r02,
    _replay_summary,
    _schedule_controls,
    _source_delta,
    _trace_exact,
    validate_flow_recurrence,
)
from .reach_progress import TARGET_OBJECT_NAME, capture_reach_snapshot
from .runner import SafeLiberoCase, policy_observation


PAYLOAD_TYPE = "r05a_reference_trajectory_lift_raw_payload"
LEDGER_TYPE = "r05a_reference_trajectory_lift_request_ledger"
TENSOR_FILENAME = "trl00a-tensors.npz"
LEDGER_FILENAME = "request-ledger.json"
PAYLOAD_FILENAME = "trl00a-raw-payload.json"
RESULTS_FILENAME = "results.json"
EXACT_FINITE_REQUESTS = 18
REFERENCE_SHAPE = (6, 10, 32)
NORMALIZED_SHAPE = (10, 32)
FLOW_SHAPE = (10, 10, 32)
TERMINAL_RESPONSE_KEY = "__crfs_terminal__"
RAW_TERMINAL_TYPE = "RAW_FLOAT32_UNREPRESENTABLE"
AF_RESULTS_SHA256 = "507f25bc381b7bfeaa30abe3a7df970681f3b175f39221034c6e768e72dae914"
AF_TENSOR_SHA256 = "ada53973653ba7c21dab33cdbc4b4498c7b2b1840f2fc284bd3d3c76baf1d383"
SOURCE_BUDGET_FLOAT32 = np.float32(3.6398398876190186)

FINITE_REQUEST_PHASES = (
    "compiled_frozen_pre",
    "eager_source_trace_pre",
    "eager_normalized_final_pre",
    "zero_schedule_pre",
    "arm_a_equal_split",
    "arm_a_equal_split",
    "budgeted_lift_generation",
    "budgeted_lift_generation",
    "budgeted_lift_replay",
    "budgeted_lift_replay",
    "raw_lift_generation",
    "raw_lift_generation",
    "raw_lift_replay",
    "raw_lift_replay",
    "zero_schedule_post",
    "eager_normalized_final_post",
    "eager_source_trace_post",
    "compiled_frozen_post",
)

RAW_TERMINAL_LEAVES = frozenset(
    {
        "state",
        "base_velocity",
        "reference_next",
        "uncontrolled_next",
        "raw_increment",
        "requested_increment",
        "requested_velocity",
        "executed_increment",
        "total_velocity",
        "next_state",
    }
)

SHARED_RECURRENCE_LEAVES = (
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
    "canonical_replay_final",
    "final_normalized",
    "final_normalized_physical",
)


class ReferenceTrajectoryLiftCanaryError(RuntimeError):
    """The TRL-00A source, policy transaction, or raw artifact is invalid."""


def _bytes_digest(value: Any) -> Mapping[str, Any]:
    raw = np.asarray(value)
    array = np.ascontiguousarray(raw) if raw.ndim else raw.copy()
    if array.dtype == np.dtype("O"):
        raise ReferenceTrajectoryLiftCanaryError("cannot digest an object array")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write one immutable, numeric-only, no-pickle NumPy archive."""

    if path.exists():
        raise ReferenceTrajectoryLiftCanaryError(
            f"immutable tensor artifact already exists: {path}"
        )
    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        raw = np.asarray(value)
        array = np.ascontiguousarray(raw) if raw.ndim else raw.copy()
        if array.dtype == np.dtype("O"):
            raise ReferenceTrajectoryLiftCanaryError(
                f"tensor artifact leaf {name} has object dtype"
            )
        if np.issubdtype(array.dtype, np.number) and not bool(np.isfinite(array).all()):
            raise ReferenceTrajectoryLiftCanaryError(
                f"tensor artifact leaf {name} is nonfinite"
            )
        normalized[name] = array
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **normalized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _scalar(value: Any, *, name: str, dtype: Any | None = None) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ReferenceTrajectoryLiftCanaryError(f"{name} must be scalar")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise ReferenceTrajectoryLiftCanaryError(
            f"{name} must preserve {np.dtype(dtype)}, got {array.dtype}"
        )
    if np.issubdtype(array.dtype, np.number) and not bool(np.isfinite(array).item()):
        raise ReferenceTrajectoryLiftCanaryError(f"{name} must be finite")
    return array.item()


def _validate_config(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != "1.0" or value.get("experiment_identity") != "TRL-00A":
        raise ReferenceTrajectoryLiftCanaryError("TRL-00A config identity changed")
    if value.get("ready_to_run") is not True or value.get("blocked_on") != []:
        raise ReferenceTrajectoryLiftCanaryError("TRL-00A config is not execution-released")
    release = value.get("execution_release")
    if not isinstance(release, Mapping) or not isinstance(release.get("run_id"), str):
        raise ReferenceTrajectoryLiftCanaryError("TRL-00A has no exact execution release")
    target = value.get("target_contract", {})
    flow = value.get("flow_contract", {})
    lift = value.get("reference_lift_contract", {})
    ledger = value.get("request_ledger", {})
    artifacts = value.get("artifact_contract", {})
    boundary = value.get("execution_boundary", {})
    preregistration = value.get("preregistration", {})
    source = value.get("frozen_source_bindings", {})
    af = source.get("af00a_source_run", {})
    checks = {
        "source budget": np.asarray(
            np.float32(target.get("source_budget_float32"))
        ).tobytes()
        == SOURCE_BUDGET_FLOAT32.tobytes(),
        "dt": np.asarray(np.float32(flow.get("dt_float32"))).tobytes()
        == DT_FLOAT32.tobytes(),
        "ten steps": flow.get("sampler_steps") == 10,
        "step five": flow.get("intervention_step") == 5,
        "reference shape": flow.get("reference_active_state_shape") == [6, 10, 32],
        "delta shape": flow.get("delta_shape") == [10, 32],
        "mode": lift.get("mode") == "reference_trajectory_lift",
        "raw terminal not accepted": lift.get("raw_guard_terminal") is None,
        "finite ledger": ledger.get("complete_finite_exact_policy_request_count")
        == EXACT_FINITE_REQUESTS,
        "payload name": artifacts.get("raw_payload_name") == PAYLOAD_FILENAME,
        "tensor name": artifacts.get("tensor_archive_name") == TENSOR_FILENAME,
        "ledger name": artifacts.get("request_ledger_name") == LEDGER_FILENAME,
        "GPU no results": artifacts.get("gpu_may_publish_results_json") is False,
        "H100 submission authorized": preregistration.get(
            "h100_submission_authorized"
        )
        is True,
        "AF result hash": af.get("results_json_sha256") == AF_RESULTS_SHA256,
        "AF tensor hash": af.get("tensor_archive_sha256") == AF_TENSOR_SHA256,
        "AF is not control": af.get("use_as_control_input") is False,
        "no simulator policy steps": boundary.get("policy_generated_action_steps_executed") == 0,
        "no simulator teacher steps": boundary.get("teacher_generated_action_steps_executed") == 0,
        "no training": boundary.get("training") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ReferenceTrajectoryLiftCanaryError(
            "TRL-00A frozen config changed: " + ", ".join(failed)
        )


def _validate_server_timing(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) not in (
        {"infer_ms"},
        {"infer_ms", "prev_total_ms"},
    ):
        raise ReferenceTrajectoryLiftCanaryError(
            "terminal server_timing keys changed"
        )
    for name, number in value.items():
        if type(number) is not float or not math.isfinite(number) or number < 0.0:
            raise ReferenceTrajectoryLiftCanaryError(
                f"terminal server_timing {name} must be one finite nonnegative float"
            )


def _normalize_raw_terminal(reply: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the only typed terminal and remove only WebSocket timing."""

    outer = set(reply)
    if outer == {TERMINAL_RESPONSE_KEY, "server_timing"}:
        _validate_server_timing(reply.get("server_timing"))
    elif outer != {TERMINAL_RESPONSE_KEY}:
        raise ReferenceTrajectoryLiftCanaryError(
            "raw terminal transport contains an unsupported outer key"
        )
    terminal = reply.get(TERMINAL_RESPONSE_KEY)
    if not isinstance(terminal, Mapping) or set(terminal) != {"type", "step", "leaf"}:
        raise ReferenceTrajectoryLiftCanaryError("raw terminal payload keys changed")
    if terminal.get("type") != RAW_TERMINAL_TYPE:
        raise ReferenceTrajectoryLiftCanaryError("raw terminal type changed")
    step = terminal.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < 10:
        raise ReferenceTrajectoryLiftCanaryError("raw terminal step is invalid")
    leaf = terminal.get("leaf")
    if leaf not in RAW_TERMINAL_LEAVES:
        raise ReferenceTrajectoryLiftCanaryError("raw terminal leaf is invalid")
    return {"type": RAW_TERMINAL_TYPE, "step": int(step), "leaf": str(leaf)}


def _alpha_float32() -> np.ndarray:
    alpha = np.zeros((6,), dtype=np.float32)
    for index in range(1, 6):
        alpha[index] = np.asarray(
            np.float32(index) / np.float32(5), dtype=np.float32
        )
    if bool(np.signbit(alpha[0])):
        raise ReferenceTrajectoryLiftCanaryError("alpha_5 lost positive zero")
    return alpha


def _zero_reference_states(trace: Mapping[str, Any]) -> np.ndarray:
    """Recover xbar_5..xbar_10 from one exact ordinary zero recurrence."""

    x_t = _finite_array(trace.get("x_t_steps"), name="zero x_t", shape=FLOW_SHAPE)
    x_next = _finite_array(
        trace.get("x_next_steps"), name="zero x_next", shape=FLOW_SHAPE
    )
    noise = _finite_array(
        trace.get("initial_noise"), name="zero initial noise", shape=NORMALIZED_SHAPE
    )
    final = _finite_array(
        trace.get("final_normalized"), name="zero final", shape=NORMALIZED_SHAPE
    )
    for name, value in (("x_t", x_t), ("x_next", x_next), ("noise", noise), ("final", final)):
        if value.dtype != np.dtype(np.float32):
            raise ReferenceTrajectoryLiftCanaryError(
                f"zero reference {name} must preserve float32"
            )
    if not _array_exact(x_t[0], noise):
        raise ReferenceTrajectoryLiftCanaryError("zero recurrence does not start at paired noise")
    if not _array_exact(x_t[1:], x_next[:-1]) or not _array_exact(x_next[-1], final):
        raise ReferenceTrajectoryLiftCanaryError("zero recurrence states are not contiguous")
    return np.ascontiguousarray(np.concatenate((x_t[5:6], x_next[5:]), axis=0))


def construct_reference_trajectory(
    zero_trace: Mapping[str, Any], delta_float32: np.ndarray, target_float32: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct exact signed-zero-preserving ``ref_5..ref_10`` bytes."""

    xbar = _zero_reference_states(zero_trace)
    delta = np.asarray(delta_float32)
    target = np.asarray(target_float32)
    if delta.shape != NORMALIZED_SHAPE or delta.dtype != np.dtype(np.float32):
        raise ReferenceTrajectoryLiftCanaryError("reference Delta must be float32 (10,32)")
    if target.shape != NORMALIZED_SHAPE or target.dtype != np.dtype(np.float32):
        raise ReferenceTrajectoryLiftCanaryError("reference target must be float32 (10,32)")
    mask = np.zeros(NORMALIZED_SHAPE, dtype=np.bool_)
    mask[:5, :3] = True
    outside = delta[~mask]
    if bool(np.count_nonzero(outside)) or bool(np.signbit(outside).any()):
        raise ReferenceTrajectoryLiftCanaryError(
            "reference Delta must be exact positive zero outside first-five XYZ"
        )
    alpha = _alpha_float32()
    weighted = np.asarray(alpha[:, None, None] * delta[None, :, :], dtype=np.float32)
    candidate = np.asarray(xbar + weighted, dtype=np.float32)
    reference = xbar.copy()
    update = np.broadcast_to(mask, REFERENCE_SHAPE) & (weighted != np.float32(0.0))
    reference[update] = candidate[update]
    if not _array_exact(reference[0], xbar[0]):
        raise ReferenceTrajectoryLiftCanaryError("reference step-5 anchor changed")
    if not _array_exact(reference[-1], target):
        raise ReferenceTrajectoryLiftCanaryError(
            "reference terminal bytes differ from the paired target"
        )
    preserved = ~update
    if reference[preserved].tobytes() != xbar[preserved].tobytes():
        raise ReferenceTrajectoryLiftCanaryError(
            "reference construction changed an untouched signed-zero byte"
        )
    return reference, xbar, alpha


def _reference_controls(
    noise: np.ndarray,
    reference: np.ndarray,
    delta: np.ndarray,
    budget: np.float32,
    *,
    projection_mode: str,
) -> Mapping[str, Any]:
    return {
        "noise": np.asarray(noise, dtype=np.float32).copy(),
        "intervention_mode": "reference_trajectory_lift",
        "intervention_step": 5,
        "return_trace": True,
        "return_normalized_final": True,
        "reference_states": np.asarray(reference, dtype=np.float32).copy(),
        "reference_space": "model",
        "delta": np.asarray(delta, dtype=np.float32).copy(),
        "delta_space": "model",
        "projection_mode": str(projection_mode),
        "model_l2_path_budget": np.float32(budget),
    }


def _compact_from_flow(value: Any, *, name: str) -> np.ndarray:
    array = _finite_array(value, name=name, shape=FLOW_SHAPE)
    if array.dtype != np.dtype(np.float32):
        raise ReferenceTrajectoryLiftCanaryError(f"{name} must preserve float32")
    return np.ascontiguousarray(array[5:, :5, :3].reshape(COMPACT_SHAPE))


def _schedule_from_compact_velocity(value: np.ndarray) -> np.ndarray:
    compact = np.asarray(value)
    if compact.shape != COMPACT_SHAPE or compact.dtype != np.dtype(np.float32):
        raise ReferenceTrajectoryLiftCanaryError(
            "compact schedule must preserve float32 (5,15)"
        )
    if not bool(np.isfinite(compact).all()):
        raise ReferenceTrajectoryLiftCanaryError("compact schedule is nonfinite")
    schedule = np.zeros(FLOW_SHAPE, dtype=np.float32)
    schedule[5:, :5, :3] = compact.reshape(5, 5, 3)
    return schedule


def _exact_scientific_reply(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        _array_exact(left.get("actions"), right.get("actions"))
        and _array_exact(
            left.get("crfs_trace", {}).get("final_normalized"),
            right.get("crfs_trace", {}).get("final_normalized"),
        )
        and _trace_exact(
            _deterministic_trace(left.get("crfs_trace", {})),
            _deterministic_trace(right.get("crfs_trace", {})),
        )
    )


def _shared_recurrence_exact(
    generated: Mapping[str, Any], replay: Mapping[str, Any]
) -> bool:
    left = generated.get("crfs_trace", {})
    right = replay.get("crfs_trace", {})
    return bool(
        _array_exact(generated.get("actions"), replay.get("actions"))
        and all(_array_exact(left.get(key), right.get(key)) for key in SHARED_RECURRENCE_LEAVES)
    )


def _reference_summary(
    reply: Mapping[str, Any],
    *,
    reference: np.ndarray,
    delta: np.ndarray,
    alpha: np.ndarray,
    budget: np.float32,
    noise: np.ndarray,
    projection_mode: str,
) -> Mapping[str, Any]:
    trace = reply.get("crfs_trace")
    if not isinstance(trace, Mapping):
        raise ReferenceTrajectoryLiftCanaryError("reference lift reply has no trace")
    expected_projection = 0 if projection_mode == "raw" else 1
    recurrence_errors = validate_flow_recurrence(trace)
    requested_velocity = _finite_array(
        trace.get("reference_requested_velocity_steps"),
        name="reference requested velocity",
        shape=FLOW_SHAPE,
    )
    executed = _finite_array(
        trace.get("reference_executed_increment_steps"),
        name="reference executed increment",
        shape=FLOW_SHAPE,
    )
    control_increment = _finite_array(
        trace.get("control_increment_steps"),
        name="reference control increment",
        shape=FLOW_SHAPE,
    )
    for name, value in (
        ("requested velocity", requested_velocity),
        ("executed increment", executed),
        ("control increment", control_increment),
    ):
        if value.dtype != np.dtype(np.float32):
            raise ReferenceTrajectoryLiftCanaryError(f"reference {name} lost float32")
    active_mask = np.zeros(FLOW_SHAPE, dtype=np.bool_)
    active_mask[5:, :5, :3] = True
    authoritative = np.zeros(FLOW_SHAPE, dtype=np.float32)
    transported = np.asarray(DT_FLOAT32 * requested_velocity, dtype=np.float32)
    authoritative[active_mask] = transported[active_mask]
    desired = _finite_array(
        trace.get("reference_desired_next_steps"),
        name="reference desired next",
        shape=FLOW_SHAPE,
    )
    raw = _finite_array(
        trace.get("reference_raw_increment_steps"),
        name="reference raw increment",
        shape=FLOW_SHAPE,
    )
    requested = _finite_array(
        trace.get("reference_requested_increment_steps"),
        name="reference requested increment",
        shape=FLOW_SHAPE,
    )
    projected = np.asarray(trace.get("reference_projected_steps"))
    if projected.shape != (10,) or projected.dtype != np.dtype(np.bool_):
        raise ReferenceTrajectoryLiftCanaryError(
            "reference projected flags must preserve bool (10,)"
        )
    checks = {
        "control_source_reference": _scalar(
            trace.get("control_source"), name="reference control_source"
        )
        == 2,
        "control_valid": bool(_scalar(trace.get("control_valid"), name="control_valid")),
        "schedule_applied": bool(
            _scalar(trace.get("schedule_applied"), name="schedule_applied")
        ),
        "recurrence_exact": not recurrence_errors,
        "initial_noise_exact": _array_exact(trace.get("initial_noise"), noise),
        "reference_exact": _array_exact(trace.get("reference_active_states"), reference),
        "delta_exact": _array_exact(trace.get("reference_delta"), delta),
        "alpha_exact": _array_exact(trace.get("reference_alpha"), alpha),
        "anchor_exact": bool(
            _scalar(trace.get("reference_anchor_exact"), name="reference anchor")
        ),
        "projection_mode_exact": _scalar(
            trace.get("reference_projection_mode"), name="reference projection mode"
        )
        == expected_projection,
        "budget_exact": np.asarray(
            _scalar(
                trace.get("reference_source_budget"),
                name="reference source budget",
                dtype=np.float32,
            ),
            dtype=np.float32,
        ).tobytes()
        == np.asarray(budget, dtype=np.float32).tobytes(),
        "cap_exact": np.asarray(
            _scalar(
                trace.get("reference_per_step_cap"),
                name="reference per-step cap",
                dtype=np.float32,
            ),
            dtype=np.float32,
        ).tobytes()
        == np.asarray(np.float32(budget) / np.float32(5), dtype=np.float32).tobytes(),
        "desired_reference_exact": _array_exact(desired[5:], reference[1:]),
        "authoritative_dt_u_exact": _array_exact(executed, authoritative),
        "flow_increment_active_exact": _array_exact(
            control_increment[active_mask], authoritative[active_mask]
        ),
        "final_physical_exact": _array_exact(
            trace.get("final_normalized_physical"), reply.get("actions")
        ),
        "raw_mode_unprojected": projection_mode != "raw" or _array_exact(raw, requested),
        "raw_mode_no_projection_flags": projection_mode != "raw" or not bool(projected.any()),
        "product_ball_applied": (
            projection_mode == "raw"
            or bool(
                _scalar(
                    trace.get("reference_product_ball_constraint_applied"),
                    name="reference product-ball applied",
                )
            )
        ),
        "product_ball_valid": (
            projection_mode == "raw"
            or bool(
                _scalar(
                    trace.get("reference_product_ball_valid"),
                    name="reference product-ball valid",
                )
            )
        ),
    }
    outside = ~active_mask
    checks["raw_increment_mask_positive_zero"] = bool(
        np.count_nonzero(raw[outside]) == 0
        and not np.signbit(raw[outside]).any()
    )
    checks["requested_increment_mask_positive_zero"] = bool(
        np.count_nonzero(requested[outside]) == 0
        and not np.signbit(requested[outside]).any()
    )
    checks["requested_velocity_mask_positive_zero"] = bool(
        np.count_nonzero(requested_velocity[outside]) == 0
        and not np.signbit(requested_velocity[outside]).any()
    )
    checks["executed_increment_mask_exact"] = bool(
        np.count_nonzero(executed[outside]) == 0
        and not np.signbit(executed[outside]).any()
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ReferenceTrajectoryLiftCanaryError(
            "reference lift trace checks failed: " + ", ".join(failed)
        )
    return {
        "checks": checks,
        "recurrence_errors": recurrence_errors,
        "requested_velocity": np.ascontiguousarray(requested_velocity),
        "executed_increment": np.ascontiguousarray(executed),
        "raw_increment": np.ascontiguousarray(raw),
        "requested_increment": np.ascontiguousarray(requested),
    }


def _smallest_float32_not_below(value_float64: float) -> np.float32:
    if not math.isfinite(value_float64) or value_float64 <= 0.0:
        raise ReferenceTrajectoryLiftCanaryError(
            "raw replay envelope seed must be finite and positive"
        )
    candidate = np.float32(value_float64)
    if not bool(np.isfinite(candidate)):
        raise ReferenceTrajectoryLiftCanaryError(
            "raw replay envelope is not finite float32"
        )
    if np.float64(candidate) < np.float64(value_float64):
        candidate = np.nextafter(candidate, np.float32(np.inf), dtype=np.float32)
    if not bool(np.isfinite(candidate)) or np.float64(candidate) < np.float64(value_float64):
        raise ReferenceTrajectoryLiftCanaryError(
            "could not form an upward float32 raw replay envelope"
        )
    return np.float32(candidate)


def raw_replay_envelope(
    executed_compact: np.ndarray, source_budget: np.float32
) -> Mapping[str, Any]:
    """Test exact B, otherwise derive and verify the frozen diagnostic envelope."""

    executed = np.asarray(executed_compact)
    if executed.shape != COMPACT_SHAPE or executed.dtype != np.dtype(np.float32):
        raise ReferenceTrajectoryLiftCanaryError(
            "raw executed increment must preserve float32 (5,15)"
        )
    per_step = np.linalg.norm(executed.reshape(5, -1), axis=1).astype(np.float32)
    path = np.asarray(np.sum(per_step, dtype=np.float32), dtype=np.float32)
    exact_budget_valid = True
    try:
        validate_executed_constraints(executed, np.float32(source_budget))
    except ActualForwardSearchError:
        exact_budget_valid = False
    if exact_budget_valid:
        seed64 = max(
            float(np.float64(path)),
            float(np.float64(5.0) * np.float64(np.max(per_step))),
            float(np.float64(np.float32(source_budget))),
        )
        replay_budget = np.float32(source_budget)
    else:
        seed64 = max(
            float(np.float64(path)),
            float(np.float64(5.0) * np.float64(np.max(per_step))),
            float(np.float64(np.float32(source_budget))),
        )
        replay_budget = _smallest_float32_not_below(seed64)
        try:
            validate_executed_constraints(executed, replay_budget)
        except ActualForwardSearchError as error:
            raise ReferenceTrajectoryLiftCanaryError(
                "derived raw replay envelope failed the unchanged validator"
            ) from error
    radius = np.asarray(np.float32(source_budget) / np.float32(5), dtype=np.float32)
    return {
        "exact_source_budget_valid": bool(exact_budget_valid),
        "replay_budget_float32": np.float32(replay_budget),
        "per_step_norm_float32": np.ascontiguousarray(per_step),
        "path_float32": np.float32(path),
        "max_step_norm_float32": np.float32(np.max(per_step)),
        "replay_seed_float64": np.float64(seed64),
        "path_over_source_budget_float64": float(
            np.float64(path) / np.float64(np.float32(source_budget))
        ),
        "max_step_over_source_radius_float64": float(
            np.float64(np.max(per_step)) / np.float64(radius)
        ),
    }


def _evaluate_reply(reply: Mapping[str, Any], target_physical: np.ndarray) -> Mapping[str, Any]:
    actions = _finite_array(reply.get("actions"), name="returned actions", shape=ACTION_SHAPE)
    if actions.dtype != np.dtype(np.float64):
        raise ReferenceTrajectoryLiftCanaryError(
            "returned actions must preserve registered float64"
        )
    error = np.ascontiguousarray(
        np.asarray(actions[:5, :7] - target_physical[:5, :7], dtype=np.float64)
    )
    objective, metrics, gates = objective_and_gates(error)
    return {
        "physical_error": error,
        "objective": float(objective),
        "metrics": tuple(float(value) for value in metrics),
        "gates": tuple(bool(value) for value in gates),
        "xyz_pass": bool(gates[0] and gates[1]),
        "full_pass": bool(all(gates)),
    }


def _stack_trace(replies: list[Mapping[str, Any]], leaf: str, *, dtype: Any) -> np.ndarray:
    values: list[np.ndarray] = []
    for reply in replies:
        trace = reply.get("crfs_trace")
        if not isinstance(trace, Mapping) or leaf not in trace:
            raise ReferenceTrajectoryLiftCanaryError(
                f"raw artifact trace omitted {leaf}"
            )
        value = np.asarray(trace[leaf])
        if value.dtype != np.dtype(dtype):
            raise ReferenceTrajectoryLiftCanaryError(
                f"raw artifact trace {leaf} must preserve {np.dtype(dtype)}, got {value.dtype}"
            )
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise ReferenceTrajectoryLiftCanaryError(
                f"raw artifact trace {leaf} is nonfinite"
            )
        values.append(np.ascontiguousarray(value) if value.ndim else value.copy())
    return np.ascontiguousarray(np.stack(values))


def _compact_group_trace(
    replies: list[Mapping[str, Any]], leaf: str, *, dtype: Any = np.float32
) -> np.ndarray:
    full = _stack_trace(replies, leaf, dtype=dtype)
    return np.ascontiguousarray(full[:, 5:, :5, :3].reshape(len(replies), 5, 15))


def _stack_actions(replies: list[Mapping[str, Any]]) -> np.ndarray:
    values: list[np.ndarray] = []
    for reply in replies:
        value = np.asarray(reply.get("actions"))
        if value.shape != ACTION_SHAPE or value.dtype != np.dtype(np.float64):
            raise ReferenceTrajectoryLiftCanaryError(
                "policy actions must preserve registered float64 (10,7)"
            )
        if not bool(np.isfinite(value).all()):
            raise ReferenceTrajectoryLiftCanaryError("policy actions are nonfinite")
        values.append(np.ascontiguousarray(value))
    return np.ascontiguousarray(np.stack(values))


def _add_transport_group(
    arrays: dict[str, np.ndarray],
    prefix: str,
    replies: list[Mapping[str, Any]],
    *,
    requested_c: np.ndarray,
    requested_velocity: np.ndarray,
    canonical_positive_zero: bool = False,
    reference_generation: bool = False,
) -> None:
    """Serialize one fixed two-reply group under the independent schema."""

    if len(replies) != 2:
        raise ReferenceTrajectoryLiftCanaryError(
            f"{prefix} must contain exactly two replies"
        )
    c = np.ascontiguousarray(np.asarray(requested_c))
    u = np.ascontiguousarray(np.asarray(requested_velocity))
    if c.shape == COMPACT_SHAPE:
        c = np.broadcast_to(c, (2, *COMPACT_SHAPE)).copy()
    if u.shape == COMPACT_SHAPE:
        u = np.broadcast_to(u, (2, *COMPACT_SHAPE)).copy()
    if c.shape != (2, *COMPACT_SHAPE) or c.dtype != np.dtype(np.float32):
        raise ReferenceTrajectoryLiftCanaryError(
            f"{prefix} requested c must preserve float32 (2,5,15)"
        )
    if u.shape != (2, *COMPACT_SHAPE) or u.dtype != np.dtype(np.float32):
        raise ReferenceTrajectoryLiftCanaryError(
            f"{prefix} requested velocity must preserve float32 (2,5,15)"
        )
    applied = _compact_group_trace(replies, "control_velocity_steps")
    executed = _compact_group_trace(replies, "control_increment_steps")
    if canonical_positive_zero:
        c = np.zeros((2, *COMPACT_SHAPE), dtype=np.float32)
        u = np.zeros_like(c)
        applied = np.zeros_like(c)
        executed = np.zeros_like(c)
    arrays.update(
        {
            f"{prefix}_requested_c_f32": c,
            f"{prefix}_requested_velocity_f32": u,
            f"{prefix}_applied_velocity_f32": applied,
            f"{prefix}_executed_c_f32": executed,
            f"{prefix}_final_normalized_f32": _stack_trace(
                replies, "final_normalized", dtype=np.float32
            ),
            f"{prefix}_final_normalized_physical_f64": _stack_trace(
                replies, "final_normalized_physical", dtype=np.float64
            ),
            f"{prefix}_returned_actions_f64": _stack_actions(replies),
            f"{prefix}_trace_step_index_i64": _stack_trace(
                replies, "step_index_steps", dtype=np.int64
            ),
            f"{prefix}_trace_time_f32": _stack_trace(
                replies, "time_steps", dtype=np.float32
            ),
            f"{prefix}_trace_active_bool": _stack_trace(
                replies, "active_steps", dtype=np.bool_
            ),
            f"{prefix}_trace_x_t_f32": _stack_trace(
                replies, "x_t_steps", dtype=np.float32
            ),
            f"{prefix}_trace_v_base_f32": _stack_trace(
                replies, "v_base_steps", dtype=np.float32
            ),
            f"{prefix}_trace_control_velocity_f32": _stack_trace(
                replies, "control_velocity_steps", dtype=np.float32
            ),
            f"{prefix}_trace_total_velocity_f32": _stack_trace(
                replies, "total_velocity_steps", dtype=np.float32
            ),
            f"{prefix}_trace_control_increment_f32": _stack_trace(
                replies, "control_increment_steps", dtype=np.float32
            ),
            f"{prefix}_trace_x_next_f32": _stack_trace(
                replies, "x_next_steps", dtype=np.float32
            ),
            f"{prefix}_trace_initial_noise_f32": _stack_trace(
                replies, "initial_noise", dtype=np.float32
            ),
        }
    )
    if not reference_generation:
        return
    reference_leaves = {
        "reference_active_states_f32": ("reference_active_states", np.float32),
        "reference_delta_f32": ("reference_delta", np.float32),
        "reference_alpha_f32": ("reference_alpha", np.float32),
        "reference_anchor_exact_bool": ("reference_anchor_exact", np.bool_),
        "reference_projection_mode_i64": ("reference_projection_mode", np.int64),
        "reference_source_budget_f32": ("reference_source_budget", np.float32),
        "reference_per_step_cap_f32": ("reference_per_step_cap", np.float32),
        "reference_desired_next_steps_f32": (
            "reference_desired_next_steps",
            np.float32,
        ),
        "reference_uncontrolled_next_steps_f32": (
            "reference_uncontrolled_next_steps",
            np.float32,
        ),
        "reference_raw_increment_steps_f32": (
            "reference_raw_increment_steps",
            np.float32,
        ),
        "reference_requested_increment_steps_f32": (
            "reference_requested_increment_steps",
            np.float32,
        ),
        "reference_requested_velocity_steps_f32": (
            "reference_requested_velocity_steps",
            np.float32,
        ),
        "reference_executed_increment_steps_f32": (
            "reference_executed_increment_steps",
            np.float32,
        ),
        "reference_raw_norm_f64_steps": ("reference_raw_norm_f64_steps", np.float64),
        "reference_requested_norm_f64_steps": (
            "reference_requested_norm_f64_steps",
            np.float64,
        ),
        "reference_executed_norm_f64_steps": (
            "reference_executed_norm_f64_steps",
            np.float64,
        ),
        "reference_projection_scale_f64_steps": (
            "reference_projection_scale_f64_steps",
            np.float64,
        ),
        "reference_projected_steps_bool": ("reference_projected_steps", np.bool_),
        "reference_tracking_error_steps_f32": (
            "reference_tracking_error_steps",
            np.float32,
        ),
        "reference_raw_path_length_f64": ("reference_raw_path_length_f64", np.float64),
        "reference_requested_path_length_f64": (
            "reference_requested_path_length_f64",
            np.float64,
        ),
        "reference_executed_path_length_f64": (
            "reference_executed_path_length_f64",
            np.float64,
        ),
        "reference_product_ball_constraint_applied_bool": (
            "reference_product_ball_constraint_applied",
            np.bool_,
        ),
        "reference_product_ball_valid_bool": (
            "reference_product_ball_valid",
            np.bool_,
        ),
    }
    for suffix, (leaf, dtype) in reference_leaves.items():
        arrays[f"{prefix}_{suffix}"] = _stack_trace(replies, leaf, dtype=dtype)


def _load_af_source(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, np.ndarray]]:
    af = value.get("frozen_source_bindings", {}).get("af00a_source_run", {})
    result_path = Path(str(af.get("results_json_path"))).resolve()
    tensor_path = Path(str(af.get("tensor_archive_path"))).resolve()
    result_hash = file_sha256(result_path)
    tensor_hash = file_sha256(tensor_path)
    if result_hash != AF_RESULTS_SHA256 or result_hash != af.get("results_json_sha256"):
        raise ReferenceTrajectoryLiftCanaryError("frozen AF-00A results hash changed")
    if tensor_hash != AF_TENSOR_SHA256 or tensor_hash != af.get("tensor_archive_sha256"):
        raise ReferenceTrajectoryLiftCanaryError("frozen AF-00A tensor hash changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    raw_evidence = result.get("raw_evidence", {}) if isinstance(result, Mapping) else {}
    if not isinstance(result, Mapping) or not (
        result.get("experiment_identity") == "AF-00A"
        and result.get("run_id") == af.get("run_id")
        and result.get("case_id") == CASE_ID
        and result.get("status") == "frozen_cem_negative"
        and result.get("execution_boundary", {}).get(
            "policy_generated_action_steps_executed"
        )
        == 0
        and result.get("execution_boundary", {}).get(
            "teacher_generated_action_steps_executed"
        )
        == 0
        and result.get("execution_boundary", {}).get("efficacy_rollouts_executed")
        == 0
        and raw_evidence.get("tensor_sha256") == tensor_hash
        and Path(str(raw_evidence.get("tensor_path"))).resolve() == tensor_path
        and raw_evidence.get("source_r02_sha256")
        == value.get("frozen_source_bindings", {}).get("source_r02", {}).get("sha256")
    ):
        raise ReferenceTrajectoryLiftCanaryError("frozen AF-00A result semantics changed")
    required = {
        "source_noise_f32",
        "source_frozen_final_f32",
        "source_delta_model_f32",
        "source_target_normalized_f32",
        "source_model_to_physical_scale_f32",
        "source_target_physical_f64",
        "source_budget_f32",
        "source_dt_f32",
        "source_radius_f32",
        "arm_a_requested_c_f32",
        "arm_a_requested_velocity_f32",
        "arm_a_applied_velocity_f32",
        "arm_a_executed_c_f32",
        "arm_a_final_normalized_f32",
        "arm_a_final_normalized_physical_f64",
        "arm_a_returned_actions_f64",
        "arm_a_trace_step_index_i64",
        "arm_a_trace_time_f32",
        "arm_a_trace_active_bool",
        "arm_a_trace_x_t_f32",
        "arm_a_trace_v_base_f32",
        "arm_a_trace_total_velocity_f32",
        "arm_a_trace_x_next_f32",
        "arm_a_trace_initial_noise_f32",
    }
    with np.load(tensor_path, allow_pickle=False) as archive:
        if not required.issubset(set(archive.files)):
            raise ReferenceTrajectoryLiftCanaryError(
                "frozen AF-00A tensor archive is incomplete"
            )
        arrays = {name: np.ascontiguousarray(archive[name].copy()) for name in required}
    return dict(result), arrays


def _af_source_checks(
    historical: Mapping[str, np.ndarray],
    *,
    noise: np.ndarray,
    frozen_final: np.ndarray,
    delta: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
    target_physical: np.ndarray,
    budget: np.float32,
) -> Mapping[str, bool]:
    return {
        "noise_exact": _array_exact(historical["source_noise_f32"], noise),
        "frozen_final_exact": _array_exact(
            historical["source_frozen_final_f32"], frozen_final
        ),
        "delta_exact": _array_exact(historical["source_delta_model_f32"], delta),
        "target_exact": _array_exact(
            historical["source_target_normalized_f32"], target
        ),
        "scale_exact": _array_exact(
            historical["source_model_to_physical_scale_f32"], scale
        ),
        "target_physical_exact": _array_exact(
            historical["source_target_physical_f64"], target_physical
        ),
        "budget_exact": _array_exact(historical["source_budget_f32"], budget),
        "dt_exact": _array_exact(historical["source_dt_f32"], DT_FLOAT32),
        "radius_exact": _array_exact(
            historical["source_radius_f32"],
            np.asarray(budget / np.float32(5), dtype=np.float32),
        ),
    }


def _af_arm_a_checks(
    historical: Mapping[str, np.ndarray],
    replies: list[Mapping[str, Any]],
    requested_c: np.ndarray,
    requested_velocity: np.ndarray,
) -> Mapping[str, bool]:
    def stack_trace(leaf: str) -> np.ndarray:
        return np.ascontiguousarray(
            np.stack([np.asarray(reply["crfs_trace"][leaf]) for reply in replies])
        )

    applied = stack_trace("control_velocity_steps")[:, 5:, :5, :3].reshape(2, 5, 15)
    executed = stack_trace("control_increment_steps")[:, 5:, :5, :3].reshape(2, 5, 15)
    checks = {
        "requested_c_exact": _array_exact(
            historical["arm_a_requested_c_f32"],
            np.broadcast_to(requested_c, (2, *COMPACT_SHAPE)).copy(),
        ),
        "requested_velocity_exact": _array_exact(
            historical["arm_a_requested_velocity_f32"],
            np.broadcast_to(requested_velocity, (2, *COMPACT_SHAPE)).copy(),
        ),
        "applied_exact": _array_exact(historical["arm_a_applied_velocity_f32"], applied),
        "executed_exact": _array_exact(historical["arm_a_executed_c_f32"], executed),
        "final_exact": _array_exact(
            historical["arm_a_final_normalized_f32"],
            stack_trace("final_normalized"),
        ),
        "final_physical_exact": _array_exact(
            historical["arm_a_final_normalized_physical_f64"],
            stack_trace("final_normalized_physical"),
        ),
        "returned_actions_exact": _array_exact(
            historical["arm_a_returned_actions_f64"],
            np.stack([np.asarray(reply["actions"]) for reply in replies]),
        ),
        "step_index_exact": _array_exact(
            historical["arm_a_trace_step_index_i64"], stack_trace("step_index_steps")
        ),
        "time_exact": _array_exact(
            historical["arm_a_trace_time_f32"], stack_trace("time_steps")
        ),
        "active_exact": _array_exact(
            historical["arm_a_trace_active_bool"], stack_trace("active_steps")
        ),
        "x_t_exact": _array_exact(
            historical["arm_a_trace_x_t_f32"], stack_trace("x_t_steps")
        ),
        "v_base_exact": _array_exact(
            historical["arm_a_trace_v_base_f32"], stack_trace("v_base_steps")
        ),
        "total_velocity_exact": _array_exact(
            historical["arm_a_trace_total_velocity_f32"],
            stack_trace("total_velocity_steps"),
        ),
        "x_next_exact": _array_exact(
            historical["arm_a_trace_x_next_f32"], stack_trace("x_next_steps")
        ),
        "noise_exact": _array_exact(
            historical["arm_a_trace_initial_noise_f32"], stack_trace("initial_noise")
        ),
    }
    return checks


def run_reference_trajectory_lift_canary(
    case: Mapping[str, Any],
    actual_config_mapping: Mapping[str, Any],
    legacy_config: Any,
    *,
    repo_root: str | Path,
    input_manifest_sha256: str,
    actual_config_path: str | Path,
    legacy_config_path: str | Path,
    client: Any = None,
    environment: Optional[SafeLiberoCase] = None,
) -> tuple[Path, str]:
    """Execute the only accepted exact 18-call finite transaction."""

    _validate_config(actual_config_mapping)
    if not os.environ.get("SLURM_JOB_ID"):
        raise ReferenceTrajectoryLiftCanaryError("TRL-00A must execute inside Slurm")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ReferenceTrajectoryLiftCanaryError(
            "TRL-00A requires one visible allocation GPU"
        )
    if socket.gethostname().split(".", 1)[0] != SOURCE_HOST:
        raise ReferenceTrajectoryLiftCanaryError(
            f"TRL-00A must remain pinned to {SOURCE_HOST}"
        )
    manifest = actual_config_mapping.get("frozen_source_bindings", {}).get("manifest", {})
    if input_manifest_sha256 != MANIFEST_SHA256 or input_manifest_sha256 != manifest.get("sha256"):
        raise R05ACanarySourceError("TRL-00A manifest binding changed")
    frozen_case = actual_config_mapping.get("frozen_case", {})
    if dict(case).get("case_id") != CASE_ID or dict(case).get("group_id") != GROUP_ID:
        raise R05ACanarySourceError("TRL-00A case/group identity changed")
    if any(
        dict(case).get(key) != frozen_case.get(key)
        for key in ("case_id", "group_id", "environment_seed", "policy_seed")
    ):
        raise R05ACanarySourceError("TRL-00A case differs from released config")

    actual_path = Path(actual_config_path).resolve()
    legacy_path = Path(legacy_config_path).resolve()
    if json.loads(actual_path.read_text(encoding="utf-8")) != dict(actual_config_mapping):
        raise ReferenceTrajectoryLiftCanaryError(
            "TRL-00A config mapping differs from its file"
        )
    release = actual_config_mapping["execution_release"]
    if legacy_config.oracle.run_id != release.get("run_id"):
        raise ReferenceTrajectoryLiftCanaryError(
            "runtime run ID differs from the exact TRL-00A release"
        )
    output_dir = Path(legacy_config.oracle.output_root) / legacy_config.oracle.run_id / CASE_ID
    tensor_path = output_dir / TENSOR_FILENAME
    ledger_path = output_dir / LEDGER_FILENAME
    payload_path = output_dir / PAYLOAD_FILENAME
    case_results_path = output_dir / RESULTS_FILENAME
    run_results_path = output_dir.parent / RESULTS_FILENAME
    if any(
        path.exists()
        for path in (
            tensor_path,
            ledger_path,
            payload_path,
            case_results_path,
            run_results_path,
        )
    ):
        raise ReferenceTrajectoryLiftCanaryError(
            "TRL-00A immutable output path is already used"
        )

    source_path, raw_r02, source_sha = _load_source_r02(legacy_config)
    source_binding = actual_config_mapping.get("frozen_source_bindings", {}).get(
        "source_r02", {}
    )
    if (
        source_sha != source_binding.get("sha256")
        or source_path.resolve() != Path(str(source_binding.get("path"))).resolve()
    ):
        raise R05ACanarySourceError("TRL-00A R02 source hash changed")
    if raw_r02.get("provenance", {}).get("case_record") != dict(case):
        raise R05ACanarySourceError("manifest row differs from immutable R02 source")
    delta64, delta_record, source_budget64 = _source_delta(raw_r02)
    registered_budget64 = float(
        actual_config_mapping.get("target_contract", {}).get(
            "source_reported_budget_float64"
        )
    )
    if source_budget64 != registered_budget64:
        raise R05ACanarySourceError(
            "R02 float64 source budget differs from TRL-00A"
        )
    delta32 = np.ascontiguousarray(np.asarray(delta64, dtype=np.float32))
    source_budget32 = np.float32(
        raw_r02["directions"]["l2_norms"]["delta_star_model"]
    )
    if source_budget32.tobytes() != SOURCE_BUDGET_FLOAT32.tobytes():
        raise R05ACanarySourceError("R02 source budget differs from TRL-00A")
    noise = np.random.default_rng(int(case["policy_seed"])).normal(size=NORMALIZED_SHAPE).astype(
        np.float32
    )
    source_noise, noise_errors = _validate_array_record(
        raw_r02.get("provenance", {}).get("noise"),
        name="source R02 noise",
        shape=NORMALIZED_SHAPE,
    )
    if noise_errors or source_noise is None or not _array_exact(
        noise, np.asarray(source_noise, dtype=np.float32)
    ):
        raise R05ACanarySourceError("reconstructed policy noise differs from R02")

    af_result, af_arrays = _load_af_source(actual_config_mapping)

    if client is None:
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(
            legacy_config.oracle.host, legacy_config.oracle.port
        )
    owns_environment = environment is None
    if environment is None:
        environment = SafeLiberoCase(dict(case), legacy_config.oracle)
    else:
        environment.configure_case(dict(case))

    ledger_rows: list[dict[str, Any]] = []
    policy_index = 0

    def request(
        phase: str,
        controls: Mapping[str, Any],
        *,
        require_trace: bool,
        phase_index: int = 0,
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, float]:
        nonlocal policy_index
        transport = copy.deepcopy(dict(policy_input))
        transport["__crfs__"] = copy.deepcopy(dict(controls))
        started = time.perf_counter_ns()
        try:
            reply = client.infer(transport)
        except Exception as error:
            raise R05ACanaryPolicyError(f"{type(error).__name__}: {error}") from error
        elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
        if not isinstance(reply, Mapping):
            raise R05ACanaryPolicyError("policy reply is not a mapping")
        terminal: Mapping[str, Any] | None = None
        normal: Mapping[str, Any] | None = None
        if TERMINAL_RESPONSE_KEY in reply:
            normalized_terminal = _normalize_raw_terminal(reply)
            raise ReferenceTrajectoryLiftCanaryError(
                f"{phase} returned {normalized_terminal['type']} at step "
                f"{normalized_terminal['step']} leaf {normalized_terminal['leaf']}; "
                "raw nonfinite execution is apparatus-inconclusive"
            )
        else:
            if "actions" not in reply:
                raise R05ACanaryPolicyError("policy reply has no actions")
            if require_trace and not isinstance(reply.get("crfs_trace"), Mapping):
                raise R05ACanaryPolicyError("policy reply has no CRFS trace")
            normal = reply
        trace_locations: dict[int, tuple[str | None, int | None]] = {
            0: (None, None),
            1: (None, None),
            2: (None, None),
            3: ("zero", 0),
            4: ("arm_a", 0),
            5: ("arm_a", 1),
            6: ("arm_b_generation", 0),
            7: ("arm_b_generation", 1),
            8: ("arm_b_replay", 0),
            9: ("arm_b_replay", 1),
            10: ("raw_generation", 0),
            11: ("raw_generation", 1),
            12: ("raw_replay", 0),
            13: ("raw_replay", 1),
            14: ("zero", 1),
            15: (None, None),
            16: (None, None),
            17: (None, None),
        }
        if policy_index not in trace_locations:
            raise ReferenceTrajectoryLiftCanaryError(
                "policy request escaped the finite 18-row ledger"
            )
        trace_group, trace_index = trace_locations[policy_index]
        assert normal is not None
        row: dict[str, Any] = {
            "ordinal": policy_index,
            "phase": phase,
            "phase_index": int(phase_index),
            "reply_kind": "action",
            "elapsed_seconds": float(elapsed),
            "trace_group": trace_group,
            "trace_index": trace_index,
            "actions": _bytes_digest(normal["actions"]),
        }
        if phase in {"budgeted_lift_generation", "raw_lift_generation"}:
            row["supplied_reference"] = _bytes_digest(controls["reference_states"])
            row["generated_schedule"] = _bytes_digest(
                normal["crfs_trace"]["control_velocity_steps"]
            )
            row["projection_mode"] = str(controls["projection_mode"])
        elif trace_group is not None:
            row["requested_schedule"] = _bytes_digest(controls["schedule"])
        if trace_group is not None:
            row["model_l2_path_budget"] = _bytes_digest(
                np.asarray(controls["model_l2_path_budget"])
            )
        ledger_rows.append(row)
        policy_index += 1
        return normal, terminal, float(elapsed)

    try:
        initial_observation = environment.reset_and_settle()
        if environment.obstacle_name is None:
            raise ReferenceTrajectoryLiftCanaryError(
                "TRL-00A did not resolve the active obstacle"
            )
        branch = capture_reach_snapshot(
            environment, TARGET_OBJECT_NAME, environment.obstacle_name
        )
        if _target_contact_at_branch(
            environment, TARGET_OBJECT_NAME
        ) or environment.env.check_success():
            raise ReferenceTrajectoryLiftCanaryError(
                "TRL-00A source is no longer a pregrasp branch"
            )
        policy_input = policy_observation(
            initial_observation, environment.prompt, legacy_config.oracle.resize_size
        )
        source_pairing = raw_r02.get("pairing")
        if not isinstance(source_pairing, Mapping):
            raise R05ACanarySourceError("R02 source has no pairing record")
        observation_fingerprint = _observation_fingerprint(policy_input)
        branch_snapshot = _json_compatible(branch.to_dict())
        pairing_checks = {
            "observation_exact": observation_fingerprint
            == source_pairing.get("policy_observation"),
            "branch_snapshot_exact": branch_snapshot
            == source_pairing.get("branch_snapshot"),
            "noise_exact": True,
        }
        if not all(pairing_checks.values()):
            raise R05ACanarySourceError(
                f"TRL-00A branch pairing failed: {pairing_checks}"
            )

        compiled_before, terminal, _ = request(
            "compiled_frozen_pre",
            _frozen_controls(noise, return_trace=False),
            require_trace=False,
        )
        assert compiled_before is not None and terminal is None
        source_before, terminal, _ = request(
            "eager_source_trace_pre",
            _frozen_controls(noise, return_trace=True),
            require_trace=True,
        )
        assert source_before is not None and terminal is None
        source_actions, action_errors = _validate_array_record(
            source_pairing.get("eager_actions"),
            name="source eager actions",
            shape=ACTION_SHAPE,
        )
        if action_errors or source_actions is None or not _array_exact(
            source_before["actions"], source_actions
        ):
            raise R05ACanarySourceError(
                "current source actions differ from immutable R02"
            )
        source_trace = _trace_from_record(
            source_pairing.get("eager_trace"), name="source eager trace"
        )
        if _trace_pairing_diagnostics(
            source_trace, source_before["crfs_trace"]
        ).get("exact_native_leaf_pairing") is not True:
            raise R05ACanarySourceError(
                "current source trace differs from immutable R02"
            )
        eager_before, terminal, _ = request(
            "eager_normalized_final_pre",
            _frozen_controls(
                noise, return_trace=True, return_normalized_final=True
            ),
            require_trace=True,
        )
        assert eager_before is not None and terminal is None
        frozen_final = _finite_array(
            eager_before["crfs_trace"].get("final_normalized"),
            name="fresh frozen final",
            shape=NORMALIZED_SHAPE,
        )
        if frozen_final.dtype != np.dtype(np.float32) or not _array_exact(
            eager_before["actions"], source_before["actions"]
        ):
            raise R05ACanaryPolicyError(
                "fresh frozen normalized path changed source actions"
            )
        target = np.where(
            delta32 == np.float32(0.0),
            frozen_final,
            np.asarray(frozen_final + delta32, dtype=np.float32),
        ).astype(np.float32, copy=False)
        scale = np.ones(NORMALIZED_SHAPE, dtype=np.float32)
        scale[:, :3] = np.asarray(REGISTERED_XYZ_SCALE, dtype=np.float32)
        target_physical = np.asarray(eager_before["actions"], dtype=np.float64).copy()
        target_physical[:5, :3] += np.asarray(
            delta32[:5, :3], dtype=np.float64
        ) * np.asarray(scale[:5, :3], dtype=np.float64)
        af_source_checks = _af_source_checks(
            af_arrays,
            noise=noise,
            frozen_final=frozen_final,
            delta=delta32,
            target=target,
            scale=scale,
            target_physical=target_physical,
            budget=source_budget32,
        )
        if not all(af_source_checks.values()):
            raise ReferenceTrajectoryLiftCanaryError(
                "fresh source differs from AF-00A: "
                + ", ".join(name for name, passed in af_source_checks.items() if not passed)
            )

        zero_schedule = np.zeros(FLOW_SHAPE, dtype=np.float32)
        zero_before, terminal, elapsed = request(
            "zero_schedule_pre",
            _schedule_controls(noise, zero_schedule, np.float32(0.0)),
            require_trace=True,
        )
        assert zero_before is not None and terminal is None
        zero_summary = _replay_summary(
            zero_before,
            requested_schedule=zero_schedule,
            source_budget_float32=np.float32(0.0),
            paired_noise=noise,
            elapsed_seconds=elapsed,
        )
        if zero_summary["passed"] is not True or not (
            _array_exact(zero_before["actions"], eager_before["actions"])
            and _array_exact(
                zero_before["crfs_trace"]["final_normalized"], frozen_final
            )
        ):
            raise ReferenceTrajectoryLiftCanaryError(
                "paired zero pre/reference check failed"
            )
        reference, xbar, alpha = construct_reference_trajectory(
            zero_before["crfs_trace"], delta32, target
        )

        arm_a_c = np.broadcast_to(
            np.asarray(
                delta32[:5, :3].reshape(15) / np.float32(5), dtype=np.float32
            ),
            COMPACT_SHAPE,
        ).copy()
        arm_a_u = np.ascontiguousarray(np.asarray(arm_a_c / DT_FLOAT32, dtype=np.float32))

        def ordinary_schedule_call(
            phase: str,
            schedule: np.ndarray,
            budget: np.float32,
            *,
            phase_index: int,
        ) -> Mapping[str, Any]:
            reply, terminal_reply, call_elapsed = request(
                phase,
                _schedule_controls(noise, schedule, budget),
                require_trace=True,
                phase_index=phase_index,
            )
            assert reply is not None and terminal_reply is None
            summary = _replay_summary(
                reply,
                requested_schedule=schedule,
                source_budget_float32=budget,
                paired_noise=noise,
                elapsed_seconds=call_elapsed,
            )
            if summary["passed"] is not True:
                raise ReferenceTrajectoryLiftCanaryError(
                    f"{phase} failed ordinary residual-schedule validation"
                )
            return reply

        arm_a_schedule = _schedule_from_compact_velocity(arm_a_u)
        arm_a_replies = [
            ordinary_schedule_call(
                "arm_a_equal_split", arm_a_schedule, source_budget32, phase_index=index
            )
            for index in range(2)
        ]
        if not _exact_scientific_reply(*arm_a_replies):
            raise ReferenceTrajectoryLiftCanaryError("Arm A duplicates differ")
        af_arm_a_checks = _af_arm_a_checks(
            af_arrays, arm_a_replies, arm_a_c, arm_a_u
        )
        arm_a_evaluation = _evaluate_reply(arm_a_replies[0], target_physical)
        historical_arm_a = af_result.get("validation", {}).get("arm_a", {})
        af_arm_a_checks = {
            **af_arm_a_checks,
            "objective_exact": arm_a_evaluation["objective"]
            == historical_arm_a.get("objective"),
            "metrics_exact": dict(
                zip(
                    ("xyz_max_abs", "xyz_rms", "full_max_abs", "full_rms"),
                    arm_a_evaluation["metrics"],
                )
            )
            == historical_arm_a.get("metrics"),
            "gate_exact": arm_a_evaluation["full_pass"]
            is bool(historical_arm_a.get("passed")),
        }
        if not all(af_arm_a_checks.values()):
            raise ReferenceTrajectoryLiftCanaryError(
                "Arm A did not reproduce AF-00A: "
                + ", ".join(name for name, passed in af_arm_a_checks.items() if not passed)
            )

        budgeted_replies: list[Mapping[str, Any]] = []
        budgeted_summaries: list[Mapping[str, Any]] = []
        for index in range(2):
            reply, terminal_reply, _ = request(
                "budgeted_lift_generation",
                _reference_controls(
                    noise,
                    reference,
                    delta32,
                    source_budget32,
                    projection_mode="product_ball",
                ),
                require_trace=True,
                phase_index=index,
            )
            assert reply is not None and terminal_reply is None
            budgeted_summaries.append(
                _reference_summary(
                    reply,
                    reference=reference,
                    delta=delta32,
                    alpha=alpha,
                    budget=source_budget32,
                    noise=noise,
                    projection_mode="product_ball",
                )
            )
            budgeted_replies.append(reply)
        if not _exact_scientific_reply(*budgeted_replies):
            raise ReferenceTrajectoryLiftCanaryError(
                "budgeted lift generations differ"
            )
        budgeted_schedule = np.ascontiguousarray(
            budgeted_summaries[0]["requested_velocity"]
        )
        if not _array_exact(
            budgeted_schedule, budgeted_summaries[1]["requested_velocity"]
        ):
            raise ReferenceTrajectoryLiftCanaryError(
                "budgeted lift generated different schedules"
            )
        validate_executed_constraints(
            _compact_from_flow(
                budgeted_summaries[0]["executed_increment"],
                name="budgeted executed increment",
            ),
            source_budget32,
        )
        budgeted_replays = [
            ordinary_schedule_call(
                "budgeted_lift_replay",
                budgeted_schedule,
                source_budget32,
                phase_index=index,
            )
            for index in range(2)
        ]
        if not _exact_scientific_reply(*budgeted_replays) or not all(
            _shared_recurrence_exact(budgeted_replies[0], reply)
            for reply in budgeted_replays
        ):
            raise ReferenceTrajectoryLiftCanaryError(
                "budgeted ordinary replay differs from generated recurrence"
            )

        raw_replies: list[Mapping[str, Any]] = []
        raw_summaries: list[Mapping[str, Any]] = []
        for index in range(2):
            reply, terminal_reply, _ = request(
                "raw_lift_generation" if index == 0 else "raw_lift_generation",
                _reference_controls(
                    noise,
                    reference,
                    delta32,
                    source_budget32,
                    projection_mode="raw",
                ),
                require_trace=True,
                phase_index=index,
            )
            assert reply is not None and terminal_reply is None
            raw_summaries.append(
                _reference_summary(
                    reply,
                    reference=reference,
                    delta=delta32,
                    alpha=alpha,
                    budget=source_budget32,
                    noise=noise,
                    projection_mode="raw",
                )
            )
            raw_replies.append(reply)
        raw_replays: list[Mapping[str, Any]] = []
        if len(raw_replies) != 2 or not _exact_scientific_reply(*raw_replies):
            raise ReferenceTrajectoryLiftCanaryError("raw lift generations differ")
        raw_schedule = np.ascontiguousarray(raw_summaries[0]["requested_velocity"])
        if not _array_exact(raw_schedule, raw_summaries[1]["requested_velocity"]):
            raise ReferenceTrajectoryLiftCanaryError(
                "raw lift generated different schedules"
            )
        raw_executed = _compact_from_flow(
            raw_summaries[0]["executed_increment"],
            name="raw executed increment",
        )
        replay_envelope = raw_replay_envelope(raw_executed, source_budget32)
        replay_budget = np.float32(replay_envelope["replay_budget_float32"])
        raw_replays = [
            ordinary_schedule_call(
                "raw_lift_replay",
                raw_schedule,
                replay_budget,
                phase_index=index,
            )
            for index in range(2)
        ]
        if not _exact_scientific_reply(*raw_replays) or not all(
            _shared_recurrence_exact(raw_replies[0], reply) for reply in raw_replays
        ):
            raise ReferenceTrajectoryLiftCanaryError(
                "raw ordinary replay differs from generated recurrence"
            )
        raw_branch = "finite"

        post_start = 14
        zero_after, terminal, zero_after_elapsed = request(
            "zero_schedule_post",
            _schedule_controls(noise, zero_schedule, np.float32(0.0)),
            require_trace=True,
        )
        assert zero_after is not None and terminal is None
        zero_after_summary = _replay_summary(
            zero_after,
            requested_schedule=zero_schedule,
            source_budget_float32=np.float32(0.0),
            paired_noise=noise,
            elapsed_seconds=zero_after_elapsed,
        )
        eager_after, terminal, _ = request(
            "eager_normalized_final_post",
            _frozen_controls(
                noise, return_trace=True, return_normalized_final=True
            ),
            require_trace=True,
        )
        assert eager_after is not None and terminal is None
        source_after, terminal, _ = request(
            "eager_source_trace_post",
            _frozen_controls(noise, return_trace=True),
            require_trace=True,
        )
        assert source_after is not None and terminal is None
        compiled_after, terminal, _ = request(
            "compiled_frozen_post",
            _frozen_controls(noise, return_trace=False),
            require_trace=False,
        )
        assert compiled_after is not None and terminal is None
        post_checks = {
            "zero_summary_passed": zero_after_summary["passed"] is True,
            "zero_exact": _exact_scientific_reply(zero_before, zero_after),
            "eager_exact": _exact_scientific_reply(eager_before, eager_after),
            "source_exact": _exact_scientific_reply(source_before, source_after),
            "compiled_actions_exact": _array_exact(
                compiled_before["actions"], compiled_after["actions"]
            ),
            "source_trace_pairing_exact": _trace_pairing_diagnostics(
                source_trace, source_after["crfs_trace"]
            ).get("exact_native_leaf_pairing")
            is True,
        }
        if not all(post_checks.values()):
            raise ReferenceTrajectoryLiftCanaryError(
                "post-lift references changed: "
                + ", ".join(name for name, passed in post_checks.items() if not passed)
            )
        expected_requests = EXACT_FINITE_REQUESTS
        if policy_index != expected_requests or [row["ordinal"] for row in ledger_rows] != list(
            range(expected_requests)
        ):
            raise ReferenceTrajectoryLiftCanaryError(
                f"policy request ledger has {policy_index}, expected {expected_requests}"
            )
        expected_phases = FINITE_REQUEST_PHASES
        if tuple(row["phase"] for row in ledger_rows) != expected_phases:
            raise ReferenceTrajectoryLiftCanaryError(
                "policy request ledger phase order changed"
            )
        if post_start != expected_phases.index("zero_schedule_post"):
            raise ReferenceTrajectoryLiftCanaryError(
                "post-reference request index changed"
            )

        budgeted_evaluations = [
            _evaluate_reply(reply, target_physical) for reply in budgeted_replies
        ]
        raw_evaluations = [
            _evaluate_reply(reply, target_physical) for reply in raw_replies
        ]
        arrays: dict[str, np.ndarray] = {
            "source_noise_f32": noise,
            "source_frozen_final_f32": frozen_final,
            "source_delta_model_f32": delta32,
            "source_target_normalized_f32": target,
            "source_model_to_physical_scale_f32": scale,
            "source_target_physical_f64": target_physical,
            "source_budget_f32": np.asarray(source_budget32, dtype=np.float32),
            "source_radius_f32": np.asarray(
                source_budget32 / np.float32(5), dtype=np.float32
            ),
            "source_dt_f32": np.asarray(DT_FLOAT32, dtype=np.float32),
            "reference_states_f32": reference,
            "reference_alpha_f32": alpha,
            "reference_compiled_actions_f64": _stack_actions(
                [compiled_before, compiled_after]
            ),
            "reference_source_actions_f64": _stack_actions(
                [source_before, source_after]
            ),
            "reference_eager_actions_f64": _stack_actions(
                [eager_before, eager_after]
            ),
            "reference_eager_final_f32": np.ascontiguousarray(
                np.stack(
                    [
                        np.asarray(
                            eager_before["crfs_trace"]["final_normalized"],
                            dtype=np.float32,
                        ),
                        np.asarray(
                            eager_after["crfs_trace"]["final_normalized"],
                            dtype=np.float32,
                        ),
                    ]
                )
            ),
            "raw_exact_budget_accepted_bool": np.asarray(
                replay_envelope["exact_source_budget_valid"], dtype=np.bool_
            ),
            "raw_path_f32": np.asarray(
                replay_envelope["path_float32"], dtype=np.float32
            ),
            "raw_max_step_norm_f32": np.asarray(
                replay_envelope["max_step_norm_float32"], dtype=np.float32
            ),
            "raw_replay_seed_f64": np.asarray(
                replay_envelope["replay_seed_float64"], dtype=np.float64
            ),
            "raw_replay_budget_f32": np.asarray(
                replay_envelope["replay_budget_float32"], dtype=np.float32
            ),
        }
        zero_compact = np.zeros((2, *COMPACT_SHAPE), dtype=np.float32)
        arm_a_c_group = np.broadcast_to(arm_a_c, (2, *COMPACT_SHAPE)).copy()
        arm_a_u_group = np.broadcast_to(arm_a_u, (2, *COMPACT_SHAPE)).copy()
        budgeted_c_group = _compact_group_trace(
            budgeted_replies, "reference_requested_increment_steps"
        )
        budgeted_u_group = _compact_group_trace(
            budgeted_replies, "reference_requested_velocity_steps"
        )
        raw_c_group = _compact_group_trace(
            raw_replies, "reference_requested_increment_steps"
        )
        raw_u_group = _compact_group_trace(
            raw_replies, "reference_requested_velocity_steps"
        )
        _add_transport_group(
            arrays,
            "zero",
            [zero_before, zero_after],
            requested_c=zero_compact,
            requested_velocity=zero_compact,
            canonical_positive_zero=True,
        )
        _add_transport_group(
            arrays,
            "arm_a",
            arm_a_replies,
            requested_c=arm_a_c_group,
            requested_velocity=arm_a_u_group,
        )
        _add_transport_group(
            arrays,
            "arm_b_generation",
            budgeted_replies,
            requested_c=budgeted_c_group,
            requested_velocity=budgeted_u_group,
            reference_generation=True,
        )
        _add_transport_group(
            arrays,
            "arm_b_replay",
            budgeted_replays,
            requested_c=budgeted_c_group,
            requested_velocity=_compact_group_trace(
                budgeted_replays, "control_velocity_steps"
            ),
        )
        _add_transport_group(
            arrays,
            "raw_generation",
            raw_replies,
            requested_c=raw_c_group,
            requested_velocity=raw_u_group,
            reference_generation=True,
        )
        _add_transport_group(
            arrays,
            "raw_replay",
            raw_replays,
            requested_c=raw_c_group,
            requested_velocity=_compact_group_trace(
                raw_replays, "control_velocity_steps"
            ),
        )
        _atomic_write_npz(tensor_path, arrays)
        tensor_hash = file_sha256(tensor_path)

        ledger_payload = {
            "schema_version": "1.0",
            "payload_type": LEDGER_TYPE,
            "case_id": CASE_ID,
            "branch": raw_branch,
            "exact_policy_request_count": policy_index,
            "rows": ledger_rows,
            "tensor_archive_sha256": tensor_hash,
        }
        atomic_write_json(ledger_path, ledger_payload)
        ledger_hash = file_sha256(ledger_path)
        bindings = actual_config_mapping["frozen_source_bindings"]
        source_bindings = {
            "case_id": CASE_ID,
            "group_id": GROUP_ID,
            "environment_seed": int(case["environment_seed"]),
            "policy_seed": int(case["policy_seed"]),
            "source_host": SOURCE_HOST,
            "manifest_sha256": str(bindings["manifest"]["sha256"]),
            "source_r02_sha256": str(bindings["source_r02"]["sha256"]),
            "source_r02_config_sha256": str(
                bindings["source_r02_config"]["sha256"]
            ),
            "checkpoint_model_sha256": str(
                bindings["checkpoint_model"]["sha256"]
            ),
            "checkpoint_config_sha256": str(
                bindings["checkpoint_config"]["sha256"]
            ),
            "normalization_asset_sha256": str(
                bindings["normalization_asset"]["sha256"]
            ),
            "baseline_revision": str(bindings["baseline_revision"]),
            "af00a_run_id": str(bindings["af00a_source_run"]["run_id"]),
            "af00a_tensor_sha256": AF_TENSOR_SHA256,
            "af00a_results_sha256": AF_RESULTS_SHA256,
        }
        payload = {
            "schema_version": "1.0",
            "payload_type": PAYLOAD_TYPE,
            "run_id": legacy_config.oracle.run_id,
            "status": "raw_evidence_complete",
            "branch": raw_branch,
            "case_id": CASE_ID,
            "group_id": GROUP_ID,
            "source_bindings": source_bindings,
            "source": {
                "r02_path": str(source_path),
                "r02_sha256": source_sha,
                "delta_star_model": delta_record,
                "source_budget_float64": source_budget64,
                "source_budget_float32": float(source_budget32),
                "pairing_checks": pairing_checks,
                "af00a_results_path": str(
                    actual_config_mapping["frozen_source_bindings"]["af00a_source_run"][
                        "results_json_path"
                    ]
                ),
                "af00a_results_sha256": AF_RESULTS_SHA256,
                "af00a_tensor_path": str(
                    actual_config_mapping["frozen_source_bindings"]["af00a_source_run"][
                        "tensor_archive_path"
                    ]
                ),
                "af00a_tensor_sha256": AF_TENSOR_SHA256,
                "af00a_source_checks": af_source_checks,
                "af00a_arm_a_checks": af_arm_a_checks,
            },
            "source_pairing_records": {
                "policy_observation_source": source_pairing.get("policy_observation"),
                "policy_observation_current": observation_fingerprint,
                "branch_snapshot_source": source_pairing.get("branch_snapshot"),
                "branch_snapshot_current": branch_snapshot,
                "eager_actions_source": source_pairing.get("eager_actions"),
                "eager_actions_before_current": _array_record(source_before["actions"]),
                "eager_actions_after_current": _array_record(source_after["actions"]),
                "eager_trace_source": source_pairing.get("eager_trace"),
                "eager_trace_before_current": _trace_record(source_before["crfs_trace"]),
                "eager_trace_after_current": _trace_record(source_after["crfs_trace"]),
            },
            "configs": {
                "actual_path": str(actual_path),
                "actual_sha256": file_sha256(actual_path),
                "legacy_path": str(legacy_path),
                "legacy_sha256": file_sha256(legacy_path),
                "repo_root": str(Path(repo_root).resolve()),
            },
            "request_accounting": {
                "exact": policy_index == EXACT_FINITE_REQUESTS,
                "total": policy_index,
                "finite_complete": True,
            },
            "reference": {
                "xbar": _bytes_digest(xbar),
                "states": _bytes_digest(reference),
                "alpha": _bytes_digest(alpha),
                "delta": _bytes_digest(delta32),
                "step_5_anchor_exact": _array_exact(reference[0], xbar[0]),
                "terminal_target_exact": _array_exact(reference[-1], target),
            },
            "arm_a": {
                "evaluation": {
                    "objective": arm_a_evaluation["objective"],
                    "metrics": list(arm_a_evaluation["metrics"]),
                    "gates": list(arm_a_evaluation["gates"]),
                },
                "historical_reproduction": af_arm_a_checks,
            },
            "budgeted": {
                "evaluations": [
                    {
                        "objective": value["objective"],
                        "metrics": list(value["metrics"]),
                        "gates": list(value["gates"]),
                    }
                    for value in budgeted_evaluations
                ],
                "duplicate_exact": True,
                "ordinary_replay_exact": True,
            },
            "raw": {
                "branch": raw_branch,
                "evaluations": [
                    {
                        "objective": value["objective"],
                        "metrics": list(value["metrics"]),
                        "gates": list(value["gates"]),
                    }
                    for value in raw_evaluations
                ],
                "replay_envelope": (
                    {
                        key: (
                            value.tolist()
                            if isinstance(value, np.ndarray)
                            else value.item()
                            if isinstance(value, np.generic)
                            else value
                        )
                        for key, value in replay_envelope.items()
                    }
                ),
                "duplicate_exact": True,
                "ordinary_replay_exact": True,
            },
            "post_checks": post_checks,
            "request_ledger": ledger_rows,
            "artifacts": {
                "tensor_archive": tensor_path.name,
                "tensor_archive_sha256": tensor_hash,
                "request_ledger": ledger_path.name,
                "request_ledger_sha256": ledger_hash,
                "results_json_published": False,
            },
            "provenance": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "host": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "execution_boundary": {
                "policy_requests": policy_index,
                "policy_generated_action_steps_executed": 0,
                "teacher_generated_action_steps_executed": 0,
                "efficacy_rollouts_executed": 0,
                "simulator_efficacy_evaluated": False,
                "geometry_or_planner_queries": 0,
                "training": False,
            },
        }
        atomic_write_json(payload_path, payload)
        if case_results_path.exists() or run_results_path.exists():
            raise ReferenceTrajectoryLiftCanaryError(
                "GPU runner illegally published results.json"
            )
        return payload_path, raw_branch
    except (ActualForwardSearchError, R05ACanaryPolicyError) as error:
        raise ReferenceTrajectoryLiftCanaryError(str(error)) from error
    finally:
        if owns_environment:
            environment.close()


__all__ = [
    "EXACT_FINITE_REQUESTS",
    "FINITE_REQUEST_PHASES",
    "LEDGER_FILENAME",
    "PAYLOAD_FILENAME",
    "ReferenceTrajectoryLiftCanaryError",
    "TENSOR_FILENAME",
    "construct_reference_trajectory",
    "raw_replay_envelope",
    "run_reference_trajectory_lift_canary",
]
