"""Independent NumPy validation for the TRL-00A reference-lift canary.

The H100 producer is not trusted to classify its own result.  This module has
no policy, PyTorch, simulator, optimizer, publication, or Slurm imports.  It
accepts only a fixed-shape no-pickle tensor archive, the exact request ledger,
and the frozen external-source binding.  It then reconstructs the reference,
online controls, fixed-order projection, float32 transport, ordinary Euler
recurrences, replay envelope, physical objective, gates, and outcome required
by ADR-0060.

Only a complete finite archive is scientifically valid.  A raw nonfinite,
exception, or typed terminal is an apparatus failure handled outside this
validator and cannot publish a scientific ``results.json``.  Missing or extra
members fail closed.  Stored GPU pass booleans and outcome strings are
intentionally not inputs to any decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"
CASE_ID = "crfs-1069f29a8d76463a"
GROUP_ID = "safelibero_spatial:II:0:46"
EXPECTED_BUDGET32 = np.float32(3.6398398876190186)
EXPECTED_DT32 = np.float32(-0.1)
EXPECTED_RADIUS32 = np.float32(EXPECTED_BUDGET32 / np.float32(5.0))
STEPS = 10
INTERVENTION_STEP = 5
ACTIVE_STEPS = 5
ACTION_SHAPE = (10, 32)
COMPACT_SHAPE = (5, 15)
METRIC_NAMES = ("xyz_max_abs", "xyz_rms", "full_max_abs", "full_rms")
FIDELITY_LIMITS64 = np.asarray((0.010, 0.005, 0.050, 0.015), dtype=np.float64)


class ReferenceTrajectoryLiftValidationError(ValueError):
    """TRL-00A raw evidence is incomplete or scientifically inconsistent."""


def _spec(shape: tuple[int, ...], dtype: str) -> tuple[tuple[int, ...], np.dtype[Any]]:
    return shape, np.dtype(dtype)


# These are immutable external identities, not producer-reported summaries.
EXPECTED_SOURCE_BINDINGS: dict[str, Any] = {
    "case_id": CASE_ID,
    "group_id": GROUP_ID,
    "environment_seed": 924805038,
    "policy_seed": 1179198633,
    "source_host": "worker-1",
    "manifest_sha256": "bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633",
    "source_r02_sha256": "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
    "source_r02_config_sha256": "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e",
    "checkpoint_model_sha256": "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
    "checkpoint_config_sha256": "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a",
    "normalization_asset_sha256": "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
    "baseline_revision": "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
    "af00a_run_id": "r05a-actual-forward-cem-canary-20260716c",
    "af00a_tensor_sha256": "ada53973653ba7c21dab33cdbc4b4498c7b2b1840f2fc284bd3d3c76baf1d383",
    "af00a_results_sha256": "507f25bc381b7bfeaa30abe3a7df970681f3b175f39221034c6e768e72dae914",
}


# The digest covers dtype.str, exact shape, and C-order bytes.  Binding the
# complete Arm-A recurrence prevents a new producer from merely copying the
# historical objective into a JSON summary while executing different bytes.
FROZEN_ARRAY_BINDINGS: dict[str, str] = {
    "source_noise_f32": "b2616f7872f2f4c27466c109e6ee7acf951bf9c9da508701baf8159fdbc62299",
    "source_frozen_final_f32": "199064fc9d51e24c1ee5723f1eb9f1d1349c83e43135992c31e2fbe488ac6066",
    "source_delta_model_f32": "7ef6e8a7af676fb03576449b7160f16e01d3f70f494ce7b0575161070c0685b0",
    "source_target_normalized_f32": "e1fc236904c7bbfbbc6090a8b3dca10ec6dbea818b93b2926ae9030201890a78",
    "source_model_to_physical_scale_f32": "35c1b02f3de60dea50e1f23694fe384e49e1f2198f48d42bcc5276f4140c335f",
    "source_target_physical_f64": "b023cae06f96fcbd0e2cd4df1ef10250d2b9b27f8c9f8fc1af229964618d1987",
    "source_budget_f32": "d2ef67395e3a574707c84aca7dd357045a21414e9779a7ceff50218f62143a7d",
    "source_dt_f32": "4f159cfac23f8d120db3c1d6e4abdca0919cb5d6632da7298483d863002840a6",
    "source_radius_f32": "12369533df916a3148983be19572e7adf5f8338f3ba8aeebdaee594ebf29d5c4",
    "arm_a_requested_c_f32": "a14ce4972ea93c2f022c7312d96b6429913d9ec3e4be7c04c0352e1022f13bd5",
    "arm_a_requested_velocity_f32": "01ac101e3827f3be55d14cddc9f0308afc20eb8ca1e8a766c716af8464201cd5",
    "arm_a_applied_velocity_f32": "01ac101e3827f3be55d14cddc9f0308afc20eb8ca1e8a766c716af8464201cd5",
    "arm_a_executed_c_f32": "c91234fa9be760127525fdc42f7999a17eec8fe40c6399a74884287254e40915",
    "arm_a_final_normalized_f32": "1dd2cff4f91b9028a47bb2cf2a96264804f1fd3b664115a79f2e9d1a3f0ff935",
    "arm_a_final_normalized_physical_f64": "b95c0272f7af9fec0b3d789cffbfdf8faf5a5fef3b73988d4271ed0f536af65c",
    "arm_a_returned_actions_f64": "b95c0272f7af9fec0b3d789cffbfdf8faf5a5fef3b73988d4271ed0f536af65c",
    "arm_a_trace_step_index_i64": "c02700288a210d7efbfed9f23849d7f4edd7d1ad93289f11bc7433d7b332aa7c",
    "arm_a_trace_time_f32": "ff21c0bff3f15e27a73953111a2ea598558d841f2c78fe8c7bf8a25d2eee0f09",
    "arm_a_trace_active_bool": "1fe78f7be8e5e0b5be79449c57bf7f6c8571c473b9c472ab96026db063df495c",
    "arm_a_trace_x_t_f32": "10449fb360292b9a54bd79def9ab8b99c339d2b58491b7118f94084abdaf418e",
    "arm_a_trace_v_base_f32": "358287800e83416135634dc7e114f2f89f568133b183c18f4b691ddc05807500",
    "arm_a_trace_total_velocity_f32": "8a8a9333d23e70c053a9f346840527b59c396cb09b010bb4083101cccff64099",
    "arm_a_trace_x_next_f32": "6a25f7fc93b6243ea8acacef1bccd7c7e81199a3a737ed47a8541943698ae57b",
    "arm_a_trace_initial_noise_f32": "e65888e5794683f187c189412385108199c3cd53a3c8687e7104fa29947ad84e",
}

EXPECTED_ARM_A_OBJECTIVE64 = np.float64(2218.1335502517986)
EXPECTED_ARM_A_METRICS64 = np.asarray(
    (0.7806643492412315, 0.35969860072305654, 0.7806643492412315, 0.23554385122689606),
    dtype=np.float64,
)


COMMON_ARRAY_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "source_noise_f32": _spec(ACTION_SHAPE, "<f4"),
    "source_frozen_final_f32": _spec(ACTION_SHAPE, "<f4"),
    "source_delta_model_f32": _spec(ACTION_SHAPE, "<f4"),
    "source_target_normalized_f32": _spec(ACTION_SHAPE, "<f4"),
    "source_model_to_physical_scale_f32": _spec(ACTION_SHAPE, "<f4"),
    "source_target_physical_f64": _spec((10, 7), "<f8"),
    "source_budget_f32": _spec((), "<f4"),
    "source_dt_f32": _spec((), "<f4"),
    "source_radius_f32": _spec((), "<f4"),
    "reference_alpha_f32": _spec((6,), "<f4"),
    "reference_states_f32": _spec((6, 10, 32), "<f4"),
    "reference_compiled_actions_f64": _spec((2, 10, 7), "<f8"),
    "reference_source_actions_f64": _spec((2, 10, 7), "<f8"),
    "reference_eager_actions_f64": _spec((2, 10, 7), "<f8"),
    "reference_eager_final_f32": _spec((2, 10, 32), "<f4"),
}

COMMON_GROUP_SIZES = {
    "zero": 2,
    "arm_a": 2,
    "arm_b_generation": 2,
    "arm_b_replay": 2,
}
FINITE_ONLY_GROUP_SIZES = {"raw_generation": 2, "raw_replay": 2}


def _add_group_specs(
    target: dict[str, tuple[tuple[int, ...], np.dtype[Any]]],
    prefix: str,
    count: int,
) -> None:
    target.update(
        {
            f"{prefix}_requested_c_f32": _spec((count, 5, 15), "<f4"),
            f"{prefix}_requested_velocity_f32": _spec((count, 5, 15), "<f4"),
            f"{prefix}_applied_velocity_f32": _spec((count, 5, 15), "<f4"),
            f"{prefix}_executed_c_f32": _spec((count, 5, 15), "<f4"),
            f"{prefix}_final_normalized_f32": _spec((count, 10, 32), "<f4"),
            f"{prefix}_final_normalized_physical_f64": _spec((count, 10, 7), "<f8"),
            f"{prefix}_returned_actions_f64": _spec((count, 10, 7), "<f8"),
            f"{prefix}_trace_step_index_i64": _spec((count, 10), "<i8"),
            f"{prefix}_trace_time_f32": _spec((count, 10), "<f4"),
            f"{prefix}_trace_active_bool": _spec((count, 10), "|b1"),
            f"{prefix}_trace_x_t_f32": _spec((count, 10, 10, 32), "<f4"),
            f"{prefix}_trace_v_base_f32": _spec((count, 10, 10, 32), "<f4"),
            f"{prefix}_trace_control_velocity_f32": _spec((count, 10, 10, 32), "<f4"),
            f"{prefix}_trace_total_velocity_f32": _spec((count, 10, 10, 32), "<f4"),
            f"{prefix}_trace_control_increment_f32": _spec((count, 10, 10, 32), "<f4"),
            f"{prefix}_trace_x_next_f32": _spec((count, 10, 10, 32), "<f4"),
            f"{prefix}_trace_initial_noise_f32": _spec((count, 10, 32), "<f4"),
        }
    )


REFERENCE_TRACE_SUFFIX_SPECS = {
    "reference_active_states_f32": _spec((2, 6, 10, 32), "<f4"),
    "reference_delta_f32": _spec((2, 10, 32), "<f4"),
    "reference_alpha_f32": _spec((2, 6), "<f4"),
    "reference_anchor_exact_bool": _spec((2,), "|b1"),
    "reference_projection_mode_i64": _spec((2,), "<i8"),
    "reference_source_budget_f32": _spec((2,), "<f4"),
    "reference_per_step_cap_f32": _spec((2,), "<f4"),
    "reference_desired_next_steps_f32": _spec((2, 10, 10, 32), "<f4"),
    "reference_uncontrolled_next_steps_f32": _spec((2, 10, 10, 32), "<f4"),
    "reference_raw_increment_steps_f32": _spec((2, 10, 10, 32), "<f4"),
    "reference_requested_increment_steps_f32": _spec((2, 10, 10, 32), "<f4"),
    "reference_requested_velocity_steps_f32": _spec((2, 10, 10, 32), "<f4"),
    "reference_executed_increment_steps_f32": _spec((2, 10, 10, 32), "<f4"),
    "reference_raw_norm_f64_steps": _spec((2, 10), "<f8"),
    "reference_requested_norm_f64_steps": _spec((2, 10), "<f8"),
    "reference_executed_norm_f64_steps": _spec((2, 10), "<f8"),
    "reference_projection_scale_f64_steps": _spec((2, 10), "<f8"),
    "reference_projected_steps_bool": _spec((2, 10), "|b1"),
    "reference_tracking_error_steps_f32": _spec((2, 10, 10, 32), "<f4"),
    "reference_raw_path_length_f64": _spec((2,), "<f8"),
    "reference_requested_path_length_f64": _spec((2,), "<f8"),
    "reference_executed_path_length_f64": _spec((2,), "<f8"),
    "reference_product_ball_constraint_applied_bool": _spec((2,), "|b1"),
    "reference_product_ball_valid_bool": _spec((2,), "|b1"),
}


for _prefix, _count in COMMON_GROUP_SIZES.items():
    _add_group_specs(COMMON_ARRAY_SPECS, _prefix, _count)
for _prefix in ("arm_b_generation",):
    for _suffix, _value in REFERENCE_TRACE_SUFFIX_SPECS.items():
        COMMON_ARRAY_SPECS[f"{_prefix}_{_suffix}"] = _value

FINITE_ARRAY_SPECS = dict(COMMON_ARRAY_SPECS)
for _prefix, _count in FINITE_ONLY_GROUP_SIZES.items():
    _add_group_specs(FINITE_ARRAY_SPECS, _prefix, _count)
for _suffix, _value in REFERENCE_TRACE_SUFFIX_SPECS.items():
    FINITE_ARRAY_SPECS[f"raw_generation_{_suffix}"] = _value
FINITE_ARRAY_SPECS.update(
    {
        "raw_exact_budget_accepted_bool": _spec((), "|b1"),
        "raw_path_f32": _spec((), "<f4"),
        "raw_max_step_norm_f32": _spec((), "<f4"),
        "raw_replay_seed_f64": _spec((), "<f8"),
        "raw_replay_budget_f32": _spec((), "<f4"),
    }
)

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceTrajectoryLiftValidationError(message)


def _bytes_equal(left: Any, right: Any) -> bool:
    first = np.asarray(left)
    second = np.asarray(right)
    return bool(
        first.shape == second.shape
        and first.dtype == second.dtype
        and np.ascontiguousarray(first).tobytes()
        == np.ascontiguousarray(second).tobytes()
    )


def _array_sha256(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _bytes_digest(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }


def _load_arrays(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    keys = set(raw)
    specs = FINITE_ARRAY_SPECS
    _require(
        keys == set(specs),
        "TRL-00A finite tensor members changed: "
        f"missing={sorted(set(specs) - keys)}, extra={sorted(keys - set(specs))}",
    )
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in specs.items():
        value = np.asarray(raw[name])
        _require(value.shape == shape, f"{name} must have shape {shape}, got {value.shape}")
        _require(value.dtype == dtype, f"{name} must preserve {dtype}, got {value.dtype}")
        _require(
            not value.dtype.hasobject and value.dtype.fields is None,
            f"{name} must be a plain no-pickle array",
        )
        if np.issubdtype(value.dtype, np.floating):
            _require(bool(np.isfinite(value).all()), f"{name} contains a nonfinite value")
        # ``np.ascontiguousarray`` promotes a zero-dimensional scalar to
        # shape ``(1,)``.  Preserve the registered scalar shape while still
        # copying every archive member out of the NPZ handle.
        arrays[name] = np.array(value, dtype=value.dtype, copy=True, order="C")
    return arrays


def _validate_source_bindings(source_bindings: Mapping[str, Any]) -> None:
    _require(isinstance(source_bindings, Mapping), "source bindings must be an object")
    _require(
        set(source_bindings) == set(EXPECTED_SOURCE_BINDINGS),
        "source binding keys changed",
    )
    for key, expected in EXPECTED_SOURCE_BINDINGS.items():
        observed = source_bindings.get(key)
        _require(
            observed == expected and type(observed) is type(expected),
            f"source binding {key} changed",
        )


def _validate_frozen_arrays(
    arrays: Mapping[str, np.ndarray], expected_bindings: Mapping[str, str]
) -> None:
    _require(
        set(expected_bindings) == set(FROZEN_ARRAY_BINDINGS),
        "frozen source/Arm-A binding names changed",
    )
    for name, expected in expected_bindings.items():
        _require(name in arrays, f"frozen bound array {name} is absent")
        _require(_array_sha256(arrays[name]) == expected, f"frozen bound array {name} changed")


def _expected_alpha() -> np.ndarray:
    alpha = np.zeros((6,), dtype=np.float32)
    for index in range(1, 6):
        alpha[index] = np.asarray(
            np.float32(index) / np.float32(5), dtype=np.float32
        )
    return alpha


def _mask() -> np.ndarray:
    mask = np.zeros(ACTION_SHAPE, dtype=np.bool_)
    mask[:5, :3] = True
    return mask


def _expected_times() -> np.ndarray:
    result = np.empty((10,), dtype=np.float32)
    current = np.float32(1.0)
    for index in range(10):
        result[index] = current
        current = np.asarray(current + EXPECTED_DT32, dtype=np.float32)
    return result


def _fixed_norm64(block: np.ndarray) -> np.float64:
    values = np.asarray(block, dtype=np.float32).reshape(15)
    total = np.float64(0.0)
    for coordinate in range(15):
        item = np.float64(values[coordinate])
        product = np.float64(item * item)
        total = np.float64(total + product)
    return np.asarray(np.sqrt(total), dtype=np.float64).reshape(())[()]


def _project_block(raw: np.ndarray, radius32: np.float32) -> tuple[np.ndarray, np.float64, np.float64, bool]:
    value = np.asarray(raw, dtype=np.float32).reshape(15)
    norm64 = _fixed_norm64(value)
    if norm64 <= np.float64(radius32):
        return value.copy(), norm64, np.float64(1.0), False
    scale64 = np.asarray(np.float64(radius32) / norm64, dtype=np.float64).reshape(())[()]
    projected = np.asarray(
        np.asarray(value, dtype=np.float64) * scale64, dtype=np.float32
    )
    return projected, norm64, scale64, True


def _allowed_after_ulps(value: np.float32, count: int = 8) -> np.float32:
    result = np.float32(value)
    for _ in range(count):
        result = np.nextafter(result, np.float32(np.inf), dtype=np.float32)
    return result


def _constraint_values(executed: np.ndarray) -> tuple[np.ndarray, np.float32]:
    value = np.asarray(executed, dtype=np.float32)
    norms = np.linalg.norm(value, axis=1).astype(np.float32)
    path = np.asarray(np.sum(norms, dtype=np.float32), dtype=np.float32).reshape(())[()]
    return norms, path


def _constraints_accept(executed: np.ndarray, budget32: np.float32) -> bool:
    norms, path = _constraint_values(executed)
    radius32 = np.asarray(budget32 / np.float32(5.0), dtype=np.float32).reshape(())[()]
    return bool(
        np.all(norms <= _allowed_after_ulps(radius32))
        and path <= _allowed_after_ulps(np.float32(budget32))
    )


def _require_constraints(executed: np.ndarray, budget32: np.float32, *, name: str) -> None:
    _require(_constraints_accept(executed, budget32), f"{name} violates the unchanged eight-ULP constraints")


def _upward_float32(seed64: np.float64) -> np.float32:
    _require(bool(np.isfinite(seed64)) and seed64 >= 0, "raw replay seed must be finite and nonnegative")
    value = np.float32(seed64)
    if np.float64(value) < seed64:
        value = np.nextafter(value, np.float32(np.inf), dtype=np.float32)
    _require(bool(np.isfinite(value)) and np.float64(value) >= seed64, "raw replay envelope is not float32 representable")
    return value


def _objective_metrics(
    actions: np.ndarray, target: np.ndarray
) -> tuple[np.float64, np.ndarray, np.ndarray]:
    physical = np.asarray(actions, dtype=np.float64)
    wanted = np.asarray(target, dtype=np.float64)
    _require(physical.shape == (10, 7) and wanted.shape == (10, 7), "physical fidelity inputs must be 10x7")
    error = np.asarray(physical[:5, :7] - wanted[:5, :7], dtype=np.float64)
    weighted = np.concatenate(
        (
            np.asarray(error[:, :3] / np.float64(0.005), dtype=np.float64).reshape(-1),
            np.asarray(error[:, 3:7] / np.float64(0.015), dtype=np.float64).reshape(-1),
        )
    )
    objective = np.asarray(np.mean(np.square(weighted), dtype=np.float64), dtype=np.float64).reshape(())[()]
    xyz = error[:, :3]
    metrics = np.asarray(
        (
            np.max(np.abs(xyz)),
            np.sqrt(np.mean(np.square(xyz), dtype=np.float64), dtype=np.float64),
            np.max(np.abs(error)),
            np.sqrt(np.mean(np.square(error), dtype=np.float64), dtype=np.float64),
        ),
        dtype=np.float64,
    )
    return objective, metrics, np.asarray(metrics <= FIDELITY_LIMITS64, dtype=np.bool_)


def _full_schedule(compact_velocity: np.ndarray) -> np.ndarray:
    compact = np.asarray(compact_velocity)
    _require(compact.shape == COMPACT_SHAPE and compact.dtype == np.dtype(np.float32), "compact schedule changed")
    result = np.zeros((10, 10, 32), dtype=np.float32)
    result[5:10, :5, :3] = compact.reshape(5, 5, 3)
    return result


def _expanded_control(compact: np.ndarray) -> np.ndarray:
    value = np.asarray(compact, dtype=np.float32)
    _require(value.ndim == 3 and value.shape[1:] == COMPACT_SHAPE, "compact control group changed")
    result = np.zeros((value.shape[0], 10, 10, 32), dtype=np.float32)
    result[:, 5:10, :5, :3] = value.reshape(value.shape[0], 5, 5, 3)
    return result


def _validate_transport_group(
    arrays: Mapping[str, np.ndarray], prefix: str, *, budget32: np.float32 | None
) -> None:
    requested_c = arrays[f"{prefix}_requested_c_f32"]
    requested_u = arrays[f"{prefix}_requested_velocity_f32"]
    applied_u = arrays[f"{prefix}_applied_velocity_f32"]
    executed_c = arrays[f"{prefix}_executed_c_f32"]
    if prefix == "zero":
        positive_zero = np.zeros_like(requested_c, dtype=np.float32)
        for name, value in (
            ("requested c", requested_c),
            ("requested velocity", requested_u),
            ("applied velocity", applied_u),
            ("executed c", executed_c),
        ):
            _require(
                _bytes_equal(value, positive_zero),
                f"zero {name} is not canonical positive zero",
            )
        return
    expected_u = np.asarray(requested_c / EXPECTED_DT32, dtype=np.float32)
    _require(_bytes_equal(requested_u, expected_u), f"{prefix} c-to-u float32 transport changed")
    _require(_bytes_equal(applied_u, requested_u), f"{prefix} applied velocity differs from requested velocity")
    expected_executed = np.asarray(EXPECTED_DT32 * applied_u, dtype=np.float32)
    _require(_bytes_equal(executed_c, expected_executed), f"{prefix} authoritative executed c differs from float32 dt*u")
    if budget32 is not None:
        for index in range(executed_c.shape[0]):
            _require_constraints(executed_c[index], budget32, name=f"{prefix}[{index}]")


ORDINARY_TRACE_SUFFIXES = (
    "trace_step_index_i64",
    "trace_time_f32",
    "trace_active_bool",
    "trace_x_t_f32",
    "trace_v_base_f32",
    "trace_control_velocity_f32",
    "trace_total_velocity_f32",
    "trace_control_increment_f32",
    "trace_x_next_f32",
    "trace_initial_noise_f32",
    "requested_c_f32",
    "requested_velocity_f32",
    "applied_velocity_f32",
    "executed_c_f32",
    "final_normalized_f32",
    "final_normalized_physical_f64",
    "returned_actions_f64",
)


def _validate_recurrence_group(
    arrays: Mapping[str, np.ndarray], prefix: str, source_noise: np.ndarray
) -> None:
    count = arrays[f"{prefix}_requested_c_f32"].shape[0]
    steps = np.broadcast_to(np.arange(10, dtype=np.int64), (count, 10)).copy()
    times = np.broadcast_to(_expected_times(), (count, 10)).copy()
    active = np.broadcast_to(
        np.asarray([False] * 5 + [True] * 5, dtype=np.bool_), (count, 10)
    ).copy()
    initial = np.broadcast_to(np.asarray(source_noise, dtype=np.float32), (count, 10, 32)).copy()
    _require(_bytes_equal(arrays[f"{prefix}_trace_step_index_i64"], steps), f"{prefix} step indices changed")
    _require(_bytes_equal(arrays[f"{prefix}_trace_time_f32"], times), f"{prefix} times changed")
    _require(_bytes_equal(arrays[f"{prefix}_trace_active_bool"], active), f"{prefix} active flags changed")
    _require(_bytes_equal(arrays[f"{prefix}_trace_initial_noise_f32"], initial), f"{prefix} initial noise changed")

    control = _expanded_control(arrays[f"{prefix}_applied_velocity_f32"])
    _require(_bytes_equal(arrays[f"{prefix}_trace_control_velocity_f32"], control), f"{prefix} full control trace changed")
    v_base = arrays[f"{prefix}_trace_v_base_f32"]
    expected_total = np.where(control == np.float32(0.0), v_base, v_base + control)
    expected_increment = np.asarray(EXPECTED_DT32 * control, dtype=np.float32)
    x_t = arrays[f"{prefix}_trace_x_t_f32"]
    expected_next = np.asarray(x_t + EXPECTED_DT32 * expected_total, dtype=np.float32)
    _require(_bytes_equal(arrays[f"{prefix}_trace_total_velocity_f32"], expected_total), f"{prefix} total velocity violates zero-preserving addition")
    _require(_bytes_equal(arrays[f"{prefix}_trace_control_increment_f32"], expected_increment), f"{prefix} control increment is not float32 dt*u")
    _require(_bytes_equal(arrays[f"{prefix}_trace_x_next_f32"], expected_next), f"{prefix} Euler recurrence changed")
    _require(_bytes_equal(x_t[:, 0], initial), f"{prefix} first state differs from paired noise")
    _require(_bytes_equal(x_t[:, 1:], expected_next[:, :-1]), f"{prefix} recurrence states are disconnected")
    _require(_bytes_equal(arrays[f"{prefix}_final_normalized_f32"], expected_next[:, -1]), f"{prefix} final normalized action differs from recurrence")
    _require(
        _bytes_equal(
            arrays[f"{prefix}_final_normalized_physical_f64"],
            arrays[f"{prefix}_returned_actions_f64"],
        ),
        f"{prefix} physical terminal trace differs from returned actions",
    )


def _require_duplicate_group(arrays: Mapping[str, np.ndarray], prefix: str) -> None:
    for suffix in ORDINARY_TRACE_SUFFIXES:
        value = arrays[f"{prefix}_{suffix}"]
        _require(_bytes_equal(value[0], value[1]), f"{prefix} duplicate {suffix} changed")
    if prefix.endswith("generation"):
        for suffix in REFERENCE_TRACE_SUFFIX_SPECS:
            value = arrays[f"{prefix}_{suffix}"]
            _require(_bytes_equal(value[0], value[1]), f"{prefix} duplicate {suffix} changed")


def _require_ordinary_replay(
    arrays: Mapping[str, np.ndarray], generation: str, replay: str
) -> None:
    for replay_index in range(2):
        for suffix in ORDINARY_TRACE_SUFFIXES:
            _require(
                _bytes_equal(arrays[f"{generation}_{suffix}"][0], arrays[f"{replay}_{suffix}"][replay_index]),
                f"{replay}[{replay_index}] {suffix} differs from generated fixed-schedule witness",
            )


def _validate_identical_schedule_outputs(arrays: Mapping[str, np.ndarray]) -> None:
    """Require deterministic output whenever two ordinary schedules coincide."""

    groups = (*COMMON_GROUP_SIZES, *FINITE_ONLY_GROUP_SIZES)
    observed: dict[bytes, tuple[bytes, ...]] = {}
    provenance: dict[bytes, str] = {}
    output_suffixes = (
        "trace_step_index_i64",
        "trace_time_f32",
        "trace_active_bool",
        "trace_x_t_f32",
        "trace_v_base_f32",
        "trace_control_velocity_f32",
        "trace_total_velocity_f32",
        "trace_control_increment_f32",
        "trace_x_next_f32",
        "trace_initial_noise_f32",
        "applied_velocity_f32",
        "executed_c_f32",
        "final_normalized_f32",
        "final_normalized_physical_f64",
        "returned_actions_f64",
    )
    for prefix in groups:
        count = arrays[f"{prefix}_requested_velocity_f32"].shape[0]
        for index in range(count):
            key = np.ascontiguousarray(
                arrays[f"{prefix}_requested_velocity_f32"][index]
            ).tobytes()
            output = tuple(
                np.ascontiguousarray(arrays[f"{prefix}_{suffix}"][index]).tobytes()
                for suffix in output_suffixes
            )
            name = f"{prefix}[{index}]"
            if key in observed:
                _require(
                    observed[key] == output,
                    f"identical schedule output changed between {provenance[key]} and {name}",
                )
            else:
                observed[key] = output
                provenance[key] = name


def _validate_reference_generation(
    arrays: Mapping[str, np.ndarray], prefix: str, *, projection_mode: int
) -> np.ndarray:
    count = 2
    expected_reference = np.broadcast_to(arrays["reference_states_f32"], (count, 6, 10, 32)).copy()
    expected_delta = np.broadcast_to(arrays["source_delta_model_f32"], (count, 10, 32)).copy()
    expected_alpha = np.broadcast_to(arrays["reference_alpha_f32"], (count, 6)).copy()
    _require(_bytes_equal(arrays[f"{prefix}_reference_active_states_f32"], expected_reference), f"{prefix} supplied full reference changed")
    _require(_bytes_equal(arrays[f"{prefix}_reference_delta_f32"], expected_delta), f"{prefix} supplied delta changed")
    _require(_bytes_equal(arrays[f"{prefix}_reference_alpha_f32"], expected_alpha), f"{prefix} alpha changed")
    _require(bool(np.all(arrays[f"{prefix}_reference_anchor_exact_bool"])), f"{prefix} live step-5 reference anchor failed")
    _require(_bytes_equal(arrays[f"{prefix}_reference_projection_mode_i64"], np.full((2,), projection_mode, dtype=np.int64)), f"{prefix} projection mode changed")
    _require(_bytes_equal(arrays[f"{prefix}_reference_source_budget_f32"], np.full((2,), EXPECTED_BUDGET32, dtype=np.float32)), f"{prefix} source budget changed")
    _require(_bytes_equal(arrays[f"{prefix}_reference_per_step_cap_f32"], np.full((2,), EXPECTED_RADIUS32, dtype=np.float32)), f"{prefix} radius changed")

    x_t = arrays[f"{prefix}_trace_x_t_f32"]
    v_base = arrays[f"{prefix}_trace_v_base_f32"]
    desired = arrays[f"{prefix}_reference_desired_next_steps_f32"]
    uncontrolled = arrays[f"{prefix}_reference_uncontrolled_next_steps_f32"]
    raw = arrays[f"{prefix}_reference_raw_increment_steps_f32"]
    requested = arrays[f"{prefix}_reference_requested_increment_steps_f32"]
    requested_u = arrays[f"{prefix}_reference_requested_velocity_steps_f32"]
    executed = arrays[f"{prefix}_reference_executed_increment_steps_f32"]
    raw_norm = arrays[f"{prefix}_reference_raw_norm_f64_steps"]
    requested_norm = arrays[f"{prefix}_reference_requested_norm_f64_steps"]
    executed_norm = arrays[f"{prefix}_reference_executed_norm_f64_steps"]
    scale = arrays[f"{prefix}_reference_projection_scale_f64_steps"]
    projected = arrays[f"{prefix}_reference_projected_steps_bool"]
    tracking = arrays[f"{prefix}_reference_tracking_error_steps_f32"]
    mask = _mask()
    expected_raw_path = np.zeros((2,), dtype=np.float64)
    expected_requested_path = np.zeros((2,), dtype=np.float64)
    expected_executed_path = np.zeros((2,), dtype=np.float64)

    for replicate in range(2):
        for step in range(10):
            expected_uncontrolled = np.asarray(
                x_t[replicate, step] + EXPECTED_DT32 * v_base[replicate, step],
                dtype=np.float32,
            )
            _require(_bytes_equal(uncontrolled[replicate, step], expected_uncontrolled), f"{prefix}[{replicate}] step {step} uncontrolled proposal changed")
            if step < 5:
                positive_zero = np.zeros(ACTION_SHAPE, dtype=np.float32)
                for leaf_name, leaf in (
                    ("desired", desired),
                    ("raw", raw),
                    ("requested", requested),
                    ("requested velocity", requested_u),
                    ("executed", executed),
                    ("tracking", tracking),
                ):
                    _require(_bytes_equal(leaf[replicate, step], positive_zero), f"{prefix}[{replicate}] inactive {leaf_name} is not positive zero")
                _require(raw_norm[replicate, step] == np.float64(0.0), f"{prefix} inactive raw norm changed")
                _require(requested_norm[replicate, step] == np.float64(0.0), f"{prefix} inactive requested norm changed")
                _require(executed_norm[replicate, step] == np.float64(0.0), f"{prefix} inactive executed norm changed")
                _require(scale[replicate, step] == np.float64(1.0) and not bool(projected[replicate, step]), f"{prefix} inactive projection metadata changed")
                continue

            active = step - 5
            expected_desired = arrays["reference_states_f32"][active + 1]
            # The supplied state is already xbar + alpha*Delta.  Comparing
            # directly here makes adding Delta a second time impossible.
            _require(_bytes_equal(desired[replicate, step], expected_desired), f"{prefix}[{replicate}] step {step} did not consume the full reference directly")
            expected_raw = np.zeros(ACTION_SHAPE, dtype=np.float32)
            difference = np.asarray(expected_desired - expected_uncontrolled, dtype=np.float32)
            expected_raw[mask] = difference[mask]
            _require(_bytes_equal(raw[replicate, step], expected_raw), f"{prefix}[{replicate}] step {step} raw online increment changed")

            compact_raw = expected_raw[:5, :3].reshape(15)
            if projection_mode == 1:
                compact_requested, expected_raw_norm, expected_scale, expected_projected = _project_block(compact_raw, EXPECTED_RADIUS32)
            else:
                compact_requested = compact_raw.copy()
                expected_raw_norm = _fixed_norm64(compact_raw)
                expected_scale = np.float64(1.0)
                expected_projected = False
            expected_requested = np.zeros(ACTION_SHAPE, dtype=np.float32)
            expected_requested[:5, :3] = compact_requested.reshape(5, 3)
            # ADR-0060 converts only the mask.  Dividing the full positive-zero
            # tensor by negative dt would fabricate -0 outside the mask and
            # make the generated field differ bytewise from ordinary replay.
            expected_u = np.zeros(ACTION_SHAPE, dtype=np.float32)
            expected_u[mask] = np.asarray(
                expected_requested[mask] / EXPECTED_DT32, dtype=np.float32
            )
            expected_executed = np.zeros(ACTION_SHAPE, dtype=np.float32)
            expected_executed[mask] = np.asarray(
                EXPECTED_DT32 * expected_u[mask], dtype=np.float32
            )
            expected_requested_norm = _fixed_norm64(expected_requested[:5, :3])
            expected_executed_norm = _fixed_norm64(expected_executed[:5, :3])
            expected_tracking = np.zeros(ACTION_SHAPE, dtype=np.float32)
            expected_tracking[mask] = np.asarray(
                arrays[f"{prefix}_trace_x_next_f32"][replicate, step] - expected_desired,
                dtype=np.float32,
            )[mask]

            _require(_bytes_equal(requested[replicate, step], expected_requested), f"{prefix}[{replicate}] step {step} projection changed")
            _require(_bytes_equal(requested_u[replicate, step], expected_u), f"{prefix}[{replicate}] step {step} c-to-u changed")
            _require(_bytes_equal(executed[replicate, step], expected_executed), f"{prefix}[{replicate}] step {step} authoritative dt*u changed")
            _require(raw_norm[replicate, step] == expected_raw_norm, f"{prefix}[{replicate}] step {step} fixed-order raw norm changed")
            _require(requested_norm[replicate, step] == expected_requested_norm, f"{prefix}[{replicate}] step {step} requested norm changed")
            _require(executed_norm[replicate, step] == expected_executed_norm, f"{prefix}[{replicate}] step {step} executed norm changed")
            _require(scale[replicate, step] == expected_scale, f"{prefix}[{replicate}] step {step} projection scale changed")
            _require(bool(projected[replicate, step]) is expected_projected, f"{prefix}[{replicate}] step {step} projection decision changed")
            _require(_bytes_equal(tracking[replicate, step], expected_tracking), f"{prefix}[{replicate}] step {step} tracking error changed")
            expected_raw_path[replicate] = np.float64(expected_raw_path[replicate] + expected_raw_norm)
            expected_requested_path[replicate] = np.float64(expected_requested_path[replicate] + expected_requested_norm)
            expected_executed_path[replicate] = np.float64(expected_executed_path[replicate] + expected_executed_norm)

    _require(_bytes_equal(arrays[f"{prefix}_reference_raw_path_length_f64"], expected_raw_path), f"{prefix} raw path changed")
    _require(_bytes_equal(arrays[f"{prefix}_reference_requested_path_length_f64"], expected_requested_path), f"{prefix} requested path changed")
    _require(_bytes_equal(arrays[f"{prefix}_reference_executed_path_length_f64"], expected_executed_path), f"{prefix} executed path changed")
    _require(_bytes_equal(arrays[f"{prefix}_reference_requested_velocity_steps_f32"], arrays[f"{prefix}_trace_control_velocity_f32"]), f"{prefix} reference velocity differs from ordinary trace")
    reference_executed = arrays[
        f"{prefix}_reference_executed_increment_steps_f32"
    ]
    generic_increment = arrays[f"{prefix}_trace_control_increment_f32"]
    active_mask = np.zeros((10, 10, 32), dtype=np.bool_)
    active_mask[5:, :5, :3] = True
    expanded_mask = np.broadcast_to(active_mask, reference_executed.shape)
    _require(
        _bytes_equal(reference_executed[expanded_mask], generic_increment[expanded_mask]),
        f"{prefix} authoritative executed increments differ on the control mask",
    )
    _require(
        _bytes_equal(
            reference_executed[~expanded_mask],
            np.zeros(reference_executed[~expanded_mask].shape, dtype=np.float32),
        ),
        f"{prefix} specialized executed increments are not positive zero outside the mask",
    )
    applied = np.asarray([projection_mode == 1] * 2, dtype=np.bool_)
    valid = np.asarray([projection_mode == 1] * 2, dtype=np.bool_)
    _require(_bytes_equal(arrays[f"{prefix}_reference_product_ball_constraint_applied_bool"], applied), f"{prefix} product-ball applied metadata changed")
    _require(_bytes_equal(arrays[f"{prefix}_reference_product_ball_valid_bool"], valid), f"{prefix} product-ball valid metadata changed")
    return raw_norm[:, 5:].copy()


def _reconstruct_and_validate_reference(arrays: Mapping[str, np.ndarray]) -> None:
    budget = arrays["source_budget_f32"].reshape(())[()]
    dt = arrays["source_dt_f32"].reshape(())[()]
    radius = arrays["source_radius_f32"].reshape(())[()]
    _require(np.asarray(budget, dtype=np.float32).tobytes() == EXPECTED_BUDGET32.tobytes(), "source budget bytes changed")
    _require(np.asarray(dt, dtype=np.float32).tobytes() == EXPECTED_DT32.tobytes(), "source dt bytes changed")
    _require(np.asarray(radius, dtype=np.float32).tobytes() == EXPECTED_RADIUS32.tobytes(), "source radius bytes changed")
    delta = arrays["source_delta_model_f32"]
    mask = _mask()
    outside = delta[~mask]
    _require(not bool(np.count_nonzero(outside)) and not bool(np.signbit(outside).any()), "source delta must be positive zero outside first-five XYZ")
    target = arrays["source_frozen_final_f32"].copy()
    summed = np.asarray(target + delta, dtype=np.float32)
    changed = mask & (delta != np.float32(0.0))
    target[changed] = summed[changed]
    _require(_bytes_equal(arrays["source_target_normalized_f32"], target), "normalized target reconstruction changed")

    alpha = _expected_alpha()
    _require(_bytes_equal(arrays["reference_alpha_f32"], alpha), "reference alpha bytes changed")
    zero_x = arrays["zero_trace_x_t_f32"][0]
    zero_next = arrays["zero_trace_x_next_f32"][0]
    xbar = np.empty((11, 10, 32), dtype=np.float32)
    xbar[0] = zero_x[0]
    xbar[1:] = zero_next
    expected = np.empty((6, 10, 32), dtype=np.float32)
    for active in range(6):
        state = xbar[5 + active].copy()
        weighted = np.asarray(alpha[active] * delta, dtype=np.float32)
        shifted = np.asarray(state + weighted, dtype=np.float32)
        update = mask & (weighted != np.float32(0.0))
        state[update] = shifted[update]
        expected[active] = state
    _require(_bytes_equal(arrays["reference_states_f32"], expected), "reference reconstruction from the fresh zero trace changed")
    _require(_bytes_equal(expected[0], xbar[5]), "reference step-5 anchor changed")
    _require(_bytes_equal(expected[-1][mask], target[mask]), "reference endpoint does not match target on the control mask")
    _require(_bytes_equal(expected[:, ~mask], xbar[5:11, ~mask]), "reference changed outside-mask or signed-zero bytes")


def _validate_reference_and_arm_a(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    compiled = arrays["reference_compiled_actions_f64"]
    source = arrays["reference_source_actions_f64"]
    eager = arrays["reference_eager_actions_f64"]
    eager_final = arrays["reference_eager_final_f32"]
    for name, pair in (("compiled", compiled), ("source", source), ("eager", eager), ("eager final", eager_final)):
        _require(_bytes_equal(pair[0], pair[1]), f"pre/post {name} reference changed")
    _require(_bytes_equal(source, eager), "eager source and eager normalized-final actions differ")
    _require(_bytes_equal(eager_final[0], arrays["source_frozen_final_f32"]), "fresh frozen final changed")
    target_physical = eager[0].copy()
    target_physical[:5, :3] += np.asarray(arrays["source_delta_model_f32"][:5, :3], dtype=np.float64) * np.asarray(
        arrays["source_model_to_physical_scale_f32"][:5, :3], dtype=np.float64
    )
    _require(_bytes_equal(arrays["source_target_physical_f64"], target_physical), "physical target must use scale only")

    positive_zero = np.zeros((2, 5, 15), dtype=np.float32)
    for suffix in ("requested_c_f32", "requested_velocity_f32", "applied_velocity_f32", "executed_c_f32"):
        _require(_bytes_equal(arrays[f"zero_{suffix}"], positive_zero), f"zero {suffix} is not positive zero")
    _require(_bytes_equal(arrays["zero_final_normalized_f32"][0], arrays["source_frozen_final_f32"]), "zero final differs from frozen final")
    _require(_bytes_equal(arrays["zero_returned_actions_f64"][0], eager[0]), "zero returned action differs from frozen action")

    compact_delta = arrays["source_delta_model_f32"][:5, :3].reshape(15)
    row = np.asarray(compact_delta / np.float32(5.0), dtype=np.float32)
    expected_a = np.broadcast_to(row, (2, 5, 15)).copy()
    _require(_bytes_equal(arrays["arm_a_requested_c_f32"], expected_a), "Arm A requested equal split changed")
    objective, metrics, gates = _objective_metrics(arrays["arm_a_returned_actions_f64"][0], target_physical)
    _require(objective == EXPECTED_ARM_A_OBJECTIVE64, "Arm A historical objective changed")
    _require(_bytes_equal(metrics, EXPECTED_ARM_A_METRICS64), "Arm A historical metrics changed")
    _require(not bool(np.any(gates)), "Arm A historical failed-gate pattern changed")
    return target_physical, gates


def _ledger_action(arrays: Mapping[str, np.ndarray], ordinal: int) -> np.ndarray:
    if ordinal == 0:
        return arrays["reference_compiled_actions_f64"][0]
    if ordinal == 1:
        return arrays["reference_source_actions_f64"][0]
    if ordinal == 2:
        return arrays["reference_eager_actions_f64"][0]
    if ordinal == 3:
        return arrays["zero_returned_actions_f64"][0]
    if ordinal in (4, 5):
        return arrays["arm_a_returned_actions_f64"][ordinal - 4]
    if ordinal in (6, 7):
        return arrays["arm_b_generation_returned_actions_f64"][ordinal - 6]
    if ordinal in (8, 9):
        return arrays["arm_b_replay_returned_actions_f64"][ordinal - 8]
    if ordinal in (10, 11):
        return arrays["raw_generation_returned_actions_f64"][ordinal - 10]
    if ordinal in (12, 13):
        return arrays["raw_replay_returned_actions_f64"][ordinal - 12]
    post = ordinal - 14
    if post == 0:
        return arrays["zero_returned_actions_f64"][1]
    if post == 1:
        return arrays["reference_eager_actions_f64"][1]
    if post == 2:
        return arrays["reference_source_actions_f64"][1]
    if post == 3:
        return arrays["reference_compiled_actions_f64"][1]
    raise ReferenceTrajectoryLiftValidationError("request ordinal escaped branch ledger")


def _ledger_phase(ordinal: int) -> tuple[str, int, str | None, int | None]:
    fixed = {
        0: ("compiled_frozen_pre", 0, None, None),
        1: ("eager_source_trace_pre", 0, None, None),
        2: ("eager_normalized_final_pre", 0, None, None),
        3: ("zero_schedule_pre", 0, "zero", 0),
        4: ("arm_a_equal_split", 0, "arm_a", 0),
        5: ("arm_a_equal_split", 1, "arm_a", 1),
        6: ("budgeted_lift_generation", 0, "arm_b_generation", 0),
        7: ("budgeted_lift_generation", 1, "arm_b_generation", 1),
        8: ("budgeted_lift_replay", 0, "arm_b_replay", 0),
        9: ("budgeted_lift_replay", 1, "arm_b_replay", 1),
    }
    if ordinal in fixed:
        return fixed[ordinal]
    if ordinal in (10, 11):
        return "raw_lift_generation", ordinal - 10, "raw_generation", ordinal - 10
    if ordinal in (12, 13):
        return "raw_lift_replay", ordinal - 12, "raw_replay", ordinal - 12
    post_start = 14
    post_names = (
        "zero_schedule_post",
        "eager_normalized_final_post",
        "eager_source_trace_post",
        "compiled_frozen_post",
    )
    offset = ordinal - post_start
    if 0 <= offset < 4:
        return post_names[offset], 0, "zero" if offset == 0 else None, 1 if offset == 0 else None
    raise ReferenceTrajectoryLiftValidationError("request ordinal escaped branch ledger")


def validate_request_ledger(
    ledger: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray]
) -> None:
    _require(not isinstance(ledger, (str, bytes)), "request ledger must be a sequence")
    _require(len(ledger) == 18, "finite request ledger must contain exactly 18 rows")
    for ordinal, row in enumerate(ledger):
        _require(isinstance(row, Mapping), f"request ledger row {ordinal} is not an object")
        phase, phase_index, trace_group, trace_index = _ledger_phase(ordinal)
        action = _ledger_action(arrays, ordinal)
        is_generation = phase in {"budgeted_lift_generation", "raw_lift_generation"}
        is_schedule = trace_group is not None and not is_generation
        expected_keys = {
            "ordinal",
            "phase",
            "phase_index",
            "reply_kind",
            "elapsed_seconds",
            "trace_group",
            "trace_index",
            "actions",
        }
        if is_generation:
            expected_keys.add("supplied_reference")
        if is_generation:
            expected_keys.add("generated_schedule")
            expected_keys.add("projection_mode")
        if is_schedule:
            expected_keys.add("requested_schedule")
        if trace_group is not None:
            expected_keys.add("model_l2_path_budget")
        _require(set(row) == expected_keys, f"request ledger row {ordinal} keys changed")
        expected_scalars = {
            "ordinal": ordinal,
            "phase": phase,
            "phase_index": phase_index,
            "reply_kind": "action",
            "trace_group": trace_group,
            "trace_index": trace_index,
        }
        for key, expected in expected_scalars.items():
            observed = row.get(key)
            _require(observed == expected and type(observed) is type(expected), f"request ledger row {ordinal}.{key} changed")
        elapsed = row.get("elapsed_seconds")
        _require(type(elapsed) is float and math.isfinite(elapsed) and elapsed >= 0.0, f"request ledger row {ordinal} elapsed time is invalid")
        _require(row.get("actions") == _bytes_digest(action), f"request ledger row {ordinal} action digest changed")
        if is_generation:
            _require(row.get("supplied_reference") == _bytes_digest(arrays["reference_states_f32"]), f"request ledger row {ordinal} reference digest changed")
            _require(row.get("generated_schedule") == _bytes_digest(_full_schedule(arrays[f"{trace_group}_requested_velocity_f32"][trace_index])), f"request ledger row {ordinal} generated schedule digest changed")
            expected_projection = (
                "product_ball"
                if trace_group == "arm_b_generation"
                else "raw"
            )
            _require(
                row.get("projection_mode") == expected_projection,
                f"request ledger row {ordinal} projection mode changed",
            )
        if is_schedule:
            _require(row.get("requested_schedule") == _bytes_digest(_full_schedule(arrays[f"{trace_group}_requested_velocity_f32"][trace_index])), f"request ledger row {ordinal} requested schedule digest changed")
        if trace_group is not None:
            expected_budget = (
                np.asarray(np.float32(0.0), dtype=np.float32)
                if trace_group == "zero"
                else (
                    arrays["raw_replay_budget_f32"]
                    if trace_group == "raw_replay"
                    else arrays["source_budget_f32"]
                )
            )
            _require(
                row.get("model_l2_path_budget")
                == _bytes_digest(np.asarray(expected_budget, dtype=np.float32)),
                f"request ledger row {ordinal} model-space budget changed",
            )


def _summary(objective: np.float64, metrics: np.ndarray, gates: np.ndarray) -> dict[str, Any]:
    return {
        "objective": float(objective),
        "metrics": {name: float(value) for name, value in zip(METRIC_NAMES, metrics)},
        "gates": {name: bool(value) for name, value in zip(METRIC_NAMES, gates)},
        "xyz_pass": bool(np.all(gates[:2])),
        "full_pass": bool(np.all(gates)),
    }


def validate_reference_trajectory_lift_tensors(
    raw: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one TRL-00A archive and independently classify its outcome."""

    arrays = _load_arrays(raw)
    _validate_source_bindings(source_bindings)
    _validate_frozen_arrays(arrays, FROZEN_ARRAY_BINDINGS)
    validate_request_ledger(ledger, arrays)

    _validate_transport_group(arrays, "zero", budget32=EXPECTED_BUDGET32)
    _validate_transport_group(arrays, "arm_a", budget32=EXPECTED_BUDGET32)
    _validate_transport_group(arrays, "arm_b_generation", budget32=EXPECTED_BUDGET32)
    _validate_transport_group(arrays, "arm_b_replay", budget32=EXPECTED_BUDGET32)
    for prefix in COMMON_GROUP_SIZES:
        _validate_recurrence_group(arrays, prefix, arrays["source_noise_f32"])
        _require_duplicate_group(arrays, prefix)
    _reconstruct_and_validate_reference(arrays)
    target_physical, arm_a_gates = _validate_reference_and_arm_a(arrays)
    budgeted_raw_norms = _validate_reference_generation(arrays, "arm_b_generation", projection_mode=1)
    _require_ordinary_replay(arrays, "arm_b_generation", "arm_b_replay")

    _validate_transport_group(arrays, "raw_generation", budget32=None)
    _validate_transport_group(arrays, "raw_replay", budget32=None)
    for prefix in FINITE_ONLY_GROUP_SIZES:
        _validate_recurrence_group(arrays, prefix, arrays["source_noise_f32"])
        _require_duplicate_group(arrays, prefix)
    raw_norms = _validate_reference_generation(arrays, "raw_generation", projection_mode=0)
    _require_ordinary_replay(arrays, "raw_generation", "raw_replay")
    _validate_identical_schedule_outputs(arrays)

    executed = arrays["raw_generation_executed_c_f32"][0]
    norms32, path32 = _constraint_values(executed)
    max_step32 = np.asarray(np.max(norms32), dtype=np.float32).reshape(())[()]
    raw_exact_budget_accepted = _constraints_accept(executed, EXPECTED_BUDGET32)
    seed64 = np.asarray(
        max(
            np.float64(path32),
            np.float64(5.0) * np.float64(max_step32),
            np.float64(EXPECTED_BUDGET32),
        ),
        dtype=np.float64,
    ).reshape(())[()]
    replay_budget32 = (
        EXPECTED_BUDGET32
        if raw_exact_budget_accepted
        else _upward_float32(seed64)
    )
    _require(bool(arrays["raw_exact_budget_accepted_bool"].reshape(())[()]) is raw_exact_budget_accepted, "stored raw exact-budget decision changed")
    _require(np.asarray(arrays["raw_path_f32"].reshape(())[()], dtype=np.float32).tobytes() == np.asarray(path32, dtype=np.float32).tobytes(), "stored raw path changed")
    _require(np.asarray(arrays["raw_max_step_norm_f32"].reshape(())[()], dtype=np.float32).tobytes() == np.asarray(max_step32, dtype=np.float32).tobytes(), "stored raw max-step norm changed")
    _require(np.asarray(arrays["raw_replay_seed_f64"].reshape(())[()], dtype=np.float64).tobytes() == np.asarray(seed64, dtype=np.float64).tobytes(), "stored raw replay seed changed")
    _require(np.asarray(arrays["raw_replay_budget_f32"].reshape(())[()], dtype=np.float32).tobytes() == np.asarray(replay_budget32, dtype=np.float32).tobytes(), "stored raw replay envelope changed")
    _require_constraints(arrays["raw_replay_executed_c_f32"][0], replay_budget32, name="raw replay envelope")
    raw_authority = {
        "exact_budget_accepted": raw_exact_budget_accepted,
        "path": float(path32),
        "max_step_norm": float(max_step32),
        "path_over_source_budget": float(np.float64(path32) / np.float64(EXPECTED_BUDGET32)),
        "max_step_over_source_radius": float(np.float64(max_step32) / np.float64(EXPECTED_RADIUS32)),
        "replay_budget": float(replay_budget32),
    }

    if bool(np.all(raw_norms <= np.float64(EXPECTED_RADIUS32))):
        for suffix in ORDINARY_TRACE_SUFFIXES:
            _require(_bytes_equal(arrays[f"raw_generation_{suffix}"], arrays[f"arm_b_generation_{suffix}"]), f"projection no-op requires raw/B equality for {suffix}")

    raw_objective, raw_metrics, raw_gates = _objective_metrics(
        arrays["raw_generation_returned_actions_f64"][0], target_physical
    )
    raw_summary = _summary(raw_objective, raw_metrics, raw_gates)

    arm_a_objective, arm_a_metrics, _ = _objective_metrics(
        arrays["arm_a_returned_actions_f64"][0], target_physical
    )
    budgeted_objective, budgeted_metrics, budgeted_gates = _objective_metrics(
        arrays["arm_b_generation_returned_actions_f64"][0], target_physical
    )
    arm_a_summary = _summary(arm_a_objective, arm_a_metrics, arm_a_gates)
    budgeted_summary = _summary(budgeted_objective, budgeted_metrics, budgeted_gates)
    b_changed = not _bytes_equal(
        arrays["arm_b_generation_requested_velocity_f32"][0],
        arrays["arm_a_requested_velocity_f32"][0],
    )
    raw_changed = not _bytes_equal(
        arrays["raw_generation_requested_velocity_f32"][0],
        arrays["arm_a_requested_velocity_f32"][0],
    )
    b_full = bool(np.all(budgeted_gates))
    b_xyz = bool(np.all(budgeted_gates[:2]))
    raw_full = bool(raw_gates is not None and np.all(raw_gates))
    raw_xyz = bool(raw_gates is not None and np.all(raw_gates[:2]))

    # Exhaustive ADR-0060 precedence after every apparatus check above.
    if (b_changed and b_full) or (
        bool(raw_exact_budget_accepted)
        and bool(raw_changed)
        and raw_full
    ):
        outcome = "same_budget_lift_pass"
    elif not b_full and raw_full and not bool(raw_exact_budget_accepted):
        outcome = "canonical_budget_bottleneck"
    elif (b_xyz and not b_full) or (raw_xyz and not raw_full):
        outcome = "canonical_mask_coupling"
    elif not b_xyz and not raw_xyz:
        outcome = "canonical_lift_negative"
    else:  # Defensive fail-closed coverage: no valid evidence may escape a class.
        raise ReferenceTrajectoryLiftValidationError("valid TRL evidence did not match the exhaustive outcome precedence")

    return {
        "schema_version": SCHEMA_VERSION,
        "branch": "finite",
        "request_count": 18,
        "arm_a": arm_a_summary,
        "budgeted": {**budgeted_summary, "changed_from_arm_a": b_changed},
        "raw": {**raw_summary, "changed_from_arm_a": bool(raw_changed)},
        "raw_authority": raw_authority,
        "budgeted_raw_projection_noop": bool(np.all(budgeted_raw_norms <= np.float64(EXPECTED_RADIUS32))),
        "outcome": outcome,
        "claims": {
            "simulator_efficacy_evaluated": False,
            "collision_or_progress_claim_allowed": False,
            "global_infeasibility_claim_allowed": False,
            "raw_authority_is_minimum_required_claim_allowed": False,
            "probe_or_mlp_training_authorized": False,
            "automatic_next_gate_authorized": False,
        },
    }


def validate_reference_trajectory_lift_npz(
    path: str | Path,
    ledger: Sequence[Mapping[str, Any]],
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Load an exact no-pickle NPZ and run the independent reconstruction."""

    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            _require(
                len(archive.files) == len(set(archive.files)),
                "TRL-00A NPZ contains duplicate member names",
            )
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, TypeError) as error:
        raise ReferenceTrajectoryLiftValidationError(
            f"TRL-00A NPZ cannot be loaded without pickle: {error}"
        ) from error
    return validate_reference_trajectory_lift_tensors(
        arrays,
        ledger,
        source_bindings,
    )


__all__ = [
    "COMMON_ARRAY_SPECS",
    "EXPECTED_SOURCE_BINDINGS",
    "FIDELITY_LIMITS64",
    "FINITE_ARRAY_SPECS",
    "FROZEN_ARRAY_BINDINGS",
    "METRIC_NAMES",
    "ReferenceTrajectoryLiftValidationError",
    "validate_reference_trajectory_lift_npz",
    "validate_reference_trajectory_lift_tensors",
    "validate_request_ledger",
]
