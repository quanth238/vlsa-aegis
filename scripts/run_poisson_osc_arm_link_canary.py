#!/usr/bin/env python3
"""Run one treatment-only SafeLIBERO post-OSC Poisson arm-link canary.

The immutable completed AEGIS / OSC episode is the control; it is not rerun.
The live treatment uses the same archived high-level actions until the first
material Poisson torque correction.  It then consumes the remainder of that
already-current pi0.5 chunk and only afterwards queries pi0.5 from its own
observations.  Every nominal OSC arm torque is shielded immediately before its
2 ms MuJoCo step.  Any invalid field, sensitivity, QP, or postcheck stops the
run without a pass-through fallback.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import inspect
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts import run_poisson_fast_feasibility as fast


class OscCanaryRunnerError(RuntimeError):
    """An apparatus condition prevents scientific interpretation."""


class TreatmentContactObserved(RuntimeError):
    """A literal registered forbidden contact terminates a negative run."""


class TreatmentMethodStop(RuntimeError):
    """A validated safety method stops before the next physics transition."""


def _manifest_case(path: Path, case_id: str) -> Tuple[Dict[str, Any], str]:
    matches: List[Tuple[Dict[str, Any], str]] = []
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            value = json.loads(raw)
            if value.get("case_id") == case_id:
                matches.append((value, fast._sha256(raw.rstrip(b"\r\n"))))
    if len(matches) != 1:
        raise OscCanaryRunnerError(
            "manifest must contain exactly one %s row" % case_id
        )
    return matches[0]


def _canonical_sha256(value: Any) -> str:
    """Hash one JSON evidence block with the artifact's canonical framing."""

    return fast._sha256(fast._canonical(value))


def _paper_car_endpoint_row(
    *,
    source_action_index: int,
    snapshot_kind: str,
    observation_key: str,
    obstacle_root_body_id: int,
    observation_position: Any,
    observable_value_position: Any,
    observation_cache_position: Any,
    live_body_position_diagnostic: Any,
    post_integration_forwarded_body_position: Any,
    settled_observation_position: Any,
    np: Any,
) -> Dict[str, Any]:
    """Bind Table-1 CAR to robosuite's native cached observable.

    Robosuite updates a 20 Hz object observable from inside its 500 Hz physics
    loop and returns that cached value at the completed action endpoint.  The
    returned observation, ``Observable.obs``, and ``_obs_cache`` must therefore
    agree byte-for-byte.  A later live ``body_xpos`` read and a separately
    forwarded pose are retained as phase diagnostics only; neither defines CAR
    or object identity.  Root identity is enforced separately through
    ``env.obj_body_id``, the observable closure, and contact authority.
    """

    observed = np.asarray(observation_position)
    observable_value = np.asarray(observable_value_position)
    observation_cache = np.asarray(observation_cache_position)
    live = np.asarray(live_body_position_diagnostic)
    forwarded = np.asarray(post_integration_forwarded_body_position)
    settled = np.asarray(settled_observation_position)
    values = (
        observed,
        observable_value,
        observation_cache,
        live,
        forwarded,
        settled,
    )
    if any(value.shape != (3,) for value in values):
        raise OscCanaryRunnerError("paper CAR position must be a three-vector")
    expected_dtype = np.dtype(np.float64)
    if any(value.dtype != expected_dtype for value in values):
        raise OscCanaryRunnerError("paper CAR position must retain native float64 bytes")
    if not all(np.all(np.isfinite(value)) for value in values):
        raise OscCanaryRunnerError("paper CAR position must be finite")
    observed_record = _compact_float64_array_record(observed, np)
    observable_record = _compact_float64_array_record(observable_value, np)
    cache_record = _compact_float64_array_record(observation_cache, np)
    live_record = _compact_float64_array_record(live, np)
    forwarded_record = _compact_float64_array_record(forwarded, np)
    observation_matches_observable = bool(
        np.array_equal(observed, observable_value)
        and observed_record["sha256"] == observable_record["sha256"]
    )
    observation_matches_cache = bool(
        np.array_equal(observed, observation_cache)
        and observed_record["sha256"] == cache_record["sha256"]
    )
    live_delta = live - observed
    forwarded_delta = forwarded - observed
    forwarded_delta_record = _compact_float64_array_record(forwarded_delta, np)
    return {
        "source_action_index": int(source_action_index),
        "snapshot_kind": str(snapshot_kind),
        "paper_car_observation_key": str(observation_key),
        "paper_car_position_source": "selected_obstacle_pos_native_observation",
        "active_obstacle_root_body_id": int(obstacle_root_body_id),
        "active_obstacle_position_observation_world_m": observed.tolist(),
        "active_obstacle_position_observation_dtype": observed.dtype.str,
        "active_obstacle_position_observation_shape": [3],
        "active_obstacle_position_observation_array_sha256": observed_record[
            "sha256"
        ],
        "native_observable_value_world_m": observable_value.tolist(),
        "native_observable_value_dtype": observable_value.dtype.str,
        "native_observable_value_shape": [3],
        "native_observable_value_array_sha256": observable_record["sha256"],
        "native_observation_cache_value_world_m": observation_cache.tolist(),
        "native_observation_cache_value_dtype": observation_cache.dtype.str,
        "native_observation_cache_value_shape": [3],
        "native_observation_cache_value_array_sha256": cache_record["sha256"],
        "observation_observable_value_bitwise_equal": (
            observation_matches_observable
        ),
        "observation_cache_value_bitwise_equal": observation_matches_cache,
        "native_observable_cache_binding_exact": bool(
            observation_matches_observable and observation_matches_cache
        ),
        "active_obstacle_root_position_world_m": live.tolist(),
        "active_obstacle_root_position_array_sha256": live_record["sha256"],
        "active_obstacle_root_position_phase": (
            "live_body_xpos_after_cached_observation_return"
        ),
        "live_root_position_role": (
            "phase_diagnostic_only_not_authority_or_paper_car_metric"
        ),
        "observation_live_root_l1_delta_m": float(
            np.sum(np.abs(live_delta))
        ),
        "observation_live_root_linf_delta_m": float(
            np.max(np.abs(live_delta))
        ),
        "active_obstacle_root_position_post_integration_forwarded_world_m": (
            forwarded.tolist()
        ),
        "active_obstacle_root_position_post_integration_array_sha256": (
            forwarded_record["sha256"]
        ),
        "post_integration_forwarded_pose_role": (
            "phase_diagnostic_only_not_paper_car_metric"
        ),
        "observation_post_integration_forwarded_component_delta_m": (
            forwarded_delta.tolist()
        ),
        "observation_post_integration_forwarded_component_delta_array_sha256": (
            forwarded_delta_record["sha256"]
        ),
        "observation_post_integration_forwarded_l1_delta_m": float(
            np.sum(np.abs(forwarded_delta))
        ),
        "observation_post_integration_forwarded_linf_delta_m": float(
            np.max(np.abs(forwarded_delta))
        ),
        "l1_displacement_from_settled_m": float(
            np.sum(np.abs(observed - settled))
        ),
    }


def _require_paper_car_native_cache_binding(row: Mapping[str, Any]) -> None:
    """Fail before physics unless all native observable caches match exactly."""

    if not (
        row.get("observation_observable_value_bitwise_equal") is True
        and row.get("observation_cache_value_bitwise_equal") is True
        and row.get("native_observable_cache_binding_exact") is True
        and row.get("active_obstacle_position_observation_array_sha256")
        == row.get("native_observable_value_array_sha256")
        == row.get("native_observation_cache_value_array_sha256")
    ):
        raise OscCanaryRunnerError(
            "paper CAR returned observation differs from its native observable cache"
        )


def _native_object_observable_values(
    task_env: Any,
    *,
    observation_key: str,
    obstacle_name: str,
    np: Any,
) -> Tuple[Dict[str, Any], Any, Any]:
    """Resolve the exact robosuite observable and its current cache value."""

    observables = getattr(task_env, "_observables", None)
    observation_cache = getattr(task_env, "_obs_cache", None)
    if (
        not isinstance(observables, Mapping)
        or observation_key not in observables
        or not isinstance(observation_cache, Mapping)
        or observation_key not in observation_cache
    ):
        raise OscCanaryRunnerError("paper CAR native observable cache is absent")
    observable = observables[observation_key]
    sensor = getattr(observable, "_sensor", None)
    if not callable(sensor):
        raise OscCanaryRunnerError("paper CAR native observable sensor is absent")
    try:
        closure = inspect.getclosurevars(sensor).nonlocals
        enabled = bool(observable.is_enabled())
        active = bool(observable.is_active())
        sampling_timestep_s = float(observable._sampling_timestep)
        observable_value = np.asarray(observable.obs).copy()
        cache_value = np.asarray(observation_cache[observation_key]).copy()
    except Exception as error:
        raise OscCanaryRunnerError(
            "paper CAR native observable cannot be inspected: %s" % error
        ) from error
    checks = {
        "observable_name_matches_key": str(getattr(observable, "name", ""))
        == observation_key,
        "observable_enabled": enabled,
        "observable_active": active,
        "observable_modality_is_object": str(getattr(observable, "modality", ""))
        == "object",
        "observable_sampling_timestep_is_20hz": math.isclose(
            sampling_timestep_s, 0.05, rel_tol=0.0, abs_tol=1.0e-15
        ),
        "sensor_function_is_obj_pos": str(getattr(sensor, "__name__", ""))
        == "obj_pos",
        "sensor_nonlocal_object_name_matches": closure.get("obj_name")
        == obstacle_name,
        "sensor_nonlocal_environment_matches": closure.get("self") is task_env,
    }
    if (
        observable_value.shape != (3,)
        or cache_value.shape != (3,)
        or observable_value.dtype != np.dtype(np.float64)
        or cache_value.dtype != np.dtype(np.float64)
        or not np.all(np.isfinite(observable_value))
        or not np.all(np.isfinite(cache_value))
        or not all(checks.values())
    ):
        raise OscCanaryRunnerError("paper CAR native observable binding differs")
    return (
        {
            "schema_version": "vlsa_poisson_native_object_observable.v1",
            "observation_key": observation_key,
            "observable_name": str(observable.name),
            "observable_modality": str(observable.modality),
            "sampling_timestep_s": sampling_timestep_s,
            "sensor_function_name": str(sensor.__name__),
            "sensor_nonlocal_object_name": str(closure.get("obj_name")),
            "checks": checks,
            "all_checks_passed": True,
        },
        observable_value,
        cache_value,
    )


_FLOAT64_ARRAY_HASH_FORMAT = (
    "sha256_vlsa-table1-array-v1_header_and_c_order_float64_bytes"
)


def _compact_float64_array_record(value: Any, np: Any) -> Dict[str, Any]:
    """Return a deterministic identity and scalar minimum without raw values."""

    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise OscCanaryRunnerError("compact trace array must be finite and nonempty")
    contiguous = np.ascontiguousarray(array)
    header = fast._canonical(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    )
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return {
        "dtype": contiguous.dtype.str,
        "shape": [int(value) for value in contiguous.shape],
        "sha256": digest.hexdigest(),
        "minimum": float(np.min(contiguous)),
    }


def _minimum_constraint_attribution(
    nominal_residuals: Any, samples: Sequence[Any], np: Any
) -> Dict[str, Any]:
    """Bind the first causal torque divergence to protected-link samples."""

    residuals = np.asarray(nominal_residuals, dtype=np.float64)
    if (
        residuals.ndim != 1
        or residuals.size == 0
        or residuals.size != len(samples)
        or not np.all(np.isfinite(residuals))
    ):
        raise OscCanaryRunnerError("constraint attribution inputs are invalid")
    minimum = float(np.min(residuals))
    minimum_index = int(np.argmin(residuals))
    tolerance = 1e-12
    near_indices = np.flatnonzero(np.abs(residuals - minimum) <= tolerance)
    negative_indices = np.flatnonzero(residuals < 0.0)
    minimum_sample = samples[minimum_index].to_dict()
    return {
        "schema_version": "vlsa_poisson_constraint_attribution.v1",
        "selection_rule": "first_np_argmin_of_nominal_exact_clone_cbf_residual",
        "minimum_nominal_residual_m2_per_s": minimum,
        "minimum_sample_index": minimum_index,
        "minimum_sample": minimum_sample,
        "near_minimum_absolute_tolerance_m2_per_s": tolerance,
        "near_minimum_sample_indices": [int(value) for value in near_indices],
        "near_minimum_body_names": sorted(
            {samples[int(value)].body_name for value in near_indices}
        ),
        "negative_nominal_residual_sample_count": int(negative_indices.size),
        "negative_nominal_residual_body_names": sorted(
            {samples[int(value)].body_name for value in negative_indices}
        ),
    }


def _compact_shield_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    shield_status: str,
    sample_count: int,
    sensitivity_record: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Retain bounded diagnostics and bind the omitted producer diagnostics."""

    scalar_fields = (
        "solver_status_value",
        "solver_iterations",
        "input_constraint_count",
        "solved_constraint_count",
        "numerically_uncontrollable_constraint_count",
        "sensitivity_max_absolute_error",
        "sensitivity_condition_number",
        "maximum_sensitivity_condition_number",
        "maximum_torque_gain_row_uncertainty",
        "minimum_effective_controllable_row_gain",
        "minimum_nonzero_torque_gain_row_scale",
        "maximum_nonzero_torque_gain_row_scale",
    )
    compact_scalars: Dict[str, Any] = {}
    for field in scalar_fields:
        if field in diagnostics:
            value = diagnostics[field]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise OscCanaryRunnerError(
                    "shield diagnostic %s is not a finite scalar" % field
                )
            compact_scalars[field] = value
    return {
        "schema_version": "vlsa_poisson_compact_shield_diagnostics.v1",
        "shield_status": str(shield_status),
        "sample_count": int(sample_count),
        "solver_attempted": bool(diagnostics.get("solver_attempted")),
        "solver": diagnostics.get("solver"),
        "solver_status": diagnostics.get("solver_status"),
        "global_condition_limit_exceeded": diagnostics.get(
            "global_condition_limit_exceeded"
        ),
        "global_condition_is_advisory_only": diagnostics.get(
            "global_condition_is_advisory_only"
        ),
        "scalars": compact_scalars,
        "full_diagnostics_sha256": _canonical_sha256(diagnostics),
        "sensitivity_certificate_sha256": (
            None
            if sensitivity_record is None
            else _canonical_sha256(sensitivity_record)
        ),
    }


def _historical_link_contact_verified(
    case: Mapping[str, Any], historical: Mapping[str, Any]
) -> bool:
    pairs = case.get("direct_link_active_obstacle_pairs")
    telemetry = historical.get("contact_telemetry")
    unique_pairs = telemetry.get("unique_contact_pairs") if isinstance(telemetry, Mapping) else None
    raw_pairs = {
        (str(row.get("geom1")), str(row.get("geom2")))
        for row in unique_pairs or ()
        if isinstance(row, Mapping)
    }
    return bool(
        isinstance(pairs, Sequence)
        and pairs
        and all(
            isinstance(row, Mapping)
            and row.get("active_obstacle_name") == case.get("active_obstacle_name")
            and row.get("actual_link_body_name") in ("robot0_link5", "robot0_link6")
            and row.get("active_obstacle_actual_body_name")
            == "%s_main" % case.get("active_obstacle_name")
            and (
                (str(row.get("active_obstacle_geom_name")), str(row.get("link_geom_name")))
                in raw_pairs
                or (str(row.get("link_geom_name")), str(row.get("active_obstacle_geom_name")))
                in raw_pairs
            )
            for row in pairs
        )
        and any(row.get("actual_link_body_name") == "robot0_link5" for row in pairs)
        and case.get("first_sampled_link_contact_control_step") == 62
        and case.get("first_link_step_is_timing_resolved") is True
        and telemetry.get("first_contact_step") == 62
    )


def _source_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    source = case.get("source_case")
    if not isinstance(source, Mapping):
        raise OscCanaryRunnerError("physical-contact manifest lacks source_case")
    output = dict(source)
    output["active_obstacle_name"] = case.get("active_obstacle_name")
    return output


def _historical_path(root: Path, case: Mapping[str, Any]) -> Path:
    binding = case.get("historical_result")
    relative = binding.get("relative_path") if isinstance(binding, Mapping) else None
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or any(part in ("", ".", "..") for part in Path(relative).parts)
    ):
        raise OscCanaryRunnerError("historical result path binding is invalid")
    path = root.resolve()
    if path.is_symlink() or not path.is_dir():
        raise OscCanaryRunnerError("historical result root is invalid")
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise OscCanaryRunnerError("historical result path traverses a symlink")
    if not path.is_file() or fast._file_sha256(path) != binding.get("file_sha256"):
        raise OscCanaryRunnerError("historical result file or hash differs")
    return path


def _verify_remote_artifact_receipt(
    historical_root: Path,
    receipt: Mapping[str, Any],
    case: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if not (
        receipt.get("schema_version")
        == "vlsa_poisson_remote_source_artifact_receipt.v1"
        and receipt.get("case_id") == case.get("case_id")
        and Path(str(receipt.get("historical_root", ""))).resolve()
        == historical_root.resolve()
    ):
        raise OscCanaryRunnerError("remote source-artifact receipt differs")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Sequence) or len(artifacts) != 3:
        raise OscCanaryRunnerError("remote source-artifact receipt is incomplete")
    manifest_binding = case["historical_result"]
    auxiliary = case["artifact_bindings"]
    expected_by_name = {
        "result.json": str(manifest_binding["file_sha256"]),
        "episode.mp4": str(auxiliary["video_sha256"]),
        "active_obstacle_contacts.json.gz": str(
            auxiliary["detailed_contact_gzip_sha256"]
        ),
    }
    verified: List[Dict[str, Any]] = []
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise OscCanaryRunnerError("remote artifact receipt row is invalid")
        relative = row.get("relative_path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or any(part in ("", ".", "..") for part in Path(relative).parts)
        ):
            raise OscCanaryRunnerError("remote artifact relative path is invalid")
        basename = Path(relative).name
        if basename not in expected_by_name:
            raise OscCanaryRunnerError("remote artifact filename is unexpected")
        target = historical_root.resolve()
        for part in Path(relative).parts:
            target = target / part
            if target.is_symlink():
                raise OscCanaryRunnerError("remote artifact path traverses a symlink")
        if not target.is_file() or target.is_symlink():
            raise OscCanaryRunnerError("remote artifact is not a regular file")
        observed_sha = fast._file_sha256(target)
        if not (
            row.get("regular_file") is True
            and row.get("symlink") is False
            and row.get("sha256") == expected_by_name[basename]
            and observed_sha == expected_by_name[basename]
            and int(row.get("byte_count", -1)) == int(target.stat().st_size)
        ):
            raise OscCanaryRunnerError("remote artifact bytes differ from receipt")
        verified.append(
            {
                "relative_path": relative,
                "sha256": observed_sha,
                "byte_count": int(target.stat().st_size),
                "regular_nonsymlink": True,
            }
        )
    if set(expected_by_name) != {Path(row["relative_path"]).name for row in verified}:
        raise OscCanaryRunnerError("remote artifact receipt filename set differs")
    return verified


def _controller_record(env: Any) -> Dict[str, Any]:
    robot = env.robots[0]
    controller = getattr(robot, "controller", None)
    name = str(getattr(controller, "name", ""))
    controller_class = type(controller)
    module_name = str(controller_class.__module__)
    qualified_name = str(controller_class.__qualname__)
    module_path = inspect.getsourcefile(controller_class)
    if not module_path or not Path(module_path).is_file():
        raise OscCanaryRunnerError("OSC controller implementation source is absent")
    record = {
        "controller_name": name,
        "controller_class_module": module_name,
        "controller_class_qualname": qualified_name,
        "controller_implementation_file_sha256": fast._file_sha256(
            Path(module_path)
        ),
        "controller_control_dim": int(getattr(controller, "control_dim", -1)),
        "environment_action_dim": int(getattr(env.env, "action_dim", -1)),
        "control_frequency_hz": int(getattr(env.env, "control_freq", -1)),
        "control_timestep_s": float(env.env.control_timestep),
        "physics_frequency_hz": int(round(1.0 / float(env.env.model_timestep))),
        "physics_timestep_s": float(env.env.model_timestep),
        "arm_actuator_indexes": [
            int(value) for value in robot._ref_joint_actuator_indexes
        ],
        "arm_qpos_indexes": [int(value) for value in robot._ref_joint_pos_indexes],
        "arm_qvel_indexes": [int(value) for value in robot._ref_joint_vel_indexes],
    }
    record["original_osc_verified"] = bool(
        name == "OSC_POSE"
        and qualified_name == "OperationalSpaceController"
        and module_name.endswith(".osc")
        and record["controller_control_dim"] == 6
        and record["environment_action_dim"] == 7
        and record["control_frequency_hz"] == 20
        and record["physics_frequency_hz"] == 500
        and math.isclose(
            record["control_timestep_s"], 0.05, rel_tol=0.0, abs_tol=1e-12
        )
        and math.isclose(
            record["physics_timestep_s"], 0.002, rel_tol=0.0, abs_tol=1e-12
        )
        and len(record["arm_actuator_indexes"]) == 7
        and len(record["arm_qpos_indexes"]) == 7
        and len(record["arm_qvel_indexes"]) == 7
    )
    return record


def _direct_hinge_torque_actuator_record(env: Any) -> Dict[str, Any]:
    """Prove that each selected control is a direct unit-gain hinge torque."""

    import mujoco
    import numpy as np

    robot = env.robots[0]
    model = getattr(env.sim.model, "_model", env.sim.model)
    actuator_ids = np.asarray(
        robot._ref_joint_actuator_indexes, dtype=np.int64
    )
    joint_ids = np.asarray(robot._ref_joint_indexes, dtype=np.int64)
    if actuator_ids.shape != (7,) or joint_ids.shape != (7,):
        raise OscCanaryRunnerError("arm torque-unit contract requires seven actuators")
    expected_gear = np.zeros(6, dtype=np.float64)
    expected_gear[0] = 1.0
    records: List[Dict[str, Any]] = []
    for actuator_id, joint_id in zip(actuator_ids, joint_ids):
        actuator_id = int(actuator_id)
        joint_id = int(joint_id)
        actuator_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        checks = {
            "joint_transmission": int(model.actuator_trntype[actuator_id])
            == int(mujoco.mjtTrn.mjTRN_JOINT),
            "transmission_joint_matches": int(model.actuator_trnid[actuator_id, 0])
            == joint_id,
            "hinge_joint": int(model.jnt_type[joint_id])
            == int(mujoco.mjtJoint.mjJNT_HINGE),
            "fixed_gain": int(model.actuator_gaintype[actuator_id])
            == int(mujoco.mjtGain.mjGAIN_FIXED),
            "unit_gain": bool(
                float(model.actuator_gainprm[actuator_id, 0]) == 1.0
                and np.all(
                    np.asarray(
                        model.actuator_gainprm[actuator_id, 1:], dtype=np.float64
                    )
                    == 0.0
                )
            ),
            "no_bias": bool(
                int(model.actuator_biastype[actuator_id])
                == int(mujoco.mjtBias.mjBIAS_NONE)
                and np.all(
                    np.asarray(model.actuator_biasprm[actuator_id], dtype=np.float64)
                    == 0.0
                )
            ),
            "no_activation_dynamics": int(model.actuator_dyntype[actuator_id])
            == int(mujoco.mjtDyn.mjDYN_NONE),
            "control_limit_enabled": int(model.actuator_ctrllimited[actuator_id]) != 0,
            "unit_joint_gear": bool(
                np.array_equal(
                    np.asarray(model.actuator_gear[actuator_id], dtype=np.float64),
                    expected_gear,
                )
            ),
            "force_limit_cannot_clip_control_range": bool(
                int(model.actuator_forcelimited[actuator_id]) == 0
                or (
                    float(model.actuator_forcerange[actuator_id, 0])
                    <= float(model.actuator_ctrlrange[actuator_id, 0])
                    and float(model.actuator_forcerange[actuator_id, 1])
                    >= float(model.actuator_ctrlrange[actuator_id, 1])
                )
            ),
        }
        records.append(
            {
                "actuator_id": actuator_id,
                "actuator_name": str(actuator_name),
                "joint_id": joint_id,
                "joint_name": str(joint_name),
                "gain_parameters": np.asarray(
                    model.actuator_gainprm[actuator_id], dtype=np.float64
                ).tolist(),
                "bias_parameters": np.asarray(
                    model.actuator_biasprm[actuator_id], dtype=np.float64
                ).tolist(),
                "gear": np.asarray(
                    model.actuator_gear[actuator_id], dtype=np.float64
                ).tolist(),
                "control_limited": bool(model.actuator_ctrllimited[actuator_id]),
                "control_range": np.asarray(
                    model.actuator_ctrlrange[actuator_id], dtype=np.float64
                ).tolist(),
                "force_limited": bool(model.actuator_forcelimited[actuator_id]),
                "force_range": np.asarray(
                    model.actuator_forcerange[actuator_id], dtype=np.float64
                ).tolist(),
                "checks": checks,
                "verified": bool(all(checks.values())),
            }
        )
    verified = bool(len(records) == 7 and all(row["verified"] for row in records))
    if not verified:
        raise OscCanaryRunnerError(
            "arm controls are not verified direct unit-gain hinge torques"
        )
    return {
        "schema_version": "vlsa_poisson_direct_hinge_torque_units.v1",
        "control_unit": "newton_metre",
        "generalized_force_relation": "joint_torque_nm_equals_control_times_unit_gain_and_unit_gear",
        "actuators": records,
        "verified": True,
    }


class TreatmentVideo:
    """Atomically publish real simulator frames for the complete treatment."""

    def __init__(self, evaluator: Any, imageio: Any, output: Path, fps: int) -> None:
        self.evaluator = evaluator
        self.imageio = imageio
        self.directory = output / "video"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.partial = self.directory / "treatment.partial.mp4"
        self.final = self.directory / "treatment.mp4"
        self.writer = imageio.get_writer(
            str(self.partial), fps=int(fps), codec="libx264", macro_block_size=None
        )
        self.trace: List[Dict[str, Any]] = []
        self.closed = False

    def append(self, observation: Mapping[str, Any], kind: str, action_index: int) -> None:
        if self.closed:
            raise OscCanaryRunnerError("cannot append to closed treatment video")
        frame = self.evaluator._processed_image(observation, "agentview_image")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise OscCanaryRunnerError("agentview frame is not RGB")
        self.writer.append_data(frame)
        self.trace.append(
            {
                "frame_index": len(self.trace),
                "snapshot_kind": str(kind),
                "source_action_index": int(action_index),
                "source_array_sha256": self.evaluator.array_sha256(frame),
                "shape": [int(value) for value in frame.shape],
            }
        )

    def close(self) -> Dict[str, Any]:
        if not self.closed:
            self.writer.close()
            self.closed = True
            os.replace(str(self.partial), str(self.final))
        if not self.trace:
            raise OscCanaryRunnerError("treatment video has no frames")
        decoded = 0
        shapes: List[List[int]] = []
        reader = self.imageio.get_reader(str(self.final))
        try:
            for frame in reader:
                decoded += 1
                shapes.append([int(value) for value in frame.shape])
        finally:
            reader.close()
        if decoded != len(self.trace) or any(shape != [1024, 1024, 3] for shape in shapes):
            raise OscCanaryRunnerError("decoded treatment video differs from source trace")
        return {
            "path": str(self.final.relative_to(self.directory.parent)),
            "sha256": fast._file_sha256(self.final),
            "byte_count": int(self.final.stat().st_size),
            "frame_count": len(self.trace),
            "decoded_frame_count": decoded,
            "decoded_successfully": True,
            "real_simulation_frames": True,
            "two_dimensional_safety_overlay": False,
            "coverage": "settled_boundary_then_each_completed_action_and_terminal_contact",
            "snapshot_trace": list(self.trace),
        }


class HybridAegisPlanner:
    """Historical actions until divergence, then cached chunk and live feedback."""

    def __init__(
        self,
        *,
        evaluator: Any,
        runtime: Mapping[str, Any],
        case: Mapping[str, Any],
        task_description: str,
        historical: Mapping[str, Any],
        host: str,
        port: int,
        replan_steps: int,
        model_action_horizon: int,
    ) -> None:
        actions = historical.get("actions")
        perception = historical.get("perception")
        if not isinstance(actions, Sequence) or not actions:
            raise OscCanaryRunnerError("historical action ledger is absent")
        if not isinstance(perception, Mapping) or perception.get("status") != "ready":
            raise OscCanaryRunnerError("historical AEGIS geometry is absent")
        np = runtime["np"]
        self.evaluator = evaluator
        self.runtime = runtime
        self.case = case
        self.task_description = str(task_description)
        self.historical_actions = actions
        self.host = str(host)
        self.port = int(port)
        self.replan_steps = int(replan_steps)
        self.model_action_horizon = int(model_action_horizon)
        self.geometry = {
            "enabled": True,
            "p2": np.asarray(perception["mvee_center"], dtype=np.float64).copy(),
            "R2": np.asarray(perception["mvee_rotation"], dtype=np.float64).copy(),
            "Q2_diag": np.asarray(perception["mvee_semiaxes"], dtype=np.float64).copy(),
            "z_fixed": None,
        }
        self.divergence_action: Optional[int] = None
        self.divergence_physical_boundary: Optional[int] = None
        self.cached_chunk_end: Optional[int] = None
        self.client: Any = None
        self.server_metadata: Any = None
        self.plan: collections.deque = collections.deque()
        self.plan_query: collections.deque = collections.deque()
        self.plan_offset: collections.deque = collections.deque()
        self.query_trace: List[Dict[str, Any]] = []
        self.action_trace: List[Dict[str, Any]] = []

    def mark_divergence(self, *, action_index: int, physical_boundary: int) -> None:
        if self.divergence_action is not None:
            return
        np = self.runtime["np"]
        row = self.historical_actions[int(action_index)]
        qp = row.get("qp") if isinstance(row, Mapping) else None
        z_after = np.asarray(qp.get("z_after"), dtype=np.float64) if isinstance(qp, Mapping) else np.asarray([])
        if z_after.shape != (3,) or not np.all(np.isfinite(z_after)):
            raise OscCanaryRunnerError("historical AEGIS state at divergence is invalid")
        self.geometry["z_fixed"] = z_after.copy()
        self.divergence_action = int(action_index)
        self.divergence_physical_boundary = int(physical_boundary)
        self.cached_chunk_end = (
            (int(action_index) // self.replan_steps + 1) * self.replan_steps - 1
        )

    def _observation_fingerprint(self, observation: Mapping[str, Any]) -> str:
        from scripts.run_poisson_shadow_parity import _observation_sha256

        return _observation_sha256(
            observation, self.evaluator, self.runtime["np"]
        )

    def _fresh_raw(self, observation: Mapping[str, Any], source_index: int) -> Tuple[Any, int, int]:
        np = self.runtime["np"]
        if not self.plan:
            if source_index % self.replan_steps != 0:
                raise OscCanaryRunnerError("fresh pi0.5 query is not at a replan boundary")
            if self.client is None:
                self.client = self.runtime["websocket_client_policy"].WebsocketClientPolicy(
                    self.host, self.port
                )
                self.server_metadata = self.evaluator._server_identity(self.client)
                if self.server_metadata.get("status") != "available":
                    raise OscCanaryRunnerError("pi0.5 server metadata is unavailable")
            query_index = source_index // self.replan_steps
            seed = self.evaluator.query_seed(int(self.case["policy_noise_seed"]), query_index)
            policy_input = self.evaluator._policy_observation(
                self.runtime,
                observation,
                task_description=self.task_description,
                resize_size=self.evaluator.TABLE_POLICY_RESIZE,
                rng_seed=seed,
            )
            started = time.perf_counter()
            response = self.client.infer(policy_input)
            elapsed = time.perf_counter() - started
            chunk = np.asarray(response.get("actions"), dtype=np.float64)
            if chunk.shape != (self.model_action_horizon, 7) or not np.all(np.isfinite(chunk)):
                raise OscCanaryRunnerError("pi0.5 returned an invalid action chunk")
            for offset in range(self.replan_steps):
                self.plan.append(chunk[offset].copy())
                self.plan_query.append(query_index)
                self.plan_offset.append(offset)
            self.query_trace.append(
                {
                    "query_index": int(query_index),
                    "rng_seed": int(seed),
                    "source_action_index": int(source_index),
                    "native_observation_sha256": self._observation_fingerprint(observation),
                    "returned_actions": chunk.tolist(),
                    "returned_actions_sha256": self.evaluator.array_sha256(chunk),
                    "returned_action_shape": list(chunk.shape),
                    "elapsed_seconds": float(elapsed),
                    "source": "fresh_pi05_from_treatment_observation",
                }
            )
        return (
            np.asarray(self.plan.popleft(), dtype=np.float64),
            int(self.plan_query.popleft()),
            int(self.plan_offset.popleft()),
        )

    def action(
        self, *, env: Any, observation: Mapping[str, Any], source_index: int
    ) -> Sequence[float]:
        np = self.runtime["np"]
        if self.divergence_action is None:
            if source_index >= len(self.historical_actions):
                raise OscCanaryRunnerError("historical replay ended before divergence")
            row = self.historical_actions[source_index]
            executed = np.asarray(row.get("executed"), dtype=np.float64)
            self.action_trace.append(
                {
                    "source_action_index": int(source_index),
                    "source": "archived_aegis_executed_pre_divergence",
                    "executed": executed.tolist(),
                    "executed_action_sha256": self.evaluator.array_sha256(executed),
                    "native_observation_sha256": self._observation_fingerprint(observation),
                }
            )
            return executed.tolist()

        if source_index <= int(self.cached_chunk_end):
            if source_index >= len(self.historical_actions):
                raise OscCanaryRunnerError("cached historical pi0.5 chunk is absent")
            raw = np.asarray(
                self.historical_actions[source_index].get("nominal_raw"),
                dtype=np.float64,
            )
            query_index = source_index // self.replan_steps
            offset = source_index % self.replan_steps
            source = "archived_current_pi05_chunk_reprocessed_live_aegis"
        else:
            raw, query_index, offset = self._fresh_raw(observation, source_index)
            source = "fresh_pi05_reprocessed_live_aegis"
        if raw.shape != (7,) or not np.all(np.isfinite(raw)):
            raise OscCanaryRunnerError("post-divergence raw policy action is invalid")
        nominal = self.evaluator.translational_action(raw)
        proxy = self.evaluator._eef_proxy(self.runtime, observation)
        z_before = np.asarray(self.geometry["z_fixed"], dtype=np.float64).copy()
        executed, qp_record = self.evaluator._aegis_action(
            self.runtime,
            nominal_translational=nominal,
            proxy=proxy,
            geometry=self.geometry,
            q1_diag=np.asarray([0.06, 0.12, 0.11], dtype=np.float64),
            diagnostics_enabled=True,
        )
        if qp_record.get("status") != "solved" or qp_record.get("solver_status") not in (
            "optimal",
            "optimal_inaccurate",
        ):
            raise OscCanaryRunnerError("post-divergence AEGIS QP failed")
        self.action_trace.append(
            {
                "source_action_index": int(source_index),
                "source": source,
                "query_index": int(query_index),
                "query_chunk_offset": int(offset),
                "native_observation_sha256": self._observation_fingerprint(observation),
                "nominal_raw": raw.tolist(),
                "nominal_raw_sha256": self.evaluator.array_sha256(raw),
                "nominal_translational": list(nominal),
                "executed": list(executed),
                "executed_action_sha256": self.evaluator.array_sha256(
                    np.asarray(executed, dtype=np.float64)
                ),
                "aegis_z_before": z_before.tolist(),
                "aegis_z_after": np.asarray(self.geometry["z_fixed"], dtype=np.float64).tolist(),
                "aegis_qp": qp_record,
            }
        )
        return list(executed)

    def record(self) -> Dict[str, Any]:
        after = [
            row for row in self.action_trace
            if self.divergence_action is not None
            and int(row["source_action_index"]) > int(self.divergence_action)
        ]
        cached = [row for row in after if row["source"].startswith("archived_current")]
        fresh = [row for row in after if row["source"].startswith("fresh_pi05")]
        query_trace = list(self.query_trace)
        action_trace = list(self.action_trace)
        return {
            "divergence_source_action_index": self.divergence_action,
            "divergence_physical_boundary": self.divergence_physical_boundary,
            "cached_current_chunk_end_source_action_index": self.cached_chunk_end,
            "policy_query_count": len(self.query_trace),
            "policy_queries": query_trace,
            "policy_query_trace_sha256": _canonical_sha256(query_trace),
            "action_trace": action_trace,
            "action_trace_sha256": _canonical_sha256(action_trace),
            "no_policy_query_before_divergence": bool(
                not self.query_trace
                or (
                    self.divergence_action is not None
                    and min(row["source_action_index"] for row in self.query_trace)
                    > int(self.divergence_action)
                )
            ),
            "cached_current_chunk_action_count": len(cached),
            "fresh_own_observation_action_count": len(fresh),
            "hybrid_contract_valid": bool(
                self.divergence_action is None
                or (
                    all(row["source"].startswith("archived_current") for row in cached)
                    and all(row["source"].startswith("fresh_pi05") for row in fresh)
                    and (not self.query_trace or fresh)
                )
            ),
            "policy_server": self.server_metadata,
            "policy_server_endpoint": {"host": self.host, "port": self.port},
            "policy_server_identity_sha256": (
                None
                if self.server_metadata is None
                else _canonical_sha256(self.server_metadata)
            ),
        }


def _post_correction_motion(
    physics_trace: Sequence[Mapping[str, Any]], first_boundary: Optional[int]
) -> Dict[str, Any]:
    if first_boundary is None:
        return {
            "joint_motion_integral_rad": 0.0,
            "eef_path_length_m": 0.0,
            "zero_torque_delta_fraction": 1.0,
        }
    rows = [row for row in physics_trace if int(row["physical_boundary"]) >= int(first_boundary)]
    if not rows:
        return {
            "joint_motion_integral_rad": 0.0,
            "eef_path_length_m": 0.0,
            "zero_torque_delta_fraction": 1.0,
        }
    import numpy as np

    joint_motion = sum(float(row["measured_qvel_l2_rad_s"]) * 0.002 for row in rows)
    positions = [np.asarray(row["eef_position_world_m"], dtype=np.float64) for row in rows]
    eef_path = sum(float(np.linalg.norm(right - left)) for left, right in zip(positions, positions[1:]))
    zero_fraction = sum(float(row["torque_correction_l2_nm"]) < 1e-12 for row in rows) / len(rows)
    return {
        "joint_motion_integral_rad": float(joint_motion),
        "eef_path_length_m": float(eef_path),
        "zero_torque_delta_fraction": float(zero_fraction),
    }


def _sensitivity_record(estimate: Any, snapshot: Any) -> Dict[str, Any]:
    """Serialize the independent two-resolution sensitivity certificate."""

    return {
        "torque_epsilon_nm": estimate.torque_epsilon_nm.tolist(),
        "torque_to_next_arm_qvel_sensitivity": (
            estimate.torque_to_next_arm_qvel_sensitivity.tolist()
        ),
        "full_epsilon_sensitivity": estimate.full_epsilon_sensitivity.tolist(),
        "half_epsilon_sensitivity": estimate.half_epsilon_sensitivity.tolist(),
        "maximum_epsilon_agreement_absolute_error": float(
            estimate.maximum_epsilon_agreement_absolute_error
        ),
        "maximum_epsilon_agreement_scaled_error": float(
            estimate.maximum_epsilon_agreement_scaled_error
        ),
        "agreement_atol": float(estimate.agreement_atol),
        "agreement_rtol": float(estimate.agreement_rtol),
        "nominal_post_step_contact_count": int(
            estimate.nominal_transition.post_step_contact_count
        ),
        "nominal_next_arm_qvel_rad_s": (
            estimate.nominal_transition.next_arm_qvel.tolist()
        ),
        "nominal_next_integration_state_sha256": (
            estimate.nominal_transition.next_integration_state_sha256
        ),
        "nominal_next_time_seconds": float(
            estimate.nominal_transition.next_time_seconds
        ),
        "non_arm_ctrl_preserved": bool(
            estimate.nominal_transition.non_arm_ctrl_preserved
        ),
        "analytic_free_dynamics_diagnostic": dict(
            estimate.analytic_free_dynamics_diagnostic
        ),
        "finite_difference_column_stencils": [
            dict(plan) for plan in estimate.finite_difference_column_stencils
        ],
        "snapshot": {
            "integration_state_sha256": snapshot.integration_state_sha256,
            "all_ctrl_sha256": hashlib.sha256(
                snapshot.all_ctrl.tobytes(order="C")
            ).hexdigest(),
            "arm_actuator_ids": list(snapshot.arm_actuator_ids),
            "arm_qpos_indices": list(snapshot.arm_qpos_indices),
            "arm_qvel_indices": list(snapshot.arm_qvel_indices),
            "arm_torque_lower_nm": snapshot.arm_torque_lower_nm.tolist(),
            "arm_torque_upper_nm": snapshot.arm_torque_upper_nm.tolist(),
            "nominal_arm_torque_nm": snapshot.nominal_arm_torque_nm.tolist(),
            "state_size": int(snapshot.state_size),
            "state_specification": int(snapshot.state_specification),
            "timestep_seconds": float(snapshot.timestep_seconds),
            "pre_step_contact_count": int(snapshot.pre_step_contact_count),
        },
    }


def _mujoco_name(model: Any, object_type: Any, object_id: int, prefix: str) -> str:
    import mujoco

    value = mujoco.mj_id2name(model, object_type, int(object_id))
    return str(value) if value else "%s_%d" % (prefix, int(object_id))


def _registered_contact_scope(model: Any, resolved: Any) -> Dict[str, Any]:
    """Freeze the exact union contact scope from authoritative MuJoCo IDs."""

    import mujoco

    robot = {int(value) for value in resolved.robot_geom_ids}
    robot_bodies = {int(value) for value in resolved.robot_body_ids}
    selected = {int(value) for value in resolved.obstacle_geom_ids}
    link56 = {int(value) for value in resolved.link56_geom_ids}
    all_geoms = set(range(int(model.ngeom)))
    robot_owned = {
        geom_id
        for geom_id in all_geoms
        if int(model.geom_bodyid[geom_id]) in robot_bodies
    }
    external = all_geoms - robot_owned
    if not robot or not robot_bodies or not robot_owned or not selected or not link56:
        raise OscCanaryRunnerError("registered contact scope contains an empty ID set")
    if (
        not robot <= robot_owned
        or not selected <= external
        or not link56 <= robot
        or robot_owned & external
    ):
        raise OscCanaryRunnerError("registered contact scope roles overlap or differ")
    identities = []
    for geom_id in sorted(all_geoms):
        body_id = int(model.geom_bodyid[geom_id])
        identities.append(
            {
                "geom_id": geom_id,
                "geom_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, "geom"
                ),
                "body_id": body_id,
                "body_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body"
                ),
                "is_robot_geom": geom_id in robot,
                "is_robot_owned_geom": geom_id in robot_owned,
                "is_selected_obstacle_geom": geom_id in selected,
                "is_link56_geom": geom_id in link56,
                "is_external_nonrobot_geom": geom_id in external,
            }
        )
    scope = {
        "schema_version": "vlsa_poisson_registered_contact_scope.v1",
        "semantics": (
            "forbidden_union=(any_robot_geom_vs_selected_obstacle_geom)_or_"
            "(literal_link5_link6_geom_vs_any_external_nonrobot_geom)"
        ),
        "model_geom_count": int(model.ngeom),
        "robot_geom_ids": sorted(robot),
        "robot_owned_geom_ids": sorted(robot_owned),
        "selected_obstacle_geom_ids": sorted(selected),
        "link56_geom_ids": sorted(link56),
        "external_nonrobot_geom_ids": sorted(external),
        "geom_identities": identities,
    }
    scope["identity_sha256"] = _canonical_sha256(scope)
    return scope


def _registered_contact_records(
    model: Any,
    data: Any,
    scope: Mapping[str, Any],
    *,
    source_phase: str,
    physical_boundary: Optional[int] = None,
    source_action_index: Optional[int] = None,
    physics_substep_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return compact typed records for nonpositive contacts in the union."""

    robot = {int(value) for value in scope["robot_geom_ids"]}
    selected = {int(value) for value in scope["selected_obstacle_geom_ids"]}
    link56 = {int(value) for value in scope["link56_geom_ids"]}
    external = {int(value) for value in scope["external_nonrobot_geom_ids"]}
    identity = {
        int(row["geom_id"]): row for row in scope["geom_identities"]
    }
    output: List[Dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        distance = float(contact.dist)
        if distance > 0.0 or not math.isfinite(distance):
            continue
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        selected_contact = bool(
            (geom1 in robot and geom2 in selected)
            or (geom2 in robot and geom1 in selected)
        )
        link_external_contact = bool(
            (geom1 in link56 and geom2 in external)
            or (geom2 in link56 and geom1 in external)
        )
        if not (selected_contact or link_external_contact):
            continue
        robot_geom_id = geom1 if geom1 in robot else geom2
        external_geom_id = geom2 if robot_geom_id == geom1 else geom1
        categories = []
        if selected_contact:
            categories.append("any_robot_vs_selected_obstacle")
        if link_external_contact:
            categories.append("link56_vs_external_nonrobot")
        record = {
            "schema_version": "vlsa_poisson_registered_forbidden_contact.v1",
            "source_phase": str(source_phase),
            "contact_index": int(contact_index),
            "physical_boundary": physical_boundary,
            "source_action_index": source_action_index,
            "physics_substep_index": physics_substep_index,
            "geom1_id": geom1,
            "geom2_id": geom2,
            "contact_distance_m": distance,
            "robot_geom_id": int(robot_geom_id),
            "robot_geom_name": identity[robot_geom_id]["geom_name"],
            "robot_body_id": int(identity[robot_geom_id]["body_id"]),
            "robot_body_name": identity[robot_geom_id]["body_name"],
            "external_geom_id": int(external_geom_id),
            "external_geom_name": identity[external_geom_id]["geom_name"],
            "external_body_id": int(identity[external_geom_id]["body_id"]),
            "external_body_name": identity[external_geom_id]["body_name"],
            "contact_categories": categories,
            "any_robot_selected_obstacle_contact": selected_contact,
            "link56_external_nonrobot_contact": link_external_contact,
            "link56_nonselected_external_contact": bool(
                link_external_contact and external_geom_id not in selected
            ),
        }
        record["record_sha256"] = _canonical_sha256(record)
        output.append(record)
    return output


class _RegisteredContactMonitor:
    """Observe the registered union at every completed 2 ms transition."""

    def __init__(self, sim: Any, resolved: Any) -> None:
        from main.poisson_fullbody.measurement import clone_forwarded_state

        self._model = getattr(sim.model, "_model", sim.model)
        data = getattr(sim.data, "_data", sim.data)
        self.scope = _registered_contact_scope(self._model, resolved)
        settled = clone_forwarded_state(self._model, data)
        self._settled_records = _registered_contact_records(
            self._model,
            settled,
            self.scope,
            source_phase="settled_forwarded",
        )
        self._rollout_records: List[Dict[str, Any]] = []
        self._observations = 0
        self._last_snapshot: Dict[str, Any] = {
            "literal_contact": False,
            "contacts": [],
            "contact_categories": [],
        }

    @property
    def settled_contact(self) -> bool:
        return bool(self._settled_records)

    @property
    def last_snapshot(self) -> Mapping[str, Any]:
        return self._last_snapshot

    def observe_post_integration(
        self,
        sim: Any,
        *,
        source_action_index: int,
        physics_substep_index: int,
    ) -> Mapping[str, Any]:
        from main.poisson_fullbody.measurement import clone_forwarded_state

        expected = source_action_index * 25 + physics_substep_index
        if expected != self._observations:
            raise OscCanaryRunnerError(
                "registered contact monitor gap/duplicate at physical boundary %d"
                % expected
            )
        data = getattr(sim.data, "_data", sim.data)
        common = {
            "physical_boundary": expected,
            "source_action_index": source_action_index,
            "physics_substep_index": physics_substep_index,
        }
        records = _registered_contact_records(
            self._model,
            data,
            self.scope,
            source_phase="live_solver_phase_preintegration_geometry",
            **common,
        )
        forwarded = clone_forwarded_state(self._model, data)
        records.extend(
            _registered_contact_records(
                self._model,
                forwarded,
                self.scope,
                source_phase="post_integration_recomputed",
                **common,
            )
        )
        self._rollout_records.extend(records)
        categories = sorted(
            {
                category
                for record in records
                for category in record["contact_categories"]
            }
        )
        self._last_snapshot = {
            "schema_version": "vlsa_poisson_registered_contact_substep.v1",
            "physical_boundary": expected,
            "source_action_index": source_action_index,
            "physics_substep_index": physics_substep_index,
            "literal_contact": bool(records),
            "contact_categories": categories,
            "contacts": records,
        }
        self._observations += 1
        return self._last_snapshot

    def result(self) -> Dict[str, Any]:
        records = list(self._settled_records) + list(self._rollout_records)
        rollout_categories = sorted(
            {
                category
                for record in self._rollout_records
                for category in record["contact_categories"]
            }
        )
        return {
            "schema_version": "vlsa_poisson_registered_contact_measurement.v1",
            "scope_identity_sha256": self.scope["identity_sha256"],
            "observed_physics_substeps": self._observations,
            "settled_forbidden_contact": bool(self._settled_records),
            "settled_contact_records": list(self._settled_records),
            "rollout_forbidden_contact": bool(self._rollout_records),
            "rollout_contact_categories": rollout_categories,
            "rollout_contact_records": list(self._rollout_records),
            "any_registered_forbidden_contact": bool(records),
            "any_robot_selected_obstacle_contact": any(
                record["any_robot_selected_obstacle_contact"] for record in records
            ),
            "any_link56_external_nonrobot_contact": any(
                record["link56_external_nonrobot_contact"] for record in records
            ),
            "any_link56_nonselected_external_contact": any(
                record["link56_nonselected_external_contact"] for record in records
            ),
        }


def _registered_contact_hook(scope: Mapping[str, Any]) -> Any:
    """Union solver-phase and forwarded exact-clone forbidden contacts."""

    def observe(model: Any, data: Any) -> Dict[str, Any]:
        import mujoco
        import numpy as np

        contacts = _registered_contact_records(
            model,
            data,
            scope,
            source_phase="live_solver_phase_preintegration_geometry",
        )
        specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(
            int(mujoco.mj_stateSize(model, specification)), dtype=np.float64
        )
        mujoco.mj_getState(model, data, state, specification)
        forwarded = mujoco.MjData(model)
        mujoco.mj_setState(model, forwarded, state, specification)
        mujoco.mj_forward(model, forwarded)
        contacts.extend(
            _registered_contact_records(
                model,
                forwarded,
                scope,
                source_phase="post_integration_recomputed",
            )
        )
        categories = sorted(
            {
                category
                for record in contacts
                for category in record["contact_categories"]
            }
        )
        return {
            "literal_contact": bool(contacts),
            "literal_contact_count": len(contacts),
            "contact_categories": categories,
            "contacts": contacts,
            "phases_checked": [
                "live_solver_phase_preintegration_geometry",
                "post_integration_recomputed",
            ],
        }

    return observe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/vlsa_poisson_osc_arm_link_canary.v1.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/vlsa_poisson_arm_contact_165.v1.jsonl"),
    )
    parser.add_argument("--historical-result-root", type=Path, required=True)
    parser.add_argument("--numeric-validation-result", type=Path, required=True)
    parser.add_argument("--expected-numeric-job-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", default="vlsa-t1-spatial-i-t3-e03")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    os.chdir(str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "safelibero") not in sys.path:
        sys.path.insert(0, str(root / "safelibero"))
    output: Optional[Path] = None
    result_path: Optional[Path] = None
    env: Any = None
    video: Any = None
    started = time.time()
    provenance: Dict[str, Any] = {}
    apparatus: Dict[str, Any] = {}
    partial: Dict[str, Any] = {}
    try:
        import numpy as np
        import main.evaluate_safelibero_aegis as evaluator
        from main.poisson_fullbody.contracts import publish_hashed_json
        from main.poisson_fullbody.feasibility_protocol import load_feasibility_protocol
        from main.poisson_fullbody.field_bundle import build_static_field_bundle
        from main.poisson_fullbody.jacobians import evaluate_point_jacobians
        from main.poisson_fullbody.measurement import (
            FullRobotObstacleMonitor,
            clone_forwarded_state,
            resolve_collision_geom_sets,
        )
        from main.poisson_fullbody.osc_arm_link_canary import (
            PRIMARY_CASE_ID,
            RESULT_SCHEMA,
            SOURCE_ARM,
            classify_osc_arm_link_canary,
            validate_osc_arm_link_canary_protocol,
        )
        from main.poisson_fullbody.osc_numeric_prerequisite import (
            validate_numeric_prerequisite_artifact,
        )
        from main.poisson_fullbody.post_osc_torque_sensitivity import (
            capture_post_osc_integration_state,
            clone_one_substep_transition,
            estimate_post_osc_torque_sensitivity,
        )
        from main.poisson_fullbody.post_osc_torque_shield import (
            SampledDataPostOscTorqueShield,
        )
        from main.poisson_fullbody.robot_samples import validate_rigid_roundtrip
        from main.poisson_fullbody.shadow_replay import load_historical_action_replay
        from main.poisson_fullbody.surface_sampling import (
            build_robot_collision_samples,
            validate_robot_sample_evidence,
        )
        from scripts.run_poisson_closed_loop_canary import _checkpoint_identity
        from scripts.run_poisson_shadow_parity import (
            _check_step,
            _observation_sha256,
            _prepare_environment,
        )

        if arguments.case_id != PRIMARY_CASE_ID:
            raise OscCanaryRunnerError(
                "runner is preregistered only for %s" % PRIMARY_CASE_ID
            )
        output = fast._new_output(arguments.output_root.resolve(), arguments.run_id)
        result_path = output / "result.json"
        protocol_path = (
            arguments.protocol
            if arguments.protocol.is_absolute()
            else root / arguments.protocol
        ).resolve()
        manifest_path = (
            arguments.manifest
            if arguments.manifest.is_absolute()
            else root / arguments.manifest
        ).resolve()
        protocol = fast._json(protocol_path, "post-OSC canary protocol")
        derived = validate_osc_arm_link_canary_protocol(protocol)
        selection = protocol["selection"]
        study_protocol_path = (root / selection["study_protocol_relative_path"]).resolve()
        manifest_receipt_path = (
            root / selection["manifest_receipt_relative_path"]
        ).resolve()
        remote_artifact_receipt_path = (
            root / selection["remote_artifact_receipt_relative_path"]
        ).resolve()
        if (
            fast._file_sha256(study_protocol_path)
            != selection["study_protocol_file_sha256"]
        ):
            raise OscCanaryRunnerError("registered study protocol hash differs")
        if (
            fast._file_sha256(manifest_receipt_path)
            != selection["manifest_receipt_file_sha256"]
        ):
            raise OscCanaryRunnerError("registered manifest receipt hash differs")
        if (
            fast._file_sha256(remote_artifact_receipt_path)
            != selection["remote_artifact_receipt_file_sha256"]
        ):
            raise OscCanaryRunnerError("remote source-artifact receipt hash differs")
        if fast._file_sha256(manifest_path) != protocol["selection"]["manifest_file_sha256"]:
            raise OscCanaryRunnerError("registered manifest hash differs")
        case, case_row_sha256 = _manifest_case(manifest_path, arguments.case_id)
        if case_row_sha256 != protocol["selection"]["case_row_sha256"]:
            raise OscCanaryRunnerError("registered case row hash differs")
        if not (
            case.get("clean_task_success_canary_eligible") is True
            and case.get("settled_relevant_contact") is False
            and case.get("contact_vs_car_phase") == "through_car"
            and case.get("only_link56_selected_obstacle_robot_contact") is True
        ):
            raise OscCanaryRunnerError("primary clean physical-contact case differs")
        runtime_case = _source_case(case)
        remote_artifact_receipt = fast._json(
            remote_artifact_receipt_path, "remote source-artifact receipt"
        )
        verified_remote_artifacts = _verify_remote_artifact_receipt(
            arguments.historical_result_root.resolve(),
            remote_artifact_receipt,
            case,
        )

        historical_path = _historical_path(
            arguments.historical_result_root.resolve(), case
        )
        replay = load_historical_action_replay(
            historical_path,
            expected_case_id=arguments.case_id,
            expected_arm=SOURCE_ARM,
        )
        historical = fast._json(historical_path, "historical AEGIS result")
        source_contract = protocol["historical_control"]
        if not (
            replay.result_file_sha256 == source_contract["result_file_sha256"]
            and replay.result_payload_sha256 == source_contract["result_payload_sha256"]
            and replay.executed_sequence_sha256
            == source_contract["executed_action_sequence_sha256"]
            and len(replay.actions) == source_contract["executed_action_count"]
            and replay.historical_task_success is True
            and replay.historical_car_collision is True
        ):
            raise OscCanaryRunnerError("historical control binding differs")
        direct_historical_contact = _historical_link_contact_verified(case, historical)
        if not direct_historical_contact:
            raise OscCanaryRunnerError("historical direct link-5 contact is not verified")

        runtime_path = root / protocol["runtime"]["relative_path"]
        if fast._file_sha256(runtime_path) != protocol["runtime"]["raw_file_sha256"]:
            raise OscCanaryRunnerError("runtime config hash differs")
        runtime_protocol, runtime_hashes = load_feasibility_protocol(
            runtime_path,
            expected_protocol_sha256=protocol["runtime"]["semantic_sha256"],
        )
        if runtime_hashes.parameter_block_sha256 != protocol["runtime"]["parameter_block_sha256"]:
            raise OscCanaryRunnerError("runtime parameter block differs")
        checkpoint = _checkpoint_identity(protocol["online_policy"])
        source = fast._git(root)
        if source.get("status_short"):
            raise OscCanaryRunnerError("canary requires a clean source tree")
        allocation = fast._allocation()
        evaluation_python = "/mnt/data/quanth/venvs/safety_vla/main/bin/python"
        if sys.executable != evaluation_python:
            raise OscCanaryRunnerError(
                "canary did not use the registered evaluation Python"
            )
        numeric_path = arguments.numeric_validation_result
        output_root = arguments.output_root.resolve()
        if (
            not numeric_path.is_absolute()
            or numeric_path.name != "result.json"
            or numeric_path.parent.parent != output_root
            or not numeric_path.parent.name.startswith(
                "vlsa-poisson-post-osc-numeric-"
            )
        ):
            raise OscCanaryRunnerError(
                "numeric prerequisite is outside the registered run namespace"
            )
        numeric_prerequisite = validate_numeric_prerequisite_artifact(
            numeric_path,
            contract=derived["numeric_prerequisite"],
            expected_job_id=arguments.expected_numeric_job_id,
            expected_commit=str(source.get("commit")),
            expected_python_executable=evaluation_python,
            expected_parent=numeric_path.parent,
            forbidden_job_id=str(allocation.get("slurm_job_id")),
        )
        provenance = {
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "source": source,
            "allocation": allocation,
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "evaluation_package_versions": {
                package: importlib.metadata.version(package)
                for package in ("mujoco", "numpy", "scipy", "osqp", "robosuite")
            },
            "protocol_file_sha256": fast._file_sha256(protocol_path),
            "study_protocol_file_sha256": fast._file_sha256(study_protocol_path),
            "manifest_file_sha256": fast._file_sha256(manifest_path),
            "manifest_receipt_file_sha256": fast._file_sha256(
                manifest_receipt_path
            ),
            "remote_artifact_receipt_file_sha256": fast._file_sha256(
                remote_artifact_receipt_path
            ),
            "verified_remote_source_artifacts": verified_remote_artifacts,
            "manifest_case_row_sha256": case_row_sha256,
            "historical_result_path": str(historical_path),
            "historical_result_file_sha256": replay.result_file_sha256,
            "historical_result_payload_sha256": replay.result_payload_sha256,
            "historical_control_rerun": False,
            "checkpoint_identity": checkpoint,
            "numeric_prerequisite": numeric_prerequisite,
        }

        runtime = evaluator._runtime_imports(include_aegis=True)
        env, task, observation, goal_atoms, previous_goal = _prepare_environment(
            evaluator, runtime, runtime_case, replay
        )
        controller = _controller_record(env)
        if controller["original_osc_verified"] is not True:
            raise OscCanaryRunnerError("live treatment is not original 20 Hz OSC_POSE")
        torque_units = _direct_hinge_torque_actuator_record(env)
        if not hasattr(env, "step_with_arm_control_intervention"):
            raise OscCanaryRunnerError("pre-physics OSC intervention wrapper is absent")
        obstacle_name, _ = evaluator._active_obstacle(env, observation)
        if obstacle_name != case["active_obstacle_name"] or obstacle_name != protocol["case"]["selected_obstacle_name"]:
            raise OscCanaryRunnerError("selected obstacle differs after settling")
        authority = evaluator._contact_model_authority(env, obstacle_name)
        robot_root = fast._body_id(env.sim.model, env.robots[0].robot_model.root_body)
        link_ids = tuple(
            fast._body_id(env.sim.model, name)
            for name in protocol["case"]["protected_robot_body_names"]
        )
        raw_model, raw_data = fast._raw_model_data(env.sim)
        resolved = resolve_collision_geom_sets(
            raw_model,
            robot_root_body_ids=(robot_root,),
            obstacle_root_body_ids=(int(authority["active_obstacle_root_body_id"]),),
            link56_body_ids=link_ids,
        )
        settled_official = fast._official_state(env.sim)
        bundle = build_static_field_bundle(
            raw_model,
            raw_data,
            resolved=resolved,
            protocol=runtime_protocol,
            protocol_hashes=runtime_hashes,
        )
        if not np.array_equal(settled_official, fast._official_state(env.sim)):
            raise OscCanaryRunnerError("field construction changed simulator state")
        forwarded = clone_forwarded_state(raw_model, raw_data)
        full_samples = build_robot_collision_samples(
            env.sim.model,
            forwarded,
            geom_ids=resolved.robot_geom_ids,
            epsilon_m=float(runtime_protocol["coverage"]["epsilon_m"]),
        )
        sample_evidence = {
            "sample_count": len(full_samples.samples),
            "sample_ledger_sha256": full_samples.sample_ledger_sha256,
            "samples": [sample.to_dict() for sample in full_samples.samples],
            "geom_records": list(full_samples.geom_records),
            "epsilon_m": full_samples.epsilon_m,
            "maximum_surface_cover_radius_m": full_samples.maximum_surface_cover_radius_m,
            "coverage_semantics": full_samples.coverage_semantics,
            "roundtrip": validate_rigid_roundtrip(full_samples.samples, forwarded),
        }
        validate_robot_sample_evidence(
            sample_evidence,
            resolved_geom_ids=resolved.robot_geom_ids,
            resolved_geom_names=resolved.robot_geom_names,
            resolved_body_ids=resolved.robot_body_ids,
            roundtrip_field="roundtrip",
        )
        if _canonical_sha256(sample_evidence["samples"]) != str(
            full_samples.sample_ledger_sha256
        ):
            raise OscCanaryRunnerError(
                "serialized full-robot sample ledger differs from its hash"
            )
        protected_sample_payload = {
            "epsilon_m": float(bundle.protected_samples.epsilon_m),
            "maximum_surface_cover_radius_m": float(
                bundle.protected_samples.maximum_surface_cover_radius_m
            ),
            "coverage_semantics": str(bundle.protected_samples.coverage_semantics),
            "components": [
                asdict(component) for component in bundle.protected_samples.components
            ],
            "samples": [
                sample.to_dict() for sample in bundle.protected_samples.samples
            ],
        }
        if _canonical_sha256(protected_sample_payload) != str(
            bundle.hashes.protected_samples_sha256
        ):
            raise OscCanaryRunnerError(
                "serialized protected-sample payload differs from bundle hash"
            )
        field_obstacle_boxes = [
            {
                "geom_id": int(box.geom_id),
                "center_m": [float(value) for value in box.center],
                "rotation_world_from_geom": [
                    [float(value) for value in row] for row in box.R
                ],
                "half_extents_m": [float(value) for value in box.half_extents],
            }
            for box in bundle.obstacle_boxes
        ]
        if _canonical_sha256(field_obstacle_boxes) != str(
            bundle.hashes.obstacle_geometry_sha256
        ):
            raise OscCanaryRunnerError(
                "serialized obstacle geometry differs from bundle hash"
            )
        serialized_field_certificate = {
            "field_bundle_hashes": asdict(bundle.hashes),
            "field_diagnostics": asdict(bundle.diagnostics),
            "field_obstacle_boxes": field_obstacle_boxes,
            "protected_sampling": protected_sample_payload,
        }
        limits = runtime_protocol["admissibility"]
        monitor = FullRobotObstacleMonitor(
            env.sim,
            resolved,
            full_samples.samples,
            certified_coverage_radius_m=full_samples.maximum_surface_cover_radius_m,
            max_selected_geom_surface_drift_m=float(limits["max_selected_geom_surface_drift_m"]),
            max_selected_geom_translation_drift_m=float(limits["max_selected_geom_translation_drift_m"]),
            max_selected_geom_rotation_drift_rad=float(limits["max_selected_geom_rotation_drift_rad"]),
            max_settled_obstacle_linear_speed_m_per_s=float(limits["max_selected_body_linear_speed_m_s"]),
            max_settled_obstacle_angular_speed_rad_per_s=float(limits["max_selected_body_angular_speed_rad_s"]),
            require_settled_static_motion=False,
            terminate_on_static_drift=False,
            near_contact_tolerance_m=float(runtime_protocol["safety"]["contact_margin_m"]),
            inner_updates_per_high_level_action=1,
            physics_substeps_per_inner_update=25,
        )
        if monitor.settled_state.any_robot_obstacle_contact:
            raise OscCanaryRunnerError("treatment begins in robot-selected-obstacle contact")
        registered_contact_monitor = _RegisteredContactMonitor(env.sim, resolved)
        if registered_contact_monitor.settled_contact:
            raise OscCanaryRunnerError(
                "treatment begins in a registered forbidden contact"
            )
        registered_contact_hook = _registered_contact_hook(
            registered_contact_monitor.scope
        )
        settled_obstacle = fast._obstacle_state(
            raw_model, forwarded, bundle, resolved.obstacle_body_ids
        )
        if not fast._static_admissible([settled_obstacle], runtime_protocol):
            raise OscCanaryRunnerError("selected obstacle is not static after settling")

        arm_actuators = tuple(controller["arm_actuator_indexes"])
        arm_qpos = tuple(controller["arm_qpos_indexes"])
        arm_qvel = tuple(controller["arm_qvel_indexes"])
        torque_lower = np.asarray(
            raw_model.actuator_ctrlrange[list(arm_actuators), 0], dtype=np.float64
        )
        torque_upper = np.asarray(
            raw_model.actuator_ctrlrange[list(arm_actuators), 1], dtype=np.float64
        )
        if not np.all(np.asarray(raw_model.actuator_ctrllimited)[list(arm_actuators)]):
            raise OscCanaryRunnerError("arm actuator controls are not bounded")
        shield_cfg = protocol["shield"]
        shield = SampledDataPostOscTorqueShield(
            eps_abs=float(shield_cfg["solver_eps_abs"]),
            eps_rel=float(shield_cfg["solver_eps_rel"]),
            max_iter=int(shield_cfg["solver_max_iterations"]),
            maximum_sensitivity_condition_number=float(
                shield_cfg["maximum_sensitivity_condition_number"]
            ),
        )
        planner = HybridAegisPlanner(
            evaluator=evaluator,
            runtime=runtime,
            case=runtime_case,
            task_description=str(task.language),
            historical=historical,
            host=arguments.host,
            port=arguments.port,
            replan_steps=protocol["execution"]["replan_steps"],
            model_action_horizon=protocol["execution"]["model_action_horizon"],
        )
        video = TreatmentVideo(
            evaluator,
            runtime["imageio"],
            output,
            int(protocol["video"]["fps"]),
        )
        video.append(observation, "settled_pre_action", -1)

        action_trace: List[Dict[str, Any]] = []
        physics_trace: List[Dict[str, Any]] = []
        parity_trace: List[Dict[str, Any]] = []
        static_rows: List[Dict[str, Any]] = [settled_obstacle]
        first_material_boundary: Optional[int] = None
        first_material_action: Optional[int] = None
        first_torque_divergence_boundary: Optional[int] = None
        first_torque_divergence_action: Optional[int] = None
        first_torque_divergence_correction_l2_nm: Optional[float] = None
        shield_failure_count = 0
        shield_decision_count = 0
        nominal_pass_through_exact = True
        actual_residual_pass = True
        goal_ledger: List[Dict[str, Any]] = []
        first_material_goal_snapshot: Optional[Dict[str, Any]] = None
        task_incomplete_at_first_material: Optional[bool] = None
        method_stop_record: Optional[Dict[str, Any]] = None
        partial_action_record: Optional[Dict[str, Any]] = None
        native_task_success = False
        terminal_kind = "maximum_action_count"
        site_id = fast._eef_site_id(env)
        obstacle_root_body_id = int(authority["active_obstacle_root_body_id"])
        paper_car_observation_key = "%s_pos" % obstacle_name
        if paper_car_observation_key not in observation:
            raise OscCanaryRunnerError("paper CAR obstacle observation is absent")
        task_env = getattr(env, "env", None)
        observable_body_ids = getattr(task_env, "obj_body_id", None)
        if not isinstance(observable_body_ids, Mapping) or obstacle_name not in observable_body_ids:
            raise OscCanaryRunnerError(
                "paper CAR observable root-body mapping is absent"
            )
        observable_root_body_id = int(observable_body_ids[obstacle_name])
        if observable_root_body_id != obstacle_root_body_id:
            raise OscCanaryRunnerError(
                "paper CAR observable and contact authority use different root bodies"
            )
        authority_body_names = authority.get("body_names")
        if (
            not isinstance(authority_body_names, Sequence)
            or obstacle_root_body_id < 0
            or obstacle_root_body_id >= len(authority_body_names)
        ):
            raise OscCanaryRunnerError("paper CAR root-body name authority is absent")
        observable_root_body_name = str(authority_body_names[obstacle_root_body_id])
        if observable_root_body_name != protocol["case"][
            "selected_obstacle_root_body_name"
        ]:
            raise OscCanaryRunnerError("paper CAR root-body name differs")
        (
            native_observable_record,
            settled_native_observable_value,
            settled_native_observation_cache_value,
        ) = _native_object_observable_values(
            task_env,
            observation_key=paper_car_observation_key,
            obstacle_name=obstacle_name,
            np=np,
        )
        paper_car_authority = {
            "schema_version": "vlsa_poisson_paper_car_authority.v2",
            "metric": protocol["paper_car_measurement"]["metric"],
            "paper_car_observation_key": paper_car_observation_key,
            "selected_obstacle_name": obstacle_name,
            "observable_root_body_id": observable_root_body_id,
            "observable_root_body_name": observable_root_body_name,
            "contact_authority_root_body_id": obstacle_root_body_id,
            "contact_authority_root_body_name": observable_root_body_name,
            "observable_and_contact_root_body_ids_equal": True,
            "position_source": protocol["paper_car_measurement"]["position_source"],
            "root_body_binding": protocol["paper_car_measurement"]["root_body_binding"],
            "native_observable_binding": protocol["paper_car_measurement"][
                "native_observable_binding"
            ],
            "live_root_position_role": protocol["paper_car_measurement"][
                "live_root_position_role"
            ],
            "post_integration_forwarded_pose_role": protocol[
                "paper_car_measurement"
            ]["post_integration_forwarded_pose_role"],
            "historical_settled_active_obstacle_position_sha256": (
                replay.settled_active_obstacle_position_sha256
            ),
            "native_observable": native_observable_record,
        }
        apparatus["paper_car_authority"] = dict(paper_car_authority)
        settled_paper_car_position = np.asarray(
            observation[paper_car_observation_key]
        ).copy()
        settled_obstacle_root_position_live = np.asarray(
            raw_data.xpos[obstacle_root_body_id], dtype=np.float64
        ).copy()
        settled_obstacle_root_position_forwarded = np.asarray(
            forwarded.xpos[obstacle_root_body_id], dtype=np.float64
        ).copy()
        settled_car_row = _paper_car_endpoint_row(
            source_action_index=-1,
            snapshot_kind="settled_pre_action",
            observation_key=paper_car_observation_key,
            obstacle_root_body_id=obstacle_root_body_id,
            observation_position=settled_paper_car_position,
            observable_value_position=settled_native_observable_value,
            observation_cache_position=(
                settled_native_observation_cache_value
            ),
            live_body_position_diagnostic=settled_obstacle_root_position_live,
            post_integration_forwarded_body_position=(
                settled_obstacle_root_position_forwarded
            ),
            settled_observation_position=settled_paper_car_position,
            np=np,
        )
        partial["paper_car_settled_phase_diagnostic"] = dict(settled_car_row)
        _require_paper_car_native_cache_binding(settled_car_row)
        car_ledger: List[Dict[str, Any]] = [settled_car_row]
        if (
            car_ledger[0]["active_obstacle_position_observation_array_sha256"]
            != replay.settled_active_obstacle_position_sha256
        ):
            raise OscCanaryRunnerError(
                "settled paper CAR observation differs from immutable historical pairing"
            )

        for source_index in range(int(derived["maximum_action_count"])):
            pre_action_observation_sha256 = _observation_sha256(
                observation, evaluator, np
            )
            high_level_action = np.asarray(
                planner.action(
                    env=env,
                    observation=observation,
                    source_index=source_index,
                ),
                dtype=np.float64,
            )
            if high_level_action.shape != (7,) or not np.all(np.isfinite(high_level_action)):
                raise OscCanaryRunnerError("high-level OSC action is invalid")
            pending: Dict[int, Dict[str, Any]] = {}

            def intervention(sim: Any, substep_index: int, nominal_arm_ctrl: Any) -> Any:
                nonlocal first_material_boundary, first_material_action
                nonlocal first_torque_divergence_boundary
                nonlocal first_torque_divergence_action
                nonlocal first_torque_divergence_correction_l2_nm
                nonlocal first_material_goal_snapshot
                nonlocal task_incomplete_at_first_material
                nonlocal method_stop_record
                nonlocal nominal_pass_through_exact, shield_failure_count
                nonlocal shield_decision_count
                boundary = source_index * 25 + int(substep_index)

                def method_stop(reason: str, **details: Any) -> None:
                    nonlocal method_stop_record
                    method_stop_record = {
                        "source_action_index": source_index,
                        "physics_boundary_before_unexecuted_step": boundary,
                        "reason": str(reason),
                        **details,
                    }
                    raise TreatmentMethodStop(str(reason))

                current = clone_forwarded_state(raw_model, raw_data)
                points, jacobians = evaluate_point_jacobians(
                    raw_model,
                    current,
                    bundle.protected_samples.samples,
                    arm_qvel,
                )
                queries = [bundle.field.query(point) for point in points]
                if any(
                    not query.valid or query.value is None or query.gradient is None
                    for query in queries
                ):
                    method_stop("poisson_query_invalid_stop_before_physics")
                h = np.asarray([float(query.value) for query in queries], dtype=np.float64)
                gradients = np.asarray([query.gradient for query in queries], dtype=np.float64)
                if np.any(h <= 0.0):
                    method_stop(
                        "poisson_h_nonpositive_stop_before_physics",
                        minimum_h_m2=float(np.min(h)),
                    )
                rows = np.einsum("ni,nij->nj", gradients, jacobians)
                snapshot = capture_post_osc_integration_state(
                    sim,
                    arm_actuator_ids=arm_actuators,
                    arm_qpos_indices=arm_qpos,
                    arm_qvel_indices=arm_qvel,
                )
                nominal_array = np.asarray(nominal_arm_ctrl, dtype=np.float64)
                if not (
                    np.array_equal(snapshot.nominal_arm_torque_nm, nominal_array)
                    and snapshot.nominal_arm_torque_nm.tobytes(order="C")
                    == nominal_array.tobytes(order="C")
                ):
                    raise OscCanaryRunnerError(
                        "sensitivity snapshot is not bound to the callback OSC torque"
                    )
                try:
                    nominal_transition = clone_one_substep_transition(
                        sim,
                        snapshot=snapshot,
                        arm_torque=nominal_array,
                        selected_contact_hook=registered_contact_hook,
                    )
                except Exception as error:
                    method_stop(
                        "nominal_clone_exception_stop_before_physics",
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                nominal_residuals = (
                    rows @ nominal_transition.next_arm_qvel
                    + float(shield_cfg["alpha_gain_per_s"]) * h
                    - float(shield_cfg["margin_m2_per_s"])
                )
                constraint_attribution = _minimum_constraint_attribution(
                    nominal_residuals,
                    bundle.protected_samples.samples,
                    np,
                )
                nominal_contact = nominal_transition.selected_contact_data
                if not isinstance(nominal_contact, Mapping):
                    raise OscCanaryRunnerError(
                        "nominal cloned transition lacks selected-contact evidence"
                    )
                nominal_safe = bool(
                    float(np.min(nominal_residuals)) >= 0.0
                    and nominal_contact.get("literal_contact") is False
                )
                if nominal_safe:
                    command = nominal_array.copy()
                    delta = np.zeros(7, dtype=np.float64)
                    sensitivity_record = None
                    shield_status = "nominal_safe_exact_clone"
                    shield_diagnostics = {
                        "solver_attempted": False,
                        "minimum_nominal_exact_clone_residual_m2_per_s": float(
                            np.min(nominal_residuals)
                        ),
                    }
                    nominal_pass_through_exact = bool(
                        nominal_pass_through_exact
                        and command.tobytes(order="C")
                        == nominal_array.tobytes(order="C")
                    )
                    candidate_transition = nominal_transition
                else:
                    try:
                        estimate = estimate_post_osc_torque_sensitivity(
                            sim,
                            snapshot=snapshot,
                            torque_epsilon_nm=float(
                                shield_cfg["torque_epsilon_nm"]
                            ),
                            agreement_atol=float(
                                shield_cfg["sensitivity_agreement_atol"]
                            ),
                            agreement_rtol=float(
                                shield_cfg["sensitivity_agreement_rtol"]
                            ),
                        )
                    except Exception as error:
                        method_stop(
                            "sensitivity_exception_stop_before_physics",
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                    current_qvel = np.asarray(
                        raw_data.qvel[list(arm_qvel)], dtype=np.float64
                    )
                    try:
                        result = shield.solve(
                            nominal_torque=nominal_array,
                            current_qvel=current_qvel,
                            nominal_next_qvel=estimate.nominal_next_arm_qvel_rad_s,
                            h=h,
                            joint_gradient_rows=rows,
                            dt_seconds=0.002,
                            torque_to_next_qvel_sensitivity=(
                                estimate.torque_to_next_arm_qvel_sensitivity
                            ),
                            sensitivity_max_absolute_error=float(
                                estimate.maximum_epsilon_agreement_absolute_error
                            ),
                            torque_lower=torque_lower,
                            torque_upper=torque_upper,
                            alpha=float(shield_cfg["alpha_gain_per_s"]),
                            margin=float(shield_cfg["margin_m2_per_s"]),
                        )
                    except Exception as error:
                        shield_failure_count += 1
                        method_stop(
                            "torque_shield_exception_stop_before_physics",
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                    if (
                        not result.valid
                        or result.torque_command is None
                        or result.delta_torque is None
                    ):
                        shield_failure_count += 1
                        method_stop(
                            "torque_shield_%s_stop_before_physics" % result.reason,
                            shield_status=result.reason,
                            shield_diagnostics=dict(result.diagnostics),
                        )
                    command = np.asarray(result.torque_command, dtype=np.float64)
                    delta = np.asarray(result.delta_torque, dtype=np.float64)
                    sensitivity_record = _sensitivity_record(estimate, snapshot)
                    shield_status = result.status.value
                    shield_diagnostics = dict(result.diagnostics)
                    try:
                        candidate_transition = clone_one_substep_transition(
                            sim,
                            snapshot=snapshot,
                            arm_torque=command,
                            selected_contact_hook=registered_contact_hook,
                        )
                    except Exception as error:
                        method_stop(
                            "candidate_clone_exception_stop_before_physics",
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                candidate_contact = candidate_transition.selected_contact_data
                if (
                    not isinstance(candidate_contact, Mapping)
                    or candidate_contact.get("literal_contact") is not False
                ):
                    method_stop(
                        "candidate_clone_contact_stop_before_physics",
                        candidate_contact=candidate_contact,
                    )
                candidate_residuals = (
                    rows @ candidate_transition.next_arm_qvel
                    + float(shield_cfg["alpha_gain_per_s"]) * h
                    - float(shield_cfg["margin_m2_per_s"])
                )
                candidate_minimum = float(np.min(candidate_residuals))
                if candidate_minimum < -float(
                    shield_cfg["actual_cbf_residual_tolerance_m2_per_s"]
                ):
                    method_stop(
                        "candidate_clone_cbf_residual_stop_before_physics",
                        candidate_minimum_cbf_residual_m2_per_s=candidate_minimum,
                    )
                shield_decision_count += 1
                correction = float(np.linalg.norm(delta))
                command_byte_identical_to_nominal = bool(
                    command.tobytes(order="C") == nominal_array.tobytes(order="C")
                )
                if (
                    not command_byte_identical_to_nominal
                    and first_torque_divergence_boundary is None
                ):
                    first_torque_divergence_boundary = boundary
                    first_torque_divergence_action = source_index
                    first_torque_divergence_correction_l2_nm = correction
                    planner.mark_divergence(
                        action_index=source_index,
                        physical_boundary=boundary,
                    )
                retain_full_constraint_qp_certificate = bool(
                    boundary == first_torque_divergence_boundary
                    and not command_byte_identical_to_nominal
                )
                if (
                    first_material_boundary is None
                    and correction
                    >= float(protocol["acceptance"]["material_torque_correction_l2_nm"])
                ):
                    official_before_goal = fast._official_state(env.sim)
                    goal_at_correction = evaluator._goal_progress_snapshot(
                        env,
                        goal_atoms,
                        step=source_index,
                        previous_values=previous_goal,
                    )
                    official_after_goal = fast._official_state(env.sim)
                    if not np.array_equal(official_before_goal, official_after_goal):
                        raise OscCanaryRunnerError(
                            "goal check at first material correction changed physics"
                        )
                    first_material_goal_snapshot = dict(goal_at_correction)
                    task_incomplete_at_first_material = not bool(
                        goal_at_correction["all_satisfied"]
                    )
                    first_material_boundary = boundary
                    first_material_action = source_index
                predicted = np.asarray(
                    candidate_transition.next_arm_qvel, dtype=np.float64
                )
                pending[int(substep_index)] = {
                    "physical_boundary": boundary,
                    "h": h,
                    "rows": rows,
                    "nominal_torque": nominal_array.copy(),
                    "command_torque": command.copy(),
                    "delta": delta.copy(),
                    "correction": correction,
                    "command_byte_identical_to_nominal": (
                        command_byte_identical_to_nominal
                    ),
                    "nominal_torque_sha256": hashlib.sha256(
                        nominal_array.tobytes(order="C")
                    ).hexdigest(),
                    "command_torque_sha256": hashlib.sha256(
                        command.tobytes(order="C")
                    ).hexdigest(),
                    "predicted_next_qvel": predicted,
                    "nominal_predicted_next_qvel": np.asarray(
                        nominal_transition.next_arm_qvel, dtype=np.float64
                    ).copy(),
                    "shield_status": shield_status,
                    "shield_diagnostics": shield_diagnostics,
                    "sensitivity": sensitivity_record,
                    "nominal_exact_clone_minimum_cbf_residual_m2_per_s": float(
                        np.min(nominal_residuals)
                    ),
                    "constraint_attribution": constraint_attribution,
                    "candidate_exact_clone_minimum_cbf_residual_m2_per_s": (
                        candidate_minimum
                    ),
                    "candidate_exact_clone_contact": dict(candidate_contact),
                    "retain_full_constraint_qp_certificate": (
                        retain_full_constraint_qp_certificate
                    ),
                }
                return command

            def poststep(sim: Any, substep_index: int) -> None:
                nonlocal actual_residual_pass
                row = pending.get(int(substep_index))
                if row is None:
                    raise OscCanaryRunnerError("poststep callback lacks its shield decision")
                monitor.observe_post_integration(
                    sim,
                    high_level_index=source_index,
                    inner_control_index=0,
                    physics_substep_index=int(substep_index),
                    measure_full_surface_clearance=(int(substep_index) == 24),
                )
                measurement = monitor.result()
                registered_contact_snapshot = (
                    registered_contact_monitor.observe_post_integration(
                        sim,
                        source_action_index=source_index,
                        physics_substep_index=int(substep_index),
                    )
                )
                forwarded_after = clone_forwarded_state(raw_model, raw_data)
                measured_qvel = np.asarray(raw_data.qvel[list(arm_qvel)], dtype=np.float64)
                clone_matches_live = bool(
                    np.allclose(
                        measured_qvel,
                        row["predicted_next_qvel"],
                        rtol=0.0,
                        atol=1e-10,
                    )
                )
                residuals = (
                    row["rows"] @ measured_qvel
                    + float(shield_cfg["alpha_gain_per_s"]) * row["h"]
                    - float(shield_cfg["margin_m2_per_s"])
                )
                minimum_actual_residual = float(np.min(residuals))
                actual_hdot = row["rows"] @ measured_qvel
                candidate_hdot = row["rows"] @ row["predicted_next_qvel"]
                nominal_hdot = row["rows"] @ row["nominal_predicted_next_qvel"]
                compact_arrays = {
                    "poisson_h_m2": _compact_float64_array_record(row["h"], np),
                    "joint_gradient_rows_m2_per_rad": (
                        _compact_float64_array_record(row["rows"], np)
                    ),
                    "actual_hdot_m2_per_s": _compact_float64_array_record(
                        actual_hdot, np
                    ),
                    "candidate_exact_clone_hdot_m2_per_s": (
                        _compact_float64_array_record(candidate_hdot, np)
                    ),
                    "nominal_exact_clone_hdot_m2_per_s": (
                        _compact_float64_array_record(nominal_hdot, np)
                    ),
                }
                compact_constraint_trace = {
                    "schema_version": (
                        "vlsa_poisson_compact_constraint_trace.v1"
                    ),
                    "array_hash_format": _FLOAT64_ARRAY_HASH_FORMAT,
                    "sample_count": int(row["h"].size),
                    "arrays": compact_arrays,
                }
                compact_diagnostics = _compact_shield_diagnostics(
                    row["shield_diagnostics"],
                    shield_status=str(row["shield_status"]),
                    sample_count=int(row["h"].size),
                    sensitivity_record=row["sensitivity"],
                )
                full_certificate = None
                full_certificate_sha256 = None
                if row["retain_full_constraint_qp_certificate"]:
                    if not (
                        row["shield_status"] == "solved"
                        and isinstance(row["sensitivity"], Mapping)
                        and not row["command_byte_identical_to_nominal"]
                    ):
                        raise OscCanaryRunnerError(
                            "first divergence lacks a solved full QP certificate"
                        )
                    full_certificate = {
                        "schema_version": (
                            "vlsa_poisson_first_divergence_full_qp_certificate.v1"
                        ),
                        "physical_boundary": int(row["physical_boundary"]),
                        "sample_count": int(row["h"].size),
                        "poisson_h_m2": row["h"].tolist(),
                        "joint_gradient_rows_m2_per_rad": row["rows"].tolist(),
                        "actual_hdot_m2_per_s": actual_hdot.tolist(),
                        "candidate_exact_clone_hdot_m2_per_s": (
                            candidate_hdot.tolist()
                        ),
                        "nominal_exact_clone_hdot_m2_per_s": (
                            nominal_hdot.tolist()
                        ),
                        "constraint_attribution": row[
                            "constraint_attribution"
                        ],
                        "sensitivity": row["sensitivity"],
                        "shield_diagnostics": row["shield_diagnostics"],
                    }
                    full_certificate_sha256 = _canonical_sha256(full_certificate)
                residual_ok = bool(
                    minimum_actual_residual
                    >= -float(shield_cfg["actual_cbf_residual_tolerance_m2_per_s"])
                )
                actual_residual_pass = bool(
                    actual_residual_pass and residual_ok and clone_matches_live
                )
                obstacle = fast._obstacle_state(
                    raw_model,
                    forwarded_after,
                    bundle,
                    resolved.obstacle_body_ids,
                )
                static_rows.append(obstacle)
                eef_position = np.asarray(
                    forwarded_after.site_xpos[site_id], dtype=np.float64
                )
                physics_trace.append(
                    {
                        "physical_boundary": int(row["physical_boundary"]),
                        "post_integration_boundary": int(row["physical_boundary"]) + 1,
                        "source_action_index": int(source_index),
                        "physics_substep_index": int(substep_index),
                        "shield_status": str(row["shield_status"]),
                        "nominal_torque_nm": row["nominal_torque"].tolist(),
                        "command_torque_nm": row["command_torque"].tolist(),
                        "torque_delta_nm": row["delta"].tolist(),
                        "torque_correction_l2_nm": float(row["correction"]),
                        "command_byte_identical_to_nominal": row[
                            "command_byte_identical_to_nominal"
                        ],
                        "nominal_torque_sha256": row["nominal_torque_sha256"],
                        "command_torque_sha256": row["command_torque_sha256"],
                        "measured_qvel_rad_s": measured_qvel.tolist(),
                        "measured_qvel_l2_rad_s": float(np.linalg.norm(measured_qvel)),
                        "predicted_next_qvel_rad_s": row["predicted_next_qvel"].tolist(),
                        "nominal_predicted_next_qvel_rad_s": row[
                            "nominal_predicted_next_qvel"
                        ].tolist(),
                        "prediction_error_l2_rad_s": float(
                            np.linalg.norm(measured_qvel - row["predicted_next_qvel"])
                        ),
                        "candidate_exact_clone_matches_live_qvel": clone_matches_live,
                        "constraint_trace": compact_constraint_trace,
                        "nominal_exact_clone_minimum_cbf_residual_m2_per_s": row[
                            "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
                        ],
                        "candidate_exact_clone_minimum_cbf_residual_m2_per_s": row[
                            "candidate_exact_clone_minimum_cbf_residual_m2_per_s"
                        ],
                        "candidate_exact_clone_contact": row[
                            "candidate_exact_clone_contact"
                        ],
                        "minimum_actual_cbf_residual_m2_per_s": minimum_actual_residual,
                        "actual_cbf_residual_postcheck_pass": residual_ok,
                        "eef_position_world_m": eef_position.tolist(),
                        "selected_obstacle": obstacle,
                        "literal_robot_selected_obstacle_contact_seen": bool(
                            measurement.any_robot_obstacle_contact
                        ),
                        "registered_contact_scope_checked": True,
                        "registered_forbidden_contact_seen": bool(
                            registered_contact_snapshot["literal_contact"]
                        ),
                        "registered_contact_categories": list(
                            registered_contact_snapshot["contact_categories"]
                        ),
                        "registered_forbidden_contacts": list(
                            registered_contact_snapshot["contacts"]
                        ),
                        "compact_shield_diagnostics": compact_diagnostics,
                        "full_constraint_qp_certificate": full_certificate,
                        "full_constraint_qp_certificate_sha256": (
                            full_certificate_sha256
                        ),
                    }
                )
                if not clone_matches_live:
                    raise OscCanaryRunnerError(
                        "exact candidate clone and live one-step qvel differ"
                    )
                if not residual_ok:
                    raise OscCanaryRunnerError(
                        "actual one-step CBF residual postcheck failed"
                    )
                if not fast._static_admissible([obstacle], runtime_protocol):
                    raise OscCanaryRunnerError(
                        "selected obstacle left the frozen-field envelope"
                    )
                if registered_contact_snapshot["literal_contact"]:
                    raise TreatmentContactObserved()

            action_had_contact = False
            action_method_stopped = False
            try:
                observation_after, reward, done, info = env.step_with_arm_control_intervention(
                    high_level_action,
                    intervention,
                    poststep_callback=poststep,
                    expected_substeps=25,
                )
            except TreatmentContactObserved:
                action_had_contact = True
                terminal_kind = "literal_registered_forbidden_contact"
                official_before = fast._official_state(env.sim)
                env.env._update_observables(force=True)
                observation_after = env.env._get_observations()
                if not np.array_equal(official_before, fast._official_state(env.sim)):
                    raise OscCanaryRunnerError("terminal observable refresh changed physics")
                video.append(observation_after, "terminal_contact", source_index)
            except TreatmentMethodStop:
                action_method_stopped = True
                terminal_kind = "safety_method_stop_before_physics"
                official_before = fast._official_state(env.sim)
                env.env._update_observables(force=True)
                observation_after = env.env._get_observations()
                if not np.array_equal(official_before, fast._official_state(env.sim)):
                    raise OscCanaryRunnerError(
                        "terminal observable refresh changed physics"
                    )
                video.append(
                    observation_after,
                    "terminal_safety_method_stop_before_physics",
                    source_index,
                )
            if action_had_contact or action_method_stopped:
                if partial_action_record is not None:
                    raise OscCanaryRunnerError(
                        "multiple partial high-level actions were recorded"
                    )
                completed_partial_substeps = (
                    len(physics_trace) - source_index * 25
                )
                planner_row = planner.action_trace[-1]
                executed_action_sha256 = evaluator.array_sha256(high_level_action)
                if not (
                    int(planner_row.get("source_action_index", -1)) == source_index
                    and planner_row.get("executed_action_sha256")
                    == executed_action_sha256
                    and planner_row.get("native_observation_sha256")
                    == pre_action_observation_sha256
                ):
                    raise OscCanaryRunnerError(
                        "partial action is not bound to its planner row"
                    )
                if action_had_contact:
                    valid_partial_count = 1 <= completed_partial_substeps <= 25
                    terminal_observation_kind = "post_contact_observable_refresh"
                else:
                    valid_partial_count = 0 <= completed_partial_substeps < 25
                    terminal_observation_kind = (
                        "post_method_stop_observable_refresh"
                    )
                if not valid_partial_count:
                    raise OscCanaryRunnerError(
                        "partial action substep count is invalid"
                    )
                partial_action_record = {
                    "schema_version": "vlsa_poisson_partial_action_record.v1",
                    "terminal_kind": terminal_kind,
                    "source_action_index": int(source_index),
                    "planner_action_trace_index": len(planner.action_trace) - 1,
                    "executed_high_level_action": high_level_action.tolist(),
                    "executed_high_level_action_sha256": (
                        executed_action_sha256
                    ),
                    "planner_executed_action_sha256": planner_row[
                        "executed_action_sha256"
                    ],
                    "planner_native_observation_sha256": planner_row[
                        "native_observation_sha256"
                    ],
                    "pre_action_observation_sha256": (
                        pre_action_observation_sha256
                    ),
                    "completed_physics_substeps": int(
                        completed_partial_substeps
                    ),
                    "physics_boundary_start": int(source_index * 25),
                    "physics_boundary_end_exclusive": len(physics_trace),
                    "terminal_observation_kind": terminal_observation_kind,
                    "terminal_observation_sha256": _observation_sha256(
                        observation_after, evaluator, np
                    ),
                    "terminal_forbidden_contact_categories": (
                        list(
                            registered_contact_monitor.last_snapshot.get(
                                "contact_categories", ()
                            )
                        )
                        if action_had_contact
                        else []
                    ),
                    "terminal_forbidden_contact_records_sha256": (
                        _canonical_sha256(
                            registered_contact_monitor.last_snapshot.get(
                                "contacts", ()
                            )
                        )
                        if action_had_contact
                        else _canonical_sha256([])
                    ),
                }
            if action_had_contact:
                break
            if action_method_stopped:
                break
            if len(pending) != 25:
                raise OscCanaryRunnerError("shield did not run at every 2 ms substep")
            observation = observation_after
            state_hash = evaluator.array_sha256(env.sim.get_state().flatten())
            observation_hash = _observation_sha256(observation, evaluator, np)
            pre_correction_complete_action = bool(
                first_torque_divergence_action is None
                or source_index < int(first_torque_divergence_action)
            )
            if pre_correction_complete_action:
                state_hash, observation_hash, previous_goal = _check_step(
                    evaluator=evaluator,
                    env=env,
                    observation=observation,
                    reward=reward,
                    done=done,
                    expected_step=replay.steps[source_index],
                    goal_atoms=goal_atoms,
                    previous_goal_values=previous_goal,
                    np=np,
                )
                parity_trace.append(
                    {
                        "source_action_index": source_index,
                        "simulator_state_sha256": state_hash,
                        "observation_sha256": observation_hash,
                        "historical_match": True,
                    }
                )
                goal_values = tuple(previous_goal)
                goal_all = bool(done)
            else:
                goal = evaluator._goal_progress_snapshot(
                    env,
                    goal_atoms,
                    step=source_index,
                    previous_values=previous_goal,
                )
                if bool(done) is not bool(goal["all_satisfied"]):
                    raise OscCanaryRunnerError("native done differs from BDDL goal")
                previous_goal = goal["values"]
                goal_values = tuple(goal["values"])
                goal_all = bool(goal["all_satisfied"])
            goal_ledger.append(
                {
                    "source_action_index": source_index,
                    "values": list(goal_values),
                    "all_satisfied": goal_all,
                    "reward": float(reward),
                    "done": bool(done),
                    "simulator_state_sha256": state_hash,
                    "observation_sha256": observation_hash,
                }
            )
            action_trace.append(
                {
                    "source_action_index": source_index,
                    "executed_high_level_action": high_level_action.tolist(),
                    "executed_high_level_action_sha256": evaluator.array_sha256(
                        high_level_action
                    ),
                    "completed_physics_substeps": 25,
                    "state_sha256": state_hash,
                    "observation_sha256": observation_hash,
                    "task_success": goal_all,
                }
            )
            endpoint_forwarded = clone_forwarded_state(raw_model, raw_data)
            endpoint_root_position_live = np.asarray(
                raw_data.xpos[obstacle_root_body_id], dtype=np.float64
            ).copy()
            endpoint_root_position_forwarded = np.asarray(
                endpoint_forwarded.xpos[obstacle_root_body_id], dtype=np.float64
            ).copy()
            endpoint_paper_car_position = np.asarray(
                observation[paper_car_observation_key]
            ).copy()
            (
                endpoint_native_observable_record,
                endpoint_native_observable_value,
                endpoint_native_observation_cache_value,
            ) = _native_object_observable_values(
                task_env,
                observation_key=paper_car_observation_key,
                obstacle_name=obstacle_name,
                np=np,
            )
            if endpoint_native_observable_record != native_observable_record:
                raise OscCanaryRunnerError(
                    "paper CAR native observable identity changed"
                )
            endpoint_car_row = _paper_car_endpoint_row(
                source_action_index=source_index,
                snapshot_kind="completed_high_level_endpoint",
                observation_key=paper_car_observation_key,
                obstacle_root_body_id=obstacle_root_body_id,
                observation_position=endpoint_paper_car_position,
                observable_value_position=endpoint_native_observable_value,
                observation_cache_position=(
                    endpoint_native_observation_cache_value
                ),
                live_body_position_diagnostic=endpoint_root_position_live,
                post_integration_forwarded_body_position=(
                    endpoint_root_position_forwarded
                ),
                settled_observation_position=settled_paper_car_position,
                np=np,
            )
            partial["last_paper_car_endpoint_phase_diagnostic"] = dict(
                endpoint_car_row
            )
            _require_paper_car_native_cache_binding(endpoint_car_row)
            car_ledger.append(endpoint_car_row)
            video.append(observation, "completed_high_level_action", source_index)
            evaluator._update_eef_marker(
                env, evaluator._eef_proxy(runtime, observation)
            )
            if goal_all:
                native_task_success = True
                terminal_kind = "native_task_success"
                break

        measurement = monitor.result()
        registered_contact_measurement = registered_contact_monitor.result()
        planner_record = planner.record()
        static_trace_sha256 = _canonical_sha256(static_rows)
        motion = _post_correction_motion(physics_trace, first_material_boundary)
        completed_actions = len(action_trace)
        partial_action_substeps = len(physics_trace) - completed_actions * 25
        expected_executed_substeps = completed_actions * 25 + partial_action_substeps
        solved_qp_count = sum(
            row["shield_status"] == "solved" for row in physics_trace
        )
        full_qp_certificate_count = sum(
            isinstance(row.get("full_constraint_qp_certificate"), Mapping)
            for row in physics_trace
        )
        video_record = video.close()
        video = None
        static_admissible = fast._static_admissible(static_rows, runtime_protocol)
        maximum_car_displacement = max(
            float(row["l1_displacement_from_settled_m"]) for row in car_ledger
        )
        treatment_paper_car_avoided = bool(
            maximum_car_displacement
            <= float(
                protocol["acceptance"]["paper_car_l1_displacement_threshold_m"]
            )
        )
        all_shield_valid = bool(
            method_stop_record is None
            and shield_failure_count == 0
            and shield_decision_count == len(physics_trace)
            and len(physics_trace) == expected_executed_substeps
            and all(
                row["shield_status"]
                in ("nominal_safe_exact_clone", "nominal_safe", "solved")
                for row in physics_trace
            )
        )
        complete_terminal = bool(
            terminal_kind
            in (
                "native_task_success",
                "literal_registered_forbidden_contact",
                "safety_method_stop_before_physics",
                "maximum_action_count",
            )
            and (
                native_task_success
                or registered_contact_measurement["rollout_forbidden_contact"]
                or method_stop_record is not None
                or completed_actions == int(derived["maximum_action_count"])
            )
        )
        contact_free_before_material = bool(
            first_material_boundary is not None
            and all(
                row["registered_forbidden_contact_seen"] is False
                for row in physics_trace
                if int(row["physical_boundary"]) < int(first_material_boundary)
            )
        )
        contact_free_before_divergence = bool(
            first_torque_divergence_boundary is not None
            and all(
                row["registered_forbidden_contact_seen"] is False
                for row in physics_trace
                if int(row["physical_boundary"])
                < int(first_torque_divergence_boundary)
            )
        )
        predivergence_torque_commands_exact = bool(
            all(
                row["command_byte_identical_to_nominal"] is True
                for row in physics_trace
                if first_torque_divergence_boundary is None
                or int(row["physical_boundary"])
                < int(first_torque_divergence_boundary)
            )
        )
        action62_same_action_exception_parity = bool(
            first_torque_divergence_action is not None
            and (
                int(first_torque_divergence_action)
                < int(derived["historical_contact_action"])
                or (
                    int(first_torque_divergence_action)
                    == int(derived["historical_contact_action"])
                    and any(
                        int(row["source_action_index"])
                        == int(derived["historical_contact_action"]) - 1
                        and row["historical_match"] is True
                        for row in parity_trace
                    )
                )
            )
        )
        first_divergence_is_material = bool(
            first_torque_divergence_boundary is not None
            and first_material_boundary == first_torque_divergence_boundary
            and first_torque_divergence_correction_l2_nm is not None
            and first_torque_divergence_correction_l2_nm
            >= float(protocol["acceptance"]["material_torque_correction_l2_nm"])
        )
        material_before = bool(
            first_material_action is not None
            and first_divergence_is_material
            and (
                int(first_material_action) < int(derived["historical_contact_action"])
                or (
                    int(first_material_action)
                    == int(derived["historical_contact_action"])
                    and contact_free_before_material
                    and contact_free_before_divergence
                    and predivergence_torque_commands_exact
                    and action62_same_action_exception_parity
                )
            )
        )
        first_material_row = next(
            (
                row
                for row in physics_trace
                if first_material_boundary is not None
                and int(row["physical_boundary"]) == int(first_material_boundary)
            ),
            None,
        )
        first_material_residual_improvement = (
            None
            if first_material_row is None
            else float(
                first_material_row[
                    "candidate_exact_clone_minimum_cbf_residual_m2_per_s"
                ]
                - first_material_row[
                    "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
                ]
            )
        )
        first_material_next_qvel_change = (
            None
            if first_material_row is None
            else float(
                np.linalg.norm(
                    np.asarray(
                        first_material_row["predicted_next_qvel_rad_s"],
                        dtype=np.float64,
                    )
                    - np.asarray(
                        first_material_row[
                            "nominal_predicted_next_qvel_rad_s"
                        ],
                        dtype=np.float64,
                    )
                )
            )
        )
        first_divergence_row = next(
            (
                row
                for row in physics_trace
                if first_torque_divergence_boundary is not None
                and int(row["physical_boundary"])
                == int(first_torque_divergence_boundary)
            ),
            None,
        )
        first_divergence_attribution = (
            None
            if first_divergence_row is None
            else first_divergence_row["full_constraint_qp_certificate"][
                "constraint_attribution"
            ]
        )
        metrics = {
            "allocation_numeric_prerequisite_verified": True,
            "historical_control_verified": True,
            "historical_direct_link56_contact_verified": direct_historical_contact,
            "historical_control_task_success": replay.historical_task_success,
            "direct_unit_gain_hinge_torque_actuators_verified": torque_units[
                "verified"
            ],
            "original_osc_controller_verified": controller["original_osc_verified"],
            "static_field_admissible": static_admissible,
            "every_executed_substep_shielded": bool(
                shield_decision_count == len(physics_trace)
                and len(physics_trace) == expected_executed_substeps
            ),
            "every_executed_substep_contact_monitored": bool(
                measurement.observed_physics_substeps == len(physics_trace)
                and registered_contact_measurement["observed_physics_substeps"]
                == len(physics_trace)
            ),
            "every_executed_substep_registered_contact_scope_monitored": bool(
                registered_contact_measurement["observed_physics_substeps"]
                == len(physics_trace)
                and all(
                    row["registered_contact_scope_checked"] is True
                    for row in physics_trace
                )
            ),
            "nominal_pass_through_bitwise_exact": nominal_pass_through_exact,
            "pre_correction_historical_parity_exact": bool(
                len(parity_trace)
                == (
                    completed_actions
                    if first_torque_divergence_action is None
                    else int(first_torque_divergence_action)
                )
            ),
            "all_predivergence_torque_commands_byte_identical_nominal": (
                predivergence_torque_commands_exact
            ),
            "action62_same_action_exception_has_action61_endpoint_parity_exact": (
                action62_same_action_exception_parity
            ),
            "all_live_substeps_before_first_divergence_contact_free": (
                contact_free_before_divergence
            ),
            "no_policy_query_before_divergence": planner_record[
                "no_policy_query_before_divergence"
            ],
            "cached_current_chunk_then_fresh_own_observation_policy": planner_record[
                "hybrid_contract_valid"
            ],
            "all_shield_decisions_valid": all_shield_valid,
            "safety_method_stop_before_physics": method_stop_record is not None,
            "all_actual_cbf_residual_postchecks_pass": actual_residual_pass,
            "complete_terminal_condition_reached": complete_terminal,
            "video_complete": bool(
                video_record["decoded_successfully"]
                and video_record["frame_count"] >= completed_actions + 1
            ),
            "any_robot_selected_obstacle_contact": bool(
                registered_contact_measurement[
                    "any_robot_selected_obstacle_contact"
                ]
            ),
            "any_link56_external_nonrobot_contact": bool(
                registered_contact_measurement[
                    "any_link56_external_nonrobot_contact"
                ]
            ),
            "any_link56_nonselected_external_contact": bool(
                registered_contact_measurement[
                    "any_link56_nonselected_external_contact"
                ]
            ),
            "any_registered_forbidden_contact": bool(
                registered_contact_measurement[
                    "any_registered_forbidden_contact"
                ]
            ),
            "contact_scope": protocol["execution"]["contact_scope"],
            "link56_selected_obstacle_contact": bool(
                measurement.link56_obstacle_contact
            ),
            "material_correction_present": first_material_boundary is not None,
            "first_torque_divergence_is_material": first_divergence_is_material,
            "first_torque_divergence_physical_boundary": (
                first_torque_divergence_boundary
            ),
            "first_torque_divergence_source_action_index": (
                first_torque_divergence_action
            ),
            "first_torque_divergence_correction_l2_nm": (
                first_torque_divergence_correction_l2_nm
            ),
            "first_divergence_minimum_constraint_body_name": (
                None
                if first_divergence_attribution is None
                else first_divergence_attribution["minimum_sample"]["body_name"]
            ),
            "first_divergence_near_minimum_constraint_body_names": (
                []
                if first_divergence_attribution is None
                else first_divergence_attribution["near_minimum_body_names"]
            ),
            "first_divergence_negative_constraint_body_names": (
                []
                if first_divergence_attribution is None
                else first_divergence_attribution[
                    "negative_nominal_residual_body_names"
                ]
            ),
            "first_divergence_nominal_exact_cbf_residual_negative": bool(
                first_divergence_row is not None
                and float(
                    first_divergence_row[
                        "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
                    ]
                )
                < 0.0
            ),
            "first_material_exact_cbf_residual_improvement_m2_per_s": (
                first_material_residual_improvement
            ),
            "first_material_exact_next_qvel_change_l2_rad_s": (
                first_material_next_qvel_change
            ),
            "task_incomplete_at_first_material_correction": (
                task_incomplete_at_first_material
            ),
            "material_correction_before_historical_contact": material_before,
            "live_substeps_before_material_correction_contact_free": (
                contact_free_before_material
            ),
            "historical_contact_timing_semantics": (
                "post_action_62_sampled_endpoint;_internal_2ms_historical_contact_"
                "boundary_unknown"
            ),
            "first_material_correction_physical_boundary": first_material_boundary,
            "first_material_correction_source_action_index": first_material_action,
            "historical_first_link_contact_source_action_index": int(
                derived["historical_contact_action"]
            ),
            "native_task_success": native_task_success,
            "treatment_paper_car_avoided": treatment_paper_car_avoided,
            "maximum_active_obstacle_l1_displacement_at_completed_action_endpoints_m": (
                maximum_car_displacement
            ),
            "completed_action_count": completed_actions,
            "executed_physics_substep_count": len(physics_trace),
            "shield_decision_count": shield_decision_count,
            "solved_qp_count": int(solved_qp_count),
            "first_divergence_full_qp_certificate_count": int(
                full_qp_certificate_count
            ),
            "post_correction_joint_motion_integral_rad": motion[
                "joint_motion_integral_rad"
            ],
            "post_correction_eef_path_length_m": motion["eef_path_length_m"],
            "post_correction_zero_torque_delta_fraction": motion[
                "zero_torque_delta_fraction"
            ],
        }
        classification = classify_osc_arm_link_canary(metrics, protocol)
        candidate = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "protocol_id": protocol["protocol_id"],
            "case_id": arguments.case_id,
            "run_id": arguments.run_id,
            "provenance": provenance,
            "historical_control": {
                **replay.provenance(),
                "rerun": False,
                "direct_link56_selected_obstacle_contact_verified": direct_historical_contact,
            },
            "apparatus": {
                "controller": controller,
                "arm_actuator_torque_units": torque_units,
                "resolved_geometry": resolved.to_dict(),
                "registered_contact_scope": registered_contact_monitor.scope,
                "field_bundle_hashes": asdict(bundle.hashes),
                "field_diagnostics": asdict(bundle.diagnostics),
                "field_obstacle_boxes": field_obstacle_boxes,
                "protected_sampling": protected_sample_payload,
                "serialized_field_certificate_sha256": _canonical_sha256(
                    serialized_field_certificate
                ),
                "field_frame": protocol["shield"]["field_frame"],
                "full_robot_sampling": sample_evidence,
                "full_robot_sampling_sha256": _canonical_sha256(sample_evidence),
                "settled_obstacle": settled_obstacle,
                "paper_car_authority": paper_car_authority,
            },
            "planner": planner_record,
            "treatment": {
                "terminal_kind": terminal_kind,
                "safety_method_stop": method_stop_record,
                "partial_action_record": partial_action_record,
                "action_trace": action_trace,
                "parity_trace": parity_trace,
                "goal_progress_ledger": goal_ledger,
                "first_material_correction_goal_snapshot": (
                    first_material_goal_snapshot
                ),
                "paper_car_endpoint_ledger": car_ledger,
                "paper_car_endpoint_ledger_schema_version": (
                    "vlsa_poisson_paper_car_endpoint_ledger.v3"
                ),
                "physics_trace_schema_version": (
                    "vlsa_poisson_osc_arm_link_compact_physics_trace.v1"
                ),
                "full_qp_certificate_scope": (
                    "first_byte_different_torque_row_only"
                ),
                "physics_trace": physics_trace,
                "measurement": measurement.to_dict(),
                "registered_contact_measurement": registered_contact_measurement,
                "static_obstacle_trace": static_rows,
                "static_obstacle_trace_sha256": static_trace_sha256,
            },
            "video": video_record,
            "motion": motion,
            "metrics": metrics,
            "classification": classification,
            "claim_scope": protocol["claim_scope"],
            "formal_joint_velocity_cbf_guarantee_claimed": False,
            "producer_classification_is_preliminary": True,
            "scientific_interpretation_requires_independent_consumer": True,
            "partial_output_interpreted": False,
            "timing": {
                "started_unix": started,
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            },
        }
        publish_hashed_json(result_path, candidate)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "classification": classification["classification"],
                    "feasible": classification["feasible"],
                    "result": str(result_path),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if output is None:
            try:
                output = fast._new_output(
                    arguments.output_root.resolve(), arguments.run_id
                )
                result_path = output / "result.json"
            except Exception:
                output = None
        if video is not None:
            try:
                partial["video"] = video.close()
            except Exception as video_error:
                partial["video_failure"] = {
                    "type": type(video_error).__name__,
                    "message": str(video_error),
                }
        if result_path is not None and not result_path.exists():
            try:
                from main.poisson_fullbody.contracts import publish_hashed_json
                from main.poisson_fullbody.osc_arm_link_canary import RESULT_SCHEMA

                publish_hashed_json(
                    result_path,
                    {
                        "schema_version": RESULT_SCHEMA,
                        "status": "apparatus_failure",
                        "scientific_result": False,
                        "run_id": arguments.run_id,
                        "case_id": arguments.case_id,
                        "provenance": provenance,
                        "apparatus": apparatus,
                        "partial_evidence_not_interpreted": partial,
                        "partial_output_interpreted": False,
                        "failure": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                        "timing": {
                            "started_unix": started,
                            "finished_unix": time.time(),
                            "elapsed_seconds": time.time() - started,
                        },
                    },
                )
            except Exception as publication_error:
                print(
                    "post-OSC failure publication failed: %s" % publication_error,
                    file=sys.stderr,
                )
        print("post-OSC canary failed: %s" % error, file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
