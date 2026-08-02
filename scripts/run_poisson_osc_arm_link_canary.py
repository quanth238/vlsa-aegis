#!/usr/bin/env python3
"""Run one treatment-only SafeLIBERO post-OSC Poisson canary.

The immutable completed AEGIS / OSC episode is the control; it is not rerun.
The live treatment uses the same archived high-level actions until the first
material Poisson torque correction.  It then consumes the remainder of that
already-current pi0.5 chunk and only afterwards queries pi0.5 from its own
observations.  The v4 experiment constrains the exact ordered link-5/link-6
surface ledger used to construct the static field before each 2 ms MuJoCo
step.  Every robot geom remains monitored against the selected obstacle and
literal link-5/link-6 geoms remain monitored against every external nonrobot
geom.  Only the seven original arm torques may change and every non-arm control
stays nominal.  Any invalid field, sensitivity, QP, or postcheck stops the run
without a pass-through fallback.  The completed whole-manipulator v3 contract
remains supported as immutable historical apparatus.
"""

from __future__ import annotations

import argparse
import collections
import copy
import gzip
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


def _settled_field_query_diagnostics(
    *,
    samples: Sequence[Any],
    world_points: Any,
    queries: Sequence[Any],
    field: Any,
    sample_ledger_sha256: str,
    np: Any,
) -> Dict[str, Any]:
    """Serialize every invalid settled query with typed, reproducible context."""

    points = np.asarray(world_points, dtype=np.float64)
    if points.shape != (len(samples), 3) or len(queries) != len(samples):
        raise OscCanaryRunnerError("settled field-query diagnostic shape differs")
    lower = np.asarray(field.lower, dtype=np.float64)
    upper = np.asarray(field.upper, dtype=np.float64)
    spacing = np.asarray(field.spacing, dtype=np.float64)
    shape = tuple(int(value) for value in field.shape)
    magnitude = max(
        1.0,
        float(np.max(np.abs(lower))),
        float(np.max(np.abs(upper))),
    )
    bound_tolerance = 16.0 * float(np.finfo(np.float64).eps) * magnitude
    maximum_index = np.asarray(shape, dtype=np.int64) - 1
    invalid_rows: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = collections.Counter()
    geom_counts: Dict[str, int] = collections.Counter()
    for index, (sample, point, query) in enumerate(
        zip(samples, points, queries)
    ):
        complete = bool(
            query.valid
            and query.value is not None
            and query.gradient is not None
            and query.outer_boundary_clearance_m is not None
        )
        if complete:
            continue
        reason = getattr(query.reason, "value", None)
        if not isinstance(reason, str) or not reason:
            reason = "incomplete_query_without_typed_reason"
        reason_counts[reason] += 1
        geom_key = "%d:%s" % (int(sample.geom_id), str(sample.geom_name))
        geom_counts[geom_key] += 1
        finite_point = bool(np.all(np.isfinite(point)))
        grid_coordinate = (
            (point - lower) / spacing
            if finite_point
            else np.full(3, np.nan, dtype=np.float64)
        )
        incident_per_axis: List[List[int]] = []
        axes_on_internal_faces: List[int] = []
        if finite_point:
            for axis in range(3):
                coordinate = float(grid_coordinate[axis])
                nearest = int(round(coordinate))
                scaled_tolerance = bound_tolerance / float(spacing[axis])
                if abs(coordinate - nearest) <= scaled_tolerance:
                    candidates = [
                        value
                        for value in (nearest - 1, nearest)
                        if 0 <= value < int(maximum_index[axis])
                    ]
                    if len(candidates) > 1:
                        axes_on_internal_faces.append(axis)
                else:
                    floor_value = int(math.floor(coordinate))
                    candidates = (
                        [floor_value]
                        if 0 <= floor_value < int(maximum_index[axis])
                        else []
                    )
                incident_per_axis.append(candidates)
        else:
            incident_per_axis = [[], [], []]
        incident_cells: List[Dict[str, Any]] = []
        if all(incident_per_axis):
            for i in incident_per_axis[0]:
                for j in incident_per_axis[1]:
                    for k in incident_per_axis[2]:
                        cell_values = np.asarray(
                            field.values[i : i + 2, j : j + 2, k : k + 2],
                            dtype=np.float64,
                        )
                        vertex_mask = np.asarray(
                            field.valid_mask[i : i + 2, j : j + 2, k : k + 2]
                        )
                        incident_cells.append(
                            {
                                "cell_index": [int(i), int(j), int(k)],
                                "valid_cell": bool(field.valid_cell_mask[i, j, k]),
                                "all_vertices_valid": bool(
                                    vertex_mask.shape == (2, 2, 2)
                                    and np.all(vertex_mask)
                                ),
                                "all_values_finite": bool(
                                    cell_values.shape == (2, 2, 2)
                                    and np.all(np.isfinite(cell_values))
                                ),
                                "all_values_zero": bool(
                                    cell_values.shape == (2, 2, 2)
                                    and np.all(cell_values == 0.0)
                                ),
                            }
                        )
        direct_outer_clearance = None
        if finite_point:
            direct_outer_clearance = float(
                np.min(np.concatenate((point - lower, upper - point)))
            )
        row = {
            "sample_index": int(index),
            "sample_id": int(sample.sample_id),
            "sample_identity": sample.to_dict(),
            "world_point_m": [float(value) for value in point],
            "world_point_array_record": _compact_float64_array_record(point, np),
            "reason": reason,
            "query_valid": bool(query.valid),
            "query_value_m2": (
                None if query.value is None else float(query.value)
            ),
            "query_gradient_m": (
                None
                if query.gradient is None
                else [float(value) for value in query.gradient]
            ),
            "query_cell_index": (
                None
                if query.cell_index is None
                else [int(value) for value in query.cell_index]
            ),
            "query_local_coordinates": (
                None
                if query.local_coordinates is None
                else [float(value) for value in query.local_coordinates]
            ),
            "query_outer_boundary_clearance_m": (
                None
                if query.outer_boundary_clearance_m is None
                else float(query.outer_boundary_clearance_m)
            ),
            "unclamped_grid_coordinate": [
                float(value) for value in grid_coordinate
            ],
            "axes_on_internal_faces": axes_on_internal_faces,
            "incident_cells": incident_cells,
            "direct_outer_boundary_clearance_m": direct_outer_clearance,
        }
        row["row_sha256"] = _canonical_sha256(row)
        invalid_rows.append(row)
    evidence = {
        "schema_version": "vlsa_poisson_settled_field_query_diagnostics.v1",
        "sample_count": len(samples),
        "sample_ledger_sha256": str(sample_ledger_sha256),
        "invalid_query_count": len(invalid_rows),
        "all_invalid_queries_serialized": True,
        "reason_counts": dict(sorted(reason_counts.items())),
        "geom_counts": dict(sorted(geom_counts.items())),
        "invalid_rows": invalid_rows,
        "invalid_rows_sha256": _canonical_sha256(invalid_rows),
        "partial_output_interpreted": False,
    }
    evidence["evidence_sha256"] = _canonical_sha256(evidence)
    return evidence


def _robot_tree_qvel_binding(
    model: Any,
    resolved: Any,
    *,
    arm_qvel_indices: Sequence[int],
    arm_actuator_ids: Sequence[int],
) -> Dict[str, Any]:
    """Resolve every joint velocity owned by the authoritative robot tree.

    The full-body safety derivative must include gripper motion as well as the
    seven Panda arm joints.  The QP still changes only the seven original arm
    torques; every other robot velocity is an exact, measured exogenous term.
    Object/free-joint velocities are excluded by binding each MuJoCo DOF to
    the body that owns its joint and then selecting the resolved robot tree.
    """

    import mujoco

    robot_body_ids = {int(value) for value in resolved.robot_body_ids}
    if not robot_body_ids:
        raise OscCanaryRunnerError("resolved robot body tree is empty")
    records: List[Dict[str, Any]] = []
    for qvel_index in range(int(model.nv)):
        joint_id = int(model.dof_jntid[qvel_index])
        joint_body_id = int(model.jnt_bodyid[joint_id])
        if joint_body_id not in robot_body_ids:
            continue
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, joint_body_id
        )
        if not joint_name or not body_name:
            raise OscCanaryRunnerError(
                "robot-tree qvel authority contains an unnamed joint or body"
            )
        records.append(
            {
                "qvel_index": int(qvel_index),
                "joint_id": joint_id,
                "joint_name": str(joint_name),
                "joint_body_id": joint_body_id,
                "joint_body_name": str(body_name),
            }
        )
    qvel_indices = [int(row["qvel_index"]) for row in records]
    arm_qvel = [int(value) for value in arm_qvel_indices]
    arm_actuators = [int(value) for value in arm_actuator_ids]
    if (
        not qvel_indices
        or qvel_indices != sorted(qvel_indices)
        or len(qvel_indices) != len(set(qvel_indices))
        or len(arm_qvel) != 7
        or len(set(arm_qvel)) != 7
        or not set(arm_qvel) <= set(qvel_indices)
        or len(arm_actuators) != 7
        or len(set(arm_actuators)) != 7
    ):
        raise OscCanaryRunnerError(
            "robot-tree qvel authority does not contain the seven arm controls"
        )
    if len(qvel_indices) <= len(arm_qvel):
        raise OscCanaryRunnerError(
            "movable-manipulator shield does not include any non-arm robot velocity"
        )
    return {
        "robot_qvel_indices": qvel_indices,
        "robot_qvel_indices_sha256": _canonical_sha256(qvel_indices),
        "robot_qvel_records": records,
        "robot_qvel_records_sha256": _canonical_sha256(records),
        "velocity_dimension": len(qvel_indices),
        "arm_qvel_indices": arm_qvel,
        "arm_qvel_indices_included": True,
        "decision_arm_actuator_ids": arm_actuators,
        "decision_dimension": 7,
        "nonarm_robot_qvel_included": True,
    }


def _protected_link_arm_qvel_binding(
    *,
    arm_qvel_indices: Sequence[int],
    arm_actuator_ids: Sequence[int],
    full_robot_qvel_authority: Mapping[str, Any],
    full_robot_influence_partition: Mapping[str, Any],
    resolved_protected_link_geom_ids: Sequence[int],
) -> Dict[str, Any]:
    """Bind v4 protected-surface rows to every influencing arm qvel."""

    arm_qvel = [int(value) for value in arm_qvel_indices]
    arm_actuators = [int(value) for value in arm_actuator_ids]
    protected_geoms = {int(value) for value in resolved_protected_link_geom_ids}
    full_records = full_robot_qvel_authority.get("robot_qvel_records")
    geom_records = full_robot_influence_partition.get("geom_records")
    if not (
        len(arm_qvel) == 7
        and len(set(arm_qvel)) == 7
        and arm_qvel == sorted(arm_qvel)
        and len(arm_actuators) == 7
        and len(set(arm_actuators)) == 7
        and protected_geoms
        and isinstance(full_records, Sequence)
        and isinstance(geom_records, Sequence)
    ):
        raise OscCanaryRunnerError(
            "v4 protected-link arm-qvel binding inputs are invalid"
        )
    protected_records = [
        row
        for row in geom_records
        if isinstance(row, Mapping)
        and int(row.get("geom_id", -1)) in protected_geoms
    ]
    if {int(row["geom_id"]) for row in protected_records} != protected_geoms:
        raise OscCanaryRunnerError(
            "v4 structural influence ledger does not cover every configured "
            "protected-link geom"
        )
    influencing = sorted(
        {
            int(qvel_index)
            for row in protected_records
            for qvel_index in row.get("influencing_robot_qvel_indices", ())
        }
    )
    if not influencing or not set(influencing) <= set(arm_qvel):
        raise OscCanaryRunnerError(
            "v4 arm qvel rows omit a configured protected-link velocity influence"
        )
    records_by_index = {
        int(row["qvel_index"]): dict(row)
        for row in full_records
        if isinstance(row, Mapping)
    }
    if not set(arm_qvel) <= set(records_by_index):
        raise OscCanaryRunnerError(
            "v4 arm qvels are outside the authoritative robot-tree ledger"
        )
    records = [records_by_index[index] for index in arm_qvel]
    return {
        "schema_version": "vlsa_poisson_protected_link_arm_qvel_binding.v1",
        "robot_qvel_indices": arm_qvel,
        "robot_qvel_indices_sha256": _canonical_sha256(arm_qvel),
        "robot_qvel_records": records,
        "robot_qvel_records_sha256": _canonical_sha256(records),
        "velocity_dimension": len(arm_qvel),
        "arm_qvel_indices": arm_qvel,
        "arm_qvel_indices_included": True,
        "decision_arm_actuator_ids": arm_actuators,
        "decision_dimension": 7,
        "nonarm_robot_qvel_included": False,
        "resolved_protected_link_geom_ids": sorted(protected_geoms),
        "structurally_influencing_protected_link_qvel_indices": influencing,
        "structurally_influencing_protected_link_qvel_indices_sha256": (
            _canonical_sha256(influencing)
        ),
        "all_structural_protected_link_qvel_influences_included": True,
        "omitted_robot_tree_qvels_proven_noninfluential_for_protected_link": sorted(
            set(int(value) for value in full_robot_qvel_authority[
                "robot_qvel_indices"
            ])
            - set(arm_qvel)
        ),
    }


def _minimum_constraint_attribution(
    nominal_residuals: Any,
    samples: Sequence[Any],
    np: Any,
    *,
    sample_scope: str = "all_structurally_movable_manipulator_collision_surfaces",
) -> Dict[str, Any]:
    """Bind the first causal torque divergence to movable-manipulator samples."""

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
        "schema_version": "vlsa_poisson_constraint_attribution.v2",
        "sample_scope": str(sample_scope),
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


def _historical_v3_link_contact_verified(
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


def _remote_artifact_path(
    historical_root: Path,
    verified: Sequence[Mapping[str, Any]],
    basename: str,
) -> Path:
    """Resolve one already byte-verified remote artifact without ambiguity."""

    matches = [
        row
        for row in verified
        if Path(str(row.get("relative_path", ""))).name == basename
    ]
    if len(matches) != 1:
        raise OscCanaryRunnerError(
            "verified remote artifact set has no unique %s" % basename
        )
    target = historical_root.resolve()
    for part in Path(str(matches[0]["relative_path"])).parts:
        target = target / part
        if target.is_symlink():
            raise OscCanaryRunnerError(
                "verified remote artifact became a symlink"
            )
    if (
        not target.is_file()
        or target.is_symlink()
        or fast._file_sha256(target) != matches[0].get("sha256")
    ):
        raise OscCanaryRunnerError(
            "verified remote artifact changed after receipt validation"
        )
    return target


def _pair_body_names(row: Mapping[str, Any]) -> Tuple[str, str]:
    lineage1 = row.get("body_lineage1")
    lineage2 = row.get("body_lineage2")
    if not (
        isinstance(lineage1, Sequence)
        and not isinstance(lineage1, (str, bytes))
        and lineage1
        and isinstance(lineage2, Sequence)
        and not isinstance(lineage2, (str, bytes))
        and lineage2
    ):
        raise OscCanaryRunnerError(
            "historical detailed contact pair lacks body lineages"
        )
    return str(lineage1[0]), str(lineage2[0])


def _target_pair_body(
    row: Mapping[str, Any],
    *,
    obstacle_root_body_name: str,
    literal_link56_body_names: Sequence[str],
) -> Optional[str]:
    body1, body2 = _pair_body_names(row)
    link56 = {str(value) for value in literal_link56_body_names}
    if body1 == obstacle_root_body_name and body2 in link56:
        return body2
    if body2 == obstacle_root_body_name and body1 in link56:
        return body1
    return None


def _historical_target_link_contact_evidence(
    *,
    case: Mapping[str, Any],
    historical: Mapping[str, Any],
    detailed_contact_path: Path,
    target_link_body_names: Sequence[str],
    literal_link56_body_names: Sequence[str],
    expected_contact_action: int,
    expected_obstacle_root_body_name: str,
) -> Dict[str, Any]:
    """Prove the configured target is the first historical link contact.

    The compact result proves the episode-level contact population and sampled
    first-contact action.  The immutable detailed gzip is the timing authority
    for which literal link body is present at that exact action.
    """

    target_links = tuple(str(value) for value in target_link_body_names)
    literal_links = tuple(str(value) for value in literal_link56_body_names)
    if (
        not target_links
        or len(target_links) != len(set(target_links))
        or not set(target_links) <= set(literal_links)
    ):
        raise OscCanaryRunnerError(
            "v4 historical target links must be a nonempty unique subset of "
            "the broad link5/link6 monitor set"
        )
    target_link_set = set(target_links)
    pairs = case.get("direct_link_active_obstacle_pairs")
    telemetry = historical.get("contact_telemetry")
    unique_pairs = (
        telemetry.get("unique_contact_pairs")
        if isinstance(telemetry, Mapping)
        else None
    )
    raw_pairs = {
        (str(row.get("geom1")), str(row.get("geom2")))
        for row in unique_pairs or ()
        if isinstance(row, Mapping)
    }
    target_manifest_pairs = [
        row
        for row in pairs or ()
        if isinstance(row, Mapping)
        and row.get("active_obstacle_name") == case.get("active_obstacle_name")
        and row.get("active_obstacle_actual_body_name")
        == expected_obstacle_root_body_name
        and row.get("actual_link_body_name") in target_link_set
    ]
    compact_verified = bool(
        isinstance(pairs, Sequence)
        and target_manifest_pairs
        and {
            str(row.get("actual_link_body_name"))
            for row in target_manifest_pairs
        }
        == target_link_set
        and all(
            (
                (
                    str(row.get("active_obstacle_geom_name")),
                    str(row.get("link_geom_name")),
                )
                in raw_pairs
                or (
                    str(row.get("link_geom_name")),
                    str(row.get("active_obstacle_geom_name")),
                )
                in raw_pairs
            )
            for row in target_manifest_pairs
        )
        and case.get("first_sampled_link_contact_control_step")
        == int(expected_contact_action)
        and case.get("first_link_step_is_timing_resolved") is True
        and isinstance(telemetry, Mapping)
        and telemetry.get("first_contact_step") == int(expected_contact_action)
    )
    if not compact_verified:
        raise OscCanaryRunnerError(
            "historical compact target-link contact binding differs"
        )

    try:
        with gzip.open(detailed_contact_path, "rt", encoding="utf-8") as stream:
            detailed = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OscCanaryRunnerError(
            "historical detailed contact artifact is unreadable"
        ) from error
    snapshots = detailed.get("snapshots") if isinstance(detailed, Mapping) else None
    if not (
        isinstance(detailed, Mapping)
        and detailed.get("schema_version")
        == "vlsa_table1_active_obstacle_contacts.v3"
        and detailed.get("case_id") == case.get("case_id")
        and detailed.get("active_obstacle_name") == case.get("active_obstacle_name")
        and isinstance(snapshots, Sequence)
        and not isinstance(snapshots, (str, bytes))
    ):
        raise OscCanaryRunnerError(
            "historical detailed contact artifact authority differs"
        )

    target_steps_by_body: Dict[str, List[int]] = {
        name: [] for name in target_links
    }
    exact_snapshot: Optional[Mapping[str, Any]] = None
    exact_link56_bodies: List[str] = []
    exact_robot_pair_link56_bodies: List[str] = []
    exact_nonpositive_target_distances_m: List[float] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            raise OscCanaryRunnerError(
                "historical detailed contact snapshot is invalid"
            )
        step = snapshot.get("step")
        robot_pairs = snapshot.get("robot_pairs")
        events = snapshot.get("events")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not isinstance(robot_pairs, Sequence)
            or isinstance(robot_pairs, (str, bytes))
            or not isinstance(events, Sequence)
            or isinstance(events, (str, bytes))
        ):
            raise OscCanaryRunnerError(
                "historical detailed contact snapshot schema differs"
            )
        bodies = [
            body
            for row in robot_pairs
            if isinstance(row, Mapping)
            for body in (
                _target_pair_body(
                    row,
                    obstacle_root_body_name=expected_obstacle_root_body_name,
                    literal_link56_body_names=literal_links,
                ),
            )
            if body is not None
        ]
        nonpositive_bodies: List[str] = []
        nonpositive_distances: List[float] = []
        for event in events:
            if not isinstance(event, Mapping):
                raise OscCanaryRunnerError(
                    "historical detailed contact event is invalid"
                )
            obstacle = event.get("obstacle")
            other = event.get("other")
            distance = event.get("distance")
            if not (
                isinstance(obstacle, Mapping)
                and isinstance(other, Mapping)
                and isinstance(distance, (int, float))
                and not isinstance(distance, bool)
                and math.isfinite(float(distance))
            ):
                raise OscCanaryRunnerError(
                    "historical detailed contact event authority differs"
                )
            body_name = str(other.get("body_name"))
            if (
                obstacle.get("body_name") == expected_obstacle_root_body_name
                and other.get("classification") == "robot"
                and body_name in literal_links
                and float(distance) <= 0.0
            ):
                nonpositive_bodies.append(body_name)
                nonpositive_distances.append(float(distance))
        for target_link in target_links:
            if target_link in nonpositive_bodies:
                target_steps_by_body[target_link].append(int(step))
        if int(step) == int(expected_contact_action):
            if exact_snapshot is not None:
                raise OscCanaryRunnerError(
                    "historical detailed contact action is duplicated"
                )
            exact_snapshot = snapshot
            exact_link56_bodies = sorted(set(nonpositive_bodies))
            exact_robot_pair_link56_bodies = sorted(set(bodies))
            exact_nonpositive_target_distances_m = [
                value
                for body, value in zip(
                    nonpositive_bodies, nonpositive_distances
                )
                if body in target_link_set
            ]
    if not (
        exact_snapshot is not None
        and all(target_steps_by_body.values())
        and all(
            min(steps) == int(expected_contact_action)
            for steps in target_steps_by_body.values()
        )
        and target_link_set <= set(exact_link56_bodies)
        and target_link_set <= set(exact_robot_pair_link56_bodies)
        and exact_nonpositive_target_distances_m
        and all(value <= 0.0 for value in exact_nonpositive_target_distances_m)
    ):
        raise OscCanaryRunnerError(
            "historical detailed contact does not attribute the first action "
            "exactly to the configured target link"
        )
    return {
        "schema_version": (
            "vlsa_poisson_historical_target_link_contact_evidence.v1"
        ),
        "case_id": str(case.get("case_id")),
        "selected_obstacle_name": str(case.get("active_obstacle_name")),
        "selected_obstacle_root_body_name": expected_obstacle_root_body_name,
        "target_link_body_names": list(target_links),
        "first_target_link_contact_source_action_by_body": {
            name: int(min(target_steps_by_body[name]))
            for name in target_links
        },
        "configured_historical_contact_source_action": int(
            expected_contact_action
        ),
        "exact_action_link56_contact_body_names": exact_link56_bodies,
        "exact_action_robot_pair_link56_body_names": (
            exact_robot_pair_link56_bodies
        ),
        "exact_action_target_link_nonpositive_contact_distances_m": (
            exact_nonpositive_target_distances_m
        ),
        "manifest_target_pair_count": len(target_manifest_pairs),
        "detailed_contact_file_sha256": fast._file_sha256(
            detailed_contact_path
        ),
        "compact_result_binding_verified": True,
        "detailed_timing_and_target_attribution_verified": True,
        "verified": True,
    }


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

    joint_motion = sum(
        float(row["measured_arm_qvel_l2_rad_s"]) * 0.002 for row in rows
    )
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
        "output_qvel_indices": list(estimate.output_qvel_indices),
        "nominal_next_output_qvel_rad_s": (
            estimate.nominal_next_output_qvel_rad_s.tolist()
        ),
        "torque_to_next_output_qvel_sensitivity": (
            estimate.torque_to_next_output_qvel_sensitivity.tolist()
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
            "output_qvel_indices": list(estimate.output_qvel_indices),
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
    controller_update_index: Optional[int] = None,
    physics_substep_within_controller_update: Optional[int] = None,
    callback_endpoint_action_inner_substep: Optional[Sequence[int]] = None,
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
        if physical_boundary is None:
            executed_transition_start_boundary = None
            observed_state_boundary = (
                0 if source_phase == "settled_forwarded" else None
            )
        else:
            executed_transition_start_boundary = int(physical_boundary)
            observed_state_boundary = int(physical_boundary) + 1
        record = {
            "schema_version": (
                "vlsa_poisson_registered_forbidden_contact.v2"
                if callback_endpoint_action_inner_substep is not None
                else "vlsa_poisson_registered_forbidden_contact.v1"
            ),
            "source_phase": str(source_phase),
            "contact_index": int(contact_index),
            "physical_boundary": physical_boundary,
            "executed_transition_start_boundary": (
                executed_transition_start_boundary
            ),
            "observed_state_boundary": observed_state_boundary,
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
        if callback_endpoint_action_inner_substep is not None:
            record.update(
                {
                    "controller_update_index": controller_update_index,
                    "physics_substep_within_controller_update": (
                        physics_substep_within_controller_update
                    ),
                    "callback_endpoint_action_inner_substep": [
                        int(value)
                        for value in callback_endpoint_action_inner_substep
                    ],
                }
            )
        record["record_sha256"] = _canonical_sha256(record)
        output.append(record)
    return output


class _RegisteredContactMonitor:
    """Observe the registered union at every completed 2 ms transition."""

    def __init__(
        self,
        sim: Any,
        resolved: Any,
        *,
        physics_substeps_per_action: int = 25,
        controller_updates_per_action: Optional[int] = None,
        physics_substeps_per_controller_update: Optional[int] = None,
    ) -> None:
        from main.poisson_fullbody.measurement import clone_forwarded_state

        self._model = getattr(sim.model, "_model", sim.model)
        if (
            isinstance(physics_substeps_per_action, bool)
            or not isinstance(physics_substeps_per_action, int)
            or physics_substeps_per_action <= 0
        ):
            raise OscCanaryRunnerError(
                "registered contact-monitor cadence is invalid"
            )
        self._physics_substeps_per_action = int(
            physics_substeps_per_action
        )
        self._typed_callback_cadence = bool(
            controller_updates_per_action is not None
            or physics_substeps_per_controller_update is not None
        )
        if self._typed_callback_cadence:
            if (
                isinstance(controller_updates_per_action, bool)
                or not isinstance(controller_updates_per_action, int)
                or isinstance(physics_substeps_per_controller_update, bool)
                or not isinstance(physics_substeps_per_controller_update, int)
                or controller_updates_per_action <= 0
                or physics_substeps_per_controller_update <= 0
                or controller_updates_per_action
                * physics_substeps_per_controller_update
                != self._physics_substeps_per_action
            ):
                raise OscCanaryRunnerError(
                    "registered contact-monitor typed cadence is invalid"
                )
            self._controller_updates_per_action = int(
                controller_updates_per_action
            )
            self._physics_substeps_per_controller_update = int(
                physics_substeps_per_controller_update
            )
        else:
            self._controller_updates_per_action = None
            self._physics_substeps_per_controller_update = None
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

        expected = (
            source_action_index * self._physics_substeps_per_action
            + physics_substep_index
        )
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
        if self._typed_callback_cadence:
            inner = int(physics_substep_index) // int(
                self._physics_substeps_per_controller_update
            )
            local_substep = int(physics_substep_index) % int(
                self._physics_substeps_per_controller_update
            )
            common.update(
                {
                    "controller_update_index": inner,
                    "physics_substep_within_controller_update": local_substep,
                    "callback_endpoint_action_inner_substep": [
                        int(source_action_index),
                        inner,
                        local_substep,
                    ],
                }
            )
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
            "schema_version": (
                "vlsa_poisson_registered_contact_substep.v2"
                if self._typed_callback_cadence
                else "vlsa_poisson_registered_contact_substep.v1"
            ),
            "physical_boundary": expected,
            "executed_transition_start_boundary": expected,
            "observed_state_boundary": expected + 1,
            "source_action_index": source_action_index,
            "physics_substep_index": physics_substep_index,
            "literal_contact": bool(records),
            "contact_categories": categories,
            "contacts": records,
        }
        if self._typed_callback_cadence:
            self._last_snapshot.update(
                {
                    "controller_update_index": common[
                        "controller_update_index"
                    ],
                    "physics_substep_within_controller_update": common[
                        "physics_substep_within_controller_update"
                    ],
                    "callback_endpoint_action_inner_substep": common[
                        "callback_endpoint_action_inner_substep"
                    ],
                }
            )
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
        result = {
            "schema_version": (
                "vlsa_poisson_registered_contact_measurement.v2"
                if self._typed_callback_cadence
                else "vlsa_poisson_registered_contact_measurement.v1"
            ),
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
        if self._typed_callback_cadence:
            result["callback_cadence"] = {
                "controller_updates_per_action": (
                    self._controller_updates_per_action
                ),
                "physics_substeps_per_controller_update": (
                    self._physics_substeps_per_controller_update
                ),
                "physics_substeps_per_action": (
                    self._physics_substeps_per_action
                ),
            }
        return result


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
        default=Path("configs/vlsa_poisson_osc_arm_link_canary.v2.json"),
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
    result_schema: Optional[str] = None
    is_target_link_v4 = False
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
            partition_robot_geoms_by_qvel_influence,
            resolve_collision_geom_sets,
        )
        from main.poisson_fullbody import osc_arm_link_canary as canary_contract
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
        is_target_link_v4 = bool(
            protocol.get("schema_version")
            == canary_contract.TARGET_LINK_PROTOCOL_SCHEMA
        )
        if is_target_link_v4:
            result_schema = canary_contract.TARGET_LINK_RESULT_SCHEMA
            derived = canary_contract.validate_osc_target_link_canary_protocol(
                protocol
            )
            classifier = canary_contract.classify_osc_target_link_canary
            source_arm = getattr(
                canary_contract,
                "TARGET_LINK_SOURCE_ARM",
                canary_contract.SOURCE_ARM,
            )
        else:
            derived = canary_contract.validate_osc_arm_link_canary_protocol(
                protocol
            )
            result_schema = canary_contract.RESULT_SCHEMA
            classifier = canary_contract.classify_osc_arm_link_canary
            source_arm = canary_contract.SOURCE_ARM
        expected_case_id = (
            str(derived["case_id"])
            if is_target_link_v4
            else canary_contract.PRIMARY_CASE_ID
        )
        if arguments.case_id != expected_case_id:
            raise OscCanaryRunnerError(
                "runner case %s differs from validated protocol case %s"
                % (arguments.case_id, expected_case_id)
            )
        physics_substeps_per_action = int(
            derived.get(
                "expected_substeps_per_action",
                protocol["execution"]["physics_substeps_per_action"],
            )
        )
        if physics_substeps_per_action != int(
            protocol["execution"]["physics_substeps_per_action"]
        ):
            raise OscCanaryRunnerError(
                "validated protocol and execution physics cadences differ"
            )
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
            expected_arm=source_arm,
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
        historical_target_link_contact_evidence: Optional[Dict[str, Any]] = None
        if is_target_link_v4:
            if source_contract.get("archived_direct_target_link_contact") is not True:
                raise OscCanaryRunnerError(
                    "v4 historical target-link contact flag is disabled"
                )
            detailed_contact_path = _remote_artifact_path(
                arguments.historical_result_root.resolve(),
                verified_remote_artifacts,
                "active_obstacle_contacts.json.gz",
            )
            historical_target_link_contact_evidence = (
                _historical_target_link_contact_evidence(
                    case=case,
                    historical=historical,
                    detailed_contact_path=detailed_contact_path,
                    target_link_body_names=derived["target_link_body_names"],
                    literal_link56_body_names=protocol["case"][
                        "literal_link56_body_names"
                    ],
                    expected_contact_action=int(
                        derived["historical_contact_action"]
                    ),
                    expected_obstacle_root_body_name=protocol["case"][
                        "selected_obstacle_root_body_name"
                    ],
                )
            )
            direct_historical_contact = bool(
                historical_target_link_contact_evidence.get("verified") is True
            )
            apparatus["historical_target_link_contact_evidence"] = (
                historical_target_link_contact_evidence
            )
        else:
            direct_historical_contact = _historical_v3_link_contact_verified(
                case, historical
            )
        if not direct_historical_contact:
            raise OscCanaryRunnerError(
                "historical direct configured link contact is not verified"
            )

        runtime_path = root / protocol["runtime"]["relative_path"]
        if fast._file_sha256(runtime_path) != protocol["runtime"]["raw_file_sha256"]:
            raise OscCanaryRunnerError("runtime config hash differs")
        runtime_protocol, runtime_hashes = load_feasibility_protocol(
            runtime_path,
            expected_protocol_sha256=protocol["runtime"]["semantic_sha256"],
        )
        if runtime_hashes.parameter_block_sha256 != protocol["runtime"]["parameter_block_sha256"]:
            raise OscCanaryRunnerError("runtime parameter block differs")
        active_cadence = runtime_protocol.get("cadence", {}).get("active", {})
        controller_updates_per_action = int(
            active_cadence.get("filter_updates_per_high_level_action", -1)
        )
        physics_substeps_per_controller_update = int(
            active_cadence.get("physics_substeps_per_filter_update", -1)
        )
        if (
            controller_updates_per_action <= 0
            or physics_substeps_per_controller_update <= 0
            or controller_updates_per_action
            * physics_substeps_per_controller_update
            != physics_substeps_per_action
        ):
            raise OscCanaryRunnerError(
                "frozen runtime does not provide a phase-complete controller/"
                "physics callback cadence"
            )

        def callback_endpoint(
            source_action_index: int, flat_physics_substep_index: int
        ) -> List[int]:
            flat_index = int(flat_physics_substep_index)
            if flat_index < 0 or flat_index >= physics_substeps_per_action:
                raise OscCanaryRunnerError(
                    "flat physics callback index is outside one high-level action"
                )
            return [
                int(source_action_index),
                flat_index // physics_substeps_per_controller_update,
                flat_index % physics_substeps_per_controller_update,
            ]

        apparatus["callback_cadence"] = {
            "schema_version": "vlsa_poisson_callback_cadence.v1",
            "high_level_actions_hz": int(
                protocol["execution"]["high_level_frequency_hz"]
            ),
            "controller_updates_per_high_level_action": (
                controller_updates_per_action
            ),
            "physics_substeps_per_controller_update": (
                physics_substeps_per_controller_update
            ),
            "physics_substeps_per_high_level_action": (
                physics_substeps_per_action
            ),
            "typed_endpoint_order": [
                "source_action_index",
                "controller_update_index",
                "physics_substep_within_controller_update",
            ],
        }
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
        literal_link56_ids = tuple(
            fast._body_id(env.sim.model, name)
            for name in protocol["case"]["literal_link56_body_names"]
        )
        target_link_body_names = tuple(
            str(value)
            for value in (
                derived["target_link_body_names"]
                if is_target_link_v4
                else protocol["case"]["literal_link56_body_names"]
            )
        )
        protected_link_body_names = tuple(
            str(value)
            for value in (
                derived.get("protected_link_body_names", ())
                if is_target_link_v4
                else protocol["case"]["literal_link56_body_names"]
            )
        )
        target_link_ids = tuple(
            fast._body_id(env.sim.model, name)
            for name in target_link_body_names
        )
        protected_link_ids = tuple(
            fast._body_id(env.sim.model, name)
            for name in protected_link_body_names
        )
        if (
            not target_link_ids
            or len(target_link_ids) != len(set(target_link_ids))
            or not protected_link_ids
            or len(protected_link_ids) != len(set(protected_link_ids))
            or not set(target_link_ids) <= set(protected_link_ids)
            or not set(protected_link_ids) <= set(literal_link56_ids)
        ):
            raise OscCanaryRunnerError(
                "configured target and protected bodies do not form nonempty "
                "target <= protected <= broad-monitor body sets"
            )
        raw_model, raw_data = fast._raw_model_data(env.sim)
        resolved = resolve_collision_geom_sets(
            raw_model,
            robot_root_body_ids=(robot_root,),
            obstacle_root_body_ids=(int(authority["active_obstacle_root_body_id"]),),
            link56_body_ids=literal_link56_ids,
        )
        target_resolved = resolve_collision_geom_sets(
            raw_model,
            robot_root_body_ids=(robot_root,),
            obstacle_root_body_ids=(int(authority["active_obstacle_root_body_id"]),),
            link56_body_ids=target_link_ids,
        )
        protected_resolved = resolve_collision_geom_sets(
            raw_model,
            robot_root_body_ids=(robot_root,),
            obstacle_root_body_ids=(int(authority["active_obstacle_root_body_id"]),),
            link56_body_ids=protected_link_ids,
        )
        resolution_shared_fields = (
            "robot_root_body_ids",
            "robot_body_ids",
            "obstacle_root_body_ids",
            "obstacle_body_ids",
            "robot_geom_ids",
            "obstacle_geom_ids",
            "collision_enabled_pairs",
            "robot_body_names",
            "obstacle_body_names",
            "robot_geom_names",
            "obstacle_geom_names",
        )
        if any(
            getattr(candidate, field) != getattr(resolved, field)
            for candidate in (target_resolved, protected_resolved)
            for field in resolution_shared_fields
        ):
            raise OscCanaryRunnerError(
                "target/protected resolution changed broad robot/obstacle authority"
            )
        target_link_geom_ids = tuple(
            int(value) for value in target_resolved.link56_geom_ids
        )
        target_link_geom_names = tuple(
            str(value) for value in target_resolved.link56_geom_names
        )
        protected_link_geom_ids = tuple(
            int(value) for value in protected_resolved.link56_geom_ids
        )
        protected_link_geom_names = tuple(
            str(value) for value in protected_resolved.link56_geom_names
        )
        if (
            target_resolved.link56_body_ids != target_link_ids
            or not target_link_geom_ids
            or protected_resolved.link56_body_ids != protected_link_ids
            or not protected_link_geom_ids
            or not set(target_link_geom_ids) <= set(protected_link_geom_ids)
            or not set(protected_link_geom_ids) <= set(resolved.link56_geom_ids)
            or any(
                int(raw_model.geom_bodyid[geom_id]) not in set(target_link_ids)
                for geom_id in target_link_geom_ids
            )
            or any(
                int(raw_model.geom_bodyid[geom_id]) not in set(protected_link_ids)
                for geom_id in protected_link_geom_ids
            )
        ):
            raise OscCanaryRunnerError(
                "target/protected resolution is not exact direct-attached "
                "collision geometry of its configured bodies"
            )
        target_resolved_geometry = {
            "schema_version": "vlsa_poisson_target_link_resolved_geometry.v1",
            "target_link_body_ids": list(target_link_ids),
            "target_link_body_names": list(target_link_body_names),
            "target_link_geom_ids": list(target_link_geom_ids),
            "target_link_geom_names": list(target_link_geom_names),
            "geometry_selection": (
                "collision_enabled_geoms_directly_attached_to_exact_configured_"
                "target_link_bodies_no_descendants"
            ),
            "broad_robot_obstacle_authority_unchanged": True,
            "target_link_geoms_subset_of_broad_literal_link56_geoms": True,
        }
        protected_resolved_geometry = {
            "schema_version": (
                "vlsa_poisson_protected_link_resolved_geometry.v1"
            ),
            "protected_link_body_ids": list(protected_link_ids),
            "protected_link_body_names": list(protected_link_body_names),
            "protected_link_geom_ids": list(protected_link_geom_ids),
            "protected_link_geom_names": list(protected_link_geom_names),
            "geometry_selection": (
                "collision_enabled_geoms_directly_attached_to_exact_configured_"
                "protected_link_bodies_no_descendants"
            ),
            "broad_robot_obstacle_authority_unchanged": True,
            "target_link_geoms_subset_of_protected_link_geoms": True,
            "protected_link_geoms_subset_of_broad_literal_link56_geoms": True,
        }
        apparatus.update(
            {
                "controller": controller,
                "arm_actuator_torque_units": torque_units,
                "resolved_geometry": resolved.to_dict(),
                "target_link_resolved_geometry": target_resolved_geometry,
                "target_link_resolved_geometry_sha256": _canonical_sha256(
                    target_resolved_geometry
                ),
                "protected_link_resolved_geometry": protected_resolved_geometry,
                "protected_link_resolved_geometry_sha256": _canonical_sha256(
                    protected_resolved_geometry
                ),
            }
        )
        arm_actuators = tuple(controller["arm_actuator_indexes"])
        arm_qpos = tuple(controller["arm_qpos_indexes"])
        arm_qvel = tuple(controller["arm_qvel_indexes"])
        model_actuator_count = int(raw_model.nu)
        if (
            model_actuator_count <= len(arm_actuators)
            or len(set(arm_actuators)) != len(arm_actuators)
            or any(
                int(actuator_id) < 0
                or int(actuator_id) >= model_actuator_count
                for actuator_id in arm_actuators
            )
        ):
            raise OscCanaryRunnerError(
                "compiled model does not expose a valid non-arm control complement"
            )
        arm_actuator_set = {int(value) for value in arm_actuators}
        nonarm_ctrl_indices = tuple(
            index
            for index in range(model_actuator_count)
            if index not in arm_actuator_set
        )
        if not nonarm_ctrl_indices:
            raise OscCanaryRunnerError("compiled model has no non-arm controls")
        full_robot_qvel_authority = _robot_tree_qvel_binding(
            raw_model,
            resolved,
            arm_qvel_indices=arm_qvel,
            arm_actuator_ids=arm_actuators,
        )
        full_robot_qvel = tuple(
            full_robot_qvel_authority["robot_qvel_indices"]
        )
        full_robot_geom_influence_partition = (
            partition_robot_geoms_by_qvel_influence(
                raw_model,
                resolved,
                full_robot_qvel,
            )
        )
        if is_target_link_v4:
            robot_qvel_authority = _protected_link_arm_qvel_binding(
                arm_qvel_indices=arm_qvel,
                arm_actuator_ids=arm_actuators,
                full_robot_qvel_authority=full_robot_qvel_authority,
                full_robot_influence_partition=(
                    full_robot_geom_influence_partition
                ),
                resolved_protected_link_geom_ids=protected_link_geom_ids,
            )
            robot_qvel = tuple(robot_qvel_authority["robot_qvel_indices"])
            robot_geom_influence_partition = (
                partition_robot_geoms_by_qvel_influence(
                    raw_model,
                    resolved,
                    robot_qvel,
                )
            )
            active_protected_link_influences = {
                int(row["geom_id"]): tuple(
                    int(value)
                    for value in row["influencing_robot_qvel_indices"]
                )
                for row in robot_geom_influence_partition["geom_records"]
                if int(row["geom_id"]) in set(protected_link_geom_ids)
            }
            full_protected_link_influences = {
                int(row["geom_id"]): tuple(
                    int(value)
                    for value in row["influencing_robot_qvel_indices"]
                )
                for row in full_robot_geom_influence_partition["geom_records"]
                if int(row["geom_id"]) in set(protected_link_geom_ids)
            }
            if active_protected_link_influences != full_protected_link_influences:
                raise OscCanaryRunnerError(
                    "v4 arm-qvel Jacobian columns omit a configured protected-link "
                    "velocity influence"
                )
        else:
            robot_qvel_authority = full_robot_qvel_authority
            robot_qvel = full_robot_qvel
            robot_geom_influence_partition = (
                full_robot_geom_influence_partition
            )
        robot_geom_influence_partition_sha256 = _canonical_sha256(
            robot_geom_influence_partition
        )
        apparatus["full_robot_qvel_authority"] = full_robot_qvel_authority
        apparatus["active_shield_qvel_authority"] = robot_qvel_authority
        apparatus["full_robot_geom_influence_partition"] = (
            full_robot_geom_influence_partition
        )
        apparatus["robot_geom_influence_partition"] = (
            robot_geom_influence_partition
        )
        apparatus["robot_geom_influence_partition_sha256"] = (
            robot_geom_influence_partition_sha256
        )
        settled_official = fast._official_state(env.sim)
        field_bundle_kwargs: Dict[str, Any] = {
            "resolved": protected_resolved if is_target_link_v4 else resolved,
            "protocol": runtime_protocol,
            "protocol_hashes": runtime_hashes,
        }
        if is_target_link_v4:
            field_bundle_kwargs["protected_body_names"] = (
                protected_link_body_names
            )
        bundle = build_static_field_bundle(
            raw_model,
            raw_data,
            **field_bundle_kwargs,
        )
        apparatus["field_bundle_hashes"] = asdict(bundle.hashes)
        apparatus["field_diagnostics"] = asdict(bundle.diagnostics)
        if not np.array_equal(settled_official, fast._official_state(env.sim)):
            raise OscCanaryRunnerError("field construction changed simulator state")
        forwarded = clone_forwarded_state(raw_model, raw_data)
        epsilon_m = float(runtime_protocol["coverage"]["epsilon_m"])
        structurally_movable_geom_ids = tuple(
            int(value)
            for value in robot_geom_influence_partition[
                "shield_manipulator_geom_ids"
            ]
        )
        link56_geom_ids = tuple(int(value) for value in resolved.link56_geom_ids)
        movable_geom_ids = (
            protected_link_geom_ids
            if is_target_link_v4
            else structurally_movable_geom_ids
        )
        fixed_geom_ids = tuple(
            int(value)
            for value in robot_geom_influence_partition[
                "fixed_robot_infrastructure_geom_ids"
            ]
        )
        all_robot_samples = build_robot_collision_samples(
            env.sim.model,
            forwarded,
            geom_ids=resolved.robot_geom_ids,
            epsilon_m=epsilon_m,
        )
        shield_samples = build_robot_collision_samples(
            env.sim.model,
            forwarded,
            geom_ids=movable_geom_ids,
            epsilon_m=epsilon_m,
        )
        fixed_samples = build_robot_collision_samples(
            env.sim.model,
            forwarded,
            geom_ids=fixed_geom_ids,
            epsilon_m=epsilon_m,
        )

        def surface_evidence(sample_set: Any) -> Dict[str, Any]:
            return {
                "sample_count": len(sample_set.samples),
                "sample_ledger_sha256": sample_set.sample_ledger_sha256,
                "samples": [sample.to_dict() for sample in sample_set.samples],
                "geom_records": list(sample_set.geom_records),
                "epsilon_m": sample_set.epsilon_m,
                "maximum_surface_cover_radius_m": (
                    sample_set.maximum_surface_cover_radius_m
                ),
                "coverage_semantics": sample_set.coverage_semantics,
                "roundtrip": validate_rigid_roundtrip(
                    sample_set.samples, forwarded
                ),
            }

        all_robot_sample_evidence = surface_evidence(all_robot_samples)
        movable_sample_evidence = surface_evidence(shield_samples)
        fixed_sample_evidence = surface_evidence(fixed_samples)
        shield_sample_payload = [
            sample.to_dict() for sample in shield_samples.samples
        ]
        field_protected_sample_payload = [
            sample.to_dict() for sample in bundle.protected_samples.samples
        ]
        field_protected_geom_ids = [
            int(component.geom_id)
            for component in bundle.protected_samples.components
        ]
        protected_link_shield_ledger_exact = bool(
            is_target_link_v4
            and movable_geom_ids == protected_link_geom_ids
            and field_protected_geom_ids == list(protected_link_geom_ids)
            and shield_sample_payload == field_protected_sample_payload
            and _canonical_sha256(shield_sample_payload)
            == shield_samples.sample_ledger_sha256
        )
        if is_target_link_v4 and not protected_link_shield_ledger_exact:
            raise OscCanaryRunnerError(
                "v4 shield rows differ from the exact ordered field-bundle "
                "configured protected-link sample ledger"
            )
        geom_name_by_id = dict(
            zip(resolved.robot_geom_ids, resolved.robot_geom_names)
        )
        for label, evidence, sample_set, geom_ids in (
            (
                "all-robot",
                all_robot_sample_evidence,
                all_robot_samples,
                tuple(int(value) for value in resolved.robot_geom_ids),
            ),
            (
                "movable-manipulator",
                movable_sample_evidence,
                shield_samples,
                movable_geom_ids,
            ),
            (
                "fixed-infrastructure",
                fixed_sample_evidence,
                fixed_samples,
                fixed_geom_ids,
            ),
        ):
            validate_robot_sample_evidence(
                evidence,
                resolved_geom_ids=geom_ids,
                resolved_geom_names=tuple(
                    str(geom_name_by_id[geom_id]) for geom_id in geom_ids
                ),
                resolved_body_ids=resolved.robot_body_ids,
                roundtrip_field="roundtrip",
            )
            if _canonical_sha256(evidence["samples"]) != str(
                sample_set.sample_ledger_sha256
            ):
                raise OscCanaryRunnerError(
                    "serialized %s sample ledger differs from its hash" % label
                )
        apparatus.update(
            {
                "all_robot_sampling": all_robot_sample_evidence,
                "all_robot_sampling_sha256": _canonical_sha256(
                    all_robot_sample_evidence
                ),
                "movable_manipulator_sampling": movable_sample_evidence,
                "movable_manipulator_sampling_sha256": _canonical_sha256(
                    movable_sample_evidence
                ),
                "protected_link_shield_sampling": (
                    movable_sample_evidence if is_target_link_v4 else None
                ),
                "protected_link_shield_sampling_sha256": (
                    _canonical_sha256(movable_sample_evidence)
                    if is_target_link_v4
                    else None
                ),
                "fixed_infrastructure_sampling": fixed_sample_evidence,
                "fixed_infrastructure_sampling_sha256": _canonical_sha256(
                    fixed_sample_evidence
                ),
            }
        )

        fixed_world_points, fixed_jacobians = evaluate_point_jacobians(
            raw_model,
            forwarded,
            fixed_samples.samples,
            robot_qvel,
        )
        fixed_jacobians = np.asarray(fixed_jacobians, dtype=np.float64)
        expected_fixed_jacobian_shape = (
            len(fixed_samples.samples),
            3,
            len(robot_qvel),
        )
        fixed_jacobians_exactly_zero = bool(
            fixed_jacobians.shape == expected_fixed_jacobian_shape
            and np.all(np.isfinite(fixed_jacobians))
            and np.all(fixed_jacobians == 0.0)
        )
        maximum_abs_fixed_jacobian = (
            None
            if fixed_jacobians.shape != expected_fixed_jacobian_shape
            or not np.all(np.isfinite(fixed_jacobians))
            else float(np.max(np.abs(fixed_jacobians)))
        )
        fixed_jacobian_certificate_array = (
            np.zeros(expected_fixed_jacobian_shape, dtype=np.float64)
            if fixed_jacobians_exactly_zero
            else fixed_jacobians
        )
        registered_contact_monitor = _RegisteredContactMonitor(
            env.sim,
            resolved,
            physics_substeps_per_action=physics_substeps_per_action,
            controller_updates_per_action=(
                controller_updates_per_action if is_target_link_v4 else None
            ),
            physics_substeps_per_controller_update=(
                physics_substeps_per_controller_update
                if is_target_link_v4
                else None
            ),
        )
        settled_registered_contact = registered_contact_monitor.result()
        fixed_geom_id_set = set(fixed_geom_ids)
        fixed_settled_contact_records = [
            row
            for row in settled_registered_contact["settled_contact_records"]
            if int(row["robot_geom_id"]) in fixed_geom_id_set
        ]
        fixed_sample_ids = [
            int(sample.sample_id) for sample in fixed_samples.samples
        ]
        all_fixed_geoms_contact_monitored = bool(
            fixed_geom_id_set
            <= {
                int(value)
                for value in registered_contact_monitor.scope["robot_geom_ids"]
            }
        )
        fixed_infrastructure_settled_certificate = {
            "schema_version": (
                "vlsa_poisson_fixed_infrastructure_settled_certificate.v1"
            ),
            "fixed_geom_ids": list(fixed_geom_ids),
            "fixed_geom_ids_sha256": _canonical_sha256(list(fixed_geom_ids)),
            "fixed_sample_ids": fixed_sample_ids,
            "fixed_sample_ids_sha256": _canonical_sha256(fixed_sample_ids),
            "fixed_sample_count": len(fixed_samples.samples),
            "fixed_sample_ledger_sha256": fixed_samples.sample_ledger_sha256,
            "fixed_world_points_array_record": _compact_float64_array_record(
                fixed_world_points, np
            ),
            "fixed_point_jacobian_array_record": _compact_float64_array_record(
                fixed_jacobian_certificate_array, np
            ),
            "structural_partition_sha256": (
                robot_geom_influence_partition_sha256
            ),
            "all_fixed_geoms_zero_structural_qvel_influence": bool(
                robot_geom_influence_partition[
                    "all_fixed_geoms_have_zero_structural_qvel_influence"
                ]
            ),
            "all_fixed_sample_jacobians_exactly_zero": (
                fixed_jacobians_exactly_zero
            ),
            "maximum_abs_fixed_sample_jacobian": (
                maximum_abs_fixed_jacobian
            ),
            "settled_registered_contact_count": len(
                fixed_settled_contact_records
            ),
            "settled_contact_records": fixed_settled_contact_records,
            "settled_contact_free": not fixed_settled_contact_records,
            "all_fixed_geoms_contact_monitored": (
                all_fixed_geoms_contact_monitored
            ),
            "contact_monitor_scope_identity_sha256": (
                registered_contact_monitor.scope["identity_sha256"]
            ),
        }
        apparatus["registered_contact_scope"] = registered_contact_monitor.scope
        apparatus["fixed_infrastructure_settled_certificate"] = (
            fixed_infrastructure_settled_certificate
        )
        apparatus["fixed_infrastructure_settled_certificate_sha256"] = (
            _canonical_sha256(fixed_infrastructure_settled_certificate)
        )
        if not (
            fixed_infrastructure_settled_certificate[
                "all_fixed_geoms_zero_structural_qvel_influence"
            ]
            and fixed_infrastructure_settled_certificate[
                "all_fixed_sample_jacobians_exactly_zero"
            ]
            and fixed_infrastructure_settled_certificate[
                "settled_contact_free"
            ]
            and fixed_infrastructure_settled_certificate[
                "all_fixed_geoms_contact_monitored"
            ]
        ):
            raise OscCanaryRunnerError(
                "fixed robot infrastructure lacks a zero-authority, "
                "settled-contact-free certificate"
            )
        if registered_contact_monitor.settled_contact:
            raise OscCanaryRunnerError(
                "treatment begins in a registered forbidden contact"
            )
        settled_shield_points, settled_shield_jacobians = (
            evaluate_point_jacobians(
                raw_model,
                forwarded,
                shield_samples.samples,
                robot_qvel,
            )
        )
        settled_shield_queries = [
            bundle.field.query(point) for point in settled_shield_points
        ]
        settled_field_query_diagnostics = _settled_field_query_diagnostics(
            samples=shield_samples.samples,
            world_points=settled_shield_points,
            queries=settled_shield_queries,
            field=bundle.field,
            sample_ledger_sha256=shield_samples.sample_ledger_sha256,
            np=np,
        )
        apparatus["settled_field_query_diagnostics"] = (
            settled_field_query_diagnostics
        )
        if settled_field_query_diagnostics["invalid_query_count"]:
            raise OscCanaryRunnerError(
                "movable-manipulator shield has %d invalid settled field "
                "queries: %r"
                % (
                    settled_field_query_diagnostics["invalid_query_count"],
                    settled_field_query_diagnostics["reason_counts"],
                )
            )
        settled_shield_h = np.asarray(
            [float(query.value) for query in settled_shield_queries],
            dtype=np.float64,
        )
        if (
            settled_shield_h.shape != (len(shield_samples.samples),)
            or not np.all(np.isfinite(settled_shield_h))
            or np.any(settled_shield_h <= 0.0)
        ):
            raise OscCanaryRunnerError(
                "movable-manipulator shield does not begin with strictly positive h"
            )
        settled_shield_points = np.asarray(
            settled_shield_points, dtype=np.float64
        )
        settled_outer_clearance = np.asarray(
            [
                float(query.outer_boundary_clearance_m)
                for query in settled_shield_queries
            ],
            dtype=np.float64,
        )
        workspace_lower = np.asarray(
            runtime_protocol["workspace"]["minimum_m"], dtype=np.float64
        )
        workspace_upper = np.asarray(
            runtime_protocol["workspace"]["maximum_m"], dtype=np.float64
        )
        direct_outer_clearance = np.min(
            np.concatenate(
                (
                    settled_shield_points - workspace_lower[None, :],
                    workspace_upper[None, :] - settled_shield_points,
                ),
                axis=1,
            ),
            axis=1,
        )
        required_outer_clearance = float(
            runtime_protocol["occupancy"]["outer_boundary_clearance_m"]
        )
        workspace_scale = max(
            1.0,
            float(np.max(np.abs(workspace_lower))),
            float(np.max(np.abs(workspace_upper))),
        )
        outer_clearance_tolerance = (
            64.0 * float(np.finfo(np.float64).eps) * workspace_scale
        )
        all_outer_clearances_pass = bool(
            np.all(
                settled_outer_clearance + outer_clearance_tolerance
                >= required_outer_clearance
            )
        )
        if (
            settled_shield_points.shape != (len(shield_samples.samples), 3)
            or settled_outer_clearance.shape != (len(shield_samples.samples),)
            or not np.all(np.isfinite(settled_shield_points))
            or not np.all(np.isfinite(settled_outer_clearance))
            or not np.allclose(
                settled_outer_clearance,
                direct_outer_clearance,
                rtol=0.0,
                atol=1.0e-12,
            )
            or not all_outer_clearances_pass
        ):
            raise OscCanaryRunnerError(
                "movable-manipulator settled samples violate expanded-grid outer clearance"
            )
        settled_zero_jacobian = np.all(
            settled_shield_jacobians == 0.0, axis=(1, 2)
        )
        settled_field_query_certificate = {
            "schema_version": (
                "vlsa_poisson_movable_manipulator_settled_field_query.v1"
            ),
            "sample_count": len(shield_samples.samples),
            "sample_ledger_sha256": shield_samples.sample_ledger_sha256,
            "all_queries_valid": True,
            "all_h_strictly_positive": True,
            "minimum_h_m2": float(np.min(settled_shield_h)),
            "h_m2": settled_shield_h.tolist(),
            "h_array_record": _compact_float64_array_record(
                settled_shield_h, np
            ),
            "world_points_m": settled_shield_points.tolist(),
            "world_points_array_record": _compact_float64_array_record(
                settled_shield_points, np
            ),
            "outer_boundary_clearance_m": settled_outer_clearance.tolist(),
            "outer_boundary_clearance_array_record": (
                _compact_float64_array_record(settled_outer_clearance, np)
            ),
            "required_outer_boundary_clearance_m": (
                required_outer_clearance
            ),
            "minimum_outer_boundary_clearance_m": float(
                np.min(settled_outer_clearance)
            ),
            "outer_boundary_clearance_comparison_tolerance_m": (
                outer_clearance_tolerance
            ),
            "all_outer_boundary_clearances_pass": all_outer_clearances_pass,
        }
        resolved_robot_geom_ids = [
            int(value) for value in resolved.robot_geom_ids
        ]
        shield_sampling_binding = {
            "schema_version": (
                "vlsa_poisson_protected_link_shield_sampling_binding.v1"
                if is_target_link_v4
                else (
                    "vlsa_poisson_movable_manipulator_shield_sampling_binding.v1"
                )
            ),
            "scope": (
                str(protocol["shield"]["binding_scope"])
                if is_target_link_v4
                else (
                    "all_structurally_movable_manipulator_collision_surfaces_"
                    "vs_selected_obstacle"
                )
            ),
            "sample_source": (
                "apparatus.protected_sampling.samples"
                if is_target_link_v4
                else "apparatus.movable_manipulator_sampling.samples"
            ),
            "protected_samples_contract": (
                str(protocol["shield"]["protected_samples"])
                if is_target_link_v4
                else None
            ),
            "field_bundle_samples_contract": (
                str(protocol["shield"]["field_bundle_samples"])
                if is_target_link_v4
                else None
            ),
            "shield_body_authority": (
                str(protocol["shield"]["shield_body_authority"])
                if is_target_link_v4
                else None
            ),
            "shield_geometry_selection": (
                str(protocol["shield"]["shield_geometry_selection"])
                if is_target_link_v4
                else None
            ),
            "sample_count": len(shield_samples.samples),
            "sample_ledger_sha256": shield_samples.sample_ledger_sha256,
            "resolved_robot_geom_ids": resolved_robot_geom_ids,
            "resolved_robot_geom_ids_sha256": _canonical_sha256(
                resolved_robot_geom_ids
            ),
            "shield_manipulator_geom_ids": list(movable_geom_ids),
            "shield_manipulator_geom_ids_sha256": _canonical_sha256(
                list(movable_geom_ids)
            ),
            "protected_link_body_ids": (
                list(protected_link_ids) if is_target_link_v4 else None
            ),
            "protected_link_body_names": (
                list(protected_link_body_names) if is_target_link_v4 else None
            ),
            "shield_protected_link_geom_ids": (
                list(protected_link_geom_ids) if is_target_link_v4 else None
            ),
            "shield_protected_link_geom_ids_sha256": (
                _canonical_sha256(list(protected_link_geom_ids))
                if is_target_link_v4
                else None
            ),
            "evaluation_target_link_body_names": (
                list(target_link_body_names) if is_target_link_v4 else None
            ),
            "evaluation_target_link_geom_ids": (
                list(target_link_geom_ids) if is_target_link_v4 else None
            ),
            "resolved_link56_geom_ids": list(link56_geom_ids),
            "resolved_link56_geom_ids_sha256": _canonical_sha256(
                list(link56_geom_ids)
            ),
            "field_bundle_protected_sample_count": len(
                bundle.protected_samples.samples
            ),
            "field_bundle_protected_body_ids": list(
                bundle.protected_body_ids
            ),
            "field_bundle_protected_body_names": list(
                bundle.protected_body_names
            ),
            "field_bundle_protected_samples_sha256": (
                bundle.hashes.protected_samples_sha256
            ),
            "exact_ordered_field_bundle_protected_sample_ledger": bool(
                protected_link_shield_ledger_exact
            ),
            "fixed_robot_infrastructure_geom_ids": list(fixed_geom_ids),
            "fixed_robot_infrastructure_geom_ids_sha256": _canonical_sha256(
                list(fixed_geom_ids)
            ),
            "robot_geom_influence_partition_sha256": (
                robot_geom_influence_partition_sha256
            ),
            "all_robot_contact_monitor_scope_identity_sha256": (
                registered_contact_monitor.scope["identity_sha256"]
            ),
            "robot_qvel_selection_rule": (
                str(protocol["shield"]["point_velocity_scope"])
                if is_target_link_v4
                else (
                    "ascending_dof_index_whose_dof_joint_body_is_in_"
                    "resolved_robot_body_ids"
                )
            ),
            **(
                {}
                if is_target_link_v4
                else robot_qvel_authority
            ),
            "protected_link_qvel_authority": (
                dict(robot_qvel_authority) if is_target_link_v4 else None
            ),
            "nonarm_ctrl_policy": (
                "unchanged_byte_exact_in_all_cloned_and_live_transitions"
            ),
            "model_actuator_count": model_actuator_count,
            "nonarm_ctrl_selection_rule": (
                "ascending_actuator_index_excluding_decision_arm_actuator_ids"
            ),
            "nonarm_ctrl_indices": list(nonarm_ctrl_indices),
            "nonarm_ctrl_indices_sha256": _canonical_sha256(
                list(nonarm_ctrl_indices)
            ),
            "nonarm_ctrl_count": len(nonarm_ctrl_indices),
            "live_nonarm_ctrl_array_hash_format": _FLOAT64_ARRAY_HASH_FORMAT,
            "field_seed_sample_scope": (
                str(protocol["shield"]["field_bundle_samples"])
                if is_target_link_v4
                else "link56_only_not_shield_scope"
            ),
            "settled_field_query_certificate": (
                settled_field_query_certificate
            ),
            "settled_zero_jacobian_sample_count": int(
                np.count_nonzero(settled_zero_jacobian)
            ),
            "settled_nonzero_jacobian_sample_count": int(
                settled_zero_jacobian.size
                - np.count_nonzero(settled_zero_jacobian)
            ),
        }
        apparatus["shield_sampling_binding"] = shield_sampling_binding
        apparatus["shield_sampling_binding_sha256"] = _canonical_sha256(
            shield_sampling_binding
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
        apparatus.update(
            {
                "field_obstacle_boxes": field_obstacle_boxes,
                "protected_sampling": protected_sample_payload,
                "serialized_field_certificate_sha256": _canonical_sha256(
                    serialized_field_certificate
                ),
                "field_frame": protocol["shield"]["field_frame"],
            }
        )
        limits = runtime_protocol["admissibility"]
        monitor = FullRobotObstacleMonitor(
            env.sim,
            resolved,
            all_robot_samples.samples,
            certified_coverage_radius_m=(
                all_robot_samples.maximum_surface_cover_radius_m
            ),
            max_selected_geom_surface_drift_m=float(limits["max_selected_geom_surface_drift_m"]),
            max_selected_geom_translation_drift_m=float(limits["max_selected_geom_translation_drift_m"]),
            max_selected_geom_rotation_drift_rad=float(limits["max_selected_geom_rotation_drift_rad"]),
            max_settled_obstacle_linear_speed_m_per_s=float(limits["max_selected_body_linear_speed_m_s"]),
            max_settled_obstacle_angular_speed_rad_per_s=float(limits["max_selected_body_angular_speed_rad_s"]),
            require_settled_static_motion=False,
            terminate_on_static_drift=False,
            near_contact_tolerance_m=float(runtime_protocol["safety"]["contact_margin_m"]),
            inner_updates_per_high_level_action=controller_updates_per_action,
            physics_substeps_per_inner_update=(
                physics_substeps_per_controller_update
            ),
        )
        if monitor.settled_state.any_robot_obstacle_contact:
            raise OscCanaryRunnerError("treatment begins in robot-selected-obstacle contact")
        registered_contact_hook = _registered_contact_hook(
            registered_contact_monitor.scope
        )
        settled_obstacle = fast._obstacle_state(
            raw_model, forwarded, bundle, resolved.obstacle_body_ids
        )
        if not fast._static_admissible([settled_obstacle], runtime_protocol):
            raise OscCanaryRunnerError("selected obstacle is not static after settling")

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
                boundary = (
                    source_index * physics_substeps_per_action
                    + int(substep_index)
                )

                def method_stop(reason: str, **details: Any) -> None:
                    nonlocal method_stop_record
                    method_stop_record = {
                        "schema_version": "vlsa_poisson_safety_method_stop.v2",
                        "source_action_index": source_index,
                        "physics_substep_index": int(substep_index),
                        "callback_endpoint_action_inner_substep": (
                            callback_endpoint(source_index, int(substep_index))
                        ),
                        "physics_boundary_before_unexecuted_step": boundary,
                        "unexecuted_post_integration_boundary": boundary + 1,
                        "sample_count": len(shield_samples.samples),
                        "sample_ledger_sha256": (
                            shield_samples.sample_ledger_sha256
                        ),
                        "robot_qvel_indices": list(robot_qvel),
                        "robot_qvel_indices_sha256": (
                            robot_qvel_authority[
                                "robot_qvel_indices_sha256"
                            ]
                        ),
                        "velocity_dimension": len(robot_qvel),
                        "reason": str(reason),
                        **details,
                    }
                    method_stop_record["record_payload_sha256"] = (
                        _canonical_sha256(method_stop_record)
                    )
                    raise TreatmentMethodStop(str(reason))

                current = clone_forwarded_state(raw_model, raw_data)
                points, jacobians = evaluate_point_jacobians(
                    raw_model,
                    current,
                    shield_samples.samples,
                    robot_qvel,
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
                    rows
                    @ nominal_transition.next_qvel[
                        np.asarray(robot_qvel, dtype=np.int64)
                    ]
                    + float(shield_cfg["alpha_gain_per_s"]) * h
                    - float(shield_cfg["margin_m2_per_s"])
                )
                constraint_attribution = _minimum_constraint_attribution(
                    nominal_residuals,
                    shield_samples.samples,
                    np,
                    sample_scope=(
                        str(protocol["shield"]["binding_scope"])
                        if is_target_link_v4
                        else (
                            "all_structurally_movable_manipulator_collision_surfaces"
                        )
                    ),
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
                            output_qvel_indices=robot_qvel,
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
                        raw_data.qvel[list(robot_qvel)], dtype=np.float64
                    )
                    try:
                        result = shield.solve(
                            nominal_torque=nominal_array,
                            current_qvel=current_qvel,
                            nominal_next_qvel=(
                                estimate.nominal_next_output_qvel_rad_s
                            ),
                            h=h,
                            joint_gradient_rows=rows,
                            dt_seconds=0.002,
                            torque_to_next_qvel_sensitivity=(
                                estimate.torque_to_next_output_qvel_sensitivity
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
                    rows
                    @ candidate_transition.next_qvel[
                        np.asarray(robot_qvel, dtype=np.int64)
                    ]
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
                is_first_byte_divergence = bool(
                    not command_byte_identical_to_nominal
                    and first_torque_divergence_boundary is None
                )
                if is_first_byte_divergence:
                    first_torque_divergence_boundary = boundary
                    first_torque_divergence_action = source_index
                    first_torque_divergence_correction_l2_nm = correction
                    planner.mark_divergence(
                        action_index=source_index,
                        physical_boundary=boundary,
                    )
                is_first_material_correction = bool(
                    first_material_boundary is None
                    and correction
                    >= float(protocol["acceptance"]["material_torque_correction_l2_nm"])
                )
                if is_first_material_correction:
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
                full_constraint_qp_certificate_roles: List[str] = []
                if is_first_byte_divergence:
                    full_constraint_qp_certificate_roles.append(
                        "first_byte_divergence"
                    )
                if is_target_link_v4 and is_first_material_correction:
                    full_constraint_qp_certificate_roles.append(
                        "first_material_correction"
                    )
                retain_full_constraint_qp_certificate = bool(
                    full_constraint_qp_certificate_roles
                )
                predicted_shield_qvel = np.asarray(
                    candidate_transition.next_qvel[
                        np.asarray(robot_qvel, dtype=np.int64)
                    ],
                    dtype=np.float64,
                )
                predicted_arm_qvel = np.asarray(
                    candidate_transition.next_arm_qvel, dtype=np.float64
                )
                live_nonarm_ctrl_before = np.asarray(
                    raw_data.ctrl[list(nonarm_ctrl_indices)], dtype=np.float64
                ).copy()
                if (
                    live_nonarm_ctrl_before.shape
                    != (len(nonarm_ctrl_indices),)
                    or not np.all(np.isfinite(live_nonarm_ctrl_before))
                ):
                    raise OscCanaryRunnerError(
                        "live pre-transition non-arm controls are invalid"
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
                    "predicted_next_shield_qvel": predicted_shield_qvel,
                    "nominal_predicted_next_shield_qvel": np.asarray(
                        nominal_transition.next_qvel[
                            np.asarray(robot_qvel, dtype=np.int64)
                        ],
                        dtype=np.float64,
                    ).copy(),
                    "predicted_next_arm_qvel": predicted_arm_qvel,
                    "nominal_predicted_next_arm_qvel": np.asarray(
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
                    "nominal_non_arm_ctrl_preserved": bool(
                        nominal_transition.non_arm_ctrl_preserved
                    ),
                    "candidate_non_arm_ctrl_preserved": bool(
                        candidate_transition.non_arm_ctrl_preserved
                    ),
                    "live_nonarm_ctrl_before": live_nonarm_ctrl_before,
                    "retain_full_constraint_qp_certificate": (
                        retain_full_constraint_qp_certificate
                    ),
                    "full_constraint_qp_certificate_roles": (
                        full_constraint_qp_certificate_roles
                    ),
                }
                return command

            def poststep(sim: Any, substep_index: int) -> None:
                nonlocal actual_residual_pass
                row = pending.get(int(substep_index))
                if row is None:
                    raise OscCanaryRunnerError("poststep callback lacks its shield decision")
                live_nonarm_ctrl_after = np.asarray(
                    raw_data.ctrl[list(nonarm_ctrl_indices)], dtype=np.float64
                ).copy()
                if (
                    live_nonarm_ctrl_after.shape
                    != (len(nonarm_ctrl_indices),)
                    or not np.all(np.isfinite(live_nonarm_ctrl_after))
                ):
                    raise OscCanaryRunnerError(
                        "live post-transition non-arm controls are invalid"
                    )
                live_nonarm_ctrl_before_record = _compact_float64_array_record(
                    row["live_nonarm_ctrl_before"], np
                )
                live_nonarm_ctrl_after_record = _compact_float64_array_record(
                    live_nonarm_ctrl_after, np
                )
                live_nonarm_ctrl_byte_identical = bool(
                    row["live_nonarm_ctrl_before"].tobytes(order="C")
                    == live_nonarm_ctrl_after.tobytes(order="C")
                )
                monitor.observe_post_integration(
                    sim,
                    high_level_index=source_index,
                    inner_control_index=(
                        int(substep_index)
                        // physics_substeps_per_controller_update
                    ),
                    physics_substep_index=(
                        int(substep_index)
                        % physics_substeps_per_controller_update
                    ),
                    measure_full_surface_clearance=(
                        int(substep_index)
                        == physics_substeps_per_action - 1
                    ),
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
                measured_arm_qvel = np.asarray(
                    raw_data.qvel[list(arm_qvel)], dtype=np.float64
                )
                measured_shield_qvel = np.asarray(
                    raw_data.qvel[list(robot_qvel)], dtype=np.float64
                )
                clone_matches_live = bool(
                    np.allclose(
                        measured_shield_qvel,
                        row["predicted_next_shield_qvel"],
                        rtol=0.0,
                        atol=1e-10,
                    )
                    and np.allclose(
                        measured_arm_qvel,
                        row["predicted_next_arm_qvel"],
                        rtol=0.0,
                        atol=1e-10,
                    )
                )
                residuals = (
                    row["rows"] @ measured_shield_qvel
                    + float(shield_cfg["alpha_gain_per_s"]) * row["h"]
                    - float(shield_cfg["margin_m2_per_s"])
                )
                minimum_actual_residual = float(np.min(residuals))
                actual_hdot = row["rows"] @ measured_shield_qvel
                candidate_hdot = (
                    row["rows"] @ row["predicted_next_shield_qvel"]
                )
                nominal_hdot = (
                    row["rows"] @ row["nominal_predicted_next_shield_qvel"]
                )
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
                        "vlsa_poisson_compact_constraint_trace.v2"
                    ),
                    "array_hash_format": _FLOAT64_ARRAY_HASH_FORMAT,
                    "sample_count": int(row["h"].size),
                    "sample_ledger_sha256": (
                        shield_samples.sample_ledger_sha256
                    ),
                    "robot_qvel_indices": list(robot_qvel),
                    "robot_qvel_indices_sha256": (
                        robot_qvel_authority["robot_qvel_indices_sha256"]
                    ),
                    "velocity_dimension": len(robot_qvel),
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
                            "registered divergence/material row lacks a solved "
                            "full QP certificate"
                        )
                    full_certificate = {
                        "schema_version": (
                            "vlsa_poisson_divergence_or_material_full_qp_"
                            "certificate.v4"
                            if is_target_link_v4
                            else (
                                "vlsa_poisson_first_divergence_full_qp_"
                                "certificate.v2"
                            )
                        ),
                        "physical_boundary": int(row["physical_boundary"]),
                        "source_action_index": int(source_index),
                        "callback_endpoint_action_inner_substep": (
                            callback_endpoint(source_index, int(substep_index))
                        ),
                        "sample_count": int(row["h"].size),
                        "sample_ledger_sha256": (
                            shield_samples.sample_ledger_sha256
                        ),
                        "robot_qvel_indices": list(robot_qvel),
                        "robot_qvel_indices_sha256": (
                            robot_qvel_authority["robot_qvel_indices_sha256"]
                        ),
                        "velocity_dimension": len(robot_qvel),
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
                    if is_target_link_v4:
                        full_certificate["certificate_roles"] = list(
                            row["full_constraint_qp_certificate_roles"]
                        )
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
                        "controller_update_index": (
                            int(substep_index)
                            // physics_substeps_per_controller_update
                        ),
                        "physics_substep_within_controller_update": (
                            int(substep_index)
                            % physics_substeps_per_controller_update
                        ),
                        "callback_endpoint_action_inner_substep": (
                            callback_endpoint(source_index, int(substep_index))
                        ),
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
                        "measured_arm_qvel_rad_s": measured_arm_qvel.tolist(),
                        "measured_arm_qvel_l2_rad_s": float(
                            np.linalg.norm(measured_arm_qvel)
                        ),
                        "measured_shield_qvel_rad_s": (
                            measured_shield_qvel.tolist()
                        ),
                        "predicted_next_arm_qvel_rad_s": row[
                            "predicted_next_arm_qvel"
                        ].tolist(),
                        "nominal_predicted_next_arm_qvel_rad_s": row[
                            "nominal_predicted_next_arm_qvel"
                        ].tolist(),
                        "predicted_next_shield_qvel_rad_s": row[
                            "predicted_next_shield_qvel"
                        ].tolist(),
                        "nominal_predicted_next_shield_qvel_rad_s": row[
                            "nominal_predicted_next_shield_qvel"
                        ].tolist(),
                        "shield_prediction_error_l2_rad_s": float(
                            np.linalg.norm(
                                measured_shield_qvel
                                - row["predicted_next_shield_qvel"]
                            )
                        ),
                        "candidate_exact_clone_matches_live_shield_qvel": (
                            clone_matches_live
                        ),
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
                        "nominal_non_arm_ctrl_preserved": row[
                            "nominal_non_arm_ctrl_preserved"
                        ],
                        "candidate_non_arm_ctrl_preserved": row[
                            "candidate_non_arm_ctrl_preserved"
                        ],
                        "live_nonarm_ctrl_count": len(nonarm_ctrl_indices),
                        "live_nonarm_ctrl_indices_sha256": _canonical_sha256(
                            list(nonarm_ctrl_indices)
                        ),
                        "live_nonarm_ctrl_before_array_record": (
                            live_nonarm_ctrl_before_record
                        ),
                        "live_nonarm_ctrl_after_array_record": (
                            live_nonarm_ctrl_after_record
                        ),
                        "live_nonarm_ctrl_byte_identical": (
                            live_nonarm_ctrl_byte_identical
                        ),
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
                    expected_substeps=physics_substeps_per_action,
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
                    len(physics_trace)
                    - source_index * physics_substeps_per_action
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
                    valid_partial_count = (
                        1
                        <= completed_partial_substeps
                        <= physics_substeps_per_action
                    )
                    terminal_observation_kind = "post_contact_observable_refresh"
                else:
                    valid_partial_count = (
                        0
                        <= completed_partial_substeps
                        < physics_substeps_per_action
                    )
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
                    "physics_boundary_start": int(
                        source_index * physics_substeps_per_action
                    ),
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
            if len(pending) != physics_substeps_per_action:
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
                    "completed_physics_substeps": physics_substeps_per_action,
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
        partial_action_substeps = (
            len(physics_trace)
            - completed_actions * physics_substeps_per_action
        )
        expected_executed_substeps = (
            completed_actions * physics_substeps_per_action
            + partial_action_substeps
        )
        solved_qp_count = sum(
            row["shield_status"] == "solved" for row in physics_trace
        )
        full_qp_certificate_count = sum(
            isinstance(row.get("full_constraint_qp_certificate"), Mapping)
            for row in physics_trace
        )
        full_qp_certificate_role_counts = {
            role: sum(
                role
                in row.get("full_constraint_qp_certificate", {}).get(
                    "certificate_roles", ()
                )
                for row in physics_trace
                if isinstance(row.get("full_constraint_qp_certificate"), Mapping)
            )
            for role in (
                "first_byte_divergence",
                "first_material_correction",
            )
        }
        if is_target_link_v4:
            phase_endpoint_ledger_exact = all(
                int(row["physical_boundary"])
                == int(row["source_action_index"])
                * physics_substeps_per_action
                + int(row["physics_substep_index"])
                and int(row["controller_update_index"])
                == int(row["physics_substep_index"])
                // physics_substeps_per_controller_update
                and int(row["physics_substep_within_controller_update"])
                == int(row["physics_substep_index"])
                % physics_substeps_per_controller_update
                and row["callback_endpoint_action_inner_substep"]
                == callback_endpoint(
                    int(row["source_action_index"]),
                    int(row["physics_substep_index"]),
                )
                and all(
                    contact.get(
                        "callback_endpoint_action_inner_substep"
                    )
                    == row["callback_endpoint_action_inner_substep"]
                    and contact.get("controller_update_index")
                    == row["controller_update_index"]
                    and contact.get(
                        "physics_substep_within_controller_update"
                    )
                    == row["physics_substep_within_controller_update"]
                    for contact in row["registered_forbidden_contacts"]
                )
                for row in physics_trace
            )
            if not phase_endpoint_ledger_exact:
                raise OscCanaryRunnerError(
                    "v4 physics/contact trace differs from the frozen typed "
                    "callback cadence"
                )
            certificate_rows = [
                row
                for row in physics_trace
                if isinstance(row.get("full_constraint_qp_certificate"), Mapping)
            ]
            expected_certificate_boundaries = {
                int(value)
                for value in (
                    first_torque_divergence_boundary,
                    first_material_boundary,
                )
                if value is not None
            }
            if not (
                {
                    int(row["physical_boundary"])
                    for row in certificate_rows
                }
                == expected_certificate_boundaries
                and full_qp_certificate_count
                == len(expected_certificate_boundaries)
                and full_qp_certificate_role_counts[
                    "first_byte_divergence"
                ]
                == int(first_torque_divergence_boundary is not None)
                and full_qp_certificate_role_counts[
                    "first_material_correction"
                ]
                == int(first_material_boundary is not None)
                and all(
                    row["full_constraint_qp_certificate"]["schema_version"]
                    == (
                        "vlsa_poisson_divergence_or_material_full_qp_"
                        "certificate.v4"
                    )
                    and row["full_constraint_qp_certificate"][
                        "physical_boundary"
                    ]
                    == int(row["physical_boundary"])
                    and row["full_constraint_qp_certificate"][
                        "source_action_index"
                    ]
                    == int(row["source_action_index"])
                    and row["full_constraint_qp_certificate"][
                        "callback_endpoint_action_inner_substep"
                    ]
                    == row["callback_endpoint_action_inner_substep"]
                    and row["callback_endpoint_action_inner_substep"]
                    == callback_endpoint(
                        int(row["source_action_index"]),
                        int(row["physics_substep_index"]),
                    )
                    and set(
                        row["full_constraint_qp_certificate"][
                            "certificate_roles"
                        ]
                    )
                    <= {
                        "first_byte_divergence",
                        "first_material_correction",
                    }
                    and len(
                        row["full_constraint_qp_certificate"][
                            "certificate_roles"
                        ]
                    )
                    == len(
                        set(
                            row["full_constraint_qp_certificate"][
                                "certificate_roles"
                            ]
                        )
                    )
                    for row in certificate_rows
                )
            ):
                raise OscCanaryRunnerError(
                    "v4 full QP certificates do not exactly cover the deduplicated "
                    "first-byte-divergence and first-material rows"
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
        executed_shield_rows_valid = bool(
            shield_decision_count == len(physics_trace)
            and len(physics_trace) == expected_executed_substeps
            and all(
                row["shield_status"]
                in ("nominal_safe_exact_clone", "nominal_safe", "solved")
                for row in physics_trace
            )
        )
        all_shield_valid = bool(
            executed_shield_rows_valid
            and (
                is_target_link_v4
                or (method_stop_record is None and shield_failure_count == 0)
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
            all(
                row["registered_forbidden_contact_seen"] is False
                for row in physics_trace
                if int(row["physical_boundary"])
                < (
                    int(first_torque_divergence_boundary)
                    if first_torque_divergence_boundary is not None
                    else len(physics_trace)
                )
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
        historical_prior_endpoint_action = int(
            derived.get(
                "historical_prior_endpoint_action",
                int(derived["historical_contact_action"]) - 1,
            )
        )
        historical_prior_endpoint_parity_exact = bool(
            any(
                int(row["source_action_index"])
                == historical_prior_endpoint_action
                and row["historical_match"] is True
                for row in parity_trace
            )
        )
        historical_prior_endpoint_contact_free = bool(
            any(
                int(row["source_action_index"])
                == historical_prior_endpoint_action
                for row in action_trace
            )
            and all(
                row["registered_forbidden_contact_seen"] is False
                for row in physics_trace
                if int(row["source_action_index"])
                <= historical_prior_endpoint_action
            )
        )
        timing_action = (
            first_material_action
            if is_target_link_v4
            else first_torque_divergence_action
        )
        historical_same_action_exception_parity = bool(
            timing_action is None
            or (
                int(timing_action) < int(derived["historical_contact_action"])
                or (
                    int(timing_action) == int(derived["historical_contact_action"])
                    and historical_prior_endpoint_parity_exact
                    and historical_prior_endpoint_contact_free
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
        first_material_row = next(
            (
                row
                for row in physics_trace
                if first_material_boundary is not None
                and int(row["physical_boundary"]) == int(first_material_boundary)
            ),
            None,
        )
        first_material_callback_endpoint = (
            None
            if first_material_row is None
            else list(
                first_material_row[
                    "callback_endpoint_action_inner_substep"
                ]
            )
        )
        if first_material_row is not None:
            reconstructed_first_material_callback_endpoint = callback_endpoint(
                int(first_material_row["source_action_index"]),
                int(first_material_row["physics_substep_index"]),
            )
            if not (
                first_material_callback_endpoint
                == reconstructed_first_material_callback_endpoint
                and int(first_material_row["controller_update_index"])
                == reconstructed_first_material_callback_endpoint[1]
                and int(
                    first_material_row[
                        "physics_substep_within_controller_update"
                    ]
                )
                == reconstructed_first_material_callback_endpoint[2]
                and int(first_material_row["physical_boundary"])
                == reconstructed_first_material_callback_endpoint[0]
                * physics_substeps_per_action
                + int(first_material_row["physics_substep_index"])
            ):
                raise OscCanaryRunnerError(
                    "first-material typed callback endpoint is inconsistent with "
                    "the frozen controller/physics cadence"
                )
        same_source_action_correction_is_prephysics_callback_zero = bool(
            first_material_row is not None
            and first_material_action is not None
            and int(first_material_action)
            == int(derived["historical_contact_action"])
            and first_material_callback_endpoint
            == [
                int(derived["historical_contact_action"]),
                *[
                    int(value)
                    for value in derived.get(
                        "same_source_action_contact_exception_callback_suffix",
                        (0, 0),
                    )
                ],
            ]
            and int(first_material_row["physical_boundary"])
            == int(derived["historical_contact_action"])
            * physics_substeps_per_action
        )
        material_before = bool(
            first_material_action is not None
            and (is_target_link_v4 or first_divergence_is_material)
            and (
                int(first_material_action) < int(derived["historical_contact_action"])
                or (
                    int(first_material_action)
                    == int(derived["historical_contact_action"])
                    and same_source_action_correction_is_prephysics_callback_zero
                    and contact_free_before_material
                    and historical_same_action_exception_parity
                    and (
                        is_target_link_v4
                        or (
                            contact_free_before_divergence
                            and predivergence_torque_commands_exact
                        )
                    )
                )
            )
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
        first_material_correction_l2_nm = (
            None
            if first_material_row is None
            else float(first_material_row["torque_correction_l2_nm"])
        )
        first_material_next_qvel_change = (
            None
            if first_material_row is None
            else float(
                np.linalg.norm(
                    np.asarray(
                        first_material_row["predicted_next_arm_qvel_rad_s"],
                        dtype=np.float64,
                    )
                    - np.asarray(
                        first_material_row[
                            "nominal_predicted_next_arm_qvel_rad_s"
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
        first_material_certificate = (
            None
            if first_material_row is None
            else first_material_row.get("full_constraint_qp_certificate")
        )
        if is_target_link_v4 and first_material_row is not None:
            if not (
                isinstance(first_material_certificate, Mapping)
                and first_material_certificate.get("schema_version")
                == (
                    "vlsa_poisson_divergence_or_material_full_qp_"
                    "certificate.v4"
                )
                and "first_material_correction"
                in first_material_certificate.get("certificate_roles", ())
                and isinstance(
                    first_material_certificate.get("constraint_attribution"),
                    Mapping,
                )
            ):
                raise OscCanaryRunnerError(
                    "first material correction lacks its registered full target-link "
                    "QP certificate"
                )
        first_material_attribution = (
            first_material_certificate["constraint_attribution"]
            if is_target_link_v4
            and isinstance(first_material_certificate, Mapping)
            else first_divergence_attribution
        )
        literal_link56_body_names = {
            str(value)
            for value in protocol["case"]["literal_link56_body_names"]
        }
        first_material_link56_attributed = bool(
            first_material_attribution is not None
            and first_material_attribution["minimum_sample"]["body_name"]
            in literal_link56_body_names
        )
        protected_link_body_name_set = {
            str(value) for value in protected_link_body_names
        }
        first_material_protected_link_attributed = bool(
            first_material_attribution is not None
            and first_material_attribution["minimum_sample"]["body_name"]
            in protected_link_body_name_set
        )
        target_link_body_name_set = {
            str(value)
            for value in (
                derived["target_link_body_names"]
                if is_target_link_v4
                else protocol["case"]["literal_link56_body_names"]
            )
        }
        first_divergence_target_link_attributed = bool(
            first_divergence_attribution is not None
            and first_divergence_attribution["minimum_sample"]["body_name"]
            in target_link_body_name_set
        )
        first_material_target_link_attributed = bool(
            first_material_attribution is not None
            and first_material_attribution["minimum_sample"]["body_name"]
            in target_link_body_name_set
        )
        first_material_protected_row_diagnostics = None
        if is_target_link_v4 and first_material_row is not None:
            material_certificate = first_material_row[
                "full_constraint_qp_certificate"
            ]
            material_nominal_residuals = (
                np.asarray(
                    material_certificate[
                        "nominal_exact_clone_hdot_m2_per_s"
                    ],
                    dtype=np.float64,
                )
                + float(shield_cfg["alpha_gain_per_s"])
                * np.asarray(
                    material_certificate["poisson_h_m2"],
                    dtype=np.float64,
                )
                - float(shield_cfg["margin_m2_per_s"])
            )
            if material_nominal_residuals.shape != (
                len(shield_samples.samples),
            ) or not np.all(np.isfinite(material_nominal_residuals)):
                raise OscCanaryRunnerError(
                    "first-material protected-row residual reconstruction differs"
                )
            reconstructed_minimum_index = int(
                np.argmin(material_nominal_residuals)
            )
            if not (
                first_material_attribution is not None
                and int(first_material_attribution["minimum_sample_index"])
                == reconstructed_minimum_index
                and first_material_attribution["minimum_sample"]
                == shield_samples.samples[
                    reconstructed_minimum_index
                ].to_dict()
            ):
                raise OscCanaryRunnerError(
                    "first-material minimum protected-row attribution differs"
                )
            active_indices = np.flatnonzero(material_nominal_residuals < 0.0)
            first_material_protected_row_diagnostics = {
                "schema_version": (
                    "vlsa_poisson_first_material_protected_rows.v1"
                ),
                "sample_scope": protocol["shield"]["binding_scope"],
                "minimum_row": {
                    "sample_index": reconstructed_minimum_index,
                    "sample": shield_samples.samples[
                        reconstructed_minimum_index
                    ].to_dict(),
                    "nominal_exact_cbf_residual_m2_per_s": float(
                        np.min(material_nominal_residuals)
                    ),
                },
                "negative_nominal_exact_cbf_rows": [
                    {
                        "sample_index": int(index),
                        "sample_id": int(shield_samples.samples[int(index)].sample_id),
                        "body_name": str(
                            shield_samples.samples[int(index)].body_name
                        ),
                        "geom_name": str(
                            shield_samples.samples[int(index)].geom_name
                        ),
                        "nominal_exact_cbf_residual_m2_per_s": float(
                            material_nominal_residuals[int(index)]
                        ),
                    }
                    for index in active_indices
                ],
                "negative_nominal_exact_cbf_row_count": int(
                    active_indices.size
                ),
            }
        nonstopping_motion_after_correction = bool(
            first_material_boundary is not None
            and float(motion["joint_motion_integral_rad"])
            >= float(
                protocol["acceptance"][
                    "minimum_post_correction_joint_motion_integral_rad"
                ]
            )
            and float(motion["eef_path_length_m"])
            >= float(
                protocol["acceptance"][
                    "minimum_post_correction_eef_path_length_m"
                ]
            )
        )
        treatment_stalled_after_correction = bool(
            first_material_boundary is not None
            and not nonstopping_motion_after_correction
        )
        material_correction_metric_l2_nm = (
            first_material_correction_l2_nm
            if is_target_link_v4
            else first_torque_divergence_correction_l2_nm
        )
        reconstructed_strict_material_before = bool(
            first_material_action is not None
            and (
                int(first_material_action) < int(derived["historical_contact_action"])
                or (
                    int(first_material_action)
                    == int(derived["historical_contact_action"])
                    and same_source_action_correction_is_prephysics_callback_zero
                    and historical_prior_endpoint_parity_exact
                    and historical_prior_endpoint_contact_free
                    and contact_free_before_material
                )
            )
        )
        strict_same_action_timing_rule_enforced = bool(
            is_target_link_v4
            and material_before is reconstructed_strict_material_before
            and (
                first_material_row is not None
                or (
                    first_material_action is None
                    and first_material_boundary is None
                    and first_material_callback_endpoint is None
                    and material_before is False
                )
            )
        )
        first_material_timing_certificate = {
            "schema_version": (
                "vlsa_poisson_first_material_historical_contact_timing.v1"
            ),
            "historical_target_contact_source_action": int(
                derived["historical_contact_action"]
            ),
            "historical_within_action_contact_substep_known": False,
            "historical_prior_completed_action_endpoint": (
                historical_prior_endpoint_action
            ),
            "historical_prior_completed_endpoint_parity_exact": (
                historical_prior_endpoint_parity_exact
            ),
            "historical_prior_completed_endpoint_contact_free": (
                historical_prior_endpoint_contact_free
            ),
            "first_material_callback_endpoint_action_inner_substep": (
                first_material_callback_endpoint
            ),
            "same_source_action_correction_is_prephysics_callback_zero": (
                same_source_action_correction_is_prephysics_callback_zero
            ),
            "all_earlier_live_substeps_registered_contact_free": (
                contact_free_before_material
            ),
            "correction_strictly_before_historical_target_contact": (
                material_before
            ),
            "independently_reconstructible_strict_before_result": (
                reconstructed_strict_material_before
            ),
            "strict_same_action_timing_rule_enforced": (
                strict_same_action_timing_rule_enforced
            ),
            "same_source_action_positive_rule": (
                "only_callback_endpoint_action_0_0_with_exact_contact_free_"
                "prior_completed_endpoint"
            ),
            "registered_same_source_action_callback_suffix": list(
                derived.get(
                    "same_source_action_contact_exception_callback_suffix",
                    (0, 0),
                )
            ),
        }
        metrics = {
            "allocation_numeric_prerequisite_verified": True,
            "allowed_case_registry_verified": bool(
                is_target_link_v4
                and arguments.case_id == derived["case_id"]
                and protocol["case"]["target_link_body_names"]
                == list(derived["target_link_body_names"])
                and protocol["shield"]["protected_link_body_names"]
                == list(derived["protected_link_body_names"])
            ),
            "historical_control_verified": True,
            "historical_direct_link56_contact_verified": direct_historical_contact,
            "historical_direct_target_link_contact_verified": bool(
                direct_historical_contact
                and (
                    not is_target_link_v4
                    or (
                        historical_target_link_contact_evidence is not None
                        and historical_target_link_contact_evidence.get("verified")
                        is True
                    )
                )
            ),
            "historical_control_car_failure": bool(
                replay.historical_car_collision is True
            ),
            "historical_control_task_success": replay.historical_task_success,
            "archived_baseline_not_rerun": bool(
                source_contract.get("rerun_baseline") is False
                and provenance["historical_control_rerun"] is False
            ),
            "frozen_runtime_parameter_hash_verified": bool(
                is_target_link_v4
                and runtime_hashes.parameter_block_sha256
                == derived["frozen_runtime_parameter_sha256"]
            ),
            "direct_unit_gain_hinge_torque_actuators_verified": torque_units[
                "verified"
            ],
            "original_osc_controller_verified": controller["original_osc_verified"],
            "all_structurally_movable_manipulator_collision_surfaces_shielded": bool(
                not is_target_link_v4
                and shield_sampling_binding["sample_count"]
                == movable_sample_evidence["sample_count"]
                and shield_sampling_binding["sample_ledger_sha256"]
                == movable_sample_evidence["sample_ledger_sha256"]
                and shield_sampling_binding["shield_manipulator_geom_ids"]
                == list(movable_geom_ids)
                and set(movable_geom_ids)
                == {
                    int(row["geom_id"])
                    for row in movable_sample_evidence["geom_records"]
                }
                and all(
                    int(row["constraint_trace"]["sample_count"])
                    == int(movable_sample_evidence["sample_count"])
                    and row["constraint_trace"]["sample_ledger_sha256"]
                    == movable_sample_evidence["sample_ledger_sha256"]
                    for row in physics_trace
                )
            ),
            "protected_link_shield_sampling_exact": bool(
                is_target_link_v4
                and protected_link_shield_ledger_exact
                and shield_sampling_binding["schema_version"]
                == "vlsa_poisson_protected_link_shield_sampling_binding.v1"
                and shield_sampling_binding["scope"]
                == protocol["shield"]["binding_scope"]
                and shield_sampling_binding["sample_source"]
                == "apparatus.protected_sampling.samples"
                and shield_sampling_binding["sample_count"]
                == len(bundle.protected_samples.samples)
                and shield_sampling_binding["sample_ledger_sha256"]
                == movable_sample_evidence["sample_ledger_sha256"]
                and shield_sampling_binding["protected_link_body_ids"]
                == list(protected_link_ids)
                and shield_sampling_binding["protected_link_body_names"]
                == list(protected_link_body_names)
                and shield_sampling_binding["field_bundle_protected_body_ids"]
                == list(protected_link_ids)
                and shield_sampling_binding["field_bundle_protected_body_names"]
                == list(protected_link_body_names)
                and shield_sampling_binding["shield_protected_link_geom_ids"]
                == list(protected_link_geom_ids)
                and shield_sampling_binding["shield_manipulator_geom_ids"]
                == list(protected_link_geom_ids)
                and shield_sampling_binding[
                    "exact_ordered_field_bundle_protected_sample_ledger"
                ]
                is True
            ),
            "link56_shield_sampling_exact": bool(
                is_target_link_v4 and protected_link_shield_ledger_exact
            ),
            "literal_link56_collision_surfaces_only_shielded": bool(
                is_target_link_v4
                and set(link56_geom_ids)
                == {
                    int(row["geom_id"])
                    for row in movable_sample_evidence["geom_records"]
                }
                and shield_sampling_binding["shield_manipulator_geom_ids"]
                == list(link56_geom_ids)
            ),
            "literal_link56_surface_coverage_complete": bool(
                is_target_link_v4
                and field_protected_geom_ids == list(link56_geom_ids)
                and float(bundle.protected_samples.maximum_surface_cover_radius_m)
                < float(bundle.protected_samples.epsilon_m)
            ),
            "protected_link_surface_coverage_complete": bool(
                is_target_link_v4
                and protected_link_shield_ledger_exact
                and field_protected_geom_ids == list(protected_link_geom_ids)
                and float(bundle.protected_samples.maximum_surface_cover_radius_m)
                < float(bundle.protected_samples.epsilon_m)
            ),
            "excluded_fixed_infrastructure_zero_qvel_influence_and_settled_contact_free": bool(
                fixed_infrastructure_settled_certificate[
                    "all_fixed_geoms_zero_structural_qvel_influence"
                ]
                and fixed_infrastructure_settled_certificate[
                    "all_fixed_sample_jacobians_exactly_zero"
                ]
                and fixed_infrastructure_settled_certificate[
                    "settled_contact_free"
                ]
                and fixed_infrastructure_settled_certificate[
                    "all_fixed_geoms_contact_monitored"
                ]
                and apparatus[
                    "fixed_infrastructure_settled_certificate_sha256"
                ]
                == _canonical_sha256(fixed_infrastructure_settled_certificate)
            ),
            "all_authoritative_robot_collision_surfaces_contact_monitored": bool(
                registered_contact_monitor.scope["robot_geom_ids"]
                == [int(value) for value in resolved.robot_geom_ids]
                and registered_contact_measurement["scope_identity_sha256"]
                == registered_contact_monitor.scope["identity_sha256"]
                and robot_geom_influence_partition[
                    "contact_monitor_robot_geom_ids"
                ]
                == [int(value) for value in resolved.robot_geom_ids]
            ),
            "literal_link56_external_nonrobot_contacts_monitored": bool(
                registered_contact_monitor.scope["link56_geom_ids"]
                == sorted(link56_geom_ids)
                and registered_contact_monitor.scope[
                    "external_nonrobot_geom_ids"
                ]
                and registered_contact_measurement[
                    "scope_identity_sha256"
                ]
                == registered_contact_monitor.scope["identity_sha256"]
            ),
            "robot_tree_qvel_scope_verified": bool(
                robot_qvel_authority["arm_qvel_indices_included"] is True
                and robot_qvel_authority["nonarm_robot_qvel_included"] is True
                and robot_qvel_authority["velocity_dimension"]
                == len(robot_qvel)
                and all(
                    row["constraint_trace"]["robot_qvel_indices"]
                    == list(robot_qvel)
                    for row in physics_trace
                )
            ),
            "link56_structural_qvel_scope_verified": bool(
                is_target_link_v4
                and robot_qvel_authority.get(
                    "all_structural_protected_link_qvel_influences_included"
                )
                is True
                and robot_qvel_authority["robot_qvel_indices"]
                == list(arm_qvel)
                and len(robot_qvel) == 7
            ),
            "protected_link_arm_qvel_scope_verified": bool(
                is_target_link_v4
                and robot_qvel_authority.get(
                    "all_structural_protected_link_qvel_influences_included"
                )
                is True
                and robot_qvel_authority["resolved_protected_link_geom_ids"]
                == sorted(protected_link_geom_ids)
                and robot_qvel_authority["robot_qvel_indices"]
                == list(arm_qvel)
                and len(robot_qvel) == 7
            ),
            "link56_arm_qvel_scope_verified": bool(
                is_target_link_v4
                and robot_qvel_authority.get(
                    "all_structural_protected_link_qvel_influences_included"
                )
                is True
                and robot_qvel_authority["robot_qvel_indices"]
                == list(arm_qvel)
                and len(robot_qvel) == 7
            ),
            "seven_arm_torque_decision_verified": bool(
                len(arm_actuators) == 7
                and len(set(arm_actuators)) == 7
                and torque_units["verified"] is True
            ),
            "nonarm_controls_unchanged": bool(
                all(
                    row["nominal_non_arm_ctrl_preserved"] is True
                    and row["candidate_non_arm_ctrl_preserved"] is True
                    and row["live_nonarm_ctrl_byte_identical"] is True
                    and row["live_nonarm_ctrl_before_array_record"]
                    == row["live_nonarm_ctrl_after_array_record"]
                    for row in physics_trace
                )
            ),
            "link56_bundle_samples_field_seed_only": bool(
                not is_target_link_v4
                and len(bundle.protected_samples.samples)
                < len(shield_samples.samples)
                and shield_sampling_binding["field_seed_sample_scope"]
                == "link56_only_not_shield_scope"
            ),
            "static_field_admissible": static_admissible,
            "every_executed_substep_shielded": bool(
                shield_decision_count == len(physics_trace)
                and len(physics_trace) == expected_executed_substeps
            ),
            "every_executed_substep_protected_link_shielded": bool(
                is_target_link_v4
                and phase_endpoint_ledger_exact
                and shield_decision_count == len(physics_trace)
                and len(physics_trace) == expected_executed_substeps
                and all(
                    int(row["constraint_trace"]["sample_count"])
                    == len(bundle.protected_samples.samples)
                    and row["constraint_trace"]["sample_ledger_sha256"]
                    == shield_samples.sample_ledger_sha256
                    for row in physics_trace
                )
            ),
            "every_executed_substep_link56_shielded": bool(
                is_target_link_v4
                and shield_decision_count == len(physics_trace)
                and len(physics_trace) == expected_executed_substeps
                and all(
                    int(row["constraint_trace"]["sample_count"])
                    == len(bundle.protected_samples.samples)
                    and row["constraint_trace"]["sample_ledger_sha256"]
                    == shield_samples.sample_ledger_sha256
                    for row in physics_trace
                )
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
            "complete_exposure_verified": bool(
                complete_terminal
                and shield_decision_count == len(physics_trace)
                and measurement.observed_physics_substeps == len(physics_trace)
                and registered_contact_measurement[
                    "observed_physics_substeps"
                ]
                == len(physics_trace)
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
            "predivergence_archived_state_action_observation_reward_goal_parity_exact": bool(
                len(parity_trace)
                == (
                    completed_actions
                    if first_torque_divergence_action is None
                    else int(first_torque_divergence_action)
                )
                and all(row["historical_match"] is True for row in parity_trace)
            ),
            "all_predivergence_torque_commands_byte_identical_nominal": (
                predivergence_torque_commands_exact
            ),
            "action62_same_action_exception_has_action61_endpoint_parity_exact": (
                historical_same_action_exception_parity
            ),
            "historical_contact_same_action_exception_has_prior_endpoint_parity_exact": (
                historical_same_action_exception_parity
            ),
            "historical_contact_same_source_action_exception_has_exact_prior_completed_endpoint_parity": (
                historical_same_action_exception_parity
            ),
            "historical_contact_same_source_action_exception_requires_exact_callback_action_0_0_and_exact_contact_free_prior_completed_endpoint": (
                strict_same_action_timing_rule_enforced
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
            "first_any_byte_different_torque_registered_as_divergence": bool(
                first_torque_divergence_boundary is not None
                and planner_record["divergence_physical_boundary"]
                == first_torque_divergence_boundary
                and planner_record["divergence_source_action_index"]
                == first_torque_divergence_action
            ),
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
            "nonstopping_motion_after_correction": (
                nonstopping_motion_after_correction
            ),
            "treatment_stalled_after_correction": (
                treatment_stalled_after_correction
            ),
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
            "first_material_correction_minimum_constraint_is_literal_link56": (
                first_material_link56_attributed
            ),
            "first_material_correction_minimum_constraint_is_target_link": (
                first_material_target_link_attributed
            ),
            "first_material_correction_minimum_constraint_is_protected_link": (
                first_material_protected_link_attributed
            ),
            "first_material_correction_protected_link_attributed": (
                first_material_protected_link_attributed
            ),
            "first_material_correction_target_link_attributed": (
                first_material_target_link_attributed
            ),
            "first_material_correction_minimum_constraint_body_name": (
                None
                if first_material_attribution is None
                else first_material_attribution["minimum_sample"]["body_name"]
            ),
            "first_material_correction_near_minimum_constraint_body_names": (
                []
                if first_material_attribution is None
                else first_material_attribution["near_minimum_body_names"]
            ),
            "first_material_correction_negative_nominal_constraint_body_names": (
                []
                if first_material_attribution is None
                else first_material_attribution[
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
            "first_divergence_nominal_exact_target_link_cbf_residual_negative": bool(
                first_divergence_row is not None
                and first_divergence_target_link_attributed
                and float(
                    first_divergence_row[
                        "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
                    ]
                )
                < 0.0
            ),
            "first_material_nominal_exact_target_link_cbf_residual_negative": bool(
                first_material_row is not None
                and first_material_target_link_attributed
                and float(
                    first_material_row[
                        "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
                    ]
                )
                < 0.0
            ),
            "first_material_nominal_exact_protected_link_cbf_residual_negative": bool(
                first_material_row is not None
                and first_material_protected_link_attributed
                and float(
                    first_material_row[
                        "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
                    ]
                )
                < 0.0
            ),
            "first_material_exact_cbf_residual_improvement_m2_per_s": (
                0.0
                if is_target_link_v4
                and first_material_residual_improvement is None
                else first_material_residual_improvement
            ),
            "first_material_exact_next_qvel_change_l2_rad_s": (
                0.0
                if is_target_link_v4
                and first_material_next_qvel_change is None
                else first_material_next_qvel_change
            ),
            "task_incomplete_at_first_material_correction": (
                bool(task_incomplete_at_first_material)
                if is_target_link_v4
                else task_incomplete_at_first_material
            ),
            "material_correction_before_historical_contact": material_before,
            "first_material_correction_before_historical_target_link_contact": (
                material_before
            ),
            "live_substeps_before_material_correction_contact_free": (
                contact_free_before_material
            ),
            "historical_contact_timing_semantics": (
                "post_action_%d_sampled_endpoint;_internal_2ms_historical_"
                "contact_boundary_unknown"
                % int(derived["historical_contact_action"])
            ),
            "first_material_correction_physical_boundary": first_material_boundary,
            "first_material_correction_source_action_index": first_material_action,
            "first_material_correction_callback_endpoint_action_inner_substep": (
                first_material_callback_endpoint
            ),
            "same_source_action_correction_is_prephysics_callback_zero": (
                same_source_action_correction_is_prephysics_callback_zero
            ),
            "historical_first_link_contact_source_action_index": int(
                derived["historical_contact_action"]
            ),
            "historical_prior_completed_action_endpoint_index": (
                historical_prior_endpoint_action
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
            "material_torque_correction_l2_nm": float(
                0.0
                if material_correction_metric_l2_nm is None
                else material_correction_metric_l2_nm
            ),
            "first_divergence_full_qp_certificate_count": int(
                full_qp_certificate_count
            ),
            "deduplicated_divergence_or_material_full_qp_certificate_count": int(
                full_qp_certificate_count
            ),
            "full_qp_certificate_role_counts": dict(
                full_qp_certificate_role_counts
            ),
            "post_correction_joint_motion_integral_rad": motion[
                "joint_motion_integral_rad"
            ],
            "post_correction_eef_path_length_m": motion["eef_path_length_m"],
            "post_correction_zero_torque_delta_fraction": motion[
                "zero_torque_delta_fraction"
            ],
        }
        classification = classifier(metrics, protocol)
        candidate = {
            "schema_version": result_schema,
            "status": "complete",
            "scientific_result": True,
            "protocol_id": protocol["protocol_id"],
            "case_id": arguments.case_id,
            "protected_link_body_names": (
                list(protected_link_body_names)
                if is_target_link_v4
                else None
            ),
            "historical_target_link_body_names": (
                list(target_link_body_names) if is_target_link_v4 else None
            ),
            "run_id": arguments.run_id,
            "provenance": provenance,
            "historical_control": {
                **replay.provenance(),
                "rerun": False,
                "direct_link56_selected_obstacle_contact_verified": direct_historical_contact,
                "direct_target_link_selected_obstacle_contact_verified": bool(
                    direct_historical_contact
                    and is_target_link_v4
                    and historical_target_link_contact_evidence is not None
                    and historical_target_link_contact_evidence.get("verified")
                    is True
                ),
                "archived_direct_target_link_contact": bool(
                    direct_historical_contact
                    and is_target_link_v4
                    and historical_target_link_contact_evidence is not None
                    and historical_target_link_contact_evidence.get("verified")
                    is True
                ),
                "historical_target_link_contact_evidence": (
                    historical_target_link_contact_evidence
                ),
            },
            "apparatus": {
                **apparatus,
                "field_bundle_hashes": asdict(bundle.hashes),
                "field_diagnostics": asdict(bundle.diagnostics),
                "field_obstacle_boxes": field_obstacle_boxes,
                "protected_sampling": protected_sample_payload,
                "serialized_field_certificate_sha256": _canonical_sha256(
                    serialized_field_certificate
                ),
                "field_frame": protocol["shield"]["field_frame"],
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
                "first_material_constraint_attribution": (
                    first_material_attribution
                ),
                "first_material_protected_row_diagnostics": (
                    first_material_protected_row_diagnostics
                ),
                "first_material_timing_certificate": (
                    first_material_timing_certificate
                ),
                "paper_car_endpoint_ledger": car_ledger,
                "paper_car_endpoint_ledger_schema_version": (
                    "vlsa_poisson_paper_car_endpoint_ledger.v3"
                ),
                "physics_trace_schema_version": (
                    "vlsa_poisson_osc_movable_manipulator_compact_physics_trace.v3"
                ),
                "full_qp_certificate_scope": (
                    (
                        "first_byte_divergence_and_first_material_correction_"
                        "rows_deduplicated"
                    )
                    if is_target_link_v4
                    else "first_byte_different_torque_row_only"
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
                from main.poisson_fullbody import (
                    osc_arm_link_canary as canary_contract,
                )

                failure_result_schema = (
                    result_schema
                    if result_schema is not None
                    else canary_contract.RESULT_SCHEMA
                )

                publish_hashed_json(
                    result_path,
                    {
                        "schema_version": failure_result_schema,
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
