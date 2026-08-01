#!/usr/bin/env python3
"""Run one lean, paired link-5/6 Poisson-CBF feasibility window.

This is deliberately not a full-episode safety evaluation.  It restores the
exact historical state at action 180, then compares the same eight recorded
AEGIS-executed actions under the joint-velocity adapter with and without the
link-5/6 Poisson filter.  There are no policy queries and no exhaustive shadow
scan.  A single immutable ``result.json`` is published atomically.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Sequence, Tuple


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PHYSICS_DT_S = 0.002
INNER_DT_S = 0.01


class FastRunnerError(RuntimeError):
    """The lean apparatus cannot support a scientific interpretation."""


class FastAssumptionInvalid(RuntimeError):
    """The preregistered static-field assumption is not valid for this case."""


class FastArmExecutionFailure(FastRunnerError):
    """One paired arm failed after producing auditable partial evidence."""

    def __init__(self, message: str, evidence: Mapping[str, Any]):
        super().__init__(message)
        self.evidence = dict(evidence)


class _PsfContactObserved(RuntimeError):
    """Private clean terminal after a literal PSF-arm contact is recorded."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FastRunnerError("%s must be an existing real file" % label)

    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate key %r" % key)
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant %r" % item)
            ),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise FastRunnerError("%s is invalid JSON" % label) from error
    if not isinstance(value, dict):
        raise FastRunnerError("%s must contain one JSON object" % label)
    return value


def _new_output(root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
        raise FastRunnerError("run ID is not portable")
    if root.is_symlink() or not root.is_dir():
        raise FastRunnerError("output root must be an existing real directory")
    parent = root.resolve()
    output = parent / run_id
    if output.parent != parent or output.exists() or output.is_symlink():
        raise FastRunnerError("run output must be new and remain inside output root")
    output.mkdir(mode=0o755)
    return output


def _manifest_case(path: Path, case_id: str) -> Tuple[Dict[str, Any], str]:
    matches = []
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            value = json.loads(raw)
            if value.get("case_id") == case_id:
                matches.append((value, _sha256(raw.rstrip(b"\r\n"))))
    if len(matches) != 1:
        raise FastRunnerError("manifest must contain exactly one registered case")
    case, digest = matches[0]
    if (
        case.get("split") != "bringup_canary"
        or case.get("study_partition") != "development"
        or case.get("settle_actions") != 20
    ):
        raise FastRunnerError("fast feasibility is restricted to the bring-up case")
    return case, digest


def _historical_path(root: Path, case: Mapping[str, Any]) -> Path:
    record = case.get("historical_aegis_result")
    relative = record.get("source_relative_path") if isinstance(record, Mapping) else None
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or any(part in ("", ".", "..") for part in Path(relative).parts)
    ):
        raise FastRunnerError("historical result binding is invalid")
    if root.is_symlink() or not root.is_dir():
        raise FastRunnerError("historical result root is missing or symlinked")
    current = root.resolve()
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise FastRunnerError("historical result path traverses a symlink")
    if not current.is_file() or _file_sha256(current) != record.get("raw_file_sha256"):
        raise FastRunnerError("historical result file or hash differs")
    return current


def _git(root: Path) -> Dict[str, str]:
    import subprocess

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(args),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

    status = run("status", "--short")
    if status:
        raise FastRunnerError("fast scientific execution requires a clean worktree")
    baseline = "1592aa59361f431ba96c6ddcbebcb596f6c20853"
    if subprocess.call(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", baseline, "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) != 0:
        raise FastRunnerError("registered Table-1 baseline is not an ancestor")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current")}


def _allocation() -> Dict[str, Any]:
    import subprocess

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise FastRunnerError("fast feasibility requires a Slurm allocation")
    lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=30,
    ).strip().splitlines()
    if len(lines) != 1:
        raise FastRunnerError("exactly one GPU must be visible")
    parts = [item.strip() for item in lines[0].split(",")]
    if len(parts) != 3 or "H100" not in parts[0]:
        raise FastRunnerError("the registered exploratory run requires one H100")
    return {
        "slurm_job_id": job_id,
        "host": socket.gethostname(),
        "gpu_name": parts[0],
        "gpu_uuid": parts[1],
        "driver_version": parts[2],
    }


def _raw_model_data(sim: Any) -> Tuple[Any, Any]:
    return getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)


def _body_id(model: Any, name: str) -> int:
    import mujoco

    raw = getattr(model, "_model", model)
    value = int(mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_BODY, str(name)))
    if value < 0:
        raise FastRunnerError("required body %s is absent" % name)
    return value


def _eef_site_id(env: Any) -> int:
    value = getattr(env.robots[0], "eef_site_id", None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    sites = getattr(getattr(env.robots[0], "gripper", None), "important_sites", None)
    name = sites.get("grip_site") if isinstance(sites, Mapping) else None
    if not isinstance(name, str):
        raise FastRunnerError("Panda grip site is unavailable")
    return int(env.sim.model.site_name2id(name))


def _synchronize_visual_marker_model(source_env: Any, target_env: Any) -> Dict[str, Any]:
    """Copy the mutable visual marker pose before physical-model identity checks."""

    import numpy as np

    marker_name = "eef_marker"
    source_id = int(source_env.sim.model.body_name2id(marker_name))
    target_id = int(target_env.sim.model.body_name2id(marker_name))
    source_pos = np.asarray(source_env.sim.model.body_pos[source_id], dtype=np.float64).copy()
    source_quat = np.asarray(source_env.sim.model.body_quat[source_id], dtype=np.float64).copy()
    target_env.sim.model.body_pos[target_id] = source_pos
    target_env.sim.model.body_quat[target_id] = source_quat
    target_pos = np.asarray(target_env.sim.model.body_pos[target_id], dtype=np.float64)
    target_quat = np.asarray(target_env.sim.model.body_quat[target_id], dtype=np.float64)
    if not np.array_equal(source_pos, target_pos) or not np.array_equal(
        source_quat, target_quat
    ):
        raise FastRunnerError("visual marker model pose did not synchronize exactly")
    return {
        "body_name": marker_name,
        "source_body_id": source_id,
        "target_body_id": target_id,
        "position_exact": True,
        "quaternion_exact": True,
        "position_m": source_pos.tolist(),
        "quaternion_wxyz": source_quat.tolist(),
    }


def _eef_kinematics(env: Any, site_id: int, arm_dofs: Sequence[int]) -> Tuple[Any, Any, Any]:
    import mujoco
    import numpy as np

    model, data = _raw_model_data(env.sim)
    position = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    jac_pos = np.zeros((3, int(model.nv)), dtype=np.float64)
    jac_rot = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, int(site_id))
    indexes = np.asarray(arm_dofs, dtype=np.int64)
    jacobian = np.vstack((jac_pos[:, indexes], jac_rot[:, indexes]))
    if jacobian.shape != (6, 7) or not np.all(np.isfinite(jacobian)):
        raise FastRunnerError("grip-site Jacobian is invalid")
    return position, rotation, jacobian


def _joint_limits(env: Any) -> Tuple[Any, Any]:
    import numpy as np

    joint_ids = np.asarray(env.robots[0]._ref_joint_indexes, dtype=np.int64)
    model, _ = _raw_model_data(env.sim)
    ranges = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    if joint_ids.shape != (7,) or ranges.shape != (7, 2) or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise FastRunnerError("Panda joint limits are invalid")
    return ranges[:, 0].copy(), ranges[:, 1].copy()


def _official_state(sim: Any) -> Any:
    import mujoco
    import numpy as np

    model, data = _raw_model_data(sim)
    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    output = np.empty(int(mujoco.mj_stateSize(model, specification)), dtype=np.float64)
    mujoco.mj_getState(model, data, output, specification)
    if output.ndim != 1 or not np.all(np.isfinite(output)):
        raise FastRunnerError("official integration state is invalid")
    return output


def _current_obstacle_boxes(model: Any, data: Any, reference_boxes: Sequence[Any]) -> Tuple[Any, ...]:
    import numpy as np
    from main.poisson_fullbody.geometry import OrientedBox

    return tuple(
        OrientedBox(
            center=np.asarray(data.geom_xpos[int(box.geom_id)], dtype=np.float64),
            R=np.asarray(data.geom_xmat[int(box.geom_id)], dtype=np.float64).reshape(3, 3),
            half_extents=np.asarray(model.geom_size[int(box.geom_id)][:3], dtype=np.float64),
            geom_id=int(box.geom_id),
        )
        for box in reference_boxes
    )


def _rotation_angle(reference: Any, current: Any) -> float:
    import numpy as np

    cosine = (float(np.trace(reference.T @ current)) - 1.0) * 0.5
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def _obstacle_state(model: Any, data: Any, bundle: Any, body_ids: Sequence[int]) -> Dict[str, float]:
    import mujoco
    import numpy as np

    current = {int(box.geom_id): box for box in _current_obstacle_boxes(model, data, bundle.obstacle_boxes)}
    translation = rotation = surface = 0.0
    for reference in bundle.obstacle_boxes:
        box = current[int(reference.geom_id)]
        translation = max(translation, float(np.linalg.norm(box.center - reference.center)))
        rotation = max(rotation, _rotation_angle(reference.R, box.R))
        surface = max(
            surface,
            float(np.max(np.linalg.norm(box.vertices() - reference.vertices(), axis=1))),
        )
    linear = angular = 0.0
    for body_id in sorted(set(int(value) for value in body_ids)):
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        if not np.all(np.isfinite(velocity)):
            raise FastRunnerError("selected-obstacle velocity is invalid")
        angular = max(angular, float(np.linalg.norm(velocity[:3])))
        linear = max(linear, float(np.linalg.norm(velocity[3:])))
    return {
        "translation_drift_m": translation,
        "rotation_drift_rad": rotation,
        "surface_drift_m": surface,
        "maximum_body_linear_speed_m_s": linear,
        "maximum_body_angular_speed_rad_s": angular,
    }


def _static_admissible(rows: Sequence[Mapping[str, float]], protocol: Mapping[str, Any]) -> bool:
    limits = protocol["admissibility"]
    return bool(
        rows
        and max(row["translation_drift_m"] for row in rows)
        <= float(limits["max_selected_geom_translation_drift_m"])
        and max(row["rotation_drift_rad"] for row in rows)
        <= float(limits["max_selected_geom_rotation_drift_rad"])
        and max(row["surface_drift_m"] for row in rows)
        <= float(limits["max_selected_geom_surface_drift_m"])
        and max(row["maximum_body_linear_speed_m_s"] for row in rows)
        <= float(limits["max_selected_body_linear_speed_m_s"])
        and max(row["maximum_body_angular_speed_rad_s"] for row in rows)
        <= float(limits["max_selected_body_angular_speed_rad_s"])
    )


def _tracking(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    import numpy as np

    if not rows:
        return {"observation_count": 0, "linf_rad_s": 0.0, "rmse_rad_s": 0.0}
    errors = np.asarray([row["tracking_error_rad_s"] for row in rows], dtype=np.float64)
    return {
        "observation_count": len(rows),
        "linf_rad_s": float(np.max(np.abs(errors))),
        "rmse_rad_s": float(np.sqrt(np.mean(np.square(errors)))),
    }


def _active_interval_motion(
    command_rows: Sequence[Mapping[str, Any]],
    physics_rows: Sequence[Mapping[str, Any]],
    active_keys: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    """Measure motion only while a material Poisson correction is active."""

    import numpy as np

    keys = set((int(source), int(inner)) for source, inner in active_keys)
    commands = [
        row
        for row in command_rows
        if (int(row["source_action_index"]), int(row["inner_control_index"])) in keys
    ]
    physics = [
        row
        for row in physics_rows
        if (int(row["source_action_index"]), int(row["inner_control_index"])) in keys
        and row.get("eef_position_world_m") is not None
    ]
    nominal_integral = sum(
        float(np.linalg.norm(np.asarray(row["nominal_qdot_rad_s"], dtype=np.float64)))
        * INNER_DT_S
        for row in commands
    )
    executed_integral = sum(
        float(np.linalg.norm(np.asarray(row["executed_qdot_rad_s"], dtype=np.float64)))
        * INNER_DT_S
        for row in commands
    )
    measured_integral = sum(
        float(np.linalg.norm(np.asarray(row["measured_qvel_rad_s"], dtype=np.float64)))
        * PHYSICS_DT_S
        for row in physics
    )
    cartesian_path = 0.0
    target_progress = 0.0
    complete_intervals = 0
    for command in commands:
        key = (
            int(command["source_action_index"]),
            int(command["inner_control_index"]),
        )
        rows = sorted(
            (
                row
                for row in physics
                if (int(row["source_action_index"]), int(row["inner_control_index"]))
                == key
            ),
            key=lambda row: int(row["physics_substep_index"]),
        )
        if len(rows) != 5:
            continue
        previous = np.asarray(
            command["eef_position_before_update_world_m"], dtype=np.float64
        )
        for row in rows:
            current = np.asarray(row["eef_position_world_m"], dtype=np.float64)
            cartesian_path += float(np.linalg.norm(current - previous))
            previous = current
        target_progress += float(command["target_error_before_update_m"]) - float(
            rows[-1]["target_error_after_substep_m"]
        )
        complete_intervals += 1
    return {
        "active_update_count": len(commands),
        "complete_active_interval_count": complete_intervals,
        "maximum_filter_correction_norm_rad_s": max(
            (float(row["correction_l2_rad_s"]) for row in commands), default=0.0
        ),
        "maximum_executed_command_norm_rad_s": max(
            (
                float(
                    np.linalg.norm(
                        np.asarray(row["executed_qdot_rad_s"], dtype=np.float64)
                    )
                )
                for row in commands
            ),
            default=0.0,
        ),
        "safe_to_nominal_command_motion_ratio": (
            0.0 if nominal_integral == 0.0 else executed_integral / nominal_integral
        ),
        "measured_joint_motion_integral_rad": measured_integral,
        "cartesian_path_length_m": cartesian_path,
        "cartesian_target_progress_m_diagnostic_only": target_progress,
    }


def _protected_clearance(model: Any, data: Any, bundle: Any) -> float:
    from main.poisson_fullbody.geometry import minimum_point_to_oriented_boxes_distance
    from main.poisson_fullbody.robot_samples import evaluate_world_points

    current_boxes = _current_obstacle_boxes(model, data, bundle.obstacle_boxes)
    points = evaluate_world_points(bundle.protected_samples.samples, data)
    if len(points) != len(bundle.protected_samples.samples):
        raise FastRunnerError("protected-link point population is incomplete")
    minimum = min(
        minimum_point_to_oriented_boxes_distance(point, current_boxes)
        for point in points
    )
    return float(minimum - bundle.protected_samples.maximum_surface_cover_radius_m)


def _contact_physical_boundary(record: Any, start_boundary: int) -> int:
    if record.observation_index is None:
        raise FastRunnerError("rollout contact lacks an observation index")
    observation = int(record.observation_index)
    if record.source_phase == "live_solver_phase_preintegration_geometry":
        return int(start_boundary) + observation
    if record.source_phase == "post_integration_recomputed":
        return int(start_boundary) + observation + 1
    raise FastRunnerError("rollout contact has an unknown source phase")


def _first_contact_boundary(
    measurement: Any,
    *,
    start_boundary: int,
    robot_geom_ids: Sequence[int] | None = None,
) -> Any:
    allowed = None if robot_geom_ids is None else set(int(value) for value in robot_geom_ids)
    records = list(measurement.live_solver_phase_contact_point_records)
    records.extend(measurement.post_state_physical_contact_point_records)
    boundaries = [
        _contact_physical_boundary(row, start_boundary)
        for row in records
        if row.is_physical_nonpositive_distance_contact
        and (allowed is None or int(row.robot_geom_id) in allowed)
    ]
    return min(boundaries) if boundaries else None


def _first_link_contact_observation(measurement: Any, link_geom_ids: Sequence[int]) -> Any:
    """Compatibility diagnostic only; scientific cutoffs use physical boundaries."""

    ids = set(int(value) for value in link_geom_ids)
    records = list(measurement.live_solver_phase_contact_point_records)
    records.extend(measurement.post_state_physical_contact_point_records)
    indexes = [
        int(row.observation_index)
        for row in records
        if row.is_physical_nonpositive_distance_contact
        and row.robot_geom_id in ids
        and row.observation_index is not None
    ]
    return min(indexes) if indexes else None


def _first_any_contact_observation(measurement: Any) -> Any:
    """Compatibility diagnostic only; scientific cutoffs use physical boundaries."""

    records = list(measurement.live_solver_phase_contact_point_records)
    records.extend(measurement.post_state_physical_contact_point_records)
    indexes = [
        int(row.observation_index)
        for row in records
        if row.is_physical_nonpositive_distance_contact
        and row.observation_index is not None
    ]
    return min(indexes) if indexes else None


def _run_arm(
    *,
    arm_name: str,
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    source_env: Any,
    state_at_B: Any,
    actions: Sequence[Sequence[float]],
    source_start_action: int,
    bundle: Any,
    resolved: Any,
    full_samples: Any,
    runtime_protocol: Mapping[str, Any],
    nominal_activation_threshold_m2_per_s: float,
) -> Dict[str, Any]:
    import numpy as np
    from main.poisson_fullbody.cbf_qp import HardCbfQp, joint_velocity_bounds
    from main.poisson_fullbody.controller_bridge import restore_osc_settled_state_into_joint_velocity_env
    from main.poisson_fullbody.jacobians import evaluate_point_jacobians
    from main.poisson_fullbody.joint_velocity_adapter import (
        TranslationalJointVelocityAdapter,
        normalized_joint_velocity_action,
    )
    from main.poisson_fullbody.measurement import FullRobotObstacleMonitor, clone_forwarded_state

    if arm_name not in ("joint_velocity_adapter_only", "joint_velocity_adapter_plus_link56_psf"):
        raise FastRunnerError("unknown fast arm")
    psf_enabled = arm_name.endswith("psf")
    env = None
    monitor = None
    failure_stage = "build_joint_velocity_environment"
    command_rows: List[Dict[str, Any]] = []
    physics_rows: List[Dict[str, Any]] = []
    action_progress_rows: List[Dict[str, Any]] = []
    static_rows: List[Dict[str, float]] = []
    protected_clearances: List[float] = []
    activation_rows: List[Dict[str, Any]] = []
    last_qp_attempt: Any = None
    invalid_field_queries = 0
    post_state_field_observation_count = 0
    post_state_field_query_count = 0
    nonpositive_post_state_field_query_count = 0
    qp_solve_count = 0
    qp_postcheck_count = 0
    joint_limit_postcheck_count = 0
    issued_bound_check_count = 0
    nominal_dynamic_bound_check_count = 0
    nominal_dynamic_bound_violation_count = 0
    minimum_safe_residual = float("inf")
    nominal_command_integral = 0.0
    executed_command_integral = 0.0
    measured_motion_integral = 0.0
    maximum_correction = 0.0
    maximum_safe_command = 0.0
    minimum_nominal_residual = float("inf")
    contact_terminated_early = False
    pending: Dict[str, Any] = {}
    try:
        failure_stage = "build_joint_velocity_environment"
        env, _, _, _ = evaluator._build_environment(
            runtime,
            case,
            render_resolution=evaluator.TABLE_RENDER_RESOLUTION,
            controller="JOINT_VELOCITY",
            control_frequency_hz=100,
        )
        marker_sync = _synchronize_visual_marker_model(source_env, env)
        limits = runtime_protocol["admissibility"]
        restore = restore_osc_settled_state_into_joint_velocity_env(
            source_env,
            env,
            state_at_B,
            max_arm_qpos_error_rad=float(limits["max_state_restore_qpos_error_rad"]),
            max_arm_qvel_error_rad_s=float(limits["max_state_restore_qvel_error_rad_s"]),
        )
        if (
            restore.get("official_integration_state_available") is not True
            or restore.get("official_integration_state_sha256")
            != restore.get("target_official_integration_state_sha256")
            or restore.get("exact_flattened_state") is not True
            or restore.get("settled_state_sha256") != restore.get("target_state_sha256")
        ):
            raise FastRunnerError("JV arm did not receive exact boundary B")
        pid = restore.get("pid_memory_reset")
        if not isinstance(pid, Mapping) or not all(
            (
                pid.get("goal_velocity_zero") is True,
                pid.get("current_velocity_zero") is True,
                pid.get("last_error_zero") is True,
                pid.get("summed_error_zero") is True,
                pid.get("derivative_buffer_size") == 0,
                pid.get("saturated") is False,
            )
        ):
            raise FastRunnerError("fresh JV PID memory was not reset")

        arm_dofs = tuple(int(value) for value in env.robots[0]._ref_joint_vel_indexes)
        arm_qpos = tuple(int(value) for value in env.robots[0]._ref_joint_pos_indexes)
        if len(arm_dofs) != 7 or len(arm_qpos) != 7:
            raise FastRunnerError("Panda arm index contract differs")
        site_id = _eef_site_id(env)
        q_min, q_max = _joint_limits(env)
        adapter_cfg = runtime_protocol["adapter"]
        adapter = TranslationalJointVelocityAdapter(
            damping=float(adapter_cfg["dls_damping"]),
            position_gain_s_inv=float(adapter_cfg["position_gain_per_s"]),
            orientation_gain_s_inv=float(adapter_cfg["orientation_gain_per_s"]),
        )
        if adapter.target is not None or adapter.inner_update_index != 0:
            raise FastRunnerError("adapter did not start fresh")
        qp_cfg = runtime_protocol["qp"]
        qp = HardCbfQp(
            eps_abs=float(qp_cfg["eps_abs"]),
            eps_rel=float(qp_cfg["eps_rel"]),
            max_iter=int(qp_cfg["max_iterations"]),
            postcheck_cbf_tolerance=float(qp_cfg["postcheck_cbf_tolerance"]),
            postcheck_bound_tolerance=float(qp_cfg["postcheck_bound_tolerance_rad_s"]),
        )
        failure_stage = "construct_contact_monitor"
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
            inner_updates_per_high_level_action=5,
            physics_substeps_per_inner_update=5,
        )
        if monitor.settled_state.any_robot_obstacle_contact:
            raise FastRunnerError("boundary B already has robot-selected-obstacle contact")

        model, data = _raw_model_data(env.sim)
        start_official_sha256 = _sha256(_official_state(env.sim).tobytes())
        static_rows = [_obstacle_state(model, data, bundle, resolved.obstacle_body_ids)]
        protected_clearances = [_protected_clearance(model, data, bundle)]

        for local_index, action_value in enumerate(actions):
            source_index = source_start_action + local_index
            source_action = np.asarray(action_value, dtype=np.float64)
            if source_action.shape != (7,) or not np.all(np.isfinite(source_action)):
                raise FastRunnerError("frozen source action is invalid")
            action_target: Dict[str, Any] = {}

            def provider(sim: Any, inner_index: int) -> Any:
                nonlocal invalid_field_queries, qp_solve_count, qp_postcheck_count
                nonlocal joint_limit_postcheck_count, issued_bound_check_count
                nonlocal minimum_safe_residual, nominal_command_integral
                nonlocal executed_command_integral, maximum_correction, maximum_safe_command
                nonlocal minimum_nominal_residual
                nonlocal nominal_dynamic_bound_check_count
                nonlocal nominal_dynamic_bound_violation_count
                nonlocal last_qp_attempt, failure_stage

                failure_stage = "compute_fresh_adapter_target_or_nominal"
                position_now, rotation_now, eef_jacobian = _eef_kinematics(env, site_id, arm_dofs)
                if int(inner_index) == 0:
                    target = adapter.begin_high_level_action(
                        source_action, position_now, rotation_now
                    )
                    action_target.update(
                        {
                            "target": target,
                            "start_error_m": float(
                                np.linalg.norm(target.target_position - position_now)
                            ),
                        }
                    )
                elif "target" not in action_target:
                    raise FastRunnerError("adapter target was not initialized at inner update zero")
                measured = np.asarray(env.sim.data.qvel[list(arm_dofs)], dtype=np.float64)
                step = adapter.compute_inner_action(
                    position_now,
                    rotation_now,
                    eef_jacobian,
                    measured_joint_velocity=measured,
                )
                nominal = np.asarray(step.qdot_physical, dtype=np.float64)
                executed = nominal.copy()
                qp_record = None
                independent_minimum = None
                nominal_residual_minimum = None
                nominal_argmin_sample = None
                update_physical_boundary = (
                    source_start_action * 25
                    + local_index * 25
                    + int(inner_index) * 5
                )
                q = np.asarray(data.qpos[list(arm_qpos)], dtype=np.float64)
                lower, upper = joint_velocity_bounds(
                    q,
                    q_min,
                    q_max,
                    qp_cfg["velocity_lower_rad_s"],
                    qp_cfg["velocity_upper_rad_s"],
                    alpha_joint=float(qp_cfg["joint_limit_alpha_per_s"]),
                    control_dt_seconds=INNER_DT_S,
                    position_margin_rad=float(qp_cfg["joint_position_margin_rad"]),
                )
                bound_tolerance = float(qp_cfg["postcheck_bound_tolerance_rad_s"])
                nominal_within_dynamic_bounds = bool(
                    np.all(nominal >= lower - bound_tolerance)
                    and np.all(nominal <= upper + bound_tolerance)
                )
                nominal_dynamic_bound_check_count += 1
                if not nominal_within_dynamic_bounds:
                    nominal_dynamic_bound_violation_count += 1
                if psf_enabled:
                    failure_stage = "query_poisson_field_and_solve_hard_qp"
                    forwarded = clone_forwarded_state(model, data)
                    points, jacobians = evaluate_point_jacobians(
                        model,
                        forwarded,
                        bundle.protected_samples.samples,
                        arm_dofs,
                    )
                    queries = [bundle.field.query(point) for point in points]
                    invalid = [query for query in queries if not query.valid or query.value is None or query.gradient is None]
                    invalid_field_queries += len(invalid)
                    if invalid:
                        raise FastRunnerError("PSF field query invalid before physics")
                    h = np.asarray([float(query.value) for query in queries], dtype=np.float64)
                    gradients = np.asarray([query.gradient for query in queries], dtype=np.float64)
                    if np.any(h <= 0.0):
                        raise FastRunnerError("PSF protected sample left strict h>0 before physics")
                    result = qp.solve_from_field(
                        nominal,
                        h,
                        gradients,
                        jacobians,
                        lower,
                        upper,
                        alpha=float(runtime_protocol["cbf"]["alpha_gain_per_s"]),
                        weight_diagonal=qp_cfg["weight_diagonal"],
                        require_safe_start=True,
                    )
                    last_qp_attempt = {
                        "local_action_index": local_index,
                        "source_action_index": source_index,
                        "inner_control_index": int(inner_index),
                        "physical_boundary": update_physical_boundary,
                        "valid": bool(result.valid),
                        "reason": str(result.reason),
                        "diagnostics": dict(result.diagnostics),
                    }
                    if not result.valid or result.qdot_safe is None:
                        raise FastRunnerError("hard PSF QP failed before physics: %s" % result.reason)
                    qp_solve_count += 1
                    executed = np.asarray(result.qdot_safe, dtype=np.float64)
                    rows = np.einsum("ni,nij->nj", gradients, jacobians)
                    nominal_residuals = (
                        rows @ nominal
                        + float(runtime_protocol["cbf"]["alpha_gain_per_s"]) * h
                    )
                    nominal_argmin = int(np.argmin(nominal_residuals))
                    nominal_residual_minimum = float(
                        nominal_residuals[nominal_argmin]
                    )
                    minimum_nominal_residual = min(
                        minimum_nominal_residual, nominal_residual_minimum
                    )
                    sample = bundle.protected_samples.samples[nominal_argmin]
                    nominal_argmin_sample = {
                        "sample_id": int(sample.sample_id),
                        "body_id": int(sample.body_id),
                        "body_name": str(sample.body_name),
                        "geom_id": int(sample.geom_id),
                        "geom_name": str(sample.geom_name),
                    }
                    residuals = rows @ executed + float(runtime_protocol["cbf"]["alpha_gain_per_s"]) * h
                    independent_minimum = float(np.min(residuals))
                    if independent_minimum < -float(qp_cfg["postcheck_cbf_tolerance"]):
                        raise FastRunnerError("independent safe CBF residual postcheck failed")
                    if np.any(executed < lower - float(qp_cfg["postcheck_bound_tolerance_rad_s"])) or np.any(
                        executed > upper + float(qp_cfg["postcheck_bound_tolerance_rad_s"])
                    ):
                        raise FastRunnerError("independent QP velocity-bound postcheck failed")
                    qp_postcheck_count += 1
                    next_q = q + INNER_DT_S * executed
                    margin = float(qp_cfg["joint_position_margin_rad"])
                    if np.any(next_q < q_min + margin - 1e-10) or np.any(next_q > q_max - margin + 1e-10):
                        raise FastRunnerError("independent one-step joint-limit postcheck failed")
                    joint_limit_postcheck_count += 1
                    minimum_safe_residual = min(minimum_safe_residual, independent_minimum)
                    qp_record = dict(result.diagnostics)
                if np.any(np.abs(executed) > 0.5 + 64.0 * np.finfo(np.float64).eps):
                    raise FastRunnerError("issued command exceeds physical bounds")
                issued_bound_check_count += 1
                execution = adapter.record_executed_joint_velocity(executed)
                normalized = normalized_joint_velocity_action(executed, float(source_action[6]))
                nominal_norm = float(np.linalg.norm(nominal))
                executed_norm = float(np.linalg.norm(executed))
                correction_norm = float(np.linalg.norm(executed - nominal))
                nominal_command_integral += nominal_norm * INNER_DT_S
                executed_command_integral += executed_norm * INNER_DT_S
                maximum_safe_command = max(maximum_safe_command, executed_norm)
                maximum_correction = max(maximum_correction, correction_norm)
                if (
                    psf_enabled
                    and nominal_residual_minimum is not None
                    and nominal_residual_minimum
                    <= float(nominal_activation_threshold_m2_per_s)
                ):
                    activation_rows.append(
                        {
                            "local_action_index": local_index,
                            "source_action_index": source_index,
                            "inner_control_index": int(inner_index),
                            "physical_boundary": update_physical_boundary,
                            "nominal_minimum_cbf_residual_m2_per_s": nominal_residual_minimum,
                            "activation_threshold_m2_per_s": float(
                                nominal_activation_threshold_m2_per_s
                            ),
                            "argmin_protected_sample": nominal_argmin_sample,
                            "safe_minimum_cbf_residual_m2_per_s": independent_minimum,
                            "filter_correction_l2_rad_s": correction_norm,
                            "nominal_within_dynamic_joint_bounds": nominal_within_dynamic_bounds,
                        }
                    )
                pending.clear()
                pending.update(
                    {
                        "local_action_index": local_index,
                        "source_action_index": source_index,
                        "inner_index": int(inner_index),
                        "nominal": nominal,
                        "executed": executed,
                    }
                )
                command_rows.append(
                    {
                        "local_action_index": local_index,
                        "source_action_index": source_index,
                        "inner_control_index": int(inner_index),
                        "physical_boundary": update_physical_boundary,
                        "nominal_qdot_rad_s": nominal.tolist(),
                        "executed_qdot_rad_s": executed.tolist(),
                        "correction_l2_rad_s": correction_norm,
                        "safe_cbf_residual_minimum_m2_per_s": independent_minimum,
                        "nominal_cbf_residual_minimum_m2_per_s": nominal_residual_minimum,
                        "nominal_argmin_protected_sample": nominal_argmin_sample,
                        "nominal_dynamic_lower_bound_rad_s": lower.tolist(),
                        "nominal_dynamic_upper_bound_rad_s": upper.tolist(),
                        "nominal_within_dynamic_joint_bounds": nominal_within_dynamic_bounds,
                        "eef_position_before_update_world_m": position_now.tolist(),
                        "target_error_before_update_m": float(
                            np.linalg.norm(
                                action_target["target"].target_position - position_now
                            )
                        ),
                        "adapter": step.to_record(),
                        "execution": execution,
                        "qp": qp_record,
                    }
                )
                failure_stage = "integrate_and_measure_physics"
                return normalized

            def callback(sim: Any, inner_index: int, physics_index: int) -> None:
                nonlocal measured_motion_integral, invalid_field_queries
                nonlocal post_state_field_observation_count
                nonlocal post_state_field_query_count
                nonlocal nonpositive_post_state_field_query_count
                nonlocal failure_stage
                if pending.get("inner_index") != int(inner_index):
                    raise FastRunnerError("physics callback lacks its issued command")
                observation_index = len(physics_rows)
                post_state_physical_boundary = (
                    source_start_action * 25 + observation_index + 1
                )
                measured = np.asarray(env.sim.data.qvel[list(arm_dofs)], dtype=np.float64)
                measured_motion_integral += float(np.linalg.norm(measured)) * PHYSICS_DT_S
                error = measured - pending["executed"]
                monitor.observe_post_integration(
                    sim,
                    high_level_index=local_index,
                    inner_control_index=int(inner_index),
                    physics_substep_index=int(physics_index),
                )
                current_measurement = monitor.result()
                any_contact_seen = bool(current_measurement.any_robot_obstacle_contact)
                if psf_enabled and any_contact_seen:
                    physics_rows.append(
                        {
                            "observation_index": observation_index,
                            "post_state_physical_boundary": post_state_physical_boundary,
                            "local_action_index": local_index,
                            "source_action_index": source_index,
                            "inner_control_index": int(inner_index),
                            "physics_substep_index": int(physics_index),
                            "measured_qvel_rad_s": measured.tolist(),
                            "issued_qvel_rad_s": pending["executed"].tolist(),
                            "tracking_error_rad_s": error.tolist(),
                            "literal_contact_observed": True,
                            "protected_link_full_surface_clearance_lower_bound_m": None,
                            "post_state_minimum_h_m2": None,
                            "selected_obstacle": None,
                        }
                    )
                    raise _PsfContactObserved()
                forwarded = clone_forwarded_state(model, data)
                static = _obstacle_state(model, forwarded, bundle, resolved.obstacle_body_ids)
                static_rows.append(static)
                if not any_contact_seen and not _static_admissible([static], runtime_protocol):
                    failure_stage = "selected_obstacle_static_assumption_crossed"
                    raise FastRunnerError(
                        "selected obstacle left the frozen-field static envelope before contact"
                    )
                protected_clearances.append(_protected_clearance(model, forwarded, bundle))
                eef_position_post = np.asarray(
                    forwarded.site_xpos[site_id], dtype=np.float64
                ).copy()
                post_minimum_h = None
                if psf_enabled:
                    from main.poisson_fullbody.robot_samples import evaluate_world_points

                    post_points = evaluate_world_points(
                        bundle.protected_samples.samples, forwarded
                    )
                    post_queries = [bundle.field.query(point) for point in post_points]
                    post_state_field_observation_count += 1
                    post_state_field_query_count += len(post_queries)
                    invalid = [
                        query
                        for query in post_queries
                        if not query.valid or query.value is None
                    ]
                    invalid_field_queries += len(invalid)
                    if invalid:
                        raise FastRunnerError(
                            "post-state PSF field query invalid after physics"
                        )
                    post_h = np.asarray(
                        [float(query.value) for query in post_queries],
                        dtype=np.float64,
                    )
                    post_minimum_h = float(np.min(post_h))
                    nonpositive = int(np.count_nonzero(post_h <= 0.0))
                    nonpositive_post_state_field_query_count += nonpositive
                    if nonpositive:
                        raise FastRunnerError(
                            "post-state protected sample left strict h>0 after physics"
                        )
                physics_rows.append(
                    {
                        "observation_index": observation_index,
                        "post_state_physical_boundary": post_state_physical_boundary,
                        "local_action_index": local_index,
                        "source_action_index": source_index,
                        "inner_control_index": int(inner_index),
                        "physics_substep_index": int(physics_index),
                        "measured_qvel_rad_s": measured.tolist(),
                        "issued_qvel_rad_s": pending["executed"].tolist(),
                        "tracking_error_rad_s": error.tolist(),
                        "protected_link_full_surface_clearance_lower_bound_m": protected_clearances[-1],
                        "post_state_minimum_h_m2": post_minimum_h,
                        "selected_obstacle": static,
                        "eef_position_world_m": eef_position_post.tolist(),
                        "target_error_after_substep_m": float(
                            np.linalg.norm(
                                action_target["target"].target_position
                                - eef_position_post
                            )
                        ),
                        "literal_contact_observed": False,
                    }
                )

            try:
                env.step_grouped_actions_with_substep_callback(
                    provider,
                    callback,
                    expected_inner_updates=5,
                    expected_substeps_per_inner=5,
                    expected_high_level_dt=0.05,
                )
            except _PsfContactObserved:
                contact_terminated_early = True
                break
            post_model, post_data = _raw_model_data(env.sim)
            post_forwarded = clone_forwarded_state(post_model, post_data)
            position_after = np.asarray(
                post_forwarded.site_xpos[site_id], dtype=np.float64
            ).copy()
            target = action_target.get("target")
            start_error = action_target.get("start_error_m")
            if target is None or not isinstance(start_error, float):
                raise FastRunnerError("completed action lacks its fresh adapter target")
            end_error = float(np.linalg.norm(target.target_position - position_after))
            action_progress_rows.append(
                {
                    "local_action_index": local_index,
                    "source_action_index": source_index,
                    "target_error_start_m": start_error,
                    "target_error_after_five_updates_m": end_error,
                    "target_progress_m": start_error - end_error,
                }
            )

        measurement = monitor.result()
        first_link_contact = _first_link_contact_observation(measurement, resolved.link56_geom_ids)
        first_any_contact = _first_any_contact_observation(measurement)
        start_boundary = int(source_start_action) * 25
        first_link_contact_boundary = _first_contact_boundary(
            measurement,
            start_boundary=start_boundary,
            robot_geom_ids=resolved.link56_geom_ids,
        )
        first_any_contact_boundary = _first_contact_boundary(
            measurement,
            start_boundary=start_boundary,
        )
        validity_contact_boundary = first_any_contact_boundary
        precontact_rows = (
            physics_rows
            if validity_contact_boundary is None
            else [
                row
                for row in physics_rows
                if row["post_state_physical_boundary"] < validity_contact_boundary
            ]
        )
        precontact_static_rows = [static_rows[0]] + [
            row["selected_obstacle"]
            for row in precontact_rows
            if row.get("selected_obstacle") is not None
        ]
        precontact_tracking = _tracking(precontact_rows)
        precontact_static_admissible = _static_admissible(
            precontact_static_rows, runtime_protocol
        )
        precontact_field_valid = bool(
            not psf_enabled
            or (
                post_state_field_observation_count == len(precontact_rows)
                and nonpositive_post_state_field_query_count == 0
                and invalid_field_queries == 0
            )
        )
        precontact_tracking_valid = bool(
            precontact_tracking["observation_count"] == 0
            or (
                precontact_tracking["linf_rad_s"]
                <= float(limits["max_joint_velocity_tracking_linf_rad_s"])
                and precontact_tracking["rmse_rad_s"]
                <= float(limits["max_joint_velocity_tracking_rmse_rad_s"])
            )
        )
        precontact_execution_valid = bool(
            not psf_enabled
            or (
                qp_solve_count == len(command_rows)
                and qp_postcheck_count == len(command_rows)
                and joint_limit_postcheck_count == len(command_rows)
                and issued_bound_check_count == len(command_rows)
                and precontact_static_admissible
                and precontact_field_valid
                and precontact_tracking_valid
                and minimum_safe_residual
                >= float(runtime_protocol["qp"]["postcheck_cbf_tolerance"]) * -1.0
            )
        )
        return {
            "arm_name": arm_name,
            "restore": restore,
            "visual_marker_model_sync": marker_sync,
            "fresh_adapter": True,
            "start_official_raw_bytes_sha256": start_official_sha256,
            "source_action_indexes": list(range(source_start_action, source_start_action + len(actions))),
            "filter_update_count": len(command_rows),
            "physics_substep_count": len(physics_rows),
            "exposure_complete": bool(len(command_rows) == 40 and len(physics_rows) == 200),
            "contact_terminated_early": contact_terminated_early,
            "precontact_execution_valid": precontact_execution_valid,
            "qp_solve_count": qp_solve_count,
            "qp_postcheck_count": qp_postcheck_count,
            "joint_limit_postcheck_count": joint_limit_postcheck_count,
            "issued_command_bound_check_count": issued_bound_check_count,
            "all_issued_commands_within_physical_bounds": bool(
                issued_bound_check_count > 0
                and issued_bound_check_count == len(command_rows)
            ),
            "nominal_dynamic_bound_check_count": nominal_dynamic_bound_check_count,
            "nominal_dynamic_bound_violation_count": nominal_dynamic_bound_violation_count,
            "all_nominal_commands_within_dynamic_joint_bounds": bool(
                nominal_dynamic_bound_check_count > 0
                and nominal_dynamic_bound_check_count == len(command_rows)
                and nominal_dynamic_bound_violation_count == 0
            ),
            "protected_sample_count": len(bundle.protected_samples.samples),
            "invalid_field_query_count": invalid_field_queries,
            "post_state_field_observation_count": post_state_field_observation_count,
            "post_state_field_query_count": post_state_field_query_count,
            "nonpositive_post_state_field_query_count": nonpositive_post_state_field_query_count,
            "all_post_state_field_queries_valid_and_positive": bool(
                not psf_enabled
                or (
                    post_state_field_observation_count == 200
                    and post_state_field_query_count
                    == 200 * len(bundle.protected_samples.samples)
                    and nonpositive_post_state_field_query_count == 0
                    and invalid_field_queries == 0
                )
            ),
            "minimum_safe_cbf_residual_m2_per_s": (
                minimum_safe_residual if psf_enabled else None
            ),
            "literal_contact": {
                "link56_present": bool(measurement.link56_obstacle_contact),
                "any_robot_selected_obstacle_present": bool(measurement.any_robot_obstacle_contact),
                "first_link56_observation_index": first_link_contact,
                "first_any_robot_observation_index": first_any_contact,
                "first_link56_physical_boundary": first_link_contact_boundary,
                "first_any_robot_physical_boundary": first_any_contact_boundary,
            },
            "minimum_D_sim_m_diagnostic_only": float(measurement.D_sim_min_m),
            "protected_link_clearance": {
                "available": bool(protected_clearances and all(math.isfinite(value) for value in protected_clearances)),
                "minimum_full_surface_lower_bound_m": float(min(protected_clearances)),
                "semantics": "sample_to_current_selected_obstacle_obb_minus_certified_protected_link_cover_radius",
            },
            "tracking_full_window": _tracking(physics_rows),
            "tracking_precontact": precontact_tracking,
            "static_full_window_admissible": _static_admissible(static_rows, runtime_protocol),
            "static_precontact_admissible": precontact_static_admissible,
            "precontact_field_queries_valid_and_positive": precontact_field_valid,
            "minimum_nominal_cbf_residual_m2_per_s": (
                minimum_nominal_residual if psf_enabled else None
            ),
            "activation_trace": activation_rows,
            "motion": {
                "nominal_command_integral_rad": nominal_command_integral,
                "executed_command_integral_rad": executed_command_integral,
                "safe_to_nominal_command_motion_ratio": (
                    0.0 if nominal_command_integral == 0.0 else executed_command_integral / nominal_command_integral
                ),
                "measured_joint_motion_integral_rad": measured_motion_integral,
                "maximum_filter_correction_norm_rad_s": maximum_correction,
                "maximum_executed_command_norm_rad_s": maximum_safe_command,
                "cumulative_cartesian_target_progress_m": float(
                    sum(row["target_progress_m"] for row in action_progress_rows)
                ),
            },
            "action_target_progress": action_progress_rows,
            "command_trace": command_rows,
            "physics_trace": physics_rows,
            "measurement": measurement.to_dict(),
        }
    except FastArmExecutionFailure:
        raise
    except Exception as error:
        measurement_record = None
        if monitor is not None:
            try:
                measurement_record = monitor.result().to_dict()
            except Exception:
                measurement_record = None
        evidence = {
            "arm_name": arm_name,
            "psf_enabled": psf_enabled,
            "failure_stage": failure_stage,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "filter_update_count": len(command_rows),
            "physics_substep_count": len(physics_rows),
            "qp_solve_count": qp_solve_count,
            "qp_postcheck_count": qp_postcheck_count,
            "joint_limit_postcheck_count": joint_limit_postcheck_count,
            "issued_command_bound_check_count": issued_bound_check_count,
            "nominal_dynamic_bound_check_count": nominal_dynamic_bound_check_count,
            "nominal_dynamic_bound_violation_count": nominal_dynamic_bound_violation_count,
            "invalid_field_query_count": invalid_field_queries,
            "post_state_field_observation_count": post_state_field_observation_count,
            "post_state_field_query_count": post_state_field_query_count,
            "nonpositive_post_state_field_query_count": nonpositive_post_state_field_query_count,
            "last_qp_attempt": last_qp_attempt,
            "command_trace": command_rows,
            "physics_trace": physics_rows,
            "activation_trace": activation_rows,
            "action_target_progress": action_progress_rows,
            "measurement": measurement_record,
        }
        raise FastArmExecutionFailure(
            "%s failed during %s: %s" % (arm_name, failure_stage, error),
            evidence,
        ) from error
    finally:
        if env is not None:
            env.close()


def _pair_exact(adapter: Mapping[str, Any], psf: Mapping[str, Any]) -> bool:
    fields = (
        "model_topology_sha256",
        "physical_model_sha256",
        "compiled_mjb_sha256",
        "official_integration_state_sha256",
        "target_official_integration_state_sha256",
        "settled_state_sha256",
        "target_state_sha256",
        "controller",
        "controller_software_state",
        "pid_memory_reset",
    )
    return bool(
        all(_canonical(adapter["restore"].get(field)) == _canonical(psf["restore"].get(field)) for field in fields)
        and adapter["start_official_raw_bytes_sha256"] == psf["start_official_raw_bytes_sha256"]
        and adapter["fresh_adapter"] is True
        and psf["fresh_adapter"] is True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path("configs/vlsa_poisson_fast_feasibility.v1.json"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"))
    parser.add_argument("--historical-result-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", default="vlsa-t1-goal-ii-t0-e05")
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    os.chdir(str(root))
    for path in (root, root / "main", root / "safelibero"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        output = _new_output(arguments.output_root, arguments.run_id)
    except FastRunnerError as error:
        print("fast feasibility refused: %s" % error, file=sys.stderr)
        return 2
    result_path = output / "result.json"
    started = time.time()
    source_env = None
    provenance: Dict[str, Any] = {}
    assumption_evidence: Dict[str, Any] = {}
    arm_evidence: Dict[str, Any] = {}
    try:
        import numpy as np
        from main.poisson_fullbody.contracts import publish_hashed_json
        from main.poisson_fullbody.fast_feasibility import (
            CASE_ID,
            RESULT_SCHEMA,
            SOURCE_ARM,
            classify_fast_feasibility,
            validate_fast_feasibility_protocol,
        )
        from main.poisson_fullbody.feasibility_protocol import load_feasibility_protocol
        from main.poisson_fullbody.field_bundle import build_static_field_bundle
        from main.poisson_fullbody.measurement import clone_forwarded_state, resolve_collision_geom_sets
        from main.poisson_fullbody.robot_samples import validate_rigid_roundtrip
        from main.poisson_fullbody.shadow_replay import load_historical_action_replay
        from main.poisson_fullbody.surface_sampling import (
            build_robot_collision_samples,
            validate_robot_sample_evidence,
        )
        from scripts.run_poisson_shadow_parity import _check_step, _prepare_environment
        import main.evaluate_safelibero_aegis as evaluator

        protocol_path = arguments.protocol if arguments.protocol.is_absolute() else root / arguments.protocol
        manifest_path = arguments.manifest if arguments.manifest.is_absolute() else root / arguments.manifest
        protocol = _json(protocol_path.resolve(), "fast protocol")
        derived = validate_fast_feasibility_protocol(protocol)
        if arguments.case_id != CASE_ID:
            raise FastRunnerError("fast runner is frozen to %s" % CASE_ID)
        source = _git(root)
        allocation = _allocation()
        case, case_row_sha256 = _manifest_case(manifest_path.resolve(), arguments.case_id)
        if _file_sha256(manifest_path.resolve()) != protocol["selection"]["manifest_file_sha256"]:
            raise FastRunnerError("manifest raw hash differs from fast protocol")
        selection_path = root / protocol["selection"]["relative_path"]
        if _file_sha256(selection_path) != protocol["selection"]["raw_file_sha256"]:
            raise FastRunnerError("selection protocol raw hash differs")
        if case.get("protocol_config_sha256") != protocol["selection"]["raw_file_sha256"]:
            raise FastRunnerError("case-to-selection binding differs")
        runtime_path = root / protocol["runtime"]["relative_path"]
        if _file_sha256(runtime_path) != protocol["runtime"]["raw_file_sha256"]:
            raise FastRunnerError("runtime protocol raw hash differs")
        runtime_protocol, runtime_hashes = load_feasibility_protocol(
            runtime_path,
            expected_protocol_sha256=protocol["runtime"]["semantic_sha256"],
        )
        if runtime_hashes.parameter_block_sha256 != protocol["runtime"]["parameter_block_sha256"]:
            raise FastRunnerError("runtime parameter-block hash differs")
        historical_path = _historical_path(arguments.historical_result_root.resolve(), case)
        replay = load_historical_action_replay(
            historical_path,
            expected_case_id=arguments.case_id,
            expected_arm=SOURCE_ARM,
        )
        historical = protocol["source"]
        if (
            replay.result_file_sha256 != historical["historical_result_file_sha256"]
            or replay.result_payload_sha256 != historical["historical_result_payload_sha256"]
            or replay.executed_sequence_sha256 != historical["historical_executed_action_sequence_sha256"]
            or len(replay.actions) != historical["historical_executed_action_count"]
        ):
            raise FastRunnerError("historical replay differs from frozen fast protocol")
        window_actions = replay.actions[derived["start_action"] : derived["end_action"] + 1]
        window_record = {
            "source_arm": SOURCE_ARM,
            "start_action_index": derived["start_action"],
            "end_action_index_inclusive": derived["end_action"],
            "actions": [list(row) for row in window_actions],
        }
        if _sha256(_canonical(window_record)) != historical["window_action_record_sha256"]:
            raise FastRunnerError("frozen eight-action window hash differs")
        if any(any(float(value) != 0.0 for value in action[3:6]) for action in window_actions):
            raise FastRunnerError("translation-only adapter would discard nonzero rotation")

        provenance = {
            "source": source,
            "allocation": allocation,
            "python": platform.python_version(),
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "case_row_sha256": case_row_sha256,
            "protocol_file_sha256": _file_sha256(protocol_path.resolve()),
            "historical_result_file_sha256": replay.result_file_sha256,
            "historical_result_payload_sha256": replay.result_payload_sha256,
            "historical_executed_action_sequence_sha256": replay.executed_sequence_sha256,
            "online_policy_query_count": 0,
        }
        runtime = evaluator._runtime_imports(include_aegis=False)
        source_env, _, observation, goal_atoms, previous_goal = _prepare_environment(
            evaluator,
            runtime,
            case,
            replay,
        )

        # The user's minimal protocol freezes the field immediately after the
        # 20 settling actions.  The subsequent OSC prefix is measured against
        # this exact geometry; the field is never refreshed at boundary B.
        obstacle_name, _ = evaluator._active_obstacle(source_env, observation)
        if obstacle_name != protocol["case"]["selected_obstacle_name"] or obstacle_name != case["active_obstacle_name"]:
            raise FastRunnerError("selected obstacle identity differs after settling")
        authority = evaluator._contact_model_authority(source_env, obstacle_name)
        robot_root = _body_id(source_env.sim.model, source_env.robots[0].robot_model.root_body)
        link_ids = tuple(_body_id(source_env.sim.model, name) for name in protocol["case"]["protected_robot_body_names"])
        raw_model, raw_data = _raw_model_data(source_env.sim)
        resolved = resolve_collision_geom_sets(
            raw_model,
            robot_root_body_ids=(robot_root,),
            obstacle_root_body_ids=(int(authority["active_obstacle_root_body_id"]),),
            link56_body_ids=link_ids,
        )
        settled_official_before_field = _official_state(source_env.sim)
        bundle = build_static_field_bundle(
            raw_model,
            raw_data,
            resolved=resolved,
            protocol=runtime_protocol,
            protocol_hashes=runtime_hashes,
        )
        if not np.array_equal(settled_official_before_field, _official_state(source_env.sim)):
            raise FastRunnerError("settled field construction changed simulator state")
        forwarded = clone_forwarded_state(raw_model, raw_data)
        full_samples = build_robot_collision_samples(
            source_env.sim.model,
            forwarded,
            geom_ids=resolved.robot_geom_ids,
            epsilon_m=float(runtime_protocol["coverage"]["epsilon_m"]),
        )
        roundtrip = validate_rigid_roundtrip(full_samples.samples, forwarded)
        evidence = {
            "sample_count": len(full_samples.samples),
            "sample_ledger_sha256": full_samples.sample_ledger_sha256,
            "geom_records": list(full_samples.geom_records),
            "epsilon_m": full_samples.epsilon_m,
            "maximum_surface_cover_radius_m": full_samples.maximum_surface_cover_radius_m,
            "coverage_semantics": full_samples.coverage_semantics,
            "roundtrip": roundtrip,
        }
        validate_robot_sample_evidence(
            evidence,
            resolved_geom_ids=resolved.robot_geom_ids,
            resolved_geom_names=resolved.robot_geom_names,
            resolved_body_ids=resolved.robot_body_ids,
            roundtrip_field="roundtrip",
        )
        settled_obstacle_state = _obstacle_state(raw_model, forwarded, bundle, resolved.obstacle_body_ids)
        if not _static_admissible([settled_obstacle_state], runtime_protocol):
            assumption_evidence = {
                "field_construction_boundary": 0,
                "field_construction_state": "after_20_settling_actions",
                "first_invalid_prefix_action_index": None,
                "first_invalid_obstacle_state": settled_obstacle_state,
                "admissible": False,
            }
            raise FastAssumptionInvalid(
                "selected obstacle is not static at field construction"
            )
        prefix_obstacle_rows: List[Dict[str, Any]] = []
        for expected_step in replay.steps[: derived["start_action"]]:
            observation, reward, done, _ = source_env.step(expected_step.action)
            _, _, previous_goal = _check_step(
                evaluator=evaluator,
                env=source_env,
                observation=observation,
                reward=reward,
                done=done,
                expected_step=expected_step,
                goal_atoms=goal_atoms,
                previous_goal_values=previous_goal,
                np=np,
            )
            evaluator._update_eef_marker(source_env, evaluator._eef_proxy(runtime, observation))
            prefix_forwarded = clone_forwarded_state(raw_model, raw_data)
            prefix_obstacle_rows.append(
                {
                    "source_action_index": int(expected_step.step),
                    **_obstacle_state(
                        raw_model,
                        prefix_forwarded,
                        bundle,
                        resolved.obstacle_body_ids,
                    ),
                }
            )
            if not _static_admissible(
                [settled_obstacle_state] + prefix_obstacle_rows,
                runtime_protocol,
            ):
                assumption_evidence = {
                    "field_construction_boundary": 0,
                    "field_construction_state": "after_20_settling_actions",
                    "first_invalid_prefix_action_index": int(expected_step.step),
                    "first_invalid_obstacle_state": prefix_obstacle_rows[-1],
                    "validated_prefix_action_boundary_count": len(prefix_obstacle_rows),
                    "admissible": False,
                }
                raise FastAssumptionInvalid(
                    "settled static field became inadmissible at OSC prefix action %d"
                    % int(expected_step.step)
                )
        flattened_B = np.asarray(source_env.sim.get_state().flatten(), dtype=np.float64).copy()
        observed_B_hash = evaluator.array_sha256(flattened_B)
        expected_B_hash = replay.steps[derived["start_action"] - 1].simulator_state_sha256
        if not (
            observed_B_hash == expected_B_hash == historical["post_action_179_flattened_state_sha256"]
        ):
            raise FastRunnerError("post-action-179 boundary state differs before official snapshot")
        official_B = _official_state(source_env.sim)
        B_forwarded = clone_forwarded_state(raw_model, raw_data)
        B_obstacle_state = _obstacle_state(raw_model, B_forwarded, bundle, resolved.obstacle_body_ids)
        drift_rows = [settled_obstacle_state] + prefix_obstacle_rows
        limits = runtime_protocol["admissibility"]
        prefix_drift_admissible = bool(
            max(row["translation_drift_m"] for row in drift_rows)
            <= float(limits["max_selected_geom_translation_drift_m"])
            and max(row["rotation_drift_rad"] for row in drift_rows)
            <= float(limits["max_selected_geom_rotation_drift_rad"])
            and max(row["surface_drift_m"] for row in drift_rows)
            <= float(limits["max_selected_geom_surface_drift_m"])
        )
        prefix_speed_admissible = bool(
            max(row["maximum_body_linear_speed_m_s"] for row in drift_rows)
            <= float(limits["max_selected_body_linear_speed_m_s"])
            and max(row["maximum_body_angular_speed_rad_s"] for row in drift_rows)
            <= float(limits["max_selected_body_angular_speed_rad_s"])
        )
        assumption_evidence = {
            "field_construction_boundary": 0,
            "field_construction_state": "after_20_settling_actions",
            "prefix_action_boundary_count": len(prefix_obstacle_rows),
            "prefix_action_boundary_indexes": [row["source_action_index"] for row in prefix_obstacle_rows],
            "maximum_prefix_translation_drift_m": max(row["translation_drift_m"] for row in drift_rows),
            "maximum_prefix_rotation_drift_rad": max(row["rotation_drift_rad"] for row in drift_rows),
            "maximum_prefix_surface_drift_m": max(row["surface_drift_m"] for row in drift_rows),
            "settled_obstacle_state": settled_obstacle_state,
            "boundary_B_obstacle_state": B_obstacle_state,
            "prefix_pose_surface_admissible": prefix_drift_admissible,
            "prefix_action_boundary_speed_admissible": prefix_speed_admissible,
            "admissible": bool(prefix_drift_admissible and prefix_speed_admissible),
            "measurement_cadence": "settled_and_every_completed_OSC_action_0_through_179_including_boundary_B",
        }
        if not assumption_evidence["admissible"]:
            raise FastAssumptionInvalid(
                "settled static field is inadmissible over the exact OSC prefix"
            )

        adapter = _run_arm(
            arm_name="joint_velocity_adapter_only",
            evaluator=evaluator,
            runtime=runtime,
            case=case,
            source_env=source_env,
            state_at_B=flattened_B,
            actions=window_actions,
            source_start_action=derived["start_action"],
            bundle=bundle,
            resolved=resolved,
            full_samples=full_samples,
            runtime_protocol=runtime_protocol,
            nominal_activation_threshold_m2_per_s=derived["thresholds"][
                "maximum_nominal_cbf_residual_for_activation_m2_per_s"
            ],
        )
        arm_evidence["adapter_only"] = adapter
        psf = _run_arm(
            arm_name="joint_velocity_adapter_plus_link56_psf",
            evaluator=evaluator,
            runtime=runtime,
            case=case,
            source_env=source_env,
            state_at_B=flattened_B,
            actions=window_actions,
            source_start_action=derived["start_action"],
            bundle=bundle,
            resolved=resolved,
            full_samples=full_samples,
            runtime_protocol=runtime_protocol,
            nominal_activation_threshold_m2_per_s=derived["thresholds"][
                "maximum_nominal_cbf_residual_for_activation_m2_per_s"
            ],
        )
        arm_evidence["adapter_plus_psf"] = psf
        exact_pair = _pair_exact(adapter, psf)
        adapter_tracking = adapter["tracking_precontact"]
        psf_tracking = psf["tracking_full_window"]
        psf_precontact_tracking = psf["tracking_precontact"]
        adapter_contact_physical_boundary = adapter["literal_contact"][
            "first_link56_physical_boundary"
        ]
        adapter_contact_boundary = (
            derived["end_boundary"] + 1
            if adapter_contact_physical_boundary is None
            else int(adapter_contact_physical_boundary)
        )
        precontact_commands = [
            row
            for row in psf["command_trace"]
            if row["physical_boundary"] < adapter_contact_boundary
            and row["nominal_cbf_residual_minimum_m2_per_s"] is not None
        ]
        material_activation_rows = [
            row
            for row in psf["activation_trace"]
            if row["physical_boundary"] < adapter_contact_boundary
            and row["filter_correction_l2_rad_s"]
            >= derived["thresholds"]["minimum_filter_correction_norm_rad_s"]
            and row["nominal_within_dynamic_joint_bounds"] is True
        ]
        minimum_precontact_nominal_residual = min(
            row["nominal_cbf_residual_minimum_m2_per_s"]
            for row in precontact_commands
        )
        maximum_activation_correction = max(
            (row["filter_correction_l2_rad_s"] for row in material_activation_rows),
            default=0.0,
        )
        first_activation_boundary = (
            material_activation_rows[0]["physical_boundary"]
            if material_activation_rows
            else derived["end_boundary"] + 1
        )
        active_keys = [
            (row["source_action_index"], row["inner_control_index"])
            for row in material_activation_rows
        ]
        active_motion = _active_interval_motion(
            psf["command_trace"], psf["physics_trace"], active_keys
        )
        metrics = {
            "exact_paired_start": exact_pair,
            "adapter_exposure_complete": adapter["exposure_complete"],
            "psf_exposure_complete": psf["exposure_complete"],
            "adapter_precontact_static_obstacle_admissible": adapter["static_precontact_admissible"],
            "psf_static_obstacle_admissible": psf["static_full_window_admissible"],
            "psf_all_qp_solved": psf["qp_solve_count"] == derived["expected_updates"],
            "psf_all_qp_postchecks_passed": psf["qp_postcheck_count"] == derived["expected_updates"],
            "psf_all_joint_limit_postchecks_passed": psf["joint_limit_postcheck_count"] == derived["expected_updates"],
            "psf_all_post_state_field_queries_valid_and_positive": psf["all_post_state_field_queries_valid_and_positive"],
            "psf_precontact_execution_valid": psf["precontact_execution_valid"],
            "psf_precontact_static_obstacle_admissible": psf["static_precontact_admissible"],
            "psf_precontact_field_queries_valid_and_positive": psf["precontact_field_queries_valid_and_positive"],
            "psf_nominal_cbf_activation_with_material_correction_before_adapter_contact": bool(material_activation_rows),
            "psf_active_interval_motion_complete": bool(
                active_motion["active_update_count"] > 0
                and active_motion["complete_active_interval_count"]
                == active_motion["active_update_count"]
            ),
            "all_issued_commands_within_physical_bounds": bool(
                adapter["all_issued_commands_within_physical_bounds"]
                and psf["all_issued_commands_within_physical_bounds"]
            ),
            "adapter_all_nominal_commands_within_dynamic_joint_bounds": adapter[
                "all_nominal_commands_within_dynamic_joint_bounds"
            ],
            "psf_all_nominal_commands_within_dynamic_joint_bounds": psf[
                "all_nominal_commands_within_dynamic_joint_bounds"
            ],
            "adapter_link56_contact_present": adapter["literal_contact"]["link56_present"],
            "adapter_first_selected_obstacle_contact_is_link56": bool(
                adapter["literal_contact"]["first_link56_physical_boundary"]
                is not None
                and adapter["literal_contact"]["first_link56_physical_boundary"]
                == adapter["literal_contact"]["first_any_robot_physical_boundary"]
            ),
            "psf_link56_contact_present": psf["literal_contact"]["link56_present"],
            "psf_any_robot_selected_obstacle_contact_present": psf["literal_contact"]["any_robot_selected_obstacle_present"],
            "psf_protected_link_clearance_lower_bound_available": psf["protected_link_clearance"]["available"],
            "adapter_filter_update_count": adapter["filter_update_count"],
            "psf_filter_update_count": psf["filter_update_count"],
            "adapter_physics_substep_count": adapter["physics_substep_count"],
            "psf_physics_substep_count": psf["physics_substep_count"],
            "psf_qp_solve_count": psf["qp_solve_count"],
            "psf_qp_postcheck_count": psf["qp_postcheck_count"],
            "psf_joint_limit_postcheck_count": psf["joint_limit_postcheck_count"],
            "adapter_issued_command_bound_check_count": adapter["issued_command_bound_check_count"],
            "psf_issued_command_bound_check_count": psf["issued_command_bound_check_count"],
            "adapter_precontact_tracking_observation_count": adapter_tracking["observation_count"],
            "psf_tracking_observation_count": psf_tracking["observation_count"],
            "psf_post_state_field_observation_count": psf["post_state_field_observation_count"],
            "psf_post_state_field_query_count": psf["post_state_field_query_count"],
            "psf_nonpositive_post_state_field_query_count": psf["nonpositive_post_state_field_query_count"],
            "psf_precontact_tracking_observation_count": psf_precontact_tracking["observation_count"],
            "psf_activation_update_count_before_adapter_contact": len(material_activation_rows),
            "adapter_first_link56_contact_physical_boundary": adapter_contact_boundary,
            "psf_first_activation_physical_boundary": first_activation_boundary,
            "psf_invalid_field_query_count": psf["invalid_field_query_count"],
            "adapter_nominal_dynamic_bound_check_count": adapter[
                "nominal_dynamic_bound_check_count"
            ],
            "psf_nominal_dynamic_bound_check_count": psf[
                "nominal_dynamic_bound_check_count"
            ],
            "adapter_nominal_dynamic_bound_violation_count": adapter[
                "nominal_dynamic_bound_violation_count"
            ],
            "psf_nominal_dynamic_bound_violation_count": psf[
                "nominal_dynamic_bound_violation_count"
            ],
            "psf_protected_sample_count": psf["protected_sample_count"],
            "psf_minimum_D_sim_m": psf["minimum_D_sim_m_diagnostic_only"],
            "psf_minimum_safe_cbf_residual_m2_per_s": psf["minimum_safe_cbf_residual_m2_per_s"],
            "adapter_precontact_tracking_linf_rad_s": adapter_tracking["linf_rad_s"],
            "adapter_precontact_tracking_rmse_rad_s": adapter_tracking["rmse_rad_s"],
            "psf_tracking_linf_rad_s": psf_tracking["linf_rad_s"],
            "psf_tracking_rmse_rad_s": psf_tracking["rmse_rad_s"],
            "psf_precontact_tracking_linf_rad_s": psf_precontact_tracking["linf_rad_s"],
            "psf_precontact_tracking_rmse_rad_s": psf_precontact_tracking["rmse_rad_s"],
            "maximum_active_filter_correction_norm_rad_s": active_motion[
                "maximum_filter_correction_norm_rad_s"
            ],
            "maximum_active_safe_command_norm_rad_s": active_motion[
                "maximum_executed_command_norm_rad_s"
            ],
            "active_safe_to_nominal_command_motion_ratio": active_motion[
                "safe_to_nominal_command_motion_ratio"
            ],
            "active_safe_measured_joint_motion_integral_rad": active_motion[
                "measured_joint_motion_integral_rad"
            ],
            "active_cartesian_path_length_m": active_motion[
                "cartesian_path_length_m"
            ],
            "psf_protected_link_full_surface_clearance_lower_bound_m": psf["protected_link_clearance"]["minimum_full_surface_lower_bound_m"],
            "psf_minimum_nominal_cbf_residual_before_adapter_contact_m2_per_s": minimum_precontact_nominal_residual,
            "psf_maximum_activation_correction_norm_before_adapter_contact_rad_s": maximum_activation_correction,
        }
        classification = classify_fast_feasibility(metrics, protocol)
        candidate = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "provenance": provenance,
            "window": {
                "start_boundary": derived["start_boundary"],
                "end_boundary": derived["end_boundary"],
                "source_action_indexes": list(range(derived["start_action"], derived["end_action"] + 1)),
                "post_action_179_flattened_state_sha256": observed_B_hash,
                "official_boundary_B_raw_bytes_sha256": _sha256(official_B.tobytes()),
                "field_construction_boundary": 0,
                "settled_to_B_static_field_assumption": assumption_evidence,
            },
            "field": {
                "bundle_hashes": asdict(bundle.hashes),
                "diagnostics": asdict(bundle.diagnostics),
                "resolved_geometry": resolved.to_dict(),
                "full_robot_sampling": evidence,
            },
            "metrics": metrics,
            "activation_evidence": {
                "threshold_m2_per_s": derived["thresholds"]["maximum_nominal_cbf_residual_for_activation_m2_per_s"],
                "adapter_contact_physical_boundary": adapter_contact_boundary,
                "first_material_activation": material_activation_rows[0] if material_activation_rows else None,
                "material_activation_count": len(material_activation_rows),
                "active_interval_motion": active_motion,
            },
            "classification": classification,
            "arms": arm_evidence,
            "claim_scope": "one_post_hoc_0.4_second_window_not_task_success_not_full_episode_not_population_safety",
            "partial_output_interpreted": False,
            "timing": {
                "started_unix": started,
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            },
        }
        publish_hashed_json(result_path, candidate)
        print(json.dumps({"status": "complete", "outcome": classification["primary_outcome"], "result": str(result_path)}, sort_keys=True))
        return 0
    except Exception as error:
        try:
            from main.poisson_fullbody.contracts import publish_hashed_json
            from main.poisson_fullbody.fast_feasibility import RESULT_SCHEMA

            assumption_invalid = isinstance(error, FastAssumptionInvalid)
            if isinstance(error, FastArmExecutionFailure):
                arm_evidence[error.evidence.get("arm_name", "failed_arm")] = error.evidence
            failure = {
                "schema_version": RESULT_SCHEMA,
                "status": "apparatus_failure",
                "run_id": arguments.run_id,
                "case_id": arguments.case_id,
                "primary_outcome": "APPARATUS_FAILURE",
                "safety_mechanism": "NOT_APPLICABLE",
                "partial_output_interpreted": False,
                "provenance": provenance,
                "static_field_assumption": assumption_evidence,
                "arms": arm_evidence,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                    "static_assumption_invalid": assumption_invalid,
                },
                "timing": {
                    "started_unix": started,
                    "finished_unix": time.time(),
                    "elapsed_seconds": time.time() - started,
                },
            }
            publish_hashed_json(result_path, failure)
        except Exception as publication_error:
            print("failure artifact publication failed: %s" % publication_error, file=sys.stderr)
        print("fast feasibility failed: %s" % error, file=sys.stderr)
        return 1
    finally:
        if source_env is not None:
            source_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
