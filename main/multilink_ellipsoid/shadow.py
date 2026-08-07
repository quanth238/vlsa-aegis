"""Read-only multi-link ellipsoid QP observer for an AEGIS rollout."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any, Mapping, Sequence

from .barrier import build_pair_constraint
from .geometry import Ellipsoid, primitive_bounding_radii
from .qp import MultiConstraintQp


SHADOW_SCHEMA = "vlsa_multilink_ellipsoid_shadow.v1"
STEP_SCHEMA = "vlsa_multilink_ellipsoid_shadow_step.v1"
_BODY_PATTERN = re.compile(r"^robot0_link[1-7]$")
_GEOM_KIND_FALLBACK = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
}


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("the ellipsoid shadow observer requires NumPy") from error
    return np


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


def load_shadow_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the preregistered shadow configuration."""

    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("multi-link ellipsoid shadow config is invalid JSON") from error
    if not isinstance(config, dict) or config.get("schema_version") != SHADOW_SCHEMA:
        raise ValueError("multi-link ellipsoid shadow config schema differs")
    required = {
        "schema_version",
        "protocol_id",
        "case_ids",
        "control_effect",
        "protected_body_names",
        "obstacle_geometry",
        "optimizer",
        "simulator_verification",
        "claim_scope",
    }
    if set(config) != required:
        raise ValueError("multi-link ellipsoid shadow config keys differ")
    if config["control_effect"] != "read_only_no_executed_action_change":
        raise ValueError("shadow configuration must be read-only")
    bodies = config["protected_body_names"]
    expected = ["robot0_link%d" % index for index in range(1, 8)]
    if bodies != expected or any(_BODY_PATTERN.fullmatch(name) is None for name in bodies):
        raise ValueError("protected bodies must be the ordered Panda link1-link7 set")
    if config["obstacle_geometry"] != "released_aegis_frozen_perception_mvee":
        raise ValueError("shadow obstacle geometry must remain the released AEGIS MVEE")
    case_ids = config["case_ids"]
    if not isinstance(case_ids, list) or not case_ids or any(
        not isinstance(value, str) or not value for value in case_ids
    ):
        raise ValueError("shadow case_ids must be a nonempty string list")
    optimizer = config["optimizer"]
    optimizer_keys = {
        "alpha_s_inv",
        "optimizer_clearance_m",
        "joint_velocity_limit_rad_s",
        "resolved_rate_damping",
        "eef_preservation_weight",
        "eef_weight_diagonal",
        "eps_abs",
        "eps_rel",
        "max_iter",
        "residual_tolerance",
        "bound_tolerance_rad_s",
    }
    if not isinstance(optimizer, dict) or set(optimizer) != optimizer_keys:
        raise ValueError("shadow optimizer keys differ")
    positive = (
        "alpha_s_inv",
        "joint_velocity_limit_rad_s",
        "resolved_rate_damping",
        "eef_preservation_weight",
        "eps_abs",
        "eps_rel",
        "residual_tolerance",
        "bound_tolerance_rad_s",
    )
    for key in positive:
        value = optimizer[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("optimizer.%s must be finite and positive" % key)
    clearance = optimizer["optimizer_clearance_m"]
    if isinstance(clearance, bool) or not math.isfinite(float(clearance)) or float(clearance) < 0.0:
        raise ValueError("optimizer.optimizer_clearance_m must be nonnegative")
    weights = optimizer["eef_weight_diagonal"]
    if not isinstance(weights, list) or len(weights) != 6 or any(
        isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0
        for value in weights
    ):
        raise ValueError("optimizer.eef_weight_diagonal must contain six positive values")
    if isinstance(optimizer["max_iter"], bool) or not isinstance(optimizer["max_iter"], int):
        raise ValueError("optimizer.max_iter must be an integer")
    verification = config["simulator_verification"]
    if not isinstance(verification, dict) or verification != {
        "D_sim": "raw_mujoco_nonpositive_contact_distance_and_active_obstacle_displacement",
        "distinct_from_D_opt": True,
    }:
        raise ValueError("simulator verification must keep D_sim distinct from D_opt")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def allocation_record() -> dict[str, Any]:
    """Return the allocation/device identity for a claim-bearing shadow run."""

    import os

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise ValueError("multi-link SafeLIBERO shadow execution requires Slurm")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=30,
    ).strip().splitlines()
    if len(output) != 1:
        raise ValueError("multi-link shadow requires exactly one visible GPU")
    parts = [part.strip() for part in output[0].split(",")]
    if len(parts) != 3 or "H100" not in parts[0]:
        raise ValueError("multi-link SafeLIBERO shadow requires one H100")
    return {
        "slurm_job_id": job_id,
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": socket.gethostname(),
        "device": {
            "name": parts[0],
            "uuid": parts[1],
            "driver_version": parts[2],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def _raw_model_data(sim: Any) -> tuple[Any, Any]:
    return (
        getattr(sim.model, "_model", sim.model),
        getattr(sim.data, "_data", sim.data),
    )


def _name(model: Any, kind: str, identifier: int) -> str | None:
    wrapper = getattr(model, "_model", model)
    method = getattr(model, "%s_id2name" % kind, None)
    if callable(method):
        try:
            value = method(int(identifier))
            return None if value is None else str(value)
        except Exception:
            pass
    try:
        import mujoco

        enum = mujoco.mjtObj.mjOBJ_BODY if kind == "body" else mujoco.mjtObj.mjOBJ_GEOM
        value = mujoco.mj_id2name(wrapper, enum, int(identifier))
        return None if value is None else str(value)
    except Exception:
        return None


def _geom_kind(raw_type: int) -> str:
    try:
        import mujoco

        for name in ("PLANE", "HFIELD", "SPHERE", "CAPSULE", "ELLIPSOID", "CYLINDER", "BOX", "MESH"):
            if int(getattr(mujoco.mjtGeom, "mjGEOM_%s" % name)) == int(raw_type):
                return name.lower()
    except Exception:
        pass
    return _GEOM_KIND_FALLBACK.get(int(raw_type), "unknown_%d" % int(raw_type))


def _arm_dof_indices(env: Any) -> tuple[int, ...]:
    indexes = getattr(env.robots[0], "_ref_joint_vel_indexes", None)
    if indexes is None or len(indexes) != 7:
        raise ValueError("Panda arm velocity indexes are unavailable")
    result = tuple(int(value) for value in indexes)
    if len(set(result)) != 7 or any(value < 0 for value in result):
        raise ValueError("Panda arm velocity indexes are invalid")
    return result


def _eef_site_id(env: Any) -> int:
    value = getattr(env.robots[0], "eef_site_id", None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    sites = getattr(getattr(env.robots[0], "gripper", None), "important_sites", None)
    if isinstance(sites, Mapping):
        name = sites.get("grip_site")
        if isinstance(name, str) and name:
            return int(env.sim.model.site_name2id(name))
    raise ValueError("authoritative Panda grip site is unavailable")


def _eef_jacobian(env: Any, arm_dofs: Sequence[int]) -> Any:
    import mujoco

    np = _numpy()
    model, data = _raw_model_data(env.sim)
    jac_position = np.zeros((3, int(model.nv)), dtype=np.float64)
    jac_rotation = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jacSite(
        model,
        data,
        jac_position,
        jac_rotation,
        _eef_site_id(env),
    )
    indexes = np.asarray(arm_dofs, dtype=np.int64)
    result = np.vstack((jac_position[:, indexes], jac_rotation[:, indexes]))
    if result.shape != (6, 7) or not np.all(np.isfinite(result)):
        raise ValueError("Panda grip-site Jacobian is invalid")
    return result


def _geom_jacobians(env: Any, ellipsoid: Ellipsoid, arm_dofs: Sequence[int]) -> tuple[Any, Any]:
    import mujoco

    np = _numpy()
    model, data = _raw_model_data(env.sim)
    jac_position = np.zeros((3, int(model.nv)), dtype=np.float64)
    jac_rotation = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jac(
        model,
        data,
        jac_position,
        jac_rotation,
        ellipsoid.center,
        int(ellipsoid.body_id),
    )
    indexes = np.asarray(arm_dofs, dtype=np.int64)
    return jac_position[:, indexes], jac_rotation[:, indexes]


def _link_ellipsoids(env: Any, protected_names: Sequence[str]) -> list[Ellipsoid]:
    np = _numpy()
    model, data = _raw_model_data(env.sim)
    protected = set(protected_names)
    output: list[Ellipsoid] = []
    observed_bodies: set[str] = set()
    geom_bodyid = np.asarray(model.geom_bodyid, dtype=np.int64)
    geom_contype = np.asarray(model.geom_contype, dtype=np.int64)
    geom_conaffinity = np.asarray(model.geom_conaffinity, dtype=np.int64)
    for geom_id in range(int(model.ngeom)):
        body_id = int(geom_bodyid[geom_id])
        body_name = _name(env.sim.model, "body", body_id)
        if body_name not in protected:
            continue
        if int(geom_contype[geom_id]) == 0 and int(geom_conaffinity[geom_id]) == 0:
            continue
        geom_name = _name(env.sim.model, "geom", geom_id) or "unnamed_geom_%d" % geom_id
        kind = _geom_kind(int(model.geom_type[geom_id]))
        rbound = float(model.geom_rbound[geom_id])
        semiaxes, source = primitive_bounding_radii(
            kind,
            np.asarray(model.geom_size[geom_id], dtype=np.float64),
            rbound,
        )
        output.append(
            Ellipsoid(
                center=np.asarray(data.geom_xpos[geom_id], dtype=np.float64),
                rotation=np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3),
                semiaxes_m=semiaxes,
                body_id=body_id,
                body_name=body_name,
                geom_id=geom_id,
                geom_name=geom_name,
                bound_source=source,
                source_rbound_m=rbound,
            )
        )
        observed_bodies.add(body_name)
    missing = [name for name in protected_names if name not in observed_bodies]
    if missing:
        raise ValueError("protected robot bodies lack collision ellipsoids: %s" % missing)
    if not output:
        raise ValueError("no protected robot collision ellipsoids were constructed")
    return output


def _resolved_rate_nominal(
    executed_action: Sequence[float],
    eef_jacobian: Any,
    *,
    damping: float,
    joint_velocity_limit: float,
) -> tuple[Any, dict[str, Any]]:
    np = _numpy()
    action = np.asarray(executed_action, dtype=np.float64)
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise ValueError("executed AEGIS action must be a finite length-7 vector")
    # SafeLIBERO OSC clips normalized XYZ and maps it to a 0.05 m target over
    # one 0.05 s high-level action.  The implied translational twist therefore
    # equals the clipped XYZ in m/s.  The registered arm zeros rotation.
    desired_twist = np.concatenate((np.clip(action[:3], -1.0, 1.0), np.zeros(3)))
    regularized = eef_jacobian @ eef_jacobian.T + damping * damping * np.eye(6)
    raw = eef_jacobian.T @ np.linalg.solve(regularized, desired_twist)
    clipped = np.clip(raw, -joint_velocity_limit, joint_velocity_limit)
    return clipped, {
        "source": "resolved_rate_estimate_from_executed_aegis_translational_osc_action",
        "desired_twist": desired_twist.tolist(),
        "qdot_unclipped_rad_s": raw.tolist(),
        "qdot_nominal_rad_s": clipped.tolist(),
        "saturated_joint_count": int(np.count_nonzero(raw != clipped)),
    }


class MultilinkEllipsoidShadow:
    """Evaluate, but never execute, the whole-arm multi-constraint QP."""

    def __init__(self, config: Mapping[str, Any], obstacle: Ellipsoid) -> None:
        if config.get("schema_version") != SHADOW_SCHEMA:
            raise ValueError("shadow configuration was not validated")
        self.config = dict(config)
        self.obstacle = obstacle
        optimizer = config["optimizer"]
        self.qp = MultiConstraintQp(
            eps_abs=float(optimizer["eps_abs"]),
            eps_rel=float(optimizer["eps_rel"]),
            max_iter=int(optimizer["max_iter"]),
            residual_tolerance=float(optimizer["residual_tolerance"]),
            bound_tolerance=float(optimizer["bound_tolerance_rad_s"]),
        )

    @classmethod
    def from_aegis_geometry(
        cls,
        config: Mapping[str, Any],
        geometry: Mapping[str, Any],
    ) -> "MultilinkEllipsoidShadow":
        np = _numpy()
        obstacle = Ellipsoid(
            center=np.asarray(geometry["p2"], dtype=np.float64),
            rotation=np.asarray(geometry["R2"], dtype=np.float64),
            semiaxes_m=np.asarray(geometry["Q2_diag"], dtype=np.float64),
            body_name=str(geometry.get("record", {}).get("label", "active_obstacle")),
            geom_name="released_aegis_perception_mvee",
            bound_source="released_aegis_frozen_perception_mvee",
        )
        return cls(config, obstacle)

    def geometry_record(self, env: Any) -> dict[str, Any]:
        links = _link_ellipsoids(env, self.config["protected_body_names"])
        return {
            "obstacle": self.obstacle.to_record(),
            "link_ellipsoid_count": len(links),
            "link_ellipsoids": [item.to_record() for item in links],
            "coverage_semantics": (
                "one_certified_enclosing_ellipsoid_per_contact_participating_"
                "mujoco_collision_geom_on_robot0_link1_through_robot0_link7"
            ),
        }

    def evaluate(self, env: Any, executed_action: Sequence[float], *, step: int) -> dict[str, Any]:
        np = _numpy()
        total_started = time.perf_counter_ns()
        optimizer = self.config["optimizer"]
        arm_dofs = _arm_dof_indices(env)
        eef_jacobian = _eef_jacobian(env, arm_dofs)
        nominal, nominal_record = _resolved_rate_nominal(
            executed_action,
            eef_jacobian,
            damping=float(optimizer["resolved_rate_damping"]),
            joint_velocity_limit=float(optimizer["joint_velocity_limit_rad_s"]),
        )
        geometry_started = time.perf_counter_ns()
        links = _link_ellipsoids(env, self.config["protected_body_names"])
        constraints = []
        for link in links:
            jac_position, jac_rotation = _geom_jacobians(env, link, arm_dofs)
            constraints.append(
                build_pair_constraint(
                    link,
                    self.obstacle,
                    jac_position,
                    jac_rotation,
                    alpha=float(optimizer["alpha_s_inv"]),
                    optimizer_clearance_m=float(optimizer["optimizer_clearance_m"]),
                )
            )
        geometry_finished = time.perf_counter_ns()
        rows = np.stack([item.row for item in constraints], axis=0)
        lower = np.asarray([item.lower for item in constraints], dtype=np.float64)
        weights = np.diag(np.asarray(optimizer["eef_weight_diagonal"], dtype=np.float64))
        metric = np.eye(7) + float(optimizer["eef_preservation_weight"]) * (
            eef_jacobian.T @ weights @ eef_jacobian
        )
        limit = float(optimizer["joint_velocity_limit_rad_s"])
        result = self.qp.solve(
            nominal,
            metric,
            rows,
            lower,
            -limit * np.ones(7),
            limit * np.ones(7),
        )
        closest_index = int(np.argmin([item.h_opt_m for item in constraints]))
        violated = [
            index
            for index, item in enumerate(constraints)
            if float(item.row @ nominal - item.lower) < 0.0
        ]
        record = {
            "schema_version": STEP_SCHEMA,
            "step": int(step),
            "control_effect": "read_only_no_executed_action_change",
            "arm_dof_indices": list(arm_dofs),
            "nominal": nominal_record,
            "constraint_count": len(constraints),
            "constraints": [item.to_record() for item in constraints],
            "closest_constraint_index": closest_index,
            "closest_body_name": constraints[closest_index].body_name,
            "minimum_h_opt_m": float(constraints[closest_index].h_opt_m),
            "nominal_violated_constraint_indexes": violated,
            "nominal_violated_constraint_count": len(violated),
            "qp": {
                "valid": bool(result.valid),
                "reason": result.reason,
                "qdot_safe_rad_s": (
                    None if result.qdot_safe is None else result.qdot_safe.tolist()
                ),
                "diagnostics": dict(result.diagnostics),
            },
            "D_opt": {
                "value_m": float(optimizer["optimizer_clearance_m"]),
                "semantics": "optimizer_support_gap_buffer",
            },
            "D_sim": {
                "available": False,
                "value": {
                    "minimum_robot_contact_distance_m": None,
                    "active_obstacle_l1_displacement_m": None,
                    "robot_contact_count": None,
                },
                "semantics": self.config["simulator_verification"]["D_sim"],
                "source": "post_step_raw_simulator_evidence_not_optimizer_geometry",
            },
            "timing": {
                "geometry_and_jacobian_wall_seconds": (
                    geometry_finished - geometry_started
                )
                * 1.0e-9,
                "total_shadow_wall_seconds": (
                    time.perf_counter_ns() - total_started
                )
                * 1.0e-9,
            },
        }
        return record


def summarize_shadow_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    if not records:
        return {"status": "no_records", "step_count": 0}
    qp_total = np.asarray(
        [item["qp"]["diagnostics"].get("timing", {}).get("total_wall_seconds") for item in records],
        dtype=np.float64,
    )
    qp_solve = np.asarray(
        [item["qp"]["diagnostics"].get("timing", {}).get("solve_wall_seconds") for item in records],
        dtype=np.float64,
    )
    shadow_total = np.asarray(
        [item["timing"]["total_shadow_wall_seconds"] for item in records],
        dtype=np.float64,
    )
    if not (
        np.all(np.isfinite(qp_total))
        and np.all(np.isfinite(qp_solve))
        and np.all(np.isfinite(shadow_total))
    ):
        raise ValueError("shadow timing contains missing or nonfinite values")

    def stats(values: Any) -> dict[str, float]:
        return {
            "mean_seconds": float(np.mean(values)),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.quantile(values, 0.95)),
            "maximum_seconds": float(np.max(values)),
        }

    first_violation = next(
        (
            {
                "step": int(item["step"]),
                "body_names": sorted(
                    {
                        item["constraints"][index]["body_name"]
                        for index in item["nominal_violated_constraint_indexes"]
                    }
                ),
            }
            for item in records
            if item["nominal_violated_constraint_count"] > 0
        ),
        None,
    )
    return {
        "status": "complete" if all(item["qp"]["valid"] for item in records) else "qp_failures_present",
        "step_count": len(records),
        "all_qps_valid": all(item["qp"]["valid"] for item in records),
        "qp_failure_count": sum(not item["qp"]["valid"] for item in records),
        "constraint_count_min": min(int(item["constraint_count"]) for item in records),
        "constraint_count_max": max(int(item["constraint_count"]) for item in records),
        "first_nominal_violation": first_violation,
        "minimum_h_opt_m": min(float(item["minimum_h_opt_m"]) for item in records),
        "qp_total_wall_timing": stats(qp_total),
        "qp_solve_wall_timing": stats(qp_solve),
        "total_shadow_wall_timing": stats(shadow_total),
    }
