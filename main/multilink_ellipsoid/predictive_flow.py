"""Predictive L5--L7 plus end-effector flow-guidance oracle.

This module is opt-in. It identifies a local ten-step clearance model by
rolling complete SafeLIBERO OSC transitions in a synchronized clone. The
ordinary AEGIS evaluator and ordinary OpenPI sampler never import it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .barrier import support_gap
from .geometry import Ellipsoid
from .rollout import (
    _auxiliary_sim_snapshot,
    _base_env,
    _controller_snapshot,
    _dynamic_state_vector,
    _restore_auxiliary_sim_snapshot,
    _restore_controller_snapshot,
)
from .shadow import (
    DISTAL_ELLIPSOID_SCHEMA,
    MultilinkEllipsoidShadow,
    _eef_site_id,
    _numpy,
    _raw_model_data,
)


PREDICTIVE_FLOW_SCHEMA = "vlsa_predictive_flow_guidance.v1"
GUIDANCE_ENVELOPE_SCHEMA = "crfs_predictive_flow_guidance.v1"
PROTECTED_BODY_NAMES = [
    "robot0_link5",
    "robot0_link6",
    "robot0_link7",
    "robot0_end_effector",
]


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


def load_predictive_flow_config(path: Path) -> dict[str, Any]:
    """Load and fail closed on the preregistered primary-case protocol."""

    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("predictive-flow config is invalid JSON") from error
    required = {
        "schema_version",
        "protocol_id",
        "case_ids",
        "claim_scope",
        "nominal_policy",
        "action_protocol",
        "obstacle_geometry",
        "protected_body_names",
        "robot_geometry",
        "end_effector_geometry",
        "trajectory_model",
        "flow_guidance",
        "verification",
        "success_definition",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("predictive-flow config keys differ")
    if config["schema_version"] != PREDICTIVE_FLOW_SCHEMA:
        raise ValueError("predictive-flow config schema differs")
    if config["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("predictive-flow config must select only the primary case")
    if config["protected_body_names"] != PROTECTED_BODY_NAMES:
        raise ValueError("predictive-flow protected bodies differ")
    if config["obstacle_geometry"] != "released_aegis_frozen_perception_mvee":
        raise ValueError("predictive-flow obstacle geometry differs")
    action = config["action_protocol"]
    if action != {
        "execute_actions_per_query": 5,
        "model_action_horizon": 10,
        "modified_dimensions": [0, 1, 2],
        "preserve_gripper": True,
        "rotation": "zero_translational_protocol",
    }:
        raise ValueError("predictive-flow action protocol differs")
    end_effector = config["end_effector_geometry"]
    if end_effector != {
        "center_and_orientation": (
            "authoritative_robot0_grip_site_pose_plus_released_"
            "minus_0.08m_local_z_offset"
        ),
        "semiaxes_m": [0.06, 0.12, 0.11],
        "source": "released_aegis_end_effector_proxy",
    }:
        raise ValueError("predictive-flow end-effector geometry differs")
    trajectory = config["trajectory_model"]
    if set(trajectory) != {
        "activation_h_opt_m",
        "clearance_tolerance_m",
        "finite_difference_action",
        "finite_difference_scheme",
        "horizon_steps",
        "optimizer_clearance_m",
        "transition",
    }:
        raise ValueError("predictive-flow trajectory model keys differ")
    if trajectory["horizon_steps"] != 10:
        raise ValueError("predictive-flow horizon must equal ten")
    if trajectory["finite_difference_scheme"] != "clipped_central_difference":
        raise ValueError("predictive-flow finite-difference scheme differs")
    if trajectory["transition"] != (
        "complete_cloned_safelibero_env_step_with_stateful_osc_pose_and_gripper"
    ):
        raise ValueError("predictive-flow transition model differs")
    for key in (
        "activation_h_opt_m",
        "clearance_tolerance_m",
        "finite_difference_action",
        "optimizer_clearance_m",
    ):
        value = trajectory[key]
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError("trajectory_model.%s must be positive" % key)
    flow = config["flow_guidance"]
    if flow != {
        "action_limit": 1.0,
        "barrier_decay_gamma": 0.9,
        "dykstra_projection_sweeps_per_euler_step": 64,
        "euler_steps": 10,
        "projection_residual_tolerance": 0.00005,
        "relinearization_attempts": 1,
    }:
        raise ValueError("predictive-flow sampler protocol differs")
    verification = config["verification"]
    if verification != {
        "clone_state_tolerance": 1e-10,
        "execute_only_exactly_verified_chunk": True,
        "raw_simulator_contact_and_car_authority": True,
    }:
        raise ValueError("predictive-flow verification protocol differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


class PredictiveFullBodyGeometry:
    """The accepted distal MVEEs plus the released AEGIS EE proxy."""

    def __init__(
        self,
        carrier: MultilinkEllipsoidShadow,
        *,
        end_effector_semiaxes_m: Sequence[float],
        clearance_m: float,
    ) -> None:
        np = _numpy()
        self.carrier = carrier
        self.obstacle = carrier.obstacle
        self.end_effector_semiaxes_m = np.asarray(
            end_effector_semiaxes_m, dtype=np.float64
        )
        self.clearance_m = float(clearance_m)
        if self.end_effector_semiaxes_m.shape != (3,) or np.any(
            self.end_effector_semiaxes_m <= 0.0
        ):
            raise ValueError("end-effector semiaxes are invalid")

    @classmethod
    def from_aegis_geometry(
        cls,
        config: Mapping[str, Any],
        obstacle_geometry: Mapping[str, Any],
    ) -> "PredictiveFullBodyGeometry":
        trajectory = config["trajectory_model"]
        carrier_config = {
            "schema_version": DISTAL_ELLIPSOID_SCHEMA,
            "protocol_id": "%s-geometry" % config["protocol_id"],
            "case_ids": list(config["case_ids"]),
            "control_effect": "read_only_no_executed_action_change",
            "obstacle_geometry": config["obstacle_geometry"],
            "protected_body_names": list(config["protected_body_names"][:3]),
            "robot_geometry": dict(config["robot_geometry"]),
            "optimizer": {
                "alpha_s_inv": 10.0,
                "optimizer_clearance_m": float(
                    trajectory["optimizer_clearance_m"]
                ),
                "joint_velocity_limit_rad_s": 1.0,
                "resolved_rate_damping": 0.05,
                "eef_preservation_weight": 1.0,
                "eef_weight_diagonal": [1.0] * 6,
                "eps_abs": 1.0e-7,
                "eps_rel": 1.0e-7,
                "max_iter": 10000,
                "residual_tolerance": 5.0e-7,
                "bound_tolerance_rad_s": 5.0e-8,
            },
            "simulator_verification": {
                "D_sim": "raw_mujoco_nonpositive_contact_distance_and_active_obstacle_displacement",
                "distinct_from_D_opt": True,
            },
            "claim_scope": config["claim_scope"],
        }
        carrier = MultilinkEllipsoidShadow.from_aegis_geometry(
            carrier_config, obstacle_geometry
        )
        return cls(
            carrier,
            end_effector_semiaxes_m=config["end_effector_geometry"][
                "semiaxes_m"
            ],
            clearance_m=float(trajectory["optimizer_clearance_m"]),
        )

    def ellipsoids(self, env: Any) -> list[Ellipsoid]:
        np = _numpy()
        links = self.carrier._distal_links(env)
        if [item.body_name for item in links] != PROTECTED_BODY_NAMES[:3]:
            raise ValueError("predictive distal geometry differs from L5--L7")
        model, data = _raw_model_data(env.sim)
        site_id = _eef_site_id(env)
        body_id = int(model.site_bodyid[site_id])
        rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(
            3, 3
        )
        center = np.asarray(data.site_xpos[site_id], dtype=np.float64) + rotation @ np.asarray(
            [0.0, 0.0, -0.08], dtype=np.float64
        )
        end_effector = Ellipsoid(
            center=center,
            rotation=rotation,
            semiaxes_m=self.end_effector_semiaxes_m,
            body_id=body_id,
            body_name="robot0_end_effector",
            geom_id=-1,
            geom_name="released_aegis_end_effector_proxy",
            bound_source="released_aegis_end_effector_proxy",
            source_body_names=("robot0_right_hand",),
        )
        return [*links, end_effector]

    def clearances(self, env: Any) -> Any:
        np = _numpy()
        values = np.asarray(
            [
                support_gap(
                    body,
                    self.obstacle,
                    optimizer_clearance_m=self.clearance_m,
                )
                for body in self.ellipsoids(env)
            ],
            dtype=np.float64,
        )
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("predictive full-body clearance vector is invalid")
        return values

    def geometry_record(self, env: Any) -> dict[str, Any]:
        bodies = self.ellipsoids(env)
        return {
            "obstacle": self.obstacle.to_record(),
            "protected_body_count": 4,
            "protected_body_names": list(PROTECTED_BODY_NAMES),
            "ellipsoids": [body.to_record() for body in bodies],
            "coverage_semantics": {
                "robot0_link5_link6_link7": (
                    "certified_surface_fitted_compiled_collision_mesh_MVEEs"
                ),
                "robot0_end_effector": "released_AEGIS_pose_oriented_proxy",
            },
        }


class ClonedOscTrajectoryProbe:
    """Roll complete open-loop action chunks from the exact current state."""

    def __init__(self, probe_env: Any, geometry: PredictiveFullBodyGeometry) -> None:
        self.probe_env = probe_env
        self.geometry = geometry

    def synchronize(self, main_env: Any) -> dict[str, Any]:
        np = _numpy()
        main_base = _base_env(main_env)
        probe_base = _base_env(self.probe_env)
        simulator_state = np.asarray(
            main_env.sim.get_state().flatten(), dtype=np.float64
        ).copy()
        auxiliary = _auxiliary_sim_snapshot(main_env)
        controllers = _controller_snapshot(main_env)
        self.probe_env.sim.set_state_from_flattened(simulator_state)
        _restore_auxiliary_sim_snapshot(self.probe_env, auxiliary)
        probe_base.timestep = int(main_base.timestep)
        probe_base.cur_time = float(main_base.cur_time)
        probe_base.done = bool(main_base.done)
        _restore_controller_snapshot(self.probe_env, controllers)
        self.probe_env.sim.forward()
        _restore_auxiliary_sim_snapshot(self.probe_env, auxiliary)
        main = _dynamic_state_vector(main_env)
        probe = _dynamic_state_vector(self.probe_env)
        if main.shape != probe.shape:
            raise ValueError("predictive clone dynamic-state shapes differ")
        error = float(np.max(np.abs(main - probe)))
        return {
            "state_dimension": int(main.size),
            "maximum_absolute_error": error,
            "main_state_sha256": hashlib.sha256(main.tobytes()).hexdigest(),
            "probe_state_sha256": hashlib.sha256(probe.tobytes()).hexdigest(),
        }

    def rollout(self, main_env: Any, action_chunk: Any) -> dict[str, Any]:
        np = _numpy()
        actions = np.asarray(action_chunk, dtype=np.float64)
        if actions.shape != (10, 7) or not np.all(np.isfinite(actions)):
            raise ValueError("predictive probe requires one finite 10x7 chunk")
        synchronization = self.synchronize(main_env)
        clearances = []
        next_state_vectors = []
        step_wall_seconds = []
        done_steps = []
        for index, action in enumerate(actions):
            started = time.perf_counter_ns()
            _, _, done, _ = self.probe_env.step(action.tolist())
            step_wall_seconds.append(
                (time.perf_counter_ns() - started) * 1.0e-9
            )
            clearances.append(self.geometry.clearances(self.probe_env))
            next_state_vectors.append(_dynamic_state_vector(self.probe_env))
            if done:
                done_steps.append(index)
        clearance_array = np.asarray(clearances, dtype=np.float64)
        if clearance_array.shape != (10, 4):
            raise ValueError("predictive rollout clearance trace differs")
        return {
            "actions": actions,
            "h_opt_m": clearance_array,
            "minimum_h_opt_m": float(np.min(clearance_array)),
            "next_state_vectors": next_state_vectors,
            "next_state_sha256": [
                hashlib.sha256(value.tobytes()).hexdigest()
                for value in next_state_vectors
            ],
            "synchronization": synchronization,
            "step_wall_seconds": step_wall_seconds,
            "total_env_step_wall_seconds": float(sum(step_wall_seconds)),
            "done_steps": done_steps,
        }


def translational_chunk(action_chunk: Any, *, action_limit: float = 1.0) -> Any:
    """Apply the registered translational protocol without changing gripper."""

    np = _numpy()
    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.shape != (10, 7) or not np.all(np.isfinite(actions)):
        raise ValueError("policy action chunk must be finite with shape 10x7")
    output = actions.copy()
    output[:, :3] = np.clip(output[:, :3], -float(action_limit), float(action_limit))
    output[:, 3:6] = 0.0
    return output


def trajectory_cbf_residuals(
    current_h_opt_m: Any,
    future_h_opt_m: Any,
    *,
    gamma: float,
) -> Any:
    """Return h_j - (1-gamma) h_{j-1} for all steps and bodies."""

    np = _numpy()
    current = np.asarray(current_h_opt_m, dtype=np.float64)
    future = np.asarray(future_h_opt_m, dtype=np.float64)
    if current.shape != (4,) or future.shape != (10, 4):
        raise ValueError("trajectory CBF clearance shapes differ")
    rho = 1.0 - float(gamma)
    previous = np.vstack((current[None, :], future[:-1]))
    return future - rho * previous


def identify_osc_clearance_model(
    probe: ClonedOscTrajectoryProbe,
    main_env: Any,
    center_output_actions: Any,
    *,
    perturbation_action: float,
    action_limit: float,
    clone_state_tolerance: float,
) -> dict[str, Any]:
    """Identify d h[future step, body] / d chunk-XYZ by cloned rollouts."""

    np = _numpy()
    started = time.perf_counter_ns()
    center = translational_chunk(center_output_actions, action_limit=action_limit)
    base = probe.rollout(main_env, center)
    if base["synchronization"]["maximum_absolute_error"] > clone_state_tolerance:
        raise ValueError("predictive base clone synchronization failed")
    jacobian = np.zeros((10, 4, 30), dtype=np.float64)
    denominators = np.zeros(30, dtype=np.float64)
    probe_wall = 0.0
    probe_hashes = []
    for variable in range(30):
        step, dimension = divmod(variable, 3)
        plus_actions = center.copy()
        minus_actions = center.copy()
        plus_actions[step, dimension] = min(
            action_limit,
            plus_actions[step, dimension] + perturbation_action,
        )
        minus_actions[step, dimension] = max(
            -action_limit,
            minus_actions[step, dimension] - perturbation_action,
        )
        denominator = float(
            plus_actions[step, dimension] - minus_actions[step, dimension]
        )
        if denominator <= 0.0:
            raise ValueError("predictive finite-difference denominator is nonpositive")
        plus = probe.rollout(main_env, plus_actions)
        minus = probe.rollout(main_env, minus_actions)
        for transition in (plus, minus):
            if (
                transition["synchronization"]["maximum_absolute_error"]
                > clone_state_tolerance
            ):
                raise ValueError("predictive finite-difference clone synchronization failed")
        jacobian[:, :, variable] = (
            plus["h_opt_m"] - minus["h_opt_m"]
        ) / denominator
        # A causal OSC transition cannot be affected before this command.
        if step > 0 and np.max(np.abs(jacobian[:step, :, variable])) > 1.0e-10:
            raise ValueError("predictive finite-difference model violates causality")
        denominators[variable] = denominator
        probe_wall += (
            plus["total_env_step_wall_seconds"]
            + minus["total_env_step_wall_seconds"]
        )
        probe_hashes.append(
            {
                "variable": variable,
                "plus_final_state_sha256": plus["next_state_sha256"][-1],
                "minus_final_state_sha256": minus["next_state_sha256"][-1],
            }
        )
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("predictive finite-difference Jacobian is nonfinite")
    return {
        "center_actions": center,
        "base": base,
        "jacobian": jacobian,
        "record": {
            "scheme": "clipped_central_difference",
            "variable_count": 30,
            "rollout_count": 61,
            "simulated_env_step_count": 610,
            "perturbation_action": float(perturbation_action),
            "denominators": denominators.tolist(),
            "jacobian_shape": [10, 4, 30],
            "jacobian_sha256": hashlib.sha256(
                np.ascontiguousarray(jacobian, dtype="<f8").tobytes()
            ).hexdigest(),
            "probe_final_state_hashes": probe_hashes,
            "base_env_step_wall_seconds": float(
                base["total_env_step_wall_seconds"]
            ),
            "finite_difference_env_step_wall_seconds": float(probe_wall),
            "model_wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        },
    }


def build_flow_guidance_envelope(
    model: Mapping[str, Any],
    nominal_output_actions: Any,
    current_h_opt_m: Any,
    *,
    gamma: float,
    action_limit: float,
    projection_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build 40 trajectory-CBF rows plus explicit XYZ action bounds."""

    np = _numpy()
    nominal = np.asarray(nominal_output_actions, dtype=np.float64)
    if nominal.shape != (10, 7):
        raise ValueError("guidance nominal output action shape differs")
    center = np.asarray(model["center_actions"], dtype=np.float64)
    base_h = np.asarray(model["base"]["h_opt_m"], dtype=np.float64)
    jacobian = np.asarray(model["jacobian"], dtype=np.float64)
    current = np.asarray(current_h_opt_m, dtype=np.float64)
    if center.shape != (10, 7) or base_h.shape != (10, 4) or jacobian.shape != (
        10,
        4,
        30,
    ):
        raise ValueError("guidance model shapes differ")
    rho = 1.0 - float(gamma)
    center_xyz = center[:, :3].reshape(-1)
    nominal_xyz = nominal[:, :3].reshape(-1)
    rows = []
    lower = []
    labels = []
    redundant = []
    for step in range(10):
        for body in range(4):
            previous_h = current[body] if step == 0 else base_h[step - 1, body]
            previous_gradient = (
                np.zeros(30, dtype=np.float64)
                if step == 0
                else jacobian[step - 1, body]
            )
            row = jacobian[step, body] - rho * previous_gradient
            lower_at_center = rho * previous_h - base_h[step, body]
            lower_at_nominal = float(
                lower_at_center + row @ (center_xyz - nominal_xyz)
            )
            label = {
                "kind": "trajectory_cbf",
                "step": step,
                "body_index": body,
                "body_name": PROTECTED_BODY_NAMES[body],
                "base_residual_m": float(base_h[step, body] - rho * previous_h),
            }
            if float(np.linalg.norm(row)) <= 1.0e-10:
                if lower_at_nominal > projection_tolerance:
                    raise ValueError(
                        "violated trajectory CBF row has no modeled OSC authority"
                    )
                redundant.append(label)
            else:
                rows.append(row)
                lower.append(lower_at_nominal)
                labels.append(label)
    trajectory_constraint_count = len(rows)
    for variable in range(30):
        positive = np.zeros(30, dtype=np.float64)
        positive[variable] = 1.0
        rows.append(positive)
        lower.append(float(-action_limit - nominal_xyz[variable]))
        labels.append({"kind": "lower_action_bound", "variable": variable})
        rows.append(-positive)
        lower.append(float(nominal_xyz[variable] - action_limit))
        labels.append({"kind": "upper_action_bound", "variable": variable})
    row_array = np.asarray(rows, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    envelope = {
        "schema_version": GUIDANCE_ENVELOPE_SCHEMA,
        "action_horizon": 10,
        "action_dimensions": [0, 1, 2],
        "nominal_output_actions": nominal.tolist(),
        "delta_rows": row_array.tolist(),
        "delta_lower": lower_array.tolist(),
        "projection_sweeps": 64,
        "projection_tolerance": float(projection_tolerance),
    }
    record = {
        "trajectory_constraint_definition_count": 40,
        "trajectory_projection_row_count": trajectory_constraint_count,
        "redundant_safe_trajectory_rows": redundant,
        "action_bound_row_count": 60,
        "projection_row_count": int(row_array.shape[0]),
        "labels": labels,
        "delta_rows_sha256": hashlib.sha256(
            np.ascontiguousarray(row_array, dtype="<f8").tobytes()
        ).hexdigest(),
        "delta_lower_sha256": hashlib.sha256(
            np.ascontiguousarray(lower_array, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    return envelope, record


def exact_trajectory_verification(
    current_h_opt_m: Any,
    rollout: Mapping[str, Any],
    *,
    gamma: float,
    tolerance_m: float,
) -> dict[str, Any]:
    """Evaluate the exact cloned trajectory-CBF conditions."""

    np = _numpy()
    residuals = trajectory_cbf_residuals(
        current_h_opt_m, rollout["h_opt_m"], gamma=gamma
    )
    safe = bool(np.all(residuals >= -float(tolerance_m)))
    return {
        "safe": safe,
        "minimum_h_opt_m": float(np.min(rollout["h_opt_m"])),
        "minimum_trajectory_cbf_residual_m": float(np.min(residuals)),
        "residuals_m": residuals.tolist(),
        "next_state_sha256": list(rollout["next_state_sha256"]),
        "synchronization": dict(rollout["synchronization"]),
        "env_step_wall_seconds": float(rollout["total_env_step_wall_seconds"]),
    }
