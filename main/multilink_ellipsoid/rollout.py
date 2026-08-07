"""Simulator-step finite-difference filter for the distal three ellipsoids.

The ordinary AEGIS and continuous multi-CBF paths do not import this module.
This opt-in controller estimates three next-step buffered clearances by
executing perturbed XYZ actions in a synchronized clone of the SafeLIBERO
environment, solves one three-row QP, and verifies the selected action with
one more cloned ``env.step`` before it can be executed in the primary env.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from .barrier import support_gap
from .qp import MultiConstraintQp
from .shadow import DISTAL_ELLIPSOID_SCHEMA, MultilinkEllipsoidShadow, _numpy


ROLLOUT_SCHEMA = "vlsa_distal_three_ellipsoid_rollout_multicbf.v1"
ROLLOUT_STEP_SCHEMA = "vlsa_distal_three_ellipsoid_rollout_multicbf_step.v1"
_PROTECTED_BODIES = ["robot0_link5", "robot0_link6", "robot0_link7"]
_ROBOT_GEOMETRY = {
    "source": "compiled_mujoco_collision_mesh_vertices",
    "fit": "khachiyan_mvee_exact_vertex_inflation",
    "relative_numerical_padding": 1.0e-9,
    "khachiyan_tolerance": 1.0e-4,
    "khachiyan_max_iterations": 20000,
}
_SIMULATOR_VERIFICATION = {
    "D_sim": "raw_mujoco_nonpositive_contact_distance_and_active_obstacle_displacement",
    "distinct_from_D_opt": True,
}


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


def load_rollout_config(path: Path) -> dict[str, Any]:
    """Load the frozen cloned-step multi-CBF experiment configuration."""

    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("rollout multi-CBF config is invalid JSON") from error
    required = {
        "schema_version",
        "protocol_id",
        "case_ids",
        "control_effect",
        "nominal_action_source",
        "output_action_contract",
        "obstacle_geometry",
        "protected_body_names",
        "robot_geometry",
        "finite_difference",
        "optimizer",
        "verification",
        "simulator_verification",
        "success_definition",
        "claim_scope",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("rollout multi-CBF config keys differ")
    if config.get("schema_version") != ROLLOUT_SCHEMA:
        raise ValueError("rollout multi-CBF config schema differs")
    if config["control_effect"] != "modify_archived_aegis_xyz_after_cloned_step_verification":
        raise ValueError("rollout multi-CBF control effect differs")
    if config["nominal_action_source"] != "immutable_archived_released_aegis_env_step_input":
        raise ValueError("rollout multi-CBF nominal action source differs")
    if config["output_action_contract"] != {
        "xyz": "three_row_finite_difference_qp_then_exact_cloned_step_verification",
        "rotation": "zero_translational_protocol",
        "gripper": "preserve_archived_aegis_command",
    }:
        raise ValueError("rollout multi-CBF action contract differs")
    if config["obstacle_geometry"] != "released_aegis_frozen_perception_mvee":
        raise ValueError("rollout multi-CBF obstacle geometry differs")
    if config["protected_body_names"] != _PROTECTED_BODIES:
        raise ValueError("rollout multi-CBF must protect exactly link5-link7")
    if config["robot_geometry"] != _ROBOT_GEOMETRY:
        raise ValueError("rollout multi-CBF robot geometry differs")
    case_ids = config["case_ids"]
    if not isinstance(case_ids, list) or not case_ids or any(
        not isinstance(value, str) or not value for value in case_ids
    ):
        raise ValueError("rollout multi-CBF case_ids must be nonempty strings")

    finite_difference = config["finite_difference"]
    if not isinstance(finite_difference, dict) or set(finite_difference) != {
        "action_dimensions",
        "scheme",
        "perturbation_action",
    }:
        raise ValueError("finite_difference keys differ")
    if finite_difference["action_dimensions"] != [0, 1, 2]:
        raise ValueError("finite differences must cover exactly XYZ")
    if finite_difference["scheme"] != "clipped_central_difference":
        raise ValueError("finite-difference scheme differs")
    perturbation = finite_difference["perturbation_action"]
    if (
        isinstance(perturbation, bool)
        or not math.isfinite(float(perturbation))
        or not 0.0 < float(perturbation) <= 0.25
    ):
        raise ValueError("finite-difference perturbation must be in (0, 0.25]")

    optimizer = config["optimizer"]
    optimizer_keys = {
        "action_limit",
        "action_weight_diagonal",
        "optimizer_clearance_m",
        "next_step_minimum_h_opt_m",
        "eps_abs",
        "eps_rel",
        "max_iter",
        "residual_tolerance",
        "bound_tolerance_action",
        "material_correction_tolerance",
    }
    if not isinstance(optimizer, dict) or set(optimizer) != optimizer_keys:
        raise ValueError("rollout optimizer keys differ")
    for key in (
        "action_limit",
        "eps_abs",
        "eps_rel",
        "residual_tolerance",
        "bound_tolerance_action",
        "material_correction_tolerance",
    ):
        value = optimizer[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("optimizer.%s must be finite and positive" % key)
    for key in ("optimizer_clearance_m", "next_step_minimum_h_opt_m"):
        value = optimizer[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("optimizer.%s must be finite and nonnegative" % key)
    weights = optimizer["action_weight_diagonal"]
    if not isinstance(weights, list) or len(weights) != 3 or any(
        isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0
        for value in weights
    ):
        raise ValueError("optimizer.action_weight_diagonal must contain three positive values")
    if (
        isinstance(optimizer["max_iter"], bool)
        or not isinstance(optimizer["max_iter"], int)
        or optimizer["max_iter"] < 1
    ):
        raise ValueError("optimizer.max_iter must be a positive integer")

    verification = config["verification"]
    if not isinstance(verification, dict) or set(verification) != {
        "candidate_scales_toward_stop",
        "clearance_tolerance_m",
        "clone_state_tolerance",
        "repeat_nominal_on_first_step",
    }:
        raise ValueError("rollout verification keys differ")
    if verification["candidate_scales_toward_stop"] != [1.0, 0.5, 0.25, 0.125, 0.0]:
        raise ValueError("rollout candidate scales differ")
    for key in ("clearance_tolerance_m", "clone_state_tolerance"):
        value = verification[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("verification.%s must be finite and nonnegative" % key)
    if verification["repeat_nominal_on_first_step"] is not True:
        raise ValueError("first-step repeatability audit is required")
    if config["simulator_verification"] != _SIMULATOR_VERIFICATION:
        raise ValueError("rollout multi-CBF must keep D_sim distinct from D_opt")
    if config["success_definition"] != {
        "protected_link_contact": "none_for_robot0_link5_link6_link7",
        "paper_car": "maximum_active_obstacle_l1_displacement_at_most_0.001_m",
        "task": "native_goal_satisfied_within_archived_237_action_horizon",
        "clone_fidelity": "every_executed_next_state_matches_its_verified_clone",
    }:
        raise ValueError("rollout multi-CBF success definition differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def finite_difference_clearance_jacobian(
    plus_clearances: Any,
    minus_clearances: Any,
    plus_xyz: Any,
    minus_xyz: Any,
) -> Any:
    """Return d(next-step h_i)/d(XYZ) from clipped central probes."""

    np = _numpy()
    plus_h = np.asarray(plus_clearances, dtype=np.float64)
    minus_h = np.asarray(minus_clearances, dtype=np.float64)
    plus = np.asarray(plus_xyz, dtype=np.float64)
    minus = np.asarray(minus_xyz, dtype=np.float64)
    if plus_h.shape != (3, 3) or minus_h.shape != (3, 3):
        raise ValueError("plus/minus clearances must each have shape (3, 3)")
    if plus.shape != (3, 3) or minus.shape != (3, 3):
        raise ValueError("plus/minus XYZ probes must each have shape (3, 3)")
    if not all(np.all(np.isfinite(value)) for value in (plus_h, minus_h, plus, minus)):
        raise ValueError("finite-difference probes must be finite")
    jacobian = np.empty((3, 3), dtype=np.float64)
    for dimension in range(3):
        denominator = float(plus[dimension, dimension] - minus[dimension, dimension])
        if denominator <= 0.0:
            raise ValueError("clipped finite-difference denominator is nonpositive")
        # Only the selected coordinate is allowed to differ between the pair.
        difference = plus[dimension] - minus[dimension]
        if any(index != dimension for index in np.flatnonzero(np.abs(difference) > 1.0e-15)):
            raise ValueError("finite-difference probe changed more than one XYZ dimension")
        jacobian[:, dimension] = (plus_h[dimension] - minus_h[dimension]) / denominator
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("finite-difference clearance Jacobian is nonfinite")
    return jacobian


def discrete_qp_lower_bounds(
    nominal_next_clearance: Any,
    jacobian: Any,
    nominal_xyz: Any,
    *,
    minimum_next_clearance: float,
) -> Any:
    """Convert h+(u0) + G(u-u0) >= h_min into G u >= lower."""

    np = _numpy()
    base = np.asarray(nominal_next_clearance, dtype=np.float64)
    rows = np.asarray(jacobian, dtype=np.float64)
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    minimum = float(minimum_next_clearance)
    if base.shape != (3,) or rows.shape != (3, 3) or nominal.shape != (3,):
        raise ValueError("discrete QP arrays have inconsistent shapes")
    if not all(np.all(np.isfinite(value)) for value in (base, rows, nominal)):
        raise ValueError("discrete QP arrays must be finite")
    if not math.isfinite(minimum):
        raise ValueError("minimum next clearance must be finite")
    return minimum - base + rows @ nominal


def _copy_optional_array(value: Any) -> Any:
    np = _numpy()
    if value is None:
        return None
    return np.array(value, copy=True)


def _base_env(env: Any) -> Any:
    base = getattr(env, "env", None)
    if base is None:
        raise ValueError("SafeLIBERO wrapper lacks its base environment")
    return base


def _controller_snapshot(env: Any) -> list[dict[str, Any]]:
    output = []
    for robot in env.robots:
        controller = robot.controller
        if controller.interpolator_pos is not None or controller.interpolator_ori is not None:
            raise ValueError("cloned-step protocol requires OSC interpolation to be disabled")
        output.append(
            {
                "goal_pos": _copy_optional_array(controller.goal_pos),
                "goal_ori": _copy_optional_array(controller.goal_ori),
                "relative_ori": _copy_optional_array(controller.relative_ori),
                "ori_ref": _copy_optional_array(controller.ori_ref),
                "new_update": bool(controller.new_update),
                "torques": _copy_optional_array(controller.torques),
                "robot_torques": _copy_optional_array(robot.torques),
                "gripper_current_action": _copy_optional_array(
                    robot.gripper.current_action if robot.has_gripper else None
                ),
            }
        )
    return output


def _restore_controller_snapshot(env: Any, snapshots: Sequence[Mapping[str, Any]]) -> None:
    np = _numpy()
    if len(env.robots) != len(snapshots):
        raise ValueError("main and probe robot counts differ")
    for robot, snapshot in zip(env.robots, snapshots):
        controller = robot.controller
        for name in ("goal_pos", "goal_ori", "relative_ori", "ori_ref", "torques"):
            value = snapshot[name]
            setattr(controller, name, None if value is None else np.array(value, copy=True))
        controller.new_update = bool(snapshot["new_update"])
        robot.torques = (
            None
            if snapshot["robot_torques"] is None
            else np.array(snapshot["robot_torques"], copy=True)
        )
        if robot.has_gripper:
            robot.gripper.current_action = np.array(
                snapshot["gripper_current_action"], copy=True
            )


def _auxiliary_sim_snapshot(env: Any) -> dict[str, Any]:
    np = _numpy()
    data = env.sim.data
    output = {}
    for name in (
        "ctrl",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "userdata",
    ):
        if hasattr(data, name):
            output[name] = np.array(getattr(data, name), copy=True)
    return output


def _restore_auxiliary_sim_snapshot(env: Any, snapshot: Mapping[str, Any]) -> None:
    np = _numpy()
    for name, value in snapshot.items():
        target = getattr(env.sim.data, name)
        target[...] = np.asarray(value, dtype=target.dtype)


def _dynamic_state_vector(env: Any) -> Any:
    """Vector used to verify cloned and primary one-step transitions."""

    np = _numpy()
    base = _base_env(env)
    values = [np.asarray(env.sim.get_state().flatten(), dtype=np.float64)]
    for value in _auxiliary_sim_snapshot(env).values():
        values.append(np.asarray(value, dtype=np.float64).reshape(-1))
    values.append(np.asarray([float(base.timestep), float(base.cur_time), float(base.done)]))
    for snapshot in _controller_snapshot(env):
        for name in (
            "goal_pos",
            "goal_ori",
            "relative_ori",
            "ori_ref",
            "torques",
            "robot_torques",
            "gripper_current_action",
        ):
            value = snapshot[name]
            if value is not None:
                values.append(np.asarray(value, dtype=np.float64).reshape(-1))
        values.append(np.asarray([float(snapshot["new_update"])]))
    output = np.concatenate(values)
    if not np.all(np.isfinite(output)):
        raise ValueError("cloned dynamic state vector is nonfinite")
    return output


class ClonedSimulatorStepProbe:
    """Synchronize one probe env from the primary env before every env.step."""

    def __init__(self, probe_env: Any, geometry: MultilinkEllipsoidShadow, *, clearance_m: float) -> None:
        self.probe_env = probe_env
        self.geometry = geometry
        self.clearance_m = float(clearance_m)

    def clearances(self, env: Any) -> Any:
        np = _numpy()
        links = self.geometry._distal_links(env)
        if [item.body_name for item in links] != _PROTECTED_BODIES:
            raise ValueError("live rollout clearances do not correspond to L5-L7")
        output = np.asarray(
            [
                support_gap(
                    link,
                    self.geometry.obstacle,
                    optimizer_clearance_m=self.clearance_m,
                )
                for link in links
            ],
            dtype=np.float64,
        )
        if output.shape != (3,) or not np.all(np.isfinite(output)):
            raise ValueError("rollout clearance vector is invalid")
        return output

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
        # ``forward`` may refresh solver work arrays. Restore the exact warm
        # start and applied-control values after kinematic quantities exist.
        _restore_auxiliary_sim_snapshot(self.probe_env, auxiliary)
        main_vector = _dynamic_state_vector(main_env)
        probe_vector = _dynamic_state_vector(self.probe_env)
        if main_vector.shape != probe_vector.shape:
            raise ValueError("main and probe dynamic-state shapes differ")
        difference = np.abs(main_vector - probe_vector)
        return {
            "state_dimension": int(main_vector.size),
            "maximum_absolute_error": float(np.max(difference)),
            "main_state_sha256": hashlib.sha256(main_vector.tobytes()).hexdigest(),
            "probe_state_sha256": hashlib.sha256(probe_vector.tobytes()).hexdigest(),
        }

    def transition(self, main_env: Any, action: Sequence[float]) -> dict[str, Any]:
        np = _numpy()
        synchronization = self.synchronize(main_env)
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (7,) or not np.all(np.isfinite(command)):
            raise ValueError("probe action must be finite with length seven")
        started = time.perf_counter_ns()
        self.probe_env.step(command.tolist())
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        vector = _dynamic_state_vector(self.probe_env)
        clearances = self.clearances(self.probe_env)
        return {
            "action": command.tolist(),
            "next_h_opt_m": clearances.tolist(),
            "next_minimum_h_opt_m": float(np.min(clearances)),
            "next_state_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
            "next_state_vector": vector,
            "synchronization": synchronization,
            "env_step_wall_seconds": elapsed,
        }


class DistalThreeEllipsoidRolloutMultiCbf:
    """Filter XYZ with three discrete constraints and exact step verification."""

    def __init__(
        self,
        config: Mapping[str, Any],
        geometry: MultilinkEllipsoidShadow,
        probe_env: Any,
    ) -> None:
        if config.get("schema_version") != ROLLOUT_SCHEMA:
            raise ValueError("rollout multi-CBF configuration was not validated")
        self.config = dict(config)
        self.geometry = geometry
        optimizer = config["optimizer"]
        self.qp = MultiConstraintQp(
            eps_abs=float(optimizer["eps_abs"]),
            eps_rel=float(optimizer["eps_rel"]),
            max_iter=int(optimizer["max_iter"]),
            residual_tolerance=float(optimizer["residual_tolerance"]),
            bound_tolerance=float(optimizer["bound_tolerance_action"]),
        )
        self.probe = ClonedSimulatorStepProbe(
            probe_env,
            geometry,
            clearance_m=float(optimizer["optimizer_clearance_m"]),
        )
        self._pending: Optional[dict[str, Any]] = None

    @classmethod
    def from_aegis_geometry(
        cls,
        config: Mapping[str, Any],
        geometry: Mapping[str, Any],
        probe_env: Any,
    ) -> "DistalThreeEllipsoidRolloutMultiCbf":
        optimizer = config["optimizer"]
        carrier_config = {
            "schema_version": DISTAL_ELLIPSOID_SCHEMA,
            "protocol_id": "%s-geometry" % config["protocol_id"],
            "case_ids": list(config["case_ids"]),
            "control_effect": "read_only_no_executed_action_change",
            "obstacle_geometry": config["obstacle_geometry"],
            "protected_body_names": list(config["protected_body_names"]),
            "robot_geometry": dict(config["robot_geometry"]),
            "optimizer": {
                "alpha_s_inv": 10.0,
                "optimizer_clearance_m": float(optimizer["optimizer_clearance_m"]),
                "joint_velocity_limit_rad_s": 1.0,
                "resolved_rate_damping": 0.05,
                "eef_preservation_weight": 1.0,
                "eef_weight_diagonal": [1.0] * 6,
                "eps_abs": float(optimizer["eps_abs"]),
                "eps_rel": float(optimizer["eps_rel"]),
                "max_iter": int(optimizer["max_iter"]),
                "residual_tolerance": float(optimizer["residual_tolerance"]),
                "bound_tolerance_rad_s": float(optimizer["bound_tolerance_action"]),
            },
            "simulator_verification": dict(config["simulator_verification"]),
            "claim_scope": config["claim_scope"],
        }
        carrier = MultilinkEllipsoidShadow.from_aegis_geometry(carrier_config, geometry)
        return cls(config, carrier, probe_env)

    def geometry_record(self, env: Any) -> dict[str, Any]:
        return self.geometry.geometry_record(env)

    def filter(
        self,
        env: Any,
        aegis_action: Sequence[float],
        *,
        step: int,
    ) -> tuple[Optional[list[float]], dict[str, Any]]:
        np = _numpy()
        if self._pending is not None:
            raise ValueError("previous cloned transition was not checked against env.step")
        started = time.perf_counter_ns()
        optimizer = self.config["optimizer"]
        verification = self.config["verification"]
        action = np.asarray(aegis_action, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("released AEGIS action must be finite with length seven")
        if not np.allclose(action[3:6], 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("rollout multi-CBF requires the translational AEGIS protocol")
        limit = float(optimizer["action_limit"])
        nominal_xyz = np.clip(action[:3], -limit, limit)
        nominal_action = action.copy()
        nominal_action[:3] = nominal_xyz
        nominal_action[3:6] = 0.0

        base = self.probe.transition(env, nominal_action)
        clone_tolerance = float(verification["clone_state_tolerance"])
        if base["synchronization"]["maximum_absolute_error"] > clone_tolerance:
            raise ValueError("probe clone did not synchronize to the primary state")
        repeatability = None
        if int(step) == 0 and verification["repeat_nominal_on_first_step"]:
            repeated = self.probe.transition(env, nominal_action)
            repeat_error = float(
                np.max(
                    np.abs(
                        np.asarray(base["next_state_vector"], dtype=np.float64)
                        - np.asarray(repeated["next_state_vector"], dtype=np.float64)
                    )
                )
            )
            repeatability = {
                "maximum_absolute_next_state_error": repeat_error,
                "first_next_state_sha256": base["next_state_sha256"],
                "second_next_state_sha256": repeated["next_state_sha256"],
            }
            if repeat_error > clone_tolerance:
                raise ValueError("repeated cloned nominal transition is not deterministic")

        perturbation = float(self.config["finite_difference"]["perturbation_action"])
        plus_xyz = []
        minus_xyz = []
        plus_h = []
        minus_h = []
        probe_records = []
        for dimension in range(3):
            positive_xyz = nominal_xyz.copy()
            negative_xyz = nominal_xyz.copy()
            positive_xyz[dimension] = min(limit, positive_xyz[dimension] + perturbation)
            negative_xyz[dimension] = max(-limit, negative_xyz[dimension] - perturbation)
            positive_action = nominal_action.copy()
            negative_action = nominal_action.copy()
            positive_action[:3] = positive_xyz
            negative_action[:3] = negative_xyz
            positive = self.probe.transition(env, positive_action)
            negative = self.probe.transition(env, negative_action)
            for transition in (positive, negative):
                if transition["synchronization"]["maximum_absolute_error"] > clone_tolerance:
                    raise ValueError("finite-difference probe clone synchronization failed")
            plus_xyz.append(positive_xyz)
            minus_xyz.append(negative_xyz)
            plus_h.append(positive["next_h_opt_m"])
            minus_h.append(negative["next_h_opt_m"])
            probe_records.append(
                {
                    "dimension": dimension,
                    "plus": {key: value for key, value in positive.items() if key != "next_state_vector"},
                    "minus": {key: value for key, value in negative.items() if key != "next_state_vector"},
                }
            )
        rows = finite_difference_clearance_jacobian(
            np.asarray(plus_h),
            np.asarray(minus_h),
            np.asarray(plus_xyz),
            np.asarray(minus_xyz),
        )
        minimum = float(optimizer["next_step_minimum_h_opt_m"])
        lower = discrete_qp_lower_bounds(
            base["next_h_opt_m"],
            rows,
            nominal_xyz,
            minimum_next_clearance=minimum,
        )
        metric = np.diag(np.asarray(optimizer["action_weight_diagonal"], dtype=np.float64))
        qp_result = self.qp.solve(
            nominal_xyz,
            metric,
            rows,
            lower,
            -limit * np.ones(3),
            limit * np.ones(3),
        )
        qp_candidate = None if qp_result.qdot_safe is None else np.asarray(qp_result.qdot_safe)
        verification_attempts = []
        accepted = None
        accepted_transition = None
        tolerance = float(verification["clearance_tolerance_m"])
        if qp_result.valid and qp_candidate is not None:
            if np.all(np.asarray(base["next_h_opt_m"]) >= minimum - tolerance):
                qp_candidate = nominal_xyz.copy()
            seen = set()
            for scale in verification["candidate_scales_toward_stop"]:
                candidate_xyz = float(scale) * qp_candidate
                key = tuple(float(value) for value in candidate_xyz)
                if key in seen:
                    continue
                seen.add(key)
                candidate_action = nominal_action.copy()
                candidate_action[:3] = candidate_xyz
                if np.array_equal(candidate_xyz, nominal_xyz):
                    transition = base
                else:
                    transition = self.probe.transition(env, candidate_action)
                if transition["synchronization"]["maximum_absolute_error"] > clone_tolerance:
                    raise ValueError("candidate-verification probe clone synchronization failed")
                clearances = np.asarray(transition["next_h_opt_m"], dtype=np.float64)
                safe = bool(np.all(clearances >= minimum - tolerance))
                verification_attempts.append(
                    {
                        "scale_toward_stop": float(scale),
                        "candidate_xyz": candidate_xyz.tolist(),
                        "verified_next_h_opt_m": clearances.tolist(),
                        "verified_safe": safe,
                        "next_state_sha256": transition["next_state_sha256"],
                        "env_step_wall_seconds": transition["env_step_wall_seconds"],
                    }
                )
                if safe:
                    accepted = candidate_action
                    accepted_transition = transition
                    break

        correction_l2 = (
            None
            if accepted is None
            else float(np.linalg.norm(np.asarray(accepted[:3]) - nominal_xyz))
        )
        predicted_candidate_h = None
        if qp_candidate is not None:
            predicted_candidate_h = (
                np.asarray(base["next_h_opt_m"], dtype=np.float64)
                + rows @ (qp_candidate - nominal_xyz)
            ).tolist()
        record = {
            "schema_version": ROLLOUT_STEP_SCHEMA,
            "step": int(step),
            "control_effect": "modify_archived_aegis_xyz_after_cloned_step_verification",
            "solver_variable": "normalized_translational_osc_xyz_action",
            "nominal_aegis_action": action.tolist(),
            "nominal_xyz_after_controller_clip": nominal_xyz.tolist(),
            "constraint_count": 3,
            "constraint_semantics": "linearized_next_step_buffered_clearance_for_L5_L6_L7",
            "nominal_next_h_opt_m": list(base["next_h_opt_m"]),
            "finite_difference": {
                "scheme": self.config["finite_difference"]["scheme"],
                "perturbation_action": perturbation,
                "probe_count": 6,
                "probes": probe_records,
                "clearance_jacobian_m_per_normalized_xyz_action": rows.tolist(),
            },
            "discrete_constraints": {
                "rows": rows.tolist(),
                "lower": lower.tolist(),
                "next_step_minimum_h_opt_m": minimum,
            },
            "qp": {
                "valid": bool(qp_result.valid),
                "reason": qp_result.reason,
                "candidate_xyz": None if qp_candidate is None else qp_candidate.tolist(),
                "predicted_candidate_next_h_opt_m": predicted_candidate_h,
                "diagnostics": dict(qp_result.diagnostics),
            },
            "verification": {
                "exact_cloned_env_step": True,
                "attempts": verification_attempts,
                "accepted": accepted is not None,
                "accepted_next_h_opt_m": (
                    None
                    if accepted_transition is None
                    else list(accepted_transition["next_h_opt_m"])
                ),
                "accepted_next_state_sha256": (
                    None
                    if accepted_transition is None
                    else accepted_transition["next_state_sha256"]
                ),
                "clone_synchronization": base["synchronization"],
                "first_step_nominal_repeatability": repeatability,
                "main_env_post_step_checked": False,
                "main_vs_probe_next_state_max_abs_error": None,
                "main_next_h_opt_m": None,
                "main_vs_probe_next_h_max_abs_error_m": None,
            },
            "executed_action": None if accepted is None else accepted.tolist(),
            "active_correction_l2": correction_l2,
            "material_correction_tolerance": float(
                optimizer["material_correction_tolerance"]
            ),
            "modified": bool(
                correction_l2 is not None
                and correction_l2 > float(optimizer["material_correction_tolerance"])
            ),
            "D_opt": {
                "value_m": float(optimizer["optimizer_clearance_m"]),
                "semantics": "optimizer_support_gap_buffer",
            },
            "D_sim": {
                "available": False,
                "value": {
                    "minimum_protected_link_contact_distance_m": None,
                    "active_obstacle_l1_displacement_m": None,
                    "protected_link_contact_count": None,
                },
                "semantics": self.config["simulator_verification"]["D_sim"],
                "source": "post_step_raw_simulator_evidence_not_optimizer_geometry",
            },
            "timing": {
                "nominal_probe_wall_seconds": float(base["env_step_wall_seconds"]),
                "finite_difference_probe_wall_seconds": float(
                    sum(
                        item[sign]["env_step_wall_seconds"]
                        for item in probe_records
                        for sign in ("plus", "minus")
                    )
                ),
                "candidate_verification_wall_seconds": float(
                    sum(item["env_step_wall_seconds"] for item in verification_attempts)
                ),
                "total_filter_wall_seconds": (
                    time.perf_counter_ns() - started
                )
                * 1.0e-9,
            },
        }
        if accepted_transition is not None:
            self._pending = {
                "step": int(step),
                "state_vector": np.asarray(
                    accepted_transition["next_state_vector"], dtype=np.float64
                ).copy(),
                "clearances": np.asarray(
                    accepted_transition["next_h_opt_m"], dtype=np.float64
                ).copy(),
                "record": record,
            }
        return None if accepted is None else accepted.tolist(), record

    def verify_executed_transition(self, env: Any, record: Mapping[str, Any]) -> None:
        """Bind the accepted clone rollout to the actual following env.step."""

        np = _numpy()
        pending = self._pending
        if pending is None or pending["record"] is not record:
            raise ValueError("no matching cloned transition awaits verification")
        actual_state = _dynamic_state_vector(env)
        expected_state = pending["state_vector"]
        if actual_state.shape != expected_state.shape:
            raise ValueError("actual and cloned next-state shapes differ")
        state_error = float(np.max(np.abs(actual_state - expected_state)))
        actual_clearances = self.probe.clearances(env)
        clearance_error = float(
            np.max(np.abs(actual_clearances - pending["clearances"]))
        )
        verification = record["verification"]
        verification["main_env_post_step_checked"] = True
        verification["main_vs_probe_next_state_max_abs_error"] = state_error
        verification["main_next_h_opt_m"] = actual_clearances.tolist()
        verification["main_vs_probe_next_h_max_abs_error_m"] = clearance_error
        tolerance = float(self.config["verification"]["clone_state_tolerance"])
        self._pending = None
        if state_error > tolerance or clearance_error > tolerance:
            raise ValueError("executed env.step does not match its verified clone rollout")


def summarize_rollout_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    if not records:
        return {"status": "no_records", "step_count": 0}

    def stats(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
            raise ValueError("rollout timing values are invalid")
        return {
            "mean_seconds": float(np.mean(array)),
            "median_seconds": float(np.median(array)),
            "p95_seconds": float(np.quantile(array, 0.95)),
            "maximum_seconds": float(np.max(array)),
        }

    first_intervention = next(
        (int(item["step"]) for item in records if item["modified"]), None
    )
    verified = [
        item
        for item in records
        if item["verification"]["main_env_post_step_checked"]
    ]
    return {
        "status": (
            "complete"
            if all(item["qp"]["valid"] and item["verification"]["accepted"] for item in records)
            else "method_failure"
        ),
        "step_count": len(records),
        "constraint_count_per_step": sorted(
            {int(item["constraint_count"]) for item in records}
        ),
        "all_qps_valid": all(item["qp"]["valid"] for item in records),
        "all_candidates_verified_safe": all(
            item["verification"]["accepted"] for item in records
        ),
        "all_executed_transitions_match_clone": len(verified) == len(records),
        "material_intervention_count": sum(bool(item["modified"]) for item in records),
        "first_material_intervention_step": first_intervention,
        "maximum_main_vs_probe_state_error": (
            None
            if not verified
            else max(
                float(item["verification"]["main_vs_probe_next_state_max_abs_error"])
                for item in verified
            )
        ),
        "filter_timing": stats(
            [float(item["timing"]["total_filter_wall_seconds"]) for item in records]
        ),
        "qp_timing": stats(
            [float(item["qp"]["diagnostics"]["timing"]["total_wall_seconds"]) for item in records]
        ),
        "simulator_probe_step_count": int(
            sum(
                1
                + int(item["finite_difference"]["probe_count"])
                + len(item["verification"]["attempts"])
                + (1 if item["verification"]["first_step_nominal_repeatability"] else 0)
                for item in records
            )
        ),
    }
