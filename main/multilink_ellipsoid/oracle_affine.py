"""One-state oracle-affine audit for the E05 distal-link false-safe event.

The ordinary AEGIS and existing opt-in filters never import this module.  It
measures every internal MuJoCo state visited by one complete OSC ``env.step``,
checks whether raw L5--L7 contact is represented by the registered ellipsoid
geometry, and gives the proposed action-affine neural model an oracle version
of its intended output before any learned model is trained.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from .barrier import support_gap
from .obstacle_primitives import minimum_union_support_gaps
from .qp import MultiConstraintQp
from .rollout import (
    ClonedSimulatorStepProbe,
    _base_env,
    _dynamic_state_vector,
)
from .shadow import (
    DISTAL_SLABBED_SCHEMA,
    MultilinkEllipsoidShadow,
    _numpy,
    _released_aegis_end_effector_ellipsoid,
)


ORACLE_AFFINE_SCHEMA = "vlsa_distal_oracle_affine_e05.v1"
ORACLE_AFFINE_RESULT_SCHEMA = "vlsa_distal_oracle_affine_e05_result.v1"
MARGIN8_AFFINE_SCHEMA = "vlsa_distal_oracle_affine_margin8mm_e05.v1"
MARGIN8_AFFINE_RESULT_SCHEMA = (
    "vlsa_distal_oracle_affine_margin8mm_e05_result.v1"
)
CONSTRAINT_ORDER = (
    "L5_part_0",
    "L5_part_1",
    "L5_part_2",
    "L6_part_0",
    "L6_part_1",
    "L7_part_0",
    "L7_part_1",
    "released_AEGIS_EE_proxy",
)
_PROTECTED_BODY_NAMES = ("robot0_link5", "robot0_link6", "robot0_link7")


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


def load_oracle_affine_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("oracle-affine config is invalid JSON") from error
    required = {
        "schema_version",
        "protocol_id",
        "case_ids",
        "claim_scope",
        "nominal_action_source",
        "audit_step",
        "protected_geometry",
        "candidate_set",
        "affine_model",
        "optimizer",
        "substep_measurement",
        "decision_gate",
        "false_safe_source",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("oracle-affine config keys differ")
    if config["schema_version"] not in (
        ORACLE_AFFINE_SCHEMA,
        MARGIN8_AFFINE_SCHEMA,
    ):
        raise ValueError("oracle-affine schema differs")
    margin_test = config["schema_version"] == MARGIN8_AFFINE_SCHEMA
    expected_protocol = (
        "vlsa-distal-oracle-affine-margin8mm-e05-v1"
        if margin_test
        else "vlsa-distal-oracle-affine-e05-v1"
    )
    if config["protocol_id"] != expected_protocol:
        raise ValueError("oracle-affine protocol differs")
    if config["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("oracle-affine audit must select only primary E05")
    if config["nominal_action_source"] != "immutable_job_37109_executed_sitl_action":
        raise ValueError("oracle-affine nominal action source differs")
    if config["audit_step"] != 192:
        raise ValueError("oracle-affine audit step differs")
    if config["false_safe_source"] != {
        "audit_step_endpoint_clearance_semantics": "all_eight_positive",
        "audit_step_raw_contact": "direct_robot0_link6_collision",
        "file_sha256": (
            "d79a28585e74cece727fbc3d3e6a72eb1647a782cae4962ec3af9045f448e603"
        ),
        "result_payload_sha256": (
            "2a4ecd5ff79a363141e0927d1dfbcfb26248ec9bb5e2bc123cc8f879c3395a53"
        ),
        "slurm_job_id": "37109",
        "source": "immutable_completed_distal_sitl_action_ledger",
    }:
        raise ValueError("oracle-affine false-safe source differs")
    if config["protected_geometry"] != {
        "distal_constraint_count": 7,
        "end_effector_constraint_count": 1,
        "source": (
            "accepted_v4_seven_slab_bounds_plus_unchanged_released_aegis_"
            "ee_proxy_and_frozen_released_aegis_obstacle_mvee"
        ),
    }:
        raise ValueError("oracle-affine protected geometry differs")
    if config["candidate_set"] != {
        "action_dimensions": [0, 1, 2],
        "action_limit": 1.0,
        "central_difference_action": 0.1,
        "global_lattice_values": [-1.0, 0.0, 1.0],
        "include_global_lattice": True,
        "include_reverse_nominal": True,
        "include_stop": True,
        "local_offset_magnitudes": [0.25, 0.5],
    }:
        raise ValueError("oracle-affine candidate set differs")
    if config["affine_model"] != {
        "clearance_target_m": 0.008 if margin_test else 0.0,
        "fit": "nominal_anchored_least_squares",
        "one_sided_error_padding_m": 1.0e-6,
        "trust_region_linf_action": 0.5,
    }:
        raise ValueError("oracle-affine model contract differs")
    optimizer = config["optimizer"]
    if set(optimizer) != {
        "bound_tolerance_action",
        "eps_abs",
        "eps_rel",
        "max_iter",
        "residual_tolerance",
    }:
        raise ValueError("oracle-affine optimizer keys differ")
    for key in (
        "bound_tolerance_action",
        "eps_abs",
        "eps_rel",
        "residual_tolerance",
    ):
        value = optimizer[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("optimizer.%s must be finite and positive" % key)
    if isinstance(optimizer["max_iter"], bool) or optimizer["max_iter"] < 1:
        raise ValueError("optimizer.max_iter must be positive")
    if config["substep_measurement"] != {
        "capture_hook": (
            "robosuite_update_observables_after_every_internal_mujoco_step"
        ),
        "contact_proxy_quadratic_tolerance": 1.0e-6,
        "raw_contact_distance_threshold_m": 0.0,
        "transient_obstacle_l1_limit_m": 1.0e-4,
    }:
        raise ValueError("oracle-affine substep measurement differs")
    if config["decision_gate"] != {
        "affine_candidate_false_safe_count": 0,
        "geometry_contact_witness": (
            "every_raw_L5_L6_L7_contact_point_inside_corresponding_robot_"
            "union_and_frozen_obstacle_mvee_with_nonpositive_support_gap"
        ),
        "local_controllability": (
            "at_least_one_raw_safe_and_proxy_safe_candidate_inside_trust_region"
        ),
        "qp_execution": (
            "valid_eight_constraint_qp_candidate_is_raw_safe_and_proxy_safe_"
            "under_exact_substep_trace"
        ),
    }:
        raise ValueError("oracle-affine decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def oracle_candidate_xyz(nominal_xyz: Any, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen, deduplicated local and global candidate set."""

    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("oracle-affine nominal XYZ is invalid")
    settings = config["candidate_set"]
    limit = float(settings["action_limit"])
    candidates = [("nominal", nominal.copy())]
    delta = float(settings["central_difference_action"])
    for dimension in range(3):
        for sign, label in ((-1.0, "minus"), (1.0, "plus")):
            value = nominal.copy()
            value[dimension] = np.clip(value[dimension] + sign * delta, -limit, limit)
            candidates.append(("central_d%d_%s" % (dimension, label), value))
    for magnitude in settings["local_offset_magnitudes"]:
        amount = float(magnitude)
        for offset in itertools.product((-amount, 0.0, amount), repeat=3):
            if offset == (0.0, 0.0, 0.0):
                continue
            value = np.clip(nominal + np.asarray(offset, dtype=np.float64), -limit, limit)
            candidates.append(("local_%.3g" % amount, value))
    if settings["include_global_lattice"]:
        for values in itertools.product(settings["global_lattice_values"], repeat=3):
            candidates.append(("global_lattice", np.asarray(values, dtype=np.float64)))
    if settings["include_stop"]:
        candidates.append(("stop", np.zeros(3, dtype=np.float64)))
    if settings["include_reverse_nominal"]:
        candidates.append(("reverse_nominal", np.clip(-nominal, -limit, limit)))
    seen = set()
    output = []
    for source, value in candidates:
        xyz = np.asarray(value, dtype=np.float64)
        key = tuple(float(item) for item in xyz)
        if key in seen:
            continue
        seen.add(key)
        output.append({"source": source, "xyz": xyz})
    return output


def _normalized_quadratic(point_world: Any, ellipsoid: Any) -> float:
    np = _numpy()
    point = np.asarray(point_world, dtype=np.float64)
    local = (point - ellipsoid.center) @ ellipsoid.rotation
    return float(np.sum((local / ellipsoid.semiaxes_m) ** 2))


def _body_lineage(model: Any, body_id: int) -> set[int]:
    output = set()
    current = int(body_id)
    while current >= 0 and current not in output:
        output.add(current)
        if current == 0:
            break
        current = int(model.body_parentid[current])
    return output


def _obstacle_root_body_id(model: Any, active_obstacle_name: str) -> int:
    for name in (str(active_obstacle_name), "%s_main" % active_obstacle_name):
        try:
            return int(model.body_name2id(name))
        except (KeyError, ValueError):
            continue
    raise ValueError("active obstacle body is unavailable")


class SubstepEightConstraintProbe(ClonedSimulatorStepProbe):
    """Capture every Robosuite internal physics substep for one cloned action."""

    def __init__(
        self,
        probe_env: Any,
        geometry: MultilinkEllipsoidShadow,
        *,
        active_obstacle_name: str,
        quadratic_tolerance: float,
        contact_distance_threshold_m: float,
        obstacle_primitive_union: Optional[Any] = None,
    ) -> None:
        super().__init__(probe_env, geometry, clearance_m=0.0)
        if geometry.config.get("schema_version") != DISTAL_SLABBED_SCHEMA:
            raise ValueError("substep audit requires accepted v4 slab geometry")
        self.active_obstacle_name = str(active_obstacle_name)
        self.quadratic_tolerance = float(quadratic_tolerance)
        self.contact_distance_threshold_m = float(contact_distance_threshold_m)
        self.obstacle_primitive_union = obstacle_primitive_union

    def _ellipsoids(self, env: Any) -> list[Any]:
        links = self.geometry._slabbed_links(env)
        links.append(_released_aegis_end_effector_ellipsoid(env))
        if len(links) != 8:
            raise ValueError("substep audit geometry must contain eight ellipsoids")
        return links

    def _obstacles(self, env: Any) -> list[Any]:
        if self.obstacle_primitive_union is None:
            return [self.geometry.obstacle]
        output = self.obstacle_primitive_union.ellipsoids(env)
        if not output:
            raise ValueError("obstacle primitive union is empty")
        return output

    def clearances(self, env: Any) -> Any:
        np = _numpy()
        values = np.asarray(
            minimum_union_support_gaps(
                self._ellipsoids(env), self._obstacles(env)
            ),
            dtype=np.float64,
        )
        if values.shape != (8,) or not np.all(np.isfinite(values)):
            raise ValueError("substep clearance vector is invalid")
        return values

    def _contact_events(
        self,
        env: Any,
        links: Sequence[Any],
        obstacles: Sequence[Any],
    ) -> list[dict[str, Any]]:
        np = _numpy()
        model = env.sim.model
        data = env.sim.data
        obstacle_id = _obstacle_root_body_id(model, self.active_obstacle_name)
        protected_ids = {
            int(model.body_name2id(name)): name for name in _PROTECTED_BODY_NAMES
        }
        events = []
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            distance = float(contact.dist)
            if distance > self.contact_distance_threshold_m:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            if body1 in protected_ids and obstacle_id in _body_lineage(model, body2):
                protected_geom, obstacle_geom = geom1, geom2
                protected_body = protected_ids[body1]
            elif body2 in protected_ids and obstacle_id in _body_lineage(model, body1):
                protected_geom, obstacle_geom = geom2, geom1
                protected_body = protected_ids[body2]
            else:
                continue
            position = np.asarray(contact.pos, dtype=np.float64).copy()
            matching = [item for item in links[:7] if item.body_name == protected_body]
            if not matching:
                raise ValueError("contacted protected body lacks a registered ellipsoid")
            protected_quadratics = [
                _normalized_quadratic(position, item) for item in matching
            ]
            obstacle_quadratics = [
                _normalized_quadratic(position, item) for item in obstacles
            ]
            obstacle_minimum_index = int(np.argmin(obstacle_quadratics))
            obstacle_quadratic = float(obstacle_quadratics[obstacle_minimum_index])
            obstacle_source_matches = [
                (index, item)
                for index, item in enumerate(obstacles)
                if int(item.geom_id) == obstacle_geom
            ]
            if (
                self.obstacle_primitive_union is not None
                and len(obstacle_source_matches) != 1
            ):
                raise ValueError(
                    "contacted obstacle geom lacks exactly one certified primitive"
                )
            source_quadratic = (
                obstacle_quadratic
                if not obstacle_source_matches
                else float(
                    obstacle_quadratics[int(obstacle_source_matches[0][0])]
                )
            )
            body_support_gaps = [
                support_gap(robot_proxy, obstacle_proxy)
                for robot_proxy in matching
                for obstacle_proxy in obstacles
            ]
            protected_minimum = float(min(protected_quadratics))
            minimum_gap = float(min(body_support_gaps))
            protected_covered = bool(
                protected_minimum <= 1.0 + self.quadratic_tolerance
            )
            obstacle_covered = bool(
                obstacle_quadratic <= 1.0 + self.quadratic_tolerance
            )
            source_obstacle_covered = bool(
                source_quadratic <= 1.0 + self.quadratic_tolerance
            )
            nonpositive_gap = bool(minimum_gap <= self.quadratic_tolerance)
            record = {
                "contact_index": contact_index,
                "distance_m": distance,
                "position_m": position.tolist(),
                "protected_body_name": protected_body,
                "protected_geom_id": protected_geom,
                "protected_geom_name": model.geom_id2name(protected_geom),
                "obstacle_geom_id": obstacle_geom,
                "obstacle_geom_name": model.geom_id2name(obstacle_geom),
                "minimum_protected_proxy_quadratic": protected_minimum,
                "minimum_obstacle_proxy_quadratic": obstacle_quadratic,
                "minimum_obstacle_proxy_index": obstacle_minimum_index,
                "source_obstacle_proxy_quadratic": source_quadratic,
                "obstacle_primitive_count": len(obstacles),
                "minimum_body_support_gap_m": minimum_gap,
                "protected_contact_point_covered": protected_covered,
                "obstacle_contact_point_covered": obstacle_covered,
                "source_obstacle_contact_point_covered": source_obstacle_covered,
                "nonpositive_body_support_gap": nonpositive_gap,
                "geometry_consistent": bool(
                    protected_covered
                    and obstacle_covered
                    and source_obstacle_covered
                    and nonpositive_gap
                ),
            }
            if self.obstacle_primitive_union is None:
                record["frozen_obstacle_proxy_quadratic"] = obstacle_quadratic
                record["frozen_obstacle_contact_point_covered"] = obstacle_covered
            events.append(record)
        return events

    def _capture(self, env: Any, substep_index: int, phase: str) -> dict[str, Any]:
        np = _numpy()
        links = self._ellipsoids(env)
        obstacles = self._obstacles(env)
        clearances = np.asarray(
            minimum_union_support_gaps(links, obstacles), dtype=np.float64
        )
        robot = env.robots[0]
        position_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
        velocity_indexes = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
        obstacle_id = _obstacle_root_body_id(env.sim.model, self.active_obstacle_name)
        controller = robot.controller
        return {
            "substep_index": int(substep_index),
            "phase": str(phase),
            "sim_time_s": float(env.sim.data.time),
            "clearance_m": clearances.tolist(),
            "minimum_distal_clearance_m": float(np.min(clearances[:7])),
            "robot_joint_position_rad": np.asarray(
                env.sim.data.qpos[position_indexes], dtype=np.float64
            ).tolist(),
            "robot_joint_velocity_rad_s": np.asarray(
                env.sim.data.qvel[velocity_indexes], dtype=np.float64
            ).tolist(),
            "controller_goal_position_m": (
                None
                if controller.goal_pos is None
                else np.asarray(controller.goal_pos, dtype=np.float64).tolist()
            ),
            "obstacle_position_m": np.asarray(
                env.sim.data.xpos[obstacle_id], dtype=np.float64
            ).tolist(),
            "obstacle_primitive_count": len(obstacles),
            "contact_events": self._contact_events(env, links, obstacles),
        }

    def transition(self, main_env: Any, action: Sequence[float]) -> dict[str, Any]:
        np = _numpy()
        synchronization = self.synchronize(main_env)
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (7,) or not np.all(np.isfinite(command)):
            raise ValueError("oracle-affine action must be finite with length seven")
        obstacle_id = _obstacle_root_body_id(
            self.probe_env.sim.model, self.active_obstacle_name
        )
        obstacle_before = np.asarray(
            self.probe_env.sim.data.xpos[obstacle_id], dtype=np.float64
        ).copy()
        trace = [self._capture(self.probe_env, 0, "interval_start")]
        base = _base_env(self.probe_env)
        original_update = base._update_observables

        def traced_update(*args: Any, **kwargs: Any) -> Any:
            value = original_update(*args, **kwargs)
            trace.append(
                self._capture(
                    self.probe_env,
                    len(trace),
                    "after_internal_mujoco_step",
                )
            )
            return value

        base._update_observables = traced_update
        started = time.perf_counter_ns()
        try:
            observation, reward, done, _ = self.probe_env.step(command.tolist())
        finally:
            base._update_observables = original_update
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        expected_internal = int(base.control_timestep / base.model_timestep)
        if len(trace) != expected_internal + 1:
            raise ValueError(
                "captured %d states for %d internal steps"
                % (len(trace), expected_internal)
            )
        clearance_trace = np.asarray(
            [item["clearance_m"] for item in trace], dtype=np.float64
        )
        minimum_substep = np.min(clearance_trace, axis=0)
        contact_events = []
        for substep in trace:
            for event in substep["contact_events"]:
                contact_events.append(
                    {**event, "substep_index": int(substep["substep_index"])}
                )
        obstacle_positions = np.asarray(
            [item["obstacle_position_m"] for item in trace], dtype=np.float64
        )
        displacements = np.sum(np.abs(obstacle_positions - obstacle_before), axis=1)
        vector = _dynamic_state_vector(self.probe_env)
        return {
            "action": command.tolist(),
            "synchronization": synchronization,
            "expected_internal_mujoco_step_count": expected_internal,
            "captured_state_count": len(trace),
            "minimum_substep_clearance_m": minimum_substep.tolist(),
            "endpoint_clearance_m": clearance_trace[-1].tolist(),
            "minimum_distal_substep_clearance_m": float(
                np.min(minimum_substep[:7])
            ),
            "raw_protected_contact_count": len(contact_events),
            "raw_protected_contact_events": contact_events,
            "geometry_consistency": {
                "status": (
                    "inconclusive_no_raw_contact"
                    if not contact_events
                    else (
                        "passed"
                        if all(item["geometry_consistent"] for item in contact_events)
                        else "failed"
                    )
                ),
                "every_contact_point_covered": bool(
                    contact_events
                    and all(
                        item["protected_contact_point_covered"]
                        and item["obstacle_contact_point_covered"]
                        and item["source_obstacle_contact_point_covered"]
                        for item in contact_events
                    )
                ),
                "every_contact_has_nonpositive_support_gap": bool(
                    contact_events
                    and all(item["nonpositive_body_support_gap"] for item in contact_events)
                ),
            },
            "maximum_within_step_obstacle_l1_displacement_m": float(
                np.max(displacements)
            ),
            "endpoint_obstacle_l1_displacement_m": float(displacements[-1]),
            "next_eef_position_m": np.asarray(
                observation["robot0_eef_pos"], dtype=np.float64
            ).tolist(),
            "reward": float(reward),
            "done": bool(done),
            "next_state_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
            "env_step_wall_seconds": elapsed,
            "substeps": trace,
            "obstacle_proxy": {
                "mode": (
                    "frozen_released_aegis_mvee"
                    if self.obstacle_primitive_union is None
                    else "live_certified_collision_primitive_union"
                ),
                "primitive_count": int(trace[0]["obstacle_primitive_count"]),
            },
        }


def fit_nominal_anchored_affine(
    candidate_records: Sequence[Mapping[str, Any]],
    nominal_xyz: Any,
    *,
    trust_region_linf: float,
    one_sided_padding_m: float,
    clearance_target_m: float,
) -> dict[str, Any]:
    """Fit the best local affine min-substep model and calibrate it downward."""

    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    candidates = np.asarray(
        [item["candidate_xyz"] for item in candidate_records], dtype=np.float64
    )
    actual = np.asarray(
        [item["minimum_substep_clearance_m"] for item in candidate_records],
        dtype=np.float64,
    )
    if candidates.ndim != 2 or candidates.shape[1] != 3 or actual.shape != (
        candidates.shape[0],
        8,
    ):
        raise ValueError("oracle-affine candidate arrays have inconsistent shapes")
    nominal_indexes = np.flatnonzero(
        np.all(np.isclose(candidates, nominal, rtol=0.0, atol=1.0e-15), axis=1)
    )
    if nominal_indexes.size != 1:
        raise ValueError("oracle-affine candidate set lacks one exact nominal action")
    nominal_clearance = actual[int(nominal_indexes[0])]
    offsets = candidates - nominal
    fit_mask = np.max(np.abs(offsets), axis=1) <= float(trust_region_linf) + 1.0e-12
    fit_offsets = offsets[fit_mask]
    fit_targets = actual[fit_mask] - nominal_clearance
    if int(np.count_nonzero(fit_mask)) < 4 or np.linalg.matrix_rank(fit_offsets) != 3:
        raise ValueError("oracle-affine trust-region candidates do not span XYZ")
    gradient_solution, _, _, _ = np.linalg.lstsq(
        fit_offsets, fit_targets, rcond=None
    )
    gradients = gradient_solution.T
    predicted = nominal_clearance[None, :] + offsets @ gradients.T
    overprediction = predicted - actual
    padding = float(one_sided_padding_m)
    epsilon = np.maximum(0.0, np.max(overprediction[fit_mask], axis=0)) + padding
    conservative = predicted - epsilon[None, :]
    target = float(clearance_target_m)
    false_safe = np.logical_and(
        conservative[fit_mask] >= target,
        actual[fit_mask] < target,
    )
    false_safe_rows = []
    fit_indexes = np.flatnonzero(fit_mask)
    for local_index, candidate_index in enumerate(fit_indexes):
        violated = np.flatnonzero(false_safe[local_index])
        if violated.size:
            false_safe_rows.append(
                {
                    "candidate_index": int(candidate_index),
                    "constraint_indexes": [int(item) for item in violated],
                }
            )
    errors = predicted[fit_mask] - actual[fit_mask]
    proxy_safe = np.all(actual >= target, axis=1)
    return {
        "fit_candidate_count": int(np.count_nonzero(fit_mask)),
        "trust_region_candidate_indexes": [int(item) for item in fit_indexes],
        "nominal_clearance_m": nominal_clearance.tolist(),
        "gradients_m_per_action": gradients.tolist(),
        "one_sided_overprediction_bound_m": epsilon.tolist(),
        "fit_rmse_m": np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
        "fit_maximum_absolute_error_m": np.max(np.abs(errors), axis=0).tolist(),
        "maximum_unsafe_overprediction_m": np.max(overprediction[fit_mask], axis=0).tolist(),
        "candidate_false_safe_count": len(false_safe_rows),
        "candidate_false_safe_records": false_safe_rows,
        "proxy_safe_candidate_indexes": [
            int(item) for item in np.flatnonzero(proxy_safe)
        ],
        "predicted_clearance_m": predicted.tolist(),
        "conservative_lower_clearance_m": conservative.tolist(),
    }


def run_oracle_affine_audit(
    config: Mapping[str, Any],
    geometry: MultilinkEllipsoidShadow,
    main_env: Any,
    probe_env: Any,
    active_obstacle_name: str,
    nominal_action: Sequence[float],
    obstacle_primitive_union: Optional[Any] = None,
) -> dict[str, Any]:
    """Evaluate geometry, affine structure, and the exact corrected QP action."""

    np = _numpy()
    started = time.perf_counter_ns()
    nominal = np.asarray(nominal_action, dtype=np.float64)
    if nominal.shape != (7,) or not np.all(np.isfinite(nominal)):
        raise ValueError("oracle-affine nominal action must contain seven values")
    if not np.allclose(nominal[3:6], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("oracle-affine audit requires translational AEGIS action")
    measurement = config["substep_measurement"]
    probe = SubstepEightConstraintProbe(
        probe_env,
        geometry,
        active_obstacle_name=active_obstacle_name,
        quadratic_tolerance=float(
            measurement["contact_proxy_quadratic_tolerance"]
        ),
        contact_distance_threshold_m=float(
            measurement["raw_contact_distance_threshold_m"]
        ),
        obstacle_primitive_union=obstacle_primitive_union,
    )
    candidate_records = []
    transition_cache = []
    for candidate in oracle_candidate_xyz(nominal[:3], config):
        action = nominal.copy()
        action[:3] = candidate["xyz"]
        transition = probe.transition(main_env, action)
        transition_cache.append(transition)
        candidate_records.append(
            {
                "candidate_index": len(candidate_records),
                "source": candidate["source"],
                "candidate_xyz": candidate["xyz"].tolist(),
                "minimum_substep_clearance_m": list(
                    transition["minimum_substep_clearance_m"]
                ),
                "endpoint_clearance_m": list(transition["endpoint_clearance_m"]),
                "raw_protected_contact_count": int(
                    transition["raw_protected_contact_count"]
                ),
                "maximum_within_step_obstacle_l1_displacement_m": float(
                    transition["maximum_within_step_obstacle_l1_displacement_m"]
                ),
                "endpoint_obstacle_l1_displacement_m": float(
                    transition["endpoint_obstacle_l1_displacement_m"]
                ),
                "next_eef_position_m": list(transition["next_eef_position_m"]),
                "next_state_sha256": transition["next_state_sha256"],
                "geometry_consistency": dict(
                    transition["geometry_consistency"]
                ),
                "env_step_wall_seconds": float(
                    transition["env_step_wall_seconds"]
                ),
            }
        )
    model = config["affine_model"]
    affine = fit_nominal_anchored_affine(
        candidate_records,
        nominal[:3],
        trust_region_linf=float(model["trust_region_linf_action"]),
        one_sided_padding_m=float(model["one_sided_error_padding_m"]),
        clearance_target_m=float(model["clearance_target_m"]),
    )
    nominal_index = next(
        index
        for index, item in enumerate(candidate_records)
        if item["source"] == "nominal"
    )
    nominal_transition = transition_cache[nominal_index]
    limit = float(config["candidate_set"]["action_limit"])
    gradients = np.asarray(affine["gradients_m_per_action"], dtype=np.float64)
    nominal_clearance = np.asarray(affine["nominal_clearance_m"], dtype=np.float64)
    epsilon = np.asarray(
        affine["one_sided_overprediction_bound_m"], dtype=np.float64
    )
    target = float(model["clearance_target_m"])
    lower = target - nominal_clearance + gradients @ nominal[:3] + epsilon
    trust = float(model["trust_region_linf_action"])
    action_lower = np.maximum(-limit, nominal[:3] - trust)
    action_upper = np.minimum(limit, nominal[:3] + trust)
    optimizer = config["optimizer"]
    qp = MultiConstraintQp(
        eps_abs=float(optimizer["eps_abs"]),
        eps_rel=float(optimizer["eps_rel"]),
        max_iter=int(optimizer["max_iter"]),
        residual_tolerance=float(optimizer["residual_tolerance"]),
        bound_tolerance=float(optimizer["bound_tolerance_action"]),
    )
    qp_result = qp.solve(
        nominal[:3],
        np.eye(3),
        gradients,
        lower,
        action_lower,
        action_upper,
    )
    qp_transition = None
    qp_xyz = None
    if qp_result.valid and qp_result.qdot_safe is not None:
        qp_xyz = np.asarray(qp_result.qdot_safe, dtype=np.float64)
        qp_action = nominal.copy()
        qp_action[:3] = qp_xyz
        qp_transition = probe.transition(main_env, qp_action)

    transient_limit = float(measurement["transient_obstacle_l1_limit_m"])
    candidate_xyz = np.asarray(
        [item["candidate_xyz"] for item in candidate_records], dtype=np.float64
    )
    inside_trust = np.max(np.abs(candidate_xyz - nominal[:3]), axis=1) <= trust + 1.0e-12
    actual_clearance = np.asarray(
        [item["minimum_substep_clearance_m"] for item in candidate_records],
        dtype=np.float64,
    )
    raw_safe = np.asarray(
        [
            item["raw_protected_contact_count"] == 0
            and item["maximum_within_step_obstacle_l1_displacement_m"]
            <= transient_limit
            for item in candidate_records
        ],
        dtype=bool,
    )
    proxy_safe = np.all(actual_clearance >= target, axis=1)
    jointly_safe_indexes = np.flatnonzero(inside_trust & raw_safe & proxy_safe)
    nominal_geometry_pass = bool(
        nominal_transition["raw_protected_contact_count"] > 0
        and nominal_transition["geometry_consistency"]["status"] == "passed"
    )
    qp_raw_safe = bool(
        qp_transition is not None
        and qp_transition["raw_protected_contact_count"] == 0
        and qp_transition["maximum_within_step_obstacle_l1_displacement_m"]
        <= transient_limit
    )
    qp_proxy_safe = bool(
        qp_transition is not None
        and np.all(
            np.asarray(
                qp_transition["minimum_substep_clearance_m"], dtype=np.float64
            )
            >= target
        )
    )
    decision = {
        "geometry_contact_witness_pass": nominal_geometry_pass,
        "nominal_raw_contact_observed": bool(
            nominal_transition["raw_protected_contact_count"] > 0
        ),
        "local_raw_safe_candidate_exists": bool(
            np.any(inside_trust & raw_safe)
        ),
        "local_proxy_safe_candidate_exists": bool(
            np.any(inside_trust & proxy_safe)
        ),
        "local_jointly_raw_and_proxy_safe_candidate_exists": bool(
            jointly_safe_indexes.size
        ),
        "local_jointly_safe_candidate_indexes": [
            int(item) for item in jointly_safe_indexes
        ],
        "affine_candidate_false_safe_count": int(
            affine["candidate_false_safe_count"]
        ),
        "affine_candidate_gate_pass": bool(
            affine["candidate_false_safe_count"]
            == config["decision_gate"]["affine_candidate_false_safe_count"]
        ),
        "qp_valid": bool(qp_result.valid),
        "qp_exact_raw_safe": qp_raw_safe,
        "qp_exact_proxy_safe": qp_proxy_safe,
    }
    if config.get("schema_version") == MARGIN8_AFFINE_SCHEMA:
        nominal_minimum = np.asarray(
            nominal_transition["minimum_substep_clearance_m"],
            dtype=np.float64,
        )
        decision["margin_intervention"] = {
            "formula": "h_corrected_equals_h_aegis_minus_0.008_m",
            "clearance_margin_m": 0.008,
            "nominal_declared_unsafe": bool(np.any(nominal_minimum < target)),
            "qp_action_raw_safe": qp_raw_safe,
            "qp_action_margin_safe": qp_proxy_safe,
            "test_pass": bool(qp_result.valid and qp_raw_safe and qp_proxy_safe),
        }
    decision["research_direction_go"] = bool(
        decision["geometry_contact_witness_pass"]
        and decision["local_jointly_raw_and_proxy_safe_candidate_exists"]
        and decision["affine_candidate_gate_pass"]
        and decision["qp_valid"]
        and decision["qp_exact_raw_safe"]
        and decision["qp_exact_proxy_safe"]
    )
    if not decision["geometry_contact_witness_pass"]:
        decision["stop_reason"] = "geometry_authority_failed_before_transition_learning"
    elif not decision["local_jointly_raw_and_proxy_safe_candidate_exists"]:
        decision["stop_reason"] = "no_jointly_raw_and_proxy_safe_action_in_local_trust_region"
    elif not decision["affine_candidate_gate_pass"]:
        decision["stop_reason"] = "local_action_to_minimum_clearance_map_not_conservatively_affine_on_candidates"
    elif not decision["qp_valid"]:
        decision["stop_reason"] = "oracle_affine_eight_constraint_qp_infeasible"
    elif not (decision["qp_exact_raw_safe"] and decision["qp_exact_proxy_safe"]):
        decision["stop_reason"] = "oracle_affine_qp_action_failed_exact_substep_verification"
    else:
        decision["stop_reason"] = None
    return {
        "constraint_order": list(CONSTRAINT_ORDER),
        "nominal_action": nominal.tolist(),
        "candidate_count": len(candidate_records),
        "candidate_records": candidate_records,
        "nominal_substep_transition": nominal_transition,
        "affine_model": affine,
        "oracle_affine_qp": {
            "valid": bool(qp_result.valid),
            "reason": qp_result.reason,
            "candidate_xyz": None if qp_xyz is None else qp_xyz.tolist(),
            "lower": lower.tolist(),
            "action_lower": action_lower.tolist(),
            "action_upper": action_upper.tolist(),
            "diagnostics": dict(qp_result.diagnostics),
            "exact_substep_transition": qp_transition,
        },
        "decision": decision,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
