"""Fail-closed contract for the static Poisson-CBF runtime.

This module validates the *runtime numerical protocol*, not the case manifest.
It accepts one active shape plus the immutable preceding shape, and rejects
unknown fields within each version.  The parameter-block digest can therefore
be frozen before held-out evaluation and embedded in every manifest/result
without relying on mutable filenames.

The active v4 canary protects every collision-enabled surface in the
authoritative robot body tree against one selected obstacle.  Link 5/6 samples
remain in the field-bundle identity only; they are not the active shield-row
population.  The v3 link-5/6 contract remains loadable solely so immutable
historical artifacts retain their original validator.  ISSf is represented
explicitly but disabled (epsilon0 = 0) until the runtime implements and tests
that term.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "vlsa_poisson_runtime_protocol.v4"
LEGACY_SCHEMA_VERSION = "vlsa_poisson_runtime_protocol.v3"
PARAMETER_SECTIONS = (
    "workspace",
    "occupancy",
    "coverage",
    "safety",
    "poisson",
    "cbf",
    "adapter",
    "qp",
    "cadence",
    "admissibility",
    "differential_audit",
    "claim_scope",
)

REGISTERED_DIFFERENTIAL_DIRECTIONS_RAD_S = (
    (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
    (0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
    (
        1.0 / 14.0,
        2.0 / 14.0,
        3.0 / 14.0,
        4.0 / 14.0,
        5.0 / 14.0,
        6.0 / 14.0,
        7.0 / 14.0,
    ),
)
REGISTERED_DIFFERENTIAL_ETA_LADDER_S = (
    2.0e-6,
    5.0e-7,
    1.25e-7,
    3.125e-8,
    7.8125e-9,
    1.953125e-9,
    4.8828125e-10,
)
REGISTERED_BINARY64_UNIT_ROUNDOFF = 2.0 ** -53
REGISTERED_ARM_TANGENT_ROUNDTRIP_CRITERION = (
    "scalar_hinge_two_stage_exact_fraction_binary64_roundoff"
)
REGISTERED_MUJOCO_VERSION = "3.2.3"
REGISTERED_ARM_DOF_INDICES = tuple(range(7))
REGISTERED_ARM_JOINT_IDS = tuple(range(7))
REGISTERED_ARM_QPOS_INDICES = tuple(range(7))
REGISTERED_ARM_JOINT_NAMES = tuple(
    "robot0_joint%d" % index for index in range(1, 8)
)
REGISTERED_ARM_JOINT_TYPE = "hinge"
REGISTERED_LEGACY_ARM_TANGENT_TOLERANCE_RAD_S = 1.0e-10


class FeasibilityProtocolError(ValueError):
    """Raised when a protocol is ambiguous, inconsistent, or not hash-bound."""


@dataclass(frozen=True)
class ProtocolHashes:
    """Content identities returned after complete validation."""

    protocol_sha256: str
    parameter_block_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique semantic JSON representation used for identities."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise FeasibilityProtocolError(
            "protocol must contain only finite JSON values"
        ) from error
    return encoded.encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FeasibilityProtocolError("{} must be a lowercase SHA-256".format(path))
    return value


def _object(value: Any, path: str, keys: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FeasibilityProtocolError("{} must be an object".format(path))
    if any(not isinstance(key, str) for key in value):
        raise FeasibilityProtocolError(
            "{} object keys must be strings".format(path)
        )
    expected = set(keys)
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise FeasibilityProtocolError(
            "{} is missing fields: {}".format(path, ", ".join(missing))
        )
    if unknown:
        raise FeasibilityProtocolError(
            "{} has unknown fields: {}".format(path, ", ".join(unknown))
        )
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise FeasibilityProtocolError("{} must be a non-empty string".format(path))
    return value


def _literal(value: Any, expected: str, path: str) -> None:
    if value != expected or not isinstance(value, str):
        raise FeasibilityProtocolError(
            "{} must equal {!r}".format(path, expected)
        )


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise FeasibilityProtocolError("{} must be boolean".format(path))
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FeasibilityProtocolError(
            "{} must be an integer >= {}".format(path, minimum)
        )
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: Optional[float] = None,
    strict_minimum: bool = False,
    maximum: Optional[float] = None,
    strict_maximum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeasibilityProtocolError("{} must be a finite number".format(path))
    output = float(value)
    if not math.isfinite(output):
        raise FeasibilityProtocolError("{} must be a finite number".format(path))
    if minimum is not None:
        invalid = output <= minimum if strict_minimum else output < minimum
        if invalid:
            operator = ">" if strict_minimum else ">="
            raise FeasibilityProtocolError(
                "{} must be {} {}".format(path, operator, minimum)
            )
    if maximum is not None:
        invalid = output >= maximum if strict_maximum else output > maximum
        if invalid:
            operator = "<" if strict_maximum else "<="
            raise FeasibilityProtocolError(
                "{} must be {} {}".format(path, operator, maximum)
            )
    return output


def _number_vector(
    value: Any,
    path: str,
    length: int,
    *,
    strict_positive: bool = False,
) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise FeasibilityProtocolError(
            "{} must be an array of length {}".format(path, length)
        )
    return tuple(
        _number(
            item,
            "{}[{}]".format(path, index),
            minimum=0.0 if strict_positive else None,
            strict_minimum=strict_positive,
        )
        for index, item in enumerate(value)
    )


def _integer_vector(
    value: Any, path: str, length: int, minimum: int
) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise FeasibilityProtocolError(
            "{} must be an array of length {}".format(path, length)
        )
    return tuple(
        _integer(item, "{}[{}]".format(path, index), minimum)
        for index, item in enumerate(value)
    )


def _literal_string_array(value: Any, expected: Sequence[str], path: str) -> None:
    if not isinstance(value, list) or value != list(expected):
        raise FeasibilityProtocolError(
            "{} must equal {!r}".format(path, list(expected))
        )
    if any(not isinstance(item, str) for item in value):
        raise FeasibilityProtocolError("{} must contain strings".format(path))


def _close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1.0e-12, abs_tol=1.0e-12)


def _validate_semantics(protocol: Mapping[str, Any]) -> None:
    top = _object(
        protocol,
        "protocol",
        (
            "schema_version",
            "protocol_id",
            "workspace",
            "occupancy",
            "coverage",
            "safety",
            "poisson",
            "cbf",
            "adapter",
            "qp",
            "cadence",
            "admissibility",
            "differential_audit",
            "claim_scope",
            "parameter_selection",
        ),
    )
    schema_version = top["schema_version"]
    if schema_version not in (SCHEMA_VERSION, LEGACY_SCHEMA_VERSION):
        raise FeasibilityProtocolError(
            "protocol.schema_version must equal {!r} or immutable legacy {!r}".format(
                SCHEMA_VERSION, LEGACY_SCHEMA_VERSION
            )
        )
    _string(top["protocol_id"], "protocol.protocol_id")

    workspace = _object(
        top["workspace"],
        "protocol.workspace",
        (
            "minimum_m",
            "maximum_m",
            "grid_shape_vertices",
            "grid_spacing_m",
            "boundary_condition",
            "outside_workspace_policy",
        ),
    )
    lower = _number_vector(workspace["minimum_m"], "workspace.minimum_m", 3)
    upper = _number_vector(workspace["maximum_m"], "workspace.maximum_m", 3)
    shape = _integer_vector(
        workspace["grid_shape_vertices"], "workspace.grid_shape_vertices", 3, 3
    )
    spacing = _number_vector(
        workspace["grid_spacing_m"],
        "workspace.grid_spacing_m",
        3,
        strict_positive=True,
    )
    _literal(
        workspace["boundary_condition"],
        "homogeneous_dirichlet_h_zero",
        "workspace.boundary_condition",
    )
    _literal(
        workspace["outside_workspace_policy"],
        "invalid_fail_closed",
        "workspace.outside_workspace_policy",
    )
    for axis in range(3):
        if upper[axis] <= lower[axis]:
            raise FeasibilityProtocolError(
                "workspace maximum must exceed minimum on every axis"
            )
        reconstructed = lower[axis] + spacing[axis] * (shape[axis] - 1)
        if not _close(reconstructed, upper[axis]):
            raise FeasibilityProtocolError(
                "workspace bounds, grid_shape_vertices, and grid_spacing_m disagree "
                "on axis {}".format(axis)
            )

    occupancy = _object(
        top["occupancy"],
        "protocol.occupancy",
        (
            "geometry_source",
            "rasterization",
            "obstacle_clearance_m",
            "outer_boundary_clearance_m",
            "buffer_application",
            "voxel_discretization_policy",
            "field_motion_model",
            "unknown_geometry_policy",
        ),
    )
    _literal(
        occupancy["geometry_source"],
        "selected_mujoco_collision_enabled_geoms",
        "occupancy.geometry_source",
    )
    _literal(
        occupancy["rasterization"],
        "closed_cell_obb_intersection",
        "occupancy.rasterization",
    )
    obstacle_clearance = _number(
        occupancy["obstacle_clearance_m"],
        "occupancy.obstacle_clearance_m",
        minimum=0.0,
    )
    outer_clearance = _number(
        occupancy["outer_boundary_clearance_m"],
        "occupancy.outer_boundary_clearance_m",
        minimum=0.0,
    )
    _literal(
        occupancy["buffer_application"],
        "exactly_once",
        "occupancy.buffer_application",
    )
    _literal(
        occupancy["voxel_discretization_policy"],
        "closed_cells_no_extra_half_diagonal",
        "occupancy.voxel_discretization_policy",
    )
    _literal(
        occupancy["field_motion_model"],
        "static_selected_obstacle",
        "occupancy.field_motion_model",
    )
    _literal(
        occupancy["unknown_geometry_policy"],
        "invalid_fail_closed",
        "occupancy.unknown_geometry_policy",
    )

    coverage = _object(
        top["coverage"],
        "protocol.coverage",
        (
            "epsilon_m",
            "ball_semantics",
            "certificate_method",
            "certificate_relation",
            "require_every_collision_surface_component",
        ),
    )
    epsilon = _number(
        coverage["epsilon_m"],
        "coverage.epsilon_m",
        minimum=0.0,
        strict_minimum=True,
    )
    _literal(
        coverage["ball_semantics"],
        "strict_open_ball",
        "coverage.ball_semantics",
    )
    _literal(
        coverage["certificate_method"],
        "analytic_triangle_lattice_covering_bound",
        "coverage.certificate_method",
    )
    _literal(
        coverage["certificate_relation"],
        "max_covering_radius_strictly_less_than_epsilon",
        "coverage.certificate_relation",
    )
    if not _boolean(
        coverage["require_every_collision_surface_component"],
        "coverage.require_every_collision_surface_component",
    ):
        raise FeasibilityProtocolError(
            "coverage must include every protected collision-surface component"
        )

    safety = _object(
        top["safety"],
        "protocol.safety",
        (
            "contact_margin_m",
            "physical_contact_definition",
            "positive_margin_contact_policy",
            "distance_accounting",
        ),
    )
    contact_margin = _number(
        safety["contact_margin_m"], "safety.contact_margin_m", minimum=0.0
    )
    _literal(
        safety["physical_contact_definition"],
        "mujoco_contact_dist_le_zero",
        "safety.physical_contact_definition",
    )
    _literal(
        safety["positive_margin_contact_policy"],
        "diagnostic_candidate_not_physical_contact",
        "safety.positive_margin_contact_policy",
    )
    _literal(
        safety["distance_accounting"],
        "D_opt_and_D_sim_are_distinct",
        "safety.distance_accounting",
    )

    poisson = _object(
        top["poisson"],
        "protocol.poisson",
        (
            "equation",
            "forcing_value",
            "boundary_value",
            "solver",
            "relaxation_omega",
            "max_iterations",
            "normalized_backward_error_tolerance",
            "residual_check_interval",
            "query_interpolation",
            "invalid_query_policy",
            "zero_cell_policy",
        ),
    )
    _literal(
        poisson["equation"],
        "minus_laplacian_h_equals_forcing",
        "poisson.equation",
    )
    forcing = _number(poisson["forcing_value"], "poisson.forcing_value")
    if forcing <= 0.0:
        raise FeasibilityProtocolError(
            "poisson.forcing_value must be strictly positive for positive interior h"
        )
    boundary_value = _number(
        poisson["boundary_value"], "poisson.boundary_value"
    )
    if boundary_value != 0.0:
        raise FeasibilityProtocolError("poisson.boundary_value must be exactly zero")
    _literal(poisson["solver"], "red_black_sor", "poisson.solver")
    _number(
        poisson["relaxation_omega"],
        "poisson.relaxation_omega",
        minimum=0.0,
        strict_minimum=True,
        maximum=2.0,
        strict_maximum=True,
    )
    maximum_iterations = _integer(
        poisson["max_iterations"], "poisson.max_iterations", 1
    )
    _number(
        poisson["normalized_backward_error_tolerance"],
        "poisson.normalized_backward_error_tolerance",
        minimum=0.0,
        strict_minimum=True,
    )
    residual_interval = _integer(
        poisson["residual_check_interval"],
        "poisson.residual_check_interval",
        1,
    )
    if residual_interval > maximum_iterations:
        raise FeasibilityProtocolError(
            "poisson.residual_check_interval may not exceed max_iterations"
        )
    _literal(
        poisson["query_interpolation"],
        "trilinear_discrete_C0",
        "poisson.query_interpolation",
    )
    _literal(
        poisson["invalid_query_policy"],
        "terminate_retain_case",
        "poisson.invalid_query_policy",
    )
    _literal(
        poisson["zero_cell_policy"],
        "nonregular_zero_cell_invalid_fail_closed",
        "poisson.zero_cell_policy",
    )

    cbf = _object(
        top["cbf"],
        "protocol.cbf",
        (
            "alpha_gain_per_s",
            "constraint_form",
            "time_derivative_policy",
            "issf_mode",
            "issf_epsilon0",
            "unsafe_initial_policy",
        ),
    )
    _number(
        cbf["alpha_gain_per_s"],
        "cbf.alpha_gain_per_s",
        minimum=0.0,
        strict_minimum=True,
    )
    _literal(
        cbf["constraint_form"],
        "grad_h_J_qdot_plus_alpha_h_ge_issf_epsilon0_grad_norm_squared",
        "cbf.constraint_form",
    )
    _literal(
        cbf["time_derivative_policy"],
        "zero_only_while_static_field_admissible",
        "cbf.time_derivative_policy",
    )
    _literal(
        cbf["issf_mode"],
        "disabled_initial_pilot",
        "cbf.issf_mode",
    )
    epsilon0 = _number(cbf["issf_epsilon0"], "cbf.issf_epsilon0", minimum=0.0)
    if epsilon0 != 0.0:
        raise FeasibilityProtocolError(
            "v1 has no validated ISSf runtime; issf_epsilon0 must be exactly zero"
        )
    _literal(
        cbf["unsafe_initial_policy"],
        "terminate_retain_case",
        "cbf.unsafe_initial_policy",
    )

    adapter = _object(
        top["adapter"],
        "protocol.adapter",
        (
            "dls_damping",
            "position_gain_per_s",
            "orientation_gain_per_s",
            "translation_action_scale_m",
            "input_clip_abs",
            "orientation_target",
            "physical_joint_velocity_scale_rad_s",
            "gripper_policy",
        ),
    )
    for field in ("dls_damping", "position_gain_per_s", "orientation_gain_per_s"):
        _number(
            adapter[field],
            "adapter.{}".format(field),
            minimum=0.0,
            strict_minimum=True,
        )
    translation_scale = _number(
        adapter["translation_action_scale_m"],
        "adapter.translation_action_scale_m",
        minimum=0.0,
        strict_minimum=True,
    )
    if not _close(translation_scale, 0.05):
        raise FeasibilityProtocolError(
            "adapter.translation_action_scale_m must preserve released OSC scale 0.05"
        )
    input_clip = _number(
        adapter["input_clip_abs"],
        "adapter.input_clip_abs",
        minimum=0.0,
        strict_minimum=True,
    )
    if not _close(input_clip, 1.0):
        raise FeasibilityProtocolError(
            "adapter.input_clip_abs must preserve released OSC clipping at 1"
        )
    _literal(
        adapter["orientation_target"],
        "hold_pose_at_high_level_action_start",
        "adapter.orientation_target",
    )
    velocity_scale = _number(
        adapter["physical_joint_velocity_scale_rad_s"],
        "adapter.physical_joint_velocity_scale_rad_s",
        minimum=0.0,
        strict_minimum=True,
    )
    if not _close(velocity_scale, 0.5):
        raise FeasibilityProtocolError(
            "adapter physical velocity scale must match Robosuite JOINT_VELOCITY 0.5"
        )
    _literal(
        adapter["gripper_policy"],
        "unchanged_vla_command",
        "adapter.gripper_policy",
    )

    qp = _object(
        top["qp"],
        "protocol.qp",
        (
            "solver",
            "objective",
            "weight_diagonal",
            "hard_cbf_constraints",
            "slack_enabled",
            "velocity_lower_rad_s",
            "velocity_upper_rad_s",
            "joint_limit_alpha_per_s",
            "joint_position_margin_rad",
            "eps_abs",
            "eps_rel",
            "max_iterations",
            "postcheck_cbf_tolerance",
            "postcheck_bound_tolerance_rad_s",
            "row_scaling",
            "failure_policy",
        ),
    )
    _literal(qp["solver"], "osqp", "qp.solver")
    _literal(
        qp["objective"],
        "minimize_weighted_squared_qdot_deviation",
        "qp.objective",
    )
    weights = _number_vector(
        qp["weight_diagonal"], "qp.weight_diagonal", 7, strict_positive=True
    )
    if len(weights) != 7:  # pragma: no cover - explicit readability invariant
        raise FeasibilityProtocolError("QP must operate on exactly seven joints")
    if not _boolean(qp["hard_cbf_constraints"], "qp.hard_cbf_constraints"):
        raise FeasibilityProtocolError("CBF constraints must be hard")
    if _boolean(qp["slack_enabled"], "qp.slack_enabled"):
        raise FeasibilityProtocolError("v1 does not permit safety slack")
    lower_velocity = _number_vector(
        qp["velocity_lower_rad_s"], "qp.velocity_lower_rad_s", 7
    )
    upper_velocity = _number_vector(
        qp["velocity_upper_rad_s"], "qp.velocity_upper_rad_s", 7
    )
    for index, (lower_bound, upper_bound) in enumerate(
        zip(lower_velocity, upper_velocity)
    ):
        if lower_bound > 0.0 or upper_bound < 0.0 or lower_bound >= upper_bound:
            raise FeasibilityProtocolError(
                "QP velocity bounds for joint {} must contain zero and be nonempty".format(
                    index
                )
            )
        if abs(lower_bound) > velocity_scale or abs(upper_bound) > velocity_scale:
            raise FeasibilityProtocolError(
                "QP velocity bounds exceed the registered physical actuator scale"
            )
    _number(
        qp["joint_limit_alpha_per_s"],
        "qp.joint_limit_alpha_per_s",
        minimum=0.0,
        strict_minimum=True,
    )
    position_margin = _number(
        qp["joint_position_margin_rad"],
        "qp.joint_position_margin_rad",
        minimum=0.0,
    )
    for field in (
        "eps_abs",
        "eps_rel",
        "postcheck_cbf_tolerance",
        "postcheck_bound_tolerance_rad_s",
    ):
        _number(
            qp[field],
            "qp.{}".format(field),
            minimum=0.0,
            strict_minimum=True,
        )
    _integer(qp["max_iterations"], "qp.max_iterations", 1)
    _literal(qp["row_scaling"], "enabled", "qp.row_scaling")
    _literal(
        qp["failure_policy"],
        "terminate_retain_case_no_cached_command",
        "qp.failure_policy",
    )

    cadence = _object(
        top["cadence"],
        "protocol.cadence",
        (
            "physics_timestep_s",
            "high_level_frequency_hz",
            "shadow",
            "active",
        ),
    )
    physics_dt = _number(
        cadence["physics_timestep_s"],
        "cadence.physics_timestep_s",
        minimum=0.0,
        strict_minimum=True,
    )
    high_level_hz = _number(
        cadence["high_level_frequency_hz"],
        "cadence.high_level_frequency_hz",
        minimum=0.0,
        strict_minimum=True,
    )
    if not _close(physics_dt, 0.002) or not _close(high_level_hz, 20.0):
        raise FeasibilityProtocolError(
            "v1 cadence must preserve 0.002 s physics and 20 Hz VLA actions"
        )
    shadow = _object(
        cadence["shadow"],
        "cadence.shadow",
        (
            "mode",
            "filter_updates_per_high_level_action",
            "physics_substeps_per_filter_update",
        ),
    )
    _literal(
        shadow["mode"],
        "monitor_only_no_action_mutation",
        "cadence.shadow.mode",
    )
    shadow_updates = _integer(
        shadow["filter_updates_per_high_level_action"],
        "cadence.shadow.filter_updates_per_high_level_action",
        1,
    )
    shadow_substeps = _integer(
        shadow["physics_substeps_per_filter_update"],
        "cadence.shadow.physics_substeps_per_filter_update",
        1,
    )
    active = _object(
        cadence["active"],
        "cadence.active",
        (
            "mode",
            "filter_frequency_hz",
            "filter_updates_per_high_level_action",
            "physics_substeps_per_filter_update",
        ),
    )
    _literal(
        active["mode"],
        "recompute_joint_velocity_each_filter_update",
        "cadence.active.mode",
    )
    filter_hz = _number(
        active["filter_frequency_hz"],
        "cadence.active.filter_frequency_hz",
        minimum=0.0,
        strict_minimum=True,
    )
    active_updates = _integer(
        active["filter_updates_per_high_level_action"],
        "cadence.active.filter_updates_per_high_level_action",
        1,
    )
    active_substeps = _integer(
        active["physics_substeps_per_filter_update"],
        "cadence.active.physics_substeps_per_filter_update",
        1,
    )
    if (shadow_updates, shadow_substeps) != (1, 25):
        raise FeasibilityProtocolError("shadow cadence must be exactly 1 x 25")
    if (active_updates, active_substeps) != (5, 5):
        raise FeasibilityProtocolError("active cadence must be exactly 5 x 5")
    if not _close(filter_hz, 100.0):
        raise FeasibilityProtocolError("active filter frequency must be 100 Hz")
    high_level_substeps = 1.0 / (physics_dt * high_level_hz)
    if not _close(float(shadow_updates * shadow_substeps), high_level_substeps):
        raise FeasibilityProtocolError("shadow cadence does not span one VLA action")
    if not _close(float(active_updates * active_substeps), high_level_substeps):
        raise FeasibilityProtocolError("active cadence does not span one VLA action")
    if not _close(1.0 / (physics_dt * active_substeps), filter_hz):
        raise FeasibilityProtocolError(
            "active filter frequency disagrees with physics substeps"
        )

    admissibility = _object(
        top["admissibility"],
        "protocol.admissibility",
        (
            "max_state_restore_qpos_error_rad",
            "max_state_restore_qvel_error_rad_s",
            "joint_position_margin_rad",
            "max_joint_velocity_tracking_linf_rad_s",
            "max_joint_velocity_tracking_rmse_rad_s",
            "max_selected_body_linear_speed_m_s",
            "max_selected_body_angular_speed_rad_s",
            "max_selected_geom_translation_drift_m",
            "max_selected_geom_rotation_drift_rad",
            "max_selected_geom_surface_drift_m",
            "maximum_invalid_field_queries",
            "require_safe_initial_samples",
            "static_field_refresh_policy",
            "violation_policy",
        ),
    )
    restore_qpos = _number(
        admissibility["max_state_restore_qpos_error_rad"],
        "admissibility.max_state_restore_qpos_error_rad",
        minimum=0.0,
    )
    restore_qvel = _number(
        admissibility["max_state_restore_qvel_error_rad_s"],
        "admissibility.max_state_restore_qvel_error_rad_s",
        minimum=0.0,
    )
    if restore_qpos == 0.0 or restore_qvel == 0.0:
        raise FeasibilityProtocolError(
            "state-restore tolerances must be positive finite numerical tolerances"
        )
    admissible_position_margin = _number(
        admissibility["joint_position_margin_rad"],
        "admissibility.joint_position_margin_rad",
        minimum=0.0,
    )
    if not _close(admissible_position_margin, position_margin):
        raise FeasibilityProtocolError(
            "QP and admissibility joint-position margins must match"
        )
    tracking_linf = _number(
        admissibility["max_joint_velocity_tracking_linf_rad_s"],
        "admissibility.max_joint_velocity_tracking_linf_rad_s",
        minimum=0.0,
        strict_minimum=True,
    )
    tracking_rmse = _number(
        admissibility["max_joint_velocity_tracking_rmse_rad_s"],
        "admissibility.max_joint_velocity_tracking_rmse_rad_s",
        minimum=0.0,
        strict_minimum=True,
    )
    if tracking_rmse > tracking_linf:
        raise FeasibilityProtocolError(
            "tracking RMSE threshold may not exceed the L-infinity threshold"
        )
    if restore_qpos > position_margin:
        raise FeasibilityProtocolError(
            "qpos restore tolerance may not exceed the joint-position margin"
        )
    if restore_qvel > tracking_linf:
        raise FeasibilityProtocolError(
            "qvel restore tolerance may not exceed the tracking L-infinity threshold"
        )
    _number(
        admissibility["max_selected_body_linear_speed_m_s"],
        "admissibility.max_selected_body_linear_speed_m_s",
        minimum=0.0,
    )
    _number(
        admissibility["max_selected_body_angular_speed_rad_s"],
        "admissibility.max_selected_body_angular_speed_rad_s",
        minimum=0.0,
    )
    translation_drift = _number(
        admissibility["max_selected_geom_translation_drift_m"],
        "admissibility.max_selected_geom_translation_drift_m",
        minimum=0.0,
    )
    _number(
        admissibility["max_selected_geom_rotation_drift_rad"],
        "admissibility.max_selected_geom_rotation_drift_rad",
        minimum=0.0,
    )
    surface_drift = _number(
        admissibility["max_selected_geom_surface_drift_m"],
        "admissibility.max_selected_geom_surface_drift_m",
        minimum=0.0,
    )
    if surface_drift < translation_drift:
        raise FeasibilityProtocolError(
            "surface-drift tolerance must cover translation-drift tolerance"
        )
    if _integer(
        admissibility["maximum_invalid_field_queries"],
        "admissibility.maximum_invalid_field_queries",
        0,
    ) != 0:
        raise FeasibilityProtocolError(
            "v1 fail-closed execution permits zero invalid field queries"
        )
    if not _boolean(
        admissibility["require_safe_initial_samples"],
        "admissibility.require_safe_initial_samples",
    ):
        raise FeasibilityProtocolError("initial protected samples must be safe")
    _literal(
        admissibility["static_field_refresh_policy"],
        "never_refresh_terminate_on_drift",
        "admissibility.static_field_refresh_policy",
    )
    _literal(
        admissibility["violation_policy"],
        "terminate_retain_invalid_not_collision_free",
        "admissibility.violation_policy",
    )

    required_clearance = epsilon + contact_margin + surface_drift
    if obstacle_clearance + 1.0e-15 < required_clearance:
        raise FeasibilityProtocolError(
            "obstacle clearance must cover epsilon, contact margin, and admitted "
            "static-field surface drift"
        )
    if outer_clearance + 1.0e-15 < epsilon + contact_margin:
        raise FeasibilityProtocolError(
            "outer-boundary clearance must cover epsilon and contact margin"
        )
    for axis, extent in enumerate(upper[index] - lower[index] for index in range(3)):
        if 2.0 * outer_clearance >= extent:
            raise FeasibilityProtocolError(
                "outer-boundary clearance removes the workspace on axis {}".format(axis)
            )

    differential = _object(
        top["differential_audit"],
        "protocol.differential_audit",
        (
            "state_source",
            "perturbation_integrator",
            "point_jacobian_delta_rad",
            "point_jacobian_absolute_tolerance_m_per_rad",
            "point_jacobian_relative_tolerance",
            "point_jacobian_near_zero_frobenius_m_per_rad",
            "arm_tangent_roundtrip_criterion",
            "binary64_unit_roundoff",
            "expected_mujoco_version",
            "expected_arm_dof_indices",
            "expected_arm_joint_ids",
            "expected_arm_qpos_indices",
            "expected_arm_joint_names",
            "required_arm_joint_type",
            "legacy_arm_tangent_reconstruction_tolerance_rad_s",
            "nonarm_tangent_leakage_tolerance_rad_s",
            "joint_velocity_directions_rad_s",
            "coupled_eta_ladder_s",
            "coupled_absolute_tolerance_m2_per_s",
            "coupled_relative_tolerance",
            "coupled_near_zero_m2_per_s",
            "same_trilinear_cell_required",
            "required_direction_count_per_sample",
            "stencil_selection_policy",
            "finite_difference_resolution_policy",
            "failure_policy",
        ),
    )
    _literal(
        differential["state_source"],
        "settled_mujoco_mjstate_integration_clone",
        "differential_audit.state_source",
    )
    _literal(
        differential["perturbation_integrator"],
        "mujoco_mj_integratePos_full_nv_tangent",
        "differential_audit.perturbation_integrator",
    )
    _literal(
        differential["arm_tangent_roundtrip_criterion"],
        REGISTERED_ARM_TANGENT_ROUNDTRIP_CRITERION,
        "differential_audit.arm_tangent_roundtrip_criterion",
    )
    _literal(
        differential["expected_mujoco_version"],
        REGISTERED_MUJOCO_VERSION,
        "differential_audit.expected_mujoco_version",
    )
    _literal(
        differential["required_arm_joint_type"],
        REGISTERED_ARM_JOINT_TYPE,
        "differential_audit.required_arm_joint_type",
    )
    for field in (
        "point_jacobian_delta_rad",
        "point_jacobian_absolute_tolerance_m_per_rad",
        "point_jacobian_relative_tolerance",
        "point_jacobian_near_zero_frobenius_m_per_rad",
        "binary64_unit_roundoff",
        "legacy_arm_tangent_reconstruction_tolerance_rad_s",
        "nonarm_tangent_leakage_tolerance_rad_s",
        "coupled_absolute_tolerance_m2_per_s",
        "coupled_relative_tolerance",
        "coupled_near_zero_m2_per_s",
    ):
        _number(
            differential[field],
            "differential_audit.%s" % field,
            minimum=0.0,
            strict_minimum=True,
        )
    if differential["binary64_unit_roundoff"] != REGISTERED_BINARY64_UNIT_ROUNDOFF:
        raise FeasibilityProtocolError(
            "differential_audit.binary64_unit_roundoff differs from binary64"
        )
    if (
        differential["legacy_arm_tangent_reconstruction_tolerance_rad_s"]
        != REGISTERED_LEGACY_ARM_TANGENT_TOLERANCE_RAD_S
    ):
        raise FeasibilityProtocolError(
            "differential_audit legacy tangent diagnostic must remain 1e-10 rad/s"
        )
    for field, expected in (
        ("expected_arm_dof_indices", REGISTERED_ARM_DOF_INDICES),
        ("expected_arm_joint_ids", REGISTERED_ARM_JOINT_IDS),
        ("expected_arm_qpos_indices", REGISTERED_ARM_QPOS_INDICES),
    ):
        observed = differential[field]
        if (
            not isinstance(observed, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in observed
            )
            or tuple(observed) != expected
        ):
            raise FeasibilityProtocolError(
                "differential_audit.%s differs from the registered Panda topology"
                % field
            )
    if (
        not isinstance(differential["expected_arm_joint_names"], list)
        or tuple(differential["expected_arm_joint_names"])
        != REGISTERED_ARM_JOINT_NAMES
    ):
        raise FeasibilityProtocolError(
            "differential_audit.expected_arm_joint_names differs from the "
            "registered Panda topology"
        )
    raw_directions = differential["joint_velocity_directions_rad_s"]
    if not isinstance(raw_directions, list) or len(raw_directions) != len(
        REGISTERED_DIFFERENTIAL_DIRECTIONS_RAD_S
    ):
        raise FeasibilityProtocolError(
            "differential_audit.joint_velocity_directions_rad_s must contain "
            "the seven basis and two dense registered directions"
        )
    directions = tuple(
        _number_vector(
            value,
            "differential_audit.joint_velocity_directions_rad_s[%d]" % index,
            7,
        )
        for index, value in enumerate(raw_directions)
    )
    for observed, expected in zip(
        directions, REGISTERED_DIFFERENTIAL_DIRECTIONS_RAD_S
    ):
        if any(not _close(left, right) for left, right in zip(observed, expected)):
            raise FeasibilityProtocolError(
                "differential_audit joint-velocity directions differ from the "
                "registered seven basis plus two dense directions"
            )
    raw_eta = differential["coupled_eta_ladder_s"]
    if not isinstance(raw_eta, list) or len(raw_eta) != len(
        REGISTERED_DIFFERENTIAL_ETA_LADDER_S
    ):
        raise FeasibilityProtocolError(
            "differential_audit.coupled_eta_ladder_s has the wrong length"
        )
    eta = tuple(
        _number(
            value,
            "differential_audit.coupled_eta_ladder_s[%d]" % index,
            minimum=0.0,
            strict_minimum=True,
        )
        for index, value in enumerate(raw_eta)
    )
    if any(
        not _close(observed, expected)
        for observed, expected in zip(
            eta, REGISTERED_DIFFERENTIAL_ETA_LADDER_S
        )
    ):
        raise FeasibilityProtocolError(
            "differential_audit eta ladder differs from the registered divide-by-four ladder"
        )
    if not _boolean(
        differential["same_trilinear_cell_required"],
        "differential_audit.same_trilinear_cell_required",
    ):
        raise FeasibilityProtocolError(
            "coupled differential audit must remain within one trilinear cell"
        )
    if _integer(
        differential["required_direction_count_per_sample"],
        "differential_audit.required_direction_count_per_sample",
        minimum=1,
    ) != len(REGISTERED_DIFFERENTIAL_DIRECTIONS_RAD_S):
        raise FeasibilityProtocolError(
            "every protected sample must pass all registered differential directions"
        )
    _literal(
        differential["stencil_selection_policy"],
        "largest_eta_with_valid_base_plus_minus_in_same_exact_cell",
        "differential_audit.stencil_selection_policy",
    )
    _literal(
        differential["finite_difference_resolution_policy"],
        "fail_on_no_certified_stencil_or_detected_cancellation",
        "differential_audit.finite_difference_resolution_policy",
    )
    _literal(
        differential["failure_policy"],
        "fail_before_active_physics_retain_artifact",
        "differential_audit.failure_policy",
    )

    if schema_version == LEGACY_SCHEMA_VERSION:
        scope_keys = (
            "obstacle_scope",
            "protected_robot_bodies",
            "unprotected_body_policy",
            "population_scope",
            "claim_strength",
            "prohibited_generalization",
        )
    else:
        scope_keys = (
            "obstacle_scope",
            "protected_robot_bodies",
            "protected_robot_bodies_role",
            "shield_robot_collision_surface_scope",
            "shield_robot_qvel_scope",
            "shield_decision_scope",
            "nonarm_control_policy",
            "unprotected_body_policy",
            "population_scope",
            "claim_strength",
            "prohibited_generalization",
        )
    scope = _object(top["claim_scope"], "protocol.claim_scope", scope_keys)
    _literal(
        scope["obstacle_scope"],
        "one_selected_obstacle_collision_geometry_only",
        "claim_scope.obstacle_scope",
    )
    _literal_string_array(
        scope["protected_robot_bodies"],
        ("robot0_link5", "robot0_link6"),
        "claim_scope.protected_robot_bodies",
    )
    if schema_version == LEGACY_SCHEMA_VERSION:
        _literal(
            scope["unprotected_body_policy"],
            "measure_contacts_but_exclude_from_link56_claim",
            "claim_scope.unprotected_body_policy",
        )
    else:
        for field, expected in (
            (
                "protected_robot_bodies_role",
                "field_bundle_sample_seed_only_not_shield_constraint_scope",
            ),
            (
                "shield_robot_collision_surface_scope",
                "all_collision_enabled_geoms_in_authoritative_robot_body_tree",
            ),
            (
                "shield_robot_qvel_scope",
                "all_qvel_dofs_in_authoritative_robot_body_tree_affecting_shield_samples",
            ),
            (
                "shield_decision_scope",
                "seven_registered_panda_arm_torque_controls_only",
            ),
            (
                "nonarm_control_policy",
                "nominal_nonarm_controls_unchanged_with_motion_included_in_exact_affine_dynamics",
            ),
            (
                "unprotected_body_policy",
                "no_authoritative_robot_collision_surface_excluded_against_selected_obstacle",
            ),
        ):
            _literal(scope[field], expected, "claim_scope.%s" % field)
    _literal(
        scope["population_scope"],
        "outcome_conditioned_109_case_targeted_feasibility",
        "claim_scope.population_scope",
    )
    _literal(
        scope["claim_strength"],
        "empirical_discrete_C0_static_field_feasibility_not_formal_guarantee",
        "claim_scope.claim_strength",
    )
    _literal(
        scope["prohibited_generalization"],
        (
            "not_all_obstacles_not_all_links_not_unbiased_safelibero"
            if schema_version == LEGACY_SCHEMA_VERSION
            else "not_all_obstacles_not_unbiased_safelibero"
        ),
        "claim_scope.prohibited_generalization",
    )

    selection = _object(
        top["parameter_selection"],
        "protocol.parameter_selection",
        (
            "selection_order",
            "allowed_tuning_splits",
            "heldout_split",
            "blocking_unit",
            "freeze_before_heldout",
            "heldout_parameter_changes",
            "after_change_policy",
            "parameter_sections",
            "parameter_block_sha256",
        ),
    )
    _literal_string_array(
        selection["selection_order"],
        ("bringup_canary", "parameter_freeze", "heldout_evaluation"),
        "parameter_selection.selection_order",
    )
    _literal_string_array(
        selection["allowed_tuning_splits"],
        ("bringup_canary", "parameter_freeze"),
        "parameter_selection.allowed_tuning_splits",
    )
    _literal(
        selection["heldout_split"],
        "heldout_evaluation",
        "parameter_selection.heldout_split",
    )
    _literal(
        selection["blocking_unit"],
        "whole_semantic_task_family",
        "parameter_selection.blocking_unit",
    )
    if not _boolean(
        selection["freeze_before_heldout"],
        "parameter_selection.freeze_before_heldout",
    ):
        raise FeasibilityProtocolError("parameters must freeze before held-out use")
    _literal(
        selection["heldout_parameter_changes"],
        "forbidden",
        "parameter_selection.heldout_parameter_changes",
    )
    _literal(
        selection["after_change_policy"],
        "new_protocol_and_new_heldout_required",
        "parameter_selection.after_change_policy",
    )
    _literal_string_array(
        selection["parameter_sections"],
        PARAMETER_SECTIONS,
        "parameter_selection.parameter_sections",
    )
    _require_sha256(
        selection["parameter_block_sha256"],
        "parameter_selection.parameter_block_sha256",
    )


def _parameter_block(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    return {name: protocol[name] for name in PARAMETER_SECTIONS}


def parameter_block_sha256(protocol: Mapping[str, Any]) -> str:
    """Compute the freeze digest after validating all non-digest semantics."""

    _validate_semantics(protocol)
    return _sha256(canonical_json_bytes(_parameter_block(protocol)))


def bind_parameter_block(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep copy with the computed parameter freeze digest installed.

    A syntactically valid lowercase placeholder must already be present.  This
    avoids making missing provenance look like a merely stale digest.
    """

    output = copy.deepcopy(protocol)
    _validate_semantics(output)
    output["parameter_selection"]["parameter_block_sha256"] = _sha256(
        canonical_json_bytes(_parameter_block(output))
    )
    return output


def validate_feasibility_protocol(
    protocol: Mapping[str, Any],
    *,
    expected_protocol_sha256: Optional[str] = None,
) -> ProtocolHashes:
    """Validate all fields, cross-field invariants, and both content hashes."""

    _validate_semantics(protocol)
    observed_parameter_hash = _sha256(
        canonical_json_bytes(_parameter_block(protocol))
    )
    declared_parameter_hash = protocol["parameter_selection"][
        "parameter_block_sha256"
    ]
    if observed_parameter_hash != declared_parameter_hash:
        raise FeasibilityProtocolError(
            "parameter_selection.parameter_block_sha256 does not match parameters"
        )
    observed_protocol_hash = _sha256(canonical_json_bytes(protocol))
    if expected_protocol_sha256 is not None:
        expected = _require_sha256(
            expected_protocol_sha256, "expected_protocol_sha256"
        )
        if observed_protocol_hash != expected:
            raise FeasibilityProtocolError(
                "protocol SHA-256 does not match the expected identity"
            )
    return ProtocolHashes(observed_protocol_hash, observed_parameter_hash)


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise FeasibilityProtocolError(
                "duplicate JSON object key {!r}".format(key)
            )
        output[key] = value
    return output


def load_feasibility_protocol(
    path: Path,
    *,
    expected_protocol_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], ProtocolHashes]:
    """Load one real JSON file and return its validated semantic identity."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FeasibilityProtocolError(
            "protocol path must be an existing non-symlink file"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FeasibilityProtocolError(
                    "non-finite JSON constant {!r}".format(value)
                )
            ),
        )
    except FeasibilityProtocolError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeasibilityProtocolError("protocol JSON is unreadable or invalid") from error
    if not isinstance(payload, dict):
        raise FeasibilityProtocolError("protocol JSON root must be an object")
    hashes = validate_feasibility_protocol(
        payload, expected_protocol_sha256=expected_protocol_sha256
    )
    return payload, hashes
