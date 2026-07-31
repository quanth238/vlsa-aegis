"""Physics-substep contact and clearance measurement for the static pilot.

MuJoCo nonpositive-distance contacts are the collision authority.
``mj_geomDistance`` is retained only as an explicitly labelled diagnostic
because mesh--box queries have not been validated as an authoritative
clearance oracle in this repository.  A separate geometric audit evaluates
registered robot surface samples against the selected obstacle's box
collision geoms.  If the samples have a certified covering radius
``epsilon``, the 1-Lipschitz property of Euclidean distance gives the
full-surface lower bound ``min(sample distance)-epsilon``.

MuJoCo's ``mj_step`` can leave position-dependent fields and its contact list
at the solver's pre-integration state while ``qpos`` already contains the
post-integration state.  Consequently, scientific post-substep measurements
are evaluated on an ``mjSTATE_INTEGRATION`` clone followed by ``mj_forward``.
The live solver-phase contacts are preserved as the collision authority for
the just-completed integration interval, while the forwarded clone is the
post-state authority.  Nonpositive contacts from either phase enter the union
safety endpoint and ``D_sim``; positive-margin candidates remain diagnostic.

All NumPy and MuJoCo imports are lazy so importing the released baseline does
not acquire an opt-in feasibility dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from main.poisson_fullbody.geometry import (
    OrientedBox,
    minimum_point_to_oriented_boxes_distance,
)
from main.poisson_fullbody.robot_samples import BodySample


def _modules() -> Tuple[Any, Any]:
    try:
        import mujoco
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError(
            "MuJoCo and NumPy are required for full-body measurement"
        ) from error
    return mujoco, np


def _raw_model_data(sim: Any) -> Tuple[Any, Any]:
    """Return official MuJoCo model/data through robosuite wrappers."""

    if not hasattr(sim, "model") or not hasattr(sim, "data"):
        raise TypeError("sim must expose MuJoCo model and data objects")
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    return model, data


def _body_positions(data: Any) -> Any:
    """Bridge mujoco-py/robosuite and official MuJoCo body field names."""

    return getattr(data, "body_xpos", getattr(data, "xpos", None))


def _body_rotations(data: Any) -> Any:
    return getattr(data, "body_xmat", getattr(data, "xmat", None))


def copy_integration_state(
    model: Any, live_data: Any, reusable_clone: Optional[Any] = None
) -> Any:
    """Copy MuJoCo's complete integration-state vector into separate data.

    MuJoCo 3.2.3 does not expose ``mj_copyData`` in Python.  Its supported
    complete integration-state transfer is ``mj_getState``/``mj_setState``
    with ``mjSTATE_INTEGRATION``.  That bitmask includes time, qpos, qvel,
    actuator state, warm-start acceleration, controls, applied forces,
    equality activation, mocap poses, user data, and plugin state.
    """

    mujoco, np = _modules()
    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(
        int(mujoco.mj_stateSize(model, specification)), dtype=np.float64
    )
    mujoco.mj_getState(model, live_data, state, specification)
    clone = reusable_clone if reusable_clone is not None else mujoco.MjData(model)
    mujoco.mj_setState(model, clone, state, specification)
    return clone


def clone_forwarded_state(
    model: Any, live_data: Any, reusable_clone: Optional[Any] = None
) -> Any:
    """Copy integration state, then refresh only the separate clone."""

    mujoco = _modules()[0]
    clone = copy_integration_state(model, live_data, reusable_clone)
    mujoco.mj_forward(model, clone)
    return clone


def _name(model: Any, object_type: Any, object_id: int, prefix: str) -> str:
    mujoco = _modules()[0]
    value = mujoco.mj_id2name(model, object_type, int(object_id))
    return str(value) if value else "%s_%d" % (prefix, int(object_id))


def _body_name(model: Any, body_id: int) -> str:
    mujoco = _modules()[0]
    return _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body")


def _geom_name(model: Any, geom_id: int) -> str:
    mujoco = _modules()[0]
    return _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, "geom")


def _validated_ids(
    values: Iterable[int], *, count: int, label: str, reject_world: bool = False
) -> Tuple[int, ...]:
    result: List[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("%s must contain integer IDs" % label)
        if value < 0 or value >= count or (reject_world and value == 0):
            raise ValueError("%s contains an invalid ID: %r" % (label, value))
        result.append(int(value))
    if not result:
        raise ValueError("%s must not be empty" % label)
    if len(result) != len(set(result)):
        raise ValueError("%s must not contain duplicates" % label)
    return tuple(result)


def descendant_body_ids(model: Any, root_body_ids: Iterable[int]) -> Tuple[int, ...]:
    """Resolve roots and all descendants without relying on body names.

    Body zero is deliberately rejected: treating the MuJoCo world as an
    authoritative object root would silently include the whole scene.
    """

    roots = _validated_ids(
        root_body_ids,
        count=int(model.nbody),
        label="root_body_ids",
        reject_world=True,
    )
    root_set = set(roots)
    descendants: List[int] = []
    for candidate in range(1, int(model.nbody)):
        current = candidate
        visited = set()
        while current != 0 and current not in visited:
            if current in root_set:
                descendants.append(candidate)
                break
            visited.add(current)
            current = int(model.body_parentid[current])
    return tuple(sorted(set(descendants)))


def _mask_enabled(model: Any, geom_id: int) -> bool:
    return bool(
        int(model.geom_contype[geom_id]) != 0
        or int(model.geom_conaffinity[geom_id]) != 0
    )


def _pair_mask_enabled(model: Any, robot_geom: int, obstacle_geom: int) -> bool:
    return bool(
        (
            int(model.geom_contype[robot_geom])
            & int(model.geom_conaffinity[obstacle_geom])
        )
        or (
            int(model.geom_contype[obstacle_geom])
            & int(model.geom_conaffinity[robot_geom])
        )
    )


@dataclass(frozen=True)
class ResolvedGeomSets:
    """Auditable geometry identities resolved from authoritative body IDs."""

    robot_root_body_ids: Tuple[int, ...]
    robot_body_ids: Tuple[int, ...]
    obstacle_root_body_ids: Tuple[int, ...]
    obstacle_body_ids: Tuple[int, ...]
    link56_body_ids: Tuple[int, ...]
    robot_geom_ids: Tuple[int, ...]
    obstacle_geom_ids: Tuple[int, ...]
    link56_geom_ids: Tuple[int, ...]
    collision_enabled_pairs: Tuple[Tuple[int, int], ...]
    robot_body_names: Tuple[str, ...]
    obstacle_body_names: Tuple[str, ...]
    robot_geom_names: Tuple[str, ...]
    obstacle_geom_names: Tuple[str, ...]
    link56_geom_names: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_collision_geom_sets(
    model_or_sim: Any,
    *,
    robot_root_body_ids: Iterable[int],
    obstacle_root_body_ids: Iterable[int],
    link56_body_ids: Iterable[int],
) -> ResolvedGeomSets:
    """Resolve selected robot/obstacle collision geoms from body identities.

    Robot and selected-obstacle roots expand through ``body_parentid``.  The
    link-5/6 set is intentionally *literal*: only geoms attached to the exact
    supplied link body IDs count as link-5/6 truth; descendants are not
    relabelled as those links.  Name matching is never used.
    """

    candidate = getattr(model_or_sim, "model", model_or_sim)
    model = getattr(candidate, "_model", candidate)
    robot_roots = _validated_ids(
        robot_root_body_ids,
        count=int(model.nbody),
        label="robot_root_body_ids",
        reject_world=True,
    )
    obstacle_roots = _validated_ids(
        obstacle_root_body_ids,
        count=int(model.nbody),
        label="obstacle_root_body_ids",
        reject_world=True,
    )
    robot_bodies = descendant_body_ids(model, robot_roots)
    obstacle_bodies = descendant_body_ids(model, obstacle_roots)
    overlap = set(robot_bodies) & set(obstacle_bodies)
    if overlap:
        raise ValueError(
            "robot and selected-obstacle body trees overlap at %r" % sorted(overlap)
        )
    literal_links = _validated_ids(
        link56_body_ids,
        count=int(model.nbody),
        label="link56_body_ids",
        reject_world=True,
    )
    outside = set(literal_links) - set(robot_bodies)
    if outside:
        raise ValueError(
            "link56_body_ids are outside the authoritative robot tree: %r"
            % sorted(outside)
        )

    robot_body_set = set(robot_bodies)
    obstacle_body_set = set(obstacle_bodies)
    all_robot_geoms = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in robot_body_set
    )
    all_obstacle_geoms = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in obstacle_body_set
    )
    robot_all_set = set(all_robot_geoms)
    obstacle_all_set = set(all_obstacle_geoms)
    explicit_pairs = set()
    for pair_index in range(int(model.npair)):
        geom1 = int(model.pair_geom1[pair_index])
        geom2 = int(model.pair_geom2[pair_index])
        if geom1 in robot_all_set and geom2 in obstacle_all_set:
            explicit_pairs.add((geom1, geom2))
        elif geom2 in robot_all_set and geom1 in obstacle_all_set:
            explicit_pairs.add((geom2, geom1))
    explicit_robot_geoms = {pair[0] for pair in explicit_pairs}
    explicit_obstacle_geoms = {pair[1] for pair in explicit_pairs}
    robot_geoms = tuple(
        geom_id
        for geom_id in all_robot_geoms
        if _mask_enabled(model, geom_id) or geom_id in explicit_robot_geoms
    )
    obstacle_geoms = tuple(
        geom_id
        for geom_id in all_obstacle_geoms
        if _mask_enabled(model, geom_id) or geom_id in explicit_obstacle_geoms
    )
    if not robot_geoms:
        raise RuntimeError("the authoritative robot tree has no collision-enabled geoms")
    if not obstacle_geoms:
        raise RuntimeError(
            "the selected-obstacle tree has no collision-enabled geoms"
        )
    pairs = tuple(
        sorted(
            explicit_pairs
            | {
                (robot_geom, obstacle_geom)
                for robot_geom in robot_geoms
                for obstacle_geom in obstacle_geoms
                if _pair_mask_enabled(model, robot_geom, obstacle_geom)
            }
        )
    )
    if not pairs:
        raise RuntimeError(
            "no robot/selected-obstacle geom pair has compatible contact masks"
        )
    literal_set = set(literal_links)
    link_geoms = tuple(
        geom_id
        for geom_id in robot_geoms
        if int(model.geom_bodyid[geom_id]) in literal_set
    )
    if not link_geoms:
        raise RuntimeError("the literal link-5/6 bodies have no collision-enabled geoms")

    return ResolvedGeomSets(
        robot_root_body_ids=robot_roots,
        robot_body_ids=robot_bodies,
        obstacle_root_body_ids=obstacle_roots,
        obstacle_body_ids=obstacle_bodies,
        link56_body_ids=literal_links,
        robot_geom_ids=robot_geoms,
        obstacle_geom_ids=obstacle_geoms,
        link56_geom_ids=link_geoms,
        collision_enabled_pairs=pairs,
        robot_body_names=tuple(_body_name(model, value) for value in robot_bodies),
        obstacle_body_names=tuple(
            _body_name(model, value) for value in obstacle_bodies
        ),
        robot_geom_names=tuple(_geom_name(model, value) for value in robot_geoms),
        obstacle_geom_names=tuple(
            _geom_name(model, value) for value in obstacle_geoms
        ),
        link56_geom_names=tuple(_geom_name(model, value) for value in link_geoms),
    )


@dataclass(frozen=True)
class ContactPointRecord:
    source_phase: str
    observation_index: Optional[int]
    high_level_index: Optional[int]
    inner_control_index: Optional[int]
    physics_substep_index: Optional[int]
    mujoco_contact_index: int
    is_physical_nonpositive_distance_contact: bool
    within_registered_near_contact_tolerance: bool
    solver_constraint_active: bool
    efc_address: int
    mujoco_geom1_id: int
    mujoco_geom1_name: str
    mujoco_geom2_id: int
    mujoco_geom2_name: str
    robot_geom_id: int
    robot_geom_name: str
    robot_body_id: int
    robot_body_name: str
    obstacle_geom_id: int
    obstacle_geom_name: str
    obstacle_body_id: int
    obstacle_body_name: str
    contact_distance_m: float
    contact_includemargin_m: float
    robot_geom_margin_m: float
    robot_geom_gap_m: float
    obstacle_geom_margin_m: float
    obstacle_geom_gap_m: float
    explicit_pair_id: Optional[int]
    explicit_pair_margin_m: Optional[float]
    explicit_pair_gap_m: Optional[float]
    position_world_m: Tuple[float, float, float]
    frame_normal_mujoco_geom1_to_geom2_world: Tuple[float, float, float]
    force_available: bool
    force_semantics: str
    force_contact_frame_n: Optional[Tuple[float, ...]]
    force_world_n: Optional[Tuple[float, float, float]]
    normal_force_n: Optional[float]
    impulse_estimate_contact_frame_ns: Optional[Tuple[float, ...]]
    impulse_estimate_world_ns: Optional[Tuple[float, float, float]]
    force_unavailable_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawGeomDistanceAdvisory:
    available: bool
    authority: str
    minimum_distance_m: Optional[float]
    distance_query_limit_m: float
    minimum_observation_index: Optional[int]
    robot_geom_id: Optional[int]
    robot_geom_name: Optional[str]
    obstacle_geom_id: Optional[int]
    obstacle_geom_name: Optional[str]
    fromto_world_m: Optional[Tuple[float, ...]]


@dataclass(frozen=True)
class SampleCoverageClearance:
    available: bool
    authority: str
    minimum_exact_sample_to_obb_distance_m: Optional[float]
    certified_coverage_radius_m: float
    full_surface_clearance_lower_bound_m: Optional[float]
    minimum_observation_index: Optional[int]
    sample_id: Optional[int]
    robot_geom_id: Optional[int]
    obstacle_geom_id: Optional[int]


@dataclass(frozen=True)
class ObstacleGeomDrift:
    geom_id: int
    geom_name: str
    body_id: int
    body_name: str
    maximum_translation_m: float
    maximum_rotation_rad: float
    maximum_surface_point_displacement_m: float
    final_translation_m: float
    final_rotation_rad: float
    final_surface_point_displacement_m: float
    maximum_translation_observation_index: int
    maximum_rotation_observation_index: int
    maximum_surface_displacement_observation_index: int


@dataclass(frozen=True)
class ObstacleDriftSummary:
    reference: str
    maximum_translation_m: float
    maximum_rotation_rad: float
    maximum_surface_point_displacement_m: float
    surface_drift_threshold_m: float
    surface_drift_threshold_crossed: bool
    first_surface_drift_threshold_crossing_observation_index: Optional[int]
    geoms: Tuple[ObstacleGeomDrift, ...]


@dataclass(frozen=True)
class SettledObstacleBodyVelocity:
    """One selected-obstacle body's world-frame velocity after settling."""

    body_id: int
    body_name: str
    angular_velocity_world_rad_per_s: Tuple[float, float, float]
    linear_velocity_world_m_per_s: Tuple[float, float, float]
    angular_speed_rad_per_s: float
    linear_speed_m_per_s: float
    angular_speed_exceeds_threshold: bool
    linear_speed_exceeds_threshold: bool


@dataclass(frozen=True)
class SettledObstacleMotionAdmissibility:
    reference: str
    max_linear_speed_threshold_m_per_s: float
    max_angular_speed_threshold_rad_per_s: float
    maximum_observed_linear_speed_m_per_s: float
    maximum_observed_angular_speed_rad_per_s: float
    admissible: bool
    reasons: Tuple[str, ...]
    bodies: Tuple[SettledObstacleBodyVelocity, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SettledObstacleMotionInadmissible(RuntimeError):
    """Raised before rollout when the selected obstacle is not static."""


class StaticObstacleDriftInadmissible(RuntimeError):
    """Raised after recording the first substep that invalidates a static field."""


@dataclass(frozen=True)
class SettledStateSnapshot:
    """Selected robot--obstacle state immediately after settling."""

    candidate_contact_point_record_count: int
    solver_active_contact_point_record_count: int
    within_near_contact_tolerance_point_record_count: int
    physical_contact_point_record_count: int
    any_robot_obstacle_contact: bool
    link56_obstacle_contact: bool
    first_candidate_contact_point_record: Optional[ContactPointRecord]
    first_physical_contact_point_record: Optional[ContactPointRecord]
    candidate_contact_point_records: Tuple[ContactPointRecord, ...]
    physical_contact_point_records: Tuple[ContactPointRecord, ...]
    obstacle_motion_admissibility: SettledObstacleMotionAdmissibility
    raw_mj_geom_distance: RawGeomDistanceAdvisory
    sample_clearance: SampleCoverageClearance
    D_sim_m: float
    contact_authority_clamped_D_sim: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullRobotObstacleMeasurement:
    """Typed final measurement; all distances are in physical metres.

    Contact counts are MuJoCo contact-point snapshot records summed across
    observed phases/substeps.  They are not deduplicated collision events.
    """

    observed_physics_substeps: int
    first_index: Tuple[int, int, int]
    last_index: Tuple[int, int, int]
    any_robot_obstacle_contact: bool
    link56_obstacle_contact: bool
    rollout_any_robot_obstacle_contact: bool
    rollout_link56_obstacle_contact: bool
    post_state_any_robot_obstacle_contact: bool
    post_state_link56_obstacle_contact: bool
    total_candidate_contact_point_record_count: int
    total_physical_contact_point_record_count: int
    rollout_phase_physical_contact_point_record_count: int
    post_state_candidate_contact_point_record_count: int
    post_state_solver_active_contact_point_record_count: int
    post_state_within_near_contact_tolerance_point_record_count: int
    post_state_physical_contact_point_record_count: int
    first_physical_contact_point_record: Optional[ContactPointRecord]
    first_candidate_contact_point_record: Optional[ContactPointRecord]
    first_live_solver_physical_contact_point_record: Optional[ContactPointRecord]
    first_post_state_physical_contact_point_record: Optional[ContactPointRecord]
    post_state_candidate_contact_point_records: Tuple[ContactPointRecord, ...]
    post_state_physical_contact_point_records: Tuple[ContactPointRecord, ...]
    live_solver_any_robot_obstacle_contact: bool
    live_solver_link56_obstacle_contact: bool
    live_solver_candidate_contact_point_record_count: int
    live_solver_active_contact_point_record_count: int
    live_solver_within_near_contact_tolerance_point_record_count: int
    live_solver_nonpositive_contact_point_record_count: int
    live_solver_phase_contact_point_records: Tuple[ContactPointRecord, ...]
    settled_state: SettledStateSnapshot
    raw_mj_geom_distance: RawGeomDistanceAdvisory
    sample_clearance: SampleCoverageClearance
    obstacle_pose_drift: ObstacleDriftSummary
    D_sim_min_m: float
    D_sim_semantics: str
    physical_contact_distance_semantics: str
    near_contact_tolerance_m: float
    contact_authority_clamped_D_sim: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FullRobotObstacleMonitor:
    """Measure one settled-state rollout at every post-integration substep.

    Construct this monitor immediately after the 20 settling actions.  Every
    selected obstacle collision-geom pose at construction is an immutable
    drift reference.  Inspect ``settled_state.obstacle_motion_admissibility``
    before rollout; callback creation and observation both fail closed when it
    is false.  Call :meth:`make_substep_callback` for every 100 Hz inner update;
    its callback is compatible with the additive SafeLIBERO wrapper.  The index
    contract (5 inner updates per high-level action and 5 physics substeps per
    inner update by default) rejects omissions and duplicates.
    """

    def __init__(
        self,
        sim: Any,
        resolved: ResolvedGeomSets,
        robot_surface_samples: Iterable[BodySample],
        *,
        certified_coverage_radius_m: float,
        max_selected_geom_surface_drift_m: float,
        max_selected_geom_translation_drift_m: Optional[float] = None,
        max_selected_geom_rotation_drift_rad: Optional[float] = None,
        max_settled_obstacle_linear_speed_m_per_s: float = 0.0,
        max_settled_obstacle_angular_speed_rad_per_s: float = 0.0,
        require_settled_static_motion: bool = True,
        terminate_on_static_drift: bool = True,
        raw_distance_query_limit_m: float = 1.0,
        near_contact_tolerance_m: float = 0.0,
        inner_updates_per_high_level_action: int = 5,
        physics_substeps_per_inner_update: int = 5,
    ) -> None:
        mujoco, np = _modules()
        model, data = _raw_model_data(sim)
        if not isinstance(resolved, ResolvedGeomSets):
            raise TypeError("resolved must be a ResolvedGeomSets")
        current_resolution = resolve_collision_geom_sets(
            model,
            robot_root_body_ids=resolved.robot_root_body_ids,
            obstacle_root_body_ids=resolved.obstacle_root_body_ids,
            link56_body_ids=resolved.link56_body_ids,
        )
        if current_resolution != resolved:
            raise ValueError("resolved geom identities do not match this MuJoCo model")
        coverage = float(certified_coverage_radius_m)
        if not math.isfinite(coverage) or coverage < 0.0:
            raise ValueError("certified_coverage_radius_m must be finite and nonnegative")
        query_limit = float(raw_distance_query_limit_m)
        if not math.isfinite(query_limit) or query_limit <= 0.0:
            raise ValueError("raw_distance_query_limit_m must be finite and positive")
        near_tolerance = float(near_contact_tolerance_m)
        if not math.isfinite(near_tolerance) or near_tolerance < 0.0:
            raise ValueError("near_contact_tolerance_m must be finite and nonnegative")
        surface_drift_threshold = float(max_selected_geom_surface_drift_m)
        if not math.isfinite(surface_drift_threshold) or surface_drift_threshold < 0.0:
            raise ValueError(
                "max_selected_geom_surface_drift_m must be finite and nonnegative"
            )
        translation_drift_threshold = (
            surface_drift_threshold
            if max_selected_geom_translation_drift_m is None
            else float(max_selected_geom_translation_drift_m)
        )
        rotation_drift_threshold = (
            float("inf")
            if max_selected_geom_rotation_drift_rad is None
            else float(max_selected_geom_rotation_drift_rad)
        )
        if (
            not math.isfinite(translation_drift_threshold)
            or translation_drift_threshold < 0.0
        ):
            raise ValueError(
                "max_selected_geom_translation_drift_m must be finite and nonnegative"
            )
        if (
            max_selected_geom_rotation_drift_rad is not None
            and (
                not math.isfinite(rotation_drift_threshold)
                or rotation_drift_threshold < 0.0
            )
        ):
            raise ValueError(
                "max_selected_geom_rotation_drift_rad must be finite and nonnegative"
            )
        if not isinstance(require_settled_static_motion, bool):
            raise TypeError("require_settled_static_motion must be Boolean")
        if not isinstance(terminate_on_static_drift, bool):
            raise TypeError("terminate_on_static_drift must be Boolean")
        if isinstance(max_settled_obstacle_linear_speed_m_per_s, bool):
            raise ValueError(
                "max_settled_obstacle_linear_speed_m_per_s must be finite and nonnegative"
            )
        if isinstance(max_settled_obstacle_angular_speed_rad_per_s, bool):
            raise ValueError(
                "max_settled_obstacle_angular_speed_rad_per_s must be finite and nonnegative"
            )
        try:
            settled_linear_speed_threshold = float(
                max_settled_obstacle_linear_speed_m_per_s
            )
            settled_angular_speed_threshold = float(
                max_settled_obstacle_angular_speed_rad_per_s
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "settled obstacle speed thresholds must be finite and nonnegative"
            ) from error
        if (
            not math.isfinite(settled_linear_speed_threshold)
            or settled_linear_speed_threshold < 0.0
        ):
            raise ValueError(
                "max_settled_obstacle_linear_speed_m_per_s must be finite and nonnegative"
            )
        if (
            not math.isfinite(settled_angular_speed_threshold)
            or settled_angular_speed_threshold < 0.0
        ):
            raise ValueError(
                "max_settled_obstacle_angular_speed_rad_per_s must be finite and nonnegative"
            )
        if (
            isinstance(inner_updates_per_high_level_action, bool)
            or int(inner_updates_per_high_level_action)
            != inner_updates_per_high_level_action
            or int(inner_updates_per_high_level_action) <= 0
        ):
            raise ValueError("inner_updates_per_high_level_action must be positive")
        if (
            isinstance(physics_substeps_per_inner_update, bool)
            or int(physics_substeps_per_inner_update)
            != physics_substeps_per_inner_update
            or int(physics_substeps_per_inner_update) <= 0
        ):
            raise ValueError("physics_substeps_per_inner_update must be positive")

        samples = tuple(robot_surface_samples)
        if not samples:
            raise ValueError("robot_surface_samples must not be empty")
        sample_ids = [sample.sample_id for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("robot surface sample IDs must be unique")
        resolved_robot_geoms = set(resolved.robot_geom_ids)
        sampled_geoms = set()
        for sample in samples:
            if not isinstance(sample, BodySample):
                raise TypeError("robot_surface_samples must contain BodySample records")
            if sample.body_id not in set(resolved.robot_body_ids):
                raise ValueError("sample body is outside the authoritative robot tree")
            if sample.geom_id not in resolved_robot_geoms:
                raise ValueError("sample geom is not a resolved robot collision geom")
            if int(model.geom_bodyid[sample.geom_id]) != sample.body_id:
                raise ValueError("sample geom/body provenance does not match MuJoCo")
            if sample.body_name != _body_name(model, sample.body_id):
                raise ValueError("sample body name does not match authoritative MuJoCo ID")
            if sample.geom_name != _geom_name(model, sample.geom_id):
                raise ValueError("sample geom name does not match authoritative MuJoCo ID")
            sampled_geoms.add(sample.geom_id)
        missing = resolved_robot_geoms - sampled_geoms
        if missing:
            raise ValueError(
                "full-robot clearance requires samples for every resolved robot geom; "
                "missing %r" % sorted(missing)
            )

        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        non_boxes = [
            geom_id
            for geom_id in resolved.obstacle_geom_ids
            if int(model.geom_type[geom_id]) != box_type
        ]
        if non_boxes:
            raise ValueError(
                "exact point-to-OBB clearance requires every selected-obstacle "
                "collision geom to be a MuJoCo box; non-box IDs %r" % non_boxes
            )

        self._model = model
        self._resolved = resolved
        self._samples = samples
        self._coverage_radius_m = coverage
        self._query_limit_m = query_limit
        self._near_contact_tolerance_m = near_tolerance
        self._surface_drift_threshold_m = surface_drift_threshold
        self._translation_drift_threshold_m = translation_drift_threshold
        self._rotation_drift_threshold_rad = rotation_drift_threshold
        self._require_settled_static_motion = require_settled_static_motion
        self._terminate_on_static_drift = terminate_on_static_drift
        self._settled_linear_speed_threshold_m_per_s = (
            settled_linear_speed_threshold
        )
        self._settled_angular_speed_threshold_rad_per_s = (
            settled_angular_speed_threshold
        )
        self._first_surface_drift_crossing: Optional[int] = None
        self._first_static_drift_crossing: Optional[int] = None
        self._first_static_drift_reason: Optional[str] = None
        self._inner_per_high = int(inner_updates_per_high_level_action)
        self._physics_per_inner = int(physics_substeps_per_inner_update)
        self._pair_set = set(resolved.collision_enabled_pairs)
        self._link56_geoms = set(resolved.link56_geom_ids)
        self._explicit_pair_ids: Dict[Tuple[int, int], int] = {}
        for pair_index in range(int(model.npair)):
            geom1 = int(model.pair_geom1[pair_index])
            geom2 = int(model.pair_geom2[pair_index])
            if (geom1, geom2) in self._pair_set:
                self._explicit_pair_ids[(geom1, geom2)] = pair_index
            elif (geom2, geom1) in self._pair_set:
                self._explicit_pair_ids[(geom2, geom1)] = pair_index
        self._observations = 0
        self._first_index: Optional[Tuple[int, int, int]] = None
        self._last_index: Optional[Tuple[int, int, int]] = None
        self._candidate_contacts: List[ContactPointRecord] = []
        self._physical_contacts: List[ContactPointRecord] = []
        self._live_solver_contacts: List[ContactPointRecord] = []
        self._raw_minimum = float("inf")
        self._raw_minimum_record: Optional[
            Tuple[Optional[int], int, int, Tuple[float, ...]]
        ] = None
        self._sample_minimum = float("inf")
        self._sample_minimum_record: Optional[
            Tuple[Optional[int], int, int]
        ] = None

        # The live data can contain post-integrated qpos with stale kinematics
        # and solver-phase contacts.  Even the settled snapshot therefore uses
        # a forwarded integration-state clone.
        self._forwarded_clone = clone_forwarded_state(model, data)
        settled_data = self._forwarded_clone
        settled_motion = self._measure_settled_obstacle_motion(settled_data)
        self._observe_raw_distance(settled_data, None)
        self._observe_sample_clearance(settled_data, None)
        settled_candidates = self._contact_records(
            settled_data,
            source_phase="settled_post_integration_recomputed",
            observation_index=None,
            high=None,
            inner=None,
            physics=None,
        )
        settled_physical = tuple(
            event
            for event in settled_candidates
            if event.is_physical_nonpositive_distance_contact
        )
        settled_sample = self._sample_clearance_result()
        if settled_sample.full_surface_clearance_lower_bound_m is None:
            raise RuntimeError("settled-state sample clearance is unavailable")
        settled_lower_bound = float(
            settled_sample.full_surface_clearance_lower_bound_m
        )
        if settled_physical:
            settled_d_sim = min(
                settled_lower_bound,
                min(event.contact_distance_m for event in settled_physical),
                0.0,
            )
        else:
            settled_d_sim = settled_lower_bound
        self._settled_state = SettledStateSnapshot(
            candidate_contact_point_record_count=len(settled_candidates),
            solver_active_contact_point_record_count=sum(
                event.solver_constraint_active for event in settled_candidates
            ),
            within_near_contact_tolerance_point_record_count=sum(
                event.within_registered_near_contact_tolerance
                for event in settled_candidates
            ),
            physical_contact_point_record_count=len(settled_physical),
            any_robot_obstacle_contact=bool(settled_physical),
            link56_obstacle_contact=any(
                event.robot_geom_id in self._link56_geoms
                for event in settled_physical
            ),
            first_candidate_contact_point_record=(
                settled_candidates[0] if settled_candidates else None
            ),
            first_physical_contact_point_record=(
                settled_physical[0] if settled_physical else None
            ),
            candidate_contact_point_records=settled_candidates,
            physical_contact_point_records=settled_physical,
            obstacle_motion_admissibility=settled_motion,
            raw_mj_geom_distance=self._raw_advisory_result(),
            sample_clearance=settled_sample,
            D_sim_m=float(settled_d_sim),
            contact_authority_clamped_D_sim=bool(
                settled_physical and settled_d_sim < settled_lower_bound
            ),
        )

        self._settled_pose: Dict[int, Tuple[Any, Any, Any, Any]] = {}
        self._drift: Dict[int, Dict[str, Any]] = {}
        for geom_id in resolved.obstacle_geom_ids:
            position = np.asarray(
                settled_data.geom_xpos[geom_id], dtype=np.float64
            ).copy()
            rotation = np.asarray(
                settled_data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3).copy()
            half_extents = np.asarray(
                model.geom_size[geom_id][:3], dtype=np.float64
            ).copy()
            settled_vertices = OrientedBox(
                center=position,
                R=rotation,
                half_extents=half_extents,
                geom_id=geom_id,
            ).vertices()
            self._settled_pose[geom_id] = (
                position,
                rotation,
                half_extents,
                settled_vertices,
            )
            self._drift[geom_id] = {
                "max_translation": 0.0,
                "max_rotation": 0.0,
                "max_surface": 0.0,
                "max_translation_observation": 0,
                "max_rotation_observation": 0,
                "max_surface_observation": 0,
                "final_translation": 0.0,
                "final_rotation": 0.0,
                "final_surface": 0.0,
            }

    def _measure_settled_obstacle_motion(
        self, settled_data: Any
    ) -> SettledObstacleMotionAdmissibility:
        mujoco, np = _modules()
        body_records: List[SettledObstacleBodyVelocity] = []
        reasons: List[str] = []
        for body_id in sorted(set(self._resolved.obstacle_body_ids)):
            velocity = np.empty(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self._model,
                settled_data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                velocity,
                0,
            )
            if not np.all(np.isfinite(velocity)):
                raise RuntimeError(
                    "MuJoCo returned non-finite settled obstacle body velocity"
                )
            angular = velocity[:3]
            linear = velocity[3:]
            angular_speed = float(np.linalg.norm(angular))
            linear_speed = float(np.linalg.norm(linear))
            linear_exceeded = bool(
                linear_speed > self._settled_linear_speed_threshold_m_per_s
            )
            angular_exceeded = bool(
                angular_speed > self._settled_angular_speed_threshold_rad_per_s
            )
            body_name = _body_name(self._model, body_id)
            if linear_exceeded:
                reasons.append(
                    "body_id=%d body_name=%s linear_speed_m_per_s=%.17g "
                    "exceeds_threshold=%.17g"
                    % (
                        body_id,
                        body_name,
                        linear_speed,
                        self._settled_linear_speed_threshold_m_per_s,
                    )
                )
            if angular_exceeded:
                reasons.append(
                    "body_id=%d body_name=%s angular_speed_rad_per_s=%.17g "
                    "exceeds_threshold=%.17g"
                    % (
                        body_id,
                        body_name,
                        angular_speed,
                        self._settled_angular_speed_threshold_rad_per_s,
                    )
                )
            body_records.append(
                SettledObstacleBodyVelocity(
                    body_id=int(body_id),
                    body_name=body_name,
                    angular_velocity_world_rad_per_s=tuple(
                        float(value) for value in angular
                    ),
                    linear_velocity_world_m_per_s=tuple(
                        float(value) for value in linear
                    ),
                    angular_speed_rad_per_s=angular_speed,
                    linear_speed_m_per_s=linear_speed,
                    angular_speed_exceeds_threshold=angular_exceeded,
                    linear_speed_exceeds_threshold=linear_exceeded,
                )
            )
        records = tuple(body_records)
        if not records:
            raise RuntimeError("selected obstacle body set is empty")
        return SettledObstacleMotionAdmissibility(
            reference=(
                "clone_forwarded_settled_state_mj_objectVelocity_world_"
                "orientation_rot_then_lin"
            ),
            max_linear_speed_threshold_m_per_s=(
                self._settled_linear_speed_threshold_m_per_s
            ),
            max_angular_speed_threshold_rad_per_s=(
                self._settled_angular_speed_threshold_rad_per_s
            ),
            maximum_observed_linear_speed_m_per_s=max(
                record.linear_speed_m_per_s for record in records
            ),
            maximum_observed_angular_speed_rad_per_s=max(
                record.angular_speed_rad_per_s for record in records
            ),
            admissible=not reasons,
            reasons=tuple(reasons),
            bodies=records,
        )

    def _validate_index(self, high: int, inner: int, physics: int) -> None:
        for value, label in (
            (high, "high_level_index"),
            (inner, "inner_control_index"),
            (physics, "physics_substep_index"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("%s must be a nonnegative integer" % label)
        if inner >= self._inner_per_high:
            raise ValueError("inner_control_index exceeds the registered cadence")
        if physics >= self._physics_per_inner:
            raise ValueError("physics_substep_index exceeds the registered cadence")
        expected = (
            (high * self._inner_per_high + inner) * self._physics_per_inner
            + physics
        )
        if expected != self._observations:
            raise RuntimeError(
                "physics-substep callback gap/duplicate: expected global index %d, got %d"
                % (self._observations, expected)
            )

    def make_substep_callback(self, high_level_index: int, inner_control_index: int):
        """Return ``callback(sim, physics_substep_index)`` for the wrapper."""

        if self._require_settled_static_motion:
            self.require_settled_obstacle_motion_admissible()

        def callback(sim: Any, physics_substep_index: int) -> None:
            self.observe_post_integration(
                sim,
                high_level_index=high_level_index,
                inner_control_index=inner_control_index,
                physics_substep_index=physics_substep_index,
            )

        return callback

    @property
    def settled_state(self) -> SettledStateSnapshot:
        """Return the immutable clone-forwarded snapshot captured at creation."""

        return self._settled_state

    def require_settled_obstacle_motion_admissible(self) -> None:
        """Fail before rollout when the selected obstacle is still moving."""

        motion = self._settled_state.obstacle_motion_admissibility
        if not motion.admissible:
            raise SettledObstacleMotionInadmissible(
                "settled selected-obstacle motion is inadmissible: %s"
                % "; ".join(motion.reasons)
            )

    def _live_obstacle_boxes(self, data: Any) -> Tuple[OrientedBox, ...]:
        np = _modules()[1]
        return tuple(
            OrientedBox(
                center=np.asarray(data.geom_xpos[geom_id], dtype=np.float64),
                R=np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3),
                half_extents=np.asarray(
                    self._model.geom_size[geom_id][:3], dtype=np.float64
                ),
                geom_id=geom_id,
            )
            for geom_id in self._resolved.obstacle_geom_ids
        )

    def _observe_raw_distance(
        self, data: Any, observation_index: Optional[int]
    ) -> None:
        mujoco, np = _modules()
        for robot_geom, obstacle_geom in self._resolved.collision_enabled_pairs:
            fromto = np.full(6, np.nan, dtype=np.float64)
            distance = float(
                mujoco.mj_geomDistance(
                    self._model,
                    data,
                    robot_geom,
                    obstacle_geom,
                    self._query_limit_m,
                    fromto,
                )
            )
            if not math.isfinite(distance):
                continue
            if distance < self._raw_minimum:
                endpoint = (
                    tuple(float(value) for value in fromto)
                    if np.all(np.isfinite(fromto))
                    else tuple()
                )
                self._raw_minimum = distance
                self._raw_minimum_record = (
                    observation_index,
                    robot_geom,
                    obstacle_geom,
                    endpoint,
                )

    def _observe_sample_clearance(
        self, data: Any, observation_index: Optional[int]
    ) -> None:
        np = _modules()[1]
        boxes = self._live_obstacle_boxes(data)
        for sample in self._samples:
            body_position = np.asarray(
                _body_positions(data)[sample.body_id], dtype=np.float64
            )
            body_rotation = np.asarray(
                _body_rotations(data)[sample.body_id], dtype=np.float64
            ).reshape(3, 3)
            point = body_position + np.matmul(
                body_rotation, sample.point_local_array()
            )
            for box in boxes:
                distance = minimum_point_to_oriented_boxes_distance(point, (box,))
                if distance < self._sample_minimum:
                    self._sample_minimum = float(distance)
                    self._sample_minimum_record = (
                        observation_index,
                        sample.sample_id,
                        int(box.geom_id),
                    )

    @staticmethod
    def _rotation_angle(reference: Any, current: Any) -> float:
        np = _modules()[1]
        relative = np.matmul(reference.T, current)
        cosine = (float(np.trace(relative)) - 1.0) * 0.5
        return float(math.acos(max(-1.0, min(1.0, cosine))))

    def _observe_drift(self, data: Any, observation_index: int) -> None:
        np = _modules()[1]
        for geom_id, (
            settled_position,
            settled_rotation,
            half_extents,
            settled_vertices,
        ) in self._settled_pose.items():
            position = np.asarray(
                data.geom_xpos[geom_id], dtype=np.float64
            )
            rotation = np.asarray(
                data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            translation = float(np.linalg.norm(position - settled_position))
            angle = self._rotation_angle(settled_rotation, rotation)
            current_vertices = OrientedBox(
                center=position,
                R=rotation,
                half_extents=half_extents,
                geom_id=geom_id,
            ).vertices()
            surface_displacement = float(
                np.max(np.linalg.norm(current_vertices - settled_vertices, axis=1))
            )
            record = self._drift[geom_id]
            record["final_translation"] = translation
            record["final_rotation"] = angle
            record["final_surface"] = surface_displacement
            if translation > record["max_translation"]:
                record["max_translation"] = translation
                record["max_translation_observation"] = observation_index
            if angle > record["max_rotation"]:
                record["max_rotation"] = angle
                record["max_rotation_observation"] = observation_index
            if surface_displacement > record["max_surface"]:
                record["max_surface"] = surface_displacement
                record["max_surface_observation"] = observation_index
            if (
                self._first_surface_drift_crossing is None
                and surface_displacement > self._surface_drift_threshold_m
            ):
                self._first_surface_drift_crossing = observation_index
            if self._first_static_drift_crossing is None:
                reasons = []
                if translation > self._translation_drift_threshold_m:
                    reasons.append(
                        "translation %.17g m > %.17g m"
                        % (translation, self._translation_drift_threshold_m)
                    )
                if angle > self._rotation_drift_threshold_rad:
                    reasons.append(
                        "rotation %.17g rad > %.17g rad"
                        % (angle, self._rotation_drift_threshold_rad)
                    )
                if surface_displacement > self._surface_drift_threshold_m:
                    reasons.append(
                        "surface %.17g m > %.17g m"
                        % (surface_displacement, self._surface_drift_threshold_m)
                    )
                if reasons:
                    self._first_static_drift_crossing = observation_index
                    self._first_static_drift_reason = "; ".join(reasons)

    def _contact_force(
        self, data: Any, contact_index: int, contact: Any
    ) -> Tuple[
        bool,
        Optional[Tuple[float, ...]],
        Optional[Tuple[float, float, float]],
        Optional[float],
        Optional[Tuple[float, ...]],
        Optional[Tuple[float, float, float]],
        Optional[str],
    ]:
        mujoco, np = _modules()
        if int(contact.efc_address) < 0:
            return (
                False,
                None,
                None,
                None,
                None,
                None,
                "candidate contact has no active solver constraint",
            )
        wrench = np.zeros(6, dtype=np.float64)
        try:
            mujoco.mj_contactForce(self._model, data, int(contact_index), wrench)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            return False, None, None, None, None, None, "%s: %s" % (
                type(error).__name__,
                error,
            )
        if not np.all(np.isfinite(wrench)):
            return False, None, None, None, None, None, "non-finite mj_contactForce"
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        force_world = np.matmul(frame.T, wrench[:3])
        timestep = float(self._model.opt.timestep)
        impulse = wrench * timestep
        impulse_world = force_world * timestep
        return (
            True,
            tuple(float(value) for value in wrench),
            tuple(float(value) for value in force_world),
            float(wrench[0]),
            tuple(float(value) for value in impulse),
            tuple(float(value) for value in impulse_world),
            None,
        )

    def _contact_records(
        self,
        data: Any,
        *,
        source_phase: str,
        observation_index: Optional[int],
        high: Optional[int],
        inner: Optional[int],
        physics: Optional[int],
    ) -> Tuple[ContactPointRecord, ...]:
        np = _modules()[1]
        records: List[ContactPointRecord] = []
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (geom1, geom2) in self._pair_set:
                robot_geom, obstacle_geom = geom1, geom2
            elif (geom2, geom1) in self._pair_set:
                robot_geom, obstacle_geom = geom2, geom1
            else:
                continue
            if source_phase == "live_solver_phase_preintegration_geometry":
                force = self._contact_force(data, contact_index, contact)
                force_semantics = (
                    "live_solver_constraint_wrench; impulse fields are wrench_"
                    "times_physics_timestep_estimates"
                )
            else:
                # mj_forward on the state clone recomputes a hypothetical next
                # constraint solve.  It is useful for contact geometry, but its
                # wrench was not applied during the just-finished integration.
                force = (
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "omitted because clone-forwarded wrench was not applied",
                )
                force_semantics = "omitted_recomputed_not_applied"
            robot_body = int(self._model.geom_bodyid[robot_geom])
            obstacle_body = int(self._model.geom_bodyid[obstacle_geom])
            distance = float(contact.dist)
            efc_address = int(contact.efc_address)
            explicit_pair_id = self._explicit_pair_ids.get(
                (robot_geom, obstacle_geom)
            )
            position = np.asarray(contact.pos, dtype=np.float64)
            normal = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)[0]
            if (
                not math.isfinite(distance)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(normal))
            ):
                raise RuntimeError("MuJoCo returned non-finite selected-obstacle contact data")
            records.append(
                ContactPointRecord(
                    source_phase=source_phase,
                    observation_index=observation_index,
                    high_level_index=high,
                    inner_control_index=inner,
                    physics_substep_index=physics,
                    mujoco_contact_index=contact_index,
                    is_physical_nonpositive_distance_contact=bool(distance <= 0.0),
                    within_registered_near_contact_tolerance=bool(
                        distance <= self._near_contact_tolerance_m
                    ),
                    solver_constraint_active=bool(efc_address >= 0),
                    efc_address=efc_address,
                    mujoco_geom1_id=geom1,
                    mujoco_geom1_name=_geom_name(self._model, geom1),
                    mujoco_geom2_id=geom2,
                    mujoco_geom2_name=_geom_name(self._model, geom2),
                    robot_geom_id=robot_geom,
                    robot_geom_name=_geom_name(self._model, robot_geom),
                    robot_body_id=robot_body,
                    robot_body_name=_body_name(self._model, robot_body),
                    obstacle_geom_id=obstacle_geom,
                    obstacle_geom_name=_geom_name(self._model, obstacle_geom),
                    obstacle_body_id=obstacle_body,
                    obstacle_body_name=_body_name(self._model, obstacle_body),
                    contact_distance_m=distance,
                    contact_includemargin_m=float(contact.includemargin),
                    robot_geom_margin_m=float(self._model.geom_margin[robot_geom]),
                    robot_geom_gap_m=float(self._model.geom_gap[robot_geom]),
                    obstacle_geom_margin_m=float(
                        self._model.geom_margin[obstacle_geom]
                    ),
                    obstacle_geom_gap_m=float(self._model.geom_gap[obstacle_geom]),
                    explicit_pair_id=explicit_pair_id,
                    explicit_pair_margin_m=(
                        float(self._model.pair_margin[explicit_pair_id])
                        if explicit_pair_id is not None
                        else None
                    ),
                    explicit_pair_gap_m=(
                        float(self._model.pair_gap[explicit_pair_id])
                        if explicit_pair_id is not None
                        else None
                    ),
                    position_world_m=tuple(float(value) for value in position),
                    frame_normal_mujoco_geom1_to_geom2_world=tuple(
                        float(value) for value in normal
                    ),
                    force_available=force[0],
                    force_semantics=force_semantics,
                    force_contact_frame_n=force[1],
                    force_world_n=force[2],
                    normal_force_n=force[3],
                    impulse_estimate_contact_frame_ns=force[4],
                    impulse_estimate_world_ns=force[5],
                    force_unavailable_reason=force[6],
                )
            )
        return tuple(records)

    def observe_post_integration(
        self,
        sim: Any,
        *,
        high_level_index: int,
        inner_control_index: int,
        physics_substep_index: int,
    ) -> None:
        """Observe exactly one state after its MuJoCo integration substep."""

        if self._require_settled_static_motion:
            self.require_settled_obstacle_motion_admissible()
        self._validate_index(
            high_level_index, inner_control_index, physics_substep_index
        )
        model, data = _raw_model_data(sim)
        if model is not self._model:
            raise ValueError("measurement callback received a different MuJoCo model")
        observation_index = self._observations
        index = (
            int(high_level_index),
            int(inner_control_index),
            int(physics_substep_index),
        )
        if self._first_index is None:
            self._first_index = index
        # Preserve the pre-integration solver contact/force evidence before
        # cloning.  Nonpositive records are interval-level collision authority;
        # the clone below independently measures the post-integration state.
        live_solver_contacts = self._contact_records(
            data,
            source_phase="live_solver_phase_preintegration_geometry",
            observation_index=observation_index,
            high=int(high_level_index),
            inner=int(inner_control_index),
            physics=int(physics_substep_index),
        )
        self._live_solver_contacts.extend(live_solver_contacts)

        forwarded = clone_forwarded_state(
            self._model, data, reusable_clone=self._forwarded_clone
        )
        self._forwarded_clone = forwarded
        self._observe_raw_distance(forwarded, observation_index)
        self._observe_sample_clearance(forwarded, observation_index)
        post_candidates = self._contact_records(
            forwarded,
            source_phase="post_integration_recomputed",
            observation_index=observation_index,
            high=int(high_level_index),
            inner=int(inner_control_index),
            physics=int(physics_substep_index),
        )
        self._candidate_contacts.extend(post_candidates)
        self._physical_contacts.extend(
            event
            for event in post_candidates
            if event.is_physical_nonpositive_distance_contact
        )
        self._observe_drift(forwarded, observation_index)
        self._last_index = index
        self._observations += 1
        if (
            self._terminate_on_static_drift
            and self._first_static_drift_crossing == observation_index
        ):
            raise StaticObstacleDriftInadmissible(
                "selected-obstacle static-pose assumption became inadmissible at "
                "observation %d (%s); terminate before another physics substep"
                % (observation_index, self._first_static_drift_reason)
            )

    def _raw_advisory_result(self) -> RawGeomDistanceAdvisory:
        if self._raw_minimum_record is None:
            return RawGeomDistanceAdvisory(
                available=False,
                authority="advisory_only_never_contact_authority",
                minimum_distance_m=None,
                distance_query_limit_m=self._query_limit_m,
                minimum_observation_index=None,
                robot_geom_id=None,
                robot_geom_name=None,
                obstacle_geom_id=None,
                obstacle_geom_name=None,
                fromto_world_m=None,
            )
        observation, robot_geom, obstacle_geom, fromto = self._raw_minimum_record
        return RawGeomDistanceAdvisory(
            available=True,
            authority="advisory_only_never_contact_authority",
            minimum_distance_m=float(self._raw_minimum),
            distance_query_limit_m=self._query_limit_m,
            minimum_observation_index=observation,
            robot_geom_id=robot_geom,
            robot_geom_name=_geom_name(self._model, robot_geom),
            obstacle_geom_id=obstacle_geom,
            obstacle_geom_name=_geom_name(self._model, obstacle_geom),
            fromto_world_m=fromto if fromto else None,
        )

    def _sample_clearance_result(self) -> SampleCoverageClearance:
        if self._sample_minimum_record is None or not math.isfinite(
            self._sample_minimum
        ):
            raise RuntimeError("exact sample-to-OBB clearance was not measured")
        observation, sample_id, obstacle_geom = self._sample_minimum_record
        sample_by_id = {sample.sample_id: sample for sample in self._samples}
        sample = sample_by_id[sample_id]
        return SampleCoverageClearance(
            available=True,
            authority="exact_sample_to_obb_plus_certified_coverage_lower_bound",
            minimum_exact_sample_to_obb_distance_m=float(self._sample_minimum),
            certified_coverage_radius_m=self._coverage_radius_m,
            full_surface_clearance_lower_bound_m=float(
                self._sample_minimum - self._coverage_radius_m
            ),
            minimum_observation_index=observation,
            sample_id=sample_id,
            robot_geom_id=sample.geom_id,
            obstacle_geom_id=obstacle_geom,
        )

    def result(self) -> FullRobotObstacleMeasurement:
        """Return the typed final result, failing closed on zero observations."""

        if self._observations == 0 or self._first_index is None or self._last_index is None:
            raise RuntimeError("No post-integration physics substeps were measured")
        sample_clearance = self._sample_clearance_result()
        if sample_clearance.full_surface_clearance_lower_bound_m is None:
            raise RuntimeError("coverage lower bound is unavailable")
        lower_bound = float(sample_clearance.full_surface_clearance_lower_bound_m)
        settled_physical = tuple(
            self._settled_state.physical_contact_point_records
        )
        post_state_physical = tuple(self._physical_contacts)
        live_solver = tuple(self._live_solver_contacts)
        live_solver_physical = tuple(
            record
            for record in live_solver
            if record.is_physical_nonpositive_distance_contact
        )

        def rollout_record_order(record: ContactPointRecord) -> Tuple[int, int, int]:
            observation = (
                int(record.observation_index)
                if record.observation_index is not None
                else -1
            )
            phase_order = (
                0
                if record.source_phase
                == "live_solver_phase_preintegration_geometry"
                else 1
            )
            return observation, phase_order, int(record.mujoco_contact_index)

        rollout_physical = tuple(
            sorted(
                live_solver_physical + post_state_physical,
                key=rollout_record_order,
            )
        )
        all_physical = settled_physical + rollout_physical
        any_contact = bool(all_physical)
        if all_physical:
            contact_minimum = min(
                event.contact_distance_m for event in all_physical
            )
            d_sim = min(lower_bound, contact_minimum, 0.0)
        else:
            d_sim = lower_bound
        clamped = bool(any_contact and d_sim < lower_bound)
        geom_drifts = tuple(
            ObstacleGeomDrift(
                geom_id=geom_id,
                geom_name=_geom_name(self._model, geom_id),
                body_id=int(self._model.geom_bodyid[geom_id]),
                body_name=_body_name(
                    self._model, int(self._model.geom_bodyid[geom_id])
                ),
                maximum_translation_m=float(record["max_translation"]),
                maximum_rotation_rad=float(record["max_rotation"]),
                maximum_surface_point_displacement_m=float(
                    record["max_surface"]
                ),
                final_translation_m=float(record["final_translation"]),
                final_rotation_rad=float(record["final_rotation"]),
                final_surface_point_displacement_m=float(
                    record["final_surface"]
                ),
                maximum_translation_observation_index=int(
                    record["max_translation_observation"]
                ),
                maximum_rotation_observation_index=int(
                    record["max_rotation_observation"]
                ),
                maximum_surface_displacement_observation_index=int(
                    record["max_surface_observation"]
                ),
            )
            for geom_id, record in sorted(self._drift.items())
        )
        drift = ObstacleDriftSummary(
            reference="every_selected_obstacle_collision_geom_pose_after_settling",
            maximum_translation_m=max(
                value.maximum_translation_m for value in geom_drifts
            ),
            maximum_rotation_rad=max(value.maximum_rotation_rad for value in geom_drifts),
            maximum_surface_point_displacement_m=max(
                value.maximum_surface_point_displacement_m
                for value in geom_drifts
            ),
            surface_drift_threshold_m=self._surface_drift_threshold_m,
            surface_drift_threshold_crossed=bool(
                self._first_surface_drift_crossing is not None
            ),
            first_surface_drift_threshold_crossing_observation_index=(
                self._first_surface_drift_crossing
            ),
            geoms=geom_drifts,
        )
        post_candidates = tuple(self._candidate_contacts)
        rollout_candidates = tuple(
            sorted(live_solver + post_candidates, key=rollout_record_order)
        )
        all_candidates = (
            self._settled_state.candidate_contact_point_records
            + rollout_candidates
        )
        return FullRobotObstacleMeasurement(
            observed_physics_substeps=self._observations,
            first_index=self._first_index,
            last_index=self._last_index,
            any_robot_obstacle_contact=any_contact,
            link56_obstacle_contact=any(
                event.robot_geom_id in self._link56_geoms for event in all_physical
            ),
            rollout_any_robot_obstacle_contact=bool(rollout_physical),
            rollout_link56_obstacle_contact=any(
                event.robot_geom_id in self._link56_geoms
                for event in rollout_physical
            ),
            post_state_any_robot_obstacle_contact=bool(post_state_physical),
            post_state_link56_obstacle_contact=any(
                record.robot_geom_id in self._link56_geoms
                for record in post_state_physical
            ),
            total_candidate_contact_point_record_count=len(all_candidates),
            total_physical_contact_point_record_count=len(all_physical),
            rollout_phase_physical_contact_point_record_count=len(
                rollout_physical
            ),
            post_state_candidate_contact_point_record_count=len(post_candidates),
            post_state_solver_active_contact_point_record_count=sum(
                event.solver_constraint_active for event in post_candidates
            ),
            post_state_within_near_contact_tolerance_point_record_count=sum(
                event.within_registered_near_contact_tolerance
                for event in post_candidates
            ),
            post_state_physical_contact_point_record_count=len(
                post_state_physical
            ),
            first_physical_contact_point_record=(
                all_physical[0] if all_physical else None
            ),
            first_candidate_contact_point_record=(
                all_candidates[0] if all_candidates else None
            ),
            first_live_solver_physical_contact_point_record=(
                live_solver_physical[0] if live_solver_physical else None
            ),
            first_post_state_physical_contact_point_record=(
                post_state_physical[0] if post_state_physical else None
            ),
            post_state_candidate_contact_point_records=post_candidates,
            post_state_physical_contact_point_records=post_state_physical,
            live_solver_any_robot_obstacle_contact=bool(live_solver_physical),
            live_solver_link56_obstacle_contact=any(
                record.robot_geom_id in self._link56_geoms
                for record in live_solver_physical
            ),
            live_solver_candidate_contact_point_record_count=len(live_solver),
            live_solver_active_contact_point_record_count=sum(
                event.solver_constraint_active for event in live_solver
            ),
            live_solver_within_near_contact_tolerance_point_record_count=sum(
                event.within_registered_near_contact_tolerance
                for event in live_solver
            ),
            live_solver_nonpositive_contact_point_record_count=sum(
                event.is_physical_nonpositive_distance_contact
                for event in live_solver
            ),
            live_solver_phase_contact_point_records=live_solver,
            settled_state=self._settled_state,
            raw_mj_geom_distance=self._raw_advisory_result(),
            sample_clearance=sample_clearance,
            obstacle_pose_drift=drift,
            D_sim_min_m=float(d_sim),
            D_sim_semantics=(
                "union_of_settled_live_solver_and_forwarded_post_state_"
                "nonpositive_contacts_plus_exact_obb_coverage_lower_bound"
            ),
            physical_contact_distance_semantics="mujoco_contact_dist_le_0",
            near_contact_tolerance_m=self._near_contact_tolerance_m,
            contact_authority_clamped_D_sim=clamped,
        )


__all__ = [
    "ContactPointRecord",
    "FullRobotObstacleMeasurement",
    "FullRobotObstacleMonitor",
    "ObstacleGeomDrift",
    "ObstacleDriftSummary",
    "RawGeomDistanceAdvisory",
    "ResolvedGeomSets",
    "SampleCoverageClearance",
    "SettledObstacleBodyVelocity",
    "SettledObstacleMotionAdmissibility",
    "SettledObstacleMotionInadmissible",
    "StaticObstacleDriftInadmissible",
    "SettledStateSnapshot",
    "clone_forwarded_state",
    "copy_integration_state",
    "descendant_body_ids",
    "resolve_collision_geom_sets",
]
