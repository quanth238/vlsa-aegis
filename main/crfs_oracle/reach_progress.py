"""Additive reach-phase progress measurements for the CRFS feasibility study.

The policy plans ten actions but the baseline commits only the first five.  This
module therefore annotates exactly a five-action rollout.  Positions are read
directly from MuJoCo state; callers must not substitute possibly stale
observables collected before a no-render rollout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


TARGET_OBJECT_NAME = "akita_black_bowl_1"
EXECUTED_REACH_ACTIONS = 5
EMPIRICAL_QUANTILE_METHOD = "inverted_cdf"
BRANCH_REPLAY_ATOL_M = 1e-9
TRACKED_REACH_SUBSTEPS = EXECUTED_REACH_ACTIONS * 25


def _position3(value: Sequence[float], *, label: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{label} must contain exactly three finite coordinates")
    return values


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    # The allocation's SafeLIBERO client is Python 3.8, before ``zip(strict=)``.
    # Both vectors have already passed the exact length-three snapshot check.
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _environment_chain(environment):
    """Yield a SafeLiberoCase / render wrapper / raw-domain chain once each."""
    current = environment
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = getattr(current, "env", None)


def _first_attribute(environment, attribute: str):
    for candidate in _environment_chain(environment):
        value = getattr(candidate, attribute, None)
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class ReachSnapshot:
    """Authoritative world positions captured from one MuJoCo state."""

    target_object_name: str
    active_obstacle_name: str
    eef_world_m: tuple[float, float, float]
    target_world_m: tuple[float, float, float]
    active_obstacle_world_m: tuple[float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)


def capture_reach_snapshot(
    environment,
    target_name: str,
    obstacle_name: str,
) -> ReachSnapshot:
    """Read EEF, target, and active-obstacle positions directly from MuJoCo.

    ``environment`` is the existing ``OffScreenRenderEnv``-like object.  The
    object body lookup lives on the wrapped LIBERO domain, while the EEF site
    and MuJoCo data are exposed by the outer environment.
    """
    if target_name != TARGET_OBJECT_NAME:
        raise ValueError(
            f"The task-0 reach diagnostic has a frozen target {TARGET_OBJECT_NAME!r}; "
            f"got {target_name!r}"
        )
    if not obstacle_name:
        raise ValueError("obstacle_name must be non-empty")

    body_ids = _first_attribute(environment, "obj_body_id")
    if body_ids is None:
        raise TypeError("environment does not expose the LIBERO obj_body_id mapping")
    missing = [name for name in (target_name, obstacle_name) if name not in body_ids]
    if missing:
        raise ValueError(f"Unknown LIBERO object body names: {missing}")

    robots = _first_attribute(environment, "robots")
    if not robots:
        raise TypeError("environment does not expose a robot with an EEF site")

    sim = _first_attribute(environment, "sim")
    if sim is None:
        raise TypeError("environment does not expose MuJoCo simulator data")

    eef_site_id = int(robots[0].eef_site_id)
    target_body_id = int(body_ids[target_name])
    obstacle_body_id = int(body_ids[obstacle_name])
    return ReachSnapshot(
        target_object_name=target_name,
        active_obstacle_name=obstacle_name,
        eef_world_m=_position3(sim.data.site_xpos[eef_site_id], label="EEF site position"),
        target_world_m=_position3(sim.data.body_xpos[target_body_id], label="target body position"),
        active_obstacle_world_m=_position3(
            sim.data.body_xpos[obstacle_body_id], label="active obstacle body position"
        ),
    )


@dataclass(frozen=True)
class ReachRolloutAnnotation:
    """Reach progress and scene displacement for one committed action prefix."""

    target_object_name: str
    active_obstacle_name: str
    executed_actions: int
    branch_target_world_m: tuple[float, float, float]
    start_eef_world_m: tuple[float, float, float]
    end_eef_world_m: tuple[float, float, float]
    start_target_world_m: tuple[float, float, float]
    end_target_world_m: tuple[float, float, float]
    start_active_obstacle_world_m: tuple[float, float, float]
    end_active_obstacle_world_m: tuple[float, float, float]
    start_distance_to_branch_target_m: float
    end_distance_to_branch_target_m: float
    reach_progress_m: float
    target_displacement_m: float
    active_obstacle_displacement_m: float
    maximum_target_displacement_m: float | None = None
    maximum_active_obstacle_displacement_m: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def annotate_reach_snapshots(
    start: ReachSnapshot,
    end: ReachSnapshot,
    *,
    executed_actions: int,
    maximum_target_displacement_m: float | None = None,
    maximum_active_obstacle_displacement_m: float | None = None,
) -> ReachRolloutAnnotation:
    """Annotate a five-action rollout using the branch-start bowl as target."""
    if int(executed_actions) != EXECUTED_REACH_ACTIONS:
        raise ValueError(
            f"Reach progress is registered for exactly {EXECUTED_REACH_ACTIONS} executed actions; "
            f"got {executed_actions}"
        )
    if start.target_object_name != end.target_object_name:
        raise ValueError("Start and end snapshots refer to different target objects")
    if start.active_obstacle_name != end.active_obstacle_name:
        raise ValueError("Start and end snapshots refer to different active obstacles")
    if start.target_object_name != TARGET_OBJECT_NAME:
        raise ValueError(f"Reach target must remain {TARGET_OBJECT_NAME!r}")

    # Freezing this point prevents apparent progress from pushing the bowl
    # toward the gripper during an otherwise invalid reach rollout.
    branch_target = start.target_world_m
    start_distance = _distance(start.eef_world_m, branch_target)
    end_distance = _distance(end.eef_world_m, branch_target)
    target_endpoint_displacement = _distance(start.target_world_m, end.target_world_m)
    obstacle_endpoint_displacement = _distance(
        start.active_obstacle_world_m, end.active_obstacle_world_m
    )
    target_maximum_displacement = (
        target_endpoint_displacement
        if maximum_target_displacement_m is None
        else float(maximum_target_displacement_m)
    )
    obstacle_maximum_displacement = (
        obstacle_endpoint_displacement
        if maximum_active_obstacle_displacement_m is None
        else float(maximum_active_obstacle_displacement_m)
    )
    for label, maximum, endpoint in (
        ("target", target_maximum_displacement, target_endpoint_displacement),
        ("active obstacle", obstacle_maximum_displacement, obstacle_endpoint_displacement),
    ):
        if not math.isfinite(maximum) or maximum < 0.0:
            raise ValueError(f"Maximum {label} displacement must be finite and non-negative")
        if maximum + BRANCH_REPLAY_ATOL_M < endpoint:
            raise ValueError(
                f"Maximum {label} displacement cannot be below endpoint displacement"
            )
    return ReachRolloutAnnotation(
        target_object_name=start.target_object_name,
        active_obstacle_name=start.active_obstacle_name,
        executed_actions=EXECUTED_REACH_ACTIONS,
        branch_target_world_m=branch_target,
        start_eef_world_m=start.eef_world_m,
        end_eef_world_m=end.eef_world_m,
        start_target_world_m=start.target_world_m,
        end_target_world_m=end.target_world_m,
        start_active_obstacle_world_m=start.active_obstacle_world_m,
        end_active_obstacle_world_m=end.active_obstacle_world_m,
        start_distance_to_branch_target_m=start_distance,
        end_distance_to_branch_target_m=end_distance,
        reach_progress_m=start_distance - end_distance,
        target_displacement_m=target_endpoint_displacement,
        active_obstacle_displacement_m=obstacle_endpoint_displacement,
        maximum_target_displacement_m=target_maximum_displacement,
        maximum_active_obstacle_displacement_m=obstacle_maximum_displacement,
    )


def _tracked_reach_snapshots(
    rollout: Mapping[str, Any],
    *,
    target_name: str,
    obstacle_name: str,
) -> tuple[ReachSnapshot, ReachSnapshot, float, float]:
    tracking = rollout.get("tracked_body_motion")
    if not isinstance(tracking, Mapping):
        raise RuntimeError("reach rollout did not return opt-in tracked-body measurements")
    if int(tracking.get("substep_samples", -1)) != TRACKED_REACH_SUBSTEPS:
        raise RuntimeError(
            "reach rollout must track target and obstacle across exactly "
            f"{TRACKED_REACH_SUBSTEPS} physics substeps"
        )
    bodies = tracking.get("bodies")
    if not isinstance(bodies, Mapping):
        raise RuntimeError("tracked-body measurements have no bodies object")
    missing = [name for name in (target_name, obstacle_name) if name not in bodies]
    if missing:
        raise RuntimeError(f"tracked-body measurements are missing {missing}")
    target = bodies[target_name]
    obstacle = bodies[obstacle_name]
    if not isinstance(target, Mapping) or not isinstance(obstacle, Mapping):
        raise RuntimeError("tracked target and obstacle records must be objects")

    branch = ReachSnapshot(
        target_object_name=target_name,
        active_obstacle_name=obstacle_name,
        eef_world_m=_position3(
            tracking.get("branch_eef_world_m", ()), label="tracked branch EEF position"
        ),
        target_world_m=_position3(
            target.get("branch_world_m", ()), label="tracked branch target position"
        ),
        active_obstacle_world_m=_position3(
            obstacle.get("branch_world_m", ()), label="tracked branch obstacle position"
        ),
    )
    end = ReachSnapshot(
        target_object_name=target_name,
        active_obstacle_name=obstacle_name,
        eef_world_m=_position3(
            tracking.get("end_eef_world_m", ()), label="tracked end EEF position"
        ),
        target_world_m=_position3(
            target.get("end_world_m", ()), label="tracked end target position"
        ),
        active_obstacle_world_m=_position3(
            obstacle.get("end_world_m", ()), label="tracked end obstacle position"
        ),
    )
    target_maximum = float(target.get("maximum_displacement_m", float("nan")))
    obstacle_maximum = float(obstacle.get("maximum_displacement_m", float("nan")))
    for label, record, branch_position, end_position in (
        ("target", target, branch.target_world_m, end.target_world_m),
        (
            "active obstacle",
            obstacle,
            branch.active_obstacle_world_m,
            end.active_obstacle_world_m,
        ),
    ):
        reported_endpoint = float(record.get("endpoint_displacement_m", float("nan")))
        direct_endpoint = _distance(branch_position, end_position)
        if not math.isfinite(reported_endpoint) or not math.isclose(
            reported_endpoint,
            direct_endpoint,
            rel_tol=0.0,
            abs_tol=BRANCH_REPLAY_ATOL_M,
        ):
            raise RuntimeError(f"tracked {label} endpoint displacement is inconsistent")
    return branch, end, target_maximum, obstacle_maximum


def _assert_paired_branch(reference: ReachSnapshot, actual: ReachSnapshot) -> None:
    comparisons = (
        ("EEF", reference.eef_world_m, actual.eef_world_m),
        ("target", reference.target_world_m, actual.target_world_m),
        (
            "active obstacle",
            reference.active_obstacle_world_m,
            actual.active_obstacle_world_m,
        ),
    )
    mismatches = [
        f"{label}={_distance(expected, observed):.12g} m"
        for label, expected, observed in comparisons
        if _distance(expected, observed) > BRANCH_REPLAY_ATOL_M
    ]
    if mismatches:
        raise RuntimeError(
            "paired reset branch does not match the supplied initial snapshot: "
            + ", ".join(mismatches)
        )


def annotate_reach_rollout(
    environment,
    actions,
    *,
    target_name: str,
    obstacle_name: str,
    initial_snapshot: ReachSnapshot | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute and annotate one runner rollout without changing its result.

    ``environment`` is expected to expose the additive ``SafeLiberoCase``
    ``rollout`` method. If ``initial_snapshot`` is omitted, the wrapper is
    reset and settled once to capture the authoritative branch state before
    its ordinary rollout performs the paired reset and execution.
    """
    try:
        executed_actions = len(actions)
    except TypeError as error:
        raise TypeError("actions must be a sized five-action sequence") from error
    if executed_actions != EXECUTED_REACH_ACTIONS:
        raise ValueError(
            f"Reach progress is registered for exactly {EXECUTED_REACH_ACTIONS} executed actions; "
            f"got {executed_actions}"
        )

    if initial_snapshot is None:
        reset_and_settle = getattr(environment, "reset_and_settle", None)
        if reset_and_settle is None:
            raise TypeError("initial_snapshot is required when environment has no reset_and_settle method")
        reset_and_settle()
        initial_snapshot = capture_reach_snapshot(environment, target_name, obstacle_name)
    if initial_snapshot.target_object_name != target_name:
        raise ValueError("initial_snapshot does not match target_name")
    if initial_snapshot.active_obstacle_name != obstacle_name:
        raise ValueError("initial_snapshot does not match obstacle_name")

    rollout_fn = getattr(environment, "rollout", None)
    if rollout_fn is None:
        raise TypeError("environment does not expose the additive rollout method")
    rollout = rollout_fn(
        actions,
        tracked_body_names=(target_name, obstacle_name),
    )
    if not isinstance(rollout, Mapping):
        raise TypeError("environment.rollout must return a mapping")
    actual_branch, final_snapshot, target_maximum, obstacle_maximum = _tracked_reach_snapshots(
        rollout,
        target_name=target_name,
        obstacle_name=obstacle_name,
    )
    _assert_paired_branch(initial_snapshot, actual_branch)
    annotation = annotate_reach_snapshots(
        actual_branch,
        final_snapshot,
        executed_actions=executed_actions,
        maximum_target_displacement_m=target_maximum,
        maximum_active_obstacle_displacement_m=obstacle_maximum,
    )
    return dict(rollout), annotation.to_dict()


@dataclass(frozen=True)
class ReachProgressCalibration:
    """Frozen empirical lower-quartile threshold over positive progress."""

    p_min_m: float
    quantile: float
    quantile_method: str
    total_examples: int
    positive_examples: int
    nonpositive_examples: int
    minimum_positive_examples: int

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_reach_p_min(
    annotations: Iterable[ReachRolloutAnnotation],
    *,
    minimum_positive_examples: int,
) -> ReachProgressCalibration:
    """Return Q25 of positive five-action reach progress using an order statistic.

    Population membership and safety / phase-validity filtering remain the
    caller's responsibility so that excluded cases can be reported explicitly.
    This function never hides non-positive values: their count is recorded in
    the returned calibration artifact.
    """
    if minimum_positive_examples <= 0:
        raise ValueError("minimum_positive_examples must be positive")

    values = []
    for annotation in annotations:
        if annotation.executed_actions != EXECUTED_REACH_ACTIONS:
            raise ValueError("Calibration contains a non-five-action annotation")
        value = float(annotation.reach_progress_m)
        if not math.isfinite(value):
            raise ValueError("Calibration progress values must be finite")
        values.append(value)

    positive = sorted(value for value in values if value > 0.0)
    if len(positive) < minimum_positive_examples:
        raise ValueError(
            f"Need at least {minimum_positive_examples} positive reach examples, got {len(positive)}"
        )

    quantile = 0.25
    # NumPy's `method="inverted_cdf"`: x[ceil(q * n) - 1], with a zero-based
    # lower bound.  Keeping the implementation here avoids adding a local test
    # dependency solely for a deterministic empirical order statistic.
    order_index = max(0, math.ceil(quantile * len(positive)) - 1)
    return ReachProgressCalibration(
        p_min_m=positive[order_index],
        quantile=quantile,
        quantile_method=EMPIRICAL_QUANTILE_METHOD,
        total_examples=len(values),
        positive_examples=len(positive),
        nonpositive_examples=len(values) - len(positive),
        minimum_positive_examples=int(minimum_positive_examples),
    )
