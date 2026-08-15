"""Matched obstacle-proxy diagnostics for distal-link ellipsoids.

This module keeps the registered L5--L7 robot ellipsoids fixed and replaces
only the obstacle representation.  The ground-truth diagnostic uses the
compiled MuJoCo box union directly.  It is an oracle audit of a simulation
proxy, not a deployable perception method.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Sequence

try:  # Keep configuration-only imports usable on the lightweight desktop Python.
    import numpy as _np
except ImportError:  # pragma: no cover - allocation dependency
    _np = None

from .barrier import support_gap
from .geometry import Ellipsoid, primitive_bounding_radii
from .shadow import _geom_kind, _name, _numpy, _raw_model_data
from .sitl_candidate import _body_lineage, _obstacle_root_body_id


_BATCH_ACTIVE_SET_KERNEL = None


@dataclass(frozen=True)
class CompiledObstacleBox:
    """One live MuJoCo collision box belonging to the active obstacle."""

    geom_id: int
    geom_name: str
    body_id: int
    body_name: str
    center: Any
    rotation: Any
    half_extents_m: Any

    def to_record(self) -> dict[str, Any]:
        return {
            "geom_id": int(self.geom_id),
            "geom_name": self.geom_name,
            "body_id": int(self.body_id),
            "body_name": self.body_name,
            "center_m": self.center.tolist(),
            "rotation": self.rotation.tolist(),
            "half_extents_m": self.half_extents_m.tolist(),
        }


def compiled_obstacle_boxes(env: Any, active_obstacle_name: str) -> list[CompiledObstacleBox]:
    """Return every contact-capable compiled box under the obstacle root."""

    np = _numpy()
    model, data = _raw_model_data(env.sim)
    root_id = _obstacle_root_body_id(env.sim.model, active_obstacle_name)
    output: list[CompiledObstacleBox] = []
    unsupported: list[dict[str, Any]] = []
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        if root_id not in _body_lineage(model, body_id):
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        kind = _geom_kind(int(model.geom_type[geom_id]))
        if kind != "box":
            unsupported.append({"geom_id": geom_id, "kind": kind})
            continue
        half_extents = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        if half_extents.shape != (3,) or np.any(half_extents <= 0.0):
            raise ValueError("compiled obstacle box has invalid half extents")
        output.append(
            CompiledObstacleBox(
                geom_id=geom_id,
                geom_name=_name(env.sim.model, "geom", geom_id)
                or "unnamed_geom_%d" % geom_id,
                body_id=body_id,
                body_name=_name(env.sim.model, "body", body_id)
                or "unnamed_body_%d" % body_id,
                center=np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy(),
                rotation=np.asarray(data.geom_xmat[geom_id], dtype=np.float64)
                .reshape(3, 3)
                .copy(),
                half_extents_m=half_extents.copy(),
            )
        )
    if unsupported:
        raise ValueError("active obstacle has unsupported collision geoms: %s" % unsupported)
    if not output:
        raise ValueError("active obstacle has no compiled collision boxes")
    return output


def minimum_ellipsoid_quadratic_over_box(robot: Ellipsoid, box: CompiledObstacleBox) -> float:
    """Return ``min_{x in box} (x-c)^T S^-1 (x-c)`` exactly.

    In box-local coordinates this is a three-dimensional strictly convex
    bound-constrained quadratic.  Enumerating the lower/free/upper active set
    of each coordinate yields its global minimizer without iterative solver
    tolerances.  A value at most one means the solid box intersects the solid
    robot ellipsoid.
    """

    np = _numpy()
    inverse_shape = np.linalg.inv(robot.shape_matrix())
    offset = np.asarray(box.center - robot.center, dtype=np.float64)
    rotation = np.asarray(box.rotation, dtype=np.float64)
    half = np.asarray(box.half_extents_m, dtype=np.float64)
    hessian = rotation.T @ inverse_shape @ rotation
    linear = rotation.T @ inverse_shape @ offset
    constant = float(offset @ inverse_shape @ offset)
    best = math.inf
    for active in itertools.product((-1, 0, 1), repeat=3):
        fixed = [index for index, state in enumerate(active) if state != 0]
        free = [index for index, state in enumerate(active) if state == 0]
        local = np.zeros(3, dtype=np.float64)
        for index in fixed:
            local[index] = float(active[index]) * half[index]
        if free:
            free_array = np.asarray(free, dtype=np.int64)
            rhs = -linear[free_array]
            if fixed:
                fixed_array = np.asarray(fixed, dtype=np.int64)
                rhs -= hessian[np.ix_(free_array, fixed_array)] @ local[fixed_array]
            local[free_array] = np.linalg.solve(
                hessian[np.ix_(free_array, free_array)], rhs
            )
            if np.any(local[free_array] < -half[free_array] - 1.0e-12) or np.any(
                local[free_array] > half[free_array] + 1.0e-12
            ):
                continue
        value = float(local @ hessian @ local + 2.0 * linear @ local + constant)
        best = min(best, value)
    if not math.isfinite(best):
        raise ValueError("ellipsoid-box quadratic minimization failed")
    return max(0.0, best)


def _batched_active_set_kernel(
    robot_centers: Any,
    inverse_shapes: Any,
    box_centers: Any,
    box_rotations: Any,
    box_half_extents: Any,
) -> Any:
    """Numba-compatible exact 3-D active-set enumeration."""

    output = _np.empty(
        (robot_centers.shape[0], box_centers.shape[0]), dtype=_np.float64,
    )
    for robot_index in range(robot_centers.shape[0]):
        inverse = inverse_shapes[robot_index]
        for box_index in range(box_centers.shape[0]):
            rotation = box_rotations[box_index]
            half = box_half_extents[box_index]
            offset = box_centers[box_index] - robot_centers[robot_index]
            hessian = rotation.T @ inverse @ rotation
            linear = rotation.T @ inverse @ offset
            constant = offset @ inverse @ offset
            best = _np.inf
            for code in range(27):
                remainder = code
                state = _np.empty(3, dtype=_np.int64)
                for coordinate in range(3):
                    state[coordinate] = remainder % 3 - 1
                    remainder //= 3
                local = _np.zeros(3, dtype=_np.float64)
                free = _np.empty(3, dtype=_np.int64)
                free_count = 0
                for coordinate in range(3):
                    if state[coordinate] == 0:
                        free[free_count] = coordinate
                        free_count += 1
                    else:
                        local[coordinate] = state[coordinate] * half[coordinate]
                valid = True
                if free_count == 1:
                    i = free[0]
                    rhs = -linear[i]
                    for fixed in range(3):
                        if state[fixed] != 0:
                            rhs -= hessian[i, fixed] * local[fixed]
                    local[i] = rhs / hessian[i, i]
                    valid = -half[i] - 1.0e-12 <= local[i] <= half[i] + 1.0e-12
                elif free_count == 2:
                    i = free[0]
                    j = free[1]
                    rhs_i = -linear[i]
                    rhs_j = -linear[j]
                    for fixed in range(3):
                        if state[fixed] != 0:
                            rhs_i -= hessian[i, fixed] * local[fixed]
                            rhs_j -= hessian[j, fixed] * local[fixed]
                    determinant = (
                        hessian[i, i] * hessian[j, j]
                        - hessian[i, j] * hessian[j, i]
                    )
                    local[i] = (
                        rhs_i * hessian[j, j] - hessian[i, j] * rhs_j
                    ) / determinant
                    local[j] = (
                        hessian[i, i] * rhs_j - rhs_i * hessian[j, i]
                    ) / determinant
                    valid = (
                        -half[i] - 1.0e-12 <= local[i] <= half[i] + 1.0e-12
                        and -half[j] - 1.0e-12 <= local[j] <= half[j] + 1.0e-12
                    )
                elif free_count == 3:
                    local = _np.linalg.solve(hessian, -linear)
                    for coordinate in range(3):
                        valid = valid and (
                            -half[coordinate] - 1.0e-12
                            <= local[coordinate]
                            <= half[coordinate] + 1.0e-12
                        )
                if valid:
                    value = (
                        local @ hessian @ local
                        + 2.0 * linear @ local
                        + constant
                    )
                    if value < best:
                        best = value
            output[robot_index, box_index] = max(0.0, best)
    return output


def minimum_ellipsoid_quadratics_over_boxes(
    robots: Sequence[Ellipsoid], boxes: Sequence[CompiledObstacleBox],
) -> Any:
    """Return the exact pairwise quadratic matrix with one compiled call.

    The mathematical target is identical to
    :func:`minimum_ellipsoid_quadratic_over_box`; batching only removes Python
    overhead from internal-substep replay. Numba is an allocation dependency,
    not a new approximation.
    """

    if not robots or not boxes:
        raise ValueError("batched ellipsoid-box evaluation requires both inputs")
    np = _numpy()
    global _BATCH_ACTIVE_SET_KERNEL
    if _BATCH_ACTIVE_SET_KERNEL is None:
        try:
            from numba import njit
        except ImportError as error:  # pragma: no cover - allocation dependency
            raise RuntimeError("batched exact box evaluation requires Numba") from error
        _BATCH_ACTIVE_SET_KERNEL = njit(cache=False)(_batched_active_set_kernel)
    robot_centers = np.asarray([row.center for row in robots], dtype=np.float64)
    inverse_shapes = np.asarray(
        [np.linalg.inv(row.shape_matrix()) for row in robots], dtype=np.float64,
    )
    box_centers = np.asarray([box.center for box in boxes], dtype=np.float64)
    box_rotations = np.asarray([box.rotation for box in boxes], dtype=np.float64)
    box_half_extents = np.asarray(
        [box.half_extents_m for box in boxes], dtype=np.float64,
    )
    values = _BATCH_ACTIVE_SET_KERNEL(
        robot_centers, inverse_shapes, box_centers, box_rotations,
        box_half_extents,
    )
    if values.shape != (len(robots), len(boxes)) or not np.all(np.isfinite(values)):
        raise ValueError("batched ellipsoid-box evaluation failed")
    return values


def evaluate_obstacle_representations(
    links: Sequence[Ellipsoid],
    perceived_mvee: Ellipsoid,
    boxes: Sequence[CompiledObstacleBox],
    *,
    overlap_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Evaluate perceived MVEE, compiled box union, and gap approximation."""

    np = _numpy()
    if len(links) != 7:
        raise ValueError("obstacle proxy audit requires seven fixed robot ellipsoids")
    if not boxes:
        raise ValueError("obstacle proxy audit requires compiled obstacle boxes")
    perceived = np.asarray([support_gap(link, perceived_mvee) for link in links])
    exact_rows = []
    loewner_rows = []
    overlaps = []
    for link_index, link in enumerate(links):
        for box in boxes:
            quadratic = minimum_ellipsoid_quadratic_over_box(link, box)
            normalized_slack = math.sqrt(quadratic) - 1.0
            radii, source = primitive_bounding_radii(
                "box", box.half_extents_m, float(np.linalg.norm(box.half_extents_m))
            )
            box_ellipsoid = Ellipsoid(
                center=box.center,
                rotation=box.rotation,
                semiaxes_m=radii,
                body_id=box.body_id,
                body_name=box.body_name,
                geom_id=box.geom_id,
                geom_name=box.geom_name,
                bound_source=source,
            )
            loewner_gap = support_gap(link, box_ellipsoid)
            row = {
                "link_index": int(link_index),
                "link_body_name": link.body_name,
                "link_geom_name": link.geom_name,
                "obstacle_geom_id": int(box.geom_id),
                "obstacle_geom_name": box.geom_name,
                "minimum_mahalanobis_quadratic": float(quadratic),
                "normalized_radial_slack": float(normalized_slack),
                "exact_solid_overlap": bool(quadratic <= 1.0 + overlap_tolerance),
            }
            exact_rows.append(row)
            loewner_rows.append(
                {
                    "link_index": int(link_index),
                    "obstacle_geom_id": int(box.geom_id),
                    "support_gap_m": float(loewner_gap),
                }
            )
            if row["exact_solid_overlap"]:
                overlaps.append(row)
    closest = min(exact_rows, key=lambda item: item["normalized_radial_slack"])
    closest_loewner = min(loewner_rows, key=lambda item: item["support_gap_m"])
    row_minimum_slack = [
        min(
            item["normalized_radial_slack"]
            for item in exact_rows
            if int(item["link_index"]) == link_index
        )
        for link_index in range(7)
    ]
    row_overlap = [
        any(
            bool(item["exact_solid_overlap"])
            for item in exact_rows
            if int(item["link_index"]) == link_index
        )
        for link_index in range(7)
    ]
    return {
        "perceived_mvee_row_clearance_m": perceived.tolist(),
        "perceived_mvee_minimum_clearance_m": float(np.min(perceived)),
        "compiled_box_union_minimum_normalized_radial_slack": float(
            closest["normalized_radial_slack"]
        ),
        "compiled_box_union_any_exact_solid_overlap": bool(overlaps),
        "compiled_box_union_exact_overlap_count": len(overlaps),
        "compiled_box_union_row_minimum_normalized_radial_slack": row_minimum_slack,
        "compiled_box_union_row_any_exact_solid_overlap": row_overlap,
        "compiled_box_union_closest_pair": closest,
        "compiled_box_loewner_minimum_support_gap_m": float(
            closest_loewner["support_gap_m"]
        ),
        "compiled_box_loewner_closest_pair": closest_loewner,
        "semantics": {
            "perceived_mvee": "frozen_released_single_obstacle_mvee_support_gap",
            "compiled_box_union": (
                "exact_solid_intersection_between_fixed_robot_ellipsoid_and_each_"
                "compiled_obstacle_collision_box"
            ),
            "normalized_radial_slack": "dimensionless_not_metric_clearance",
            "loewner_support_gap": (
                "diagnostic_conservative_box_enclosure_and_centerline_support_gap"
            ),
        },
    }
