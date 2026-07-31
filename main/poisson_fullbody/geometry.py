"""Conservative geometry primitives for the opt-in Poisson feasibility arm.

The routines in this module deliberately operate on MuJoCo collision boxes,
not rendered meshes.  Intersection is *inclusive*: touching closed sets count
as occupied.  That convention is important when the result is used as the
zero boundary of a safety field.

NumPy is imported lazily so importing the baseline package does not add a new
runtime dependency to ordinary VLSA / AEGIS execution.
"""

from dataclasses import dataclass
import math


_NUMPY = None


def _numpy():
    global _NUMPY
    if _NUMPY is None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised on minimal installs
            raise RuntimeError(
                "Poisson geometry requires NumPy; install the opt-in "
                "feasibility dependencies first"
            ) from exc
        _NUMPY = np
    return _NUMPY


def _readonly_bytes_copy(value, dtype=None):
    """Return an independent C-order array backed by immutable bytes."""

    np = _numpy()
    contiguous = np.array(value, dtype=dtype, order="C", copy=True)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape, order="C")


def _finite_vector3(value, name):
    np = _numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("{} must have shape (3,), got {}".format(name, array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return _readonly_bytes_copy(array, np.float64)


def validate_aabb(lower, upper):
    """Return validated, immutable endpoints of a closed 3-D AABB.

    Degenerate boxes are accepted because a grid face, edge, or point is a
    meaningful closed query set for the inclusive SAT predicate.
    """

    np = _numpy()
    lo = _finite_vector3(lower, "lower")
    hi = _finite_vector3(upper, "upper")
    if np.any(hi < lo):
        raise ValueError("upper must be greater than or equal to lower per axis")
    return lo, hi


@dataclass(frozen=True)
class OrientedBox:
    """A closed oriented box in world coordinates.

    ``R`` maps box-local coordinates to world coordinates, so its columns are
    the three unit box axes.  ``half_extents`` are physical half lengths in
    metres.  ``geom_id`` is provenance only and may be an integer, string, or
    ``None`` for synthetic tests.
    """

    center: object
    R: object
    half_extents: object
    geom_id: object = None

    def __post_init__(self):
        np = _numpy()
        center = _finite_vector3(self.center, "center")
        half_extents = _finite_vector3(self.half_extents, "half_extents")
        if np.any(half_extents <= 0.0):
            raise ValueError("half_extents must be strictly positive")

        rotation = np.asarray(self.R, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("R must have shape (3, 3), got {}".format(rotation.shape))
        if not np.all(np.isfinite(rotation)):
            raise ValueError("R must contain only finite values")
        rotation = np.array(rotation, dtype=np.float64, order="C", copy=True)

        # A reflection is not a valid rigid-body rotation.  Use a tight but
        # scale-independent tolerance because all entries are dimensionless.
        orthogonality_error = np.linalg.norm(
            np.matmul(rotation.T, rotation) - np.eye(3), ord=np.inf
        )
        determinant = float(np.linalg.det(rotation))
        if orthogonality_error > 1.0e-9 or not math.isclose(
            determinant, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9
        ):
            raise ValueError(
                "R must be a proper orthonormal rotation (error={!r}, det={!r})".format(
                    orthogonality_error, determinant
                )
            )
        object.__setattr__(self, "center", center)
        object.__setattr__(
            self, "R", _readonly_bytes_copy(rotation, np.float64)
        )
        object.__setattr__(self, "half_extents", half_extents)

    @property
    def rotation(self):
        """Alias retained for callers that spell the pose field explicitly."""

        return self.R

    def vertices(self):
        """Return the eight world-space corners as an ``(8, 3)`` array."""

        np = _numpy()
        signs = np.asarray(
            [
                (-1.0, -1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, -1.0),
                (1.0, 1.0, 1.0),
            ],
            dtype=np.float64,
        )
        local = signs * self.half_extents[None, :]
        return self.center[None, :] + np.matmul(local, self.R.T)

    def world_aabb(self):
        """Return the tight world-axis-aligned bounding box of this OBB."""

        np = _numpy()
        radius = np.matmul(np.abs(self.R), self.half_extents)
        return self.center - radius, self.center + radius


def _separated_on_axis(box, aabb_center, aabb_half_extents, axis, tolerance):
    """Return whether one nonzero SAT axis strictly separates the boxes."""

    np = _numpy()
    norm = float(np.linalg.norm(axis))
    # A cross product of parallel axes is not a separating direction.  The
    # threshold is dimensionless because all candidate directions are unit
    # vectors before the cross product.
    if norm <= 64.0 * np.finfo(np.float64).eps:
        return False
    unit = axis / norm
    center_delta = abs(float(np.dot(box.center - aabb_center, unit)))
    box_radius = float(
        np.dot(box.half_extents, np.abs(np.matmul(box.R.T, unit)))
    )
    aabb_radius = float(np.dot(aabb_half_extents, np.abs(unit)))
    return center_delta > box_radius + aabb_radius + tolerance


def obb_intersects_aabb(box, lower, upper, tolerance=None):
    """Test inclusive OBB--AABB intersection with all 15 SAT axes.

    The candidate axes are the three world AABB axes, the three OBB axes, and
    their nine pairwise cross products.  Equality is intersection, as required
    for conservative closed-cell rasterization.

    ``tolerance`` is a numerical separation tolerance in world units.  When
    omitted, it is limited to floating-point roundoff at the scale of the two
    boxes; it is not a geometric padding parameter.
    """

    np = _numpy()
    if not isinstance(box, OrientedBox):
        raise TypeError("box must be an OrientedBox")
    lo, hi = validate_aabb(lower, upper)
    center = 0.5 * (lo + hi)
    half_extents = 0.5 * (hi - lo)

    if tolerance is None:
        scale = max(
            1.0,
            float(np.max(np.abs(box.center))),
            float(np.max(np.abs(center))),
            float(np.max(box.half_extents)),
            float(np.max(half_extents)),
        )
        tolerance = 64.0 * np.finfo(np.float64).eps * scale
    else:
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")

    world_axes = np.eye(3, dtype=np.float64)
    box_axes = [box.R[:, axis] for axis in range(3)]
    candidates = [world_axes[:, axis] for axis in range(3)] + box_axes
    candidates.extend(
        np.cross(world_axis, box_axis)
        for world_axis in world_axes.T
        for box_axis in box_axes
    )
    return not any(
        _separated_on_axis(box, center, half_extents, axis, tolerance)
        for axis in candidates
    )


# More explicit spelling for audit/report code.
oriented_box_intersects_aabb = obb_intersects_aabb


def _point_to_aabb_distance(point, lower, upper):
    np = _numpy()
    outside = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    return float(np.linalg.norm(outside))


def _segment_segment_distance(first_start, first_end, second_start, second_end):
    """Exact Euclidean distance between two closed 3-D line segments."""

    np = _numpy()
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    offset = first_start - second_start
    first_sq = float(np.dot(first_direction, first_direction))
    second_sq = float(np.dot(second_direction, second_direction))
    cross_term = float(np.dot(first_direction, second_direction))
    first_offset = float(np.dot(first_direction, offset))
    second_offset = float(np.dot(second_direction, offset))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, first_sq, second_sq
    )
    if first_sq <= tolerance and second_sq <= tolerance:
        return float(np.linalg.norm(first_start - second_start))
    if first_sq <= tolerance:
        first_parameter = 0.0
        second_parameter = min(max(second_offset / second_sq, 0.0), 1.0)
    else:
        if second_sq <= tolerance:
            second_parameter = 0.0
            first_parameter = min(max(-first_offset / first_sq, 0.0), 1.0)
        else:
            denominator = first_sq * second_sq - cross_term * cross_term
            if denominator > tolerance * max(first_sq, second_sq):
                first_parameter = min(
                    max(
                        (cross_term * second_offset - first_offset * second_sq)
                        / denominator,
                        0.0,
                    ),
                    1.0,
                )
            else:
                first_parameter = 0.0
            second_parameter = (
                cross_term * first_parameter + second_offset
            ) / second_sq
            if second_parameter < 0.0:
                second_parameter = 0.0
                first_parameter = min(max(-first_offset / first_sq, 0.0), 1.0)
            elif second_parameter > 1.0:
                second_parameter = 1.0
                first_parameter = min(
                    max((cross_term - first_offset) / first_sq, 0.0), 1.0
                )
    first_closest = first_start + first_parameter * first_direction
    second_closest = second_start + second_parameter * second_direction
    return float(np.linalg.norm(first_closest - second_closest))


def _box_edges(vertices):
    """Return the twelve edges for the bit-ordered vertices of either box."""

    return tuple(
        (vertices[index], vertices[index ^ (1 << axis)])
        for index in range(8)
        for axis in range(3)
        if (index & (1 << axis)) == 0
    )


def oriented_box_to_aabb_distance(box, lower, upper, tolerance=None):
    """Return the exact nonnegative Euclidean distance between two boxes.

    The closest features of two disjoint convex polyhedra are vertex--face or
    edge--edge pairs.  We enumerate both vertex-to-box directions and all 144
    edge pairs after an inclusive 15-axis intersection test.  This is slower
    than SAT but is used only in the obstacle-local rasterization broad phase.
    """

    np = _numpy()
    if not isinstance(box, OrientedBox):
        raise TypeError("box must be an OrientedBox")
    lo, hi = validate_aabb(lower, upper)
    if tolerance is not None:
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
    if obb_intersects_aabb(box, lo, hi, tolerance=tolerance):
        return 0.0

    aabb_vertices = np.asarray(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ],
        dtype=np.float64,
    )
    obb_vertices = box.vertices()
    candidates = [
        max(0.0, point_to_oriented_box_distance(point, box))
        for point in aabb_vertices
    ]
    candidates.extend(
        _point_to_aabb_distance(point, lo, hi) for point in obb_vertices
    )
    candidates.extend(
        _segment_segment_distance(first_start, first_end, second_start, second_end)
        for first_start, first_end in _box_edges(aabb_vertices)
        for second_start, second_end in _box_edges(obb_vertices)
    )
    return float(min(candidates))


def point_to_oriented_box_distance(point, box):
    """Return exact signed Euclidean point-to-OBB distance in world units.

    The result is negative in the box interior, zero on its closed surface,
    and positive outside.  Rotation is removed with ``R.T`` before evaluating
    the exact axis-aligned-box signed-distance formula.
    """

    np = _numpy()
    if not isinstance(box, OrientedBox):
        raise TypeError("box must be an OrientedBox")
    point = _finite_vector3(point, "point")
    local = np.matmul(box.R.T, point - box.center)
    offset = np.abs(local) - box.half_extents
    outside = float(np.linalg.norm(np.maximum(offset, 0.0)))
    inside = min(float(np.max(offset)), 0.0)
    return outside + inside


def minimum_point_to_oriented_boxes_distance(point, boxes):
    """Return the minimum signed point-to-box distance over a nonempty union.

    This CSG minimum is exact for points outside the box union, which is the
    admissible region where it is used as ``D_opt``.  For overlapping solids,
    a negative result remains a valid penetration indicator but is not claimed
    to be the exact signed distance to the exterior boundary of the union.
    """

    boxes = tuple(boxes)
    if not boxes:
        raise ValueError("boxes must contain at least one OrientedBox")
    if any(not isinstance(box, OrientedBox) for box in boxes):
        raise TypeError("every box must be an OrientedBox")
    # Validate the point once, then let the scalar helper operate on the
    # immutable copy.  Keeping the scalar helper public simplifies per-sample
    # audit ledgers.
    point = _finite_vector3(point, "point")
    return min(point_to_oriented_box_distance(point, box) for box in boxes)


# Concise aliases for clearance audit code.
signed_distance_to_obb = point_to_oriented_box_distance
minimum_signed_distance_to_obbs = minimum_point_to_oriented_boxes_distance
