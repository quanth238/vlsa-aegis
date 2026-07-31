"""Conservative voxel and connected-domain construction for static PSFs.

Grid geometry has deliberately explicit semantics:

* ``cell_shape`` counts closed voxel cells;
* scalar field values live on ``vertex_shape == cell_shape + 1`` vertices;
* a raw cell is occupied when its closed AABB intersects any closed OBB;
* production epsilon buffering rasterizes the exact OBB epsilon-neighbourhood
  directly, rather than buffering an already-rasterized cell union.

The legacy cell-union helper remains for isolated morphology tests and reports
its two-rasterization conservatism explicitly.  Production uses exact
OBB--cell distance, so only the final covering cell contributes grid overreach.
"""

from collections import deque
from dataclasses import dataclass, field
import itertools
import math
import numbers

from .geometry import (
    OrientedBox,
    _numpy,
    obb_intersects_aabb,
    oriented_box_to_aabb_distance,
)


def _readonly_copy(array, dtype=None):
    np = _numpy()
    contiguous = np.array(array, dtype=dtype, order="C", copy=True)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape, order="C")


def _shape3(value, name, minimum):
    if isinstance(value, (str, bytes)):
        raise ValueError("{} must be a length-three integer sequence".format(name))
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError("{} must be a length-three integer sequence".format(name)) from exc
    if len(items) != 3:
        raise ValueError("{} must contain exactly three entries".format(name))
    parsed = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, numbers.Integral):
            raise ValueError("{} entries must be integers".format(name))
        if item < minimum:
            raise ValueError(
                "{} entries must be at least {}".format(name, minimum)
            )
        parsed.append(int(item))
    return tuple(parsed)


@dataclass(frozen=True)
class GridSpec:
    """Axis-aligned uniform-per-axis grid with explicit cell semantics."""

    lower: object
    spacing: object
    cell_shape: object

    def __post_init__(self):
        np = _numpy()
        lower = np.asarray(self.lower, dtype=np.float64)
        spacing = np.asarray(self.spacing, dtype=np.float64)
        if lower.shape != (3,):
            raise ValueError("lower must have shape (3,), got {}".format(lower.shape))
        if spacing.shape != (3,):
            raise ValueError(
                "spacing must have shape (3,), got {}".format(spacing.shape)
            )
        if not np.all(np.isfinite(lower)):
            raise ValueError("lower must contain only finite values")
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
            raise ValueError("spacing must be finite and strictly positive")
        cell_shape = _shape3(self.cell_shape, "cell_shape", 1)
        upper = lower + spacing * np.asarray(cell_shape, dtype=np.float64)
        if not np.all(np.isfinite(upper)):
            raise ValueError("grid upper bound overflows finite coordinates")

        object.__setattr__(self, "lower", _readonly_copy(lower, np.float64))
        object.__setattr__(self, "spacing", _readonly_copy(spacing, np.float64))
        object.__setattr__(self, "cell_shape", cell_shape)

    @classmethod
    def from_bounds(cls, lower, upper, vertex_shape):
        """Construct a grid from bounds while preserving vertex semantics."""

        np = _numpy()
        lo = np.asarray(lower, dtype=np.float64)
        hi = np.asarray(upper, dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("lower and upper must each have shape (3,)")
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError("lower and upper must contain only finite values")
        if np.any(hi <= lo):
            raise ValueError("upper must be strictly greater than lower per axis")
        vertices = _shape3(vertex_shape, "vertex_shape", 2)
        cells = tuple(value - 1 for value in vertices)
        spacing = (hi - lo) / np.asarray(cells, dtype=np.float64)
        return cls(lower=lo, spacing=spacing, cell_shape=cells)

    @property
    def vertex_shape(self):
        return tuple(value + 1 for value in self.cell_shape)

    @property
    def upper(self):
        np = _numpy()
        result = self.lower + self.spacing * np.asarray(
            self.cell_shape, dtype=np.float64
        )
        return _readonly_copy(result, np.float64)

    def vertex_axes(self):
        """Return the three one-dimensional vertex coordinate arrays."""

        np = _numpy()
        return tuple(
            self.lower[axis]
            + self.spacing[axis]
            * np.arange(self.vertex_shape[axis], dtype=np.float64)
            for axis in range(3)
        )

    def vertex(self, index):
        index = _index3(index, self.vertex_shape, "vertex index")
        np = _numpy()
        return self.lower + self.spacing * np.asarray(index, dtype=np.float64)

    def cell_bounds(self, index):
        index = _index3(index, self.cell_shape, "cell index")
        np = _numpy()
        lo = self.lower + self.spacing * np.asarray(index, dtype=np.float64)
        return lo, lo + self.spacing


def _index3(value, shape, name):
    parsed = _shape3_allow_zero(value, name)
    if any(index >= limit for index, limit in zip(parsed, shape)):
        raise IndexError("{} {} is outside shape {}".format(name, parsed, shape))
    return parsed


def _shape3_allow_zero(value, name):
    if isinstance(value, (str, bytes)):
        raise ValueError("{} must be a length-three integer sequence".format(name))
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError("{} must be a length-three integer sequence".format(name)) from exc
    if len(items) != 3:
        raise ValueError("{} must contain exactly three entries".format(name))
    parsed = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, numbers.Integral):
            raise ValueError("{} entries must be integers".format(name))
        if item < 0:
            raise ValueError("{} entries must be nonnegative".format(name))
        parsed.append(int(item))
    return tuple(parsed)


def _require_bool_mask(mask, expected_shape, name):
    np = _numpy()
    array = np.asarray(mask)
    if array.shape != tuple(expected_shape):
        raise ValueError(
            "{} must have shape {}, got {}".format(name, expected_shape, array.shape)
        )
    if array.dtype != np.bool_:
        raise TypeError("{} must have boolean dtype".format(name))
    return array


@dataclass(frozen=True)
class OccupancyResult:
    """Raw and exactly-once buffered occupancy with auditable semantics."""

    grid: GridSpec
    raw_cells: object
    buffered_cells: object
    epsilon: float
    numerical_tolerance: float
    grid_cell_diagonal: float = field(init=False)
    explicit_half_diagonal_padding: float = field(init=False)
    buffer_applied_once: bool = field(init=False)

    def __post_init__(self):
        if not isinstance(self.grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        raw = _require_bool_mask(self.raw_cells, self.grid.cell_shape, "raw_cells")
        buffered = _require_bool_mask(
            self.buffered_cells, self.grid.cell_shape, "buffered_cells"
        )
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) < 0.0:
            raise ValueError("epsilon must be finite and nonnegative")
        if not math.isfinite(float(self.numerical_tolerance)) or float(
            self.numerical_tolerance
        ) < 0.0:
            raise ValueError("numerical_tolerance must be finite and nonnegative")
        if _numpy().any(raw & ~buffered):
            raise ValueError("buffered_cells must contain every raw occupied cell")
        object.__setattr__(self, "raw_cells", _readonly_copy(raw, bool))
        object.__setattr__(self, "buffered_cells", _readonly_copy(buffered, bool))
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(
            self, "numerical_tolerance", float(self.numerical_tolerance)
        )
        object.__setattr__(
            self,
            "grid_cell_diagonal",
            float(_numpy().linalg.norm(self.grid.spacing)),
        )
        # Rasterizing every intersecting cell already contributes a one-cell
        # cover whose worst-case outward overreach is the full cell diagonal.
        # Record that discretization separately; never add a second half-cell
        # term to the requested epsilon.
        object.__setattr__(self, "explicit_half_diagonal_padding", 0.0)
        object.__setattr__(self, "buffer_applied_once", True)

    def audit_metadata(self):
        """Return scalar semantics suitable for a result/provenance ledger."""

        return {
            "buffer_epsilon_m": self.epsilon,
            "grid_cell_diagonal_m": self.grid_cell_diagonal,
            "raw_rasterization_conservatism_upper_bound_m": (
                self.grid_cell_diagonal
            ),
            "buffer_rasterization_conservatism_upper_bound_m": (
                self.grid_cell_diagonal
            ),
            "total_buffer_conservatism_upper_bound_m": self.grid_cell_diagonal,
            "grid_cover_conservatism_upper_bound_m": self.grid_cell_diagonal,
            "explicit_half_diagonal_padding_m": (
                self.explicit_half_diagonal_padding
            ),
            "buffer_applied_once": self.buffer_applied_once,
            "buffer_source": "direct_exact_obb_epsilon_neighborhood",
        }


def rasterize_oriented_boxes(grid, boxes, tolerance=None):
    """Conservatively rasterize a union of OBBs into closed grid cells."""

    np = _numpy()
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    boxes = tuple(boxes)
    if any(not isinstance(box, OrientedBox) for box in boxes):
        raise TypeError("every box must be an OrientedBox")
    if tolerance is not None:
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")

    occupied = np.zeros(grid.cell_shape, dtype=bool)
    axes = grid.vertex_axes()
    grid_upper = grid.upper
    for box in boxes:
        box_lower, box_upper = box.world_aabb()
        numeric_scale = max(
            1.0,
            float(np.max(np.abs(grid.lower))),
            float(np.max(np.abs(grid_upper))),
            float(np.max(np.abs(box_lower))),
            float(np.max(np.abs(box_upper))),
        )
        tau = (
            64.0 * np.finfo(np.float64).eps * numeric_scale
            if tolerance is None
            else tolerance
        )
        if np.any(box_upper < grid.lower - tau) or np.any(
            box_lower > grid_upper + tau
        ):
            continue

        ranges = []
        for axis in range(3):
            first = int(np.searchsorted(axes[axis], box_lower[axis], side="left")) - 1
            last = int(np.searchsorted(axes[axis], box_upper[axis], side="right")) - 1
            first = max(0, min(grid.cell_shape[axis] - 1, first))
            last = max(0, min(grid.cell_shape[axis] - 1, last))
            ranges.append(range(first, last + 1))

        for index in itertools.product(*ranges):
            if occupied[index]:
                continue
            lower, upper = grid.cell_bounds(index)
            if obb_intersects_aabb(box, lower, upper, tolerance=tau):
                occupied[index] = True
    return occupied


# Short audited spelling used in implementation notes.
rasterize_obb_cells = rasterize_oriented_boxes


def rasterize_buffered_oriented_boxes(grid, boxes, epsilon, tolerance=None):
    """Rasterize cells intersecting the exact OBB union epsilon-neighbourhood.

    Unlike buffering ``raw_cells``, this directly tests the closed-cell to
    exact-OBB Euclidean distance.  A marked cell therefore contains at least
    one point within epsilon of the physical obstacle, and its worst-case
    outward grid overreach is one cell diagonal, not two.
    """

    np = _numpy()
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    boxes = tuple(boxes)
    if any(not isinstance(box, OrientedBox) for box in boxes):
        raise TypeError("every box must be an OrientedBox")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative")
    tau = (
        _default_distance_tolerance(grid, epsilon)
        if tolerance is None
        else float(tolerance)
    )
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    occupied = np.zeros(grid.cell_shape, dtype=bool)
    axes = grid.vertex_axes()
    for box in boxes:
        box_lower, box_upper = box.world_aabb()
        search_lower = box_lower - epsilon - tau
        search_upper = box_upper + epsilon + tau
        if np.any(search_upper < grid.lower) or np.any(search_lower > grid.upper):
            continue
        ranges = []
        for axis in range(3):
            first = (
                int(np.searchsorted(axes[axis], search_lower[axis], side="left"))
                - 1
            )
            last = (
                int(np.searchsorted(axes[axis], search_upper[axis], side="right"))
                - 1
            )
            first = max(0, min(grid.cell_shape[axis] - 1, first))
            last = max(0, min(grid.cell_shape[axis] - 1, last))
            ranges.append(range(first, last + 1))
        for index in itertools.product(*ranges):
            if occupied[index]:
                continue
            lower, upper = grid.cell_bounds(index)
            distance = oriented_box_to_aabb_distance(
                box, lower, upper, tolerance=tau
            )
            if distance <= epsilon + tau:
                occupied[index] = True
    return occupied


def _default_distance_tolerance(grid, epsilon):
    np = _numpy()
    scale = max(
        1.0,
        abs(float(epsilon)),
        float(np.max(grid.spacing)),
        float(np.max(np.abs(grid.lower))),
        float(np.max(np.abs(grid.upper))),
    )
    return 64.0 * np.finfo(np.float64).eps * scale


def _shift_into(output, source, offset):
    if any(abs(delta) >= size for delta, size in zip(offset, source.shape)):
        return
    source_slices = []
    target_slices = []
    for delta, size in zip(offset, source.shape):
        if delta >= 0:
            source_slices.append(slice(0, size - delta))
            target_slices.append(slice(delta, size))
        else:
            source_slices.append(slice(-delta, size))
            target_slices.append(slice(0, size + delta))
    if any(item.start == item.stop for item in source_slices):
        return
    output[tuple(target_slices)] |= source[tuple(source_slices)]


def buffer_cell_union(grid, raw_cells, epsilon, tolerance=None):
    """Rasterize the Euclidean epsilon-neighbourhood of a raw cell union.

    For cells separated by integer offset ``k`` on an axis of spacing ``d``,
    the exact closed-AABB gap is ``max((abs(k)-1)*d, 0)``.  Offsets are tested
    against the 3-D Euclidean norm of those gaps.  All shifts read only from
    ``raw_cells``; newly marked cells never seed another iteration.  Thus the
    epsilon buffer is applied exactly once and no voxel half-diagonal is added.
    """

    np = _numpy()
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    raw = _require_bool_mask(raw_cells, grid.cell_shape, "raw_cells")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative")
    tau = _default_distance_tolerance(grid, epsilon) if tolerance is None else float(
        tolerance
    )
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not np.any(raw):
        return np.zeros(grid.cell_shape, dtype=bool)

    # abs(k)=1 has zero closed-cell gap.  ceil(epsilon/d)+1 is therefore the
    # largest potentially admissible integer offset on an axis.
    maxima = tuple(
        min(
            size - 1,
            int(math.ceil((epsilon + tau) / float(step))) + 1,
        )
        for step, size in zip(grid.spacing, grid.cell_shape)
    )
    buffered = np.zeros(grid.cell_shape, dtype=bool)
    offset_ranges = [range(-limit, limit + 1) for limit in maxima]
    for offset in itertools.product(*offset_ranges):
        gap = np.asarray(
            [
                max((abs(delta) - 1) * float(step), 0.0)
                for delta, step in zip(offset, grid.spacing)
            ],
            dtype=np.float64,
        )
        if float(np.linalg.norm(gap)) <= epsilon + tau:
            _shift_into(buffered, raw, offset)
    return buffered


def build_occupancy(grid, boxes, epsilon, tolerance=None):
    """Build raw occupancy and a direct exact-geometry epsilon rasterization."""

    boxes = tuple(boxes)
    if not boxes:
        raise ValueError("boxes must contain the selected obstacle geometry")
    if any(not isinstance(box, OrientedBox) for box in boxes):
        raise TypeError("every box must be an OrientedBox")
    raw = rasterize_oriented_boxes(grid, boxes, tolerance=tolerance)
    tau = (
        _default_distance_tolerance(grid, float(epsilon))
        if tolerance is None
        else float(tolerance)
    )
    buffered = rasterize_buffered_oriented_boxes(
        grid, boxes, epsilon, tolerance=tau
    )
    if not _numpy().any(raw):
        raise ValueError("selected obstacle geometry does not intersect the workspace")
    return OccupancyResult(
        grid=grid,
        raw_cells=raw,
        buffered_cells=buffered,
        epsilon=epsilon,
        numerical_tolerance=tau,
    )


def incident_cells(grid, point, tolerance=None):
    """Return every closed cell incident on a point.

    A point on an internal face belongs to both adjacent closed cells; on an
    edge it can belong to four, and on a vertex to eight.  Callers must inspect
    all returned cells instead of choosing one with ``floor``.
    """

    np = _numpy()
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point must be a finite vector with shape (3,)")
    tau = _default_distance_tolerance(grid, 0.0) if tolerance is None else float(
        tolerance
    )
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    if np.any(point < grid.lower - tau) or np.any(point > grid.upper + tau):
        raise ValueError("point lies outside the closed grid bounds")

    per_axis = []
    for axis in range(3):
        coordinate = min(
            max(float(point[axis]), float(grid.lower[axis])),
            float(grid.upper[axis]),
        )
        scaled = (coordinate - float(grid.lower[axis])) / float(
            grid.spacing[axis]
        )
        nearest = int(round(scaled))
        scaled_tolerance = tau / float(grid.spacing[axis])
        candidates = []
        if abs(scaled - nearest) <= scaled_tolerance:
            for candidate in (nearest - 1, nearest):
                if 0 <= candidate < grid.cell_shape[axis]:
                    candidates.append(candidate)
        else:
            candidate = int(math.floor(scaled))
            if 0 <= candidate < grid.cell_shape[axis]:
                candidates.append(candidate)
        if not candidates:
            raise ValueError("point has no incident cell inside the grid")
        per_axis.append(tuple(candidates))
    return tuple(itertools.product(*per_axis))


def _validated_seed_cells(shape, seed_cells):
    if seed_cells is None:
        raise ValueError("seed_cells or seed_points are required")
    try:
        seeds = tuple(seed_cells)
    except TypeError as exc:
        raise ValueError("seed_cells must be an iterable of cell indices") from exc
    # Treat one bare (i,j,k) tuple as one seed, not three scalar seeds.
    if len(seeds) == 3 and all(
        isinstance(item, numbers.Integral) and not isinstance(item, bool)
        for item in seeds
    ):
        seeds = (seeds,)
    if not seeds:
        raise ValueError("at least one seed cell is required")
    return tuple(_index3(seed, shape, "seed cell") for seed in seeds)


def connected_safe_component(blocked_cells, seed_cells):
    """Return the 6-connected free component shared by every seed cell."""

    np = _numpy()
    blocked = np.asarray(blocked_cells)
    if blocked.ndim != 3 or any(size < 1 for size in blocked.shape):
        raise ValueError("blocked_cells must be a nonempty 3-D array")
    if blocked.dtype != np.bool_:
        raise TypeError("blocked_cells must have boolean dtype")
    seeds = _validated_seed_cells(blocked.shape, seed_cells)
    if any(bool(blocked[seed]) for seed in seeds):
        raise ValueError("every seed cell must be free")

    component = np.zeros(blocked.shape, dtype=bool)
    first = seeds[0]
    component[first] = True
    queue = deque([first])
    while queue:
        current = queue.popleft()
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = list(current)
                neighbor[axis] += direction
                if neighbor[axis] < 0 or neighbor[axis] >= blocked.shape[axis]:
                    continue
                neighbor = tuple(neighbor)
                if blocked[neighbor] or component[neighbor]:
                    continue
                component[neighbor] = True
                queue.append(neighbor)
    missing = [seed for seed in seeds if not component[seed]]
    if missing:
        raise ValueError(
            "seed cells do not share one 6-connected free component: {}".format(
                missing
            )
        )
    return component


def _active_vertices(component):
    np = _numpy()
    active = np.zeros(tuple(size + 1 for size in component.shape), dtype=bool)
    for corner in itertools.product((0, 1), repeat=3):
        target = tuple(
            slice(corner[axis], corner[axis] + component.shape[axis])
            for axis in range(3)
        )
        active[target] |= component
    return active


def _component_face_boundary_vertices(component):
    np = _numpy()
    vertices = np.zeros(tuple(size + 1 for size in component.shape), dtype=bool)
    for axis in range(3):
        negative_neighbor = np.zeros(component.shape, dtype=bool)
        positive_neighbor = np.zeros(component.shape, dtype=bool)
        negative_target = [slice(None)] * 3
        negative_source = [slice(None)] * 3
        negative_target[axis] = slice(1, None)
        negative_source[axis] = slice(None, -1)
        negative_neighbor[tuple(negative_target)] = component[tuple(negative_source)]
        positive_target = [slice(None)] * 3
        positive_source = [slice(None)] * 3
        positive_target[axis] = slice(None, -1)
        positive_source[axis] = slice(1, None)
        positive_neighbor[tuple(positive_target)] = component[tuple(positive_source)]

        for positive, exposed in (
            (False, component & ~negative_neighbor),
            (True, component & ~positive_neighbor),
        ):
            indices = np.nonzero(exposed)
            if indices[0].size == 0:
                continue
            for offsets_other in itertools.product((0, 1), repeat=2):
                vertex_indices = []
                other_position = 0
                for coordinate_axis in range(3):
                    if coordinate_axis == axis:
                        vertex_indices.append(
                            indices[coordinate_axis] + (1 if positive else 0)
                        )
                    else:
                        vertex_indices.append(
                            indices[coordinate_axis]
                            + offsets_other[other_position]
                        )
                        other_position += 1
                vertices[tuple(vertex_indices)] = True
    return vertices


@dataclass(frozen=True)
class PoissonDomain:
    """One conservative connected cell component and its vertex partition."""

    component_cells: object
    active_vertices: object
    boundary_vertices: object
    interior_vertices: object

    def __post_init__(self):
        np = _numpy()
        component = np.asarray(self.component_cells)
        if component.ndim != 3 or component.dtype != np.bool_:
            raise TypeError("component_cells must be a boolean 3-D array")
        expected_vertices = tuple(size + 1 for size in component.shape)
        active = _require_bool_mask(
            self.active_vertices, expected_vertices, "active_vertices"
        )
        boundary = _require_bool_mask(
            self.boundary_vertices, expected_vertices, "boundary_vertices"
        )
        interior = _require_bool_mask(
            self.interior_vertices, expected_vertices, "interior_vertices"
        )
        if np.any(boundary & ~active) or np.any(interior & ~active):
            raise ValueError("boundary/interior vertices must be subsets of active")
        if np.any(boundary & interior):
            raise ValueError("boundary and interior vertices must be disjoint")
        if not np.array_equal(boundary | interior, active):
            raise ValueError("boundary and interior must partition active vertices")
        if not np.any(interior):
            raise ValueError("domain is too thin: it has no interior vertex")

        object.__setattr__(self, "component_cells", _readonly_copy(component, bool))
        object.__setattr__(self, "active_vertices", _readonly_copy(active, bool))
        object.__setattr__(self, "boundary_vertices", _readonly_copy(boundary, bool))
        object.__setattr__(self, "interior_vertices", _readonly_copy(interior, bool))

    @property
    def component_cell_mask(self):
        return self.component_cells

    @property
    def active_vertex_mask(self):
        return self.active_vertices

    @property
    def valid_vertex_mask(self):
        return self.active_vertices

    @property
    def free_vertex_mask(self):
        return self.active_vertices

    @property
    def boundary_vertex_mask(self):
        return self.boundary_vertices

    @property
    def interior_mask(self):
        return self.interior_vertices

    @property
    def solve_mask(self):
        return self.interior_vertices


# Concise name used by the numerical design.
Domain = PoissonDomain


def build_connected_domain(
    grid,
    blocked_cells,
    seed_cells=None,
    seed_points=None,
    tolerance=None,
):
    """Construct the common safe component and its Poisson vertex boundary.

    Points on a grid face/edge/vertex contribute *all* incident closed cells.
    If any incident cell is blocked, the point is not a valid safe seed.  All
    seeds must lie in the same 6-connected component; disconnected components
    are never silently unioned.
    """

    np = _numpy()
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    blocked = _require_bool_mask(blocked_cells, grid.cell_shape, "blocked_cells")
    if seed_cells is not None and seed_points is not None:
        raise ValueError("provide seed_cells or seed_points, not both")

    if seed_points is not None:
        try:
            points = tuple(seed_points)
        except TypeError as exc:
            raise ValueError("seed_points must be an iterable of 3-D points") from exc
        # Treat one bare 3-vector as one point.
        if len(points) == 3 and all(
            isinstance(value, numbers.Real) and not isinstance(value, bool)
            for value in points
        ):
            points = (points,)
        if not points:
            raise ValueError("at least one seed point is required")
        all_incident = []
        for point in points:
            cells = incident_cells(grid, point, tolerance=tolerance)
            if any(bool(blocked[cell]) for cell in cells):
                raise ValueError(
                    "a seed point touches a blocked cell under closed-cell semantics"
                )
            all_incident.extend(cells)
        # Preserve deterministic order while removing duplicates.
        seed_cells = tuple(dict.fromkeys(all_incident))

    seeds = _validated_seed_cells(grid.cell_shape, seed_cells)
    component = connected_safe_component(blocked, seeds)
    active = _active_vertices(component)
    boundary = _component_face_boundary_vertices(component)

    # Fail closed: every active vertex lacking a complete in-domain 6-neighbor
    # stencil is Dirichlet boundary.  This also captures the outer grid box.
    all_six_active = np.zeros(active.shape, dtype=bool)
    if all(size >= 3 for size in active.shape):
        core = (
            active[1:-1, 1:-1, 1:-1]
            & active[:-2, 1:-1, 1:-1]
            & active[2:, 1:-1, 1:-1]
            & active[1:-1, :-2, 1:-1]
            & active[1:-1, 2:, 1:-1]
            & active[1:-1, 1:-1, :-2]
            & active[1:-1, 1:-1, 2:]
        )
        all_six_active[1:-1, 1:-1, 1:-1] = core
    boundary |= active & ~all_six_active
    interior = active & ~boundary
    return PoissonDomain(
        component_cells=component,
        active_vertices=active,
        boundary_vertices=boundary,
        interior_vertices=interior,
    )
