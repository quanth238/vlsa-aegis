"""Coverage-controlled samples for MuJoCo robot collision surfaces.

The static pilot initially accepts robot collision meshes and boxes.  Any
other geom type is rejected explicitly rather than being approximated without
an audited covering radius.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from main.poisson_fullbody.robot_samples import BodySample, body_world_pose


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("NumPy is required for surface sampling") from error
    return np


@dataclass(frozen=True)
class TriangleSurfaceSamples:
    points: Any
    triangle_count: int
    certified_cover_radius_m: float
    requested_epsilon_m: float


@dataclass(frozen=True)
class RobotSampleSet:
    samples: Tuple[BodySample, ...]
    epsilon_m: float
    maximum_triangle_cover_radius_m: float
    geom_records: Tuple[Dict[str, Any], ...]
    coverage_semantics: str


def sample_triangle_surface(
    vertices_m: Any,
    faces: Any,
    *,
    epsilon_m: float,
) -> TriangleSurfaceSamples:
    """Sample each triangle on a barycentric lattice with diameter <= epsilon.

    Every point in a triangle lies in a lattice subtriangle whose vertices are
    samples.  Since that subtriangle's diameter is at most the longest parent
    edge divided by the subdivision count, its nearest sampled vertex is no
    farther than that diameter.
    """

    np = _numpy()
    vertices = np.asarray(vertices_m, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("vertices_m must have finite shape (V, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError("faces must have non-empty shape (F, 3)")
    if np.any(triangles < 0) or np.any(triangles >= vertices.shape[0]):
        raise ValueError("face index is outside vertices")
    if not math.isfinite(epsilon_m) or epsilon_m <= 0.0:
        raise ValueError("epsilon_m must be finite and positive")

    points: List[Any] = []
    maximum_bound = 0.0
    for face in triangles:
        triangle = vertices[face]
        edges = (
            np.linalg.norm(triangle[1] - triangle[0]),
            np.linalg.norm(triangle[2] - triangle[1]),
            np.linalg.norm(triangle[0] - triangle[2]),
        )
        longest = float(max(edges))
        if longest <= 1e-15:
            raise ValueError("degenerate collision triangle")
        # The paper's coverage premise uses open epsilon-balls: equality is
        # not enough. ``ceil(L / epsilon)`` can produce an exactly-epsilon
        # subtriangle when L is an integer multiple of epsilon, so take the
        # next integer strictly above the ratio.
        subdivisions = max(1, int(math.floor(longest / float(epsilon_m))) + 1)
        maximum_bound = max(maximum_bound, longest / float(subdivisions))
        for first in range(subdivisions + 1):
            for second in range(subdivisions + 1 - first):
                third = subdivisions - first - second
                point = (
                    float(first) * triangle[0]
                    + float(second) * triangle[1]
                    + float(third) * triangle[2]
                ) / float(subdivisions)
                points.append(point)
    point_array = np.asarray(points, dtype=np.float64)
    # Deduplication removes shared triangle edges without moving any sample.
    point_array = np.unique(point_array, axis=0)
    return TriangleSurfaceSamples(
        points=point_array,
        triangle_count=int(triangles.shape[0]),
        certified_cover_radius_m=float(maximum_bound),
        requested_epsilon_m=float(epsilon_m),
    )


def box_triangle_mesh(half_extents_m: Sequence[float]) -> Tuple[Any, Any]:
    np = _numpy()
    half = np.asarray(half_extents_m, dtype=np.float64)
    if half.shape != (3,) or not np.all(np.isfinite(half)) or np.any(half <= 0.0):
        raise ValueError("box half extents must be a positive finite 3-vector")
    vertices = np.array(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    # Vertex index uses bits x,y,z after the ordering above.
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _name(model: Any, raw_model: Any, mujoco: Any, kind: str, index: int) -> str:
    wrapper_method = getattr(model, "%s_id2name" % kind, None)
    if wrapper_method is not None:
        value = wrapper_method(int(index))
    else:
        object_type = (
            mujoco.mjtObj.mjOBJ_BODY if kind == "body" else mujoco.mjtObj.mjOBJ_GEOM
        )
        value = mujoco.mj_id2name(raw_model, object_type, int(index))
    return "unnamed_%s_%d" % (kind, index) if value is None else str(value)


def _mesh_vertices_and_hull(raw_model: Any, mesh_id: int) -> Tuple[Any, Any]:
    np = _numpy()
    try:
        from scipy.spatial import ConvexHull
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("SciPy is required to reconstruct collision mesh hulls") from error
    vertex_address = int(raw_model.mesh_vertadr[mesh_id])
    vertex_count = int(raw_model.mesh_vertnum[mesh_id])
    if vertex_count < 4:
        raise ValueError("collision mesh has too few vertices")
    vertices = np.asarray(
        raw_model.mesh_vert[vertex_address : vertex_address + vertex_count],
        dtype=np.float64,
    ).copy()
    hull = ConvexHull(vertices)
    return vertices, np.asarray(hull.simplices, dtype=np.int64)


def build_robot_collision_samples(
    model: Any,
    data: Any,
    *,
    body_ids: Iterable[int],
    epsilon_m: float,
) -> RobotSampleSet:
    """Build body-local samples from collision-enabled mesh/box geoms."""

    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("MuJoCo is required for collision sample extraction") from error
    np = _numpy()
    raw_model = getattr(model, "_model", model)
    selected_bodies = {int(value) for value in body_ids}
    if not selected_bodies:
        raise ValueError("body_ids must be non-empty")
    if any(value < 0 or value >= int(raw_model.nbody) for value in selected_bodies):
        raise ValueError("selected body ID is outside model.nbody")

    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    samples: List[BodySample] = []
    records: List[Dict[str, Any]] = []
    maximum_bound = 0.0
    for geom_id in range(int(raw_model.ngeom)):
        body_id = int(raw_model.geom_bodyid[geom_id])
        if body_id not in selected_bodies:
            continue
        contype = int(raw_model.geom_contype[geom_id])
        conaffinity = int(raw_model.geom_conaffinity[geom_id])
        if not (contype or conaffinity):
            continue
        geom_type = int(raw_model.geom_type[geom_id])
        if geom_type == mesh_type:
            mesh_id = int(raw_model.geom_dataid[geom_id])
            vertices, faces = _mesh_vertices_and_hull(raw_model, mesh_id)
            geometry_kind = "compiled_mesh_convex_hull"
        elif geom_type == box_type:
            vertices, faces = box_triangle_mesh(raw_model.geom_size[geom_id, :3])
            geometry_kind = "exact_box_faces"
        else:
            raise ValueError(
                "unsupported robot collision geom type %d for geom %d"
                % (geom_type, geom_id)
            )
        surface = sample_triangle_surface(vertices, faces, epsilon_m=epsilon_m)
        maximum_bound = max(maximum_bound, surface.certified_cover_radius_m)

        body_position, body_rotation = body_world_pose(data, body_id)
        geom_position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        geom_rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        rotation_body_from_geom = body_rotation.T @ geom_rotation
        position_body_from_geom = body_rotation.T @ (geom_position - body_position)
        body_points = (
            surface.points @ rotation_body_from_geom.T + position_body_from_geom
        )
        body_name = _name(model, raw_model, mujoco, "body", body_id)
        geom_name = _name(model, raw_model, mujoco, "geom", geom_id)
        for point in body_points:
            samples.append(
                BodySample(
                    sample_id=len(samples),
                    body_id=body_id,
                    body_name=body_name,
                    geom_id=geom_id,
                    geom_name=geom_name,
                    point_body_local_m=tuple(float(value) for value in point),
                )
            )
        records.append(
            {
                "geom_id": geom_id,
                "geom_name": geom_name,
                "body_id": body_id,
                "body_name": body_name,
                "geometry_kind": geometry_kind,
                "triangle_count": surface.triangle_count,
                "sample_count": int(body_points.shape[0]),
                "certified_triangle_cover_radius_m": surface.certified_cover_radius_m,
            }
        )
    if not samples:
        raise ValueError("selected bodies expose no supported collision geoms")
    return RobotSampleSet(
        samples=tuple(samples),
        epsilon_m=float(epsilon_m),
        maximum_triangle_cover_radius_m=float(maximum_bound),
        geom_records=tuple(records),
        coverage_semantics=(
            "triangle_lattice_cover_of_compiled_convex_hull_or_exact_box; "
            "MuJoCo collision-semantic equivalence requires allocation audit"
        ),
    )


def audit_finite_surface_coverage(
    sparse_points_m: Any,
    independent_dense_points_m: Any,
    *,
    epsilon_m: float,
) -> Dict[str, Any]:
    """Empirically audit an independent finite surface point set."""

    np = _numpy()
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("SciPy is required for the finite coverage audit") from error
    sparse = np.asarray(sparse_points_m, dtype=np.float64)
    dense = np.asarray(independent_dense_points_m, dtype=np.float64)
    if sparse.ndim != 2 or sparse.shape[1] != 3 or sparse.shape[0] == 0:
        raise ValueError("sparse_points_m must have non-empty shape (N, 3)")
    if dense.ndim != 2 or dense.shape[1] != 3 or dense.shape[0] == 0:
        raise ValueError("independent_dense_points_m must have non-empty shape (M, 3)")
    if not np.all(np.isfinite(sparse)) or not np.all(np.isfinite(dense)):
        raise ValueError("coverage points must be finite")
    distances, _ = cKDTree(sparse).query(dense, k=1)
    maximum = float(np.max(distances))
    return {
        "sparse_sample_count": int(sparse.shape[0]),
        "independent_audit_count": int(dense.shape[0]),
        "maximum_nearest_distance_m": maximum,
        "epsilon_m": float(epsilon_m),
        "passed": bool(maximum < float(epsilon_m)),
        "semantics": (
            "strict_open_ball_finite_independent_audit_not_continuous_proof"
        ),
    }
