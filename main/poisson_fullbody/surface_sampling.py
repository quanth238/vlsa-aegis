"""Coverage-controlled samples for MuJoCo robot collision surfaces.

Meshes and boxes use a triangle-lattice certificate.  Exact MuJoCo cylinders
use an analytic parameter-grid certificate over the lateral surface and both
end caps.  Any other geom type is rejected explicitly rather than being
approximated without an audited covering radius.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
from main.poisson_fullbody.robot_samples import BodySample, body_world_pose


MAX_CYLINDER_ANGULAR_SAMPLES = 100000
MAX_CYLINDER_AXIAL_INTERVALS = 100000
MAX_CYLINDER_RADIAL_INTERVALS = 100000
MAX_CYLINDER_RAW_SAMPLE_COUNT = 1000000
COVERAGE_SEMANTICS = (
    "strict_open_ball_surface_cover_from_triangle_lattices_for_compiled_"
    "convex_hulls_and_exact_boxes_or_analytic_parameter_grids_for_exact_"
    "cylinders; MuJoCo collision-semantic equivalence requires allocation audit"
)


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
class CylinderSurfaceSamples:
    points: Any
    angular_sample_count: int
    axial_interval_count: int
    cap_radial_interval_count: int
    certified_cover_radius_m: float
    requested_epsilon_m: float


@dataclass(frozen=True)
class RobotSampleSet:
    samples: Tuple[BodySample, ...]
    epsilon_m: float
    maximum_surface_cover_radius_m: float
    sample_ledger_sha256: str
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


def sample_cylinder_surface(
    radius_m: float,
    half_length_m: float,
    *,
    epsilon_m: float,
) -> CylinderSurfaceSamples:
    """Sample an exact z-axis MuJoCo cylinder with a strict cover certificate.

    The side uses a periodic angular grid and an endpoint-inclusive axial
    grid.  Each end cap uses the same angular grid on concentric radial rings,
    plus the center.  For angular spacing ``2*pi/n``, the worst same-radius
    chord is ``2*r*sin(pi/(2*n))``.  Combining that chord with the nearest
    axial or radial interval by Pythagoras bounds every exact cylinder-surface
    point, including the cap interiors; no polygonal approximation is used.
    """

    np = _numpy()
    radius = float(radius_m)
    half_length = float(half_length_m)
    epsilon = float(epsilon_m)
    for value, label in (
        (radius, "radius_m"),
        (half_length, "half_length_m"),
        (epsilon, "epsilon_m"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("%s must be finite and positive" % label)

    # Keeping each orthogonal contribution strictly below epsilon/sqrt(2)
    # proves that the Euclidean product-grid bound is strictly below epsilon.
    component_limit = epsilon / math.sqrt(2.0)
    if not math.isfinite(component_limit) or component_limit <= 0.0:
        raise ValueError(
            "cylinder sampling exceeds the finite float64 implementation range"
        )
    angular_ratio = component_limit / (2.0 * radius)
    if not math.isfinite(angular_ratio) or angular_ratio <= 0.0:
        raise ValueError(
            "cylinder angular sampling exceeds the finite float64 implementation range"
        )
    if angular_ratio >= 1.0:
        angular_count = 3
    else:
        threshold = math.pi / (2.0 * math.asin(angular_ratio))
        if not math.isfinite(threshold):
            raise ValueError(
                "cylinder angular sampling exceeds the finite float64 implementation range"
            )
        if threshold >= MAX_CYLINDER_ANGULAR_SAMPLES:
            raise ValueError("cylinder angular sampling exceeds the implementation cap")
        angular_count = max(3, int(math.floor(threshold)) + 1)
    angular_bound = 2.0 * radius * math.sin(
        math.pi / (2.0 * float(angular_count))
    )
    while not angular_bound < component_limit:
        angular_count += 1
        if angular_count > MAX_CYLINDER_ANGULAR_SAMPLES:
            raise ValueError("cylinder angular sampling exceeds the implementation cap")
        angular_bound = 2.0 * radius * math.sin(
            math.pi / (2.0 * float(angular_count))
        )

    axial_ratio = half_length / component_limit
    radial_ratio = radius / (2.0 * component_limit)
    if axial_ratio >= MAX_CYLINDER_AXIAL_INTERVALS:
        raise ValueError("cylinder axial sampling exceeds the implementation cap")
    if radial_ratio >= MAX_CYLINDER_RADIAL_INTERVALS:
        raise ValueError("cylinder cap sampling exceeds the implementation cap")
    axial_intervals = max(1, int(math.floor(axial_ratio)) + 1)
    radial_intervals = max(1, int(math.floor(radial_ratio)) + 1)
    raw_sample_count = (
        angular_count * (axial_intervals + 1)
        + 2 * (1 + angular_count * radial_intervals)
    )
    if raw_sample_count > MAX_CYLINDER_RAW_SAMPLE_COUNT:
        raise ValueError("cylinder surface sampling exceeds the implementation sample cap")

    side_bound = math.hypot(
        angular_bound, half_length / float(axial_intervals)
    )
    cap_bound = math.hypot(
        angular_bound, radius / (2.0 * float(radial_intervals))
    )
    certified_bound = max(side_bound, cap_bound)
    if not certified_bound < epsilon:
        raise AssertionError("constructed cylinder cover is not strictly below epsilon")

    angles = (
        2.0
        * math.pi
        * np.arange(angular_count, dtype=np.float64)
        / float(angular_count)
    )
    cosines = np.cos(angles)
    sines = np.sin(angles)
    norms = np.hypot(cosines, sines)
    cosines = cosines / norms
    sines = sines / norms
    points: List[Any] = []
    for axial in range(axial_intervals + 1):
        if axial == 0:
            z = -half_length
        elif axial == axial_intervals:
            z = half_length
        else:
            z = -half_length + (
                2.0 * half_length * float(axial) / float(axial_intervals)
            )
        for cosine, sine in zip(cosines, sines):
            points.append(np.asarray([radius * cosine, radius * sine, z]))
    for z in (-half_length, half_length):
        points.append(np.asarray([0.0, 0.0, z]))
        for radial in range(1, radial_intervals + 1):
            ring_radius = (
                radius
                if radial == radial_intervals
                else radius * float(radial) / float(radial_intervals)
            )
            for cosine, sine in zip(cosines, sines):
                points.append(
                    np.asarray([ring_radius * cosine, ring_radius * sine, z])
                )
    point_array = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    return CylinderSurfaceSamples(
        points=point_array,
        angular_sample_count=int(angular_count),
        axial_interval_count=int(axial_intervals),
        cap_radial_interval_count=int(radial_intervals),
        certified_cover_radius_m=float(certified_bound),
        requested_epsilon_m=epsilon,
    )


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
    body_ids: Optional[Iterable[int]] = None,
    geom_ids: Optional[Iterable[int]] = None,
    epsilon_m: float,
) -> RobotSampleSet:
    """Build body-local samples from supported collision-relevant geoms.

    ``geom_ids`` is the authoritative mode for a resolved collision set and
    includes explicit-pair-only geoms.  ``body_ids`` is retained for synthetic
    construction and selects mask-enabled geoms in those bodies.  Exactly one
    selection mode is required.
    """

    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("MuJoCo is required for collision sample extraction") from error
    np = _numpy()
    raw_model = getattr(model, "_model", model)
    if (body_ids is None) == (geom_ids is None):
        raise ValueError("exactly one of body_ids or geom_ids is required")
    selected_bodies = (
        set() if body_ids is None else {int(value) for value in body_ids}
    )
    selected_geoms = (
        set() if geom_ids is None else {int(value) for value in geom_ids}
    )
    selection_authority = (
        "mask_enabled_geoms_under_selected_bodies"
        if geom_ids is None
        else "authoritative_resolved_geom_ids"
    )
    if body_ids is not None:
        if not selected_bodies:
            raise ValueError("body_ids must be non-empty")
        if any(
            value < 0 or value >= int(raw_model.nbody)
            for value in selected_bodies
        ):
            raise ValueError("selected body ID is outside model.nbody")
    else:
        if not selected_geoms:
            raise ValueError("geom_ids must be non-empty")
        if any(
            value < 0 or value >= int(raw_model.ngeom)
            for value in selected_geoms
        ):
            raise ValueError("selected geom ID is outside model.ngeom")

    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    cylinder_type = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    samples: List[BodySample] = []
    records: List[Dict[str, Any]] = []
    maximum_bound = 0.0
    for geom_id in range(int(raw_model.ngeom)):
        body_id = int(raw_model.geom_bodyid[geom_id])
        if geom_ids is not None:
            if geom_id not in selected_geoms:
                continue
        else:
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
            geom_type_name = "mesh"
            surface = sample_triangle_surface(vertices, faces, epsilon_m=epsilon_m)
            certificate_kind = "analytic_triangle_lattice_covering_bound"
            surface_element_count = int(surface.triangle_count)
            certificate_parameters = {
                "triangle_count": int(surface.triangle_count),
                "requested_epsilon_m": float(surface.requested_epsilon_m),
            }
        elif geom_type == box_type:
            vertices, faces = box_triangle_mesh(raw_model.geom_size[geom_id, :3])
            geometry_kind = "exact_box_faces"
            geom_type_name = "box"
            surface = sample_triangle_surface(vertices, faces, epsilon_m=epsilon_m)
            certificate_kind = "analytic_triangle_lattice_covering_bound"
            surface_element_count = int(surface.triangle_count)
            certificate_parameters = {
                "triangle_count": int(surface.triangle_count),
                "requested_epsilon_m": float(surface.requested_epsilon_m),
            }
        elif geom_type == cylinder_type:
            surface = sample_cylinder_surface(
                float(raw_model.geom_size[geom_id, 0]),
                float(raw_model.geom_size[geom_id, 1]),
                epsilon_m=epsilon_m,
            )
            geometry_kind = "exact_cylinder_surface"
            geom_type_name = "cylinder"
            certificate_kind = "analytic_cylinder_parameter_grid_covering_bound"
            surface_element_count = int(
                surface.angular_sample_count
                * (
                    surface.axial_interval_count
                    + 2 * surface.cap_radial_interval_count
                )
            )
            certificate_parameters = {
                "angular_sample_count": int(surface.angular_sample_count),
                "axial_interval_count": int(surface.axial_interval_count),
                "cap_radial_interval_count": int(
                    surface.cap_radial_interval_count
                ),
                "requested_epsilon_m": float(surface.requested_epsilon_m),
                "implementation_caps": {
                    "maximum_angular_samples": MAX_CYLINDER_ANGULAR_SAMPLES,
                    "maximum_axial_intervals": MAX_CYLINDER_AXIAL_INTERVALS,
                    "maximum_radial_intervals": MAX_CYLINDER_RADIAL_INTERVALS,
                    "maximum_raw_sample_count": MAX_CYLINDER_RAW_SAMPLE_COUNT,
                },
            }
        else:
            raise ValueError(
                "unsupported robot collision geom type %d for geom %d"
                % (geom_type, geom_id)
            )
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
                "geom_type_id": geom_type,
                "geom_type_name": geom_type_name,
                "geom_size": [
                    float(value) for value in raw_model.geom_size[geom_id, :3]
                ],
                "contype": int(raw_model.geom_contype[geom_id]),
                "conaffinity": int(raw_model.geom_conaffinity[geom_id]),
                "mask_collision_enabled": bool(
                    int(raw_model.geom_contype[geom_id])
                    or int(raw_model.geom_conaffinity[geom_id])
                ),
                "selection_authority": selection_authority,
                "geometry_kind": geometry_kind,
                "certificate_kind": certificate_kind,
                "certificate_parameters": certificate_parameters,
                "surface_element_count": surface_element_count,
                "sample_count": int(body_points.shape[0]),
                "certified_surface_cover_radius_m": surface.certified_cover_radius_m,
            }
        )
    if not samples:
        raise ValueError("selected geometry set exposes no supported collision geoms")
    sample_ledger = [
        {
            "sample_id": int(sample.sample_id),
            "body_id": int(sample.body_id),
            "body_name": str(sample.body_name),
            "geom_id": int(sample.geom_id),
            "geom_name": str(sample.geom_name),
            "point_body_local_m": [
                float(value) for value in sample.point_body_local_m
            ],
            "source": str(sample.source),
        }
        for sample in samples
    ]
    return RobotSampleSet(
        samples=tuple(samples),
        epsilon_m=float(epsilon_m),
        maximum_surface_cover_radius_m=float(maximum_bound),
        sample_ledger_sha256=sha256_bytes(canonical_json_bytes(sample_ledger)),
        geom_records=tuple(records),
        coverage_semantics=COVERAGE_SEMANTICS,
    )


def validate_robot_sample_evidence(
    evidence: Mapping[str, Any],
    *,
    resolved_geom_ids: Sequence[int],
    resolved_geom_names: Sequence[str],
    resolved_body_ids: Sequence[int],
    roundtrip_field: str,
) -> Dict[str, int]:
    """Validate serialized full-robot sample evidence without simulator access."""

    if not isinstance(evidence, Mapping):
        raise ValueError("full-robot surface-sampling evidence must be an object")
    expected_ids = tuple(int(value) for value in resolved_geom_ids)
    expected_names = tuple(str(value) for value in resolved_geom_names)
    expected_bodies = {int(value) for value in resolved_body_ids}
    if (
        not expected_ids
        or len(expected_ids) != len(expected_names)
        or tuple(sorted(set(expected_ids))) != expected_ids
        or not expected_bodies
    ):
        raise ValueError("authoritative resolved geometry evidence is invalid")
    records = evidence.get("geom_records")
    if not isinstance(records, list) or len(records) != len(expected_ids):
        raise ValueError("surface records differ from authoritative geometry count")
    epsilon = float(evidence.get("epsilon_m"))
    maximum = float(evidence.get("maximum_surface_cover_radius_m"))
    if not (
        math.isfinite(epsilon)
        and epsilon > 0.0
        and math.isfinite(maximum)
        and 0.0 <= maximum < epsilon
    ):
        raise ValueError("serialized surface-cover radius is invalid")
    if evidence.get("coverage_semantics") != COVERAGE_SEMANTICS:
        raise ValueError("serialized surface-cover semantics differ")
    ledger_hash = evidence.get("sample_ledger_sha256")
    if not (
        isinstance(ledger_hash, str)
        and len(ledger_hash) == 64
        and all(character in "0123456789abcdef" for character in ledger_hash)
    ):
        raise ValueError("serialized surface-sample ledger hash is invalid")

    kind_contract = {
        "compiled_mesh_convex_hull": (7, "mesh", "analytic_triangle_lattice_covering_bound"),
        "exact_box_faces": (6, "box", "analytic_triangle_lattice_covering_bound"),
        "exact_cylinder_surface": (
            5,
            "cylinder",
            "analytic_cylinder_parameter_grid_covering_bound",
        ),
    }
    observed_ids = []
    observed_names = []
    sample_total = 0
    record_maximum = 0.0
    type_counts = {"mesh": 0, "box": 0, "cylinder": 0}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("surface record must be an object")
        geometry_kind = record.get("geometry_kind")
        if geometry_kind not in kind_contract:
            raise ValueError("surface record geometry kind is unsupported")
        type_id, type_name, certificate_kind = kind_contract[geometry_kind]
        if (
            record.get("geom_type_id") != type_id
            or record.get("geom_type_name") != type_name
            or record.get("certificate_kind") != certificate_kind
        ):
            raise ValueError("surface record type or certificate is inconsistent")
        observed_ids.append(int(record.get("geom_id")))
        observed_names.append(str(record.get("geom_name")))
        if int(record.get("body_id")) not in expected_bodies:
            raise ValueError("surface record body is outside authoritative robot tree")
        contype = int(record.get("contype"))
        conaffinity = int(record.get("conaffinity"))
        if record.get("mask_collision_enabled") is not bool(contype or conaffinity):
            raise ValueError("surface record collision-mask evidence is inconsistent")
        if record.get("selection_authority") != "authoritative_resolved_geom_ids":
            raise ValueError("surface record selection is not authoritative")
        count = int(record.get("sample_count"))
        elements = int(record.get("surface_element_count"))
        cover = float(record.get("certified_surface_cover_radius_m"))
        if count <= 0 or elements <= 0 or not (
            math.isfinite(cover) and 0.0 <= cover < epsilon
        ):
            raise ValueError("surface record certificate values are invalid")
        parameters = record.get("certificate_parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("surface record certificate parameters are absent")
        requested_epsilon = float(parameters.get("requested_epsilon_m"))
        if not math.isclose(
            requested_epsilon, epsilon, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("surface certificate epsilon differs")
        if type_name == "cylinder":
            if not all(
                isinstance(parameters.get(field), int)
                and not isinstance(parameters.get(field), bool)
                and parameters[field] > 0
                for field in (
                    "angular_sample_count",
                    "axial_interval_count",
                    "cap_radial_interval_count",
                )
            ):
                raise ValueError("cylinder certificate grid parameters are invalid")
            angular = int(parameters["angular_sample_count"])
            axial = int(parameters["axial_interval_count"])
            radial = int(parameters["cap_radial_interval_count"])
            expected_elements = angular * (axial + 2 * radial)
            expected_samples = angular * (axial + 1) + 2 * (
                1 + angular * (radial - 1)
            )
            if elements != expected_elements or count != expected_samples:
                raise ValueError("cylinder certificate counts are inconsistent")
            raw_size = record.get("geom_size")
            if not isinstance(raw_size, list) or len(raw_size) != 3:
                raise ValueError("cylinder geom_size is invalid")
            radius, half_length, unused_size = (
                float(raw_size[0]),
                float(raw_size[1]),
                float(raw_size[2]),
            )
            if not (
                math.isfinite(radius)
                and radius > 0.0
                and math.isfinite(half_length)
                and half_length > 0.0
                and math.isfinite(unused_size)
            ):
                raise ValueError("cylinder geom_size is invalid")
            angular_bound = 2.0 * radius * math.sin(
                math.pi / (2.0 * float(angular))
            )
            recomputed_cover = max(
                math.hypot(angular_bound, half_length / float(axial)),
                math.hypot(angular_bound, radius / (2.0 * float(radial))),
            )
            if not math.isclose(
                cover, recomputed_cover, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError("cylinder analytic cover was not reconstructed")
        elif not (
            isinstance(parameters.get("triangle_count"), int)
            and not isinstance(parameters.get("triangle_count"), bool)
            and parameters["triangle_count"] == elements
        ):
            raise ValueError("triangle certificate parameters are invalid")
        sample_total += count
        record_maximum = max(record_maximum, cover)
        type_counts[type_name] += 1
    if tuple(observed_ids) != expected_ids or tuple(observed_names) != expected_names:
        raise ValueError("surface records differ from authoritative geom identities")
    if int(evidence.get("sample_count")) != sample_total:
        raise ValueError("surface sample total differs from component records")
    if not math.isclose(maximum, record_maximum, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("surface maximum cover differs from component records")
    roundtrip = evidence.get(roundtrip_field)
    if not isinstance(roundtrip, Mapping):
        raise ValueError("surface rigid-roundtrip evidence is absent")
    roundtrip_error = float(roundtrip.get("maximum_roundtrip_error_m"))
    roundtrip_tolerance = float(roundtrip.get("tolerance_m"))
    if (
        roundtrip.get("passed") is not True
        or int(roundtrip.get("sample_count")) != sample_total
        or not math.isfinite(roundtrip_error)
        or not math.isfinite(roundtrip_tolerance)
        or roundtrip_error < 0.0
        or roundtrip_tolerance <= 0.0
        or roundtrip_error > roundtrip_tolerance
    ):
        raise ValueError("surface rigid-roundtrip evidence is invalid")
    return type_counts


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
