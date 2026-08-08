"""Conservative ellipsoid bounds for MuJoCo collision geometry.

The ordinary AEGIS path never imports this module.  The opt-in observer uses
one ellipsoid per participating collision geom.  The distal three-envelope
variant fits a tight, certified MVEE around each live link-5/link-6/link-7
collision mesh instead of using its broad-phase bounding sphere.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence, Tuple


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("multi-link ellipsoid geometry requires NumPy") from error
    return np


def _finite_vector(value: Any, length: int, label: str) -> Any:
    np = _numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError("%s must be a finite length-%d vector" % (label, length))
    return np.array(array, dtype=np.float64, copy=True)


def _rotation(value: Any) -> Any:
    np = _numpy()
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    error = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord=np.inf))
    determinant = float(np.linalg.det(matrix))
    if error > 1.0e-8 or not math.isclose(
        determinant, 1.0, rel_tol=1.0e-8, abs_tol=1.0e-8
    ):
        raise ValueError("rotation must be proper and orthonormal")
    return np.array(matrix, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class Ellipsoid:
    """One world-frame ellipsoid with explicit simulator provenance."""

    center: Any
    rotation: Any
    semiaxes_m: Any
    body_id: int = -1
    body_name: str = ""
    geom_id: int = -1
    geom_name: str = ""
    bound_source: str = "unspecified"
    source_rbound_m: float | None = None
    source_geom_kind: str | None = None
    source_geom_size_m: Any | None = None
    source_body_names: tuple[str, ...] = ()
    source_geom_names: tuple[str, ...] = ()
    enclosure_certificate: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        np = _numpy()
        center = _finite_vector(self.center, 3, "center")
        rotation = _rotation(self.rotation)
        semiaxes = _finite_vector(self.semiaxes_m, 3, "semiaxes_m")
        if np.any(semiaxes <= 0.0):
            raise ValueError("ellipsoid semiaxes must be strictly positive")
        if self.source_rbound_m is not None:
            radius = float(self.source_rbound_m)
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError("source_rbound_m must be finite and positive")
        source_size = self.source_geom_size_m
        if source_size is not None:
            source_size = _finite_vector(source_size, 3, "source_geom_size_m")
        certificate = self.enclosure_certificate
        if certificate is not None:
            if certificate.get("verified") is not True:
                raise ValueError("ellipsoid enclosure certificate must be verified")
            certificate = dict(certificate)
        source_bodies = tuple(str(value) for value in self.source_body_names)
        source_geoms = tuple(str(value) for value in self.source_geom_names)
        if any(not value for value in source_bodies):
            raise ValueError("source_body_names must not contain empty names")
        if any(not value for value in source_geoms):
            raise ValueError("source_geom_names must not contain empty names")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "semiaxes_m", semiaxes)
        object.__setattr__(self, "source_geom_size_m", source_size)
        object.__setattr__(self, "source_body_names", source_bodies)
        object.__setattr__(self, "source_geom_names", source_geoms)
        object.__setattr__(self, "enclosure_certificate", certificate)

    def shape_matrix(self) -> Any:
        np = _numpy()
        return self.rotation @ np.diag(self.semiaxes_m ** 2) @ self.rotation.T

    def support_radius(self, direction_world: Sequence[float]) -> float:
        np = _numpy()
        direction = _finite_vector(direction_world, 3, "direction_world")
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-15:
            raise ValueError("support direction must be nonzero")
        unit = direction / norm
        return float(math.sqrt(float(unit @ self.shape_matrix() @ unit)))

    def to_record(self) -> dict[str, Any]:
        return {
            "center_m": self.center.tolist(),
            "rotation": self.rotation.tolist(),
            "semiaxes_m": self.semiaxes_m.tolist(),
            "body_id": int(self.body_id),
            "body_name": self.body_name,
            "geom_id": int(self.geom_id),
            "geom_name": self.geom_name,
            "bound_source": self.bound_source,
            "source_rbound_m": self.source_rbound_m,
            "source_geom_kind": self.source_geom_kind,
            "source_geom_size_m": (
                None
                if self.source_geom_size_m is None
                else self.source_geom_size_m.tolist()
            ),
            "source_body_names": list(self.source_body_names),
            "source_geom_names": list(self.source_geom_names),
            "enclosure_certificate": self.enclosure_certificate,
        }


def minimum_volume_enclosing_ellipsoid(
    points_world: Any,
    *,
    body_name: str,
    geom_name: str,
    body_id: int = -1,
    geom_id: int = -1,
    source_body_names: Sequence[str],
    source_geom_names: Sequence[str],
    relative_padding: float = 1.0e-9,
    tolerance: float = 1.0e-4,
    max_iterations: int = 20000,
    certificate_metadata: Mapping[str, Any] | None = None,
    include_source_points: bool = False,
) -> Ellipsoid:
    """Fit a surface-following MVEE and certify exact vertex containment.

    Khachiyan's algorithm estimates the minimum-volume enclosing ellipsoid.
    A final exact Mahalanobis inflation places the farthest source vertex on
    the surface. Because an ellipsoid is convex, containing every compiled
    mesh vertex also contains their convex hull.
    """

    np = _numpy()
    points = np.asarray(points_world, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < 4
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("source points must be a finite Nx3 array with N >= 4")
    padding = float(relative_padding)
    if not math.isfinite(padding) or padding < 0.0 or padding > 1.0e-3:
        raise ValueError("relative_padding must be finite and in [0, 1e-3]")
    fit_tolerance = float(tolerance)
    if not math.isfinite(fit_tolerance) or fit_tolerance <= 0.0:
        raise ValueError("MVEE tolerance must be finite and positive")
    if isinstance(max_iterations, bool) or int(max_iterations) < 1:
        raise ValueError("MVEE max_iterations must be a positive integer")
    offset = np.mean(points, axis=0)
    shifted = points - offset
    if int(np.linalg.matrix_rank(shifted, tol=1.0e-12)) != 3:
        raise ValueError("collision vertices are degenerate")
    count, dimension = shifted.shape
    homogeneous = np.vstack((shifted.T, np.ones(count, dtype=np.float64)))
    weights = np.full(count, 1.0 / float(count), dtype=np.float64)
    update_error = math.inf
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        moment = (homogeneous * weights) @ homogeneous.T
        inverse = np.linalg.inv(moment)
        leverage = np.sum(homogeneous * (inverse @ homogeneous), axis=0)
        maximum_index = int(np.argmax(leverage))
        maximum = float(leverage[maximum_index])
        numerator = maximum - float(dimension + 1)
        denominator = float(dimension + 1) * (maximum - 1.0)
        step = 0.0 if numerator <= 0.0 else numerator / denominator
        updated = (1.0 - step) * weights
        updated[maximum_index] += step
        update_error = float(np.linalg.norm(updated - weights))
        weights = updated
        if update_error <= fit_tolerance:
            converged = True
            break
    if not converged:
        raise ValueError("MVEE fit did not converge within max_iterations")
    shifted_center = shifted.T @ weights
    center = shifted_center + offset
    covariance = (
        shifted.T @ (weights[:, None] * shifted)
        - np.outer(shifted_center, shifted_center)
    )
    shape = np.linalg.inv(covariance) / float(dimension)
    eigenvalues, rotation = np.linalg.eigh(shape)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    rotation = rotation[:, order]
    if float(np.linalg.det(rotation)) < 0.0:
        rotation[:, -1] *= -1.0
    if float(eigenvalues[0]) <= 0.0 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("MVEE shape matrix is not positive definite")
    base_semiaxes = 1.0 / np.sqrt(eigenvalues)
    local = (points - center) @ rotation
    raw_quadratic = np.sum((local / base_semiaxes) ** 2, axis=1)
    raw_maximum = float(np.max(raw_quadratic))
    scale = math.sqrt(raw_maximum) * (1.0 + padding)
    semiaxes = base_semiaxes * scale
    normalized = np.sum((local / semiaxes) ** 2, axis=1)
    normalized_maximum = float(np.max(normalized))
    if normalized_maximum > 1.0 + 1.0e-12:
        raise ValueError("fitted ellipsoid does not enclose every source vertex")
    little_endian = np.ascontiguousarray(points, dtype="<f8")
    certificate: dict[str, Any] = {
        "verified": True,
        "proof": "all_compiled_mesh_vertices_inside_convex_ellipsoid",
        "fit_method": "khachiyan_mvee_exact_vertex_inflation",
        "khachiyan_tolerance": fit_tolerance,
        "khachiyan_max_iterations": int(max_iterations),
        "khachiyan_iterations": iterations,
        "khachiyan_final_update_l2": update_error,
        "khachiyan_converged": converged,
        "source_vertex_count": int(points.shape[0]),
        "source_vertices_float64_sha256": hashlib.sha256(
            little_endian.tobytes(order="C")
        ).hexdigest(),
        "raw_maximum_mahalanobis_quadratic": raw_maximum,
        "maximum_normalized_quadratic": normalized_maximum,
        "maximum_containment_violation": max(0.0, normalized_maximum - 1.0),
        "relative_numerical_padding": padding,
        "near_surface_vertex_count": int(np.count_nonzero(normalized >= 0.95)),
        "convex_hull_contained": True,
    }
    if certificate_metadata is not None:
        certificate["source_meshes"] = [
            dict(value) for value in certificate_metadata.get("source_meshes", [])
        ]
    if include_source_points:
        certificate["source_vertices_world_m"] = points.tolist()
    return Ellipsoid(
        center=center,
        rotation=rotation,
        semiaxes_m=semiaxes,
        body_id=int(body_id),
        body_name=str(body_name),
        geom_id=int(geom_id),
        geom_name=str(geom_name),
        bound_source="compiled_mesh_vertex_mvee_enclosure",
        source_body_names=tuple(source_body_names),
        source_geom_names=tuple(source_geom_names),
        enclosure_certificate=certificate,
    )


def partitioned_convex_hull_enclosing_ellipsoids(
    points_world: Any,
    *,
    part_count: int,
    body_name: str,
    geom_name: str,
    body_id: int = -1,
    geom_id: int = -1,
    source_body_names: Sequence[str],
    source_geom_names: Sequence[str],
    relative_padding: float = 1.0e-9,
    tolerance: float = 1.0e-4,
    max_iterations: int = 20000,
    certificate_metadata: Mapping[str, Any] | None = None,
    include_source_points: bool = False,
) -> list[Ellipsoid]:
    """Cover one compiled convex collision hull with tighter ellipsoids.

    The convex hull is triangulated into boundary facets.  Every facet and a
    common certified interior point define a tetrahedral cell; those cells
    exactly fill the hull.  Facets are partitioned along the dominant PCA
    axis, and one MVEE encloses the common point plus every vertex of the
    facets in that partition.  Convexity therefore proves that the union of
    the returned ellipsoids encloses the complete collision hull, rather than
    merely a sampled point cloud.
    """

    np = _numpy()
    try:
        from scipy.spatial import ConvexHull
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("partitioned ellipsoids require SciPy") from error

    points = np.asarray(points_world, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < 4
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("partition source points must be a finite Nx3 array")
    if isinstance(part_count, bool) or int(part_count) < 1:
        raise ValueError("ellipsoid part_count must be a positive integer")
    count = int(part_count)
    hull = ConvexHull(points)
    facets = np.asarray(hull.simplices, dtype=np.int64)
    hull_vertices = np.asarray(hull.vertices, dtype=np.int64)
    if facets.ndim != 2 or facets.shape[1] != 3 or len(facets) < count:
        raise ValueError("convex hull has insufficient triangular facets")

    interior = np.mean(points[hull_vertices], axis=0)
    halfspaces = np.asarray(hull.equations, dtype=np.float64)
    maximum_halfspace = float(
        np.max(halfspaces[:, :3] @ interior + halfspaces[:, 3])
    )
    if maximum_halfspace > 1.0e-10:
        raise ValueError("partition common point is outside the convex hull")

    centered = points[hull_vertices] - interior
    covariance = centered.T @ centered / float(len(hull_vertices))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=np.float64)
    sign_index = int(np.argmax(np.abs(axis)))
    if axis[sign_index] < 0.0:
        axis = -axis
    centroids = np.mean(points[facets], axis=1)
    projections = (centroids - interior) @ axis
    ordered = np.lexsort((np.arange(len(facets), dtype=np.int64), projections))
    partitions = [np.asarray(value, dtype=np.int64) for value in np.array_split(ordered, count)]
    if any(value.size == 0 for value in partitions):
        raise ValueError("convex-hull facet partition is empty")
    assigned = np.concatenate(partitions)
    if not np.array_equal(np.sort(assigned), np.arange(len(facets), dtype=np.int64)):
        raise ValueError("convex-hull facets are not partitioned exactly once")

    source_hash = hashlib.sha256(
        np.ascontiguousarray(points, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    facet_hash = hashlib.sha256(
        np.ascontiguousarray(facets, dtype="<i8").tobytes(order="C")
    ).hexdigest()
    output: list[Ellipsoid] = []
    for partition_index, face_indices in enumerate(partitions):
        vertex_indices = np.unique(facets[face_indices].reshape(-1))
        support = np.concatenate((interior[None, :], points[vertex_indices]), axis=0)
        metadata = dict(certificate_metadata or {})
        metadata.update(
            {
                "partition_index": int(partition_index),
                "partition_count": count,
                "partition_face_count": int(len(face_indices)),
                "partition_vertex_count": int(len(vertex_indices)),
                "partition_face_indices": face_indices.tolist(),
                "partition_vertex_indices": vertex_indices.tolist(),
                "convex_hull_face_count": int(len(facets)),
                "convex_hull_vertex_count": int(len(hull_vertices)),
                "convex_hull_volume_m3": float(hull.volume),
                "convex_hull_source_vertices_float64_sha256": source_hash,
                "convex_hull_facets_int64_sha256": facet_hash,
                "common_interior_point_m": interior.tolist(),
                "common_interior_maximum_halfspace_value": maximum_halfspace,
                "partition_axis_world": axis.tolist(),
            }
        )
        fitted = minimum_volume_enclosing_ellipsoid(
            support,
            body_name=body_name,
            geom_name="%s#part%d" % (geom_name, partition_index + 1),
            body_id=body_id,
            geom_id=geom_id,
            source_body_names=source_body_names,
            source_geom_names=source_geom_names,
            relative_padding=relative_padding,
            tolerance=tolerance,
            max_iterations=max_iterations,
            include_source_points=include_source_points,
        )
        certificate = dict(fitted.enclosure_certificate or {})
        certificate.update(metadata)
        certificate.update(
            {
                "proof": (
                    "complete_convex_hull_covered_by_union_of_common_apex_"
                    "facet_partition_ellipsoids"
                ),
                "fit_method": (
                    "convex_hull_facet_partition_khachiyan_mvee_exact_inflation"
                ),
                "partition_cell_contained": True,
                "all_convex_hull_facets_assigned_exactly_once": True,
                "convex_hull_contained_by_partition_union": True,
            }
        )
        output.append(
            Ellipsoid(
                center=fitted.center,
                rotation=fitted.rotation,
                semiaxes_m=fitted.semiaxes_m,
                body_id=fitted.body_id,
                body_name=fitted.body_name,
                geom_id=fitted.geom_id,
                geom_name=fitted.geom_name,
                bound_source="certified_partitioned_convex_hull_mvee_union",
                source_body_names=fitted.source_body_names,
                source_geom_names=fitted.source_geom_names,
                enclosure_certificate=certificate,
            )
        )
    return output


def primitive_bounding_radii(
    geom_kind: str,
    geom_size: Sequence[float],
    geom_rbound_m: float,
) -> Tuple[Any, str]:
    """Return a certified ellipsoid enclosing one MuJoCo primitive.

    Meshes and unknown types use MuJoCo's broad-phase bounding sphere.  Tight
    closed-form Loewner bounds are used for standard primitives.  A sphere is
    itself an ellipsoid, so the fallback retains the artifact contract.
    """

    np = _numpy()
    kind = str(geom_kind).lower()
    size = _finite_vector(geom_size, 3, "geom_size")
    rbound = float(geom_rbound_m)
    if not math.isfinite(rbound) or rbound <= 0.0:
        raise ValueError("geom_rbound_m must be finite and positive")
    if kind == "sphere" and size[0] > 0.0:
        radii = np.repeat(size[0], 3)
        source = "exact_mujoco_sphere"
    elif kind == "ellipsoid" and np.all(size > 0.0):
        radii = size.copy()
        source = "exact_mujoco_ellipsoid"
    elif kind == "capsule" and size[0] > 0.0 and size[1] >= 0.0:
        radius = float(size[0])
        half_length = float(size[1])
        radial = math.sqrt(radius * (radius + half_length))
        radii = np.asarray([radial, radial, radius + half_length])
        source = "closed_form_capsule_enclosing_ellipsoid"
    elif kind == "cylinder" and size[0] > 0.0 and size[1] > 0.0:
        radii = np.asarray(
            [
                math.sqrt(1.5) * float(size[0]),
                math.sqrt(1.5) * float(size[0]),
                math.sqrt(3.0) * float(size[1]),
            ]
        )
        source = "loewner_cylinder_enclosing_ellipsoid"
    elif kind == "box" and np.all(size > 0.0):
        radii = math.sqrt(3.0) * size
        source = "loewner_box_enclosing_ellipsoid"
    else:
        radii = np.repeat(rbound, 3)
        source = "mujoco_geom_rbound_sphere_fallback"

    # MuJoCo's rbound supplies the conservative fallback.  Use it whenever a
    # malformed or version-dependent primitive size prevents construction of
    # the corresponding closed-form bound.
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        radii = np.repeat(rbound, 3)
        source = "mujoco_geom_rbound_sphere_invalid_primitive_fallback"
    return np.asarray(radii, dtype=np.float64), source


def primitive_enclosure_certificate(
    geom_kind: str,
    geom_size: Sequence[float],
    geom_rbound_m: float,
    semiaxes_m: Sequence[float],
    bound_source: str,
) -> dict[str, Any]:
    """Certify the closed-form enclosure used for one compiled geom."""

    np = _numpy()
    size = _finite_vector(geom_size, 3, "geom_size")
    radii = _finite_vector(semiaxes_m, 3, "semiaxes_m")
    expected, expected_source = primitive_bounding_radii(
        geom_kind, size, geom_rbound_m
    )
    if bound_source != expected_source or not np.allclose(
        radii, expected, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("ellipsoid semiaxes differ from the certified primitive bound")
    proof_by_source = {
        "exact_mujoco_sphere": "sphere_support_equals_recorded_semiaxis",
        "exact_mujoco_ellipsoid": "principal_semiaxes_equal_compiled_geom_size",
        "closed_form_capsule_enclosing_ellipsoid": (
            "support_squared_slack_is_radius_times_half_length_times_"
            "one_minus_abs_direction_z_squared"
        ),
        "loewner_cylinder_enclosing_ellipsoid": (
            "boundary_max_is_two_thirds_radial_plus_one_third_axial"
        ),
        "loewner_box_enclosing_ellipsoid": (
            "corner_max_is_one_third_per_principal_coordinate"
        ),
        "mujoco_geom_rbound_sphere_fallback": (
            "compiled_mujoco_broadphase_rbound_sphere"
        ),
        "mujoco_geom_rbound_sphere_invalid_primitive_fallback": (
            "compiled_mujoco_broadphase_rbound_sphere"
        ),
    }
    return {
        "verified": True,
        "proof": proof_by_source[bound_source],
        "maximum_normalized_quadratic": 1.0,
        "semiaxes_formula_match_tolerance_m": 1.0e-12,
    }


def ellipsoid_record_hash_payload(ellipsoids: Sequence[Ellipsoid]) -> list[Mapping[str, Any]]:
    """Stable JSON-native geometry payload used by the shadow receipt."""

    return [item.to_record() for item in ellipsoids]
