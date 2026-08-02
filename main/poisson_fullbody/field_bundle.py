"""Fail-closed construction of the registered static Poisson field bundle.

This module constructs numerical state only.  It does not install a controller,
mutate an action, or step physics.  The constructor deliberately revalidates
all provenance at the point of use: the runtime protocol hash, MuJoCo geometry
resolution, exact selected-obstacle box poses, protected surface coverage, and
the independently recomputed Poisson diagnostics must all agree before a
bundle is returned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from main.poisson_fullbody.feasibility_protocol import (
    ProtocolHashes,
    canonical_json_bytes,
    validate_feasibility_protocol,
)
from main.poisson_fullbody.geometry import OrientedBox
from main.poisson_fullbody.measurement import (
    ResolvedGeomSets,
    clone_forwarded_state,
    resolve_collision_geom_sets,
)
from main.poisson_fullbody.poisson_field import (
    PoissonDiagnostics,
    PoissonSystem,
    TrilinearPoissonField,
    build_poisson_system,
    poisson_diagnostics,
    solve_poisson_sor,
)
from main.poisson_fullbody.robot_samples import BodySample, evaluate_world_points
from main.poisson_fullbody.surface_sampling import build_robot_collision_samples
from main.poisson_fullbody.voxel_grid import (
    GridSpec,
    OccupancyResult,
    PoissonDomain,
    build_connected_domain,
    build_occupancy,
)


REGISTERED_GRID_SHAPE_VERTICES = (101, 101, 101)
REGISTERED_FULL_ROBOT_GRID_SHAPE_VERTICES = (116, 101, 111)
REGISTERED_PROTECTED_BODY_NAMES = ("robot0_link5", "robot0_link6")

_REGISTERED_WORKSPACES = {
    "vlsa_poisson_runtime_protocol.v3": {
        "protocol_id": "vlsa-poisson-link56-canary-parameters-v3",
        "minimum_m": (-1.0, -1.0, 0.0),
        "maximum_m": (1.0, 1.0, 2.0),
        "grid_shape_vertices": REGISTERED_GRID_SHAPE_VERTICES,
        "grid_spacing_m": (0.02, 0.02, 0.02),
    },
    "vlsa_poisson_runtime_protocol.v4": {
        "protocol_id": "vlsa-poisson-full-robot-canary-parameters-v4",
        "minimum_m": (-1.3, -1.0, -0.2),
        "maximum_m": (1.0, 1.0, 2.0),
        "grid_shape_vertices": REGISTERED_FULL_ROBOT_GRID_SHAPE_VERTICES,
        "grid_spacing_m": (0.02, 0.02, 0.02),
    },
    "vlsa_poisson_runtime_protocol.v5": {
        "protocol_id": "vlsa-poisson-movable-manipulator-canary-parameters-v5",
        "minimum_m": (-1.3, -1.0, -0.2),
        "maximum_m": (1.0, 1.0, 2.0),
        "grid_shape_vertices": REGISTERED_FULL_ROBOT_GRID_SHAPE_VERTICES,
        "grid_spacing_m": (0.02, 0.02, 0.02),
    },
}


class StaticFieldBundleError(ValueError):
    """Raised when no certified static field bundle can be returned."""


def _modules() -> Tuple[Any, Any]:
    try:
        import mujoco
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError(
            "MuJoCo and NumPy are required to construct a static field bundle"
        ) from error
    return mujoco, np


@dataclass(frozen=True)
class SurfaceComponentCertificate:
    """Immutable coverage certificate for one protected collision geom."""

    geom_id: int
    geom_name: str
    body_id: int
    body_name: str
    geometry_kind: str
    certificate_kind: str
    surface_element_count: int
    sample_count: int
    certified_surface_cover_radius_m: float


@dataclass(frozen=True)
class ProtectedSurfaceSamples:
    """Immutable link-5/6 samples and their strict-open coverage metadata."""

    samples: Tuple[BodySample, ...]
    components: Tuple[SurfaceComponentCertificate, ...]
    epsilon_m: float
    maximum_surface_cover_radius_m: float
    coverage_semantics: str

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        components = tuple(self.components)
        if not samples:
            raise StaticFieldBundleError("protected surface samples are empty")
        if not components:
            raise StaticFieldBundleError("protected surface components are empty")
        if any(not isinstance(sample, BodySample) for sample in samples):
            raise TypeError("samples must contain only BodySample records")
        if any(
            not isinstance(component, SurfaceComponentCertificate)
            for component in components
        ):
            raise TypeError(
                "components must contain SurfaceComponentCertificate records"
            )
        epsilon = float(self.epsilon_m)
        maximum = float(self.maximum_surface_cover_radius_m)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise StaticFieldBundleError("coverage epsilon must be finite and positive")
        if not math.isfinite(maximum) or maximum < 0.0 or not maximum < epsilon:
            raise StaticFieldBundleError(
                "protected surface covering radius must be strictly less than epsilon"
            )
        sample_ids = tuple(sample.sample_id for sample in samples)
        if sample_ids != tuple(range(len(samples))):
            raise StaticFieldBundleError(
                "protected surface sample IDs must be contiguous and deterministic"
            )
        component_geom_ids = tuple(component.geom_id for component in components)
        if len(component_geom_ids) != len(set(component_geom_ids)):
            raise StaticFieldBundleError(
                "each protected collision geom must have one component certificate"
            )
        if set(sample.geom_id for sample in samples) != set(component_geom_ids):
            raise StaticFieldBundleError(
                "sample and component collision-geom identities disagree"
            )
        for component in components:
            if component.surface_element_count <= 0 or component.sample_count <= 0:
                raise StaticFieldBundleError(
                    "every protected surface component must be nonempty"
                )
            if not (
                math.isfinite(component.certified_surface_cover_radius_m)
                and component.certified_surface_cover_radius_m < epsilon
            ):
                raise StaticFieldBundleError(
                    "every component covering radius must be strictly less than epsilon"
                )
            observed = sum(
                sample.geom_id == component.geom_id for sample in samples
            )
            if observed != component.sample_count:
                raise StaticFieldBundleError(
                    "component sample count disagrees with immutable samples"
                )
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "epsilon_m", epsilon)
        object.__setattr__(self, "maximum_surface_cover_radius_m", maximum)


class ImmutableTrilinearPoissonField(TrilinearPoissonField):
    """A sealed trilinear field whose numerical arrays have immutable storage."""

    __slots__ = ("_field_bundle_sealed",)

    def __init__(self, grid: Any, values: Any, domain: Any) -> None:
        object.__setattr__(self, "_field_bundle_sealed", False)
        super().__init__(grid=grid, values=values, domain=domain)
        object.__setattr__(self, "_field_bundle_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_field_bundle_sealed", False):
            raise AttributeError("the certified Poisson field is immutable")
        object.__setattr__(self, name, value)


@dataclass(frozen=True)
class StaticFieldBundleDiagnostics:
    obstacle_geom_count: int
    protected_surface_component_count: int
    protected_sample_count: int
    raw_occupied_cell_count: int
    buffered_occupied_cell_count: int
    connected_free_cell_count: int
    active_vertex_count: int
    boundary_vertex_count: int
    interior_vertex_count: int
    poisson_method: str
    poisson_iterations: int
    poisson: PoissonDiagnostics
    minimum_initial_h_m2: float
    minimum_outer_boundary_clearance_m: float
    required_outer_boundary_clearance_m: float


@dataclass(frozen=True)
class StaticFieldBundleHashes:
    protocol_sha256: str
    parameter_block_sha256: str
    obstacle_geometry_sha256: str
    occupancy_sha256: str
    domain_sha256: str
    system_sha256: str
    field_sha256: str
    protected_samples_sha256: str
    bundle_sha256: str


@dataclass(frozen=True)
class StaticFieldBundle:
    """Complete immutable certificate used by later shadow/active runtimes."""

    protocol_id: str
    protocol_hashes: ProtocolHashes
    protected_body_ids: Tuple[int, int]
    protected_body_names: Tuple[str, str]
    obstacle_boxes: Tuple[OrientedBox, ...]
    grid: GridSpec
    occupancy: OccupancyResult
    domain: PoissonDomain
    system: PoissonSystem
    field: ImmutableTrilinearPoissonField
    protected_samples: ProtectedSurfaceSamples
    diagnostics: StaticFieldBundleDiagnostics
    hashes: StaticFieldBundleHashes


def _validated_protocol_snapshot(
    protocol: Mapping[str, Any], protocol_hashes: ProtocolHashes
) -> Tuple[Dict[str, Any], ProtocolHashes]:
    if not isinstance(protocol_hashes, ProtocolHashes):
        raise TypeError("protocol_hashes must be a validated ProtocolHashes record")
    try:
        # A canonical round trip gives construction an isolated snapshot and
        # prevents a caller from changing a mutable mapping after validation.
        snapshot = json.loads(canonical_json_bytes(protocol).decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise StaticFieldBundleError("runtime protocol cannot be snapshotted") from error
    observed = validate_feasibility_protocol(
        snapshot,
        expected_protocol_sha256=protocol_hashes.protocol_sha256,
    )
    if observed != protocol_hashes:
        raise StaticFieldBundleError(
            "validated runtime protocol hashes do not match the supplied identity"
        )
    schema_version = snapshot.get("schema_version")
    registered_workspace = _REGISTERED_WORKSPACES.get(schema_version)
    if registered_workspace is None:
        raise StaticFieldBundleError(
            "the static field constructor does not support this runtime schema"
        )
    if snapshot.get("protocol_id") != registered_workspace["protocol_id"]:
        raise StaticFieldBundleError(
            "runtime protocol ID does not match its registered workspace"
        )
    workspace = snapshot["workspace"]
    observed_workspace = {
        "minimum_m": tuple(float(value) for value in workspace["minimum_m"]),
        "maximum_m": tuple(float(value) for value in workspace["maximum_m"]),
        "grid_shape_vertices": tuple(
            int(value) for value in workspace["grid_shape_vertices"]
        ),
        "grid_spacing_m": tuple(
            float(value) for value in workspace["grid_spacing_m"]
        ),
    }
    expected_workspace = {
        key: value
        for key, value in registered_workspace.items()
        if key != "protocol_id"
    }
    if observed_workspace != expected_workspace:
        raise StaticFieldBundleError(
            "runtime workspace does not match its registered schema"
        )
    if float(snapshot["poisson"]["forcing_value"]) != 1.0:
        raise StaticFieldBundleError("the registered static field requires forcing +1")
    if float(snapshot["poisson"]["boundary_value"]) != 0.0:
        raise StaticFieldBundleError(
            "the registered static field requires zero boundary data"
        )
    if tuple(snapshot["claim_scope"]["protected_robot_bodies"]) != (
        REGISTERED_PROTECTED_BODY_NAMES
    ):
        raise StaticFieldBundleError("protected-body protocol scope mismatch")
    return snapshot, observed


def _authoritative_model_and_forwarded_data(model: Any, data: Any) -> Tuple[Any, Any]:
    mujoco, _ = _modules()
    raw_model = getattr(model, "_model", model)
    raw_data = getattr(data, "_data", data)
    if not isinstance(raw_model, mujoco.MjModel):
        raise TypeError("model must resolve to an authoritative mujoco.MjModel")
    if not isinstance(raw_data, mujoco.MjData):
        raise TypeError("data must resolve to an authoritative mujoco.MjData")
    try:
        forwarded = clone_forwarded_state(raw_model, raw_data)
    except Exception as error:
        raise StaticFieldBundleError(
            "MuJoCo model/data do not form one readable integration state"
        ) from error
    return raw_model, forwarded


def _validated_resolution(
    model: Any,
    resolved: ResolvedGeomSets,
    expected_body_names: Sequence[str],
) -> Tuple[int, int]:
    mujoco, _ = _modules()
    if not isinstance(resolved, ResolvedGeomSets):
        raise TypeError("resolved must be a ResolvedGeomSets record")
    current = resolve_collision_geom_sets(
        model,
        robot_root_body_ids=resolved.robot_root_body_ids,
        obstacle_root_body_ids=resolved.obstacle_root_body_ids,
        link56_body_ids=resolved.link56_body_ids,
    )
    if current != resolved:
        raise StaticFieldBundleError(
            "resolved selected-obstacle geometry does not match this MuJoCo model"
        )
    protected_ids = tuple(int(value) for value in resolved.link56_body_ids)
    if len(protected_ids) != 2:
        raise StaticFieldBundleError(
            "the registered field-seed scope requires exactly two bodies"
        )
    observed_names = []
    for body_id in protected_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None:
            raise StaticFieldBundleError("a protected MuJoCo body is unnamed")
        observed_names.append(str(name))
    if tuple(observed_names) != tuple(expected_body_names):
        raise StaticFieldBundleError(
            "resolved protected bodies do not equal robot0_link5/robot0_link6"
        )
    return protected_ids  # type: ignore[return-value]


def _selected_obstacle_boxes(
    model: Any, data: Any, resolved: ResolvedGeomSets
) -> Tuple[OrientedBox, ...]:
    mujoco, np = _modules()
    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    boxes = []
    for geom_id in resolved.obstacle_geom_ids:
        if int(model.geom_type[geom_id]) != box_type:
            raise StaticFieldBundleError(
                "selected-obstacle geom {} is not a supported MuJoCo box".format(
                    geom_id
                )
            )
        boxes.append(
            OrientedBox(
                center=np.asarray(data.geom_xpos[geom_id], dtype=np.float64),
                R=np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3),
                half_extents=np.asarray(
                    model.geom_size[geom_id, :3], dtype=np.float64
                ),
                geom_id=int(geom_id),
            )
        )
    if not boxes:
        raise StaticFieldBundleError("selected-obstacle geometry is empty")
    return tuple(boxes)


def _protected_surface_samples(
    model: Any,
    data: Any,
    resolved: ResolvedGeomSets,
    protected_body_ids: Sequence[int],
    epsilon_m: float,
    certificate_method: str,
) -> ProtectedSurfaceSamples:
    mujoco, _ = _modules()
    try:
        extracted = build_robot_collision_samples(
            model,
            data,
            geom_ids=resolved.link56_geom_ids,
            epsilon_m=epsilon_m,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise StaticFieldBundleError(
            "protected collision-surface extraction failed: {}".format(error)
        ) from error
    if not extracted.samples:
        raise StaticFieldBundleError("protected collision-surface samples are empty")
    if not extracted.geom_records:
        raise StaticFieldBundleError("protected collision-surface components are empty")
    if not math.isclose(
        float(extracted.epsilon_m), float(epsilon_m), rel_tol=0.0, abs_tol=0.0
    ):
        raise StaticFieldBundleError(
            "surface sampler did not preserve the registered coverage epsilon"
        )
    expected_geoms = tuple(int(value) for value in resolved.link56_geom_ids)
    observed_geoms = tuple(int(record["geom_id"]) for record in extracted.geom_records)
    if observed_geoms != expected_geoms:
        missing = sorted(set(expected_geoms) - set(observed_geoms))
        unexpected = sorted(set(observed_geoms) - set(expected_geoms))
        raise StaticFieldBundleError(
            "protected collision-surface coverage mismatch; missing={} unexpected={}".format(
                missing, unexpected
            )
        )
    immutable_samples = tuple(
        BodySample(
            sample_id=int(sample.sample_id),
            body_id=int(sample.body_id),
            body_name=str(sample.body_name),
            geom_id=int(sample.geom_id),
            geom_name=str(sample.geom_name),
            point_body_local_m=tuple(
                float(value) for value in sample.point_body_local_m
            ),
            source=str(sample.source),
        )
        for sample in extracted.samples
    )
    protected_body_set = set(int(value) for value in protected_body_ids)
    expected_geom_set = set(expected_geoms)
    for sample in immutable_samples:
        authoritative_body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, sample.body_id
        )
        authoritative_geom = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, sample.geom_id
        )
        if (
            sample.body_id not in protected_body_set
            or sample.geom_id not in expected_geom_set
            or int(model.geom_bodyid[sample.geom_id]) != sample.body_id
            or authoritative_body is None
            or authoritative_geom is None
            or str(authoritative_body) != sample.body_name
            or str(authoritative_geom) != sample.geom_name
        ):
            raise StaticFieldBundleError(
                "protected sample provenance disagrees with authoritative MuJoCo"
            )
    components = []
    for record in extracted.geom_records:
        geom_id = int(record["geom_id"])
        body_id = int(record["body_id"])
        authoritative_body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        authoritative_geom = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        if authoritative_body is None or authoritative_geom is None:
            raise StaticFieldBundleError("a protected collision surface is unnamed")
        if (
            str(authoritative_body) != str(record["body_name"])
            or str(authoritative_geom) != str(record["geom_name"])
            or int(model.geom_bodyid[geom_id]) != body_id
        ):
            raise StaticFieldBundleError(
                "protected surface provenance disagrees with authoritative MuJoCo"
            )
        if str(record["certificate_kind"]) != str(
            certificate_method
        ):
            raise StaticFieldBundleError(
                "protected surface certificate method differs from protocol"
            )
        components.append(
            SurfaceComponentCertificate(
                geom_id=geom_id,
                geom_name=str(record["geom_name"]),
                body_id=body_id,
                body_name=str(record["body_name"]),
                geometry_kind=str(record["geometry_kind"]),
                certificate_kind=str(record["certificate_kind"]),
                surface_element_count=int(record["surface_element_count"]),
                sample_count=int(record["sample_count"]),
                certified_surface_cover_radius_m=float(
                    record["certified_surface_cover_radius_m"]
                ),
            )
        )
    return ProtectedSurfaceSamples(
        samples=immutable_samples,
        components=tuple(components),
        epsilon_m=float(extracted.epsilon_m),
        maximum_surface_cover_radius_m=float(
            extracted.maximum_surface_cover_radius_m
        ),
        coverage_semantics=str(extracted.coverage_semantics),
    )


def _minimum_outer_clearance(
    points: Any, lower: Any, upper: Any, required_m: float
) -> float:
    _, np = _modules()
    world = np.asarray(points, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3 or world.shape[0] == 0:
        raise StaticFieldBundleError("protected sample world points are empty")
    if not np.all(np.isfinite(world)):
        raise StaticFieldBundleError("protected sample world points are non-finite")
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    per_axis = np.minimum(world - lower[None, :], upper[None, :] - world)
    minimum = float(np.min(per_axis))
    scale = max(1.0, float(np.max(np.abs(lower))), float(np.max(np.abs(upper))))
    tolerance = 64.0 * float(np.finfo(np.float64).eps) * scale
    if minimum + tolerance < float(required_m):
        raise StaticFieldBundleError(
            "protected surface violates registered outer-boundary clearance: "
            "observed {:.9g} m, required {:.9g} m".format(minimum, required_m)
        )
    return minimum


def _array_digest(metadata: Mapping[str, Any], arrays: Sequence[Tuple[str, Any]]) -> str:
    """Hash typed C-order numerical payloads with deterministic framing."""

    _, np = _modules()
    digest = hashlib.sha256()
    metadata_bytes = canonical_json_bytes(metadata)
    digest.update(len(metadata_bytes).to_bytes(8, byteorder="big"))
    digest.update(metadata_bytes)
    for name, value in arrays:
        array = np.asarray(value)
        if array.dtype.kind == "b":
            canonical = np.asarray(array, dtype=np.uint8, order="C")
            dtype_name = "uint8_bool"
        elif array.dtype.kind in "iu":
            canonical = np.asarray(array, dtype="<i8", order="C")
            dtype_name = "int64_le"
        elif array.dtype.kind == "f":
            canonical = np.asarray(array, dtype="<f8", order="C")
            dtype_name = "float64_le"
        else:
            raise TypeError("unsupported array dtype in field-bundle hash")
        header = canonical_json_bytes(
            {
                "name": str(name),
                "shape": [int(value) for value in canonical.shape],
                "dtype": dtype_name,
            }
        )
        payload = canonical.tobytes(order="C")
        digest.update(len(header).to_bytes(8, byteorder="big"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def _bundle_hashes(
    protocol_hashes: ProtocolHashes,
    boxes: Sequence[OrientedBox],
    occupancy: OccupancyResult,
    domain: PoissonDomain,
    system: PoissonSystem,
    field: ImmutableTrilinearPoissonField,
    protected: ProtectedSurfaceSamples,
    diagnostics: StaticFieldBundleDiagnostics,
) -> StaticFieldBundleHashes:
    obstacle_hash = hashlib.sha256(
        canonical_json_bytes(
            [
                {
                    "geom_id": int(box.geom_id),
                    "center_m": [float(value) for value in box.center],
                    "rotation_world_from_geom": [
                        [float(value) for value in row] for row in box.R
                    ],
                    "half_extents_m": [float(value) for value in box.half_extents],
                }
                for box in boxes
            ]
        )
    ).hexdigest()
    occupancy_hash = _array_digest(
        {
            "epsilon_m": occupancy.epsilon,
            "numerical_tolerance_m": occupancy.numerical_tolerance,
            "cell_shape": list(occupancy.grid.cell_shape),
        },
        (
            ("raw_cells", occupancy.raw_cells),
            ("buffered_cells", occupancy.buffered_cells),
        ),
    )
    domain_hash = _array_digest(
        {"vertex_shape": list(system.shape)},
        (
            ("component_cells", domain.component_cells),
            ("active_vertices", domain.active_vertices),
            ("boundary_vertices", domain.boundary_vertices),
            ("interior_vertices", domain.interior_vertices),
        ),
    )
    matrix = system.matrix
    system_hash = _array_digest(
        {"shape": list(system.shape), "matrix_shape": list(matrix.shape)},
        (
            ("matrix_data", matrix.data),
            ("matrix_indices", matrix.indices),
            ("matrix_indptr", matrix.indptr),
            ("rhs", system.rhs),
            ("lower", system.lower),
            ("upper", system.upper),
            ("spacing", system.spacing),
            ("interior_mask", system.interior_mask),
            ("boundary_mask", system.boundary_mask),
            ("valid_vertex_mask", system.valid_vertex_mask),
            ("forcing", system.forcing),
            ("boundary_values", system.boundary_values),
            ("unknown_flat_indices", system.unknown_flat_indices),
        ),
    )
    field_hash = _array_digest(
        {"shape": list(field.shape), "interpolation": "trilinear_discrete_C0"},
        (
            ("lower", field.lower),
            ("upper", field.upper),
            ("spacing", field.spacing),
            ("values", field.values),
            ("valid_vertices", field.valid_mask),
            ("valid_cells", field.valid_cell_mask),
        ),
    )
    sample_payload = {
        "epsilon_m": protected.epsilon_m,
        "maximum_surface_cover_radius_m": (
            protected.maximum_surface_cover_radius_m
        ),
        "coverage_semantics": protected.coverage_semantics,
        "components": [asdict(component) for component in protected.components],
        "samples": [sample.to_dict() for sample in protected.samples],
    }
    samples_hash = hashlib.sha256(canonical_json_bytes(sample_payload)).hexdigest()
    identities = {
        "protocol_sha256": protocol_hashes.protocol_sha256,
        "parameter_block_sha256": protocol_hashes.parameter_block_sha256,
        "obstacle_geometry_sha256": obstacle_hash,
        "occupancy_sha256": occupancy_hash,
        "domain_sha256": domain_hash,
        "system_sha256": system_hash,
        "field_sha256": field_hash,
        "protected_samples_sha256": samples_hash,
        "diagnostics": asdict(diagnostics),
    }
    bundle_hash = hashlib.sha256(canonical_json_bytes(identities)).hexdigest()
    return StaticFieldBundleHashes(
        protocol_sha256=protocol_hashes.protocol_sha256,
        parameter_block_sha256=protocol_hashes.parameter_block_sha256,
        obstacle_geometry_sha256=obstacle_hash,
        occupancy_sha256=occupancy_hash,
        domain_sha256=domain_hash,
        system_sha256=system_hash,
        field_sha256=field_hash,
        protected_samples_sha256=samples_hash,
        bundle_sha256=bundle_hash,
    )


def build_static_field_bundle(
    model: Any,
    data: Any,
    *,
    resolved: ResolvedGeomSets,
    protocol: Mapping[str, Any],
    protocol_hashes: ProtocolHashes,
) -> StaticFieldBundle:
    """Construct a registered static link-5/6-seeded Poisson certificate.

    The live ``data`` object is never forwarded or modified.  Geometry and
    robot samples are read from an integration-state clone after ``mj_forward``.
    Any unsupported, incomplete, inconsistent, or numerically uncertified
    input raises :class:`StaticFieldBundleError`; no partial bundle is exposed.
    """

    snapshot, observed_hashes = _validated_protocol_snapshot(
        protocol, protocol_hashes
    )
    raw_model, forwarded_data = _authoritative_model_and_forwarded_data(model, data)
    protected_names = tuple(snapshot["claim_scope"]["protected_robot_bodies"])
    protected_ids = _validated_resolution(raw_model, resolved, protected_names)
    boxes = _selected_obstacle_boxes(raw_model, forwarded_data, resolved)

    coverage_epsilon = float(snapshot["coverage"]["epsilon_m"])
    protected = _protected_surface_samples(
        raw_model,
        forwarded_data,
        resolved,
        protected_ids,
        coverage_epsilon,
        str(snapshot["coverage"]["certificate_method"]),
    )
    world_points = evaluate_world_points(protected.samples, forwarded_data)

    workspace = snapshot["workspace"]
    grid = GridSpec.from_bounds(
        workspace["minimum_m"],
        workspace["maximum_m"],
        workspace["grid_shape_vertices"],
    )
    _, np = _modules()
    declared_spacing = np.asarray(workspace["grid_spacing_m"], dtype=np.float64)
    declared_shape = tuple(int(value) for value in workspace["grid_shape_vertices"])
    if grid.vertex_shape != declared_shape or not np.allclose(
        grid.spacing, declared_spacing, rtol=0.0, atol=1.0e-15
    ):
        raise StaticFieldBundleError("registered grid construction mismatch")
    required_outer_clearance = float(
        snapshot["occupancy"]["outer_boundary_clearance_m"]
    )
    minimum_outer_clearance = _minimum_outer_clearance(
        world_points, grid.lower, grid.upper, required_outer_clearance
    )

    occupancy = build_occupancy(
        grid,
        boxes,
        epsilon=float(snapshot["occupancy"]["obstacle_clearance_m"]),
    )
    domain = build_connected_domain(
        grid,
        occupancy.buffered_cells,
        seed_points=world_points,
    )
    system = build_poisson_system(
        grid,
        domain,
        forcing=1.0,
        boundary_values=0.0,
    )
    if not np.array_equal(system.interior_mask, domain.interior_vertices):
        raise StaticFieldBundleError("Poisson system/domain interior mismatch")
    if not np.array_equal(system.boundary_mask, domain.boundary_vertices):
        raise StaticFieldBundleError("Poisson system/domain boundary mismatch")
    if system.unknown_count <= 0:
        raise StaticFieldBundleError("Poisson system has no interior unknowns")

    poisson = snapshot["poisson"]
    solve = solve_poisson_sor(
        system,
        omega=float(poisson["relaxation_omega"]),
        tolerance=float(poisson["normalized_backward_error_tolerance"]),
        max_iterations=int(poisson["max_iterations"]),
        initial=0.0,
        check_every=int(poisson["residual_check_interval"]),
    )
    registered_tolerance = float(
        poisson["normalized_backward_error_tolerance"]
    )
    if solve.method != "red_black_sor":
        raise StaticFieldBundleError("Poisson result did not come from registered SOR")
    if solve.iterations < 0 or solve.iterations > int(poisson["max_iterations"]):
        raise StaticFieldBundleError("Poisson solver iteration diagnostics are invalid")
    if not math.isclose(
        float(solve.backward_error_target),
        registered_tolerance,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise StaticFieldBundleError("Poisson solver tolerance diagnostics mismatch")
    independent_diagnostics = poisson_diagnostics(
        system,
        solve.values,
        expected_sign="nonnegative",
        residual_tolerance=registered_tolerance,
    )
    if not solve.converged:
        raise StaticFieldBundleError(
            "registered SOR did not converge: {}".format(solve.message)
        )
    if not independent_diagnostics.passed:
        raise StaticFieldBundleError(
            "independent Poisson field diagnostics did not pass"
        )
    if (
        independent_diagnostics.interior_min is None
        or independent_diagnostics.interior_min <= 0.0
    ):
        raise StaticFieldBundleError(
            "positive forcing did not produce a strictly positive interior field"
        )

    field = ImmutableTrilinearPoissonField(
        grid=grid,
        values=solve.values,
        domain=domain,
    )
    minimum_h = float("inf")
    query_minimum_outer = float("inf")
    for sample, point in zip(protected.samples, world_points):
        query = field.query(point)
        if not query.valid:
            reason = query.reason.value if query.reason is not None else "unknown"
            raise StaticFieldBundleError(
                "initial field query is invalid for sample {}: {}".format(
                    sample.sample_id, reason
                )
            )
        if query.value is None or not math.isfinite(query.value) or query.value <= 0.0:
            raise StaticFieldBundleError(
                "an initial protected sample is not strictly inside h > 0"
            )
        if query.outer_boundary_clearance_m is None:
            raise StaticFieldBundleError(
                "initial field query omitted outer-boundary clearance"
            )
        minimum_h = min(minimum_h, float(query.value))
        query_minimum_outer = min(
            query_minimum_outer, float(query.outer_boundary_clearance_m)
        )
    if not math.isclose(
        minimum_outer_clearance,
        query_minimum_outer,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise StaticFieldBundleError(
            "direct and field-query outer-boundary clearances disagree"
        )

    diagnostics = StaticFieldBundleDiagnostics(
        obstacle_geom_count=len(boxes),
        protected_surface_component_count=len(protected.components),
        protected_sample_count=len(protected.samples),
        raw_occupied_cell_count=int(np.count_nonzero(occupancy.raw_cells)),
        buffered_occupied_cell_count=int(
            np.count_nonzero(occupancy.buffered_cells)
        ),
        connected_free_cell_count=int(np.count_nonzero(domain.component_cells)),
        active_vertex_count=int(np.count_nonzero(domain.active_vertices)),
        boundary_vertex_count=int(np.count_nonzero(domain.boundary_vertices)),
        interior_vertex_count=int(np.count_nonzero(domain.interior_vertices)),
        poisson_method=str(solve.method),
        poisson_iterations=int(solve.iterations),
        poisson=independent_diagnostics,
        minimum_initial_h_m2=float(minimum_h),
        minimum_outer_boundary_clearance_m=float(minimum_outer_clearance),
        required_outer_boundary_clearance_m=float(required_outer_clearance),
    )
    hashes = _bundle_hashes(
        observed_hashes,
        boxes,
        occupancy,
        domain,
        system,
        field,
        protected,
        diagnostics,
    )
    return StaticFieldBundle(
        protocol_id=str(snapshot["protocol_id"]),
        protocol_hashes=observed_hashes,
        protected_body_ids=protected_ids,
        protected_body_names=REGISTERED_PROTECTED_BODY_NAMES,
        obstacle_boxes=boxes,
        grid=grid,
        occupancy=occupancy,
        domain=domain,
        system=system,
        field=field,
        protected_samples=protected,
        diagnostics=diagnostics,
        hashes=hashes,
    )


# Descriptive alias retained for callers that prefer constructor terminology.
construct_static_field_bundle = build_static_field_bundle


__all__ = [
    "ImmutableTrilinearPoissonField",
    "ProtectedSurfaceSamples",
    "REGISTERED_FULL_ROBOT_GRID_SHAPE_VERTICES",
    "REGISTERED_GRID_SHAPE_VERTICES",
    "REGISTERED_PROTECTED_BODY_NAMES",
    "StaticFieldBundle",
    "StaticFieldBundleDiagnostics",
    "StaticFieldBundleError",
    "StaticFieldBundleHashes",
    "SurfaceComponentCertificate",
    "build_static_field_bundle",
    "construct_static_field_bundle",
]
