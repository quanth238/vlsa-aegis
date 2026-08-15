"""Certified compiled-palm primitive and immutable replay audit contracts.

The released AEGIS end-effector ellipsoid is intentionally left unchanged.
This opt-in diagnostic fits one minimum-volume enclosing ellipsoid to the
compiled ``gripper0_hand_collision`` mesh and evaluates it against raw MuJoCo
contacts on a frozen cohort.  Contact outcomes never enter the fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .geometry import Ellipsoid, minimum_volume_enclosing_ellipsoid
from .shadow import _geom_kind, _name, _numpy, _raw_model_data


CONFIG_SCHEMA = "vlsa_distal_palm_primitive_audit.v1"
CONFIG_SCHEMA_V2 = "vlsa_distal_palm_primitive_compiled_obstacle_audit.v1"
RESULT_SCHEMA = "vlsa_distal_palm_primitive_case_result.v1"
RESULT_SCHEMA_V2 = "vlsa_distal_palm_primitive_case_result.v2"
VALIDATION_SCHEMA = "vlsa_distal_palm_primitive_audit_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    payload.pop("validation_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    compact_v2_keys = {
        "schema_version", "protocol_id", "base_config",
        "base_config_file_sha256", "claim_scope", "compiled_obstacle",
        "comparators",
    }
    if (
        isinstance(value, dict)
        and value.get("schema_version") == CONFIG_SCHEMA_V2
        and set(value) == compact_v2_keys
    ):
        if repo_root is None:
            raise ValueError("compiled-obstacle palm config requires repo_root")
        base_path = Path(repo_root) / str(value["base_config"])
        if file_sha256(base_path) != value["base_config_file_sha256"]:
            raise ValueError("compiled-obstacle palm base config differs")
        base = load_config(base_path, repo_root=repo_root)
        base.pop("config_file_sha256", None)
        base.pop("config_payload_sha256", None)
        base.update({
            "schema_version": value["schema_version"],
            "protocol_id": value["protocol_id"],
            "claim_scope": value["claim_scope"],
            "compiled_obstacle": value["compiled_obstacle"],
            "comparators": value["comparators"],
        })
        value = base
    expected = {
        "schema_version", "protocol_id", "claim_scope", "source", "cohort",
        "primitive", "replay", "gate", "comparators", "next_if_pass",
        "forbidden",
    }
    if value.get("schema_version") == CONFIG_SCHEMA_V2:
        expected.add("compiled_obstacle")
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("palm primitive audit config keys differ")
    variants = {
        CONFIG_SCHEMA: "vlsa-distal-palm-primitive-audit-v1",
        CONFIG_SCHEMA_V2: "vlsa-distal-palm-primitive-compiled-obstacle-audit-v1",
    }
    if (
        variants.get(value["schema_version"]) != value["protocol_id"]
        or value["primitive"]["geom_name"] != "gripper0_hand_collision"
        or value["primitive"]["fit_source"]
        != "compiled_collision_mesh_vertices_only"
    ):
        raise ValueError("palm primitive audit protocol differs")
    contacts = [str(item) for item in value["cohort"]["contact_case_ids"]]
    controls = [str(item) for item in value["cohort"]["control_case_ids"]]
    if (
        len(contacts) != int(value["gate"]["required_contact_episodes"])
        or len(controls) != int(value["gate"]["required_control_episodes"])
        or len(set(contacts + controls)) != len(contacts) + len(controls)
    ):
        raise ValueError("palm primitive frozen cohort differs")
    if repo_root is not None:
        manifest = Path(repo_root) / value["source"]["population_manifest"]
        if file_sha256(manifest) != value["source"][
            "population_manifest_file_sha256"
        ]:
            raise ValueError("palm primitive population manifest differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


@dataclass(frozen=True)
class CompiledGeomEllipsoidTemplate:
    """One geom-local certified ellipsoid that follows a rigid MuJoCo geom."""

    geom_id: int
    geom_name: str
    body_id: int
    body_name: str
    center_local_m: Any
    rotation_local: Any
    semiaxes_m: Any
    enclosure_certificate: Mapping[str, Any]
    geom_rbound_m: float
    bound_source: str = "certified_compiled_mesh_mvee"

    def to_record(self) -> dict[str, Any]:
        return {
            "geom_id": int(self.geom_id),
            "geom_name": self.geom_name,
            "body_id": int(self.body_id),
            "body_name": self.body_name,
            "center_local_m": self.center_local_m.tolist(),
            "rotation_local": self.rotation_local.tolist(),
            "semiaxes_m": self.semiaxes_m.tolist(),
            "geom_rbound_m": float(self.geom_rbound_m),
            "bound_source": self.bound_source,
            "volume_m3": float(
                4.0 * math.pi * float(_numpy().prod(self.semiaxes_m)) / 3.0
            ),
            "enclosure_certificate": dict(self.enclosure_certificate),
        }


def fit_compiled_mesh_geom(
    env: Any,
    geom_name: str,
    *,
    relative_padding: float,
    tolerance: float,
    max_iterations: int,
) -> CompiledGeomEllipsoidTemplate:
    """Fit a certified geom-local MVEE without using rollout outcomes."""

    np = _numpy()
    model, _ = _raw_model_data(env.sim)
    matches = [
        geom_id for geom_id in range(int(model.ngeom))
        if _name(env.sim.model, "geom", geom_id) == geom_name
    ]
    if len(matches) != 1:
        raise ValueError("compiled palm geom identity is not unique")
    geom_id = int(matches[0])
    if (
        int(model.geom_contype[geom_id]) == 0
        and int(model.geom_conaffinity[geom_id]) == 0
    ):
        raise ValueError("compiled palm geom is not contact capable")
    kind = _geom_kind(int(model.geom_type[geom_id]))
    if kind != "mesh":
        raise ValueError("compiled palm primitive requires a mesh; got %s" % kind)
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise ValueError("compiled palm mesh lacks mesh data")
    address = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    if count < 4:
        raise ValueError("compiled palm mesh has fewer than four vertices")
    vertices = np.asarray(
        model.mesh_vert[address : address + count], dtype=np.float64,
    ).copy()
    rbound = float(model.geom_rbound[geom_id])
    maximum_radius = float(np.max(np.linalg.norm(vertices, axis=1)))
    if maximum_radius > rbound + 1.0e-8:
        raise ValueError("compiled palm vertices exceed geom_rbound")
    body_id = int(model.geom_bodyid[geom_id])
    body_name = _name(env.sim.model, "body", body_id) or "unnamed_body_%d" % body_id
    fitted = minimum_volume_enclosing_ellipsoid(
        vertices,
        body_id=body_id,
        body_name=body_name,
        geom_id=geom_id,
        geom_name=geom_name,
        source_body_names=(body_name,),
        source_geom_names=(geom_name,),
        relative_padding=float(relative_padding),
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
        certificate_metadata={
            "source_meshes": [{
                "geom_id": geom_id,
                "geom_name": geom_name,
                "mesh_id": mesh_id,
                "compiled_vertex_count": count,
                "maximum_vertex_radius_m": maximum_radius,
                "geom_rbound_m": rbound,
                "vertices_within_geom_rbound": True,
            }]
        },
    )
    return CompiledGeomEllipsoidTemplate(
        geom_id=geom_id,
        geom_name=geom_name,
        body_id=body_id,
        body_name=body_name,
        center_local_m=fitted.center,
        rotation_local=fitted.rotation,
        semiaxes_m=fitted.semiaxes_m,
        enclosure_certificate=dict(fitted.enclosure_certificate or {}),
        geom_rbound_m=rbound,
        bound_source="certified_compiled_mesh_mvee",
    )


def fit_compiled_primitive_geom(
    env: Any, geom_name: str,
) -> CompiledGeomEllipsoidTemplate:
    """Construct the registered closed-form enclosure of one primitive geom."""

    np = _numpy()
    from .geometry import primitive_bounding_radii, primitive_enclosure_certificate

    model, _ = _raw_model_data(env.sim)
    matches = [
        geom_id for geom_id in range(int(model.ngeom))
        if _name(env.sim.model, "geom", geom_id) == geom_name
    ]
    if len(matches) != 1:
        raise ValueError("compiled obstacle geom identity is not unique")
    geom_id = int(matches[0])
    kind = _geom_kind(int(model.geom_type[geom_id]))
    if kind == "mesh":
        raise ValueError("mesh geoms require compiled-vertex fitting")
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    rbound = float(model.geom_rbound[geom_id])
    semiaxes, source = primitive_bounding_radii(kind, size, rbound)
    certificate = primitive_enclosure_certificate(
        kind, size, rbound, semiaxes, source,
    )
    body_id = int(model.geom_bodyid[geom_id])
    body_name = _name(env.sim.model, "body", body_id) or "unnamed_body_%d" % body_id
    return CompiledGeomEllipsoidTemplate(
        geom_id=geom_id,
        geom_name=geom_name,
        body_id=body_id,
        body_name=body_name,
        center_local_m=np.zeros(3, dtype=np.float64),
        rotation_local=np.eye(3, dtype=np.float64),
        semiaxes_m=semiaxes,
        enclosure_certificate=certificate,
        geom_rbound_m=rbound,
        bound_source=source,
    )


def compiled_obstacle_templates(
    env: Any,
    active_obstacle_name: str,
    *,
    relative_padding: float,
    tolerance: float,
    max_iterations: int,
) -> list[CompiledGeomEllipsoidTemplate]:
    """Fit one certified live bound per contact-capable obstacle geom."""

    from .sitl_candidate import _body_lineage, _obstacle_root_body_id

    model, _ = _raw_model_data(env.sim)
    root_id = _obstacle_root_body_id(env.sim.model, active_obstacle_name)
    output = []
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        if root_id not in _body_lineage(model, body_id):
            continue
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        geom_name = _name(env.sim.model, "geom", geom_id)
        if not geom_name:
            raise ValueError("compiled obstacle geom has no name")
        kind = _geom_kind(int(model.geom_type[geom_id]))
        if kind == "mesh":
            template = fit_compiled_mesh_geom(
                env, geom_name,
                relative_padding=relative_padding,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        else:
            template = fit_compiled_primitive_geom(env, geom_name)
        output.append(template)
    if not output:
        raise ValueError("active obstacle has no contact-capable compiled geoms")
    return output


def world_ellipsoid(
    env: Any, template: CompiledGeomEllipsoidTemplate,
) -> Ellipsoid:
    """Transform a frozen geom-local primitive into the current world pose."""

    np = _numpy()
    model, data = _raw_model_data(env.sim)
    if (
        template.geom_id >= int(model.ngeom)
        or _name(env.sim.model, "geom", template.geom_id) != template.geom_name
    ):
        raise ValueError("compiled palm geom identity changed")
    rotation = np.asarray(
        data.geom_xmat[template.geom_id], dtype=np.float64,
    ).reshape(3, 3)
    position = np.asarray(data.geom_xpos[template.geom_id], dtype=np.float64)
    return Ellipsoid(
        center=position + rotation @ template.center_local_m,
        rotation=rotation @ template.rotation_local,
        semiaxes_m=template.semiaxes_m,
        body_id=template.body_id,
        body_name=template.body_name,
        geom_id=template.geom_id,
        geom_name=template.geom_name,
        bound_source=template.bound_source,
        source_rbound_m=template.geom_rbound_m,
        source_geom_kind="mesh",
        source_body_names=(template.body_name,),
        source_geom_names=(template.geom_name,),
        enclosure_certificate=template.enclosure_certificate,
    )


def point_quadratic(ellipsoid: Ellipsoid, point_world_m: Sequence[float]) -> float:
    """Return the ellipsoid-frame Mahalanobis quadratic of a world point."""

    np = _numpy()
    point = np.asarray(point_world_m, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("contact point must be a finite length-three vector")
    local = ellipsoid.rotation.T @ (point - ellipsoid.center)
    return float(np.sum((local / ellipsoid.semiaxes_m) ** 2))


def summarize_case_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate the frozen cohort without dropping failures or controls."""

    expected_contacts = set(config["cohort"]["contact_case_ids"])
    expected_controls = set(config["cohort"]["control_case_ids"])
    by_id = {str(item["case_id"]): item for item in records}
    expected = expected_contacts | expected_controls
    if set(by_id) != expected or len(by_id) != len(records):
        raise ValueError("palm primitive case population differs")
    contact_records = [by_id[item] for item in sorted(expected_contacts)]
    control_records = [by_id[item] for item in sorted(expected_controls)]
    palm_contact_samples = sum(
        int(item["physical_contact"]["internal_palm_contact_sample_count"])
        for item in records
    )
    tight_false_safe = sum(
        int(item["tight_primitive"]["physical_false_safe_sample_count"])
        for item in records
    )
    released_false_safe = sum(
        int(item["released_proxy"]["physical_false_safe_sample_count"])
        for item in records
    )
    internal_contact_free_controls = sum(
        int(item["physical_contact"]["internal_palm_contact_sample_count"] == 0)
        for item in control_records
    )
    tight_safe_controls = sum(
        int(item["tight_primitive"]["episode_minimum_support_gap_m"] > 0.0)
        for item in control_records
    )
    released_safe_controls = sum(
        int(item["released_proxy"]["episode_minimum_support_gap_m"] > 0.0)
        for item in control_records
    )
    contact_episodes_reproduced = sum(
        int(item["physical_contact"]["internal_palm_contact_sample_count"] > 0)
        for item in contact_records
    )
    gate = config["gate"]
    pass_gate = bool(
        len(records) == len(expected)
        and all(bool(item["replay"]["fidelity_pass"]) for item in records)
        and contact_episodes_reproduced
        == int(gate["required_contact_episodes"])
        and internal_contact_free_controls
        == int(gate["required_control_episodes"])
        and palm_contact_samples > 0
        and tight_false_safe == int(gate["maximum_physical_false_safe_samples"])
        and all(bool(item["primitive_fit"]["certificate_pass"]) for item in records)
    )
    return {
        "case_count": len(records),
        "contact_episode_count": len(contact_records),
        "control_episode_count": len(control_records),
        "contact_episodes_reproduced": contact_episodes_reproduced,
        "internal_contact_free_control_count": internal_contact_free_controls,
        "palm_contact_sample_count": palm_contact_samples,
        "tight_physical_false_safe_sample_count": tight_false_safe,
        "released_physical_false_safe_sample_count": released_false_safe,
        "tight_contact_free_episode_accept_count": tight_safe_controls,
        "released_contact_free_episode_accept_count": released_safe_controls,
        "tight_contact_free_episode_accept_rate": tight_safe_controls / len(control_records),
        "released_contact_free_episode_accept_rate": (
            released_safe_controls / len(control_records)
        ),
        "geometry_gate_pass": pass_gate,
        "interpretation": (
            "tight_palm_geometry_validated_boundary_collection_may_begin"
            if pass_gate else
            "tight_palm_geometry_no_go_boundary_collection_blocked"
        ),
    }
