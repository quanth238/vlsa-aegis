"""Certified live obstacle primitives for the bounded E05 mechanism audit.

This module is opt-in.  It represents every collision-active geometry in the
selected obstacle body lineage by one enclosing ellipsoid.  Mesh geometries
use an exact-vertex-inflated MVEE; supported MuJoCo primitives use the existing
closed-form enclosure certificates.  Templates are stored in each source
geom's local frame and therefore follow the live rigid obstacle pose.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .barrier import support_gap
from .geometry import (
    Ellipsoid,
    minimum_volume_enclosing_ellipsoid,
    primitive_bounding_radii,
    primitive_enclosure_certificate,
)
from .shadow import _geom_kind, _name, _numpy, _raw_model_data


OBSTACLE_PRIMITIVE_SCHEMA = "vlsa_distal_oracle_mesh_obstacle_e05.v1"
OBSTACLE_PRIMITIVE_RESULT_SCHEMA = (
    "vlsa_distal_oracle_mesh_obstacle_e05_result.v1"
)
_CASE_ID = "vlsa-t1-goal-ii-t0-e05"
_BASE_FILE_SHA256 = (
    "c7019c176e0e8b6379cdb1b83e09d7129c1237a4f28ec2f3daba49a7b57ddc95"
)
_BASE_PAYLOAD_SHA256 = (
    "dc0a9d52fe75da297a187c36a0a6c9f069948511afa940c9268ff43ed392c98d"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_obstacle_primitive_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("obstacle primitive config is invalid JSON") from error
    required = {
        "schema_version",
        "protocol_id",
        "case_ids",
        "claim_scope",
        "base_audit_config",
        "obstacle_geometry",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("obstacle primitive config keys differ")
    if config["schema_version"] != OBSTACLE_PRIMITIVE_SCHEMA:
        raise ValueError("obstacle primitive config schema differs")
    if config["protocol_id"] != "vlsa-distal-oracle-mesh-obstacle-e05-v1":
        raise ValueError("obstacle primitive protocol differs")
    if config["case_ids"] != [_CASE_ID]:
        raise ValueError("obstacle primitive audit must select only primary E05")
    if config["base_audit_config"] != {
        "config_file_sha256": _BASE_FILE_SHA256,
        "config_payload_sha256": _BASE_PAYLOAD_SHA256,
        "schema_version": "vlsa_distal_oracle_affine_e05.v1",
    }:
        raise ValueError("obstacle primitive base audit identity differs")
    if config["obstacle_geometry"] != {
        "collision_geom_filter": (
            "active_obstacle_body_lineage_and_nonzero_contype_or_conaffinity"
        ),
        "clearance_reduction": (
            "minimum_support_gap_over_every_obstacle_primitive_for_each_of_"
            "eight_robot_proxies"
        ),
        "khachiyan_max_iterations": 20000,
        "khachiyan_tolerance": 1.0e-4,
        "maximum_primitive_count": 64,
        "mesh_representation": (
            "one_certified_mvee_per_compiled_mujoco_collision_mesh"
        ),
        "non_mesh_representation": (
            "certified_closed_form_enclosing_ellipsoid"
        ),
        "pose_update": (
            "rigidly_attached_to_live_source_geom_at_every_internal_mujoco_step"
        ),
        "relative_numerical_padding": 1.0e-9,
        "source": "live_compiled_mujoco_active_obstacle_collision_geometry",
    }:
        raise ValueError("obstacle primitive geometry contract differs")
    if config["decision_gate"] != {
        "contact_witness": (
            "every_raw_L5_L6_L7_contact_point_inside_corresponding_certified_"
            "robot_union_and_live_source_obstacle_primitive_with_nonpositive_"
            "minimum_pair_support_gap"
        ),
        "retain_base_affine_candidate_solver_and_exact_execution_gates": True,
    }:
        raise ValueError("obstacle primitive decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _body_lineage(model: Any, body_id: int) -> set[int]:
    output = set()
    current = int(body_id)
    while current >= 0 and current not in output:
        output.add(current)
        if current == 0:
            break
        current = int(model.body_parentid[current])
    return output


def _obstacle_root_body_id(model: Any, active_obstacle_name: str) -> int:
    for name in (str(active_obstacle_name), "%s_main" % active_obstacle_name):
        try:
            return int(model.body_name2id(name))
        except (KeyError, ValueError):
            continue
    raise ValueError("active obstacle root body is unavailable")


def minimum_union_support_gaps(
    robot_ellipsoids: Sequence[Ellipsoid],
    obstacle_ellipsoids: Sequence[Ellipsoid],
) -> Any:
    """Return one conservative minimum pair gap for every robot proxy."""

    np = _numpy()
    if not robot_ellipsoids or not obstacle_ellipsoids:
        raise ValueError("support-gap union requires nonempty ellipsoid sets")
    matrix = np.asarray(
        [
            [support_gap(robot, obstacle) for obstacle in obstacle_ellipsoids]
            for robot in robot_ellipsoids
        ],
        dtype=np.float64,
    )
    if matrix.shape != (len(robot_ellipsoids), len(obstacle_ellipsoids)):
        raise ValueError("support-gap union matrix shape differs")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("support-gap union is nonfinite")
    return np.min(matrix, axis=1)


class ConservativeObstaclePrimitiveUnion:
    """Rigid live ellipsoid union enclosing all active collision geometries."""

    def __init__(
        self,
        config: Mapping[str, Any],
        env: Any,
        active_obstacle_name: str,
    ) -> None:
        if config.get("schema_version") != OBSTACLE_PRIMITIVE_SCHEMA:
            raise ValueError("obstacle primitive configuration was not validated")
        self.config = dict(config)
        self.active_obstacle_name = str(active_obstacle_name)
        self._templates = self._fit_templates(env)

    def _fit_templates(self, env: Any) -> list[dict[str, Any]]:
        np = _numpy()
        model, data = _raw_model_data(env.sim)
        wrapper_model = env.sim.model
        root_id = _obstacle_root_body_id(wrapper_model, self.active_obstacle_name)
        settings = self.config["obstacle_geometry"]
        templates = []
        for geom_id in range(int(model.ngeom)):
            body_id = int(model.geom_bodyid[geom_id])
            if root_id not in _body_lineage(model, body_id):
                continue
            if (
                int(model.geom_contype[geom_id]) == 0
                and int(model.geom_conaffinity[geom_id]) == 0
            ):
                continue
            kind = _geom_kind(int(model.geom_type[geom_id]))
            if kind in ("plane", "hfield") or kind.startswith("unknown_"):
                raise ValueError("unsupported active-obstacle collision geom: %s" % kind)
            geom_name = _name(wrapper_model, "geom", geom_id)
            if not geom_name:
                geom_name = "unnamed_obstacle_geom_%d" % geom_id
            body_name = _name(wrapper_model, "body", body_id)
            if not body_name:
                body_name = "unnamed_obstacle_body_%d" % body_id
            geom_rotation = np.asarray(
                data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            geom_position = np.asarray(
                data.geom_xpos[geom_id], dtype=np.float64
            )
            rbound = float(model.geom_rbound[geom_id])
            mesh_record = None
            if kind == "mesh":
                mesh_id = int(model.geom_dataid[geom_id])
                if mesh_id < 0:
                    raise ValueError("obstacle collision mesh lacks compiled data")
                address = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                if count < 4:
                    raise ValueError("obstacle collision mesh has fewer than four vertices")
                local = np.asarray(
                    model.mesh_vert[address : address + count], dtype=np.float64
                )
                points = local @ geom_rotation.T + geom_position
                maximum_radius = float(
                    np.max(np.linalg.norm(points - geom_position, axis=1))
                )
                if maximum_radius > rbound + 1.0e-8:
                    raise ValueError("obstacle mesh vertices exceed geom_rbound")
                mesh_record = {
                    "body_name": body_name,
                    "geom_id": geom_id,
                    "geom_name": geom_name,
                    "mesh_id": mesh_id,
                    "compiled_vertex_count": count,
                    "maximum_vertex_radius_m": maximum_radius,
                    "geom_rbound_m": rbound,
                    "vertices_within_geom_rbound": True,
                }
                fitted = minimum_volume_enclosing_ellipsoid(
                    points,
                    body_id=body_id,
                    body_name=body_name,
                    geom_id=geom_id,
                    geom_name=geom_name,
                    source_body_names=(body_name,),
                    source_geom_names=(geom_name,),
                    relative_padding=float(
                        settings["relative_numerical_padding"]
                    ),
                    tolerance=float(settings["khachiyan_tolerance"]),
                    max_iterations=int(settings["khachiyan_max_iterations"]),
                    certificate_metadata={"source_meshes": [mesh_record]},
                )
            else:
                source_size = np.asarray(
                    model.geom_size[geom_id], dtype=np.float64
                )
                semiaxes, source = primitive_bounding_radii(
                    kind, source_size, rbound
                )
                certificate = primitive_enclosure_certificate(
                    kind, source_size, rbound, semiaxes, source
                )
                fitted = Ellipsoid(
                    center=geom_position,
                    rotation=geom_rotation,
                    semiaxes_m=semiaxes,
                    body_id=body_id,
                    body_name=body_name,
                    geom_id=geom_id,
                    geom_name=geom_name,
                    bound_source=source,
                    source_rbound_m=rbound,
                    source_geom_kind=kind,
                    source_geom_size_m=source_size,
                    source_body_names=(body_name,),
                    source_geom_names=(geom_name,),
                    enclosure_certificate=certificate,
                )
            certificate = dict(fitted.enclosure_certificate or {})
            if certificate.get("verified") is not True:
                raise ValueError("obstacle primitive lacks enclosure certificate")
            certificate["rigid_source_geom_pose_update_preserves_enclosure"] = True
            templates.append(
                {
                    "geom_id": geom_id,
                    "geom_name": geom_name,
                    "body_id": body_id,
                    "body_name": body_name,
                    "source_geom_kind": kind,
                    "center_geom_m": (
                        geom_rotation.T @ (fitted.center - geom_position)
                    ),
                    "rotation_geom": geom_rotation.T @ fitted.rotation,
                    "semiaxes_m": fitted.semiaxes_m.copy(),
                    "bound_source": fitted.bound_source,
                    "source_rbound_m": fitted.source_rbound_m,
                    "source_geom_size_m": (
                        None
                        if fitted.source_geom_size_m is None
                        else fitted.source_geom_size_m.copy()
                    ),
                    "certificate": certificate,
                    "mesh_record": mesh_record,
                }
            )
        maximum = int(settings["maximum_primitive_count"])
        if not templates:
            raise ValueError("active obstacle has no collision-active geometries")
        if len(templates) > maximum:
            raise ValueError("active obstacle primitive count exceeds preregistered maximum")
        templates.sort(key=lambda item: int(item["geom_id"]))
        geom_ids = [int(item["geom_id"]) for item in templates]
        if len(set(geom_ids)) != len(geom_ids):
            raise ValueError("active obstacle collision geom represented more than once")
        return templates

    def ellipsoids(self, env: Any) -> list[Ellipsoid]:
        np = _numpy()
        model, data = _raw_model_data(env.sim)
        output = []
        for template in self._templates:
            geom_id = int(template["geom_id"])
            if geom_id >= int(model.ngeom):
                raise ValueError("obstacle primitive geom id is unavailable")
            live_name = _name(env.sim.model, "geom", geom_id)
            if live_name != template["geom_name"]:
                raise ValueError("obstacle primitive source geom identity differs")
            rotation = np.asarray(
                data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            output.append(
                Ellipsoid(
                    center=position + rotation @ template["center_geom_m"],
                    rotation=rotation @ template["rotation_geom"],
                    semiaxes_m=template["semiaxes_m"],
                    body_id=int(template["body_id"]),
                    body_name=str(template["body_name"]),
                    geom_id=geom_id,
                    geom_name=str(template["geom_name"]),
                    bound_source=str(template["bound_source"]),
                    source_rbound_m=template["source_rbound_m"],
                    source_geom_kind=str(template["source_geom_kind"]),
                    source_geom_size_m=template["source_geom_size_m"],
                    source_body_names=(str(template["body_name"]),),
                    source_geom_names=(str(template["geom_name"]),),
                    enclosure_certificate=template["certificate"],
                )
            )
        return output

    def geometry_record(self, env: Any) -> dict[str, Any]:
        ellipsoids = self.ellipsoids(env)
        certificates = [dict(item.enclosure_certificate or {}) for item in ellipsoids]
        mesh_count = sum(item.source_geom_kind == "mesh" for item in ellipsoids)
        return {
            "schema_version": OBSTACLE_PRIMITIVE_SCHEMA,
            "active_obstacle_name": self.active_obstacle_name,
            "primitive_count": len(ellipsoids),
            "mesh_primitive_count": int(mesh_count),
            "closed_form_primitive_count": int(len(ellipsoids) - mesh_count),
            "all_collision_geoms_represented_exactly_once": True,
            "all_enclosure_certificates_verified": all(
                item.get("verified") is True for item in certificates
            ),
            "union_proof": (
                "each_collision_geom_or_compiled_mesh_convex_hull_is_contained_"
                "by_its_rigidly_attached_ellipsoid"
            ),
            "clearance_semantics": self.config["obstacle_geometry"][
                "clearance_reduction"
            ],
            "primitives": [item.to_record() for item in ellipsoids],
        }
