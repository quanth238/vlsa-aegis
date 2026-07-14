"""MuJoCo-geometry clearance measurement at physics-substep resolution.

The raw MuJoCo geom distance is deliberately kept separate from the
conservative sphere-to-box primitive used by the controlled CRFS pilot.  H03
must establish their semantics against contact pairs before either signal is
allowed to drive an oracle optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ClearanceMeasurement:
    min_clearance_m: float
    contact: bool
    samples: int
    min_pair: tuple[str, str] | None
    distance_limit_m: float
    min_substep_index: int | None
    min_fromto_m: tuple[float, ...] | None
    min_pair_metadata: dict[str, Any] | None
    first_contact_substep_index: int | None
    first_contact_distance_m: float | None
    minimum_contact_distance_m: float | None
    raw_mj_contact_consistent: bool
    contact_tolerance_m: float
    conservative_clearance_m: float | None
    conservative_min_substep_index: int | None
    conservative_obstacle_geom: str | None
    conservative_eef_center_m: tuple[float, float, float] | None

    def to_dict(self) -> dict:
        return asdict(self)


def _raw_model_data(sim):
    """Return DeepMind MuJoCo objects through robosuite's thin wrappers."""
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    return model, data


def _geom_id(sim, name: str) -> int:
    geom_id = int(sim.model.geom_name2id(name))
    if geom_id < 0:
        raise ValueError(f"Unknown MuJoCo geom: {name!r}")
    return geom_id


def signed_distance_point_to_oriented_box(
    point_world_m: Sequence[float],
    box_center_world_m: Sequence[float],
    box_rotation_world: np.ndarray,
    box_half_size_m: Sequence[float],
) -> float:
    """Exact signed Euclidean distance from a point to an oriented box.

    Negative values are inside the box, zero is on its surface, and positive
    values are outside. MuJoCo stores box ``geom_size`` as half-extents.
    """
    point = np.asarray(point_world_m, dtype=np.float64)
    center = np.asarray(box_center_world_m, dtype=np.float64)
    rotation = np.asarray(box_rotation_world, dtype=np.float64).reshape(3, 3)
    half_size = np.asarray(box_half_size_m, dtype=np.float64)
    if point.shape != (3,) or center.shape != (3,) or half_size.shape != (3,):
        raise ValueError("Point, center, and half-size must all have shape (3,)")
    if np.any(half_size <= 0):
        raise ValueError("Box half-sizes must be positive")
    local = rotation.T @ (point - center)
    delta = np.abs(local) - half_size
    outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
    inside = float(min(np.max(delta), 0.0))
    return outside + inside


def _geom_type_name(mujoco_module, value: int) -> str:
    try:
        return str(mujoco_module.mjtGeom(int(value)).name)
    except (TypeError, ValueError):
        return f"UNKNOWN_{int(value)}"


def _geom_metadata(sim, mujoco_module, geom_id: int) -> dict[str, Any]:
    body_id = int(sim.model.geom_bodyid[geom_id])
    return {
        "id": int(geom_id),
        "name": str(sim.model.geom_id2name(geom_id)),
        "type": _geom_type_name(mujoco_module, int(sim.model.geom_type[geom_id])),
        "body_id": body_id,
        "body_name": str(sim.model.body_id2name(body_id)),
        "contype": int(sim.model.geom_contype[geom_id]),
        "conaffinity": int(sim.model.geom_conaffinity[geom_id]),
        "margin_m": float(sim.model.geom_margin[geom_id]),
        "gap_m": float(sim.model.geom_gap[geom_id]),
        "size_m": np.asarray(sim.model.geom_size[geom_id], dtype=np.float64).tolist(),
        "world_position_m": np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64).tolist(),
    }


class GeomClearanceMonitor:
    """Track raw geom distance, contacts, and a conservative pilot primitive."""

    def __init__(
        self,
        sim,
        eef_geoms: Iterable[str],
        obstacle_geoms: Iterable[str],
        *,
        distance_limit_m: float = 1.0,
        contact_tolerance_m: float = 1e-4,
        eef_site_id: int | None = None,
        eef_center_offset_local_m: Sequence[float] = (0.0, 0.0, -0.08),
        eef_radius_m: float | None = None,
    ) -> None:
        if distance_limit_m <= 0:
            raise ValueError("distance_limit_m must be positive")
        if contact_tolerance_m < 0:
            raise ValueError("contact_tolerance_m must be non-negative")
        if (eef_site_id is None) != (eef_radius_m is None):
            raise ValueError("eef_site_id and eef_radius_m must be supplied together")
        if eef_radius_m is not None and eef_radius_m <= 0:
            raise ValueError("eef_radius_m must be positive")
        offset = np.asarray(eef_center_offset_local_m, dtype=np.float64)
        if offset.shape != (3,):
            raise ValueError("eef_center_offset_local_m must have shape (3,)")

        self._distance_limit_m = float(distance_limit_m)
        self._contact_tolerance_m = float(contact_tolerance_m)
        self._eef = tuple((str(name), _geom_id(sim, str(name))) for name in eef_geoms)
        self._obstacle = tuple((str(name), _geom_id(sim, str(name))) for name in obstacle_geoms)
        if not self._eef or not self._obstacle:
            raise ValueError("Both EEF and obstacle geom groups must be non-empty")
        self._eef_ids = {identity for _, identity in self._eef}
        self._obstacle_ids = {identity for _, identity in self._obstacle}
        self._eef_site_id = int(eef_site_id) if eef_site_id is not None else None
        self._eef_center_offset = offset
        self._eef_radius_m = float(eef_radius_m) if eef_radius_m is not None else None

        self._minimum = float("inf")
        self._minimum_pair: tuple[str, str] | None = None
        self._minimum_substep: int | None = None
        self._minimum_fromto: tuple[float, ...] | None = None
        self._minimum_metadata: dict[str, Any] | None = None
        self._contact = False
        self._first_contact_substep: int | None = None
        self._first_contact_distance: float | None = None
        self._minimum_contact_distance: float | None = None
        self._samples = 0
        self._conservative_minimum = float("inf")
        self._conservative_substep: int | None = None
        self._conservative_obstacle_geom: str | None = None
        self._conservative_eef_center: tuple[float, float, float] | None = None

    def _observe_conservative_sphere_boxes(self, sim, mujoco_module, substep_index: int) -> None:
        if self._eef_site_id is None or self._eef_radius_m is None:
            return
        site_position = np.asarray(sim.data.site_xpos[self._eef_site_id], dtype=np.float64)
        site_rotation = np.asarray(sim.data.site_xmat[self._eef_site_id], dtype=np.float64).reshape(3, 3)
        eef_center = site_position + site_rotation @ self._eef_center_offset
        box_type = int(mujoco_module.mjtGeom.mjGEOM_BOX)
        for obstacle_name, obstacle_id in self._obstacle:
            if int(sim.model.geom_type[obstacle_id]) != box_type:
                continue
            point_distance = signed_distance_point_to_oriented_box(
                eef_center,
                sim.data.geom_xpos[obstacle_id],
                sim.data.geom_xmat[obstacle_id],
                sim.model.geom_size[obstacle_id],
            )
            clearance = point_distance - self._eef_radius_m
            if clearance < self._conservative_minimum:
                self._conservative_minimum = clearance
                self._conservative_substep = int(substep_index)
                self._conservative_obstacle_geom = obstacle_name
                self._conservative_eef_center = tuple(float(value) for value in eef_center)

    def observe(self, sim, substep_index: int | None = None) -> None:
        """Record one post-integration physics state."""
        import mujoco

        sample_substep = int(substep_index) if substep_index is not None else self._samples
        model, data = _raw_model_data(sim)
        fromto = np.empty(6, dtype=np.float64)
        for eef_name, eef_id in self._eef:
            for obstacle_name, obstacle_id in self._obstacle:
                distance = float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        eef_id,
                        obstacle_id,
                        self._distance_limit_m,
                        fromto,
                    )
                )
                if distance < self._minimum:
                    eef_metadata = _geom_metadata(sim, mujoco, eef_id)
                    obstacle_metadata = _geom_metadata(sim, mujoco, obstacle_id)
                    collision_enabled = bool(
                        (eef_metadata["contype"] & obstacle_metadata["conaffinity"])
                        or (obstacle_metadata["contype"] & eef_metadata["conaffinity"])
                    )
                    self._minimum = distance
                    self._minimum_pair = (eef_name, obstacle_name)
                    self._minimum_substep = sample_substep
                    self._minimum_fromto = tuple(float(value) for value in fromto)
                    self._minimum_metadata = {
                        "eef": eef_metadata,
                        "obstacle": obstacle_metadata,
                        "collision_enabled": collision_enabled,
                    }

        for contact in sim.data.contact[: sim.data.ncon]:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            paired = (
                geom1 in self._eef_ids and geom2 in self._obstacle_ids
            ) or (
                geom2 in self._eef_ids and geom1 in self._obstacle_ids
            )
            if paired:
                contact_distance = float(contact.dist)
                if not self._contact:
                    self._first_contact_substep = sample_substep
                    self._first_contact_distance = contact_distance
                self._contact = True
                if self._minimum_contact_distance is None or contact_distance < self._minimum_contact_distance:
                    self._minimum_contact_distance = contact_distance

        self._observe_conservative_sphere_boxes(sim, mujoco, sample_substep)
        self._samples += 1

    def result(self) -> ClearanceMeasurement:
        if self._samples == 0:
            raise RuntimeError("No physics substeps were measured")
        collision_enabled = bool(
            self._minimum_metadata is not None and self._minimum_metadata.get("collision_enabled", False)
        )
        raw_mj_contact_consistent = collision_enabled and not (
            (not self._contact and self._minimum < -self._contact_tolerance_m)
            or (self._contact and self._minimum > self._contact_tolerance_m)
        )
        conservative = None if np.isinf(self._conservative_minimum) else self._conservative_minimum
        return ClearanceMeasurement(
            min_clearance_m=self._minimum,
            contact=self._contact,
            samples=self._samples,
            min_pair=self._minimum_pair,
            distance_limit_m=self._distance_limit_m,
            min_substep_index=self._minimum_substep,
            min_fromto_m=self._minimum_fromto,
            min_pair_metadata=self._minimum_metadata,
            first_contact_substep_index=self._first_contact_substep,
            first_contact_distance_m=self._first_contact_distance,
            minimum_contact_distance_m=self._minimum_contact_distance,
            raw_mj_contact_consistent=raw_mj_contact_consistent,
            contact_tolerance_m=self._contact_tolerance_m,
            conservative_clearance_m=conservative,
            conservative_min_substep_index=self._conservative_substep,
            conservative_obstacle_geom=self._conservative_obstacle_geom,
            conservative_eef_center_m=self._conservative_eef_center,
        )


def resolve_crfs_geom_groups(env, obstacle_name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve the baseline Panda gripper and SafeLIBERO obstacle geoms."""
    gripper = env.robots[0].gripper
    eef_geoms = tuple(str(name) for name in gripper.contact_geoms)
    obstacle_object = env.env.objects_dict[obstacle_name]
    obstacle_geoms = tuple(str(name) for name in obstacle_object.contact_geoms)
    if not eef_geoms:
        raise RuntimeError("The baseline gripper exposes no contact geoms")
    if not obstacle_geoms:
        raise RuntimeError(f"Obstacle {obstacle_name!r} exposes no contact geoms")
    return eef_geoms, obstacle_geoms
