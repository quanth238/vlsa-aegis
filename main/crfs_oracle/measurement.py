"""MuJoCo-geometry clearance measurement at physics-substep resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ClearanceMeasurement:
    min_clearance_m: float
    contact: bool
    samples: int
    min_pair: tuple[str, str] | None
    distance_limit_m: float

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


class GeomClearanceMonitor:
    """Track exact geom distance and contact for two named geom groups."""

    def __init__(
        self,
        sim,
        eef_geoms: Iterable[str],
        obstacle_geoms: Iterable[str],
        *,
        distance_limit_m: float = 1.0,
    ) -> None:
        if distance_limit_m <= 0:
            raise ValueError("distance_limit_m must be positive")
        self._distance_limit_m = float(distance_limit_m)
        self._eef = tuple((str(name), _geom_id(sim, str(name))) for name in eef_geoms)
        self._obstacle = tuple((str(name), _geom_id(sim, str(name))) for name in obstacle_geoms)
        if not self._eef or not self._obstacle:
            raise ValueError("Both EEF and obstacle geom groups must be non-empty")
        self._eef_ids = {identity for _, identity in self._eef}
        self._obstacle_ids = {identity for _, identity in self._obstacle}
        self._minimum = float("inf")
        self._minimum_pair: tuple[str, str] | None = None
        self._contact = False
        self._samples = 0

    def observe(self, sim, _substep_index: int | None = None) -> None:
        """Record one post-integration physics state."""
        import mujoco

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
                    self._minimum = distance
                    self._minimum_pair = (eef_name, obstacle_name)

        for contact in sim.data.contact[: sim.data.ncon]:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            paired = (
                geom1 in self._eef_ids and geom2 in self._obstacle_ids
            ) or (
                geom2 in self._eef_ids and geom1 in self._obstacle_ids
            )
            if paired:
                self._contact = True
                contact_distance = float(contact.dist)
                if contact_distance < self._minimum:
                    self._minimum = contact_distance
                    name1 = sim.model.geom_id2name(geom1)
                    name2 = sim.model.geom_id2name(geom2)
                    self._minimum_pair = (str(name1), str(name2))
        self._samples += 1

    def result(self) -> ClearanceMeasurement:
        if self._samples == 0:
            raise RuntimeError("No physics substeps were measured")
        return ClearanceMeasurement(
            min_clearance_m=self._minimum,
            contact=self._contact,
            samples=self._samples,
            min_pair=self._minimum_pair,
            distance_limit_m=self._distance_limit_m,
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
