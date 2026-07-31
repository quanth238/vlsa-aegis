"""Robot body-point records and rigid body-frame transformations."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("NumPy is required for robot surface samples") from error
    return np


def body_world_pose(data: Any, body_id: int) -> Tuple[Any, Any]:
    """Return one body's world pose through either supported MuJoCo API.

    Robosuite's legacy wrapper exposes ``body_xpos/body_xmat``.  Official
    ``mujoco.MjData`` exposes ``xpos/xmat``.  Live SafeLIBERO and standalone
    numerical validation use different forms, so both are explicit here.
    """

    np = _numpy()
    raw_data = getattr(data, "_data", data)
    positions = getattr(data, "body_xpos", None)
    rotations = getattr(data, "body_xmat", None)
    if positions is None:
        positions = getattr(raw_data, "xpos", None)
    if rotations is None:
        rotations = getattr(raw_data, "xmat", None)
    if positions is None or rotations is None:
        raise TypeError(
            "data must expose Robosuite body_xpos/body_xmat or MuJoCo xpos/xmat"
        )
    if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id < 0:
        raise ValueError("body_id must be a non-negative integer")
    try:
        position = np.asarray(positions[body_id], dtype=np.float64)
        rotation = np.asarray(rotations[body_id], dtype=np.float64).reshape(3, 3)
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("body_id is outside the available body pose arrays") from error
    if position.shape != (3,) or not (
        np.all(np.isfinite(position)) and np.all(np.isfinite(rotation))
    ):
        raise ValueError("body world pose must be finite")
    return position, rotation


@dataclass(frozen=True)
class BodySample:
    """One robot collision-surface point stored in its body-local frame."""

    sample_id: int
    body_id: int
    body_name: str
    geom_id: int
    geom_name: str
    point_body_local_m: Tuple[float, float, float]
    source: str = "collision_geom_surface"

    def __post_init__(self) -> None:
        if self.sample_id < 0 or self.body_id < 0 or self.geom_id < 0:
            raise ValueError("sample, body, and geom IDs must be non-negative")
        if not self.body_name or not self.geom_name:
            raise ValueError("sample body and geom names must be non-empty")
        np = _numpy()
        point = np.asarray(self.point_body_local_m, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point_body_local_m must be a finite 3-vector")

    def point_local_array(self) -> Any:
        np = _numpy()
        return np.asarray(self.point_body_local_m, dtype=np.float64)

    def world_point(self, data: Any) -> Any:
        """Transform this point using the owning body's live MuJoCo pose."""

        position, rotation = body_world_pose(data, self.body_id)
        return position + rotation @ self.point_local_array()

    def recover_local_point(self, data: Any, point_world_m: Sequence[float]) -> Any:
        np = _numpy()
        world = np.asarray(point_world_m, dtype=np.float64)
        if world.shape != (3,) or not np.all(np.isfinite(world)):
            raise ValueError("point_world_m must be a finite 3-vector")
        position, rotation = body_world_pose(data, self.body_id)
        return rotation.T @ (world - position)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": int(self.sample_id),
            "body_id": int(self.body_id),
            "body_name": self.body_name,
            "geom_id": int(self.geom_id),
            "geom_name": self.geom_name,
            "point_body_local_m": [float(value) for value in self.point_body_local_m],
            "source": self.source,
        }


def evaluate_world_points(samples: Iterable[BodySample], data: Any) -> Any:
    np = _numpy()
    records = list(samples)
    if not records:
        return np.empty((0, 3), dtype=np.float64)
    identifiers = [sample.sample_id for sample in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("body sample IDs must be unique")
    return np.stack([sample.world_point(data) for sample in records], axis=0)


def validate_rigid_roundtrip(
    samples: Iterable[BodySample],
    data: Any,
    *,
    tolerance_m: float = 1e-10,
) -> Dict[str, Any]:
    """Audit that local/world transforms are mutually consistent."""

    np = _numpy()
    if tolerance_m <= 0.0:
        raise ValueError("tolerance_m must be positive")
    records = list(samples)
    errors: List[float] = []
    for sample in records:
        world = sample.world_point(data)
        recovered = sample.recover_local_point(data, world)
        errors.append(float(np.linalg.norm(recovered - sample.point_local_array())))
    maximum = max(errors) if errors else 0.0
    return {
        "sample_count": len(records),
        "maximum_roundtrip_error_m": maximum,
        "tolerance_m": float(tolerance_m),
        "passed": bool(maximum <= tolerance_m),
    }
