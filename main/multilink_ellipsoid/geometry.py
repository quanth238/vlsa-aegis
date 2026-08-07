"""Conservative ellipsoid bounds for MuJoCo collision geometry.

The ordinary AEGIS path never imports this module.  The opt-in observer uses
one ellipsoid per participating collision geom, so articulated links remain
separate instead of being hidden inside one giant whole-robot primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "semiaxes_m", semiaxes)

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
        }


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


def ellipsoid_record_hash_payload(ellipsoids: Sequence[Ellipsoid]) -> list[Mapping[str, Any]]:
    """Stable JSON-native geometry payload used by the shadow receipt."""

    return [item.to_record() for item in ellipsoids]
