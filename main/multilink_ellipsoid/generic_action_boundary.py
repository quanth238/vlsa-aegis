"""Generic symmetric Cartesian candidates for future-risk identification."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _uniform_profile() -> list[float]:
    value = 1.0 / math.sqrt(5.0)
    return [value] * 5


def candidate_definitions(
    nominal_actions: Sequence[Sequence[float]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return nominal plus symmetric axis candidates at registered radii.

    The basis is intentionally independent of obstacle normals and future
    outcomes.  Only XYZ translation is perturbed; rotation and gripper remain
    exactly equal to the frozen nominal chunk.
    """

    import numpy as np

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    if nominal.shape != (5, 7) or not np.all(np.isfinite(nominal)):
        raise ValueError("generic boundary nominal chunk differs")
    if list(config["axis_order"]) != ["x", "y", "z"]:
        raise ValueError("generic boundary axis order differs")
    if [int(item) for item in config["sign_order"]] != [-1, 1]:
        raise ValueError("generic boundary sign order differs")
    radii = [float(item) for item in config["radii"]]
    if radii != [0.5, 1.5]:
        raise ValueError("generic boundary radii differ")
    if config["temporal_profile"] != "uniform_unit_L2_1_1_1_1_1":
        raise ValueError("generic boundary temporal profile differs")

    profile = np.asarray(_uniform_profile(), dtype=np.float64)
    limit = float(config["action_limit"])
    axes = np.eye(3, dtype=np.float64)
    output = [{
        "name": "nominal",
        "order": 0,
        "direction": [0.0, 0.0, 0.0],
        "sign": 0,
        "temporal_profile": profile.tolist(),
        "requested_alpha": 0.0,
        "requested_correction_l2_action": 0.0,
        "applied_correction_l2_action": 0.0,
        "clipped": False,
        "actions": nominal.tolist(),
    }]
    for radius in radii:
        for axis_index, axis_name in enumerate(config["axis_order"]):
            direction = axes[axis_index]
            for sign in config["sign_order"]:
                sign = int(sign)
                requested_delta = (
                    float(radius) * float(sign)
                    * profile[:, None] * direction[None, :]
                )
                actions = nominal.copy()
                actions[:, :3] = np.clip(
                    actions[:, :3] + requested_delta, -limit, limit,
                )
                applied = actions[:, :3] - nominal[:, :3]
                output.append({
                    "name": "%s_%s_r%0.2f" % (
                        axis_name, "pos" if sign > 0 else "neg", radius,
                    ),
                    "order": len(output),
                    "direction": direction.tolist(),
                    "axis": str(axis_name),
                    "sign": sign,
                    "temporal_profile": profile.tolist(),
                    "requested_alpha": float(radius),
                    "requested_correction_l2_action": float(radius),
                    "applied_correction_l2_action": float(np.linalg.norm(applied)),
                    "clipped": bool(
                        np.max(np.abs(applied - requested_delta)) > 1.0e-12
                    ),
                    "actions": actions.tolist(),
                })
    return output
