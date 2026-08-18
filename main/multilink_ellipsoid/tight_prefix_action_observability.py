"""Contracts for the symmetric normal/tangent action-observability audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_tight_prefix_action_observability.v1"
CASE_SCHEMA = "vlsa_tight_prefix_action_observability_case.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_action_observability_validation.v1"
PRIMARY_ROWS = tuple(range(8))
DIAGNOSTIC_ROWS = (8, 9)
DIRECTION_NAMES = (
    "normal_pos", "normal_neg", "tangent_up_pos", "tangent_up_neg",
    "tangent_side_pos", "tangent_side_neg",
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "source", "cases",
        "perturbation", "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("action-observability config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-tight-prefix-action-observability-v1"
    ):
        raise ValueError("action-observability protocol differs")
    cases = value["cases"]
    if (
        len(cases) != 4
        or [int(item["tight_dataset_case_index"]) for item in cases]
        != [0, 2, 10, 15]
        or any(item.get("split") != "validation" for item in cases)
    ):
        raise ValueError("action-observability cases differ")
    perturbation = value["perturbation"]
    if perturbation != {
        "basis": ["normal", "tangent_up", "tangent_side"],
        "directions": list(DIRECTION_NAMES),
        "temporal_profile": "front_loaded_unit_L2_5_4_3_2_1",
        "maximum_translation_l2_action": 0.25,
        "minimum_translation_l2_action": 0.01,
        "symmetric_headroom_fraction": 0.8,
        "action_limit": 1.0,
        "primary_rows": list(PRIMARY_ROWS),
        "diagnostic_rows": list(DIAGNOSTIC_ROWS),
        "execution": "direct_effective_action_through_unchanged_OSC_five_actions_only",
    }:
        raise ValueError("action-observability perturbation differs")
    gate = value["gate"]
    if (
        int(gate.get("required_case_count", 0)) != 4
        or int(gate.get("minimum_structural_counterexample_count", 0)) != 1
        or int(gate.get("minimum_material_tangent_descent_case_count", 0)) != 2
        or float(gate.get("material_absolute_risk_change", -1.0)) != 0.01
        or float(gate.get("maximum_tangent_7D_feature_change", -1.0)) != 1e-10
        or float(gate.get("maximum_post_clipping_symmetry_error", -1.0)) != 1e-12
    ):
        raise ValueError("action-observability gate differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"allocation", "result_payload_sha256", "source"}
    }


def _orthonormal_frame(normal: Sequence[float]) -> dict[str, list[float]]:
    import numpy as np

    normal_array = np.asarray(normal, dtype=np.float64)
    normal_array /= float(np.linalg.norm(normal_array))
    tangent_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    tangent_up -= float(tangent_up @ normal_array) * normal_array
    if float(np.linalg.norm(tangent_up)) <= 1e-12:
        fallback = np.eye(3)[int(np.argmin(np.abs(normal_array)))]
        tangent_up = fallback - float(fallback @ normal_array) * normal_array
    tangent_up /= float(np.linalg.norm(tangent_up))
    tangent_side = np.cross(normal_array, tangent_up)
    tangent_side /= float(np.linalg.norm(tangent_side))
    frame = np.stack([normal_array, tangent_up, tangent_side], axis=1)
    if float(np.max(np.abs(frame.T @ frame - np.eye(3)))) > 1e-12:
        raise ValueError("action-observability frame is not orthonormal")
    return {
        "normal": normal_array.tolist(),
        "tangent_up": tangent_up.tolist(),
        "tangent_side": tangent_side.tolist(),
    }


def active_frame(
    exact_case: Mapping[str, Any], primary_rows: Sequence[int],
) -> dict[str, Any]:
    """Return the closest represented primary row/box and its local frame."""
    import numpy as np
    from main.multilink_ellipsoid.geometry import Ellipsoid
    from main.multilink_ellipsoid.obstacle_proxy_audit import (
        CompiledObstacleBox, minimum_ellipsoid_quadratic_over_box,
    )

    rows = []
    for record in exact_case["initial_exact_robot_rows"]:
        rows.append(Ellipsoid(
            center=record["center_m"], rotation=record["rotation"],
            semiaxes_m=record["semiaxes_m"],
            body_name=str(record["body_name"]),
            geom_name=str(record.get("geom_name", "")),
            bound_source=str(record.get("bound_source", "artifact")),
        ))
    boxes = []
    for record in exact_case["initial_compiled_obstacle_boxes"]:
        boxes.append(CompiledObstacleBox(
            geom_id=int(record["geom_id"]), geom_name=str(record["geom_name"]),
            body_id=int(record["body_id"]), body_name=str(record["body_name"]),
            center=np.asarray(record["center_m"], dtype=np.float64),
            rotation=np.asarray(record["rotation"], dtype=np.float64),
            half_extents_m=np.asarray(record["half_extents_m"], dtype=np.float64),
        ))
    choices = []
    for row_index in (int(value) for value in primary_rows):
        for box_index, box in enumerate(boxes):
            choices.append((
                float(minimum_ellipsoid_quadratic_over_box(rows[row_index], box)),
                row_index, box_index,
            ))
    quadratic, row_index, box_index = min(choices)
    normal = np.asarray(rows[row_index].center) - np.asarray(boxes[box_index].center)
    if float(np.linalg.norm(normal)) <= 1e-12:
        raise ValueError("action-observability normal is degenerate")
    return {
        **_orthonormal_frame(normal),
        "active_row": int(row_index),
        "active_box": int(box_index),
        "active_initial_slack": float(math.sqrt(quadratic) - 1.0),
    }


def symmetric_action_overrides(
    nominal_actions: Sequence[Sequence[float]], frame: Mapping[str, Any],
    *, maximum_radius: float, minimum_radius: float,
    headroom_fraction: float, action_limit: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate equal-radius, exactly symmetric local-frame perturbations."""
    import numpy as np
    from main.multilink_ellipsoid.query_action_risk import temporal_profile

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    if nominal.shape != (5, 7) or not np.all(np.isfinite(nominal)):
        raise ValueError("action-observability nominal differs")
    profile = np.asarray(temporal_profile("front_loaded"), dtype=np.float64)
    directions = {
        "normal_pos": np.asarray(frame["normal"], dtype=np.float64),
        "normal_neg": -np.asarray(frame["normal"], dtype=np.float64),
        "tangent_up_pos": np.asarray(frame["tangent_up"], dtype=np.float64),
        "tangent_up_neg": -np.asarray(frame["tangent_up"], dtype=np.float64),
        "tangent_side_pos": np.asarray(frame["tangent_side"], dtype=np.float64),
        "tangent_side_neg": -np.asarray(frame["tangent_side"], dtype=np.float64),
    }
    headroom = float(action_limit) - np.abs(nominal[:, :3])
    radius_bound = math.inf
    for direction in directions.values():
        unit = profile[:, None] * np.abs(direction[None, :])
        active = unit > 1e-15
        if np.any(active):
            radius_bound = min(
                radius_bound, float(np.min(headroom[active] / unit[active])),
            )
    radius = min(float(maximum_radius), float(headroom_fraction) * radius_bound)
    if not math.isfinite(radius) or radius < float(minimum_radius):
        raise ValueError("action-observability symmetric radius is unavailable")
    output = [{
        "name": "nominal", "actions": nominal.tolist(),
        "requested_alpha": 0.0, "effective_correction_l2_action": 0.0,
        "metadata": {"axis": "nominal", "sign": 0},
    }]
    deltas = {}
    for name in DIRECTION_NAMES:
        direction = directions[name]
        delta = radius * profile[:, None] * direction[None, :]
        actions = nominal.copy()
        actions[:, :3] += delta
        if float(np.max(np.abs(actions[:, :3]))) > float(action_limit) + 1e-12:
            raise ValueError("action-observability action bound differs")
        deltas[name] = delta
        axis, sign = name.rsplit("_", 1)
        output.append({
            "name": name, "actions": actions.tolist(),
            "requested_alpha": float(radius),
            "effective_correction_l2_action": float(np.linalg.norm(delta)),
            "metadata": {
                "axis": axis, "sign": 1 if sign == "pos" else -1,
                "direction": direction.tolist(),
            },
        })
    symmetry = max(
        float(np.max(np.abs(deltas[positive] + deltas[negative])))
        for positive, negative in (
            ("normal_pos", "normal_neg"),
            ("tangent_up_pos", "tangent_up_neg"),
            ("tangent_side_pos", "tangent_side_neg"),
        )
    )
    norm_error = max(
        abs(float(np.linalg.norm(value)) - radius) for value in deltas.values()
    )
    return output, {
        "common_translation_l2_action": float(radius),
        "maximum_pair_symmetry_error": float(symmetry),
        "maximum_equal_norm_error": float(norm_error),
    }


def row_future_risk(candidate: Mapping[str, Any], row: int) -> float:
    trace = candidate["exact_group_target"]["trace"]
    if not trace:
        raise ValueError("action-observability trace is empty")
    return -min(
        float(sample["row_normalized_radial_slack"][int(row)])
        for sample in trace
    )
