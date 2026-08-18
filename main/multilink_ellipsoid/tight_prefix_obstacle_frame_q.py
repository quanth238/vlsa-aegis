"""Matched 7D/17D tight-prefix future-risk critic contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_tight_prefix_obstacle_frame_q.v1"
RESULT_SCHEMA = "vlsa_tight_prefix_obstacle_frame_q_result.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_obstacle_frame_q_validation.v1"
MODEL_ROWS = tuple(range(10))
PRIMARY_ROWS = tuple(range(8))
DIAGNOSTIC_ROWS = (8, 9)
COMPACT_INPUT_DIMENSION = 7
OBSTACLE_FRAME_INPUT_DIMENSION = 17


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
        "schema_version", "protocol_id", "claim_scope", "source",
        "dataset", "representations", "frame", "model", "evaluation",
        "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("obstacle-frame Q config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-tight-prefix-obstacle-frame-q-v1"
    ):
        raise ValueError("obstacle-frame Q protocol differs")
    if value["dataset"] != {
        "fit_splits": ["train"],
        "evaluation_splits": ["train", "validation"],
        "exclude_initially_unsafe_from_fit": True,
        "test_access": False,
    }:
        raise ValueError("obstacle-frame Q dataset protocol differs")
    representations = value["representations"]
    if representations != {
        "compact_7D": {
            "input_dimension": 7,
            "definition": (
                "[initial_exact_radial_slack,"
                "nominal_first_step_outward_projection,"
                "five_effective_step_outward_projections]"
            ),
        },
        "obstacle_frame_17D": {
            "input_dimension": 17,
            "definition": (
                "[initial_exact_radial_slack,"
                "nominal_first_step_outward_projection,"
                "five_effective_step_normal_tangent1_tangent2_projections]"
            ),
        },
        "translation_scale_m_per_action_unit": 0.05,
        "model_rows": list(MODEL_ROWS),
        "primary_rows": list(PRIMARY_ROWS),
        "diagnostic_rows": list(DIAGNOSTIC_ROWS),
    }:
        raise ValueError("obstacle-frame Q representations differ")
    if value["frame"] != {
        "normal": "robot_center_minus_closest_compiled_box_center",
        "tangent1": (
            "largest_norm_compiled_box_axis_projection_onto_normal_plane_"
            "lowest_index_tie_break"
        ),
        "tangent2": "normal_cross_tangent1",
        "normalization": "ellipsoid_support_radius_along_normal",
        "orientation_source": "compiled_box_rotation",
    }:
        raise ValueError("obstacle-frame Q frame differs")
    model = value["model"]
    if (
        model.get("hidden_widths") != [32, 32]
        or model.get("activation") != "silu"
        or int(model.get("seed", -1)) != 20260814
        or int(model.get("epochs", -1)) != 2000
        or float(model.get("weight_decay", -1.0)) != 0.0001
        or model.get("checkpoint") != "fixed_final_epoch_no_selection"
    ):
        raise ValueError("obstacle-frame Q model differs")
    gate = value["gate"]
    if (
        float(gate.get("minimum_validation_pairwise_rank_improvement", -1.0))
        != 0.02
        or float(gate.get("minimum_17D_tangent_slope_sign_accuracy", -1.0))
        != 0.75
        or float(gate.get("maximum_validation_near_boundary_RMSE_ratio", -1.0))
        != 1.1
        or gate.get("require_17D_tangent_slope_RMSE_below_7D") is not True
        or gate.get("require_17D_validation_false_safes_no_more_than_7D")
        is not True
    ):
        raise ValueError("obstacle-frame Q gate differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"allocation", "result_payload_sha256", "source"}
    }


def constraint_action_frame(
    exact_case: Mapping[str, Any], row_index: int,
    *, model_rows: Sequence[int] = MODEL_ROWS,
) -> dict[str, Any]:
    """Construct one deterministic compiled-obstacle-relative action frame."""
    import numpy as np
    from main.multilink_ellipsoid.geometry import Ellipsoid
    from main.multilink_ellipsoid.obstacle_proxy_audit import (
        CompiledObstacleBox, minimum_ellipsoid_quadratic_over_box,
    )

    row_index = int(row_index)
    allowed = tuple(int(row) for row in model_rows)
    if row_index not in allowed or len(set(allowed)) != len(allowed):
        raise ValueError("obstacle-frame row differs")
    record = exact_case["initial_exact_robot_rows"][row_index]
    robot = Ellipsoid(
        center=record["center_m"], rotation=record["rotation"],
        semiaxes_m=record["semiaxes_m"],
        body_name=str(record["body_name"]),
        geom_name=str(record.get("geom_name", "")),
        bound_source=str(record.get("bound_source", "artifact")),
    )
    boxes = []
    for item in exact_case["initial_compiled_obstacle_boxes"]:
        boxes.append(CompiledObstacleBox(
            geom_id=int(item["geom_id"]), geom_name=str(item["geom_name"]),
            body_id=int(item["body_id"]), body_name=str(item["body_name"]),
            center=np.asarray(item["center_m"], dtype=np.float64),
            rotation=np.asarray(item["rotation"], dtype=np.float64),
            half_extents_m=np.asarray(item["half_extents_m"], dtype=np.float64),
        ))
    quadratics = np.asarray([
        minimum_ellipsoid_quadratic_over_box(robot, box) for box in boxes
    ], dtype=np.float64)
    box_index = int(np.argmin(quadratics))
    box = boxes[box_index]
    slack = float(math.sqrt(float(quadratics[box_index])) - 1.0)
    stored = float(exact_case["exact_group_target"][
        "initial_row_normalized_radial_slack"
    ][row_index])
    if not math.isclose(slack, stored, abs_tol=1.0e-8):
        raise ValueError("obstacle-frame initial slack differs")
    normal = np.asarray(robot.center - box.center, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise ValueError("obstacle-frame normal is degenerate")
    normal /= norm
    axes = np.asarray(box.rotation, dtype=np.float64)
    projections = axes - normal[:, None] * (normal @ axes)[None, :]
    projection_norms = np.linalg.norm(projections, axis=0)
    tangent_axis = int(np.argmax(projection_norms))
    if float(projection_norms[tangent_axis]) <= 1.0e-12:
        raise ValueError("obstacle-frame tangent is degenerate")
    tangent1 = projections[:, tangent_axis] / projection_norms[tangent_axis]
    tangent2 = np.cross(normal, tangent1)
    tangent2 /= float(np.linalg.norm(tangent2))
    frame = np.stack([normal, tangent1, tangent2], axis=1)
    if float(np.max(np.abs(frame.T @ frame - np.eye(3)))) > 1.0e-12:
        raise ValueError("obstacle-frame basis is not orthonormal")
    local_normal = np.asarray(robot.rotation, dtype=np.float64).T @ normal
    support_radius = float(np.linalg.norm(
        np.asarray(robot.semiaxes_m, dtype=np.float64) * local_normal
    ))
    if support_radius <= 1.0e-12:
        raise ValueError("obstacle-frame support radius is degenerate")
    return {
        "normal": normal.tolist(),
        "tangent1": tangent1.tolist(),
        "tangent2": tangent2.tolist(),
        "closest_box_index": box_index,
        "tangent_box_axis_index": tangent_axis,
        "support_radius_m": support_radius,
        "initial_exact_radial_slack": stored,
    }


def obstacle_frame_feature(
    exact_case: Mapping[str, Any], candidate: Mapping[str, Any], row_index: int,
    *, translation_scale: float,
    model_rows: Sequence[int] = MODEL_ROWS,
) -> list[float]:
    """Return the matched 17D feature with full local Cartesian action data."""
    import numpy as np

    frame = constraint_action_frame(
        exact_case, row_index, model_rows=model_rows,
    )
    basis = np.asarray([
        frame["normal"], frame["tangent1"], frame["tangent2"],
    ], dtype=np.float64)
    support_radius = float(frame["support_radius_m"])

    def projection(action: Sequence[float]) -> list[float]:
        if len(action) != 7:
            raise ValueError("obstacle-frame action differs")
        displacement = float(translation_scale) * np.asarray(
            action[:3], dtype=np.float64,
        )
        return (basis @ displacement / support_radius).tolist()

    nominal = exact_case["source_nominal_five_action_chunk"]
    effective = candidate["source_executed_actions"]
    if len(nominal) != 5 or len(effective) != 5:
        raise ValueError("obstacle-frame action chunk differs")
    feature = [
        float(frame["initial_exact_radial_slack"]), projection(nominal[0])[0],
    ]
    for action in effective:
        feature.extend(projection(action))
    if len(feature) != OBSTACLE_FRAME_INPUT_DIMENSION or not all(
        math.isfinite(value) for value in feature
    ):
        raise ValueError("obstacle-frame feature is invalid")
    return feature


def same_state_pairwise_rank_accuracy(
    samples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[float]],
    *, rows: Sequence[int], tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Compare all non-tied candidate pairs within each physical state."""
    import numpy as np

    estimate = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if len(samples) != len(estimate):
        raise ValueError("obstacle-frame rank prediction shape differs")
    selected_rows = tuple(int(row) for row in rows)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for sample, predicted in zip(samples, estimate):
        row = int(sample["row_index"])
        if row not in selected_rows:
            continue
        key = (str(sample["state_id"]), str(sample["candidate_name"]))
        entry = grouped.setdefault(key, {"actual": {}, "predicted": {}})
        entry["actual"][row] = float(sample["risk"])
        entry["predicted"][row] = float(predicted)
    by_state: dict[str, list[dict[str, float]]] = {}
    for (state_id, _), value in grouped.items():
        if sorted(value["actual"]) != sorted(selected_rows):
            raise ValueError("obstacle-frame rank rows differ")
        by_state.setdefault(state_id, []).append({
            "actual": max(value["actual"].values()),
            "predicted": max(value["predicted"].values()),
        })
    correct = 0
    pairs = 0
    state_reports = []
    for state_id in sorted(by_state):
        rows_for_state = by_state[state_id]
        state_correct = 0
        state_pairs = 0
        for left in range(len(rows_for_state)):
            for right in range(left + 1, len(rows_for_state)):
                actual_delta = (
                    rows_for_state[left]["actual"]
                    - rows_for_state[right]["actual"]
                )
                if abs(actual_delta) <= float(tolerance):
                    continue
                predicted_delta = (
                    rows_for_state[left]["predicted"]
                    - rows_for_state[right]["predicted"]
                )
                hit = bool(actual_delta * predicted_delta > 0.0)
                state_correct += int(hit)
                state_pairs += 1
        correct += state_correct
        pairs += state_pairs
        state_reports.append({
            "state_id": state_id,
            "pair_count": state_pairs,
            "correct_pair_count": state_correct,
            "accuracy": None if not state_pairs else state_correct / state_pairs,
        })
    return {
        "pair_count": pairs,
        "correct_pair_count": correct,
        "accuracy": None if not pairs else correct / pairs,
        "states": state_reports,
    }
