"""Compact inference-only safety-coordinate Q predictor contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_compact_safety_coordinate_q.v1"
RESULT_SCHEMA = "vlsa_distal_compact_safety_coordinate_q_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_compact_safety_coordinate_q_validation.v1"
INPUT_DIMENSION = 7
MODEL_ROWS = (0, 1, 2, 3, 4, 5, 6)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("compact safety-coordinate config schema differs")
    if value.get("protocol_id") != "vlsa-distal-compact-safety-coordinate-q-v1":
        raise ValueError("compact safety-coordinate protocol differs")
    feature = value.get("feature", {})
    if (
        feature.get("input_dimension") != INPUT_DIMENSION
        or feature.get("model_rows") != list(MODEL_ROWS)
        or feature.get("definition")
        != "[initial_exact_radial_slack,nominal_first_step_outward_projection,"
        "five_effective_step_outward_projections]"
        or float(feature.get("translation_scale_m_per_action_unit", -1.0)) != 0.05
    ):
        raise ValueError("compact safety-coordinate feature differs")
    groups = value.get("group_rows", {})
    if groups != {
        "end_effector": [0], "palm": [1], "L5": [2, 3, 4],
        "L6_diagnostic": [5, 6],
    }:
        raise ValueError("compact safety-coordinate groups differ")
    model = value.get("model", {})
    if (
        model.get("hidden_widths") != [32, 32]
        or model.get("seed") != 20260814
        or float(model.get("weight_decay", -1.0)) != 0.0001
        or model.get("checkpoint") != "fixed_final_epoch_no_selection"
    ):
        raise ValueError("compact safety-coordinate model differs")
    if value.get("simulation_rollouts") is not False:
        raise ValueError("compact safety-coordinate experiment must not simulate")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def safety_coordinate_feature(
    exact_case: Mapping[str, Any], candidate: Mapping[str, Any], row_index: int,
    *, translation_scale: float,
    model_rows: Sequence[int] = MODEL_ROWS,
) -> list[float]:
    """Return a causal 7D current-geometry/candidate representation."""
    import numpy as np
    from main.multilink_ellipsoid.geometry import Ellipsoid
    from main.multilink_ellipsoid.obstacle_proxy_audit import (
        CompiledObstacleBox, minimum_ellipsoid_quadratic_over_box,
    )

    row_index = int(row_index)
    allowed_rows = tuple(int(row) for row in model_rows)
    if row_index not in allowed_rows or len(set(allowed_rows)) != len(allowed_rows):
        raise ValueError("compact safety-coordinate row differs")
    row_record = exact_case["initial_exact_robot_rows"][row_index]
    robot = Ellipsoid(
        center=row_record["center_m"], rotation=row_record["rotation"],
        semiaxes_m=row_record["semiaxes_m"],
        body_name=str(row_record["body_name"]),
        geom_name=str(row_record.get("geom_name", "")),
        bound_source=str(row_record.get("bound_source", "artifact")),
    )
    boxes = []
    for record in exact_case["initial_compiled_obstacle_boxes"]:
        boxes.append(CompiledObstacleBox(
            geom_id=int(record["geom_id"]), geom_name=str(record["geom_name"]),
            body_id=int(record["body_id"]), body_name=str(record["body_name"]),
            center=np.asarray(record["center_m"], dtype=np.float64),
            rotation=np.asarray(record["rotation"], dtype=np.float64),
            half_extents_m=np.asarray(record["half_extents_m"], dtype=np.float64),
        ))
    values = [minimum_ellipsoid_quadratic_over_box(robot, box) for box in boxes]
    box = boxes[int(np.argmin(np.asarray(values, dtype=np.float64)))]
    computed_slack = math.sqrt(float(min(values))) - 1.0
    stored_slack = float(exact_case["exact_group_target"][
        "initial_row_normalized_radial_slack"
    ][row_index])
    if not math.isclose(computed_slack, stored_slack, abs_tol=1.0e-8):
        raise ValueError("compact safety-coordinate initial slack differs")

    normal = np.asarray(robot.center - box.center, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise ValueError("compact safety-coordinate outward direction is degenerate")
    normal /= norm
    local_normal = np.asarray(robot.rotation, dtype=np.float64).T @ normal
    support_radius = float(np.linalg.norm(
        np.asarray(robot.semiaxes_m, dtype=np.float64) * local_normal
    ))
    if support_radius <= 1.0e-12:
        raise ValueError("compact safety-coordinate support radius is degenerate")

    def projection(action: Sequence[float]) -> float:
        if len(action) != 7:
            raise ValueError("compact safety-coordinate action differs")
        displacement = float(translation_scale) * np.asarray(
            action[:3], dtype=np.float64,
        )
        return float(normal @ displacement / support_radius)

    nominal = exact_case["source_nominal_five_action_chunk"]
    effective = candidate["source_executed_actions"]
    if len(nominal) != 5 or len(effective) != 5:
        raise ValueError("compact safety-coordinate chunk differs")
    feature = [stored_slack, projection(nominal[0])]
    feature.extend(projection(action) for action in effective)
    if len(feature) != INPUT_DIMENSION or not all(
        math.isfinite(value) for value in feature
    ):
        raise ValueError("compact safety-coordinate feature is invalid")
    return feature


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"allocation", "result_payload_sha256", "source"}
    }


def candidate_records(
    samples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[float]],
    *,
    primary_rows: Sequence[int],
    physical_rows: Sequence[int],
) -> list[dict[str, Any]]:
    """Aggregate shared row predictions into inference-time candidates."""
    if len(samples) != len(predictions):
        raise ValueError("compact safety-coordinate prediction shape differs")
    primary = tuple(int(row) for row in primary_rows)
    physical = tuple(int(row) for row in physical_rows)
    required = sorted(set(primary) | set(physical))
    if required != list(MODEL_ROWS):
        raise ValueError("compact safety-coordinate aggregate rows differ")
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for sample, prediction in zip(samples, predictions):
        if len(prediction) != 1:
            raise ValueError("compact safety-coordinate prediction is not scalar")
        key = (str(sample["state_id"]), str(sample["candidate_name"]))
        known = bool(sample["known_outcome"])
        entry = grouped.setdefault(key, {
            "state_id": key[0],
            "split": str(sample["split"]),
            "candidate_name": key[1],
            "candidate_order": int(sample["candidate_order"]),
            "correction": float(sample["applied_correction_l2_action"]),
            "known_outcome": known,
            "physical_veto": bool(sample["physical_veto"]),
            "actual_by_row": {},
            "predicted_by_row": {},
        })
        if (
            entry["known_outcome"] != known
            or entry["physical_veto"] != bool(sample["physical_veto"])
        ):
            raise ValueError("compact safety-coordinate candidate metadata differs")
        row = int(sample["row_index"])
        if row in entry["predicted_by_row"]:
            raise ValueError("compact safety-coordinate row repeats")
        entry["predicted_by_row"][row] = float(prediction[0])
        if known:
            entry["actual_by_row"][row] = float(sample["risk"])
    output = []
    for entry in grouped.values():
        if sorted(entry["predicted_by_row"]) != required or (
            entry["known_outcome"]
            and sorted(entry["actual_by_row"]) != required
        ):
            raise ValueError("compact safety-coordinate candidate rows differ")
        predicted_primary = max(
            entry["predicted_by_row"][row] for row in primary
        )
        actual_primary = None
        actual_physical = None
        if entry["known_outcome"]:
            actual_primary = max(entry["actual_by_row"][row] for row in primary)
            actual_physical = max(entry["actual_by_row"][row] for row in physical)
        output.append({
            **entry,
            "predicted_primary": float(predicted_primary),
            "actual_primary": actual_primary,
            "actual_physical": actual_physical,
            "actual_primary_safe": (
                None if actual_primary is None else actual_primary <= 0.0
            ),
            "actual_all_physical_safe": (
                None if actual_physical is None else bool(
                    actual_physical <= 0.0 and not entry["physical_veto"]
                )
            ),
        })
    return sorted(
        output, key=lambda row: (row["state_id"], row["candidate_order"])
    )


def evaluate_selector(
    records: Sequence[Mapping[str, Any]], *, rule: str,
) -> dict[str, Any]:
    """Evaluate an inference-only selector against already stored outcomes."""
    by_state: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_state.setdefault(str(record["state_id"]), []).append(record)
    states = []
    for state_id, rows in sorted(by_state.items()):
        recoverable = [
            row for row in rows if row["actual_primary_safe"] is True
        ]
        if rule == "minimum_predicted_primary_risk":
            selected = min(rows, key=lambda row: (
                float(row["predicted_primary"]), float(row["correction"]),
                int(row["candidate_order"]),
            ))
        elif rule == "predicted_safe_minimum_intervention_else_minimum_risk":
            accepted = [
                row for row in rows if float(row["predicted_primary"]) <= 0.0
            ]
            selected = min(accepted, key=lambda row: (
                float(row["correction"]), int(row["candidate_order"]),
            )) if accepted else min(rows, key=lambda row: (
                float(row["predicted_primary"]), float(row["correction"]),
                int(row["candidate_order"]),
            ))
        elif rule == "nominal":
            nominal = [row for row in rows if row["candidate_name"] == "nominal"]
            if len(nominal) != 1:
                raise ValueError("compact safety-coordinate nominal differs")
            selected = nominal[0]
        else:
            raise ValueError("compact safety-coordinate selector differs")
        failed_rows = []
        if selected["known_outcome"]:
            failed_rows = sorted(
                int(row) for row, risk in selected["actual_by_row"].items()
                if float(risk) > 0.0
            )
        states.append({
            "state_id": state_id,
            "recoverable_primary": bool(recoverable),
            "actual_primary_safe_candidate_count": len(recoverable),
            "selected_candidate": selected["candidate_name"],
            "selected_outcome_known": bool(selected["known_outcome"]),
            "selected_predicted_primary": selected["predicted_primary"],
            "selected_actual_primary_safe": selected["actual_primary_safe"],
            "selected_actual_all_physical_safe": (
                selected["actual_all_physical_safe"]
            ),
            "selected_failed_rows": failed_rows,
            "selected_L6_failure": any(row in (5, 6) for row in failed_rows),
            "selected_correction": selected["correction"],
        })
    recoverable_states = [row for row in states if row["recoverable_primary"]]
    return {
        "rule": rule,
        "state_count": len(states),
        "recoverable_primary_state_count": len(recoverable_states),
        "known_safe_primary_selection_count": sum(
            row["selected_actual_primary_safe"] is True for row in states
        ),
        "known_false_safe_primary_selection_count": sum(
            row["selected_actual_primary_safe"] is False for row in states
        ),
        "known_all_physical_safe_selection_count": sum(
            row["selected_actual_all_physical_safe"] is True for row in states
        ),
        "known_all_physical_unsafe_selection_count": sum(
            row["selected_actual_all_physical_safe"] is False for row in states
        ),
        "unknown_selection_count": sum(
            not row["selected_outcome_known"] for row in states
        ),
        "ignored_L6_failure_count": sum(
            row["selected_L6_failure"] for row in states
        ),
        "safe_primary_support_fraction": (
            1.0 if not recoverable_states else sum(
                row["selected_actual_primary_safe"] is True
                for row in recoverable_states
            ) / len(recoverable_states)
        ),
        "states": states,
    }
