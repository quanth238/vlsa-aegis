"""Frozen split/support audit for the restricted L5 capacity diagnostic."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA = "vlsa_distal_l5_action_risk_root_cause_audit.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_action_risk_root_cause_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    payload.pop("validation_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def feature_support(
    train_features: Sequence[Sequence[float]],
    query_features: Sequence[Sequence[float]],
    train_ids: Sequence[str], query_ids: Sequence[str],
    feature_scale: Sequence[float],
) -> dict[str, Any]:
    """Measure query support relative to the fitted population."""

    import math

    train = [[float(value) for value in row] for row in train_features]
    query = [[float(value) for value in row] for row in query_features]
    scale = [float(value) for value in feature_scale]
    if (
        not train or not query or any(len(row) != 86 for row in train)
        or any(len(row) != 86 for row in query) or len(scale) != 86
        or len(train_ids) != len(train) or len(query_ids) != len(query)
        or any(value <= 0.0 for value in scale)
    ):
        raise ValueError("L5 root-cause feature support shape differs")
    groups = {
        "geometry_context_0_15": (0, 16),
        "nominal_action_16_50": (16, 51),
        "state_context_0_50": (0, 51),
        "candidate_residual_51_85": (51, 86),
        "complete_86D": (0, 86),
    }
    rows = []
    for query_index, (values, state_id) in enumerate(zip(query, query_ids)):
        record: dict[str, Any] = {
            "query_index": int(query_index),
            "state_id": str(state_id),
            "same_state_in_fit": str(state_id) in set(map(str, train_ids)),
        }
        for name, (start, stop) in groups.items():
            distances = [math.sqrt(sum(
                ((candidate[index] - values[index]) / scale[index]) ** 2
                for index in range(start, stop)
            ) / float(stop - start)) for candidate in train]
            nearest = min(range(len(distances)), key=distances.__getitem__)
            record[name] = {
                "nearest_rms_z": float(distances[nearest]),
                "nearest_train_state_id": str(train_ids[nearest]),
            }
        outside = sum(
            value < min(candidate[index] for candidate in train)
            or value > max(candidate[index] for candidate in train)
            for index, value in enumerate(values)
        )
        record["outside_train_range_dimension_count"] = int(outside)
        rows.append(record)

    def aggregate(name: str) -> dict[str, float]:
        values = sorted(float(item[name]["nearest_rms_z"]) for item in rows)
        middle = len(values) // 2
        median = (
            values[middle] if len(values) % 2
            else 0.5 * (values[middle - 1] + values[middle])
        )
        return {
            "minimum_nearest_rms_z": min(values),
            "median_nearest_rms_z": median,
            "maximum_nearest_rms_z": max(values),
        }

    return {
        "sample_count": int(len(rows)),
        "same_state_in_fit_count": int(sum(item["same_state_in_fit"] for item in rows)),
        "groups": {name: aggregate(name) for name in groups},
        "maximum_outside_train_range_dimension_count": max(
            item["outside_train_range_dimension_count"] for item in rows
        ),
        "records": rows,
    }


def root_cause_decision(
    *, train_rmse_m: float, action_holdout_rmse_m: float,
    grouped_validation_rmse_m: float, train_boundary_states: Sequence[int],
    validation_boundary_states: Sequence[int], train_active_witnesses: Sequence[int],
    train_per_row_false_safe: Sequence[int],
) -> dict[str, Any]:
    if any(len(item) != 3 for item in (
        train_boundary_states, validation_boundary_states,
        train_active_witnesses, train_per_row_false_safe,
    )):
        raise ValueError("L5 root-cause row vector differs")
    interpolation_pass = float(action_holdout_rmse_m) <= 0.001
    grouped_gap = float(grouped_validation_rmse_m) >= max(
        10.0 * float(action_holdout_rmse_m), 0.005
    )
    rows_1_2_unsupported = (
        list(validation_boundary_states)[1:] == [0, 0]
        and list(train_active_witnesses)[1:] == [0, 0]
    )
    return {
        "training_fit_submillimetre": float(train_rmse_m) <= 0.001,
        "same_state_action_interpolation_submillimetre": interpolation_pass,
        "grouped_error_at_least_10x_action_holdout": grouped_gap,
        "rows_1_2_missing_validation_boundary_support":
        list(validation_boundary_states)[1:] == [0, 0],
        "rows_1_2_missing_train_active_witnesses":
        list(train_active_witnesses)[1:] == [0, 0],
        "symmetric_fit_retains_per_row_false_safe":
        any(int(value) > 0 for value in train_per_row_false_safe),
        "primary_root_cause": (
            "insufficient_grouped_state_and_active_boundary_coverage"
            if interpolation_pass and grouped_gap and rows_1_2_unsupported
            else "not_isolated"
        ),
        "secondary_root_causes": [
            "86D_representation_omits_recorded_OSC_and_joint_state_unresolved",
            "symmetric_regression_loss_is_not_a_conservative_safety_gate",
        ],
        "model_capacity_primary_failure": False,
        "input_representation_causality_proven": False,
        "generalizable_filter_authorized": False,
    }
