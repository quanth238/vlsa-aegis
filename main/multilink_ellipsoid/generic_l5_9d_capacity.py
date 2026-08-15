"""Contracts for the generic L5 9D future-risk capacity diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_generic_l5_9d_capacity.v1"
RESULT_SCHEMA = "vlsa_distal_generic_l5_9d_capacity_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_generic_l5_9d_capacity_validation.v1"
INPUT_DIMENSION = 9
OUTPUT_COUNT = 3


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "protocol_id", "claim_scope", "sources", "dataset",
        "features", "model", "metrics", "decision", "forbidden",
    }:
        raise ValueError("generic L5 9D config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-generic-l5-9d-capacity-v1"
        or value["features"]["input_dimension"] != INPUT_DIMENSION
        or value["features"]["translation_scale_m_per_action_unit"] != 0.05
        or value["model"]["input_dimension"] != INPUT_DIMENSION
        or value["model"]["output_count"] != OUTPUT_COUNT
        or value["model"]["hidden_widths"] != [32, 32]
        or value["model"]["seed"] != 20260814
        or value["dataset"]["L5_row_indices"] != [1, 2, 3]
        or len(value["dataset"]["prevention_case_ids"]) != 4
    ):
        raise ValueError("generic L5 9D protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def feature_vector(
    *, eef_position_m: Sequence[float], candidate_actions: Sequence[Sequence[float]],
    obstacle_center_m: Sequence[float], obstacle_semiaxes_m: Sequence[float],
    translation_scale_m_per_action_unit: float,
) -> list[float]:
    start = [float(item) for item in eef_position_m]
    center = [float(item) for item in obstacle_center_m]
    semiaxes = [float(item) for item in obstacle_semiaxes_m]
    actions = [[float(item) for item in row] for row in candidate_actions]
    raw = start + center + semiaxes + [item for row in actions for item in row]
    if (
        len(start) != 3 or len(center) != 3 or len(semiaxes) != 3
        or len(actions) != 5 or any(len(row) != 7 for row in actions)
        or any(item <= 0.0 for item in semiaxes)
        or not all(math.isfinite(item) for item in raw)
    ):
        raise ValueError("generic L5 9D feature source differs")
    scale = float(translation_scale_m_per_action_unit)
    end = [
        start[index] + scale * sum(row[index] for row in actions)
        for index in range(3)
    ]
    feature = (
        [start[index] - center[index] for index in range(3)]
        + [end[index] - center[index] for index in range(3)]
        + semiaxes
    )
    if len(feature) != INPUT_DIMENSION or not all(math.isfinite(x) for x in feature):
        raise ValueError("generic L5 9D feature differs")
    return feature


def build_model(torch: Any, hidden_widths: Sequence[int]) -> Any:
    widths = [INPUT_DIMENSION, *[int(item) for item in hidden_widths], OUTPUT_COUNT]
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


def state_balanced_weights(state_ids: Sequence[str]) -> list[float]:
    counts = {state_id: state_ids.count(state_id) for state_id in set(state_ids)}
    weights = [1.0 / float(counts[state_id]) for state_id in state_ids]
    total = sum(weights)
    return [weight * len(weights) / total for weight in weights]


def weighted_mean_scale(
    values: Any, weights: Any, minimum_scale: float, *, fallback_scale: float,
) -> tuple[Any, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    raw_weights = np.asarray(weights, dtype=np.float64)
    normalized = raw_weights / np.sum(raw_weights)
    mean = np.sum(array * normalized[:, None], axis=0)
    variance = np.sum((array - mean) ** 2 * normalized[:, None], axis=0)
    scale = np.where(
        np.sqrt(variance) >= minimum_scale, np.sqrt(variance), float(fallback_scale),
    )
    return mean, scale


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[cursor]]):
            end += 1
        rank = 0.5 * (cursor + end - 1)
        for index in order[cursor:end]:
            output[index] = rank
        cursor = end
    return output


def spearman(actual: Sequence[float], predicted: Sequence[float]) -> float | None:
    import numpy as np

    if len(actual) < 2:
        return None
    left = np.asarray(rankdata(actual), dtype=np.float64)
    right = np.asarray(rankdata(predicted), dtype=np.float64)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def diagnostic_metrics(
    samples: Sequence[Mapping[str, Any]], predictions: Sequence[Sequence[float]],
    *, near_boundary_abs_risk: float,
) -> dict[str, Any]:
    import numpy as np

    target = np.asarray([sample["risk_rows"] for sample in samples], dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if target.shape != predicted.shape or target.shape[1:] != (OUTPUT_COUNT,):
        raise ValueError("generic L5 9D prediction shape differs")
    actual_global = np.max(target, axis=1)
    predicted_global = np.max(predicted, axis=1)
    actual_safe = actual_global <= 0.0
    predicted_safe = predicted_global <= 0.0
    near = np.abs(actual_global) <= float(near_boundary_abs_risk)
    states = []
    direction_correct = direction_total = 0
    ordering_correct = ordering_total = 0
    for state_id in sorted({str(sample["state_id"]) for sample in samples}):
        indices = [
            index for index, sample in enumerate(samples)
            if str(sample["state_id"]) == state_id
        ]
        nominal = next(
            (index for index in indices if samples[index]["candidate_name"] == "nominal"),
            None,
        )
        if nominal is not None:
            for index in indices:
                if index == nominal:
                    continue
                actual_delta = actual_global[index] - actual_global[nominal]
                predicted_delta = predicted_global[index] - predicted_global[nominal]
                if abs(actual_delta) > 1.0e-12:
                    direction_total += 1
                    direction_correct += int(actual_delta * predicted_delta > 0.0)
        safe_indices = [index for index in indices if actual_safe[index]]
        unsafe_indices = [index for index in indices if not actual_safe[index]]
        for safe_index in safe_indices:
            for unsafe_index in unsafe_indices:
                ordering_total += 1
                ordering_correct += int(
                    predicted_global[safe_index] < predicted_global[unsafe_index]
                )
        accepted = [index for index in indices if predicted_safe[index]]
        selected = None if not accepted else min(accepted, key=lambda index: (
            float(samples[index]["applied_correction_l2_action"]),
            int(samples[index]["candidate_order"]),
        ))
        oracle = min(indices, key=lambda index: (
            float(actual_global[index]), int(samples[index]["candidate_order"]),
        ))
        states.append({
            "state_id": state_id,
            "known_candidate_count": len(indices),
            "actual_safe_candidate_count": int(np.sum(actual_safe[indices])),
            "predicted_safe_candidate_count": int(np.sum(predicted_safe[indices])),
            "predicted_safe_support": bool(accepted),
            "selected_candidate": None if selected is None else samples[selected]["candidate_name"],
            "selected_actual_safe": None if selected is None else bool(actual_safe[selected]),
            "selected_actual_global_risk": None if selected is None else float(actual_global[selected]),
            "lowest_predicted_candidate": samples[int(indices[int(np.argmin(predicted_global[indices]))])]["candidate_name"],
            "oracle_lowest_risk_candidate": samples[oracle]["candidate_name"],
            "oracle_lowest_global_risk": float(actual_global[oracle]),
            "rank_spearman": spearman(
                [actual_global[index] for index in indices],
                [predicted_global[index] for index in indices],
            ),
        })
    row_rmse = np.sqrt(np.mean((predicted - target) ** 2, axis=0))
    return {
        "sample_count": len(samples),
        "state_count": len(states),
        "row_RMSE": row_rmse.tolist(),
        "global_RMSE": float(np.sqrt(np.mean((predicted_global - actual_global) ** 2))),
        "global_MAE": float(np.mean(np.abs(predicted_global - actual_global))),
        "near_boundary_count": int(np.sum(near)),
        "near_boundary_global_RMSE": None if not np.any(near) else float(
            np.sqrt(np.mean((predicted_global[near] - actual_global[near]) ** 2))
        ),
        "false_safe_count": int(np.sum(predicted_safe & ~actual_safe)),
        "false_unsafe_count": int(np.sum(~predicted_safe & actual_safe)),
        "actual_safe_count": int(np.sum(actual_safe)),
        "predicted_safe_count": int(np.sum(predicted_safe)),
        "safe_recall": None if not np.any(actual_safe) else float(
            np.sum(predicted_safe & actual_safe) / np.sum(actual_safe)
        ),
        "global_rank_spearman": spearman(actual_global.tolist(), predicted_global.tolist()),
        "improvement_direction_correct": direction_correct,
        "improvement_direction_total": direction_total,
        "improvement_direction_accuracy": None if not direction_total else float(
            direction_correct / direction_total
        ),
        "safe_unsafe_ordering_correct": ordering_correct,
        "safe_unsafe_ordering_total": ordering_total,
        "safe_unsafe_ordering_accuracy": None if not ordering_total else float(
            ordering_correct / ordering_total
        ),
        "supported_state_count": sum(item["predicted_safe_support"] for item in states),
        "selected_exact_safe_state_count": sum(
            item["selected_actual_safe"] is True for item in states
        ),
        "states": states,
    }
