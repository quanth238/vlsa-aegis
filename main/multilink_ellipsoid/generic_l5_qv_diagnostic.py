"""Contracts for the matched direct-L5 Q-only versus Q-plus-V diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_generic_l5_qv_diagnostic.v1"
RESULT_SCHEMA = "vlsa_distal_generic_l5_qv_diagnostic_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_generic_l5_qv_diagnostic_validation.v1"
STATE_DIMENSION = 15
ACTION_DIMENSION = 3


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
        raise ValueError("generic L5 Q/V diagnostic config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-generic-l5-qv-diagnostic-v1"
        or value["features"]["state_dimension"] != STATE_DIMENSION
        or value["features"]["action_dimension"] != ACTION_DIMENSION
        or value["features"]["translation_scale_m_per_action_unit"] != 0.05
        or value["dataset"]["L5_row_indices"] != [1, 2, 3]
        or len(value["dataset"]["boundary_case_ids"]) != 3
        or len(value["dataset"]["safe_auxiliary_case_ids"]) != 1
        or len(value["dataset"]["recovery_case_ids"]) != 1
        or value["model"]["hidden_width"] != 32
        or value["model"]["seed"] != 20260816
        or value["model"]["arms"] != ["q_only", "q_plus_v"]
    ):
        raise ValueError("generic L5 Q/V diagnostic protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _obstacle_aabb(boxes: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    lower = [math.inf, math.inf, math.inf]
    upper = [-math.inf, -math.inf, -math.inf]
    if not boxes:
        raise ValueError("direct-L5 context has no compiled obstacle boxes")
    for box in boxes:
        center = [float(item) for item in box["center_m"]]
        rotation = [[float(item) for item in row] for row in box["rotation"]]
        half = [float(item) for item in box["half_extents_m"]]
        if len(center) != 3 or len(rotation) != 3 or any(len(row) != 3 for row in rotation) or len(half) != 3:
            raise ValueError("direct-L5 compiled obstacle box differs")
        world_half = [
            sum(abs(rotation[row][column]) * half[column] for column in range(3))
            for row in range(3)
        ]
        lower = [min(lower[index], center[index] - world_half[index]) for index in range(3)]
        upper = [max(upper[index], center[index] + world_half[index]) for index in range(3)]
    center = [0.5 * (lower[index] + upper[index]) for index in range(3)]
    half = [0.5 * (upper[index] - lower[index]) for index in range(3)]
    if not all(math.isfinite(item) for item in center) or not all(item > 0.0 for item in half):
        raise ValueError("direct-L5 obstacle AABB differs")
    return center, half


def direct_l5_state_feature(
    boundary: Mapping[str, Any], *, row_indices: Sequence[int],
) -> list[float]:
    """Return a compact state feature tied directly to current L5 geometry.

    The closest current L5 primitive supplies the relative center and shape.
    All three current L5 slacks and a one-hot active-row identity preserve
    witness-switch information.  The feature deliberately excludes future
    executed state and candidate action.
    """

    rows = boundary["exact_robot_rows"]
    slacks = [float(item) for item in boundary["row_normalized_radial_slack"]]
    indexes = [int(item) for item in row_indices]
    if indexes != [1, 2, 3] or len(rows) <= max(indexes) or len(slacks) <= max(indexes):
        raise ValueError("direct-L5 row binding differs")
    if any(rows[index]["body_name"] != "robot0_link5" for index in indexes):
        raise ValueError("direct-L5 body binding differs")
    active_local = min(range(3), key=lambda local: slacks[indexes[local]])
    active = rows[indexes[active_local]]
    obstacle_center, obstacle_half = _obstacle_aabb(
        boundary["compiled_obstacle_boxes"],
    )
    center = [float(item) for item in active["center_m"]]
    semiaxes = [float(item) for item in active["semiaxes_m"]]
    feature = (
        [center[index] - obstacle_center[index] for index in range(3)]
        + semiaxes
        + obstacle_half
        + [slacks[index] for index in indexes]
        + [1.0 if index == active_local else 0.0 for index in range(3)]
    )
    if len(feature) != STATE_DIMENSION or not all(math.isfinite(item) for item in feature):
        raise ValueError("direct-L5 state feature differs")
    return feature


def action_endpoint_feature(
    actions: Sequence[Sequence[float]], *, translation_scale_m_per_action_unit: float,
) -> list[float]:
    parsed = [[float(item) for item in row] for row in actions]
    if len(parsed) != 5 or any(len(row) != 7 for row in parsed):
        raise ValueError("direct-L5 action chunk differs")
    scale = float(translation_scale_m_per_action_unit)
    feature = [scale * sum(row[index] for row in parsed) for index in range(3)]
    if not all(math.isfinite(item) for item in feature):
        raise ValueError("direct-L5 action endpoint differs")
    return feature


def build_model(torch: Any, hidden_width: int) -> Any:
    width = int(hidden_width)

    class SharedQV(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = torch.nn.Sequential(
                torch.nn.Linear(STATE_DIMENSION, width), torch.nn.SiLU(),
                torch.nn.Linear(width, width), torch.nn.SiLU(),
            )
            self.q_head = torch.nn.Sequential(
                torch.nn.Linear(width + ACTION_DIMENSION, width), torch.nn.SiLU(),
                torch.nn.Linear(width, 1),
            )
            self.v_head = torch.nn.Sequential(
                torch.nn.Linear(width, width), torch.nn.SiLU(),
                torch.nn.Linear(width, 1),
            )

        def encode(self, state: Any) -> Any:
            return self.state_encoder(state)

        def q(self, state: Any, action: Any) -> Any:
            return self.q_head(torch.cat((self.encode(state), action), dim=-1))

        def v(self, state: Any) -> Any:
            return self.v_head(self.encode(state))

    return SharedQV()


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


def diagnostic_metrics(samples: Sequence[Mapping[str, Any]], predictions: Sequence[float]) -> dict[str, Any]:
    actual = [float(item["target"]) for item in samples]
    predicted = [float(item) for item in predictions]
    if len(actual) != len(predicted) or not actual:
        raise ValueError("generic L5 Q/V metric input differs")
    safe = [item <= 0.0 for item in actual]
    predicted_safe = [item <= 0.0 for item in predicted]
    by_state = {}
    for state_id in sorted({str(item["state_id"]) for item in samples}):
        indexes = [index for index, item in enumerate(samples) if item["state_id"] == state_id]
        state_safe = [safe[index] for index in indexes]
        state_predicted_safe = [predicted_safe[index] for index in indexes]
        by_state[state_id] = {
            "sample_count": len(indexes),
            "exact_safe_count": sum(state_safe),
            "predicted_safe_count": sum(state_predicted_safe),
            "false_safe_count": sum(
                proposed and not truth
                for proposed, truth in zip(state_predicted_safe, state_safe)
            ),
            "safe_support": any(
                truth and proposed
                for truth, proposed in zip(state_safe, state_predicted_safe)
            ),
            "mean_signed_error": sum(
                predicted[index] - actual[index] for index in indexes
            ) / len(indexes),
        }
    squared = [(guess - truth) ** 2 for guess, truth in zip(predicted, actual)]
    absolute = [abs(guess - truth) for guess, truth in zip(predicted, actual)]
    exact_safe_count = sum(safe)
    return {
        "sample_count": len(actual),
        "rmse": math.sqrt(sum(squared) / len(squared)),
        "mae": sum(absolute) / len(absolute),
        "mean_signed_error": sum(
            guess - truth for guess, truth in zip(predicted, actual)
        ) / len(actual),
        "false_safe_count": sum(
            proposed and not truth for proposed, truth in zip(predicted_safe, safe)
        ),
        "false_unsafe_count": sum(
            not proposed and truth for proposed, truth in zip(predicted_safe, safe)
        ),
        "exact_safe_count": exact_safe_count,
        "predicted_safe_count": sum(predicted_safe),
        "safe_recall": None if not exact_safe_count else sum(
            proposed and truth for proposed, truth in zip(predicted_safe, safe)
        ) / exact_safe_count,
        "safe_support_state_count": sum(bool(item["safe_support"]) for item in by_state.values()),
        "state_count": len(by_state),
        "per_state": by_state,
    }
