"""Contracts for obstacle-relative EE endpoint L5 risk ablation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_row01_obstacle_endpoint_ablation.v1"
RESULT_SCHEMA = "vlsa_distal_l5_row01_obstacle_endpoint_ablation_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_row01_obstacle_endpoint_ablation_validation.v1"
INPUT_DIMENSION = 9


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    payload.pop("validation_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "protocol_id", "claim_scope", "source", "features",
        "matched_model", "decision", "forbidden",
    }:
        raise ValueError("obstacle-endpoint config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-distal-l5-row01-obstacle-endpoint-ablation-v1"
        or value["features"]["input_dimension"] != INPUT_DIMENSION
        or value["features"]["translation_scale_m_per_action_unit"] != 0.05
        or value["matched_model"]["input_dimension"] != INPUT_DIMENSION
        or value["matched_model"]["hidden_widths"] != [32, 32]
        or value["matched_model"]["seed"] != 20260814
    ):
        raise ValueError("obstacle-endpoint protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def obstacle_endpoint_feature_vector(
    *, eef_position_m: Sequence[float], candidate_actions: Any,
    obstacle_center_m: Sequence[float], obstacle_semiaxes_m: Sequence[float],
    translation_scale_m_per_action_unit: float,
) -> list[float]:
    start = [float(item) for item in eef_position_m]
    center = [float(item) for item in obstacle_center_m]
    semiaxes = [float(item) for item in obstacle_semiaxes_m]
    actions = [[float(item) for item in row] for row in candidate_actions]
    if any(len(item) != 3 for item in (start, center, semiaxes)) or (
        len(actions) != 5 or any(len(row) != 7 for row in actions)
    ):
        raise ValueError("obstacle-endpoint source shape differs")
    raw = start + center + semiaxes + [item for row in actions for item in row]
    if not all(math.isfinite(item) for item in raw) or any(
        item <= 0.0 for item in semiaxes
    ):
        raise ValueError("obstacle-endpoint source differs")
    scale = float(translation_scale_m_per_action_unit)
    end = [
        start[index] + scale * sum(row[index] for row in actions)
        for index in range(3)
    ]
    output = (
        [start[index] - center[index] for index in range(3)]
        + [end[index] - center[index] for index in range(3)]
        + semiaxes
    )
    if len(output) != INPUT_DIMENSION or not all(
        math.isfinite(item) for item in output
    ):
        raise ValueError("obstacle-endpoint feature differs")
    return output


def load_bundle(
    torch: Any, payload: Mapping[str, Any], model_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.l5_row01_selection import build_model

    model = build_model(torch, INPUT_DIMENSION, model_config["hidden_widths"])
    template = model.state_dict()
    if set(payload["state_dict"]) != set(template):
        raise ValueError("obstacle-endpoint state keys differ")
    state = {}
    for name, value in template.items():
        raw = torch.as_tensor(payload["state_dict"][name], dtype=value.dtype)
        if raw.shape != value.shape:
            raise ValueError("obstacle-endpoint parameter shape differs")
        state[name] = raw
    model.load_state_dict(state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output = {
        "model": model, "device": device,
        "feature_mean": np.asarray(payload["feature_mean"], dtype=np.float64),
        "feature_scale": np.asarray(payload["feature_scale"], dtype=np.float64),
        "target_mean": np.asarray(payload["target_mean"], dtype=np.float64),
        "target_scale": np.asarray(payload["target_scale"], dtype=np.float64),
    }
    if output["feature_mean"].shape != (INPUT_DIMENSION,):
        raise ValueError("obstacle-endpoint normalization shape differs")
    return output


def classify(
    current: Mapping[str, Any], endpoint: Mapping[str, Any],
    *, minimum_relative_rmse_improvement: float,
) -> tuple[str, dict[str, Any]]:
    current_rmse = float(current["rmse_m"])
    endpoint_rmse = float(endpoint["rmse_m"])
    relative = (endpoint_rmse - current_rmse) / max(endpoint_rmse, 1.0e-12)
    comparison = {
        "obstacle_relative_minus_endpoint_validation_RMSE_m":
        current_rmse - endpoint_rmse,
        "relative_validation_RMSE_improvement_over_endpoint": relative,
        "obstacle_relative_minus_endpoint_false_safe_count": int(
            current["row01_false_safe_count"]
        ) - int(endpoint["row01_false_safe_count"]),
        "obstacle_relative_minus_endpoint_supported_state_count": int(
            current["supported_recoverable_state_count"]
        ) - int(endpoint["supported_recoverable_state_count"]),
    }
    promising = (
        relative >= float(minimum_relative_rmse_improvement)
        and int(current["row01_false_safe_count"])
        <= int(endpoint["row01_false_safe_count"])
        and int(current["supported_recoverable_state_count"])
        >= int(endpoint["supported_recoverable_state_count"])
    )
    return (
        "obstacle_relative_endpoint_group_is_useful_continue_minimal_ablation"
        if promising else
        "obstacle_relative_endpoint_group_insufficient_add_current_L5_clearances_next"
    ), comparison
