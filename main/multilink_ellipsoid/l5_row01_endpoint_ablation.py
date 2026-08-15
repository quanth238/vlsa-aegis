"""Contracts for the six-dimensional EE start/end L5 risk ablation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_row01_endpoint_ablation.v1"
RESULT_SCHEMA = "vlsa_distal_l5_row01_endpoint_ablation_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_row01_endpoint_ablation_validation.v1"
ENDPOINT_DIMENSION = 6


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
        "schema_version", "protocol_id", "claim_scope", "source", "endpoint",
        "matched_model", "decision", "forbidden",
    }:
        raise ValueError("L5 row01 endpoint config keys differ")
    endpoint = value["endpoint"]
    model = value["matched_model"]
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-l5-row01-endpoint-ablation-v1"
        or endpoint["input_dimension"] != ENDPOINT_DIMENSION
        or endpoint["translation_scale_m_per_action_unit"] != 0.05
        or model["input_dimension"] != ENDPOINT_DIMENSION
        or model["hidden_widths"] != [32, 32]
        or model["seed"] != 20260814
    ):
        raise ValueError("L5 row01 endpoint protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def endpoint_feature_vector(
    *, eef_position_m: Sequence[float], candidate_actions: Any,
    translation_scale_m_per_action_unit: float,
) -> list[float]:
    start = [float(item) for item in eef_position_m]
    actions = [[float(item) for item in row] for row in candidate_actions]
    if len(start) != 3 or len(actions) != 5 or any(
        len(row) != 7 for row in actions
    ):
        raise ValueError("endpoint source shape differs")
    if not all(math.isfinite(item) for item in start) or not all(
        math.isfinite(item) for row in actions for item in row
    ):
        raise ValueError("endpoint source is nonfinite")
    scale = float(translation_scale_m_per_action_unit)
    end = [
        start[index] + scale * sum(row[index] for row in actions)
        for index in range(3)
    ]
    output = start + end
    if len(output) != ENDPOINT_DIMENSION or not all(
        math.isfinite(item) for item in output
    ):
        raise ValueError("endpoint feature differs")
    return output


def load_endpoint_bundle(
    torch: Any, payload: Mapping[str, Any], model_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_selection import build_model

    model = build_model(
        torch, ENDPOINT_DIMENSION, model_config["hidden_widths"]
    )
    template = model.state_dict()
    if set(payload["state_dict"]) != set(template):
        raise ValueError("endpoint frozen state keys differ")
    state = {}
    for name, value in template.items():
        raw = torch.as_tensor(payload["state_dict"][name], dtype=value.dtype)
        if raw.shape != value.shape:
            raise ValueError("endpoint frozen parameter shape differs")
        state[name] = raw
    model.load_state_dict(state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output = {
        "model": model,
        "device": device,
        "feature_mean": np.asarray(payload["feature_mean"], dtype=np.float64),
        "feature_scale": np.asarray(payload["feature_scale"], dtype=np.float64),
        "target_mean": np.asarray(payload["target_mean"], dtype=np.float64),
        "target_scale": np.asarray(payload["target_scale"], dtype=np.float64),
    }
    if (
        output["feature_mean"].shape != (ENDPOINT_DIMENSION,)
        or output["feature_scale"].shape != (ENDPOINT_DIMENSION,)
        or output["target_mean"].shape != (2,)
        or output["target_scale"].shape != (2,)
    ):
        raise ValueError("endpoint frozen normalization shape differs")
    return output


def classify(
    endpoint: Mapping[str, Any], complete: Mapping[str, Any],
    *, minimum_relative_rmse_improvement: float,
) -> tuple[str, dict[str, Any]]:
    endpoint_rmse = float(endpoint["rmse_m"])
    complete_rmse = float(complete["rmse_m"])
    relative = (complete_rmse - endpoint_rmse) / max(complete_rmse, 1.0e-12)
    comparison = {
        "endpoint_minus_complete_validation_RMSE_m": endpoint_rmse - complete_rmse,
        "relative_validation_RMSE_improvement_over_complete": relative,
        "endpoint_minus_complete_false_safe_count": int(
            endpoint["row01_false_safe_count"]
        ) - int(complete["row01_false_safe_count"]),
        "endpoint_minus_complete_supported_state_count": int(
            endpoint["supported_recoverable_state_count"]
        ) - int(complete["supported_recoverable_state_count"]),
    }
    promising = (
        relative >= float(minimum_relative_rmse_improvement)
        and int(endpoint["row01_false_safe_count"])
        < int(complete["row01_false_safe_count"])
        and int(endpoint["supported_recoverable_state_count"])
        >= int(complete["supported_recoverable_state_count"])
    )
    return (
        "endpoint_only_materially_improves_transfer_add_inputs_only_if_needed"
        if promising else
        "endpoint_only_insufficient_add_one_physical_context_group_next"
    ), comparison
