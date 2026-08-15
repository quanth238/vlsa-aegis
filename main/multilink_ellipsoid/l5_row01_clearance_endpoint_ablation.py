"""Contracts for the 12D relative endpoint plus current L5 clearance ablation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_row01_clearance_endpoint_ablation.v1"
RESULT_SCHEMA = "vlsa_distal_l5_row01_clearance_endpoint_ablation_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_row01_clearance_endpoint_ablation_validation.v1"
INPUT_DIMENSION = 12


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


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
        raise ValueError("clearance-endpoint config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-l5-row01-clearance-endpoint-ablation-v1"
        or value["features"]["input_dimension"] != INPUT_DIMENSION
        or value["features"]["translation_scale_m_per_action_unit"] != 0.05
        or value["matched_model"]["input_dimension"] != INPUT_DIMENSION
        or value["matched_model"]["hidden_widths"] != [32, 32]
        or value["matched_model"]["seed"] != 20260814
    ):
        raise ValueError("clearance-endpoint protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def clearance_endpoint_feature_vector(
    *, obstacle_endpoint_feature: Sequence[float], current_L5_clearances_m: Sequence[float],
) -> list[float]:
    base = [float(item) for item in obstacle_endpoint_feature]
    clearance = [float(item) for item in current_L5_clearances_m]
    if len(base) != 9 or len(clearance) != 3 or not all(
        math.isfinite(item) for item in base + clearance
    ):
        raise ValueError("clearance-endpoint feature source differs")
    return base + clearance


def load_bundle(torch: Any, payload: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.l5_row01_selection import build_model

    model = build_model(torch, INPUT_DIMENSION, model_config["hidden_widths"])
    template = model.state_dict()
    if set(payload["state_dict"]) != set(template):
        raise ValueError("clearance-endpoint state keys differ")
    state = {}
    for name, value in template.items():
        raw = torch.as_tensor(payload["state_dict"][name], dtype=value.dtype)
        if raw.shape != value.shape:
            raise ValueError("clearance-endpoint parameter shape differs")
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
        raise ValueError("clearance-endpoint normalization shape differs")
    return output


def classify(current: Mapping[str, Any], prior: Mapping[str, Any], *, minimum_relative_rmse_improvement: float) -> tuple[str, dict[str, Any]]:
    current_rmse = float(current["rmse_m"])
    prior_rmse = float(prior["rmse_m"])
    relative = (prior_rmse - current_rmse) / max(prior_rmse, 1.0e-12)
    comparison = {
        "clearance_minus_9D_validation_RMSE_m": current_rmse - prior_rmse,
        "relative_validation_RMSE_improvement_over_9D": relative,
        "clearance_minus_9D_false_safe_count": int(current["row01_false_safe_count"]) - int(prior["row01_false_safe_count"]),
        "clearance_minus_9D_supported_state_count": int(current["supported_recoverable_state_count"]) - int(prior["supported_recoverable_state_count"]),
        "clearance_minus_9D_exact_safe_selection_count": int(current["selected_exact_safe_recoverable_state_count"]) - int(prior["selected_exact_safe_recoverable_state_count"]),
    }
    useful = (
        relative >= float(minimum_relative_rmse_improvement)
        and int(current["row01_false_safe_count"]) <= int(prior["row01_false_safe_count"])
        and int(current["supported_recoverable_state_count"]) >= int(prior["supported_recoverable_state_count"])
    )
    return (
        "current_L5_clearances_are_useful_continue_minimal_ablation"
        if useful else
        "current_L5_clearances_insufficient_stop_and_review_before_more_inputs"
    ), comparison
