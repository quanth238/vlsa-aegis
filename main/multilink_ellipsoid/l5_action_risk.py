"""Contracts for the prediction-only three-output L5 feasibility pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_action_risk_mlp.v1"
AUDIT_SCHEMA = "vlsa_distal_l5_action_risk_dataset_audit.v1"
MODEL_SCHEMA = "vlsa_distal_l5_action_risk_model.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "learned_scope", "dataset_gate", "features", "model",
        "prediction_gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("L5 action-risk config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("L5 action-risk schema differs")
    if value["protocol_id"] != "vlsa-distal-l5-action-risk-mlp-v1":
        raise ValueError("L5 action-risk protocol differs")
    if (
        value["learned_scope"]["rows"] != [0, 1, 2]
        or value["learned_scope"]["output_count"] != 3
    ):
        raise ValueError("L5 action-risk learned scope differs")
    if value["features"]["input_dimension"] != 86:
        raise ValueError("L5 action-risk feature dimension differs")
    if (
        value["model"]["input_dimension"] != 86
        or value["model"]["output_count"] != 3
    ):
        raise ValueError("L5 action-risk model shape differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def feature_vector(
    *, initial_clearance: Sequence[float], local_frame: Mapping[str, Sequence[float]],
    nominal_actions: Sequence[Sequence[float]], candidate_actions: Sequence[Sequence[float]],
) -> list[float]:
    import numpy as np

    clearance = np.asarray(initial_clearance, dtype=np.float64)
    nominal = np.asarray(nominal_actions, dtype=np.float64)
    candidate = np.asarray(candidate_actions, dtype=np.float64)
    frame = np.concatenate([
        np.asarray(local_frame[name], dtype=np.float64)
        for name in ("normal", "tangent_up", "tangent_side")
    ])
    if clearance.shape != (7,) or frame.shape != (9,):
        raise ValueError("L5 action-risk state feature shape differs")
    if nominal.shape != (5, 7) or candidate.shape != nominal.shape:
        raise ValueError("L5 action-risk action feature shape differs")
    output = np.concatenate([
        clearance, frame, nominal.reshape(-1), (candidate - nominal).reshape(-1)
    ])
    if output.shape != (86,) or not np.all(np.isfinite(output)):
        raise ValueError("L5 action-risk feature differs")
    return output.tolist()


def prediction_metrics(
    predictions: Sequence[Sequence[float]], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    import numpy as np

    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray([item["risk_l5"] for item in samples], dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("L5 action-risk prediction shape differs")
    predicted_safe = np.max(predicted, axis=1) <= 0.0
    actual_l5_safe = np.max(target, axis=1) <= 0.0
    exact_safe = np.asarray([bool(item["exact_safe"]) for item in samples])
    false_safe = predicted_safe & ~actual_l5_safe
    exact_safe_recalled = predicted_safe & exact_safe
    state_ids = sorted({str(item["state_id"]) for item in samples})
    state_records = []
    for state_id in state_ids:
        indices = np.asarray([
            index for index, item in enumerate(samples)
            if str(item["state_id"]) == state_id
        ], dtype=np.int64)
        recoverable = bool(np.any(exact_safe[indices]))
        supported = bool(np.any(exact_safe_recalled[indices]))
        physical_veto_count = int(np.sum(predicted_safe[indices] & ~exact_safe[indices]))
        state_records.append({
            "state_id": state_id,
            "recoverable": recoverable,
            "predicted_safe_exact_safe_support": supported,
            "predicted_L5_safe_but_all_seven_physical_veto_count": physical_veto_count,
        })
    safe_count = int(np.sum(exact_safe))
    return {
        "sample_count": len(samples),
        "rmse_m": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "near_boundary_rmse_m": float(np.sqrt(np.mean(
            (predicted[np.abs(target) <= 0.005] - target[np.abs(target) <= 0.005]) ** 2
        ))) if np.any(np.abs(target) <= 0.005) else None,
        "L5_false_safe_count": int(np.sum(false_safe)),
        "exact_safe_candidate_count": safe_count,
        "exact_safe_candidate_recall": (
            float(np.sum(exact_safe_recalled)) / safe_count if safe_count else 0.0
        ),
        "recoverable_state_count": sum(item["recoverable"] for item in state_records),
        "supported_recoverable_state_count": sum(
            item["recoverable"] and item["predicted_safe_exact_safe_support"]
            for item in state_records
        ),
        "predicted_L5_safe_but_all_seven_physical_veto_count": int(np.sum(
            predicted_safe & ~exact_safe
        )),
        "state_records": state_records,
    }
