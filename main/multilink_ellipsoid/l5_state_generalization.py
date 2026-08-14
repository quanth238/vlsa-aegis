"""Contracts for the matched L5 action-versus-state generalization audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_state_generalization.v1"
RESULT_SCHEMA = "vlsa_distal_l5_state_generalization_result.v1"


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
        "matched_split", "diagnostic_gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("L5 state-generalization config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("L5 state-generalization schema differs")
    if value["protocol_id"] != "vlsa-distal-l5-state-generalization-v1":
        raise ValueError("L5 state-generalization protocol differs")
    if value["matched_split"] != {
        "fit_episode_split": "train",
        "fit_temporal_profiles": [None, "constant"],
        "test_A_episode_split": "train",
        "test_A_temporal_profile": "front_loaded",
        "test_A_definition": "new_actions_at_known_states",
        "test_B_episode_split": "validation",
        "test_B_temporal_profile": "front_loaded",
        "test_B_definition": "same_action_family_at_new_episode_states",
        "state_grouping": "complete_episode",
    }:
        raise ValueError("L5 state-generalization matched split differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def prediction_diagnostic(
    predictions: Sequence[Sequence[float]],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray([item["risk_l5"] for item in samples], dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[1:] != (3,):
        raise ValueError("L5 state-generalization prediction shape differs")
    predicted_safe = np.max(predicted, axis=1) <= 0.0
    actual_safe = np.max(target, axis=1) <= 0.0
    per_row_false_safe = np.sum(
        (predicted <= 0.0) & (target > 0.0), axis=0
    )
    state_records = []
    for state_id in sorted({str(item["state_id"]) for item in samples}):
        indices = np.asarray([
            index for index, item in enumerate(samples)
            if str(item["state_id"]) == state_id
        ], dtype=np.int64)
        recoverable = bool(np.any(actual_safe[indices]))
        supported = bool(np.any(predicted_safe[indices] & actual_safe[indices]))
        state_records.append({
            "state_id": state_id,
            "candidate_count": int(indices.size),
            "actual_L5_safe_candidate_count": int(np.sum(actual_safe[indices])),
            "predicted_L5_safe_candidate_count": int(
                np.sum(predicted_safe[indices])
            ),
            "L5_false_safe_candidate_count": int(np.sum(
                predicted_safe[indices] & ~actual_safe[indices]
            )),
            "recoverable": recoverable,
            "predicted_safe_support": supported,
        })
    safe_count = int(np.sum(actual_safe))
    near = np.abs(target) <= 0.005
    return {
        "sample_count": len(samples),
        "state_count": len(state_records),
        "rmse_m": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "near_boundary_rmse_m": (
            float(np.sqrt(np.mean((predicted[near] - target[near]) ** 2)))
            if np.any(near) else None
        ),
        "L5_false_safe_candidate_count": int(np.sum(
            predicted_safe & ~actual_safe
        )),
        "per_row_false_safe_count": [int(item) for item in per_row_false_safe],
        "actual_L5_safe_candidate_count": safe_count,
        "L5_safe_candidate_recall": (
            float(np.sum(predicted_safe & actual_safe)) / safe_count
            if safe_count else 0.0
        ),
        "recoverable_state_count": sum(
            item["recoverable"] for item in state_records
        ),
        "supported_recoverable_state_count": sum(
            item["recoverable"] and item["predicted_safe_support"]
            for item in state_records
        ),
        "state_records": state_records,
    }


def prediction_gate(
    metrics: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "zero_observed_L5_false_safe_candidates":
        int(metrics["L5_false_safe_candidate_count"]) == 0,
        "minimum_L5_safe_candidate_recall":
        float(metrics["L5_safe_candidate_recall"])
        >= float(gate["minimum_L5_safe_candidate_recall"]),
        "safe_support_every_recoverable_state":
        int(metrics["supported_recoverable_state_count"])
        == int(metrics["recoverable_state_count"]),
        "contains_recoverable_state": int(metrics["recoverable_state_count"]) > 0,
    }
