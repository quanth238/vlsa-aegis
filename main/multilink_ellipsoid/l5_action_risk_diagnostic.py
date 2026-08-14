"""Contracts for the restricted three-output L5 capacity diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_action_risk_diagnostic.v1"
MODEL_SCHEMA = "vlsa_distal_l5_action_risk_diagnostic_model.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_action_risk_diagnostic_validation.v1"


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
        "schema_version", "protocol_id", "claim_scope", "source",
        "learned_scope", "sample_protocol", "features", "model",
        "reporting", "forbidden",
    }:
        raise ValueError("L5 diagnostic config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-distal-l5-action-risk-diagnostic-v1"
        or value["learned_scope"]["rows"] != [0, 1, 2]
        or value["learned_scope"]["output_count"] != 3
        or value["features"]["input_dimension"] != 86
        or value["model"]["input_dimension"] != 86
        or value["model"]["output_count"] != 3
        or value["model"]["hidden_widths"] != [32, 32]
        or value["reporting"]["no_deployment_pass_gate"] is not True
    ):
        raise ValueError("L5 diagnostic protocol differs")
    expected = {
        "train_fit": 26,
        "action_heldout": 6,
        "grouped_validation": 16,
        "grouped_row_0": 10,
        "diagnostic_positive_control": 6,
    }
    if value["sample_protocol"]["expected_sample_counts"] != expected:
        raise ValueError("L5 diagnostic sample counts differ")
    if value["forbidden"] != [
        "generalizable_L5_safety_filter_claim",
        "row_1_or_row_2_grouped_generalization_claim",
        "reserved_test_label_access",
        "calibration", "QP", "closed_loop_execution", "neural_CBF_claim",
    ]:
        raise ValueError("L5 diagnostic forbidden set differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def split_train_actions(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out the largest registered action order in each nontrivial state."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in samples:
        grouped.setdefault(str(item["state_id"]), []).append(dict(item))
    fit: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for state_id in sorted(grouped):
        ordered = sorted(grouped[state_id], key=lambda item: (
            int(item["candidate_order"]), str(item["candidate_name"])
        ))
        if len(ordered) >= 2:
            heldout.append(ordered[-1])
            fit.extend(ordered[:-1])
        else:
            fit.extend(ordered)
    return fit, heldout


def state_has_row_boundary(
    samples: Sequence[Mapping[str, Any]], *, row: int, near_m: float,
) -> bool:
    values = [float(item["risk_l5"][int(row)]) for item in samples]
    return bool(
        values
        and any(value <= 0.0 for value in values)
        and any(value > 0.0 for value in values)
        and any(abs(value) <= float(near_m) for value in values)
    )


def per_row_metrics(
    predictions: Sequence[Sequence[float]],
    samples: Sequence[Mapping[str, Any]],
    *, near_m: float,
) -> list[dict[str, Any]]:
    import numpy as np

    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray([item["risk_l5"] for item in samples], dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[1:] != (3,):
        raise ValueError("L5 diagnostic prediction shape differs")
    output = []
    for row in range(3):
        prediction = predicted[:, row]
        truth = target[:, row]
        near = np.abs(truth) <= float(near_m)
        false_safe = (prediction <= 0.0) & (truth > 0.0)
        false_unsafe = (prediction > 0.0) & (truth <= 0.0)
        output.append({
            "row": row,
            "sample_count": int(len(truth)),
            "known_safe_count": int(np.sum(truth <= 0.0)),
            "known_unsafe_count": int(np.sum(truth > 0.0)),
            "near_boundary_count": int(np.sum(near)),
            "rmse_m": float(np.sqrt(np.mean((prediction - truth) ** 2))),
            "near_boundary_rmse_m": (
                float(np.sqrt(np.mean((prediction[near] - truth[near]) ** 2)))
                if np.any(near) else None
            ),
            "boundary_side_accuracy": float(np.mean(
                (prediction <= 0.0) == (truth <= 0.0)
            )),
            "false_safe_count": int(np.sum(false_safe)),
            "false_unsafe_count": int(np.sum(false_unsafe)),
            "maximum_optimistic_risk_underprediction_m": float(np.max(
                truth - prediction
            )),
            "mean_signed_error_m": float(np.mean(prediction - truth)),
        })
    return output


def report_metrics(
    predictions: Sequence[Sequence[float]],
    samples: Sequence[Mapping[str, Any]],
    *, near_m: float,
) -> dict[str, Any]:
    import numpy as np

    from .l5_action_risk import prediction_metrics

    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray([item["risk_l5"] for item in samples], dtype=np.float64)
    conjunction = prediction_metrics(predicted, samples)
    conjunction["per_row"] = per_row_metrics(
        predicted, samples, near_m=float(near_m)
    )
    conjunction["target_row_counts"] = [
        sum(item.get("target_row") == row for item in samples)
        for row in range(3)
    ]
    conjunction["zero_baseline_rmse_m"] = float(np.sqrt(np.mean(target ** 2)))
    return conjunction
