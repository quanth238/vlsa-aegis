"""Frozen compact-selector inference ablations over immutable candidates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_compact_selector_ablation.v1"
RESULT_SCHEMA = "vlsa_distal_compact_selector_ablation_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_compact_selector_ablation_validation.v1"
PRIMARY_GROUP_ROWS = {
    "end_effector": (0,),
    "palm": (1,),
    "L5": (2, 3, 4),
}
AUDIT_GROUP_ROWS = {**PRIMARY_GROUP_ROWS, "L6": (5, 6)}


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
    expected = {
        "schema_version", "protocol_id", "claim_scope", "compact_config",
        "compact_result", "L5_specialist_result", "margin", "arms",
        "evaluation", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("compact selector ablation config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-distal-compact-selector-ablation-v1"
        or value["margin"] != {
            "fit_split": "validation",
            "labels": "known_only_UNKNOWN_censored",
            "definition": "max(0,max_candidate(actual_group-predicted_group))",
            "test_access": "evaluation_only_after_validation_parameters_frozen",
        }
        or value["evaluation"]["splits"] != ["validation", "test"]
        or value["evaluation"]["new_simulator_rollout_count"] != 0
        or value["evaluation"]["retraining"] is not False
        or value["arms"] != [
            "compact_minimum_risk",
            "compact_zero_margin_safe_or_abstain",
            "compact_all_group_margin_safe_or_abstain",
            "compact_L5_margin_safe_or_abstain",
            "compact_33D_L5_hybrid_safe_or_abstain",
        ]
    ):
        raise ValueError("compact selector ablation protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def group_max(values: Mapping[int, float], rows: Sequence[int]) -> float:
    return max(float(values[int(row)]) for row in rows)


def enrich_records(
    records: Sequence[Mapping[str, Any]],
    specialist_predictions: Mapping[tuple[str, str], Sequence[float]],
) -> list[dict[str, Any]]:
    """Attach compact, specialist, fused, and exact per-group values."""
    output = []
    for source in records:
        row = dict(source)
        key = (str(row["state_id"]), str(row["candidate_name"]))
        specialist = specialist_predictions.get(key)
        if specialist is None or len(specialist) != 3:
            raise ValueError("compact selector specialist prediction differs")
        compact = {
            group: group_max(row["predicted_by_row"], rows)
            for group, rows in PRIMARY_GROUP_ROWS.items()
        }
        specialist_l5 = max(float(value) for value in specialist)
        hybrid = dict(compact)
        hybrid["L5"] = max(compact["L5"], specialist_l5)
        actual = None
        if bool(row["known_outcome"]):
            actual = {
                group: group_max(row["actual_by_row"], rows)
                for group, rows in AUDIT_GROUP_ROWS.items()
            }
        row["predicted_by_group"] = {
            "compact": compact,
            "compact_33D_L5_hybrid": hybrid,
        }
        row["predicted_33D_L5_by_row"] = [
            float(value) for value in specialist
        ]
        row["predicted_33D_L5"] = specialist_l5
        row["actual_by_group"] = actual
        output.append(row)
    return output


def validation_margins(
    records: Sequence[Mapping[str, Any]], *, prediction: str,
) -> dict[str, float]:
    """Maximum observed optimism by group, using known validation only."""
    known = [row for row in records if bool(row["known_outcome"])]
    if not known or any(str(row["split"]) != "validation" for row in known):
        raise ValueError("compact selector margin data differ")
    return {
        group: max(0.0, max(
            float(row["actual_by_group"][group])
            - float(row["predicted_by_group"][prediction][group])
            for row in known
        ))
        for group in PRIMARY_GROUP_ROWS
    }


def _failed_groups(row: Mapping[str, Any]) -> list[str]:
    if not bool(row["known_outcome"]):
        return []
    return sorted(
        group for group, value in row["actual_by_group"].items()
        if float(value) > 0.0
    )


def evaluate_arm(
    records: Sequence[Mapping[str, Any]], *, name: str, prediction: str,
    margins: Mapping[str, float], selection: str,
) -> dict[str, Any]:
    """Evaluate a frozen inference rule; abstention never becomes safety."""
    groups = set(PRIMARY_GROUP_ROWS)
    if set(margins) != groups:
        raise ValueError("compact selector margin groups differ")
    by_state: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        by_state.setdefault(str(row["state_id"]), []).append(row)
    states = []
    for state_id, rows in sorted(by_state.items()):
        known_primary_safe = [
            row for row in rows
            if bool(row["known_outcome"])
            and max(float(row["actual_by_group"][group]) for group in groups)
            <= 0.0
        ]
        known_physical_safe = [
            row for row in rows
            if row["actual_all_physical_safe"] is True
        ]
        if selection == "minimum_risk":
            selected = min(rows, key=lambda row: (
                max(float(value) for value in row[
                    "predicted_by_group"
                ][prediction].values()),
                float(row["correction"]), int(row["candidate_order"]),
            ))
        elif selection == "safe_or_abstain":
            accepted = [
                row for row in rows
                if all(
                    float(row["predicted_by_group"][prediction][group])
                    + float(margins[group]) <= 0.0
                    for group in groups
                )
            ]
            selected = min(accepted, key=lambda row: (
                float(row["correction"]), int(row["candidate_order"]),
            )) if accepted else None
        else:
            raise ValueError("compact selector selection rule differs")
        if selected is None:
            states.append({
                "state_id": state_id,
                "recoverable_primary": bool(known_primary_safe),
                "recoverable_all_physical": bool(known_physical_safe),
                "actual_primary_safe_candidate_count": len(known_primary_safe),
                "actual_all_physical_safe_candidate_count": len(known_physical_safe),
                "abstained": True,
                "selected_candidate": None,
                "selected_outcome_known": None,
                "selected_actual_primary_safe": None,
                "selected_actual_all_physical_safe": None,
                "selected_failed_groups": [],
                "selected_prediction_by_group": None,
                "selected_actual_by_group": None,
                "selected_correction": None,
            })
            continue
        actual_primary_safe = None
        if bool(selected["known_outcome"]):
            actual_primary_safe = bool(max(
                float(selected["actual_by_group"][group]) for group in groups
            ) <= 0.0)
        states.append({
            "state_id": state_id,
            "recoverable_primary": bool(known_primary_safe),
            "recoverable_all_physical": bool(known_physical_safe),
            "actual_primary_safe_candidate_count": len(known_primary_safe),
            "actual_all_physical_safe_candidate_count": len(known_physical_safe),
            "abstained": False,
            "selected_candidate": str(selected["candidate_name"]),
            "selected_outcome_known": bool(selected["known_outcome"]),
            "selected_actual_primary_safe": actual_primary_safe,
            "selected_actual_all_physical_safe": selected[
                "actual_all_physical_safe"
            ],
            "selected_failed_groups": _failed_groups(selected),
            "selected_prediction_by_group": selected[
                "predicted_by_group"
            ][prediction],
            "selected_actual_by_group": selected["actual_by_group"],
            "selected_correction": float(selected["correction"]),
        })
    selected_states = [row for row in states if not row["abstained"]]
    recoverable = [row for row in states if row["recoverable_primary"]]
    corrections = [
        float(row["selected_correction"]) for row in selected_states
    ]
    per_group_failure = {
        group: sum(group in row["selected_failed_groups"] for row in states)
        for group in AUDIT_GROUP_ROWS
    }
    safe = sum(
        row["selected_actual_all_physical_safe"] is True for row in states
    )
    unsafe = sum(
        row["selected_actual_all_physical_safe"] is False for row in states
    )
    unknown = sum(
        row["selected_outcome_known"] is False for row in selected_states
    )
    return {
        "arm": name,
        "prediction": prediction,
        "selection": selection,
        "margins": {group: float(margins[group]) for group in sorted(groups)},
        "state_count": len(states),
        "recoverable_primary_state_count": len(recoverable),
        "selected_state_count": len(selected_states),
        "abstention_count": len(states) - len(selected_states),
        "known_all_physical_safe_selection_count": safe,
        "known_all_physical_unsafe_selection_count": unsafe,
        "known_primary_false_safe_selection_count": sum(
            row["selected_actual_primary_safe"] is False for row in states
        ),
        "unknown_selection_count": unknown,
        "selected_safe_fraction": (
            None if not selected_states else safe / len(selected_states)
        ),
        "safe_primary_support_fraction": (
            1.0 if not recoverable else sum(
                row["selected_actual_primary_safe"] is True for row in recoverable
            ) / len(recoverable)
        ),
        "per_group_selected_failure_count": per_group_failure,
        "mean_selected_correction": (
            None if not corrections else sum(corrections) / len(corrections)
        ),
        "max_selected_correction": (
            None if not corrections else max(corrections)
        ),
        "states": states,
    }


def exact_float_lists_close(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]],
    *, tolerance: float,
) -> tuple[bool, float]:
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        return False, math.inf
    error = max(
        (abs(float(a) - float(b)) for x, y in zip(left, right)
         for a, b in zip(x, y)),
        default=0.0,
    )
    return error <= float(tolerance), error


def predict_serialized_mlp_float32(
    features: Sequence[Sequence[float]], state_payload: Mapping[str, Any],
) -> list[list[float]]:
    """Replay the frozen Torch MLP with float32 layer arithmetic.

    Normalization and output de-normalization follow the original trainer:
    float64 statistics bracket a float32 network. This matters for strongly
    out-of-distribution candidates and is only an apparatus replay operation.
    """
    import numpy as np

    mean = np.asarray(state_payload["feature_mean"], dtype=np.float64)
    scale = np.asarray(state_payload["feature_scale"], dtype=np.float64)
    target_mean = np.asarray(state_payload["target_mean"], dtype=np.float64)
    target_scale = np.asarray(state_payload["target_scale"], dtype=np.float64)
    state = state_payload["state_dict"]
    output = []
    for feature in features:
        raw = np.asarray(feature, dtype=np.float64)
        if raw.shape != mean.shape or scale.shape != mean.shape:
            raise ValueError("compact selector serialized feature shape differs")
        value = ((raw - mean) / scale).astype(np.float32)
        for layer_index, key in enumerate(("0", "2", "4")):
            weight = np.asarray(state[f"{key}.weight"], dtype=np.float32)
            bias = np.asarray(state[f"{key}.bias"], dtype=np.float32)
            value = (weight @ value + bias).astype(np.float32)
            if layer_index < 2:
                sigmoid = np.empty_like(value)
                positive = value >= np.float32(0.0)
                sigmoid[positive] = np.float32(1.0) / (
                    np.float32(1.0) + np.exp(-value[positive])
                )
                negative_exp = np.exp(value[~positive])
                sigmoid[~positive] = negative_exp / (
                    np.float32(1.0) + negative_exp
                )
                value = (value * sigmoid).astype(np.float32)
        if value.shape != target_mean.shape:
            raise ValueError("compact selector serialized output shape differs")
        physical = value.astype(np.float64) * target_scale + target_mean
        if not np.all(np.isfinite(physical)):
            raise ValueError("compact selector serialized output is non-finite")
        output.append(physical.tolist())
    return output


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"allocation", "result_payload_sha256", "source"}
    }
