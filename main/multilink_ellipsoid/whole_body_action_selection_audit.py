"""Offline finite-bank selection rules for frozen whole-body Q predictions."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


GROUP_ROWS = {"palm": (1,), "L5": (2, 3, 4), "L6": (5, 6)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def candidate_records(
    samples: Sequence[Mapping[str, Any]], predictions: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    """Aggregate row-level predictions into physical candidate risk records."""

    _require(len(samples) == len(predictions), "selection prediction shape differs")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for sample, prediction_row in zip(samples, predictions):
        prediction = list(prediction_row)
        _require(len(prediction) == 1, "selection prediction is not scalar")
        state_id = str(sample["state_id"])
        candidate_name = str(sample["candidate_name"])
        key = (state_id, candidate_name)
        record = records.setdefault(key, {
            "state_id": state_id,
            "candidate_name": candidate_name,
            "candidate_order": int(sample["candidate_order"]),
            "correction": float(sample["applied_correction_l2_action"]),
            "actual_by_row": {},
            "predicted_by_row": {},
            "known_outcome": bool(sample.get("known_outcome", True)),
        })
        _require(
            record["known_outcome"] == bool(sample.get("known_outcome", True)),
            "selection candidate outcome status differs by row",
        )
        row = int(sample["row_index"])
        _require(row not in record["actual_by_row"], "selection row repeats")
        if record["known_outcome"]:
            record["actual_by_row"][row] = float(sample["risk"])
        record["predicted_by_row"][row] = float(prediction[0])
    required = sorted(row for rows in GROUP_ROWS.values() for row in rows)
    output = []
    for record in records.values():
        _require(
            (
                not record["known_outcome"]
                or sorted(record["actual_by_row"]) == required
            ) and sorted(record["predicted_by_row"]) == required,
            "selection candidate rows differ",
        )
        actual_group = None if not record["known_outcome"] else {
            group: max(record["actual_by_row"][row] for row in rows)
            for group, rows in GROUP_ROWS.items()
        }
        predicted_group = {
            group: max(record["predicted_by_row"][row] for row in rows)
            for group, rows in GROUP_ROWS.items()
        }
        output.append({
            **record,
            "actual_by_group": actual_group,
            "predicted_by_group": predicted_group,
            "actual_global": (
                None if actual_group is None else max(actual_group.values())
            ),
            "predicted_global": max(predicted_group.values()),
        })
    return sorted(
        output, key=lambda row: (row["state_id"], row["candidate_order"])
    )


def predict_serialized_mlp(
    features: Sequence[Sequence[float]], state_payload: Mapping[str, Any],
) -> list[list[float]]:
    """Evaluate the frozen three-linear-layer SiLU MLP without retraining."""

    import math

    mean = [float(value) for value in state_payload["feature_mean"]]
    scale = [float(value) for value in state_payload["feature_scale"]]
    _require(len(mean) == len(scale), "selection serialized MLP scale differs")
    state = state_payload["state_dict"]
    target_mean = [float(value) for value in state_payload["target_mean"]]
    target_scale = [float(value) for value in state_payload["target_scale"]]
    output = []
    for raw_row in features:
        _require(len(raw_row) == len(mean),
                 "selection serialized MLP feature shape differs")
        value = [
            (float(item) - mean[index]) / scale[index]
            for index, item in enumerate(raw_row)
        ]
        for layer_index, key in enumerate(("0", "2", "4")):
            weights = state[f"{key}.weight"]
            biases = state[f"{key}.bias"]
            _require(len(weights) == len(biases),
                     "selection serialized MLP layer shape differs")
            value = [
                float(bias) + sum(
                    float(left) * float(right)
                    for left, right in zip(row, value)
                )
                for row, bias in zip(weights, biases)
            ]
            if layer_index < 2:
                value = [
                    item * (
                        1.0 / (1.0 + math.exp(-item))
                        if item >= 0.0 else
                        math.exp(item) / (1.0 + math.exp(item))
                    )
                    for item in value
                ]
        _require(len(value) == len(target_mean) == len(target_scale),
                 "selection serialized MLP output shape differs")
        value = [
            item * target_scale[index] + target_mean[index]
            for index, item in enumerate(value)
        ]
        _require(all(math.isfinite(item) for item in value),
                 "selection serialized MLP is non-finite")
        output.append(value)
    return output


def validation_optimistic_margin(records: Sequence[Mapping[str, Any]]) -> float:
    """Return the smallest observed validation error bound preventing optimism."""

    _require(bool(records), "selection validation records are empty")
    return max(
        0.0,
        max(
            float(row["actual_global"]) - float(row["predicted_global"])
            for row in records if row["actual_global"] is not None
        ),
    )


def _select(
    rows: Sequence[Mapping[str, Any]], *, rule: str, margin: float = 0.0,
) -> Mapping[str, Any] | None:
    if rule == "nominal":
        eligible = [row for row in rows if row["candidate_name"] == "nominal"]
    elif rule == "least_intervention_predicted_safe":
        eligible = [
            row for row in rows
            if float(row["predicted_global"]) <= -float(margin)
        ]
    elif rule == "minimum_predicted_risk":
        eligible = list(rows)
        return min(
            eligible,
            key=lambda row: (
                float(row["predicted_global"]), float(row["correction"]),
                int(row["candidate_order"]),
            ),
        ) if eligible else None
    elif rule == "exact_oracle_minimum_intervention":
        eligible = [
            row for row in rows
            if row["actual_global"] is not None
            and float(row["actual_global"]) <= 0.0
        ]
    else:
        raise ValueError("unknown selection rule")
    return min(
        eligible,
        key=lambda row: (float(row["correction"]), int(row["candidate_order"])),
    ) if eligible else None


def evaluate_rule(
    records: Sequence[Mapping[str, Any]], *, rule: str, margin: float = 0.0,
) -> dict[str, Any]:
    by_state: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        by_state.setdefault(str(row["state_id"]), []).append(row)
    states = []
    witnesses: Counter[str] = Counter()
    for state_id, rows in sorted(by_state.items()):
        actual_safe = [
            row for row in rows
            if row["actual_global"] is not None
            and float(row["actual_global"]) <= 0.0
        ]
        selected = _select(rows, rule=rule, margin=margin)
        selected_safe = (
            None if selected is None or selected["actual_global"] is None
            else float(selected["actual_global"]) <= 0.0
        )
        failed_groups = [] if selected is None or selected["actual_by_group"] is None else [
            group for group, risk in selected["actual_by_group"].items()
            if float(risk) > 0.0
        ]
        witnesses.update(failed_groups)
        states.append({
            "state_id": state_id,
            "recoverable": bool(actual_safe),
            "actual_safe_candidate_count": len(actual_safe),
            "selected_candidate": None if selected is None else selected["candidate_name"],
            "selected_actual_safe": selected_safe,
            "selected_outcome_known": (
                None if selected is None else bool(selected["known_outcome"])
            ),
            "selected_correction": None if selected is None else selected["correction"],
            "selected_actual_global": None if selected is None else selected["actual_global"],
            "selected_predicted_global": None if selected is None else selected["predicted_global"],
            "selected_failed_groups": failed_groups,
        })
    recoverable = [row for row in states if row["recoverable"]]
    selected_recoverable = [
        row for row in recoverable if row["selected_candidate"] is not None
    ]
    false_safe = [
        row for row in selected_recoverable if row["selected_actual_safe"] is False
    ]
    selected_unknown = [
        row for row in selected_recoverable if row["selected_actual_safe"] is None
    ]
    return {
        "rule": rule,
        "margin": float(margin),
        "state_count": len(states),
        "recoverable_state_count": len(recoverable),
        "selected_recoverable_state_count": len(selected_recoverable),
        "safe_selected_recoverable_state_count": sum(
            row["selected_actual_safe"] is True for row in selected_recoverable
        ),
        "false_safe_selected_state_count": len(false_safe),
        "unknown_selected_state_count": len(selected_unknown),
        "abstained_recoverable_state_count": len(recoverable) - len(selected_recoverable),
        "false_safe_group_count": dict(sorted(witnesses.items())),
        "states": states,
    }


def evaluate_ranked_exact_verification(
    records: Sequence[Mapping[str, Any]], *, top_ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Measure how many MLP-ranked proposals precede the first exact-safe one."""

    by_state: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        by_state.setdefault(str(row["state_id"]), []).append(row)
    states = []
    for state_id, rows in sorted(by_state.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                float(row["predicted_global"]), float(row["correction"]),
                int(row["candidate_order"]),
            ),
        )
        safe_ranks = [
            rank for rank, row in enumerate(ranked, start=1)
            if row["actual_global"] is not None
            and float(row["actual_global"]) <= 0.0
        ]
        if not safe_ranks:
            continue
        first = int(safe_ranks[0])
        preceding = ranked[:first - 1]
        states.append({
            "state_id": state_id,
            "first_exact_safe_rank": first,
            "first_exact_safe_candidate": ranked[first - 1]["candidate_name"],
            "first_exact_safe_correction": ranked[first - 1]["correction"],
            "unknown_checks_before_safe": sum(
                row["actual_global"] is None for row in preceding
            ),
            "unsafe_checks_before_safe": sum(
                row["actual_global"] is not None
                and float(row["actual_global"]) > 0.0
                for row in preceding
            ),
        })
    ranks = [row["first_exact_safe_rank"] for row in states]
    return {
        "recoverable_state_count": len(states),
        "maximum_candidates_checked": max(ranks, default=None),
        "mean_candidates_checked": (
            None if not ranks else sum(ranks) / len(ranks)
        ),
        "unknown_checks_before_safe": sum(
            row["unknown_checks_before_safe"] for row in states
        ),
        "unsafe_checks_before_safe": sum(
            row["unsafe_checks_before_safe"] for row in states
        ),
        "top_k_safe_support": {
            str(int(k)): sum(rank <= int(k) for rank in ranks)
            for k in top_ks
        },
        "states": states,
    }
