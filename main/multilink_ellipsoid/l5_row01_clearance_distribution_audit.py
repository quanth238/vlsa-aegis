"""State-balanced distribution audit for the three current-L5 clearances."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_row01_clearance_distribution_audit.v1"
RESULT_SCHEMA = "vlsa_distal_l5_row01_clearance_distribution_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_row01_clearance_distribution_audit_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
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
        "schema_version", "protocol_id", "claim_scope", "source", "audit",
        "required_outputs", "forbidden",
    }:
        raise ValueError("clearance distribution audit config keys differ")
    audit = value["audit"]
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-distal-l5-row01-clearance-distribution-audit-v1"
        or audit["clearance_feature_indices"] != [9, 10, 11]
        or audit["clearance_rows"] != [0, 1, 2]
        or audit["state_balanced"] is not True
        or audit["within_state_bitwise_equality_required"] is not True
    ):
        raise ValueError("clearance distribution audit protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _rankdata(values: Any) -> Any:
    """Average ranks, implemented locally to avoid a SciPy dependency."""
    import numpy as np

    raw = np.asarray(values, dtype=np.float64)
    order = np.argsort(raw, kind="mergesort")
    ranks = np.empty(raw.size, dtype=np.float64)
    cursor = 0
    while cursor < raw.size:
        end = cursor + 1
        while end < raw.size and raw[order[end]] == raw[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1) + 1.0
        cursor = end
    return ranks


def _correlation(left: Any, right: Any) -> float | None:
    import numpy as np

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _state_groups(
    samples: Sequence[Mapping[str, Any]], predictions_9d: Any,
    predictions_12d: Any,
) -> dict[str, dict[str, Any]]:
    import numpy as np

    prediction_9d = np.asarray(predictions_9d, dtype=np.float64)
    prediction_12d = np.asarray(predictions_12d, dtype=np.float64)
    if (
        prediction_9d.shape != (len(samples), 2)
        or prediction_12d.shape != (len(samples), 2)
    ):
        raise ValueError("clearance audit prediction shape differs")
    groups: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        state_id = str(sample["state_id"])
        feature = np.asarray(
            sample["clearance_endpoint_feature_vector"], dtype=np.float64,
        )
        if feature.shape != (12,) or not np.all(np.isfinite(feature)):
            raise ValueError("clearance audit feature differs")
        clearances = feature[-3:].copy()
        group = groups.setdefault(state_id, {
            "state_id": state_id,
            "case_id": str(sample["case_id"]),
            "clearances": clearances,
            "indices": [],
        })
        if not np.array_equal(group["clearances"], clearances):
            raise ValueError("clearances vary across candidates of one state")
        group["indices"].append(index)
    for group in groups.values():
        indices = group.pop("indices")
        target = np.asarray(
            [samples[index]["risk_row01"] for index in indices],
            dtype=np.float64,
        )
        p9 = prediction_9d[indices]
        p12 = prediction_12d[indices]
        safe9 = np.max(p9, axis=1) <= 0.0
        safe12 = np.max(p12, axis=1) <= 0.0
        actual = np.max(target, axis=1) <= 0.0
        exact = np.asarray(
            [bool(samples[index]["exact_safe"]) for index in indices], dtype=bool,
        )
        group.update({
            "candidate_count": len(indices),
            "recoverable": bool(np.any(exact)),
            "exact_safe_candidate_count": int(np.sum(exact)),
            "rmse_9D_m": float(np.sqrt(np.mean((p9 - target) ** 2))),
            "rmse_12D_m": float(np.sqrt(np.mean((p12 - target) ** 2))),
            "rmse_12D_minus_9D_m": float(
                np.sqrt(np.mean((p12 - target) ** 2))
                - np.sqrt(np.mean((p9 - target) ** 2))
            ),
            "false_safe_9D_count": int(np.sum(safe9 & ~actual)),
            "false_safe_12D_count": int(np.sum(safe12 & ~actual)),
            "predicted_safe_9D_count": int(np.sum(safe9)),
            "predicted_safe_12D_count": int(np.sum(safe12)),
            "support_9D": bool(np.any(safe9)),
            "support_12D": bool(np.any(safe12)),
            "support_lost_by_12D": bool(np.any(safe9) and not np.any(safe12)),
        })
    return groups


def audit_distribution(
    *, train_samples: Sequence[Mapping[str, Any]],
    validation_samples: Sequence[Mapping[str, Any]], train_predictions_9d: Any,
    validation_predictions_9d: Any, train_predictions_12d: Any,
    validation_predictions_12d: Any, standard_deviation_floor_m: float,
    range_tolerance_m: float, high_distance_threshold: float,
) -> dict[str, Any]:
    import numpy as np

    train_groups = _state_groups(
        train_samples, train_predictions_9d, train_predictions_12d,
    )
    validation_groups = _state_groups(
        validation_samples, validation_predictions_9d,
        validation_predictions_12d,
    )
    train_state_clearances = np.stack([
        train_groups[state_id]["clearances"] for state_id in sorted(train_groups)
    ])
    train_candidate_clearances = np.asarray([
        sample["clearance_endpoint_feature_vector"][-3:]
        for sample in train_samples
    ], dtype=np.float64)
    state_mean = np.mean(train_state_clearances, axis=0)
    state_std_raw = np.std(train_state_clearances, axis=0)
    state_std = np.maximum(
        state_std_raw, float(standard_deviation_floor_m),
    )
    state_min = np.min(train_state_clearances, axis=0)
    state_max = np.max(train_state_clearances, axis=0)
    candidate_mean = np.mean(train_candidate_clearances, axis=0)
    candidate_std = np.std(train_candidate_clearances, axis=0)
    records = []
    for state_id in sorted(validation_groups):
        group = validation_groups[state_id]
        clearance = np.asarray(group["clearances"], dtype=np.float64)
        below = clearance < state_min - float(range_tolerance_m)
        above = clearance > state_max + float(range_tolerance_m)
        z = (clearance - state_mean) / state_std
        distances = np.sqrt(np.mean(
            ((train_state_clearances - clearance) / state_std) ** 2, axis=1,
        ))
        nearest_index = int(np.argmin(distances))
        nearest_id = sorted(train_groups)[nearest_index]
        record = {
            key: value for key, value in group.items() if key != "clearances"
        }
        record.update({
            "clearances_m": clearance.tolist(),
            "below_training_range_by_row": below.tolist(),
            "above_training_range_by_row": above.tolist(),
            "outside_training_range_by_row": (below | above).tolist(),
            "outside_training_range_count": int(np.sum(below | above)),
            "state_balanced_train_z_by_row": z.tolist(),
            "maximum_absolute_train_z": float(np.max(np.abs(z))),
            "nearest_training_state_id": nearest_id,
            "nearest_training_state_rms_z_distance": float(
                distances[nearest_index]
            ),
            "high_nearest_training_state_distance": bool(
                distances[nearest_index] >= float(high_distance_threshold)
            ),
        })
        records.append(record)
    distances = [
        item["nearest_training_state_rms_z_distance"] for item in records
    ]
    error_changes = [item["rmse_12D_minus_9D_m"] for item in records]
    outside = [bool(item["outside_training_range_count"]) for item in records]
    outside_changes = [
        delta for delta, flag in zip(error_changes, outside) if flag
    ]
    inside_changes = [
        delta for delta, flag in zip(error_changes, outside) if not flag
    ]
    return {
        "state_counting_rule": "one_record_per_distinct_physical_state",
        "train_candidate_count": len(train_samples),
        "validation_candidate_count": len(validation_samples),
        "train_distinct_state_count": len(train_groups),
        "validation_distinct_state_count": len(validation_groups),
        "train_candidates_per_state": {
            state_id: int(train_groups[state_id]["candidate_count"])
            for state_id in sorted(train_groups)
        },
        "state_balanced_train_clearance_statistics": {
            "minimum_m": state_min.tolist(),
            "maximum_m": state_max.tolist(),
            "mean_m": state_mean.tolist(),
            "standard_deviation_m": state_std_raw.tolist(),
            "standard_deviation_used_for_z_m": state_std.tolist(),
        },
        "candidate_weighted_train_clearance_statistics": {
            "mean_m": candidate_mean.tolist(),
            "standard_deviation_m": candidate_std.tolist(),
            "mean_minus_state_balanced_mean_m": (
                candidate_mean - state_mean
            ).tolist(),
        },
        "validation_state_records": records,
        "validation_state_outside_training_range_count": int(sum(outside)),
        "validation_state_high_distance_count": int(sum(
            item["high_nearest_training_state_distance"] for item in records
        )),
        "validation_support_lost_by_12D_count": int(sum(
            item["support_lost_by_12D"] for item in records
        )),
        "nearest_distance_vs_12D_minus_9D_error_pearson": _correlation(
            distances, error_changes,
        ),
        "nearest_distance_vs_12D_minus_9D_error_spearman": _correlation(
            _rankdata(distances), _rankdata(error_changes),
        ),
        "mean_12D_minus_9D_error_outside_range_m": (
            float(np.mean(outside_changes)) if outside_changes else None
        ),
        "mean_12D_minus_9D_error_inside_range_m": (
            float(np.mean(inside_changes)) if inside_changes else None
        ),
    }


def classify(audit: Mapping[str, Any]) -> str:
    records = audit["validation_state_records"]
    lost = [item for item in records if item["support_lost_by_12D"]]
    all_lost_shifted = bool(lost) and all(
        item["outside_training_range_count"] > 0
        or item["high_nearest_training_state_distance"]
        for item in lost
    )
    shifted_worse = [
        item for item in records
        if (
            item["outside_training_range_count"] > 0
            or item["high_nearest_training_state_distance"]
        ) and item["rmse_12D_minus_9D_m"] > 0.0
    ]
    if all_lost_shifted and shifted_worse:
        return (
            "clearance_distribution_shift_associated_with_12D_degradation_"
            "retain_9D_and_collect_state_support"
        )
    if not audit["validation_state_outside_training_range_count"]:
        return (
            "clearances_within_train_ranges_but_12D_degrades_"
            "representation_or_model_interaction"
        )
    return (
        "mixed_clearance_support_no_single_root_cause_"
        "retain_9D_and_do_not_add_inputs"
    )
