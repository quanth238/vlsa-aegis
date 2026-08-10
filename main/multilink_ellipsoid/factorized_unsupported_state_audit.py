"""Frozen summaries for the one-sided model's unsupported-state audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .complete_osc_margin import _canonical, _numpy


CONFIG_SCHEMA = "vlsa_distal_factorized_unsupported_state_audit_config.v1"
RESULT_SCHEMA = "vlsa_distal_factorized_unsupported_state_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_unsupported_state_audit_validation.v1"


def load_unsupported_state_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("unsupported-state config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "measurements", "interpretation",
        "final_evaluation_policy", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("unsupported-state config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-unsupported-state-audit-moka10-v1"
    ):
        raise ValueError("unsupported-state protocol differs")
    if config["population"] != {
        "test_state_count": 15, "random_candidate_count_per_state": 64,
        "expected_unsupported_state_count": 3,
        "exact_safe_definition": "all_seven_dynamic_rollout_minimum_ellipsoid_margins_nonnegative",
        "accepted_definition": "all_seven_predicted_static_minimum_margins_nonnegative",
        "unsupported_definition": "at_least_one_exact_safe_candidate_and_zero_accepted_exact_safe_candidates",
    }:
        raise ValueError("unsupported-state population differs")
    if config["forbidden_actions"] != {
        "training": True, "new_controller_rollouts": True,
        "bias_or_quantile_correction_fit": True, "poisson_or_SDF": True,
        "binary_classifier": True, "calibration": True, "QP": True,
        "closed_loop": True, "verdict_change": True,
    }:
        raise ValueError("unsupported-state forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
    return output


def unsupported_state_indexes(
    *, exact_margin: Any, predicted_margin: Any, state_index: Any,
    selected: Any, expected_count: int,
) -> list[int]:
    np = _numpy()
    exact = np.asarray(exact_margin, dtype=np.float64)
    predicted = np.asarray(predicted_margin, dtype=np.float64)
    states = np.asarray(state_index, dtype=np.int64)
    mask = np.asarray(selected, dtype=bool)
    if exact.shape != predicted.shape or exact.shape[1:] != (7,):
        raise ValueError("unsupported-state margin arrays differ")
    output = []
    for state in sorted(set(states[mask].tolist())):
        rows = mask & (states == state)
        exact_safe = np.all(exact[rows] >= 0.0, axis=1)
        accepted = exact_safe & np.all(predicted[rows] >= 0.0, axis=1)
        if np.any(exact_safe) and not np.any(accepted):
            output.append(int(state))
    if len(output) != int(expected_count):
        raise ValueError("unsupported-state count differs")
    return output


def standardized_state_support(
    *, state_features: Any, state_indexes: Sequence[int], splits: Sequence[str],
    case_ids: Sequence[str], exact_safe_support: Sequence[bool],
    query_indexes: Sequence[int], maximum_abs_z: float,
) -> dict[str, Any]:
    """Measure query support against exact-safe-supported training states."""

    np = _numpy()
    features = np.asarray(state_features, dtype=np.float64)
    indexes = np.asarray(state_indexes, dtype=np.int64)
    split = np.asarray(splits, dtype=object)
    cases = np.asarray(case_ids, dtype=object)
    supported = np.asarray(exact_safe_support, dtype=bool)
    if features.shape[0] != len(indexes) or features.ndim != 2:
        raise ValueError("unsupported-state feature shape differs")
    train = (split == "train") & supported
    if np.count_nonzero(train) < 2:
        raise ValueError("unsupported-state training support is empty")
    mean = np.mean(features[train], axis=0)
    std = np.maximum(np.std(features[train], axis=0), 1e-6)
    z = (features - mean[None, :]) / std[None, :]

    def nearest(position: int, choices: Any, exclude_same_case: bool) -> dict[str, Any]:
        valid = np.flatnonzero(choices)
        if exclude_same_case:
            valid = valid[cases[valid] != cases[position]]
        valid = valid[indexes[valid] != indexes[position]]
        if not len(valid):
            raise ValueError("unsupported-state nearest support is empty")
        difference = z[valid] - z[position][None, :]
        rms = np.sqrt(np.mean(difference ** 2, axis=1))
        order = np.lexsort((indexes[valid], rms))
        chosen = int(valid[int(order[0])])
        delta = z[chosen] - z[position]
        return {
            "neighbor_state_index": int(indexes[chosen]),
            "neighbor_case_id": str(cases[chosen]),
            "distance_RMS_z": float(np.sqrt(np.mean(delta ** 2))),
            "maximum_absolute_z_difference": float(np.max(np.abs(delta))),
        }

    reference = []
    for position in np.flatnonzero(train):
        reference.append(nearest(int(position), train, True)["distance_RMS_z"])
    threshold = float(np.quantile(reference, 0.95))
    queries = {}
    for state in query_indexes:
        matches = np.flatnonzero(indexes == int(state))
        if len(matches) != 1:
            raise ValueError("unsupported-state query identity differs")
        neighbor = nearest(int(matches[0]), train, False)
        neighbor["support_reference_p95_RMS_z"] = threshold
        neighbor["support_pass"] = bool(
            neighbor["distance_RMS_z"] <= threshold + 1e-12
            and neighbor["maximum_absolute_z_difference"]
            <= float(maximum_abs_z) + 1e-12
        )
        queries[str(int(state))] = neighbor
    return {
        "supported_training_state_count": int(np.count_nonzero(train)),
        "reference_cross_episode_neighbor_count": int(len(reference)),
        "support_reference_p95_RMS_z": threshold,
        "queries": queries,
    }


def interpret_state(
    *, closest_predicted_margin_m: float, support_pass: bool,
    member_accepted_counts: Sequence[int], candidate_exact_safe_count: int,
    large_trajectory_error: bool, config: Mapping[str, Any],
) -> dict[str, Any]:
    member = [int(value) for value in member_accepted_counts]
    some_member_support = any(value > 0 for value in member)
    all_member_support = all(value > 0 for value in member)
    flags = {
        "conditional_bias_candidate": bool(
            -float(config["measurements"]["small_systematic_negative_bias_m"])
            <= float(closest_predicted_margin_m) < 0.0
            and support_pass and not large_trajectory_error
        ),
        "coverage_failure": not bool(support_pass),
        "aggregation_failure": bool(some_member_support and not all_member_support),
        "candidate_region_failure": int(candidate_exact_safe_count) == 0,
        "decoder_failure": bool(large_trajectory_error),
    }
    if flags["coverage_failure"]:
        primary = "unsupported_robot_or_controller_state"
    elif flags["decoder_failure"]:
        primary = "large_execution_trajectory_error"
    elif flags["aggregation_failure"]:
        primary = "ensemble_aggregation_rejects_member_supported_actions"
    elif flags["conditional_bias_candidate"]:
        primary = "small_systematic_negative_bias"
    else:
        primary = "systematic_rejection_not_explained_by_registered_simple_cases"
    return {"flags": flags, "primary_explanation": primary}
