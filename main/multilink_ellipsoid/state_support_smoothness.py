"""No-training feature-support and regional-oracle smoothness audit.

The audit consumes immutable controller-rollout artifacts.  It never invokes
MuJoCo, trains a model, solves a QP, or changes the released AEGIS path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import _canonical, _numpy
from .region_aware_mlp import (
    REGION_FEATURE_NAMES,
    regional_training_arrays,
)
from .two_step_margin import PAIR_CANDIDATE_SLICE, PAIR_FEATURE_NAMES


CONFIG_SCHEMA = "vlsa_distal_state_support_smoothness_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_state_support_smoothness_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_state_support_smoothness_moka10_validation.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("support/smoothness config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "split", "feature_shift", "state_support", "oracle_smoothness",
        "decision",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("support/smoothness config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-state-support-smoothness-moka10-v1"
    ):
        raise ValueError("support/smoothness protocol differs")
    source = config["immutable_source"]
    if set(source) != {
        "dataset_file_sha256", "dataset_payload_sha256",
        "multi_region_result_file_sha256",
        "multi_region_result_payload_sha256",
        "multi_region_validation_file_sha256",
        "learned_result_file_sha256", "learned_result_payload_sha256",
        "learned_validation_file_sha256", "expected_state_count",
        "expected_region_count", "expected_constraint_count",
        "fit_actions_per_state",
    }:
        raise ValueError("support/smoothness immutable source differs")
    if (
        int(source["expected_state_count"]) != 50
        or int(source["expected_region_count"]) != 27
        or int(source["expected_constraint_count"]) != 7
        or int(source["fit_actions_per_state"]) != 125
    ):
        raise ValueError("support/smoothness source cardinality differs")
    if config["split"] != {
        "unit": "complete_episode_and_task_level_group",
        "expected_state_counts": {"train": 30, "validation": 5, "test": 15},
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "reference_neighbor_exclusion": "same_case_id",
        "test_labels_never_used_to_define_thresholds": True,
    }:
        raise ValueError("support/smoothness split differs")
    if config["feature_shift"] != {
        "representation": "frozen_62D_region_aware_MLP_input",
        "normalization": "training_rows_mean_std_with_1e-6_floor",
        "absolute_z_shift_threshold": 5.0,
        "rank_by": ["test_max_abs_z", "test_p95_abs_z"],
        "report_top_feature_count": 20,
    }:
        raise ValueError("support/smoothness feature audit differs")
    if config["state_support"] != {
        "context": (
            "frozen_53D_pair_features_excluding_constraint_one_hot_and_"
            "candidate_action"
        ),
        "distance": "RMS_training_standardized_context_difference",
        "normalization": "training_constraint_rows_mean_std_with_1e-6_floor",
        "reference": (
            "nearest_training_state_from_a_different_complete_episode"
        ),
        "reference_quantile": 0.95,
        "maximum_single_feature_abs_z": 5.0,
        "required_supported_test_state_count": 15,
    }:
        raise ValueError("support/smoothness state support differs")
    if config["oracle_smoothness"] != {
        "comparison": (
            "test_state_to_its_support_nearest_training_state_at_matched_"
            "normalized_action_grid_coordinates"
        ),
        "value": (
            "union_margin_max_over_regions_of_minimum_seven_row_lower_value"
        ),
        "decision": "zero_margin_accepted_set_Jaccard",
        "reference": (
            "same_metrics_for_cross_episode_nearest_training_pairs"
        ),
        "maximum_value_RMSE_reference_quantile": 0.95,
        "minimum_Jaccard_reference_quantile": 0.05,
        "required_smooth_test_state_count": 15,
    }:
        raise ValueError("support/smoothness oracle audit differs")
    if config["decision"] != {
        "insufficient_support": (
            "collect_additional_grouped_boundary_states_from_new_episodes"
        ),
        "supported_but_nonsmooth": (
            "collect_denser_local_boundary_states_or_reduce_region_size"
        ),
        "supported_and_smooth_after_prior_MLP_failure": (
            "authorize_preregistration_of_action_conditioned_conservative_"
            "safety_value_model"
        ),
        "closed_loop_E05_authorized": False,
        "training_in_this_gate": False,
        "simulation_in_this_gate": False,
    }:
        raise ValueError("support/smoothness decision differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def linear_quantile(values: Sequence[float], probability: float) -> float:
    """Version-independent linear sample quantile."""

    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= float(probability) <= 1.0:
        raise ValueError("quantile input differs")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def boolean_jaccard(left: Sequence[bool], right: Sequence[bool]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Jaccard input differs")
    intersection = sum(bool(a) and bool(b) for a, b in zip(left, right))
    union = sum(bool(a) or bool(b) for a, b in zip(left, right))
    return 1.0 if union == 0 else float(intersection) / float(union)


def context_feature_indexes() -> list[int]:
    output = []
    for index, name in enumerate(PAIR_FEATURE_NAMES):
        if name.startswith("constraint_one_hot_"):
            continue
        if PAIR_CANDIDATE_SLICE.start <= index < PAIR_CANDIDATE_SLICE.stop:
            continue
        output.append(index)
    if len(output) != 43:
        raise ValueError("state-context feature count differs")
    return output


def context_arrays(
    states: Sequence[Mapping[str, Any]], expected_train_state_count: int = 30,
) -> dict[str, Any]:
    np = _numpy()
    indexes = context_feature_indexes()
    train_values = np.asarray([
        state["pair_state_feature_vectors"] for state in states
        if state["split"] == "train"
    ], dtype=np.float64)[:, :, indexes]
    if train_values.shape != (int(expected_train_state_count), 7, len(indexes)):
        raise ValueError("training context shape differs")
    mean = np.mean(train_values.reshape(-1, len(indexes)), axis=0)
    standard_deviation = np.maximum(
        np.std(train_values.reshape(-1, len(indexes)), axis=0), 1.0e-6
    )
    standardized = {}
    for state in states:
        values = np.asarray(
            state["pair_state_feature_vectors"], dtype=np.float64
        )[:, indexes]
        standardized[int(state["state_index"])] = (
            values - mean[None, :]
        ) / standard_deviation[None, :]
    return {
        "feature_indexes": indexes,
        "feature_names": [PAIR_FEATURE_NAMES[index] for index in indexes],
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standardized": standardized,
    }


def context_distance(left: Any, right: Any) -> tuple[float, float, Any]:
    np = _numpy()
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    if difference.ndim != 2 or difference.shape[0] != 7:
        raise ValueError("context distance shape differs")
    squared_by_feature = np.mean(difference * difference, axis=0)
    return (
        float(np.sqrt(np.mean(squared_by_feature))),
        float(np.max(np.abs(difference))),
        squared_by_feature,
    )


def nearest_state(
    query: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    standardized: Mapping[int, Any], exclude_same_case: bool = False,
) -> dict[str, Any]:
    choices = []
    query_index = int(query["state_index"])
    for candidate in candidates:
        if int(candidate["state_index"]) == query_index:
            continue
        if exclude_same_case and candidate["case_id"] == query["case_id"]:
            continue
        distance, maximum, contributions = context_distance(
            standardized[query_index], standardized[int(candidate["state_index"])]
        )
        choices.append((distance, int(candidate["state_index"]), maximum, contributions))
    if not choices:
        raise ValueError("no eligible support neighbor")
    distance, state_index, maximum, contributions = min(
        choices, key=lambda item: (item[0], item[1])
    )
    return {
        "neighbor_state_index": state_index,
        "distance_rms_z": float(distance),
        "pair_maximum_abs_z_difference": float(maximum),
        "squared_difference_by_feature": contributions,
    }


def oracle_signature(
    state: Mapping[str, Any], oracle_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the induced 27-region lower oracle on its normalized grid."""

    np = _numpy()
    candidate_xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
    lower = np.asarray(state["action_lower"], dtype=np.float64)
    upper = np.asarray(state["action_upper"], dtype=np.float64)
    if candidate_xyz.shape != (125, 3) or np.any(upper <= lower):
        raise ValueError("oracle signature action box differs")
    normalized = 2.0 * (candidate_xyz - lower[None, :]) / (
        upper - lower
    )[None, :] - 1.0
    canonical_axis = np.linspace(-1.0, 1.0, 5)
    canonical = np.asarray(
        [(x, y, z) for x in canonical_axis for y in canonical_axis for z in canonical_axis],
        dtype=np.float64,
    )
    normalized_error = float(np.max(np.abs(normalized - canonical)))
    if normalized_error > 1.0e-10:
        raise ValueError("normalized action grid differs")
    regions = oracle_state["regions"]
    targets = oracle_state["regional_targets"]
    if len(regions) != 27 or len(targets) != 27:
        raise ValueError("oracle region count differs")
    regional_values = []
    membership = [[] for _ in range(125)]
    region_value_maps = {}
    for region, target in zip(regions, targets):
        region_index = int(region["region_index"])
        if region_index != int(target["region_index"]):
            raise ValueError("oracle region order differs")
        indexes = [int(value) for value in region["fit_candidate_indexes"]]
        anchor = np.asarray(target["anchor_xyz"], dtype=np.float64)
        lower_anchor = np.asarray(target["anchor_margin_m"], dtype=np.float64) - np.asarray(
            target["one_sided_error_m"], dtype=np.float64
        )
        gradient = np.asarray(target["gradient_m_per_action"], dtype=np.float64)
        values = lower_anchor[None, :] + (
            candidate_xyz[indexes] - anchor[None, :]
        ) @ gradient.T
        if values.shape != (27, 7) or not np.all(np.isfinite(values)):
            raise ValueError("oracle regional value differs")
        regional_values.append(values.reshape(-1))
        region_value_maps[region_index] = {
            candidate_index: values[local_index]
            for local_index, candidate_index in enumerate(indexes)
        }
        for candidate_index in indexes:
            membership[candidate_index].append(region_index)
    envelope = []
    accepted = []
    for candidate_index, region_indexes in enumerate(membership):
        if not region_indexes:
            raise ValueError("oracle candidate has no region")
        region_minima = [
            float(np.min(region_value_maps[index][candidate_index]))
            for index in region_indexes
        ]
        value = max(region_minima)
        envelope.append(value)
        accepted.append(value >= 0.0)
    return {
        "normalized_grid_maximum_error": normalized_error,
        "regional_values": np.concatenate(regional_values),
        "union_margin_m": np.asarray(envelope, dtype=np.float64),
        "accepted_flags": accepted,
    }


def compare_oracles(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    left_union = np.asarray(left["union_margin_m"], dtype=np.float64)
    right_union = np.asarray(right["union_margin_m"], dtype=np.float64)
    left_regional = np.asarray(left["regional_values"], dtype=np.float64)
    right_regional = np.asarray(right["regional_values"], dtype=np.float64)
    return {
        "union_margin_RMSE_m": float(np.sqrt(np.mean((left_union - right_union) ** 2))),
        "regional_row_value_RMSE_m": float(
            np.sqrt(np.mean((left_regional - right_regional) ** 2))
        ),
        "accepted_set_Jaccard": boolean_jaccard(
            left["accepted_flags"], right["accepted_flags"]
        ),
        "left_accepted_count": int(sum(left["accepted_flags"])),
        "right_accepted_count": int(sum(right["accepted_flags"])),
    }


def analyze(
    dataset: Mapping[str, Any], multi_region: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    states = dataset["state_records"]
    oracle_states = multi_region["state_results"]
    expected = config["immutable_source"]
    if len(states) != int(expected["expected_state_count"]):
        raise ValueError("support/smoothness state count differs")
    split_counts = {
        name: sum(state["split"] == name for state in states)
        for name in ("train", "validation", "test")
    }
    if split_counts != config["split"]["expected_state_counts"]:
        raise ValueError("support/smoothness split count differs")
    train = [state for state in states if state["split"] == "train"]
    validation = [state for state in states if state["split"] == "validation"]
    test = [state for state in states if state["split"] == "test"]
    if sorted(set(state["case_id"] for state in test)) != sorted(
        config["split"]["test_case_ids"]
    ):
        raise ValueError("support/smoothness test identities differ")
    oracle_by_index = {int(item["state_index"]): item for item in oracle_states}
    if len(oracle_by_index) != len(states):
        raise ValueError("support/smoothness oracle state count differs")

    arrays = regional_training_arrays(states, oracle_states)
    train_mask = arrays["split"] == "train"
    test_mask = arrays["split"] == "test"
    feature_mean = np.mean(arrays["features"][train_mask], axis=0)
    feature_std = np.maximum(np.std(arrays["features"][train_mask], axis=0), 1.0e-6)
    train_abs = np.abs((arrays["features"][train_mask] - feature_mean) / feature_std)
    test_abs = np.abs((arrays["features"][test_mask] - feature_mean) / feature_std)
    shift_threshold = float(config["feature_shift"]["absolute_z_shift_threshold"])
    feature_records = []
    for index, name in enumerate(REGION_FEATURE_NAMES):
        feature_records.append({
            "feature_index": index,
            "feature_name": name,
            "train_maximum_abs_z": float(np.max(train_abs[:, index])),
            "test_p95_abs_z": linear_quantile(test_abs[:, index], 0.95),
            "test_maximum_abs_z": float(np.max(test_abs[:, index])),
            "test_fraction_above_shift_threshold": float(
                np.mean(test_abs[:, index] > shift_threshold)
            ),
        })
    feature_records.sort(
        key=lambda item: (
            -item["test_maximum_abs_z"], -item["test_p95_abs_z"],
            item["feature_index"],
        )
    )
    top_count = int(config["feature_shift"]["report_top_feature_count"])

    context = context_arrays(
        states,
        int(config["split"]["expected_state_counts"]["train"]),
    )
    reference_neighbors = []
    oracle_signatures = {
        int(state["state_index"]): oracle_signature(
            state, oracle_by_index[int(state["state_index"])]
        ) for state in states
    }
    state_by_index = {int(state["state_index"]): state for state in states}
    for state in train:
        neighbor = nearest_state(
            state, train, context["standardized"], exclude_same_case=True
        )
        comparison = compare_oracles(
            oracle_signatures[int(state["state_index"])],
            oracle_signatures[int(neighbor["neighbor_state_index"])],
        )
        reference_neighbors.append({
            "state_index": int(state["state_index"]),
            "case_id": state["case_id"], "state_step": int(state["state_step"]),
            "neighbor_state_index": int(neighbor["neighbor_state_index"]),
            "neighbor_case_id": state_by_index[
                int(neighbor["neighbor_state_index"])
            ]["case_id"],
            "distance_rms_z": neighbor["distance_rms_z"],
            "pair_maximum_abs_z_difference": neighbor[
                "pair_maximum_abs_z_difference"
            ],
            "oracle": comparison,
        })
    support_threshold = linear_quantile(
        [item["distance_rms_z"] for item in reference_neighbors],
        float(config["state_support"]["reference_quantile"]),
    )
    value_threshold = linear_quantile(
        [item["oracle"]["union_margin_RMSE_m"] for item in reference_neighbors],
        float(config["oracle_smoothness"]["maximum_value_RMSE_reference_quantile"]),
    )
    jaccard_threshold = linear_quantile(
        [item["oracle"]["accepted_set_Jaccard"] for item in reference_neighbors],
        float(config["oracle_smoothness"]["minimum_Jaccard_reference_quantile"]),
    )

    contribution_sum = np.zeros(len(context["feature_names"]), dtype=np.float64)
    def evaluate_population(population):
        records = []
        for state in population:
            state_index = int(state["state_index"])
            neighbor = nearest_state(
                state, train, context["standardized"], exclude_same_case=False
            )
            contribution_sum[:] += neighbor["squared_difference_by_feature"]
            maximum_state_abs_z = float(np.max(np.abs(
                context["standardized"][state_index]
            )))
            support_pass = bool(
                neighbor["distance_rms_z"] <= support_threshold + 1.0e-12
                and maximum_state_abs_z <= float(
                    config["state_support"]["maximum_single_feature_abs_z"]
                ) + 1.0e-12
            )
            comparison = compare_oracles(
                oracle_signatures[state_index],
                oracle_signatures[int(neighbor["neighbor_state_index"])],
            )
            smoothness_pass = bool(
                comparison["union_margin_RMSE_m"] <= value_threshold + 1.0e-12
                and comparison["accepted_set_Jaccard"] + 1.0e-12
                >= jaccard_threshold
            )
            neighbor_state = state_by_index[int(neighbor["neighbor_state_index"])]
            records.append({
                "state_index": state_index, "case_id": state["case_id"],
                "state_step": int(state["state_step"]),
                "nearest_training_state_index": int(neighbor["neighbor_state_index"]),
                "nearest_training_case_id": neighbor_state["case_id"],
                "nearest_training_state_step": int(neighbor_state["state_step"]),
                "distance_rms_z": neighbor["distance_rms_z"],
                "pair_maximum_abs_z_difference": neighbor[
                    "pair_maximum_abs_z_difference"
                ],
                "state_maximum_training_standardized_abs_z": maximum_state_abs_z,
                "support_pass": support_pass,
                "oracle": comparison,
                "smoothness_pass": smoothness_pass,
            })
        return records

    validation_results = evaluate_population(validation)
    contribution_sum[:] = 0.0
    test_results = evaluate_population(test)
    contribution_total = float(np.sum(contribution_sum))
    contribution_records = []
    for index, name in enumerate(context["feature_names"]):
        contribution_records.append({
            "feature_name": name,
            "mean_squared_z_difference": float(contribution_sum[index] / len(test)),
            "fraction_of_test_neighbor_distance_squared": (
                0.0 if contribution_total == 0.0
                else float(contribution_sum[index] / contribution_total)
            ),
        })
    contribution_records.sort(
        key=lambda item: (-item["fraction_of_test_neighbor_distance_squared"], item["feature_name"])
    )

    support_count = sum(item["support_pass"] for item in test_results)
    smooth_count = sum(item["smoothness_pass"] for item in test_results)
    support_sufficient = support_count == int(
        config["state_support"]["required_supported_test_state_count"]
    )
    smoothness_sufficient = smooth_count == int(
        config["oracle_smoothness"]["required_smooth_test_state_count"]
    )
    if not support_sufficient:
        next_action = config["decision"]["insufficient_support"]
    elif not smoothness_sufficient:
        next_action = config["decision"]["supported_but_nonsmooth"]
    else:
        next_action = config["decision"][
            "supported_and_smooth_after_prior_MLP_failure"
        ]
    return {
        "feature_shift": {
            "training_row_count": int(np.sum(train_mask)),
            "test_row_count": int(np.sum(test_mask)),
            "global_training_maximum_abs_z": float(np.max(train_abs)),
            "global_test_maximum_abs_z": float(np.max(test_abs)),
            "global_test_fraction_above_shift_threshold": float(
                np.mean(test_abs > shift_threshold)
            ),
            "shifted_feature_count": sum(
                item["test_maximum_abs_z"] > shift_threshold
                for item in feature_records
            ),
            "top_features": feature_records[:top_count],
            "test_neighbor_distance_top_contributors": contribution_records[:top_count],
        },
        "reference": {
            "cross_episode_training_pairs": reference_neighbors,
            "support_distance_RMS_z_p95": support_threshold,
            "oracle_union_margin_RMSE_m_p95": value_threshold,
            "oracle_accepted_set_Jaccard_p05": jaccard_threshold,
        },
        "validation_state_results": validation_results,
        "test_state_results": test_results,
        "aggregates": {
            "test_state_count": len(test_results),
            "supported_test_state_count": support_count,
            "smooth_test_state_count": smooth_count,
            "support_sufficient": support_sufficient,
            "oracle_smoothness_sufficient": smoothness_sufficient,
            "action_conditioned_model_preregistration_authorized": bool(
                support_sufficient and smoothness_sufficient
            ),
            "closed_loop_E05_authorized": False,
        },
        "decision": {
            "next_action": next_action,
            "closed_loop_E05_remains_blocked": True,
            "new_training_executed": False,
            "new_simulation_executed": False,
        },
    }
