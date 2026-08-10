"""Leave-one-episode-out local upper error bounds for executed margins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_conditioned_margin import (
    predict_member_margins, solve_regional_qps,
)
from .execution_margin_nn import CONSTRAINT_ORDER, _canonical, _numpy
from .multi_region_affine_oracle import fit_region_target, region_affine_values
from .state_support_smoothness import context_feature_indexes


CONFIG_SCHEMA = "vlsa_distal_local_residual_bound_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_local_residual_bound_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_local_residual_bound_moka10_validation.v1"
OOF_SCHEMA = "vlsa_distal_local_residual_bound_moka10_oof.v1"
LOCAL_FEATURE_NAMES = tuple(
    ["context_%02d" % index for index in context_feature_indexes()]
    + ["candidate_normalized_%d" % index for index in range(3)]
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("local-residual config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "source_population_manifest_sha256", "test_selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "split", "leave_one_episode_out", "local_features", "local_bound",
        "regionalization", "projection", "learned_gate",
        "exact_verification", "decision",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("local-residual config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"] != "vlsa-distal-local-residual-bound-moka10-v1"
    ):
        raise ValueError("local-residual protocol differs")
    source = config["immutable_source"]
    source_keys = {
        "expanded_dataset_file_sha256", "expanded_dataset_payload_sha256",
        "expanded_oracle_file_sha256", "expanded_oracle_payload_sha256",
        "validation_fresh_file_sha256", "validation_fresh_payload_sha256",
        "action_conditioned_config_file_sha256",
        "action_conditioned_result_file_sha256",
        "action_conditioned_result_payload_sha256",
        "action_conditioned_validation_file_sha256",
        "action_conditioned_model_file_sha256",
        "target_authority_result_file_sha256",
        "target_authority_result_payload_sha256",
        "target_authority_validation_file_sha256", "archived_e05_file_sha256",
        "archived_e05_payload_sha256", "expected_state_count",
        "fit_actions_per_state", "fresh_actions_per_validation_or_test_state",
    }
    if (
        set(source) != source_keys
        or int(source["expected_state_count"]) != 85
        or int(source["fit_actions_per_state"]) != 125
        or int(source["fresh_actions_per_validation_or_test_state"]) != 96
    ):
        raise ValueError("local-residual immutable source differs")
    if config["split"] != {
        "unit": "complete_episode_and_task_level_group",
        "expected_state_counts": {"train": 60, "validation": 10, "test": 15},
        "expected_training_episode_count": 12, "states_per_episode": 5,
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "test_never_used_for_fold_training_residual_estimation_normalization_or_thresholds": True,
    }:
        raise ValueError("local-residual grouped split differs")
    if config["leave_one_episode_out"] != {
        "fold_unit": "one_complete_training_episode", "fold_count": 12,
        "held_out_states_per_fold": 5,
        "fit_population": "other_55_training_states",
        "early_stopping_population": "unchanged_10_validation_states",
        "base_model": "unchanged_five_member_action_conditioned_margin_MLP",
        "base_model_prediction": "ensemble_mean_exact_margin",
        "dangerous_residual": "predicted_margin_minus_exact_OSC_margin",
        "expected_residual_count": 52500,
    }:
        raise ValueError("local-residual LOO contract differs")
    if config["local_features"] != {
        "context": (
            "validated_53D_pair_features_excluding_constraint_one_hot_and_"
            "candidate_action"
        ),
        "context_dimension": 43,
        "candidate": "normalized_candidate_xyz_within_state_action_box",
        "candidate_dimension": 3, "input_dimension": len(LOCAL_FEATURE_NAMES),
        "normalization": (
            "leave_one_episode_out_training_residual_pool_mean_std_with_1e-6_floor"
        ),
        "constraint_handling": (
            "separate_neighbor_pool_per_each_of_seven_constraints"
        ),
    }:
        raise ValueError("local-residual feature contract differs")
    if config["local_bound"] != {
        "estimator": "fixed_k_nearest_neighbor_upper_residual",
        "distance": "RMS_standardized_local_feature_distance",
        "neighbor_count": 256, "quantile_probability": 0.999,
        "finite_sample_quantile": (
            "higher_order_statistic_equals_neighbor_maximum"
        ),
        "minimum_quantile_m": 0.0, "fixed_padding_m": 0.001,
        "validation_labels_used_for_adaptation": False,
        "test_labels_used_for_adaptation": False,
        "conservative_margin": (
            "ensemble_mean_margin_minus_local_upper_residual_minus_fixed_padding"
        ),
    }:
        raise ValueError("local-residual bound differs")
    if config["decision"] != {
        "closed_loop_in_this_gate": False,
        "if_every_gate_passes": "preregister_receding_QP_closed_loop_E05",
        "if_gate_fails": (
            "retain_execution_aware_direction_as_unvalidated_and_stop_before_"
            "closed_loop"
        ),
    }:
        raise ValueError("local-residual decision differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def local_feature_matrix(
    state: Mapping[str, Any], actions_xyz: Sequence[Sequence[float]],
) -> Any:
    """Return [action, constraint, 46] local distance features."""

    np = _numpy()
    actions = np.asarray(actions_xyz, dtype=np.float64)
    pair = np.asarray(state["pair_state_feature_vectors"], dtype=np.float64)
    lower = np.asarray(state["action_lower"], dtype=np.float64)
    upper = np.asarray(state["action_upper"], dtype=np.float64)
    indexes = context_feature_indexes()
    if (
        actions.ndim != 2 or actions.shape[1] != 3 or pair.shape[0] != 7
        or lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower)
    ):
        raise ValueError("local-residual feature input differs")
    normalized = 2.0 * (actions - lower[None, :]) / (
        upper - lower
    )[None, :] - 1.0
    if np.any(normalized < -1.0 - 1.0e-10) or np.any(normalized > 1.0 + 1.0e-10):
        raise ValueError("local-residual candidate lies outside action box")
    context = pair[:, indexes]
    output = np.concatenate((
        np.repeat(context[None, :, :], len(actions), axis=0),
        np.repeat(normalized[:, None, :], 7, axis=1),
    ), axis=2)
    if (
        output.shape != (len(actions), 7, len(LOCAL_FEATURE_NAMES))
        or not np.all(np.isfinite(output))
    ):
        raise ValueError("local-residual feature output differs")
    return output


def build_pool(
    features: Sequence[Sequence[float]], residual_m: Sequence[float],
    constraint_index: Sequence[int],
) -> dict[str, Any]:
    np = _numpy()
    x = np.asarray(features, dtype=np.float64)
    residual = np.asarray(residual_m, dtype=np.float64)
    constraint = np.asarray(constraint_index, dtype=np.int64)
    if (
        x.ndim != 2 or x.shape[1] != len(LOCAL_FEATURE_NAMES)
        or residual.shape != (len(x),) or constraint.shape != (len(x),)
        or set(constraint.tolist()) != set(range(7))
    ):
        raise ValueError("local-residual pool differs")
    mean = np.mean(x, axis=0)
    std = np.maximum(np.std(x, axis=0), 1.0e-6)
    normalized = (x - mean[None, :]) / std[None, :]
    return {
        "features": x, "normalized_features": normalized,
        "residual_m": residual, "constraint_index": constraint,
        "feature_mean": mean, "feature_standard_deviation": std,
    }


def query_upper_bound(
    pool: Mapping[str, Any], query_features: Any,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Fixed-k per-constraint nearest-neighbor maximum dangerous residual."""

    np = _numpy()
    query = np.asarray(query_features, dtype=np.float64)
    if query.ndim != 3 or query.shape[1:] != (7, len(LOCAL_FEATURE_NAMES)):
        raise ValueError("local-residual query shape differs")
    normalized = (
        query - np.asarray(pool["feature_mean"], dtype=np.float64)[None, None, :]
    ) / np.asarray(
        pool["feature_standard_deviation"], dtype=np.float64
    )[None, None, :]
    pool_x = np.asarray(pool["normalized_features"], dtype=np.float64)
    residual = np.asarray(pool["residual_m"], dtype=np.float64)
    constraints = np.asarray(pool["constraint_index"], dtype=np.int64)
    k = int(settings["neighbor_count"])
    upper = np.empty((len(query), 7), dtype=np.float64)
    maximum_distance = np.empty((len(query), 7), dtype=np.float64)
    for row in range(7):
        mask = constraints == row
        candidates = pool_x[mask]
        candidate_residual = residual[mask]
        if len(candidates) < k:
            raise ValueError("local-residual neighbor pool is too small")
        difference = normalized[:, row, None, :] - candidates[None, :, :]
        squared = np.mean(difference * difference, axis=2)
        nearest = np.argpartition(squared, k - 1, axis=1)[:, :k]
        selected_residual = candidate_residual[nearest]
        selected_distance = np.sqrt(np.take_along_axis(squared, nearest, axis=1))
        upper[:, row] = np.maximum(
            float(settings["minimum_quantile_m"]),
            np.max(selected_residual, axis=1),
        ) + float(settings["fixed_padding_m"])
        maximum_distance[:, row] = np.max(selected_distance, axis=1)
    return {
        "upper_error_m": upper,
        "maximum_neighbor_distance_rms_z": maximum_distance,
    }


def local_lower_values(
    models: Sequence[Any], model_state: Mapping[str, Any],
    state: Mapping[str, Any], actions_xyz: Sequence[Sequence[float]],
    pool: Mapping[str, Any], settings: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    members = predict_member_margins(
        models, model_state, state, actions_xyz
    )
    mean = np.mean(members, axis=0)
    query = local_feature_matrix(state, actions_xyz)
    bound = query_upper_bound(pool, query, settings)
    return {
        "mean_m": mean,
        "upper_error_m": bound["upper_error_m"],
        "lower_m": mean - bound["upper_error_m"],
        "maximum_neighbor_distance_rms_z": bound[
            "maximum_neighbor_distance_rms_z"
        ],
    }


def learned_regional_targets(
    models: Sequence[Any], model_state: Mapping[str, Any],
    state: Mapping[str, Any], regions: Sequence[Mapping[str, Any]],
    pool: Mapping[str, Any], config: Mapping[str, Any],
    ridge_huber: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    np = _numpy()
    xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
    values = local_lower_values(
        models, model_state, state, xyz, pool, config["local_bound"]
    )
    targets = []
    for region in regions:
        fitted = fit_region_target(
            xyz, values["lower_m"], region, ridge_huber
        )
        fitted["lower_anchor_m"] = (
            np.asarray(fitted["anchor_margin_m"], dtype=np.float64)
            - np.asarray(fitted["one_sided_error_m"], dtype=np.float64)
        ).tolist()
        fitted["value_source"] = "LOO_local_upper_residual_bound"
        targets.append(fitted)
    return targets, values


def target_values(target: Mapping[str, Any], xyz: Sequence[float]) -> Any:
    return region_affine_values(target, xyz)


def save_oof_artifact(
    path: Path, *, features: Any, residual_m: Any, state_index: Any,
    action_index: Any, constraint_index: Any, case_index: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    arrays = {
        "features": np.asarray(features, dtype=np.float64),
        "residual_m": np.asarray(residual_m, dtype=np.float64),
        "state_index": np.asarray(state_index, dtype=np.int64),
        "action_index": np.asarray(action_index, dtype=np.int64),
        "constraint_index": np.asarray(constraint_index, dtype=np.int64),
        "case_index": np.asarray(case_index, dtype=np.int64),
        "metadata_utf8": np.frombuffer(_canonical(metadata), dtype=np.uint8),
    }
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    raw = path.read_bytes()
    return {"path": str(path), "file_sha256": _sha256(raw), "size_bytes": len(raw)}


__all__ = [
    "CONFIG_SCHEMA", "LOCAL_FEATURE_NAMES", "OOF_SCHEMA", "RESULT_SCHEMA",
    "VALIDATION_SCHEMA", "build_pool", "learned_regional_targets",
    "load_config", "local_feature_matrix", "local_lower_values",
    "query_upper_bound", "save_oof_artifact", "solve_regional_qps",
    "target_values",
]
