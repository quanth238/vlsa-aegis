"""Matched old-56D versus complete-OSC input ablation.

Both arms consume the same immutable rollout-margin rows.  Only the input
projection differs.  This module deliberately contains no calibration, QP,
simulation, or closed-loop control path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_conditioned_margin import ACTION_FEATURE_NAMES, action_feature_matrix
from .complete_osc_margin import (
    DATASET_SCHEMA, _canonical, _numpy, load_weights, payload_sha256, predict,
    save_weights, train_ensemble, training_arrays,
)
from .two_step_margin import PAIR_FEATURE_NAMES


CONFIG_SCHEMA = "vlsa_distal_matched_input_ablation_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_matched_input_ablation_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_matched_input_ablation_moka10_validation.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("matched-input config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "split", "paired_arms", "model", "training", "prediction_gate",
        "interpretation", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("matched-input config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-matched-input-ablation-moka10-v1"
    ):
        raise ValueError("matched-input protocol differs")
    source = config["immutable_source"]
    if set(source) != {
        "dataset_file_sha256", "dataset_payload_sha256",
        "collection_result_file_sha256", "collection_result_payload_sha256",
        "dataset_schema_version", "expected_state_count",
        "expected_candidate_count", "expected_row_count",
    }:
        raise ValueError("matched-input immutable source differs")
    if (
        source["dataset_schema_version"] != DATASET_SCHEMA
        or int(source["expected_state_count"]) != 85
        or int(source["expected_candidate_count"]) != 10625
        or int(source["expected_row_count"]) != 74375
    ):
        raise ValueError("matched-input immutable population differs")
    if config["split"] != {
        "unit": "complete_episode",
        "expected_state_counts": {"train": 60, "validation": 10, "test": 15},
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "test_only": True,
    }:
        raise ValueError("matched-input grouped split differs")
    arms = config["paired_arms"]
    if arms != {
        "old56": {
            "dimension": 56,
            "projection": "fresh_q_qdot_relative_proxy_geometry_and_XYZ_action",
            "action": "first_translation_and_fixed_second_translation",
        },
        "completeOSC": {
            "dimension": 2131,
            "projection": "complete_serialized_OSC_snapshot_and_semantic_geometry",
            "action": "full_two_action_translation_rotation_gripper_commands",
        },
        "same_rows_labels_splits_architecture_loss_seeds_schedule": True,
        "old56_geometry_reconstruction_tolerance_m": 1.0e-10,
    }:
        raise ValueError("matched-input arm contract differs")
    if config["model"] != {
        "class": "shared_constraint_action_conditioned_residual_MLP_ensemble",
        "ensemble_seeds": [20260831, 20260832, 20260833, 20260834, 20260835],
        "hidden_widths": [256, 256, 128], "activation": "silu",
        "output_count": 1,
    }:
        raise ValueError("matched-input model differs")
    if config["training"] != {
        "device": "cpu_inside_H100_allocation",
        "batch_size": 4096, "epochs": 500, "patience": 60,
        "learning_rate": 0.001, "weight_decay": 1.0e-6,
        "huber_delta_mm": 2.0, "boundary_band_m": 0.005,
        "boundary_weight_multiplier": 9.0,
        "one_sided_overestimate_loss_weight": 4.0,
        "early_stopping_split": "validation_complete_episodes",
        "normalization": "training_rows_mean_std_with_1e-6_floor",
    }:
        raise ValueError("matched-input training differs")
    if config["prediction_gate"] != {
        "proxy_false_safe_action_count": 0,
        "minimum_exact_safe_action_recall": 0.5,
        "required_test_state_safe_support_count": 15,
        "maximum_near_boundary_RMSE_m": 0.002,
        "near_boundary_absolute_margin_m": 0.005,
        "maximum_overall_RMSE_m": 0.003808,
        "training_fit_requires_overall_and_near_boundary_gates": True,
    }:
        raise ValueError("matched-input prediction gate differs")
    if config["interpretation"] != {
        "complete_input_root_cause": "completeOSC_test_pass_and_old56_test_fail",
        "state_setup_coverage_failure": "both_train_fit_and_both_test_fail",
        "target_or_model_representation_failure": "both_train_fail",
        "otherwise": "mixed_or_inconclusive",
    }:
        raise ValueError("matched-input interpretation differs")
    if config["forbidden_actions"] != {
        "additional_simulation": True, "new_data_collection": True,
        "poisson_fields": True, "calibration": True, "QP": True,
        "closed_loop_E05": True,
    }:
        raise ValueError("matched-input forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _support_radius(
    rotation: Any, size: Any, direction_world: Any, *, box: bool,
) -> float:
    np = _numpy()
    local = np.asarray(rotation, dtype=np.float64).T @ np.asarray(
        direction_world, dtype=np.float64
    )
    extent = np.asarray(size, dtype=np.float64)
    return float(
        np.sum(extent * np.abs(local))
        if box else np.sqrt(np.sum((extent * local) ** 2))
    )


def _old_pair_state_features(
    state: Mapping[str, Any], tolerance_m: float,
) -> tuple[Any, Any, Any, float]:
    """Reconstruct the historical 53D pair rows from fresh semantics."""

    np = _numpy()
    semantic = state["semantic_state"]
    q = np.asarray(semantic["q_rad"], dtype=np.float64)
    qdot = np.asarray(semantic["qdot_rad_s"], dtype=np.float64)
    goal = np.asarray(semantic["goal_EE_position_m"], dtype=np.float64)
    links = semantic["seven_robot_ellipsoid_transforms"]
    obstacles = semantic["exact_obstacle_primitive_transforms"]
    current = np.asarray(state["current_ellipsoid_margin_m"], dtype=np.float64)
    if (
        q.shape != (7,) or qdot.shape != (7,) or goal.shape != (3,)
        or len(links) != 7 or not obstacles or current.shape != (7,)
    ):
        raise ValueError("old56 fresh semantic state differs")
    pair_prefixes = []
    reconstructed = []
    for row, link in enumerate(links):
        link_center = np.asarray(link["center_m"], dtype=np.float64)
        link_rotation = np.asarray(
            link["rotation_world_from_local"], dtype=np.float64
        )
        link_size = np.asarray(
            link["semiaxes_or_half_size_m"], dtype=np.float64
        )
        gaps = []
        for obstacle in obstacles:
            obstacle_center = np.asarray(obstacle["center_m"], dtype=np.float64)
            displacement = obstacle_center - link_center
            distance = float(np.linalg.norm(displacement))
            if not math.isfinite(distance) or distance <= 1.0e-12:
                raise ValueError("old56 proxy centers differ")
            direction = displacement / distance
            gaps.append(
                distance
                - _support_radius(link_rotation, link_size, direction, box=False)
                - _support_radius(
                    obstacle["rotation_world_from_local"],
                    obstacle["semiaxes_or_half_size_m"], direction, box=True,
                )
            )
        witness = int(np.argmin(np.asarray(gaps, dtype=np.float64)))
        reconstructed.append(float(gaps[witness]))
        obstacle = obstacles[witness]
        obstacle_center = np.asarray(obstacle["center_m"], dtype=np.float64)
        obstacle_rotation = np.asarray(
            obstacle["rotation_world_from_local"], dtype=np.float64
        )
        relative_center = link_rotation.T @ (obstacle_center - link_center)
        relative_rotation = link_rotation.T @ obstacle_rotation
        goal_local = link_rotation.T @ (goal - link_center)
        one_hot = np.zeros(7, dtype=np.float64)
        one_hot[row] = 1.0
        pair_prefixes.append(np.concatenate((
            q, qdot, goal_local, relative_center, relative_rotation.reshape(-1),
            link_size,
            np.asarray(obstacle["semiaxes_or_half_size_m"], dtype=np.float64),
            np.asarray([
                current[row], np.linalg.norm(obstacle_center - link_center),
            ], dtype=np.float64),
            one_hot,
        )))
    reconstructed_array = np.asarray(reconstructed, dtype=np.float64)
    maximum_error = float(np.max(np.abs(reconstructed_array - current)))
    if maximum_error > float(tolerance_m):
        raise ValueError(
            "old56 reconstructed fresh clearance differs: %.17g" % maximum_error
        )
    prefixes = np.asarray(pair_prefixes, dtype=np.float64)
    nominal_first = np.asarray(state["nominal_first_action"], dtype=np.float64)[:3]
    nominal_second = np.asarray(state["nominal_second_action"], dtype=np.float64)[:3]
    pair_state = np.concatenate((
        prefixes,
        np.repeat(np.concatenate((
            nominal_first, nominal_first, nominal_second,
        ))[None, :], 7, axis=0),
    ), axis=1)
    if pair_state.shape != (7, len(PAIR_FEATURE_NAMES)):
        raise ValueError("old56 pair feature shape differs")
    return pair_state, nominal_first, nominal_second, maximum_error


def old56_training_arrays(
    dataset: Mapping[str, Any], tolerance_m: float = 1.0e-10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project fresh state/action records into exactly 56 historical inputs."""

    np = _numpy()
    features = []
    margins = []
    current_values = []
    splits = []
    state_indexes = []
    action_indexes = []
    constraint_indexes = []
    maximum_reconstruction_error = 0.0
    for state in dataset["state_records"]:
        pair_state, _, _, reconstruction_error = _old_pair_state_features(
            state, tolerance_m
        )
        maximum_reconstruction_error = max(
            maximum_reconstruction_error, reconstruction_error
        )
        current = np.asarray(state["current_ellipsoid_margin_m"], dtype=np.float64)
        action_xyz = np.asarray([
            candidate["full_two_action_commands"][0][:3]
            for candidate in state["candidates"]
        ], dtype=np.float64)
        lower = np.min(action_xyz, axis=0)
        upper = np.max(action_xyz, axis=0)
        if np.any(upper <= lower):
            raise ValueError("old56 candidate action box differs")
        for action_index, candidate in enumerate(state["candidates"]):
            xyz = np.asarray(
                candidate["full_two_action_commands"][0][:3], dtype=np.float64
            )
            rows = action_feature_matrix(pair_state, xyz, lower, upper)
            exact = np.asarray(
                candidate["rollout_minimum_ellipsoid_margin_m"], dtype=np.float64
            )
            for constraint_index in range(7):
                features.append(rows[constraint_index])
                margins.append(exact[constraint_index])
                current_values.append(current[constraint_index])
                splits.append(state["split"])
                state_indexes.append(int(state["state_index"]))
                action_indexes.append(int(candidate["candidate_index"]))
                constraint_indexes.append(constraint_index)
    output = {
        "features": np.asarray(features, dtype=np.float64),
        "margin_m": np.asarray(margins, dtype=np.float64),
        "current_margin_m": np.asarray(current_values, dtype=np.float64),
        "split": np.asarray(splits, dtype=object),
        "state_index": np.asarray(state_indexes, dtype=np.int64),
        "action_index": np.asarray(action_indexes, dtype=np.int64),
        "constraint_index": np.asarray(constraint_indexes, dtype=np.int64),
    }
    expected = int(dataset["summary"]["pair_count"]) * 7
    if (
        output["features"].shape != (expected, len(ACTION_FEATURE_NAMES))
        or any(output[key].shape != (expected,) for key in (
            "margin_m", "current_margin_m", "split", "state_index",
            "action_index", "constraint_index",
        ))
        or not np.all(np.isfinite(output["features"]))
    ):
        raise ValueError("old56 training arrays differ")
    return output, {
        "input_dimension": int(output["features"].shape[1]),
        "maximum_fresh_clearance_reconstruction_error_m": (
            maximum_reconstruction_error
        ),
    }


def paired_row_receipt(
    old_arrays: Mapping[str, Any], complete_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    exact_keys = (
        "margin_m", "current_margin_m", "state_index", "action_index",
        "constraint_index",
    )
    equal = all(np.array_equal(old_arrays[key], complete_arrays[key]) for key in exact_keys)
    equal = bool(equal and np.array_equal(
        np.asarray(old_arrays["split"], dtype=str),
        np.asarray(complete_arrays["split"], dtype=str),
    ))
    identity = np.column_stack((
        np.asarray(old_arrays["state_index"], dtype=np.int64),
        np.asarray(old_arrays["action_index"], dtype=np.int64),
        np.asarray(old_arrays["constraint_index"], dtype=np.int64),
    ))
    labels = np.column_stack((
        np.asarray(old_arrays["current_margin_m"], dtype=np.float64),
        np.asarray(old_arrays["margin_m"], dtype=np.float64),
    ))
    return {
        "exact_row_identity_and_label_match": equal,
        "row_count": int(labels.shape[0]),
        "row_identity_sha256": _sha256(identity.tobytes()),
        "current_and_target_margin_sha256": _sha256(labels.tobytes()),
    }


def split_metrics(
    arrays: Mapping[str, Any], prediction_m: Sequence[float],
    config: Mapping[str, Any], split_name: str,
) -> dict[str, Any]:
    np = _numpy()
    exact = np.asarray(arrays["margin_m"], dtype=np.float64)
    prediction = np.asarray(prediction_m, dtype=np.float64)
    split = np.asarray(arrays["split"], dtype=object)
    selected = split == split_name
    if prediction.shape != exact.shape or not np.any(selected):
        raise ValueError("matched-input prediction population differs")
    state_indexes = np.asarray(arrays["state_index"], dtype=np.int64)
    action_indexes = np.asarray(arrays["action_index"], dtype=np.int64)
    states = sorted(set(state_indexes[selected].tolist()))
    false_safe = accepted_safe = exact_safe = support = action_count = 0
    for state_index in states:
        state_mask = selected & (state_indexes == state_index)
        actions = sorted(set(action_indexes[state_mask].tolist()))
        action_count += len(actions)
        accepted_in_state = 0
        for action_index in actions:
            mask = state_mask & (action_indexes == action_index)
            predicted_safe = bool(np.all(prediction[mask] >= 0.0))
            truly_safe = bool(np.all(exact[mask] >= 0.0))
            false_safe += int(predicted_safe and not truly_safe)
            accepted_safe += int(predicted_safe and truly_safe)
            accepted_in_state += int(predicted_safe and truly_safe)
            exact_safe += int(truly_safe)
        support += int(accepted_in_state > 0)
    error = prediction[selected] - exact[selected]
    gate = config["prediction_gate"]
    boundary = float(gate["near_boundary_absolute_margin_m"])
    near = selected & (np.abs(exact) <= boundary)
    near_rmse = (
        float(np.sqrt(np.mean((prediction[near] - exact[near]) ** 2)))
        if np.any(near) else None
    )
    metrics = {
        "split": split_name, "state_count": len(states),
        "action_count": int(action_count),
        "row_count": int(np.count_nonzero(selected)),
        "proxy_false_safe_action_count": int(false_safe),
        "exact_safe_action_count": int(exact_safe),
        "accepted_exact_safe_action_count": int(accepted_safe),
        "exact_safe_action_recall": (
            0.0 if exact_safe == 0 else float(accepted_safe / exact_safe)
        ),
        "state_safe_support_count": int(support),
        "overall_RMSE_m": float(np.sqrt(np.mean(error ** 2))),
        "near_boundary_row_count": int(np.count_nonzero(near)),
        "near_boundary_RMSE_m": near_rmse,
    }
    metrics["fit_gate_pass"] = bool(
        metrics["overall_RMSE_m"] <= float(gate["maximum_overall_RMSE_m"])
        and near_rmse is not None
        and near_rmse <= float(gate["maximum_near_boundary_RMSE_m"])
    )
    metrics["full_prediction_gate_pass"] = bool(
        metrics["fit_gate_pass"]
        and metrics["proxy_false_safe_action_count"]
        == int(gate["proxy_false_safe_action_count"])
        and metrics["exact_safe_action_recall"]
        >= float(gate["minimum_exact_safe_action_recall"])
        and (
            split_name != "test"
            or metrics["state_safe_support_count"]
            == int(gate["required_test_state_safe_support_count"])
        )
    )
    return metrics


def matched_decision(arm_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    old_train = bool(arm_metrics["old56"]["train"]["fit_gate_pass"])
    complete_train = bool(arm_metrics["completeOSC"]["train"]["fit_gate_pass"])
    old_test = bool(arm_metrics["old56"]["test"]["full_prediction_gate_pass"])
    complete_test = bool(
        arm_metrics["completeOSC"]["test"]["full_prediction_gate_pass"]
    )
    if complete_test and not old_test:
        conclusion = "missing_inputs_were_root_cause"
    elif old_train and complete_train and not old_test and not complete_test:
        conclusion = "insufficient_state_or_setup_coverage"
    elif not old_train and not complete_train:
        conclusion = "output_target_or_plain_MLP_representation_problem"
    else:
        conclusion = "mixed_or_inconclusive"
    return {
        "old56_training_fit": old_train,
        "completeOSC_training_fit": complete_train,
        "old56_test_gate_pass": old_test,
        "completeOSC_test_gate_pass": complete_test,
        "conclusion": conclusion,
        "calibration_QP_or_closed_loop_authorized": False,
    }


def load_arm_weights(path: Path) -> tuple[list[Any], dict[str, Any]]:
    return load_weights(path)


def save_arm_weights(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    return save_weights(path, state)
