"""Fixed joint-mask confirmation for duplicated obstacle orientation inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import _canonical, _numpy
from .factorized_execution_pilot import dataset_arrays, load_weights, predict
from .factorized_input_representation_audit import (
    _predict_flat_normalized, _predict_time_normalized, model_feature_names,
    semantic_group, terminal_rmse,
)
from .factorized_time_conditioned_decoder import (
    load_time_weights, predict_time_conditioned,
)


CONFIG_SCHEMA = "vlsa_distal_factorized_orientation_joint_mask_config.v1"
RESULT_SCHEMA = "vlsa_distal_factorized_orientation_joint_mask_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_orientation_joint_mask_validation.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha256(_canonical(payload))


def load_joint_mask_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "fixed_masks", "evaluation", "decision", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("orientation joint-mask config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-orientation-joint-mask-moka10-v1"
    ):
        raise ValueError("orientation joint-mask protocol differs")
    if config["fixed_masks"] != {
        "primitive_rotation_only": [
            "semantic_state.exact_obstacle_primitive_transforms[].rotation_world_from_local[][]"
        ],
        "root_rotation_only": [
            "semantic_state.obstacle_root_orientation_world_from_body[][]"
        ],
        "joint_redundant_orientation": [
            "semantic_state.exact_obstacle_primitive_transforms[].rotation_world_from_local[][]",
            "semantic_state.obstacle_root_orientation_world_from_body[][]",
        ],
        "all_variance_floored_features": ["__all_training_raw_std_below_1e-6__"],
    }:
        raise ValueError("orientation joint-mask definitions differ")
    if not all(config["forbidden_actions"].values()):
        raise ValueError("orientation joint-mask forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _selected_rows(arrays: Mapping[str, Any], split_name: str) -> Any:
    np = _numpy()
    return np.flatnonzero(
        (np.asarray(arrays["split"], dtype=object) == split_name)
        & (np.asarray(arrays["source_code"], dtype=np.int8) == 2)
    )


def _prediction_metrics(predicted: Any, exact: Any) -> dict[str, float]:
    np = _numpy()
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    return {
        "terminal_joint_RMSE_rad": float(np.sqrt(np.mean(error[:, -1] ** 2))),
        "overall_joint_RMSE_rad": float(np.sqrt(np.mean(error ** 2))),
        "maximum_absolute_terminal_joint_error_rad": float(
            np.max(np.abs(error[:, -1]))
        ),
    }


def _mask_indexes(
    mask_groups: Sequence[str], groups: Any, raw_std: Any, floor: float,
) -> Any:
    np = _numpy()
    if list(mask_groups) == ["__all_training_raw_std_below_1e-6__"]:
        return np.flatnonzero(np.asarray(raw_std, dtype=np.float64) < floor)
    wanted = set(mask_groups)
    return np.flatnonzero(np.asarray([group in wanted for group in groups], dtype=bool))


def run_joint_mask_confirmation(
    *, config: Mapping[str, Any], audit_config: Mapping[str, Any],
    complete_dataset: Mapping[str, Any], metadata: Mapping[str, Any],
    archive: Mapping[str, Any], flat_model_path: Path, flat_predictions_path: Path,
    time_model_path: Path, time_predictions_path: Path,
) -> dict[str, Any]:
    np = _numpy()
    arrays = dataset_arrays(metadata, archive)
    names = model_feature_names(complete_dataset)
    groups = np.asarray([semantic_group(name) for name in names], dtype=object)
    flat_models, flat_state = load_weights(flat_model_path)
    time_models, time_state = load_time_weights(time_model_path)
    flat_prediction = predict(flat_models, flat_state, arrays)
    time_prediction = predict_time_conditioned(time_models, time_state, arrays)
    stored_flat = np.load(flat_predictions_path, allow_pickle=False)[
        "experimental_joint_position_rad"
    ].astype(np.float64)
    stored_time = np.load(time_predictions_path, allow_pickle=False)[
        "experimental_joint_position_rad"
    ].astype(np.float64)
    if not (
        np.array_equal(flat_prediction, stored_flat)
        and np.array_equal(time_prediction, stored_time)
    ):
        raise ValueError("orientation joint-mask immutable prediction differs")
    x = np.asarray(arrays["features"], dtype=np.float64)
    split = np.asarray(arrays["split"], dtype=object)
    train = split == "train"
    mean = np.mean(x[train], axis=0)
    raw_std = np.std(x[train], axis=0)
    floor = float(audit_config["normalization_audit"]["standard_deviation_floor"])
    std = np.maximum(raw_std, floor)
    if not (
        np.array_equal(mean, flat_state["feature_mean"])
        and np.array_equal(std, flat_state["feature_std"])
        and np.array_equal(mean, time_state["feature_mean"])
        and np.array_equal(std, time_state["feature_std"])
    ):
        raise ValueError("orientation joint-mask normalization differs")
    normalized = (x - mean) / std
    exact_q = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
    state_to_case = {
        int(record["state_index"]): str(record["case_id"])
        for record in metadata["state_records"]
    }
    states = np.asarray(arrays["state_index"], dtype=np.int64)
    model_full = {"flat": flat_prediction, "time": time_prediction}
    full = {}
    for split_name in ("validation", "test"):
        rows = _selected_rows(arrays, split_name)
        full[split_name] = {
            model: _prediction_metrics(prediction[rows], exact_q[rows])
            for model, prediction in model_full.items()
        }
    case_full = {}
    for case_id in (
        config["evaluation"]["validation_explosion_case_id"],
        config["evaluation"]["ordinary_validation_case_id"],
    ):
        rows = np.flatnonzero(
            (split == "validation")
            & (np.asarray(arrays["source_code"], dtype=np.int8) == 2)
            & np.asarray([state_to_case[int(state)] == case_id for state in states])
        )
        case_full[case_id] = {
            model: _prediction_metrics(prediction[rows], exact_q[rows])
            for model, prediction in model_full.items()
        }
    masks = {}
    for mask_name, mask_groups in config["fixed_masks"].items():
        indexes = _mask_indexes(mask_groups, groups, raw_std, floor)
        if not len(indexes):
            raise ValueError("orientation joint-mask selected no features")
        masked_normalized = normalized.copy()
        masked_normalized[:, indexes] = 0.0
        masked_predictions = {}
        for split_name in ("validation", "test"):
            rows = _selected_rows(arrays, split_name)
            exact = exact_q[rows]
            base = np.repeat(exact[:, :1], 51, axis=1)
            masked_predictions[split_name] = {
                "flat": _predict_flat_normalized(
                    flat_models, masked_normalized[rows], base
                ),
                "time": _predict_time_normalized(
                    time_models, masked_normalized[rows], base
                ),
            }
        record = {
            "semantic_groups": list(mask_groups),
            "feature_count": int(len(indexes)),
            "split_metrics": {}, "validation_case_metrics": {},
        }
        for split_name in ("validation", "test"):
            rows = _selected_rows(arrays, split_name)
            record["split_metrics"][split_name] = {
                model: _prediction_metrics(prediction, exact_q[rows])
                for model, prediction in masked_predictions[split_name].items()
            }
        validation_rows = _selected_rows(arrays, "validation")
        validation_states = states[validation_rows]
        for case_id in case_full:
            local = np.flatnonzero(np.asarray([
                state_to_case[int(state)] == case_id for state in validation_states
            ]))
            record["validation_case_metrics"][case_id] = {
                model: _prediction_metrics(prediction[local], exact_q[validation_rows][local])
                for model, prediction in masked_predictions["validation"].items()
            }
        masks[mask_name] = record
    joint = masks["joint_redundant_orientation"]
    explosion_case = config["evaluation"]["validation_explosion_case_id"]
    ordinary_case = config["evaluation"]["ordinary_validation_case_id"]
    reductions = {}
    test_increases = {}
    for model in ("flat", "time"):
        before = case_full[explosion_case][model]["terminal_joint_RMSE_rad"]
        after = joint["validation_case_metrics"][explosion_case][model][
            "terminal_joint_RMSE_rad"
        ]
        reductions[model] = float(1.0 - after / max(before, 1.0e-15))
        test_before = full["test"][model]["terminal_joint_RMSE_rad"]
        test_after = joint["split_metrics"]["test"][model]["terminal_joint_RMSE_rad"]
        test_increases[model] = float(test_after / max(test_before, 1.0e-15) - 1.0)
    threshold = config["evaluation"]
    gate_tests = {
        "flat_explosion_reduction": reductions["flat"] >= float(threshold[
            "minimum_joint_mask_terminal_RMSE_reduction_fraction_both_models"
        ]),
        "time_explosion_reduction": reductions["time"] >= float(threshold[
            "minimum_joint_mask_terminal_RMSE_reduction_fraction_both_models"
        ]),
        "flat_ordinary_validation_remains_physical": joint[
            "validation_case_metrics"
        ][ordinary_case]["flat"]["terminal_joint_RMSE_rad"] <= float(threshold[
            "maximum_ordinary_validation_terminal_RMSE_rad_after_joint_mask"
        ]),
        "time_ordinary_validation_remains_physical": joint[
            "validation_case_metrics"
        ][ordinary_case]["time"]["terminal_joint_RMSE_rad"] <= float(threshold[
            "maximum_ordinary_validation_terminal_RMSE_rad_after_joint_mask"
        ]),
        "flat_test_not_materially_harmed": test_increases["flat"] <= float(
            threshold["maximum_test_terminal_RMSE_relative_increase"]
        ),
        "time_test_not_materially_harmed": test_increases["time"] <= float(
            threshold["maximum_test_terminal_RMSE_relative_increase"]
        ),
    }
    passed = bool(all(gate_tests.values()))
    return {
        "immutable_prediction_replay_exact": True,
        "normalization_replay_exact": True,
        "unmasked_metrics": {"split": full, "validation_case": case_full},
        "fixed_mask_metrics": masks,
        "joint_mask_explosion_RMSE_reduction_fraction": reductions,
        "joint_mask_test_RMSE_relative_increase": test_increases,
        "decision": {
            "gate_tests": gate_tests,
            "orientation_representation_causal": passed,
            "conclusion": (
                config["decision"]["pass_interpretation"] if passed
                else config["decision"]["fail_interpretation"]
            ),
            "matched_representation_ablation_authorized": passed,
            "training_directly_authorized": False,
        },
    }
