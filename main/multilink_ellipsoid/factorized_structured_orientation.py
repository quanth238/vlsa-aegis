"""Structured bounded orientation input for factorized OSC execution models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import _canonical, _numpy
from .factorized_input_representation_audit import (
    model_feature_names, semantic_group,
)


CONFIG_SCHEMA = "vlsa_distal_factorized_structured_orientation_config.v1"
RESULT_SCHEMA = "vlsa_distal_factorized_structured_orientation_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_structured_orientation_validation.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha256(_canonical(payload))


def load_structured_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "structured_representation", "matched_arms",
        "decision_gate", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("structured-orientation config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-structured-orientation-moka10-v1"
    ):
        raise ValueError("structured-orientation protocol differs")
    representation = config["structured_representation"]
    if (
        int(representation["original_input_dimension"]) != 2124
        or int(representation["removed_scalar_count"]) != 288
        or int(representation["appended_dimension"]) != 6
        or int(representation["structured_input_dimension"]) != 1842
    ):
        raise ValueError("structured-orientation dimensions differ")
    if config["forbidden_actions"] != {
        "new_simulation_labels": True, "calibration": True,
        "candidate_expansion": True, "QP": True, "closed_loop": True,
        "poisson_or_SDF": True,
    }:
        raise ValueError("structured-orientation forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _root_rotation_indexes(names: Sequence[str]) -> list[int]:
    indexes = []
    for column in (0, 1):
        for row in (0, 1, 2):
            target = (
                "semantic_state.obstacle_root_orientation_world_from_body"
                "[%d][%d].value" % (row, column)
            )
            matches = [index for index, name in enumerate(names) if name == target]
            if len(matches) != 1:
                raise ValueError("structured root rotation feature differs")
            indexes.append(matches[0])
    return indexes


def structured_orientation_arrays(
    complete_dataset: Mapping[str, Any], arrays: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replace duplicated rotations and define train-only bounded normalization."""

    np = _numpy()
    names = model_feature_names(complete_dataset)
    x = np.asarray(arrays["features"], dtype=np.float64)
    split = np.asarray(arrays["split"], dtype=object)
    train = split == "train"
    removed_groups = set(config["structured_representation"]["removed_groups"])
    groups = [semantic_group(name) for name in names]
    removed = np.asarray([
        index for index, group in enumerate(groups) if group in removed_groups
    ], dtype=np.int64)
    kept = np.asarray([
        index for index, group in enumerate(groups) if group not in removed_groups
    ], dtype=np.int64)
    if len(removed) != int(config["structured_representation"][
        "removed_scalar_count"
    ]):
        raise ValueError("structured-orientation removed scalar count differs")
    root_indexes = _root_rotation_indexes(names)
    root_6d = x[:, root_indexes]
    if np.max(np.abs(root_6d)) > 1.0 + 1.0e-9:
        raise ValueError("structured root rotation is outside the unit bound")
    # The six coordinates are the first two columns of R. Reconstruct the
    # third by cross product only for the representation certificate.
    first = root_6d[:, :3]
    second = root_6d[:, 3:]
    third = np.cross(first, second)
    orthogonality_error = float(max(
        np.max(np.abs(np.sum(first * second, axis=1))),
        np.max(np.abs(np.linalg.norm(first, axis=1) - 1.0)),
        np.max(np.abs(np.linalg.norm(second, axis=1) - 1.0)),
        np.max(np.abs(np.linalg.norm(third, axis=1) - 1.0)),
    ))
    if orthogonality_error > 1.0e-6:
        raise ValueError("structured root rotation 6D certificate differs")
    structured = np.concatenate((x[:, kept], root_6d), axis=1)
    structured_names = [names[index] for index in kept] + [
        "structured.root_rotation_6D_column0_x",
        "structured.root_rotation_6D_column0_y",
        "structured.root_rotation_6D_column0_z",
        "structured.root_rotation_6D_column1_x",
        "structured.root_rotation_6D_column1_y",
        "structured.root_rotation_6D_column1_z",
    ]
    expected = int(config["structured_representation"][
        "structured_input_dimension"
    ])
    if structured.shape != (len(x), expected) or len(structured_names) != expected:
        raise ValueError("structured-orientation output dimension differs")
    raw_mean = np.mean(structured[train], axis=0)
    raw_std = np.std(structured[train], axis=0)
    mean = raw_mean.copy()
    std = raw_std.copy()
    presence = np.asarray([
        name.endswith(".present") for name in structured_names
    ], dtype=bool)
    constant = raw_std < 1.0e-6
    mean[presence] = 0.0
    std[presence] = 1.0
    std[constant & ~presence] = 1.0
    mean[-6:] = 0.0
    std[-6:] = 1.0
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError("structured-orientation normalization differs")
    normalized = (structured - mean) / std
    maximum_by_split = {
        split_name: float(np.max(np.abs(normalized[split == split_name])))
        for split_name in ("train", "validation", "test")
    }
    structured_arrays = dict(arrays)
    structured_arrays["features"] = structured
    normalization = {"feature_mean": mean, "feature_std": std}
    receipt = {
        "original_input_dimension": int(x.shape[1]),
        "structured_input_dimension": int(structured.shape[1]),
        "removed_feature_count": int(len(removed)),
        "retained_feature_count": int(len(kept)),
        "appended_feature_count": 6,
        "presence_feature_count": int(np.count_nonzero(presence)),
        "unit_scaled_train_constant_feature_count": int(np.count_nonzero(
            constant & ~presence
        )),
        "root_rotation_6D_maximum_absolute_value": float(np.max(np.abs(root_6d))),
        "root_rotation_6D_orthonormality_error": orthogonality_error,
        "maximum_absolute_normalized_value_by_split": maximum_by_split,
        "removed_feature_indexes_sha256": _sha256(removed.tobytes()),
        "retained_feature_indexes_sha256": _sha256(kept.tobytes()),
        "structured_feature_names_sha256": _sha256(_canonical(structured_names)),
        "structured_features_sha256": _sha256(structured.tobytes()),
        "normalization_mean_sha256": _sha256(mean.tobytes()),
        "normalization_std_sha256": _sha256(std.tobytes()),
    }
    return structured_arrays, normalization, receipt


def structured_decision(
    *, representation: Mapping[str, Any], flat_metrics: Mapping[str, Any],
    time_metrics: Mapping[str, Any], flat_temporal: Mapping[str, Any],
    time_temporal: Mapping[str, Any], flat_support: Mapping[str, Any],
    time_support: Mapping[str, Any], source_time_metrics: Mapping[str, Any],
    source_time_temporal: Mapping[str, Any], source_time_support_count: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["decision_gate"]
    shared_tests = {
        "structured_validation_z_bounded": float(
            representation["maximum_absolute_normalized_value_by_split"]["validation"]
        ) <= float(gate["maximum_structured_validation_absolute_z"]),
        "flat_validation_terminal_physical": float(
            flat_temporal["validation"]["terminal_joint_RMSE_rad"]
        ) <= float(gate["maximum_validation_terminal_joint_RMSE_rad_each_arm"]),
        "time_validation_terminal_physical": float(
            time_temporal["validation"]["terminal_joint_RMSE_rad"]
        ) <= float(gate["maximum_validation_terminal_joint_RMSE_rad_each_arm"]),
    }
    flat_tests = {
        "flat_zero_test_false_safe": int(
            flat_metrics["test_safety"]["false_safe_action_count"]
        ) <= int(gate["maximum_flat_test_false_safe_action_count"]),
        "flat_test_safe_recall": float(
            flat_metrics["test_safety"]["exact_safe_action_recall"]
        ) >= float(gate["minimum_flat_test_safe_recall"]),
        "flat_test_support": int(flat_support["supported_state_count"])
        >= int(gate["minimum_flat_test_supported_eligible_state_count"]),
        "flat_test_boundary_RMSE": float(
            flat_metrics["test_safety"]["near_boundary_RMSE_m"]
        ) <= float(gate["maximum_flat_test_boundary_RMSE_m"]),
        "flat_test_terminal_joint_RMSE": float(
            flat_temporal["test"]["terminal_joint_RMSE_rad"]
        ) <= float(gate["maximum_flat_test_terminal_joint_RMSE_rad"]),
    }
    time_tests = {
        "time_zero_test_false_safe": int(
            time_metrics["test_safety"]["false_safe_action_count"]
        ) == 0,
        "time_strict_boundary_improvement": float(
            time_metrics["test_safety"]["near_boundary_RMSE_m"]
        ) < float(source_time_metrics["test_safety"]["near_boundary_RMSE_m"]),
        "time_strict_terminal_improvement": float(
            time_temporal["test"]["terminal_joint_RMSE_rad"]
        ) < float(source_time_temporal["test"]["terminal_joint_RMSE_rad"]),
        "time_strict_support_improvement": int(time_support["supported_state_count"])
        > int(source_time_support_count),
    }
    flat_go = bool(all(shared_tests.values()) and all(flat_tests.values()))
    time_go = bool(all(shared_tests.values()) and all(time_tests.values()))
    return {
        "shared_representation_tests": shared_tests,
        "flat_candidate_tests": flat_tests,
        "time_candidate_tests": time_tests,
        "structured_representation_GO": flat_go,
        "flat_model_candidate_GO": flat_go,
        "time_decoder_candidate_GO": time_go,
        "new_reserved_episode_evaluation_authorized": flat_go,
        "QP_or_closed_loop_authorized": False,
        "conclusion": (
            "structured_orientation_repairs_representation_and_flat_candidate_passes"
            if flat_go else
            "structured_orientation_does_not_pass_matched_flat_candidate_gate"
        ),
    }
