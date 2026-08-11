"""Frozen complete-input normalization and activation audit.

This module never trains a model or changes an input.  It replays the two
immutable execution predictors, locates unsupported normalized features, and
uses training-mean masking only as a causal diagnostic for validation-time
activation explosions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import _canonical, _numpy, _torch
from .factorized_execution_pilot import (
    dataset_arrays, load_weights, predict as predict_flat,
)
from .factorized_time_conditioned_decoder import (
    load_time_weights, predict_time_conditioned,
)


CONFIG_SCHEMA = "vlsa_distal_factorized_input_representation_audit_config.v1"
RESULT_SCHEMA = "vlsa_distal_factorized_input_representation_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_input_representation_audit_validation.v1"
RECORDS_SCHEMA = "vlsa_distal_factorized_input_representation_audit_records.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha256(_canonical(payload))


def load_audit_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("input-representation audit config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "normalization_audit", "activation_audit",
        "causal_mask_audit", "decision", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("input-representation audit config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-input-representation-audit-moka10-v1"
    ):
        raise ValueError("input-representation audit protocol differs")
    population = config["population"]
    if population != {
        "episode_count": 17, "state_count": 85, "rollout_count": 7905,
        "state_input_dimension": 2110, "action_input_dimension": 14,
        "model_input_dimension": 2124,
        "split_state_counts": {"train": 60, "validation": 10, "test": 15},
    }:
        raise ValueError("input-representation audit population differs")
    normalization = config["normalization_audit"]
    if (
        float(normalization["standard_deviation_floor"]) != 1.0e-6
        or float(normalization["extreme_absolute_z_threshold"]) != 10.0
        or float(normalization["explosive_absolute_z_threshold"]) != 100.0
        or int(normalization["top_feature_record_count"]) != 64
        or int(normalization["top_group_record_count"]) != 24
    ):
        raise ValueError("input-representation normalization audit differs")
    if config["forbidden_actions"] != {
        "new_simulation": True, "training": True,
        "normalization_change": True, "model_change": True,
        "calibration": True, "QP": True, "closed_loop": True,
        "poisson_or_SDF": True,
    }:
        raise ValueError("input-representation forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def model_feature_names(complete_dataset: Mapping[str, Any]) -> list[str]:
    state_names = [str(value) for value in complete_dataset[
        "complete_input_feature_names"
    ]]
    action_names = []
    coordinates = ("x", "y", "z", "rx", "ry", "rz", "gripper")
    for action in range(2):
        for coordinate in coordinates:
            action_names.append("action[%d].%s" % (action, coordinate))
    names = state_names + action_names
    if len(state_names) != 2110 or len(names) != 2124 or len(set(names)) != len(names):
        raise ValueError("input-representation semantic feature names differ")
    return names


def semantic_group(name: str) -> str:
    """Map aligned scalar features to stable, interpretable groups."""

    if name.startswith("action["):
        prefix, coordinate = name.rsplit(".", 1)
        family = "gripper"
        if coordinate in ("x", "y", "z"):
            family = "translation"
        elif coordinate in ("rx", "ry", "rz"):
            family = "rotation"
        return "%s.%s" % (prefix, family)
    base = re.sub(r"\.(value|present)$", "", name)
    base = re.sub(r"\[\d+\]", "[]", base)
    parts = base.split(".")
    if len(parts) <= 3:
        return base
    return ".".join(parts[:3])


def feature_classes(
    names: Sequence[str], features: Any, train_mask: Any, floor: float,
) -> list[str]:
    np = _numpy()
    x = np.asarray(features, dtype=np.float64)
    selected = x[np.asarray(train_mask, dtype=bool)]
    name_to_index = {str(name): index for index, name in enumerate(names)}
    output = []
    raw_std = np.std(selected, axis=0)
    for index, name in enumerate(names):
        name = str(name)
        if name.endswith(".present"):
            output.append("presence")
            continue
        if name.endswith(".value"):
            present_name = name[:-len(".value")] + ".present"
            present_index = name_to_index.get(present_name)
            if present_index is not None and np.any(x[:, present_index] < 0.5):
                output.append("padded_value")
                continue
        if raw_std[index] < floor:
            output.append("train_constant")
            continue
        unique = np.unique(selected[:, index])
        if len(unique) <= 2:
            output.append("binary_or_categorical")
            continue
        if name.startswith("action["):
            output.append("action_continuous")
        else:
            output.append("continuous_physical")
    return output


def normalization_records(
    *, arrays: Mapping[str, Any], names: Sequence[str], flat_state: Mapping[str, Any],
    time_state: Mapping[str, Any], config: Mapping[str, Any], state_to_case: Mapping[int, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    np = _numpy()
    x = np.asarray(arrays["features"], dtype=np.float64)
    split = np.asarray(arrays["split"], dtype=object)
    train = split == "train"
    floor = float(config["normalization_audit"]["standard_deviation_floor"])
    mean = np.mean(x[train], axis=0)
    raw_std = np.std(x[train], axis=0)
    std = np.maximum(raw_std, floor)
    if not (
        np.array_equal(mean, np.asarray(flat_state["feature_mean"]))
        and np.array_equal(std, np.asarray(flat_state["feature_std"]))
        and np.array_equal(mean, np.asarray(time_state["feature_mean"]))
        and np.array_equal(std, np.asarray(time_state["feature_std"]))
    ):
        raise ValueError("stored model normalization differs from fresh train rows")
    z = (x - mean) / std
    maximum_by_split = {}
    worst_row_by_split = {}
    for split_name in ("train", "validation", "test"):
        rows = np.flatnonzero(split == split_name)
        absolute = np.abs(z[rows])
        maximum_by_split[split_name] = np.max(absolute, axis=0)
        worst_row_by_split[split_name] = rows[np.argmax(absolute, axis=0)]
    classes = feature_classes(names, x, train, floor)
    groups = [semantic_group(name) for name in names]
    validation_max = maximum_by_split["validation"]
    order = np.argsort(-validation_max, kind="stable")
    top_count = int(config["normalization_audit"]["top_feature_record_count"])
    states = np.asarray(arrays["state_index"], dtype=np.int64)
    top_features = []
    for index in order[:top_count]:
        row = int(worst_row_by_split["validation"][index])
        top_features.append({
            "feature_index": int(index), "feature_name": str(names[index]),
            "semantic_group": groups[index], "feature_class": classes[index],
            "training_mean": float(mean[index]),
            "training_raw_standard_deviation": float(raw_std[index]),
            "normalization_standard_deviation": float(std[index]),
            "maximum_absolute_z": {
                key: float(maximum_by_split[key][index])
                for key in ("train", "validation", "test")
            },
            "worst_validation_row_index": row,
            "worst_validation_state_index": int(states[row]),
            "worst_validation_case_id": str(state_to_case[int(states[row])]),
            "worst_validation_raw_value": float(x[row, index]),
        })
    group_indexes = {}
    for index, group in enumerate(groups):
        group_indexes.setdefault(group, []).append(index)
    group_records = []
    for group, indexes in group_indexes.items():
        values = validation_max[indexes]
        local = int(np.argmax(values))
        feature_index = int(indexes[local])
        group_records.append({
            "semantic_group": str(group),
            "feature_indexes": [int(value) for value in indexes],
            "maximum_validation_absolute_z": float(values[local]),
            "maximum_feature_index": feature_index,
            "maximum_feature_name": str(names[feature_index]),
            "maximum_feature_class": classes[feature_index],
        })
    group_records.sort(key=lambda value: -value["maximum_validation_absolute_z"])
    group_records = group_records[:int(
        config["normalization_audit"]["top_group_record_count"]
    )]
    episode_records = []
    for state in sorted(set(states.tolist())):
        rows = np.flatnonzero(states == state)
        state_split = str(split[rows[0]])
        absolute = np.abs(z[rows])
        flat_index = int(np.argmax(absolute))
        local_row, feature_index = np.unravel_index(flat_index, absolute.shape)
        row = int(rows[local_row])
        episode_records.append({
            "state_index": int(state), "case_id": str(state_to_case[int(state)]),
            "split": state_split, "maximum_absolute_z": float(absolute.flat[flat_index]),
            "feature_index": int(feature_index),
            "feature_name": str(names[feature_index]),
            "feature_class": classes[feature_index],
            "row_index": row,
        })
    summary = {
        "input_dimension": int(x.shape[1]),
        "train_mean_and_std_exactly_match_both_models": True,
        "training_standard_deviation_floor_count": int(np.count_nonzero(raw_std < floor)),
        "maximum_absolute_z_by_split": {
            key: float(np.max(value)) for key, value in maximum_by_split.items()
        },
        "feature_count_at_or_above_extreme_validation_z": int(np.count_nonzero(
            validation_max >= float(config["normalization_audit"][
                "extreme_absolute_z_threshold"
            ])
        )),
        "feature_count_at_or_above_explosive_validation_z": int(np.count_nonzero(
            validation_max >= float(config["normalization_audit"][
                "explosive_absolute_z_threshold"
            ])
        )),
        "top_features": top_features,
        "top_groups": group_records,
        "per_state": episode_records,
    }
    full = {
        "feature_names": np.asarray(names, dtype="U512"),
        "feature_class": np.asarray(classes, dtype="U32"),
        "semantic_group": np.asarray(groups, dtype="U256"),
        "training_mean": mean, "training_raw_std": raw_std,
        "normalization_std": std,
        "maximum_abs_z_train": maximum_by_split["train"],
        "maximum_abs_z_validation": maximum_by_split["validation"],
        "maximum_abs_z_test": maximum_by_split["test"],
        "normalized_features": z,
    }
    return summary, full


def _accumulate_activation(target: dict[str, Any], name: str, value: Any) -> None:
    np = _numpy()
    array = value.detach().cpu().numpy().astype(np.float64, copy=False)
    record = target.setdefault(name, {
        "maximum_absolute_value": 0.0, "sum_of_squares": 0.0,
        "element_count": 0, "nonfinite_count": 0,
    })
    finite = np.isfinite(array)
    record["nonfinite_count"] += int(np.count_nonzero(~finite))
    if np.any(finite):
        selected = array[finite]
        record["maximum_absolute_value"] = max(
            float(record["maximum_absolute_value"]), float(np.max(np.abs(selected)))
        )
        record["sum_of_squares"] += float(np.sum(selected ** 2))
        record["element_count"] += int(len(selected))


def _member_activation_summary(model: Any, normalized: Any, kind: str) -> list[dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    accumulator: dict[str, Any] = {}
    handles = []
    for name, module in model.named_modules():
        if name and isinstance(module, (torch.nn.Linear, torch.nn.SiLU)):
            def hook(_module: Any, _inputs: Any, output: Any, label: str = name) -> None:
                _accumulate_activation(accumulator, label, output)
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            for start in range(0, len(normalized), 512):
                tensor = torch.as_tensor(
                    np.asarray(normalized[start:start + 512]), dtype=torch.float32
                )
                output = model(tensor)
                _accumulate_activation(accumulator, "model_output", output)
    finally:
        for handle in handles:
            handle.remove()
    records = []
    for order, (name, values) in enumerate(accumulator.items()):
        count = int(values["element_count"])
        records.append({
            "order": int(order), "module": name, "model_kind": kind,
            "maximum_absolute_value": float(values["maximum_absolute_value"]),
            "RMS": float(math.sqrt(values["sum_of_squares"] / max(count, 1))),
            "element_count": count,
            "nonfinite_count": int(values["nonfinite_count"]),
        })
    return records


def activation_audit(
    *, flat_models: Sequence[Any], time_models: Sequence[Any], normalized: Any,
    split: Any, config: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    split_array = np.asarray(split, dtype=object)
    by_model = {}
    first_exploding = {}
    threshold = config["activation_audit"]
    for kind, models in (("flat", flat_models), ("time", time_models)):
        members = []
        aggregate = {}
        for member_index, model in enumerate(models):
            split_records = {}
            for split_name in ("train", "validation", "test"):
                rows = np.flatnonzero(split_array == split_name)
                records = _member_activation_summary(model, normalized[rows], kind)
                split_records[split_name] = records
                for record in records:
                    key = record["module"]
                    item = aggregate.setdefault(key, {
                        "order": record["order"], "module": key,
                        "train": 0.0, "validation": 0.0, "test": 0.0,
                    })
                    item[split_name] = max(
                        float(item[split_name]), record["maximum_absolute_value"]
                    )
            members.append({"member_index": int(member_index), "splits": split_records})
        ordered = sorted(aggregate.values(), key=lambda value: value["order"])
        first = None
        for item in ordered:
            ratio = float(item["validation"] / max(item["train"], 1.0e-12))
            item["validation_to_train_maximum_ratio"] = ratio
            if (
                first is None
                and item["validation"] >= float(threshold[
                    "first_exploding_layer_minimum_validation_max_abs"
                ])
                and ratio >= float(threshold[
                    "first_exploding_layer_minimum_validation_to_train_ratio"
                ])
            ):
                first = dict(item)
        by_model[kind] = {"members": members, "aggregate_modules": ordered}
        first_exploding[kind] = first
    return {"models": by_model, "first_exploding_module": first_exploding}


def _predict_flat_normalized(models: Sequence[Any], normalized: Any, base: Any) -> Any:
    np = _numpy()
    torch = _torch()
    members = []
    with torch.no_grad():
        tensor = torch.as_tensor(normalized, dtype=torch.float32)
        for model in models:
            members.append(model(tensor).cpu().numpy().reshape(-1, 51, 7))
    return np.asarray(base, dtype=np.float64) + np.mean(np.asarray(members), axis=0)


def _predict_time_normalized(models: Sequence[Any], normalized: Any, base: Any) -> Any:
    np = _numpy()
    torch = _torch()
    members = []
    with torch.no_grad():
        tensor = torch.as_tensor(normalized, dtype=torch.float32)
        for model in models:
            members.append(model(tensor).cpu().numpy())
    return np.asarray(base, dtype=np.float64) + np.mean(np.asarray(members), axis=0)


def terminal_rmse(predicted: Any, exact: Any) -> float:
    np = _numpy()
    error = np.asarray(predicted, dtype=np.float64)[:, -1] - np.asarray(
        exact, dtype=np.float64
    )[:, -1]
    return float(np.sqrt(np.mean(error ** 2)))


def causal_group_mask_audit(
    *, flat_models: Sequence[Any], time_models: Sequence[Any], arrays: Mapping[str, Any],
    normalized: Any, normalization_summary: Mapping[str, Any],
    full_normalization: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    split = np.asarray(arrays["split"], dtype=object)
    source = np.asarray(arrays["source_code"], dtype=np.int8)
    rows = np.flatnonzero((split == "validation") & (source == 2))
    exact = np.asarray(arrays["joint_position_rad"], dtype=np.float64)[rows]
    base = np.repeat(exact[:, :1], 51, axis=1)
    z = np.asarray(normalized, dtype=np.float64)[rows]
    flat_full = _predict_flat_normalized(flat_models, z, base)
    time_full = _predict_time_normalized(time_models, z, base)
    full_rmse = {
        "flat": terminal_rmse(flat_full, exact),
        "time": terminal_rmse(time_full, exact),
    }
    group_records = []
    classes = np.asarray(full_normalization["feature_class"], dtype=object)
    groups = np.asarray(full_normalization["semantic_group"], dtype=object)
    names = np.asarray(full_normalization["feature_names"], dtype=object)
    maximum_z = np.asarray(
        full_normalization["maximum_abs_z_validation"], dtype=np.float64
    )
    artifact_classes = set(config["normalization_audit"]["artifact_feature_classes"])
    for source_group in normalization_summary["top_groups"]:
        group = str(source_group["semantic_group"])
        indexes = np.flatnonzero(groups == group)
        masked = z.copy()
        masked[:, indexes] = 0.0
        flat_prediction = _predict_flat_normalized(flat_models, masked, base)
        time_prediction = _predict_time_normalized(time_models, masked, base)
        flat_rmse = terminal_rmse(flat_prediction, exact)
        time_rmse = terminal_rmse(time_prediction, exact)
        feature_index = int(indexes[np.argmax(maximum_z[indexes])])
        feature_class = str(classes[feature_index])
        group_records.append({
            "semantic_group": group,
            "feature_count": int(len(indexes)),
            "maximum_validation_absolute_z": float(maximum_z[feature_index]),
            "maximum_feature_name": str(names[feature_index]),
            "maximum_feature_class": feature_class,
            "artifact_class": feature_class in artifact_classes,
            "terminal_joint_RMSE_rad": {
                "flat": float(flat_rmse), "time": float(time_rmse),
            },
            "terminal_RMSE_reduction_fraction": {
                "flat": float(1.0 - flat_rmse / max(full_rmse["flat"], 1.0e-15)),
                "time": float(1.0 - time_rmse / max(full_rmse["time"], 1.0e-15)),
            },
        })
    reduction = float(config["causal_mask_audit"][
        "strong_terminal_RMSE_reduction_fraction"
    ])
    for record in group_records:
        record["strong_causal_reduction_both_models"] = bool(
            record["terminal_RMSE_reduction_fraction"]["flat"] >= reduction
            and record["terminal_RMSE_reduction_fraction"]["time"] >= reduction
        )
    group_records.sort(key=lambda value: -min(
        value["terminal_RMSE_reduction_fraction"]["flat"],
        value["terminal_RMSE_reduction_fraction"]["time"],
    ))
    return {
        "validation_random_action_count": int(len(rows)),
        "unmasked_terminal_joint_RMSE_rad": full_rmse,
        "group_masks": group_records,
    }


def per_case_prediction_audit(
    *, arrays: Mapping[str, Any], predictions: Mapping[str, Any],
    state_to_case: Mapping[int, str],
) -> list[dict[str, Any]]:
    np = _numpy()
    states = np.asarray(arrays["state_index"], dtype=np.int64)
    split = np.asarray(arrays["split"], dtype=object)
    source = np.asarray(arrays["source_code"], dtype=np.int8)
    exact = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
    records = []
    for case_id in sorted(set(state_to_case.values())):
        state_set = {state for state, value in state_to_case.items() if value == case_id}
        rows = np.asarray([
            index for index, state in enumerate(states)
            if int(state) in state_set and source[index] == 2
        ], dtype=np.int64)
        if not len(rows):
            continue
        item = {
            "case_id": str(case_id), "split": str(split[rows[0]]),
            "random_action_count": int(len(rows)),
        }
        for name, predicted in predictions.items():
            error = np.asarray(predicted, dtype=np.float64)[rows] - exact[rows]
            item[name] = {
                "terminal_joint_RMSE_rad": float(np.sqrt(np.mean(error[:, -1] ** 2))),
                "maximum_absolute_terminal_joint_error_rad": float(
                    np.max(np.abs(error[:, -1]))
                ),
                "overall_joint_RMSE_rad": float(np.sqrt(np.mean(error ** 2))),
            }
        records.append(item)
    return records


def audit_decision(
    *, normalization: Mapping[str, Any], activation: Mapping[str, Any],
    masking: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    explosive = float(config["normalization_audit"]["explosive_absolute_z_threshold"])
    extreme = float(config["normalization_audit"]["extreme_absolute_z_threshold"])
    artifact = False
    physical = False
    strong_groups = []
    for record in masking["group_masks"]:
        if not record["strong_causal_reduction_both_models"]:
            continue
        strong_groups.append(record["semantic_group"])
        if record["artifact_class"] and record["maximum_validation_absolute_z"] >= explosive:
            artifact = True
        if (
            record["maximum_feature_class"] in ("continuous_physical", "action_continuous")
            and record["maximum_validation_absolute_z"] >= extreme
        ):
            physical = True
    if artifact and physical:
        cause = "mixed_representation_artifact_and_physical_support_shift"
    elif artifact:
        cause = "representation_artifact"
    elif physical:
        cause = "physical_state_or_action_support_shift"
    elif any(value is not None for value in activation["first_exploding_module"].values()):
        cause = "model_internal_instability_without_strong_input_group_cause"
    else:
        cause = "no_explosive_representation_root_identified"
    return {
        "root_cause_classification": cause,
        "representation_artifact_identified": artifact,
        "physical_support_shift_identified": physical,
        "strong_causal_groups": strong_groups,
        "normalization_pathology_present": bool(
            normalization["maximum_absolute_z_by_split"]["validation"] >= explosive
        ),
        "first_exploding_module": activation["first_exploding_module"],
        "retraining_or_architecture_change_authorized": False,
        "candidate_expansion_authorized": False,
        "next_action": (
            "repair_and_preregister_a_matched_input_representation_ablation"
            if artifact else
            "collect_grouped_coverage_for_the_causal_physical_feature_group"
            if physical else
            "audit_model_numerics_and_output_parameterization"
        ),
    }


def run_audit(
    *, config: Mapping[str, Any], complete_dataset: Mapping[str, Any],
    metadata: Mapping[str, Any], archive: Mapping[str, Any],
    flat_model_path: Path, flat_predictions_path: Path,
    time_model_path: Path, time_predictions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    np = _numpy()
    arrays = dataset_arrays(metadata, archive)
    names = model_feature_names(complete_dataset)
    flat_models, flat_state = load_weights(flat_model_path)
    time_models, time_state = load_time_weights(time_model_path)
    state_to_case = {
        int(record["state_index"]): str(record["case_id"])
        for record in metadata["state_records"]
    }
    normalization, full = normalization_records(
        arrays=arrays, names=names, flat_state=flat_state, time_state=time_state,
        config=config, state_to_case=state_to_case,
    )
    flat_prediction = predict_flat(flat_models, flat_state, arrays)
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
        raise ValueError("immutable model prediction replay differs")
    activation = activation_audit(
        flat_models=flat_models, time_models=time_models,
        normalized=full["normalized_features"], split=arrays["split"], config=config,
    )
    masking = causal_group_mask_audit(
        flat_models=flat_models, time_models=time_models, arrays=arrays,
        normalized=full["normalized_features"], normalization_summary=normalization,
        full_normalization=full, config=config,
    )
    prediction = {
        "immutable_flat_prediction_exact": True,
        "immutable_time_prediction_exact": True,
        "per_case": per_case_prediction_audit(
            arrays=arrays,
            predictions={"flat": flat_prediction, "time": time_prediction},
            state_to_case=state_to_case,
        ),
    }
    core = {
        "normalization": normalization, "activation": activation,
        "causal_group_mask": masking, "prediction": prediction,
    }
    core["decision"] = audit_decision(
        normalization=normalization, activation=activation, masking=masking,
        config=config,
    )
    records = {
        key: value for key, value in full.items() if key != "normalized_features"
    }
    records["schema_version"] = np.asarray(RECORDS_SCHEMA)
    return core, records
