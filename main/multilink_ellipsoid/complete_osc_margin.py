"""Complete-state action-conditioned execution-margin diagnostic.

This opt-in module tests whether omitted OSC state caused the earlier
action-conditioned surrogate failure.  It deliberately contains no safety
calibration, QP, or closed-loop execution path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import _canonical, _numpy, _torch


CONFIG_SCHEMA = "vlsa_distal_complete_osc_margin_moka10.v1"
DATASET_SCHEMA = "vlsa_distal_complete_osc_margin_dataset.v1"
COLLECTION_RESULT_SCHEMA = "vlsa_distal_complete_osc_margin_collection_result.v1"
RESULT_SCHEMA = "vlsa_distal_complete_osc_margin_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_complete_osc_margin_validation.v1"
WEIGHTS_SCHEMA = "vlsa_distal_complete_osc_margin_weights.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha256(_canonical(payload))


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("complete-OSC config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "split", "paired_collection", "complete_input", "model", "training",
        "prediction_gate", "forbidden_actions", "decision",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("complete-OSC config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"] != "vlsa-distal-complete-osc-margin-moka10-v1"
    ):
        raise ValueError("complete-OSC protocol differs")
    source = config["immutable_source"]
    if set(source) != {
        "expanded_dataset_file_sha256", "expanded_dataset_payload_sha256",
        "source_population_manifest_sha256", "selected_manifest_file_sha256",
        "same_task_manifest_file_sha256", "targeted_manifest_file_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "expected_episode_count", "expected_state_count",
        "expected_candidates_per_state", "expected_pair_count",
    }:
        raise ValueError("complete-OSC immutable source differs")
    if (
        int(source["expected_episode_count"]) != 17
        or int(source["expected_state_count"]) != 85
        or int(source["expected_candidates_per_state"]) != 125
        or int(source["expected_pair_count"]) != 10625
    ):
        raise ValueError("complete-OSC population count differs")
    if config["split"] != {
        "unit": "complete_episode",
        "expected_state_counts": {"train": 60, "validation": 10, "test": 15},
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "test_only": True,
    }:
        raise ValueError("complete-OSC grouped split differs")
    paired = config["paired_collection"]
    if paired != {
        "replay_count_per_snapshot_action_pair": 2,
        "margin_comparison": "bitwise_float64_equal",
        "contact_comparison": "canonical_raw_contact_receipt_equal",
        "next_state_comparison": "sha256_equal_for_both_actions",
        "training_requires_every_pair_deterministic": True,
        "horizon_actions": 2,
        "record_every_internal_OSC_substep": True,
    }:
        raise ValueError("complete-OSC paired replay contract differs")
    complete = config["complete_input"]
    if complete != {
        "state_vector": "complete_serialized_simulator_auxiliary_controller_snapshot",
        "semantic_state": [
            "q", "qdot", "current_EE_position_orientation",
            "goal_EE_position_orientation", "obstacle_pose",
            "seven_robot_ellipsoid_transforms",
            "all_exact_obstacle_primitive_transforms",
        ],
        "action": "full_two_action_translation_rotation_gripper_commands",
        "constraint_identity": "seven_way_one_hot",
        "normalization": "training_rows_mean_std_with_1e-6_floor",
        "output": "rollout_minimum_ellipsoid_margin_residual_from_k0_margin_mm",
    }:
        raise ValueError("complete-OSC input contract differs")
    model = config["model"]
    if model != {
        "class": "shared_constraint_complete_state_action_conditioned_residual_MLP_ensemble",
        "ensemble_seeds": [20260831, 20260832, 20260833, 20260834, 20260835],
        "hidden_widths": [256, 256, 128],
        "activation": "silu", "output_count": 1,
    }:
        raise ValueError("complete-OSC model differs")
    training = config["training"]
    expected_training = {
        "device": "cpu_inside_H100_allocation",
        "batch_size": 4096, "epochs": 500, "patience": 60,
        "learning_rate": 0.001, "weight_decay": 1.0e-6,
        "huber_delta_mm": 2.0, "boundary_band_m": 0.005,
        "boundary_weight_multiplier": 9.0,
        "one_sided_overestimate_loss_weight": 4.0,
        "early_stopping_split": "validation_complete_episodes",
    }
    if training != expected_training:
        raise ValueError("complete-OSC training differs")
    if config["prediction_gate"] != {
        "test_proxy_false_safe_action_count": 0,
        "minimum_test_exact_safe_action_recall": 0.5,
        "required_test_state_safe_support_count": 15,
        "maximum_test_near_boundary_RMSE_m": 0.002,
        "near_boundary_absolute_margin_m": 0.005,
        "maximum_test_overall_RMSE_m": 0.003808,
    }:
        raise ValueError("complete-OSC prediction gate differs")
    if config["forbidden_actions"] != {
        "calibration": True, "QP": True, "closed_loop_E05": True,
        "pi05_features": True,
    }:
        raise ValueError("complete-OSC forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def jsonable(value: Any) -> Any:
    """Convert a nested simulator snapshot to canonical JSON values."""

    np = _numpy()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    raise TypeError("unsupported complete-OSC snapshot value: %s" % type(value))


def flatten_numeric_tree(value: Any, prefix: str = "") -> tuple[list[str], Any]:
    """Flatten a fixed nested numeric tree and preserve semantic names."""

    np = _numpy()
    names: list[str] = []
    values: list[float] = []

    def visit(item: Any, name: str) -> None:
        if item is None:
            names.extend((name + ".present", name + ".value"))
            values.extend((0.0, 0.0))
        elif isinstance(item, bool):
            names.append(name)
            values.append(float(item))
        elif isinstance(item, (int, float, np.generic)):
            scalar = float(item)
            if not math.isfinite(scalar):
                raise ValueError("complete-OSC feature is nonfinite: %s" % name)
            names.append(name)
            values.append(scalar)
        elif isinstance(item, Mapping):
            for key in sorted(item):
                visit(item[key], "%s.%s" % (name, key) if name else str(key))
        elif isinstance(item, (list, tuple, np.ndarray)):
            array = item.tolist() if isinstance(item, np.ndarray) else list(item)
            for index, child in enumerate(array):
                visit(child, "%s[%d]" % (name, index))
        else:
            raise TypeError("complete-OSC numeric tree differs: %s" % type(item))

    visit(value, prefix)
    output = np.asarray(values, dtype=np.float64)
    if output.ndim != 1 or not np.all(np.isfinite(output)):
        raise ValueError("complete-OSC flattened vector differs")
    return names, output


def align_named_complete_inputs(
    named_values: Sequence[Mapping[str, float]],
) -> tuple[list[str], list[Any]]:
    """Losslessly align variable-size task snapshots with presence masks."""

    np = _numpy()
    if not named_values:
        raise ValueError("complete-OSC named input population is empty")
    base_names = sorted({name for item in named_values for name in item})
    if not base_names:
        raise ValueError("complete-OSC named input schema is empty")
    feature_names = []
    for name in base_names:
        feature_names.extend((name + ".value", name + ".present"))
    vectors = []
    for item in named_values:
        unknown = set(item) - set(base_names)
        if unknown:
            raise ValueError("complete-OSC named input contains unknown fields")
        values = []
        for name in base_names:
            if name in item:
                scalar = float(item[name])
                if not math.isfinite(scalar):
                    raise ValueError("complete-OSC aligned input is nonfinite")
                values.extend((scalar, 1.0))
            else:
                values.extend((0.0, 0.0))
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (len(feature_names),):
            raise ValueError("complete-OSC aligned vector differs")
        vectors.append(vector)
    return feature_names, vectors


def action_row_features(
    state_vector: Sequence[float], action_pair: Sequence[Sequence[float]],
    constraint_index: int,
) -> Any:
    np = _numpy()
    state = np.asarray(state_vector, dtype=np.float64)
    actions = np.asarray(action_pair, dtype=np.float64)
    if (
        state.ndim != 1 or actions.shape != (2, 7)
        or not 0 <= int(constraint_index) < 7
        or not np.all(np.isfinite(state)) or not np.all(np.isfinite(actions))
    ):
        raise ValueError("complete-OSC action row input differs")
    identity = np.zeros(7, dtype=np.float64)
    identity[int(constraint_index)] = 1.0
    return np.concatenate((state, actions.reshape(-1), identity))


def training_arrays(dataset: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    features = []
    margins = []
    current = []
    splits = []
    state_indexes = []
    action_indexes = []
    constraint_indexes = []
    for state in dataset["state_records"]:
        state_vector = state["complete_input_vector"]
        k0 = np.asarray(state["current_ellipsoid_margin_m"], dtype=np.float64)
        for action_index, candidate in enumerate(state["candidates"]):
            target = np.asarray(candidate["rollout_minimum_ellipsoid_margin_m"], dtype=np.float64)
            for constraint_index in range(7):
                features.append(action_row_features(
                    state_vector, candidate["full_two_action_commands"], constraint_index
                ))
                margins.append(target[constraint_index])
                current.append(k0[constraint_index])
                splits.append(state["split"])
                state_indexes.append(int(state["state_index"]))
                action_indexes.append(action_index)
                constraint_indexes.append(constraint_index)
    count = int(dataset["summary"]["pair_count"]) * 7
    output = {
        "features": np.asarray(features, dtype=np.float64),
        "margin_m": np.asarray(margins, dtype=np.float64),
        "current_margin_m": np.asarray(current, dtype=np.float64),
        "split": np.asarray(splits, dtype=object),
        "state_index": np.asarray(state_indexes, dtype=np.int64),
        "action_index": np.asarray(action_indexes, dtype=np.int64),
        "constraint_index": np.asarray(constraint_indexes, dtype=np.int64),
    }
    if (
        output["features"].shape[0] != count
        or output["features"].ndim != 2
        or any(output[key].shape != (count,) for key in (
            "margin_m", "current_margin_m", "split", "state_index",
            "action_index", "constraint_index",
        ))
    ):
        raise ValueError("complete-OSC training arrays differ")
    return output


def _build_model(input_count: int, hidden_widths: Sequence[int]) -> Any:
    torch = _torch()
    layers = []
    previous = int(input_count)
    for width in hidden_widths:
        layers.extend((torch.nn.Linear(previous, int(width)), torch.nn.SiLU()))
        previous = int(width)
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


def train_ensemble(arrays: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Train only after the caller has validated deterministic paired replay."""

    np = _numpy()
    torch = _torch()
    x = np.asarray(arrays["features"], dtype=np.float64)
    exact = np.asarray(arrays["margin_m"], dtype=np.float64)
    current = np.asarray(arrays["current_margin_m"], dtype=np.float64)
    split = np.asarray(arrays["split"], dtype=object)
    train_mask = split == "train"
    validation_mask = split == "validation"
    if not np.any(train_mask) or not np.any(validation_mask):
        raise ValueError("complete-OSC grouped training split is empty")
    mean = np.mean(x[train_mask], axis=0)
    std = np.std(x[train_mask], axis=0)
    std = np.maximum(std, 1.0e-6)
    normalized = (x - mean) / std
    target_mm = 1000.0 * (exact - current)
    settings = config["training"]
    delta = float(settings["huber_delta_mm"])
    boundary = float(settings["boundary_band_m"])
    multiplier = float(settings["boundary_weight_multiplier"])
    one_sided = float(settings["one_sided_overestimate_loss_weight"])
    batch_size = int(settings["batch_size"])
    train_indexes = np.flatnonzero(train_mask)
    validation_indexes = np.flatnonzero(validation_mask)
    models = []
    audits = []
    saved_states = []
    for seed in config["model"]["ensemble_seeds"]:
        torch.manual_seed(int(seed))
        model = _build_model(x.shape[1], config["model"]["hidden_widths"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        generator = np.random.default_rng(int(seed))
        best_score = math.inf
        best_epoch = -1
        best = None
        stale = 0
        for epoch in range(int(settings["epochs"])):
            model.train()
            order = generator.permutation(train_indexes)
            for start in range(0, len(order), batch_size):
                indexes = order[start:start + batch_size]
                xb = torch.as_tensor(normalized[indexes], dtype=torch.float32)
                yb = torch.as_tensor(target_mm[indexes], dtype=torch.float32)
                exact_b = torch.as_tensor(exact[indexes], dtype=torch.float32)
                prediction = model(xb).reshape(-1)
                difference = prediction - yb
                absolute = torch.abs(difference)
                huber = torch.where(
                    absolute <= delta, 0.5 * difference ** 2,
                    delta * (absolute - 0.5 * delta),
                )
                weights = torch.where(
                    torch.abs(exact_b) <= boundary,
                    torch.full_like(exact_b, multiplier),
                    torch.ones_like(exact_b),
                )
                over = torch.relu(difference)
                loss = torch.mean(weights * (huber + one_sided * over ** 2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                vi = validation_indexes
                prediction = model(torch.as_tensor(
                    normalized[vi], dtype=torch.float32
                )).reshape(-1).numpy()
            error = prediction - target_mm[vi]
            score = float(np.sqrt(np.mean(error ** 2)) + 0.5 * np.sqrt(
                np.mean(np.maximum(error, 0.0) ** 2)
            ))
            if score < best_score - 1.0e-9:
                best_score = score
                best_epoch = epoch
                best = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= int(settings["patience"]):
                break
        if best is None:
            raise RuntimeError("complete-OSC training produced no checkpoint")
        model.load_state_dict(best)
        model.eval()
        models.append(model)
        saved_states.append({
            key: value.detach().cpu().numpy() for key, value in best.items()
        })
        audits.append({
            "seed": int(seed), "best_epoch": int(best_epoch),
            "validation_score_mm": float(best_score),
            "completed_epoch_count": int(epoch + 1),
        })
    state = {
        "feature_mean": mean, "feature_std": std,
        "model_states": saved_states, "input_dimension": int(x.shape[1]),
        "hidden_widths": list(config["model"]["hidden_widths"]),
        "ensemble_seeds": list(config["model"]["ensemble_seeds"]),
    }
    return models, state, {
        "member_audits": audits,
        "training_row_count": int(np.count_nonzero(train_mask)),
        "validation_row_count": int(np.count_nonzero(validation_mask)),
    }


def predict(models: Sequence[Any], state: Mapping[str, Any], features: Any, current: Any) -> Any:
    np = _numpy()
    torch = _torch()
    x = np.asarray(features, dtype=np.float64)
    normalized = (x - state["feature_mean"]) / state["feature_std"]
    predictions = []
    with torch.no_grad():
        tensor = torch.as_tensor(normalized, dtype=torch.float32)
        for model in models:
            predictions.append(model(tensor).reshape(-1).numpy())
    residual_m = 0.001 * np.mean(np.asarray(predictions), axis=0)
    return np.asarray(current, dtype=np.float64) + residual_m


def save_weights(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    arrays: dict[str, Any] = {
        "schema_version": np.asarray(WEIGHTS_SCHEMA),
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float64),
        "feature_std": np.asarray(state["feature_std"], dtype=np.float64),
        "input_dimension": np.asarray(int(state["input_dimension"]), dtype=np.int64),
        "hidden_widths": np.asarray(state["hidden_widths"], dtype=np.int64),
        "ensemble_seeds": np.asarray(state["ensemble_seeds"], dtype=np.int64),
    }
    for member, member_state in enumerate(state["model_states"]):
        for key, value in member_state.items():
            arrays["member_%d.%s" % (member, key)] = np.asarray(value)
    temporary = path.with_name(".%s.tmp.npz" % path.name)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return {"path": str(path), "file_sha256": _sha256(path.read_bytes())}


def load_weights(path: Path) -> tuple[list[Any], dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    archive = np.load(path, allow_pickle=False)
    if str(archive["schema_version"].item()) != WEIGHTS_SCHEMA:
        raise ValueError("complete-OSC weights schema differs")
    hidden = archive["hidden_widths"].astype(np.int64).tolist()
    seeds = archive["ensemble_seeds"].astype(np.int64).tolist()
    input_dimension = int(archive["input_dimension"].item())
    models = []
    for member in range(len(seeds)):
        model = _build_model(input_dimension, hidden)
        prefix = "member_%d." % member
        state_dict = {
            key[len(prefix):]: torch.as_tensor(archive[key])
            for key in archive.files if key.startswith(prefix)
        }
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models, {
        "feature_mean": archive["feature_mean"].astype(np.float64),
        "feature_std": archive["feature_std"].astype(np.float64),
        "input_dimension": input_dimension, "hidden_widths": hidden,
        "ensemble_seeds": seeds,
    }


def prediction_metrics(arrays: Mapping[str, Any], prediction_m: Sequence[float], config: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    exact = np.asarray(arrays["margin_m"], dtype=np.float64)
    prediction = np.asarray(prediction_m, dtype=np.float64)
    split = np.asarray(arrays["split"], dtype=object)
    test = split == "test"
    if prediction.shape != exact.shape:
        raise ValueError("complete-OSC prediction shape differs")
    state_index = np.asarray(arrays["state_index"], dtype=np.int64)
    action_index = np.asarray(arrays["action_index"], dtype=np.int64)
    test_states = sorted(set(state_index[test].tolist()))
    false_safe = 0
    accepted_safe = 0
    exact_safe = 0
    support = 0
    state_results = []
    for state in test_states:
        state_mask = test & (state_index == state)
        actions = sorted(set(action_index[state_mask].tolist()))
        state_false = 0
        state_accepted_safe = 0
        state_exact_safe = 0
        for action in actions:
            mask = state_mask & (action_index == action)
            predicted_safe = bool(np.all(prediction[mask] >= 0.0))
            truly_safe = bool(np.all(exact[mask] >= 0.0))
            state_false += int(predicted_safe and not truly_safe)
            state_accepted_safe += int(predicted_safe and truly_safe)
            state_exact_safe += int(truly_safe)
        false_safe += state_false
        accepted_safe += state_accepted_safe
        exact_safe += state_exact_safe
        support += int(state_accepted_safe > 0)
        state_results.append({
            "state_index": int(state), "action_count": len(actions),
            "proxy_false_safe_action_count": state_false,
            "accepted_exact_safe_action_count": state_accepted_safe,
            "exact_safe_action_count": state_exact_safe,
        })
    error = prediction[test] - exact[test]
    band = float(config["prediction_gate"]["near_boundary_absolute_margin_m"])
    near = test & (np.abs(exact) <= band)
    metrics = {
        "test_state_count": len(test_states),
        "test_action_count": int(len(test_states) * 125),
        "test_proxy_false_safe_action_count": int(false_safe),
        "test_exact_safe_action_recall": (
            0.0 if exact_safe == 0 else float(accepted_safe / exact_safe)
        ),
        "test_state_safe_support_count": int(support),
        "test_overall_RMSE_m": float(np.sqrt(np.mean(error ** 2))),
        "test_near_boundary_row_count": int(np.count_nonzero(near)),
        "test_near_boundary_RMSE_m": float(np.sqrt(np.mean(
            (prediction[near] - exact[near]) ** 2
        ))) if np.any(near) else None,
        "state_results": state_results,
    }
    gate = config["prediction_gate"]
    metrics["prediction_gate_pass"] = bool(
        metrics["test_proxy_false_safe_action_count"]
        == int(gate["test_proxy_false_safe_action_count"])
        and metrics["test_exact_safe_action_recall"]
        >= float(gate["minimum_test_exact_safe_action_recall"])
        and metrics["test_state_safe_support_count"]
        == int(gate["required_test_state_safe_support_count"])
        and metrics["test_near_boundary_RMSE_m"] is not None
        and metrics["test_near_boundary_RMSE_m"]
        <= float(gate["maximum_test_near_boundary_RMSE_m"])
        and metrics["test_overall_RMSE_m"]
        <= float(gate["maximum_test_overall_RMSE_m"])
    )
    return metrics
