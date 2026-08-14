"""Prefix/backup target decomposition for the L5 row-0/row-1 predictor."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_row01_component_audit.v1"
RESULT_SCHEMA = "vlsa_distal_l5_row01_component_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_row01_component_audit_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    payload.pop("validation_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source", "targets",
        "matched_model", "E39_curve", "gates", "interpretation", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("component-audit config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-l5-row01-component-audit-v1"
        or value["targets"]["learned_rows"] != [0, 1]
        or value["matched_model"]["input_dimension"] != 134
        or value["matched_model"]["output_count"] != 2
        or value["matched_model"]["hidden_widths"] != [32, 32]
    ):
        raise ValueError("component-audit protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def train_component_model(
    train_x: Any, train_y: Any, validation_x: Any, validation_y: Any,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    from main.multilink_ellipsoid.l5_row01_input_ablation import _weighted_huber
    from main.multilink_ellipsoid.l5_row01_selection import build_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("component audit requires exactly one H100")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError("component audit requires an H100")
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    x_validation = np.asarray(validation_x, dtype=np.float64)
    y_validation = np.asarray(validation_y, dtype=np.float64)
    if (
        x_train.ndim != 2 or x_train.shape[1] != 134
        or y_train.shape != (x_train.shape[0], 2)
        or x_validation.ndim != 2 or x_validation.shape[1] != 134
        or y_validation.shape != (x_validation.shape[0], 2)
        or min(x_train.shape[0], x_validation.shape[0]) < 1
        or not all(np.all(np.isfinite(item)) for item in (
            x_train, y_train, x_validation, y_validation
        ))
    ):
        raise ValueError("component training arrays differ")
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, int(model_config["cpu_threads"])))
    device = torch.device("cuda:0")
    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.where(
        np.std(x_train, axis=0) >= 1.0e-6, np.std(x_train, axis=0), 1.0
    )
    target_mean = np.mean(y_train, axis=0)
    target_scale = np.maximum(
        np.std(y_train, axis=0), float(model_config["minimum_target_scale_m"])
    )

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    tx = tensor((x_train - feature_mean) / feature_scale)
    ty = tensor((y_train - target_mean) / target_scale)
    ty_physical = tensor(y_train)
    vx = tensor((x_validation - feature_mean) / feature_scale)
    vy = tensor((y_validation - target_mean) / target_scale)
    vy_physical = tensor(y_validation)
    model = build_model(torch, 134, model_config["hidden_widths"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_train_loss = math.inf
    for _ in range(int(model_config["epochs"])):
        model.train()
        loss = _weighted_huber(torch, model(tx), ty, ty_physical, model_config)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(model_config["gradient_clip_norm"])
        )
        optimizer.step()
        final_train_loss = float(loss.item())
    model.eval()
    with torch.no_grad():
        validation_loss = float(_weighted_huber(
            torch, model(vx), vy, vy_physical, model_config
        ).item())
    state_payload = {
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "state_dict": {
            name: value.detach().cpu().numpy().tolist()
            for name, value in sorted(model.state_dict().items())
        },
    }
    return {
        "model": model, "device": device,
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "state_payload": state_payload,
        "model_sha256": hashlib.sha256(canonical(state_payload)).hexdigest(),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "device_name": device_name,
        "epochs_completed": int(model_config["epochs"]),
        "final_train_loss": final_train_loss,
        "validation_loss": validation_loss,
    }


def component_metrics(predictions: Any, targets: Any, threshold_m: float) -> dict[str, Any]:
    import numpy as np

    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("component metric arrays differ")
    error = prediction - target
    near = np.abs(target) <= float(threshold_m)
    predicted_safe = np.max(prediction, axis=1) <= 0.0
    exact_safe = np.max(target, axis=1) <= 0.0
    return {
        "sample_count": int(target.shape[0]),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "per_row_rmse_m": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "near_boundary_element_count": int(np.sum(near)),
        "near_boundary_rmse_m": (
            float(np.sqrt(np.mean(error[near] ** 2))) if np.any(near) else None
        ),
        "per_row_near_boundary_rmse_m": [
            float(np.sqrt(np.mean(error[near[:, row], row] ** 2)))
            if np.any(near[:, row]) else None for row in range(2)
        ],
        "false_safe_count": int(np.sum(predicted_safe & ~exact_safe)),
        "safe_recall": (
            float(np.sum(predicted_safe & exact_safe)) / int(np.sum(exact_safe))
            if np.any(exact_safe) else 0.0
        ),
    }


def false_safe_phase_attribution(
    predictions: Any, samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray([sample["combined_target"] for sample in samples], dtype=np.float64)
    predicted_safe = np.max(prediction, axis=1) <= 0.0
    exact_unsafe = np.max(target, axis=1) > 0.0
    counts = {"prefix": 0, "backup": 0}
    records = []
    for index in np.flatnonzero(predicted_safe & exact_unsafe):
        active_row = int(np.argmax(target[index]))
        phase = str(samples[int(index)]["witness_phase"][active_row])
        counts[phase] += 1
        records.append({
            "state_id": samples[int(index)]["state_id"],
            "candidate_name": samples[int(index)]["candidate_name"],
            "active_row": active_row,
            "active_phase": phase,
            "exact_active_risk_m": float(target[index, active_row]),
            "predicted_active_risk_m": float(prediction[index, active_row]),
        })
    return {"count_by_active_phase": counts, "records": records}


def response_curve_audit(
    train_samples: Sequence[Mapping[str, Any]],
    validation_samples: Sequence[Mapping[str, Any]],
    validation_state_contains: str, minimum_actions: int,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_relative_root_cause import STATE_INDICES

    def grouped(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
        output = {}
        for sample in samples:
            output.setdefault(sample["state_id"], []).append(sample)
        return output

    train_states = grouped(train_samples)
    validation_states = grouped(validation_samples)
    matches = [key for key in validation_states if validation_state_contains in key]
    if len(matches) != 1:
        raise ValueError("E39 response state differs")
    query_id = matches[0]
    eligible = {
        key: values for key, values in train_states.items()
        if len({float(item["applied_correction_l2_action"]) for item in values})
        >= int(minimum_actions)
    }
    if not eligible:
        raise ValueError("no response-capable training state")
    indices = np.asarray(STATE_INDICES, dtype=np.int64)
    train_matrix = np.asarray([
        np.asarray(values[0]["relative_feature_vector"])[indices]
        for values in eligible.values()
    ], dtype=np.float64)
    all_train_matrix = np.asarray([
        np.asarray(values[0]["relative_feature_vector"])[indices]
        for values in train_states.values()
    ], dtype=np.float64)
    mean = np.mean(all_train_matrix, axis=0)
    scale = np.where(np.std(all_train_matrix, axis=0) >= 1.0e-6,
                     np.std(all_train_matrix, axis=0), 1.0)
    query = np.asarray(validation_states[query_id][0]["relative_feature_vector"])[indices]
    distance = np.sqrt(np.mean(((train_matrix - query) / scale) ** 2, axis=1))
    nearest_index = int(np.argmin(distance))
    nearest_id = list(eligible)[nearest_index]

    def curve(state_id: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        ordered = sorted(items, key=lambda item: (
            float(item["applied_correction_l2_action"]), int(item["candidate_order"])
        ))
        coordinate = np.asarray([
            float(item["applied_correction_l2_action"]) for item in ordered
        ], dtype=np.float64)
        if len(np.unique(coordinate)) < int(minimum_actions):
            raise ValueError("response curve lacks distinct coordinates")
        output = {"state_id": state_id, "points": [], "slopes": {}}
        for item in ordered:
            output["points"].append({
                "candidate_name": item["candidate_name"],
                "correction_l2_action": float(item["applied_correction_l2_action"]),
                "prefix_target_m": item["prefix_target"],
                "backup_target_m": item["backup_target"],
                "combined_target_m": item["combined_target"],
            })
        for component in ("prefix", "backup", "combined"):
            for row in range(2):
                pairs = [
                    (float(item["applied_correction_l2_action"]), item[f"{component}_target"][row])
                    for item in ordered if item[f"{component}_target"] is not None
                ]
                key = "%s_row%d" % (component, row)
                if len({pair[0] for pair in pairs}) < int(minimum_actions):
                    output["slopes"][key] = None
                    continue
                x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
                y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
                output["slopes"][key] = float(np.polyfit(x, y, 1)[0])
        return output

    query_curve = curve(query_id, validation_states[query_id])
    nearest_curve = curve(nearest_id, eligible[nearest_id])
    signs = {}
    for key, query_slope in query_curve["slopes"].items():
        nearest_slope = nearest_curve["slopes"][key]
        if query_slope is None or nearest_slope is None:
            signs[key] = None
        else:
            signs[key] = bool(np.sign(query_slope) == np.sign(nearest_slope))
    identifiable = [value for value in signs.values() if value is not None]
    return {
        "validation_curve": query_curve,
        "nearest_response_capable_training_curve": nearest_curve,
        "nearest_state_RMS_z_distance": float(distance[nearest_index]),
        "slope_sign_match": signs,
        "identifiable_slope_count": len(identifiable),
        "matching_slope_sign_count": int(sum(identifiable)),
        "all_identifiable_slope_signs_match": bool(identifiable) and all(identifiable),
    }
