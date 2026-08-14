"""Contracts for the prediction-only three-output L5 feasibility pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_action_risk_mlp.v1"
AUDIT_SCHEMA = "vlsa_distal_l5_action_risk_dataset_audit.v1"
MODEL_SCHEMA = "vlsa_distal_l5_action_risk_model.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "learned_scope", "dataset_gate", "features", "model",
        "prediction_gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("L5 action-risk config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("L5 action-risk schema differs")
    if value["protocol_id"] != "vlsa-distal-l5-action-risk-mlp-v1":
        raise ValueError("L5 action-risk protocol differs")
    if (
        value["learned_scope"]["rows"] != [0, 1, 2]
        or value["learned_scope"]["output_count"] != 3
    ):
        raise ValueError("L5 action-risk learned scope differs")
    if value["features"]["input_dimension"] != 86:
        raise ValueError("L5 action-risk feature dimension differs")
    if (
        value["model"]["input_dimension"] != 86
        or value["model"]["output_count"] != 3
    ):
        raise ValueError("L5 action-risk model shape differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def feature_vector(
    *, initial_clearance: Sequence[float], local_frame: Mapping[str, Sequence[float]],
    nominal_actions: Sequence[Sequence[float]], candidate_actions: Sequence[Sequence[float]],
) -> list[float]:
    import numpy as np

    clearance = np.asarray(initial_clearance, dtype=np.float64)
    nominal = np.asarray(nominal_actions, dtype=np.float64)
    candidate = np.asarray(candidate_actions, dtype=np.float64)
    frame = np.concatenate([
        np.asarray(local_frame[name], dtype=np.float64)
        for name in ("normal", "tangent_up", "tangent_side")
    ])
    if clearance.shape != (7,) or frame.shape != (9,):
        raise ValueError("L5 action-risk state feature shape differs")
    if nominal.shape != (5, 7) or candidate.shape != nominal.shape:
        raise ValueError("L5 action-risk action feature shape differs")
    output = np.concatenate([
        clearance, frame, nominal.reshape(-1), (candidate - nominal).reshape(-1)
    ])
    if output.shape != (86,) or not np.all(np.isfinite(output)):
        raise ValueError("L5 action-risk feature differs")
    return output.tolist()


def prediction_metrics(
    predictions: Sequence[Sequence[float]], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    import numpy as np

    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray([item["risk_l5"] for item in samples], dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("L5 action-risk prediction shape differs")
    predicted_safe = np.max(predicted, axis=1) <= 0.0
    actual_l5_safe = np.max(target, axis=1) <= 0.0
    exact_safe = np.asarray([bool(item["exact_safe"]) for item in samples])
    false_safe = predicted_safe & ~actual_l5_safe
    exact_safe_recalled = predicted_safe & exact_safe
    state_ids = sorted({str(item["state_id"]) for item in samples})
    state_records = []
    for state_id in state_ids:
        indices = np.asarray([
            index for index, item in enumerate(samples)
            if str(item["state_id"]) == state_id
        ], dtype=np.int64)
        recoverable = bool(np.any(exact_safe[indices]))
        supported = bool(np.any(exact_safe_recalled[indices]))
        physical_veto_count = int(np.sum(predicted_safe[indices] & ~exact_safe[indices]))
        accepted = [
            int(index) for index in indices if bool(exact_safe_recalled[index])
        ]
        selected_index = None if not accepted else min(
            accepted, key=lambda index: (
                float(samples[index]["applied_correction_l2_action"]),
                int(samples[index]["candidate_order"]),
            )
        )
        state_records.append({
            "state_id": state_id,
            "recoverable": recoverable,
            "predicted_safe_exact_safe_support": supported,
            "predicted_L5_safe_but_all_seven_physical_veto_count": physical_veto_count,
            "selected_candidate_name": (
                None if selected_index is None
                else samples[selected_index]["candidate_name"]
            ),
            "selected_candidate_exact_all_seven_safe": (
                None if selected_index is None else bool(exact_safe[selected_index])
            ),
        })
    safe_count = int(np.sum(exact_safe))
    return {
        "sample_count": len(samples),
        "rmse_m": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "near_boundary_rmse_m": float(np.sqrt(np.mean(
            (predicted[np.abs(target) <= 0.005] - target[np.abs(target) <= 0.005]) ** 2
        ))) if np.any(np.abs(target) <= 0.005) else None,
        "L5_false_safe_count": int(np.sum(false_safe)),
        "exact_safe_candidate_count": safe_count,
        "exact_safe_candidate_recall": (
            float(np.sum(exact_safe_recalled)) / safe_count if safe_count else 0.0
        ),
        "recoverable_state_count": sum(item["recoverable"] for item in state_records),
        "supported_recoverable_state_count": sum(
            item["recoverable"] and item["predicted_safe_exact_safe_support"]
            for item in state_records
        ),
        "predicted_L5_safe_but_all_seven_physical_veto_count": int(np.sum(
            predicted_safe & ~exact_safe
        )),
        "selected_candidate_count": sum(
            item["selected_candidate_name"] is not None for item in state_records
        ),
        "all_selected_candidates_exact_all_seven_safe": all(
            item["selected_candidate_exact_all_seven_safe"] is not False
            for item in state_records
        ),
        "state_records": state_records,
    }


def build_model(torch: Any, input_dimension: int, hidden_widths: Sequence[int]) -> Any:
    widths = [int(input_dimension)] + [int(item) for item in hidden_widths] + [3]
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


def _weighted_huber(
    torch: Any, prediction: Any, normalized_target: Any, physical_target: Any,
    model_config: Mapping[str, Any],
) -> Any:
    element = torch.nn.functional.smooth_l1_loss(
        prediction, normalized_target,
        beta=float(model_config["huber_beta_normalized"]), reduction="none",
    )
    weight = 1.0 + float(model_config["boundary_weight_multiplier"]) * torch.exp(
        -torch.abs(physical_target) / float(model_config["boundary_scale_m"])
    )
    return torch.sum(weight * element) / torch.sum(weight)


def train_model(
    train_x: Any, train_y: Any, validation_x: Any, validation_y: Any,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit one fixed-epoch model without validation checkpoint selection."""

    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("L5 action-risk training requires one H100 allocation")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError("L5 action-risk training requires an H100")
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    x_validation = np.asarray(validation_x, dtype=np.float64)
    y_validation = np.asarray(validation_y, dtype=np.float64)
    if (
        x_train.ndim != 2
        or x_train.shape[1] != int(model_config["input_dimension"])
        or x_validation.ndim != 2
        or x_validation.shape[1] != x_train.shape[1]
        or y_train.shape != (x_train.shape[0], 3)
        or y_validation.shape != (x_validation.shape[0], 3)
    ):
        raise ValueError("L5 action-risk training arrays differ")
    if not all(np.all(np.isfinite(item)) for item in (
        x_train, y_train, x_validation, y_validation
    )):
        raise ValueError("L5 action-risk training arrays are nonfinite")
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, int(model_config["cpu_threads"])))
    device = torch.device("cuda:0")
    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.where(np.std(x_train, axis=0) >= 1.0e-6,
                             np.std(x_train, axis=0), 1.0)
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
    model = build_model(
        torch, int(model_config["input_dimension"]), model_config["hidden_widths"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_train_loss = math.inf
    for _epoch in range(int(model_config["epochs"])):
        model.train()
        prediction = model(tx)
        loss = _weighted_huber(torch, prediction, ty, ty_physical, model_config)
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
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    payload = {
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "state_dict": {
            name: value.numpy().tolist() for name, value in sorted(state.items())
        },
    }
    return {
        "model": model,
        "device": device,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "epochs_completed": int(model_config["epochs"]),
        "final_train_loss": final_train_loss,
        "validation_loss": validation_loss,
        "model_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
        "state_payload": payload,
        "parameter_count": int(sum(item.numel() for item in model.parameters())),
        "device_name": device_name,
    }


def predict(bundle: Mapping[str, Any], features: Any) -> Any:
    import numpy as np
    import torch

    raw = np.asarray(features, dtype=np.float64)
    normalized = torch.as_tensor(
        (raw - bundle["feature_mean"]) / bundle["feature_scale"],
        dtype=torch.float32, device=bundle["device"],
    )
    bundle["model"].eval()
    with torch.no_grad():
        output = bundle["model"](normalized).detach().cpu().numpy()
    return output * bundle["target_scale"] + bundle["target_mean"]
