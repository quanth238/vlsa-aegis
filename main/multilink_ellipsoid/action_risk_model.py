"""Prediction-only seven-row action-risk MLP.

This module contains no action selection, QP, or simulator control.  It is
usable only after the exact candidate-plus-backup dataset gate authorizes
training.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def build_model(torch: Any, input_dimension: int, hidden_widths: list[int]) -> Any:
    widths = [int(input_dimension)] + [int(item) for item in hidden_widths] + [7]
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


def _weighted_huber(
    torch: Any,
    prediction: Any,
    normalized_target: Any,
    physical_target: Any,
    *,
    beta: float,
    boundary_scale_m: float,
    boundary_weight_multiplier: float,
) -> Any:
    element = torch.nn.functional.smooth_l1_loss(
        prediction, normalized_target, beta=float(beta), reduction="none"
    )
    weight = 1.0 + float(boundary_weight_multiplier) * torch.exp(
        -torch.abs(physical_target) / float(boundary_scale_m)
    )
    return torch.sum(weight * element) / torch.sum(weight)


def train_model(
    train_x: Any,
    train_y: Any,
    validation_x: Any,
    validation_y: Any,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train one preregistered model and return its frozen bundle."""

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("clean action-risk training requires an H100 allocation")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name or torch.cuda.device_count() != 1:
        raise RuntimeError("clean action-risk training requires exactly one H100")
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    x_validation = np.asarray(validation_x, dtype=np.float64)
    y_validation = np.asarray(validation_y, dtype=np.float64)
    if (
        x_train.ndim != 2
        or x_validation.ndim != 2
        or x_train.shape[1] != int(model_config["input_dimension"])
        or x_validation.shape[1] != x_train.shape[1]
        or y_train.shape != (x_train.shape[0], 7)
        or y_validation.shape != (x_validation.shape[0], 7)
    ):
        raise ValueError("clean action-risk training arrays differ")
    if not all(
        np.all(np.isfinite(value))
        for value in (x_train, y_train, x_validation, y_validation)
    ):
        raise ValueError("clean action-risk training arrays are nonfinite")

    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, int(model_config["cpu_threads"])))
    device = torch.device("cuda:0")
    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.std(x_train, axis=0)
    feature_scale = np.where(feature_scale >= 1.0e-6, feature_scale, 1.0)
    target_mean = np.mean(y_train, axis=0)
    target_scale = np.maximum(
        np.std(y_train, axis=0), float(model_config["minimum_target_scale_m"])
    )

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    tx = tensor((x_train - feature_mean) / feature_scale)
    vx = tensor((x_validation - feature_mean) / feature_scale)
    ty_physical = tensor(y_train)
    vy_physical = tensor(y_validation)
    ty = tensor((y_train - target_mean) / target_scale)
    vy = tensor((y_validation - target_mean) / target_scale)
    model = build_model(
        torch, x_train.shape[1], list(model_config["hidden_widths"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    stale = 0
    for epoch in range(int(model_config["epochs"])):
        model.train()
        prediction = model(tx)
        loss = _weighted_huber(
            torch, prediction, ty, ty_physical,
            beta=float(model_config["huber_beta_normalized"]),
            boundary_scale_m=float(model_config["boundary_scale_m"]),
            boundary_weight_multiplier=float(
                model_config["boundary_weight_multiplier"]
            ),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(model_config["gradient_clip_norm"])
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(_weighted_huber(
                torch, model(vx), vy, vy_physical,
                beta=float(model_config["huber_beta_normalized"]),
                boundary_scale_m=float(model_config["boundary_scale_m"]),
                boundary_weight_multiplier=float(
                    model_config["boundary_weight_multiplier"]
                ),
            ).item())
        if validation_loss < best_loss - 1.0e-9:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(model_config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("clean action-risk training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    payload = {
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "state_dict": {
            name: value.numpy().tolist()
            for name, value in sorted(best_state.items())
        },
    }
    return {
        "model": model,
        "device": device,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "epochs_completed": int(epoch + 1),
        "model_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "state_payload": payload,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "device_name": device_name,
    }


def predict(bundle: Mapping[str, Any], features: Any) -> Any:
    import numpy as np
    import torch

    raw = np.asarray(features, dtype=np.float64)
    mean = np.asarray(bundle["feature_mean"], dtype=np.float64)
    scale = np.asarray(bundle["feature_scale"], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != mean.size or scale.shape != mean.shape:
        raise ValueError("clean action-risk prediction features differ")
    normalized = torch.as_tensor(
        (raw - mean) / scale, dtype=torch.float32, device=bundle["device"]
    )
    bundle["model"].eval()
    with torch.no_grad():
        output = bundle["model"](normalized).detach().cpu().numpy()
    return (
        output * np.asarray(bundle["target_scale"], dtype=np.float64)
        + np.asarray(bundle["target_mean"], dtype=np.float64)
    )


def load_frozen_bundle(
    torch: Any, payload: Mapping[str, Any], model_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconstruct a prediction-only bundle from the immutable JSON payload."""

    import numpy as np

    model = build_model(
        torch,
        int(model_config["input_dimension"]),
        list(model_config["hidden_widths"]),
    )
    template = model.state_dict()
    recorded = payload["state_dict"]
    if set(template) != set(recorded):
        raise ValueError("clean action-risk frozen state keys differ")
    state = {}
    for name, value in template.items():
        raw = torch.as_tensor(recorded[name], dtype=value.dtype)
        if raw.shape != value.shape:
            raise ValueError("clean action-risk frozen parameter shape differs")
        state[name] = raw
    model.load_state_dict(state)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    bundle = {
        "model": model,
        "device": device,
        "feature_mean": np.asarray(payload["feature_mean"], dtype=np.float64),
        "feature_scale": np.asarray(payload["feature_scale"], dtype=np.float64),
        "target_mean": np.asarray(payload["target_mean"], dtype=np.float64),
        "target_scale": np.asarray(payload["target_scale"], dtype=np.float64),
    }
    if (
        bundle["feature_mean"].shape != (int(model_config["input_dimension"]),)
        or bundle["feature_scale"].shape != bundle["feature_mean"].shape
        or bundle["target_mean"].shape != (7,)
        or bundle["target_scale"].shape != (7,)
    ):
        raise ValueError("clean action-risk frozen normalization differs")
    return bundle
