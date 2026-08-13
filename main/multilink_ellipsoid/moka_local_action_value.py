"""Matched scalar-response and nonlinear local action-value models.

This module is prediction-only.  It deliberately contains no optimizer,
action correction, simulator execution, or QP integration.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


def _numpy() -> Any:
    import numpy as np

    return np


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def build_scalar_model(torch: Any, input_dimension: int, hidden_units: int) -> Any:
    """Return the shared scalar architecture used by both matched arms."""

    return torch.nn.Sequential(
        torch.nn.Linear(int(input_dimension), int(hidden_units)),
        torch.nn.SiLU(),
        torch.nn.Linear(int(hidden_units), int(hidden_units)),
        torch.nn.SiLU(),
        torch.nn.Linear(int(hidden_units), 1),
    )


def train_scalar_model(
    train_x: Any,
    train_y: Any,
    validation_x: Any,
    validation_y: Any,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train one scalar regressor under the frozen matched protocol."""

    import torch

    np = _numpy()
    if not torch.cuda.is_available():
        raise RuntimeError("Moka local action-value training requires H100 allocation")
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64).reshape(-1)
    x_validation = np.asarray(validation_x, dtype=np.float64)
    y_validation = np.asarray(validation_y, dtype=np.float64).reshape(-1)
    if (
        x_train.ndim != 2
        or x_validation.ndim != 2
        or x_train.shape[1] != x_validation.shape[1]
        or x_train.shape[0] != y_train.size
        or x_validation.shape[0] != y_validation.size
    ):
        raise ValueError("scalar training arrays differ")
    if not all(
        np.all(np.isfinite(value))
        for value in (x_train, y_train, x_validation, y_validation)
    ):
        raise ValueError("scalar training arrays are nonfinite")

    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(8, int(model_config.get("cpu_threads", 8))))
    device = torch.device("cpu")
    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.std(x_train, axis=0)
    feature_scale = np.where(feature_scale >= 1.0e-6, feature_scale, 1.0)
    target_mean = float(np.mean(y_train))
    target_scale = max(float(np.std(y_train)), 1.0e-4)

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    normalized_train_x = tensor((x_train - feature_mean) / feature_scale)
    normalized_validation_x = tensor(
        (x_validation - feature_mean) / feature_scale
    )
    normalized_train_y = tensor((y_train - target_mean) / target_scale)
    normalized_validation_y = tensor(
        (y_validation - target_mean) / target_scale
    )
    model = build_scalar_model(
        torch, x_train.shape[1], int(model_config["hidden_units"])
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
        prediction = model(normalized_train_x).reshape(-1)
        loss = torch.nn.functional.smooth_l1_loss(
            prediction, normalized_train_y, beta=0.1
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(normalized_validation_x).reshape(-1)
            validation_loss = float(
                torch.mean(
                    (validation_prediction - normalized_validation_y) ** 2
                ).item()
            )
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
        raise RuntimeError("scalar training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    payload = {
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "state_dict": {
            name: value.numpy().tolist() for name, value in sorted(best_state.items())
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
        "model_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "state_payload": payload,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
    }


def predict_scalar(bundle: Mapping[str, Any], features: Any) -> Any:
    """Evaluate a frozen scalar bundle and return physical-unit predictions."""

    import torch

    np = _numpy()
    raw = np.asarray(features, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != np.asarray(bundle["feature_mean"]).size:
        raise ValueError("scalar prediction features differ")
    normalized = (raw - np.asarray(bundle["feature_mean"])) / np.asarray(
        bundle["feature_scale"]
    )
    value = torch.as_tensor(normalized, dtype=torch.float32, device=bundle["device"])
    bundle["model"].eval()
    with torch.no_grad():
        prediction = bundle["model"](value).reshape(-1).detach().cpu().numpy()
    return prediction * float(bundle["target_scale"]) + float(bundle["target_mean"])
