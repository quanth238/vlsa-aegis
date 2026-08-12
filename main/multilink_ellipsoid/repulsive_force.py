"""Physical-basis learned repulsive potential for the local E05 gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .shadow import _numpy


REPULSIVE_FORCE_SCHEMA = "vlsa_distal_repulsive_force_direction_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_repulsive_force_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("repulsive-force config is invalid JSON") from error
    required = {
        "case_ids",
        "claim_scope",
        "comparators",
        "continuation",
        "finite_difference",
        "gate",
        "model",
        "nominal_action_source",
        "protected_geometry",
        "protocol_id",
        "sampling",
        "schema_version",
        "score",
        "state_split",
        "success_definition",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("repulsive-force config keys differ")
    if value["schema_version"] != REPULSIVE_FORCE_SCHEMA:
        raise ValueError("repulsive-force schema differs")
    if value["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("repulsive-force case differs")
    if value["state_split"] != {
        "test_steps": [185],
        "train_steps": [182, 183],
        "validation_steps": [184],
    }:
        raise ValueError("repulsive-force state split differs")
    finite_difference = value["finite_difference"]
    if finite_difference != {
        "action_dimensions": [0, 1, 2],
        "perturbation_action": 0.05,
        "scheme": "clipped_central_difference_of_two_action_rollout_minimum",
    }:
        raise ValueError("repulsive-force finite difference differs")
    sampling = value["sampling"]
    if sampling["paired_direction_count_per_state"] != 128:
        raise ValueError("repulsive-force training direction count differs")
    if sampling["random_test_direction_count"] != 256:
        raise ValueError("repulsive-force random comparator count differs")
    if sampling["train_radii_action"] != [0.1, 0.25]:
        raise ValueError("repulsive-force training radii differ")
    if sampling["test_radii_action"] != [0.1, 0.25]:
        raise ValueError("repulsive-force test radii differ")
    score = value["score"]
    if score != {
        "future_only_excludes_k0": True,
        "rollout_actions": 2,
        "softmin_temperature_m": 0.002,
        "target_clearance_m": 0.0,
    }:
        raise ValueError("repulsive-force score differs")
    if value["protected_geometry"] != {
        "constraint_count": 7,
        "end_effector_proxy": "diagnostic_only_unchanged_released_aegis_proxy",
        "optimizer_clearance_m": 0.0,
        "source": "accepted_v4_seven_slab_bounds_for_link5_link6_link7",
    }:
        raise ValueError("repulsive-force geometry differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def softmin(values: Any, temperature: float) -> float:
    np = _numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    tau = float(temperature)
    if array.size < 1 or not np.all(np.isfinite(array)) or tau <= 0.0:
        raise ValueError("softmin inputs are invalid")
    offset = float(np.min(array))
    return float(offset - tau * np.log(np.sum(np.exp(-(array - offset) / tau))))


def softmin_weights(values: Any, temperature: float) -> Any:
    np = _numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    tau = float(temperature)
    if array.size < 1 or not np.all(np.isfinite(array)) or tau <= 0.0:
        raise ValueError("softmin weight inputs are invalid")
    logits = -(array - float(np.min(array))) / tau
    weights = np.exp(logits)
    return weights / np.sum(weights)


def normalized_direction(vector: Any) -> Any:
    np = _numpy()
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.all(np.isfinite(value)) or norm <= 1.0e-12:
        raise ValueError("repulsive direction is degenerate")
    return value / norm


def fixed_repulsive_direction(base_margins: Any, rows: Any, temperature: float) -> Any:
    np = _numpy()
    base = np.asarray(base_margins, dtype=np.float64)
    matrix = np.asarray(rows, dtype=np.float64)
    if base.shape != (7,) or matrix.shape != (7, 3):
        raise ValueError("fixed repulsive basis shape differs")
    return normalized_direction(matrix.T.dot(softmin_weights(base, temperature)))


def paired_feasible_directions(
    count: int,
    seed: int,
    nominal_xyz: Any,
    maximum_radius: float,
    action_limit: float,
) -> Any:
    """Generate antithetic unit directions whose equal norms need no clipping."""

    np = _numpy()
    if count < 2 or count % 2:
        raise ValueError("paired direction count must be a positive even number")
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("nominal XYZ is invalid")
    radius = float(maximum_radius)
    limit = float(action_limit)
    if radius <= 0.0 or limit <= 0.0:
        raise ValueError("direction radius and limit must be positive")
    rng = np.random.RandomState(int(seed))
    positive = []
    attempts = 0
    while len(positive) < count // 2:
        attempts += 1
        if attempts > count * 1000:
            raise ValueError("could not sample enough paired feasible directions")
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            continue
        direction /= norm
        plus = nominal + radius * direction
        minus = nominal - radius * direction
        if np.max(np.abs(plus)) > limit or np.max(np.abs(minus)) > limit:
            continue
        positive.append(direction)
    output = []
    for direction in positive:
        output.extend([direction, -direction])
    return np.asarray(output, dtype=np.float64)


def matched_random_p_value(learned_gain: float, random_gains: Any) -> float:
    np = _numpy()
    values = np.asarray(random_gains, dtype=np.float64).reshape(-1)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("random gains are invalid")
    equally_or_better = int(np.sum(values >= float(learned_gain)))
    return float((equally_or_better + 1) / (values.size + 1))


def build_monotone_model(torch: Any, hidden_units: int) -> Any:
    """Create a scalar potential monotone in each of seven clearances."""

    class PositiveLinear(torch.nn.Module):
        def __init__(self, inputs: int, outputs: int, *, bias: bool = True) -> None:
            super().__init__()
            self.raw_weight = torch.nn.Parameter(torch.empty(outputs, inputs))
            self.bias = torch.nn.Parameter(torch.zeros(outputs)) if bias else None
            torch.nn.init.normal_(self.raw_weight, mean=-1.5, std=0.15)

        def forward(self, value: Any) -> Any:
            weight = torch.nn.functional.softplus(self.raw_weight)
            return torch.nn.functional.linear(value, weight, self.bias)

    class MonotonePotential(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = PositiveLinear(7, int(hidden_units))
            self.second = PositiveLinear(int(hidden_units), 1)

        def forward(self, value: Any) -> Any:
            hidden = torch.tanh(self.first(value))
            return self.second(hidden).squeeze(-1)

    return MonotonePotential()


def train_monotone_potential(
    train_inputs: Any,
    train_targets: Any,
    validation_inputs: Any,
    validation_targets: Any,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train on H100 allocation CPU and freeze by validation loss only."""

    import torch

    np = _numpy()
    torch.set_num_threads(1)
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    train_x = np.asarray(train_inputs, dtype=np.float64)
    train_y = np.asarray(train_targets, dtype=np.float64).reshape(-1)
    validation_x = np.asarray(validation_inputs, dtype=np.float64)
    validation_y = np.asarray(validation_targets, dtype=np.float64).reshape(-1)
    if train_x.ndim != 2 or train_x.shape[1] != 7 or train_x.shape[0] != train_y.size:
        raise ValueError("repulsive-force training arrays differ")
    if validation_x.ndim != 2 or validation_x.shape[1] != 7 or validation_x.shape[0] != validation_y.size:
        raise ValueError("repulsive-force validation arrays differ")
    input_scale = np.maximum(np.sqrt(np.mean(train_x * train_x, axis=0)), 1.0e-4)
    target_scale = max(float(np.sqrt(np.mean(train_y * train_y))), 1.0e-4)
    train_x_t = torch.as_tensor(train_x / input_scale, dtype=torch.float64)
    train_y_t = torch.as_tensor(train_y / target_scale, dtype=torch.float64)
    validation_x_t = torch.as_tensor(validation_x / input_scale, dtype=torch.float64)
    validation_y_t = torch.as_tensor(validation_y / target_scale, dtype=torch.float64)
    model = build_monotone_model(torch, int(model_config["hidden_units"])).double()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed + 1)
    batch_size = min(int(model_config["batch_size"]), train_x.shape[0])
    best_loss = math.inf
    best_epoch = None
    best_state = None
    epochs = int(model_config["epochs"])
    for epoch in range(epochs):
        permutation = torch.randperm(train_x.shape[0], generator=generator)
        model.train()
        for start in range(0, train_x.shape[0], batch_size):
            indexes = permutation[start : start + batch_size]
            prediction = model(train_x_t[indexes])
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, train_y_t[indexes], beta=0.1
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x_t)
            validation_loss = float(
                torch.mean((validation_prediction - validation_y_t) ** 2).item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None or best_epoch is None:
        raise ValueError("repulsive-force training did not create a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_prediction = (
            model(train_x_t).detach().cpu().numpy() * target_scale
        )
        validation_prediction = (
            model(validation_x_t).detach().cpu().numpy() * target_scale
        )
    state_payload = {
        "input_scale": input_scale.tolist(),
        "target_scale": target_scale,
        "state_dict": {
            name: tensor.numpy().tolist() for name, tensor in sorted(best_state.items())
        },
    }
    return {
        "model": model,
        "input_scale": input_scale,
        "target_scale": target_scale,
        "model_sha256": hashlib.sha256(_canonical(state_payload)).hexdigest(),
        "best_epoch": int(best_epoch),
        "best_validation_normalized_mse": float(best_loss),
        "train_rmse_m": float(np.sqrt(np.mean((train_prediction - train_y) ** 2))),
        "validation_rmse_m": float(
            np.sqrt(np.mean((validation_prediction - validation_y) ** 2))
        ),
        "state_payload": state_payload,
    }


def learned_repulsive_direction(bundle: Mapping[str, Any], base_margins: Any, rows: Any) -> tuple[Any, Any]:
    """Differentiate the frozen monotone potential through the physical rows."""

    import torch

    np = _numpy()
    base = np.asarray(base_margins, dtype=np.float64)
    matrix = np.asarray(rows, dtype=np.float64)
    if base.shape != (7,) or matrix.shape != (7, 3):
        raise ValueError("learned repulsive basis shape differs")
    input_scale = np.asarray(bundle["input_scale"], dtype=np.float64)
    target_scale = float(bundle["target_scale"])
    value = torch.tensor(
        base / input_scale,
        dtype=torch.float64,
        requires_grad=True,
    )
    prediction = bundle["model"](value[None, :]).squeeze() * target_scale
    gradient = torch.autograd.grad(prediction, value)[0].detach().cpu().numpy()
    gradient_physical = gradient / input_scale
    if np.any(gradient_physical < -1.0e-12):
        raise ValueError("monotone potential produced a negative clearance weight")
    direction = normalized_direction(matrix.T.dot(gradient_physical))
    return direction, gradient_physical
