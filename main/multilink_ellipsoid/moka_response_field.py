"""One-task controller-conditioned response field for the E05 Moka pilot.

The learned object is deliberately narrow: for one explicit future
``(action offset, robot ellipsoid)`` witness it predicts the witness value and
its 15-dimensional response to changes in the first five Cartesian XYZ
commands.  Geometry stays outside the network and the compiled Moka obstacle
is represented by its individual collision boxes rather than one MVEE.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .barrier import support_gap
from .geometry import Ellipsoid, primitive_bounding_radii
from .shadow import _numpy


MOKA_RESPONSE_SCHEMA = "vlsa_distal_moka_response_field_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_moka_response_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "action_space",
        "case_ids",
        "claim_scope",
        "comparators",
        "gate",
        "geometry",
        "model",
        "protocol_id",
        "sampling",
        "score",
        "state_split",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Moka response config keys differ")
    if value.get("protocol_id") != "vlsa-distal-moka-response-field-e05-v1":
        raise ValueError("Moka response protocol differs")
    if value.get("case_ids") != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("Moka response case differs")
    if value.get("state_split") != {
        "test_steps": [185, 186],
        "train_steps": [178, 179, 180, 181, 182],
        "validation_steps": [183, 184],
    }:
        raise ValueError("Moka response state split differs")
    if value.get("action_space") != {
        "action_limit": 1.0,
        "corrected_dimensions": [0, 1, 2],
        "corrected_horizon": 5,
        "evaluation_horizon": 20,
        "endpoint_preservation": False,
    }:
        raise ValueError("Moka response action space differs")
    sampling = value.get("sampling", {})
    if sampling.get("paired_direction_count_per_state") != 32:
        raise ValueError("Moka response paired count differs")
    if sampling.get("fit_direction_count") != 24:
        raise ValueError("Moka response fit count differs")
    if sampling.get("paired_perturbation_action") != 0.05:
        raise ValueError("Moka response perturbation differs")
    if sampling.get("test_radii_action") != [0.1, 0.25, 0.5]:
        raise ValueError("Moka response radii differ")
    if sampling.get("matched_random_direction_count") != 64:
        raise ValueError("Moka response random count differs")
    geometry = value.get("geometry", {})
    if geometry.get("expected_compiled_moka_box_count") != 15:
        raise ValueError("Moka response primitive count differs")
    if geometry.get("robot_ellipsoid_count") != 7:
        raise ValueError("Moka response robot row count differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["schema_version"] = MOKA_RESPONSE_SCHEMA
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def compiled_box_ellipsoids(boxes: Sequence[Any]) -> list[Ellipsoid]:
    """Return one tight Loewner ellipsoid for every compiled collision box."""

    np = _numpy()
    output = []
    for box in boxes:
        radii, source = primitive_bounding_radii(
            "box",
            np.asarray(box.half_extents_m, dtype=np.float64),
            float(np.linalg.norm(box.half_extents_m)),
        )
        output.append(
            Ellipsoid(
                center=np.asarray(box.center, dtype=np.float64),
                rotation=np.asarray(box.rotation, dtype=np.float64),
                semiaxes_m=np.asarray(radii, dtype=np.float64),
                body_id=int(box.body_id),
                body_name=str(box.body_name),
                geom_id=int(box.geom_id),
                geom_name=str(box.geom_name),
                bound_source=str(source),
            )
        )
    if not output:
        raise ValueError("Moka response requires compiled obstacle boxes")
    return output


def multi_primitive_link_margins(links: Sequence[Ellipsoid], boxes: Sequence[Any]) -> Any:
    """Minimum support gap to the union proxy, separately for seven robot rows.

    The result is a smooth quantitative optimization proxy in metres.  It is
    *not* the physical contact authority; raw MuJoCo contacts are retained for
    acceptance and exact box/ellipsoid overlap is reported separately.
    """

    np = _numpy()
    if len(links) != 7:
        raise ValueError("Moka response requires seven robot ellipsoids")
    obstacles = (
        list(boxes)
        if boxes and isinstance(boxes[0], Ellipsoid)
        else compiled_box_ellipsoids(boxes)
    )
    values = np.asarray(
        [min(support_gap(link, obstacle) for obstacle in obstacles) for link in links],
        dtype=np.float64,
    )
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("Moka response margin vector differs")
    return values


def witness_features(context: Any, horizon: int = 20, rows: int = 7) -> Any:
    """Expand one physical context into explicit time/link witness inputs."""

    np = _numpy()
    base = np.asarray(context, dtype=np.float64).reshape(-1)
    if base.size < 1 or not np.all(np.isfinite(base)):
        raise ValueError("Moka response context differs")
    features = []
    for offset in range(int(horizon)):
        phase = 0.0 if horizon <= 1 else float(offset) / float(horizon - 1)
        time_features = np.asarray(
            [phase, math.sin(math.pi * phase), math.cos(math.pi * phase)],
            dtype=np.float64,
        )
        for row in range(int(rows)):
            identity = np.zeros(rows, dtype=np.float64)
            identity[row] = 1.0
            features.append(np.concatenate([base, time_features, identity]))
    return np.asarray(features, dtype=np.float64)


def build_response_model(torch: Any, input_dimension: int, hidden_units: int) -> Any:
    class ResponseModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(hidden_units)
            self.network = torch.nn.Sequential(
                torch.nn.Linear(int(input_dimension), width),
                torch.nn.SiLU(),
                torch.nn.Linear(width, width),
                torch.nn.SiLU(),
                torch.nn.Linear(width, 16),
            )

        def forward(self, value: Any) -> tuple[Any, Any]:
            output = self.network(value)
            return output[..., 0], output[..., 1:]

    return ResponseModel()


def _materialize(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    features = []
    values = []
    directions = []
    targets = []
    witness_indexes = []
    state_indexes = []
    for state_index, state in enumerate(states):
        state_features = witness_features(state["context"])
        base = np.asarray(state["base_values_m"], dtype=np.float64).reshape(-1)
        design = np.asarray(state["fit_directions"], dtype=np.float64)
        directional = np.asarray(state["fit_targets_m_per_action"], dtype=np.float64)
        if state_features.shape[0] != 140 or base.shape != (140,):
            raise ValueError("Moka response witness arrays differ")
        if design.ndim != 2 or design.shape[1] != 15:
            raise ValueError("Moka response direction arrays differ")
        if directional.shape != (design.shape[0], 140):
            raise ValueError("Moka response directional targets differ")
        feature_start = len(features)
        features.extend(state_features)
        values.extend(base)
        for direction_index, direction in enumerate(design):
            for witness_index in range(140):
                directions.append(direction)
                targets.append(directional[direction_index, witness_index])
                witness_indexes.append(feature_start + witness_index)
                state_indexes.append(state_index)
    return {
        "features": np.asarray(features, dtype=np.float64),
        "values": np.asarray(values, dtype=np.float64),
        "directions": np.asarray(directions, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.float64),
        "witness_indexes": np.asarray(witness_indexes, dtype=np.int64),
        "state_indexes": np.asarray(state_indexes, dtype=np.int64),
    }


def train_response_model(
    train_states: Sequence[Mapping[str, Any]],
    validation_states: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the explicit per-witness value/response model on an H100."""

    import torch

    np = _numpy()
    if not torch.cuda.is_available():
        raise RuntimeError("Moka response training requires H100 CUDA allocation")
    train = _materialize(train_states)
    validation = _materialize(validation_states)
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    feature_mean = np.mean(train["features"], axis=0)
    feature_scale = np.std(train["features"], axis=0)
    feature_scale = np.where(feature_scale >= 1.0e-6, feature_scale, 1.0)
    value_scale = max(float(np.sqrt(np.mean(train["values"] ** 2))), 1.0e-3)
    response_scale = max(float(np.sqrt(np.mean(train["targets"] ** 2))), 1.0e-3)

    def tensor(value: Any, dtype: Any = torch.float32) -> Any:
        return torch.as_tensor(value, dtype=dtype, device=device)

    train_x = tensor((train["features"] - feature_mean) / feature_scale)
    train_values = tensor(train["values"] / value_scale)
    train_directions = tensor(train["directions"])
    train_targets = tensor(train["targets"] / response_scale)
    train_indexes = tensor(train["witness_indexes"], torch.long)
    validation_x = tensor((validation["features"] - feature_mean) / feature_scale)
    validation_values = tensor(validation["values"] / value_scale)
    validation_directions = tensor(validation["directions"])
    validation_targets = tensor(validation["targets"] / response_scale)
    validation_indexes = tensor(validation["witness_indexes"], torch.long)

    model = build_response_model(
        torch, train_x.shape[1], int(model_config["hidden_units"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    patience = int(model_config["patience"])
    stale = 0
    for epoch in range(int(model_config["epochs"])):
        model.train()
        predicted_values, predicted_gradients = model(train_x)
        response_prediction = torch.sum(
            predicted_gradients[train_indexes] * train_directions, dim=1
        )
        value_loss = torch.nn.functional.smooth_l1_loss(
            predicted_values, train_values, beta=0.1
        )
        response_loss = torch.nn.functional.smooth_l1_loss(
            response_prediction, train_targets, beta=0.1
        )
        loss = value_loss + float(model_config["response_loss_weight"]) * response_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_value_prediction, validation_gradients = model(validation_x)
            validation_response = torch.sum(
                validation_gradients[validation_indexes] * validation_directions, dim=1
            )
            validation_loss = float(
                (
                    torch.mean((validation_value_prediction - validation_values) ** 2)
                    + float(model_config["response_loss_weight"])
                    * torch.mean((validation_response - validation_targets) ** 2)
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
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Moka response training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    state_payload = {
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "value_scale": value_scale,
        "response_scale": response_scale,
        "state_dict": {
            name: value.numpy().tolist() for name, value in sorted(best_state.items())
        },
    }
    return {
        "model": model,
        "device": device,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "value_scale": value_scale,
        "response_scale": response_scale,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "model_sha256": hashlib.sha256(_canonical(state_payload)).hexdigest(),
        "state_payload": state_payload,
        "train_witness_count": int(train["features"].shape[0]),
        "train_directional_equation_count": int(train["targets"].shape[0]),
        "validation_witness_count": int(validation["features"].shape[0]),
        "validation_directional_equation_count": int(validation["targets"].shape[0]),
    }


def predict_response(bundle: Mapping[str, Any], context: Any) -> tuple[Any, Any]:
    """Predict 20x7 values and their 15D Cartesian response rows."""

    import torch

    np = _numpy()
    features = witness_features(context)
    normalized = (features - np.asarray(bundle["feature_mean"])) / np.asarray(
        bundle["feature_scale"]
    )
    value = torch.as_tensor(normalized, dtype=torch.float32, device=bundle["device"])
    bundle["model"].eval()
    with torch.no_grad():
        predicted_values, predicted_gradients = bundle["model"](value)
    values = predicted_values.detach().cpu().numpy() * float(bundle["value_scale"])
    gradients = predicted_gradients.detach().cpu().numpy() * float(
        bundle["response_scale"]
    )
    return values.reshape(20, 7), gradients.reshape(20, 7, 15)
