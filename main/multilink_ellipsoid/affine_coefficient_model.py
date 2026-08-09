"""State-conditioned affine safety-row targets and neural surrogate.

This module is opt-in.  It learns the quantities a bounded safety QP actually
uses: nominal margin, local action coefficient, and a nonnegative one-sided
error.  Exact cloned OSC rollouts remain the label and verification authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import CONSTRAINT_ORDER, _canonical, _numpy, _torch
from .two_step_margin import PAIR_FEATURE_NAMES


AFFINE_COEFFICIENT_SCHEMA = "vlsa_distal_affine_coefficient_moka10.v1"
AFFINE_COEFFICIENT_DATASET_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_dataset.v1"
)
AFFINE_COEFFICIENT_DATASET_RESULT_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_dataset_result.v1"
)
AFFINE_COEFFICIENT_TRAINING_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_training.v1"
)
AFFINE_COEFFICIENT_RESULT_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_result.v1"
)
AFFINE_COEFFICIENT_VALIDATION_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_validation.v1"
)
AFFINE_COEFFICIENT_WEIGHTS_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_weights.v1"
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_affine_coefficient_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("affine-coefficient config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope",
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "split", "state_sampling", "action_sampling", "coefficient_target",
        "model", "training", "calibration", "projection", "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("affine-coefficient config keys differ")
    if (
        config["schema_version"] != AFFINE_COEFFICIENT_SCHEMA
        or config["protocol_id"] != "vlsa-distal-affine-coefficient-moka10-v1"
    ):
        raise ValueError("affine-coefficient protocol differs")
    split = config["split"]
    groups = {
        name: set(split[name])
        for name in (
            "train_task_groups", "validation_task_groups", "test_task_groups"
        )
    }
    if (
        split.get("unit") != "complete_episode_and_task_level_group"
        or split.get("primary_e05_use")
        != "test_only_never_training_or_calibration"
        or not all(groups.values())
        or groups["train_task_groups"] & groups["validation_task_groups"]
        or groups["train_task_groups"] & groups["test_task_groups"]
        or groups["validation_task_groups"] & groups["test_task_groups"]
    ):
        raise ValueError("affine-coefficient grouped split differs")
    state = config["state_sampling"]
    if state != {
        "search_start_offset_actions": 20,
        "crossing": "first_two_step_exact_proxy_crossing_in_registered_window",
        "state_offsets_from_crossing": [-4, -3, -2, -1, 0],
        "expected_states_per_episode": 5,
    }:
        raise ValueError("affine-coefficient state sampling differs")
    sampling = config["action_sampling"]
    if sampling != {
        "action_limit": 1.0, "trust_region_linf_action": 0.5,
        "grid_points_per_dimension": 5,
        "expected_grid_action_count_per_state": 125,
        "include_exact_nominal_certificate_anchor": True,
        "expected_certificate_action_count_per_state": 126,
        "second_action": "immutable_released_AEGIS_nominal",
    }:
        raise ValueError("affine-coefficient action sampling differs")
    target = config["coefficient_target"]
    if target != {
        "candidate_order": "exact_safe_then_l2_from_nominal_then_grid_index",
        "fit": "candidate_conditioned_minimum_l1_sampled_grid_lower_envelope",
        "one_sided_padding_m": 1.0e-6,
        "target_clearance_m": 0.0,
        "coefficient_postcheck_tolerance_m": 1.0e-8,
        "error_target": "exact_nominal_margin_minus_lower_envelope_intercept",
        "require_every_registered_state": True,
    }:
        raise ValueError("affine-coefficient target contract differs")
    if config["model"] != {
        "input": "factorized_state_nominal_chunk_and_relative_geometry_features",
        "shared_across_constraint_rows": True,
        "hidden_widths": [128, 128], "hidden_activation": "silu",
        "outputs": [
            "nominal_margin_mm", "gradient_x_mm_per_action",
            "gradient_y_mm_per_action", "gradient_z_mm_per_action",
            "nonnegative_error_mm",
        ],
    }:
        raise ValueError("affine-coefficient model differs")
    projection = config["projection"]
    for key in (
        "action_limit", "trust_region_linf_action", "eps_abs", "eps_rel",
        "residual_tolerance", "bound_tolerance_action",
    ):
        if not math.isfinite(float(projection[key])) or float(projection[key]) <= 0:
            raise ValueError("affine-coefficient projection is invalid")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def coefficient_grid_actions(
    nominal_xyz: Sequence[float], config: Mapping[str, Any]
) -> tuple[Any, Any, list[Any]]:
    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    settings = config["action_sampling"]
    limit = float(settings["action_limit"])
    trust = float(settings["trust_region_linf_action"])
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("affine-coefficient nominal action is invalid")
    lower = np.maximum(-limit, nominal - trust)
    upper = np.minimum(limit, nominal + trust)
    axes = [
        np.linspace(lower[index], upper[index], int(settings["grid_points_per_dimension"]))
        for index in range(3)
    ]
    import itertools
    candidates = [
        np.asarray(values, dtype=np.float64) for values in itertools.product(*axes)
    ]
    if len(candidates) != int(settings["expected_grid_action_count_per_state"]):
        raise ValueError("affine-coefficient grid count differs")
    return lower, upper, candidates


def coefficient_targets(
    exact_nominal_margin_m: Sequence[float], certificate: Mapping[str, Any]
) -> dict[str, Any]:
    np = _numpy()
    nominal = np.asarray(exact_nominal_margin_m, dtype=np.float64)
    intercept = np.asarray(certificate["intercept_at_nominal_m"], dtype=np.float64)
    gradient = np.asarray(certificate["gradients_m_per_action"], dtype=np.float64)
    error = nominal - intercept
    if (
        nominal.shape != (7,) or intercept.shape != (7,)
        or gradient.shape != (7, 3) or np.any(error < -1.0e-8)
        or not all(np.all(np.isfinite(value)) for value in (nominal, intercept, gradient))
    ):
        raise ValueError("affine-coefficient targets are invalid")
    error = np.maximum(error, 0.0)
    return {
        "nominal_margin_m": nominal.tolist(),
        "gradient_m_per_action": gradient.tolist(),
        "state_conditioned_error_m": error.tolist(),
        "lower_intercept_m": (nominal - error).tolist(),
    }


def affine_values(
    nominal_margin: Any, gradient: Any, error: Any,
    candidate_xyz: Any, nominal_xyz: Any,
) -> Any:
    np = _numpy()
    return (
        np.asarray(nominal_margin, dtype=np.float64)
        - np.asarray(error, dtype=np.float64)
        + np.asarray(gradient, dtype=np.float64)
        @ (
            np.asarray(candidate_xyz, dtype=np.float64)
            - np.asarray(nominal_xyz, dtype=np.float64)
        )
    )


def build_affine_coefficient_model(hidden_widths: Sequence[int]) -> Any:
    torch = _torch()
    nn = torch.nn

    class AffineCoefficientNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            previous = len(PAIR_FEATURE_NAMES)
            for width in hidden_widths:
                layers.extend((nn.Linear(previous, int(width)), nn.SiLU()))
                previous = int(width)
            layers.append(nn.Linear(previous, 5))
            self.network = nn.Sequential(*layers)

        def forward(self, values: Any) -> Any:
            return self.network(values)

    return AffineCoefficientNet()


def _state_arrays(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    features = np.asarray(
        [item["pair_state_feature_vectors"] for item in states], dtype=np.float64
    )
    nominal = np.asarray(
        [item["coefficient_target"]["nominal_margin_m"] for item in states],
        dtype=np.float64,
    )
    gradients = np.asarray(
        [item["coefficient_target"]["gradient_m_per_action"] for item in states],
        dtype=np.float64,
    )
    errors = np.asarray(
        [item["coefficient_target"]["state_conditioned_error_m"] for item in states],
        dtype=np.float64,
    )
    if (
        features.shape != (len(states), 7, len(PAIR_FEATURE_NAMES))
        or nominal.shape != (len(states), 7)
        or gradients.shape != (len(states), 7, 3)
        or errors.shape != (len(states), 7)
        or not all(np.all(np.isfinite(value)) for value in (
            features, nominal, gradients, errors
        ))
        or np.any(errors < 0.0)
    ):
        raise ValueError("affine-coefficient training arrays are invalid")
    return {
        "features": features, "nominal": nominal, "gradients": gradients,
        "errors": errors,
        "split": np.asarray([item["split"] for item in states], dtype=object),
        "case_id": np.asarray([item["case_id"] for item in states], dtype=object),
    }


def _decode_coefficients(
    raw: Any, output_mean: Any, output_scale: Any, error_scale: float
) -> tuple[Any, Any, Any]:
    torch = _torch()
    first = raw[..., :4] * output_scale + output_mean
    error = torch.nn.functional.softplus(raw[..., 4]) * float(error_scale)
    return first[..., 0], first[..., 1:4], error


def train_affine_coefficient_model(
    states: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Fit `(b,a,e)` from complete state groups; returns metre predictions."""

    import time

    np = _numpy()
    torch = _torch()
    arrays = _state_arrays(states)
    masks = {
        name: arrays["split"] == name
        for name in ("train", "validation", "test")
    }
    if not all(np.any(mask) for mask in masks.values()):
        raise ValueError("affine-coefficient split is empty")
    e05 = "vlsa-t1-goal-ii-t0-e05"
    if e05 not in set(arrays["case_id"][masks["test"]].tolist()):
        raise ValueError("affine-coefficient E05 leaked outside test")
    features = arrays["features"]
    train_features = features[masks["train"]].reshape(-1, features.shape[-1])
    feature_mean = np.mean(train_features, axis=0)
    feature_scale = np.maximum(np.std(train_features, axis=0), 1.0e-6)
    normalized = (features - feature_mean) / feature_scale
    first_four_mm = np.concatenate(
        (arrays["nominal"][..., None], arrays["gradients"]), axis=2
    ) * 1000.0
    train_first = first_four_mm[masks["train"]].reshape(-1, 4)
    output_mean = np.mean(train_first, axis=0)
    output_scale = np.maximum(np.std(train_first, axis=0), 1.0)
    error_mm = arrays["errors"] * 1000.0
    error_scale = max(
        1.0, float(np.sqrt(np.mean(error_mm[masks["train"]] ** 2)))
    )
    target_first = (first_four_mm - output_mean) / output_scale
    target_error = error_mm / error_scale

    settings = config["training"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("affine-coefficient training requires CUDA on H100")
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    model = build_affine_coefficient_model(
        config["model"]["hidden_widths"]
    ).to(device=device, dtype=torch.float64)
    x = torch.as_tensor(normalized, dtype=torch.float64, device=device)
    y_first = torch.as_tensor(target_first, dtype=torch.float64, device=device)
    y_error = torch.as_tensor(target_error, dtype=torch.float64, device=device)
    mean_t = torch.as_tensor(output_mean, dtype=torch.float64, device=device)
    scale_t = torch.as_tensor(output_scale, dtype=torch.float64, device=device)
    mask_t = {
        name: torch.as_tensor(mask, dtype=torch.bool, device=device)
        for name, mask in masks.items()
    }
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    huber = torch.nn.HuberLoss(
        delta=float(settings["huber_delta_standardized"]), reduction="mean"
    )

    def loss_for(name: str) -> Any:
        raw = model(x[mask_t[name]])
        error_prediction = torch.nn.functional.softplus(raw[..., 4])
        return huber(raw[..., :4], y_first[mask_t[name]]) + huber(
            error_prediction, y_error[mask_t[name]]
        )

    best_state = None
    best_epoch = None
    best_validation = math.inf
    stale = 0
    history = []
    started = time.perf_counter_ns()
    for epoch in range(int(settings["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = loss_for("train")
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_for("validation")
        value = float(validation_loss.cpu())
        if epoch == 0 or (epoch + 1) % 100 == 0:
            history.append({
                "epoch": int(epoch), "train_loss": float(train_loss.detach().cpu()),
                "validation_loss": value,
            })
        if value < best_validation - 1.0e-12:
            best_validation = value
            best_epoch = int(epoch)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(settings["patience"]):
            break
    if best_state is None or best_epoch is None:
        raise RuntimeError("affine-coefficient training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw = model(x)
        predicted_b_mm, predicted_a_mm, predicted_e_mm = _decode_coefficients(
            raw, mean_t, scale_t, error_scale
        )
    predicted = {
        "nominal_margin_m": predicted_b_mm.cpu().numpy() / 1000.0,
        "gradient_m_per_action": predicted_a_mm.cpu().numpy() / 1000.0,
        "state_conditioned_error_m": predicted_e_mm.cpu().numpy() / 1000.0,
    }
    state = {
        "state_dict": best_state, "feature_mean": feature_mean,
        "feature_standard_deviation": feature_scale,
        "output_mean_first_four_mm": output_mean,
        "output_scale_first_four_mm": output_scale,
        "error_scale_mm": error_scale,
        "hidden_widths": list(config["model"]["hidden_widths"]),
    }
    audit = {
        "model_class": "shared_state_conditioned_affine_coefficient",
        "device": str(device), "torch_version": str(torch.__version__),
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": best_epoch, "completed_epoch_count": int(epoch + 1),
        "best_validation_objective": best_validation, "history": history,
        "output_mean_first_four_mm": output_mean.tolist(),
        "output_scale_first_four_mm": output_scale.tolist(),
        "error_scale_mm": error_scale,
        "cases_by_split": {
            name: sorted(set(arrays["case_id"][mask].tolist()))
            for name, mask in masks.items()
        },
        "training_wall_seconds": (
            time.perf_counter_ns() - started
        ) * 1.0e-9,
    }
    return model, {**state, "predicted": predicted}, audit


def save_affine_coefficient_model(
    path: Path, state: Mapping[str, Any]
) -> dict[str, Any]:
    np = _numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    names = list(state["state_dict"])
    metadata = {
        "schema_version": AFFINE_COEFFICIENT_WEIGHTS_SCHEMA,
        "feature_names": list(PAIR_FEATURE_NAMES),
        "constraint_order": list(CONSTRAINT_ORDER),
        "hidden_widths": list(state["hidden_widths"]),
        "parameter_names": names,
    }
    arrays = {
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float64),
        "feature_standard_deviation": np.asarray(
            state["feature_standard_deviation"], dtype=np.float64
        ),
        "output_mean_first_four_mm": np.asarray(
            state["output_mean_first_four_mm"], dtype=np.float64
        ),
        "output_scale_first_four_mm": np.asarray(
            state["output_scale_first_four_mm"], dtype=np.float64
        ),
        "error_scale_mm": np.asarray([state["error_scale_mm"]], dtype=np.float64),
        "calibration_m": np.asarray(state["calibration_m"], dtype=np.float64),
        "metadata_utf8": np.frombuffer(_canonical(metadata), dtype=np.uint8),
    }
    for index, name in enumerate(names):
        arrays["parameter_%03d" % index] = (
            state["state_dict"][name].detach().cpu().numpy().astype(np.float64)
        )
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    raw = path.read_bytes()
    return {"path": str(path), "file_sha256": _sha256(raw), "size_bytes": len(raw)}


def load_affine_coefficient_model(
    path: Path, *, device: str = "cpu"
) -> tuple[Any, dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["metadata_utf8"].tolist()).decode("utf-8"))
        if (
            metadata.get("schema_version") != AFFINE_COEFFICIENT_WEIGHTS_SCHEMA
            or metadata.get("feature_names") != list(PAIR_FEATURE_NAMES)
            or metadata.get("constraint_order") != list(CONSTRAINT_ORDER)
        ):
            raise ValueError("affine-coefficient model artifact identity differs")
        model = build_affine_coefficient_model(metadata["hidden_widths"]).to(
            device=torch.device(device), dtype=torch.float64
        )
        names = metadata.get("parameter_names")
        if names != list(model.state_dict()):
            raise ValueError("affine-coefficient parameter names differ")
        state_dict = {}
        for index, name in enumerate(names):
            values = np.asarray(archive["parameter_%03d" % index], dtype=np.float64)
            if values.shape != tuple(model.state_dict()[name].shape):
                raise ValueError("affine-coefficient parameter shape differs")
            state_dict[name] = torch.as_tensor(values, dtype=torch.float64, device=device)
        model.load_state_dict(state_dict)
        model.eval()
        state = {
            "feature_mean": np.asarray(archive["feature_mean"], dtype=np.float64),
            "feature_standard_deviation": np.asarray(
                archive["feature_standard_deviation"], dtype=np.float64
            ),
            "output_mean_first_four_mm": np.asarray(
                archive["output_mean_first_four_mm"], dtype=np.float64
            ),
            "output_scale_first_four_mm": np.asarray(
                archive["output_scale_first_four_mm"], dtype=np.float64
            ),
            "error_scale_mm": float(archive["error_scale_mm"][0]),
            "calibration_m": np.asarray(archive["calibration_m"], dtype=np.float64),
            "hidden_widths": list(metadata["hidden_widths"]),
        }
    return model, state


def predict_affine_coefficients(
    model: Any, model_state: Mapping[str, Any], pair_state_features: Any
) -> tuple[Any, Any, Any]:
    np = _numpy()
    torch = _torch()
    features = np.asarray(pair_state_features, dtype=np.float64)
    if features.shape != (7, len(PAIR_FEATURE_NAMES)):
        raise ValueError("affine-coefficient prediction features differ")
    normalized = (
        features - np.asarray(model_state["feature_mean"], dtype=np.float64)
    ) / np.asarray(model_state["feature_standard_deviation"], dtype=np.float64)
    device = next(model.parameters()).device
    with torch.no_grad():
        raw = model(torch.as_tensor(normalized, dtype=torch.float64, device=device))
        b, a, e = _decode_coefficients(
            raw,
            torch.as_tensor(
                model_state["output_mean_first_four_mm"],
                dtype=torch.float64, device=device,
            ),
            torch.as_tensor(
                model_state["output_scale_first_four_mm"],
                dtype=torch.float64, device=device,
            ),
            float(model_state["error_scale_mm"]),
        )
    return (
        b.cpu().numpy() / 1000.0,
        a.cpu().numpy() / 1000.0,
        e.cpu().numpy() / 1000.0,
    )
