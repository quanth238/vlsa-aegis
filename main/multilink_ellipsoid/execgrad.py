"""Direction-first controller-conditioned execution-gradient pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .complete_osc_margin import _canonical, _numpy, _torch
from .factorized_direct_horizon_displacement import (
    _huber,
    horizon_time_encoding,
    temporal_error_metrics,
)
from .factorized_execution_pilot import _arm_targets


CONFIG_SCHEMA = "vlsa_distal_execgrad_direction_config.v1"
PREPROCESS_SCHEMA = "vlsa_distal_execgrad_kinematic_preprocess.v1"
WEIGHTS_SCHEMA = "vlsa_distal_execgrad_weights.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_execgrad_config(path: Path) -> dict[str, Any]:
    """Load the frozen direction-first protocol without permissive defaults."""

    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version",
        "protocol_id",
        "claim_scope",
        "immutable_source",
        "population",
        "nominal_rollout",
        "architecture",
        "training",
        "direction_score",
        "fitted_direction_gate",
        "conditional_next_stage",
        "forbidden_before_direction_pass",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("ExecGrad config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-execgrad-direction-moka10-v1"
    ):
        raise ValueError("ExecGrad protocol differs")
    if config["architecture"] != {
        "input_representation": "structured_complete_OSC_state_and_full_two_action_chunk",
        "state_action_encoder_widths": [256, 256],
        "time_decoder_width": 192,
        "activation": "silu",
        "trace_state_count": 51,
        "joint_dimension": 7,
        "time_encoding_dimension": 9,
        "nominal_input_dimension": 7,
        "output": "direct_nonrecursive_residual_over_deterministic_kinematic_nominal_q",
        "exact_initial_condition": "qhat_0_equals_q0_architecturally",
    }:
        raise ValueError("ExecGrad architecture differs")
    nominal = config["nominal_rollout"]
    if nominal != {
        "method": "two_segment_damped_resolved_rate_endpoint_interpolation",
        "controller_action_scaling": "authoritative_OSC_scale_action",
        "substeps_per_action": 25,
        "damping": 0.05,
        "maximum_absolute_joint_delta_per_segment_rad": 0.35,
        "gripper_changes_arm_nominal": False,
        "no_cloned_OSC_or_future_q_used": True,
        "validation": {
            "require_deterministic_repeat": True,
            "require_finite_in_bounds_q": True,
            "record_scaled_pose_basis": True,
            "maximum_center_JVP_linearization_RMSE_m_per_action": 0.001,
            "minimum_center_JVP_cosine": 0.95,
        },
    }:
        raise ValueError("ExecGrad nominal-rollout contract differs")
    training = config["training"]
    if training != {
        "device": "cpu_inside_H100_allocation_due_to_pinned_PyTorch_missing_sm90_kernels",
        "seed": 20260881,
        "batch_size": 512,
        "epochs": 300,
        "patience": 40,
        "learning_rate": 0.001,
        "weight_decay": 1e-06,
        "joint_huber_delta_rad": 0.002,
        "joint_trajectory_loss_weight": 1.0,
        "symmetric_safety_normal_loss_weight": 10.0,
        "geometry_huber_delta_m": 0.002,
        "near_boundary_absolute_margin_m": 0.005,
        "near_boundary_weight_multiplier": 5.0,
        "execgrad_link_center_JVP_loss_weight": 1.0,
        "JVP_scale": "training_only_RMS_L2_link_center_response_by_pose_coordinate_and_horizon",
        "JVP_minimum_RMS_m_per_action": 1e-06,
        "JVP_weight_relative_clip": [0.001, 1000.0],
        "steering_action_dimensions": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        "gripper_dimensions_are_inputs_but_not_steering_targets": [6, 13],
        "matched_arms": ["trajectory_only", "trajectory_plus_link_JVP"],
        "same_initialization_optimizer_data_and_checkpoint_metric": True,
        "checkpoint_metric": "validation_joint_RMSE_plus_0.1_validation_normalized_link_JVP_RMSE",
    }:
        raise ValueError("ExecGrad training contract differs")
    direction = config["direction_score"]
    if direction != {
        "score": "soft_min_over_51_by_7_ellipsoid_support_gaps",
        "soft_min_temperature_m": 0.001,
        "gradient": "central_secant_over_registered_pose_coordinate_pairs",
        "zero_exact_gradient_norm_threshold_m_per_action": 1e-06,
        "matched_random_direction_count_per_state": 256,
        "random_seed": 20260882,
        "action_norm": "unit_L2_in_normalized_pose_action_coordinates",
        "old_E05_E10_E15_are_diagnostic_not_final_generalization": True,
    }:
        raise ValueError("ExecGrad direction score differs")
    forbidden = config["forbidden_before_direction_pass"]
    expected_forbidden = {
        "new_unseen_episode_opening",
        "fresh_cloned_OSC_direction_rollouts",
        "flow_guidance",
        "calibration",
        "QP",
        "closed_loop",
        "poisson_or_SDF",
        "binary_classifier",
    }
    if set(forbidden) != expected_forbidden or not all(
        bool(value) for value in forbidden.values()
    ):
        raise ValueError("ExecGrad forbidden-action contract differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _build_model(input_dimension: int, architecture: Mapping[str, Any]) -> Any:
    """Build a direct-horizon residual over an explicit kinematic nominal."""

    torch = _torch()
    nn = torch.nn
    widths = [int(value) for value in architecture["state_action_encoder_widths"]]
    decoder = int(architecture["time_decoder_width"])
    time = torch.as_tensor(horizon_time_encoding(), dtype=torch.float32)

    class ExecGradNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(int(input_dimension), widths[0]),
                nn.SiLU(),
                nn.Linear(widths[0], widths[1]),
                nn.SiLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(widths[1] + 9 + 7, decoder),
                nn.SiLU(),
                nn.Linear(decoder, 7),
            )
            self.register_buffer("horizon_time_encoding", time.clone())

        def forward(self, features: Any, nominal_q: Any) -> Any:
            if nominal_q.ndim != 3 or tuple(nominal_q.shape[1:]) != (51, 7):
                raise ValueError("ExecGrad nominal-q tensor differs")
            latent = self.encoder(features)
            repeated = latent[:, None, :].expand(-1, 51, -1)
            encoded_time = self.horizon_time_encoding[None].expand(
                len(features), -1, -1
            )
            residual = self.decoder(
                torch.cat((repeated, encoded_time, nominal_q), dim=2)
            )
            predicted = nominal_q + residual
            # The current measured q is not predicted.
            return torch.cat((nominal_q[:, :1], predicted[:, 1:]), dim=1)

    return ExecGradNet()


def link_jvp_scales(
    exact: Any,
    dimensions: Any,
    training_rows: Any,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit frozen coordinate/horizon scales from training episodes only."""

    np = _numpy()
    values = np.asarray(exact, dtype=np.float64)
    dims = np.asarray(dimensions, dtype=np.int64)
    rows = np.asarray(training_rows, dtype=np.int64)
    if values.ndim != 5 or values.shape[1:] != (51, 7, 3,):
        raise ValueError("ExecGrad link JVP shape differs")
    steering = set(int(value) for value in settings["steering_action_dimensions"])
    scale = np.zeros((14, 51), dtype=np.float64)
    count = np.zeros((14, 51), dtype=np.int64)
    selected = np.zeros(len(values), dtype=bool)
    selected[rows] = True
    for dimension in steering:
        use = selected & (dims == dimension)
        if np.any(use):
            squared = np.sum(values[use] ** 2, axis=(2, 3))
            scale[dimension] = np.sqrt(np.mean(squared, axis=0))
            count[dimension] = int(np.count_nonzero(use))
    valid = scale >= float(settings["JVP_minimum_RMS_m_per_action"])
    raw = np.zeros_like(scale)
    raw[valid] = 1.0 / (scale[valid] ** 2 + 1.0e-15)
    if not np.any(valid):
        raise ValueError("ExecGrad has no valid nonzero link JVP cell")
    median = float(np.median(raw[valid]))
    lower, upper = [
        median * float(value) for value in settings["JVP_weight_relative_clip"]
    ]
    weight = np.zeros_like(scale)
    weight[valid] = np.clip(raw[valid], lower, upper)
    return {
        "RMS_scale_m_per_action": scale,
        "sample_count": count,
        "valid": valid,
        "weight": weight,
        "raw_weight_median": median,
        "weight_clip_lower": lower,
        "weight_clip_upper": upper,
        "valid_cell_count": int(np.count_nonzero(valid)),
    }


def link_jvp_loss(
    predicted: Any,
    exact: Any,
    dimensions: Any,
    scale: Mapping[str, Any],
) -> Any:
    """Return the normalized vector error for all seven link centers."""

    torch = _torch()
    dims = torch.as_tensor(dimensions, dtype=torch.long)
    weight = torch.as_tensor(scale["weight"], dtype=predicted.dtype)[dims]
    valid = torch.as_tensor(scale["valid"], dtype=torch.bool)[dims]
    squared = torch.sum((predicted - exact) ** 2, dim=(2, 3))
    denominator = torch.sum(valid)
    if int(denominator.detach().cpu()) <= 0:
        raise ValueError("ExecGrad JVP batch has no valid cells")
    return torch.sum(squared * weight * valid) / denominator


def _normalized_jvp_rmse(
    predicted: Any, exact: Any, dimensions: Any, scale: Mapping[str, Any]
) -> float:
    np = _numpy()
    dims = np.asarray(dimensions, dtype=np.int64)
    valid = np.asarray(scale["valid"], dtype=bool)[dims]
    rms = np.asarray(scale["RMS_scale_m_per_action"], dtype=np.float64)[dims]
    squared = np.sum(
        (np.asarray(predicted) - np.asarray(exact)) ** 2, axis=(2, 3)
    )
    if not np.any(valid):
        return math.inf
    return float(np.sqrt(np.mean(squared[valid] / (rms[valid] ** 2 + 1e-15))))


def train_matched_execgrad_models(
    *,
    arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any],
    local_geometry: Mapping[str, Any],
    nominal_q: Any,
    center_jacobian: Any,
    exact_center_jvp: Any,
    config: Mapping[str, Any],
    normalization: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Train the matched trajectory-only and link-JVP execution models."""

    np = _numpy()
    torch = _torch()
    torch.set_num_threads(8)
    settings = config["training"]
    x = np.asarray(arrays["features"], dtype=np.float64)
    exact_q, _, _ = _arm_targets(arrays, "factorized_execution")
    q_nom = np.asarray(nominal_q, dtype=np.float64)
    if q_nom.shape != exact_q.shape or not np.array_equal(q_nom[:, 0], exact_q[:, 0]):
        raise ValueError("ExecGrad nominal rollout initial condition differs")
    mean = np.asarray(normalization["feature_mean"], dtype=np.float64)
    std = np.asarray(normalization["feature_std"], dtype=np.float64)
    if mean.shape != (x.shape[1],) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("ExecGrad feature normalization differs")
    normalized = (x - mean) / std
    split = np.asarray(arrays["split"], dtype=object)
    train_rows = np.flatnonzero(split == "train")
    validation_rows = np.flatnonzero(split == "validation")
    state_index = np.asarray(arrays["state_index"], dtype=np.int64)
    clearance = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    safety_jacobian = np.asarray(
        local_geometry["jacobian_m_per_rad"], dtype=np.float64
    )
    center_jac = np.asarray(center_jacobian, dtype=np.float64)
    exact_link = np.asarray(exact_center_jvp, dtype=np.float64)
    pair_state = np.asarray(sensitivities["state_index"], dtype=np.int64)
    dimensions = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    negative_rows = np.asarray(sensitivities["negative_row_index"], dtype=np.int64)
    positive_rows = np.asarray(sensitivities["positive_row_index"], dtype=np.int64)
    actions = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    denominators = actions[positive_rows, dimensions] - actions[negative_rows, dimensions]
    split_by_state = {
        int(state): str(split[row]) for row, state in enumerate(state_index)
    }
    pair_split = np.asarray(
        [split_by_state[int(state)] for state in pair_state], dtype=object
    )
    train_pairs = np.flatnonzero(pair_split == "train")
    validation_pairs = np.flatnonzero(pair_split == "validation")
    scale = link_jvp_scales(exact_link, dimensions, train_pairs, settings)
    near = np.where(
        np.abs(clearance) <= float(settings["near_boundary_absolute_margin_m"]),
        float(settings["near_boundary_weight_multiplier"]),
        1.0,
    )
    outputs: dict[str, Any] = {}
    states: dict[str, Any] = {}
    audits: dict[str, Any] = {
        "JVP_scale": {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in scale.items()
        }
    }
    for arm in settings["matched_arms"]:
        seed = int(settings["seed"])
        torch.manual_seed(seed)
        model = _build_model(x.shape[1], config["architecture"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        generator = np.random.default_rng(seed)
        best_score = math.inf
        best_epoch = -1
        best = None
        stale = 0
        history = []
        for epoch in range(int(settings["epochs"])):
            model.train()
            order = generator.permutation(train_rows)
            for start in range(0, len(order), int(settings["batch_size"])):
                indexes = order[start : start + int(settings["batch_size"])]
                prediction = model(
                    torch.as_tensor(normalized[indexes], dtype=torch.float32),
                    torch.as_tensor(q_nom[indexes], dtype=torch.float32),
                )
                truth = torch.as_tensor(exact_q[indexes], dtype=torch.float32)
                q_error = prediction - truth
                q_loss = _huber(
                    q_error[:, 1:], float(settings["joint_huber_delta_rad"])
                ).mean()
                h_jac = torch.as_tensor(
                    safety_jacobian[state_index[indexes]], dtype=torch.float32
                )
                h_error = torch.einsum("bkrj,bkj->bkr", h_jac, q_error)
                weights = torch.as_tensor(near[indexes], dtype=torch.float32)
                geometry_loss = torch.mean(
                    weights
                    * _huber(h_error, float(settings["geometry_huber_delta_m"]))
                )
                loss = (
                    float(settings["joint_trajectory_loss_weight"]) * q_loss
                    + float(settings["symmetric_safety_normal_loss_weight"])
                    * geometry_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # One full paired pass per epoch preserves the exact central-pair
            # population and makes the matched-arm attribution transparent.
            model.train()
            pair_indexes = train_pairs
            negative = model(
                torch.as_tensor(normalized[negative_rows[pair_indexes]], dtype=torch.float32),
                torch.as_tensor(q_nom[negative_rows[pair_indexes]], dtype=torch.float32),
            )
            positive = model(
                torch.as_tensor(normalized[positive_rows[pair_indexes]], dtype=torch.float32),
                torch.as_tensor(q_nom[positive_rows[pair_indexes]], dtype=torch.float32),
            )
            q_secant = (positive - negative) / torch.as_tensor(
                denominators[pair_indexes], dtype=torch.float32
            )[:, None, None]
            p_jac = torch.as_tensor(
                center_jac[pair_state[pair_indexes]], dtype=torch.float32
            )
            predicted_link = torch.einsum("skpcj,skj->skpc", p_jac, q_secant)
            exact_link_tensor = torch.as_tensor(
                exact_link[pair_indexes], dtype=torch.float32
            )
            jvp_loss = link_jvp_loss(
                predicted_link,
                exact_link_tensor,
                dimensions[pair_indexes],
                scale,
            )
            if arm == "trajectory_plus_link_JVP":
                optimizer.zero_grad()
                (
                    float(settings["execgrad_link_center_JVP_loss_weight"])
                    * jvp_loss
                ).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_prediction = model(
                    torch.as_tensor(normalized[validation_rows], dtype=torch.float32),
                    torch.as_tensor(q_nom[validation_rows], dtype=torch.float32),
                ).cpu().numpy()
                val_joint = float(
                    np.sqrt(np.mean((val_prediction[:, 1:] - exact_q[validation_rows, 1:]) ** 2))
                )
                val_negative = model(
                    torch.as_tensor(normalized[negative_rows[validation_pairs]], dtype=torch.float32),
                    torch.as_tensor(q_nom[negative_rows[validation_pairs]], dtype=torch.float32),
                )
                val_positive = model(
                    torch.as_tensor(normalized[positive_rows[validation_pairs]], dtype=torch.float32),
                    torch.as_tensor(q_nom[positive_rows[validation_pairs]], dtype=torch.float32),
                )
                val_q_secant = (val_positive - val_negative) / torch.as_tensor(
                    denominators[validation_pairs], dtype=torch.float32
                )[:, None, None]
                val_jac = torch.as_tensor(
                    center_jac[pair_state[validation_pairs]], dtype=torch.float32
                )
                val_link = torch.einsum("skpcj,skj->skpc", val_jac, val_q_secant).cpu().numpy()
                val_jvp = _normalized_jvp_rmse(
                    val_link,
                    exact_link[validation_pairs],
                    dimensions[validation_pairs],
                    scale,
                )
                score = val_joint + 0.1 * val_jvp
            history.append(
                {
                    "epoch": int(epoch),
                    "validation_joint_RMSE_rad": val_joint,
                    "validation_normalized_link_JVP_RMSE": val_jvp,
                    "checkpoint_score": score,
                }
            )
            if score < best_score - 1.0e-12:
                best_score = score
                best_epoch = int(epoch)
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
            raise RuntimeError("ExecGrad training produced no checkpoint")
        model.load_state_dict(best)
        model.eval()
        outputs[arm] = model
        states[arm] = {
            key: value.detach().cpu().numpy() for key, value in best.items()
        }
        audits[arm] = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "completed_epoch_count": int(epoch + 1),
            "final_history": history[-5:],
        }
    state = {
        "feature_mean": mean,
        "feature_std": std,
        "input_dimension": int(x.shape[1]),
        "architecture": dict(config["architecture"]),
        "arms": list(settings["matched_arms"]),
        "model_states": states,
    }
    return outputs, state, audits


def predict_execgrad_models(
    models: Mapping[str, Any],
    state: Mapping[str, Any],
    arrays: Mapping[str, Any],
    nominal_q: Any,
) -> dict[str, Any]:
    np = _numpy()
    torch = _torch()
    x = np.asarray(arrays["features"], dtype=np.float64)
    normalized = (x - state["feature_mean"]) / state["feature_std"]
    nominal = np.asarray(nominal_q, dtype=np.float64)
    output = {}
    with torch.no_grad():
        for name, model in models.items():
            parts = []
            for start in range(0, len(x), 1024):
                parts.append(
                    model(
                        torch.as_tensor(normalized[start : start + 1024], dtype=torch.float32),
                        torch.as_tensor(nominal[start : start + 1024], dtype=torch.float32),
                    ).cpu().numpy()
                )
            output[name] = np.concatenate(parts, axis=0).astype(np.float64)
    return output


def save_execgrad_weights(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    arrays: dict[str, Any] = {
        "schema_version": np.asarray(WEIGHTS_SCHEMA),
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float64),
        "feature_std": np.asarray(state["feature_std"], dtype=np.float64),
        "input_dimension": np.asarray(state["input_dimension"], dtype=np.int64),
        "architecture_json": np.asarray(json.dumps(state["architecture"], sort_keys=True)),
        "arms_json": np.asarray(json.dumps(state["arms"])),
    }
    for arm, values in state["model_states"].items():
        for key, value in values.items():
            arrays["%s__%s" % (arm, key)] = np.asarray(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"path": str(Path(path).resolve()), "file_sha256": _sha256(Path(path).read_bytes())}


def soft_min_trace(trace: Any, temperature_m: float) -> Any:
    np = _numpy()
    values = np.asarray(trace, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (51, 7):
        raise ValueError("ExecGrad clearance trace shape differs")
    scaled = -values.reshape(len(values), -1) / float(temperature_m)
    maximum = np.max(scaled, axis=1, keepdims=True)
    return -float(temperature_m) * (
        maximum[:, 0] + np.log(np.sum(np.exp(scaled - maximum), axis=1))
    )


def action_gradient_from_pairs(
    score: Any,
    sensitivities: Mapping[str, Any],
    arrays: Mapping[str, Any],
    steering_dimensions: Sequence[int],
) -> tuple[Any, Any]:
    """Assemble one normalized-action score gradient per saved state."""

    np = _numpy()
    values = np.asarray(score, dtype=np.float64)
    negative = np.asarray(sensitivities["negative_row_index"], dtype=np.int64)
    positive = np.asarray(sensitivities["positive_row_index"], dtype=np.int64)
    dimensions = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    states = np.asarray(sensitivities["state_index"], dtype=np.int64)
    actions = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    selected = list(int(value) for value in steering_dimensions)
    column = {dimension: index for index, dimension in enumerate(selected)}
    unique_states = np.asarray(sorted(set(states.tolist())), dtype=np.int64)
    row_by_state = {int(state): index for index, state in enumerate(unique_states)}
    gradient = np.full((len(unique_states), len(selected)), np.nan, dtype=np.float64)
    for pair, (state, dimension) in enumerate(zip(states, dimensions)):
        if int(dimension) not in column:
            continue
        denominator = actions[positive[pair], dimension] - actions[negative[pair], dimension]
        gradient[row_by_state[int(state)], column[int(dimension)]] = (
            values[positive[pair]] - values[negative[pair]]
        ) / denominator
    if not np.all(np.isfinite(gradient)):
        raise ValueError("ExecGrad action gradient is incomplete")
    return unique_states, gradient


def direction_metrics(
    *,
    exact_gradient: Any,
    predicted_gradient: Any,
    state_ids: Any,
    state_splits: Mapping[int, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate actual first-order clearance gain and matched random rank."""

    np = _numpy()
    exact = np.asarray(exact_gradient, dtype=np.float64)
    predicted = np.asarray(predicted_gradient, dtype=np.float64)
    states = np.asarray(state_ids, dtype=np.int64)
    threshold = float(
        config["direction_score"]["zero_exact_gradient_norm_threshold_m_per_action"]
    )
    random_count = int(
        config["direction_score"]["matched_random_direction_count_per_state"]
    )
    generator = np.random.default_rng(int(config["direction_score"]["random_seed"]))
    output = {}
    for split_name in ("train", "validation", "test"):
        selected = np.asarray(
            [state_splits[int(state)] == split_name for state in states], dtype=bool
        )
        truth = exact[selected]
        estimate = predicted[selected]
        split_states = states[selected]
        exact_norm = np.linalg.norm(truth, axis=1)
        predicted_norm = np.linalg.norm(estimate, axis=1)
        valid = (exact_norm > threshold) & (predicted_norm > 1.0e-12)
        cosine = np.sum(truth[valid] * estimate[valid], axis=1) / (
            exact_norm[valid] * predicted_norm[valid]
        )
        unit = estimate[valid] / predicted_norm[valid, None]
        gain = np.sum(truth[valid] * unit, axis=1)
        norm_ratio = predicted_norm[valid] / exact_norm[valid]
        ranks = []
        random_means = []
        for exact_row, learned_gain in zip(truth[valid], gain):
            random = generator.normal(size=(random_count, exact.shape[1]))
            random /= np.linalg.norm(random, axis=1, keepdims=True)
            random_gain = random @ exact_row
            ranks.append((1 + int(np.count_nonzero(random_gain >= learned_gain))) / (random_count + 1))
            random_means.append(float(np.mean(random_gain)))
        output[split_name] = {
            "state_count": int(np.count_nonzero(selected)),
            "valid_nonzero_gradient_state_count": int(np.count_nonzero(valid)),
            "state_ids": split_states.tolist(),
            "mean_cosine": None if not len(cosine) else float(np.mean(cosine)),
            "median_cosine": None if not len(cosine) else float(np.median(cosine)),
            "wrong_direction_count": int(np.count_nonzero(gain <= 0.0)),
            "wrong_direction_rate": None if not len(gain) else float(np.mean(gain <= 0.0)),
            "median_exact_directional_gain_m_per_unit_action": None if not len(gain) else float(np.median(gain)),
            "mean_exact_directional_gain_m_per_unit_action": None if not len(gain) else float(np.mean(gain)),
            "median_predicted_to_exact_gradient_norm_ratio": None if not len(norm_ratio) else float(np.median(norm_ratio)),
            "mean_random_rank_add_one_p": None if not ranks else float(np.mean(ranks)),
            "state_random_rank_add_one_p": ranks,
            "state_fraction_p_le_0_05": None if not ranks else float(np.mean(np.asarray(ranks) <= 0.05)),
        }
    return output


def execgrad_fitted_decision(
    *,
    trajectory_only: Mapping[str, Any],
    execgrad: Mapping[str, Any],
    temporal: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["fitted_direction_gate"]
    validation = execgrad["validation"]
    baseline = trajectory_only["validation"]
    tests = {
        "validation_direction_cosine": bool(
            validation["mean_cosine"] is not None
            and float(validation["mean_cosine"]) >= float(gate["minimum_validation_mean_cosine"])
        ),
        "validation_wrong_direction_rate": bool(
            validation["wrong_direction_rate"] is not None
            and float(validation["wrong_direction_rate"]) <= float(gate["maximum_validation_wrong_direction_rate"])
        ),
        "validation_random_rank": bool(
            validation["mean_random_rank_add_one_p"] is not None
            and float(validation["mean_random_rank_add_one_p"]) <= float(gate["maximum_validation_mean_random_rank_p"])
        ),
        "execgrad_gain_beats_trajectory_only": bool(
            validation["median_exact_directional_gain_m_per_unit_action"] is not None
            and baseline["median_exact_directional_gain_m_per_unit_action"] is not None
            and float(validation["median_exact_directional_gain_m_per_unit_action"])
            >= float(baseline["median_exact_directional_gain_m_per_unit_action"])
            + float(gate["minimum_median_gain_improvement_over_trajectory_only_m_per_unit_action"])
        ),
        "no_material_joint_regression": bool(
            float(temporal["trajectory_plus_link_JVP"]["validation"]["overall_joint_RMSE_rad"])
            <= float(temporal["trajectory_only"]["validation"]["overall_joint_RMSE_rad"])
            * float(gate["maximum_validation_joint_RMSE_ratio_to_trajectory_only"])
        ),
    }
    passed = bool(all(tests.values()))
    return {
        "classification": "GO_direction_mechanism" if passed else "NO_GO_direction_mechanism",
        "tests": tests,
        "fitted_direction_gate_pass": passed,
        "absolute_false_safe_metrics_are_diagnostic_not_gate": True,
        "authorized_next_action": (
            "open_new_episode_groups_and_run_fresh_exact_OSC_learned_kinematic_random_direction_comparison"
            if passed
            else "stop_before_new_episode_flow_calibration_QP_and_closed_loop"
        ),
    }


def split_temporal_metrics(
    predicted: Any, exact: Any, split: Any
) -> dict[str, Any]:
    np = _numpy()
    labels = np.asarray(split, dtype=object)
    return {
        name: temporal_error_metrics(
            np.asarray(predicted)[labels == name], np.asarray(exact)[labels == name]
        )
        for name in ("train", "validation", "test")
    }
