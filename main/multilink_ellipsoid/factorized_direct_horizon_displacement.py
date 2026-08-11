"""Direct-horizon joint-displacement model for black-box OSC execution."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .complete_osc_margin import _canonical, _numpy, _torch
from .factorized_execution_pilot import _arm_targets
from .factorized_input_representation_audit import model_feature_names, semantic_group
from .factorized_structured_orientation import _root_rotation_indexes


CONFIG_SCHEMA = "vlsa_distal_factorized_direct_horizon_displacement_config.v1"
CONFIG_SCHEMA_V2 = "vlsa_distal_factorized_direct_horizon_displacement_config.v2"
COLLECTION_SCHEMA = "vlsa_distal_factorized_direct_horizon_reserved_collection.v1"
DATASET_SCHEMA = "vlsa_distal_factorized_direct_horizon_reserved_dataset.v1"
COLLECTION_VALIDATION_SCHEMA = (
    "vlsa_distal_factorized_direct_horizon_reserved_collection_validation.v1"
)
RESULT_SCHEMA = "vlsa_distal_factorized_direct_horizon_displacement_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_direct_horizon_displacement_validation.v1"
WEIGHTS_SCHEMA = "vlsa_distal_factorized_direct_horizon_displacement_weights.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_direct_horizon_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "reserved_state_selection", "candidate_design",
        "architecture", "training", "prediction_gate", "residual_bound_gate",
        "future_untouched_policy", "forbidden_before_prediction_pass",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("direct-horizon config keys differ")
    protocol = str(config["protocol_id"])
    if (config["schema_version"], protocol) not in {
        (CONFIG_SCHEMA,
         "vlsa-distal-factorized-direct-horizon-displacement-moka10-v1"),
        (CONFIG_SCHEMA_V2,
         "vlsa-distal-factorized-direct-horizon-normalized-secant-moka10-v1"),
    }:
        raise ValueError("direct-horizon protocol differs")
    architecture = config["architecture"]
    if (
        architecture["state_action_encoder_widths"] != [256, 256]
        or int(architecture["time_decoder_width"]) != 128
        or int(architecture["trace_state_count"]) != 51
        or int(architecture["joint_dimension"]) != 7
        or int(architecture["time_encoding_dimension"]) != 9
        or architecture["output"]
        != "direct_delta_q_k_from_q0_for_each_k_without_recursive_integration"
        or architecture["exact_initial_condition"]
        != "delta_q_0_is_architectural_zero"
    ):
        raise ValueError("direct-horizon architecture differs")
    training = config["training"]
    common_training_valid = (
        training["ensemble_seeds"]
        == [20260861, 20260862, 20260863, 20260864, 20260865]
        and int(training["batch_size"]) == 512
        and int(training["epochs"]) == 300
        and int(training["patience"]) == 40
        and float(training["joint_huber_delta_rad"]) == 0.002
        and float(training["joint_displacement_loss_weight"]) == 1.0
        and float(training["finite_difference_sensitivity_loss_weight"]) == 1.0
        and float(training["symmetric_safety_normal_loss_weight"]) == 10.0
        and float(training["geometry_huber_delta_m"]) == 0.002
        and float(training["near_boundary_absolute_margin_m"]) == 0.005
        and float(training["near_boundary_weight_multiplier"]) == 5.0
        and training["dangerous_overestimate_extra_weight"] == 0.0
    )
    if not common_training_valid:
        raise ValueError("direct-horizon training differs")
    if protocol.endswith("displacement-moka10-v1"):
        if set(training) != {
            "device", "ensemble_seeds", "batch_size", "epochs", "patience",
            "learning_rate", "weight_decay", "joint_huber_delta_rad",
            "joint_displacement_loss_weight",
            "finite_difference_sensitivity_loss_weight",
            "symmetric_safety_normal_loss_weight", "geometry_huber_delta_m",
            "near_boundary_absolute_margin_m", "near_boundary_weight_multiplier",
            "dangerous_overestimate_extra_weight", "training_substeps",
            "early_stopping",
        }:
            raise ValueError("direct-horizon v1 training keys differ")
    else:
        normalized = training.get("normalized_paired_secant")
        if (
            set(training) != {
                "device", "ensemble_seeds", "batch_size", "epochs",
                "patience", "learning_rate", "weight_decay",
                "joint_huber_delta_rad", "joint_displacement_loss_weight",
                "finite_difference_sensitivity_loss_weight",
                "sensitivity_loss", "normalized_paired_secant",
                "symmetric_safety_normal_loss_weight",
                "geometry_huber_delta_m", "near_boundary_absolute_margin_m",
                "near_boundary_weight_multiplier",
                "dangerous_overestimate_extra_weight", "training_substeps",
                "early_stopping",
            }
            or float(training["learning_rate"]) != 0.001
            or float(training["weight_decay"]) != 1.0e-6
            or training.get("sensitivity_loss")
            != "training_RMS_normalized_paired_secant_vector_MSE"
            or not isinstance(normalized, dict)
            or normalized != {
                "scale": "training_only_RMS_L2_joint_sensitivity_by_action_coordinate_and_horizon",
                "physically_zero_RMS_norm_threshold_rad_per_action": 1e-05,
                "denominator_epsilon_squared": 1e-12,
                "minimum_relative_weight": 0.001,
                "maximum_relative_weight": 1000.0,
                "exclude_zero_or_noisy_cells": True,
            }
            or training["training_substeps"] != "all_1_through_50"
            or training["early_stopping"]
            != "validation_joint_RMSE_plus_0.1_sensitivity_RMSE_plus_symmetric_safety_normal_RMSE"
        ):
            raise ValueError("normalized paired-secant training differs")
    forbidden = config["forbidden_before_prediction_pass"]
    if set(forbidden) != {
        "residual_bound_calibration", "QP", "closed_loop", "poisson_or_SDF",
        "binary_classifier",
    } or not all(bool(value) for value in forbidden.values()):
        raise ValueError("direct-horizon forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def horizon_time_encoding() -> Any:
    """Return fixed time features for trace states zero through fifty."""

    np = _numpy()
    tau = np.arange(51, dtype=np.float64) / 50.0
    parts = [tau]
    for frequency in (1.0, 2.0, 4.0, 8.0):
        parts.extend((
            np.sin(math.pi * frequency * tau),
            np.cos(math.pi * frequency * tau),
        ))
    output = np.stack(parts, axis=1)
    if output.shape != (51, 9):
        raise ValueError("direct-horizon time encoding differs")
    return output


def _build_model(input_dimension: int, architecture: Mapping[str, Any]) -> Any:
    torch = _torch()
    nn = torch.nn
    widths = [int(value) for value in architecture["state_action_encoder_widths"]]
    decoder = int(architecture["time_decoder_width"])
    encoding = torch.as_tensor(horizon_time_encoding(), dtype=torch.float32)

    class DirectHorizonDisplacementNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(int(input_dimension), widths[0]), nn.SiLU(),
                nn.Linear(widths[0], widths[1]), nn.SiLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(widths[1] + 9, decoder), nn.SiLU(),
                nn.Linear(decoder, 7),
            )
            self.register_buffer("horizon_time_encoding", encoding.clone())

        def forward(self, features: Any) -> Any:
            latent = self.encoder(features)
            expanded = latent[:, None, :].expand(-1, 51, -1)
            time = self.horizon_time_encoding[None, :, :].expand(
                len(features), -1, -1
            )
            displacement = self.decoder(torch.cat((expanded, time), dim=2))
            # There is no recurrent sum and no prediction at k=0.
            zero = torch.zeros(
                (len(features), 1, 7), dtype=displacement.dtype,
                device=displacement.device,
            )
            return torch.cat((zero, displacement[:, 1:]), dim=1)

    return DirectHorizonDisplacementNet()


def _huber(error: Any, beta: float) -> Any:
    torch = _torch()
    absolute = torch.abs(error)
    return torch.where(
        absolute <= beta, 0.5 * error ** 2,
        beta * (absolute - 0.5 * beta),
    )


def normalized_paired_secant_scales(
    exact_sensitivity: Any, dimensions: Any, training_rows: Any,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit fixed action-coordinate/horizon scales from training rows only."""

    np = _numpy()
    exact = np.asarray(exact_sensitivity, dtype=np.float64)
    dimension = np.asarray(dimensions, dtype=np.int64)
    rows = np.asarray(training_rows, dtype=np.int64)
    if exact.ndim != 3 or exact.shape[1:] != (51, 7):
        raise ValueError("paired-secant sensitivity shape differs")
    if dimension.shape != (len(exact),) or np.any((dimension < 0) | (dimension >= 14)):
        raise ValueError("paired-secant action dimensions differ")
    selected = np.zeros(len(exact), dtype=bool)
    selected[rows] = True
    scale = np.zeros((14, 51), dtype=np.float64)
    sample_count = np.zeros((14, 51), dtype=np.int64)
    for action_dimension in range(14):
        use = selected & (dimension == action_dimension)
        if np.any(use):
            squared_norm = np.sum(exact[use] ** 2, axis=2)
            scale[action_dimension] = np.sqrt(np.mean(squared_norm, axis=0))
            sample_count[action_dimension] = int(np.count_nonzero(use))
    specification = settings["normalized_paired_secant"]
    threshold = float(
        specification["physically_zero_RMS_norm_threshold_rad_per_action"]
    )
    valid = scale >= threshold
    epsilon = float(specification["denominator_epsilon_squared"])
    raw_weight = np.zeros_like(scale)
    raw_weight[valid] = 1.0 / (scale[valid] ** 2 + epsilon)
    if not np.any(valid):
        raise ValueError("paired-secant training has no nonzero cells")
    median_weight = float(np.median(raw_weight[valid]))
    lower = median_weight * float(specification["minimum_relative_weight"])
    upper = median_weight * float(specification["maximum_relative_weight"])
    weight = np.zeros_like(scale)
    weight[valid] = np.clip(raw_weight[valid], lower, upper)
    return {
        "RMS_scale_rad_per_action": scale,
        "sample_count": sample_count,
        "valid": valid,
        "weight": weight,
        "raw_weight_median": median_weight,
        "weight_clip_lower": lower,
        "weight_clip_upper": upper,
        "valid_cell_count": int(np.count_nonzero(valid)),
        "excluded_cell_count": int(valid.size - np.count_nonzero(valid)),
        "clipped_low_cell_count": int(np.count_nonzero(
            valid & (raw_weight < lower)
        )),
        "clipped_high_cell_count": int(np.count_nonzero(
            valid & (raw_weight > upper)
        )),
    }


def normalized_paired_secant_loss(
    predicted: Any, exact: Any, dimensions: Any, scale: Mapping[str, Any],
) -> Any:
    """Return one vector-MSE loss normalized by fixed training-only scales."""

    torch = _torch()
    dimension = torch.as_tensor(dimensions, dtype=torch.long)
    weight = torch.as_tensor(scale["weight"], dtype=predicted.dtype)[dimension]
    valid = torch.as_tensor(scale["valid"], dtype=torch.bool)[dimension]
    squared_vector_error = torch.sum((predicted - exact) ** 2, dim=2)
    denominator = torch.sum(valid)
    if int(denominator.detach().cpu()) <= 0:
        raise ValueError("paired-secant batch has no valid cells")
    return torch.sum(squared_vector_error * weight * valid) / denominator


def apply_structured_representation(
    complete_dataset: Mapping[str, Any], arrays: Mapping[str, Any],
    structured_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the frozen bounded-orientation representation without refitting."""

    np = _numpy()
    names = model_feature_names(complete_dataset)
    x = np.asarray(arrays["features"], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError("direct-horizon unstructured feature dimension differs")
    removed_groups = set(
        structured_config["structured_representation"]["removed_groups"]
    )
    groups = [semantic_group(name) for name in names]
    removed = np.asarray([
        index for index, group in enumerate(groups) if group in removed_groups
    ], dtype=np.int64)
    kept = np.asarray([
        index for index, group in enumerate(groups) if group not in removed_groups
    ], dtype=np.int64)
    root_indexes = _root_rotation_indexes(names)
    root_6d = x[:, root_indexes]
    structured = np.concatenate((x[:, kept], root_6d), axis=1)
    expected = int(structured_config["structured_representation"][
        "structured_input_dimension"
    ])
    if (
        len(removed) != int(structured_config["structured_representation"][
            "removed_scalar_count"
        ])
        or structured.shape != (len(x), expected)
        or np.max(np.abs(root_6d)) > 1.0 + 1.0e-9
    ):
        raise ValueError("direct-horizon structured representation differs")
    output = dict(arrays)
    output["features"] = structured
    return output, {
        "removed_feature_count": int(len(removed)),
        "retained_feature_count": int(len(kept)),
        "structured_input_dimension": int(structured.shape[1]),
        "removed_indexes_sha256": _sha256(removed.tobytes()),
        "retained_indexes_sha256": _sha256(kept.tobytes()),
        "structured_features_sha256": _sha256(structured.tobytes()),
    }


def train_direct_horizon_ensemble(
    arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
    geometry: Mapping[str, Any], config: Mapping[str, Any],
    normalization: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Train direct displacements with symmetric safety-normal supervision."""

    np = _numpy()
    torch = _torch()
    torch.set_num_threads(8)
    x = np.asarray(arrays["features"], dtype=np.float64)
    target, base, _ = _arm_targets(arrays, "factorized_execution")
    displacement = target - base
    split = np.asarray(arrays["split"], dtype=object)
    train_indexes = np.flatnonzero(split == "train")
    validation_indexes = np.flatnonzero(split == "validation")
    mean = np.asarray(normalization["feature_mean"], dtype=np.float64)
    std = np.asarray(normalization["feature_std"], dtype=np.float64)
    if mean.shape != (x.shape[1],) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("direct-horizon normalization differs")
    normalized = (x - mean) / std
    state_index = np.asarray(arrays["state_index"], dtype=np.int64)
    clearance = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    jacobians = np.asarray(geometry["jacobian_m_per_rad"], dtype=np.float64)
    exact_sensitivity = np.asarray(
        sensitivities["joint_sensitivity_rad_per_action"], dtype=np.float64,
    )
    sensitivity_states = np.asarray(sensitivities["state_index"], dtype=np.int64)
    state_split = {
        int(state): str(split[row]) for row, state in enumerate(state_index)
    }
    sensitivity_split = np.asarray([
        state_split[int(state)] for state in sensitivity_states
    ], dtype=object)
    train_sensitivity = np.flatnonzero(sensitivity_split == "train")
    validation_sensitivity = np.flatnonzero(sensitivity_split == "validation")
    negative_rows = np.asarray(
        sensitivities["negative_row_index"], dtype=np.int64
    )
    positive_rows = np.asarray(
        sensitivities["positive_row_index"], dtype=np.int64
    )
    dimensions = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    actions = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    denominators = (
        actions[positive_rows, dimensions] - actions[negative_rows, dimensions]
    )
    settings = config["training"]
    sensitivity_mode = str(settings.get("sensitivity_loss", "raw_secant_huber"))
    normalized_scale = (
        normalized_paired_secant_scales(
            exact_sensitivity, dimensions, train_sensitivity, settings,
        )
        if sensitivity_mode
        == "training_RMS_normalized_paired_secant_vector_MSE"
        else None
    )
    joint_beta = float(settings["joint_huber_delta_rad"])
    geometry_beta = float(settings["geometry_huber_delta_m"])
    boundary = float(settings["near_boundary_absolute_margin_m"])
    element_weight = np.where(
        np.abs(clearance) <= boundary,
        float(settings["near_boundary_weight_multiplier"]), 1.0,
    )
    models = []
    model_states = []
    audits = []
    for seed in settings["ensemble_seeds"]:
        torch.manual_seed(int(seed))
        model = _build_model(x.shape[1], config["architecture"])
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
            for start in range(0, len(order), int(settings["batch_size"])):
                indexes = order[start:start + int(settings["batch_size"])]
                prediction = model(torch.as_tensor(
                    normalized[indexes], dtype=torch.float32,
                ))
                truth = torch.as_tensor(
                    displacement[indexes], dtype=torch.float32,
                )
                q_error = prediction - truth
                q_loss = _huber(q_error[:, 1:], joint_beta).mean()
                jacobian = torch.as_tensor(
                    jacobians[state_index[indexes]], dtype=torch.float32,
                )
                normal_error = torch.einsum(
                    "bkrj,bkj->bkr", jacobian, q_error
                )
                weights = torch.as_tensor(
                    element_weight[indexes], dtype=torch.float32,
                )
                # Deliberately symmetric: optimistic and conservative signed
                # normal errors receive identical penalties.
                geometry_loss = torch.mean(
                    weights * _huber(normal_error, geometry_beta)
                )
                loss = (
                    float(settings["joint_displacement_loss_weight"]) * q_loss
                    + float(settings["symmetric_safety_normal_loss_weight"])
                    * geometry_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            selected = train_sensitivity
            model.train()
            negative = model(torch.as_tensor(
                normalized[negative_rows[selected]], dtype=torch.float32,
            ))
            positive = model(torch.as_tensor(
                normalized[positive_rows[selected]], dtype=torch.float32,
            ))
            predicted_sensitivity = (positive - negative) / torch.as_tensor(
                denominators[selected], dtype=torch.float32,
            )[:, None, None]
            exact = torch.as_tensor(
                exact_sensitivity[selected], dtype=torch.float32,
            )
            if normalized_scale is None:
                sensitivity_loss = torch.nn.functional.smooth_l1_loss(
                    predicted_sensitivity[:, 1:], exact[:, 1:], beta=joint_beta,
                )
            else:
                sensitivity_loss = normalized_paired_secant_loss(
                    predicted_sensitivity, exact, dimensions[selected],
                    normalized_scale,
                )
            optimizer.zero_grad()
            (float(settings["finite_difference_sensitivity_loss_weight"])
             * sensitivity_loss).backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_prediction = model(torch.as_tensor(
                    normalized[validation_indexes], dtype=torch.float32,
                )).cpu().numpy()
                selected = validation_sensitivity
                negative = model(torch.as_tensor(
                    normalized[negative_rows[selected]], dtype=torch.float32,
                ))
                positive = model(torch.as_tensor(
                    normalized[positive_rows[selected]], dtype=torch.float32,
                ))
                validation_sensitivity_prediction = (
                    (positive - negative).cpu().numpy()
                    / denominators[selected, None, None]
                )
            q_rmse = float(np.sqrt(np.mean(
                (validation_prediction[:, 1:]
                 - displacement[validation_indexes, 1:]) ** 2
            )))
            sensitivity_rmse = float(np.sqrt(np.mean(
                (validation_sensitivity_prediction[:, 1:]
                 - exact_sensitivity[selected, 1:]) ** 2
            )))
            q_error = validation_prediction - displacement[validation_indexes]
            normal_error = np.einsum(
                "bkrj,bkj->bkr",
                jacobians[state_index[validation_indexes]], q_error,
            )
            geometry_rmse = float(np.sqrt(np.mean(normal_error ** 2)))
            score = q_rmse + 0.1 * sensitivity_rmse + geometry_rmse
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
            raise RuntimeError("direct-horizon training produced no checkpoint")
        model.load_state_dict(best)
        model.eval()
        models.append(model)
        model_states.append({key: value.numpy() for key, value in best.items()})
        audits.append({
            "seed": int(seed), "best_epoch": int(best_epoch),
            "validation_score": float(best_score),
            "completed_epoch_count": int(epoch + 1),
        })
    state = {
        "feature_mean": mean, "feature_std": std,
        "input_dimension": int(x.shape[1]), "output_shape": [51, 7],
        "ensemble_seeds": list(settings["ensemble_seeds"]),
        "architecture": dict(config["architecture"]),
        "model_states": model_states,
    }
    training_audit = {
        "member_audits": audits,
        "training_row_count": int(len(train_indexes)),
        "validation_row_count": int(len(validation_indexes)),
        "training_sensitivity_row_count": int(len(train_sensitivity)),
        "validation_sensitivity_row_count": int(len(validation_sensitivity)),
    }
    if normalized_scale is not None:
        training_audit["normalized_paired_secant"] = {
            "scale_fit_split": "train_only",
            "RMS_scale_rad_per_action": normalized_scale[
                "RMS_scale_rad_per_action"
            ].tolist(),
            "sample_count": normalized_scale["sample_count"].tolist(),
            "valid": normalized_scale["valid"].tolist(),
            "weight": normalized_scale["weight"].tolist(),
            "raw_weight_median": float(normalized_scale["raw_weight_median"]),
            "weight_clip_lower": float(normalized_scale["weight_clip_lower"]),
            "weight_clip_upper": float(normalized_scale["weight_clip_upper"]),
            "valid_cell_count": int(normalized_scale["valid_cell_count"]),
            "excluded_cell_count": int(normalized_scale["excluded_cell_count"]),
            "clipped_low_cell_count": int(
                normalized_scale["clipped_low_cell_count"]
            ),
            "clipped_high_cell_count": int(
                normalized_scale["clipped_high_cell_count"]
            ),
        }
    return models, state, training_audit


def predict_direct_horizon(
    models: Sequence[Any], state: Mapping[str, Any], arrays: Mapping[str, Any],
) -> Any:
    np = _numpy()
    torch = _torch()
    normalized = (
        np.asarray(arrays["features"], dtype=np.float64) - state["feature_mean"]
    ) / state["feature_std"]
    members = []
    with torch.no_grad():
        for model in models:
            parts = []
            for start in range(0, len(normalized), 512):
                parts.append(model(torch.as_tensor(
                    normalized[start:start + 512], dtype=torch.float32,
                )).cpu().numpy())
            members.append(np.concatenate(parts, axis=0))
    displacement = np.mean(np.asarray(members), axis=0)
    _, base, _ = _arm_targets(arrays, "factorized_execution")
    return base + displacement


def save_direct_horizon_weights(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    archive: dict[str, Any] = {
        "schema_version": np.asarray(WEIGHTS_SCHEMA),
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float64),
        "feature_std": np.asarray(state["feature_std"], dtype=np.float64),
        "input_dimension": np.asarray(state["input_dimension"], dtype=np.int64),
        "output_shape": np.asarray([51, 7], dtype=np.int64),
        "ensemble_seeds": np.asarray(state["ensemble_seeds"], dtype=np.int64),
        "architecture_json": np.asarray(json.dumps(
            state["architecture"], sort_keys=True, separators=(",", ":")
        )),
    }
    for member, values in enumerate(state["model_states"]):
        for key, value in values.items():
            archive["member_%d__%s" % (member, key)] = np.asarray(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **archive)
    return {
        "path": str(Path(path).resolve()),
        "file_sha256": _sha256(Path(path).read_bytes()),
    }


def load_direct_horizon_weights(path: Path) -> tuple[list[Any], dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    archive = np.load(path, allow_pickle=False)
    if str(archive["schema_version"].item()) != WEIGHTS_SCHEMA:
        raise ValueError("direct-horizon weights schema differs")
    architecture = json.loads(str(archive["architecture_json"].item()))
    input_dimension = int(archive["input_dimension"].item())
    seeds = archive["ensemble_seeds"].astype(int).tolist()
    models = []
    states = []
    for member in range(len(seeds)):
        model = _build_model(input_dimension, architecture)
        prefix = "member_%d__" % member
        values = {
            key[len(prefix):]: torch.as_tensor(archive[key])
            for key in archive.files if key.startswith(prefix)
        }
        model.load_state_dict(values)
        model.eval()
        models.append(model)
        states.append({key: value.numpy() for key, value in values.items()})
    return models, {
        "feature_mean": archive["feature_mean"].astype(np.float64),
        "feature_std": archive["feature_std"].astype(np.float64),
        "input_dimension": input_dimension, "output_shape": [51, 7],
        "ensemble_seeds": seeds, "architecture": architecture,
        "model_states": states,
    }


def temporal_error_metrics(predicted_q: Any, exact_q: Any) -> dict[str, Any]:
    np = _numpy()
    error = np.asarray(predicted_q, dtype=np.float64) - np.asarray(
        exact_q, dtype=np.float64
    )
    if error.ndim != 3 or error.shape[1:] != (51, 7):
        raise ValueError("direct-horizon temporal metric shape differs")
    by_substep = np.sqrt(np.mean(error ** 2, axis=(0, 2)))
    indexes = np.arange(1, 51, dtype=np.float64)
    slope = float(np.polyfit(indexes, by_substep[1:], 1)[0])
    overall = float(np.sqrt(np.mean(error[:, 1:] ** 2)))
    terminal = float(by_substep[-1])
    return {
        "action_count": int(len(error)),
        "joint_RMSE_by_substep_rad": by_substep.tolist(),
        "overall_joint_RMSE_rad": overall,
        "terminal_joint_RMSE_rad": terminal,
        "linear_RMSE_slope_rad_per_substep": slope,
        "terminal_to_overall_RMSE_ratio": (
            None if overall <= 0.0 else float(terminal / overall)
        ),
        "maximum_initial_joint_error_rad": float(np.max(np.abs(error[:, 0]))),
    }


def sensitivity_magnitude_metrics(exact: Any, predicted: Any) -> dict[str, Any]:
    np = _numpy()
    truth = np.asarray(exact, dtype=np.float64).reshape(len(exact), -1)
    estimate = np.asarray(predicted, dtype=np.float64).reshape(len(predicted), -1)
    truth_norm = np.linalg.norm(truth, axis=1)
    estimate_norm = np.linalg.norm(estimate, axis=1)
    valid = truth_norm > 1.0e-12
    relative = np.abs(estimate_norm[valid] - truth_norm[valid]) / truth_norm[valid]
    ratio = estimate_norm[valid] / truth_norm[valid]
    return {
        "valid_row_count": int(np.count_nonzero(valid)),
        "norm_RMSE": float(np.sqrt(np.mean(
            (estimate_norm[valid] - truth_norm[valid]) ** 2
        ))),
        "mean_relative_norm_error": float(np.mean(relative)),
        "median_relative_norm_error": float(np.median(relative)),
        "median_predicted_to_exact_norm_ratio": float(np.median(ratio)),
    }


def normalized_secant_fitted_gate(
    *, sensitivity: Mapping[str, Any], validation_temporal: Mapping[str, Any],
    baseline_validation_joint_RMSE_rad: float, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate fitted-state action sensitivity before any unseen evaluation."""

    gate = config["prediction_gate"]

    def metric_tests(item: Mapping[str, Any]) -> dict[str, bool]:
        return {
            "direction": bool(
                item["mean_cosine"] is not None
                and float(item["mean_cosine"])
                >= float(gate["minimum_joint_sensitivity_cosine"])
            ),
            "magnitude_ratio": bool(
                item["median_norm_ratio"] is not None
                and float(gate["minimum_joint_sensitivity_median_norm_ratio"])
                <= float(item["median_norm_ratio"])
                <= float(gate["maximum_joint_sensitivity_median_norm_ratio"])
            ),
            "magnitude_error": bool(
                item["mean_relative_norm_error"] is not None
                and float(item["mean_relative_norm_error"])
                <= float(gate[
                    "maximum_joint_sensitivity_mean_relative_norm_error"
                ])
            ),
        }

    metrics = {}
    tests = {}
    for split in ("train", "validation"):
        aggregate = sensitivity[split]["all_horizon_trace"]["joint"]
        terminal = sensitivity[split]["all_horizon_trace"]["by_horizon"][50][
            "joint"
        ]
        metrics[split] = {"aggregate": aggregate, "terminal": terminal}
        for scope, item in (("aggregate", aggregate), ("terminal", terminal)):
            scoped = metric_tests(item)
            tests["%s_%s" % (split, scope)] = bool(all(scoped.values()))
            tests["%s_%s_components" % (split, scope)] = scoped
    maximum_validation = (
        float(baseline_validation_joint_RMSE_rad)
        * float(gate[
            "maximum_validation_joint_RMSE_relative_to_frozen_direct_model"
        ])
    )
    tests["validation_joint_trajectory"] = bool(
        float(validation_temporal["overall_joint_RMSE_rad"])
        <= maximum_validation
    )
    required = [
        tests["train_aggregate"], tests["validation_aggregate"],
        tests["train_terminal"], tests["validation_terminal"],
        tests["validation_joint_trajectory"],
    ]
    passed = bool(all(required))
    return {
        "metrics": metrics,
        "validation_joint_trajectory": {
            "observed_RMSE_rad": float(
                validation_temporal["overall_joint_RMSE_rad"]
            ),
            "frozen_baseline_RMSE_rad": float(
                baseline_validation_joint_RMSE_rad
            ),
            "maximum_RMSE_rad": maximum_validation,
        },
        "tests": tests,
        "fitted_sensitivity_gate_pass": passed,
        "authorized_next_action": (
            "evaluate_on_new_grouped_unseen_episodes" if passed
            else "change_time_decoder_without_adding_losses"
        ),
        "calibration_QP_closed_loop_authorized": False,
    }


def direct_horizon_decision(
    *, validation_metrics: Mapping[str, Any], reserved_metrics: Mapping[str, Any],
    reserved_exact_static_metrics: Mapping[str, Any],
    validation_support: Mapping[str, Any], reserved_support: Mapping[str, Any],
    reserved_temporal: Mapping[str, Any], joint_sensitivity: Mapping[str, Any],
    safety_sensitivity: Mapping[str, Any], matched_cumulative: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["prediction_gate"]
    tests = {
        "reserved_exact_q_static_zero_false_safe": int(
            reserved_exact_static_metrics["false_safe_action_count"]
        ) <= int(gate[
            "maximum_reserved_exact_q_static_false_safe_action_count"
        ]),
        "reserved_exact_q_static_safe_recall": float(
            reserved_exact_static_metrics["exact_safe_action_recall"]
        ) >= float(gate["minimum_reserved_exact_q_static_safe_recall"]),
        "zero_validation_false_safe": int(
            validation_metrics["false_safe_action_count"]
        ) <= int(gate["maximum_validation_false_safe_action_count"]),
        "zero_reserved_false_safe": int(
            reserved_metrics["false_safe_action_count"]
        ) <= int(gate["maximum_reserved_false_safe_action_count"]),
        "all_validation_recoverable_states_supported": bool(
            validation_support["all_eligible_states_supported"]
        ),
        "all_reserved_recoverable_states_supported": bool(
            reserved_support["all_eligible_states_supported"]
        ),
        "reserved_safe_recall": float(
            reserved_metrics["exact_safe_action_recall"]
        ) >= float(gate["minimum_reserved_safe_recall"]),
        "reserved_boundary_RMSE": float(
            reserved_metrics["near_boundary_RMSE_m"]
        ) <= float(gate["maximum_reserved_near_boundary_RMSE_m"]),
        "joint_sensitivity_cosine": float(
            joint_sensitivity["mean_cosine"]
        ) > float(gate["minimum_joint_sensitivity_cosine"]),
        "safety_sensitivity_cosine": float(
            safety_sensitivity["mean_cosine"]
        ) > float(gate["minimum_safety_sensitivity_cosine"]),
        "overall_joint_error": float(
            reserved_temporal["overall_joint_RMSE_rad"]
        ) <= float(gate["maximum_reserved_overall_joint_RMSE_rad"]),
        "terminal_joint_error": float(
            reserved_temporal["terminal_joint_RMSE_rad"]
        ) <= float(gate["maximum_reserved_terminal_joint_RMSE_rad"]),
        "horizon_slope": float(
            reserved_temporal["linear_RMSE_slope_rad_per_substep"]
        ) <= float(gate["maximum_RMSE_slope_rad_per_substep"]),
        "terminal_to_overall_ratio": float(
            reserved_temporal["terminal_to_overall_RMSE_ratio"]
        ) <= float(gate["maximum_terminal_to_overall_RMSE_ratio"]),
        "exact_initial_condition": float(
            reserved_temporal["maximum_initial_joint_error_rad"]
        ) <= 1.0e-12,
        "strictly_less_drift_than_cumulative": float(
            reserved_temporal["linear_RMSE_slope_rad_per_substep"]
        ) < float(matched_cumulative["linear_RMSE_slope_rad_per_substep"]),
    }
    passed = bool(all(tests.values()))
    return {
        "gate_tests": tests,
        "prediction_gate_GO": passed,
        "execution_model_frozen": passed,
        "residual_bound_gate_authorized": passed,
        "QP_or_closed_loop_authorized": False,
        "conclusion": (
            "direct_horizon_displacement_passes_reserved_prediction_gate"
            if passed else
            "direct_horizon_displacement_fails_reserved_prediction_gate"
        ),
    }
