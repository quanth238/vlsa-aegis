"""Explicit local-affine OSC execution intercept and Jacobian decoder."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .complete_osc_margin import _canonical, _numpy, _torch
from .factorized_direct_horizon_displacement import (
    _huber, horizon_time_encoding, normalized_paired_secant_loss,
    normalized_paired_secant_scales,
)
from .factorized_execution_pilot import _arm_targets


CONFIG_SCHEMA = "vlsa_distal_factorized_explicit_execution_jacobian_config.v1"
WEIGHTS_SCHEMA = "vlsa_distal_factorized_explicit_execution_jacobian_weights.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_explicit_jacobian_config(path: Path) -> Dict[str, Any]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "reserved_state_selection", "candidate_design",
        "architecture", "causal_contract", "training", "prediction_gate",
        "residual_bound_gate", "future_untouched_policy",
        "forbidden_before_prediction_pass",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("explicit-jacobian config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-explicit-execution-jacobian-moka10-v1"
    ):
        raise ValueError("explicit-jacobian protocol differs")
    architecture = config["architecture"]
    if architecture != {
        "input_representation": (
            "job_38095_structured_complete_OSC_state_plus_nominal_2_by_7_action"
        ),
        "state_action_encoder_widths": [256, 256],
        "time_decoder_width": 128,
        "activation": "silu",
        "trace_state_count": 51,
        "joint_dimension": 7,
        "action_dimension": 14,
        "time_encoding": (
            "tau_plus_sin_cos_pi_tau_frequencies_1_2_4_8_at_k_0_through_50"
        ),
        "time_encoding_dimension": 9,
        "action_phase_encoding": "one_hot_first_action_then_second_action",
        "action_phase_encoding_dimension": 2,
        "output": (
            "direct_nominal_delta_q_k_and_local_execution_J_k_without_"
            "recursive_integration"
        ),
        "candidate_prediction": (
            "q0_plus_nominal_delta_q_k_plus_J_k_times_candidate_minus_nominal"
        ),
        "exact_initial_condition": "delta_q_0_and_J_0_are_architectural_zero",
    }:
        raise ValueError("explicit-jacobian architecture differs")
    if config["causal_contract"] != {
        "controller_substeps_per_action": 25,
        "first_action_active_substeps": [1, 25],
        "second_action_starts_at_substep": 26,
        "second_action_J_columns_zero_through_substep": 25,
        "first_action_J_columns": [0, 6],
        "second_action_J_columns": [7, 13],
    }:
        raise ValueError("explicit-jacobian causal contract differs")
    training = config["training"]
    if training != {
        "device": (
            "cpu_inside_H100_allocation_due_to_pinned_PyTorch_missing_sm90_kernels"
        ),
        "ensemble_seeds": [20260861, 20260862, 20260863, 20260864, 20260865],
        "batch_size": 512,
        "epochs": 300,
        "patience": 40,
        "learning_rate": 0.001,
        "weight_decay": 1e-06,
        "joint_huber_delta_rad": 0.002,
        "joint_displacement_loss_weight": 1.0,
        "finite_difference_sensitivity_loss_weight": 1.0,
        "sensitivity_loss": "training_RMS_normalized_paired_secant_vector_MSE",
        "normalized_paired_secant": {
            "scale": (
                "training_only_RMS_L2_joint_sensitivity_by_action_coordinate_"
                "and_horizon"
            ),
            "physically_zero_RMS_norm_threshold_rad_per_action": 1e-05,
            "denominator_epsilon_squared": 1e-12,
            "minimum_relative_weight": 0.001,
            "maximum_relative_weight": 1000.0,
            "exclude_zero_or_noisy_cells": True,
        },
        "symmetric_safety_normal_loss_weight": 10.0,
        "geometry_huber_delta_m": 0.002,
        "near_boundary_absolute_margin_m": 0.005,
        "near_boundary_weight_multiplier": 5.0,
        "dangerous_overestimate_extra_weight": 0.0,
        "training_substeps": "all_1_through_50",
        "early_stopping": (
            "validation_joint_RMSE_plus_0.1_sensitivity_RMSE_plus_"
            "symmetric_safety_normal_RMSE"
        ),
    }:
        raise ValueError("explicit-jacobian training differs")
    forbidden = config["forbidden_before_prediction_pass"]
    if set(forbidden) != {
        "residual_bound_calibration", "QP", "closed_loop", "poisson_or_SDF",
        "binary_classifier", "new_unseen_episode_evaluation",
    } or not all(bool(value) for value in forbidden.values()):
        raise ValueError("explicit-jacobian forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def action_phase_encoding() -> Any:
    np = _numpy()
    phase = np.zeros((51, 2), dtype=np.float64)
    phase[1:26, 0] = 1.0
    phase[26:, 1] = 1.0
    return phase


def causal_execution_jacobian_mask() -> Any:
    np = _numpy()
    mask = np.ones((51, 7, 14), dtype=np.float64)
    mask[0] = 0.0
    mask[1:26, :, 7:14] = 0.0
    return mask


def _build_explicit_model(input_dimension: int, architecture: Mapping[str, Any]) -> Any:
    torch = _torch()
    nn = torch.nn
    widths = [int(value) for value in architecture["state_action_encoder_widths"]]
    decoder = int(architecture["time_decoder_width"])
    time = torch.as_tensor(horizon_time_encoding(), dtype=torch.float32)
    phase = torch.as_tensor(action_phase_encoding(), dtype=torch.float32)
    mask = torch.as_tensor(causal_execution_jacobian_mask(), dtype=torch.float32)

    class ExplicitExecutionJacobianNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(int(input_dimension), widths[0]), nn.SiLU(),
                nn.Linear(widths[0], widths[1]), nn.SiLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(widths[1] + 11, decoder), nn.SiLU(),
                nn.Linear(decoder, 7 + 7 * 14),
            )
            self.register_buffer("horizon_time_encoding", time.clone())
            self.register_buffer("action_phase_encoding", phase.clone())
            self.register_buffer("causal_jacobian_mask", mask.clone())

        def forward(self, nominal_features: Any) -> Tuple[Any, Any]:
            latent = self.encoder(nominal_features)
            expanded = latent[:, None, :].expand(-1, 51, -1)
            time_features = self.horizon_time_encoding[None].expand(
                len(nominal_features), -1, -1
            )
            phase_features = self.action_phase_encoding[None].expand(
                len(nominal_features), -1, -1
            )
            raw = self.decoder(torch.cat(
                (expanded, time_features, phase_features), dim=2,
            ))
            nominal_displacement = raw[:, :, :7]
            jacobian = raw[:, :, 7:].reshape(-1, 51, 7, 14)
            zero = torch.zeros(
                (len(nominal_features), 1, 7),
                dtype=nominal_displacement.dtype,
                device=nominal_displacement.device,
            )
            nominal_displacement = torch.cat(
                (zero, nominal_displacement[:, 1:]), dim=1,
            )
            jacobian = jacobian * self.causal_jacobian_mask[None]
            return nominal_displacement, jacobian

    return ExplicitExecutionJacobianNet()


def nominal_row_mapping(arrays: Mapping[str, Any]) -> Dict[str, Any]:
    np = _numpy()
    states = np.asarray(arrays["state_index"], dtype=np.int64)
    candidates = np.asarray(arrays["candidate_index"], dtype=np.int64)
    unique_states = np.asarray(sorted(set(states.tolist())), dtype=np.int64)
    anchors = []
    state_to_ordinal = {}
    for ordinal, state in enumerate(unique_states):
        rows = np.flatnonzero((states == state) & (candidates == 0))
        if len(rows) != 1:
            raise ValueError("explicit-jacobian nominal row differs")
        anchors.append(int(rows[0]))
        state_to_ordinal[int(state)] = int(ordinal)
    row_to_ordinal = np.asarray(
        [state_to_ordinal[int(state)] for state in states], dtype=np.int64,
    )
    return {
        "state_index": unique_states,
        "anchor_row_index": np.asarray(anchors, dtype=np.int64),
        "row_to_anchor_ordinal": row_to_ordinal,
        "state_to_anchor_ordinal": state_to_ordinal,
    }


def _gather_action_dimension(jacobian: Any, dimensions: Any) -> Any:
    torch = _torch()
    dimension = torch.as_tensor(
        dimensions, dtype=torch.long, device=jacobian.device,
    )
    indexes = dimension[:, None, None, None].expand(-1, 51, 7, 1)
    return torch.gather(jacobian, 3, indexes).squeeze(3)


def _expand_local_affine(
    nominal_displacement: Any, jacobian: Any, delta_action: Any,
) -> Any:
    torch = _torch()
    return nominal_displacement + torch.einsum(
        "bkjd,bd->bkj", jacobian, delta_action,
    )


def train_explicit_jacobian_ensemble(
    arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
    geometry: Mapping[str, Any], config: Mapping[str, Any],
    normalization: Mapping[str, Any],
) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
    """Fit nominal rollout intercepts and local execution Jacobians."""

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
        raise ValueError("explicit-jacobian normalization differs")
    normalized = (x - mean) / std
    mapping = nominal_row_mapping(arrays)
    anchor_rows = mapping["anchor_row_index"]
    row_to_anchor = mapping["row_to_anchor_ordinal"]
    actions = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    delta_action = actions - actions[anchor_rows[row_to_anchor]]
    state_index = np.asarray(arrays["state_index"], dtype=np.int64)
    clearance = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    jacobians = np.asarray(geometry["jacobian_m_per_rad"], dtype=np.float64)
    exact_sensitivity = np.asarray(
        sensitivities["joint_sensitivity_rad_per_action"], dtype=np.float64,
    )
    sensitivity_states = np.asarray(sensitivities["state_index"], dtype=np.int64)
    dimensions = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    state_split = {
        int(state): str(split[row]) for row, state in enumerate(state_index)
    }
    sensitivity_split = np.asarray([
        state_split[int(state)] for state in sensitivity_states
    ], dtype=object)
    train_sensitivity = np.flatnonzero(sensitivity_split == "train")
    validation_sensitivity = np.flatnonzero(sensitivity_split == "validation")
    sensitivity_anchor_ordinal = np.asarray([
        mapping["state_to_anchor_ordinal"][int(state)]
        for state in sensitivity_states
    ], dtype=np.int64)
    settings = config["training"]
    normalized_scale = normalized_paired_secant_scales(
        exact_sensitivity, dimensions, train_sensitivity, settings,
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
    anchor_features = normalized[anchor_rows]
    for seed in settings["ensemble_seeds"]:
        torch.manual_seed(int(seed))
        model = _build_explicit_model(x.shape[1], config["architecture"])
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
                nominal, execution_jacobian = model(torch.as_tensor(
                    anchor_features[row_to_anchor[indexes]], dtype=torch.float32,
                ))
                prediction = _expand_local_affine(
                    nominal, execution_jacobian, torch.as_tensor(
                        delta_action[indexes], dtype=torch.float32,
                    ),
                )
                truth = torch.as_tensor(displacement[indexes], dtype=torch.float32)
                q_error = prediction - truth
                q_loss = _huber(q_error[:, 1:], joint_beta).mean()
                geometry_jacobian = torch.as_tensor(
                    jacobians[state_index[indexes]], dtype=torch.float32,
                )
                normal_error = torch.einsum(
                    "bkrj,bkj->bkr", geometry_jacobian, q_error,
                )
                weights = torch.as_tensor(
                    element_weight[indexes], dtype=torch.float32,
                )
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
            _, execution_jacobian = model(torch.as_tensor(
                anchor_features[sensitivity_anchor_ordinal[selected]],
                dtype=torch.float32,
            ))
            predicted_sensitivity = _gather_action_dimension(
                execution_jacobian, dimensions[selected],
            )
            exact = torch.as_tensor(
                exact_sensitivity[selected], dtype=torch.float32,
            )
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
                nominal, execution_jacobian = model(torch.as_tensor(
                    anchor_features[row_to_anchor[validation_indexes]],
                    dtype=torch.float32,
                ))
                validation_prediction = _expand_local_affine(
                    nominal, execution_jacobian, torch.as_tensor(
                        delta_action[validation_indexes], dtype=torch.float32,
                    ),
                ).cpu().numpy()
                selected = validation_sensitivity
                _, execution_jacobian = model(torch.as_tensor(
                    anchor_features[sensitivity_anchor_ordinal[selected]],
                    dtype=torch.float32,
                ))
                validation_sensitivity_prediction = _gather_action_dimension(
                    execution_jacobian, dimensions[selected],
                ).cpu().numpy()
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
            raise RuntimeError("explicit-jacobian training produced no checkpoint")
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
        "input_dimension": int(x.shape[1]),
        "nominal_output_shape": [51, 7],
        "jacobian_output_shape": [51, 7, 14],
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
        "local_affine_anchor_count": int(len(anchor_rows)),
        "normalized_paired_secant": {
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
        },
    }
    return models, state, training_audit


def predict_explicit_jacobian(
    models: Sequence[Any], state: Mapping[str, Any], arrays: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    x = np.asarray(arrays["features"], dtype=np.float64)
    normalized = (
        x - np.asarray(state["feature_mean"], dtype=np.float64)
    ) / np.asarray(state["feature_std"], dtype=np.float64)
    mapping = nominal_row_mapping(arrays)
    anchor_rows = mapping["anchor_row_index"]
    row_to_anchor = mapping["row_to_anchor_ordinal"]
    actions = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    delta_action = actions - actions[anchor_rows[row_to_anchor]]
    member_nominal = []
    member_jacobian = []
    with torch.no_grad():
        for model in models:
            nominal, jacobian = model(torch.as_tensor(
                normalized[anchor_rows], dtype=torch.float32,
            ))
            member_nominal.append(nominal.cpu().numpy())
            member_jacobian.append(jacobian.cpu().numpy())
    nominal = np.mean(np.asarray(member_nominal), axis=0)
    jacobian = np.mean(np.asarray(member_jacobian), axis=0)
    predicted_displacement = nominal[row_to_anchor] + np.einsum(
        "nkjd,nd->nkj", jacobian[row_to_anchor], delta_action,
    )
    _, base, _ = _arm_targets(arrays, "factorized_execution")
    return base + predicted_displacement, {
        "state_index": mapping["state_index"],
        "nominal_displacement_rad": nominal,
        "execution_jacobian_rad_per_action": jacobian,
    }


def explicit_jacobian_consistency(
    *, arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
    predicted_q: Any, prediction: Mapping[str, Any],
) -> Dict[str, Any]:
    np = _numpy()
    negative = np.asarray(sensitivities["negative_row_index"], dtype=np.int64)
    positive = np.asarray(sensitivities["positive_row_index"], dtype=np.int64)
    dimensions = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    states = np.asarray(sensitivities["state_index"], dtype=np.int64)
    actions = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    denominator = actions[positive, dimensions] - actions[negative, dimensions]
    secant = (
        np.asarray(predicted_q)[positive] - np.asarray(predicted_q)[negative]
    ) / denominator[:, None, None]
    state_to_ordinal = {
        int(state): index for index, state in enumerate(
            np.asarray(prediction["state_index"], dtype=np.int64)
        )
    }
    jacobian = np.asarray(
        prediction["execution_jacobian_rad_per_action"], dtype=np.float64,
    )
    explicit = np.asarray([
        jacobian[state_to_ordinal[int(state)], :, :, int(dimension)]
        for state, dimension in zip(states, dimensions)
    ])
    error = secant - explicit
    exact = np.asarray(
        sensitivities["joint_sensitivity_rad_per_action"], dtype=np.float64,
    )
    return {
        "finite_difference_pair_count": int(len(secant)),
        "predicted_secant_vs_explicit_J_RMSE_rad_per_action": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "predicted_secant_vs_explicit_J_maximum_absolute_rad_per_action": float(
            np.max(np.abs(error))
        ),
        "maximum_absolute_predicted_J_at_substep_zero": float(
            np.max(np.abs(jacobian[:, 0]))
        ),
        "maximum_absolute_predicted_second_action_J_through_substep_25": float(
            np.max(np.abs(jacobian[:, 1:26, :, 7:14]))
        ),
        "maximum_absolute_exact_second_action_sensitivity_through_substep_25": (
            float(np.max(np.abs(exact[dimensions >= 7, 1:26])))
        ),
    }


def save_explicit_jacobian_weights(
    path: Path, state: Mapping[str, Any],
) -> Dict[str, Any]:
    np = _numpy()
    archive: Dict[str, Any] = {
        "schema_version": np.asarray(WEIGHTS_SCHEMA),
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float64),
        "feature_std": np.asarray(state["feature_std"], dtype=np.float64),
        "input_dimension": np.asarray(state["input_dimension"], dtype=np.int64),
        "nominal_output_shape": np.asarray([51, 7], dtype=np.int64),
        "jacobian_output_shape": np.asarray([51, 7, 14], dtype=np.int64),
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


def load_explicit_jacobian_weights(
    path: Path,
) -> Tuple[List[Any], Dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    archive = np.load(path, allow_pickle=False)
    if str(archive["schema_version"].item()) != WEIGHTS_SCHEMA:
        raise ValueError("explicit-jacobian weights schema differs")
    architecture = json.loads(str(archive["architecture_json"].item()))
    input_dimension = int(archive["input_dimension"].item())
    seeds = archive["ensemble_seeds"].astype(int).tolist()
    models = []
    states = []
    for member in range(len(seeds)):
        model = _build_explicit_model(input_dimension, architecture)
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
        "input_dimension": input_dimension,
        "nominal_output_shape": [51, 7],
        "jacobian_output_shape": [51, 7, 14],
        "ensemble_seeds": seeds, "architecture": architecture,
        "model_states": states,
    }
