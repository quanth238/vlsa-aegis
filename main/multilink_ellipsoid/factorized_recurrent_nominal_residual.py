"""Recurrent nominal-plus-residual model of two-action OSC execution."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .complete_osc_margin import _canonical, _numpy, _torch
from .factorized_direct_horizon_displacement import (
    _huber, horizon_time_encoding, normalized_paired_secant_loss,
    normalized_paired_secant_scales,
)
from .factorized_execution_pilot import _arm_targets
from .factorized_explicit_execution_jacobian import (
    action_phase_encoding, nominal_row_mapping,
)


CONFIG_SCHEMA = "vlsa_distal_factorized_recurrent_nominal_residual_config.v1"
WEIGHTS_SCHEMA = "vlsa_distal_factorized_recurrent_nominal_residual_weights.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_recurrent_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "architecture", "causal_contract", "training",
        "fitted_prediction_gate", "untouched_prediction_gate",
        "forbidden_before_fitted_pass",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("recurrent nominal-residual config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-recurrent-nominal-residual-moka10-v1"
    ):
        raise ValueError("recurrent nominal-residual protocol differs")
    architecture = config["architecture"]
    if architecture != {
        "input_representation": (
            "structured_complete_OSC_nominal_anchor_plus_raw_candidate_delta"
        ),
        "anchor_encoder_widths": [256, 256],
        "recurrent_hidden_width": 192,
        "activation": "silu",
        "trace_state_count": 51,
        "joint_dimension": 7,
        "action_dimension": 14,
        "time_encoding_dimension": 9,
        "action_phase_encoding_dimension": 2,
        "nominal_branch": "GRU_over_causal_anchor_actions",
        "residual_branch": "shared_GRU_candidate_delta_minus_zero_delta",
        "output": "direct_delta_q_k_from_q0_without_increment_integration",
        "exact_initial_condition": "delta_q_0_is_architectural_zero",
        "exact_zero_residual": "candidate_equals_anchor_gives_zero_residual",
    }:
        raise ValueError("recurrent nominal-residual architecture differs")
    if config["causal_contract"] != {
        "controller_substeps_per_action": 25,
        "first_action_active_substeps": [1, 25],
        "second_action_starts_at_substep": 26,
        "second_action_delta_has_no_effect_through_substep": 25,
        "joint_outputs_are_direct_from_q0_not_cumulative": True,
    }:
        raise ValueError("recurrent nominal-residual causal contract differs")
    training = config["training"]
    if training != {
        "device": (
            "cpu_inside_H100_allocation_due_to_pinned_PyTorch_missing_sm90_kernels"
        ),
        "ensemble_seeds": [20260861, 20260862, 20260863, 20260864, 20260865],
        "batch_size": 512,
        "prediction_batch_size": 256,
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
        "checkpoint_selection": (
            "among_validation_joint_gate_checkpoints_minimize_training_"
            "normalized_validation_paired_secant_then_geometry_then_joint"
        ),
    }:
        raise ValueError("recurrent nominal-residual training differs")
    forbidden = config["forbidden_before_fitted_pass"]
    if set(forbidden) != {
        "untouched_episode_opening", "flow_guidance", "matched_random_rollout",
        "calibration", "QP", "closed_loop", "poisson_or_SDF",
        "binary_classifier",
    } or not all(bool(value) for value in forbidden.values()):
        raise ValueError("recurrent nominal-residual forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _active_action_sequence(action: Any) -> Any:
    """Return action 0 for k=1..25 and action 1 for k=26..50."""

    torch = _torch()
    if action.ndim != 3 or tuple(action.shape[1:]) != (2, 7):
        raise ValueError("recurrent action tensor differs")
    return torch.cat((
        action[:, 0:1].expand(-1, 25, -1),
        action[:, 1:2].expand(-1, 25, -1),
    ), dim=1)


def _build_recurrent_model(
    input_dimension: int, architecture: Mapping[str, Any],
) -> Any:
    torch = _torch()
    nn = torch.nn
    widths = [int(value) for value in architecture["anchor_encoder_widths"]]
    hidden = int(architecture["recurrent_hidden_width"])
    time = torch.as_tensor(horizon_time_encoding()[1:], dtype=torch.float32)
    phase = torch.as_tensor(action_phase_encoding()[1:], dtype=torch.float32)

    class RecurrentNominalResidualNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor_encoder = nn.Sequential(
                nn.Linear(int(input_dimension), widths[0]), nn.SiLU(),
                nn.Linear(widths[0], widths[1]), nn.SiLU(),
            )
            self.nominal_initial = nn.Linear(widths[1], hidden)
            self.residual_initial = nn.Linear(widths[1], hidden)
            self.nominal_gru = nn.GRU(
                input_size=9 + 2 + 7, hidden_size=hidden, batch_first=True,
            )
            self.residual_gru = nn.GRU(
                input_size=9 + 2 + 7 + 7,
                hidden_size=hidden, batch_first=True,
            )
            self.nominal_head = nn.Linear(hidden, 7)
            self.residual_head = nn.Linear(hidden, 7)
            self.register_buffer("time_encoding", time.clone())
            self.register_buffer("phase_encoding", phase.clone())

        def forward(
            self, anchor_features: Any, anchor_action: Any,
            candidate_delta: Any,
        ) -> Any:
            batch = len(anchor_features)
            latent = self.anchor_encoder(anchor_features)
            time_features = self.time_encoding[None].expand(batch, -1, -1)
            phase_features = self.phase_encoding[None].expand(batch, -1, -1)
            active_anchor = _active_action_sequence(anchor_action)
            active_delta = _active_action_sequence(candidate_delta)
            nominal_input = torch.cat(
                (time_features, phase_features, active_anchor), dim=2,
            )
            nominal_hidden, _ = self.nominal_gru(
                nominal_input, self.nominal_initial(latent)[None],
            )
            nominal = self.nominal_head(nominal_hidden)
            common = torch.cat(
                (time_features, phase_features, active_anchor), dim=2,
            )
            residual_input = torch.cat((common, active_delta), dim=2)
            zero_input = torch.cat((common, torch.zeros_like(active_delta)), dim=2)
            combined_input = torch.cat((residual_input, zero_input), dim=0)
            initial = self.residual_initial(latent)[None]
            combined_initial = torch.cat((initial, initial), dim=1)
            combined_hidden, _ = self.residual_gru(
                combined_input, combined_initial,
            )
            candidate_hidden, zero_hidden = torch.chunk(combined_hidden, 2, dim=0)
            residual = (
                self.residual_head(candidate_hidden)
                - self.residual_head(zero_hidden)
            )
            direct_displacement = nominal + residual
            zero = torch.zeros(
                (batch, 1, 7), dtype=direct_displacement.dtype,
                device=direct_displacement.device,
            )
            return torch.cat((zero, direct_displacement), dim=1)

    return RecurrentNominalResidualNet()


def _prepared_inputs(
    arrays: Mapping[str, Any], normalization: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    x = np.asarray(arrays["features"], dtype=np.float64)
    mean = np.asarray(normalization["feature_mean"], dtype=np.float64)
    std = np.asarray(normalization["feature_std"], dtype=np.float64)
    if mean.shape != (x.shape[1],) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("recurrent normalization differs")
    mapping = nominal_row_mapping(arrays)
    anchor_rows = mapping["anchor_row_index"]
    row_to_anchor = mapping["row_to_anchor_ordinal"]
    action = np.asarray(arrays["action_chunk"], dtype=np.float64)
    if action.shape != (len(x), 2, 7):
        raise ValueError("recurrent action population differs")
    anchor_action = action[anchor_rows]
    delta = action - anchor_action[row_to_anchor]
    return {
        "normalized_anchor_features": ((x - mean) / std)[anchor_rows],
        "anchor_action": anchor_action,
        "candidate_delta": delta,
        "mapping": mapping,
        "feature_mean": mean,
        "feature_std": std,
    }


def train_recurrent_ensemble(
    arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
    geometry: Mapping[str, Any], config: Mapping[str, Any],
    normalization: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Fit direct horizon displacements with a centered recurrent residual."""

    np = _numpy()
    torch = _torch()
    torch.set_num_threads(8)
    prepared = _prepared_inputs(arrays, normalization)
    anchor_features = prepared["normalized_anchor_features"]
    anchor_action = prepared["anchor_action"]
    delta = prepared["candidate_delta"]
    mapping = prepared["mapping"]
    row_to_anchor = mapping["row_to_anchor_ordinal"]
    target, base, _ = _arm_targets(arrays, "factorized_execution")
    displacement = target - base
    split = np.asarray(arrays["split"], dtype=object)
    train_rows = np.flatnonzero(split == "train")
    validation_rows = np.flatnonzero(split == "validation")
    state_index = np.asarray(arrays["state_index"], dtype=np.int64)
    clearance = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    geometry_jacobian = np.asarray(
        geometry["jacobian_m_per_rad"], dtype=np.float64,
    )
    exact_sensitivity = np.asarray(
        sensitivities["joint_sensitivity_rad_per_action"], dtype=np.float64,
    )
    sensitivity_states = np.asarray(
        sensitivities["state_index"], dtype=np.int64,
    )
    dimensions = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    negative_rows = np.asarray(
        sensitivities["negative_row_index"], dtype=np.int64,
    )
    positive_rows = np.asarray(
        sensitivities["positive_row_index"], dtype=np.int64,
    )
    flat_action = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    denominator = (
        flat_action[positive_rows, dimensions]
        - flat_action[negative_rows, dimensions]
    )
    state_to_split = {
        int(state): str(name) for state, name in zip(state_index, split)
    }
    sensitivity_split = np.asarray([
        state_to_split[int(state)] for state in sensitivity_states
    ], dtype=object)
    train_sensitivity = np.flatnonzero(sensitivity_split == "train")
    validation_sensitivity = np.flatnonzero(sensitivity_split == "validation")
    settings = config["training"]
    scale = normalized_paired_secant_scales(
        exact_sensitivity, dimensions, train_sensitivity, settings,
    )
    joint_beta = float(settings["joint_huber_delta_rad"])
    geometry_beta = float(settings["geometry_huber_delta_m"])
    boundary = float(settings["near_boundary_absolute_margin_m"])
    element_weight = np.where(
        np.abs(clearance) <= boundary,
        float(settings["near_boundary_weight_multiplier"]), 1.0,
    )
    maximum_validation_q = float(
        config["fitted_prediction_gate"]["maximum_validation_joint_RMSE_rad"]
    )

    def forward_rows(model: Any, rows: Any) -> Any:
        ordinal = row_to_anchor[rows]
        return model(
            torch.as_tensor(anchor_features[ordinal], dtype=torch.float32),
            torch.as_tensor(anchor_action[ordinal], dtype=torch.float32),
            torch.as_tensor(delta[rows], dtype=torch.float32),
        )

    models = []
    states = []
    member_audits = []
    for seed in settings["ensemble_seeds"]:
        torch.manual_seed(int(seed))
        model = _build_recurrent_model(
            anchor_features.shape[1], config["architecture"],
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        generator = np.random.default_rng(int(seed))
        best_key = None
        best = None
        best_epoch = -1
        best_record = None
        stale = 0
        for epoch in range(int(settings["epochs"])):
            model.train()
            order = generator.permutation(train_rows)
            for start in range(0, len(order), int(settings["batch_size"])):
                rows = order[start:start + int(settings["batch_size"])]
                prediction = forward_rows(model, rows)
                truth = torch.as_tensor(displacement[rows], dtype=torch.float32)
                q_error = prediction - truth
                q_loss = _huber(q_error[:, 1:], joint_beta).mean()
                dhdq = torch.as_tensor(
                    geometry_jacobian[state_index[rows]], dtype=torch.float32,
                )
                normal_error = torch.einsum("bkrj,bkj->bkr", dhdq, q_error)
                weights = torch.as_tensor(
                    element_weight[rows], dtype=torch.float32,
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
            model.train()
            selected = train_sensitivity
            negative = forward_rows(model, negative_rows[selected])
            positive = forward_rows(model, positive_rows[selected])
            predicted_sensitivity = (
                (positive - negative)
                / torch.as_tensor(
                    denominator[selected], dtype=torch.float32,
                )[:, None, None]
            )
            sensitivity_loss = normalized_paired_secant_loss(
                predicted_sensitivity,
                torch.as_tensor(exact_sensitivity[selected], dtype=torch.float32),
                dimensions[selected], scale,
            )
            optimizer.zero_grad()
            (
                float(settings["finite_difference_sensitivity_loss_weight"])
                * sensitivity_loss
            ).backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_parts = []
                for start in range(
                    0, len(validation_rows), int(settings["prediction_batch_size"])
                ):
                    validation_parts.append(forward_rows(
                        model,
                        validation_rows[
                            start:start + int(settings["prediction_batch_size"])
                        ],
                    ).cpu().numpy())
                validation_prediction = np.concatenate(validation_parts, axis=0)
                selected = validation_sensitivity
                negative = forward_rows(model, negative_rows[selected])
                positive = forward_rows(model, positive_rows[selected])
                validation_sensitivity_prediction = (
                    (positive - negative)
                    / torch.as_tensor(
                        denominator[selected], dtype=torch.float32,
                    )[:, None, None]
                )
                validation_sensitivity_loss = float(
                    normalized_paired_secant_loss(
                        validation_sensitivity_prediction,
                        torch.as_tensor(
                            exact_sensitivity[selected], dtype=torch.float32,
                        ),
                        dimensions[selected], scale,
                    ).cpu()
                )
            q_error = (
                validation_prediction - displacement[validation_rows]
            )
            q_rmse = float(np.sqrt(np.mean(q_error[:, 1:] ** 2)))
            normal_error = np.einsum(
                "bkrj,bkj->bkr",
                geometry_jacobian[state_index[validation_rows]], q_error,
            )
            geometry_rmse = float(np.sqrt(np.mean(normal_error ** 2)))
            qualified = q_rmse <= maximum_validation_q
            key = (
                (0, validation_sensitivity_loss, geometry_rmse, q_rmse)
                if qualified else
                (1, q_rmse - maximum_validation_q,
                 validation_sensitivity_loss, geometry_rmse)
            )
            if best_key is None or key < best_key:
                best_key = key
                best_epoch = int(epoch)
                best = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_record = {
                    "validation_joint_RMSE_rad": q_rmse,
                    "validation_normalized_secant_loss": (
                        validation_sensitivity_loss
                    ),
                    "validation_safety_normal_RMSE_m": geometry_rmse,
                    "validation_joint_gate_qualified": bool(qualified),
                }
                stale = 0
            else:
                stale += 1
            if stale >= int(settings["patience"]):
                break
        if best is None or best_record is None:
            raise RuntimeError("recurrent training produced no checkpoint")
        model.load_state_dict(best)
        model.eval()
        models.append(model)
        states.append({name: value.numpy() for name, value in best.items()})
        member_audits.append({
            "seed": int(seed), "best_epoch": best_epoch,
            "completed_epoch_count": int(epoch + 1), **best_record,
        })
    state = {
        "feature_mean": prepared["feature_mean"],
        "feature_std": prepared["feature_std"],
        "input_dimension": int(anchor_features.shape[1]),
        "ensemble_seeds": list(settings["ensemble_seeds"]),
        "architecture": dict(config["architecture"]),
        "model_states": states,
    }
    audit = {
        "member_audits": member_audits,
        "training_row_count": int(len(train_rows)),
        "validation_row_count": int(len(validation_rows)),
        "training_sensitivity_row_count": int(len(train_sensitivity)),
        "validation_sensitivity_row_count": int(len(validation_sensitivity)),
        "nominal_anchor_count": int(len(mapping["anchor_row_index"])),
        "normalized_paired_secant": {
            "scale_fit_split": "train_only",
            "RMS_scale_rad_per_action": scale[
                "RMS_scale_rad_per_action"
            ].tolist(),
            "sample_count": scale["sample_count"].tolist(),
            "valid": scale["valid"].tolist(),
            "weight": scale["weight"].tolist(),
            "valid_cell_count": int(scale["valid_cell_count"]),
            "excluded_cell_count": int(scale["excluded_cell_count"]),
            "clipped_low_cell_count": int(scale["clipped_low_cell_count"]),
            "clipped_high_cell_count": int(scale["clipped_high_cell_count"]),
        },
    }
    return models, state, audit


def predict_recurrent(
    models: Sequence[Any], state: Mapping[str, Any], arrays: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    prepared = _prepared_inputs(arrays, state)
    anchor_features = prepared["normalized_anchor_features"]
    anchor_action = prepared["anchor_action"]
    delta = prepared["candidate_delta"]
    row_to_anchor = prepared["mapping"]["row_to_anchor_ordinal"]
    members = []
    batch_size = 256
    with torch.no_grad():
        for model in models:
            parts = []
            for start in range(0, len(delta), batch_size):
                rows = np.arange(start, min(start + batch_size, len(delta)))
                ordinal = row_to_anchor[rows]
                parts.append(model(
                    torch.as_tensor(anchor_features[ordinal], dtype=torch.float32),
                    torch.as_tensor(anchor_action[ordinal], dtype=torch.float32),
                    torch.as_tensor(delta[rows], dtype=torch.float32),
                ).cpu().numpy())
            members.append(np.concatenate(parts, axis=0))
    member_displacement = np.asarray(members, dtype=np.float64)
    displacement = np.mean(member_displacement, axis=0)
    _, base, _ = _arm_targets(arrays, "factorized_execution")
    return base + displacement, {
        "member_joint_position_rad": base[None] + member_displacement,
        "member_displacement_rad": member_displacement,
        "state_index": prepared["mapping"]["state_index"],
    }


def save_recurrent_weights(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    arrays: dict[str, Any] = {
        "schema_version": np.asarray(WEIGHTS_SCHEMA),
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float64),
        "feature_std": np.asarray(state["feature_std"], dtype=np.float64),
        "input_dimension": np.asarray(state["input_dimension"], dtype=np.int64),
        "ensemble_seeds": np.asarray(state["ensemble_seeds"], dtype=np.int64),
        "architecture_json": np.asarray(json.dumps(
            state["architecture"], sort_keys=True, separators=(",", ":"),
        )),
    }
    for member, values in enumerate(state["model_states"]):
        for name, value in values.items():
            arrays["member_%d__%s" % (member, name)] = np.asarray(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "path": str(Path(path).resolve()),
        "file_sha256": _sha256(Path(path).read_bytes()),
    }


def load_recurrent_weights(path: Path) -> tuple[list[Any], dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    archive = np.load(path, allow_pickle=False)
    if str(archive["schema_version"].item()) != WEIGHTS_SCHEMA:
        raise ValueError("recurrent weights schema differs")
    architecture = json.loads(str(archive["architecture_json"].item()))
    input_dimension = int(archive["input_dimension"].item())
    seeds = archive["ensemble_seeds"].astype(int).tolist()
    models = []
    states = []
    for member in range(len(seeds)):
        model = _build_recurrent_model(input_dimension, architecture)
        prefix = "member_%d__" % member
        values = {
            name[len(prefix):]: torch.as_tensor(archive[name])
            for name in archive.files if name.startswith(prefix)
        }
        model.load_state_dict(values)
        model.eval()
        models.append(model)
        states.append({name: value.numpy() for name, value in values.items()})
    return models, {
        "feature_mean": archive["feature_mean"].astype(np.float64),
        "feature_std": archive["feature_std"].astype(np.float64),
        "input_dimension": input_dimension,
        "ensemble_seeds": seeds,
        "architecture": architecture,
        "model_states": states,
    }


def fitted_prediction_gate(
    *, metrics: Mapping[str, Any], support: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["fitted_prediction_gate"]
    tests: dict[str, bool] = {}
    recorded: dict[str, Any] = {}
    for split in ("train", "validation"):
        recorded[split] = {}
        for scope, horizon in (("aggregate", None), ("terminal", 50)):
            source = metrics["sensitivity"][split]["all_horizon_trace"]
            item = source["joint"] if horizon is None else source["by_horizon"][horizon]["joint"]
            safety = source["safety"] if horizon is None else source["by_horizon"][horizon]["safety"]
            recorded[split][scope] = {"joint": item, "safety": safety}
            for name, value in (("joint", item), ("safety", safety)):
                prefix = "%s_%s_%s" % (split, scope, name)
                tests[prefix + "_direction"] = bool(
                    value["mean_cosine"] is not None
                    and float(value["mean_cosine"])
                    > float(gate["minimum_sensitivity_cosine"])
                )
                tests[prefix + "_magnitude"] = bool(
                    value["median_norm_ratio"] is not None
                    and float(gate["minimum_median_norm_ratio"])
                    <= float(value["median_norm_ratio"])
                    <= float(gate["maximum_median_norm_ratio"])
                    and float(value["mean_relative_norm_error"])
                    <= float(gate["maximum_mean_relative_norm_error"])
                )
    validation_safety = metrics["safety"]["validation"]
    validation_temporal = metrics["temporal_joint"]["validation"]
    tests.update({
        "zero_validation_false_safes": int(
            validation_safety["false_safe_action_count"]
        ) == 0,
        "validation_safe_recall": float(
            validation_safety["exact_safe_action_recall"]
        ) >= float(gate["minimum_validation_safe_recall"]),
        "validation_safe_support": bool(
            support["validation"]["all_eligible_states_supported"]
        ),
        "validation_boundary_RMSE": (
            validation_safety["near_boundary_RMSE_m"] is not None
            and float(validation_safety["near_boundary_RMSE_m"])
            <= float(gate["maximum_validation_near_boundary_RMSE_m"])
        ),
        "validation_joint_RMSE": float(
            validation_temporal["overall_joint_RMSE_rad"]
        ) <= float(gate["maximum_validation_joint_RMSE_rad"]),
        "validation_terminal_joint_RMSE": float(
            validation_temporal["terminal_joint_RMSE_rad"]
        ) <= float(gate["maximum_validation_terminal_joint_RMSE_rad"]),
        "validation_horizon_slope": float(
            validation_temporal["linear_RMSE_slope_rad_per_substep"]
        ) <= float(gate["maximum_RMSE_slope_rad_per_substep"]),
        "validation_terminal_ratio": float(
            validation_temporal["terminal_to_overall_RMSE_ratio"]
        ) <= float(gate["maximum_terminal_to_overall_RMSE_ratio"]),
        "exact_q0": float(
            validation_temporal["maximum_initial_joint_error_rad"]
        ) == 0.0,
    })
    passed = bool(all(tests.values()))
    return {
        "tests": tests, "sensitivity": recorded,
        "fitted_prediction_gate_pass": passed,
        "untouched_episode_evaluation_authorized": passed,
        "flow_guidance_calibration_QP_closed_loop_authorized": False,
        "conclusion": (
            "recurrent_execution_model_passes_fitted_gate"
            if passed else "recurrent_execution_model_fails_fitted_gate"
        ),
    }
