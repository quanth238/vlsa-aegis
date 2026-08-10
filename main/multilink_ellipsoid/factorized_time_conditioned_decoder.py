"""Shared time-conditioned decoder for short-horizon OSC execution."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .complete_osc_margin import _canonical, _numpy, _torch
from .factorized_execution_pilot import _arm_targets
from .factorized_one_sided_geometry import signed_clearance_error


CONFIG_SCHEMA = "vlsa_distal_factorized_time_conditioned_decoder_config.v1"
RESULT_SCHEMA = "vlsa_distal_factorized_time_conditioned_decoder_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_time_conditioned_decoder_validation.v1"
WEIGHTS_SCHEMA = "vlsa_distal_factorized_time_conditioned_decoder_weights.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_time_decoder_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("time-conditioned decoder config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "matched_ablation", "architecture",
        "local_geometry_jacobian", "training", "decision_gate",
        "final_evaluation_policy", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("time-conditioned decoder config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-time-conditioned-decoder-moka10-v1"
    ):
        raise ValueError("time-conditioned decoder protocol differs")
    if config["population"] != {
        "episode_count": 17, "state_count": 85,
        "state_split_counts": {"train": 60, "validation": 10, "test": 15},
        "candidate_count_per_state": 93, "random_candidate_count_per_state": 64,
        "joint_trace_state_count": 51, "constraint_count": 7,
        "diagnostic_test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
    }:
        raise ValueError("time-conditioned decoder population differs")
    if config["architecture"] != {
        "state_action_encoder_widths": [256, 256],
        "time_decoder_width": 128, "joint_output_dimension": 7,
        "activation": "silu", "time_coordinate": "tau_equals_k_over_50",
        "time_encoding": "tau_plus_sin_cos_pi_tau_at_frequencies_1_2_4_8",
        "time_encoding_dimension": 9,
        "shared_weights_across_all_51_substeps": True,
        "exact_initial_condition": "predicted_joint_residual_equals_tau_times_decoder_output",
    }:
        raise ValueError("time-conditioned decoder architecture differs")
    if config["matched_ablation"] != {
        "baseline": "immutable_job_38070_flat_51_by_7_one_sided_model",
        "experimental": "shared_time_conditioned_joint_execution_decoder",
        "same_dataset_splits_complete_inputs_targets_seeds_optimizer_schedule_and_one_sided_geometry_loss": True,
        "only_model_change": "flat_357_output_decoder_to_shared_F_theta_state_action_k",
        "inference_output": "51_by_7_joint_trajectory_only",
        "deployment_safety": "known_FK_plus_existing_ellipsoid_geometry",
    }:
        raise ValueError("time-conditioned decoder matched ablation differs")
    if config["local_geometry_jacobian"] != {
        "source": "same_train_validation_only_local_dh_dq_as_job_38070",
        "fit_rows": "nominal_plus_28_full_action_finite_difference_candidates",
        "fit_state_splits": ["train", "validation"],
        "test_split_used_for_fit": False,
        "ridge_regularization": 1.0e-6,
        "maximum_validation_random_linearization_RMSE_m": 0.002,
    }:
        raise ValueError("time-conditioned decoder geometry supervision differs")
    training = config["training"]
    if (
        training["ensemble_seeds"]
        != [20260841, 20260842, 20260843, 20260844, 20260845]
        or int(training["batch_size"]) != 512
        or int(training["epochs"]) != 300
        or int(training["patience"]) != 40
        or int(training["training_substeps_per_update"]) != 4
        or float(training["joint_huber_delta_rad"]) != 0.002
        or float(training["geometry_huber_delta_m"]) != 0.002
        or float(training["finite_difference_loss_weight"]) != 1.0
        or float(training["geometry_loss_weight"]) != 10.0
        or float(training["dangerous_overestimate_weight"]) != 4.0
        or float(training["near_boundary_absolute_margin_m"]) != 0.005
        or float(training["near_boundary_weight_multiplier"]) != 5.0
        or float(training["terminal_weight_multiplier"]) != 3.0
        or int(training["terminal_weight_power"]) != 4
        or training["training_substep_sampling"]
        != "terminal_50_plus_three_without_replacement_from_1_through_49"
    ):
        raise ValueError("time-conditioned decoder training differs")
    forbidden = config["forbidden_actions"]
    if set(forbidden) != {
        "new_controller_rollout_labels", "candidate_region_expansion",
        "uncertainty_calibration", "binary_classifier", "surface_loss",
        "poisson_or_SDF", "QP", "closed_loop",
    } or not all(value is True for value in forbidden.values()):
        raise ValueError("time-conditioned decoder forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def time_encoding(substep_indexes: Any) -> Any:
    """Return fixed smooth time features for integer substeps 0 through 50."""

    np = _numpy()
    indexes = np.asarray(substep_indexes, dtype=np.int64)
    if indexes.ndim != 1 or np.any(indexes < 0) or np.any(indexes > 50):
        raise ValueError("time-conditioned decoder substep indexes differ")
    tau = indexes.astype(np.float64) / 50.0
    parts = [tau]
    for frequency in (1.0, 2.0, 4.0, 8.0):
        parts.extend((np.sin(math.pi * frequency * tau),
                      np.cos(math.pi * frequency * tau)))
    output = np.stack(parts, axis=1)
    if output.shape != (len(indexes), 9):
        raise ValueError("time-conditioned decoder time encoding differs")
    return output


def _build_time_model(input_dimension: int, architecture: Mapping[str, Any]) -> Any:
    torch = _torch()
    nn = torch.nn
    encoder_widths = [int(value) for value in architecture["state_action_encoder_widths"]]
    decoder_width = int(architecture["time_decoder_width"])
    encoding = torch.as_tensor(time_encoding(list(range(51))), dtype=torch.float32)

    class TimeConditionedExecutionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(int(input_dimension), encoder_widths[0]), nn.SiLU(),
                nn.Linear(encoder_widths[0], encoder_widths[1]), nn.SiLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoder_widths[1] + 9, decoder_width), nn.SiLU(),
                nn.Linear(decoder_width, 7),
            )
            self.register_buffer("registered_time_encoding", encoding.clone())

        def forward(self, features: Any, substeps: Any = None) -> Any:
            if substeps is None:
                indexes = torch.arange(51, device=features.device)
            else:
                indexes = torch.as_tensor(substeps, dtype=torch.long,
                                          device=features.device)
            latent = self.encoder(features)
            encoded = self.registered_time_encoding[indexes]
            expanded_latent = latent[:, None, :].expand(-1, len(indexes), -1)
            expanded_time = encoded[None, :, :].expand(len(features), -1, -1)
            raw = self.decoder(torch.cat((expanded_latent, expanded_time), dim=2))
            tau = indexes.to(dtype=raw.dtype)[None, :, None] / 50.0
            return tau * raw

    return TimeConditionedExecutionNet()


def _sample_training_substeps(generator: Any) -> Any:
    np = _numpy()
    interior = generator.choice(np.arange(1, 50), size=3, replace=False)
    return np.asarray(sorted(interior.tolist()) + [50], dtype=np.int64)


def train_time_conditioned_ensemble(
    arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
    geometry: Mapping[str, Any], config: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Train the registered shared decoder with one-sided geometry loss."""

    np = _numpy()
    torch = _torch()
    torch.set_num_threads(8)
    x = np.asarray(arrays["features"], dtype=np.float64)
    target, base, _ = _arm_targets(arrays, "factorized_execution")
    residual = target - base
    split = np.asarray(arrays["split"], dtype=object)
    train_mask = split == "train"
    validation_mask = split == "validation"
    mean = np.mean(x[train_mask], axis=0)
    std = np.maximum(np.std(x[train_mask], axis=0), 1.0e-6)
    normalized = (x - mean) / std
    train_indexes = np.flatnonzero(train_mask)
    validation_indexes = np.flatnonzero(validation_mask)
    state_index = np.asarray(arrays["state_index"], dtype=np.int64)
    clearance = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    jacobians = np.asarray(geometry["jacobian_m_per_rad"], dtype=np.float64)
    exact_sensitivity = np.asarray(
        sensitivities["joint_sensitivity_rad_per_action"], dtype=np.float64,
    )
    sensitivity_states = np.asarray(sensitivities["state_index"], dtype=np.int64)
    state_split = {int(state): str(split[row])
                   for row, state in enumerate(state_index)}
    sensitivity_split = np.asarray(
        [state_split[int(state)] for state in sensitivity_states], dtype=object,
    )
    train_sensitivity = np.flatnonzero(sensitivity_split == "train")
    validation_sensitivity = np.flatnonzero(sensitivity_split == "validation")
    neg_rows = np.asarray(sensitivities["negative_row_index"], dtype=np.int64)
    pos_rows = np.asarray(sensitivities["positive_row_index"], dtype=np.int64)
    dims = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    actions = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    denominators = actions[pos_rows, dims] - actions[neg_rows, dims]
    settings = config["training"]
    joint_delta = float(settings["joint_huber_delta_rad"])
    geometry_delta = float(settings["geometry_huber_delta_m"])
    boundary = float(settings["near_boundary_absolute_margin_m"])
    late = 1.0 + (float(settings["terminal_weight_multiplier"]) - 1.0) * (
        np.arange(51, dtype=np.float64) / 50.0
    ) ** int(settings["terminal_weight_power"])
    element_weight = np.where(
        np.abs(clearance) <= boundary,
        float(settings["near_boundary_weight_multiplier"]), 1.0,
    ) * late[None, :, None]
    models = []
    model_states = []
    audits = []
    all_substeps = np.arange(51, dtype=np.int64)
    for seed in settings["ensemble_seeds"]:
        torch.manual_seed(int(seed))
        model = _build_time_model(x.shape[1], config["architecture"])
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
                substeps = _sample_training_substeps(generator)
                prediction = model(torch.as_tensor(
                    normalized[indexes], dtype=torch.float32,
                ), substeps)
                exact_residual = torch.as_tensor(
                    residual[indexes][:, substeps], dtype=torch.float32,
                )
                q_error = prediction - exact_residual
                absolute = torch.abs(q_error)
                q_loss = torch.where(
                    absolute <= joint_delta, 0.5 * q_error ** 2,
                    joint_delta * (absolute - 0.5 * joint_delta),
                ).mean()
                jacobian = torch.as_tensor(
                    jacobians[state_index[indexes]][:, substeps],
                    dtype=torch.float32,
                )
                h_error = torch.einsum("btrj,btj->btr", jacobian, q_error)
                h_absolute = torch.abs(h_error)
                h_huber = torch.where(
                    h_absolute <= geometry_delta, 0.5 * h_error ** 2,
                    geometry_delta * (h_absolute - 0.5 * geometry_delta),
                )
                weights = torch.as_tensor(
                    element_weight[indexes][:, substeps], dtype=torch.float32,
                )
                geometry_loss = torch.mean(weights * h_huber)
                geometry_loss = geometry_loss + float(
                    settings["dangerous_overestimate_weight"]
                ) * torch.mean(weights * torch.relu(h_error) ** 2)
                loss = q_loss + float(settings["geometry_loss_weight"]) * geometry_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            substeps = _sample_training_substeps(generator)
            selected = train_sensitivity
            model.train()
            negative = model(torch.as_tensor(
                normalized[neg_rows[selected]], dtype=torch.float32,
            ), substeps)
            positive = model(torch.as_tensor(
                normalized[pos_rows[selected]], dtype=torch.float32,
            ), substeps)
            predicted_sensitivity = (positive - negative) / torch.as_tensor(
                denominators[selected], dtype=torch.float32,
            )[:, None, None]
            exact_s = torch.as_tensor(
                exact_sensitivity[selected][:, substeps], dtype=torch.float32,
            )
            sensitivity_loss = torch.nn.functional.smooth_l1_loss(
                predicted_sensitivity, exact_s, beta=joint_delta,
            )
            optimizer.zero_grad()
            (float(settings["finite_difference_loss_weight"])
             * sensitivity_loss).backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_prediction = model(torch.as_tensor(
                    normalized[validation_indexes], dtype=torch.float32,
                ), all_substeps).cpu().numpy()
                q_rmse = float(np.sqrt(np.mean(
                    (validation_prediction - residual[validation_indexes]) ** 2
                )))
                selected = validation_sensitivity
                negative = model(torch.as_tensor(
                    normalized[neg_rows[selected]], dtype=torch.float32,
                ), all_substeps)
                positive = model(torch.as_tensor(
                    normalized[pos_rows[selected]], dtype=torch.float32,
                ), all_substeps)
                predicted_s = (
                    (positive - negative).cpu().numpy()
                    / denominators[selected, None, None]
                )
                sensitivity_rmse = float(np.sqrt(np.mean(
                    (predicted_s - exact_sensitivity[selected]) ** 2
                )))
            validation_q = base[validation_indexes] + validation_prediction
            h_error = signed_clearance_error(
                validation_q, target[validation_indexes],
                state_index[validation_indexes], jacobians,
            )
            geometry_rmse = float(np.sqrt(np.mean(h_error ** 2)))
            score = q_rmse + 0.1 * sensitivity_rmse + geometry_rmse
            if score < best_score - 1.0e-12:
                best_score = score
                best_epoch = int(epoch)
                best = {key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= int(settings["patience"]):
                break
        if best is None:
            raise RuntimeError("time-conditioned decoder produced no checkpoint")
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
    return models, state, {
        "member_audits": audits, "training_row_count": int(len(train_indexes)),
        "training_sensitivity_row_count": int(len(train_sensitivity)),
        "validation_row_count": int(len(validation_indexes)),
        "validation_sensitivity_row_count": int(len(validation_sensitivity)),
    }


def predict_time_conditioned(
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
    residual = np.mean(np.asarray(members), axis=0)
    _, base, _ = _arm_targets(arrays, "factorized_execution")
    return base + residual


def save_time_weights(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    arrays: dict[str, Any] = {
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
            arrays["member_%d__%s" % (member, key)] = np.asarray(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"path": str(Path(path).resolve()),
            "file_sha256": _sha256(Path(path).read_bytes())}


def load_time_weights(path: Path) -> tuple[list[Any], dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    archive = np.load(path, allow_pickle=False)
    if str(archive["schema_version"].item()) != WEIGHTS_SCHEMA:
        raise ValueError("time-conditioned decoder weights schema differs")
    architecture = json.loads(str(archive["architecture_json"].item()))
    input_dimension = int(archive["input_dimension"].item())
    seeds = archive["ensemble_seeds"].astype(int).tolist()
    models = []
    states = []
    for member in range(len(seeds)):
        model = _build_time_model(input_dimension, architecture)
        prefix = "member_%d__" % member
        values = {key[len(prefix):]: torch.as_tensor(archive[key])
                  for key in archive.files if key.startswith(prefix)}
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


def eligible_state_support(
    *, exact_margin: Any, predicted_margin: Any, arrays: Mapping[str, Any],
    split_name: str,
) -> dict[str, Any]:
    np = _numpy()
    exact = np.asarray(exact_margin, dtype=np.float64)
    predicted = np.asarray(predicted_margin, dtype=np.float64)
    split = np.asarray(arrays["split"], dtype=object)
    source = np.asarray(arrays["source_code"], dtype=np.int8)
    states = np.asarray(arrays["state_index"], dtype=np.int64)
    selected = (split == split_name) & (source == 2)
    eligible = []
    supported = []
    for state in sorted(set(states[selected].tolist())):
        rows = selected & (states == state)
        exact_safe = np.all(exact[rows] >= 0.0, axis=1)
        accepted = exact_safe & np.all(predicted[rows] >= 0.0, axis=1)
        if np.any(exact_safe):
            eligible.append(int(state))
            if np.any(accepted):
                supported.append(int(state))
    return {
        "eligible_state_indexes": eligible,
        "supported_state_indexes": supported,
        "eligible_state_count": int(len(eligible)),
        "supported_state_count": int(len(supported)),
        "all_eligible_states_supported": eligible == supported,
    }


def temporal_joint_metrics(
    predicted_q: Any, exact_q: Any, arrays: Mapping[str, Any], split_name: str,
) -> dict[str, Any]:
    np = _numpy()
    split = np.asarray(arrays["split"], dtype=object)
    source = np.asarray(arrays["source_code"], dtype=np.int8)
    selected = (split == split_name) & (source == 2)
    error = np.asarray(predicted_q)[selected] - np.asarray(exact_q)[selected]
    by_substep = np.sqrt(np.mean(error ** 2, axis=(0, 2)))
    return {
        "action_count": int(np.count_nonzero(selected)),
        "overall_joint_RMSE_rad": float(np.sqrt(np.mean(error ** 2))),
        "terminal_joint_RMSE_rad": float(by_substep[-1]),
        "joint_RMSE_by_substep_rad": by_substep.tolist(),
    }


def time_decoder_decision(
    *, source_metrics: Mapping[str, Any], source_temporal: Mapping[str, Any],
    experimental_metrics: Mapping[str, Any], experimental_temporal: Mapping[str, Any],
    validation_support: Mapping[str, Any], test_support: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["decision_gate"]
    tests = {
        "zero_validation_false_safe": int(
            experimental_metrics["validation_safety"]["false_safe_action_count"]
        ) <= int(gate["maximum_validation_false_safe_action_count"]),
        "zero_test_false_safe": int(
            experimental_metrics["test_safety"]["false_safe_action_count"]
        ) <= int(gate["maximum_test_false_safe_action_count"]),
        "all_eligible_validation_states_supported": bool(
            validation_support["all_eligible_states_supported"]
        ),
        "all_eligible_test_states_supported": bool(
            test_support["all_eligible_states_supported"]
        ),
        "validation_safe_recall": float(
            experimental_metrics["validation_safety"]["exact_safe_action_recall"]
        ) >= float(gate["minimum_validation_safe_recall"]),
        "test_safe_recall": float(
            experimental_metrics["test_safety"]["exact_safe_action_recall"]
        ) >= float(gate["minimum_test_safe_recall"]),
        "test_boundary_RMSE": float(
            experimental_metrics["test_safety"]["near_boundary_RMSE_m"]
        ) <= float(gate["maximum_test_near_boundary_RMSE_m"]),
        "joint_sensitivity": float(
            experimental_metrics["joint_sensitivity"]["mean_cosine"]
        ) >= float(gate["minimum_joint_sensitivity_cosine"]),
        "margin_sensitivity": float(
            experimental_metrics["margin_sensitivity"]["mean_cosine"]
        ) >= float(gate["minimum_margin_sensitivity_cosine"]),
        "strict_validation_terminal_joint_improvement": float(
            experimental_temporal["validation"]["terminal_joint_RMSE_rad"]
        ) < float(source_temporal["validation"]["terminal_joint_RMSE_rad"]),
        "strict_test_terminal_joint_improvement": float(
            experimental_temporal["test"]["terminal_joint_RMSE_rad"]
        ) < float(source_temporal["test"]["terminal_joint_RMSE_rad"]),
        "strict_test_support_improvement": int(test_support["supported_state_count"])
        > int(source_metrics["test_safety"]["state_safe_support_count"]),
    }
    passed = bool(all(tests.values()))
    return {
        "gate_tests": tests, "time_conditioned_mechanism_GO": passed,
        "conclusion": (
            "shared_time_conditioned_decoder_passes_diagnostic_mechanism_gate"
            if passed else
            "shared_time_conditioned_decoder_does_not_pass_diagnostic_mechanism_gate"
        ),
        "new_unseen_episode_evaluation_authorized": passed,
        "QP_or_closed_loop_authorized": False,
    }
