"""Exact-q0 cumulative joint-displacement model for black-box OSC execution."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .complete_osc_margin import _canonical, _numpy, _torch
from .factorized_execution_pilot import _arm_targets
from .factorized_one_sided_geometry import signed_clearance_error


CONFIG_SCHEMA = "vlsa_distal_factorized_cumulative_delta_q_config.v1"
RESULT_SCHEMA = "vlsa_distal_factorized_cumulative_delta_q_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_cumulative_delta_q_validation.v1"
WEIGHTS_SCHEMA = "vlsa_distal_factorized_cumulative_delta_q_weights.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_cumulative_delta_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "architecture", "training", "mechanism_decision_gate",
        "reserved_evaluation_policy", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("cumulative-delta-q config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-cumulative-delta-q-moka10-v1"
    ):
        raise ValueError("cumulative-delta-q protocol differs")
    architecture = config["architecture"]
    if (
        architecture["state_action_encoder_widths"] != [256, 256]
        or int(architecture["increment_decoder_width"]) != 128
        or int(architecture["transition_count"]) != 50
        or int(architecture["joint_dimension"]) != 7
        or int(architecture["time_encoding_dimension"]) != 9
        or architecture["exact_initial_condition"]
        != "delta_q_0_equals_zero_and_q_hat_0_equals_measured_q_0"
    ):
        raise ValueError("cumulative-delta-q architecture differs")
    training = config["training"]
    if (
        training["ensemble_seeds"]
        != [20260861, 20260862, 20260863, 20260864, 20260865]
        or int(training["batch_size"]) != 512
        or int(training["epochs"]) != 300
        or int(training["patience"]) != 40
        or float(training["joint_huber_delta_rad"]) != 0.002
        or float(training["joint_increment_loss_weight"]) != 1.0
        or float(training["finite_difference_sensitivity_loss_weight"]) != 1.0
        or float(training["geometry_loss_weight"]) != 10.0
        or float(training["dangerous_overestimate_weight"]) != 4.0
    ):
        raise ValueError("cumulative-delta-q training differs")
    if not all(config["forbidden_actions"].values()):
        raise ValueError("cumulative-delta-q forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def transition_time_encoding() -> Any:
    """Return midpoint encodings for the 50 controller transitions."""

    np = _numpy()
    tau = (np.arange(50, dtype=np.float64) + 0.5) / 50.0
    parts = [tau]
    for frequency in (1.0, 2.0, 4.0, 8.0):
        parts.extend((
            np.sin(math.pi * frequency * tau),
            np.cos(math.pi * frequency * tau),
        ))
    output = np.stack(parts, axis=1)
    if output.shape != (50, 9):
        raise ValueError("cumulative-delta-q time encoding differs")
    return output


def _build_model(input_dimension: int, architecture: Mapping[str, Any]) -> Any:
    torch = _torch()
    nn = torch.nn
    widths = [int(value) for value in architecture["state_action_encoder_widths"]]
    decoder = int(architecture["increment_decoder_width"])
    encoding = torch.as_tensor(transition_time_encoding(), dtype=torch.float32)

    class CumulativeDeltaQNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(int(input_dimension), widths[0]), nn.SiLU(),
                nn.Linear(widths[0], widths[1]), nn.SiLU(),
            )
            self.increment_decoder = nn.Sequential(
                nn.Linear(widths[1] + 9, decoder), nn.SiLU(),
                nn.Linear(decoder, 7),
            )
            self.register_buffer("transition_time_encoding", encoding.clone())

        def forward(self, features: Any) -> Any:
            latent = self.encoder(features)
            expanded = latent[:, None, :].expand(-1, 50, -1)
            time = self.transition_time_encoding[None, :, :].expand(
                len(features), -1, -1
            )
            increments = self.increment_decoder(torch.cat((expanded, time), dim=2))
            residual = torch.cumsum(increments, dim=1)
            zero = torch.zeros(
                (len(features), 1, 7), dtype=residual.dtype,
                device=residual.device,
            )
            return torch.cat((zero, residual), dim=1)

    return CumulativeDeltaQNet()


def _huber(error: Any, beta: float) -> Any:
    torch = _torch()
    absolute = torch.abs(error)
    return torch.where(
        absolute <= beta, 0.5 * error ** 2,
        beta * (absolute - 0.5 * beta),
    )


def train_cumulative_delta_ensemble(
    arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
    geometry: Mapping[str, Any], config: Mapping[str, Any],
    normalization: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Train exact-q0 cumulative displacement with safety-relevant losses."""

    np = _numpy()
    torch = _torch()
    torch.set_num_threads(8)
    x = np.asarray(arrays["features"], dtype=np.float64)
    target, base, _ = _arm_targets(arrays, "factorized_execution")
    residual = target - base
    exact_increment = residual[:, 1:] - residual[:, :-1]
    split = np.asarray(arrays["split"], dtype=object)
    train_indexes = np.flatnonzero(split == "train")
    validation_indexes = np.flatnonzero(split == "validation")
    mean = np.asarray(normalization["feature_mean"], dtype=np.float64)
    std = np.asarray(normalization["feature_std"], dtype=np.float64)
    if mean.shape != (x.shape[1],) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("cumulative-delta-q normalization differs")
    normalized = (x - mean) / std
    state_index = np.asarray(arrays["state_index"], dtype=np.int64)
    clearance = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    jacobians = np.asarray(geometry["jacobian_m_per_rad"], dtype=np.float64)
    exact_sensitivity = np.asarray(
        sensitivities["joint_sensitivity_rad_per_action"], dtype=np.float64,
    )
    sensitivity_states = np.asarray(sensitivities["state_index"], dtype=np.int64)
    state_split = {int(state): str(split[row]) for row, state in enumerate(state_index)}
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
    joint_beta = float(settings["joint_huber_delta_rad"])
    geometry_beta = float(settings["geometry_huber_delta_m"])
    boundary = float(settings["near_boundary_absolute_margin_m"])
    late = 1.0 + (float(settings["terminal_weight_multiplier"]) - 1.0) * (
        np.arange(51, dtype=np.float64) / 50.0
    ) ** int(settings["terminal_weight_power"])
    element_weight = np.where(
        np.abs(clearance) <= boundary,
        float(settings["near_boundary_weight_multiplier"]), 1.0,
    ) * late[None, :, None]
    models = []
    states = []
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
                truth = torch.as_tensor(residual[indexes], dtype=torch.float32)
                q_error = prediction - truth
                q_loss = _huber(q_error[:, 1:], joint_beta).mean()
                predicted_increment = prediction[:, 1:] - prediction[:, :-1]
                increment_error = predicted_increment - torch.as_tensor(
                    exact_increment[indexes], dtype=torch.float32,
                )
                increment_loss = _huber(increment_error, joint_beta).mean()
                jacobian = torch.as_tensor(
                    jacobians[state_index[indexes]], dtype=torch.float32,
                )
                h_error = torch.einsum("bkrj,bkj->bkr", jacobian, q_error)
                weights = torch.as_tensor(
                    element_weight[indexes], dtype=torch.float32,
                )
                geometry_loss = torch.mean(
                    weights * _huber(h_error, geometry_beta)
                ) + float(settings["dangerous_overestimate_weight"]) * torch.mean(
                    weights * torch.relu(h_error) ** 2
                )
                loss = q_loss + float(
                    settings["joint_increment_loss_weight"]
                ) * increment_loss + float(
                    settings["geometry_loss_weight"]
                ) * geometry_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            selected = train_sensitivity
            model.train()
            negative = model(torch.as_tensor(
                normalized[neg_rows[selected]], dtype=torch.float32,
            ))
            positive = model(torch.as_tensor(
                normalized[pos_rows[selected]], dtype=torch.float32,
            ))
            predicted_s = (positive - negative) / torch.as_tensor(
                denominators[selected], dtype=torch.float32,
            )[:, None, None]
            exact_s = torch.as_tensor(
                exact_sensitivity[selected], dtype=torch.float32,
            )
            sensitivity_loss = torch.nn.functional.smooth_l1_loss(
                predicted_s, exact_s, beta=joint_beta,
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
                    normalized[neg_rows[selected]], dtype=torch.float32,
                ))
                positive = model(torch.as_tensor(
                    normalized[pos_rows[selected]], dtype=torch.float32,
                ))
                validation_s = (
                    (positive - negative).cpu().numpy()
                    / denominators[selected, None, None]
                )
            q_rmse = float(np.sqrt(np.mean(
                (validation_prediction - residual[validation_indexes]) ** 2
            )))
            sensitivity_rmse = float(np.sqrt(np.mean(
                (validation_s - exact_sensitivity[validation_sensitivity]) ** 2
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
            raise RuntimeError("cumulative-delta-q training produced no checkpoint")
        model.load_state_dict(best)
        model.eval()
        models.append(model)
        states.append({key: value.numpy() for key, value in best.items()})
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
        "model_states": states,
    }
    return models, state, {
        "member_audits": audits,
        "training_row_count": int(len(train_indexes)),
        "validation_row_count": int(len(validation_indexes)),
        "training_sensitivity_row_count": int(len(train_sensitivity)),
        "validation_sensitivity_row_count": int(len(validation_sensitivity)),
    }


def predict_cumulative_delta(
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


def save_cumulative_delta_weights(
    path: Path, state: Mapping[str, Any],
) -> dict[str, Any]:
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
    return {
        "path": str(Path(path).resolve()),
        "file_sha256": _sha256(Path(path).read_bytes()),
    }


def load_cumulative_delta_weights(
    path: Path,
) -> tuple[list[Any], dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    archive = np.load(path, allow_pickle=False)
    if str(archive["schema_version"].item()) != WEIGHTS_SCHEMA:
        raise ValueError("cumulative-delta-q weights schema differs")
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


def cumulative_delta_decision(
    *, metrics: Mapping[str, Any], validation_support: Mapping[str, Any],
    test_support: Mapping[str, Any], temporal: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["mechanism_decision_gate"]
    tests = {
        "zero_validation_false_safe": int(
            metrics["validation_safety"]["false_safe_action_count"]
        ) <= int(gate["maximum_validation_false_safe_action_count"]),
        "zero_diagnostic_test_false_safe": int(
            metrics["test_safety"]["false_safe_action_count"]
        ) <= int(gate["maximum_diagnostic_test_false_safe_action_count"]),
        "all_validation_eligible_states_supported": bool(
            validation_support["all_eligible_states_supported"]
        ),
        "all_diagnostic_test_eligible_states_supported": bool(
            test_support["all_eligible_states_supported"]
        ),
        "validation_safe_recall": float(
            metrics["validation_safety"]["exact_safe_action_recall"]
        ) >= float(gate["minimum_validation_safe_recall"]),
        "diagnostic_test_safe_recall": float(
            metrics["test_safety"]["exact_safe_action_recall"]
        ) >= float(gate["minimum_diagnostic_test_safe_recall"]),
        "diagnostic_test_boundary_RMSE": float(
            metrics["test_safety"]["near_boundary_RMSE_m"]
        ) <= float(gate["maximum_diagnostic_test_near_boundary_RMSE_m"]),
        "joint_sensitivity": float(
            metrics["joint_sensitivity"]["mean_cosine"]
        ) >= float(gate["minimum_joint_sensitivity_cosine"]),
        "margin_sensitivity": float(
            metrics["margin_sensitivity"]["mean_cosine"]
        ) >= float(gate["minimum_margin_sensitivity_cosine"]),
        "terminal_joint_RMSE": float(
            temporal["test"]["terminal_joint_RMSE_rad"]
        ) <= float(gate["maximum_diagnostic_test_terminal_joint_RMSE_rad"]),
        "exact_initial_condition": float(
            temporal["test"]["joint_RMSE_by_substep_rad"][0]
        ) <= 1.0e-12,
    }
    passed = bool(all(tests.values()))
    return {
        "gate_tests": tests, "mechanism_prediction_GO": passed,
        "reserved_episode_collection_authorized": passed,
        "QP_or_closed_loop_authorized": False,
        "conclusion": (
            "cumulative_delta_q_passes_mechanism_gate"
            if passed else "cumulative_delta_q_fails_mechanism_gate"
        ),
    }
