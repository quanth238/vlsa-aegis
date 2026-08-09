"""Direct signed two-step controller-rollout margin learning.

This opt-in diagnostic differs from the earlier execution-loss model in one
important way: the network directly regresses each signed minimum rollout
margin with a linear output.  The exact simulator remains the label and
verification authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .execution_margin_nn import _canonical, _numpy, _torch
from .two_step_margin import (
    CONSTRAINT_ORDER,
    PAIR_CANDIDATE_SLICE,
    PAIR_FEATURE_NAMES,
    _records_to_arrays,
    _sample_weights,
    feature_vectors,
)


DIRECT_MARGIN_SCHEMA = "vlsa_distal_direct_rollout_margin_e05.v1"
DIRECT_MARGIN_TRAINING_SCHEMA = (
    "vlsa_distal_direct_rollout_margin_e05_training.v1"
)
DIRECT_MARGIN_RESULT_SCHEMA = "vlsa_distal_direct_rollout_margin_e05_result.v1"
DIRECT_MARGIN_VALIDATION_SCHEMA = (
    "vlsa_distal_direct_rollout_margin_e05_validation.v1"
)
DIRECT_MARGIN_WEIGHTS_SCHEMA = "vlsa_distal_direct_rollout_margin_weights.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_direct_margin_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("direct-margin config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "case_id",
        "state_step", "immutable_sources", "split", "model", "training",
        "calibration", "matched_random", "projection", "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("direct-margin config keys differ")
    if (
        config["schema_version"] != DIRECT_MARGIN_SCHEMA
        or config["protocol_id"] != "vlsa-distal-direct-rollout-margin-e05-v1"
        or config["case_id"] != "vlsa-t1-goal-ii-t0-e05"
        or config["state_step"] != 185
    ):
        raise ValueError("direct-margin protocol identity differs")
    if config["split"] != {
        "unit": "complete_episode_and_task_level_group",
        "train_episode_count": 6,
        "validation_episode_count": 1,
        "test_episode_count": 3,
        "primary_e05_use": "test_only_never_training_or_calibration",
    }:
        raise ValueError("direct-margin grouped split differs")
    if config["model"] != {
        "input": "factorized_state_action_and_relative_geometry_features",
        "shared_across_constraint_rows": True,
        "hidden_widths": [128, 128],
        "hidden_activation": "softplus",
        "output_activation": "linear_signed_margin_mm",
        "output_count_per_constraint": 1,
        "target_clip_mm": 50.0,
    }:
        raise ValueError("direct-margin model differs")
    training = config["training"]
    if training != {
        "device": "cuda", "dtype": "float64",
        "batching": "deterministic_full_batch", "epochs": 2500,
        "patience": 250, "learning_rate": 0.001,
        "weight_decay": 1.0e-6, "huber_delta_mm": 1.0,
        "gradient_cosine_loss_weight": 0.25,
        "gradient_magnitude_loss_weight": 0.1,
        "gradient_magnitude_floor_mm_per_action": 1.0,
        "seed": 20260810,
        "sample_weighting": "equal_episode_then_equal_boundary_sign",
        "finite_difference_source": (
            "witness_stable_exact_cloned_OSC_two_step_pairs"
        ),
    }:
        raise ValueError("direct-margin training differs")
    if config["calibration"] != {
        "method": (
            "per_constraint_maximum_validation_overprediction_plus_fixed_padding"
        ),
        "fixed_padding_m": 0.001,
    }:
        raise ValueError("direct-margin calibration differs")
    if config["matched_random"] != {
        "radius_action": 0.1, "direction_count": 256, "seed": 20260810,
        "maximum_sampling_attempts": 100000,
        "empirical_equal_or_better_p_maximum": 0.05,
        "comparison": (
            "exact_change_in_minimum_seven_row_two_step_margin"
        ),
    }:
        raise ValueError("direct-margin matched-random protocol differs")
    if config["projection"] != {
        "action_limit": 1.0, "trust_region_linf_action": 0.5,
        "clearance_target_m": 0.0, "maximum_linearization_iterations": 4,
        "eps_abs": 1.0e-7, "eps_rel": 1.0e-7, "max_iter": 10000,
        "residual_tolerance": 5.0e-7,
        "bound_tolerance_action": 5.0e-8,
    }:
        raise ValueError("direct-margin projection differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def build_direct_margin_model(hidden_widths: Sequence[int]) -> Any:
    torch = _torch()
    nn = torch.nn

    class DirectMarginNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            previous = len(PAIR_FEATURE_NAMES)
            for width in hidden_widths:
                layers.extend((nn.Linear(previous, int(width)), nn.Softplus()))
                previous = int(width)
            layers.append(nn.Linear(previous, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, values: Any) -> Any:
            return self.network(values)

    return DirectMarginNet()


def direct_margins_and_gradients(
    model: Any, normalized_features: Any, candidate_scale: Any,
    *, create_graph: bool,
) -> tuple[Any, Any]:
    """Return signed margins in mm and d-margin/d-physical-action."""

    torch = _torch()
    values = normalized_features
    if not values.requires_grad:
        values = values.clone().detach().requires_grad_(True)
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    predicted = model(flat)[:, 0].reshape(shape[0], 7)
    derivative = torch.autograd.grad(
        predicted.sum(), values, create_graph=create_graph, retain_graph=True
    )[0]
    gradients = (
        derivative[:, :, PAIR_CANDIDATE_SLICE]
        / candidate_scale[None, None, :]
    )
    return predicted, gradients


def _gradient_losses(
    torch: Any, predicted: Any, target: Any, valid: Any, scale: float,
) -> tuple[Any, Any]:
    selected_predicted = predicted[valid]
    selected_target = target[valid]
    if selected_target.numel() == 0:
        zero = torch.zeros((), dtype=predicted.dtype, device=predicted.device)
        return zero, zero
    predicted_norm = torch.linalg.vector_norm(selected_predicted, dim=1)
    target_norm = torch.linalg.vector_norm(selected_target, dim=1)
    cosine = torch.sum(selected_predicted * selected_target, dim=1) / (
        predicted_norm * target_norm + 1.0e-12
    )
    cosine_loss = torch.mean(1.0 - cosine)
    magnitude_loss = torch.mean(
        ((selected_predicted - selected_target) / float(scale)) ** 2
    )
    return cosine_loss, magnitude_loss


def train_direct_margin_model(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    arrays = _records_to_arrays(records)
    masks = {
        name: arrays["split"] == name
        for name in ("train", "validation", "test")
    }
    expected_counts = config["split"]
    cases_by_split = {
        name: sorted(set(arrays["case_id"][mask].tolist()))
        for name, mask in masks.items()
    }
    for name, key in (
        ("train", "train_episode_count"),
        ("validation", "validation_episode_count"),
        ("test", "test_episode_count"),
    ):
        if len(cases_by_split[name]) != int(expected_counts[key]):
            raise ValueError("direct-margin episode split count differs")
    if config["case_id"] not in cases_by_split["test"]:
        raise ValueError("direct-margin E05 leaked outside test")

    features = arrays["pair_features"]
    train_features = features[masks["train"]]
    normalization = train_features.reshape(-1, train_features.shape[-1])
    mean = np.mean(normalization, axis=0)
    standard_deviation = np.maximum(np.std(normalization, axis=0), 1.0e-6)
    normalized = (features - mean) / standard_deviation
    target_unclipped_mm = arrays["minimum"] * 1000.0
    clip_mm = float(config["model"]["target_clip_mm"])
    target_mm = np.clip(target_unclipped_mm, -clip_mm, clip_mm)

    settings = config["training"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("direct-margin training requires CUDA on H100")
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    model = build_direct_margin_model(config["model"]["hidden_widths"]).to(
        device=device, dtype=torch.float64
    )
    feature_tensor = torch.as_tensor(
        normalized, dtype=torch.float64, device=device
    )
    target_tensor = torch.as_tensor(target_mm, dtype=torch.float64, device=device)
    raw_target_tensor = torch.as_tensor(
        target_unclipped_mm, dtype=torch.float64, device=device
    )
    gradient_target = torch.as_tensor(
        np.where(
            arrays["gradient_valid"][:, :, None], arrays["gradients"], 0.0
        ) * 1000.0,
        dtype=torch.float64, device=device,
    )
    gradient_valid = torch.as_tensor(
        arrays["gradient_valid"], dtype=torch.bool, device=device
    )
    mask_tensors = {
        name: torch.as_tensor(mask, dtype=torch.bool, device=device)
        for name, mask in masks.items()
    }
    train_weights = torch.as_tensor(
        _sample_weights(arrays, masks["train"]),
        dtype=torch.float64, device=device,
    )
    train_gradient_values = arrays["gradients"][masks["train"]] * 1000.0
    train_gradient_valid = arrays["gradient_valid"][masks["train"]]
    finite = np.repeat(train_gradient_valid[:, :, None], 3, axis=2)
    if not np.any(finite):
        raise ValueError("direct-margin training lacks gradient supervision")
    gradient_scale = max(
        float(settings["gradient_magnitude_floor_mm_per_action"]),
        float(np.sqrt(np.mean(train_gradient_values[finite] ** 2))),
    )
    candidate_scale = torch.as_tensor(
        standard_deviation[PAIR_CANDIDATE_SLICE],
        dtype=torch.float64, device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    huber = torch.nn.HuberLoss(
        reduction="none", delta=float(settings["huber_delta_mm"])
    )

    def loss_for(name: str, create_graph: bool) -> tuple[Any, Any, Any, Any]:
        selected = mask_tensors[name]
        values = feature_tensor[selected].clone().detach().requires_grad_(True)
        predicted, gradients = direct_margins_and_gradients(
            model, values, candidate_scale, create_graph=create_graph
        )
        per_record = huber(predicted, target_tensor[selected]).mean(dim=1)
        margin_loss = (
            torch.sum(per_record * train_weights)
            if name == "train" else torch.mean(per_record)
        )
        valid = gradient_valid[selected] & (
            torch.abs(raw_target_tensor[selected]) < clip_mm
        )
        cosine_loss, magnitude_loss = _gradient_losses(
            torch, gradients, gradient_target[selected], valid, gradient_scale
        )
        total = (
            margin_loss
            + float(settings["gradient_cosine_loss_weight"]) * cosine_loss
            + float(settings["gradient_magnitude_loss_weight"]) * magnitude_loss
        )
        return total, margin_loss, cosine_loss, magnitude_loss

    best_state = None
    best_epoch = None
    best_validation = math.inf
    stale = 0
    history = []
    started = time.perf_counter_ns()
    for epoch in range(int(settings["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = loss_for("train", True)
        train_losses[0].backward()
        optimizer.step()
        model.eval()
        with torch.enable_grad():
            validation_losses = loss_for("validation", False)
        value = float(validation_losses[0].detach().cpu())
        if epoch == 0 or (epoch + 1) % 100 == 0:
            history.append({
                "epoch": int(epoch),
                "train_total": float(train_losses[0].detach().cpu()),
                "train_margin": float(train_losses[1].detach().cpu()),
                "train_gradient_cosine": float(train_losses[2].detach().cpu()),
                "train_gradient_magnitude": float(train_losses[3].detach().cpu()),
                "validation_total": value,
                "validation_margin": float(validation_losses[1].detach().cpu()),
                "validation_gradient_cosine": float(validation_losses[2].detach().cpu()),
                "validation_gradient_magnitude": float(validation_losses[3].detach().cpu()),
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
        raise RuntimeError("direct-margin training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.enable_grad():
        predicted_mm, predicted_gradient_mm = direct_margins_and_gradients(
            model, feature_tensor.clone().detach().requires_grad_(True),
            candidate_scale, create_graph=False,
        )
    predicted_margin = predicted_mm.detach().cpu().numpy() / 1000.0
    predicted_gradient = predicted_gradient_mm.detach().cpu().numpy() / 1000.0
    validation_error = (
        predicted_margin[masks["validation"]]
        - arrays["minimum"][masks["validation"]]
    )
    calibration = np.maximum(0.0, np.max(validation_error, axis=0)) + float(
        config["calibration"]["fixed_padding_m"]
    )
    conservative = predicted_margin - calibration[None, :]

    def episode_metrics(case_id: str) -> dict[str, Any]:
        mask = arrays["case_id"] == case_id
        actual = arrays["minimum"][mask]
        predicted = predicted_margin[mask]
        lower = conservative[mask]
        baseline = arrays["current"][mask]
        categories = arrays["category"][mask]
        active = np.argmin(actual, axis=1)
        indexes = np.arange(len(actual))
        boundary = np.isin(categories, ["boundary_safe", "boundary_unsafe"])
        if not np.any(boundary):
            raise ValueError("direct-margin episode lacks boundary records")
        selected_indexes = indexes[boundary]
        selected_rows = active[boundary]
        error = (
            predicted[selected_indexes, selected_rows]
            - actual[selected_indexes, selected_rows]
        )
        baseline_error = (
            baseline[selected_indexes, selected_rows]
            - actual[selected_indexes, selected_rows]
        )
        false_safe = np.logical_and(lower >= 0.0, actual < 0.0)
        true_gradients = arrays["gradients"][mask]
        valid_gradients = arrays["gradient_valid"][mask]
        learned_gradients = predicted_gradient[mask]
        cosines = []
        magnitude_relative_errors = []
        for local_index in selected_indexes:
            row = int(active[local_index])
            if not bool(valid_gradients[local_index, row]):
                continue
            true = true_gradients[local_index, row]
            learned = learned_gradients[local_index, row]
            true_norm = float(np.linalg.norm(true))
            learned_norm = float(np.linalg.norm(learned))
            if true_norm > 1.0e-12 and learned_norm > 1.0e-12:
                cosines.append(float(np.dot(true, learned) / (true_norm * learned_norm)))
                magnitude_relative_errors.append(abs(learned_norm - true_norm) / true_norm)
        return {
            "record_count": int(np.count_nonzero(mask)),
            "active_boundary_record_count": int(np.count_nonzero(boundary)),
            "active_boundary_rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "active_boundary_current_clearance_baseline_rmse_m": float(
                np.sqrt(np.mean(baseline_error ** 2))
            ),
            "conservative_false_safe_candidate_count": int(
                np.count_nonzero(np.any(false_safe, axis=1))
            ),
            "active_gradient_cosine_count": len(cosines),
            "active_gradient_cosine_mean": None if not cosines else float(np.mean(cosines)),
            "active_gradient_cosine_minimum": None if not cosines else float(np.min(cosines)),
            "active_gradient_magnitude_relative_error_mean": (
                None if not magnitude_relative_errors
                else float(np.mean(magnitude_relative_errors))
            ),
        }

    per_episode = {
        case: episode_metrics(case)
        for case in sorted(set(arrays["case_id"].tolist()))
    }
    e05 = per_episode[config["case_id"]]
    e05_model_gate = bool(
        e05["active_boundary_rmse_m"]
        < e05["active_boundary_current_clearance_baseline_rmse_m"]
        and e05["active_gradient_cosine_count"] > 0
        and e05["active_gradient_cosine_mean"]
        >= float(config["decision_gate"][
            "e05_active_gradient_cosine_mean_minimum"
        ])
    )
    test_false_safe = sum(
        per_episode[case]["conservative_false_safe_candidate_count"]
        for case in cases_by_split["test"]
    )
    state = {
        "state_dict": best_state,
        "feature_mean": mean,
        "feature_standard_deviation": standard_deviation,
        "calibration_m": calibration,
        "hidden_widths": list(config["model"]["hidden_widths"]),
    }
    audit = {
        "model_class": "shared_constraint_direct_signed_two_step_margin",
        "device": str(device), "torch_version": str(torch.__version__),
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": best_epoch, "completed_epoch_count": int(epoch + 1),
        "best_validation_objective": best_validation,
        "gradient_scale_mm_per_action": gradient_scale,
        "training_valid_gradient_row_count": int(
            np.count_nonzero(train_gradient_valid)
        ),
        "history": history, "calibration_m": calibration.tolist(),
        "cases_by_split": cases_by_split, "per_episode_metrics": per_episode,
        "e05_model_gate_pass": e05_model_gate,
        "test_conservative_false_safe_candidate_count": int(test_false_safe),
        "test_false_safe_gate_pass": bool(
            test_false_safe
            == int(config["decision_gate"][
                "all_test_conservative_false_safe_count"
            ])
        ),
        "training_wall_seconds": (
            time.perf_counter_ns() - started
        ) * 1.0e-9,
    }
    return model, state, audit


def save_direct_margin_model(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    names = list(state["state_dict"])
    metadata = {
        "schema_version": DIRECT_MARGIN_WEIGHTS_SCHEMA,
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


def load_direct_margin_model(
    path: Path, *, device: str = "cpu"
) -> tuple[Any, dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(
            bytes(archive["metadata_utf8"].tolist()).decode("utf-8")
        )
        if (
            metadata.get("schema_version") != DIRECT_MARGIN_WEIGHTS_SCHEMA
            or metadata.get("feature_names") != list(PAIR_FEATURE_NAMES)
            or metadata.get("constraint_order") != list(CONSTRAINT_ORDER)
        ):
            raise ValueError("direct-margin model artifact identity differs")
        model = build_direct_margin_model(metadata["hidden_widths"]).to(
            device=torch.device(device), dtype=torch.float64
        )
        names = metadata.get("parameter_names")
        if names != list(model.state_dict()):
            raise ValueError("direct-margin parameter names differ")
        state_dict = {}
        for index, name in enumerate(names):
            values = np.asarray(archive["parameter_%03d" % index], dtype=np.float64)
            if values.shape != tuple(model.state_dict()[name].shape):
                raise ValueError("direct-margin parameter shape differs")
            state_dict[name] = torch.as_tensor(
                values, dtype=torch.float64, device=device
            )
        model.load_state_dict(state_dict)
        model.eval()
        state = {
            "feature_mean": np.asarray(archive["feature_mean"], dtype=np.float64),
            "feature_standard_deviation": np.asarray(
                archive["feature_standard_deviation"], dtype=np.float64
            ),
            "calibration_m": np.asarray(archive["calibration_m"], dtype=np.float64),
            "hidden_widths": list(metadata["hidden_widths"]),
        }
    return model, state


def predict_direct_margins_and_jacobian(
    model: Any, model_state: Mapping[str, Any], context: Mapping[str, Any],
    nominal_first_xyz: Sequence[float], candidate_first_xyz: Sequence[float],
    nominal_second_xyz: Sequence[float],
) -> tuple[Any, Any, float]:
    np = _numpy()
    torch = _torch()
    _, pair_features = feature_vectors(
        context, nominal_first_xyz, candidate_first_xyz, nominal_second_xyz
    )
    mean = np.asarray(model_state["feature_mean"], dtype=np.float64)
    standard_deviation = np.asarray(
        model_state["feature_standard_deviation"], dtype=np.float64
    )
    normalized = (pair_features - mean) / standard_deviation
    tensor = torch.as_tensor(
        normalized, dtype=torch.float64, device=next(model.parameters()).device
    )[None, ...].clone().detach().requires_grad_(True)
    candidate_scale = torch.as_tensor(
        standard_deviation[PAIR_CANDIDATE_SLICE],
        dtype=torch.float64, device=tensor.device,
    )
    started = time.perf_counter_ns()
    margins_mm, gradients_mm = direct_margins_and_gradients(
        model, tensor, candidate_scale, create_graph=False
    )
    return (
        margins_mm[0].detach().cpu().numpy() / 1000.0,
        gradients_mm[0].detach().cpu().numpy() / 1000.0,
        (time.perf_counter_ns() - started) * 1.0e-9,
    )


def project_direct_margin_action(
    model: Any, model_state: Mapping[str, Any], context: Mapping[str, Any],
    nominal_first_xyz: Sequence[float], nominal_second_xyz: Sequence[float],
    config: Mapping[str, Any], *, calibrated: bool,
) -> dict[str, Any]:
    np = _numpy()
    from .qp import MultiConstraintQp

    settings = config["projection"]
    nominal = np.asarray(nominal_first_xyz, dtype=np.float64)
    limit = float(settings["action_limit"])
    trust = float(settings["trust_region_linf_action"])
    lower_action = np.maximum(-limit, nominal - trust)
    upper_action = np.minimum(limit, nominal + trust)
    calibration = (
        np.asarray(model_state["calibration_m"], dtype=np.float64)
        if calibrated else np.zeros(7, dtype=np.float64)
    )
    solver = MultiConstraintQp(
        eps_abs=float(settings["eps_abs"]), eps_rel=float(settings["eps_rel"]),
        max_iter=int(settings["max_iter"]),
        residual_tolerance=float(settings["residual_tolerance"]),
        bound_tolerance=float(settings["bound_tolerance_action"]),
    )
    current = nominal.copy()
    iterations = []
    valid = False
    reason = "maximum_linearization_iterations_reached"
    started = time.perf_counter_ns()
    for iteration in range(int(settings["maximum_linearization_iterations"])):
        margin, jacobian, inference = predict_direct_margins_and_jacobian(
            model, model_state, context, nominal, current, nominal_second_xyz
        )
        lower_margin = margin - calibration
        if np.all(lower_margin >= float(settings["clearance_target_m"])):
            valid = True
            reason = "predicted_safe"
            iterations.append({
                "iteration": iteration, "linearization_xyz": current.tolist(),
                "predicted_margin_m": margin.tolist(),
                "lower_margin_m": lower_margin.tolist(),
                "jacobian_m_per_action": jacobian.tolist(),
                "inference_and_jacobian_wall_seconds": inference, "qp": None,
            })
            break
        row_lower = (
            float(settings["clearance_target_m"]) - lower_margin
            + jacobian @ current
        )
        qp = solver.solve(
            nominal, np.eye(3), jacobian, row_lower,
            lower_action, upper_action,
        )
        iterations.append({
            "iteration": iteration, "linearization_xyz": current.tolist(),
            "predicted_margin_m": margin.tolist(),
            "lower_margin_m": lower_margin.tolist(),
            "jacobian_m_per_action": jacobian.tolist(),
            "inference_and_jacobian_wall_seconds": inference,
            "qp": {
                "valid": bool(qp.valid), "reason": qp.reason,
                "solution_xyz": None if qp.qdot_safe is None else qp.qdot_safe.tolist(),
                "lower": row_lower.tolist(), "diagnostics": dict(qp.diagnostics),
            },
        })
        if not qp.valid or qp.qdot_safe is None:
            reason = "direct_margin_qp_%s" % qp.reason
            break
        next_value = np.asarray(qp.qdot_safe, dtype=np.float64)
        if np.max(np.abs(next_value - current)) <= 1.0e-10:
            current = next_value
            reason = "direct_margin_projection_stalled"
            break
        current = next_value
    final_margin, final_jacobian, final_inference = (
        predict_direct_margins_and_jacobian(
            model, model_state, context, nominal, current, nominal_second_xyz
        )
    )
    final_lower = final_margin - calibration
    if np.all(final_lower >= float(settings["clearance_target_m"])):
        valid = True
        reason = "predicted_safe"
    return {
        "valid": valid, "reason": reason, "calibrated": bool(calibrated),
        "nominal_xyz": nominal.tolist(), "projected_xyz": current.tolist(),
        "correction_l2": float(np.linalg.norm(current - nominal)),
        "action_lower": lower_action.tolist(), "action_upper": upper_action.tolist(),
        "calibration_m": calibration.tolist(),
        "final_predicted_margin_m": final_margin.tolist(),
        "final_lower_margin_m": final_lower.tolist(),
        "final_jacobian_m_per_action": final_jacobian.tolist(),
        "final_inference_and_jacobian_wall_seconds": final_inference,
        "iterations": iterations,
        "total_wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
