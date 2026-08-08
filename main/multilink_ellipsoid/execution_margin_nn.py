"""Execution-aware residual network for the paired E05 distal-link audit.

The model is intentionally opt-in and small.  It predicts the nonnegative
clearance loss accumulated during one complete OSC ``env.step``.  Subtracting
that loss from the measured interval-start clearance enforces the physical
identity that a minimum over an interval cannot exceed its first sample.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


EXECUTION_MARGIN_NN_SCHEMA = "vlsa_distal_execution_margin_nn_e05.v1"
EXECUTION_MARGIN_RESULT_SCHEMA = "vlsa_distal_execution_margin_nn_e05_result.v1"
CONSTRAINT_ORDER = (
    "L5_part_0",
    "L5_part_1",
    "L5_part_2",
    "L6_part_0",
    "L6_part_1",
    "L7_part_0",
    "L7_part_1",
)
FEATURE_NAMES = tuple(
    ["q_rad_%d" % index for index in range(7)]
    + ["qdot_rad_s_%d" % index for index in range(7)]
    + ["osc_goal_position_m_%d" % index for index in range(3)]
    + ["obstacle_position_m_%d" % index for index in range(3)]
    + ["current_clearance_m_%d" % index for index in range(7)]
    + ["nominal_xyz_%d" % index for index in range(3)]
    + ["candidate_xyz_%d" % index for index in range(3)]
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("execution-margin model requires NumPy") from error
    return np


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("execution-margin model requires PyTorch") from error
    return torch


def load_execution_margin_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution-margin config is invalid JSON") from error
    required = {
        "schema_version",
        "protocol_id",
        "case_ids",
        "claim_scope",
        "immutable_sources",
        "state_groups",
        "candidate_set",
        "features",
        "labels",
        "network",
        "training",
        "calibration",
        "projection",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("execution-margin config keys differ")
    if config["schema_version"] != EXECUTION_MARGIN_NN_SCHEMA:
        raise ValueError("execution-margin schema differs")
    if config["protocol_id"] != "vlsa-distal-execution-margin-nn-e05-v1":
        raise ValueError("execution-margin protocol differs")
    if config["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("execution-margin protocol must select only E05")
    if config["immutable_sources"] != {
        "archived_table1_file_sha256": "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b",
        "archived_table1_payload_sha256": "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c",
        "exact_box_config_file_sha256": "cc568a85c2acf215beda1cef4abcc31a92b3f6d3772d6c36147410afd471bf9f",
        "exact_box_config_payload_sha256": "d3e0ab883eb3b3de160417fc9547012db9154dae7015ff80bc739219b728013b",
        "exact_box_discovery_file_sha256": "3db37092b2c5573e70cfc604bbbf01f1361be822ec87c8733d2c80b84685b0fd",
        "exact_box_discovery_payload_sha256": "2eef9d767354f6692d36f57899bd511c261733e05394d00c997eacc7252fc051",
        "false_safe_action_ledger_file_sha256": "d79a28585e74cece727fbc3d3e6a72eb1647a782cae4962ec3af9045f448e603",
        "false_safe_action_ledger_payload_sha256": "2a4ecd5ff79a363141e0927d1dfbcfb26248ec9bb5e2bc123cc8f879c3395a53",
        "geometry_config_file_sha256": "fd9042ebc7605c68e71d1ab51bfdc8b4412fcb9dda11f1879d85271641436da1",
    }:
        raise ValueError("execution-margin immutable source identities differ")
    groups = config["state_groups"]
    expected_steps = list(range(180, 193))
    if (
        set(groups) != {"collect_steps", "train_steps", "validation_steps", "test_steps", "primary_projection_step"}
        or groups["collect_steps"] != expected_steps
        or groups["train_steps"] != [180, 181, 182, 183, 184, 185, 186, 187]
        or groups["validation_steps"] != [188, 189]
        or groups["test_steps"] != [190, 191, 192]
        or groups["primary_projection_step"] != 191
    ):
        raise ValueError("execution-margin state groups differ")
    split = groups["train_steps"] + groups["validation_steps"] + groups["test_steps"]
    if sorted(split) != expected_steps or len(set(split)) != len(split):
        raise ValueError("execution-margin state groups must partition collection")
    if config["candidate_set"] != {
        "action_dimensions": [0, 1, 2],
        "action_limit": 1.0,
        "central_difference_action": 0.1,
        "global_lattice_values": [-1.0, 0.0, 1.0],
        "include_global_lattice": True,
        "include_reverse_nominal": True,
        "include_stop": True,
        "local_offset_magnitudes": [0.25, 0.5],
        "expected_candidate_count_per_state": 87,
    }:
        raise ValueError("execution-margin candidate set differs")
    if config["features"] != {
        "names": list(FEATURE_NAMES),
        "normalization": "train_group_mean_and_standard_deviation_with_1e-6_floor",
    }:
        raise ValueError("execution-margin features differ")
    if config["labels"] != {
        "D_opt": "seven_minimum_exact_box_support_gaps_over_interval_start_and_every_internal_mujoco_step",
        "D_sim": "raw_nonpositive_L5_L6_L7_contact_and_maximum_within_step_obstacle_l1_displacement",
        "residual_target": "nonnegative_current_clearance_minus_minimum_substep_clearance",
        "distinct_D_opt_and_D_sim": True,
    }:
        raise ValueError("execution-margin labels differ")
    if config["network"] != {
        "hidden_widths": [128, 128],
        "hidden_activation": "silu",
        "output_activation": "softplus_nonnegative_clearance_loss_mm",
        "output_count": 7,
    }:
        raise ValueError("execution-margin network differs")
    if config["training"] != {
        "batching": "deterministic_full_batch",
        "device": "cuda",
        "epochs": 3000,
        "learning_rate": 0.001,
        "patience": 300,
        "seed": 20260809,
        "weight_decay": 1e-06,
    }:
        raise ValueError("execution-margin training differs")
    if config["calibration"] != {
        "method": "per_constraint_maximum_validation_overprediction_plus_fixed_padding",
        "fixed_padding_m": 0.001,
    }:
        raise ValueError("execution-margin calibration differs")
    projection = config["projection"]
    if projection != {
        "action_limit": 1.0,
        "activation_warning_m": 0.008,
        "bound_tolerance_action": 5e-08,
        "clearance_target_m": 0.0,
        "eps_abs": 1e-07,
        "eps_rel": 1e-07,
        "max_iter": 10000,
        "maximum_linearization_iterations": 3,
        "residual_tolerance": 5e-07,
        "trust_region_linf_action": 0.5,
    }:
        raise ValueError("execution-margin projection differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def feature_vector(
    start_substep: Mapping[str, Any],
    nominal_xyz: Sequence[float],
    candidate_xyz: Sequence[float],
) -> Any:
    np = _numpy()
    goal = start_substep.get("controller_goal_position_m")
    if goal is None:
        raise ValueError("OSC goal position is unavailable at dataset state")
    pieces = (
        start_substep.get("robot_joint_position_rad"),
        start_substep.get("robot_joint_velocity_rad_s"),
        goal,
        start_substep.get("obstacle_position_m"),
        start_substep.get("clearance_m", [])[:7],
        nominal_xyz,
        candidate_xyz,
    )
    output = np.concatenate(
        [np.asarray(item, dtype=np.float64) for item in pieces], axis=0
    )
    if output.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(output)):
        raise ValueError("execution-margin feature vector is invalid")
    return output


def _build_model(input_count: int, hidden_widths: Sequence[int], output_count: int) -> Any:
    torch = _torch()
    nn = torch.nn

    class ExecutionLossNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            previous = int(input_count)
            for width in hidden_widths:
                layers.extend((nn.Linear(previous, int(width)), nn.SiLU()))
                previous = int(width)
            layers.append(nn.Linear(previous, int(output_count)))
            self.network = nn.Sequential(*layers)
            self.softplus = nn.Softplus()

        def forward(self, values: Any) -> Any:
            return self.softplus(self.network(values))

    return ExecutionLossNet()


def _records_to_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    features = np.asarray([item["feature_vector"] for item in records], dtype=np.float64)
    current = np.asarray([item["current_clearance_m"] for item in records], dtype=np.float64)
    minimum = np.asarray([item["minimum_substep_clearance_m"] for item in records], dtype=np.float64)
    steps = np.asarray([item["state_step"] for item in records], dtype=np.int64)
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or current.shape != (features.shape[0], 7)
        or minimum.shape != current.shape
        or steps.shape != (features.shape[0],)
        or not all(np.all(np.isfinite(value)) for value in (features, current, minimum))
    ):
        raise ValueError("execution-margin dataset arrays are invalid")
    loss_mm = (current - minimum) * 1000.0
    if float(np.min(loss_mm)) < -1.0e-7:
        raise ValueError("minimum-substep clearance exceeds interval-start clearance")
    return {
        "features": features,
        "current": current,
        "minimum": minimum,
        "loss_mm": np.maximum(loss_mm, 0.0),
        "steps": steps,
    }


def _split_mask(steps: Any, selected: Sequence[int]) -> Any:
    np = _numpy()
    return np.isin(steps, np.asarray(selected, dtype=np.int64))


def train_execution_margin_model(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Train deterministically and return model, serializable state, and audit."""

    np = _numpy()
    torch = _torch()
    arrays = _records_to_arrays(records)
    groups = config["state_groups"]
    masks = {
        "train": _split_mask(arrays["steps"], groups["train_steps"]),
        "validation": _split_mask(arrays["steps"], groups["validation_steps"]),
        "test": _split_mask(arrays["steps"], groups["test_steps"]),
    }
    if any(int(np.count_nonzero(mask)) == 0 for mask in masks.values()):
        raise ValueError("execution-margin split is empty")
    if not np.all(sum(mask.astype(np.int64) for mask in masks.values()) == 1):
        raise ValueError("execution-margin records do not have one split")
    train_features = arrays["features"][masks["train"]]
    mean = np.mean(train_features, axis=0)
    standard_deviation = np.std(train_features, axis=0)
    standard_deviation = np.maximum(standard_deviation, 1.0e-6)
    normalized = (arrays["features"] - mean) / standard_deviation

    settings = config["training"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("execution-margin training requires CUDA on H100")
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    model = _build_model(
        normalized.shape[1],
        config["network"]["hidden_widths"],
        config["network"]["output_count"],
    ).to(device=device, dtype=torch.float64)
    feature_tensor = torch.as_tensor(normalized, dtype=torch.float64, device=device)
    target_tensor = torch.as_tensor(arrays["loss_mm"], dtype=torch.float64, device=device)
    mask_tensors = {
        name: torch.as_tensor(mask, dtype=torch.bool, device=device)
        for name, mask in masks.items()
    }
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_state = None
    best_epoch = None
    best_validation = math.inf
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter_ns()
    for epoch in range(int(settings["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted = model(feature_tensor[mask_tensors["train"]])
        train_loss = torch.mean(
            (predicted - target_tensor[mask_tensors["train"]]) ** 2
        )
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(feature_tensor[mask_tensors["validation"]])
            validation_loss = torch.mean(
                (validation_prediction - target_tensor[mask_tensors["validation"]]) ** 2
            )
        value = float(validation_loss.detach().cpu())
        if epoch == 0 or (epoch + 1) % 100 == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "train_mse_mm2": float(train_loss.detach().cpu()),
                    "validation_mse_mm2": value,
                }
            )
        if value < best_validation - 1.0e-12:
            best_validation = value
            best_epoch = int(epoch)
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(settings["patience"]):
            break
    if best_state is None or best_epoch is None:
        raise RuntimeError("execution-margin training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted_loss_mm = model(feature_tensor).detach().cpu().numpy()
    predicted_margin = arrays["current"] - predicted_loss_mm / 1000.0
    validation_overprediction = (
        predicted_margin[masks["validation"]] - arrays["minimum"][masks["validation"]]
    )
    calibration = np.maximum(0.0, np.max(validation_overprediction, axis=0)) + float(
        config["calibration"]["fixed_padding_m"]
    )
    lower_margin = predicted_margin - calibration[None, :]

    def metrics(mask: Any) -> dict[str, Any]:
        actual = arrays["minimum"][mask]
        predicted = predicted_margin[mask]
        conservative = lower_margin[mask]
        baseline = arrays["current"][mask]
        error = predicted - actual
        baseline_error = baseline - actual
        false_safe = np.logical_and(conservative >= 0.0, actual < 0.0)
        return {
            "sample_count": int(np.count_nonzero(mask)),
            "state_steps": sorted({int(value) for value in arrays["steps"][mask]}),
            "rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "mae_m": float(np.mean(np.abs(error))),
            "maximum_absolute_error_m": float(np.max(np.abs(error))),
            "current_clearance_baseline_rmse_m": float(
                np.sqrt(np.mean(baseline_error ** 2))
            ),
            "conservative_false_safe_element_count": int(np.count_nonzero(false_safe)),
            "conservative_false_safe_candidate_count": int(
                np.count_nonzero(np.any(false_safe, axis=1))
            ),
            "unsafe_element_count": int(np.count_nonzero(actual < 0.0)),
            "minimum_actual_margin_m": float(np.min(actual)),
            "minimum_conservative_margin_m": float(np.min(conservative)),
        }

    split_metrics = {name: metrics(mask) for name, mask in masks.items()}
    test_gate = bool(
        split_metrics["test"]["conservative_false_safe_candidate_count"] == 0
        and split_metrics["test"]["rmse_m"]
        < split_metrics["test"]["current_clearance_baseline_rmse_m"]
    )
    serializable = {
        "state_dict": best_state,
        "feature_mean": mean,
        "feature_standard_deviation": standard_deviation,
        "calibration_m": calibration,
        "feature_names": list(FEATURE_NAMES),
        "constraint_order": list(CONSTRAINT_ORDER),
        "hidden_widths": list(config["network"]["hidden_widths"]),
    }
    audit = {
        "model_class": "current_exact_clearance_minus_softplus_execution_loss",
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": best_epoch,
        "completed_epoch_count": int(epoch + 1),
        "best_validation_mse_mm2": best_validation,
        "history": history,
        "calibration_m": calibration.tolist(),
        "split_metrics": split_metrics,
        "model_learnability_gate_pass": test_gate,
        "training_wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    return model, serializable, audit


def save_model_artifact(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    parameter_names = list(state["state_dict"])
    metadata = {
        "schema_version": "vlsa_distal_execution_margin_nn_weights.v1",
        "feature_names": list(state["feature_names"]),
        "constraint_order": list(state["constraint_order"]),
        "hidden_widths": list(state["hidden_widths"]),
        "parameter_names": parameter_names,
    }
    arrays = {
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float64),
        "feature_standard_deviation": np.asarray(
            state["feature_standard_deviation"], dtype=np.float64
        ),
        "calibration_m": np.asarray(state["calibration_m"], dtype=np.float64),
        "metadata_utf8": np.frombuffer(_canonical(metadata), dtype=np.uint8),
    }
    for index, name in enumerate(parameter_names):
        arrays["parameter_%03d" % index] = (
            state["state_dict"][name].detach().cpu().numpy().astype(np.float64)
        )
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    raw = path.read_bytes()
    return {"path": str(path), "file_sha256": _sha256(raw), "size_bytes": len(raw)}


def load_model_artifact(path: Path, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["metadata_utf8"].tolist()).decode("utf-8"))
        if metadata.get("schema_version") != "vlsa_distal_execution_margin_nn_weights.v1":
            raise ValueError("execution-margin model artifact schema differs")
        if metadata.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("execution-margin model feature identity differs")
        if metadata.get("constraint_order") != list(CONSTRAINT_ORDER):
            raise ValueError("execution-margin model constraint identity differs")
        model = _build_model(
            len(FEATURE_NAMES), metadata["hidden_widths"], len(CONSTRAINT_ORDER)
        ).to(device=torch.device(device), dtype=torch.float64)
        parameter_names = metadata.get("parameter_names")
        expected_names = list(model.state_dict())
        if parameter_names != expected_names:
            raise ValueError("execution-margin model parameter names differ")
        state_dict = {}
        for index, name in enumerate(parameter_names):
            values = np.asarray(archive["parameter_%03d" % index], dtype=np.float64)
            expected_shape = tuple(model.state_dict()[name].shape)
            if values.shape != expected_shape or not np.all(np.isfinite(values)):
                raise ValueError("execution-margin model parameter is invalid")
            state_dict[name] = torch.as_tensor(values, dtype=torch.float64, device=device)
        model.load_state_dict(state_dict)
        model.eval()
        state = {
            "feature_mean": np.asarray(archive["feature_mean"], dtype=np.float64),
            "feature_standard_deviation": np.asarray(
                archive["feature_standard_deviation"], dtype=np.float64
            ),
            "calibration_m": np.asarray(archive["calibration_m"], dtype=np.float64),
            "feature_names": list(metadata["feature_names"]),
            "constraint_order": list(metadata["constraint_order"]),
            "hidden_widths": list(metadata["hidden_widths"]),
        }
    return model, state


def predict_margin_and_jacobian(
    model: Any,
    model_state: Mapping[str, Any],
    start_substep: Mapping[str, Any],
    nominal_xyz: Sequence[float],
    candidate_xyz: Sequence[float],
) -> tuple[Any, Any, float]:
    """Return seven margins and their Jacobian with respect to candidate XYZ."""

    np = _numpy()
    torch = _torch()
    device = next(model.parameters()).device
    feature = feature_vector(start_substep, nominal_xyz, candidate_xyz)
    mean = np.asarray(model_state["feature_mean"], dtype=np.float64)
    standard_deviation = np.asarray(
        model_state["feature_standard_deviation"], dtype=np.float64
    )
    normalized = (feature - mean) / standard_deviation
    tensor = torch.as_tensor(
        normalized, dtype=torch.float64, device=device
    ).clone().detach().requires_grad_(True)
    started = time.perf_counter_ns()
    predicted_loss_mm = model(tensor[None, :])[0]
    current = torch.as_tensor(
        np.asarray(start_substep["clearance_m"][:7], dtype=np.float64),
        dtype=torch.float64,
        device=device,
    )
    margin = current - predicted_loss_mm / 1000.0
    rows = []
    for index in range(7):
        gradient = torch.autograd.grad(
            margin[index], tensor, retain_graph=index < 6
        )[0]
        rows.append(gradient[-3:])
    elapsed = (time.perf_counter_ns() - started) * 1.0e-9
    return (
        margin.detach().cpu().numpy(),
        torch.stack(rows, dim=0).detach().cpu().numpy()
        / standard_deviation[-3:][None, :],
        elapsed,
    )


def project_action_with_model(
    model: Any,
    model_state: Mapping[str, Any],
    start_substep: Mapping[str, Any],
    nominal_xyz: Sequence[float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    from .qp import MultiConstraintQp

    settings = config["projection"]
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("neural projection nominal XYZ is invalid")
    limit = float(settings["action_limit"])
    trust = float(settings["trust_region_linf_action"])
    calibration = np.asarray(model_state["calibration_m"], dtype=np.float64)
    lower_action = np.maximum(-limit, nominal - trust)
    upper_action = np.minimum(limit, nominal + trust)
    solver = MultiConstraintQp(
        eps_abs=float(settings["eps_abs"]),
        eps_rel=float(settings["eps_rel"]),
        max_iter=int(settings["max_iter"]),
        residual_tolerance=float(settings["residual_tolerance"]),
        bound_tolerance=float(settings["bound_tolerance_action"]),
    )
    current = nominal.copy()
    iterations = []
    total_started = time.perf_counter_ns()
    valid = False
    reason = "maximum_linearization_iterations_reached"
    for iteration in range(int(settings["maximum_linearization_iterations"])):
        margin, jacobian, inference_seconds = predict_margin_and_jacobian(
            model, model_state, start_substep, nominal, current
        )
        conservative = margin - calibration
        if np.all(conservative >= float(settings["clearance_target_m"])):
            valid = True
            reason = "predicted_conservative_safe"
            iterations.append(
                {
                    "iteration": iteration,
                    "linearization_xyz": current.tolist(),
                    "predicted_margin_m": margin.tolist(),
                    "conservative_margin_m": conservative.tolist(),
                    "jacobian_m_per_action": jacobian.tolist(),
                    "inference_and_jacobian_wall_seconds": inference_seconds,
                    "qp": None,
                }
            )
            break
        lower = (
            float(settings["clearance_target_m"])
            - conservative
            + jacobian @ current
        )
        qp = solver.solve(
            nominal,
            np.eye(3),
            jacobian,
            lower,
            lower_action,
            upper_action,
        )
        iterations.append(
            {
                "iteration": iteration,
                "linearization_xyz": current.tolist(),
                "predicted_margin_m": margin.tolist(),
                "conservative_margin_m": conservative.tolist(),
                "jacobian_m_per_action": jacobian.tolist(),
                "inference_and_jacobian_wall_seconds": inference_seconds,
                "qp": {
                    "valid": bool(qp.valid),
                    "reason": qp.reason,
                    "solution_xyz": (
                        None if qp.qdot_safe is None else qp.qdot_safe.tolist()
                    ),
                    "lower": lower.tolist(),
                    "diagnostics": dict(qp.diagnostics),
                },
            }
        )
        if not qp.valid or qp.qdot_safe is None:
            reason = "neural_linearized_qp_%s" % qp.reason
            break
        next_value = np.asarray(qp.qdot_safe, dtype=np.float64)
        if np.max(np.abs(next_value - current)) <= 1.0e-10:
            reason = "neural_projection_stalled"
            current = next_value
            break
        current = next_value
    final_margin, final_jacobian, final_inference = predict_margin_and_jacobian(
        model, model_state, start_substep, nominal, current
    )
    final_conservative = final_margin - calibration
    if np.all(final_conservative >= float(settings["clearance_target_m"])):
        valid = True
        reason = "predicted_conservative_safe"
    return {
        "valid": valid,
        "reason": reason,
        "nominal_xyz": nominal.tolist(),
        "projected_xyz": current.tolist(),
        "correction_l2": float(np.linalg.norm(current - nominal)),
        "action_lower": lower_action.tolist(),
        "action_upper": upper_action.tolist(),
        "calibration_m": calibration.tolist(),
        "final_predicted_margin_m": final_margin.tolist(),
        "final_conservative_margin_m": final_conservative.tolist(),
        "final_jacobian_m_per_action": final_jacobian.tolist(),
        "final_inference_and_jacobian_wall_seconds": final_inference,
        "iterations": iterations,
        "total_wall_seconds": (time.perf_counter_ns() - total_started) * 1.0e-9,
    }
