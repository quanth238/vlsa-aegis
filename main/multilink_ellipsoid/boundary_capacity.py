"""Boundary-focused local capacity experiment for the E05 distal-link case."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .execution_margin_nn import (
    CONSTRAINT_ORDER,
    FEATURE_NAMES,
    _build_model,
    _canonical,
    _numpy,
    _torch,
)


BOUNDARY_CAPACITY_SCHEMA = "vlsa_distal_boundary_capacity_e05.v1"
BOUNDARY_DATASET_SCHEMA = "vlsa_distal_boundary_capacity_e05_dataset.v1"
BOUNDARY_DATASET_RESULT_SCHEMA = (
    "vlsa_distal_boundary_capacity_e05_dataset_result.v1"
)
BOUNDARY_RESULT_SCHEMA = "vlsa_distal_boundary_capacity_e05_result.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_boundary_capacity_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("boundary-capacity config is invalid JSON") from error
    required = {
        "schema_version",
        "protocol_id",
        "case_ids",
        "claim_scope",
        "immutable_sources",
        "state",
        "sampling",
        "split",
        "features",
        "network",
        "training",
        "calibration",
        "projection",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("boundary-capacity config keys differ")
    if config["schema_version"] != BOUNDARY_CAPACITY_SCHEMA:
        raise ValueError("boundary-capacity schema differs")
    if config["protocol_id"] != "vlsa-distal-boundary-capacity-e05-v1":
        raise ValueError("boundary-capacity protocol differs")
    if config["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("boundary-capacity protocol must select only E05")
    if config["claim_scope"] != (
        "post_oracle_selected_action188_same_state_local_capacity_diagnostic_"
        "not_state_generalization_not_closed_loop_efficacy_not_formal_safety_"
        "certificate"
    ):
        raise ValueError("boundary-capacity claim scope differs")
    if config["immutable_sources"] != {
        "archived_table1_file_sha256": "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b",
        "archived_table1_payload_sha256": "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c",
        "baseline_nn_result_file_sha256": "7fed872f2e189c9cdc07180f7ef3950c0a1a605065650e9650c8a9cee167fcca",
        "baseline_nn_result_payload_sha256": "22b3b9ebda2eb15c32215ab26fc08ce1e43d28dc120856c443cdac6a47f1a2a3",
        "exact_box_config_file_sha256": "cc568a85c2acf215beda1cef4abcc31a92b3f6d3772d6c36147410afd471bf9f",
        "exact_box_config_payload_sha256": "d3e0ab883eb3b3de160417fc9547012db9154dae7015ff80bc739219b728013b",
        "exact_box_discovery_file_sha256": "3db37092b2c5573e70cfc604bbbf01f1361be822ec87c8733d2c80b84685b0fd",
        "exact_box_discovery_payload_sha256": "2eef9d767354f6692d36f57899bd511c261733e05394d00c997eacc7252fc051",
        "false_safe_action_ledger_file_sha256": "d79a28585e74cece727fbc3d3e6a72eb1647a782cae4962ec3af9045f448e603",
        "false_safe_action_ledger_payload_sha256": "2a4ecd5ff79a363141e0927d1dfbcfb26248ec9bb5e2bc123cc8f879c3395a53",
        "geometry_config_file_sha256": "fd9042ebc7605c68e71d1ab51bfdc8b4412fcb9dda11f1879d85271641436da1",
        "previous_nn_config_file_sha256": "b2dc378aba6dbe91e244e83d059ed2be555c21cd78e6d7314294af3d7ba13488",
    }:
        raise ValueError("boundary-capacity immutable source identities differ")
    if config["state"] != {
        "source": "immutable_job_37109_executed_sitl_action",
        "step": 188,
        "critical_constraint_index": 1,
        "critical_constraint_name": "L5_part_1",
    }:
        raise ValueError("boundary-capacity state differs")
    if config["sampling"] != {
        "action_limit": 1.0,
        "trust_region_linf_action": 0.5,
        "grid_points_per_dimension": 10,
        "expected_grid_action_count": 1000,
        "boundary_band_m": 0.005,
        "gradient_anchor_count": 64,
        "gradient_anchor_safe_count": 32,
        "gradient_anchor_unsafe_count": 32,
        "finite_difference_epsilon_action": 0.02,
        "expected_gradient_probe_count": 384,
        "require_matching_substep_and_obstacle_witness": True,
    }:
        raise ValueError("boundary-capacity sampling differs")
    if config["split"] != {
        "unit": "base_action_with_all_finite_difference_probes",
        "seed": 20260809,
        "train_fraction": 0.7,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "stratify": "critical_row_boundary_safe_boundary_unsafe",
        "outside_boundary_band": "record_but_exclude_from_learning_splits",
    }:
        raise ValueError("boundary-capacity split differs")
    if config["features"] != {
        "source": "same_33_features_as_execution_margin_v2",
        "normalization": (
            "train_group_mean_and_standard_deviation_with_1e-6_floor"
        ),
    }:
        raise ValueError("boundary-capacity features differ")
    if config["network"] != {
        "hidden_widths": [128, 128],
        "hidden_activation": "silu",
        "output_activation": "softplus_nonnegative_clearance_loss_mm",
        "output_count": 7,
    }:
        raise ValueError("boundary-capacity network differs")
    training = config["training"]
    if training != {
        "arms": ["boundary_margin", "boundary_margin_gradient"],
        "batching": "deterministic_full_batch",
        "device": "cuda",
        "epochs": 3000,
        "gradient_loss_weight": 1.0,
        "gradient_normalization": (
            "training_valid_gradient_global_rms_with_1mm_per_action_floor"
        ),
        "huber_delta_mm": 1.0,
        "learning_rate": 0.001,
        "margin_category_target_fractions": {
            "boundary_safe": 0.5,
            "boundary_unsafe": 0.5,
        },
        "patience": 300,
        "seed": 20260809,
        "weight_decay": 1.0e-6,
    }:
        raise ValueError("boundary-capacity training differs")
    if config["calibration"] != {
        "method": (
            "per_constraint_maximum_validation_overprediction_plus_fixed_padding"
        ),
        "fixed_padding_m": 0.001,
    }:
        raise ValueError("boundary-capacity calibration differs")
    if config["projection"] != {
        "action_limit": 1.0,
        "activation_warning_m": 0.008,
        "bound_tolerance_action": 5.0e-8,
        "clearance_target_m": 0.0,
        "eps_abs": 1.0e-7,
        "eps_rel": 1.0e-7,
        "max_iter": 10000,
        "maximum_linearization_iterations": 3,
        "residual_tolerance": 5.0e-7,
        "trust_region_linf_action": 0.5,
    }:
        raise ValueError("boundary-capacity projection differs")
    if config["decision_gate"] != {
        "boundary_test_rmse": "strictly_below_current_clearance_baseline",
        "conservative_false_safe_candidate_count": 0,
        "critical_gradient_cosine_similarity_minimum": 0.8,
        "projection": (
            "valid_seven_row_QP_and_exact_proxy_raw_safe_cloned_transition"
        ),
        "research_direction_go": (
            "boundary_margin_gradient_arm_passes_all_capacity_gates"
        ),
    }:
        raise ValueError("boundary-capacity decision gate differs")
    fractions = config["split"]
    if not math.isclose(
        fractions["train_fraction"]
        + fractions["validation_fraction"]
        + fractions["test_fraction"],
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("boundary-capacity split fractions do not sum to one")
    category_total = sum(training["margin_category_target_fractions"].values())
    if not math.isclose(category_total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("boundary-capacity category weights do not sum to one")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def structured_grid_actions(
    nominal_xyz: Sequence[float], config: Mapping[str, Any]
) -> tuple[Any, Any, list[Any]]:
    """Return the frozen Cartesian grid inside action and trust bounds."""

    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("boundary-capacity nominal action is invalid")
    settings = config["sampling"]
    limit = float(settings["action_limit"])
    trust = float(settings["trust_region_linf_action"])
    lower = np.maximum(-limit, nominal - trust)
    upper = np.minimum(limit, nominal + trust)
    count = int(settings["grid_points_per_dimension"])
    axes = [np.linspace(lower[i], upper[i], count) for i in range(3)]
    actions = [
        np.asarray(values, dtype=np.float64)
        for values in itertools.product(*axes)
    ]
    if len(actions) != int(settings["expected_grid_action_count"]):
        raise ValueError("boundary-capacity grid count differs")
    if len({tuple(item.tolist()) for item in actions}) != len(actions):
        raise ValueError("boundary-capacity grid contains duplicate actions")
    return lower, upper, actions


def critical_category(margin_m: float, band_m: float) -> str:
    if not math.isfinite(float(margin_m)):
        raise ValueError("boundary-capacity critical margin is nonfinite")
    if 0.0 <= margin_m <= band_m:
        return "boundary_safe"
    if -band_m <= margin_m < 0.0:
        return "boundary_unsafe"
    return "far_safe" if margin_m > band_m else "far_unsafe"


def select_gradient_anchor_indexes(
    records: Sequence[Mapping[str, Any]],
    lower_action: Sequence[float],
    upper_action: Sequence[float],
    config: Mapping[str, Any],
) -> list[int]:
    """Select equal safe/unsafe anchors closest to the critical boundary."""

    np = _numpy()
    settings = config["sampling"]
    critical = int(config["state"]["critical_constraint_index"])
    epsilon = float(settings["finite_difference_epsilon_action"])
    lower = np.asarray(lower_action, dtype=np.float64)
    upper = np.asarray(upper_action, dtype=np.float64)
    eligible = []
    for record in records:
        action = np.asarray(record["candidate_xyz"], dtype=np.float64)
        margin = float(record["minimum_substep_clearance_m"][critical])
        if np.any(action - epsilon < lower - 1.0e-12) or np.any(
            action + epsilon > upper + 1.0e-12
        ):
            continue
        if abs(margin) > float(settings["boundary_band_m"]):
            continue
        eligible.append((abs(margin), int(record["grid_index"]), margin))
    safe = sorted(item for item in eligible if item[2] >= 0.0)
    unsafe = sorted(item for item in eligible if item[2] < 0.0)
    safe_count = int(settings["gradient_anchor_safe_count"])
    unsafe_count = int(settings["gradient_anchor_unsafe_count"])
    if len(safe) < safe_count or len(unsafe) < unsafe_count:
        raise ValueError(
            "boundary-capacity grid lacks balanced gradient anchors: "
            "safe=%d unsafe=%d" % (len(safe), len(unsafe))
        )
    selected = [item[1] for item in safe[:safe_count]] + [
        item[1] for item in unsafe[:unsafe_count]
    ]
    if len(selected) != int(settings["gradient_anchor_count"]):
        raise ValueError("boundary-capacity gradient anchor count differs")
    return sorted(selected)


def assign_grouped_splits(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, str]:
    """Assign complete base/probe groups with deterministic stratification."""

    np = _numpy()
    groups: dict[str, str] = {}
    for record in records:
        group = str(record["group_id"])
        category = str(record["category"])
        if group in groups and groups[group] != category:
            raise ValueError("boundary-capacity group category differs")
        groups[group] = category
    output: dict[str, str] = {}
    rng = np.random.default_rng(int(config["split"]["seed"]))
    for group, category in groups.items():
        if category in {"far_safe", "far_unsafe"}:
            output[group] = "excluded_far"
    for category in ("boundary_safe", "boundary_unsafe"):
        selected = sorted(group for group, value in groups.items() if value == category)
        if not selected:
            raise ValueError("boundary-capacity split category is empty: %s" % category)
        permutation = rng.permutation(len(selected))
        shuffled = [selected[int(index)] for index in permutation]
        train_count = int(math.floor(len(shuffled) * config["split"]["train_fraction"]))
        validation_count = int(
            math.floor(len(shuffled) * config["split"]["validation_fraction"])
        )
        if min(train_count, validation_count, len(shuffled) - train_count - validation_count) < 1:
            raise ValueError("boundary-capacity split category is too small")
        for index, group in enumerate(shuffled):
            if index < train_count:
                split = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            output[group] = split
    if set(output) != set(groups):
        raise ValueError("boundary-capacity group split is incomplete")
    return output


def _records_to_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    features = np.asarray([item["feature_vector"] for item in records], dtype=np.float64)
    current = np.asarray([item["current_clearance_m"] for item in records], dtype=np.float64)
    minimum = np.asarray(
        [item["minimum_substep_clearance_m"] for item in records], dtype=np.float64
    )
    split = np.asarray([str(item["split"]) for item in records])
    categories = np.asarray([str(item["category"]) for item in records])
    gradients = np.full((len(records), 7, 3), np.nan, dtype=np.float64)
    gradient_valid = np.zeros((len(records), 7), dtype=bool)
    for index, record in enumerate(records):
        if record.get("gradient_m_per_action") is None:
            continue
        gradients[index] = np.asarray(
            record["gradient_m_per_action"], dtype=np.float64
        )
        gradient_valid[index] = np.asarray(
            record["gradient_valid_rows"], dtype=bool
        )
    if (
        features.shape != (len(records), len(FEATURE_NAMES))
        or current.shape != (len(records), 7)
        or minimum.shape != current.shape
        or not all(np.all(np.isfinite(item)) for item in (features, current, minimum))
        or np.any(np.logical_and(gradient_valid[:, :, None], ~np.isfinite(gradients)))
    ):
        raise ValueError("boundary-capacity dataset arrays are invalid")
    loss_mm = (current - minimum) * 1000.0
    if float(np.min(loss_mm)) < -1.0e-7:
        raise ValueError("boundary-capacity interval minimum exceeds its start")
    return {
        "features": features,
        "current": current,
        "minimum": minimum,
        "loss_mm": np.maximum(loss_mm, 0.0),
        "split": split,
        "categories": categories,
        "gradients_m_per_action": gradients,
        "gradient_valid": gradient_valid,
    }


def _category_weights(categories: Any, config: Mapping[str, Any]) -> Any:
    np = _numpy()
    targets = config["training"]["margin_category_target_fractions"]
    weights = np.zeros(len(categories), dtype=np.float64)
    for category, target in targets.items():
        mask = categories == category
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise ValueError("boundary-capacity training category is empty")
        weights[mask] = float(target) / count
    if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("boundary-capacity sample weights do not sum to one")
    return weights


def _predict_all_gradients(
    model: Any, normalized_features: Any, candidate_scale: Any, *, create_graph: bool
) -> tuple[Any, Any]:
    """Return predicted loss and margin gradients in millimetres per action."""

    torch = _torch()
    values = normalized_features
    if not values.requires_grad:
        values = values.clone().detach().requires_grad_(True)
    predicted_loss_mm = model(values)
    rows = []
    for index in range(7):
        gradient = torch.autograd.grad(
            predicted_loss_mm[:, index].sum(),
            values,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        rows.append(-gradient[:, -3:] / candidate_scale[None, :])
    return predicted_loss_mm, torch.stack(rows, dim=1)


def train_boundary_capacity_model(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    arm: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Train one paired boundary-capacity arm on an H100."""

    np = _numpy()
    torch = _torch()
    if arm not in config["training"]["arms"]:
        raise ValueError("boundary-capacity arm is not registered")
    use_gradient = arm == "boundary_margin_gradient"
    arrays = _records_to_arrays(records)
    masks = {
        name: arrays["split"] == name
        for name in ("train", "validation", "test")
    }
    if any(int(np.count_nonzero(mask)) == 0 for mask in masks.values()):
        raise ValueError("boundary-capacity split is empty")
    train_features = arrays["features"][masks["train"]]
    mean = np.mean(train_features, axis=0)
    standard_deviation = np.maximum(np.std(train_features, axis=0), 1.0e-6)
    normalized = (arrays["features"] - mean) / standard_deviation
    settings = config["training"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("boundary-capacity training requires CUDA on H100")
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    model = _build_model(
        len(FEATURE_NAMES),
        config["network"]["hidden_widths"],
        config["network"]["output_count"],
    ).to(device=device, dtype=torch.float64)
    feature_tensor = torch.as_tensor(normalized, dtype=torch.float64, device=device)
    target_loss = torch.as_tensor(
        arrays["loss_mm"], dtype=torch.float64, device=device
    )
    mask_tensors = {
        name: torch.as_tensor(mask, dtype=torch.bool, device=device)
        for name, mask in masks.items()
    }
    train_weights_np = _category_weights(
        arrays["categories"][masks["train"]], config
    )
    train_weights = torch.as_tensor(
        train_weights_np, dtype=torch.float64, device=device
    )
    finite_gradient_target = np.where(
        arrays["gradient_valid"][:, :, None],
        arrays["gradients_m_per_action"],
        0.0,
    )
    gradient_target_mm = torch.as_tensor(
        finite_gradient_target * 1000.0,
        dtype=torch.float64,
        device=device,
    )
    gradient_valid = torch.as_tensor(
        arrays["gradient_valid"], dtype=torch.bool, device=device
    )
    train_gradient_values = (
        arrays["gradients_m_per_action"][masks["train"]] * 1000.0
    )
    train_gradient_mask = arrays["gradient_valid"][masks["train"]]
    expanded_mask = np.repeat(train_gradient_mask[:, :, None], 3, axis=2)
    if use_gradient and int(np.count_nonzero(expanded_mask)) == 0:
        raise ValueError("boundary-capacity training has no valid gradients")
    gradient_scale = max(
        1.0,
        float(
            np.sqrt(np.mean(train_gradient_values[expanded_mask] ** 2))
            if np.any(expanded_mask)
            else 1.0
        ),
    )
    candidate_scale = torch.as_tensor(
        standard_deviation[-3:], dtype=torch.float64, device=device
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    huber = torch.nn.HuberLoss(
        reduction="none", delta=float(settings["huber_delta_mm"])
    )

    def loss_for_split(name: str, *, create_graph: bool) -> tuple[Any, Any, Any]:
        indexes = mask_tensors[name]
        values = feature_tensor[indexes].clone().detach().requires_grad_(use_gradient)
        if use_gradient:
            predicted, predicted_gradient = _predict_all_gradients(
                model, values, candidate_scale, create_graph=create_graph
            )
        else:
            predicted = model(values)
            predicted_gradient = None
        element = huber(predicted, target_loss[indexes]).mean(dim=1)
        if name == "train":
            margin_loss = torch.sum(element * train_weights)
        else:
            margin_loss = torch.mean(element)
        gradient_loss = torch.zeros((), dtype=torch.float64, device=device)
        if use_gradient:
            valid = gradient_valid[indexes][:, :, None].expand(-1, -1, 3)
            difference = (
                predicted_gradient - gradient_target_mm[indexes]
            ) / gradient_scale
            if torch.any(valid):
                gradient_loss = torch.mean(difference[valid] ** 2)
        total = margin_loss + float(settings["gradient_loss_weight"]) * gradient_loss
        return total, margin_loss, gradient_loss

    best_state = None
    best_epoch = None
    best_validation = math.inf
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter_ns()
    for epoch in range(int(settings["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total, train_margin, train_gradient = loss_for_split(
            "train", create_graph=use_gradient
        )
        train_total.backward()
        optimizer.step()
        model.eval()
        with torch.enable_grad():
            validation_total, validation_margin, validation_gradient = loss_for_split(
                "validation", create_graph=False
            )
        value = float(validation_total.detach().cpu())
        if epoch == 0 or (epoch + 1) % 100 == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "train_total": float(train_total.detach().cpu()),
                    "train_margin": float(train_margin.detach().cpu()),
                    "train_gradient": float(train_gradient.detach().cpu()),
                    "validation_total": value,
                    "validation_margin": float(validation_margin.detach().cpu()),
                    "validation_gradient": float(
                        validation_gradient.detach().cpu()
                    ),
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
        raise RuntimeError("boundary-capacity training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted_loss_mm = model(feature_tensor).detach().cpu().numpy()
    predicted_margin = arrays["current"] - predicted_loss_mm / 1000.0
    validation_overprediction = (
        predicted_margin[masks["validation"]]
        - arrays["minimum"][masks["validation"]]
    )
    calibration = np.maximum(
        0.0, np.max(validation_overprediction, axis=0)
    ) + float(config["calibration"]["fixed_padding_m"])
    lower_margin = predicted_margin - calibration[None, :]

    with torch.enable_grad():
        _, predicted_gradient_mm = _predict_all_gradients(
            model,
            feature_tensor.clone().detach().requires_grad_(True),
            candidate_scale,
            create_graph=False,
        )
    predicted_gradient = predicted_gradient_mm.detach().cpu().numpy() / 1000.0
    critical = int(config["state"]["critical_constraint_index"])
    band = float(config["sampling"]["boundary_band_m"])

    def metrics(mask: Any) -> dict[str, Any]:
        actual = arrays["minimum"][mask]
        predicted = predicted_margin[mask]
        conservative = lower_margin[mask]
        baseline = arrays["current"][mask]
        false_safe = np.logical_and(conservative >= 0.0, actual < 0.0)
        boundary = np.abs(actual[:, critical]) <= band
        if not np.any(boundary):
            raise ValueError("boundary-capacity metric split lacks boundary samples")
        gradient_rows = np.logical_and(
            arrays["gradient_valid"][mask, critical], boundary
        )
        cosines = []
        if np.any(gradient_rows):
            target = arrays["gradients_m_per_action"][mask, critical][gradient_rows]
            estimate = predicted_gradient[mask, critical][gradient_rows]
            for left, right in zip(target, estimate):
                denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                if denominator > 1.0e-12:
                    cosines.append(float(np.dot(left, right) / denominator))
        boundary_error = predicted[boundary] - actual[boundary]
        boundary_baseline_error = baseline[boundary] - actual[boundary]
        critical_boundary_error = (
            predicted[boundary, critical] - actual[boundary, critical]
        )
        critical_boundary_baseline_error = (
            baseline[boundary, critical] - actual[boundary, critical]
        )
        return {
            "sample_count": int(np.count_nonzero(mask)),
            "boundary_sample_count": int(np.count_nonzero(boundary)),
            "boundary_rmse_m": float(np.sqrt(np.mean(boundary_error ** 2))),
            "boundary_current_clearance_baseline_rmse_m": float(
                np.sqrt(np.mean(boundary_baseline_error ** 2))
            ),
            "critical_boundary_rmse_m": float(
                np.sqrt(np.mean(critical_boundary_error ** 2))
            ),
            "critical_boundary_current_clearance_baseline_rmse_m": float(
                np.sqrt(np.mean(critical_boundary_baseline_error ** 2))
            ),
            "all_rmse_m": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "conservative_false_safe_candidate_count": int(
                np.count_nonzero(np.any(false_safe, axis=1))
            ),
            "critical_gradient_cosine_count": len(cosines),
            "critical_gradient_cosine_mean": (
                None if not cosines else float(np.mean(cosines))
            ),
            "critical_gradient_cosine_minimum": (
                None if not cosines else float(np.min(cosines))
            ),
            "minimum_actual_margin_m": float(np.min(actual)),
            "minimum_conservative_margin_m": float(np.min(conservative)),
        }

    split_metrics = {name: metrics(mask) for name, mask in masks.items()}
    test = split_metrics["test"]
    model_gate = bool(
        test["critical_boundary_rmse_m"]
        < test["critical_boundary_current_clearance_baseline_rmse_m"]
        and test["conservative_false_safe_candidate_count"] == 0
        and test["critical_gradient_cosine_count"] > 0
        and test["critical_gradient_cosine_mean"]
        >= float(
            config["decision_gate"][
                "critical_gradient_cosine_similarity_minimum"
            ]
        )
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
        "arm": arm,
        "model_class": "current_exact_clearance_minus_softplus_execution_loss",
        "gradient_supervision": use_gradient,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": best_epoch,
        "completed_epoch_count": int(epoch + 1),
        "best_validation_objective": best_validation,
        "gradient_scale_mm_per_action": gradient_scale,
        "training_gradient_valid_row_count": int(
            np.count_nonzero(train_gradient_mask)
        ),
        "history": history,
        "calibration_m": calibration.tolist(),
        "split_metrics": split_metrics,
        "model_capacity_gate_pass": model_gate,
        "training_wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    return model, serializable, audit
