"""State-conditioned neural surrogate for the fixed multi-region safety oracle.

The module is opt-in and leaves released AEGIS untouched.  It predicts the
regional lower affine values used by the seven-row L5--L7 QPs.  Exact simulator
labels remain evaluation authority.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import CONSTRAINT_ORDER, _canonical, _numpy, _torch
from .multi_region_affine_oracle import fixed_regions
from .oracle_affine_safe_set import solve_affine_certificate_qp
from .two_step_margin import PAIR_CANDIDATE_SLICE, PAIR_FEATURE_NAMES


CONFIG_SCHEMA = "vlsa_distal_region_aware_mlp_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_region_aware_mlp_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_region_aware_mlp_moka10_validation.v1"
CLOSED_LOOP_RESULT_SCHEMA = "vlsa_distal_region_aware_mlp_closed_loop_e05_result.v1"
CLOSED_LOOP_VALIDATION_SCHEMA = (
    "vlsa_distal_region_aware_mlp_closed_loop_e05_validation.v1"
)
WEIGHTS_SCHEMA = "vlsa_distal_region_aware_mlp_moka10_weights.v1"

REGION_FEATURE_NAMES = tuple(
    list(PAIR_FEATURE_NAMES)
    + ["region_normalized_lower_%d" % index for index in range(3)]
    + ["region_normalized_upper_%d" % index for index in range(3)]
    + ["region_normalized_center_%d" % index for index in range(3)]
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("region-aware MLP config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "split", "partition", "features", "model", "training",
        "uncertainty", "projection", "learned_gate", "closed_loop",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("region-aware MLP config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"] != "vlsa-distal-region-aware-mlp-moka10-v1"
    ):
        raise ValueError("region-aware MLP protocol differs")
    source = config["immutable_source"]
    source_keys = {
        "dataset_file_sha256", "dataset_payload_sha256",
        "off_grid_result_file_sha256", "off_grid_result_payload_sha256",
        "multi_region_result_file_sha256", "multi_region_result_payload_sha256",
        "multi_region_validation_file_sha256",
        "decision_stability_result_file_sha256",
        "decision_stability_result_payload_sha256",
        "decision_stability_validation_file_sha256", "archived_e05_file_sha256",
        "archived_e05_payload_sha256", "expected_state_count",
        "fit_actions_per_state", "off_grid_actions_per_state",
    }
    if set(source) != source_keys or (
        int(source["expected_state_count"]) != 50
        or int(source["fit_actions_per_state"]) != 125
        or int(source["off_grid_actions_per_state"]) != 96
    ):
        raise ValueError("region-aware immutable source differs")
    if config["split"] != {
        "unit": "complete_episode_and_task_level_group",
        "expected_state_counts": {"train": 30, "validation": 5, "test": 15},
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "primary_e05_use": (
            "test_and_conditional_closed_loop_only_never_training_or_calibration"
        ),
    }:
        raise ValueError("region-aware grouped split differs")
    if config["partition"] != {
        "coordinates": "per_state_action_box_normalized_minus1_plus1",
        "axis_intervals": [[-1.0, 0.0], [-0.5, 0.5], [0.0, 1.0]],
        "axis_centers": [-0.5, 0.0, 0.5], "region_count": 27,
        "fit_actions_per_region": 27, "action_limit": 1.0,
        "trust_region_linf_action": 0.5, "grid_points_per_dimension": 5,
        "inclusive_membership_tolerance": 1.0e-12,
    }:
        raise ValueError("region-aware partition differs")
    if config["features"] != {
        "base": (
            "validated_53D_pair_state_geometry_feature_with_candidate_slot_"
            "replaced_by_region_anchor"
        ),
        "region_descriptor": [
            "normalized_lower_xyz", "normalized_upper_xyz",
            "normalized_center_xyz",
        ],
        "input_dimension": len(REGION_FEATURE_NAMES),
        "normalization": (
            "training_states_mean_and_standard_deviation_with_1e-6_floor"
        ),
    }:
        raise ValueError("region-aware features differ")
    model = config["model"]
    if (
        model.get("class") != "shared_constraint_and_region_MLP_ensemble"
        or model.get("ensemble_seeds")
        != [20260811, 20260812, 20260813, 20260814, 20260815]
        or model.get("hidden_widths") != [256, 256, 128]
        or model.get("hidden_activation") != "silu"
        or model.get("outputs") != [
            "regional_lower_anchor_mm", "gradient_x_mm_per_action",
            "gradient_y_mm_per_action", "gradient_z_mm_per_action",
        ]
    ):
        raise ValueError("region-aware model differs")
    if config["training"] != {
        "device": "cpu_inside_H100_allocation_due_pinned_sm90_incompatibility",
        "batching": "deterministic_full_batch", "epochs": 1800,
        "patience": 180, "learning_rate": 0.001, "weight_decay": 1.0e-6,
        "huber_delta_standardized": 1.0, "regional_value_loss_weight": 1.0,
        "one_sided_exact_overestimate_loss_weight": 4.0,
        "coefficient_loss_weight": 0.25,
    }:
        raise ValueError("region-aware training differs")
    if config["uncertainty"] != {
        "ensemble_aggregation": "mean_affine_coefficients",
        "guard": (
            "twice_maximum_ensemble_affine_standard_deviation_over_eight_"
            "region_vertices"
        ),
        "standard_deviation_multiplier": 2.0,
        "calibration": (
            "per_region_and_constraint_maximum_validation_exact_margin_"
            "overestimate_after_guard_plus_fixed_padding"
        ),
        "calibration_sources": [
            "validation_fit_grid", "validation_immutable_off_grid"
        ],
        "fixed_padding_m": 0.001,
    }:
        raise ValueError("region-aware uncertainty differs")
    if config["projection"] != {
        "eps_abs": 1.0e-7, "eps_rel": 1.0e-7, "max_iter": 10000,
        "residual_tolerance": 1.0e-6, "bound_tolerance_action": 5.0e-8,
    }:
        raise ValueError("region-aware projection differs")
    if config["learned_gate"] != {
        "test_off_grid_false_safe_action_count": 0,
        "minimum_global_accepted_set_jaccard_to_oracle": 0.9,
        "minimum_state_accepted_set_jaccard_to_oracle": 0.8,
        "required_test_state_safe_support_count": 15,
        "required_valid_selected_QP_count": 15,
        "required_fresh_exact_safe_selected_QP_count": 15,
        "required_released_AEGIS_EE_compatible_selected_QP_count": 15,
        "selected_action_shift_from_oracle_l2_p95_maximum": 0.1,
        "selected_action_shift_from_oracle_l2_maximum": 0.25,
        "closed_loop_authorized_only_if_every_gate_passes": True,
    }:
        raise ValueError("region-aware learned gate differs")
    if config["closed_loop"] != {
        "case_id": "vlsa-t1-goal-ii-t0-e05",
        "nominal_source": (
            "immutable_released_AEGIS_prefix_then_live_pi05_libero_after_"
            "first_learned_intervention"
        ),
        "released_AEGIS_EE_filter": "retained_and_recomputed_from_live_state",
        "activation_current_distal_clearance_m": 0.05,
        "inactive_behavior": (
            "execute_released_AEGIS_nominal_and_keep_exact_rollout_"
            "measurement_only"
        ),
        "safety_replanning": (
            "within_50mm_solve_27_seven_row_regional_QPs_before_every_"
            "executed_action_and_select_minimum_L2_valid_action_without_"
            "simulator_outcome"
        ),
        "prediction_horizon_actions": 2,
        "second_action_when_live_chunk_tail": (
            "repeat_current_released_AEGIS_nominal"
        ),
        "execute": "first_action_only_then_recompute_from_next_state",
        "exact_cloned_OSC_rollout_role": (
            "measurement_only_never_selection_fallback_or_stopping"
        ),
        "fallback": "none", "maximum_steps": 300,
        "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
        "success_requires": [
            "zero_L5_L6_L7_raw_contact",
            "all_executed_two_step_distal_margins_nonnegative",
            "paper_CAR_pass", "native_SafeLIBERO_task_success",
        ],
        "video": True,
    }:
        raise ValueError("region-aware closed-loop gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def region_feature_matrix(
    pair_state_features: Sequence[Sequence[float]], region: Mapping[str, Any]
) -> Any:
    """Build seven row features for one region without simulator labels."""

    np = _numpy()
    base = np.asarray(pair_state_features, dtype=np.float64).copy()
    anchor = np.asarray(region["anchor_xyz"], dtype=np.float64)
    descriptor = np.concatenate((
        np.asarray(region["normalized_lower"], dtype=np.float64),
        np.asarray(region["normalized_upper"], dtype=np.float64),
        np.asarray(region["normalized_center"], dtype=np.float64),
    ))
    if (
        base.shape != (7, len(PAIR_FEATURE_NAMES)) or anchor.shape != (3,)
        or descriptor.shape != (9,) or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(descriptor))
    ):
        raise ValueError("region-aware feature input differs")
    base[:, PAIR_CANDIDATE_SLICE] = anchor[None, :]
    output = np.concatenate((base, np.repeat(descriptor[None, :], 7, axis=0)), axis=1)
    if output.shape != (7, len(REGION_FEATURE_NAMES)):
        raise ValueError("region-aware feature shape differs")
    return output


def live_regions(
    nominal_xyz: Sequence[float], partition: Mapping[str, Any]
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Recreate the immutable five-point action box around a live nominal."""

    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    limit = float(partition["action_limit"])
    trust = float(partition["trust_region_linf_action"])
    points = int(partition["grid_points_per_dimension"])
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("region-aware live nominal differs")
    lower = np.maximum(-limit, nominal - trust)
    upper = np.minimum(limit, nominal + trust)
    axes = [np.linspace(lower[index], upper[index], points) for index in range(3)]
    candidates = np.asarray(list(itertools.product(*axes)), dtype=np.float64)
    if candidates.shape != (125, 3):
        raise ValueError("region-aware live grid differs")
    regions = fixed_regions(candidates, lower, upper, partition)
    return lower, upper, regions


def regional_training_arrays(
    states: Sequence[Mapping[str, Any]],
    oracle_states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Flatten state/region/constraint oracle rows for grouped training."""

    np = _numpy()
    oracle = {int(item["state_index"]): item for item in oracle_states}
    features = []
    targets = []
    deltas = []
    exact = []
    splits = []
    state_indexes = []
    region_indexes = []
    constraint_indexes = []
    for state in states:
        state_index = int(state["state_index"])
        source = oracle.get(state_index)
        if (
            source is None or source["case_id"] != state["case_id"]
            or int(source["state_step"]) != int(state["state_step"])
            or len(source["regions"]) != 27
            or len(source["regional_targets"]) != 27
        ):
            raise ValueError("region-aware oracle/state pairing differs")
        fit_xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
        fit_margin = np.asarray(
            state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        if fit_xyz.shape != (125, 3) or fit_margin.shape != (125, 7):
            raise ValueError("region-aware fit grid differs")
        for region, target in zip(source["regions"], source["regional_targets"]):
            if int(region["region_index"]) != int(target["region_index"]):
                raise ValueError("region-aware region target order differs")
            row_features = region_feature_matrix(
                state["pair_state_feature_vectors"], region
            )
            anchor = np.asarray(target["anchor_xyz"], dtype=np.float64)
            lower_anchor = (
                np.asarray(target["anchor_margin_m"], dtype=np.float64)
                - np.asarray(target["one_sided_error_m"], dtype=np.float64)
            )
            gradients = np.asarray(target["gradient_m_per_action"], dtype=np.float64)
            indexes = np.asarray(region["fit_candidate_indexes"], dtype=np.int64)
            region_deltas = fit_xyz[indexes] - anchor[None, :]
            for row in range(7):
                features.append(row_features[row])
                targets.append(np.concatenate(([lower_anchor[row]], gradients[row])))
                deltas.append(region_deltas)
                exact.append(fit_margin[indexes, row])
                splits.append(state["split"])
                state_indexes.append(state_index)
                region_indexes.append(int(region["region_index"]))
                constraint_indexes.append(row)
    output = {
        "features": np.asarray(features, dtype=np.float64),
        "targets_m": np.asarray(targets, dtype=np.float64),
        "deltas": np.asarray(deltas, dtype=np.float64),
        "exact_margin_m": np.asarray(exact, dtype=np.float64),
        "split": np.asarray(splits, dtype=object),
        "state_index": np.asarray(state_indexes, dtype=np.int64),
        "region_index": np.asarray(region_indexes, dtype=np.int64),
        "constraint_index": np.asarray(constraint_indexes, dtype=np.int64),
    }
    count = len(states) * 27 * 7
    if (
        output["features"].shape != (count, len(REGION_FEATURE_NAMES))
        or output["targets_m"].shape != (count, 4)
        or output["deltas"].shape != (count, 27, 3)
        or output["exact_margin_m"].shape != (count, 27)
        or not all(np.all(np.isfinite(output[key])) for key in (
            "features", "targets_m", "deltas", "exact_margin_m"
        ))
    ):
        raise ValueError("region-aware training arrays differ")
    return output


def build_model(hidden_widths: Sequence[int]) -> Any:
    torch = _torch()
    nn = torch.nn

    class RegionAwareNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            previous = len(REGION_FEATURE_NAMES)
            for width in hidden_widths:
                layers.extend((nn.Linear(previous, int(width)), nn.SiLU()))
                previous = int(width)
            layers.append(nn.Linear(previous, 4))
            self.network = nn.Sequential(*layers)

        def forward(self, values: Any) -> Any:
            return self.network(values)

    return RegionAwareNet()


def train_ensemble(
    arrays: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Train all preregistered members and return metre-space predictions."""

    import time

    np = _numpy()
    torch = _torch()
    if config["training"]["device"] != (
        "cpu_inside_H100_allocation_due_pinned_sm90_incompatibility"
    ):
        raise ValueError("region-aware training device differs")
    torch.set_num_threads(8)
    masks = {
        name: np.asarray(arrays["split"] == name, dtype=bool)
        for name in ("train", "validation", "test")
    }
    if not all(np.any(value) for value in masks.values()):
        raise ValueError("region-aware grouped split is empty")
    features = np.asarray(arrays["features"], dtype=np.float64)
    targets_mm = np.asarray(arrays["targets_m"], dtype=np.float64) * 1000.0
    deltas = np.asarray(arrays["deltas"], dtype=np.float64)
    exact_mm = np.asarray(arrays["exact_margin_m"], dtype=np.float64) * 1000.0
    mean = np.mean(features[masks["train"]], axis=0)
    std = np.maximum(np.std(features[masks["train"]], axis=0), 1.0e-6)
    normalized = (features - mean) / std
    output_mean = np.mean(targets_mm[masks["train"]], axis=0)
    output_scale = np.maximum(np.std(targets_mm[masks["train"]], axis=0), 1.0)
    target_standardized = (targets_mm - output_mean) / output_scale
    device = torch.device("cpu")
    x = torch.as_tensor(normalized, dtype=torch.float64, device=device)
    y = torch.as_tensor(target_standardized, dtype=torch.float64, device=device)
    delta_t = torch.as_tensor(deltas, dtype=torch.float64, device=device)
    exact_t = torch.as_tensor(exact_mm, dtype=torch.float64, device=device)
    mean_t = torch.as_tensor(output_mean, dtype=torch.float64, device=device)
    scale_t = torch.as_tensor(output_scale, dtype=torch.float64, device=device)
    mask_t = {
        name: torch.as_tensor(mask, dtype=torch.bool, device=device)
        for name, mask in masks.items()
    }
    settings = config["training"]
    huber = torch.nn.HuberLoss(
        delta=float(settings["huber_delta_standardized"]), reduction="mean"
    )
    models = []
    audits = []
    prediction_members = []
    started = time.perf_counter_ns()

    def member_loss(model: Any, name: str) -> Any:
        selected = mask_t[name]
        raw = model(x[selected])
        physical = raw * scale_t + mean_t
        target_physical = targets_mm[selected.cpu().numpy()]
        target_physical_t = torch.as_tensor(
            target_physical, dtype=torch.float64, device=device
        )
        predicted_values = physical[:, 0, None] + torch.einsum(
            "bi,bji->bj", physical[:, 1:4], delta_t[selected]
        )
        oracle_values = target_physical_t[:, 0, None] + torch.einsum(
            "bi,bji->bj", target_physical_t[:, 1:4], delta_t[selected]
        )
        coefficient = huber(raw, y[selected])
        regional = huber(predicted_values / 10.0, oracle_values / 10.0)
        over = torch.relu((predicted_values - exact_t[selected]) / 5.0)
        one_sided = torch.mean(over * over)
        return (
            float(settings["coefficient_loss_weight"]) * coefficient
            + float(settings["regional_value_loss_weight"]) * regional
            + float(settings["one_sided_exact_overestimate_loss_weight"])
            * one_sided
        )

    for seed in config["model"]["ensemble_seeds"]:
        torch.manual_seed(int(seed))
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        model = build_model(config["model"]["hidden_widths"]).to(
            device=device, dtype=torch.float64
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        best_state = None
        best_epoch = None
        best_validation = math.inf
        stale = 0
        history = []
        member_started = time.perf_counter_ns()
        for epoch in range(int(settings["epochs"])):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = member_loss(model, "train")
            train_loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = member_loss(model, "validation")
            value = float(validation_loss.cpu())
            if epoch == 0 or (epoch + 1) % 100 == 0:
                history.append({
                    "epoch": int(epoch),
                    "train_loss": float(train_loss.detach().cpu()),
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
            raise RuntimeError("region-aware training produced no checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            physical = model(x) * scale_t + mean_t
        prediction_members.append(physical.cpu().numpy() / 1000.0)
        models.append(model)
        audits.append({
            "seed": int(seed), "best_epoch": best_epoch,
            "completed_epoch_count": int(epoch + 1),
            "best_validation_objective": best_validation, "history": history,
            "training_wall_seconds": (
                time.perf_counter_ns() - member_started
            ) * 1.0e-9,
        })
    state = {
        "state_dicts": [
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            for model in models
        ],
        "feature_mean": mean, "feature_standard_deviation": std,
        "output_mean_mm": output_mean, "output_scale_mm": output_scale,
        "hidden_widths": list(config["model"]["hidden_widths"]),
        "ensemble_seeds": list(config["model"]["ensemble_seeds"]),
        "calibration_m": np.zeros((27, 7), dtype=np.float64),
    }
    audit = {
        "model_class": config["model"]["class"], "device": str(device),
        "torch_version": str(torch.__version__),
        "allocation_cuda_available": bool(torch.cuda.is_available()),
        "allocation_cuda_device_name": (
            None if not torch.cuda.is_available()
            else str(torch.cuda.get_device_name(0))
        ),
        "ensemble_member_count": len(models),
        "parameter_count_per_member": int(sum(
            value.numel() for value in models[0].parameters()
        )),
        "members": audits,
        "training_wall_seconds": (
            time.perf_counter_ns() - started
        ) * 1.0e-9,
    }
    return models, {**state, "predictions_m": np.asarray(prediction_members)}, audit


def predict_members(
    models: Sequence[Any], state: Mapping[str, Any], features: Any
) -> Any:
    np = _numpy()
    torch = _torch()
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(REGION_FEATURE_NAMES):
        raise ValueError("region-aware prediction features differ")
    normalized = (
        values - np.asarray(state["feature_mean"], dtype=np.float64)
    ) / np.asarray(state["feature_standard_deviation"], dtype=np.float64)
    mean = torch.as_tensor(state["output_mean_mm"], dtype=torch.float64)
    scale = torch.as_tensor(state["output_scale_mm"], dtype=torch.float64)
    tensor = torch.as_tensor(normalized, dtype=torch.float64)
    output = []
    for model in models:
        model.eval()
        with torch.no_grad():
            output.append(((model(tensor) * scale + mean) / 1000.0).cpu().numpy())
    return np.asarray(output, dtype=np.float64)


def region_vertices(region: Mapping[str, Any]) -> Any:
    np = _numpy()
    lower = np.asarray(region["action_lower"], dtype=np.float64)
    upper = np.asarray(region["action_upper"], dtype=np.float64)
    return np.asarray(list(itertools.product(*zip(lower, upper))), dtype=np.float64)


def guarded_targets(
    models: Sequence[Any], state: Mapping[str, Any], pair_state_features: Any,
    regions: Sequence[Mapping[str, Any]], uncertainty: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Predict 27 conservative targets using no exact rollout outcome."""

    np = _numpy()
    calibration = np.asarray(state["calibration_m"], dtype=np.float64)
    if calibration.shape != (27, 7) or len(regions) != 27:
        raise ValueError("region-aware calibration or regions differ")
    outputs = []
    multiplier = float(uncertainty["standard_deviation_multiplier"])
    for region in regions:
        region_index = int(region["region_index"])
        features = region_feature_matrix(pair_state_features, region)
        members = predict_members(models, state, features)
        mean = np.mean(members, axis=0)
        anchor = np.asarray(region["anchor_xyz"], dtype=np.float64)
        delta = region_vertices(region) - anchor[None, :]
        member_values = members[:, :, 0, None] + np.einsum(
            "mri,vi->mrv", members[:, :, 1:4], delta
        )
        maximum_std = np.max(np.std(member_values, axis=0), axis=1)
        guard = multiplier * maximum_std
        lower_anchor = mean[:, 0] - guard - calibration[region_index]
        outputs.append({
            "region_index": region_index, "anchor_xyz": anchor.tolist(),
            "lower_anchor_m": lower_anchor.tolist(),
            "gradient_m_per_action": mean[:, 1:4].tolist(),
            "ensemble_guard_m": guard.tolist(),
            "calibration_m": calibration[region_index].tolist(),
        })
    return outputs


def target_values(target: Mapping[str, Any], xyz: Sequence[float]) -> Any:
    np = _numpy()
    return (
        np.asarray(target["lower_anchor_m"], dtype=np.float64)
        + np.asarray(target["gradient_m_per_action"], dtype=np.float64)
        @ (
            np.asarray(xyz, dtype=np.float64)
            - np.asarray(target["anchor_xyz"], dtype=np.float64)
        )
    )


def solve_regional_qps(
    nominal_xyz: Sequence[float], regions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]], projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve all regional QPs and select without simulator outcome."""

    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if len(regions) != 27 or len(targets) != 27:
        raise ValueError("region-aware QP region count differs")
    proposals = []
    for region, target in zip(regions, targets):
        if int(region["region_index"]) != int(target["region_index"]):
            raise ValueError("region-aware QP target order differs")
        gradient = np.asarray(target["gradient_m_per_action"], dtype=np.float64)
        intercept = target_values(target, nominal)
        certificate = {
            "valid": True, "intercept_at_nominal_m": intercept.tolist(),
            "gradients_m_per_action": gradient.tolist(),
        }
        solution = solve_affine_certificate_qp(
            nominal, region["action_lower"], region["action_upper"],
            certificate, projection,
        )
        proposals.append({
            "region_index": int(region["region_index"]), **solution,
        })
    valid = [item for item in proposals if item.get("valid")]
    valid.sort(key=lambda item: (
        float(item["correction_l2"]), int(item["region_index"])
    ))
    return {
        "regional_proposals": proposals,
        "valid_proposal_count": len(valid),
        "selected": None if not valid else valid[0],
    }


def save_model(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    names = list(state["state_dicts"][0])
    metadata = {
        "schema_version": WEIGHTS_SCHEMA,
        "feature_names": list(REGION_FEATURE_NAMES),
        "constraint_order": list(CONSTRAINT_ORDER),
        "hidden_widths": list(state["hidden_widths"]),
        "ensemble_seeds": list(state["ensemble_seeds"]),
        "parameter_names": names,
    }
    arrays = {
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float64),
        "feature_standard_deviation": np.asarray(
            state["feature_standard_deviation"], dtype=np.float64
        ),
        "output_mean_mm": np.asarray(state["output_mean_mm"], dtype=np.float64),
        "output_scale_mm": np.asarray(state["output_scale_mm"], dtype=np.float64),
        "calibration_m": np.asarray(state["calibration_m"], dtype=np.float64),
        "metadata_utf8": np.frombuffer(_canonical(metadata), dtype=np.uint8),
    }
    for member_index, member in enumerate(state["state_dicts"]):
        for parameter_index, name in enumerate(names):
            arrays["member_%02d_parameter_%03d" % (
                member_index, parameter_index
            )] = member[name].detach().cpu().numpy().astype(np.float64)
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    raw = path.read_bytes()
    return {"path": str(path), "file_sha256": _sha256(raw), "size_bytes": len(raw)}


def load_model(path: Path) -> tuple[list[Any], dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["metadata_utf8"].tolist()).decode("utf-8"))
        if (
            metadata.get("schema_version") != WEIGHTS_SCHEMA
            or metadata.get("feature_names") != list(REGION_FEATURE_NAMES)
            or metadata.get("constraint_order") != list(CONSTRAINT_ORDER)
        ):
            raise ValueError("region-aware model identity differs")
        models = []
        names = metadata["parameter_names"]
        for member_index, _ in enumerate(metadata["ensemble_seeds"]):
            model = build_model(metadata["hidden_widths"]).to(dtype=torch.float64)
            if names != list(model.state_dict()):
                raise ValueError("region-aware parameter names differ")
            member = {}
            for parameter_index, name in enumerate(names):
                values = np.asarray(
                    archive["member_%02d_parameter_%03d" % (
                        member_index, parameter_index
                    )], dtype=np.float64,
                )
                if values.shape != tuple(model.state_dict()[name].shape):
                    raise ValueError("region-aware parameter shape differs")
                member[name] = torch.as_tensor(values, dtype=torch.float64)
            model.load_state_dict(member)
            model.eval()
            models.append(model)
        state = {
            "feature_mean": np.asarray(archive["feature_mean"], dtype=np.float64),
            "feature_standard_deviation": np.asarray(
                archive["feature_standard_deviation"], dtype=np.float64
            ),
            "output_mean_mm": np.asarray(archive["output_mean_mm"], dtype=np.float64),
            "output_scale_mm": np.asarray(archive["output_scale_mm"], dtype=np.float64),
            "calibration_m": np.asarray(archive["calibration_m"], dtype=np.float64),
            "hidden_widths": list(metadata["hidden_widths"]),
            "ensemble_seeds": list(metadata["ensemble_seeds"]),
        }
    return models, state


def jaccard(left: Sequence[bool], right: Sequence[bool]) -> float:
    np = _numpy()
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("region-aware Jaccard shape differs")
    union = int(np.count_nonzero(np.logical_or(a, b)))
    intersection = int(np.count_nonzero(np.logical_and(a, b)))
    return 1.0 if union == 0 else intersection / union
