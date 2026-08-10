"""Action-conditioned execution-margin surrogate for distal-link QPs.

This module is opt-in and leaves released AEGIS unchanged.  Unlike the prior
coefficient-output model, it learns exact two-action cloned-OSC margins at a
candidate Cartesian action.  At inference its guarded values are converted to
fixed regional affine rows by the already validated ridge-Huber procedure.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import CONSTRAINT_ORDER, _canonical, _numpy, _torch
from .multi_region_affine_oracle import fit_region_target, region_affine_values
from .region_aware_mlp import solve_regional_qps
from .two_step_margin import PAIR_CANDIDATE_SLICE, PAIR_FEATURE_NAMES


CONFIG_SCHEMA = "vlsa_distal_action_conditioned_margin_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_action_conditioned_margin_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_action_conditioned_margin_moka10_validation.v1"
WEIGHTS_SCHEMA = "vlsa_distal_action_conditioned_margin_moka10_weights.v1"

NORMALIZED_ACTION_NAMES = tuple(
    "candidate_normalized_within_current_action_box_%d" % index
    for index in range(3)
)
ACTION_FEATURE_NAMES = tuple(list(PAIR_FEATURE_NAMES) + list(NORMALIZED_ACTION_NAMES))
CURRENT_CLEARANCE_INDEX = PAIR_FEATURE_NAMES.index("current_clearance_m")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("action-conditioned config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "source_population_manifest_sha256", "test_selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "split", "features", "model", "training", "uncertainty",
        "regionalization", "ridge_huber", "projection", "learned_gate",
        "exact_verification", "decision",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("action-conditioned config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-action-conditioned-margin-moka10-v1"
    ):
        raise ValueError("action-conditioned protocol differs")
    source = config["immutable_source"]
    source_keys = {
        "expanded_dataset_file_sha256", "expanded_dataset_payload_sha256",
        "expanded_oracle_file_sha256", "expanded_oracle_payload_sha256",
        "support_result_file_sha256", "support_result_payload_sha256",
        "support_validation_file_sha256", "supported_mlp_result_file_sha256",
        "supported_mlp_result_payload_sha256",
        "supported_mlp_validation_file_sha256",
        "validation_fresh_file_sha256", "validation_fresh_payload_sha256",
        "archived_e05_file_sha256", "archived_e05_payload_sha256",
        "expected_state_count", "fit_actions_per_state",
        "off_grid_actions_per_validation_or_test_state",
    }
    if (
        set(source) != source_keys
        or int(source["expected_state_count"]) != 85
        or int(source["fit_actions_per_state"]) != 125
        or int(source["off_grid_actions_per_validation_or_test_state"]) != 96
    ):
        raise ValueError("action-conditioned immutable source differs")
    if config["split"] != {
        "unit": "complete_episode_and_task_level_group",
        "expected_state_counts": {"train": 60, "validation": 10, "test": 15},
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "held_out_test_never_used_for_training_calibration_early_stopping_or_model_selection": True,
    }:
        raise ValueError("action-conditioned grouped split differs")
    if (
        config["features"].get("input_dimension") != len(ACTION_FEATURE_NAMES)
        or config["features"].get("output_parameterization")
        != "exact_two_step_margin_equals_current_clearance_plus_learned_residual"
    ):
        raise ValueError("action-conditioned feature contract differs")
    if config["model"] != {
        "class": "shared_constraint_action_conditioned_residual_margin_MLP_ensemble",
        "ensemble_seeds": [20260821, 20260822, 20260823, 20260824, 20260825],
        "hidden_widths": [256, 256, 128], "hidden_activation": "silu",
        "output": "scalar_margin_residual_mm",
    }:
        raise ValueError("action-conditioned model differs")
    if config["training"] != {
        "device": "cpu_inside_H100_allocation_due_pinned_sm90_incompatibility",
        "batch_size": 4096, "epochs": 800, "patience": 80,
        "learning_rate": 0.001, "weight_decay": 1.0e-6,
        "huber_delta_mm": 2.0, "boundary_band_m": 0.005,
        "boundary_weight_multiplier": 9.0,
        "one_sided_overestimate_loss_weight": 4.0,
        "one_sided_scale_mm": 5.0,
        "early_stopping_population": "validation_fit_grid_only",
    }:
        raise ValueError("action-conditioned training differs")
    if config["uncertainty"] != {
        "ensemble_aggregation": "mean_exact_margin",
        "standard_deviation_multiplier": 2.0,
        "calibration": (
            "per_constraint_maximum_validation_overestimate_after_"
            "pointwise_ensemble_guard"
        ),
        "calibration_sources": [
            "validation_fit_grid", "validation_immutable_off_grid"
        ],
        "fixed_padding_m": 0.001,
    }:
        raise ValueError("action-conditioned uncertainty differs")
    regional = config["regionalization"]
    if (
        regional.get("region_count") != 27
        or regional.get("fit_actions_per_region") != 27
        or regional.get("runtime_value_queries_per_state") != 125
        or regional.get("affine_fit_source")
        != "guarded_action_conditioned_values_not_learned_coefficients"
    ):
        raise ValueError("action-conditioned regionalization differs")
    if config["decision"] != {
        "closed_loop_in_this_gate": False,
        "if_every_learned_gate_passes": (
            "preregister_receding_QP_closed_loop_E05"
        ),
        "if_supported_action_conditioned_model_fails": (
            "reject_plain_MLP_generalization_and_test_nonparametric_or_"
            "latent_local_adaptation_before_closed_loop"
        ),
    }:
        raise ValueError("action-conditioned decision differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def action_feature_matrix(
    pair_state_features: Sequence[Sequence[float]],
    action_xyz: Sequence[float], action_lower: Sequence[float],
    action_upper: Sequence[float],
) -> Any:
    """Return seven constraint rows for one candidate action."""

    np = _numpy()
    base = np.asarray(pair_state_features, dtype=np.float64).copy()
    action = np.asarray(action_xyz, dtype=np.float64)
    lower = np.asarray(action_lower, dtype=np.float64)
    upper = np.asarray(action_upper, dtype=np.float64)
    if (
        base.shape != (7, len(PAIR_FEATURE_NAMES))
        or action.shape != (3,) or lower.shape != (3,) or upper.shape != (3,)
        or np.any(upper <= lower)
        or np.any(action < lower - 1.0e-12)
        or np.any(action > upper + 1.0e-12)
    ):
        raise ValueError("action-conditioned feature input differs")
    base[:, PAIR_CANDIDATE_SLICE] = action[None, :]
    normalized = 2.0 * (action - lower) / (upper - lower) - 1.0
    output = np.concatenate(
        (base, np.repeat(normalized[None, :], 7, axis=0)), axis=1
    )
    if (
        output.shape != (7, len(ACTION_FEATURE_NAMES))
        or not np.all(np.isfinite(output))
    ):
        raise ValueError("action-conditioned feature output differs")
    return output


def action_features_for_state(
    state: Mapping[str, Any], actions_xyz: Sequence[Sequence[float]],
) -> Any:
    np = _numpy()
    actions = np.asarray(actions_xyz, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError("action-conditioned action matrix differs")
    return np.concatenate([
        action_feature_matrix(
            state["pair_state_feature_vectors"], action,
            state["action_lower"], state["action_upper"],
        ) for action in actions
    ], axis=0)


def value_training_arrays(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Flatten the immutable 125-action grids without regional duplication."""

    np = _numpy()
    features = []
    margins = []
    splits = []
    state_indexes = []
    action_indexes = []
    constraint_indexes = []
    for state in states:
        xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
        exact = np.asarray(
            state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        if xyz.shape != (125, 3) or exact.shape != (125, 7):
            raise ValueError("action-conditioned fit grid differs")
        rows = action_features_for_state(state, xyz).reshape(125, 7, -1)
        for action_index in range(125):
            for constraint_index in range(7):
                features.append(rows[action_index, constraint_index])
                margins.append(exact[action_index, constraint_index])
                splits.append(state["split"])
                state_indexes.append(int(state["state_index"]))
                action_indexes.append(action_index)
                constraint_indexes.append(constraint_index)
    output = {
        "features": np.asarray(features, dtype=np.float64),
        "exact_margin_m": np.asarray(margins, dtype=np.float64),
        "split": np.asarray(splits, dtype=object),
        "state_index": np.asarray(state_indexes, dtype=np.int64),
        "action_index": np.asarray(action_indexes, dtype=np.int64),
        "constraint_index": np.asarray(constraint_indexes, dtype=np.int64),
    }
    count = len(states) * 125 * 7
    if (
        output["features"].shape != (count, len(ACTION_FEATURE_NAMES))
        or output["exact_margin_m"].shape != (count,)
        or not np.all(np.isfinite(output["features"]))
        or not np.all(np.isfinite(output["exact_margin_m"]))
    ):
        raise ValueError("action-conditioned training arrays differ")
    return output


def build_model(hidden_widths: Sequence[int]) -> Any:
    torch = _torch()
    nn = torch.nn

    class ActionMarginNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            previous = len(ACTION_FEATURE_NAMES)
            for width in hidden_widths:
                layers.extend((nn.Linear(previous, int(width)), nn.SiLU()))
                previous = int(width)
            layers.append(nn.Linear(previous, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, values: Any) -> Any:
            return self.network(values).squeeze(-1)

    return ActionMarginNet()


def train_ensemble(
    arrays: Mapping[str, Any], config: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Train the preregistered grouped action-value ensemble."""

    import time

    np = _numpy()
    torch = _torch()
    torch.set_num_threads(8)
    features = np.asarray(arrays["features"], dtype=np.float64)
    exact_mm = np.asarray(arrays["exact_margin_m"], dtype=np.float64) * 1000.0
    splits = np.asarray(arrays["split"], dtype=object)
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    if not np.any(train_mask) or not np.any(validation_mask):
        raise ValueError("action-conditioned grouped split is empty")
    mean = np.mean(features[train_mask], axis=0)
    std = np.maximum(np.std(features[train_mask], axis=0), 1.0e-6)
    normalized = (features - mean) / std
    current_mm = features[:, CURRENT_CLEARANCE_INDEX] * 1000.0
    residual_mm = exact_mm - current_mm
    output_mean = float(np.mean(residual_mm[train_mask]))
    output_scale = max(float(np.std(residual_mm[train_mask])), 1.0)
    settings = config["training"]
    boundary = np.where(
        np.abs(exact_mm) <= float(settings["boundary_band_m"]) * 1000.0,
        float(settings["boundary_weight_multiplier"]), 1.0,
    )
    device = torch.device("cpu")
    x = torch.as_tensor(normalized, dtype=torch.float64, device=device)
    exact_t = torch.as_tensor(exact_mm, dtype=torch.float64, device=device)
    current_t = torch.as_tensor(current_mm, dtype=torch.float64, device=device)
    weight_t = torch.as_tensor(boundary, dtype=torch.float64, device=device)
    train_indexes = np.flatnonzero(train_mask)
    validation_indexes = torch.as_tensor(
        np.flatnonzero(validation_mask), dtype=torch.long, device=device
    )
    delta = float(settings["huber_delta_mm"])
    over_scale = float(settings["one_sided_scale_mm"])
    over_weight = float(settings["one_sided_overestimate_loss_weight"])

    def loss_for(model: Any, indexes: Any) -> Any:
        residual = model(x[indexes]) * output_scale + output_mean
        predicted = current_t[indexes] + residual
        error = predicted - exact_t[indexes]
        absolute = torch.abs(error)
        huber = torch.where(
            absolute <= delta,
            0.5 * error * error,
            delta * (absolute - 0.5 * delta),
        )
        weights = weight_t[indexes]
        regression = torch.sum(weights * huber) / torch.sum(weights)
        over = torch.relu(error / over_scale)
        one_sided = torch.sum(weights * over * over) / torch.sum(weights)
        return regression + over_weight * one_sided

    models = []
    member_audits = []
    started = time.perf_counter_ns()
    batch_size = int(settings["batch_size"])
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
        rng = np.random.RandomState(int(seed))
        best_state = None
        best_epoch = None
        best_validation = math.inf
        stale = 0
        history = []
        member_started = time.perf_counter_ns()
        for epoch in range(int(settings["epochs"])):
            model.train()
            permutation = rng.permutation(train_indexes)
            train_sum = 0.0
            batch_count = 0
            for offset in range(0, len(permutation), batch_size):
                indexes = torch.as_tensor(
                    permutation[offset:offset + batch_size],
                    dtype=torch.long, device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(model, indexes)
                loss.backward()
                optimizer.step()
                train_sum += float(loss.detach().cpu())
                batch_count += 1
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_for(model, validation_indexes).cpu()
                )
            if epoch == 0 or (epoch + 1) % 25 == 0:
                history.append({
                    "epoch": int(epoch),
                    "train_loss": train_sum / max(batch_count, 1),
                    "validation_loss": validation_loss,
                })
            if validation_loss < best_validation - 1.0e-10:
                best_validation = validation_loss
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
            raise RuntimeError("action-conditioned training produced no checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        member_audits.append({
            "seed": int(seed), "best_epoch": best_epoch,
            "completed_epoch_count": int(epoch + 1),
            "best_validation_objective": best_validation,
            "history": history,
            "training_wall_seconds": (
                time.perf_counter_ns() - member_started
            ) * 1.0e-9,
        })
    state = {
        "state_dicts": [{
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        } for model in models],
        "feature_mean": mean, "feature_standard_deviation": std,
        "output_mean_mm": output_mean, "output_scale_mm": output_scale,
        "hidden_widths": list(config["model"]["hidden_widths"]),
        "ensemble_seeds": list(config["model"]["ensemble_seeds"]),
        "calibration_m": np.zeros(7, dtype=np.float64),
    }
    audit = {
        "model_class": config["model"]["class"], "device": str(device),
        "torch_version": str(torch.__version__),
        "allocation_cuda_available": bool(torch.cuda.is_available()),
        "allocation_cuda_device_name": (
            None if not torch.cuda.is_available()
            else str(torch.cuda.get_device_name(0))
        ),
        "training_row_count": int(np.sum(train_mask)),
        "validation_grid_row_count": int(np.sum(validation_mask)),
        "parameter_count_per_member": int(sum(
            value.numel() for value in models[0].parameters()
        )),
        "members": member_audits,
        "training_wall_seconds": (
            time.perf_counter_ns() - started
        ) * 1.0e-9,
    }
    return models, state, audit


def predict_member_margins(
    models: Sequence[Any], model_state: Mapping[str, Any],
    state: Mapping[str, Any], actions_xyz: Sequence[Sequence[float]],
) -> Any:
    np = _numpy()
    torch = _torch()
    actions = np.asarray(actions_xyz, dtype=np.float64)
    features = action_features_for_state(state, actions)
    normalized = (
        features - np.asarray(model_state["feature_mean"], dtype=np.float64)
    ) / np.asarray(model_state["feature_standard_deviation"], dtype=np.float64)
    current_mm = features[:, CURRENT_CLEARANCE_INDEX] * 1000.0
    tensor = torch.as_tensor(normalized, dtype=torch.float64)
    output = []
    for model in models:
        model.eval()
        with torch.no_grad():
            residual = (
                model(tensor) * float(model_state["output_scale_mm"])
                + float(model_state["output_mean_mm"])
            ).cpu().numpy()
        output.append((current_mm + residual).reshape(len(actions), 7) / 1000.0)
    result = np.asarray(output, dtype=np.float64)
    if result.shape != (len(models), len(actions), 7):
        raise ValueError("action-conditioned prediction shape differs")
    return result


def guarded_margin_values(
    models: Sequence[Any], model_state: Mapping[str, Any],
    state: Mapping[str, Any], actions_xyz: Sequence[Sequence[float]],
    uncertainty: Mapping[str, Any], *, calibrated: bool = True,
) -> dict[str, Any]:
    np = _numpy()
    members = predict_member_margins(models, model_state, state, actions_xyz)
    mean = np.mean(members, axis=0)
    guard = float(uncertainty["standard_deviation_multiplier"]) * np.std(
        members, axis=0
    )
    calibration = (
        np.asarray(model_state["calibration_m"], dtype=np.float64)[None, :]
        if calibrated else np.zeros((1, 7), dtype=np.float64)
    )
    return {
        "members_m": members, "mean_m": mean, "guard_m": guard,
        "lower_m": mean - guard - calibration,
    }


def calibrate_validation(
    models: Sequence[Any], model_state: dict[str, Any],
    states: Sequence[Mapping[str, Any]], oracle_states: Sequence[Mapping[str, Any]],
    uncertainty: Mapping[str, Any],
) -> dict[str, Any]:
    """Maximum validation overestimate after pointwise ensemble guarding."""

    np = _numpy()
    oracle = {int(item["state_index"]): item for item in oracle_states}
    maximum = np.full(7, -np.inf, dtype=np.float64)
    fit_count = 0
    off_grid_count = 0
    validation_indexes = []
    for state in states:
        if state["split"] != "validation":
            continue
        validation_indexes.append(int(state["state_index"]))
        xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
        exact = np.asarray(
            state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        predicted = guarded_margin_values(
            models, model_state, state, xyz, uncertainty, calibrated=False,
        )["lower_m"]
        maximum = np.maximum(maximum, np.max(predicted - exact, axis=0))
        fit_count += len(xyz)
        fresh = oracle[int(state["state_index"])].get("fresh_actions", [])
        if len(fresh) != 96:
            raise ValueError("action-conditioned validation off-grid count differs")
        fresh_xyz = np.asarray(
            [item["candidate_xyz"] for item in fresh], dtype=np.float64
        )
        fresh_exact = np.asarray(
            [item["minimum_distal_margin_m"] for item in fresh],
            dtype=np.float64,
        )
        fresh_predicted = guarded_margin_values(
            models, model_state, state, fresh_xyz, uncertainty, calibrated=False,
        )["lower_m"]
        maximum = np.maximum(
            maximum, np.max(fresh_predicted - fresh_exact, axis=0)
        )
        off_grid_count += len(fresh)
    if not np.all(np.isfinite(maximum)) or len(validation_indexes) != 10:
        raise ValueError("action-conditioned validation calibration lacks support")
    calibration = np.maximum(maximum, 0.0) + float(
        uncertainty["fixed_padding_m"]
    )
    model_state["calibration_m"] = calibration
    return {
        "calibration_m": calibration.tolist(),
        "maximum_m": float(np.max(calibration)),
        "mean_m": float(np.mean(calibration)),
        "fit_action_count": fit_count,
        "off_grid_action_count": off_grid_count,
        "validation_state_indexes": sorted(validation_indexes),
    }


def learned_regional_targets(
    models: Sequence[Any], model_state: Mapping[str, Any],
    state: Mapping[str, Any], regions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Query 125 values, then deterministically fit 27 conservative rows."""

    np = _numpy()
    xyz = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
    values = guarded_margin_values(
        models, model_state, state, xyz, config["uncertainty"], calibrated=True,
    )
    lower = np.asarray(values["lower_m"], dtype=np.float64)
    targets = []
    for region in regions:
        fitted = fit_region_target(
            xyz, lower, region, config["ridge_huber"]
        )
        fitted["lower_anchor_m"] = (
            np.asarray(fitted["anchor_margin_m"], dtype=np.float64)
            - np.asarray(fitted["one_sided_error_m"], dtype=np.float64)
        ).tolist()
        fitted["value_source"] = "guarded_action_conditioned_grid"
        targets.append(fitted)
    return targets, {
        "grid_mean_m": values["mean_m"],
        "grid_guard_m": values["guard_m"],
        "grid_lower_m": lower,
    }


def target_values(target: Mapping[str, Any], xyz: Sequence[float]) -> Any:
    return region_affine_values(target, xyz)


def save_model(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    names = list(state["state_dicts"][0])
    metadata = {
        "schema_version": WEIGHTS_SCHEMA,
        "feature_names": list(ACTION_FEATURE_NAMES),
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
        "output_mean_mm": np.asarray([state["output_mean_mm"]], dtype=np.float64),
        "output_scale_mm": np.asarray([state["output_scale_mm"]], dtype=np.float64),
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
    """Load an immutable action-conditioned ensemble artifact."""

    np = _numpy()
    torch = _torch()
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(
            bytes(archive["metadata_utf8"].tolist()).decode("utf-8")
        )
        if (
            metadata.get("schema_version") != WEIGHTS_SCHEMA
            or metadata.get("feature_names") != list(ACTION_FEATURE_NAMES)
            or metadata.get("constraint_order") != list(CONSTRAINT_ORDER)
        ):
            raise ValueError("action-conditioned model identity differs")
        names = metadata["parameter_names"]
        models = []
        for member_index, _ in enumerate(metadata["ensemble_seeds"]):
            model = build_model(metadata["hidden_widths"]).to(dtype=torch.float64)
            if names != list(model.state_dict()):
                raise ValueError("action-conditioned parameter names differ")
            member = {}
            for parameter_index, name in enumerate(names):
                values = np.asarray(
                    archive["member_%02d_parameter_%03d" % (
                        member_index, parameter_index
                    )], dtype=np.float64,
                )
                if values.shape != tuple(model.state_dict()[name].shape):
                    raise ValueError("action-conditioned parameter shape differs")
                member[name] = torch.as_tensor(values, dtype=torch.float64)
            model.load_state_dict(member)
            model.eval()
            models.append(model)
        state = {
            "feature_mean": np.asarray(archive["feature_mean"], dtype=np.float64),
            "feature_standard_deviation": np.asarray(
                archive["feature_standard_deviation"], dtype=np.float64
            ),
            "output_mean_mm": float(np.asarray(
                archive["output_mean_mm"], dtype=np.float64
            )[0]),
            "output_scale_mm": float(np.asarray(
                archive["output_scale_mm"], dtype=np.float64
            )[0]),
            "calibration_m": np.asarray(
                archive["calibration_m"], dtype=np.float64
            ),
            "hidden_widths": list(metadata["hidden_widths"]),
            "ensemble_seeds": list(metadata["ensemble_seeds"]),
        }
    return models, state


def jaccard(left: Sequence[bool], right: Sequence[bool]) -> float:
    np = _numpy()
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = int(np.sum(np.logical_or(a, b)))
    return 1.0 if union == 0 else float(np.sum(np.logical_and(a, b)) / union)


__all__ = [
    "ACTION_FEATURE_NAMES", "CONFIG_SCHEMA", "CURRENT_CLEARANCE_INDEX", "RESULT_SCHEMA",
    "VALIDATION_SCHEMA", "action_feature_matrix", "action_features_for_state",
    "build_model", "calibrate_validation", "guarded_margin_values", "jaccard",
    "learned_regional_targets", "load_config", "load_model",
    "predict_member_margins",
    "save_model", "solve_regional_qps", "target_values", "train_ensemble",
    "value_training_arrays",
]
