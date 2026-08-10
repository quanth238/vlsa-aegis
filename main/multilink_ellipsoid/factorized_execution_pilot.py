"""Factorized two-action OSC execution pilot.

This opt-in module learns either the complete 51-state arm-joint rollout or
the seven rollout-minimum ellipsoid margins from exactly the same state/action
population.  It contains no QP, calibration, Poisson/SDF, policy inference, or
closed-loop execution path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .complete_osc_margin import _canonical, _numpy, _torch


CONFIG_SCHEMA = "vlsa_distal_factorized_execution_moka10.v1"
DATASET_SCHEMA = "vlsa_distal_factorized_execution_dataset.v1"
COLLECTION_SCHEMA = "vlsa_distal_factorized_execution_collection_result.v1"
DATASET_VALIDATION_SCHEMA = "vlsa_distal_factorized_execution_dataset_validation.v1"
RESULT_SCHEMA = "vlsa_distal_factorized_execution_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_execution_validation.v1"
WEIGHTS_SCHEMA = "vlsa_distal_factorized_execution_weights.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha256(_canonical(payload))


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("factorized-execution config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "candidate_design", "rollout_target", "paired_models", "model",
        "training", "decision_gate", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("factorized-execution config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"] != "vlsa-distal-factorized-execution-moka10-v1"
    ):
        raise ValueError("factorized-execution protocol differs")
    source = config["immutable_source"]
    if (
        int(source["expected_episode_count"]) != 17
        or int(source["expected_state_count"]) != 85
        or source["expected_state_split_counts"]
        != {"train": 60, "validation": 10, "test": 15}
        or int(source["complete_state_input_dimension"]) != 2110
        or source["test_case_ids"] != [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ]
    ):
        raise ValueError("factorized-execution immutable population differs")
    design = config["candidate_design"]
    if (
        design["action_chunk_shape"] != [2, 7]
        or int(design["flattened_action_dimension"]) != 14
        or len(design["action_coordinate_order"]) != 14
        or float(design["pose_action_limit"]) != 1.0
        or design["gripper_clipping"]
        != "none_preserve_archived_center_and_local_perturbation"
        or design["finite_difference_step"]
        != {"translation": 0.05, "rotation": 0.05, "gripper": 0.25}
        or int(design["random_antithetic_pair_count"]) != 32
        or design["random_linf_radius"]
        != {"translation": 0.10, "rotation": 0.10, "gripper": 0.25}
        or int(design["expected_candidate_count_per_state"]) != 93
        or int(design["expected_rollout_count"]) != 7905
        or design["repeat_candidate_indexes"] != [0, 1, 2, 7, 8]
        or int(design["expected_repeat_rollout_count"]) != 425
    ):
        raise ValueError("factorized-execution candidate design differs")
    target = config["rollout_target"]
    if (
        int(target["horizon_actions"]) != 2
        or int(target["expected_internal_mujoco_steps_per_action"]) != 25
        or int(target["joint_trace_state_count"]) != 51
        or len(target["joint_order"]) != 7
    ):
        raise ValueError("factorized-execution rollout target differs")
    paired = config["paired_models"]
    if (
        "both_complete_7D_actions" not in paired["common_input"]
        or paired["same_state_action_rows_episode_splits_trunk_widths_seeds_optimizer_and_schedule"]
        is not True
    ):
        raise ValueError("factorized-execution paired arms differ")
    model = config["model"]
    if (
        model["ensemble_seeds"]
        != [20260841, 20260842, 20260843, 20260844, 20260845]
        or model["hidden_widths"] != [256, 256, 128]
        or model["activation"] != "silu"
    ):
        raise ValueError("factorized-execution model differs")
    training = config["training"]
    if (
        training["device"]
        != "cpu_inside_H100_allocation_due_to_pinned_PyTorch_missing_sm90_kernels"
        or int(training["batch_size"]) != 512
        or int(training["epochs"]) != 300
        or int(training["patience"]) != 40
        or training["early_stopping_split"] != "validation_complete_episodes"
    ):
        raise ValueError("factorized-execution training differs")
    forbidden = config["forbidden_actions"]
    if set(forbidden) != {
        "poisson_or_SDF", "calibration", "QP", "closed_loop_E05",
        "VLA_policy_inference", "whole_body_or_invariance_claim",
    } or not all(value is True for value in forbidden.values()):
        raise ValueError("factorized-execution forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _coordinate_kind(flat_index: int) -> str:
    local = int(flat_index) % 7
    if local < 3:
        return "translation"
    if local < 6:
        return "rotation"
    return "gripper"


def candidate_chunks(
    nominal_actions: Sequence[Sequence[float]], state_index: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the registered 14D star plus antithetic local population."""

    np = _numpy()
    nominal = np.asarray(nominal_actions, dtype=np.float64)
    if nominal.shape != (2, 7) or not np.all(np.isfinite(nominal)):
        raise ValueError("factorized-execution nominal action chunk differs")
    design = config["candidate_design"]
    limit = float(design["pose_action_limit"])
    pose_indexes = np.asarray([
        index for index in range(14) if _coordinate_kind(index) != "gripper"
    ], dtype=np.int64)
    nominal_flat = nominal.reshape(-1)
    if (
        np.any(nominal_flat[pose_indexes] < -limit)
        or np.any(nominal_flat[pose_indexes] > limit)
    ):
        raise ValueError("factorized-execution nominal pose action exceeds bounds")
    flat = nominal.reshape(-1)
    records: list[dict[str, Any]] = [{
        "candidate_index": 0, "source": "nominal",
        "finite_difference_dimension": None, "finite_difference_sign": None,
        "full_two_action_commands": nominal.tolist(),
    }]
    for dimension in range(14):
        kind = _coordinate_kind(dimension)
        step = float(design["finite_difference_step"][kind])
        for sign in (-1, 1):
            value = flat.copy()
            value[dimension] = value[dimension] + sign * step
            if kind != "gripper":
                value[dimension] = np.clip(value[dimension], -limit, limit)
            if value[dimension] == flat[dimension]:
                # At an action bound the outward probe is the nominal point;
                # the opposite probe still supplies a valid one-sided secant.
                source = "finite_difference_bound_nominal"
            else:
                source = "finite_difference"
            records.append({
                "candidate_index": len(records), "source": source,
                "finite_difference_dimension": int(dimension),
                "finite_difference_sign": int(sign),
                "full_two_action_commands": value.reshape(2, 7).tolist(),
            })
    rng = np.random.default_rng(
        int(design["random_seed"]) + 1000003 * int(state_index)
    )
    radii = np.asarray([
        float(design["random_linf_radius"][_coordinate_kind(index)])
        for index in range(14)
    ], dtype=np.float64)
    for pair_index in range(int(design["random_antithetic_pair_count"])):
        direction = rng.uniform(-1.0, 1.0, size=14)
        norm = float(np.max(np.abs(direction)))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError("factorized-execution random direction differs")
        offset = direction * radii / norm
        for sign in (-1, 1):
            value = flat + sign * offset
            value[pose_indexes] = np.clip(
                value[pose_indexes], -limit, limit
            )
            records.append({
                "candidate_index": len(records),
                "source": "random_antithetic",
                "random_pair_index": int(pair_index),
                "random_sign": int(sign),
                "finite_difference_dimension": None,
                "finite_difference_sign": None,
                "full_two_action_commands": value.reshape(2, 7).tolist(),
            })
    expected = int(design["expected_candidate_count_per_state"])
    if len(records) != expected:
        raise ValueError("factorized-execution candidate count differs")
    for index, record in enumerate(records):
        if int(record["candidate_index"]) != index:
            raise ValueError("factorized-execution candidate order differs")
    return records


def finite_difference_candidate_indexes(dimension: int) -> tuple[int, int]:
    if not 0 <= int(dimension) < 14:
        raise ValueError("factorized-execution finite-difference dimension differs")
    return 1 + 2 * int(dimension), 2 + 2 * int(dimension)


def trace_arrays(chunk: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the nonduplicated 51-state q/ellipsoid trace."""

    np = _numpy()
    transitions = list(chunk.get("transitions", []))
    if len(transitions) != 2:
        raise ValueError("factorized-execution chunk length differs")
    expected_internal = int(
        config["rollout_target"]["expected_internal_mujoco_steps_per_action"]
    )
    for transition in transitions:
        if (
            int(transition["expected_internal_mujoco_step_count"])
            != expected_internal
            or int(transition["captured_state_count"]) != expected_internal + 1
        ):
            raise ValueError("factorized-execution internal trace length differs")
    trace = list(transitions[0]["substeps"]) + list(transitions[1]["substeps"])[1:]
    q = np.asarray(
        [item["robot_joint_position_rad"] for item in trace], dtype=np.float64
    )
    clearance = np.asarray(
        [item["clearance_m"][:7] for item in trace], dtype=np.float64
    )
    expected_states = int(config["rollout_target"]["joint_trace_state_count"])
    if (
        q.shape != (expected_states, 7)
        or clearance.shape != (expected_states, 7)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(clearance))
    ):
        raise ValueError("factorized-execution trace arrays differ")
    contacts = sum(
        int(item["raw_protected_contact_count"]) for item in transitions
    )
    return {
        "joint_position_rad": q,
        "ellipsoid_clearance_m": clearance,
        "minimum_ellipsoid_margin_m": np.min(clearance, axis=0),
        "raw_protected_contact_count": int(contacts),
        "next_state_sha256_per_action": [
            str(item["next_state_sha256"]) for item in transitions
        ],
        "maximum_obstacle_motion_m": float(max(
            item["maximum_within_step_obstacle_l1_displacement_m"]
            for item in transitions
        )),
    }


def dataset_arrays(
    metadata: Mapping[str, Any], archive: Mapping[str, Any],
) -> dict[str, Any]:
    """Build paired state/action learning arrays from validated artifacts."""

    np = _numpy()
    state_vectors = np.asarray(archive["state_input_vector"], dtype=np.float64)
    actions = np.asarray(archive["action_chunk"], dtype=np.float64)
    q = np.asarray(archive["joint_position_rad"], dtype=np.float64)
    clearance = np.asarray(archive["ellipsoid_clearance_m"], dtype=np.float64)
    state_index = np.asarray(archive["state_index"], dtype=np.int64)
    candidate_index = np.asarray(archive["candidate_index"], dtype=np.int64)
    split_code = np.asarray(archive["split_code"], dtype=np.int8)
    source_code = np.asarray(archive["source_code"], dtype=np.int8)
    count = int(metadata["summary"]["rollout_count"])
    trace_count = int(metadata["summary"]["joint_trace_state_count"])
    state_dim = int(metadata["summary"]["complete_state_input_dimension"])
    if (
        state_vectors.shape != (count, state_dim)
        or actions.shape != (count, 2, 7)
        or q.shape != (count, trace_count, 7)
        or clearance.shape != (count, trace_count, 7)
        or any(value.shape != (count,) for value in (
            state_index, candidate_index, split_code, source_code,
        ))
        or not all(np.all(np.isfinite(value)) for value in (
            state_vectors, actions, q, clearance,
        ))
    ):
        raise ValueError("factorized-execution dataset arrays differ")
    features = np.concatenate((state_vectors, actions.reshape(count, 14)), axis=1)
    split_names = np.asarray(["train", "validation", "test"], dtype=object)
    if np.any(split_code < 0) or np.any(split_code > 2):
        raise ValueError("factorized-execution split codes differ")
    return {
        "features": features,
        "action_chunk": actions,
        "joint_position_rad": q,
        "ellipsoid_clearance_m": clearance,
        "minimum_margin_m": np.min(clearance, axis=1),
        "current_margin_m": clearance[:, 0],
        "state_index": state_index,
        "candidate_index": candidate_index,
        "split": split_names[split_code],
        "source_code": source_code,
        "raw_contact_count": np.asarray(
            archive["raw_contact_count"], dtype=np.int64
        ),
    }


def sensitivity_arrays(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Return actual clipped finite-difference q and margin sensitivities."""

    np = _numpy()
    states = sorted(set(np.asarray(arrays["state_index"], dtype=np.int64).tolist()))
    q_rows = []
    margin_rows = []
    state_rows = []
    dimension_rows = []
    negative_rows = []
    positive_rows = []
    for state in states:
        state_mask = np.flatnonzero(np.asarray(arrays["state_index"]) == state)
        by_candidate = {
            int(np.asarray(arrays["candidate_index"])[row]): int(row)
            for row in state_mask
        }
        for dimension in range(14):
            negative_index, positive_index = finite_difference_candidate_indexes(
                dimension
            )
            negative = by_candidate[negative_index]
            positive = by_candidate[positive_index]
            action_negative = np.asarray(
                arrays["action_chunk"][negative], dtype=np.float64
            ).reshape(-1)
            action_positive = np.asarray(
                arrays["action_chunk"][positive], dtype=np.float64
            ).reshape(-1)
            denominator = float(
                action_positive[dimension] - action_negative[dimension]
            )
            if denominator <= 0.0:
                raise ValueError("factorized-execution FD denominator is nonpositive")
            changed = np.flatnonzero(
                np.abs(action_positive - action_negative) > 1.0e-15
            )
            if changed.tolist() != [dimension]:
                raise ValueError("factorized-execution FD pair changed multiple coordinates")
            q_rows.append(
                (arrays["joint_position_rad"][positive]
                 - arrays["joint_position_rad"][negative]) / denominator
            )
            margin_rows.append(
                (arrays["minimum_margin_m"][positive]
                 - arrays["minimum_margin_m"][negative]) / denominator
            )
            state_rows.append(int(state))
            dimension_rows.append(int(dimension))
            negative_rows.append(int(negative))
            positive_rows.append(int(positive))
    q_sensitivity = np.asarray(q_rows, dtype=np.float64)
    margin_sensitivity = np.asarray(margin_rows, dtype=np.float64)
    if (
        q_sensitivity.shape[1:] != arrays["joint_position_rad"].shape[1:]
        or margin_sensitivity.shape[1:] != (7,)
        or not np.all(np.isfinite(q_sensitivity))
        or not np.all(np.isfinite(margin_sensitivity))
    ):
        raise ValueError("factorized-execution sensitivity arrays differ")
    return {
        "joint_sensitivity_rad_per_action": q_sensitivity,
        "margin_sensitivity_m_per_action": margin_sensitivity,
        "state_index": np.asarray(state_rows, dtype=np.int64),
        "dimension_index": np.asarray(dimension_rows, dtype=np.int64),
        "negative_row_index": np.asarray(negative_rows, dtype=np.int64),
        "positive_row_index": np.asarray(positive_rows, dtype=np.int64),
    }


def _build_model(input_count: int, output_count: int, widths: Sequence[int]) -> Any:
    torch = _torch()
    layers = []
    previous = int(input_count)
    for width in widths:
        layers.extend((torch.nn.Linear(previous, int(width)), torch.nn.SiLU()))
        previous = int(width)
    layers.append(torch.nn.Linear(previous, int(output_count)))
    return torch.nn.Sequential(*layers)


def _arm_targets(arrays: Mapping[str, Any], arm: str) -> tuple[Any, Any, float]:
    np = _numpy()
    if arm == "direct_margin":
        target = np.asarray(arrays["minimum_margin_m"], dtype=np.float64)
        base = np.asarray(arrays["current_margin_m"], dtype=np.float64)
        return target, base, 1.0
    if arm == "factorized_execution":
        q = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
        base = np.repeat(q[:, :1, :], q.shape[1], axis=1)
        return q, base, 1.0
    raise ValueError("factorized-execution arm differs")


def train_ensemble(
    arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
    config: Mapping[str, Any], *, arm: str,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Train a matched arm with value and explicit secant sensitivity loss."""

    np = _numpy()
    torch = _torch()
    # The pinned evaluation PyTorch lacks sm_90 kernels. Simulation remains
    # inside the H100 allocation; both matched arms use its registered CPUs.
    device = torch.device("cpu")
    torch.set_num_threads(8)
    x = np.asarray(arrays["features"], dtype=np.float64)
    target, base, _ = _arm_targets(arrays, arm)
    split = np.asarray(arrays["split"], dtype=object)
    train_mask = split == "train"
    validation_mask = split == "validation"
    mean = np.mean(x[train_mask], axis=0)
    std = np.maximum(np.std(x[train_mask], axis=0), 1.0e-6)
    normalized = (x - mean) / std
    residual = target - base
    output_shape = residual.shape[1:]
    output_count = int(np.prod(output_shape))
    exact_min = np.min(np.asarray(arrays["minimum_margin_m"]), axis=1)
    settings = config["training"]
    boundary = float(settings["near_boundary_absolute_margin_m"])
    sample_weight = np.where(
        np.abs(exact_min) <= boundary,
        float(settings["near_boundary_weight_multiplier"]), 1.0,
    )
    sensitivity_key = (
        "margin_sensitivity_m_per_action"
        if arm == "direct_margin" else "joint_sensitivity_rad_per_action"
    )
    exact_sensitivity = np.asarray(sensitivities[sensitivity_key], dtype=np.float64)
    sensitivity_states = np.asarray(sensitivities["state_index"], dtype=np.int64)
    state_split = {
        int(state): str(np.asarray(arrays["split"])[row])
        for row, state in enumerate(np.asarray(arrays["state_index"], dtype=np.int64))
    }
    sensitivity_split = np.asarray(
        [state_split[int(state)] for state in sensitivity_states], dtype=object
    )
    neg_rows = np.asarray(sensitivities["negative_row_index"], dtype=np.int64)
    pos_rows = np.asarray(sensitivities["positive_row_index"], dtype=np.int64)
    action_flat = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    dims = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    denominators = (
        action_flat[pos_rows, dims] - action_flat[neg_rows, dims]
    )
    train_indexes = np.flatnonzero(train_mask)
    validation_indexes = np.flatnonzero(validation_mask)
    train_sensitivity = np.flatnonzero(sensitivity_split == "train")
    validation_sensitivity = np.flatnonzero(sensitivity_split == "validation")
    delta = float(
        settings["direct_huber_delta_m"]
        if arm == "direct_margin" else settings["joint_huber_delta_rad"]
    )
    models = []
    states = []
    member_audits = []
    batch_size = int(settings["batch_size"])
    for seed in config["model"]["ensemble_seeds"]:
        torch.manual_seed(int(seed))
        model = _build_model(
            x.shape[1], output_count, config["model"]["hidden_widths"]
        ).to(device)
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
            for start in range(0, len(order), batch_size):
                indexes = order[start:start + batch_size]
                xb = torch.as_tensor(normalized[indexes], dtype=torch.float32, device=device)
                rb = torch.as_tensor(
                    residual[indexes].reshape(len(indexes), -1),
                    dtype=torch.float32, device=device,
                )
                prediction = model(xb)
                error = prediction - rb
                absolute = torch.abs(error)
                huber = torch.where(
                    absolute <= delta, 0.5 * error ** 2,
                    delta * (absolute - 0.5 * delta),
                ).mean(dim=1)
                weights = torch.as_tensor(
                    sample_weight[indexes], dtype=torch.float32, device=device
                )
                loss = torch.mean(weights * huber)
                if arm == "direct_margin":
                    over = torch.relu(error)
                    loss = loss + float(
                        settings["direct_overestimate_loss_weight"]
                    ) * torch.mean(weights[:, None] * over ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Explicitly supervise the action-to-output secant on every
            # training state.  This is not inferred from classification.
            model.train()
            selected = train_sensitivity
            negative_prediction = model(torch.as_tensor(
                normalized[neg_rows[selected]], dtype=torch.float32, device=device
            ))
            positive_prediction = model(torch.as_tensor(
                normalized[pos_rows[selected]], dtype=torch.float32, device=device
            ))
            denominator = torch.as_tensor(
                denominators[selected], dtype=torch.float32, device=device
            )[:, None]
            predicted_sensitivity = (
                positive_prediction - negative_prediction
            ) / denominator
            exact_s = torch.as_tensor(
                exact_sensitivity[selected].reshape(len(selected), -1),
                dtype=torch.float32, device=device,
            )
            sensitivity_loss = torch.nn.functional.smooth_l1_loss(
                predicted_sensitivity, exact_s, beta=delta
            )
            optimizer.zero_grad()
            (float(settings["finite_difference_loss_weight"]) * sensitivity_loss).backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_prediction = model(torch.as_tensor(
                    normalized[validation_indexes], dtype=torch.float32, device=device
                )).cpu().numpy().reshape((len(validation_indexes),) + output_shape)
                value_rmse = float(np.sqrt(np.mean(
                    (validation_prediction - residual[validation_indexes]) ** 2
                )))
                selected = validation_sensitivity
                negative_prediction = model(torch.as_tensor(
                    normalized[neg_rows[selected]], dtype=torch.float32, device=device
                ))
                positive_prediction = model(torch.as_tensor(
                    normalized[pos_rows[selected]], dtype=torch.float32, device=device
                ))
                predicted_s = (
                    (positive_prediction - negative_prediction).cpu().numpy()
                    / denominators[selected, None]
                )
                sensitivity_rmse = float(np.sqrt(np.mean(
                    (predicted_s - exact_sensitivity[selected].reshape(
                        len(selected), -1
                    )) ** 2
                )))
            score = value_rmse + 0.1 * sensitivity_rmse
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
            raise RuntimeError("factorized-execution training produced no checkpoint")
        model.load_state_dict(best)
        model.eval()
        models.append(model)
        states.append({key: value.numpy() for key, value in best.items()})
        member_audits.append({
            "seed": int(seed), "best_epoch": int(best_epoch),
            "validation_score": float(best_score),
            "completed_epoch_count": int(epoch + 1),
        })
    state = {
        "arm": arm, "feature_mean": mean, "feature_std": std,
        "input_dimension": int(x.shape[1]), "output_shape": list(output_shape),
        "hidden_widths": list(config["model"]["hidden_widths"]),
        "ensemble_seeds": list(config["model"]["ensemble_seeds"]),
        "model_states": states,
    }
    return models, state, {
        "arm": arm, "member_audits": member_audits,
        "training_row_count": int(len(train_indexes)),
        "training_sensitivity_row_count": int(len(train_sensitivity)),
        "validation_row_count": int(len(validation_indexes)),
        "validation_sensitivity_row_count": int(len(validation_sensitivity)),
    }


def predict(
    models: Sequence[Any], state: Mapping[str, Any], arrays: Mapping[str, Any],
) -> Any:
    np = _numpy()
    torch = _torch()
    device = next(models[0].parameters()).device
    normalized = (
        np.asarray(arrays["features"], dtype=np.float64) - state["feature_mean"]
    ) / state["feature_std"]
    outputs = []
    batch = 2048
    with torch.no_grad():
        for model in models:
            parts = []
            for start in range(0, len(normalized), batch):
                parts.append(model(torch.as_tensor(
                    normalized[start:start + batch], dtype=torch.float32,
                    device=device,
                )).cpu().numpy())
            outputs.append(np.concatenate(parts, axis=0))
    residual = np.mean(np.asarray(outputs), axis=0).reshape(
        (len(normalized),) + tuple(state["output_shape"])
    )
    _, base, _ = _arm_targets(arrays, str(state["arm"]))
    return base + residual


def save_weights(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    arrays: dict[str, Any] = {
        "schema_version": np.asarray(WEIGHTS_SCHEMA),
        "arm": np.asarray(str(state["arm"])),
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float64),
        "feature_std": np.asarray(state["feature_std"], dtype=np.float64),
        "input_dimension": np.asarray(state["input_dimension"], dtype=np.int64),
        "output_shape": np.asarray(state["output_shape"], dtype=np.int64),
        "hidden_widths": np.asarray(state["hidden_widths"], dtype=np.int64),
        "ensemble_seeds": np.asarray(state["ensemble_seeds"], dtype=np.int64),
    }
    for member, values in enumerate(state["model_states"]):
        for key, value in values.items():
            arrays["member_%d__%s" % (member, key)] = np.asarray(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    raw = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "file_sha256": _sha256(raw)}


def load_weights(path: Path) -> tuple[list[Any], dict[str, Any]]:
    np = _numpy()
    archive = np.load(path, allow_pickle=False)
    if str(archive["schema_version"].item()) != WEIGHTS_SCHEMA:
        raise ValueError("factorized-execution weights schema differs")
    arm = str(archive["arm"].item())
    input_dimension = int(archive["input_dimension"].item())
    output_shape = archive["output_shape"].astype(int).tolist()
    widths = archive["hidden_widths"].astype(int).tolist()
    seeds = archive["ensemble_seeds"].astype(int).tolist()
    torch = _torch()
    device = torch.device("cpu")
    models = []
    states = []
    for member in range(len(seeds)):
        model = _build_model(input_dimension, int(np.prod(output_shape)), widths)
        values = {}
        prefix = "member_%d__" % member
        for key in archive.files:
            if key.startswith(prefix):
                values[key[len(prefix):]] = torch.as_tensor(archive[key])
        model.load_state_dict(values)
        model.to(device).eval()
        models.append(model)
        states.append({key: value.cpu().numpy() for key, value in values.items()})
    return models, {
        "arm": arm,
        "feature_mean": archive["feature_mean"].astype(np.float64),
        "feature_std": archive["feature_std"].astype(np.float64),
        "input_dimension": input_dimension, "output_shape": output_shape,
        "hidden_widths": widths, "ensemble_seeds": seeds,
        "model_states": states,
    }


def cosine_summary(exact: Any, predicted: Any) -> dict[str, Any]:
    np = _numpy()
    truth = np.asarray(exact, dtype=np.float64).reshape(len(exact), -1)
    estimate = np.asarray(predicted, dtype=np.float64).reshape(len(predicted), -1)
    denominator = np.linalg.norm(truth, axis=1) * np.linalg.norm(estimate, axis=1)
    valid = denominator > 1.0e-12
    cosine = np.full(len(truth), np.nan, dtype=np.float64)
    cosine[valid] = np.sum(truth[valid] * estimate[valid], axis=1) / denominator[valid]
    return {
        "row_count": int(len(truth)), "valid_row_count": int(np.count_nonzero(valid)),
        "mean_cosine": None if not np.any(valid) else float(np.mean(cosine[valid])),
        "median_cosine": None if not np.any(valid) else float(np.median(cosine[valid])),
        "minimum_cosine": None if not np.any(valid) else float(np.min(cosine[valid])),
        "RMSE": float(np.sqrt(np.mean((estimate - truth) ** 2))),
    }


def secant_predictions(
    prediction: Any, sensitivities: Mapping[str, Any], arrays: Mapping[str, Any],
) -> Any:
    np = _numpy()
    negative = np.asarray(sensitivities["negative_row_index"], dtype=np.int64)
    positive = np.asarray(sensitivities["positive_row_index"], dtype=np.int64)
    dimensions = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    actions = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    denominator = actions[positive, dimensions] - actions[negative, dimensions]
    shape = (len(negative),) + (1,) * (np.asarray(prediction).ndim - 1)
    return (
        np.asarray(prediction)[positive] - np.asarray(prediction)[negative]
    ) / denominator.reshape(shape)


def safety_metrics(
    exact_margin: Any, predicted_margin: Any, arrays: Mapping[str, Any],
    config: Mapping[str, Any], *, split_name: str, evaluation_only: bool,
) -> dict[str, Any]:
    np = _numpy()
    exact = np.asarray(exact_margin, dtype=np.float64)
    predicted = np.asarray(predicted_margin, dtype=np.float64)
    split = np.asarray(arrays["split"], dtype=object)
    selected = split == split_name
    if evaluation_only:
        selected &= np.asarray(arrays["source_code"], dtype=np.int8) == 2
    if exact.shape != predicted.shape or exact.shape[1:] != (7,) or not np.any(selected):
        raise ValueError("factorized-execution safety metric population differs")
    state_ids = np.asarray(arrays["state_index"], dtype=np.int64)
    states = sorted(set(state_ids[selected].tolist()))
    predicted_safe = np.all(predicted[selected] >= 0.0, axis=1)
    exact_safe = np.all(exact[selected] >= 0.0, axis=1)
    false_safe = int(np.count_nonzero(predicted_safe & ~exact_safe))
    accepted_safe = int(np.count_nonzero(predicted_safe & exact_safe))
    safe_count = int(np.count_nonzero(exact_safe))
    support = 0
    for state in states:
        mask = selected & (state_ids == state)
        support += int(np.any(
            np.all(predicted[mask] >= 0.0, axis=1)
            & np.all(exact[mask] >= 0.0, axis=1)
        ))
    error = predicted[selected] - exact[selected]
    band = float(config["training"]["near_boundary_absolute_margin_m"])
    near = selected[:, None] & (np.abs(exact) <= band)
    return {
        "split": split_name, "evaluation_only": bool(evaluation_only),
        "state_count": len(states), "action_count": int(np.count_nonzero(selected)),
        "false_safe_action_count": false_safe,
        "exact_safe_action_count": safe_count,
        "accepted_exact_safe_action_count": accepted_safe,
        "exact_safe_action_recall": 0.0 if safe_count == 0 else float(accepted_safe / safe_count),
        "state_safe_support_count": int(support),
        "overall_RMSE_m": float(np.sqrt(np.mean(error ** 2))),
        "near_boundary_row_count": int(np.count_nonzero(near)),
        "near_boundary_RMSE_m": (
            None if not np.any(near) else
            float(np.sqrt(np.mean((predicted[near] - exact[near]) ** 2)))
        ),
    }


def factorized_decision(
    *, collection_pass: bool, direct_metrics: Mapping[str, Any],
    exact_geometry_metrics: Mapping[str, Any],
    factorized_metrics: Mapping[str, Any], joint_rmse_rad: float,
    link_center_rmse_m: float, joint_sensitivity_cosine: float,
    margin_sensitivity_cosine: float, config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["decision_gate"]
    tests = {
        "collection_determinism_and_pairing": bool(collection_pass),
        "exact_q_static_geometry_false_safe": int(
            exact_geometry_metrics["false_safe_action_count"]
        ) <= int(gate["maximum_exact_q_static_geometry_false_safe_actions"]),
        "exact_q_static_geometry_safe_recall": float(
            exact_geometry_metrics["exact_safe_action_recall"]
        ) >= float(gate["minimum_exact_q_static_geometry_safe_recall"]),
        "joint_trajectory_RMSE": float(joint_rmse_rad)
        <= float(gate["maximum_test_joint_trajectory_RMSE_rad"]),
        "link_center_RMSE": float(link_center_rmse_m)
        <= float(gate["maximum_test_link_center_RMSE_m"]),
        "joint_sensitivity_cosine": float(joint_sensitivity_cosine)
        >= float(gate["minimum_test_joint_sensitivity_cosine"]),
        "margin_sensitivity_cosine": float(margin_sensitivity_cosine)
        >= float(gate["minimum_test_action_space_margin_sensitivity_cosine"]),
        "false_safe_absolute": int(factorized_metrics["false_safe_action_count"])
        <= int(gate["maximum_factorized_test_false_safe_actions"]),
        "beats_fresh_direct_false_safe": int(factorized_metrics["false_safe_action_count"])
        < int(direct_metrics["false_safe_action_count"]),
        "beats_prior_best_direct_false_safe": int(factorized_metrics["false_safe_action_count"])
        < 72,
        "safe_recall": float(factorized_metrics["exact_safe_action_recall"])
        >= float(gate["minimum_test_safe_action_recall"]),
        "safe_support": int(factorized_metrics["state_safe_support_count"])
        == int(gate["required_test_state_safe_support_count"]),
        "near_boundary_margin_RMSE": (
            factorized_metrics["near_boundary_RMSE_m"] is not None
            and float(factorized_metrics["near_boundary_RMSE_m"])
            <= float(gate["maximum_test_near_boundary_margin_RMSE_m"])
        ),
    }
    passed = bool(all(tests.values()))
    return {
        "gate_tests": tests, "factorization_GO": passed,
        "poisson_or_SDF_authorized": passed,
        "calibration_QP_or_closed_loop_authorized": False,
        "conclusion": (
            "factorized_execution_improves_boundary_prediction"
            if passed else "factorized_execution_pilot_NO_GO_or_inconclusive"
        ),
    }
