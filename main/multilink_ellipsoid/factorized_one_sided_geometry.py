"""One-sided geometry supervision for the factorized OSC pilot.

The learned output remains the 51-by-7 joint trajectory.  During training, a
frozen per-state local Jacobian of the known ellipsoid clearance converts joint
prediction error into signed clearance error.  This supplies a differentiable
normal-direction loss without introducing a learned collision classifier or a
second safety output at deployment.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .complete_osc_margin import _canonical, _numpy, _torch
from .factorized_execution_pilot import _arm_targets, _build_model


CONFIG_SCHEMA = "vlsa_distal_factorized_one_sided_geometry_config.v1"
CONFIG_SCHEMA_V2 = "vlsa_distal_factorized_one_sided_geometry_config.v2"
RESULT_SCHEMA = "vlsa_distal_factorized_one_sided_geometry_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_one_sided_geometry_validation.v1"


def load_one_sided_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("one-sided geometry config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "validation_pattern_gate", "local_geometry_jacobian",
        "matched_ablation", "training", "decision_gate", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("one-sided geometry config keys differ")
    protocol = str(config["protocol_id"])
    if (config["schema_version"], protocol) not in {
        (CONFIG_SCHEMA, "vlsa-distal-factorized-one-sided-geometry-moka10-v1"),
        (CONFIG_SCHEMA_V2, "vlsa-distal-factorized-one-sided-geometry-moka10-v2"),
    }:
        raise ValueError("one-sided geometry protocol differs")
    if config["population"] != {
        "episode_count": 17,
        "state_count": 85,
        "state_split_counts": {"train": 60, "validation": 10, "test": 15},
        "candidate_count_per_state": 93,
        "joint_trace_state_count": 51,
        "constraint_count": 7,
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
    }:
        raise ValueError("one-sided geometry population differs")
    expected_pattern = (
        {
            "source_model": "immutable_job_37980_factorized_execution",
            "minimum_validation_false_safe_action_count": 1,
            "minimum_terminal_L5_fraction": 0.5,
            "terminal_substep_index": 50,
            "L5_constraint_rows": [0, 1, 2],
            "train_only_if_both_pass": True,
        }
        if protocol.endswith("-v1") else
        {
            "source_model": "immutable_job_37980_factorized_execution",
            "adaptation_source": "validated_v1_audit_job_38064",
            "minimum_validation_false_safe_action_count": 1,
            "minimum_terminal_distal_fraction": 0.5,
            "terminal_substep_index": 50,
            "distal_constraint_rows": [0, 1, 2, 3, 4, 5, 6],
            "train_only_if_both_pass": True,
        }
    )
    if config["validation_pattern_gate"] != expected_pattern:
        raise ValueError("one-sided validation pattern gate differs")
    if config["local_geometry_jacobian"] != {
        "fit_rows": "nominal_plus_28_full_action_finite_difference_candidates",
        "fit_state_splits": ["train", "validation"],
        "test_split_used_for_fit": False,
        "center_on_nominal_candidate": True,
        "ridge_regularization": 1e-06,
        "target": "seven_per_step_ellipsoid_clearances",
        "shape": [85, 51, 7, 7],
        "maximum_validation_random_linearization_RMSE_m": 0.002,
    }:
        raise ValueError("one-sided local geometry Jacobian differs")
    matched = config["matched_ablation"]
    if matched != {
        "baseline": "immutable_job_37980_factorized_execution_weights",
        "experimental": "same_factorized_joint_model_plus_one_sided_geometry_loss",
        "same_dataset_splits_features_architecture_seeds_optimizer_schedule": True,
        "only_training_difference": "one_sided_per_step_local_geometry_loss",
        "inference_output": "51_by_7_joint_trajectory_only",
        "deployment_safety": "known_FK_plus_existing_ellipsoid_geometry",
    }:
        raise ValueError("one-sided matched ablation differs")
    training = config["training"]
    if (
        training["ensemble_seeds"]
        != [20260841, 20260842, 20260843, 20260844, 20260845]
        or training["hidden_widths"] != [256, 256, 128]
        or int(training["batch_size"]) != 512
        or int(training["epochs"]) != 300
        or int(training["patience"]) != 40
        or float(training["joint_huber_delta_rad"]) != 0.002
        or float(training["geometry_huber_delta_m"]) != 0.002
        or float(training["finite_difference_loss_weight"]) != 1.0
        or float(training["geometry_loss_weight"]) != 10.0
        or float(training["dangerous_overestimate_weight"]) != 4.0
        or float(training["near_boundary_absolute_margin_m"]) != 0.005
        or float(training["near_boundary_weight_multiplier"]) != 5.0
        or float(training["terminal_weight_multiplier"]) != 3.0
        or int(training["terminal_weight_power"]) != 4
    ):
        raise ValueError("one-sided geometry training differs")
    forbidden = config["forbidden_actions"]
    expected_forbidden = {
        "new_simulation_labels": True, "surface_position_loss": True,
        "binary_classifier": True, "uncertainty_calibration": True,
        "poisson_or_SDF": True, "QP": True, "closed_loop": True,
    }
    if forbidden != expected_forbidden:
        raise ValueError("one-sided geometry forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
    return output


def fit_local_geometry_jacobians(
    arrays: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit dh/dq from only nominal and registered star probes per state.

    The fit never sees a test episode.  Its random-antithetic residual is an
    out-of-fit diagnostic of whether the signed local geometry loss is usable.
    """

    np = _numpy()
    q = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
    clearance = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    state_index = np.asarray(arrays["state_index"], dtype=np.int64)
    candidate_index = np.asarray(arrays["candidate_index"], dtype=np.int64)
    source_code = np.asarray(arrays["source_code"], dtype=np.int8)
    split = np.asarray(arrays["split"], dtype=object)
    if q.shape[1:] != (51, 7) or clearance.shape != q.shape:
        raise ValueError("one-sided geometry trace shape differs")
    state_count = int(config["population"]["state_count"])
    jacobian = np.full((state_count, 51, 7, 7), np.nan, dtype=np.float64)
    nominal_q = np.full((state_count, 51, 7), np.nan, dtype=np.float64)
    nominal_h = np.full((state_count, 51, 7), np.nan, dtype=np.float64)
    allowed_splits = set(config["local_geometry_jacobian"]["fit_state_splits"])
    ridge = float(config["local_geometry_jacobian"]["ridge_regularization"])
    audit_error = {name: [] for name in sorted(allowed_splits)}
    audit_exact_delta = {name: [] for name in sorted(allowed_splits)}
    state_splits: dict[int, str] = {}
    for state in range(state_count):
        rows = np.flatnonzero(state_index == state)
        if len(rows) != int(config["population"]["candidate_count_per_state"]):
            raise ValueError("one-sided geometry state population differs")
        unique_split = sorted(set(split[rows].tolist()))
        if len(unique_split) != 1:
            raise ValueError("one-sided geometry state crosses episode splits")
        state_split = str(unique_split[0])
        state_splits[state] = state_split
        if state_split not in allowed_splits:
            continue
        by_candidate = {int(candidate_index[row]): int(row) for row in rows}
        if sorted(by_candidate) != list(range(93)):
            raise ValueError("one-sided geometry candidate identity differs")
        nominal = by_candidate[0]
        fit_rows = np.asarray([by_candidate[index] for index in range(1, 29)])
        if not np.all(np.isin(source_code[fit_rows], [0, 1])):
            raise ValueError("one-sided geometry fit row source differs")
        q0 = q[nominal]
        h0 = clearance[nominal]
        nominal_q[state] = q0
        nominal_h[state] = h0
        for substep in range(51):
            x = q[fit_rows, substep] - q0[substep]
            y = clearance[fit_rows, substep] - h0[substep]
            gram = x.T @ x + ridge * np.eye(7, dtype=np.float64)
            coefficient = np.linalg.solve(gram, x.T @ y)
            jacobian[state, substep] = coefficient.T
        random_rows = np.asarray([
            row for row in rows if int(source_code[row]) == 2
        ], dtype=np.int64)
        delta_q = q[random_rows] - q0[None, :, :]
        predicted_delta = np.einsum(
            "krj,nkj->nkr", jacobian[state], delta_q,
        )
        exact_delta = clearance[random_rows] - h0[None, :, :]
        audit_error[state_split].append((predicted_delta - exact_delta).reshape(-1))
        audit_exact_delta[state_split].append(exact_delta.reshape(-1))
    fit_mask = np.isfinite(jacobian).all(axis=(1, 2, 3))
    expected_fit = sum(
        count for name, count in config["population"]["state_split_counts"].items()
        if name in allowed_splits
    )
    if int(np.count_nonzero(fit_mask)) != int(expected_fit):
        raise ValueError("one-sided geometry fitted state count differs")
    audits = {}
    for name in sorted(allowed_splits):
        error = np.concatenate(audit_error[name])
        truth = np.concatenate(audit_exact_delta[name])
        denominator = float(np.linalg.norm(error + truth) * np.linalg.norm(truth))
        # Correlation of predicted and exact deltas is more interpretable than
        # coefficient identity, which is not unique in redundant joint motion.
        predicted = error + truth
        cosine = 0.0 if denominator <= 1.0e-15 else float(predicted @ truth / denominator)
        audits[name] = {
            "random_scalar_count": int(len(error)),
            "linearization_RMSE_m": float(np.sqrt(np.mean(error ** 2))),
            "linearization_absolute_error_p95_m": float(np.quantile(np.abs(error), 0.95)),
            "clearance_delta_cosine": cosine,
        }
    return {
        "jacobian_m_per_rad": jacobian,
        "nominal_joint_position_rad": nominal_q,
        "nominal_clearance_m": nominal_h,
        "state_split": state_splits,
        "fit_state_mask": fit_mask,
        "audit": audits,
    }


def signed_clearance_error(
    predicted_q: Any, exact_q: Any, state_index: Any, jacobians: Any,
) -> Any:
    """Return the local signed clearance error induced by joint error."""

    np = _numpy()
    predicted = np.asarray(predicted_q, dtype=np.float64)
    exact = np.asarray(exact_q, dtype=np.float64)
    states = np.asarray(state_index, dtype=np.int64)
    jacobian = np.asarray(jacobians, dtype=np.float64)[states]
    if predicted.shape != exact.shape or predicted.shape[1:] != (51, 7):
        raise ValueError("one-sided geometry q error shape differs")
    if jacobian.shape != (len(predicted), 51, 7, 7):
        raise ValueError("one-sided geometry Jacobian row shape differs")
    return np.einsum("bkrj,bkj->bkr", jacobian, predicted - exact)


def validation_pattern_gate_tests(
    audit: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, bool]:
    """Return the frozen v1 L5-specific or v2 distal-terminal audit."""

    pattern = config["validation_pattern_gate"]
    tests = {
        "validation_has_false_safe": int(audit["false_safe_action_count"])
        >= int(pattern["minimum_validation_false_safe_action_count"]),
    }
    if "minimum_terminal_L5_fraction" in pattern:
        tests["validation_terminal_L5_pattern"] = float(
            audit["terminal_L5_false_safe_fraction"]
        ) >= float(pattern["minimum_terminal_L5_fraction"])
    else:
        tests["validation_terminal_distal_pattern"] = float(
            audit["terminal_distal_false_safe_fraction"]
        ) >= float(pattern["minimum_terminal_distal_fraction"])
    return tests


def train_one_sided_geometry_ensemble(
    arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
    geometry: Mapping[str, Any], factorized_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Train q prediction with the registered directional geometry loss."""

    np = _numpy()
    torch = _torch()
    device = torch.device("cpu")
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
    state_split = {
        int(state): str(split[row])
        for row, state in enumerate(state_index)
    }
    sensitivity_split = np.asarray([
        state_split[int(state)] for state in sensitivity_states
    ], dtype=object)
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
    boundary_multiplier = float(settings["near_boundary_weight_multiplier"])
    terminal_multiplier = float(settings["terminal_weight_multiplier"])
    terminal_power = int(settings["terminal_weight_power"])
    late = 1.0 + (terminal_multiplier - 1.0) * (
        np.arange(51, dtype=np.float64) / 50.0
    ) ** terminal_power
    element_weight = np.where(
        np.abs(clearance) <= boundary, boundary_multiplier, 1.0,
    ) * late[None, :, None]
    batch_size = int(settings["batch_size"])
    models = []
    model_states = []
    member_audits = []
    for seed in settings["ensemble_seeds"]:
        torch.manual_seed(int(seed))
        model = _build_model(
            x.shape[1], 51 * 7, settings["hidden_widths"],
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
                xb = torch.as_tensor(normalized[indexes], dtype=torch.float32)
                target_residual = torch.as_tensor(
                    residual[indexes], dtype=torch.float32,
                )
                prediction = model(xb).reshape(-1, 51, 7)
                q_error = prediction - target_residual
                q_absolute = torch.abs(q_error)
                q_huber = torch.where(
                    q_absolute <= joint_delta, 0.5 * q_error ** 2,
                    joint_delta * (q_absolute - 0.5 * joint_delta),
                ).mean()
                jacobian = torch.as_tensor(
                    jacobians[state_index[indexes]], dtype=torch.float32,
                )
                h_error = torch.einsum("bkrj,bkj->bkr", jacobian, q_error)
                h_absolute = torch.abs(h_error)
                h_huber = torch.where(
                    h_absolute <= geometry_delta, 0.5 * h_error ** 2,
                    geometry_delta * (h_absolute - 0.5 * geometry_delta),
                )
                weights = torch.as_tensor(
                    element_weight[indexes], dtype=torch.float32,
                )
                geometry_loss = torch.mean(weights * h_huber)
                dangerous = torch.relu(h_error)
                geometry_loss = geometry_loss + float(
                    settings["dangerous_overestimate_weight"]
                ) * torch.mean(weights * dangerous ** 2)
                loss = q_huber + float(
                    settings["geometry_loss_weight"]
                ) * geometry_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            selected = train_sensitivity
            model.train()
            negative_prediction = model(torch.as_tensor(
                normalized[neg_rows[selected]], dtype=torch.float32,
            ))
            positive_prediction = model(torch.as_tensor(
                normalized[pos_rows[selected]], dtype=torch.float32,
            ))
            predicted_sensitivity = (
                positive_prediction - negative_prediction
            ) / torch.as_tensor(
                denominators[selected], dtype=torch.float32,
            )[:, None]
            exact_s = torch.as_tensor(
                exact_sensitivity[selected].reshape(len(selected), -1),
                dtype=torch.float32,
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
                )).cpu().numpy().reshape(-1, 51, 7)
                q_rmse = float(np.sqrt(np.mean(
                    (validation_prediction - residual[validation_indexes]) ** 2
                )))
                selected = validation_sensitivity
                negative_prediction = model(torch.as_tensor(
                    normalized[neg_rows[selected]], dtype=torch.float32,
                ))
                positive_prediction = model(torch.as_tensor(
                    normalized[pos_rows[selected]], dtype=torch.float32,
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
            validation_q = (
                base[validation_indexes] + validation_prediction
            )
            validation_h_error = signed_clearance_error(
                validation_q, target[validation_indexes],
                state_index[validation_indexes], jacobians,
            )
            geometry_rmse = float(np.sqrt(np.mean(validation_h_error ** 2)))
            # Selection remains value/sensitivity based, as in the baseline,
            # with only the registered signed-geometry value added.
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
            raise RuntimeError("one-sided geometry training produced no checkpoint")
        model.load_state_dict(best)
        model.eval()
        models.append(model)
        model_states.append({key: value.numpy() for key, value in best.items()})
        member_audits.append({
            "seed": int(seed), "best_epoch": int(best_epoch),
            "validation_score": float(best_score),
            "completed_epoch_count": int(epoch + 1),
        })
    state = {
        "arm": "factorized_execution", "feature_mean": mean,
        "feature_std": std, "input_dimension": int(x.shape[1]),
        "output_shape": [51, 7], "hidden_widths": settings["hidden_widths"],
        "ensemble_seeds": settings["ensemble_seeds"],
        "model_states": model_states,
    }
    return models, state, {
        "arm": "factorized_execution_one_sided_geometry",
        "member_audits": member_audits,
        "training_row_count": int(len(train_indexes)),
        "training_sensitivity_row_count": int(len(train_sensitivity)),
        "validation_row_count": int(len(validation_indexes)),
        "validation_sensitivity_row_count": int(len(validation_sensitivity)),
    }


def one_sided_decision(
    *, validation_audit: Mapping[str, Any], jacobian_audit: Mapping[str, Any],
    baseline_test: Mapping[str, Any], experimental_test: Mapping[str, Any],
    joint_sensitivity_cosine: float, margin_sensitivity_cosine: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["decision_gate"]
    tests = {
        **validation_pattern_gate_tests(validation_audit, config),
        "validation_linearization": float(
            jacobian_audit["validation"]["linearization_RMSE_m"]
        ) <= float(config["local_geometry_jacobian"][
            "maximum_validation_random_linearization_RMSE_m"
        ]),
        "zero_test_false_safe": int(experimental_test["false_safe_action_count"])
        <= int(gate["maximum_test_false_safe_action_count"]),
        "all_test_states_have_safe_support": int(
            experimental_test["state_safe_support_count"]
        ) >= int(gate["required_test_safe_support_state_count"]),
        "test_safe_recall": float(experimental_test["exact_safe_action_recall"])
        >= float(gate["minimum_test_safe_recall"]),
        "test_boundary_RMSE": float(experimental_test["near_boundary_RMSE_m"])
        <= float(gate["maximum_test_near_boundary_RMSE_m"]),
        "joint_sensitivity": float(joint_sensitivity_cosine)
        >= float(gate["minimum_joint_sensitivity_cosine"]),
        "margin_sensitivity": float(margin_sensitivity_cosine)
        >= float(gate["minimum_margin_sensitivity_cosine"]),
        "strict_false_safe_improvement": int(
            experimental_test["false_safe_action_count"]
        ) < int(baseline_test["false_safe_action_count"]),
    }
    passed = bool(all(tests.values()))
    return {
        "gate_tests": tests,
        "one_sided_geometry_GO": passed,
        "conclusion": (
            "one_sided_geometry_supervision_passes_prediction_gate"
            if passed else
            "one_sided_geometry_supervision_not_conservative_on_unseen_states"
        ),
        "uncertainty_QP_or_closed_loop_authorized": False,
    }
