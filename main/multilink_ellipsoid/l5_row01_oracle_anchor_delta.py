"""Exact-anchor action-response diagnostic for L5 rows 0 and 1."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_row01_oracle_anchor_delta.v1"
RESULT_SCHEMA = "vlsa_distal_l5_row01_oracle_anchor_delta_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_row01_oracle_anchor_delta_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    payload.pop("validation_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source", "dataset",
        "representation", "model", "evaluation", "gates", "interpretation",
        "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("oracle-anchor delta config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-l5-row01-oracle-anchor-delta-v1"
        or value["representation"]["context_dimension"] != 99
        or value["representation"]["residual_action_dimension"] != 35
        or value["representation"]["input_dimension"] != 134
        or value["model"]["output_count"] != 2
        or value["model"]["hidden_widths"] != [32, 32]
        or value["evaluation"]["learned_rows"] != [0, 1]
    ):
        raise ValueError("oracle-anchor delta protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def build_model(torch: Any, input_dimension: int,
                hidden_widths: Sequence[int]) -> Any:
    widths = [int(input_dimension), *[int(item) for item in hidden_widths], 2]
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


def _normalized_inputs(torch: Any, context: Any, residual: Any,
                       bundle: Mapping[str, Any]) -> tuple[Any, Any]:
    context_z = (context - bundle["context_mean"]) / bundle["context_scale"]
    residual_z = residual / bundle["residual_scale"]
    zeros = torch.zeros_like(residual_z)
    return (
        torch.cat((context_z, residual_z), dim=1),
        torch.cat((context_z, zeros), dim=1),
    )


def _delta_output(torch: Any, model: Any, context: Any, residual: Any,
                  bundle: Mapping[str, Any]) -> Any:
    with_residual, at_zero = _normalized_inputs(
        torch, context, residual, bundle
    )
    return model(with_residual) - model(at_zero)


def train_model(train_context: Any, train_residual: Any, train_delta: Any,
                train_anchor: Any, validation_context: Any,
                validation_residual: Any, validation_delta: Any,
                validation_anchor: Any,
                model_config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("oracle-anchor delta training requires exactly one H100")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError("oracle-anchor delta training requires an H100")
    arrays = [np.asarray(item, dtype=np.float64) for item in (
        train_context, train_residual, train_delta, train_anchor,
        validation_context, validation_residual, validation_delta,
        validation_anchor,
    )]
    (tc, tr, td, tb, vc, vr, vd, vb) = arrays
    if (
        tc.shape[1:] != (99,) or tr.shape != (tc.shape[0], 35)
        or td.shape != (tc.shape[0], 2) or tb.shape != td.shape
        or vc.shape[1:] != (99,) or vr.shape != (vc.shape[0], 35)
        or vd.shape != (vc.shape[0], 2) or vb.shape != vd.shape
        or min(tc.shape[0], vc.shape[0]) < 1
        or not all(np.all(np.isfinite(item)) for item in arrays)
    ):
        raise ValueError("oracle-anchor delta training arrays differ")
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, int(model_config["cpu_threads"])))
    device = torch.device("cuda:0")
    context_mean = np.mean(tc, axis=0)
    context_scale = np.where(np.std(tc, axis=0) >= 1.0e-6,
                             np.std(tc, axis=0), 1.0)
    residual_scale = np.where(
        np.sqrt(np.mean(tr ** 2, axis=0)) >= 1.0e-6,
        np.sqrt(np.mean(tr ** 2, axis=0)), 1.0,
    )
    target_scale = np.maximum(
        np.std(td, axis=0), float(model_config["minimum_target_scale_m"])
    )

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    bundle: dict[str, Any] = {
        "context_mean": tensor(context_mean),
        "context_scale": tensor(context_scale),
        "residual_scale": tensor(residual_scale),
        "target_scale": tensor(target_scale),
    }
    tc_t, tr_t, td_t, tb_t = map(tensor, (tc, tr, td, tb))
    vc_t, vr_t, vd_t, vb_t = map(tensor, (vc, vr, vd, vb))
    model = build_model(torch, 134, model_config["hidden_widths"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )

    def loss(context: Any, residual: Any, delta: Any, anchor: Any) -> Any:
        prediction = _delta_output(torch, model, context, residual, bundle)
        normalized_target = delta / bundle["target_scale"]
        element = torch.nn.functional.smooth_l1_loss(
            prediction, normalized_target,
            beta=float(model_config["huber_beta_normalized"]), reduction="none",
        )
        exact_risk = anchor + delta
        weight = 1.0 + float(model_config["boundary_weight_multiplier"]) * torch.exp(
            -torch.abs(exact_risk) / float(model_config["boundary_scale_m"])
        )
        return torch.sum(weight * element) / torch.sum(weight)

    final_train_loss = math.inf
    for _ in range(int(model_config["epochs"])):
        model.train()
        train_loss = loss(tc_t, tr_t, td_t, tb_t)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(model_config["gradient_clip_norm"])
        )
        optimizer.step()
        final_train_loss = float(train_loss.item())
    model.eval()
    with torch.no_grad():
        validation_loss = float(loss(vc_t, vr_t, vd_t, vb_t).item())
        zero_error = float(torch.max(torch.abs(_delta_output(
            torch, model, vc_t, torch.zeros_like(vr_t), bundle
        ) * bundle["target_scale"])).item())
    state_payload = {
        "context_mean": context_mean.tolist(),
        "context_scale": context_scale.tolist(),
        "residual_scale": residual_scale.tolist(),
        "target_scale": target_scale.tolist(),
        "state_dict": {
            name: value.detach().cpu().numpy().tolist()
            for name, value in sorted(model.state_dict().items())
        },
    }
    return {
        "model": model, "device": device,
        **bundle,
        "state_payload": state_payload,
        "model_sha256": hashlib.sha256(canonical(state_payload)).hexdigest(),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "device_name": device_name,
        "epochs_completed": int(model_config["epochs"]),
        "final_train_loss": final_train_loss,
        "validation_loss": validation_loss,
        "architectural_zero_response_maximum_error_m": zero_error,
    }


def predict(bundle: Mapping[str, Any], context: Any, residual: Any) -> Any:
    import numpy as np
    import torch

    context_t = torch.as_tensor(
        np.asarray(context, dtype=np.float64), dtype=torch.float32,
        device=bundle["device"],
    )
    residual_t = torch.as_tensor(
        np.asarray(residual, dtype=np.float64), dtype=torch.float32,
        device=bundle["device"],
    )
    bundle["model"].eval()
    with torch.no_grad():
        output = _delta_output(
            torch, bundle["model"], context_t, residual_t, bundle
        ) * bundle["target_scale"]
    return output.detach().cpu().numpy()


def load_bundle(torch: Any, payload: Mapping[str, Any],
                model_config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    model = build_model(torch, 134, model_config["hidden_widths"])
    template = model.state_dict()
    if set(payload) != {
        "context_mean", "context_scale", "residual_scale", "target_scale",
        "state_dict",
    } or set(payload["state_dict"]) != set(template):
        raise ValueError("oracle-anchor frozen bundle differs")
    state = {}
    for name, value in template.items():
        raw = torch.as_tensor(payload["state_dict"][name], dtype=value.dtype)
        if raw.shape != value.shape:
            raise ValueError("oracle-anchor frozen parameter shape differs")
        state[name] = raw
    model.load_state_dict(state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def tensor(name: str) -> Any:
        return torch.as_tensor(payload[name], dtype=torch.float32, device=device)

    output = {
        "model": model, "device": device,
        "context_mean": tensor("context_mean"),
        "context_scale": tensor("context_scale"),
        "residual_scale": tensor("residual_scale"),
        "target_scale": tensor("target_scale"),
    }
    if (
        output["context_mean"].shape != (99,)
        or output["context_scale"].shape != (99,)
        or output["residual_scale"].shape != (35,)
        or output["target_scale"].shape != (2,)
    ):
        raise ValueError("oracle-anchor frozen normalization shape differs")
    return output


def response_metrics(predicted_delta: Any,
                     samples: Sequence[Mapping[str, Any]], *,
                     sign_deadband_m: float, random_seed: int,
                     random_draws: int) -> dict[str, Any]:
    import numpy as np

    prediction = np.asarray(predicted_delta, dtype=np.float64)
    delta = np.asarray([item["delta_target"] for item in samples], dtype=np.float64)
    anchor = np.asarray([item["anchor_target"] for item in samples], dtype=np.float64)
    exact = np.asarray([item["combined_target"] for item in samples], dtype=np.float64)
    if prediction.shape != delta.shape or prediction.shape[1:] != (2,):
        raise ValueError("oracle-anchor metric arrays differ")
    predicted_risk = anchor + prediction
    exact_scalar = np.max(exact, axis=1)
    predicted_scalar = np.max(predicted_risk, axis=1)
    anchor_scalar = np.max(anchor, axis=1)
    exact_improvement = anchor_scalar - exact_scalar
    predicted_improvement = anchor_scalar - predicted_scalar
    response = np.asarray([
        float(item["applied_correction_l2_action"]) > 1.0e-12
        for item in samples
    ], dtype=bool)
    sign_mask = response & (np.abs(exact_improvement) >= float(sign_deadband_m))
    predicted_safe = predicted_scalar <= 0.0
    exact_safe = exact_scalar <= 0.0
    state_records = []
    pair_correct = pair_count = 0
    all_pair_correct = all_pair_count = 0
    random_safe = random_total = 0
    rng = np.random.default_rng(int(random_seed))
    for state_id in sorted({str(item["state_id"]) for item in samples}):
        indices = np.asarray([
            index for index, item in enumerate(samples)
            if str(item["state_id"]) == state_id
        ], dtype=np.int64)
        safe_indices = [int(index) for index in indices if exact_safe[index]]
        unsafe_indices = [int(index) for index in indices if not exact_safe[index]]
        for safe_index in safe_indices:
            for unsafe_index in unsafe_indices:
                pair_count += 1
                pair_correct += int(
                    predicted_scalar[safe_index] < predicted_scalar[unsafe_index]
                )
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1:]:
                exact_difference = exact_scalar[left] - exact_scalar[right]
                if abs(exact_difference) < float(sign_deadband_m):
                    continue
                all_pair_count += 1
                predicted_difference = predicted_scalar[left] - predicted_scalar[right]
                all_pair_correct += int(
                    np.sign(predicted_difference) == np.sign(exact_difference)
                )
        accepted = [int(index) for index in indices if predicted_safe[index]]
        selected = None if not accepted else min(accepted, key=lambda index: (
            float(samples[index]["applied_correction_l2_action"]),
            int(samples[index]["candidate_order"]),
        ))
        ranked = int(indices[int(np.argmin(predicted_scalar[indices]))])
        oracle = None if not safe_indices else min(safe_indices, key=lambda index: (
            float(samples[index]["applied_correction_l2_action"]),
            int(samples[index]["candidate_order"]),
        ))
        non_nominal = [
            int(index) for index in indices
            if float(samples[index]["applied_correction_l2_action"]) > 1.0e-12
        ]
        strongest = max(non_nominal, key=lambda index: (
            float(samples[index]["applied_correction_l2_action"]),
            -int(samples[index]["candidate_order"]),
        ))
        draws = rng.choice(indices, size=int(random_draws), replace=True)
        state_random_safe = int(np.sum(exact_safe[draws]))
        random_safe += state_random_safe
        random_total += int(random_draws)
        selected_safe = None if selected is None else bool(exact_safe[selected])
        strongest_safe = bool(exact_safe[strongest])
        win_or_tie = False
        if selected is not None and selected_safe:
            if not strongest_safe:
                win_or_tie = True
            else:
                win_or_tie = bool(
                    float(samples[selected]["applied_correction_l2_action"])
                    <= float(samples[strongest]["applied_correction_l2_action"])
                    + 1.0e-12
                )
        state_records.append({
            "state_id": state_id,
            "case_id": samples[int(indices[0])]["case_id"],
            "candidate_count": int(len(indices)),
            "exact_row01_safe_candidate_count": len(safe_indices),
            "recoverable": bool(safe_indices),
            "predicted_safe_candidate_count": len(accepted),
            "selected_candidate_name": None if selected is None else samples[selected]["candidate_name"],
            "selected_exact_row01_safe": selected_safe,
            "selected_exact_all_seven_safe": None if selected is None else bool(samples[selected]["exact_safe"]),
            "selected_correction_l2_action": None if selected is None else float(samples[selected]["applied_correction_l2_action"]),
            "selected_exact_risk_m": None if selected is None else float(exact_scalar[selected]),
            "selected_exact_improvement_m": None if selected is None else float(exact_improvement[selected]),
            "ranked_candidate_name": samples[ranked]["candidate_name"],
            "ranked_exact_row01_safe": bool(exact_safe[ranked]),
            "oracle_closest_safe_candidate_name": None if oracle is None else samples[oracle]["candidate_name"],
            "strongest_registered_path_candidate_name": samples[strongest]["candidate_name"],
            "strongest_registered_path_exact_row01_safe": strongest_safe,
            "strongest_registered_path_correction_l2_action": float(samples[strongest]["applied_correction_l2_action"]),
            "task_preserving_win_or_tie_vs_strongest": win_or_tie,
            "seeded_random_row01_safe_rate": float(state_random_safe) / float(random_draws),
        })
    recoverable = [item for item in state_records if item["recoverable"]]
    supported = [item for item in recoverable if item["selected_candidate_name"] is not None]
    exact_selected = [item for item in supported if item["selected_exact_row01_safe"]]
    return {
        "sample_count": len(samples),
        "response_sample_count": int(np.sum(response)),
        "delta_rmse_m": float(np.sqrt(np.mean((prediction[response] - delta[response]) ** 2))),
        "zero_change_delta_rmse_m": float(np.sqrt(np.mean(delta[response] ** 2))),
        "per_row_delta_rmse_m": np.sqrt(np.mean(
            (prediction[response] - delta[response]) ** 2, axis=0
        )).tolist(),
        "per_row_zero_change_delta_rmse_m": np.sqrt(np.mean(
            delta[response] ** 2, axis=0
        )).tolist(),
        "improvement_sign_evaluable_count": int(np.sum(sign_mask)),
        "improvement_sign_accuracy": (
            float(np.mean(
                np.sign(predicted_improvement[sign_mask])
                == np.sign(exact_improvement[sign_mask])
            )) if np.any(sign_mask) else 0.0
        ),
        "safe_unsafe_pair_count": pair_count,
        "safe_unsafe_pair_order_accuracy": (
            float(pair_correct) / pair_count if pair_count else 0.0
        ),
        "all_pair_count": all_pair_count,
        "all_pair_order_accuracy": (
            float(all_pair_correct) / all_pair_count if all_pair_count else 0.0
        ),
        "false_safe_candidate_count": int(np.sum(predicted_safe & ~exact_safe)),
        "exact_safe_candidate_recall": (
            float(np.sum(predicted_safe & exact_safe)) / int(np.sum(exact_safe))
            if np.any(exact_safe) else 0.0
        ),
        "recoverable_state_count": len(recoverable),
        "supported_recoverable_state_count": len(supported),
        "selected_exact_row01_safe_state_count": len(exact_selected),
        "selected_exact_row01_safe_rate": (
            float(len(exact_selected)) / len(recoverable) if recoverable else 0.0
        ),
        "seeded_random_row01_safe_rate": (
            float(random_safe) / random_total if random_total else 0.0
        ),
        "task_preserving_win_or_tie_vs_strongest_state_count": sum(
            bool(item["task_preserving_win_or_tie_vs_strongest"])
            for item in recoverable
        ),
        "state_records": state_records,
    }
