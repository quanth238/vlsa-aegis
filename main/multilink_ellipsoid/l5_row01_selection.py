"""Contracts for the two-output L5 row-0/row-1 selection feasibility gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_row01_selection.v1"
MODEL_SCHEMA = "vlsa_distal_l5_row01_selection_model.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_row01_selection_validation.v1"


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
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "protocol_id", "claim_scope", "sources", "dataset",
        "features", "model", "selection", "gates", "forbidden",
    }:
        raise ValueError("L5 row01 selection config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-l5-row01-selection-v1"
        or value["model"]["output_count"] != 2
        or value["model"]["input_dimension"] != 86
        or value["model"]["hidden_widths"] != [32, 32]
        or value["selection"]["learned_rows"] != [0, 1]
        or value["dataset"]["expected_known_candidate_counts"]
        != {"train": 43, "validation": 29}
    ):
        raise ValueError("L5 row01 selection protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def build_model(torch: Any, input_dimension: int, hidden_widths: Sequence[int]) -> Any:
    widths = [int(input_dimension), *[int(item) for item in hidden_widths], 2]
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


def _weighted_huber(
    torch: Any, prediction: Any, normalized_target: Any, physical_target: Any,
    model_config: Mapping[str, Any],
) -> Any:
    element = torch.nn.functional.smooth_l1_loss(
        prediction, normalized_target,
        beta=float(model_config["huber_beta_normalized"]), reduction="none",
    )
    weight = 1.0 + float(model_config["boundary_weight_multiplier"]) * torch.exp(
        -torch.abs(physical_target) / float(model_config["boundary_scale_m"])
    )
    return torch.sum(weight * element) / torch.sum(weight)


def train_model(
    train_x: Any, train_y: Any, validation_x: Any, validation_y: Any,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("L5 row01 training requires exactly one H100")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError("L5 row01 training requires an H100")
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    x_validation = np.asarray(validation_x, dtype=np.float64)
    y_validation = np.asarray(validation_y, dtype=np.float64)
    if (
        x_train.shape != (43, int(model_config["input_dimension"]))
        or y_train.shape != (43, 2)
        or x_validation.shape != (29, int(model_config["input_dimension"]))
        or y_validation.shape != (29, 2)
    ):
        raise ValueError("L5 row01 training arrays differ")
    if not all(np.all(np.isfinite(item)) for item in (
        x_train, y_train, x_validation, y_validation
    )):
        raise ValueError("L5 row01 arrays are nonfinite")
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, int(model_config["cpu_threads"])))
    device = torch.device("cuda:0")
    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.where(
        np.std(x_train, axis=0) >= 1.0e-6, np.std(x_train, axis=0), 1.0
    )
    target_mean = np.mean(y_train, axis=0)
    target_scale = np.maximum(
        np.std(y_train, axis=0), float(model_config["minimum_target_scale_m"])
    )

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    tx = tensor((x_train - feature_mean) / feature_scale)
    ty = tensor((y_train - target_mean) / target_scale)
    ty_physical = tensor(y_train)
    vx = tensor((x_validation - feature_mean) / feature_scale)
    vy = tensor((y_validation - target_mean) / target_scale)
    vy_physical = tensor(y_validation)
    model = build_model(
        torch, int(model_config["input_dimension"]), model_config["hidden_widths"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_train_loss = math.inf
    for _ in range(int(model_config["epochs"])):
        model.train()
        loss = _weighted_huber(
            torch, model(tx), ty, ty_physical, model_config
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(model_config["gradient_clip_norm"])
        )
        optimizer.step()
        final_train_loss = float(loss.item())
    model.eval()
    with torch.no_grad():
        validation_loss = float(_weighted_huber(
            torch, model(vx), vy, vy_physical, model_config
        ).item())
    state_payload = {
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "state_dict": {
            name: value.detach().cpu().numpy().tolist()
            for name, value in sorted(model.state_dict().items())
        },
    }
    return {
        "model": model,
        "device": device,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "state_payload": state_payload,
        "model_sha256": hashlib.sha256(canonical(state_payload)).hexdigest(),
        "parameter_count": int(sum(item.numel() for item in model.parameters())),
        "device_name": device_name,
        "epochs_completed": int(model_config["epochs"]),
        "final_train_loss": final_train_loss,
        "validation_loss": validation_loss,
    }


def predict(bundle: Mapping[str, Any], features: Any) -> Any:
    import numpy as np
    import torch

    raw = np.asarray(features, dtype=np.float64)
    normalized = torch.as_tensor(
        (raw - bundle["feature_mean"]) / bundle["feature_scale"],
        dtype=torch.float32, device=bundle["device"],
    )
    bundle["model"].eval()
    with torch.no_grad():
        output = bundle["model"](normalized).detach().cpu().numpy()
    return output * bundle["target_scale"] + bundle["target_mean"]


def load_frozen_bundle(
    torch: Any, payload: Mapping[str, Any], model_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    model = build_model(
        torch, int(model_config["input_dimension"]), model_config["hidden_widths"]
    )
    template = model.state_dict()
    if set(payload) != {
        "feature_mean", "feature_scale", "target_mean", "target_scale", "state_dict"
    } or set(payload["state_dict"]) != set(template):
        raise ValueError("L5 row01 frozen payload differs")
    state = {}
    for name, value in template.items():
        raw = torch.as_tensor(payload["state_dict"][name], dtype=value.dtype)
        if raw.shape != value.shape:
            raise ValueError("L5 row01 frozen parameter shape differs")
        state[name] = raw
    model.load_state_dict(state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output = {
        "model": model,
        "device": device,
        "feature_mean": np.asarray(payload["feature_mean"], dtype=np.float64),
        "feature_scale": np.asarray(payload["feature_scale"], dtype=np.float64),
        "target_mean": np.asarray(payload["target_mean"], dtype=np.float64),
        "target_scale": np.asarray(payload["target_scale"], dtype=np.float64),
    }
    if (
        output["feature_mean"].shape != (86,)
        or output["feature_scale"].shape != (86,)
        or output["target_mean"].shape != (2,)
        or output["target_scale"].shape != (2,)
    ):
        raise ValueError("L5 row01 frozen normalization shape differs")
    return output


def selection_metrics(
    predictions: Sequence[Sequence[float]], samples: Sequence[Mapping[str, Any]],
    *, random_seed: int, random_draws: int,
) -> dict[str, Any]:
    import numpy as np

    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray([item["risk_row01"] for item in samples], dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[1:] != (2,):
        raise ValueError("L5 row01 prediction shape differs")
    predicted_safe = np.max(predicted, axis=1) <= 0.0
    actual_row01_safe = np.max(target, axis=1) <= 0.0
    false_safe = predicted_safe & ~actual_row01_safe
    state_records = []
    random_successes = random_total = 0
    rng = np.random.default_rng(int(random_seed))
    for state_id in sorted({str(item["state_id"]) for item in samples}):
        indices = [
            index for index, item in enumerate(samples)
            if str(item["state_id"]) == state_id
        ]
        exact = [index for index in indices if bool(samples[index]["exact_safe"])]
        accepted = [index for index in indices if bool(predicted_safe[index])]
        selected = None if not accepted else min(accepted, key=lambda index: (
            float(samples[index]["applied_correction_l2_action"]),
            int(samples[index]["candidate_order"]),
        ))
        oracle = None if not exact else min(exact, key=lambda index: (
            float(samples[index]["applied_correction_l2_action"]),
            int(samples[index]["candidate_order"]),
        ))
        draws = rng.choice(indices, size=int(random_draws), replace=True)
        random_safe = sum(bool(samples[int(index)]["exact_safe"]) for index in draws)
        random_successes += int(random_safe)
        random_total += int(random_draws)
        nominal = next(
            (index for index in indices if int(samples[index]["candidate_order"]) == 0),
            None,
        )
        selected_risk = None if selected is None else float(
            max(samples[selected]["risk_row01"])
        )
        nominal_risk = None if nominal is None else float(
            max(samples[nominal]["risk_row01"])
        )
        state_records.append({
            "state_id": state_id,
            "case_id": samples[indices[0]]["case_id"],
            "candidate_count": len(indices),
            "recoverable": bool(exact),
            "exact_safe_candidate_count": len(exact),
            "predicted_safe_candidate_count": len(accepted),
            "selected_candidate_name": None if selected is None else samples[selected]["candidate_name"],
            "selected_candidate_order": None if selected is None else int(samples[selected]["candidate_order"]),
            "selected_exact_row01_safe": None if selected is None else bool(actual_row01_safe[selected]),
            "selected_exact_all_seven_safe": None if selected is None else bool(samples[selected]["exact_safe"]),
            "selected_original_AEGIS_EE_compatible": None if selected is None else bool(samples[selected]["aegis_compatible"]),
            "selected_row01_risk_m": selected_risk,
            "selected_all_row_risk_m": None if selected is None else samples[selected]["risk_all_rows"],
            "selected_applied_correction_l2_action": None if selected is None else float(samples[selected]["applied_correction_l2_action"]),
            "oracle_candidate_name": None if oracle is None else samples[oracle]["candidate_name"],
            "oracle_applied_correction_l2_action": None if oracle is None else float(samples[oracle]["applied_correction_l2_action"]),
            "nominal_target_known": nominal is not None,
            "nominal_row01_risk_m": nominal_risk,
            "selected_improves_row01_risk_over_known_nominal": None if selected is None or nominal is None else bool(selected_risk < nominal_risk),
            "random_exact_safe_rate": float(random_safe) / float(random_draws),
        })
    recoverable = [item for item in state_records if item["recoverable"]]
    selected_records = [
        item for item in state_records if item["selected_candidate_name"] is not None
    ]
    selected_recoverable = [
        item for item in recoverable if item["selected_candidate_name"] is not None
    ]
    selected_exact = [
        item for item in selected_recoverable if item["selected_exact_all_seven_safe"]
    ]
    known_nominal = [
        item for item in selected_recoverable if item["nominal_target_known"]
    ]
    return {
        "sample_count": len(samples),
        "rmse_m": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "per_row_rmse_m": [
            float(np.sqrt(np.mean((predicted[:, row] - target[:, row]) ** 2)))
            for row in range(2)
        ],
        "row01_false_safe_count": int(np.sum(false_safe)),
        "recoverable_state_count": len(recoverable),
        "supported_recoverable_state_count": len(selected_recoverable),
        "selected_exact_safe_recoverable_state_count": len(selected_exact),
        "selected_exact_safe_rate": (
            float(len(selected_exact)) / len(recoverable) if recoverable else 0.0
        ),
        "all_selected_original_AEGIS_EE_compatible": all(
            item["selected_original_AEGIS_EE_compatible"] is not False
            for item in selected_records
        ),
        "all_selected_exact_all_seven_safe": bool(
            selected_records and all(
                item["selected_exact_all_seven_safe"] for item in selected_records
            )
        ),
        "all_selected_improve_over_known_nominal": bool(
            known_nominal and all(
                item["selected_improves_row01_risk_over_known_nominal"]
                for item in known_nominal
            )
        ),
        "seeded_random_exact_safe_rate": (
            float(random_successes) / random_total if random_total else 0.0
        ),
        "state_records": state_records,
    }
