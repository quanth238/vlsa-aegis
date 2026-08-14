"""Contracts for the matched compact-versus-complete L5 input ablation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_row01_input_ablation.v1"
RESULT_SCHEMA = "vlsa_distal_l5_row01_input_ablation_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_row01_input_ablation_validation.v1"
COMPLETE_DIMENSION = 354


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
        "schema_version", "protocol_id", "claim_scope", "source", "arms",
        "matched", "gates", "interpretation", "forbidden",
    }:
        raise ValueError("L5 row01 input-ablation config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-l5-row01-input-ablation-v1"
        or value["arms"]["compact_86D"]["input_dimension"] != 86
        or value["arms"]["complete_physical_OSC_354D"]["input_dimension"]
        != COMPLETE_DIMENSION
        or value["matched"]["hidden_widths"] != [32, 32]
        or value["matched"]["seed"] != 20260814
    ):
        raise ValueError("L5 row01 input-ablation protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _flat(value: Any, *, shape: tuple[int, ...], name: str) -> list[float]:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("complete-input field differs: %s" % name)
    return array.reshape(-1).tolist()


def complete_feature_vector(
    *, physical_context: Mapping[str, Any], nominal_actions: Any,
    candidate_actions: Any, aegis: Mapping[str, Any],
) -> list[float]:
    """Flatten only the preregistered online physical/controller quantities."""

    import numpy as np

    controller = physical_context["controller_snapshot"]
    output: list[float] = []
    output += _flat(physical_context["arm_joint_position_rad"], shape=(7,), name="q")
    output += _flat(physical_context["arm_joint_velocity_rad_s"], shape=(7,), name="dq")
    output += _flat(physical_context["eef_position_m"], shape=(3,), name="eef_position")
    output += _flat(physical_context["eef_quaternion_xyzw"], shape=(4,), name="eef_quaternion")
    output += _flat(controller["goal_ori"], shape=(3, 3), name="goal_ori")
    output += _flat(controller["goal_pos"], shape=(3,), name="goal_pos")
    output += _flat(
        controller["gripper_current_action"], shape=(2,), name="gripper_current_action"
    )
    output.append(float(bool(controller["new_update"])))
    ori_ref = controller["ori_ref"]
    output.append(float(ori_ref is not None))
    output += ([0.0] * 9 if ori_ref is None else _flat(
        ori_ref, shape=(3, 3), name="ori_ref"
    ))
    output += _flat(controller["relative_ori"], shape=(3,), name="relative_ori")
    output += _flat(controller["robot_torques"], shape=(7,), name="robot_torques")
    output += _flat(controller["torques"], shape=(7,), name="torques")
    output += _flat(nominal_actions, shape=(5, 7), name="nominal_actions")
    output += _flat(candidate_actions, shape=(5, 7), name="candidate_actions")
    obstacle = physical_context["obstacle"]
    output += _flat(obstacle["center_m"], shape=(3,), name="obstacle_center")
    output += _flat(obstacle["rotation"], shape=(3, 3), name="obstacle_rotation")
    output += _flat(obstacle["semiaxes_m"], shape=(3,), name="obstacle_semiaxes")
    rows = physical_context["geometry_rows"]
    if len(rows) != 7 or any(row["body_name"] != "robot0_link5" for row in rows[:3]):
        raise ValueError("complete-input L5 row identities differ")
    for index, row in enumerate(rows[:3]):
        output += _flat(row["center_m"], shape=(3,), name=f"row{index}_center")
        output += _flat(row["rotation"], shape=(3, 3), name=f"row{index}_rotation")
        output += _flat(row["semiaxes_m"], shape=(3,), name=f"row{index}_semiaxes")
        output += _flat(
            row["obstacle_relative_center_m"], shape=(3,), name=f"row{index}_relative"
        )
        output += _flat(row["outward_normal"], shape=(3,), name=f"row{index}_normal")
        clearance = float(row["current_clearance_m"])
        if not math.isfinite(clearance):
            raise ValueError("complete-input row clearance is nonfinite")
        output.append(clearance)
    if aegis["enabled"] is not True or len(aegis["qp_records"]) != 5:
        raise ValueError("complete-input AEGIS record differs")
    output += _flat(aegis["proposed_actions"], shape=(5, 7), name="aegis_proposed")
    output += [
        float(aegis["correction_l2_action"]),
        float(aegis["maximum_absolute_action_change"]),
    ]
    output += _flat(aegis["z_before"], shape=(3,), name="aegis_z_before")
    output += _flat(aegis["z_after_by_action"], shape=(5, 3), name="aegis_z_after")
    for index, qp in enumerate(aegis["qp_records"]):
        output += [
            float(qp["status"] == "solved"),
            float(qp["solver_status"] in ("optimal", "optimal_inaccurate")),
            float(qp["barrier_h"]),
            float(qp["constraint_lhs"]),
            float(qp["objective"]),
        ]
        output += _flat(qp["u_solution"], shape=(6,), name=f"qp{index}_u")
        output += _flat(qp["z_before"], shape=(3,), name=f"qp{index}_z_before")
        output += _flat(qp["z_after"], shape=(3,), name=f"qp{index}_z_after")
    array = np.asarray(output, dtype=np.float64)
    if array.shape != (COMPLETE_DIMENSION,) or not np.all(np.isfinite(array)):
        raise ValueError("complete physical/OSC feature vector differs")
    return array.tolist()


def build_model(torch: Any, input_dimension: int, hidden_widths: Sequence[int]) -> Any:
    widths = [int(input_dimension), *[int(value) for value in hidden_widths], 2]
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


def train_complete_model(
    train_x: Any, train_y: Any, validation_x: Any, validation_y: Any,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("complete-input ablation requires exactly one H100")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError("complete-input ablation requires an H100")
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    x_validation = np.asarray(validation_x, dtype=np.float64)
    y_validation = np.asarray(validation_y, dtype=np.float64)
    if (
        x_train.shape != (43, COMPLETE_DIMENSION)
        or y_train.shape != (43, 2)
        or x_validation.shape != (29, COMPLETE_DIMENSION)
        or y_validation.shape != (29, 2)
    ):
        raise ValueError("complete-input training arrays differ")
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
        torch, COMPLETE_DIMENSION, model_config["hidden_widths"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_train_loss = math.inf
    for _ in range(int(model_config["epochs"])):
        model.train()
        loss = _weighted_huber(torch, model(tx), ty, ty_physical, model_config)
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
        "model": model, "device": device,
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "state_payload": state_payload,
        "model_sha256": hashlib.sha256(canonical(state_payload)).hexdigest(),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "device_name": device_name,
        "epochs_completed": int(model_config["epochs"]),
        "final_train_loss": final_train_loss,
        "validation_loss": validation_loss,
    }


def load_complete_bundle(
    torch: Any, payload: Mapping[str, Any], model_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    model = build_model(torch, COMPLETE_DIMENSION, model_config["hidden_widths"])
    template = model.state_dict()
    if set(payload["state_dict"]) != set(template):
        raise ValueError("complete-input frozen state differs")
    state = {}
    for name, value in template.items():
        raw = torch.as_tensor(payload["state_dict"][name], dtype=value.dtype)
        if raw.shape != value.shape:
            raise ValueError("complete-input frozen parameter shape differs")
        state[name] = raw
    model.load_state_dict(state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return {
        "model": model, "device": device,
        "feature_mean": np.asarray(payload["feature_mean"], dtype=np.float64),
        "feature_scale": np.asarray(payload["feature_scale"], dtype=np.float64),
        "target_mean": np.asarray(payload["target_mean"], dtype=np.float64),
        "target_scale": np.asarray(payload["target_scale"], dtype=np.float64),
    }


def ablation_metrics(
    predictions: Any, samples: Sequence[Mapping[str, Any]], *, random_seed: int,
    random_draws: int, near_boundary_m: float = 0.005,
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.l5_row01_selection import selection_metrics

    report = selection_metrics(
        predictions, samples, random_seed=random_seed, random_draws=random_draws
    )
    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray([item["risk_row01"] for item in samples], dtype=np.float64)
    near = np.abs(target) <= float(near_boundary_m)
    report["near_boundary_element_count"] = int(np.sum(near))
    report["near_boundary_rmse_m"] = (
        float(np.sqrt(np.mean((predicted[near] - target[near]) ** 2)))
        if np.any(near) else None
    )
    report["per_row_near_boundary_rmse_m"] = [
        float(np.sqrt(np.mean(
            (predicted[near[:, row], row] - target[near[:, row], row]) ** 2
        ))) if np.any(near[:, row]) else None
        for row in range(2)
    ]
    return report
