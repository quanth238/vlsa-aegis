"""Contracts and causal features for the prospective L5 context ablation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_prospective_l5_context_ablation.v1"
RESULT_SCHEMA = "vlsa_distal_prospective_l5_context_ablation_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_prospective_l5_context_ablation_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "sources", "arms",
        "matched_factors", "decision", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("prospective L5 context ablation config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-prospective-l5-context-ablation-v1"
        or value["arms"]["relative_endpoint_9D"]["input_dimension"] != 9
        or value["arms"]["direct_L5_OSC_33D"]["input_dimension"] != 33
        or value["arms"]["direct_L5_OSC_33D"]["initial_L5_primitive_row_indices"] != [0, 1, 2]
        or value["arms"]["direct_L5_OSC_33D"]["initial_exact_slack_indices"] != [1, 2, 3]
        or not all(bool(item) for item in value["matched_factors"].values())
        or "candidate_selection_or_correction" not in value["forbidden"]
        or "QP_or_gradient_correction" not in value["forbidden"]
    ):
        raise ValueError("prospective L5 context ablation protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def obstacle_aabb(boxes: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    lower = [math.inf, math.inf, math.inf]
    upper = [-math.inf, -math.inf, -math.inf]
    if not boxes:
        raise ValueError("compiled obstacle box set is empty")
    for box in boxes:
        center = [float(item) for item in box["center_m"]]
        rotation = [[float(item) for item in row] for row in box["rotation"]]
        half = [float(item) for item in box["half_extents_m"]]
        if len(center) != 3 or len(rotation) != 3 or any(len(row) != 3 for row in rotation) or len(half) != 3:
            raise ValueError("compiled obstacle box shape differs")
        extent = [sum(abs(rotation[i][k]) * half[k] for k in range(3)) for i in range(3)]
        for i in range(3):
            lower[i] = min(lower[i], center[i] - extent[i])
            upper[i] = max(upper[i], center[i] + extent[i])
    center = [(lower[i] + upper[i]) * 0.5 for i in range(3)]
    half = [(upper[i] - lower[i]) * 0.5 for i in range(3)]
    return center, half


def direct_l5_osc_feature(
    base_9d: Sequence[float], exact_case: Mapping[str, Any],
    *, primitive_indices: Sequence[int] = (0, 1, 2),
    slack_indices: Sequence[int] = (1, 2, 3),
) -> list[float]:
    """Build the causal 33D feature; no future rollout state enters this vector."""
    base = [float(item) for item in base_9d]
    rows = exact_case["initial_empirical_robot_rows"]
    boxes = exact_case["initial_compiled_obstacle_boxes"]
    obstacle_center, _ = obstacle_aabb(boxes)
    relative_centers: list[float] = []
    semiaxes: list[float] = []
    for index in primitive_indices:
        row = rows[int(index)]
        if str(row["body_name"]) != "robot0_link5":
            raise ValueError("direct context primitive is not L5")
        center = [float(item) for item in row["center_m"]]
        axes = [float(item) for item in row["semiaxes_m"]]
        relative_centers.extend(center[i] - obstacle_center[i] for i in range(3))
        semiaxes.extend(axes)
    all_slacks = exact_case["exact_group_target"]["initial_row_normalized_radial_slack"]
    slacks = [float(all_slacks[int(index)]) for index in slack_indices]
    context = exact_case["physical_context"]
    goal = [float(item) for item in context["controller_snapshot"]["goal_pos"]]
    eef = [float(item) for item in context["eef_position_m"]]
    osc_error = [goal[i] - eef[i] for i in range(3)]
    feature = base + relative_centers + semiaxes + slacks + osc_error
    if len(feature) != 33 or not all(math.isfinite(item) for item in feature):
        raise ValueError("direct L5 OSC feature differs")
    return feature


def build_model(torch: Any, input_dimension: int, hidden_widths: Sequence[int]) -> Any:
    widths = [int(input_dimension), *[int(item) for item in hidden_widths], 3]
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


def classify_ablation(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    btest, ctest = baseline["test"], candidate["test"]
    checks = {
        "test_RMSE_reduced_by_25_percent": (
            float(ctest["global_RMSE"])
            <= float(decision["maximum_test_RMSE_ratio_to_9D"]) * float(btest["global_RMSE"])
        ),
        "strictly_fewer_test_false_safes": (
            int(ctest["false_safe_count"]) < int(btest["false_safe_count"])
        ),
        "strictly_better_test_improvement_direction": (
            ctest["improvement_direction_accuracy"] is not None
            and btest["improvement_direction_accuracy"] is not None
            and float(ctest["improvement_direction_accuracy"])
            > float(btest["improvement_direction_accuracy"])
        ),
        "no_validation_support_loss": (
            int(candidate["validation"]["supported_state_count"])
            >= int(baseline["validation"]["supported_state_count"])
        ),
        "no_test_support_loss": (
            int(ctest["supported_state_count"]) >= int(btest["supported_state_count"])
        ),
    }
    strong = {}
    for split in ("validation", "test"):
        metrics = candidate[split]
        strong[f"{split}_zero_false_safes"] = (
            int(metrics["false_safe_count"]) <= int(decision["strong_maximum_false_safe_count"])
        )
        strong[f"{split}_safe_recall"] = (
            metrics["safe_recall"] is not None
            and float(metrics["safe_recall"]) >= float(decision["strong_minimum_safe_recall"])
        )
        strong[f"{split}_safe_support"] = (
            int(metrics["supported_state_count"])
            >= int(decision["strong_required_safe_support_state_count"])
        )
        strong[f"{split}_near_boundary_RMSE"] = (
            metrics["near_boundary_global_RMSE"] is not None
            and float(metrics["near_boundary_global_RMSE"])
            <= float(decision["strong_maximum_near_boundary_global_RMSE"])
        )
    checks.update({f"strong_{key}": value for key, value in strong.items()})
    material = all(value for key, value in checks.items() if not key.startswith("strong_"))
    strong_pass = all(strong.values())
    if strong_pass:
        verdict = "strong_diagnostic_pass_opened_test_only"
    elif material:
        verdict = "material_representation_improvement_but_prediction_gate_failed"
    else:
        verdict = "representation_ablation_no_go"
    return verdict, checks
