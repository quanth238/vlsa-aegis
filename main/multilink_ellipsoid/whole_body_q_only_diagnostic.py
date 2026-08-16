"""Contracts and causal features for the whole-body Q-only diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_whole_body_q_only_diagnostic.v1"
RESULT_SCHEMA = "vlsa_distal_whole_body_q_only_diagnostic_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_whole_body_q_only_diagnostic_validation.v1"
CLAIMED_ROWS = (1, 2, 3, 4, 5, 6)
SHARED_INPUT_DIMENSION = 135


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Any) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("whole-body diagnostic schema differs")
    if value.get("protocol_id") != "vlsa-distal-whole-body-q-only-diagnostic-v1":
        raise ValueError("whole-body diagnostic protocol differs")
    if value.get("dataset", {}).get("fit_splits") != ["train"]:
        raise ValueError("whole-body diagnostic fit split differs")
    if value.get("dataset", {}).get("evaluation_splits") != ["train", "validation"]:
        raise ValueError("whole-body diagnostic evaluation split differs")
    if value.get("dataset", {}).get("test_access") is not False:
        raise ValueError("whole-body diagnostic test must remain sealed")
    arms = value.get("arms", {})
    if (
        arms.get("relative_endpoint_9D", {}).get("input_dimension") != 9
        or arms.get("direct_L5_OSC_33D", {}).get("input_dimension") != 33
        or arms.get("shared_constraint_135D", {}).get("input_dimension")
        != SHARED_INPUT_DIMENSION
        or arms.get("shared_constraint_135D", {}).get("claimed_rows")
        != list(CLAIMED_ROWS)
    ):
        raise ValueError("whole-body diagnostic arm contract differs")
    model = value.get("model", {})
    if (
        model.get("hidden_widths") != [32, 32]
        or model.get("seed") != 20260814
        or float(model.get("weight_decay", -1.0)) != 0.0001
        or model.get("checkpoint") != "fixed_final_epoch_no_validation_selection"
    ):
        raise ValueError("whole-body diagnostic model contract differs")
    if value.get("timeout_rule") != "UNKNOWN_censored_from_fit_and_metrics":
        raise ValueError("whole-body diagnostic timeout rule differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _flatten_actions(actions: Sequence[Sequence[float]]) -> list[float]:
    output = [float(value) for row in actions for value in row]
    if len(actions) != 5 or any(len(row) != 7 for row in actions):
        raise ValueError("whole-body diagnostic action chunk differs")
    return output


def _transpose_matvec(rotation: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError("whole-body diagnostic rotation differs")
    return [
        sum(float(rotation[j][i]) * float(vector[j]) for j in range(3))
        for i in range(3)
    ]


def direct_l5_33d_feature(
    base_9d: Sequence[float], exact_case: Mapping[str, Any],
) -> list[float]:
    from main.multilink_ellipsoid.prospective_l5_context_ablation import obstacle_aabb

    rows = exact_case["initial_exact_robot_rows"]
    obstacle_center, _ = obstacle_aabb(exact_case["initial_compiled_obstacle_boxes"])
    relative_centers = []
    semiaxes = []
    for index in (2, 3, 4):
        row = rows[index]
        if str(row["body_name"]) != "robot0_link5":
            raise ValueError("whole-body diagnostic L5 row differs")
        relative_centers.extend(
            float(row["center_m"][axis]) - float(obstacle_center[axis])
            for axis in range(3)
        )
        semiaxes.extend(float(value) for value in row["semiaxes_m"])
    initial_slack = exact_case["exact_group_target"][
        "initial_row_normalized_radial_slack"
    ]
    slacks = [float(initial_slack[index]) for index in (2, 3, 4)]
    context = exact_case["physical_context"]
    goal = [float(value) for value in context["controller_snapshot"]["goal_pos"]]
    eef = [float(value) for value in context["eef_position_m"]]
    osc_error = [goal[index] - eef[index] for index in range(3)]
    feature = [float(value) for value in base_9d]
    feature.extend(relative_centers + semiaxes + slacks + osc_error)
    if len(feature) != 33 or not all(math.isfinite(value) for value in feature):
        raise ValueError("whole-body diagnostic 33D feature differs")
    return feature


def shared_constraint_feature(
    exact_case: Mapping[str, Any], candidate: Mapping[str, Any], row_index: int,
) -> list[float]:
    from main.multilink_ellipsoid.prospective_l5_context_ablation import obstacle_aabb

    row_index = int(row_index)
    if row_index not in CLAIMED_ROWS:
        raise ValueError("whole-body diagnostic row is not claimed")
    row = exact_case["initial_exact_robot_rows"][row_index]
    center = [float(value) for value in row["center_m"]]
    semiaxes = [float(value) for value in row["semiaxes_m"]]
    obstacle_center, _ = obstacle_aabb(exact_case["initial_compiled_obstacle_boxes"])
    world_delta = [float(obstacle_center[i]) - center[i] for i in range(3)]
    local_delta = _transpose_matvec(row["rotation"], world_delta)
    normalized_relative = [local_delta[i] / semiaxes[i] for i in range(3)]
    initial_slack = float(exact_case["exact_group_target"][
        "initial_row_normalized_radial_slack"
    ][row_index])
    context = exact_case["physical_context"]
    q = [float(value) for value in context["arm_joint_position_rad"]]
    qdot = [float(value) for value in context["arm_joint_velocity_rad_s"]]
    goal = [float(value) for value in context["controller_snapshot"]["goal_pos"]]
    eef = [float(value) for value in context["eef_position_m"]]
    osc_error = [goal[i] - eef[i] for i in range(3)]
    nominal = _flatten_actions(exact_case["source_nominal_five_action_chunk"])
    effective = _flatten_actions(candidate["source_executed_actions"])
    residual = [effective[i] - nominal[i] for i in range(35)]
    one_hot = [float(index == row_index) for index in CLAIMED_ROWS]
    feature = (
        normalized_relative + semiaxes + [initial_slack] + q + qdot
        + osc_error + effective + nominal + residual + one_hot
    )
    if (
        len(q) != 7 or len(qdot) != 7
        or len(feature) != SHARED_INPUT_DIMENSION
        or not all(math.isfinite(value) for value in feature)
    ):
        raise ValueError("whole-body diagnostic shared feature differs")
    return feature


def row_future_risk(candidate: Mapping[str, Any], row_index: int) -> float:
    trace = candidate["exact_group_target"]["trace"]
    if not trace:
        raise ValueError("whole-body diagnostic known candidate trace is empty")
    minimum = min(
        float(sample["row_normalized_radial_slack"][int(row_index)])
        for sample in trace
    )
    return -minimum


def build_model(torch: Any, input_dimension: int, output_dimension: int,
                hidden_widths: Sequence[int]) -> Any:
    widths = [int(input_dimension), *[int(value) for value in hidden_widths],
              int(output_dimension)]
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)
