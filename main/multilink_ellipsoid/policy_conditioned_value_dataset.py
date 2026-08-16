"""Dataset contract for policy-conditioned future-violation learning.

The learned object is a finite-horizon approximation of a PNCBF-style policy
value.  It is not an OSC execution model.  OSC, the robot, contacts, and the
fixed continuation policy are all part of the plant used to create labels.

Positive targets are unsafe.  Query samples represent

    Q_j(z, A) = max(current/prefix violation, V_j^pi(z_plus))

and value samples represent the exact reverse-suffix maximum under the fixed
continuation policy.  Internal MuJoCo substeps remain label authority; they are
not treated as independent training states.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


CONFIG_SCHEMA = "vlsa_distal_policy_conditioned_value_dataset.v1"
DATASET_SCHEMA = "vlsa_distal_policy_conditioned_value_dataset_result.v1"
PHASE_ORDER = ("prefix", "backup", "terminal_hold")
MAX_ROWS_PER_GROUP = 3


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def _finite_vector(value: Any, length: int, label: str) -> List[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(label + " is not a sequence")
    output = [float(item) for item in value]
    if len(output) != length or not all(math.isfinite(item) for item in output):
        raise ValueError(label + " differs")
    return output


def _finite_matrix(value: Any, rows: int, columns: int, label: str) -> List[List[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(label + " is not a sequence")
    output = [_finite_vector(row, columns, label + " row") for row in value]
    if len(output) != rows:
        raise ValueError(label + " differs")
    return output


def _obstacle_aabb(boxes: Sequence[Mapping[str, Any]]) -> Tuple[List[float], List[float]]:
    lower = [math.inf, math.inf, math.inf]
    upper = [-math.inf, -math.inf, -math.inf]
    if not boxes:
        raise ValueError("policy-value context has no compiled obstacle boxes")
    for box in boxes:
        center = _finite_vector(box["center_m"], 3, "compiled-box center")
        rotation = _finite_matrix(box["rotation"], 3, 3, "compiled-box rotation")
        half = _finite_vector(box["half_extents_m"], 3, "compiled-box half extents")
        world_half = [
            sum(abs(rotation[row][column]) * half[column] for column in range(3))
            for row in range(3)
        ]
        for index in range(3):
            lower[index] = min(lower[index], center[index] - world_half[index])
            upper[index] = max(upper[index], center[index] + world_half[index])
    center = [0.5 * (lower[index] + upper[index]) for index in range(3)]
    half = [0.5 * (upper[index] - lower[index]) for index in range(3)]
    if not all(math.isfinite(item) for item in center + half) or not all(
        item > 0.0 for item in half
    ):
        raise ValueError("policy-value obstacle AABB differs")
    return center, half


def constraint_state_feature(
    boundary: Mapping[str, Any], *, group: str,
    group_order: Sequence[str], group_rows: Mapping[str, Sequence[int]],
) -> List[float]:
    """Build a compact constraint-conditioned Markov-state approximation.

    The feature uses only information available at the controller boundary:
    current exact relative geometry, arm state, OSC position error, policy
    phase, and constraint identity.  It contains no future state or label.
    """

    groups = [str(item) for item in group_order]
    if group not in groups or len(groups) != len(set(groups)):
        raise ValueError("policy-value group order differs")
    indexes = [int(item) for item in group_rows[group]]
    if not indexes or len(indexes) > MAX_ROWS_PER_GROUP or len(set(indexes)) != len(indexes):
        raise ValueError("policy-value group rows differ")
    rows = boundary["exact_robot_rows"]
    slacks = [float(item) for item in boundary["row_normalized_radial_slack"]]
    if max(indexes) >= len(rows) or max(indexes) >= len(slacks):
        raise ValueError("policy-value group row binding differs")
    if not all(math.isfinite(slacks[index]) for index in indexes):
        raise ValueError("policy-value current slack differs")
    active_index = min(indexes, key=lambda index: slacks[index])
    active = rows[active_index]
    center = _finite_vector(active["center_m"], 3, "robot-row center")
    semiaxes = _finite_vector(active["semiaxes_m"], 3, "robot-row semiaxes")
    obstacle_center, obstacle_half = _obstacle_aabb(
        boundary["compiled_obstacle_boxes"],
    )
    qpos = _finite_vector(boundary["arm_joint_position_rad"], 7, "arm qpos")
    qvel = _finite_vector(boundary["arm_joint_velocity_rad_s"], 7, "arm qvel")
    eef = _finite_vector(boundary["eef_position_m"], 3, "EEF position")
    controller = boundary["controller_snapshot"]
    goal = _finite_vector(controller["goal_pos"], 3, "OSC goal position")
    phase = str(boundary["phase"])
    if phase not in PHASE_ORDER:
        raise ValueError("policy-value phase differs")
    padded_slacks = [slacks[index] for index in indexes]
    row_mask = [1.0] * len(indexes)
    while len(padded_slacks) < MAX_ROWS_PER_GROUP:
        padded_slacks.append(0.0)
        row_mask.append(0.0)
    feature = (
        [center[index] - obstacle_center[index] for index in range(3)]
        + semiaxes
        + obstacle_half
        + padded_slacks
        + row_mask
        + qpos
        + qvel
        + [goal[index] - eef[index] for index in range(3)]
        + [1.0 if phase == item else 0.0 for item in PHASE_ORDER]
        + [1.0 if group == item else 0.0 for item in groups]
    )
    if not all(math.isfinite(item) for item in feature):
        raise ValueError("policy-value state feature is nonfinite")
    return feature


def action_chunk_feature(
    boundaries: Sequence[Mapping[str, Any]], *, prefix_action_count: int,
    translation_scale_m_per_action_unit: float,
) -> List[float]:
    """Return the complete Cartesian translation prefix in metric units."""

    count = int(prefix_action_count)
    scale = float(translation_scale_m_per_action_unit)
    if count <= 0 or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("policy-value action feature contract differs")
    prefix = [row for row in boundaries if str(row["phase"]) == "prefix"]
    if len(prefix) != count:
        raise ValueError("policy-value prefix action count differs")
    output = []
    for expected_offset, boundary in enumerate(prefix):
        if int(boundary["action_offset"]) != expected_offset:
            raise ValueError("policy-value prefix offsets differ")
        action = _finite_vector(boundary["executed_action"], 7, "executed action")
        output.extend(scale * action[index] for index in range(3))
    return output


def extract_case_samples(
    case: Mapping[str, Any], *, learned_groups: Sequence[str],
    group_order: Sequence[str], group_rows: Mapping[str, Sequence[int]],
    eligible_value_phases: Sequence[str], prefix_action_count: int,
    translation_scale_m_per_action_unit: float,
) -> Dict[str, Any]:
    """Extract exact query-Q and fixed-policy-V samples from one episode state."""

    groups = [str(item) for item in group_order]
    learned = [str(item) for item in learned_groups]
    eligible = {str(item) for item in eligible_value_phases}
    if not learned or not set(learned).issubset(groups):
        raise ValueError("policy-value learned groups differ")
    if not eligible or not eligible.issubset(set(PHASE_ORDER) - {"prefix"}):
        raise ValueError("policy-value eligible phases differ")
    case_id = str(case["case_id"])
    split = str(case["selection"].get("split", "development"))
    q_samples = []
    v_samples = []
    unknown_count = 0
    for candidate in case["exact_case"]["candidates"]:
        exact = candidate["exact_group_target"]
        boundaries = exact.get("action_boundaries", [])
        trajectory = exact.get("trajectory_policy_value", {})
        records = trajectory.get("records", [])
        if len(boundaries) != len(records) or not records:
            raise ValueError("policy-value boundary records differ")
        if not bool(exact["known_outcome"]):
            unknown_count += 1
            if any(bool(record.get("training_sample_eligible")) for record in records):
                raise ValueError("unknown policy-value trajectory is training eligible")
            continue
        action_feature = action_chunk_feature(
            boundaries, prefix_action_count=prefix_action_count,
            translation_scale_m_per_action_unit=translation_scale_m_per_action_unit,
        )
        first = records[0]
        if int(first["action_offset"]) != 0 or str(first["phase"]) != "prefix":
            raise ValueError("policy-value first Bellman record differs")
        for group in learned:
            # The first exact Bellman record includes the current state, every
            # prefix substep, and the complete fixed continuation.  This is the
            # authoritative Q target; do not substitute a one-step OSC target.
            q_samples.append({
                "state_id": case_id,
                "split": split,
                "trajectory_id": case_id + "/" + str(candidate["name"]),
                "candidate_name": str(candidate["name"]),
                "constraint": group,
                "state_feature": constraint_state_feature(
                    boundaries[0], group=group, group_order=groups,
                    group_rows=group_rows,
                ),
                "action_feature": action_feature,
                "target": float(first["value"][group]),
                "safe": float(first["value"][group]) <= 0.0,
            })
        for boundary, record in zip(boundaries, records):
            if (
                int(boundary["action_offset"]) != int(record["action_offset"])
                or str(boundary["phase"]) != str(record["phase"])
            ):
                raise ValueError("policy-value boundary binding differs")
            is_eligible = str(record["phase"]) in eligible
            if bool(record.get("training_sample_eligible")) != is_eligible:
                raise ValueError("policy-value eligibility differs")
            if not is_eligible:
                continue
            for group in learned:
                v_samples.append({
                    "state_id": case_id,
                    "split": split,
                    "trajectory_id": case_id + "/" + str(candidate["name"]),
                    "candidate_name": str(candidate["name"]),
                    "action_offset": int(record["action_offset"]),
                    "phase": str(record["phase"]),
                    "constraint": group,
                    "state_feature": constraint_state_feature(
                        boundary, group=group, group_order=groups,
                        group_rows=group_rows,
                    ),
                    "target": float(record["value"][group]),
                    "safe": float(record["value"][group]) <= 0.0,
                })
    return {
        "case_id": case_id,
        "split": split,
        "query_samples": q_samples,
        "value_samples": v_samples,
        "unknown_candidate_count": unknown_count,
    }


def coverage_summary(
    case_samples: Sequence[Mapping[str, Any]], *, learned_groups: Sequence[str],
) -> Dict[str, Any]:
    """Summarize independent state support; correlated actions do not add states."""

    groups = [str(item) for item in learned_groups]
    splits = sorted({str(case["split"]) for case in case_samples})
    output = {}
    for split in splits:
        split_cases = [case for case in case_samples if case["split"] == split]
        by_group = {}
        for group in groups:
            two_sided = []
            q_count = 0
            v_count = 0
            v_states = set()
            for case in split_cases:
                q_rows = [
                    row for row in case["query_samples"]
                    if row["constraint"] == group
                ]
                v_rows = [
                    row for row in case["value_samples"]
                    if row["constraint"] == group
                ]
                q_count += len(q_rows)
                v_count += len(v_rows)
                if v_rows:
                    v_states.add(str(case["case_id"]))
                if any(bool(row["safe"]) for row in q_rows) and any(
                    not bool(row["safe"]) for row in q_rows
                ):
                    two_sided.append(str(case["case_id"]))
            by_group[group] = {
                "independent_state_count": len(split_cases),
                "two_sided_state_count": len(two_sided),
                "two_sided_state_ids": two_sided,
                "query_sample_count": q_count,
                "value_sample_count": v_count,
                "value_episode_group_count": len(v_states),
            }
        output[split] = {"state_count": len(split_cases), "groups": by_group}
    return output


def readiness_checks(
    coverage: Mapping[str, Any], *, learned_groups: Sequence[str],
    required_two_sided_states: Mapping[str, int],
    required_value_episode_groups: Mapping[str, int],
) -> Dict[str, Any]:
    checks = {}
    for split, minimum in required_two_sided_states.items():
        for group in learned_groups:
            key = "two_sided/{}/{}".format(split, group)
            checks[key] = (
                int(coverage.get(split, {}).get("groups", {}).get(group, {}).get(
                    "two_sided_state_count", 0
                )) >= int(minimum)
            )
    for split, minimum in required_value_episode_groups.items():
        for group in learned_groups:
            key = "value_groups/{}/{}".format(split, group)
            checks[key] = (
                int(coverage.get(split, {}).get("groups", {}).get(group, {}).get(
                    "value_episode_group_count", 0
                )) >= int(minimum)
            )
    checks["training_ready"] = bool(checks and all(checks.values()))
    return checks


def load_config(path: Path) -> Dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "dataset", "features", "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("policy-value dataset config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-policy-conditioned-value-dataset-v1"
        or value["features"]["prefix_action_count"] != 5
        or value["features"]["translation_scale_m_per_action_unit"] != 0.05
        or value["dataset"]["eligible_value_phases"] != ["backup", "terminal_hold"]
        or value["dataset"]["internal_substeps"] != "label_authority_only"
        or value["dataset"]["timeouts"] != "censored_unknown"
    ):
        raise ValueError("policy-value dataset protocol differs")
    groups = value["dataset"]["group_order"]
    learned = value["dataset"]["learned_groups"]
    if not groups or not learned or not set(learned).issubset(set(groups)):
        raise ValueError("policy-value dataset learned groups differ")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output
