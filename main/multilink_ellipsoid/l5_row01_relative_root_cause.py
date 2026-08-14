"""Physics-aligned representation contracts for the L5 row-0/row-1 audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_row01_relative_root_cause.v1"
RESULT_SCHEMA = "vlsa_distal_l5_row01_relative_root_cause_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_row01_relative_root_cause_validation.v1"
RELATIVE_DIMENSION = 134
STATE_INDICES = tuple(range(20)) + tuple(range(90, RELATIVE_DIMENSION))


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
        "schema_version", "protocol_id", "claim_scope", "source",
        "relative_input", "matched_MLP", "baselines", "audit",
        "interpretation", "gates", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("relative root-cause config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-distal-l5-row01-relative-root-cause-v1"
        or value["relative_input"]["input_dimension"] != RELATIVE_DIMENSION
        or value["matched_MLP"]["input_dimension"] != RELATIVE_DIMENSION
        or value["matched_MLP"]["output_count"] != 2
        or value["matched_MLP"]["hidden_widths"] != [32, 32]
        or value["relative_input"]["learned_rows"] != [0, 1]
    ):
        raise ValueError("relative root-cause protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _array(value: Any, shape: tuple[int, ...], name: str) -> Any:
    import numpy as np

    output = np.asarray(value, dtype=np.float64)
    if output.shape != shape or not np.all(np.isfinite(output)):
        raise ValueError("relative field differs: %s" % name)
    return output


def _actions_in_frame(actions: Any, frame_rotation: Any) -> Any:
    import numpy as np

    raw = _array(actions, (5, 7), "actions")
    rotation = _array(frame_rotation, (3, 3), "action_frame_rotation")
    output = raw.copy()
    output[:, :3] = raw[:, :3] @ rotation
    output[:, 3:6] = raw[:, 3:6] @ rotation
    if not np.all(np.isfinite(output)):
        raise ValueError("relative actions are nonfinite")
    return output


def relative_feature_vector(
    *, physical_context: Mapping[str, Any], nominal_actions: Any,
    candidate_actions: Any,
) -> list[float]:
    """Represent geometry and Cartesian vectors in attached physical frames."""

    import numpy as np

    obstacle = physical_context["obstacle"]
    obstacle_center = _array(obstacle["center_m"], (3,), "obstacle_center")
    obstacle_rotation = _array(obstacle["rotation"], (3, 3), "obstacle_rotation")
    obstacle_semiaxes = _array(obstacle["semiaxes_m"], (3,), "obstacle_semiaxes")
    controller = physical_context["controller_snapshot"]
    eef_position = _array(physical_context["eef_position_m"], (3,), "eef_position")
    goal_error = obstacle_rotation.T @ (
        _array(controller["goal_pos"], (3,), "goal_pos") - eef_position
    )
    orientation_error = obstacle_rotation.T @ _array(
        controller["relative_ori"], (3,), "relative_ori"
    )
    output = []
    output.extend(_array(
        physical_context["arm_joint_position_rad"], (7,), "q"
    ).tolist())
    output.extend(_array(
        physical_context["arm_joint_velocity_rad_s"], (7,), "dq"
    ).tolist())
    output.extend(goal_error.tolist())
    output.extend(orientation_error.tolist())
    output.extend(_actions_in_frame(
        nominal_actions, obstacle_rotation
    ).reshape(-1).tolist())
    output.extend(_actions_in_frame(
        candidate_actions, obstacle_rotation
    ).reshape(-1).tolist())
    rows = physical_context["geometry_rows"]
    if len(rows) != 7 or any(row["body_name"] != "robot0_link5" for row in rows[:2]):
        raise ValueError("relative L5 row identities differ")
    for index, row in enumerate(rows[:2]):
        center = _array(row["center_m"], (3,), "row%d_center" % index)
        rotation = _array(row["rotation"], (3, 3), "row%d_rotation" % index)
        relative_center = rotation.T @ (obstacle_center - center)
        relative_rotation = rotation.T @ obstacle_rotation
        normal = rotation.T @ _array(
            row["outward_normal"], (3,), "row%d_normal" % index
        )
        clearance = float(row["current_clearance_m"])
        if not math.isfinite(clearance):
            raise ValueError("relative clearance is nonfinite")
        output.extend(relative_center.tolist())
        output.extend(relative_rotation.reshape(-1).tolist())
        output.extend(_array(
            row["semiaxes_m"], (3,), "row%d_semiaxes" % index
        ).tolist())
        output.extend(obstacle_semiaxes.tolist())
        output.append(clearance)
        output.extend(normal.tolist())
    result = np.asarray(output, dtype=np.float64)
    if result.shape != (RELATIVE_DIMENSION,) or not np.all(np.isfinite(result)):
        raise ValueError("relative feature vector differs")
    return result.tolist()


def witness_phase(candidate: Mapping[str, Any]) -> list[str]:
    prefix = _array(candidate["candidate_prefix_risk"], (7,), "prefix_risk")
    backup_raw = candidate.get("backup_risk")
    if backup_raw is None:
        return ["prefix", "prefix"]
    backup = _array(backup_raw, (7,), "backup_risk")
    combined = _array(candidate["combined_risk"], (7,), "combined_risk")
    if not all(abs(combined[index] - max(prefix[index], backup[index])) <= 1.0e-12
               for index in range(7)):
        raise ValueError("combined risk is not prefix-plus-backup maximum")
    return [
        "prefix" if prefix[index] >= backup[index] else "backup"
        for index in range(2)
    ]


def _normalization(train_x: Any) -> tuple[Any, Any]:
    import numpy as np

    mean = np.mean(train_x, axis=0)
    scale = np.where(np.std(train_x, axis=0) >= 1.0e-6, np.std(train_x, axis=0), 1.0)
    return mean, scale


def knn_predict(train_x: Any, train_y: Any, query_x: Any, k: int) -> Any:
    import numpy as np

    train = np.asarray(train_x, dtype=np.float64)
    target = np.asarray(train_y, dtype=np.float64)
    query = np.asarray(query_x, dtype=np.float64)
    mean, scale = _normalization(train)
    train_z = (train - mean) / scale
    query_z = (query - mean) / scale
    output = []
    for row in query_z:
        distance = np.sqrt(np.mean((train_z - row) ** 2, axis=1))
        nearest = np.argsort(distance, kind="stable")[:int(k)]
        weight = 1.0 / (distance[nearest] + 1.0e-6)
        output.append(np.sum(target[nearest] * weight[:, None], axis=0) / np.sum(weight))
    return np.asarray(output, dtype=np.float64)


def ridge_predict(train_x: Any, train_y: Any, query_x: Any, ridge_lambda: float) -> Any:
    import numpy as np

    train = np.asarray(train_x, dtype=np.float64)
    target = np.asarray(train_y, dtype=np.float64)
    query = np.asarray(query_x, dtype=np.float64)
    mean, scale = _normalization(train)
    train_z = (train - mean) / scale
    query_z = (query - mean) / scale
    design = np.concatenate([train_z, np.ones((train_z.shape[0], 1))], axis=1)
    query_design = np.concatenate([query_z, np.ones((query_z.shape[0], 1))], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_lambda)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ target
    )
    return query_design @ coefficients


def support_audit(
    train_x: Any, train_y: Any, train_samples: Sequence[Mapping[str, Any]],
    validation_x: Any, validation_y: Any,
    validation_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    train = np.asarray(train_x, dtype=np.float64)
    validation = np.asarray(validation_x, dtype=np.float64)
    train_target = np.asarray(train_y, dtype=np.float64)
    validation_target = np.asarray(validation_y, dtype=np.float64)
    mean, scale = _normalization(train)
    train_z = (train - mean) / scale
    validation_z = (validation - mean) / scale
    nearest_records = []
    target_errors = []
    phase_disagreements = [0, 0]
    for index, row in enumerate(validation_z):
        distance = np.sqrt(np.mean((train_z - row) ** 2, axis=1))
        nearest = int(np.argmin(distance))
        difference = np.abs(validation_target[index] - train_target[nearest])
        target_errors.append(difference)
        for output in range(2):
            phase_disagreements[output] += int(
                validation_samples[index]["witness_phase"][output]
                != train_samples[nearest]["witness_phase"][output]
            )
        nearest_records.append({
            "validation_state_id": validation_samples[index]["state_id"],
            "validation_candidate_name": validation_samples[index]["candidate_name"],
            "nearest_train_state_id": train_samples[nearest]["state_id"],
            "nearest_train_candidate_name": train_samples[nearest]["candidate_name"],
            "relative_134D_RMS_z_distance": float(distance[nearest]),
            "absolute_target_difference_m": difference.tolist(),
            "validation_witness_phase": validation_samples[index]["witness_phase"],
            "nearest_train_witness_phase": train_samples[nearest]["witness_phase"],
        })
    state_indices = np.asarray(STATE_INDICES, dtype=np.int64)
    train_states = {}
    validation_states = {}
    for features, samples, destination in (
        (train, train_samples, train_states),
        (validation, validation_samples, validation_states),
    ):
        for index, sample in enumerate(samples):
            destination.setdefault(sample["state_id"], features[index, state_indices])
    train_state_matrix = np.asarray(list(train_states.values()), dtype=np.float64)
    state_mean, state_scale = _normalization(train_state_matrix)
    train_state_z = (train_state_matrix - state_mean) / state_scale
    train_min = np.min(train_state_matrix, axis=0)
    train_max = np.max(train_state_matrix, axis=0)
    state_records = []
    for state_id, features in validation_states.items():
        z = (features - state_mean) / state_scale
        distance = np.sqrt(np.mean((train_state_z - z) ** 2, axis=1))
        nearest = int(np.argmin(distance))
        outside = (features < train_min) | (features > train_max)
        state_records.append({
            "validation_state_id": state_id,
            "nearest_train_state_id": list(train_states)[nearest],
            "relative_state_RMS_z_distance": float(distance[nearest]),
            "outside_training_range_feature_count": int(np.sum(outside)),
            "state_feature_count": int(features.shape[0]),
        })
    target_errors_array = np.asarray(target_errors, dtype=np.float64)
    return {
        "train_unique_state_count": len(train_states),
        "validation_unique_state_count": len(validation_states),
        "state_records": state_records,
        "nearest_candidate_records": nearest_records,
        "nearest_candidate_mean_absolute_target_difference_m": np.mean(
            target_errors_array, axis=0
        ).tolist(),
        "nearest_candidate_witness_phase_disagreement_rate": [
            value / len(validation_samples) for value in phase_disagreements
        ],
    }
