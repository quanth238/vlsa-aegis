"""Grouped multi-task boundary generalization pilot for distal L5--L7 safety."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import _canonical, _numpy


GENERALIZATION_SCHEMA = "vlsa_distal_boundary_generalization_moka10.v1"
GENERALIZATION_DATASET_SCHEMA = "vlsa_distal_boundary_generalization_moka10_dataset.v1"
GENERALIZATION_DATASET_RESULT_SCHEMA = "vlsa_distal_boundary_generalization_moka10_dataset_result.v1"
GENERALIZATION_TRAINING_SCHEMA = "vlsa_distal_boundary_generalization_moka10_training.v1"
GENERALIZATION_RESULT_SCHEMA = "vlsa_distal_boundary_generalization_moka10_result.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_generalization_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("boundary-generalization config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope",
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "table1_population_run_id", "geometry_config_file_sha256",
        "exact_box_config_file_sha256", "split", "state", "sampling",
        "features", "network", "training", "calibration", "projection",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("boundary-generalization config keys differ")
    if config["schema_version"] != GENERALIZATION_SCHEMA:
        raise ValueError("boundary-generalization schema differs")
    if config["protocol_id"] != "vlsa-distal-boundary-generalization-moka10-v1":
        raise ValueError("boundary-generalization protocol differs")
    split = config["split"]
    groups = {
        name: set(split[name])
        for name in (
            "train_task_groups", "validation_task_groups", "test_task_groups"
        )
    }
    if (
        split.get("unit") != "complete_episode_and_task_level_group"
        or split.get("primary_e05_use") != "test_only_never_training_or_calibration"
        or not all(groups.values())
        or groups["train_task_groups"] & groups["validation_task_groups"]
        or groups["train_task_groups"] & groups["test_task_groups"]
        or groups["validation_task_groups"] & groups["test_task_groups"]
    ):
        raise ValueError("boundary-generalization grouped split differs")
    sampling = config["sampling"]
    if sampling != {
        "action_limit": 1.0,
        "trust_region_linf_action": 0.5,
        "grid_points_per_dimension": 8,
        "expected_grid_action_count": 512,
        "boundary_band_m": 0.005,
        "gradient_anchor_count": 32,
        "gradient_anchor_safe_count": 16,
        "gradient_anchor_unsafe_count": 16,
        "finite_difference_epsilon_action": 0.02,
        "expected_gradient_probe_count_per_episode": 192,
        "require_matching_substep_and_obstacle_witness": True,
    }:
        raise ValueError("boundary-generalization sampling differs")
    if config["state"] != {
        "search_start_offset_actions": 20,
        "selection": "first_exact_seven_proxy_nominal_crossing_in_registered_window",
        "critical_constraint_index": 0,
    }:
        raise ValueError("boundary-generalization crossing rule differs")
    if config["network"].get("output_count") != 7:
        raise ValueError("boundary-generalization network output differs")
    if config["training"].get("arms") != [
        "boundary_margin", "boundary_margin_gradient"
    ]:
        raise ValueError("boundary-generalization training arms differ")
    if not math.isclose(
        sum(config["training"]["margin_category_target_fractions"].values()),
        1.0, rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise ValueError("boundary-generalization category weights differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def load_selected_manifest(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    if _sha256(raw) != config["selected_manifest_sha256"]:
        raise ValueError("selected boundary-generalization manifest hash differs")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(rows) != 10 or len({row.get("case_id") for row in rows}) != 10:
        raise ValueError("selected boundary-generalization cases differ")
    group_to_split: dict[str, str] = {}
    for row in rows:
        if set(row) != {
            "action_count", "archived_file_sha256", "archived_payload_sha256",
            "archived_relative_path", "case_id", "collision_first_step",
            "executed_sequence_sha256", "first_relevant_contact_step",
            "protected_links", "split", "stratum", "task_level_group_id",
        }:
            raise ValueError("selected boundary-generalization row keys differ")
        group = str(row["task_level_group_id"])
        split = str(row["split"])
        if split not in {"train", "validation", "test"}:
            raise ValueError("selected boundary-generalization split differs")
        if group in group_to_split and group_to_split[group] != split:
            raise ValueError("task group crosses boundary-generalization splits")
        group_to_split[group] = split
    expected = {
        group: split
        for split, key in (
            ("train", "train_task_groups"),
            ("validation", "validation_task_groups"),
            ("test", "test_task_groups"),
        )
        for group in config["split"][key]
    }
    if group_to_split != expected:
        raise ValueError("selected manifest does not realize registered task split")
    e05 = next(row for row in rows if row["case_id"] == "vlsa-t1-goal-ii-t0-e05")
    if e05["split"] != "test":
        raise ValueError("primary E05 leaked outside test")
    return rows


def grid_actions(nominal_xyz: Sequence[float], config: Mapping[str, Any]) -> tuple[Any, Any, list[Any]]:
    import itertools

    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    settings = config["sampling"]
    limit = float(settings["action_limit"])
    trust = float(settings["trust_region_linf_action"])
    lower = np.maximum(-limit, nominal - trust)
    upper = np.minimum(limit, nominal + trust)
    count = int(settings["grid_points_per_dimension"])
    axes = [np.linspace(lower[index], upper[index], count) for index in range(3)]
    actions = [np.asarray(values, dtype=np.float64) for values in itertools.product(*axes)]
    if len(actions) != int(settings["expected_grid_action_count"]):
        raise ValueError("boundary-generalization grid count differs")
    return lower, upper, actions


def worst_category(minimum_clearance_m: Sequence[float], band_m: float) -> str:
    value = float(min(minimum_clearance_m))
    if 0.0 <= value <= band_m:
        return "boundary_safe"
    if -band_m <= value < 0.0:
        return "boundary_unsafe"
    return "far_safe" if value > band_m else "far_unsafe"


def select_balanced_anchors(
    records: Sequence[Mapping[str, Any]], lower: Sequence[float],
    upper: Sequence[float], config: Mapping[str, Any]
) -> list[int]:
    np = _numpy()
    settings = config["sampling"]
    epsilon = float(settings["finite_difference_epsilon_action"])
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    eligible = []
    for record in records:
        action = np.asarray(record["candidate_xyz"], dtype=np.float64)
        if np.any(action - epsilon < lower_array - 1e-12) or np.any(
            action + epsilon > upper_array + 1e-12
        ):
            continue
        margin = float(min(record["minimum_substep_clearance_m"]))
        if abs(margin) <= float(settings["boundary_band_m"]):
            eligible.append((abs(margin), int(record["grid_index"]), margin))
    safe = sorted(item for item in eligible if item[2] >= 0.0)
    unsafe = sorted(item for item in eligible if item[2] < 0.0)
    safe_count = int(settings["gradient_anchor_safe_count"])
    unsafe_count = int(settings["gradient_anchor_unsafe_count"])
    if len(safe) < safe_count or len(unsafe) < unsafe_count:
        raise ValueError(
            "episode lacks balanced boundary anchors: safe=%d unsafe=%d"
            % (len(safe), len(unsafe))
        )
    return sorted(
        [item[1] for item in safe[:safe_count]]
        + [item[1] for item in unsafe[:unsafe_count]]
    )
