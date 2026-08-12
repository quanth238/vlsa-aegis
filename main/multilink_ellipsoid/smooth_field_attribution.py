"""Contracts and pure helpers for the direct smooth-field attribution gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .shadow import _numpy


SMOOTH_FIELD_ATTRIBUTION_SCHEMA = "vlsa_distal_smooth_field_attribution_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_smooth_field_attribution_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "action_space",
        "case_ids",
        "claim_scope",
        "comparators",
        "field_estimation",
        "gate",
        "internal_verification",
        "protected_geometry",
        "protocol_id",
        "registered_inputs",
        "state_protocol",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("smooth-field attribution config keys differ")
    if value["protocol_id"] != "vlsa-distal-smooth-field-attribution-e05-v1":
        raise ValueError("smooth-field attribution protocol differs")
    if value["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("smooth-field attribution case differs")
    if value["comparators"]["arms"] != [
        "raw_aegis",
        "earlier_five_action_detour",
        "detour_plus_smooth_compound",
        "direct_raw_aegis_smooth_field",
    ]:
        raise ValueError("smooth-field attribution arms differ")
    action = value["action_space"]
    if action != {
        "action_limit": 1.0,
        "corrected_dimensions": [0, 1, 2],
        "corrected_horizon": 5,
        "line_search_fractions": [1.0, 0.5, 0.25],
        "maximum_iterations": 10,
        "maximum_total_correction_l2_action": 1.0,
        "maximum_total_path_length_action": 1.0,
        "trust_radius_action": 0.1,
    }:
        raise ValueError("smooth-field attribution action space differs")
    estimation = value["field_estimation"]
    if estimation != {
        "direction_seed": 26081241,
        "heldout_direction_count_per_iteration": 8,
        "heldout_direction_seed": 26081242,
        "paired_direction_count_per_iteration": 32,
        "paired_perturbation_action": 0.05,
        "ridge": 1e-06,
        "temperature_m": 0.002,
    }:
        raise ValueError("smooth-field attribution field estimation differs")
    if value["gate"] != {
        "internal_substep_clearance_buffer_m": 0.001,
        "minimum_heldout_direction_cosine": 0.8,
        "minimum_heldout_sign_accuracy": 0.75,
        "paper_car_threshold_m": 0.001,
        "protected_raw_contact_count": 0,
    }:
        raise ValueError("smooth-field attribution gate differs")
    if value["internal_verification"] != {
        "expected_mujoco_substeps_per_action": 25,
        "include_initial_state": True,
        "measurement": "all_mujoco_model_steps_and_initial_state",
        "ordinary_env_step_boundary_equivalence_tolerance": 1e-12,
        "scale_grid_step": 0.025,
    }:
        raise ValueError("smooth-field attribution internal verification differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["schema_version"] = SMOOTH_FIELD_ATTRIBUTION_SCHEMA
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def smooth_min_value(clearances: Any, temperature_m: float) -> float:
    """Stable smooth minimum matching the registered smooth direction."""

    np = _numpy()
    values = np.asarray(clearances, dtype=np.float64).reshape(-1)
    temperature = float(temperature_m)
    if values.size < 1 or not np.all(np.isfinite(values)) or temperature <= 0.0:
        raise ValueError("smooth-field value input differs")
    minimum = float(np.min(values))
    return float(
        minimum
        - temperature
        * np.log(np.sum(np.exp(-(values - minimum) / temperature)))
    )


def scale_grid(step: float) -> list[float]:
    value = float(step)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("smooth-field scale step differs")
    count = int(round(1.0 / value))
    if abs(count * value - 1.0) > 1.0e-12:
        raise ValueError("smooth-field scale step must divide one")
    return [float(index * value) for index in range(count + 1)]


def internal_verification_gate(record: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    return bool(
        float(record["minimum_clearance_m"])
        >= float(gate["internal_substep_clearance_buffer_m"])
        and len(record["protected_contacts"])
        == int(gate["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
    )


def select_smallest_verified_scale(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    safe = [item for item in records if bool(item.get("verification_gate"))]
    if not safe:
        return None
    return min(safe, key=lambda item: float(item["scale"]))
