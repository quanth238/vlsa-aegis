"""Contracts for the raw-AEGIS multi-start Gate-0 binding diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


RAW_MULTISTART_GATE0_SCHEMA = "vlsa_distal_raw_multistart_gate0_e05_config.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_raw_multistart_gate0_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("protocol_id") != "vlsa-distal-raw-multistart-gate0-e05-v1":
        raise ValueError("raw multi-start Gate-0 protocol differs")
    if value.get("case_ids") != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("raw multi-start Gate-0 case differs")
    if value["controls"]["arms"] != [
        "raw_aegis",
        "full_compound_trajectory",
        "transplanted_compound_prefix_raw_suffix",
    ]:
        raise ValueError("raw multi-start Gate-0 arms differ")
    if value["state_protocol"]["compound_prefix_steps"] != list(range(182, 187)):
        raise ValueError("raw multi-start Gate-0 prefix differs")
    if value["state_protocol"]["evaluation_steps"] != list(range(182, 202)):
        raise ValueError("raw multi-start Gate-0 continuation differs")
    if value["normalization_contract"]["displacement_conversion"] != (
        "scale_only_no_normalization_mean_subtraction"
    ):
        raise ValueError("raw multi-start Gate-0 normalization differs")
    if [float(item) for item in value["search_authorization"]["conditional_radii_l2_bounds"]] != [
        1.0,
        1.5,
        2.0,
    ]:
        raise ValueError("raw multi-start Gate-0 conditional radii differ")
    gate = value["gate"]
    if not math.isclose(float(gate["internal_substep_clearance_buffer_m"]), 0.001):
        raise ValueError("raw multi-start Gate-0 clearance gate differs")
    if not math.isclose(float(gate["paper_car_threshold_m"]), 0.001):
        raise ValueError("raw multi-start Gate-0 CAR gate differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["schema_version"] = RAW_MULTISTART_GATE0_SCHEMA
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def internal_gate(record: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    return bool(
        float(record["minimum_clearance_m"])
        >= float(gate["internal_substep_clearance_buffer_m"])
        and len(record["protected_contacts"]) == int(gate["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
    )


def transplant_compound_prefix(raw_actions: Any, compound_actions: Any, action_limit: float) -> dict[str, Any]:
    """Put only the compound first-five XYZ displacement on the raw suffix."""

    import numpy as np

    raw = np.asarray(raw_actions, dtype=np.float64)
    compound = np.asarray(compound_actions, dtype=np.float64)
    if raw.shape != (20, 7) or compound.shape != (20, 7):
        raise ValueError("raw multi-start Gate-0 action shape differs")
    delta = compound[:5, :3] - raw[:5, :3]
    unbounded = raw.copy()
    unbounded[:5, :3] += delta
    clipped = unbounded.copy()
    clipped[:5, :3] = np.clip(clipped[:5, :3], -float(action_limit), float(action_limit))
    applied = clipped[:5, :3] - raw[:5, :3]
    return {
        "actions": clipped,
        "requested_correction": delta,
        "applied_correction": applied,
        "clipping_delta": clipped[:5, :3] - unbounded[:5, :3],
        "clipped_coordinate_count": int(np.count_nonzero(clipped[:5, :3] != unbounded[:5, :3])),
        "requested_l2_action": float(np.linalg.norm(delta)),
        "applied_l2_action": float(np.linalg.norm(applied)),
    }
