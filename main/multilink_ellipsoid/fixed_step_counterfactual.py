"""Contract helpers for the E05 fixed-step counterfactual field gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .shadow import _numpy


FIXED_STEP_SCHEMA = "vlsa_distal_fixed_step_counterfactual_field_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_fixed_step_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "action_space",
        "case_ids",
        "claim_scope",
        "comparators",
        "field_estimation",
        "gate",
        "prior_iterative_result",
        "protected_geometry",
        "protocol_id",
        "reference_continuation",
        "risk",
        "state_protocol",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("fixed-step config keys differ")
    if value.get("protocol_id") != "vlsa-distal-fixed-step-counterfactual-field-e05-v1":
        raise ValueError("fixed-step protocol differs")
    if value.get("case_ids") != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("fixed-step case differs")
    if value.get("action_space") != {
        "action_limit": 1.0,
        "corrected_dimensions": [0, 1, 2],
        "corrected_horizon": 5,
        "endpoint_preservation": False,
        "maximum_total_correction_l2_action": 1.0,
    }:
        raise ValueError("fixed-step action space differs")
    estimation = value.get("field_estimation", {})
    if estimation != {
        "direction_seed": 26081221,
        "fixed_normalized_step_action": 0.1,
        "maximum_iterations": 10,
        "paired_direction_count_per_iteration": 32,
        "paired_perturbation_action": 0.05,
        "ridge": 1e-06,
    }:
        raise ValueError("fixed-step field schedule differs")
    if value.get("risk") != {
        "definition": "maximum_over_link_and_action_of_negative_ellipsoid_margin",
        "evaluation_horizon_actions": 20,
        "positive_part_disabled": True,
        "task_penalty_excluded": True,
    }:
        raise ValueError("fixed-step risk differs")
    if value.get("gate") != {
        "minimum_exact_clearance_m": 0.0,
        "paper_car_threshold_m": 0.001,
        "protected_raw_contact_count": 0,
    }:
        raise ValueError("fixed-step gate differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["schema_version"] = FIXED_STEP_SCHEMA
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def fixed_step_update(
    correction: Any,
    unit_direction: Any,
    *,
    step: float,
    maximum_norm: float,
) -> Any:
    """Take the registered full step, refusing implicit clipping or backtracking."""

    np = _numpy()
    current = np.asarray(correction, dtype=np.float64).reshape(15)
    direction = np.asarray(unit_direction, dtype=np.float64).reshape(15)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1.0e-9:
        raise ValueError("fixed-step direction is not unit norm")
    proposed = current + float(step) * direction
    if float(np.linalg.norm(proposed)) > float(maximum_norm) + 1.0e-10:
        raise ValueError("fixed-step total correction budget exceeded")
    if abs(float(np.linalg.norm(proposed - current)) - float(step)) > 1.0e-9:
        raise ValueError("fixed-step update was implicitly rescaled")
    return proposed
