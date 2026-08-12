"""Preregistered fixed-repulsion guidance inside final pi0.5 flow steps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .repulsive_force import normalized_direction, softmin_weights
from .shadow import _numpy


REPULSIVE_FLOW_SCHEMA = "vlsa_fixed_repulsion_flow_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_repulsive_flow_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "case_ids",
        "claim_scope",
        "comparison",
        "flow_guidance",
        "nominal_action_source",
        "protected_geometry",
        "protocol_id",
        "repulsive_direction",
        "schema_version",
        "state_protocol",
        "success_definition",
        "verification",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("repulsive-flow config keys differ")
    if value["schema_version"] != REPULSIVE_FLOW_SCHEMA:
        raise ValueError("repulsive-flow schema differs")
    if value["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("repulsive-flow case differs")
    if value["state_protocol"] != {
        "activation_query_step": 180,
        "executed_action_steps": [180, 181, 182, 183, 184],
        "guided_executed_steps": [182, 183, 184],
    }:
        raise ValueError("repulsive-flow state protocol differs")
    if value["flow_guidance"] != {
        "action_limit": 1.0,
        "guided_action_slots": [2, 3, 4],
        "guided_euler_steps": [5, 6, 7, 8, 9],
        "per_euler_step_size_action": 0.05,
        "total_nominal_guidance_budget_action": 0.25,
    }:
        raise ValueError("repulsive-flow schedule differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def five_action_row_model(jacobian: Any, base_h: Any) -> tuple[Any, Any]:
    """Collapse seven-row future traces to active future minima."""

    np = _numpy()
    gradients = np.asarray(jacobian, dtype=np.float64)
    trace = np.asarray(base_h, dtype=np.float64)
    if gradients.shape != (5, 7, 15) or trace.shape != (5, 7):
        raise ValueError("five-action repulsive model shape differs")
    active_steps = np.argmin(trace, axis=0)
    rows = np.asarray(
        [gradients[active_steps[row], row] for row in range(7)],
        dtype=np.float64,
    )
    minima = np.min(trace, axis=0)
    return minima, rows


def fixed_chunk_direction(
    row_minima: Any,
    rows: Any,
    *,
    temperature_m: float,
    guided_slots: list[int],
) -> tuple[Any, Any]:
    """Return one physical XYZ direction repeated over selected action slots."""

    np = _numpy()
    minima = np.asarray(row_minima, dtype=np.float64)
    matrix = np.asarray(rows, dtype=np.float64)
    if minima.shape != (7,) or matrix.shape != (7, 15):
        raise ValueError("repulsive chunk model shape differs")
    weights = softmin_weights(minima, float(temperature_m))
    chunk_gradient = matrix.T.dot(weights).reshape(5, 3)
    selected = np.sum(chunk_gradient[guided_slots], axis=0)
    direction = normalized_direction(selected)
    return direction, weights


def posthoc_chunk(
    nominal_actions: Any,
    direction: Any,
    *,
    guided_slots: list[int],
    total_budget_action: float,
    action_limit: float,
) -> Any:
    np = _numpy()
    actions = np.asarray(nominal_actions, dtype=np.float64).copy()
    unit = np.asarray(direction, dtype=np.float64)
    if actions.shape != (10, 7) or unit.shape != (3,):
        raise ValueError("posthoc repulsive chunk shape differs")
    for slot in guided_slots:
        actions[slot, :3] = np.clip(
            actions[slot, :3] + float(total_budget_action) * unit,
            -float(action_limit),
            float(action_limit),
        )
    return actions


def build_repulsive_flow_envelope(
    nominal_actions: Any,
    direction: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    nominal = np.asarray(nominal_actions, dtype=np.float64)
    unit = normalized_direction(direction)
    flow = config["flow_guidance"]
    if nominal.shape != (10, 7):
        raise ValueError("repulsive-flow nominal chunk shape differs")
    return {
        "schema_version": "crfs_fixed_repulsive_flow_guidance.v1",
        "action_horizon": 10,
        "action_dimensions": [0, 1, 2],
        "physical_output_direction": unit.tolist(),
        "nominal_output_actions": nominal.tolist(),
        "guided_action_slots": list(flow["guided_action_slots"]),
        "guided_euler_steps": list(flow["guided_euler_steps"]),
        "step_size_action": float(flow["per_euler_step_size_action"]),
        "action_limit": float(flow["action_limit"]),
    }
