"""Exact discrete policy-value targets for a fixed backup rollout.

The sign convention follows PNCBF: positive values are unsafe.  These helpers
construct finite-horizon targets only; they do not claim that a neural
approximation is a control barrier function.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def exact_group_action_boundary_values(
    boundary_records: Sequence[Mapping[str, Any]],
    substep_trace: Sequence[Mapping[str, Any]],
    *,
    group_order: Sequence[str],
) -> dict[str, Any]:
    """Construct exact finite-horizon values at controller action boundaries.

    ``boundary_records`` are control-time states sampled before each executed
    action.  ``substep_trace`` remains the safety authority and contains every
    MuJoCo substep produced by that action.  This deliberately avoids treating
    highly correlated internal simulator substeps as independent training
    states while retaining their worst violation in each value target.

    The sign convention is positive-is-unsafe:
    ``c_j = -normalized_radial_slack_j``.
    """

    groups = [str(item) for item in group_order]
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("exact-group policy-value group order differs")
    if not boundary_records:
        raise ValueError("exact-group policy-value has no action boundaries")

    parsed_boundaries = []
    previous_offset = None
    for raw in boundary_records:
        offset = int(raw["action_offset"])
        phase = str(raw["phase"])
        if previous_offset is not None and offset != previous_offset + 1:
            raise ValueError("exact-group action boundaries are not contiguous")
        previous_offset = offset
        slacks = raw["group_normalized_radial_slack"]
        if set(slacks) != set(groups):
            raise ValueError("exact-group boundary slack keys differ")
        current = {group: float(slacks[group]) for group in groups}
        if not all(math.isfinite(item) for item in current.values()):
            raise ValueError("exact-group boundary slack is not finite")
        parsed_boundaries.append((offset, phase, current))

    samples_by_offset: dict[int, list[Mapping[str, Any]]] = {
        offset: [] for offset, _, _ in parsed_boundaries
    }
    for raw in substep_trace:
        offset = int(raw["action_offset"])
        if offset not in samples_by_offset:
            raise ValueError("exact-group substep has no action boundary")
        slacks = raw["group_normalized_radial_slack"]
        if set(slacks) != set(groups):
            raise ValueError("exact-group substep slack keys differ")
        samples_by_offset[offset].append(raw)
    if any(not samples for samples in samples_by_offset.values()):
        raise ValueError("exact-group action has no internal substeps")

    successor: dict[str, float] | None = None
    records_reversed = []
    for offset, phase, current_slack in reversed(parsed_boundaries):
        action_max = {}
        for group in groups:
            action_max[group] = max(
                [-current_slack[group]]
                + [
                    -float(sample["group_normalized_radial_slack"][group])
                    for sample in samples_by_offset[offset]
                ]
            )
        value = dict(action_max) if successor is None else {
            group: max(action_max[group], successor[group]) for group in groups
        }
        residual = max(
            abs(
                value[group]
                - (
                    action_max[group]
                    if successor is None
                    else max(action_max[group], successor[group])
                )
            )
            for group in groups
        )
        records_reversed.append({
            "action_offset": offset,
            "phase": phase,
            "current_violation": {
                group: -current_slack[group] for group in groups
            },
            "action_max_violation": action_max,
            "successor_value": None if successor is None else dict(successor),
            "value": value,
            "bellman_residual": residual,
            "all_groups_safe": max(value.values()) <= 0.0,
        })
        successor = value

    records = list(reversed(records_reversed))
    return {
        "schema_version": "distal_exact_group_action_boundary_value.v1",
        "sign_convention": "c_j=-normalized_radial_slack_j; positive_is_unsafe",
        "value_definition": "V_j(z_t)=max_over_exact_candidate_plus_continuation_c_j",
        "finite_horizon_only": True,
        "group_order": groups,
        "action_boundary_count": len(records),
        "safe_action_boundary_count": sum(
            bool(item["all_groups_safe"]) for item in records
        ),
        "maximum_bellman_residual": max(
            float(item["bellman_residual"]) for item in records
        ),
        "records": records,
    }


def _matrix(value: Any, *, columns: int = 7) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("policy-value trace must be a sequence")
    output = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("policy-value trace row must be a sequence")
        converted = [float(item) for item in row]
        if len(converted) != columns or not all(math.isfinite(item) for item in converted):
            raise ValueError("policy-value trace row differs")
        output.append(converted)
    if not output:
        raise ValueError("policy-value trace is empty")
    return output


def violation_trace(clearance_trace_m: Any, safety_buffer_m: float) -> list[list[float]]:
    """Convert clearance into the paper's positive-is-unsafe convention."""

    buffer_m = float(safety_buffer_m)
    if not math.isfinite(buffer_m) or buffer_m < 0.0:
        raise ValueError("policy-value safety buffer differs")
    return [[buffer_m - item for item in row] for row in _matrix(clearance_trace_m)]


def exact_suffix_policy_values(
    windows: Sequence[Mapping[str, Any]],
    *,
    terminal_tail_clearance_trace_m: Any,
    safety_buffer_m: float,
    expected_substeps_per_action: int,
    boundary_tolerance_m: float,
) -> dict[str, Any]:
    """Construct exact per-row suffix maxima at receding-policy boundaries.

    Each window must contain the clearance trace of the action prefix that was
    actually selected and executed by the fixed policy.  The terminal tail is
    a registered hold/retreat rollout from the final policy state.
    """

    substeps = int(expected_substeps_per_action)
    tolerance = float(boundary_tolerance_m)
    if substeps <= 0 or not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("policy-value sampling contract differs")
    if not windows:
        raise ValueError("policy-value rollout has no windows")

    parsed = []
    previous_step = None
    for raw in windows:
        step = int(raw["step"])
        action_count = int(raw["executed_action_count"])
        if action_count <= 0:
            raise ValueError("policy-value window executed no actions")
        if previous_step is not None and step != previous_step:
            raise ValueError("policy-value window steps are not contiguous")
        previous_step = step + action_count
        trace = _matrix(raw["selected_clearance_trace_m"])
        expected_rows = 1 + substeps * action_count
        if len(trace) != expected_rows:
            raise ValueError("policy-value window trace length differs")
        parsed.append((step, action_count, trace))

    tail_clearance = _matrix(terminal_tail_clearance_trace_m)
    terminal_boundary_error = max(
        abs(parsed[-1][2][-1][row_index] - tail_clearance[0][row_index])
        for row_index in range(7)
    )
    tail_h = violation_trace(tail_clearance, safety_buffer_m)
    successor = [max(row[index] for row in tail_h) for index in range(7)]
    tail_step = parsed[-1][0] + parsed[-1][1]
    records_reversed = []
    maximum_boundary_error = terminal_boundary_error

    for index in range(len(parsed) - 1, -1, -1):
        step, action_count, clearance = parsed[index]
        h_trace = violation_trace(clearance, safety_buffer_m)
        prefix = [max(row[row_index] for row in h_trace) for row_index in range(7)]
        value = [max(prefix[row_index], successor[row_index]) for row_index in range(7)]
        residual = max(
            abs(value[row_index] - max(prefix[row_index], successor[row_index]))
            for row_index in range(7)
        )
        if index + 1 < len(parsed):
            next_clearance = parsed[index + 1][2][0]
            boundary_error = max(
                abs(clearance[-1][row_index] - next_clearance[row_index])
                for row_index in range(7)
            )
            maximum_boundary_error = max(maximum_boundary_error, boundary_error)
        records_reversed.append(
            {
                "state_step": step,
                "executed_action_count": action_count,
                "current_h": h_trace[0],
                "prefix_max_h": prefix,
                "successor_value": list(successor),
                "value": value,
                "bellman_residual": residual,
                "all_constraints_safe": max(value) <= 0.0,
            }
        )
        successor = value

    records = list(reversed(records_reversed))
    maximum_residual = max(item["bellman_residual"] for item in records)
    monotonic = all(
        all(
            records[index]["value"][row] + tolerance
            >= records[index + 1]["value"][row]
            for row in range(7)
        )
        for index in range(len(records) - 1)
    )
    safe_count = sum(bool(item["all_constraints_safe"]) for item in records)
    return {
        "schema_version": "distal_exact_backup_policy_value.v1",
        "sign_convention": "h_j=safety_buffer_m-clearance_j; positive_is_unsafe",
        "value_definition": "V_j(z_t)=max_over_registered_backup_continuation_h_j",
        "finite_horizon_only": True,
        "safety_buffer_m": float(safety_buffer_m),
        "row_count": 7,
        "decision_state_count": len(records),
        "safe_decision_state_count": safe_count,
        "all_decision_states_safe": safe_count == len(records),
        "tail_state_step": tail_step,
        "terminal_tail_value": [float(item) for item in [max(row[index] for row in tail_h) for index in range(7)]],
        "maximum_bellman_residual": maximum_residual,
        "maximum_successor_boundary_error_m": maximum_boundary_error,
        "successor_boundaries_consistent": maximum_boundary_error <= tolerance,
        "nonincreasing_along_backup": monotonic,
        "records": records,
    }
