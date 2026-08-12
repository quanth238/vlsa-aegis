"""Task-rejoining five-action sequential-QP detour for archived E05."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .shadow import _numpy


FIVE_ACTION_DETOUR_SCHEMA = "vlsa_distal_five_action_detour_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_five_action_detour_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "case_ids",
        "claim_scope",
        "protocol_id",
        "schema_version",
        "state_protocol",
        "protected_geometry",
        "finite_difference",
        "sequential_qp",
        "success_definition",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("five-action detour config keys differ")
    if value["schema_version"] != FIVE_ACTION_DETOUR_SCHEMA:
        raise ValueError("five-action detour schema differs")
    if value["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("five-action detour case differs")
    if value["state_protocol"] != {
        "activation_step": 182,
        "detour_action_steps": [182, 183, 184, 185, 186],
        "post_detour_live_replan_step": 187,
        "endpoint_preservation": "sum_xyz_correction_over_five_actions_equals_zero",
    }:
        raise ValueError("five-action detour state protocol differs")
    finite = value["finite_difference"]
    if finite != {
        "action_dimensions": [0, 1, 2],
        "perturbation_action": 0.05,
        "scheme": "clipped_central_difference_at_every_sqp_iteration",
    }:
        raise ValueError("five-action detour finite difference differs")
    qp = value["sequential_qp"]
    expected_keys = {
        "action_limit",
        "clearance_buffer_m",
        "clearance_slack_weight",
        "iterations",
        "maximum_abs_total_correction_action",
        "maximum_abs_iteration_step_action",
        "minimum_progress_ratio",
        "maximum_terminal_eef_error_m",
        "nominal_deviation_weight",
        "smoothness_weight",
        "terminal_eef_weight",
    }
    if set(qp) != expected_keys:
        raise ValueError("five-action detour QP keys differ")
    for key in expected_keys - {"iterations"}:
        item = qp[key]
        if isinstance(item, bool) or not math.isfinite(float(item)) or float(item) < 0:
            raise ValueError("five-action detour QP value differs: %s" % key)
    if int(qp["iterations"]) != 5:
        raise ValueError("five-action detour iteration count differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def solve_detour_qp(
    *,
    cp: Any,
    nominal_actions: Any,
    center_correction: Any,
    clearance_trace_m: Any,
    clearance_jacobian_m_per_action: Any,
    terminal_eef_error_m: Any,
    terminal_eef_jacobian_m_per_action: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve one convex local SQP subproblem around an exact rollout."""

    np = _numpy()
    nominal = np.asarray(nominal_actions, dtype=np.float64)
    center = np.asarray(center_correction, dtype=np.float64).reshape(15)
    h = np.asarray(clearance_trace_m, dtype=np.float64).reshape(35)
    rows = np.asarray(clearance_jacobian_m_per_action, dtype=np.float64)
    terminal_error = np.asarray(terminal_eef_error_m, dtype=np.float64).reshape(3)
    terminal_rows = np.asarray(terminal_eef_jacobian_m_per_action, dtype=np.float64)
    if nominal.shape != (5, 7) or rows.shape != (35, 15) or terminal_rows.shape != (3, 15):
        raise ValueError("five-action detour QP shape differs")
    if not all(
        np.all(np.isfinite(item))
        for item in (nominal, center, h, rows, terminal_error, terminal_rows)
    ):
        raise ValueError("five-action detour QP input is nonfinite")
    qp = config["sequential_qp"]
    step = cp.Variable(15)
    clearance_slack = cp.Variable(35, nonneg=True)
    total = center + step
    total_matrix = cp.reshape(total, (5, 3), order="C")
    corrected_xyz = nominal[:, :3] + total_matrix
    smooth = total_matrix[1:, :] - total_matrix[:-1, :]
    objective = cp.Minimize(
        float(qp["nominal_deviation_weight"]) * cp.sum_squares(total)
        + float(qp["smoothness_weight"]) * cp.sum_squares(smooth)
        + float(qp["terminal_eef_weight"])
        * cp.sum_squares(terminal_error + terminal_rows @ step)
        + float(qp["clearance_slack_weight"]) * cp.sum_squares(clearance_slack)
    )
    limit = float(qp["action_limit"])
    correction_limit = float(qp["maximum_abs_total_correction_action"])
    iteration_limit = float(qp["maximum_abs_iteration_step_action"])
    constraints = [
        h + rows @ step + clearance_slack >= float(qp["clearance_buffer_m"]),
        corrected_xyz >= -limit,
        corrected_xyz <= limit,
        total >= -correction_limit,
        total <= correction_limit,
        step >= -iteration_limit,
        step <= iteration_limit,
        cp.sum(total_matrix, axis=0) == 0.0,
    ]
    problem = cp.Problem(objective, constraints)
    started = __import__("time").perf_counter_ns()
    problem.solve(
        solver=cp.OSQP,
        eps_abs=1.0e-7,
        eps_rel=1.0e-7,
        max_iter=20000,
        polish=True,
        warm_start=False,
        verbose=False,
    )
    elapsed = (__import__("time").perf_counter_ns() - started) * 1.0e-9
    valid = problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and step.value is not None
    if not valid:
        return {
            "valid": False,
            "status": str(problem.status),
            "step": None,
            "total_correction": None,
            "objective": None,
            "solve_wall_seconds": elapsed,
        }
    step_value = np.asarray(step.value, dtype=np.float64).reshape(15)
    total_value = center + step_value
    affine_h = h + rows.dot(step_value)
    slack_value = np.asarray(clearance_slack.value, dtype=np.float64).reshape(35)
    endpoint_error = np.sum(total_value.reshape(5, 3), axis=0)
    tolerance = 1.0e-6
    valid = bool(
        np.max(np.abs(endpoint_error)) <= tolerance
        and np.max(np.abs(total_value)) <= correction_limit + tolerance
        and np.max(np.abs(step_value)) <= iteration_limit + tolerance
    )
    return {
        "valid": valid,
        "status": str(problem.status),
        "step": step_value.tolist(),
        "total_correction": total_value.tolist(),
        "affine_minimum_clearance_m": float(np.min(affine_h)),
        "maximum_clearance_slack_m": float(np.max(slack_value)),
        "endpoint_correction_sum": endpoint_error.tolist(),
        "objective": None if problem.value is None else float(problem.value),
        "solve_wall_seconds": elapsed,
    }
