"""Inference-only trust-region SQP over the frozen compact future-risk model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_learned_risk_sqp_feasibility.v1"
RESULT_SCHEMA = "vlsa_learned_risk_sqp_feasibility_result.v1"
VALIDATION_SCHEMA = "vlsa_learned_risk_sqp_feasibility_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("learned-risk SQP config schema differs")
    if value.get("protocol_id") != "vlsa-learned-risk-sqp-feasibility-v1":
        raise ValueError("learned-risk SQP protocol differs")
    if value.get("case_id") != "vlsa-t1-goal-ii-t0-e05":
        raise ValueError("learned-risk SQP case differs")
    source = value.get("source_query", {})
    if not (
        source.get("query_index") == 2
        and source.get("replay_action_count") == 10
        and source.get("terminal_reason") == "all_predicted_unsafe_abstention"
    ):
        raise ValueError("learned-risk SQP source query differs")
    qp = value.get("SQP", {})
    if not (
        qp.get("variable") == "five_by_three_terminal_XYZ_correction"
        and qp.get("arms") == {"EE_only": [0], "EE_palm_L5": [0, 1, 2, 3, 4]}
        and float(qp.get("risk_margin", 1.0)) == 0.0
        and qp.get("maximum_iterations") == 6
        and float(qp.get("finite_difference_action", 0.0)) == 0.005
        and float(qp.get("trust_region_action", 0.0)) == 0.25
        and float(qp.get("maximum_total_correction_action", 0.0)) == 0.75
        and qp.get("line_search_fractions") == [1.0, 0.5, 0.25, 0.125]
    ):
        raise ValueError("learned-risk SQP settings differ")
    forbidden = value.get("forbidden", {})
    if any(forbidden.get(key) is not False for key in (
        "QP_action_execution", "candidate_future_rollout",
        "simulator_lookahead", "model_training", "new_labels",
        "released_AEGIS_EE_QP",
    )):
        raise ValueError("learned-risk SQP forbidden component enabled")
    output = json.loads(canonical(value))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def solve_sqp(
    nominal_actions: Sequence[Sequence[float]],
    risk_function: Callable[[Any], Sequence[float]],
    *, constraint_rows: Sequence[int], maximum_iterations: int,
    finite_difference_action: float, trust_region_action: float,
    maximum_total_correction_action: float,
    line_search_fractions: Sequence[float], risk_margin: float,
) -> dict[str, Any]:
    """Minimize terminal XYZ correction under linearized learned-risk rows."""
    import numpy as np

    from main.multilink_ellipsoid.qp import MultiConstraintQp

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    if nominal.shape != (5, 7) or not np.all(np.isfinite(nominal)):
        raise ValueError("learned-risk SQP nominal actions differ")
    rows = tuple(int(row) for row in constraint_rows)
    if not rows or not set(rows).issubset(set(range(7))):
        raise ValueError("learned-risk SQP constraint rows differ")
    epsilon = float(finite_difference_action)
    trust = float(trust_region_action)
    maximum = float(maximum_total_correction_action)
    margin = float(risk_margin)
    if not all(math.isfinite(value) and value >= 0.0 for value in (
        epsilon, trust, maximum, margin,
    )) or min(epsilon, trust, maximum) <= 0.0:
        raise ValueError("learned-risk SQP scalar differs")
    fractions = tuple(float(value) for value in line_search_fractions)
    if fractions != (1.0, 0.5, 0.25, 0.125):
        raise ValueError("learned-risk SQP line search differs")

    correction = np.zeros(15, dtype=np.float64)
    solver = MultiConstraintQp()

    def candidate(vector: Any) -> Any:
        output = nominal.copy()
        output[:, :3] = np.clip(
            nominal[:, :3] + np.asarray(vector).reshape(5, 3), -1.0, 1.0,
        )
        return output

    def risks(vector: Any) -> Any:
        value = np.asarray(risk_function(candidate(vector)), dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("learned-risk SQP prediction differs")
        return value

    initial = risks(correction)
    trace = []
    converged = bool(np.max(initial[list(rows)]) <= -margin)
    reason = "nominal_feasible" if converged else "maximum_iterations"
    for iteration in range(int(maximum_iterations)):
        current = risks(correction)
        constrained = current[list(rows)]
        if float(np.max(constrained)) <= -margin:
            converged = True
            reason = "nonlinear_constraints_satisfied"
            break
        jacobian = np.empty((len(rows), 15), dtype=np.float64)
        for dimension in range(15):
            plus = correction.copy()
            minus = correction.copy()
            plus[dimension] += epsilon
            minus[dimension] -= epsilon
            jacobian[:, dimension] = (
                risks(plus)[list(rows)] - risks(minus)[list(rows)]
            ) / (2.0 * epsilon)
        base = nominal[:, :3].reshape(-1)
        lower = np.maximum.reduce((
            np.full(15, -trust), -1.0 - base - correction,
            np.full(15, -maximum) - correction,
        ))
        upper = np.minimum.reduce((
            np.full(15, trust), 1.0 - base - correction,
            np.full(15, maximum) - correction,
        ))
        solved = solver.solve(
            qdot_nominal=-correction,
            metric=np.eye(15, dtype=np.float64),
            rows=-jacobian,
            lower_rows=constrained + margin,
            velocity_lower=lower,
            velocity_upper=upper,
        )
        record = {
            "iteration": int(iteration),
            "risk_by_row_before": current.tolist(),
            "maximum_constrained_risk_before": float(np.max(constrained)),
            "finite_difference_jacobian_sha256": hashlib.sha256(
                jacobian.tobytes()
            ).hexdigest(),
            "QP_valid": bool(solved.valid),
            "QP_reason": str(solved.reason),
            "QP_diagnostics": dict(solved.diagnostics),
            "accepted_line_search_fraction": None,
            "maximum_constrained_risk_after": None,
        }
        if not solved.valid or solved.qdot_safe is None:
            reason = str(solved.reason)
            trace.append(record)
            break
        delta = np.asarray(solved.qdot_safe, dtype=np.float64)
        before = float(np.max(constrained))
        accepted = None
        for fraction in fractions:
            proposal = correction + fraction * delta
            proposal_risk = risks(proposal)
            after = float(np.max(proposal_risk[list(rows)]))
            if after < before - 1.0e-10:
                accepted = (fraction, proposal, proposal_risk, after)
                break
        if accepted is None:
            reason = "line_search_no_improvement"
            trace.append(record)
            break
        fraction, correction, proposal_risk, after = accepted
        record["accepted_line_search_fraction"] = float(fraction)
        record["maximum_constrained_risk_after"] = float(after)
        trace.append(record)
        if after <= -margin:
            converged = True
            reason = "nonlinear_constraints_satisfied"
            break
    final_risk = risks(correction)
    nonlinear_feasible = bool(np.max(final_risk[list(rows)]) <= -margin)
    if nonlinear_feasible:
        converged = True
    return {
        "constraint_rows": list(rows),
        "initial_predicted_risk_by_row": initial.tolist(),
        "final_predicted_risk_by_row": final_risk.tolist(),
        "initial_maximum_constrained_risk": float(np.max(initial[list(rows)])),
        "final_maximum_constrained_risk": float(np.max(final_risk[list(rows)])),
        "all_primary_rows_safe": bool(np.max(final_risk[:5]) <= -margin),
        "nonlinear_constraints_satisfied": nonlinear_feasible,
        "converged": bool(converged),
        "reason": reason,
        "iteration_count": len(trace),
        "correction_l2_action": float(np.linalg.norm(correction)),
        "correction_linf_action": float(np.max(np.abs(correction))),
        "correction_XYZ": correction.reshape(5, 3).tolist(),
        "effective_five_actions": candidate(correction).tolist(),
        "trace": trace,
        "candidate_future_rollout_count": 0,
        "QP_action_executed": False,
    }


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    def without_timing(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): without_timing(item)
                for key, item in value.items() if str(key) != "timing"
            }
        if isinstance(value, list):
            return [without_timing(item) for item in value]
        return value

    return {
        "case_id": result["case_id"],
        "pairing_sha256": result["pairing_sha256"],
        "query_dynamic_state_sha256": result["query_dynamic_state_sha256"],
        "ordinary_terminal_actions_sha256": result[
            "ordinary_terminal_actions_sha256"
        ],
        "nominal_predicted_risk_by_row": result["nominal_predicted_risk_by_row"],
        "arms": without_timing(result["arms"]),
        "candidate_future_rollout_count": result["candidate_future_rollout_count"],
        "QP_action_executed": result["QP_action_executed"],
    }
