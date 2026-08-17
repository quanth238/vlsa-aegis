"""Inference-only late-flow pullback QP contracts.

The numerical pullback is a diagnostic approximation to
``d V(x, T_theta(A_tau)) / d A_tau``.  It perturbs the late-flow latent in
physical action-displacement coordinates, terminalizes through the unchanged
remaining pi0.5 flow, and scores only the resulting executable actions.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_late_flow_risk_qp_diagnostic.v1"
RESULT_SCHEMA = "vlsa_late_flow_risk_qp_diagnostic_result.v1"
VARIABLE_DIMENSION = 15
CONSTRAINT_ROWS = (0, 1, 2, 3, 4)


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
        raise ValueError("late-flow risk QP config schema differs")
    if value.get("protocol_id") != "vlsa-late-flow-risk-qp-diagnostic-v1":
        raise ValueError("late-flow risk QP protocol differs")
    source = value.get("source_query", {})
    if not (
        value.get("case_id") == "vlsa-t1-goal-ii-t0-e05"
        and source.get("query_index") == 2
        and source.get("replay_action_count") == 10
        and source.get("terminal_reason") == "all_predicted_unsafe_abstention"
    ):
        raise ValueError("late-flow risk QP source query differs")
    pullback = value.get("numerical_pullback", {})
    if not (
        pullback.get("coordinate")
        == "late_flow_physical_output_displacement_first_five_XYZ"
        and pullback.get("branch_after_euler_step") == 8
        and pullback.get("remaining_euler_steps") == 2
        and pullback.get("variable_dimension") == VARIABLE_DIMENSION
        and float(pullback.get("central_difference_action", 0.0)) == 0.005
        and pullback.get("batch_count") == 3
        and pullback.get("branches_per_batch") == 13
    ):
        raise ValueError("late-flow risk QP pullback differs")
    qp = value.get("QP", {})
    if not (
        qp.get("constraint_rows") == list(CONSTRAINT_ROWS)
        and qp.get("constraint_groups") == ["end_effector", "palm", "L5"]
        and float(qp.get("risk_margin", -1.0)) == 0.0
        and float(qp.get("linearized_tightening", -1.0)) == 0.0001
        and float(qp.get("trust_region_action", 0.0)) == 0.25
        and qp.get("nonlinear_terminal_recheck") is True
    ):
        raise ValueError("late-flow risk QP settings differ")
    forbidden = value.get("forbidden", {})
    if forbidden != {
        "new_data_collection": True,
        "model_training": True,
        "simulator_candidate_rollout": True,
        "QP_action_execution": True,
        "fixed_13_candidate_selection": True,
        "released_AEGIS_EE_QP": True,
    }:
        raise ValueError("late-flow risk QP forbidden settings differ")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def central_difference_batches(
    *, epsilon: float, horizon: int = 10,
) -> list[dict[str, Any]]:
    """Return three fixed 13-branch batches covering +/- all 15 variables."""
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("late-flow pullback epsilon is invalid")
    if int(horizon) != 10:
        raise ValueError("late-flow pullback horizon differs")
    dimensions = list(range(VARIABLE_DIMENSION))
    batches = []
    for batch_index in range(3):
        selected = dimensions[batch_index * 6:(batch_index + 1) * 6]
        residuals = [
            [[0.0, 0.0, 0.0] for _ in range(horizon)] for _ in range(13)
        ]
        records = []
        branch_index = 1
        for dimension in selected:
            slot, axis = divmod(dimension, 3)
            residuals[branch_index][slot][axis] = epsilon
            records.append({
                "dimension": dimension, "sign": 1,
                "branch_index": branch_index,
            })
            branch_index += 1
            residuals[branch_index][slot][axis] = -epsilon
            records.append({
                "dimension": dimension, "sign": -1,
                "branch_index": branch_index,
            })
            branch_index += 1
        # The final batch has only three dimensions.  Unused branches remain
        # zero and are ignored; they are present solely because the already
        # validated terminal-branching server contract has a 13-row batch.
        batches.append({
            "batch_index": batch_index,
            "residuals": residuals,
            "records": records,
        })
    covered = sorted(
        record["dimension"] for batch in batches for record in batch["records"]
        if record["sign"] == 1
    )
    if covered != dimensions:
        raise ValueError("late-flow pullback dimensions differ")
    return batches


def pullback_jacobian(
    *, scored_batches: Sequence[Mapping[str, Any]], epsilon: float,
) -> tuple[list[float], list[list[float]]]:
    """Recover the seven-row central-difference Jacobian from scored banks."""
    import numpy as np

    batches = central_difference_batches(epsilon=float(epsilon))
    if len(scored_batches) != len(batches):
        raise ValueError("late-flow scored batch count differs")
    nominal = None
    jacobian = np.empty((7, VARIABLE_DIMENSION), dtype=np.float64)
    for expected, scored in zip(batches, scored_batches):
        risks = np.asarray(scored["risk_by_branch_and_row"], dtype=np.float64)
        if risks.shape != (13, 7) or not np.all(np.isfinite(risks)):
            raise ValueError("late-flow scored risk bank differs")
        if nominal is None:
            nominal = risks[0].copy()
        elif not np.array_equal(nominal, risks[0]):
            raise ValueError("late-flow pullback nominal risk is not exact")
        by_dimension: dict[int, dict[int, Any]] = {}
        for record in expected["records"]:
            by_dimension.setdefault(int(record["dimension"]), {})[
                int(record["sign"])
            ] = risks[int(record["branch_index"])]
        for dimension, signed in by_dimension.items():
            if set(signed) != {-1, 1}:
                raise ValueError("late-flow pullback central pair differs")
            jacobian[:, dimension] = (
                signed[1] - signed[-1]
            ) / (2.0 * float(epsilon))
    if nominal is None or not np.all(np.isfinite(jacobian)):
        raise ValueError("late-flow pullback Jacobian is invalid")
    return nominal.tolist(), jacobian.tolist()


def solve_pullback_qp(
    *, nominal_risk: Sequence[float], jacobian: Sequence[Sequence[float]],
    constraint_rows: Sequence[int] = CONSTRAINT_ROWS,
    risk_margin: float, linearized_tightening: float,
    trust_region_action: float,
) -> dict[str, Any]:
    """Solve one minimum-norm QP in the late-flow correction coordinates."""
    import numpy as np

    from main.multilink_ellipsoid.qp import MultiConstraintQp

    risk = np.asarray(nominal_risk, dtype=np.float64)
    matrix = np.asarray(jacobian, dtype=np.float64)
    rows = tuple(int(row) for row in constraint_rows)
    margin = float(risk_margin)
    tightening = float(linearized_tightening)
    trust = float(trust_region_action)
    if (
        risk.shape != (7,) or matrix.shape != (7, VARIABLE_DIMENSION)
        or not np.all(np.isfinite(risk)) or not np.all(np.isfinite(matrix))
        or rows != CONSTRAINT_ROWS
        or min(margin, tightening) < 0.0 or trust <= 0.0
    ):
        raise ValueError("late-flow pullback QP inputs differ")
    active_risk = risk[list(rows)]
    active_jacobian = matrix[list(rows)]
    solved = MultiConstraintQp().solve(
        qdot_nominal=np.zeros(VARIABLE_DIMENSION, dtype=np.float64),
        metric=np.eye(VARIABLE_DIMENSION, dtype=np.float64),
        rows=-active_jacobian,
        lower_rows=active_risk + margin + tightening,
        velocity_lower=np.full(VARIABLE_DIMENSION, -trust, dtype=np.float64),
        velocity_upper=np.full(VARIABLE_DIMENSION, trust, dtype=np.float64),
    )
    correction = None
    linearized = None
    if solved.valid and solved.qdot_safe is not None:
        correction_array = np.asarray(solved.qdot_safe, dtype=np.float64)
        if correction_array.shape != (VARIABLE_DIMENSION,):
            raise ValueError("late-flow pullback QP solution differs")
        correction = correction_array.tolist()
        linearized = (risk + matrix @ correction_array).tolist()
    return {
        "constraint_rows": list(rows),
        "QP_valid": bool(solved.valid),
        "QP_reason": str(solved.reason),
        "QP_diagnostics": dict(solved.diagnostics),
        "correction": correction,
        "correction_l2_action": (
            None if correction is None else float(np.linalg.norm(correction))
        ),
        "correction_linf_action": (
            None if correction is None
            else float(np.max(np.abs(np.asarray(correction))))
        ),
        "linearized_risk_by_row": linearized,
        "linearized_constraints_satisfied": bool(
            linearized is not None
            and max(float(linearized[row]) for row in rows) <= -margin
        ),
    }


def correction_residual_bank(correction: Sequence[float]) -> list[list[list[float]]]:
    """Encode one QP correction and zero fillers for terminal re-evaluation."""
    import numpy as np

    value = np.asarray(correction, dtype=np.float64)
    if value.shape != (VARIABLE_DIMENSION,) or not np.all(np.isfinite(value)):
        raise ValueError("late-flow QP correction differs")
    bank = np.zeros((13, 10, 3), dtype=np.float64)
    bank[1, :5] = value.reshape(5, 3)
    return bank.tolist()
