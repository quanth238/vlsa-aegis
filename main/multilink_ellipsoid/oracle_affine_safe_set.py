"""Oracle conservative affine safe-set representation diagnostic.

This module is opt-in and never enters the ordinary AEGIS execution path.  It
asks whether exact two-step grid labels admit seven affine lower envelopes
whose bounded minimum-deviation QP survives a fresh exact OSC rollout.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ORACLE_AFFINE_SAFE_SET_CONFIG_SCHEMA = (
    "vlsa_distal_two_step_oracle_affine_safe_set_moka_test.v1"
)
ORACLE_AFFINE_SAFE_SET_RESULT_SCHEMA = (
    "vlsa_distal_two_step_oracle_affine_safe_set_moka_test_result.v1"
)
ORACLE_AFFINE_SAFE_SET_VALIDATION_SCHEMA = (
    "vlsa_distal_two_step_oracle_affine_safe_set_moka_test_validation.v1"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_oracle_affine_safe_set_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("oracle affine safe-set config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "source_dataset",
        "test_case_ids", "constraint_order", "affine_certificate",
        "optimizer", "exact_verification", "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("oracle affine safe-set config keys differ")
    if (
        config["schema_version"] != ORACLE_AFFINE_SAFE_SET_CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-two-step-oracle-affine-safe-set-moka-test-v1"
    ):
        raise ValueError("oracle affine safe-set protocol differs")
    if config["test_case_ids"] != [
        "vlsa-t1-goal-ii-t0-e05",
        "vlsa-t1-goal-ii-t0-e10",
        "vlsa-t1-goal-ii-t0-e15",
    ]:
        raise ValueError("oracle affine safe-set test cases differ")
    if config["constraint_order"] != [
        "L5_part_0", "L5_part_1", "L5_part_2", "L6_part_0",
        "L6_part_1", "L7_part_0", "L7_part_1",
    ]:
        raise ValueError("oracle affine safe-set constraint order differs")
    source = config["source_dataset"]
    if set(source) != {
        "dataset_file_sha256", "dataset_result_file_sha256",
        "dataset_validation_file_sha256", "dataset_source_commit",
        "expected_grid_count_per_case", "use_records",
    } or source["use_records"] != "source_equals_grid_only":
        raise ValueError("oracle affine safe-set dataset identity differs")
    certificate = config["affine_certificate"]
    if certificate != {
        "candidate_order": "exact_safe_then_l2_from_nominal_then_grid_index",
        "fit": "candidate_conditioned_minimum_l1_sampled_grid_lower_envelope",
        "one_sided_padding_m": 1.0e-6,
        "target_clearance_m": 0.0,
        "coefficient_postcheck_tolerance_m": 1.0e-8,
        "candidate_requires_distal_proxy_and_raw_simulator_safety": True,
    }:
        raise ValueError("oracle affine safe-set certificate contract differs")
    optimizer = config["optimizer"]
    if set(optimizer) != {
        "bound_tolerance_action", "eps_abs", "eps_rel", "max_iter",
        "residual_tolerance",
    }:
        raise ValueError("oracle affine safe-set optimizer keys differ")
    for key in (
        "bound_tolerance_action", "eps_abs", "eps_rel", "residual_tolerance"
    ):
        if (
            isinstance(optimizer[key], bool)
            or not math.isfinite(float(optimizer[key]))
            or float(optimizer[key]) <= 0.0
        ):
            raise ValueError("optimizer.%s must be finite and positive" % key)
    if isinstance(optimizer["max_iter"], bool) or int(optimizer["max_iter"]) < 1:
        raise ValueError("optimizer.max_iter must be positive")
    if config["exact_verification"] != {
        "horizon_actions": 2,
        "second_action": "immutable_released_AEGIS_nominal",
        "osc_internal_substeps": "all",
        "require_all_seven_distal_margins_nonnegative": True,
        "require_released_AEGIS_EE_proxy_nonnegative": True,
        "require_zero_raw_L5_L6_L7_contact": True,
        "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
    }:
        raise ValueError("oracle affine safe-set exact verification differs")
    if config["decision_gate"] != {
        "per_case": (
            "sampled_grid_safe_candidate_and_seven_affine_certificates_and_"
            "valid_qp_and_exact_two_step_all_eight_proxy_raw_safe"
        ),
        "representation_go": "every_registered_test_case_passes",
        "go_authorizes": (
            "multi_state_affine_coefficient_target_collection_not_neural_"
            "closed_loop_efficacy"
        ),
        "no_go_interpretation": (
            "registered_single_affine_sampled_grid_certificate_is_"
            "insufficient_not_all_piecewise_or_nonlinear_safe_sets"
        ),
    }:
        raise ValueError("oracle affine safe-set decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def fit_candidate_conditioned_affine_certificate(
    candidate_xyz: Sequence[Sequence[float]],
    minimum_clearance_m: Sequence[Sequence[float]],
    nominal_xyz: Sequence[float],
    raw_safe: Sequence[bool],
    grid_indexes: Sequence[int],
    *,
    one_sided_padding_m: float,
    target_clearance_m: float,
    postcheck_tolerance_m: float,
    expected_constraint_count: int = 7,
) -> dict[str, Any]:
    """Find the closest sampled-safe action admitting all affine envelopes.

    Each row is a minimum-L1-gradient affine function constrained to lie below
    every exact grid label (minus fixed padding) and to certify the selected
    target action.  The selected target is the first feasible candidate under
    the registered distance/index ordering.
    """

    import numpy as np
    from scipy.optimize import linprog

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    margins = np.asarray(minimum_clearance_m, dtype=np.float64)
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    raw = np.asarray(raw_safe, dtype=bool)
    indexes = np.asarray(grid_indexes, dtype=np.int64)
    row_count = int(expected_constraint_count)
    if (
        xyz.ndim != 2 or xyz.shape[1] != 3
        or row_count < 1 or margins.shape != (xyz.shape[0], row_count)
        or nominal.shape != (3,) or raw.shape != (xyz.shape[0],)
        or indexes.shape != (xyz.shape[0],)
        or len(set(indexes.tolist())) != len(indexes)
        or any(not np.all(np.isfinite(value)) for value in (xyz, margins, nominal))
    ):
        raise ValueError("oracle affine safe-set arrays are invalid")
    padding = float(one_sided_padding_m)
    target = float(target_clearance_m)
    tolerance = float(postcheck_tolerance_m)
    if padding <= 0.0 or tolerance <= 0.0 or not all(
        math.isfinite(value) for value in (padding, target, tolerance)
    ):
        raise ValueError("oracle affine safe-set thresholds are invalid")
    offsets = xyz - nominal[None, :]
    exact_safe = np.logical_and(np.all(margins >= target + padding, axis=1), raw)
    eligible = np.flatnonzero(exact_safe)
    order = sorted(
        eligible.tolist(),
        key=lambda item: (
            float(np.linalg.norm(offsets[item])), int(indexes[item])
        ),
    )
    base_grid = np.column_stack(
        (np.ones(len(xyz), dtype=np.float64), offsets, np.zeros((len(xyz), 3)))
    )
    absolute_rows = []
    for dimension in range(3):
        positive = np.zeros(7, dtype=np.float64)
        positive[1 + dimension] = 1.0
        positive[4 + dimension] = -1.0
        negative = np.zeros(7, dtype=np.float64)
        negative[1 + dimension] = -1.0
        negative[4 + dimension] = -1.0
        absolute_rows.extend((positive, negative))
    objective = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    bounds = [(None, None)] * 4 + [(0.0, None)] * 3
    attempts = []
    for candidate_index in order:
        coefficients = []
        row_audits = []
        all_feasible = True
        target_row = np.zeros(7, dtype=np.float64)
        target_row[:4] = -np.concatenate(([1.0], offsets[candidate_index]))
        for row in range(row_count):
            a_ub = np.vstack((base_grid, absolute_rows, target_row[None, :]))
            b_ub = np.concatenate(
                (
                    margins[:, row] - padding,
                    np.zeros(len(absolute_rows), dtype=np.float64),
                    np.asarray([-target], dtype=np.float64),
                )
            )
            solution = linprog(
                objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs"
            )
            audit = {
                "constraint_index": row,
                "status": int(solution.status),
                "success": bool(solution.success),
                "message": str(solution.message),
            }
            row_audits.append(audit)
            if not solution.success or solution.x is None:
                all_feasible = False
                break
            coefficients.append(np.asarray(solution.x[:4], dtype=np.float64))
        attempts.append(
            {
                "grid_index": int(indexes[candidate_index]),
                "candidate_array_index": int(candidate_index),
                "correction_l2": float(np.linalg.norm(offsets[candidate_index])),
                "all_rows_feasible": all_feasible,
                "row_audits": row_audits,
            }
        )
        if not all_feasible:
            continue
        values = np.asarray(coefficients, dtype=np.float64)
        intercept = values[:, 0]
        gradients = values[:, 1:4]
        lower = intercept[None, :] + offsets @ gradients.T
        maximum_overbound = float(np.max(lower - (margins - padding)))
        target_lower = lower[candidate_index]
        false_safe = np.logical_and(lower >= target, margins < target)
        if (
            maximum_overbound > tolerance
            or float(np.min(target_lower)) < target - tolerance
            or np.any(false_safe)
        ):
            raise RuntimeError("oracle affine safe-set certificate postcheck failed")
        return {
            "valid": True,
            "reason": "certificate_found",
            "grid_record_count": int(len(xyz)),
            "exact_proxy_raw_safe_grid_candidate_count": int(len(eligible)),
            "selected_candidate_array_index": int(candidate_index),
            "selected_grid_index": int(indexes[candidate_index]),
            "selected_candidate_xyz": xyz[candidate_index].tolist(),
            "selected_candidate_exact_margin_m": margins[candidate_index].tolist(),
            "selected_candidate_lower_bound_m": target_lower.tolist(),
            "selected_candidate_correction_l2": float(
                np.linalg.norm(offsets[candidate_index])
            ),
            "intercept_at_nominal_m": intercept.tolist(),
            "gradients_m_per_action": gradients.tolist(),
            "maximum_sampled_grid_overbound_m": maximum_overbound,
            "sampled_grid_false_safe_candidate_count": int(
                np.count_nonzero(np.any(false_safe, axis=1))
            ),
            "attempt_count": len(attempts),
            "attempts": attempts,
        }
    return {
        "valid": False,
        "reason": (
            "no_exact_proxy_raw_safe_grid_candidate"
            if not len(eligible)
            else (
                "no_seven_row_affine_certificate"
                if row_count == 7
                else "no_%d_row_affine_certificate" % row_count
            )
        ),
        "grid_record_count": int(len(xyz)),
        "exact_proxy_raw_safe_grid_candidate_count": int(len(eligible)),
        "attempt_count": len(attempts),
        "attempts": attempts,
    }


def solve_affine_certificate_qp(
    nominal_xyz: Sequence[float],
    action_lower: Sequence[float],
    action_upper: Sequence[float],
    certificate: Mapping[str, Any],
    optimizer: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from .qp import MultiConstraintQp

    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    lower = np.asarray(action_lower, dtype=np.float64)
    upper = np.asarray(action_upper, dtype=np.float64)
    if not certificate.get("valid"):
        return {
            "valid": False, "reason": "affine_certificate_invalid",
            "candidate_xyz": None, "diagnostics": {},
        }
    intercept = np.asarray(certificate["intercept_at_nominal_m"], dtype=np.float64)
    gradients = np.asarray(certificate["gradients_m_per_action"], dtype=np.float64)
    qp = MultiConstraintQp(
        eps_abs=float(optimizer["eps_abs"]),
        eps_rel=float(optimizer["eps_rel"]),
        max_iter=int(optimizer["max_iter"]),
        residual_tolerance=float(optimizer["residual_tolerance"]),
        bound_tolerance=float(optimizer["bound_tolerance_action"]),
    )
    result = qp.solve(
        nominal, np.eye(3), gradients,
        -intercept + gradients @ nominal,
        lower, upper,
    )
    candidate = None if result.qdot_safe is None else np.asarray(
        result.qdot_safe, dtype=np.float64
    )
    lower_bound = None
    if candidate is not None:
        lower_bound = intercept + gradients @ (candidate - nominal)
    return {
        "valid": bool(result.valid),
        "reason": result.reason,
        "candidate_xyz": None if candidate is None else candidate.tolist(),
        "correction_l2": (
            None if candidate is None else float(np.linalg.norm(candidate - nominal))
        ),
        "predicted_conservative_lower_bound_m": (
            None if lower_bound is None else lower_bound.tolist()
        ),
        "diagnostics": dict(result.diagnostics),
    }
