"""Pure helpers for the frozen factorized-execution surface-error audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SURFACE_CONFIG_SCHEMA = "vlsa_distal_factorized_surface_error_config.v1"
SURFACE_RESULT_SCHEMA = "vlsa_distal_factorized_surface_error_result.v1"
SURFACE_VALIDATION_SCHEMA = "vlsa_distal_factorized_surface_error_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_surface_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("surface-error config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "surface_measurement", "matched_control",
        "decision_gate", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("surface-error config keys differ")
    if config["schema_version"] != SURFACE_CONFIG_SCHEMA:
        raise ValueError("surface-error config schema differs")
    if config["protocol_id"] != "vlsa-distal-factorized-surface-error-moka10-v1":
        raise ValueError("surface-error protocol differs")
    if config["population"] != {
        "candidate_scope": "untouched_test_random_antithetic_actions_only",
        "expected_action_count": 960,
        "expected_false_safe_action_count": 44,
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05",
            "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
    }:
        raise ValueError("surface-error population differs")
    if config["surface_measurement"] != {
        "constraint_scope": "seven_L5_L6_L7_ellipsoid_rows",
        "geometry": "fixed_k0_exact_box_union_identical_to_factorized_pilot",
        "surface_point": "ellipsoid_support_point_on_exact_active_center_axis_normal",
        "witness": "globally_minimum_exact_q_static_constraint_and_substep",
        "clearance_overestimate_sign": "negative_normal_surface_position_error",
        "compare_center_and_support_contributions": True,
    }:
        raise ValueError("surface-error measurement differs")
    if config["matched_control"] != {
        "class": "correctly_rejected_exact_unsafe_action",
        "matching": "same_state_nearest_exact_dynamic_margin_greedy_without_reuse_then_reuse",
    }:
        raise ValueError("surface-error control differs")
    if config["decision_gate"] != {
        "minimum_false_safe_surface_error_median_excess_m": 0.0005,
        "minimum_paired_surface_error_win_fraction": 0.65,
        "minimum_surface_retraction_margin_overestimate_spearman": 0.5,
        "surface_loss_pilot_only_if_all_pass": True,
    }:
        raise ValueError("surface-error decision gate differs")
    expected_forbidden = {
        "new_simulation_labels": True, "training": True,
        "surface_loss_training": True, "poisson_or_SDF": True,
        "calibration": True, "QP": True, "closed_loop": True,
    }
    if config["forbidden_actions"] != expected_forbidden:
        raise ValueError("surface-error forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
    return output


def ellipsoid_support_point(ellipsoid: Any, direction_world: Any) -> Any:
    """Return the world point maximizing the ellipsoid support direction."""

    import numpy as np

    direction = np.asarray(direction_world, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or not math.isfinite(norm) or norm <= 1.0e-15:
        raise ValueError("surface-error support direction is invalid")
    unit = direction / norm
    shape = np.asarray(ellipsoid.shape_matrix(), dtype=np.float64)
    denominator = float(math.sqrt(float(unit @ shape @ unit)))
    point = np.asarray(ellipsoid.center, dtype=np.float64) + shape @ unit / denominator
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("surface-error support point is invalid")
    return point


def _average_ranks(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("surface-error ranks require a finite vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(left: Any, right: Any) -> float:
    import numpy as np

    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if len(left_rank) < 2:
        raise ValueError("surface-error correlation needs at least two rows")
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator <= 1.0e-15:
        return 0.0
    return float(left_centered @ right_centered / denominator)


def matched_control_indexes(
    *, false_safe_indexes: Any, control_indexes: Any, state_index: Any,
    exact_margin_m: Any,
) -> Any:
    """Match each false-safe row to a same-state boundary-nearest control."""

    import numpy as np

    false_rows = np.asarray(false_safe_indexes, dtype=np.int64)
    controls = np.asarray(control_indexes, dtype=np.int64)
    states = np.asarray(state_index, dtype=np.int64)
    margin = np.asarray(exact_margin_m, dtype=np.float64)
    if not len(false_rows) or not len(controls):
        raise ValueError("surface-error matching population is empty")
    output = []
    for state in sorted(set(states[false_rows].tolist())):
        state_false = false_rows[states[false_rows] == state]
        state_controls = controls[states[controls] == state]
        if not len(state_controls):
            raise ValueError("surface-error false-safe state has no unsafe control")
        available = list(state_controls.tolist())
        for false_row in sorted(state_false.tolist(), key=lambda row: margin[row]):
            pool = available if available else state_controls.tolist()
            selected = min(
                pool,
                key=lambda row: (abs(margin[row] - margin[false_row]), int(row)),
            )
            output.append((int(false_row), int(selected)))
            if selected in available:
                available.remove(selected)
    output.sort(key=lambda pair: pair[0])
    return np.asarray([pair[1] for pair in output], dtype=np.int64)


def vector_summary(values: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("surface-error summary requires a nonempty finite vector")
    return {
        "count": int(len(array)), "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)), "maximum": float(np.max(array)),
    }


def surface_loss_decision(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gate = config["decision_gate"]
    tests = {
        "source_false_safe_count": int(metrics["false_safe_action_count"])
        == int(config["population"]["expected_false_safe_action_count"]),
        "recomputed_geometry_exact": bool(metrics["recomputed_geometry_exact"]),
        "false_safe_surface_error_excess": float(
            metrics["false_safe_surface_error_median_excess_m"]
        ) >= float(gate["minimum_false_safe_surface_error_median_excess_m"]),
        "paired_surface_error_win_fraction": float(
            metrics["paired_surface_error_win_fraction"]
        ) >= float(gate["minimum_paired_surface_error_win_fraction"]),
        "surface_retraction_correlation": float(
            metrics["unsafe_surface_retraction_margin_overestimate_spearman"]
        ) >= float(gate["minimum_surface_retraction_margin_overestimate_spearman"]),
    }
    supported = bool(all(tests.values()))
    return {
        "gate_tests": tests,
        "surface_loss_pilot_supported": supported,
        "conclusion": (
            "false_safe_surface_error_supports_targeted_surface_loss_pilot"
            if supported else
            "surface_position_not_decisive_skip_surface_loss_and_audit_one_sided_uncertainty"
        ),
        "training_authorized_in_this_diagnostic": False,
        "poisson_calibration_QP_or_closed_loop_authorized": False,
    }
