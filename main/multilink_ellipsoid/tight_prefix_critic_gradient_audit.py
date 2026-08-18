"""Contracts and pure metrics for the compact-critic gradient audit."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


CONFIG_SCHEMA = "vlsa_tight_prefix_critic_gradient_audit.v1"
RESULT_SCHEMA = "vlsa_tight_prefix_critic_gradient_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_critic_gradient_audit_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "source", "audit",
        "feasibility_gate", "structural_audit", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("critic gradient audit config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-tight-prefix-critic-gradient-audit-v1"
    ):
        raise ValueError("critic gradient audit protocol differs")
    audit = value["audit"]
    expected_pairs = [
        ["grid_m1_m1_m1_front_loaded_r2.0",
         "grid_p1_p1_p1_front_loaded_r2.0"],
        ["grid_m1_m1_z0_front_loaded_r2.0",
         "grid_p1_p1_z0_front_loaded_r2.0"],
        ["grid_m1_z0_m1_front_loaded_r2.0",
         "grid_p1_z0_p1_front_loaded_r2.0"],
        ["grid_m1_p1_z0_front_loaded_r2.0",
         "grid_p1_m1_z0_front_loaded_r2.0"],
        ["grid_z0_m1_m1_front_loaded_r2.0",
         "grid_z0_p1_p1_front_loaded_r2.0"],
        ["grid_z0_z0_m1_front_loaded_r2.0",
         "grid_z0_z0_p1_front_loaded_r2.0"],
    ]
    if (
        audit.get("report_splits") != ["train", "validation", "test"]
        or audit.get("decision_split") != "validation"
        or audit.get("test_status") != "already_opened_diagnostic"
        or audit.get("primary_rows") != list(range(8))
        or audit.get("diagnostic_rows") != [8, 9]
        or audit.get("optimized_action_dimensions")
        != "five_effective_XYZ_actions_only"
        or audit.get("global_gradient")
        != "beta20_logmeanexp_over_primary_row_predictions"
        or float(audit.get("smoothmax_beta", -1.0)) != 20.0
        or audit.get("candidate_pairs") != expected_pairs
    ):
        raise ValueError("critic gradient audit definition differs")
    gate = value["feasibility_gate"]
    if (
        float(gate.get("minimum_validation_unsafe_recall", -1.0)) != 0.8
        or float(gate.get("minimum_validation_direction_accuracy", -1.0))
        != 0.75
        or float(gate.get("minimum_validation_descent_rate", -1.0)) != 0.7
        or int(gate.get("minimum_validation_eligible_pair_count", -1)) != 8
        or int(gate.get("minimum_validation_contributing_root_count", -1)) != 2
        or int(gate.get("maximum_validation_safe_to_unsafe_regressions", -1))
        != 1
        or gate.get("require_negative_validation_median_true_risk_change")
        is not True
        or gate.get("require_autograd_finite_difference_pass") is not True
    ):
        raise ValueError("critic gradient audit gate differs")
    structural = value["structural_audit"]
    if structural != {
        "per_row_action_dimension": 15,
        "per_row_represented_normal_coordinates": 5,
        "per_row_unrepresented_tangent_dimensions": 10,
        "warning": (
            "A global smooth maximum can combine row normals, but one active "
            "row cannot express either tangent direction at any action step."
        ),
    }:
        raise ValueError("critic gradient structural audit differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def smoothmax_weights(values: Sequence[float], beta: float) -> list[float]:
    """Return stable softmax weights for the derivative of log-mean-exp."""
    if not values or not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("critic gradient smooth maximum differs")
    scaled = [float(beta) * float(value) for value in values]
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("critic gradient value is not finite")
    offset = max(scaled)
    weights = [math.exp(value - offset) for value in scaled]
    total = sum(weights)
    return [float(value / total) for value in weights]


def pair_direction_record(
    *, state_id: str, pair_names: Sequence[str], symmetry_error: float,
    predicted_minus_score: float, predicted_plus_score: float,
    nominal_true_risk: float, minus_true_risk: float, plus_true_risk: float,
    symmetry_tolerance: float, score_tolerance: float, risk_tolerance: float,
) -> dict[str, Any]:
    """Classify one matched post-clipping candidate pair."""
    if len(pair_names) != 2:
        raise ValueError("critic gradient pair names differ")
    eligible = bool(float(symmetry_error) <= float(symmetry_tolerance))
    score_delta = float(plus_true_risk) - float(minus_true_risk)
    predicted_delta = float(predicted_plus_score) - float(predicted_minus_score)
    selected = None
    selected_risk = None
    opposite_risk = None
    if eligible and abs(predicted_delta) > float(score_tolerance):
        if predicted_delta < 0.0:
            selected = str(pair_names[1])
            selected_risk = float(plus_true_risk)
            opposite_risk = float(minus_true_risk)
        else:
            selected = str(pair_names[0])
            selected_risk = float(minus_true_risk)
            opposite_risk = float(plus_true_risk)
    actual_pair_tied = abs(score_delta) <= float(risk_tolerance)
    direction_correct = None
    descends = None
    safe_conversion = None
    safety_regression = None
    true_risk_change = None
    if selected_risk is not None:
        if not actual_pair_tied:
            direction_correct = bool(
                selected_risk < float(opposite_risk) - float(risk_tolerance)
            )
        descends = bool(
            selected_risk < float(nominal_true_risk) - float(risk_tolerance)
        )
        safe_conversion = bool(
            float(nominal_true_risk) > 0.0 and selected_risk <= 0.0
        )
        safety_regression = bool(
            float(nominal_true_risk) <= 0.0 and selected_risk > 0.0
        )
        true_risk_change = float(selected_risk - float(nominal_true_risk))
    return {
        "state_id": str(state_id),
        "pair_names": [str(name) for name in pair_names],
        "post_clipping_symmetry_error": float(symmetry_error),
        "eligible_symmetric_pair": eligible,
        "predicted_minus_directional_score": float(predicted_minus_score),
        "predicted_plus_directional_score": float(predicted_plus_score),
        "predicted_selected_candidate": selected,
        "nominal_true_risk": float(nominal_true_risk),
        "minus_true_risk": float(minus_true_risk),
        "plus_true_risk": float(plus_true_risk),
        "actual_pair_tied": actual_pair_tied,
        "direction_correct": direction_correct,
        "true_risk_descends_from_nominal": descends,
        "unsafe_nominal_converted_safe": safe_conversion,
        "safe_nominal_regressed_unsafe": safety_regression,
        "selected_true_risk_change": true_risk_change,
    }


def aggregate_pair_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in records if item["eligible_symmetric_pair"]]
    directed = [
        item for item in eligible
        if item["predicted_selected_candidate"] is not None
    ]
    comparable = [
        item for item in directed if item["direction_correct"] is not None
    ]
    changes = [
        float(item["selected_true_risk_change"]) for item in directed
        if item["selected_true_risk_change"] is not None
    ]
    unsafe = [item for item in directed if float(item["nominal_true_risk"]) > 0.0]
    safe = [item for item in directed if float(item["nominal_true_risk"]) <= 0.0]

    def ratio(numerator: int, denominator: int) -> Optional[float]:
        return None if denominator == 0 else float(numerator / denominator)

    return {
        "registered_pair_count": len(records),
        "eligible_symmetric_pair_count": len(eligible),
        "asymmetric_pair_count": len(records) - len(eligible),
        "directed_pair_count": len(directed),
        "actual_nontied_comparable_pair_count": len(comparable),
        "contributing_root_count": len({item["state_id"] for item in directed}),
        "direction_correct_count": sum(
            bool(item["direction_correct"]) for item in comparable
        ),
        "direction_accuracy": ratio(
            sum(bool(item["direction_correct"]) for item in comparable),
            len(comparable),
        ),
        "descent_count": sum(
            bool(item["true_risk_descends_from_nominal"]) for item in directed
        ),
        "descent_rate": ratio(
            sum(bool(item["true_risk_descends_from_nominal"]) for item in directed),
            len(directed),
        ),
        "unsafe_nominal_pair_count": len(unsafe),
        "safe_conversion_count": sum(
            bool(item["unsafe_nominal_converted_safe"]) for item in unsafe
        ),
        "safe_conversion_rate": ratio(
            sum(bool(item["unsafe_nominal_converted_safe"]) for item in unsafe),
            len(unsafe),
        ),
        "safe_nominal_pair_count": len(safe),
        "safe_to_unsafe_regression_count": sum(
            bool(item["safe_nominal_regressed_unsafe"]) for item in safe
        ),
        "median_selected_true_risk_change": (
            None if not changes else float(statistics.median(changes))
        ),
    }


def unsafe_recall(records: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    actual_unsafe = [item for item in records if float(item["actual"]) > 0.0]
    detected = sum(float(item["predicted"]) > 0.0 for item in actual_unsafe)
    return {
        "candidate_count": len(records),
        "actual_unsafe_candidate_count": len(actual_unsafe),
        "detected_unsafe_candidate_count": int(detected),
        "false_safe_candidate_count": int(len(actual_unsafe) - detected),
        "unsafe_recall": (
            None if not actual_unsafe else float(detected / len(actual_unsafe))
        ),
    }


def feasibility_verdict(
    *, trigger: Mapping[str, Any], pairs: Mapping[str, Any],
    numerical_gradient_pass: bool, gate: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "unsafe_recall": bool(
            trigger.get("unsafe_recall") is not None
            and float(trigger["unsafe_recall"])
            >= float(gate["minimum_validation_unsafe_recall"])
        ),
        "direction_accuracy": bool(
            pairs.get("direction_accuracy") is not None
            and float(pairs["direction_accuracy"])
            >= float(gate["minimum_validation_direction_accuracy"])
        ),
        "descent_rate": bool(
            pairs.get("descent_rate") is not None
            and float(pairs["descent_rate"])
            >= float(gate["minimum_validation_descent_rate"])
        ),
        "pair_support": bool(
            int(pairs["eligible_symmetric_pair_count"])
            >= int(gate["minimum_validation_eligible_pair_count"])
            and int(pairs["contributing_root_count"])
            >= int(gate["minimum_validation_contributing_root_count"])
        ),
        "safe_regression": bool(
            int(pairs["safe_to_unsafe_regression_count"])
            <= int(gate["maximum_validation_safe_to_unsafe_regressions"])
        ),
        "median_true_risk_change": bool(
            pairs.get("median_selected_true_risk_change") is not None
            and float(pairs["median_selected_true_risk_change"]) < 0.0
        ),
        "autograd_finite_difference": bool(numerical_gradient_pass),
    }
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "action_space_exact_gradient_probe_authorized": all(checks.values()),
        "flow_guidance_authorized": False,
        "interpretation": (
            "support_scale_directional_gate_pass"
            if all(checks.values())
            else "current_compact_critic_not_gradient_ready"
        ),
    }


def projection_geometry(
    exact_case: Mapping[str, Any], row_index: int,
) -> tuple[float, list[float], float]:
    """Return stored slack, fixed outward normal, and ellipsoid support radius."""
    import numpy as np
    from main.multilink_ellipsoid.geometry import Ellipsoid
    from main.multilink_ellipsoid.obstacle_proxy_audit import (
        CompiledObstacleBox, minimum_ellipsoid_quadratic_over_box,
    )

    row = exact_case["initial_exact_robot_rows"][int(row_index)]
    robot = Ellipsoid(
        center=row["center_m"], rotation=row["rotation"],
        semiaxes_m=row["semiaxes_m"], body_name=str(row["body_name"]),
        geom_name=str(row.get("geom_name", "")),
        bound_source=str(row.get("bound_source", "artifact")),
    )
    boxes = [
        CompiledObstacleBox(
            geom_id=int(item["geom_id"]), geom_name=str(item["geom_name"]),
            body_id=int(item["body_id"]), body_name=str(item["body_name"]),
            center=np.asarray(item["center_m"], dtype=np.float64),
            rotation=np.asarray(item["rotation"], dtype=np.float64),
            half_extents_m=np.asarray(item["half_extents_m"], dtype=np.float64),
        )
        for item in exact_case["initial_compiled_obstacle_boxes"]
    ]
    values = [minimum_ellipsoid_quadratic_over_box(robot, box) for box in boxes]
    box = boxes[int(np.argmin(np.asarray(values, dtype=np.float64)))]
    computed = math.sqrt(float(min(values))) - 1.0
    stored = float(exact_case["exact_group_target"][
        "initial_row_normalized_radial_slack"
    ][int(row_index)])
    if not math.isclose(computed, stored, abs_tol=1.0e-8):
        raise ValueError("critic gradient initial slack differs")
    normal = np.asarray(robot.center - box.center, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise ValueError("critic gradient outward normal is degenerate")
    normal /= norm
    local = np.asarray(robot.rotation, dtype=np.float64).T @ normal
    support = float(np.linalg.norm(
        np.asarray(robot.semiaxes_m, dtype=np.float64) * local
    ))
    if support <= 1.0e-12:
        raise ValueError("critic gradient support radius is degenerate")
    return stored, normal.tolist(), support


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"allocation", "result_payload_sha256", "source"}
    }
