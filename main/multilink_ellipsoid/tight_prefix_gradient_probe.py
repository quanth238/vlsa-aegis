"""Contracts and pure metrics for the exact compact-critic gradient probe."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_tight_prefix_gradient_probe.v1"
CASE_SCHEMA = "vlsa_tight_prefix_gradient_probe_case.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_gradient_probe_validation.v1"


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
        "schema_version", "protocol_id", "claim_scope", "source",
        "probe", "cases", "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("tight gradient-probe config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-tight-prefix-gradient-probe-v1"
    ):
        raise ValueError("tight gradient-probe protocol differs")
    probe = value["probe"]
    if (
        probe.get("decision_split") != "validation"
        or probe.get("primary_rows") != list(range(8))
        or probe.get("diagnostic_rows") != [8, 9]
        or probe.get("probe_names")
        != ["gradient_down", "gradient_up", "matched_random"]
        or float(probe.get("requested_translation_l2_action", -1.0)) != 0.25
        or float(probe.get("minimum_translation_l2_action", -1.0)) != 0.01
        or float(probe.get("symmetric_bound_headroom", -1.0)) != 0.8
        or int(probe.get("random_seed", -1)) != 20260818
    ):
        raise ValueError("tight gradient-probe definition differs")
    cases = value["cases"]
    expected_cases = {
        "vlsa-t1-goal-ii-t0-e39": {
            "grid_m1_m1_z0_front_loaded_r2.0",
            "grid_z0_z0_p1_front_loaded_r2.0",
        },
        "vlsa-t1-goal-ii-t2-e44": {
            "grid_m1_m1_m1_front_loaded_r2.0",
            "grid_m1_z0_m1_front_loaded_r2.0",
            "grid_m1_p1_z0_front_loaded_r2.0",
            "grid_z0_z0_m1_front_loaded_r2.0",
        },
    }
    if (
        len(cases) != 2
        or {str(item["case_id"]) for item in cases} != set(expected_cases)
        or any(
            item.get("split") != "validation"
            or set(item.get("unsafe_anchor_candidates", []))
            != expected_cases[str(item["case_id"])]
            for item in cases
        )
    ):
        raise ValueError("tight gradient-probe anchor population differs")
    gate = value["gate"]
    if gate != {
        "required_anchor_count": 6,
        "required_probe_rollout_count": 18,
        "minimum_direction_accuracy": 2.0 / 3.0,
        "minimum_descent_rate": 2.0 / 3.0,
        "minimum_random_advantage_rate": 0.5,
        "minimum_safe_conversion_count": 1,
        "require_negative_median_down_risk_change": True,
        "maximum_anchor_prediction_absolute_error": 0.00005,
        "maximum_post_clipping_symmetry_error": 1e-12,
        "maximum_independent_replay_absolute_error": 1e-9,
    }:
        raise ValueError("tight gradient-probe gate differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def symmetric_probe_actions(
    anchor_actions: Sequence[Sequence[float]],
    gradient: Sequence[Sequence[float]],
    random_direction: Sequence[Sequence[float]],
    *, requested_radius: float, minimum_radius: float,
    bound_headroom: float, bound: float = 1.0,
) -> dict[str, Any]:
    """Build unclipped, equal-norm probes in the symmetric box tangent space."""
    import numpy as np

    anchor = np.asarray(anchor_actions, dtype=np.float64)
    grad = np.asarray(gradient, dtype=np.float64)
    random = np.asarray(random_direction, dtype=np.float64)
    if (
        anchor.shape != (5, 7) or grad.shape != (5, 3)
        or random.shape != (5, 3)
        or not np.all(np.isfinite(anchor))
        or not np.all(np.isfinite(grad))
        or not np.all(np.isfinite(random))
    ):
        raise ValueError("tight gradient-probe action shape differs")
    translation = anchor[:, :3]
    free = np.abs(translation) < float(bound) - 1e-12
    grad = np.where(free, grad, 0.0)
    random = np.where(free, random, 0.0)
    grad_norm = float(np.linalg.norm(grad))
    random_norm = float(np.linalg.norm(random))
    if grad_norm <= 1e-12 or random_norm <= 1e-12:
        raise ValueError("tight gradient-probe feasible direction is degenerate")
    grad /= grad_norm
    random /= random_norm

    def maximum_symmetric_radius(direction: Any) -> float:
        active = np.abs(direction) > 1e-15
        values = (
            (float(bound) - np.abs(translation[active]))
            / np.abs(direction[active])
        )
        return float(np.min(values))

    maximum = min(
        maximum_symmetric_radius(grad),
        maximum_symmetric_radius(random),
    )
    radius = min(
        float(requested_radius), float(bound_headroom) * float(maximum),
    )
    if not math.isfinite(radius) or radius < float(minimum_radius):
        raise ValueError("tight gradient-probe symmetric radius is unsupported")

    def apply(delta: Any) -> Any:
        output = anchor.copy()
        output[:, :3] += delta
        if np.max(np.abs(output[:, :3])) > float(bound) + 1e-12:
            raise ValueError("tight gradient-probe unexpectedly clips")
        return output

    down = apply(-radius * grad)
    up = apply(radius * grad)
    matched = apply(radius * random)
    down_delta = down[:, :3] - translation
    up_delta = up[:, :3] - translation
    random_delta = matched[:, :3] - translation
    symmetry_error = float(np.max(np.abs(down_delta + up_delta)))
    norm_error = max(
        abs(float(np.linalg.norm(down_delta)) - radius),
        abs(float(np.linalg.norm(up_delta)) - radius),
        abs(float(np.linalg.norm(random_delta)) - radius),
    )
    return {
        "radius_l2_action": float(radius),
        "free_translation_coordinate_count": int(np.sum(free)),
        "projected_gradient_l2_before_normalization": grad_norm,
        "post_clipping_symmetry_error": symmetry_error,
        "equal_norm_maximum_error": float(norm_error),
        "gradient_unit_direction": grad.tolist(),
        "random_unit_direction": random.tolist(),
        "gradient_down": down.tolist(),
        "gradient_up": up.tolist(),
        "matched_random": matched.tolist(),
    }


def aggregate_anchor_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("tight gradient-probe records are empty")
    changes = [float(item["down_risk_change"]) for item in records]
    return {
        "anchor_count": len(records),
        "probe_rollout_count": 3 * len(records),
        "contributing_root_count": len({str(item["case_id"]) for item in records}),
        "direction_correct_count": sum(bool(item["direction_correct"]) for item in records),
        "direction_accuracy": float(sum(
            bool(item["direction_correct"]) for item in records
        ) / len(records)),
        "descent_count": sum(bool(item["gradient_down_descends"]) for item in records),
        "descent_rate": float(sum(
            bool(item["gradient_down_descends"]) for item in records
        ) / len(records)),
        "random_advantage_count": sum(bool(
            item["gradient_down_beats_random"]
        ) for item in records),
        "random_advantage_rate": float(sum(
            bool(item["gradient_down_beats_random"]) for item in records
        ) / len(records)),
        "safe_conversion_count": sum(bool(
            item["gradient_down_converts_safe"]
        ) for item in records),
        "median_down_risk_change": float(statistics.median(changes)),
        "maximum_anchor_prediction_absolute_error": max(
            float(item["anchor_prediction_absolute_error"])
            for item in records
        ),
        "maximum_post_clipping_symmetry_error": max(
            float(item["post_clipping_symmetry_error"]) for item in records
        ),
    }


def feasibility_verdict(
    metrics: Mapping[str, Any], gate: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "anchor_count": int(metrics["anchor_count"])
        == int(gate["required_anchor_count"]),
        "probe_rollout_count": int(metrics["probe_rollout_count"])
        == int(gate["required_probe_rollout_count"]),
        "direction_accuracy": float(metrics["direction_accuracy"])
        >= float(gate["minimum_direction_accuracy"]),
        "descent_rate": float(metrics["descent_rate"])
        >= float(gate["minimum_descent_rate"]),
        "random_advantage": float(metrics["random_advantage_rate"])
        >= float(gate["minimum_random_advantage_rate"]),
        "safe_conversion": int(metrics["safe_conversion_count"])
        >= int(gate["minimum_safe_conversion_count"]),
        "median_down_risk_change": float(metrics["median_down_risk_change"])
        < 0.0,
        "anchor_prediction_match": float(
            metrics["maximum_anchor_prediction_absolute_error"]
        ) <= float(gate["maximum_anchor_prediction_absolute_error"]),
        "symmetric_actions": float(
            metrics["maximum_post_clipping_symmetry_error"]
        ) <= float(gate["maximum_post_clipping_symmetry_error"]),
    }
    passed = bool(all(checks.values()))
    return {
        "checks": checks,
        "all_checks_pass": passed,
        "negative_gradient_exact_risk_feasible": passed,
        "flow_guidance_or_QP_authorized": False,
        "interpretation": (
            "compact_critic_negative_gradient_lowers_exact_risk_on_opened_validation_anchors"
            if passed else
            "compact_critic_negative_gradient_not_yet_shown_to_lower_exact_risk"
        ),
    }


def scientific_view(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in record.items()
        if key not in {"source", "allocation", "result_payload_sha256"}
    }
