"""Contracts for the frozen-critic immediately-earlier-query probe."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_tight_prefix_early_gradient_probe.v1"
PREP_SCHEMA = "vlsa_tight_prefix_early_gradient_prep.v1"
CASE_SCHEMA = "vlsa_tight_prefix_early_gradient_case.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_early_gradient_validation.v1"


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
    if set(value) != {
        "schema_version", "protocol_id", "claim_scope", "source", "probe",
        "cases", "gate", "forbidden",
    }:
        raise ValueError("early gradient-probe config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-tight-prefix-early-gradient-probe-v1"
    ):
        raise ValueError("early gradient-probe protocol differs")
    if value["probe"] != {
        "decision_split": "validation",
        "query_offset_actions_from_ADR_0207": -5,
        "primary_rows": list(range(8)),
        "diagnostic_rows": [8, 9],
        "smoothmax_beta": 20.0,
        "optimized_action_dimensions": "five_effective_XYZ_actions_only",
        "requested_translation_l2_action": 0.25,
        "minimum_translation_l2_action": 0.01,
        "symmetric_bound_headroom": 0.8,
        "bound_active_tolerance": 1e-12,
        "random_direction_count": 4,
        "random_seed": 20260819,
        "trigger": {
            "rule": "hard_predicted_primary_Q_gt_negative_margin",
            "margin": 0.1421400248048467,
            "margin_source": "ADR_0201_validation_primary_near_boundary_RMSE",
        },
        "eligibility": [
            "initial_primary_tight_geometry_safe_and_contact_free",
            "nominal_five_action_exact_safe_and_CAR_pass",
            "frozen_critic_trigger_true",
        ],
        "model_device": "allocated_H100_float32_matching_frozen_inference",
        "execution": "direct_effective_action_through_unchanged_OSC_five_actions_only_no_QP",
    }:
        raise ValueError("early gradient-probe definition differs")
    cases = value["cases"]
    if cases != [
        {
            "case_id": "vlsa-t1-goal-ii-t0-e39", "split": "validation",
            "ADR_0207_state_step": 185, "early_state_step": 180,
            "query_index": 36,
            "active_obstacle_name": "milk_obstacle_1",
            "archived_result_relative_path": (
                "tasks/task-12/results/aegis/"
                "vlsa-t1-goal-ii-t0-e39/result.json"
            ),
            "archived_result_file_sha256": (
                "8a31d708cea13bc8cd476513477fc9c5cffe3a69c8c7bc0ec8c44c2160210cee"
            ),
            "archived_result_payload_sha256": (
                "56384611b130711d1bde861345c9c3f9ea17c7d7ce40ed4a0983f8721ee2c763"
            ),
        },
        {
            "case_id": "vlsa-t1-goal-ii-t2-e44", "split": "validation",
            "ADR_0207_state_step": 25, "early_state_step": 20,
            "query_index": 4,
            "active_obstacle_name": "milk_obstacle_1",
            "archived_result_relative_path": (
                "tasks/task-14/results/aegis/"
                "vlsa-t1-goal-ii-t2-e44/result.json"
            ),
            "archived_result_file_sha256": (
                "6eea09121169a9b3005476b19f278a7bdcea03c721fd4ffe9635dc8c280cc054"
            ),
            "archived_result_payload_sha256": (
                "9d0c51305371a6fdd3e0baaa626beb7de14d01ed35d09f446dd3e924dbe3c9f1"
            ),
        },
    ]:
        raise ValueError("early gradient-probe cases differ")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def symmetric_multi_probe_actions(
    anchor_actions: Sequence[Sequence[float]],
    gradient: Sequence[Sequence[float]],
    random_directions: Sequence[Sequence[Sequence[float]]],
    *, requested_radius: float, minimum_radius: float,
    bound_headroom: float, bound: float = 1.0,
) -> dict[str, Any]:
    """Construct equal-norm, unclipped gradient and random probes."""
    import numpy as np

    anchor = np.asarray(anchor_actions, dtype=np.float64)
    grad = np.asarray(gradient, dtype=np.float64)
    random = np.asarray(random_directions, dtype=np.float64)
    if (
        anchor.shape != (5, 7) or grad.shape != (5, 3)
        or random.ndim != 3 or random.shape[1:] != (5, 3)
        or len(random) == 0 or not np.all(np.isfinite(anchor))
        or not np.all(np.isfinite(grad)) or not np.all(np.isfinite(random))
    ):
        raise ValueError("early gradient-probe action shape differs")
    translation = anchor[:, :3]
    free = np.abs(translation) < float(bound) - 1e-12
    directions = [np.where(free, grad, 0.0)]
    directions.extend(np.where(free, item, 0.0) for item in random)
    normalized = []
    raw_norms = []
    for direction in directions:
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("early gradient-probe direction is degenerate")
        raw_norms.append(norm)
        normalized.append(direction / norm)

    def maximum_symmetric_radius(direction: Any) -> float:
        active = np.abs(direction) > 1e-15
        values = (
            (float(bound) - np.abs(translation[active]))
            / np.abs(direction[active])
        )
        return float(np.min(values))

    maximum = min(maximum_symmetric_radius(item) for item in normalized)
    radius = min(float(requested_radius), float(bound_headroom) * maximum)
    if not math.isfinite(radius) or radius < float(minimum_radius):
        raise ValueError("early gradient-probe symmetric radius is unsupported")

    def apply(direction: Any) -> list[list[float]]:
        output = anchor.copy()
        output[:, :3] += radius * direction
        if float(np.max(np.abs(output[:, :3]))) > float(bound) + 1e-12:
            raise ValueError("early gradient-probe unexpectedly clips")
        return output.tolist()

    gradient_unit = normalized[0]
    down = apply(-gradient_unit)
    up = apply(gradient_unit)
    random_actions = [apply(item) for item in normalized[1:]]
    deltas = [
        np.asarray(item, dtype=np.float64)[:, :3] - translation
        for item in [down, up, *random_actions]
    ]
    return {
        "radius_l2_action": float(radius),
        "free_translation_coordinate_count": int(np.sum(free)),
        "projected_gradient_l2_before_normalization": raw_norms[0],
        "post_clipping_symmetry_error": float(np.max(np.abs(
            deltas[0] + deltas[1]
        ))),
        "equal_norm_maximum_error": float(max(
            abs(float(np.linalg.norm(delta)) - radius) for delta in deltas
        )),
        "gradient_unit_direction": gradient_unit.tolist(),
        "random_unit_directions": [item.tolist() for item in normalized[1:]],
        "gradient_down": down,
        "gradient_up": up,
        "matched_random": random_actions,
    }


def summarize_cases(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in records if item["eligible"]]
    comparisons = [
        bool(value)
        for item in eligible for value in item["gradient_down_beats_random"]
    ]
    changes = [float(item["down_risk_change"]) for item in eligible]
    return {
        "case_count": len(records),
        "eligible_case_count": len(eligible),
        "nominal_exact_safe_case_count": sum(
            bool(item["nominal_exact_safe"]) for item in records
        ),
        "triggered_case_count": sum(bool(item["triggered"]) for item in records),
        "gradient_down_exact_safe_count": sum(
            bool(item["gradient_down_exact_safe"]) for item in eligible
        ),
        "gradient_down_descends_count": sum(
            bool(item["gradient_down_descends"]) for item in eligible
        ),
        "gradient_down_beats_up_count": sum(
            bool(item["gradient_down_beats_up"]) for item in eligible
        ),
        "random_comparison_count": len(comparisons),
        "gradient_down_random_win_count": sum(comparisons),
        "gradient_down_random_win_rate": (
            None if not comparisons else float(sum(comparisons) / len(comparisons))
        ),
        "median_down_risk_change": (
            None if not changes else float(statistics.median(changes))
        ),
    }


def hypothesis_verdict(
    summary: Mapping[str, Any], gate: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = int(summary["eligible_case_count"])
    checks = {
        "eligible_support": eligible >= int(gate["minimum_eligible_case_count"]),
        "guided_safe": (
            eligible > 0
            and int(summary["gradient_down_exact_safe_count"]) == eligible
        ),
        "exact_descent": (
            eligible > 0
            and int(summary["gradient_down_descends_count"]) == eligible
        ),
        "beats_positive_gradient": (
            eligible > 0
            and int(summary["gradient_down_beats_up_count"]) == eligible
        ),
        "beats_random": (
            summary["gradient_down_random_win_rate"] is not None
            and float(summary["gradient_down_random_win_rate"])
            >= float(gate["minimum_random_win_rate"])
        ),
        "negative_median_change": (
            summary["median_down_risk_change"] is not None
            and float(summary["median_down_risk_change"]) < 0.0
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "checks": checks,
        "all_checks_pass": passed,
        "early_bounded_gradient_hypothesis_confirmed": passed,
        "retraining_authorized": False,
        "full_episode_or_deployment_authorized": False,
    }


def normalized_exact_case(value: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(canonical(value).decode("utf-8"))
    for key in (
        "source_result", "source_result_file_sha256",
        "source_result_payload_sha256",
    ):
        output.pop(key, None)
    return output


def scientific_view(record: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        key: value for key, value in record.items()
        if key not in {
            "source", "allocation", "prep_binding", "result_payload_sha256",
        }
    }
    if output.get("exact_rollout") is not None:
        output["exact_rollout"] = normalized_exact_case(output["exact_rollout"])
    return output
