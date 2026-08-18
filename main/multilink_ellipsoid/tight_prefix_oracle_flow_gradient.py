"""Contracts for the late-flow physical-oracle gradient canary.

The canary is deliberately diagnostic.  It freezes the trained critic and
uses exact cloned-OSC rollouts to ask whether the *correct* local future-risk
direction can steer a late pi0.5 flow branch.  Nothing in this module trains a
model or authorizes online correction.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_tight_prefix_oracle_flow_gradient.v1"
CASE_SCHEMA = "vlsa_tight_prefix_oracle_flow_gradient_case.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_oracle_flow_gradient_validation.v1"


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
        "schema_version", "protocol_id", "claim_scope", "source", "cases",
        "flow", "localization", "finite_difference", "comparison", "gate",
        "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("oracle-flow-gradient config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-tight-prefix-oracle-flow-gradient-v1"
    ):
        raise ValueError("oracle-flow-gradient protocol differs")
    cases = value["cases"]
    expected_cases = [
        ("vlsa-t1-goal-ii-t0-e39", 185, "boundary"),
        ("vlsa-t1-goal-ii-t2-e44", 25, "boundary"),
        ("vlsa-t1-goal-ii-t0-e05", 180, "far_safe_control"),
    ]
    if [
        (str(row.get("case_id")), int(row.get("state_step", -1)), row.get("role"))
        for row in cases
    ] != expected_cases or any(row.get("split") != "validation" for row in cases):
        raise ValueError("oracle-flow-gradient case population differs")
    if value["flow"] != {
        "sampler_steps": 10,
        "branch_after_euler_step": 8,
        "remaining_euler_steps": 2,
        "optimized_action_slots": 5,
        "optimized_coordinates": [0, 1, 2],
        "branch_coordinates": "physical_output_action_displacement_scale_only",
        "execution": "terminalize_then_clip_XYZ_then_unchanged_OSC_five_actions",
    }:
        raise ValueError("oracle-flow-gradient flow schedule differs")
    if value["localization"] != {
        "source_bank_candidate_count": 13,
        "maximum_pair_distance_l2": 2.0,
        "interpolation_segment_count": 8,
        "maximum_local_witness_distance_l2": 0.25,
        "minimum_local_witness_distance_l2": 0.01,
        "unsafe_definition": "hard_primary_future_risk_gt_zero",
        "safe_definition": "hard_primary_future_risk_le_zero_and_no_primary_contact_and_CAR_pass",
        "pair_selection": "minimum_residual_distance_then_source_order",
        "crossing_selection": "first_adjacent_unsafe_safe_segment",
    }:
        raise ValueError("oracle-flow-gradient localization differs")
    if value["finite_difference"] != {
        "dimension": 15,
        "maximum_epsilon": 0.05,
        "witness_fraction": 0.25,
        "minimum_epsilon": 0.0025,
        "scheme": "central",
        "exact_objective": "hard_max_over_primary_row_future_risks",
        "learned_objective": "beta20_logmeanexp_over_frozen_17D_primary_row_predictions",
        "primary_rows": list(range(8)),
        "diagnostic_rows": [8, 9],
    }:
        raise ValueError("oracle-flow-gradient finite difference differs")
    comparison = value["comparison"]
    if comparison != {
        "direction_names": [
            "learned_down", "oracle_down", "oracle_up",
            "random_0", "random_1", "random_2", "random_3",
        ],
        "random_seed": 20260819,
        "equal_norm": "localized_safe_witness_distance_in_branch_coordinates",
        "same_state_seed_flow_OSC_horizon": True,
    }:
        raise ValueError("oracle-flow-gradient comparison differs")
    gate = value["gate"]
    if gate != {
        "required_case_count": 3,
        "minimum_eligible_boundary_root_count": 1,
        "require_oracle_descent_on_every_eligible_root": True,
        "require_oracle_beats_opposite_on_every_eligible_root": True,
        "minimum_oracle_random_advantage_rate": 0.75,
        "minimum_oracle_safe_conversion_count": 1,
        "maximum_independent_replay_absolute_error": 1e-9,
        "maximum_physical_false_safe_count": 0,
        "paper_CAR_threshold_m": 0.001,
    }:
        raise ValueError("oracle-flow-gradient gate differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def terminal_branch_envelope(
    names: Sequence[str], residuals: Sequence[Sequence[Sequence[float]]],
    *, branch_after_euler_step: int = 8,
) -> dict[str, Any]:
    """Build the opt-in terminal-branch request for an arbitrary fixed bank."""
    import numpy as np

    labels = [str(name) for name in names]
    bank = np.asarray(residuals, dtype=np.float64)
    if (
        len(labels) < 2 or len(set(labels)) != len(labels)
        or bank.shape != (len(labels), 10, 3)
        or not np.all(np.isfinite(bank))
        or int(branch_after_euler_step) != 8
    ):
        raise ValueError("oracle-flow-gradient terminal branch bank differs")
    return {
        "schema_version": "crfs_terminal_branching.v1",
        "action_horizon": 10,
        "action_dimensions": [0, 1, 2],
        "candidate_names": labels,
        "candidate_output_residuals": bank.tolist(),
        "branch_after_euler_step": int(branch_after_euler_step),
    }


def choose_closest_unsafe_safe_pair(
    records: Sequence[Mapping[str, Any]], residuals: Sequence[Any],
    *, maximum_distance: float,
) -> dict[str, Any] | None:
    """Choose the easiest frozen-bank recovery pair without cherry-picking risk."""
    import numpy as np

    if len(records) != len(residuals):
        raise ValueError("oracle-flow-gradient bank pairing differs")
    bank = np.asarray(residuals, dtype=np.float64)
    if bank.shape != (len(records), 10, 3):
        raise ValueError("oracle-flow-gradient residual shape differs")
    options = []
    for unsafe_index, unsafe in enumerate(records):
        if float(unsafe["hard_primary_future_risk"]) <= 0.0:
            continue
        for safe_index, safe in enumerate(records):
            if not bool(safe["physical_primary_safe"]):
                continue
            distance = float(np.linalg.norm(bank[safe_index] - bank[unsafe_index]))
            if distance <= float(maximum_distance) + 1e-12:
                options.append((distance, unsafe_index, safe_index))
    if not options:
        return None
    distance, unsafe_index, safe_index = min(options)
    return {
        "unsafe_index": int(unsafe_index),
        "unsafe_name": str(records[unsafe_index]["name"]),
        "safe_index": int(safe_index),
        "safe_name": str(records[safe_index]["name"]),
        "residual_distance_l2": float(distance),
    }


def interpolation_residuals(
    unsafe: Sequence[Any], safe: Sequence[Any], *, segment_count: int,
) -> tuple[list[str], list[Any]]:
    import numpy as np

    left = np.asarray(unsafe, dtype=np.float64)
    right = np.asarray(safe, dtype=np.float64)
    if left.shape != (10, 3) or right.shape != left.shape or segment_count <= 0:
        raise ValueError("oracle-flow-gradient interpolation differs")
    names = [f"interpolation_{index:02d}" for index in range(segment_count + 1)]
    values = [
        ((1.0 - index / segment_count) * left + (index / segment_count) * right)
        for index in range(segment_count + 1)
    ]
    return names, [value.tolist() for value in values]


def choose_local_crossing(
    records: Sequence[Mapping[str, Any]], residuals: Sequence[Any],
    *, minimum_distance: float, maximum_distance: float,
) -> dict[str, Any] | None:
    """Choose the first adjacent unsafe-to-safe interpolation segment."""
    import numpy as np

    bank = np.asarray(residuals, dtype=np.float64)
    if bank.shape != (len(records), 10, 3):
        raise ValueError("oracle-flow-gradient crossing bank differs")
    for left in range(len(records) - 1):
        for unsafe_index, safe_index in ((left, left + 1), (left + 1, left)):
            if (
                float(records[unsafe_index]["hard_primary_future_risk"]) > 0.0
                and bool(records[safe_index]["physical_primary_safe"])
            ):
                distance = float(np.linalg.norm(
                    bank[safe_index] - bank[unsafe_index]
                ))
                if (
                    float(minimum_distance) - 1e-12 <= distance
                    <= float(maximum_distance) + 1e-12
                ):
                    return {
                        "unsafe_index": int(unsafe_index),
                        "unsafe_name": str(records[unsafe_index]["name"]),
                        "safe_index": int(safe_index),
                        "safe_name": str(records[safe_index]["name"]),
                        "witness_distance_l2": float(distance),
                    }
    return None


def finite_difference_residuals(
    anchor: Sequence[Any], *, epsilon: float,
) -> tuple[list[str], list[Any]]:
    import numpy as np

    base = np.asarray(anchor, dtype=np.float64)
    if base.shape != (10, 3) or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("oracle-flow-gradient finite-difference anchor differs")
    names = ["fd_anchor"]
    residuals = [base.copy()]
    for coordinate in range(15):
        slot, axis = divmod(coordinate, 3)
        negative = base.copy()
        positive = base.copy()
        negative[slot, axis] -= float(epsilon)
        positive[slot, axis] += float(epsilon)
        names.extend([f"fd_{coordinate:02d}_neg", f"fd_{coordinate:02d}_pos"])
        residuals.extend([negative, positive])
    return names, [value.tolist() for value in residuals]


def central_gradient(
    records: Sequence[Mapping[str, Any]], *, key: str, epsilon: float,
) -> list[float]:
    if len(records) != 31 or epsilon <= 0.0:
        raise ValueError("oracle-flow-gradient central records differ")
    values = []
    for coordinate in range(15):
        negative = float(records[1 + 2 * coordinate][key])
        positive = float(records[2 + 2 * coordinate][key])
        values.append((positive - negative) / (2.0 * float(epsilon)))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("oracle-flow-gradient central gradient is non-finite")
    return values


def comparison_residuals(
    anchor: Sequence[Any], learned_gradient: Sequence[float],
    oracle_gradient: Sequence[float], *, correction_norm: float,
    random_seed: int,
) -> tuple[list[str], list[Any], dict[str, Any]]:
    import numpy as np

    base = np.asarray(anchor, dtype=np.float64)
    learned = np.asarray(learned_gradient, dtype=np.float64).reshape(5, 3)
    oracle = np.asarray(oracle_gradient, dtype=np.float64).reshape(5, 3)
    learned_norm = float(np.linalg.norm(learned))
    oracle_norm = float(np.linalg.norm(oracle))
    if (
        base.shape != (10, 3) or correction_norm <= 0.0
        or learned_norm <= 1e-12 or oracle_norm <= 1e-12
    ):
        raise ValueError("oracle-flow-gradient comparison direction is degenerate")
    learned /= learned_norm
    oracle /= oracle_norm
    rng = np.random.default_rng(int(random_seed))
    random = []
    for _ in range(4):
        direction = rng.normal(size=(5, 3))
        direction /= float(np.linalg.norm(direction))
        random.append(direction)

    def shifted(direction: Any) -> list[list[float]]:
        value = base.copy()
        value[:5] += float(correction_norm) * direction
        return value.tolist()

    names = [
        "comparison_anchor", "learned_down", "oracle_down", "oracle_up",
        "random_0", "random_1", "random_2", "random_3",
    ]
    residuals = [
        base.tolist(), shifted(-learned), shifted(-oracle), shifted(oracle),
        *[shifted(direction) for direction in random],
    ]
    audit = {
        "correction_norm_l2": float(correction_norm),
        "learned_gradient_l2": learned_norm,
        "oracle_gradient_l2": oracle_norm,
        "learned_unit_direction": learned.tolist(),
        "oracle_unit_direction": oracle.tolist(),
        "random_unit_directions": [value.tolist() for value in random],
        "maximum_requested_norm_error": max(
            abs(float(np.linalg.norm(np.asarray(value)[:5] - base[:5])) - correction_norm)
            for value in residuals[1:]
        ),
    }
    return names, residuals, audit


def case_direction_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(record["name"]): record for record in records}
    required = {
        "comparison_anchor", "learned_down", "oracle_down", "oracle_up",
        "random_0", "random_1", "random_2", "random_3",
    }
    if set(by_name) != required:
        raise ValueError("oracle-flow-gradient comparison records differ")
    anchor = float(by_name["comparison_anchor"]["hard_primary_future_risk"])
    learned = float(by_name["learned_down"]["hard_primary_future_risk"])
    oracle = float(by_name["oracle_down"]["hard_primary_future_risk"])
    opposite = float(by_name["oracle_up"]["hard_primary_future_risk"])
    random = [
        float(by_name[f"random_{index}"]["hard_primary_future_risk"])
        for index in range(4)
    ]
    return {
        "anchor_risk": anchor,
        "learned_risk": learned,
        "oracle_risk": oracle,
        "opposite_oracle_risk": opposite,
        "random_risks": random,
        "learned_descends": learned < anchor,
        "oracle_descends": oracle < anchor,
        "oracle_beats_opposite": oracle < opposite,
        "oracle_random_advantage_count": sum(oracle < value for value in random),
        "oracle_random_advantage_rate": sum(oracle < value for value in random) / 4.0,
        "learned_random_advantage_count": sum(learned < value for value in random),
        "oracle_safe_conversion": bool(
            anchor > 0.0 and by_name["oracle_down"]["physical_primary_safe"]
        ),
        "learned_safe_conversion": bool(
            anchor > 0.0 and by_name["learned_down"]["physical_primary_safe"]
        ),
        "random_safe_conversion_count": sum(
            anchor > 0.0 and by_name[f"random_{index}"]["physical_primary_safe"]
            for index in range(4)
        ),
    }


def scientific_view(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in record.items()
        if key not in {
            "source", "allocation", "result_payload_sha256", "timing",
            "replica", "server_identity",
        }
    }
