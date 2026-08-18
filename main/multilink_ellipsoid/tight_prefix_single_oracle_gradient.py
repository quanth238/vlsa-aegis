"""Contracts for one exact-action-replayed oracle/critic gradient decision.

The protocol contains one already-opened warning query.  It first requires the
ordinary terminal pi0.5 chunk to be physically unsafe, then asks whether exact
oracle descent reaches an exact-safe chunk inside the unchanged local radius.
Only a reachable case receives learned/opposite/random comparisons.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_tight_prefix_single_oracle_gradient.v1"
RESULT_SCHEMA = "vlsa_tight_prefix_single_oracle_gradient_result.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_single_oracle_gradient_validation.v1"


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
        "schema_version", "protocol_id", "claim_scope", "source", "case",
        "flow", "finite_difference", "support", "comparison", "gate",
        "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("single-oracle-gradient config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-tight-prefix-single-oracle-gradient-v1"
    ):
        raise ValueError("single-oracle-gradient protocol differs")
    if value["case"] != {
        "case_id": "vlsa-t1-spatial-i-t3-e15",
        "state_step": 65,
        "split": "opened_diagnostic_test",
        "tight_dataset_case_index": 22,
        "tight_case_file_sha256": (
            "7008b85902230a4ab24acd53867a128c8a5566580f962ab34290b1a89bf7cfab"
        ),
        "tight_case_payload_sha256": (
            "e22f5cc3de3f0f008369f1723c7c257c6d183c1a9cce3b3c58dd165d6851f00e"
        ),
    }:
        raise ValueError("single-oracle-gradient case differs")
    if value["flow"] != {
        "sampler_steps": 10,
        "branch_after_euler_step": 8,
        "remaining_euler_steps": 2,
        "optimized_action_slots": 5,
        "optimized_coordinates": [0, 1, 2],
        "branch_coordinates": "physical_output_action_displacement_scale_only",
        "execution": "terminalize_once_then_clip_XYZ_then_unchanged_OSC_five_actions",
    }:
        raise ValueError("single-oracle-gradient flow differs")
    if value["finite_difference"] != {
        "dimension": 15,
        "epsilon": 0.005,
        "scheme": "central",
        "exact_objective": "hard_max_over_primary_row_future_risks",
        "learned_objective": "beta20_logmeanexp_over_frozen_17D_primary_row_predictions",
        "primary_rows": list(range(8)),
        "diagnostic_rows": [8, 9],
    }:
        raise ValueError("single-oracle-gradient finite difference differs")
    if value["support"] != {
        "correction_radius_l2": 0.25,
        "oracle_line_radii_l2": [0.0625, 0.125, 0.1875, 0.25],
        "raw_unsafe_definition": "hard_primary_future_risk_gt_zero",
        "safe_definition": "hard_primary_future_risk_le_zero_and_no_primary_contact_and_CAR_pass",
        "select_safe_witness": "smallest_registered_oracle_radius",
    }:
        raise ValueError("single-oracle-gradient support differs")
    if value["comparison"] != {
        "direction_names": [
            "learned_down", "oracle_down", "oracle_up",
            "random_0", "random_1", "random_2", "random_3",
        ],
        "random_seed": 20260819,
        "equal_norm": "selected_exact_safe_oracle_witness_radius",
        "same_state_seed_flow_OSC_horizon": True,
    }:
        raise ValueError("single-oracle-gradient comparison differs")
    gate = value["gate"]
    if gate != {
        "require_raw_terminal_unsafe": True,
        "require_exact_safe_oracle_within_radius": True,
        "require_exact_action_replay": True,
        "maximum_exact_replay_absolute_error": 1e-9,
        "maximum_physical_false_safe_count": 0,
        "paper_CAR_threshold_m": 0.001,
    }:
        raise ValueError("single-oracle-gradient gate differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def unit_direction(gradient: Sequence[float]) -> list[list[float]]:
    import numpy as np

    value = np.asarray(gradient, dtype=np.float64).reshape(5, 3)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("single-oracle-gradient direction is degenerate")
    return (value / norm).tolist()


def oracle_line_residuals(
    oracle_gradient: Sequence[float], radii: Sequence[float],
) -> tuple[list[str], list[Any]]:
    import numpy as np

    direction = np.asarray(unit_direction(oracle_gradient), dtype=np.float64)
    names = []
    residuals = []
    for index, radius in enumerate(radii):
        if not math.isfinite(float(radius)) or float(radius) <= 0.0:
            raise ValueError("single-oracle-gradient radius differs")
        value = np.zeros((10, 3), dtype=np.float64)
        value[:5] = -float(radius) * direction
        names.append(f"oracle_down_r{index}")
        residuals.append(value.tolist())
    return names, residuals


def terminal_branch_batches(
    names: Sequence[str], residuals: Sequence[Any], *, batch_width: int = 13,
) -> list[dict[str, Any]]:
    """Pack arbitrary probes into the sampler's frozen 13-branch envelope."""
    import numpy as np

    labels = [str(name) for name in names]
    values = np.asarray(residuals, dtype=np.float64)
    if (
        batch_width != 13 or len(labels) != len(set(labels))
        or any(not name or name == "nominal" for name in labels)
        or values.shape != (len(labels), 10, 3)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("single-oracle-gradient terminal batch differs")
    output = []
    for batch_index, start in enumerate(range(0, len(labels), batch_width - 1)):
        stop = min(start + batch_width - 1, len(labels))
        selected_names = labels[start:stop]
        selected_values = values[start:stop]
        pad_count = batch_width - 1 - len(selected_names)
        request_names = ["nominal", *selected_names]
        request_values = [np.zeros((10, 3), dtype=np.float64), *selected_values]
        for pad_index in range(pad_count):
            request_names.append(f"padding_{batch_index:02d}_{pad_index:02d}")
            request_values.append(np.zeros((10, 3), dtype=np.float64))
        output.append({
            "batch_index": int(batch_index),
            "scientific_start": int(start),
            "scientific_stop": int(stop),
            "scientific_names": selected_names,
            "request_names": request_names,
            "request_residuals": np.asarray(request_values).tolist(),
        })
    return output


def comparison_residuals(
    learned_gradient: Sequence[float], oracle_gradient: Sequence[float],
    *, radius: float, random_seed: int,
) -> tuple[list[str], list[Any], dict[str, Any]]:
    import numpy as np

    learned = np.asarray(unit_direction(learned_gradient), dtype=np.float64)
    oracle = np.asarray(unit_direction(oracle_gradient), dtype=np.float64)
    rng = np.random.default_rng(int(random_seed))
    random = []
    for _ in range(4):
        value = rng.normal(size=(5, 3))
        value /= float(np.linalg.norm(value))
        random.append(value)

    def residual(direction: Any) -> list[list[float]]:
        value = np.zeros((10, 3), dtype=np.float64)
        value[:5] = float(radius) * direction
        return value.tolist()

    names = [
        "comparison_anchor", "learned_down", "oracle_up",
        "random_0", "random_1", "random_2", "random_3",
    ]
    residuals = [
        np.zeros((10, 3), dtype=np.float64).tolist(),
        residual(-learned), residual(oracle),
        *[residual(value) for value in random],
    ]
    audit = {
        "radius_l2": float(radius),
        "learned_unit_direction": learned.tolist(),
        "oracle_unit_direction": oracle.tolist(),
        "random_unit_directions": [value.tolist() for value in random],
        "maximum_requested_norm_error": max(
            abs(float(np.linalg.norm(np.asarray(value)[:5])) - float(radius))
            for value in residuals[1:]
        ),
    }
    return names, residuals, audit


def comparison_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(record["name"]): record for record in records}
    required = {
        "comparison_anchor", "learned_down", "oracle_down", "oracle_up",
        "random_0", "random_1", "random_2", "random_3",
    }
    if set(by_name) != required:
        raise ValueError("single-oracle-gradient comparison records differ")
    risks = {
        name: float(record["hard_primary_future_risk"])
        for name, record in by_name.items()
    }
    random = [risks[f"random_{index}"] for index in range(4)]
    return {
        "anchor_risk": risks["comparison_anchor"],
        "oracle_risk": risks["oracle_down"],
        "learned_risk": risks["learned_down"],
        "opposite_risk": risks["oracle_up"],
        "random_risks": random,
        "oracle_safe": bool(by_name["oracle_down"]["physical_primary_safe"]),
        "learned_safe": bool(by_name["learned_down"]["physical_primary_safe"]),
        "opposite_safe": bool(by_name["oracle_up"]["physical_primary_safe"]),
        "random_safe_count": sum(
            bool(by_name[f"random_{index}"]["physical_primary_safe"])
            for index in range(4)
        ),
        "oracle_descends": risks["oracle_down"] < risks["comparison_anchor"],
        "learned_descends": risks["learned_down"] < risks["comparison_anchor"],
        "oracle_beats_opposite": risks["oracle_down"] < risks["oracle_up"],
        "oracle_random_advantage_rate": sum(
            risks["oracle_down"] < value for value in random
        ) / 4.0,
        "learned_random_advantage_rate": sum(
            risks["learned_down"] < value for value in random
        ) / 4.0,
    }


def scientific_view(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in record.items()
        if key not in {
            "source", "allocation", "result_payload_sha256", "timing",
            "replica", "frozen_producer_binding", "server_identity",
        }
    }
