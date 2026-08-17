"""Contracts for late-flow branching with terminal executable-action scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


CONFIG_SCHEMA = "vlsa_distal_terminalized_late_flow.v1"
ENVELOPE_SCHEMA = "crfs_terminal_branching.v1"
RESULT_SCHEMA = "vlsa_distal_terminal_branch_sampler_canary_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_terminal_branch_sampler_canary_validation.v1"
PILOT_RESULT_SCHEMA = "vlsa_distal_terminalized_late_flow_pilot_result.v1"
PILOT_VALIDATION_SCHEMA = "vlsa_distal_terminalized_late_flow_pilot_validation.v1"


FROZEN_CANDIDATE_NAMES = (
    "nominal",
    "grid_m1_m1_m1_front_loaded_r2.0",
    "grid_p1_p1_p1_front_loaded_r2.0",
    "grid_m1_m1_z0_front_loaded_r2.0",
    "grid_p1_p1_z0_front_loaded_r2.0",
    "grid_m1_z0_m1_front_loaded_r2.0",
    "grid_p1_z0_p1_front_loaded_r2.0",
    "grid_m1_p1_z0_front_loaded_r2.0",
    "grid_p1_m1_z0_front_loaded_r2.0",
    "grid_z0_m1_m1_front_loaded_r2.0",
    "grid_z0_p1_p1_front_loaded_r2.0",
    "grid_z0_z0_m1_front_loaded_r2.0",
    "grid_z0_z0_p1_front_loaded_r2.0",
)


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
        raise ValueError("terminalized late-flow config schema differs")
    if value.get("protocol_id") != "vlsa-distal-terminalized-late-flow-v1":
        raise ValueError("terminalized late-flow protocol differs")
    flow = value.get("flow", {})
    if flow != {
        "sampler_steps": 10,
        "branch_after_euler_step": 8,
        "remaining_euler_steps": 2,
        "candidate_count": 13,
        "candidate_names": list(FROZEN_CANDIDATE_NAMES),
        "residual_coordinates": "physical_output_action_displacement",
        "residual_normalization": "scale_only_never_subtract_mean",
        "terminalization": "complete_each_branch_through_frozen_pi05_to_t0",
    }:
        raise ValueError("terminalized late-flow schedule differs")
    scoring = value.get("scoring", {})
    if scoring != {
        "critic": "frozen_ADR_0191_compact_shared_7D",
        "critic_input": "effective_terminal_five_action_chunk_after_clipping",
        "primary_rows": [0, 1, 2, 3, 4],
        "selection": "minimum_predicted_primary_risk_then_intervention_tie_break",
        "risk_scored_inside_sampler": False,
    }:
        raise ValueError("terminalized late-flow scoring differs")
    if value.get("sampler_canary", {}).get("case_id") != "vlsa-t1-goal-ii-t0-e05":
        raise ValueError("terminalized late-flow canary case differs")
    validated = value.get("validated_sampler_canary", {})
    if not (
        validated.get("file_sha256")
        == "0e97f798ec2c1d1138c06a5c102b85b80bd5e468603104ed1c739311d95bb181"
        and validated.get("payload_sha256")
        == "65d8d293d7a8b5b50d313e5cf7cfb9b2165700c278e26034fe2cb2a7b1b22d2b"
    ):
        raise ValueError("terminalized late-flow sampler validation differs")
    mechanism = value.get("mechanism_case", {})
    if not (
        mechanism.get("case_id") == "vlsa-t1-goal-ii-t0-e05"
        and mechanism.get("state_step") == 180
        and mechanism.get("query_index") == 36
        and mechanism.get("source_case_index") == 11
        and mechanism.get("source_snapshot_sha256")
        == "81a2a29dc065715f5fb096b3e7fbf0443d126844eeaf9b2c0632c6557401682e"
    ):
        raise ValueError("terminalized late-flow mechanism case differs")
    pilot = value.get("paired_pilot", {})
    if not (
        pilot.get("arms") == [
            "ordinary_pi05", "terminal_compact_selector",
            "late_flow_terminalized_compact_selector",
        ]
        and pilot.get("execute_only_one_selected_chunk_per_arm") is True
        and pilot.get("selected_arm_count_per_replica") == 3
        and pilot.get("independent_replica_count") == 2
        and pilot.get("source_nominal_determinism_checks_per_replica") == 2
        and pilot.get("exact_outcome_replay_each_selected_arm") is True
        and pilot.get("released_AEGIS_EE_QP_enabled") is False
        and pilot.get("learned_QP_enabled") is False
        and pilot.get("exact_rollout_verifier_at_inference") is False
    ):
        raise ValueError("terminalized late-flow paired pilot differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def build_envelope(
    nominal_actions: Sequence[Sequence[float]],
    candidate_actions: Sequence[Sequence[Sequence[float]]],
    candidate_names: Sequence[str],
    *,
    branch_after_euler_step: int,
) -> dict[str, Any]:
    """Build a strict server request from complete physical action chunks."""

    import numpy as np

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    candidates = np.asarray(candidate_actions, dtype=np.float64)
    names = [str(name) for name in candidate_names]
    if nominal.shape != (10, 7):
        raise ValueError("terminal-branch nominal action shape differs")
    if candidates.shape != (len(names), 10, 7):
        raise ValueError("terminal-branch candidate action bank differs")
    if tuple(names) != FROZEN_CANDIDATE_NAMES:
        raise ValueError("terminal-branch frozen candidate names differ")
    residuals = candidates[:, :, :3] - nominal[None, :, :3]
    if not np.array_equal(residuals[0], np.zeros((10, 3), dtype=np.float64)):
        raise ValueError("terminal-branch nominal residual differs")
    if not np.all(np.isfinite(residuals)) or np.max(np.abs(residuals)) > 2.0:
        raise ValueError("terminal-branch residual bank is invalid")
    return {
        "schema_version": ENVELOPE_SCHEMA,
        "action_horizon": 10,
        "action_dimensions": [0, 1, 2],
        "candidate_names": names,
        "candidate_output_residuals": residuals.tolist(),
        "branch_after_euler_step": int(branch_after_euler_step),
    }


def score_terminal_bank(
    *,
    exact_case: Mapping[str, Any],
    ordinary_terminal_actions: Sequence[Sequence[float]],
    terminal_action_bank: Sequence[Sequence[Sequence[float]]],
    candidate_names: Sequence[str],
    state_payload: Mapping[str, Any],
    primary_rows: Sequence[int] = (0, 1, 2, 3, 4),
    model_rows: Optional[Sequence[int]] = None,
    translation_scale: float = 0.05,
) -> dict[str, Any]:
    """Score only clipped, terminal executable chunks with the frozen critic."""

    import copy
    import math
    import numpy as np

    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        MODEL_ROWS, safety_coordinate_feature,
    )
    from main.multilink_ellipsoid.compact_selector_ablation import (
        predict_serialized_mlp_float32,
    )

    names = [str(name) for name in candidate_names]
    if tuple(names) != FROZEN_CANDIDATE_NAMES:
        raise ValueError("terminal scorer candidate names differ")
    ordinary = np.asarray(ordinary_terminal_actions, dtype=np.float64)
    terminal = np.asarray(terminal_action_bank, dtype=np.float64)
    if ordinary.shape != (10, 7) or terminal.shape != (13, 10, 7):
        raise ValueError("terminal scorer action shapes differ")
    if not np.all(np.isfinite(ordinary)) or not np.all(np.isfinite(terminal)):
        raise ValueError("terminal scorer actions are non-finite")
    effective_ordinary = ordinary.copy()
    effective_terminal = terminal.copy()
    effective_ordinary[:, :3] = np.clip(effective_ordinary[:, :3], -1.0, 1.0)
    effective_terminal[:, :, :3] = np.clip(
        effective_terminal[:, :, :3], -1.0, 1.0
    )
    if float(translation_scale) != 0.05:
        raise ValueError("terminal scorer translation scale differs")
    explicit_model_rows = model_rows is not None
    rows = tuple(
        int(row) for row in (
            MODEL_ROWS if model_rows is None else model_rows
        )
    )
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("terminal scorer model rows differ")
    primary = tuple(int(row) for row in primary_rows)
    if not primary or not set(primary).issubset(rows):
        raise ValueError("terminal scorer primary rows differ")

    feature_case = copy.deepcopy(dict(exact_case))
    feature_case["source_nominal_five_action_chunk"] = (
        effective_ordinary[:5].tolist()
    )
    features = []
    metadata = []
    for candidate_index, name in enumerate(names):
        candidate = {
            "source_executed_actions": effective_terminal[
                candidate_index, :5
            ].tolist(),
        }
        for row in rows:
            feature_kwargs = {"translation_scale": float(translation_scale)}
            if explicit_model_rows:
                feature_kwargs["model_rows"] = rows
            features.append(safety_coordinate_feature(
                feature_case, candidate, row, **feature_kwargs
            ))
            metadata.append((candidate_index, row))
    values = predict_serialized_mlp_float32(features, state_payload)
    if len(values) != len(metadata) or any(len(value) != 1 for value in values):
        raise ValueError("terminal scorer prediction shape differs")
    predictions = np.empty((len(names), len(rows)), dtype=np.float64)
    for (candidate_index, row), value in zip(metadata, values):
        predictions[candidate_index, rows.index(row)] = float(value[0])
    if not np.all(np.isfinite(predictions)):
        raise ValueError("terminal scorer prediction is non-finite")
    corrections = np.linalg.norm(
        effective_terminal[:, :5, :3] - effective_ordinary[None, :5, :3],
        axis=(1, 2),
    )
    records = []
    for candidate_index, name in enumerate(names):
        by_row = {
            str(row): float(predictions[candidate_index, rows.index(row)])
            for row in rows
        }
        predicted_primary = max(by_row[str(row)] for row in primary)
        records.append({
            "candidate_name": name,
            "candidate_order": candidate_index,
            "predicted_by_row": by_row,
            "predicted_primary": float(predicted_primary),
            "effective_correction_l2_action": float(corrections[candidate_index]),
            "effective_first_five_actions": effective_terminal[
                candidate_index, :5
            ].tolist(),
        })
    selected = min(records, key=lambda record: (
        float(record["predicted_primary"]),
        float(record["effective_correction_l2_action"]),
        int(record["candidate_order"]),
    ))
    if not math.isfinite(float(selected["predicted_primary"])):
        raise ValueError("terminal scorer selection differs")
    return {
        "selection_rule": (
            "minimum_predicted_primary_risk_then_intervention_tie_break"
        ),
        "primary_rows": list(primary),
        "translation_scale_m_per_action_unit": float(translation_scale),
        "candidate_count": len(records),
        "selected_candidate": str(selected["candidate_name"]),
        "selected_candidate_order": int(selected["candidate_order"]),
        "selected_predicted_primary": float(selected["predicted_primary"]),
        "selected_effective_first_five_actions": selected[
            "effective_first_five_actions"
        ],
        "records": records,
    }
