"""Contracts for late-flow branching with terminal executable-action scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_terminalized_late_flow.v1"
ENVELOPE_SCHEMA = "crfs_terminal_branching.v1"
RESULT_SCHEMA = "vlsa_distal_terminal_branch_sampler_canary_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_terminal_branch_sampler_canary_validation.v1"


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
