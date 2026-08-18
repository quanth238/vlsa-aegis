"""Contracts for one nominal-first detour-and-rejoin full-episode pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.tight_prefix_full_episode import (
    CASE_ID,
    canonical,
    file_sha256,
    payload_sha256,
    scientific_view,
)

TRANSFER_CASE_ID = "vlsa-t1-spatial-i-t3-e42"
ALLOWED_CASE_IDS = (CASE_ID, TRANSFER_CASE_ID)


CONFIG_SCHEMA = "vlsa_tight_prefix_nominal_first_full_episode.v1"
RESULT_SCHEMA = "vlsa_tight_prefix_nominal_first_full_episode_result.v1"
VALIDATION_SCHEMA = (
    "vlsa_tight_prefix_nominal_first_full_episode_validation.v1"
)

DETOUR_CANDIDATE_NAMES = (
    "nominal",
    "detour_normal_m1_r1.0",
    "detour_normal_p1_r1.0",
    "detour_normal_m1_r2.0",
    "detour_normal_p1_r2.0",
    "detour_tangent_up_m1_r1.0",
    "detour_tangent_up_p1_r1.0",
    "detour_tangent_up_m1_r2.0",
    "detour_tangent_up_p1_r2.0",
    "detour_tangent_side_m1_r1.0",
    "detour_tangent_side_p1_r1.0",
    "detour_tangent_side_m1_r2.0",
    "detour_tangent_side_p1_r2.0",
)


def detour_and_rejoin_candidates(
    nominal_actions: Sequence[Sequence[float]],
    local_frame: Mapping[str, Sequence[float]],
    *,
    magnitudes: Sequence[float] = (1.0, 2.0),
    action_limit: float = 1.0,
) -> list[dict[str, Any]]:
    """Build 12 zero-sum translation detours without relying on clipping."""

    import numpy as np

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    if nominal.shape != (5, 7) or not np.all(np.isfinite(nominal)):
        raise ValueError("nominal-first detour nominal differs")
    if float(action_limit) != 1.0:
        raise ValueError("nominal-first detour action limit differs")
    if tuple(float(value) for value in magnitudes) != (1.0, 2.0):
        raise ValueError("nominal-first detour magnitudes differ")
    effective_nominal = nominal.copy()
    effective_nominal[:, :3] = np.clip(
        effective_nominal[:, :3], -float(action_limit), float(action_limit),
    )
    profile = np.asarray([0.5, 0.5, 0.0, -0.5, -0.5], dtype=np.float64)
    output = [{
        "name": "nominal",
        "order": 0,
        "axis": None,
        "sign": 0,
        "requested_magnitude": 0.0,
        "applied_magnitude": 0.0,
        "correction_sum_xyz": [0.0, 0.0, 0.0],
        "endpoint_preserved": True,
        "actions": effective_nominal.tolist(),
    }]
    for axis in ("normal", "tangent_up", "tangent_side"):
        direction = np.asarray(local_frame[axis], dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("nominal-first detour frame differs")
        norm = float(np.linalg.norm(direction))
        if abs(norm - 1.0) > 1.0e-12:
            raise ValueError("nominal-first detour frame is not normalized")
        for magnitude in magnitudes:
            for sign in (-1, 1):
                unit_correction = (
                    profile[:, None] * float(sign) * direction[None, :]
                )
                feasible = float(magnitude)
                for row in range(5):
                    for coordinate in range(3):
                        coefficient = float(unit_correction[row, coordinate])
                        if coefficient > 0.0:
                            feasible = min(
                                feasible,
                                (float(action_limit) - effective_nominal[
                                    row, coordinate
                                ]) / coefficient,
                            )
                        elif coefficient < 0.0:
                            feasible = min(
                                feasible,
                                (-float(action_limit) - effective_nominal[
                                    row, coordinate
                                ]) / coefficient,
                            )
                feasible = max(0.0, min(float(magnitude), feasible))
                correction = feasible * unit_correction
                actions = effective_nominal.copy()
                actions[:, :3] += correction
                correction_sum = np.sum(correction, axis=0)
                if (
                    float(np.max(np.abs(actions[:, :3]))) > 1.0 + 1.0e-12
                    or float(np.max(np.abs(correction_sum))) > 1.0e-12
                ):
                    raise ValueError("nominal-first detour endpoint differs")
                output.append({
                    "name": "detour_%s_%s_r%.1f" % (
                        axis, "p1" if sign > 0 else "m1", magnitude,
                    ),
                    "order": len(output),
                    "axis": axis,
                    "sign": int(sign),
                    "requested_magnitude": float(magnitude),
                    "applied_magnitude": float(feasible),
                    "correction_sum_xyz": correction_sum.tolist(),
                    "endpoint_preserved": True,
                    "actions": actions.tolist(),
                })
    by_name = {row["name"]: row for row in output}
    if set(by_name) != set(DETOUR_CANDIDATE_NAMES):
        raise ValueError("nominal-first detour candidate names differ")
    return [by_name[name] for name in DETOUR_CANDIDATE_NAMES]


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "case_id",
        "source", "method", "controller", "termination",
        "physical_contact_groups", "verification", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("nominal-first full-episode config keys differ")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("protocol_id")
        != "vlsa-tight-prefix-nominal-first-full-episode-v1"
        or value.get("case_id") not in ALLOWED_CASE_IDS
    ):
        raise ValueError("nominal-first full-episode protocol differs")
    method = value["method"]
    if method != {
        "start_step": 0,
        "replan_stride": 5,
        "warning": {
            "rows": list(range(8)),
            "normalized_radial_slack_strictly_below": 1.0,
            "outside_warning": "execute_nominal_without_critic",
        },
        "candidate_count_inside_warning": 13,
        "candidate_names": list(DETOUR_CANDIDATE_NAMES),
        "candidate_generation": {
            "coordinates": "current_minimum_slack_tight_row_local_frame",
            "profile": [0.5, 0.5, 0.0, -0.5, -0.5],
            "translation_axes": ["normal", "tangent_up", "tangent_side"],
            "signs": [-1, 1],
            "magnitudes": [1.0, 2.0],
            "action_limit": 1.0,
            "endpoint_preservation": (
                "sum_effective_xyz_correction_over_five_actions_equals_zero"
            ),
            "rotation_and_gripper_preserved": True,
        },
        "critic": "frozen_ADR_0201_compact_shared_7D_tight_prefix",
        "model_rows": list(range(10)),
        "primary_rows": list(range(8)),
        "diagnostic_rows": [8, 9],
        "acceptance": {
            "rule": "all_primary_rows_at_or_below_negative_margin",
            "margin": 0.1421400248048467,
            "margin_source": (
                "ADR_0201_validation_primary_near_boundary_RMSE"
            ),
            "margin_is_empirical_not_formal": True,
        },
        "selection": (
            "nominal_if_accepted_else_minimum_effective_correction_"
            "accepted_else_abstain"
        ),
        "translation_scale_m_per_action_unit": 0.05,
    }:
        raise ValueError("nominal-first full-episode method differs")
    source = value["source"]
    required_source = {
        "training_config", "training_config_file_sha256",
        "training_config_payload_sha256", "training_result",
        "training_result_file_sha256", "training_result_payload_sha256",
        "model_sha256", "training_validation",
        "training_validation_file_sha256",
        "training_validation_payload_sha256", "tight_dataset_config",
        "tight_dataset_config_file_sha256",
        "tight_dataset_config_payload_sha256", "geometry_archive",
        "geometry_archive_file_sha256", "geometry_archive_payload_sha256",
        "raw_pi05_archive", "raw_pi05_archive_file_sha256",
        "raw_pi05_archive_payload_sha256", "prior_full_episode_validation",
        "prior_full_episode_validation_file_sha256",
        "prior_full_episode_validation_payload_sha256",
    }
    if set(source) != required_source:
        raise ValueError("nominal-first full-episode source keys differ")
    forbidden = value["forbidden"]
    if any(forbidden.get(key) is not False for key in (
        "released_AEGIS_EE_QP", "learned_QP", "late_denoising",
        "simulator_candidate_rollout", "model_training", "data_collection",
    )):
        raise ValueError("nominal-first full-episode forbidden component enabled")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output
