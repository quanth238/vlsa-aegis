"""Contracts for exact outward-normal magnitude risk curves."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_normal_risk_curve.v1"
RESULT_SCHEMA = "vlsa_distal_normal_risk_curve_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "base_risk_config",
        "selection_manifest", "warning_steps", "query_indices",
        "requested_alpha", "direction", "temporal_profile", "action_limit",
        "risk_definition", "monotonicity_tolerance_m",
        "unknown_timeout_handling", "selection_rule", "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("normal risk-curve config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("normal risk-curve schema differs")
    if value["protocol_id"] != "vlsa-distal-normal-risk-curve-v1":
        raise ValueError("normal risk-curve protocol differs")
    if value["warning_steps"] != [25, 100, 110]:
        raise ValueError("normal risk-curve warning states differ")
    if value["query_indices"] != [5, 20, 22]:
        raise ValueError("normal risk-curve query indices differ")
    alpha = [float(item) for item in value["requested_alpha"]]
    if alpha != [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        raise ValueError("normal risk-curve alpha grid differs")
    if any(right <= left for left, right in zip(alpha, alpha[1:])):
        raise ValueError("normal risk-curve alpha grid is not increasing")
    if value["direction"] != "current_closest_slab_outward_normal":
        raise ValueError("normal risk-curve direction differs")
    if value["temporal_profile"] != "front_loaded_unit_L2_5_4_3_2_1":
        raise ValueError("normal risk-curve temporal profile differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _front_loaded_profile() -> list[float]:
    values = [5.0, 4.0, 3.0, 2.0, 1.0]
    norm = math.sqrt(sum(item * item for item in values))
    return [item / norm for item in values]


def candidate_definitions(
    nominal_actions: Sequence[Sequence[float]],
    local_frame: Mapping[str, Sequence[float]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Create one clipped five-action proposal for each requested alpha."""

    import numpy as np

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    if nominal.shape != (5, 7) or not np.all(np.isfinite(nominal)):
        raise ValueError("normal risk-curve nominal chunk differs")
    direction = np.asarray(local_frame["normal"], dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("normal risk-curve direction differs")
    direction /= np.linalg.norm(direction)
    profile = np.asarray(_front_loaded_profile(), dtype=np.float64)
    limit = float(config["action_limit"])
    output = []
    for order, requested in enumerate(config["requested_alpha"]):
        requested = float(requested)
        actions = nominal.copy()
        requested_delta = requested * profile[:, None] * direction[None, :]
        actions[:, :3] = np.clip(actions[:, :3] + requested_delta, -limit, limit)
        applied = actions[:, :3] - nominal[:, :3]
        output.append({
            "name": "nominal" if requested == 0.0 else "normal_alpha_%0.2f" % requested,
            "order": order,
            "direction": direction.tolist(),
            "sign": 1,
            "temporal_profile": profile.tolist(),
            "requested_alpha": requested,
            "requested_correction_l2_action": requested,
            "applied_correction_l2_action": float(np.linalg.norm(applied)),
            "clipped": bool(np.max(np.abs(applied - requested_delta)) > 1.0e-12),
            "actions": actions.tolist(),
        })
    return output


def curve_summary(
    candidates: Sequence[Mapping[str, Any]], *, tolerance_m: float,
) -> dict[str, Any]:
    """Summarize known labels without promoting timeouts to outcomes."""

    ordered = sorted(candidates, key=lambda row: float(row["requested_alpha"]))
    known = [row for row in ordered if row["terminal_status"] != "UNKNOWN_TIMEOUT"]
    monotonic_pairs = []
    row_violations = [0] * 7
    global_violations = 0
    for left, right in zip(known, known[1:]):
        row_delta = [
            float(r) - float(l)
            for l, r in zip(left["combined_risk"], right["combined_risk"])
        ]
        violations = [delta > float(tolerance_m) for delta in row_delta]
        for index, violation in enumerate(violations):
            row_violations[index] += int(violation)
        global_delta = (
            max(float(item) for item in right["combined_risk"])
            - max(float(item) for item in left["combined_risk"])
        )
        global_violation = global_delta > float(tolerance_m)
        global_violations += int(global_violation)
        monotonic_pairs.append({
            "left_alpha": float(left["requested_alpha"]),
            "right_alpha": float(right["requested_alpha"]),
            "per_row_risk_delta_m": row_delta,
            "per_row_violation": violations,
            "global_worst_risk_delta_m": global_delta,
            "global_violation": global_violation,
        })
    safe = [row for row in ordered if bool(row["exact_safe"])]
    selected = min(safe, key=lambda row: float(row["requested_alpha"])) if safe else None
    return {
        "candidate_count": len(ordered),
        "known_candidate_count": len(known),
        "unknown_timeout_count": len(ordered) - len(known),
        "safe_candidate_count": len(safe),
        "unsafe_known_candidate_count": sum(
            not bool(row["exact_safe"]) for row in known
        ),
        "minimum_exact_safe_requested_alpha": (
            None if selected is None else float(selected["requested_alpha"])
        ),
        "minimum_exact_safe_effective_post_AEGIS_l2_action": (
            None if selected is None
            else float(selected["effective_post_AEGIS_correction_l2_action"])
        ),
        "minimum_exact_safe_candidate_name": (
            None if selected is None else str(selected["name"])
        ),
        "per_row_monotonicity_violation_count": row_violations,
        "global_worst_risk_monotonicity_violation_count": global_violations,
        "known_adjacent_pairs": monotonic_pairs,
    }
