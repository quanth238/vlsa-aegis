"""Deterministic, VLA-ledger-independent backup-policy primitives.

The functions in this module deliberately know nothing about a VLA action or
proposal ledger.  They construct a small registered action family from the
measured robot/obstacle geometry and choose a physically verified member with
fixed tie breaking.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


PURE_BACKUP_PROTOCOL = "vlsa-distal-pncbf-pure-backup-audit-e05-v1"


def load_pure_backup_config(path: Any) -> dict[str, Any]:
    import json
    from pathlib import Path

    value = json.loads(Path(path).read_text())
    if value.get("protocol_id") != PURE_BACKUP_PROTOCOL:
        raise ValueError("pure backup protocol differs")
    if value.get("case_ids") != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("pure backup case differs")
    if value.get("decision_steps") != [187, 222, 288]:
        raise ValueError("pure backup decision states differ")
    candidate = value.get("candidate_family", {})
    if candidate.get("directions") != "hold_plus_world_axes_plus_local_normal_tangents":
        raise ValueError("pure backup directions differ")
    if candidate.get("amplitudes_action") != [0.5, 1.0]:
        raise ValueError("pure backup amplitudes differ")
    if candidate.get("execute_actions") != 1:
        raise ValueError("pure backup must replan after one action")
    if candidate.get("terminal_hold_actions") != 25:
        raise ValueError("pure backup terminal tail differs")
    if float(value.get("safety_buffer_m", -1.0)) != 0.001:
        raise ValueError("pure backup buffer differs")
    if float(value.get("paper_car_threshold_m", -1.0)) != 0.001:
        raise ValueError("pure backup CAR threshold differs")
    return value


def temporal_profile(action_count: int) -> list[float]:
    """Return a nonnegative unit-L1 profile for incremental actions."""

    if int(action_count) <= 0:
        raise ValueError("action_count must be positive")
    return [1.0 / float(action_count)] * int(action_count)


def orthonormal_local_frame(normal: Sequence[float]) -> dict[str, list[float]]:
    """Construct obstacle normal/up/side directions with deterministic fallback."""

    import math

    n = [float(item) for item in normal]
    if len(n) != 3:
        raise ValueError("normal must have length three")
    norm = math.sqrt(sum(item * item for item in n))
    if not norm > 1.0e-12:
        raise ValueError("normal is degenerate")
    n = [item / norm for item in n]

    def tangent(reference: Sequence[float]) -> list[float]:
        dot = sum(n[i] * float(reference[i]) for i in range(3))
        return [float(reference[i]) - n[i] * dot for i in range(3)]

    up = tangent([0.0, 0.0, 1.0])
    up_norm = math.sqrt(sum(item * item for item in up))
    if up_norm <= 1.0e-8:
        up = tangent([0.0, 1.0, 0.0])
        up_norm = math.sqrt(sum(item * item for item in up))
    up = [item / up_norm for item in up]
    side = [
        n[1] * up[2] - n[2] * up[1],
        n[2] * up[0] - n[0] * up[2],
        n[0] * up[1] - n[1] * up[0],
    ]
    side_norm = math.sqrt(sum(item * item for item in side))
    side = [item / side_norm for item in side]
    return {
        "normal": n,
        "tangent_up": up,
        "tangent_side": side,
    }


def registered_directions(normal: Sequence[float]) -> list[tuple[str, list[float]]]:
    """Return the fixed world-axis and state-derived local-frame candidate order."""

    frame = orthonormal_local_frame(normal)
    base = [
        ("world_pos_x", [1.0, 0.0, 0.0]),
        ("world_neg_x", [-1.0, 0.0, 0.0]),
        ("world_pos_y", [0.0, 1.0, 0.0]),
        ("world_neg_y", [0.0, -1.0, 0.0]),
        ("world_pos_z", [0.0, 0.0, 1.0]),
        ("world_neg_z", [0.0, 0.0, -1.0]),
    ]
    for name in ("normal", "tangent_up", "tangent_side"):
        value = frame[name]
        base.append(("local_pos_%s" % name, value))
        base.append(("local_neg_%s" % name, [-float(item) for item in value]))
    return base


def select_verified_backup(
    candidates: Sequence[Mapping[str, Any]],
    *,
    safety_buffer_m: float,
    paper_car_threshold_m: float,
) -> Mapping[str, Any] | None:
    """Choose maximum verified clearance with a fixed order/name tie break."""

    eligible = []
    for item in candidates:
        record = item["record"]
        if int(record["protected_contact_count"]) != 0:
            continue
        if float(record["maximum_active_obstacle_l1_displacement_m"]) > float(
            paper_car_threshold_m
        ):
            continue
        if float(record["minimum_clearance_m"]) < float(safety_buffer_m):
            continue
        eligible.append(item)
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            -float(item["record"]["future_minimum_clearance_m"]),
            int(item["order"]),
            str(item["name"]),
        ),
    )
