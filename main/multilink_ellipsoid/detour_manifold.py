"""Task-relative candidate manifold for the no-learning E05 oracle audit."""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

from .shadow import _numpy


def task_relative_basis(first_five_xyz: Any) -> dict[str, Any]:
    """Return generic soft/free and endpoint-preserving detour bases.

    Columns are normalized in the 15-dimensional normalized Cartesian action
    space.  The construction depends only on the nominal task-progress
    direction and world up; it is not fitted to a successful E05 trajectory.
    """

    np = _numpy()
    actions = np.asarray(first_five_xyz, dtype=np.float64)
    if actions.shape != (5, 3) or not np.all(np.isfinite(actions)):
        raise ValueError("detour basis action shape differs")
    progress = np.sum(actions, axis=0)
    if float(np.linalg.norm(progress)) <= 1.0e-10:
        progress = np.mean(actions, axis=0)
    if float(np.linalg.norm(progress)) <= 1.0e-10:
        raise ValueError("nominal task progress is degenerate")
    progress = progress / np.linalg.norm(progress)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    up = world_up - progress * float(np.dot(progress, world_up))
    if float(np.linalg.norm(up)) <= 1.0e-8:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        up = fallback - progress * float(np.dot(progress, fallback))
    up = up / np.linalg.norm(up)
    side = np.cross(progress, up)
    side = side / np.linalg.norm(side)

    def normalized_column(temporal: Any, spatial: Any) -> Any:
        column = np.outer(
            np.asarray(temporal, dtype=np.float64),
            np.asarray(spatial, dtype=np.float64),
        ).reshape(15)
        norm = float(np.linalg.norm(column))
        if norm <= 1.0e-12:
            raise ValueError("detour basis column is degenerate")
        return column / norm

    early = np.asarray([1.0, 0.75, 0.25, 0.0, 0.0], dtype=np.float64)
    late = np.asarray([0.0, 0.0, 0.25, 0.75, 1.0], dtype=np.float64)
    slowdown = -actions.reshape(15)
    slowdown_norm = float(np.linalg.norm(slowdown))
    if slowdown_norm <= 1.0e-12:
        slowdown = -np.tile(progress, 5)
        slowdown_norm = float(np.linalg.norm(slowdown))
    slowdown = slowdown / slowdown_norm
    soft = np.stack(
        [
            normalized_column(early, up),
            normalized_column(early, side),
            normalized_column(late, up),
            normalized_column(late, side),
            slowdown,
        ],
        axis=1,
    )

    early_zero = np.asarray([1.0, 0.5, -0.5, -1.0, 0.0], dtype=np.float64)
    late_zero = np.asarray([0.0, 1.0, 0.5, -0.5, -1.0], dtype=np.float64)
    endpoint = np.stack(
        [
            normalized_column(early_zero, up),
            normalized_column(early_zero, side),
            normalized_column(late_zero, up),
            normalized_column(late_zero, side),
        ],
        axis=1,
    )
    if not np.allclose(endpoint.reshape(5, 3, 4).sum(axis=0), 0.0, atol=1.0e-12):
        raise ValueError("endpoint-preserving detour basis does not close")
    return {
        "progress_direction": progress,
        "transverse_up": up,
        "transverse_side": side,
        "soft_matrix": soft,
        "endpoint_matrix": endpoint,
        "soft_names": [
            "early_up",
            "early_side",
            "late_up",
            "late_side",
            "slowdown",
        ],
        "endpoint_names": [
            "early_up_zero_sum",
            "early_side_zero_sum",
            "late_up_zero_sum",
            "late_side_zero_sum",
        ],
    }


def finite_soft_library() -> list[dict[str, Any]]:
    """Return the preregistered 54-point discretization of the soft manifold."""

    np = _numpy()
    rows: list[tuple[str, Sequence[float]]] = [("nominal", [0.0] * 5)]
    for index in range(4):
        for amplitude in (0.25, 0.5, 0.75):
            for sign in (-1.0, 1.0):
                value = np.zeros(5, dtype=np.float64)
                value[index] = sign * amplitude
                rows.append(("transverse_axis", value.tolist()))
    for amplitude in (0.25, 0.5, 0.75):
        rows.append(("slowdown", [0.0, 0.0, 0.0, 0.0, amplitude]))
    for axis, sign in itertools.product((0, 1), (-1.0, 1.0)):
        late_axis = axis + 2
        for early_amplitude, late_amplitude in (
            (0.25, 0.5),
            (0.5, 0.5),
            (0.75, 0.5),
            (0.75, 0.75),
        ):
            value = np.zeros(5, dtype=np.float64)
            value[axis] = sign * early_amplitude
            value[late_axis] = sign * late_amplitude
            rows.append(("coherent_arc", value.tolist()))
    for early_axis, late_axis, early_sign, late_sign in (
        (0, 3, 1.0, 1.0),
        (0, 3, 1.0, -1.0),
        (0, 3, -1.0, 1.0),
        (0, 3, -1.0, -1.0),
        (1, 2, 1.0, 1.0),
        (1, 2, 1.0, -1.0),
        (1, 2, -1.0, 1.0),
        (1, 2, -1.0, -1.0),
    ):
        value = np.zeros(5, dtype=np.float64)
        value[early_axis] = 0.5 * early_sign
        value[late_axis] = 0.5 * late_sign
        rows.append(("around_arc", value.tolist()))
    rows.extend(
        [
            ("up_slowdown_backup", [0.5, 0.0, 0.0, 0.0, 0.5]),
            ("side_slowdown_backup", [0.0, 0.5, 0.0, 0.0, 0.5]),
        ]
    )
    if len(rows) != 54:
        raise ValueError("finite detour library count differs")
    coefficients = np.asarray([row[1] for row in rows], dtype=np.float64)
    if np.unique(coefficients, axis=0).shape[0] != 54:
        raise ValueError("finite detour library contains duplicates")
    return [
        {"index": index, "family": family, "coefficients": list(coefficients[index])}
        for index, (family, _) in enumerate(rows)
    ]


def candidate_gate(record: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate physical safety and task compatibility without proxy conflation."""

    checks = {
        "exact_box_nonoverlap": int(record["exact_overlap_sample_count"]) == 0
        and float(record["minimum_exact_normalized_radial_slack"]) >= 0.0,
        "zero_protected_contact": int(record["protected_contact_count"]) == 0,
        "paper_car": float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"]),
        "terminal_eef": float(record["terminal_eef_error_m"])
        <= float(gate["maximum_terminal_eef_error_m"]),
        "task_progress": float(record["task_progress_ratio"])
        >= float(gate["minimum_task_progress_ratio"]),
    }
    physical = bool(
        checks["exact_box_nonoverlap"]
        and checks["zero_protected_contact"]
        and checks["paper_car"]
    )
    task = bool(checks["terminal_eef"] and checks["task_progress"])
    return {"checks": checks, "physical_safe": physical, "task_compatible": task, "pass": physical and task}


def controllability_gate(record: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "initial_exact_safe": not bool(record["initial_exact_overlap"]),
        "initial_contact_free": int(record["initial_protected_contact_count"]) == 0,
        "command_influence_precedes_violation": int(record["first_influence_sample_index"])
        < int(record["first_physical_violation_sample_index"]),
        "cartesian_authority": float(record["maximum_previolation_link_center_change_m"])
        >= float(gate["minimum_previolation_link_center_change_m"]),
        "genuine_future_physical_violation": int(record["nominal_protected_contact_count"]) > 0,
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}
