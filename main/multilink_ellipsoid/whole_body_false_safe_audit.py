"""Exact attribution of selected-action false-safes by physical constraint."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


RESULT_SCHEMA = "vlsa_distal_whole_body_false_safe_audit.v1"
PHYSICAL_GROUPS = ("palm", "L5", "L6")
REPRESENTED_GROUPS = ("end_effector", "palm", "L5", "L6", "L7")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def selected_false_safe_states(prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen shared-model held-out selections that are not exact-safe."""

    arm = prediction["arms"]["shared_constraint_135D"]["metrics"]
    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        states = arm[split]["global"]["states"]
        for state in states:
            if (
                state.get("selected_candidate") is not None
                and state.get("selected_actual_safe") is False
            ):
                rows.append({"split": split, **dict(state)})
    return rows


def attribute_selection(
    selection: Mapping[str, Any], case: Mapping[str, Any]
) -> dict[str, Any]:
    """Join one selected action to its authoritative per-constraint rollout target."""

    _require(case["case_id"] == selection["state_id"], "case identity differs")
    candidates = case["exact_case"]["candidates"]
    matches = [
        row for row in candidates
        if row["name"] == selection["selected_candidate"]
    ]
    _require(len(matches) == 1, "selected candidate identity differs")
    candidate = matches[0]
    target = candidate["exact_group_target"]
    _require(target["known_outcome"] is True, "selected outcome is unknown")
    risks = {
        group: float(target["group_future_violation"][group])
        for group in REPRESENTED_GROUPS
    }
    physical_witness = max(PHYSICAL_GROUPS, key=lambda group: risks[group])
    represented_witness = max(REPRESENTED_GROUPS, key=lambda group: risks[group])
    _require(risks[physical_witness] > 0.0, "selection is not physical false-safe")
    contacts = []
    for group in REPRESENTED_GROUPS:
        for event in target["group_contact_events"].get(group, []):
            contacts.append({
                "group": group,
                "phase": event["phase"],
                "action_offset": int(event["action_offset"]),
                "substep": int(event["substep"]),
                "robot_geom": event["other"]["geom_name"],
                "obstacle_geom": event["obstacle"]["geom_name"],
            })
    return {
        "state_id": selection["state_id"],
        "split": selection["split"],
        "selected_candidate": selection["selected_candidate"],
        "selected_predicted_global_risk": float(
            selection["selected_predicted_global_risk"]
        ),
        "selected_actual_safe": False,
        "future_risk": risks,
        "active_physical_witness": physical_witness,
        "active_physical_witness_Q": risks[physical_witness],
        "active_represented_witness": represented_witness,
        "active_represented_witness_Q": risks[represented_witness],
        "raw_contact_events": contacts,
    }


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    physical = Counter(row["active_physical_witness"] for row in rows)
    represented = Counter(row["active_represented_witness"] for row in rows)
    return {
        "selected_false_safe_count": len(rows),
        "split_count": dict(sorted(Counter(row["split"] for row in rows).items())),
        "active_physical_witness_count": {
            group: int(physical.get(group, 0)) for group in PHYSICAL_GROUPS
        },
        "active_represented_witness_count": {
            group: int(represented.get(group, 0)) for group in REPRESENTED_GROUPS
        },
        "all_failures_explained_by_claimed_physical_heads": all(
            row["active_physical_witness"] in PHYSICAL_GROUPS for row in rows
        ),
    }
