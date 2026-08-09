"""Pair construction and scoring for contact-directional supervision."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def symmetric_directional_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    allowed_source_prefixes: Sequence[str],
    maximum_radius_action: float,
    matching_tolerance: float,
) -> list[dict[str, Any]]:
    """Match already-executed candidates whose offsets are exact opposites."""

    import numpy as np

    output = []
    steps = sorted(set(int(item["state_step"]) for item in records))
    for step in steps:
        selected = [
            item for item in records
            if int(item["state_step"]) == step
            and any(
                str(item["candidate_source"]).startswith(prefix)
                for prefix in allowed_source_prefixes
            )
        ]
        used = set()
        for left_index, left in enumerate(selected):
            if int(left["record_index"]) in used:
                continue
            nominal = np.asarray(left["nominal_xyz"], dtype=np.float64)
            left_xyz = np.asarray(left["candidate_xyz"], dtype=np.float64)
            left_offset = left_xyz - nominal
            radius = float(np.linalg.norm(left_offset))
            if radius <= matching_tolerance or radius > maximum_radius_action:
                continue
            matches = []
            for right in selected[left_index + 1:]:
                if right["split"] != left["split"]:
                    continue
                right_nominal = np.asarray(right["nominal_xyz"], dtype=np.float64)
                right_offset = np.asarray(right["candidate_xyz"], dtype=np.float64) - right_nominal
                if (
                    np.max(np.abs(right_nominal - nominal)) <= matching_tolerance
                    and np.max(np.abs(right_offset + left_offset)) <= matching_tolerance
                ):
                    matches.append(right)
            if len(matches) != 1:
                continue
            right = matches[0]
            used.add(int(left["record_index"]))
            used.add(int(right["record_index"]))
            left_count = int(left["raw_protected_contact_count"])
            right_count = int(right["raw_protected_contact_count"])
            if left_count == right_count:
                better = worse = None
            elif left_count < right_count:
                better, worse = left, right
            else:
                better, worse = right, left
            output.append({
                "state_step": step,
                "split": str(left["split"]),
                "radius_action": radius,
                "left_record_index": int(left["record_index"]),
                "right_record_index": int(right["record_index"]),
                "left_contact_count": left_count,
                "right_contact_count": right_count,
                "informative": better is not None,
                "better_record_index": None if better is None else int(better["record_index"]),
                "worse_record_index": None if worse is None else int(worse["record_index"]),
            })
    return output


def burden_key(burden: Mapping[str, Any]) -> tuple[Any, ...]:
    """Lower keys are safer under the registered exact simulator ordering."""

    return (
        0 if bool(burden["raw_safe"]) else 1,
        int(burden["contact_count"]),
        float(burden["summed_penetration_m"]),
    )


def empirical_burden_p(
    learned: Mapping[str, Any], random_burdens: Sequence[Mapping[str, Any]]
) -> float:
    """Add-one fraction of random directions at least as good as learned."""

    learned_key = burden_key(learned)
    count = sum(burden_key(item) <= learned_key for item in random_burdens)
    return (1.0 + count) / (1.0 + len(random_burdens))

