"""Pure support accounting for grouped AEGIS-consistent L5 risk artifacts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


ROW_COUNT = 7
LEARNED_L5_ROWS = (0, 1, 2)


def candidate_outcome(candidate: Mapping[str, Any]) -> str:
    status = candidate["terminal_status"]
    if status == "UNKNOWN_TIMEOUT":
        return "unknown"
    if bool(candidate["exact_safe"]):
        return "safe"
    return "unsafe"


def state_classification(candidates: Iterable[Mapping[str, Any]]) -> str:
    values = list(candidates)
    proxy_invalid = any(
        max(float(v) for v in candidate["combined_risk"]) <= 0.0
        and bool(candidate["physical_veto"])
        for candidate in values
        if candidate_outcome(candidate) != "unknown"
    )
    if proxy_invalid:
        return "proxy_invalid"
    outcomes = {candidate_outcome(candidate) for candidate in values}
    if "safe" in outcomes and "unsafe" in outcomes:
        return "usable_mixed_support"
    if "unsafe" in outcomes and "safe" not in outcomes:
        return "no_safe_candidate"
    if "safe" in outcomes and "unsafe" not in outcomes:
        return "no_known_unsafe_candidate"
    return "unknown_only"


def row_coverage(
    candidates: Iterable[Mapping[str, Any]], near_boundary_m: float = 0.005,
) -> list[dict[str, int]]:
    output = [{
        "known_safe_candidate_count": 0,
        "known_unsafe_candidate_count": 0,
        "near_boundary_known_candidate_count": 0,
        "active_witness_known_candidate_count": 0,
        "unknown_timeout_candidate_count": 0,
    } for _ in range(ROW_COUNT)]
    for candidate in candidates:
        risk = [float(v) for v in candidate["combined_risk"]]
        if len(risk) != ROW_COUNT:
            raise ValueError("candidate risk row count differs")
        outcome = candidate_outcome(candidate)
        if outcome == "unknown":
            for record in output:
                record["unknown_timeout_candidate_count"] += 1
            continue
        active = max(range(ROW_COUNT), key=risk.__getitem__)
        output[active]["active_witness_known_candidate_count"] += 1
        for row, value in enumerate(risk):
            key = (
                "known_safe_candidate_count"
                if value <= 0.0 else "known_unsafe_candidate_count"
            )
            output[row][key] += 1
            if abs(value) <= float(near_boundary_m):
                output[row]["near_boundary_known_candidate_count"] += 1
    return output
