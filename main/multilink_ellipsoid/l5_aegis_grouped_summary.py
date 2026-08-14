"""Pure support accounting for grouped AEGIS-consistent L5 risk artifacts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


ROW_COUNT = 7
LEARNED_L5_ROWS = (0, 1, 2)


def training_eligible_state(classification: str) -> bool:
    """Proxy-invalid geometry cannot supervise the learned ellipsoid risk."""
    return str(classification) != "proxy_invalid"


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


def training_readiness(
    *, aggregate_rows: list[Mapping[str, int]],
    useful_boundary_states: Mapping[str, list[int]],
    known_candidates: Mapping[str, int], thresholds: Mapping[str, int],
    learned_rows: tuple[int, ...] = LEARNED_L5_ROWS,
) -> tuple[dict[str, bool], list[dict[str, Any]], bool]:
    sample_gates = {
        "train": int(known_candidates.get("train", 0))
        >= int(thresholds["minimum_train_samples"]),
        "validation": int(known_candidates.get("validation", 0))
        >= int(thresholds["minimum_validation_samples"]),
    }
    row_gates = []
    for row in learned_rows:
        aggregate = aggregate_rows[row]
        gates = {
            "known_safe": int(aggregate["known_safe_candidate_count"])
            >= int(thresholds["minimum_known_safe_candidates_per_row"]),
            "known_unsafe": int(aggregate["known_unsafe_candidate_count"])
            >= int(thresholds["minimum_known_unsafe_candidates_per_row"]),
            "near_boundary": int(aggregate["near_boundary_known_candidate_count"])
            >= int(thresholds["minimum_near_boundary_candidates_per_row"]),
            "train_boundary_states": int(
                useful_boundary_states.get("train", [0] * ROW_COUNT)[row]
            ) >= int(thresholds["minimum_train_useful_boundary_states_per_row"]),
            "validation_boundary_states": int(
                useful_boundary_states.get("validation", [0] * ROW_COUNT)[row]
            ) >= int(thresholds[
                "minimum_validation_useful_boundary_states_per_row"
            ]),
        }
        row_gates.append({"row": row, "gates": gates, "passes": all(gates.values())})
    authorized = bool(
        all(sample_gates.values()) and all(record["passes"] for record in row_gates)
    )
    return sample_gates, row_gates, authorized
