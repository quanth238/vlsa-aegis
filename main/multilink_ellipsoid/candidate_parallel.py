"""Merge independently simulated candidate shards without changing labels."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


_CANDIDATE_DEPENDENT_KEYS = {
    "candidate_count",
    "candidates",
    "summary",
    "gates",
    "interpretation",
    "wall_seconds",
    "result_payload_sha256",
    "parallel_candidate_execution",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _common_view(shard: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in shard.items()
        if key not in _CANDIDATE_DEPENDENT_KEYS
    }


def merge_candidate_curve_shards(
    shards: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
    *,
    worker_count: int,
    wall_seconds: float,
) -> dict[str, Any]:
    """Merge nominal-plus-one shards into canonical frozen-bank order."""

    names = list(candidate_names)
    _require(names and names[0] == "nominal", "parallel bank nominal differs")
    _require(len(set(names)) == len(names), "parallel bank names are not unique")
    _require(len(shards) == len(names), "parallel shard count differs")
    baseline = _common_view(shards[0])
    nominal_record = None
    by_name: dict[str, dict[str, Any]] = {}
    shard_receipts = []
    for expected_name, shard in zip(names, shards):
        _require(_common_view(shard) == baseline, "parallel shard context differs")
        rows = list(shard["candidates"])
        row_by_name = {row["name"]: row for row in rows}
        expected = {"nominal"} if expected_name == "nominal" else {
            "nominal", expected_name,
        }
        _require(set(row_by_name) == expected, "parallel shard candidates differ")
        if nominal_record is None:
            nominal_record = copy.deepcopy(row_by_name["nominal"])
        else:
            _require(
                row_by_name["nominal"] == nominal_record,
                "parallel nominal candidate differs",
            )
        by_name[expected_name] = copy.deepcopy(row_by_name[expected_name])
        shard_receipts.append({
            "requested_candidate": expected_name,
            "candidate_names": [row["name"] for row in rows],
            "source_snapshot_sha256": row_by_name["nominal"][
                "source_snapshot_sha256"
            ],
        })

    records = [by_name[name] for name in names]
    result = copy.deepcopy(dict(shards[0]))
    result["candidate_count"] = len(records)
    result["candidates"] = records
    safe_count = sum(bool(row["exact_safe"]) for row in records)
    unsafe_count = sum(
        row["terminal_status"] == "UNSAFE_CONTACT_OR_CAR"
        or max(row["combined_risk"]) > 0.0
        for row in records
    )
    timeout_count = sum(
        row["terminal_status"] == "UNKNOWN_TIMEOUT" for row in records
    )
    contradiction_count = sum(
        max(row["combined_risk"]) <= 0.0 and row["physical_veto"]
        for row in records
    )
    statuses = result["base_method_config"]["risk_target"]["terminal_statuses"]
    result["summary"] = {
        "safe_candidate_count": safe_count,
        "unsafe_candidate_count": unsafe_count,
        "unknown_timeout_count": timeout_count,
        "proxy_safe_physical_collision_count": contradiction_count,
        "terminal_status_counts": {
            status: sum(row["terminal_status"] == status for row in records)
            for status in statuses
        },
        "row_active_witness_counts": [
            sum(
                min(
                    range(len(row["combined_row_minimum_clearance_m"])),
                    key=lambda index: row["combined_row_minimum_clearance_m"][index],
                ) == group_index
                for row in records
            )
            for group_index in range(7)
        ],
    }
    gates = copy.deepcopy(shards[0]["gates"])
    gates.update({
        "exact_snapshot_replay": all(
            shard["gates"]["exact_snapshot_replay"] for shard in shards
        ) and all(row["source_restore_maximum_error"] == 0.0 for row in records),
        "zero_L5_residual_reproduces_recomputed_released_aegis": all(
            shard["gates"][
                "zero_L5_residual_reproduces_recomputed_released_aegis"
            ]
            for shard in shards
        ),
        "query_boundary_is_initially_safe": all(
            shard["gates"]["query_boundary_is_initially_safe"]
            for shard in shards
        ),
        "nominal_future_is_unsafe": not bool(records[0]["exact_safe"]),
        "safe_and_unsafe_candidate_support": safe_count > 0 and unsafe_count > 0,
        "at_least_one_safe_terminal": safe_count > 0,
        "no_timeout_labeled_safe": all(
            not row["exact_safe"]
            for row in records
            if row["terminal_status"] == "UNKNOWN_TIMEOUT"
        ),
        "proxy_safe_physical_collision_count_zero": contradiction_count == 0,
        "all_terminal_statuses_registered": all(
            row["terminal_status"] in statuses for row in records
        ),
        "all_replays_boundary_exact": all(
            shard["gates"]["all_replays_boundary_exact"] for shard in shards
        ),
    })
    result["gates"] = gates
    result["interpretation"] = (
        "grouped_query_action_risk_state_pass"
        if all(gates.values())
        else "grouped_query_action_risk_state_no_go"
    )
    result["parallel_candidate_execution"] = {
        "enabled": True,
        "worker_count": int(worker_count),
        "isolation": "one_external_process_and_MuJoCo_environment_per_shard",
        "queue": "dynamic_external_process_queue",
        "canonical_candidate_order": names,
        "shards": shard_receipts,
    }
    result["wall_seconds"] = float(wall_seconds)
    result.pop("result_payload_sha256", None)
    return result
