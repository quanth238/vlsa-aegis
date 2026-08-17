"""Scientific equality view for sequential and process-isolated candidates."""

from __future__ import annotations

from typing import Any, Mapping


RESULT_SCHEMA = "vlsa_distal_candidate_worker_equivalence.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scientific_view(
    case: Mapping[str, Any], source_curve: Mapping[str, Any]
) -> dict[str, Any]:
    _require(case["status"] == "complete", "candidate case is incomplete")
    exact_by_name = {
        row["name"]: row for row in case["exact_case"]["candidates"]
    }
    rows = []
    for source in source_curve["candidates"]:
        name = source["name"]
        _require(name in exact_by_name, "exact candidate name differs")
        exact = exact_by_name[name]
        target = exact["exact_group_target"]
        rows.append({
            "name": name,
            "requested_action_chunk": source[
                "post_aegis_candidate_before_consistency"
            ],
            "proposed_action_chunk": source["proposed_actions"],
            "effective_action_chunk": source["actions"],
            "source_snapshot_sha256": source["source_snapshot_sha256"],
            "source_restore_maximum_error": source[
                "source_restore_maximum_error"
            ],
            "terminal_state_sha256": source["terminal_state_sha256"],
            "terminal_status": source["terminal_status"],
            "terminal_reason": source["terminal_reason"],
            "prefix_CAR": source["prefix"][
                "maximum_active_obstacle_l1_displacement_m"
            ],
            "backup_CAR": source["backup"][
                "maximum_active_obstacle_l1_displacement_m"
            ],
            "physical_veto": source["physical_veto"],
            "known_outcome": target["known_outcome"],
            "per_constraint_future_risk": target["group_future_violation"],
            "per_constraint_contacts": target["group_contact_events"],
            "raw_protected_contacts": exact["raw_protected_contacts"],
        })
    return {
        "case_id": case["case_id"],
        "state_step": case["state_step"],
        "target_group": case["selection"]["target_group"],
        "candidate_order": [row["name"] for row in rows],
        "initial_source_snapshot_sha256": source_curve["state"][
            "source_snapshot_sha256"
        ],
        "determinism_replay": source_curve["determinism_replay"],
        "exact_replayed_snapshot_sha256": case["exact_case"][
            "replayed_snapshot_sha256"
        ],
        "exact_source_replay": case["exact_case"]["source_replay_exact"],
        "candidates": rows,
    }


def compare(
    sequential_case: Mapping[str, Any],
    sequential_source: Mapping[str, Any],
    parallel_case: Mapping[str, Any],
    parallel_source: Mapping[str, Any],
) -> dict[str, Any]:
    sequential = scientific_view(sequential_case, sequential_source)
    parallel = scientific_view(parallel_case, parallel_source)
    parallel_receipt = parallel_source.get("parallel_candidate_execution")
    checks = {
        "parallel_receipt_present": bool(
            isinstance(parallel_receipt, dict)
            and parallel_receipt.get("enabled") is True
            and parallel_receipt.get("worker_count") == 4
        ),
        "candidate_order_equal": (
            sequential["candidate_order"] == parallel["candidate_order"]
        ),
        "requested_and_effective_actions_equal": all(
            left[key] == right[key]
            for left, right in zip(
                sequential["candidates"], parallel["candidates"]
            )
            for key in (
                "requested_action_chunk", "proposed_action_chunk",
                "effective_action_chunk",
            )
        ),
        "restored_state_hashes_equal": (
            sequential["initial_source_snapshot_sha256"]
            == parallel["initial_source_snapshot_sha256"]
            and sequential["exact_replayed_snapshot_sha256"]
            == parallel["exact_replayed_snapshot_sha256"]
            and all(
                left["source_snapshot_sha256"]
                == right["source_snapshot_sha256"]
                for left, right in zip(
                    sequential["candidates"], parallel["candidates"]
                )
            )
        ),
        "per_constraint_risks_equal": all(
            left["per_constraint_future_risk"]
            == right["per_constraint_future_risk"]
            for left, right in zip(
                sequential["candidates"], parallel["candidates"]
            )
        ),
        "contacts_and_CAR_equal": all(
            left[key] == right[key]
            for left, right in zip(
                sequential["candidates"], parallel["candidates"]
            )
            for key in (
                "prefix_CAR", "backup_CAR", "physical_veto",
                "per_constraint_contacts", "raw_protected_contacts",
            )
        ),
        "timeouts_equal": all(
            left["terminal_status"] == right["terminal_status"]
            and left["terminal_reason"] == right["terminal_reason"]
            and left["known_outcome"] == right["known_outcome"]
            for left, right in zip(
                sequential["candidates"], parallel["candidates"]
            )
        ),
        "terminal_state_hashes_equal": all(
            left["terminal_state_sha256"] == right["terminal_state_sha256"]
            for left, right in zip(
                sequential["candidates"], parallel["candidates"]
            )
        ),
    }
    return {
        "checks": checks,
        "strict_equivalence_pass": all(checks.values()),
        "sequential_scientific_view": sequential,
        "parallel_scientific_view": parallel,
        "sequential_wall_seconds": sequential_source["wall_seconds"],
        "parallel_wall_seconds": parallel_source["wall_seconds"],
        "observed_speedup": (
            float(sequential_source["wall_seconds"])
            / float(parallel_source["wall_seconds"])
        ),
    }
