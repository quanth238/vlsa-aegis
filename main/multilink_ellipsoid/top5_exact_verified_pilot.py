"""Contracts for the frozen-MLP top-five exact-verification pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_top5_exact_verified_pilot.v1"
RESULT_SCHEMA = "vlsa_distal_top5_exact_verified_pilot_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_top5_exact_verified_pilot_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("top-five pilot config schema differs")
    if value.get("protocol_id") != "vlsa-distal-top5-exact-verified-pilot-v1":
        raise ValueError("top-five pilot protocol differs")
    if int(value.get("maximum_ranked_candidates", 0)) != 5:
        raise ValueError("top-five pilot rank cap differs")
    if value.get("physical_acceptance_groups") != ["palm", "L5", "L6", "L7"]:
        raise ValueError("top-five pilot physical groups differ")
    cases = value.get("cases", [])
    if [case.get("case_id") for case in cases] != [
        "vlsa-t1-goal-ii-t0-e39",
        "vlsa-t1-long-ii-t3-e46",
        "vlsa-t1-spatial-i-t3-e42",
    ]:
        raise ValueError("top-five pilot cases differ")
    if any(len(case.get("ranked_candidate_names", [])) != 5 for case in cases):
        raise ValueError("top-five pilot ranked prefix differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def candidate_summary(
    candidate: Mapping[str, Any], physical_groups: Sequence[str],
) -> dict[str, Any]:
    target = candidate["exact_group_target"]
    known = bool(target["known_outcome"])
    risks = {
        group: float(value)
        for group, value in target["group_future_violation"].items()
    }
    contacts = {
        group: int(value)
        for group, value in target["group_contact_sample_count"].items()
    }
    physical_safe = bool(
        known
        and bool(target["safe_terminal"])
        and all(risks[group] <= 0.0 for group in physical_groups)
        and all(contacts[group] == 0 for group in physical_groups)
        and int(candidate["source_raw_protected_contact_count"]) == 0
        and int(candidate["raw_protected_contact_sample_count"]) == 0
        and not bool(candidate["source_physical_veto"])
        and not bool(candidate["replayed_physical_veto"])
    )
    return {
        "candidate_name": str(candidate["name"]),
        "known_outcome": known,
        "safe_terminal": bool(target["safe_terminal"]),
        "physical_safe": physical_safe,
        "group_future_violation": risks,
        "group_contact_sample_count": contacts,
        "source_raw_protected_contact_count": int(
            candidate["source_raw_protected_contact_count"]
        ),
        "replayed_raw_protected_contact_sample_count": int(
            candidate["raw_protected_contact_sample_count"]
        ),
        "source_physical_veto": bool(candidate["source_physical_veto"]),
        "replayed_physical_veto": bool(candidate["replayed_physical_veto"]),
        "source_maximum_CAR_m": float(candidate["source_maximum_CAR_m"]),
        "replayed_maximum_CAR_m": float(candidate["replayed_maximum_CAR_m"]),
        "effective_correction_l2_action": float(
            candidate["source_effective_post_AEGIS_correction_l2_action"]
        ),
        "executed_actions": candidate["source_executed_actions"],
    }


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    attempts = [
        {
            key: value for key, value in attempt.items()
            if key not in {
                "fresh_result_file_sha256", "fresh_result_payload_sha256",
            }
        }
        for attempt in result["attempts"]
    ]
    return {
        "case_id": result["case_id"],
        "split": result["split"],
        "attempts": attempts,
        "selected_rank": result["selected_rank"],
        "selected_candidate": result["selected_candidate"],
        "selected_action_chunk": result["selected_action_chunk"],
        "top_one_freshly_unsafe": result["top_one_freshly_unsafe"],
        "verified_safe_selection": result["verified_safe_selection"],
        "method_avoids_top_one_collision": result[
            "method_avoids_top_one_collision"
        ],
    }
