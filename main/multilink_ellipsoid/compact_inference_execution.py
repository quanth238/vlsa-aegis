"""Contracts for paired execution of the frozen compact inference selector."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

CONFIG_SCHEMA = "vlsa_distal_compact_inference_execution.v1"
RESULT_SCHEMA = "vlsa_distal_compact_inference_execution_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_compact_inference_execution_validation.v1"


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
        raise ValueError("compact inference execution config schema differs")
    if value.get("protocol_id") != "vlsa-distal-compact-inference-execution-v1":
        raise ValueError("compact inference execution protocol differs")
    if value.get("selection_rule") != "minimum_predicted_primary_risk":
        raise ValueError("compact inference execution selection rule differs")
    if value.get("physical_outcome_groups") != ["palm", "L5", "L6", "L7"]:
        raise ValueError("compact inference execution physical groups differ")
    execution = value.get("candidate_execution", {})
    if execution != {
        "candidate_names": ["nominal", "frozen_7D_minimum_risk_selection"],
        "candidate_count_per_state": 2,
        "released_AEGIS_EE_QP_enabled": False,
        "learned_QP_enabled": False,
        "exact_rollout_verifier_at_inference": False,
        "outcome_used_to_change_selection": False,
        "unchanged_OSC": True,
        "complete_fixed_continuation": True,
        "timeout_rule": "UNKNOWN_not_safe_or_unsafe",
    }:
        raise ValueError("compact inference execution action contract differs")
    cases = value.get("cases", [])
    if [case.get("case_id") for case in cases] != [
        "vlsa-t1-goal-ii-t0-e39",
        "vlsa-t1-long-ii-t3-e46",
        "vlsa-t1-spatial-i-t3-e15",
        "vlsa-t1-spatial-i-t3-e42",
    ]:
        raise ValueError("compact inference execution cases differ")
    if any(
        not isinstance(case.get("source_case_index"), int)
        or not isinstance(case.get("selected_candidate"), str)
        or not math.isfinite(float(case.get("selected_predicted_primary")))
        for case in cases
    ):
        raise ValueError("compact inference execution case binding differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def case_definition(config: Mapping[str, Any], case_index: int) -> dict[str, Any]:
    cases = list(config["cases"])
    if not 0 <= int(case_index) < len(cases):
        raise ValueError("compact inference execution case index differs")
    return dict(cases[int(case_index)])


def frozen_model_selection(
    compact_result: Mapping[str, Any], case: Mapping[str, Any], rule: str,
) -> dict[str, Any]:
    """Extract only the causal frozen selection fields from the fit artifact."""
    states = compact_result["compact_shared_7D"]["selectors"][
        str(case["split"])
    ][str(rule)]["states"]
    matches = [row for row in states if row["state_id"] == case["case_id"]]
    if len(matches) != 1:
        raise ValueError("compact inference execution model state differs")
    row = matches[0]
    if (
        row["selected_candidate"] != case["selected_candidate"]
        or not math.isclose(
            float(row["selected_predicted_primary"]),
            float(case["selected_predicted_primary"]),
            rel_tol=0.0, abs_tol=1.0e-12,
        )
    ):
        raise ValueError("compact inference execution frozen selection differs")
    return {
        "state_id": str(row["state_id"]),
        "split": str(case["split"]),
        "rule": str(rule),
        "selected_candidate": str(row["selected_candidate"]),
        "selected_predicted_primary": float(row["selected_predicted_primary"]),
    }


def summarize_pair(
    fresh_case: Mapping[str, Any], selected_candidate: str,
    physical_groups: Sequence[str],
) -> dict[str, Any]:
    exact = fresh_case.get("exact_case") or {}
    candidates = exact.get("candidates", [])
    if [row.get("name") for row in candidates] != [
        "nominal", str(selected_candidate),
    ]:
        raise ValueError("compact inference execution fresh pair differs")
    summaries = []
    for candidate in candidates:
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
        terminal_status = str(candidate["source_terminal_status"])
        physical_safe = bool(
            known
            and terminal_status == "SAFE_TERMINAL"
            and all(risks[group] <= 0.0 for group in physical_groups)
            and all(contacts[group] == 0 for group in physical_groups)
            and int(candidate["source_raw_protected_contact_count"]) == 0
            and int(candidate["raw_protected_contact_sample_count"]) == 0
            and not bool(candidate["source_physical_veto"])
            and not bool(candidate["replayed_physical_veto"])
        )
        summary = {
            "candidate_name": str(candidate["name"]),
            "known_outcome": known,
            "source_terminal_status": terminal_status,
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
            "replayed_maximum_CAR_m": float(
                candidate["replayed_maximum_CAR_m"]
            ),
            "effective_correction_l2_action": float(
                candidate["source_effective_post_AEGIS_correction_l2_action"]
            ),
            "executed_actions": candidate["source_executed_actions"],
            "diagnostic_EE_future_violation": float(
                risks["end_effector"]
            ),
        }
        summaries.append(summary)
    nominal, selected = summaries
    return {
        "source_snapshot_sha256": str(exact["source_snapshot_sha256"]),
        "replayed_snapshot_sha256": str(exact["replayed_snapshot_sha256"]),
        "state_hash_matches": bool(exact["state_hash_matches"]),
        "source_replay_exact": bool(exact["source_replay_exact"]),
        "initial_group_normalized_radial_slack": dict(
            exact["exact_group_target"]["initial_group_normalized_radial_slack"]
        ),
        "nominal": nominal,
        "selected": selected,
        "nominal_physical_safe": bool(nominal["physical_safe"]),
        "selected_physical_safe": bool(selected["physical_safe"]),
        "selected_outcome_known": bool(selected["known_outcome"]),
        "collision_avoided": bool(
            not nominal["physical_safe"] and selected["physical_safe"]
        ),
        "collision_persisted": bool(
            not nominal["physical_safe"] and not selected["physical_safe"]
        ),
    }


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": result["case_id"],
        "split": result["split"],
        "stratum": result["stratum"],
        "model_selection": result["model_selection"],
        "pair": result["pair"],
        "inference_used_future_outcome": result["inference_used_future_outcome"],
        "released_AEGIS_EE_QP_enabled": result[
            "released_AEGIS_EE_QP_enabled"
        ],
        "learned_QP_enabled": result["learned_QP_enabled"],
    }
