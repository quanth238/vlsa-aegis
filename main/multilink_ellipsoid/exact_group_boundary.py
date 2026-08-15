"""Contracts for the palm/L5/L6 exact-geometry boundary canary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_exact_group_boundary_canary.v1"
NO_QP_L5_CONFIG_SCHEMA = "vlsa_distal_no_qp_l5_boundary_canary.v1"
CASE_SCHEMA = "vlsa_distal_exact_group_boundary_case_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_exact_group_boundary_validation.v1"
GROUPS = ("palm", "L5", "L6")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str = "result_payload_sha256") -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    schema = value.get("schema_version")
    if schema not in (CONFIG_SCHEMA, NO_QP_L5_CONFIG_SCHEMA):
        raise ValueError("exact-group boundary config schema differs")
    expected_protocol = (
        "vlsa-distal-exact-group-boundary-canary-v1"
        if schema == CONFIG_SCHEMA
        else "vlsa-distal-no-qp-l5-boundary-canary-v1"
    )
    if value.get("protocol_id") != expected_protocol:
        raise ValueError("exact-group boundary protocol differs")
    bank = value["candidate_bank"]
    if [float(item) for item in bank["requested_alpha"]] != [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
    ]:
        raise ValueError("exact-group boundary alpha bank differs")
    expected_aegis = schema == CONFIG_SCHEMA
    if bank["released_AEGIS_EE_applied_to_every_candidate"] is not expected_aegis:
        raise ValueError("released AEGIS EE-QP execution mode differs")
    if value["learned_correction_QP_enabled"] is not False:
        raise ValueError("learned correction QP must remain disabled")
    if value["exact_group_target"]["group_order"] != list(GROUPS):
        raise ValueError("exact-group order differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def load_cases(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if len(cases) != int(config["gate"]["required_case_count"]):
        raise ValueError("exact-group boundary case count differs")
    expected_groups = list(config.get("target_group_sequence", GROUPS))
    if [case["target_group"] for case in cases] != expected_groups:
        raise ValueError("exact-group boundary target cases differ")
    if len({case["task_level_group_id"] for case in cases}) != len(cases):
        raise ValueError("exact-group boundary episode groups are not independent")
    return cases


def warning_step(case: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    lead = int(config["state_selection"]["lead_actions_before_first_target_contact"])
    step = int(math.floor((int(case["first_target_contact_step"]) - lead) / 5.0) * 5)
    if step < 0 or step % 5:
        raise ValueError("exact-group warning step differs")
    return step


def summarize_cases(cases: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    near = float(config["gate"]["near_boundary_abs_slack"])
    summaries = []
    proxy_false_safe = 0
    replay_pass = True
    source_state_hash_pass = True
    for case in cases:
        target = str(case["selection"]["target_group"])
        initial = case["exact_case"]["exact_group_target"]
        candidates = case["exact_case"]["candidates"]
        known = [
            row for row in candidates
            if bool(row["exact_group_target"]["known_outcome"])
        ]
        safe = [
            row for row in known
            if float(row["exact_group_target"]["group_future_violation"][target]) <= 0.0
            and int(row["exact_group_target"]["group_contact_sample_count"][target]) == 0
        ]
        unsafe = [row for row in known if row not in safe]
        boundary = [
            row for row in known
            if abs(float(row["exact_group_target"]["group_minimum_normalized_radial_slack"][target])) <= near
        ]
        false_safe = sum(
            int(row["exact_group_target"]["group_contact_sample_count"][group]) > 0
            and float(row["exact_group_target"]["group_minimum_normalized_radial_slack"][group]) > 0.0
            for row in known for group in GROUPS
        )
        proxy_false_safe += false_safe
        replay_pass = replay_pass and bool(case["exact_case"]["source_replay_exact"])
        source_state_hash_pass = source_state_hash_pass and bool(
            case["exact_case"]["state_hash_matches"]
        )
        summaries.append({
            "case_id": case["case_id"],
            "target_group": target,
            "candidate_count": len(candidates),
            "known_candidate_count": len(known),
            "unknown_timeout_count": len(candidates) - len(known),
            "target_safe_candidate_count": len(safe),
            "target_unsafe_candidate_count": len(unsafe),
            "target_near_boundary_candidate_count": len(boundary),
            "target_two_sided_support": bool(safe and unsafe),
            "initial_target_slack": float(initial["initial_group_normalized_radial_slack"][target]),
            "initial_target_contact_count": int(initial["initial_group_contact_sample_count"][target]),
            "robot_primitive_certificate_pass": bool(initial["robot_primitive_certificate_pass"]),
            "physical_false_safe_count": false_safe,
        })
    replay_gate = (
        source_state_hash_pass
        if config["gate"].get("legacy_source_proxy_replay") == "diagnostic_only"
        else replay_pass
    )
    apparatus = bool(
        len(cases) == int(config["gate"]["required_case_count"])
        and all(item["candidate_count"] == int(config["gate"]["required_candidates_per_case"])
                for item in summaries)
        and all(item["robot_primitive_certificate_pass"] for item in summaries)
        and all(item["initial_target_slack"] > 0.0 and item["initial_target_contact_count"] == 0
                for item in summaries)
        and replay_gate and proxy_false_safe == 0
    )
    same_bank = bool(apparatus and all(item["target_two_sided_support"] for item in summaries))
    return {
        "case_count": len(cases),
        "candidate_count": sum(item["candidate_count"] for item in summaries),
        "source_replay_exact": replay_pass,
        "source_state_hash_exact": source_state_hash_pass,
        "legacy_source_proxy_replay": config["gate"].get(
            "legacy_source_proxy_replay", "strict"
        ),
        "physical_false_safe_count": proxy_false_safe,
        "per_case": summaries,
        "apparatus_pass": apparatus,
        "same_bank_grouped_collection_authorized": same_bank,
        "training_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
