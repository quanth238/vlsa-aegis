"""Contracts for the L6 perception-MVEE versus compiled-box risk audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_compiled_box_risk_target_audit.v1"
RESULT_SCHEMA = "vlsa_distal_compiled_box_risk_target_audit_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "population_manifest",
        "population_manifest_file_sha256", "geometry_config",
        "geometry_config_file_sha256", "cases", "target", "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("compiled-box target-audit config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("compiled-box target-audit schema differs")
    if value["protocol_id"] != "vlsa-distal-compiled-box-risk-target-audit-v1":
        raise ValueError("compiled-box target-audit protocol differs")
    cases = value["cases"]
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("compiled-box target-audit case count differs")
    expected_ids = ["vlsa-t1-goal-ii-t3-e44", "vlsa-t1-goal-ii-t3-e42"]
    if [case["case_id"] for case in cases] != expected_ids:
        raise ValueError("compiled-box target-audit cases differ")
    if [case["state_step"] for case in cases] != [110, 105]:
        raise ValueError("compiled-box target-audit states differ")
    if value["target"] != {
        "compiled_obstacle_source": "mujoco_contact_capable_box_union",
        "positive_is_unsafe": True,
        "risk_definition": "negative_minimum_normalized_radial_slack",
        "risk_units": "dimensionless_not_metric_clearance",
        "robot_geometry": "unchanged_seven_certified_L5_L6_L7_slab_ellipsoids",
    }:
        raise ValueError("compiled-box target definition differs")
    if value["gate"].get("require_initial_states_safe") is not True:
        raise ValueError("compiled-box initial-state gate differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def candidate_action_sequence(
    candidate: Mapping[str, Any],
    *,
    rollout_scope: str = "complete_candidate_plus_continuation",
) -> tuple[list[Any], list[str]]:
    """Return the registered executed ledger for the requested rollout scope.

    The historical default remains byte-compatible.  The opt-in prefix-only
    scope is used by the finite-horizon critic dataset and never executes the
    registered backup or terminal hold.
    """

    actions = [list(row) for row in candidate["actions"]]
    phases = ["prefix"] * len(actions)
    if rollout_scope == "candidate_five_action_prefix_only":
        if len(actions) != 5:
            raise ValueError("compiled-box candidate prefix length differs")
        return actions, phases
    if rollout_scope != "complete_candidate_plus_continuation":
        raise ValueError("compiled-box candidate rollout scope differs")
    backup = candidate["backup"]
    for decision in backup["decisions"]:
        actions.append(list(decision["selected_action"]))
        phases.append("backup")
    terminal = backup.get("terminal_hold")
    if terminal is not None:
        hold_actions = terminal["executed_actions"]
        actions.extend(list(row) for row in hold_actions)
        phases.extend(["terminal_hold"] * len(hold_actions))
    if not actions or len(actions) != len(phases):
        raise ValueError("compiled-box candidate ledger differs")
    return actions, phases


def classify(cases: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [candidate for case in cases for candidate in case["candidates"]]
    contact_controls = [
        candidate for candidate in candidates
        if int(candidate["source_raw_protected_contact_count"]) > 0
    ]
    safe_controls = [
        candidate for candidate in candidates
        if candidate["source_terminal_status"] == "SAFE_TERMINAL"
        and not candidate["source_physical_veto"]
    ]
    timeout_controls = [
        candidate for candidate in candidates
        if candidate["source_terminal_status"] == "UNKNOWN_TIMEOUT"
    ]
    physical_false_safe = [
        candidate for candidate in contact_controls
        if not candidate["compiled_box_any_exact_overlap"]
    ]
    rejected_safe_rescued = [
        candidate for candidate in safe_controls
        if not candidate["source_proxy_nonoverlap"]
        and candidate["compiled_box_safe_terminal"]
    ]
    states_with_safe_support = {
        candidate["case_id"] for candidate in safe_controls
        if candidate["compiled_box_safe_terminal"]
    }
    gates = {
        "case_count": len(cases) == int(gate["require_case_count"]),
        "candidate_count": len(candidates) == int(gate["require_candidate_count"]),
        "initial_states_safe": all(
            not bool(case["initial_compiled_box_any_exact_overlap"])
            and int(case["initial_raw_protected_contact_count"]) == 0
            for case in cases
        ),
        "source_replay_exact": all(bool(case["source_replay_exact"]) for case in cases),
        "minimum_contact_controls": len(contact_controls) >= int(
            gate["minimum_contact_controls"]
        ),
        "minimum_safe_terminal_controls": len(safe_controls) >= int(
            gate["minimum_safe_terminal_controls"]
        ),
        "zero_compiled_box_physical_false_safe": len(physical_false_safe) == 0,
        "safe_support_in_every_state": len(states_with_safe_support) == len(cases),
        "minimum_proxy_rejected_safe_rescued": len(rejected_safe_rescued) >= int(
            gate["minimum_proxy_rejected_safe_rescued"]
        ),
        "timeouts_remain_unknown": all(
            not candidate["compiled_box_safe_terminal"] for candidate in timeout_controls
        ),
    }
    return {
        "gates": gates,
        "strict_gate_pass": bool(all(gates.values())),
        "candidate_count": len(candidates),
        "contact_control_count": len(contact_controls),
        "safe_terminal_control_count": len(safe_controls),
        "timeout_control_count": len(timeout_controls),
        "compiled_box_physical_false_safe_count": len(physical_false_safe),
        "proxy_rejected_safe_rescued_count": len(rejected_safe_rescued),
        "states_with_compiled_box_safe_support": sorted(states_with_safe_support),
    }
