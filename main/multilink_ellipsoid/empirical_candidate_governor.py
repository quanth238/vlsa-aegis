"""Contracts for the four-state empirical-geometry candidate governor oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_empirical_candidate_governor.v1"
RESULT_SCHEMA = "vlsa_distal_empirical_candidate_governor_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version",
        "protocol_id",
        "claim_scope",
        "population_manifest",
        "population_manifest_file_sha256",
        "geometry_config",
        "geometry_config_file_sha256",
        "empirical_l6_proxy_config",
        "empirical_l6_proxy_config_file_sha256",
        "cases",
        "gate",
        "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("empirical candidate-governor config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("empirical candidate-governor schema differs")
    if value["protocol_id"] != "vlsa-distal-empirical-candidate-governor-v1":
        raise ValueError("empirical candidate-governor protocol differs")
    cases = value["cases"]
    expected = [
        ("vlsa-t1-goal-ii-t2-e42", 25),
        ("vlsa-t1-goal-ii-t3-e42", 100),
        ("vlsa-t1-goal-ii-t3-e44", 110),
        ("vlsa-t1-goal-ii-t3-e42", 105),
    ]
    if [(item["case_id"], item["state_step"]) for item in cases] != expected:
        raise ValueError("empirical candidate-governor states differ")
    expected_names = ["nominal"] + [
        "normal_alpha_%.2f" % (0.25 * index) for index in range(1, 9)
    ]
    if any(item["candidate_names"] != expected_names for item in cases):
        raise ValueError("empirical candidate-governor bank differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def classify(cases: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [candidate for case in cases for candidate in case["candidates"]]
    contact_controls = [
        candidate for candidate in candidates
        if int(candidate["source_raw_protected_contact_count"]) > 0
    ]
    stable_controls = [
        candidate for candidate in candidates
        if candidate["source_terminal_status"] == "SAFE_TERMINAL"
        and not bool(candidate["source_physical_veto"])
    ]
    timeouts = [
        candidate for candidate in candidates
        if candidate["source_terminal_status"] == "UNKNOWN_TIMEOUT"
    ]
    selections = []
    for case in cases:
        safe = [
            candidate for candidate in case["candidates"]
            if candidate["compiled_box_safe_terminal"]
        ]
        selected = None if not safe else min(
            safe,
            key=lambda item: (
                float(item["source_effective_post_AEGIS_correction_l2_action"]),
                float(item["requested_alpha"]),
            ),
        )
        strongest = [
            candidate for candidate in case["candidates"]
            if candidate["name"] == "normal_alpha_2.00"
        ][0]
        selections.append(
            {
                "case_id": case["case_id"],
                "state_step": int(case["state_step"]),
                "selected": selected,
                "strongest": strongest,
                "strictly_smaller_than_strongest": bool(
                    selected is not None
                    and float(selected["source_effective_post_AEGIS_correction_l2_action"])
                    < float(strongest["source_effective_post_AEGIS_correction_l2_action"])
                    - 1.0e-12
                ),
            }
        )
    gates = {
        "case_count": len(cases) == int(gate["require_case_count"]),
        "candidate_count": len(candidates) == int(gate["require_candidate_count"]),
        "source_replay_exact": all(bool(case["source_replay_exact"]) for case in cases),
        "initial_states_safe": all(
            not bool(case["initial_compiled_box_any_exact_overlap"])
            and int(case["initial_raw_protected_contact_count"]) == 0
            for case in cases
        ),
        "all_contact_controls_detected": all(
            candidate["compiled_box_any_exact_overlap"]
            for candidate in contact_controls
        ),
        "all_stable_controls_accepted": all(
            candidate["compiled_box_safe_terminal"] for candidate in stable_controls
        ),
        "safe_support_every_state": all(item["selected"] is not None for item in selections),
        "strongest_control_safe_every_state": all(
            item["strongest"]["compiled_box_safe_terminal"] for item in selections
        ),
        "minimum_intervention_strictly_smaller_every_state": all(
            item["strictly_smaller_than_strongest"] for item in selections
        ),
        "timeouts_remain_unknown": all(
            not candidate["compiled_box_safe_terminal"] for candidate in timeouts
        ),
    }
    reductions = [
        float(item["strongest"]["source_effective_post_AEGIS_correction_l2_action"])
        - float(item["selected"]["source_effective_post_AEGIS_correction_l2_action"])
        for item in selections if item["selected"] is not None
    ]
    return {
        "gates": gates,
        "strict_gate_pass": bool(all(gates.values())),
        "contact_control_count": len(contact_controls),
        "stable_control_count": len(stable_controls),
        "timeout_count": len(timeouts),
        "selections": selections,
        "mean_effective_intervention_reduction": (
            None if not reductions else sum(reductions) / len(reductions)
        ),
    }
