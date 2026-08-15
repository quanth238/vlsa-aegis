"""Contracts for grouped empirical candidate-risk data collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_empirical_candidate_risk_dataset.v1"
CASE_RESULT_SCHEMA = "vlsa_distal_empirical_candidate_risk_case.v1"
SUMMARY_SCHEMA = "vlsa_distal_empirical_candidate_risk_summary.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "population_manifest",
        "population_manifest_file_sha256", "selection_manifest",
        "selection_manifest_file_sha256", "included_splits", "episode_counts",
        "state_selection", "candidate_bank", "base_risk_config",
        "base_risk_config_file_sha256", "geometry_config",
        "geometry_config_file_sha256", "empirical_l6_proxy_config",
        "empirical_l6_proxy_config_file_sha256", "risk_target", "gate",
        "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("empirical candidate-risk config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("empirical candidate-risk config schema differs")
    if value["protocol_id"] != "vlsa-distal-empirical-candidate-risk-dataset-v1":
        raise ValueError("empirical candidate-risk protocol differs")
    if value["included_splits"] != ["train", "validation"]:
        raise ValueError("empirical candidate-risk splits differ")
    if value["episode_counts"] != {"train": 10, "validation": 3}:
        raise ValueError("empirical candidate-risk episode counts differ")
    if value["state_selection"] != {
        "unit": "real_five_action_VLA_query_boundary",
        "lead_actions_before_first_protected_contact": 3,
        "formula": "floor((first_relevant_contact_step-3)/5)*5",
        "include_k0": True,
        "minimum_initial_empirical_slack": 0.0,
    }:
        raise ValueError("empirical candidate-risk state selection differs")
    if value["candidate_bank"] != {
        "requested_alpha": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        "direction": "current_closest_slab_outward_normal",
        "temporal_profile": "front_loaded_unit_L2_5_4_3_2_1",
        "action_limit": 1.0,
        "released_AEGIS_EE_applied_to_every_candidate": True,
        "selection_cost": "realized_post_AEGIS_five_action_L2",
    }:
        raise ValueError("empirical candidate-risk bank differs")
    if value["risk_target"] != {
        "primary": "negative_minimum_empirical_normalized_radial_slack_over_prefix_plus_fixed_backup",
        "positive_is_unsafe": True,
        "physical_authority": "raw_MuJoCo_L5_L6_L7_contact_and_paper_CAR",
        "unknown_timeout": "censored_not_safe_not_unsafe",
        "output": "one_scalar_per_complete_candidate",
        "metric_clearance_claim": False,
    }:
        raise ValueError("empirical candidate-risk target differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = sha256(raw)
    output["config_payload_sha256"] = sha256(canonical(value))
    return output


def load_cases(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    if sha256(raw) != config["selection_manifest_file_sha256"]:
        raise ValueError("empirical candidate-risk manifest differs")
    all_rows = [
        json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()
    ]
    cases = [row for row in all_rows if row.get("split") in config["included_splits"]]
    counts = {
        split: sum(row.get("split") == split for row in cases)
        for split in config["included_splits"]
    }
    if counts != config["episode_counts"] or len(cases) != 13:
        raise ValueError("empirical candidate-risk case population differs")
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("empirical candidate-risk case repeats")
    return cases


def warning_step(case: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    contact = int(case["first_relevant_contact_step"])
    lead = int(config["state_selection"]["lead_actions_before_first_protected_contact"])
    step = ((contact - lead) // 5) * 5
    if step < 5 or step >= contact or step % 5:
        raise ValueError("empirical candidate-risk warning step differs")
    return step


def classify(cases: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    by_split: dict[str, list[Mapping[str, Any]]] = {"train": [], "validation": []}
    all_candidates = []
    per_case = []
    for item in cases:
        split = str(item["split"])
        empirical = item["empirical_case"]
        candidates = empirical["candidates"]
        by_split[split].append(item)
        all_candidates.extend(candidates)
        safe = [row for row in candidates if bool(row["compiled_box_safe_terminal"])]
        known_unsafe = [
            row for row in candidates
            if row["source_terminal_status"] != "UNKNOWN_TIMEOUT"
            and not bool(row["compiled_box_safe_terminal"])
        ]
        per_case.append({
            "case_id": item["case_id"],
            "split": split,
            "state_step": int(item["state_step"]),
            "safe_candidate_count": len(safe),
            "known_unsafe_candidate_count": len(known_unsafe),
            "timeout_count": sum(
                row["source_terminal_status"] == "UNKNOWN_TIMEOUT"
                for row in candidates
            ),
            "two_sided_support": bool(safe and known_unsafe),
            "minimum_safe": None if not safe else min(
                safe,
                key=lambda row: (
                    float(row["source_effective_post_AEGIS_correction_l2_action"]),
                    float(row["requested_alpha"]),
                ),
            )["name"],
        })
    contact_controls = [
        row for row in all_candidates
        if int(row["source_raw_protected_contact_count"]) > 0
    ]
    stable_controls = [
        row for row in all_candidates
        if row["source_terminal_status"] == "SAFE_TERMINAL"
        and not bool(row["source_physical_veto"])
    ]
    timeouts = [
        row for row in all_candidates
        if row["source_terminal_status"] == "UNKNOWN_TIMEOUT"
    ]
    gate = config["gate"]
    gates = {
        "case_count": len(cases) == int(gate["require_case_count"]),
        "candidate_count": len(all_candidates) == int(gate["require_candidate_count"]),
        "split_counts": {
            split: len(rows) for split, rows in by_split.items()
        } == config["episode_counts"],
        "source_replay_exact": all(
            bool(item["empirical_case"]["source_replay_exact"]) for item in cases
        ),
        "initial_states_empirically_safe": all(
            not bool(item["empirical_case"]["initial_compiled_box_any_exact_overlap"])
            and int(item["empirical_case"]["initial_raw_protected_contact_count"]) == 0
            for item in cases
        ),
        "all_raw_contact_controls_detected": all(
            bool(row["compiled_box_any_exact_overlap"]) for row in contact_controls
        ),
        "all_stable_controls_accepted": all(
            bool(row["compiled_box_safe_terminal"]) for row in stable_controls
        ),
        "timeouts_remain_unknown": all(
            not bool(row["compiled_box_safe_terminal"]) for row in timeouts
        ),
        "two_sided_support_every_state": all(
            bool(item["two_sided_support"]) for item in per_case
        ),
        "recoverable_training_states": sum(
            item["safe_candidate_count"] > 0 and item["split"] == "train"
            for item in per_case
        ) >= int(gate["minimum_recoverable_train_states"]),
        "recoverable_validation_states": sum(
            item["safe_candidate_count"] > 0 and item["split"] == "validation"
            for item in per_case
        ) >= int(gate["minimum_recoverable_validation_states"]),
    }
    return {
        "gates": gates,
        "strict_gate_pass": bool(all(gates.values())),
        "training_authorized": bool(all(gates.values())),
        "case_reports": per_case,
        "counts": {
            "cases": len(cases),
            "candidates": len(all_candidates),
            "raw_contact_controls": len(contact_controls),
            "stable_controls": len(stable_controls),
            "timeouts": len(timeouts),
            "recoverable_train_states": sum(
                item["safe_candidate_count"] > 0 and item["split"] == "train"
                for item in per_case
            ),
            "recoverable_validation_states": sum(
                item["safe_candidate_count"] > 0 and item["split"] == "validation"
                for item in per_case
            ),
        },
    }
