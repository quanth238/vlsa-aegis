"""Gate a combined whole-body future-risk cohort without fitting a model."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.whole_body_support_audit import canonical


CONFIG_SCHEMA = "vlsa_distal_whole_body_combined_audit.v1"
CONFIG_SCHEMA_V2 = "vlsa_distal_whole_body_combined_audit.v2"
RESULT_SCHEMA = "vlsa_distal_whole_body_combined_audit_result.v1"


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Any) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    schema = value.get("schema_version")
    if schema not in (CONFIG_SCHEMA, CONFIG_SCHEMA_V2):
        raise ValueError("combined whole-body audit schema differs")
    expected_protocol = {
        CONFIG_SCHEMA: "vlsa-distal-whole-body-combined-audit-v1",
        CONFIG_SCHEMA_V2: "vlsa-distal-whole-body-combined-audit-v2",
    }[schema]
    if value.get("protocol_id") != expected_protocol:
        raise ValueError("combined whole-body audit protocol differs")
    if value.get("group_order") != [
        "end_effector", "palm", "L5", "L6", "L7",
    ]:
        raise ValueError("combined whole-body group order differs")
    if value.get("claimed_physical_groups") != ["palm", "L5", "L6"]:
        raise ValueError("combined whole-body claimed groups differ")
    if value.get("diagnostic_groups") != ["end_effector", "L7"]:
        raise ValueError("combined whole-body diagnostic groups differ")
    required_counts = value.get("required_split_case_count")
    if (
        not isinstance(required_counts, dict)
        or set(required_counts) != {"train", "validation", "test"}
        or any(int(count) <= 0 for count in required_counts.values())
        or (
            schema == CONFIG_SCHEMA
            and required_counts != {"train": 16, "validation": 4, "test": 4}
        )
    ):
        raise ValueError("combined whole-body split counts differ")
    if value.get("minimum_two_sided_state_count") != {
        "train": 4, "validation": 2, "test": 2,
    }:
        raise ValueError("combined whole-body support gate differs")
    if value.get("timeout_rule") != "UNKNOWN_excluded_from_safe_unsafe_support":
        raise ValueError("combined whole-body timeout rule differs")
    if value.get("initial_safety_rule") != (
        "every_prevention_state_positive_and_contact_free_for_palm_L5_L6_L7"
    ):
        raise ValueError("combined whole-body initial-safety rule differs")
    expected_source_count = 2 if schema == CONFIG_SCHEMA else 3
    if len(value.get("sources", [])) != expected_source_count:
        raise ValueError("combined whole-body source count differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def evaluate_gate(
    summary: Mapping[str, Any], config: Mapping[str, Any], *,
    source_replay_exact: bool, source_state_hash_exact: bool,
    physical_false_safe_count: int, context_complete: bool,
    maximum_bellman_residual: float,
) -> dict[str, Any]:
    required_splits = config["required_split_case_count"]
    required_support = config["minimum_two_sided_state_count"]
    split_summary = summary["split_summary"]
    split_count_pass = all(
        int(split_summary[split]["case_count"]) == int(required)
        for split, required in required_splits.items()
    )

    group_gates = {}
    for group in config["group_order"]:
        counts = {
            split: int(split_summary[split]["per_group"][group][
                "two_sided_state_count"
            ])
            for split in required_splits
        }
        passes = all(
            counts[split] >= int(required_support[split])
            for split in required_splits
        )
        group_gates[group] = {
            "two_sided_state_count": counts,
            "minimum_required": dict(required_support),
            "passes": passes,
            "claim_status": (
                "claimed_physical" if group in config["claimed_physical_groups"]
                else "diagnostic"
            ),
        }

    unsafe = [
        row["case_id"] for row in summary["per_case"]
        if not bool(row["initially_safe_across_physical_groups"])
    ]
    missing_global_safe = {
        split: [
            row["case_id"] for row in summary["per_case"]
            if row["split"] == split
            and bool(row["initially_safe_across_physical_groups"])
            and int(row["global_support"]["physical_safe_candidate_count"]) == 0
        ]
        for split in ("validation", "test")
    }
    apparatus_pass = bool(
        source_replay_exact
        and source_state_hash_exact
        and int(physical_false_safe_count) == 0
        and context_complete
        and float(maximum_bellman_residual) == 0.0
    )
    claimed_groups_pass = all(
        group_gates[group]["passes"]
        for group in config["claimed_physical_groups"]
    )
    initial_safety_pass = not unsafe
    global_safe_support_pass = not any(missing_global_safe.values())
    training_authorized = bool(
        apparatus_pass and split_count_pass and initial_safety_pass
        and claimed_groups_pass and global_safe_support_pass
    )
    return {
        "apparatus_pass": apparatus_pass,
        "split_count_pass": split_count_pass,
        "initial_safety_pass": initial_safety_pass,
        "initially_unsafe_case_ids": unsafe,
        "group_gates": group_gates,
        "claimed_groups_pass": claimed_groups_pass,
        "missing_global_safe_case_ids": missing_global_safe,
        "global_safe_support_pass": global_safe_support_pass,
        "source_replay_exact": bool(source_replay_exact),
        "source_state_hash_exact": bool(source_state_hash_exact),
        "physical_false_safe_count": int(physical_false_safe_count),
        "context_complete": bool(context_complete),
        "maximum_bellman_residual": float(maximum_bellman_residual),
        "training_authorized": training_authorized,
        "authorized_prediction_groups": (
            list(config["claimed_physical_groups"]) if training_authorized else []
        ),
    }
