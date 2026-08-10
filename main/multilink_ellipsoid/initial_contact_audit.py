"""Classify paired controller states by zero-cutoff protected contact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adaptive_native_distance import _pair_query, _raw_pair_contacts
from .execution_margin_nn import _canonical
from .native_geom_margin import _native_model_data


CONFIG_SCHEMA = "vlsa_distal_initial_contact_audit_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_initial_contact_audit_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_initial_contact_audit_moka10_validation.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("initial-contact config is invalid JSON") from error
    if not isinstance(config, dict) or set(config) != {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "measurement", "gate",
    }:
        raise ValueError("initial-contact config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"] != "vlsa-distal-initial-contact-audit-moka10-v1"
    ):
        raise ValueError("initial-contact protocol differs")
    if set(config["immutable_source"]) != {
        "source_population_manifest_sha256",
        "boundary_generalization_manifest_sha256",
        "same_task_expansion_manifest_sha256",
        "targeted_expansion_manifest_sha256",
        "expanded_dataset_file_sha256", "expanded_dataset_payload_sha256",
        "proxy_contact_result_file_sha256", "proxy_contact_result_payload_sha256",
        "proxy_contact_validation_file_sha256",
    }:
        raise ValueError("initial-contact immutable source differs")
    if config["population"] != {
        "expected_state_count": 85,
        "expected_split_counts": {"train": 60, "validation": 10, "test": 15},
        "expected_complete_episode_count": 17,
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "state_identity": (
            "exact_paired_settled_state_plus_canonical_archived_action_"
            "prefix_replayed_in_current_runtime"
        ),
        "no_state_discarded": True,
    }:
        raise ValueError("initial-contact population differs")
    if config["measurement"] != {
        "protected_bodies": ["robot0_link5", "robot0_link6", "robot0_link7"],
        "obstacle_scope": "collision_eligible_active_obstacle_body_lineage",
        "distance_api": "mujoco.mj_geomDistance",
        "distmax_m": 0.0, "negative_tolerance_m": 1.0e-6,
        "raw_contact_distance_threshold_m": 0.0,
        "classification": {
            "prevention": "zero_registered_protected_contact_at_k0",
            "recovery": "at_least_one_registered_protected_contact_at_k0",
        },
        "positive_native_distance_forbidden": True,
    }:
        raise ValueError("initial-contact measurement differs")
    if config["gate"] != {
        "semantic_inventory_hash_count": 1,
        "maximum_negative_without_raw_pair_contact_count": 0,
        "maximum_raw_pair_contact_without_negative_count": 0,
        "maximum_unregistered_raw_pair_count": 0,
        "required_state_count": 85, "required_test_state_count": 15,
        "training_in_this_gate": False, "QP_in_this_gate": False,
        "closed_loop_in_this_gate": False,
    }:
        raise ValueError("initial-contact gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def measure_initial_contact(
    env: Any, inventory: Mapping[str, Any], measurement: Mapping[str, Any],
) -> dict[str, Any]:
    native_model, native_data = _native_model_data(env)
    tolerance = float(measurement["negative_tolerance_m"])
    raw_pairs, raw_records = _raw_pair_contacts(
        env, inventory, float(measurement["raw_contact_distance_threshold_m"])
    )
    registered = {
        (int(group["protected_geom_id"]), int(obstacle_geom))
        for group in inventory["groups"]
        for obstacle_geom in group["obstacle_geom_ids"]
    }
    pair_records = []
    group_records = []
    for group in inventory["groups"]:
        group_pairs = []
        for obstacle_geom in group["obstacle_geom_ids"]:
            pair = (int(group["protected_geom_id"]), int(obstacle_geom))
            query = _pair_query(native_model, native_data, pair[0], pair[1], 0.0)
            negative = bool(float(query["distance_m"]) < -tolerance)
            raw = pair in raw_pairs
            record = {
                "protected_group_index": int(group["group_index"]),
                "protected_geom_id": pair[0], "obstacle_geom_id": pair[1],
                "zero_query_distance_m": float(query["distance_m"]),
                "zero_query_negative": negative, "raw_pair_contact": raw,
                "negative_without_raw_pair_contact": bool(negative and not raw),
                "raw_pair_contact_without_negative": bool(raw and not negative),
                "finite": bool(query["finite"]),
            }
            pair_records.append(record)
            group_pairs.append(record)
        group_records.append({
            "group_index": int(group["group_index"]),
            "body_name": str(group["body_name"]),
            "raw_contact": any(item["raw_pair_contact"] for item in group_pairs),
            "zero_query_negative": any(
                item["zero_query_negative"] for item in group_pairs
            ),
        })
    unregistered = sorted(set(raw_pairs) - registered)
    raw_count = len(raw_records)
    return {
        "cohort": "recovery" if raw_count else "prevention",
        "initially_safe": raw_count == 0,
        "raw_protected_contact_count": raw_count,
        "raw_protected_contact_records": raw_records,
        "negative_pair_count": sum(
            item["zero_query_negative"] for item in pair_records
        ),
        "negative_without_raw_pair_contact_count": sum(
            item["negative_without_raw_pair_contact"] for item in pair_records
        ),
        "raw_pair_contact_without_negative_count": sum(
            item["raw_pair_contact_without_negative"] for item in pair_records
        ),
        "unregistered_raw_pair_count": len(unregistered),
        "all_queries_finite": all(item["finite"] for item in pair_records),
        "group_records": group_records,
    }


def summarize(
    state_records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    semantic_hash_count: int,
) -> dict[str, Any]:
    gate = config["gate"]
    split_counts = {
        split: sum(item["split"] == split for item in state_records)
        for split in ("train", "validation", "test")
    }
    split_cohorts = {
        split: {
            cohort: sum(
                item["split"] == split and item["cohort"] == cohort
                for item in state_records
            )
            for cohort in ("prevention", "recovery")
        }
        for split in ("train", "validation", "test")
    }
    aggregates = {
        "state_count": len(state_records), "split_counts": split_counts,
        "semantic_inventory_hash_count": int(semantic_hash_count),
        "prevention_state_count": sum(
            item["cohort"] == "prevention" for item in state_records
        ),
        "recovery_state_count": sum(
            item["cohort"] == "recovery" for item in state_records
        ),
        "test_prevention_state_count": split_cohorts["test"]["prevention"],
        "test_recovery_state_count": split_cohorts["test"]["recovery"],
        "negative_without_raw_pair_contact_count": sum(
            int(item["negative_without_raw_pair_contact_count"])
            for item in state_records
        ),
        "raw_pair_contact_without_negative_count": sum(
            int(item["raw_pair_contact_without_negative_count"])
            for item in state_records
        ),
        "unregistered_raw_pair_count": sum(
            int(item["unregistered_raw_pair_count"]) for item in state_records
        ),
        "all_queries_finite": all(
            bool(item["all_queries_finite"]) for item in state_records
        ),
        "split_cohort_counts": split_cohorts,
    }
    complete = bool(
        aggregates["state_count"] == int(gate["required_state_count"])
        and split_counts["test"] == int(gate["required_test_state_count"])
        and aggregates["semantic_inventory_hash_count"]
        == int(gate["semantic_inventory_hash_count"])
        and aggregates["negative_without_raw_pair_contact_count"]
        <= int(gate["maximum_negative_without_raw_pair_contact_count"])
        and aggregates["raw_pair_contact_without_negative_count"]
        <= int(gate["maximum_raw_pair_contact_without_negative_count"])
        and aggregates["unregistered_raw_pair_count"]
        <= int(gate["maximum_unregistered_raw_pair_count"])
        and aggregates["all_queries_finite"]
    )
    return {
        "aggregates": aggregates,
        "decision": {
            "initial_contact_audit_complete": complete,
            "prevention_cohort_training_preregistration_authorized": complete,
            "recovery_states_must_exclude_k0_from_future_margin": bool(
                aggregates["recovery_state_count"]
            ),
            "training_executed": False, "QP_executed": False,
            "closed_loop_E05_authorized": False,
        },
    }


__all__ = [
    "CONFIG_SCHEMA", "RESULT_SCHEMA", "VALIDATION_SCHEMA", "load_config",
    "measure_initial_contact", "summarize",
]
