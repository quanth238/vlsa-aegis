"""Contracts for the tight-geometry five-action prefix-risk dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


CONFIG_SCHEMA = "vlsa_tight_prefix_risk_dataset.v1"
CASE_SCHEMA = "vlsa_tight_prefix_risk_case.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_risk_validation.v1"
ROLLOUT_SCOPE = "candidate_five_action_prefix_only"
PRIMARY_GROUPS = (
    "palm",
    "finger1_base",
    "finger1_pad",
    "finger2_base",
    "finger2_pad",
    "L5",
)
DIAGNOSTIC_GROUPS = ("L6",)
GROUPS = PRIMARY_GROUPS + DIAGNOSTIC_GROUPS


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    payload.pop("validation_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path, *, repo_root: Optional[Path] = None) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "rollout_scope", "candidate_bank", "tight_geometry_validation",
        "exact_group_target", "cases", "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("tight prefix-risk config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-tight-prefix-risk-dataset-v1"
        or value["rollout_scope"] != ROLLOUT_SCOPE
    ):
        raise ValueError("tight prefix-risk protocol differs")
    cases = value["cases"]
    if len(cases) != 24 or len({item["case_id"] for item in cases}) != 24:
        raise ValueError("tight prefix-risk case population differs")
    split_counts = {
        split: sum(item["split"] == split for item in cases)
        for split in ("train", "validation", "test")
    }
    if split_counts != {"train": 14, "validation": 4, "test": 6}:
        raise ValueError("tight prefix-risk split differs")
    names = value["candidate_bank"]["candidate_names"]
    if len(names) != 13 or names[0] != "nominal" or len(set(names)) != 13:
        raise ValueError("tight prefix-risk candidate bank differs")
    target = value["exact_group_target"]
    if (
        target["group_order"] != list(GROUPS)
        or target["robot_rows"] != {
            "palm": [0], "finger1_base": [1], "finger1_pad": [2],
            "finger2_base": [3], "finger2_pad": [4],
            "L5": [5, 6, 7], "L6": [8, 9],
        }
        or target.get("include_released_aegis_end_effector_proxy") is not False
    ):
        raise ValueError("tight prefix-risk geometry rows differ")
    if repo_root is not None:
        source = value["source"]
        population = Path(repo_root) / source["population_manifest"]
        geometry = Path(repo_root) / source["legacy_replay_geometry_config"]
        distal = Path(repo_root) / target["robot_geometry_config"]
        if file_sha256(population) != source["population_manifest_file_sha256"]:
            raise ValueError("tight prefix-risk population binding differs")
        if file_sha256(geometry) != source["legacy_replay_geometry_config_file_sha256"]:
            raise ValueError("tight prefix-risk replay geometry binding differs")
        if file_sha256(distal) != target["robot_geometry_config_file_sha256"]:
            raise ValueError("tight prefix-risk distal geometry binding differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def scientific_view(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_index": record["case_index"],
        "case_id": record["case_id"],
        "split": record["split"],
        "config_file_sha256": record["config_file_sha256"],
        "config_payload_sha256": record["config_payload_sha256"],
        "case": record["case"],
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {str(record["case_id"]): record for record in records}
    expected_ids = [str(item["case_id"]) for item in config["cases"]]
    if len(by_id) != 24 or set(by_id) != set(expected_ids):
        raise ValueError("tight prefix-risk result population differs")
    split_summary = {}
    total_false_safes = {group: 0 for group in GROUPS}
    all_replay_exact = True
    all_prefix_only = True
    all_certified = True
    for split in ("train", "validation", "test"):
        split_records = [
            by_id[item["case_id"]]
            for item in config["cases"] if item["split"] == split
        ]
        per_group = {}
        global_safe_support = 0
        global_two_sided = 0
        initially_primary_safe = 0
        for record in split_records:
            case = record["case"]
            all_replay_exact = all_replay_exact and bool(case["source_replay_exact"])
            all_prefix_only = all_prefix_only and bool(
                case["rollout_scope"] == ROLLOUT_SCOPE
                and all(
                    candidate["action_count"] == 5
                    and candidate["phase_sample_counts"] == {
                        "prefix": candidate["sample_count"],
                        "backup": 0,
                        "terminal_hold": 0,
                    }
                    for candidate in case["candidates"]
                )
            )
            all_certified = all_certified and bool(
                case["exact_group_target"]["robot_primitive_certificate_pass"]
            )
            initial = case["exact_group_target"]
            if all(
                float(initial["initial_group_normalized_radial_slack"][group]) > 0.0
                and int(initial["initial_group_contact_sample_count"][group]) == 0
                for group in PRIMARY_GROUPS
            ):
                initially_primary_safe += 1
            candidate_global_safe = []
            candidate_global_unsafe = []
            for candidate in case["candidates"]:
                exact = candidate["exact_group_target"]
                primary_safe = all(
                    float(exact["group_future_violation"][group]) <= 0.0
                    and int(exact["group_contact_sample_count"][group]) == 0
                    for group in PRIMARY_GROUPS
                )
                car_safe = float(candidate["replayed_maximum_CAR_m"]) <= float(
                    config["gate"]["paper_car_threshold_m"]
                )
                candidate_global_safe.append(bool(primary_safe and car_safe))
                candidate_global_unsafe.append(not bool(primary_safe and car_safe))
            global_safe_support += int(any(candidate_global_safe))
            global_two_sided += int(
                any(candidate_global_safe) and any(candidate_global_unsafe)
            )
        for group in GROUPS:
            safe_count = unsafe_count = two_sided = false_safe = 0
            for record in split_records:
                candidate_safe = []
                for candidate in record["case"]["candidates"]:
                    exact = candidate["exact_group_target"]
                    q = float(exact["group_future_violation"][group])
                    contacts = int(exact["group_contact_sample_count"][group])
                    safe = q <= 0.0 and contacts == 0
                    candidate_safe.append(safe)
                    safe_count += int(safe)
                    unsafe_count += int(not safe)
                    false_safe += int(q <= 0.0 and contacts > 0)
                two_sided += int(any(candidate_safe) and not all(candidate_safe))
            total_false_safes[group] += false_safe
            per_group[group] = {
                "safe_candidate_count": safe_count,
                "unsafe_candidate_count": unsafe_count,
                "two_sided_state_count": two_sided,
                "physical_false_safe_count": false_safe,
            }
        split_summary[split] = {
            "case_count": len(split_records),
            "initially_primary_safe_case_count": initially_primary_safe,
            "global_safe_support_case_count": global_safe_support,
            "global_two_sided_case_count": global_two_sided,
            "per_group": per_group,
        }
    gate = {
        "case_count": len(records) == 24,
        "candidate_count": sum(len(record["case"]["candidates"]) for record in records) == 312,
        "source_replay_exact": all_replay_exact,
        "prefix_only": all_prefix_only,
        "robot_primitives_certified": all_certified,
        "zero_physical_false_safes": all(
            value == 0 for value in total_false_safes.values()
        ),
    }
    return {
        "case_count": len(records),
        "candidate_count": sum(len(record["case"]["candidates"]) for record in records),
        "split_summary": split_summary,
        "physical_false_safe_count_by_group": total_false_safes,
        "gate": gate,
        "dataset_gate_pass": bool(all(gate.values())),
        "diagnostic_Q_only_training_authorized": bool(all(gate.values())),
        "paper_scale_or_untouched_test_claim_authorized": False,
    }
