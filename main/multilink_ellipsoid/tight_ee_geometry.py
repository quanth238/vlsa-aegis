"""Contracts for a contact-aligned tight end-effector geometry audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_tight_ee_geometry_audit.v1"
RESULT_SCHEMA = "vlsa_tight_ee_geometry_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_tight_ee_geometry_audit_validation.v1"


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


def load_config(path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "tight_ee_primitives", "cohort", "fit", "replay", "geometry",
        "visualization", "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("tight EE config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-tight-ee-geometry-audit-v1"
    ):
        raise ValueError("tight EE protocol differs")
    primitives = value["tight_ee_primitives"]
    expected_primitives = {
        "palm": "gripper0_hand_collision",
        "finger1_base": "gripper0_finger1_collision",
        "finger1_pad": "gripper0_finger1_pad_collision",
        "finger2_base": "gripper0_finger2_collision",
        "finger2_pad": "gripper0_finger2_pad_collision",
    }
    if primitives != expected_primitives:
        raise ValueError("tight EE primitive map differs")
    cases = value["cohort"]["cases"]
    identities = [str(item["case_id"]) for item in cases]
    if len(cases) != 4 or len(identities) != len(set(identities)):
        raise ValueError("tight EE frozen cohort differs")
    if value["visualization"]["case_id"] not in identities:
        raise ValueError("tight EE visualization case is outside the cohort")
    if repo_root is not None:
        source = value["source"]
        manifest = Path(repo_root) / str(source["population_manifest"])
        if file_sha256(manifest) != source["population_manifest_file_sha256"]:
            raise ValueError("tight EE population manifest differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def scientific_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that independent replay must reproduce exactly."""

    return {
        "case_id": record["case_id"],
        "expected_contact_groups": record["expected_contact_groups"],
        "pairing": record["pairing"],
        "primitive_templates": record["primitive_templates"],
        "replay": record["replay"],
        "raw_contact": record["raw_contact"],
        "geometry": record["geometry"],
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    cases = config["cohort"]["cases"]
    expected_ids = [str(item["case_id"]) for item in cases]
    by_id = {str(record["case_id"]): record for record in records}
    if len(records) != len(by_id) or set(by_id) != set(expected_ids):
        raise ValueError("tight EE result population differs")
    groups = list(config["tight_ee_primitives"])
    contact_cases = [
        item for item in cases if item["expected_contact_groups"]
    ]
    control_cases = [
        item for item in cases if not item["expected_contact_groups"]
    ]
    group_contact_samples = {
        group: sum(
            int(by_id[case_id]["raw_contact"]["sample_count_by_group"][group])
            for case_id in expected_ids
        )
        for group in groups
    }
    group_false_safes = {
        group: sum(
            int(by_id[case_id]["geometry"]["physical_false_safe_sample_count_by_group"][group])
            for case_id in expected_ids
        )
        for group in groups
    }
    missing_expected = {
        str(item["case_id"]): sorted(
            set(item["expected_contact_groups"])
            - set(by_id[str(item["case_id"])]["raw_contact"]["observed_groups"])
        )
        for item in contact_cases
    }
    control_positive = {
        str(item["case_id"]): all(
            float(by_id[str(item["case_id"])]["geometry"]["episode_minimum_radial_slack_by_group"][group])
            > 0.0
            for group in groups
        )
        for item in control_cases
    }
    gate = config["gate"]
    pass_gate = bool(
        all(record["replay"]["fidelity_pass"] for record in records)
        and all(record["geometry"]["primitive_certificate_pass"] for record in records)
        and all(not missing for missing in missing_expected.values())
        and all(value == 0 for value in group_false_safes.values())
        and (
            not gate["require_contact_free_control_positive_for_every_primitive"]
            or all(control_positive.values())
        )
    )
    return {
        "case_count": len(records),
        "contact_case_count": len(contact_cases),
        "contact_free_control_count": len(control_cases),
        "primitive_groups": groups,
        "raw_contact_sample_count_by_group": group_contact_samples,
        "physical_false_safe_sample_count_by_group": group_false_safes,
        "missing_expected_contact_groups_by_case": missing_expected,
        "contact_free_control_positive_by_case": control_positive,
        "released_proxy_is_comparator_only": True,
        "tight_ee_geometry_gate_pass": pass_gate,
    }
