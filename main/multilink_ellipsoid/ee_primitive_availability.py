"""Contracts for the immutable EE/distal primitive availability audit."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_ee_primitive_availability_audit.v1"
RESULT_SCHEMA = "vlsa_distal_ee_primitive_availability_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_ee_primitive_availability_audit_validation.v1"
CONTACT_SCHEMA = "vlsa_table1_active_obstacle_contacts.v3"


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
        "population", "constraint_groups", "clean_contact_eligibility",
        "geometry_audit_gate", "contact_semantics", "next_if_ready",
        "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("EE primitive availability config keys differ")
    population = value["population"]
    names = [str(item["name"]) for item in value["constraint_groups"]]
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-distal-ee-primitive-availability-audit-v1"
        or population["expected_case_count"] != 1600
        or population["logical_tasks"] != [0, 1, 2, 3]
        or population["safety_levels"] != ["I", "II"]
        or names != ["palm", "finger1", "finger2", "L5", "L6", "L7"]
        or len(names) != len(set(names))
    ):
        raise ValueError("EE primitive availability protocol differs")
    for group in value["constraint_groups"]:
        if not group["body_names"] and not group["geom_names"]:
            raise ValueError("EE primitive group has no physical identity")
    if repo_root is not None:
        manifest = Path(repo_root) / value["source"]["population_manifest"]
        if file_sha256(manifest) != value["source"][
            "population_manifest_file_sha256"
        ]:
            raise ValueError("EE primitive population manifest differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def load_population(
    path: Path, config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    population = config["population"]
    selected = [
        row for row in rows
        if row["suite"] in population["suites"]
        and int(row["logical_task_index"]) in population["logical_tasks"]
        and row["safety_level"] in population["safety_levels"]
    ]
    if len(selected) != int(population["expected_case_count"]):
        raise ValueError("EE primitive population count differs")
    identities = [str(row["case_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("EE primitive population identities repeat")
    expected = int(population["episodes_per_task_level"])
    counts = Counter(
        (
            str(row["suite"]), int(row["logical_task_index"]),
            str(row["safety_level"]),
        )
        for row in selected
    )
    if any(
        counts[(suite, task, level)] != expected
        for suite in population["suites"]
        for task in population["logical_tasks"]
        for level in population["safety_levels"]
    ):
        raise ValueError("EE primitive task-level counts differ")
    return sorted(selected, key=lambda row: int(row["case_ordinal"]))


def archived_result_path(
    table1_root: Path, row: Mapping[str, Any],
) -> Path:
    task_ordinal = int(row["case_ordinal"]) // 50
    return (
        Path(table1_root) / "tasks" / f"task-{task_ordinal}" / "results"
        / "aegis" / str(row["case_id"]) / "result.json"
    )


def classify_robot_group(
    *, body_name: str, geom_name: str, config: Mapping[str, Any],
) -> str | None:
    # The collision geom is the most specific authority.  A finger geom can
    # legally inherit a hand body lineage, so body aliases are fallback only.
    geom_matches = [
        str(group["name"]) for group in config["constraint_groups"]
        if geom_name in group["geom_names"]
    ]
    if len(geom_matches) > 1:
        raise ValueError("robot geom maps to multiple primitive groups")
    if geom_matches:
        return geom_matches[0]
    matches = []
    for group in config["constraint_groups"]:
        if body_name in group["body_names"]:
            matches.append(str(group["name"]))
    if len(matches) > 1:
        raise ValueError("robot contact maps to multiple primitive groups")
    return matches[0] if matches else None


def contact_evidence(
    path: Path, config: Mapping[str, Any],
) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("schema_version") != CONTACT_SCHEMA:
        raise ValueError("EE primitive contact artifact schema differs")
    role_counts: Counter[str] = Counter()
    body_counts: Counter[str] = Counter()
    geom_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    first_group_step: dict[str, int] = {}
    unmapped_counts: Counter[str] = Counter()
    settled_relevant = False
    for snapshot in artifact["snapshots"]:
        for event in snapshot["events"]:
            other = event["other"]
            role = str(other["classification"])
            step = int(event["step"])
            role_counts[role] += 1
            if step < 0 and role in {
                "robot", "dynamic_task_object", "dynamic_other",
            }:
                settled_relevant = True
            if role != "robot":
                continue
            body = str(other["body_name"])
            geom = str(other["geom_name"])
            body_counts[body] += 1
            geom_counts[geom] += 1
            group = classify_robot_group(
                body_name=body, geom_name=geom, config=config,
            )
            if group is None:
                unmapped_counts[f"{body}|{geom}"] += 1
                continue
            group_counts[group] += 1
            first_group_step[group] = min(
                first_group_step.get(group, step), step,
            )
    return {
        "event_counts_by_role": dict(sorted(role_counts.items())),
        "robot_contact_counts_by_body": dict(sorted(body_counts.items())),
        "robot_contact_counts_by_geom": dict(sorted(geom_counts.items())),
        "robot_contact_counts_by_constraint_group": dict(
            sorted(group_counts.items())
        ),
        "first_robot_contact_step_by_constraint_group": dict(
            sorted(first_group_step.items())
        ),
        "unmapped_robot_contact_counts": dict(sorted(unmapped_counts.items())),
        "settled_relevant_contact": settled_relevant,
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    groups = [str(item["name"]) for item in config["constraint_groups"]]
    gate = config["geometry_audit_gate"]
    per_group = []
    for group in groups:
        contact_records = [
            record for record in records
            if group in record.get("contact_groups", [])
        ]
        clean = [
            record for record in records
            if group in record.get("eligible_clean_contact_groups", [])
        ]
        cohorts = sorted({
            str(record["task_level_group_id"]) for record in clean
        })
        matched_controls = [
            record for record in records
            if bool(record.get("eligible_clean_contact_free_control"))
            and str(record["task_level_group_id"]) in set(cohorts)
        ]
        ready = bool(
            len(clean) >= int(gate["minimum_clean_contact_episodes"])
            and len(cohorts)
            >= int(gate["minimum_distinct_task_level_groups"])
            and len(matched_controls)
            >= int(gate["minimum_matched_contact_free_controls"])
        )
        per_group.append({
            "constraint_group": group,
            "contact_episode_count": len(contact_records),
            "clean_task_success_contact_episode_count": len(clean),
            "clean_task_success_contact_case_ids": [
                str(record["case_id"]) for record in clean
            ],
            "distinct_clean_task_level_groups": cohorts,
            "matched_clean_contact_free_control_count": len(matched_controls),
            "matched_clean_contact_free_control_case_ids": [
                str(record["case_id"]) for record in matched_controls
            ],
            "tighter_geometry_audit_ready": ready,
        })
    ready_groups = [
        item["constraint_group"] for item in per_group
        if item["tighter_geometry_audit_ready"]
    ]
    classifications = Counter(
        str(record["classification"]) for record in records
    )
    unmapped = Counter()
    geom_counts = Counter()
    for record in records:
        contact = record.get("contacts", {})
        unmapped.update(contact.get("unmapped_robot_contact_counts", {}))
        geom_counts.update(contact.get("robot_contact_counts_by_geom", {}))
    return {
        "case_count": len(records),
        "classification_counts": dict(sorted(classifications.items())),
        "constraint_group_coverage": per_group,
        "tighter_geometry_audit_ready_groups": ready_groups,
        "robot_contact_event_counts_by_geom": dict(sorted(geom_counts.items())),
        "unmapped_robot_contact_event_counts": dict(sorted(unmapped.items())),
        "next_action": (
            config["next_if_ready"] if ready_groups
            else "expand_outcome_blind_population_before_geometry_or_learning"
        ),
    }
