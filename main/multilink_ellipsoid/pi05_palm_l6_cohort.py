"""Freeze a strict-split palm/L6 cohort from a validated source audit."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
    canonical, file_sha256,
)


CONFIG_SCHEMA = "vlsa_distal_pi05_palm_l6_cohort_selection.v1"
RESULT_SCHEMA = "vlsa_distal_pi05_palm_l6_cohort_selection_result.v1"
CASE_SCHEMA = "vlsa_distal_targeted_pi05_palm_l6_case.v1"
SPLITS = ("train", "validation", "test")
TARGETS = ("palm", "L6")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "source_audit",
        "split_task_level_groups", "target_order", "target_split_counts",
        "selection_rule", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("pi05 palm/L6 cohort config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-pi05-palm-l6-cohort-selection-v1"
        or value["target_order"] != list(TARGETS)
        or set(value["split_task_level_groups"]) != set(SPLITS)
        or set(value["target_split_counts"]) != set(TARGETS)
        or any(
            set(value["target_split_counts"][target]) != set(SPLITS)
            for target in TARGETS
        )
        or value["selection_rule"]["candidate_outcomes_accessed"] is not False
        or value["selection_rule"]["one_state_per_root_episode"] is not True
    ):
        raise ValueError("pi05 palm/L6 cohort protocol differs")
    group_splits: dict[str, str] = {}
    for split in SPLITS:
        groups = [str(item) for item in value["split_task_level_groups"][split]]
        if len(groups) != len(set(groups)):
            raise ValueError("pi05 cohort split groups repeat")
        for group in groups:
            if group in group_splits:
                raise ValueError("pi05 cohort task group crosses splits")
            group_splits[group] = split
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _rank(record: Mapping[str, Any], target: str) -> tuple[Any, ...]:
    digest = hashlib.sha256(
        f"{target}|{record['case_id']}".encode("utf-8")
    ).hexdigest()
    return (not bool(record["task_success"]), digest, str(record["case_id"]))


def choose_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used_cases: set[str] = set()
    target_order = list(config["target_order"])
    for target_index, target in enumerate(target_order):
        later_targets = set(target_order[target_index + 1:])
        for split in SPLITS:
            allowed = set(config["split_task_level_groups"][split])
            count = int(config["target_split_counts"][target][split])
            def select(*, reserve_later: bool) -> list[Mapping[str, Any]]:
                by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
                for record in records:
                    eligible = set(record.get("eligible_target_groups", []))
                    if (
                        target in eligible
                        and record["task_level_group_id"] in allowed
                        and record["case_id"] not in used_cases
                        and not (reserve_later and eligible.intersection(later_targets))
                    ):
                        by_group[str(record["task_level_group_id"])].append(record)
                for rows in by_group.values():
                    rows.sort(key=lambda row: _rank(row, target))
                selected: list[Mapping[str, Any]] = []
                group_names = sorted(by_group)
                depth = 0
                while len(selected) < count:
                    added = False
                    for group in group_names:
                        rows = by_group[group]
                        if depth < len(rows):
                            selected.append(rows[depth])
                            added = True
                            if len(selected) == count:
                                break
                    if not added:
                        break
                    depth += 1
                return selected

            selected = select(reserve_later=True)
            if len(selected) != count:
                selected = select(reserve_later=False)
            if len(selected) != count:
                raise ValueError(
                    f"insufficient eligible {target} {split} source episodes: "
                    f"required {count}, found {len(selected)}"
                )
            for record in selected:
                used_cases.add(str(record["case_id"]))
                chosen.append({"target_group": target, "split": split, **dict(record)})
    return chosen


def manifest_rows(
    chosen: Sequence[Mapping[str, Any]], *, table1_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for record in chosen:
        target = str(record["target_group"])
        warning = record["warning_contract_by_group"][target]
        pi05_path = Path(record["pi05_result_path"])
        aegis_path = Path(record["aegis_geometry_result_path"])
        pi05 = json.loads(pi05_path.read_text(encoding="utf-8"))
        active = str(pi05["failure_diagnostics"]["contacts"]["active_obstacle_name"])
        rows.append({
            "schema_version": CASE_SCHEMA,
            "case_id": str(record["case_id"]),
            "episode_group_id": str(record["case_id"]),
            "episode_index": int(record["episode_index"]),
            "suite": str(record["suite"]),
            "task_level_group_id": str(record["task_level_group_id"]),
            "target_group": target,
            "split": str(record["split"]),
            "state_step": int(warning["state_step"]),
            "first_target_contact_step": int(
                record["first_contact_step_by_group"][target]
            ),
            "active_obstacle_name": active,
            "archived_result_relative_path": str(pi05_path.relative_to(table1_root)),
            "archived_result_file_sha256": str(record["pi05_result_file_sha256"]),
            "archived_result_payload_sha256": str(record["pi05_result_payload_sha256"]),
            "aegis_geometry_result_relative_path": str(aegis_path.relative_to(table1_root)),
            "aegis_geometry_result_file_sha256": str(
                record["aegis_geometry_result_file_sha256"]
            ),
            "aegis_geometry_result_payload_sha256": str(
                record["aegis_geometry_result_payload_sha256"]
            ),
            "nominal_action_source": "raw_pi05_nominal_translational",
            "released_AEGIS_EE_QP_enabled": False,
            "paired_AEGIS_perception_role": (
                "fixed_backup_and_diagnostic_EE_geometry_only"
            ),
            "prospective_split_frozen_before_candidate_outcomes": True,
            "candidate_outcomes_used_for_selection": False,
            "selection_rule": "floor((first_target_raw_contact_step-5)/5)*5",
        })
    return rows


def manifest_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical(row) + b"\n" for row in rows)


def selection_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["task_level_group_id"])].add(str(row["split"]))
    return {
        "case_count": len(rows),
        "target_split_counts": {
            target: {
                split: sum(
                    row["target_group"] == target and row["split"] == split
                    for row in rows
                )
                for split in SPLITS
            }
            for target in TARGETS
        },
        "split_case_counts": {
            split: sum(row["split"] == split for row in rows) for split in SPLITS
        },
        "task_level_group_split_isolation": bool(
            group_splits
            and all(len(splits) == 1 for splits in group_splits.values())
        ),
        "candidate_outcomes_accessed": False,
    }
