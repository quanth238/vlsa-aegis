#!/usr/bin/env python3
"""Audit palm/finger and distal contact availability in immutable Table 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def classify_case(
    row: Mapping[str, Any], table1_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from main.multilink_ellipsoid.ee_primitive_availability import (
        archived_result_path, contact_evidence, file_sha256,
    )
    from main.multilink_ellipsoid.moka_noise_discovery import (
        geometry_evidence, validate_result_payload,
    )

    result_path = archived_result_path(table1_root, row)
    base = {
        "case_id": str(row["case_id"]),
        "case_ordinal": int(row["case_ordinal"]),
        "suite": str(row["suite"]),
        "logical_task_index": int(row["logical_task_index"]),
        "safety_level": str(row["safety_level"]),
        "task_level_group_id": str(row["task_level_group_id"]),
        "episode_index": int(row["episode_index"]),
        "result_path": str(result_path),
    }
    if not result_path.is_file():
        return {
            **base, "classification": "MISSING_RESULT",
            "contact_groups": [], "eligible_clean_contact_groups": [],
            "eligible_clean_contact_free_control": False,
        }
    result = _load(result_path)
    validate_result_payload(result)
    _require(result["case_id"] == row["case_id"], "EE availability case differs")
    complete = bool(
        result.get("status") == "complete" and result.get("scientific_result")
    )
    task_success = bool(result.get("task_success"))
    paper_car = bool(result.get("metrics", {}).get("paper_collision"))
    contacts_record = result.get("failure_diagnostics", {}).get("contacts", {})
    contact_path = result_path.parents[2] / str(contacts_record.get("path", ""))
    if not contact_path.is_file():
        return {
            **base, "classification": "MISSING_CONTACT_ARTIFACT",
            "complete_scientific_result": complete,
            "task_success": task_success,
            "paper_CAR_failure": paper_car,
            "contact_groups": [], "eligible_clean_contact_groups": [],
            "eligible_clean_contact_free_control": False,
            "result_file_sha256": file_sha256(result_path),
            "result_payload_sha256": result["result_payload_sha256"],
        }
    _require(
        file_sha256(contact_path) == contacts_record["sha256"],
        "EE availability contact artifact differs",
    )
    contacts = contact_evidence(contact_path, config)
    geometry_record = (
        result.get("failure_diagnostics", {}).get("geometry", {}).get("record", {})
    )
    geometry_complete = geometry_record.get("status") == "complete"
    geometry = geometry_evidence(result) if geometry_complete else {
        "proxy_valid": False,
        "status": str(geometry_record.get("status", "missing")),
    }
    contact_groups = sorted(
        contacts["robot_contact_counts_by_constraint_group"]
    )
    first_steps = contacts["first_robot_contact_step_by_constraint_group"]
    eligibility = config["clean_contact_eligibility"]
    minimum_step = int(eligibility["minimum_first_contact_step"])
    dynamic_bad = bool(
        contacts["event_counts_by_role"].get("dynamic_task_object", 0)
        or contacts["event_counts_by_role"].get("dynamic_other", 0)
    )
    common_clean = bool(
        complete and task_success and geometry["proxy_valid"]
        and not contacts["settled_relevant_contact"] and not dynamic_bad
        and (
            paper_car
            or not eligibility["paper_CAR_failure_required"]
        )
    )
    eligible_groups = sorted(
        group for group in contact_groups
        if common_clean and int(first_steps[group]) >= minimum_step
    )
    clean_control = bool(
        common_clean and not contact_groups
        and not contacts["unmapped_robot_contact_counts"]
    )
    if not complete:
        classification = "FAILED_OR_INCOMPLETE"
    elif not task_success:
        classification = "TASK_FAILURE"
    elif not geometry_complete or not geometry["proxy_valid"]:
        classification = "PROXY_INVALID_OR_INCOMPLETE"
    elif contacts["settled_relevant_contact"]:
        classification = "INITIALLY_UNSAFE"
    elif dynamic_bad:
        classification = "DYNAMIC_CONTACT_CONFOUNDED"
    elif eligible_groups:
        classification = "CLEAN_TASK_SUCCESS_ROBOT_CONTACT"
    elif contact_groups:
        classification = "ROBOT_CONTACT_TOO_EARLY"
    elif contacts["unmapped_robot_contact_counts"]:
        classification = "UNMAPPED_ROBOT_CONTACT"
    else:
        classification = "CLEAN_TASK_SUCCESS_CONTACT_FREE"
    return {
        **base,
        "classification": classification,
        "complete_scientific_result": complete,
        "task_success": task_success,
        "paper_CAR_failure": paper_car,
        "geometry": geometry,
        "contacts": contacts,
        "contact_groups": contact_groups,
        "eligible_clean_contact_groups": eligible_groups,
        "eligible_clean_contact_free_control": clean_control,
        "result_file_sha256": file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "contact_file_sha256": file_sha256(contact_path),
    }


def run(
    *, repo_root: Path, table1_root: Path, config_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.ee_primitive_availability import (
        RESULT_SCHEMA, load_config, load_population, payload_sha256,
        summarize_records,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path, repo_root=repo_root)
    _require(
        str(table1_root) == config["source"]["table1_root"],
        "EE availability Table1 root differs",
    )
    rows = load_population(
        repo_root / config["source"]["population_manifest"], config,
    )
    records = [classify_case(row, table1_root, config) for row in rows]
    summary = summarize_records(records, config)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": identity,
        "allocation": allocation_record(),
        "config": config,
        "summary": summary,
        "records": records,
        "new_simulation_performed": False,
        "tighter_geometry_audit_authorized": bool(
            summary["tighter_geometry_audit_ready_groups"]
        ),
        "boundary_collection_authorized": False,
        "training_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    result["result_payload_sha256"] = payload_sha256(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repo_root=args.repo_root.resolve(), table1_root=args.table1_root.resolve(),
        config_path=args.config.resolve(), expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "summary": result["summary"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
