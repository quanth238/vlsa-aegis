#!/usr/bin/env python3
"""Audit clean L5 availability across Goal tasks and both obstacle levels."""

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
    from main.multilink_ellipsoid.l5_four_task_availability import (
        archived_result_path, file_sha256,
    )
    from main.multilink_ellipsoid.moka_noise_discovery import (
        contact_evidence, geometry_evidence, validate_result_payload,
    )

    result_path = archived_result_path(table1_root, row)
    base = {
        "case_id": str(row["case_id"]),
        "case_ordinal": int(row["case_ordinal"]),
        "logical_task_index": int(row["logical_task_index"]),
        "safety_level": str(row["safety_level"]),
        "task_level_group_id": str(row["task_level_group_id"]),
        "episode_index": int(row["episode_index"]),
        "result_path": str(result_path),
    }
    if not result_path.is_file():
        return {**base, "classification": "MISSING_RESULT", "eligible": False}
    result = _load(result_path)
    validate_result_payload(result)
    _require(result["case_id"] == row["case_id"], "availability case binding differs")
    contacts_record = result["failure_diagnostics"]["contacts"]
    contact_path = result_path.parents[2] / contacts_record["path"]
    _require(contact_path.is_file(), "availability contact artifact missing")
    _require(
        file_sha256(contact_path) == contacts_record["sha256"],
        "availability contact artifact differs",
    )
    contacts = contact_evidence(contact_path)
    geometry_record = result["failure_diagnostics"].get("geometry", {}).get("record", {})
    geometry_complete = geometry_record.get("status") == "complete"
    geometry = geometry_evidence(result) if geometry_complete else {
        "proxy_valid": False,
        "status": str(geometry_record.get("status", "missing")),
    }
    bodies = contacts["robot_contact_counts_by_body"]
    l5 = int(bodies.get("robot0_link5", 0)) > 0
    l6_l7 = bool(
        int(bodies.get("robot0_link6", 0))
        or int(bodies.get("robot0_link7", 0))
    )
    first_l5 = contacts["first_robot_contact_step_by_body"].get("robot0_link5")
    dynamic_bad = bool(
        contacts["event_counts_by_role"].get("dynamic_task_object", 0)
        or contacts["event_counts_by_role"].get("dynamic_other", 0)
    )
    complete = bool(result.get("status") == "complete" and result.get("scientific_result"))
    task_success = bool(result.get("task_success"))
    paper_car_failure = bool(result.get("metrics", {}).get("paper_collision"))
    minimum_step = int(config["clean_L5_eligibility"]["minimum_first_L5_contact_step"])
    eligible = bool(
        complete and task_success and paper_car_failure and geometry["proxy_valid"]
        and l5 and first_l5 is not None and int(first_l5) >= minimum_step
        and not dynamic_bad and not contacts["settled_relevant_contact"]
    )
    if not complete:
        classification = "FAILED_OR_INCOMPLETE"
    elif not task_success:
        classification = "TASK_FAILURE"
    elif not paper_car_failure:
        classification = "TASK_SUCCESS_NO_CAR_FAILURE"
    elif not geometry_complete or not geometry["proxy_valid"]:
        classification = "PROXY_INVALID_OR_INCOMPLETE"
    elif contacts["settled_relevant_contact"]:
        classification = "INITIALLY_UNSAFE"
    elif dynamic_bad:
        classification = "DYNAMIC_CONTACT_OUT_OF_SCOPE"
    elif l5 and first_l5 is not None and int(first_l5) >= minimum_step:
        classification = "ELIGIBLE_TASK_SUCCESS_L5"
    elif l5:
        classification = "L5_CONTACT_TOO_EARLY"
    elif l6_l7:
        classification = "DISTAL_CONTACT_NON_L5"
    else:
        classification = "CAR_FAILURE_WITHOUT_L5_L7_CONTACT"
    return {
        **base,
        "classification": classification,
        "eligible": eligible,
        "task_success": task_success,
        "paper_CAR_failure": paper_car_failure,
        "geometry": geometry,
        "contacts": contacts,
        "first_L5_contact_step": first_l5,
        "result_file_sha256": file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "contact_file_sha256": file_sha256(contact_path),
    }


def run(
    *, repo_root: Path, table1_root: Path, config_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_four_task_availability import (
        RESULT_SCHEMA, load_config, load_population, payload_sha256,
        summarize_records,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path, repo_root=repo_root)
    _require(
        str(table1_root) == config["source"]["table1_root"],
        "availability Table1 root differs",
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
        "training_authorized": False,
        "boundary_collection_authorized": bool(
            summary["all_four_task_folds_feasible"]
        ),
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
