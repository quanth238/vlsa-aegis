#!/usr/bin/env python3
"""Audit immutable natural pi0.5 episodes for palm/L6 source states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def classify_case(
    row: Mapping[str, Any], *, repo_root: Path, table1_root: Path,
    config: Mapping[str, Any], excluded: set[str],
) -> dict[str, Any]:
    from main.multilink_ellipsoid.ee_primitive_availability import contact_evidence
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        _raw_action_contract, _result_payload_valid, _warning_step,
        file_sha256, result_path,
    )

    pi05_path = result_path(table1_root, row, "pi05")
    aegis_path = result_path(table1_root, row, "aegis")
    base = {
        "case_id": str(row["case_id"]),
        "case_ordinal": int(row["case_ordinal"]),
        "suite": str(row["suite"]),
        "logical_task_index": int(row["logical_task_index"]),
        "safety_level": str(row["safety_level"]),
        "task_level_group_id": str(row["task_level_group_id"]),
        "episode_index": int(row["episode_index"]),
        "excluded_previous_candidate_case": str(row["case_id"]) in excluded,
        "pi05_result_path": str(pi05_path),
        "aegis_geometry_result_path": str(aegis_path),
    }
    if not pi05_path.is_file() or not aegis_path.is_file():
        return {**base, "classification": "MISSING_PAIRED_RESULT", "eligible_target_groups": []}
    pi05 = _load(pi05_path)
    aegis = _load(aegis_path)
    if not _result_payload_valid(pi05) or not _result_payload_valid(aegis):
        return {**base, "classification": "INVALID_RESULT_PAYLOAD", "eligible_target_groups": []}
    contacts_record = pi05.get("failure_diagnostics", {}).get("contacts", {})
    contact_path = pi05_path.parents[2] / str(contacts_record.get("path", ""))
    if not contact_path.is_file():
        return {**base, "classification": "MISSING_CONTACT_ARTIFACT", "eligible_target_groups": []}
    if file_sha256(contact_path) != contacts_record.get("sha256"):
        return {**base, "classification": "INVALID_CONTACT_ARTIFACT", "eligible_target_groups": []}
    contacts = contact_evidence(contact_path, config)
    first_steps = contacts["first_robot_contact_step_by_constraint_group"]
    contact_groups = sorted(contacts["robot_contact_counts_by_constraint_group"])
    dynamic_bad = bool(
        contacts["event_counts_by_role"].get("dynamic_task_object", 0)
        or contacts["event_counts_by_role"].get("dynamic_other", 0)
    )
    source_complete = bool(
        pi05.get("status") == "complete" and pi05.get("scientific_result") is True
        and pi05.get("arm") == "pi05_translational" and pi05.get("mode") == "pi05"
        and pi05.get("perception", {}).get("status") == "not_run"
    )
    perception = aegis.get("perception", {})
    geometry_available = bool(
        isinstance(perception, dict)
        and all(key in perception for key in (
            "mvee_center", "mvee_rotation", "mvee_semiaxes", "obstacle_label",
        ))
    )
    minimum = int(config["eligibility"]["minimum_first_target_contact_step"])
    eligible_groups: list[str] = []
    contracts: dict[str, Any] = {}
    for group in ("palm", "L6"):
        if group not in first_steps:
            continue
        step = int(first_steps[group])
        state_step = _warning_step(step)
        action = _raw_action_contract(pi05, state_step)
        contracts[group] = {"state_step": state_step, **action}
        if (
            source_complete and geometry_available
            and not base["excluded_previous_candidate_case"]
            and not contacts["settled_relevant_contact"] and not dynamic_bad
            and not contacts["unmapped_robot_contact_counts"]
            and step >= minimum and state_step >= 0
            and action["complete_through_five_action_chunk"]
            and action["raw_pi05_action_invariant"] and action["query_bound"]
        ):
            eligible_groups.append(group)
    if base["excluded_previous_candidate_case"]:
        classification = "EXCLUDED_PREVIOUS_CANDIDATE_CASE"
    elif not source_complete:
        classification = "INCOMPLETE_OR_NON_PI05_SOURCE"
    elif contacts["settled_relevant_contact"]:
        classification = "INITIALLY_CONTACTING_SOURCE"
    elif dynamic_bad:
        classification = "DYNAMIC_CONTACT_CONFOUNDED"
    elif contacts["unmapped_robot_contact_counts"]:
        classification = "UNMAPPED_ROBOT_CONTACT"
    elif not geometry_available:
        classification = "PAIRED_AEGIS_GEOMETRY_UNAVAILABLE"
    elif eligible_groups:
        classification = "ELIGIBLE_RAW_PI05_TARGET_CONTACT"
    elif set(contact_groups).intersection({"palm", "L6"}):
        classification = "TARGET_CONTACT_NOT_ACTION_BOUNDARY_ELIGIBLE"
    else:
        classification = "NO_PALM_OR_L6_CONTACT"
    return {
        **base,
        "classification": classification,
        "complete_scientific_pi05": source_complete,
        "task_success": bool(pi05.get("task_success")),
        "paper_CAR_failure": bool(pi05.get("metrics", {}).get("paper_collision")),
        "contact_groups": contact_groups,
        "first_contact_step_by_group": first_steps,
        "eligible_target_groups": eligible_groups,
        "warning_contract_by_group": contracts,
        "contacts": contacts,
        "pi05_result_file_sha256": file_sha256(pi05_path),
        "pi05_result_payload_sha256": pi05["result_payload_sha256"],
        "aegis_geometry_result_file_sha256": file_sha256(aegis_path),
        "aegis_geometry_result_payload_sha256": aegis["result_payload_sha256"],
        "contact_file_sha256": file_sha256(contact_path),
    }


def run(
    *, repo_root: Path, table1_root: Path, config_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        RESULT_SCHEMA, cpu_allocation_record, excluded_case_ids, load_config,
        load_population, payload_sha256, summarize,
    )

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path, repo_root=repo_root)
    _require(str(table1_root) == config["source"]["table1_root"], "pi05 Table1 root differs")
    rows = load_population(repo_root / config["source"]["population_manifest"], config)
    excluded = excluded_case_ids(repo_root, config)
    records = [
        classify_case(
            row, repo_root=repo_root, table1_root=table1_root,
            config=config, excluded=excluded,
        )
        for row in rows
    ]
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": identity,
        "allocation": cpu_allocation_record(),
        "config": config,
        "summary": summarize(records),
        "excluded_case_count": len(excluded),
        "records": records,
        "new_simulation_performed": False,
        "candidate_outcomes_accessed": False,
        "boundary_collection_authorized": False,
        "training_authorized": False,
        "correction_authorized": False,
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
