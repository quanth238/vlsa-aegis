"""Contracts for grouped nominal-risk and anchored-action-response collection."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_factorized_boundary.v1"
CLASSIFICATION_SCHEMA = "vlsa_distal_l5_factorized_boundary_classification.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_factorized_boundary_case_validation.v1"
SUMMARY_SCHEMA = "vlsa_distal_l5_factorized_boundary_summary.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str = "payload_sha256") -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "population",
        "execution", "method_bindings", "collection",
        "coverage_gate_before_learning", "next_model_if_coverage_passes",
        "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("factorized boundary config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-l5-factorized-boundary-v1"
        or value["population"]["case_count"] != 12
        or value["population"]["train_case_count"] != 8
        or value["population"]["validation_case_count"] != 4
        or value["population"]["reserved_unopened_initial_episode_indices_per_task"]
        != [31, 36]
    ):
        raise ValueError("factorized boundary protocol differs")
    root = Path(repo_root)
    population = value["population"]
    for name in ("manifest", "table1_manifest"):
        if file_sha256(root / population[name]) != population[name + "_file_sha256"]:
            raise ValueError("factorized boundary population binding differs: " + name)
    for name in (
        "clean_config", "query_coverage_config", "grouped_config",
        "adaptive_config", "base_config", "geometry_config",
    ):
        if file_sha256(root / value["method_bindings"][name]) != value[
            "method_bindings"
        ][name + "_file_sha256"]:
            raise ValueError("factorized boundary method binding differs: " + name)
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def load_manifest(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    population = config["population"]
    if len(rows) != population["case_count"]:
        raise ValueError("factorized boundary manifest count differs")
    seen_cases: set[str] = set()
    seen_groups: set[str] = set()
    split_counts: Counter[str] = Counter()
    task_episodes: dict[int, list[int]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        group_id = str(row["factorized_trajectory_group_id"])
        split = str(row["factorized_split"])
        task = int(row["logical_task_index"])
        episode = int(row["episode_index"])
        if case_id in seen_cases or group_id in seen_groups:
            raise ValueError("factorized boundary manifest identity repeats")
        seen_cases.add(case_id)
        seen_groups.add(group_id)
        split_counts[split] += 1
        task_episodes.setdefault(task, []).append(episode)
        if (
            task not in population["tasks"]
            or split not in ("train", "validation")
            or episode not in population["unused_initial_episode_indices_per_task"]
            or episode in population["reserved_unopened_initial_episode_indices_per_task"]
            or row["factorized_collection_run_id"]
            != "l5-factorized-boundary-20260814a"
            or group_id != case_id + "-noise-" + str(row["policy_noise_seed"])
            or row["required_arms"]
            != ["pi05_translational", "pi05_plus_aegis_translational"]
            or int(row["max_steps"]) != 300
            or int(row["replan_steps"]) != 5
        ):
            raise ValueError("factorized boundary manifest protocol differs")
    if split_counts != Counter({"train": 8, "validation": 4}):
        raise ValueError("factorized boundary split counts differ")
    if any(sorted(values) != [1, 6, 11, 16, 21, 26]
           for values in task_episodes.values()):
        raise ValueError("factorized boundary episode population differs")
    return rows


def classify_evaluation(
    *, row: Mapping[str, Any], result_path: Path, results_root: Path,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.moka_noise_discovery import (
        contact_evidence, geometry_evidence, validate_result_payload,
    )

    if not Path(result_path).is_file():
        return {
            "schema_version": CLASSIFICATION_SCHEMA,
            "case_id": row["case_id"], "split": row["factorized_split"],
            "classification": "MISSING_OR_FAILED", "eligible": False,
            "result_path": str(result_path),
        }
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    validate_result_payload(result)
    if (
        result["case_id"] != row["case_id"]
        or int(result["case"]["policy_noise_seed"]) != int(row["policy_noise_seed"])
    ):
        raise ValueError("factorized evaluation case binding differs")
    contact_path = Path(results_root) / result["failure_diagnostics"]["contacts"]["path"]
    if not contact_path.is_file():
        raise ValueError("factorized contact artifact is missing")
    if file_sha256(contact_path) != result["failure_diagnostics"]["contacts"]["sha256"]:
        raise ValueError("factorized contact artifact differs")
    geometry = geometry_evidence(result)
    contacts = contact_evidence(contact_path)
    bodies = contacts["robot_contact_counts_by_body"]
    dynamic_bad = bool(
        contacts["event_counts_by_role"].get("dynamic_task_object", 0)
        or contacts["event_counts_by_role"].get("dynamic_other", 0)
    )
    complete = bool(result.get("status") == "complete" and result.get("scientific_result"))
    task_success = bool(result.get("task_success"))
    link5 = int(bodies.get("robot0_link5", 0)) > 0
    eligible = bool(
        complete and task_success and geometry["proxy_valid"] and link5
        and not dynamic_bad and not contacts["settled_relevant_contact"]
    )
    if not complete:
        classification = "FAILED_OR_INCOMPLETE"
    elif not task_success:
        classification = "TASK_FAILURE"
    elif not geometry["proxy_valid"]:
        classification = "PROXY_INVALID"
    elif contacts["settled_relevant_contact"]:
        classification = "INITIALLY_UNSAFE"
    elif dynamic_bad:
        classification = "DYNAMIC_CONTACT_OUT_OF_SCOPE"
    elif link5:
        classification = "ELIGIBLE_TASK_SUCCESS_L5"
    elif bodies.get("robot0_link6", 0) or bodies.get("robot0_link7", 0):
        classification = "DISTAL_CONTACT_NON_L5"
    else:
        classification = "TASK_SUCCESS_NO_L5_CONTACT"
    first_l5 = contacts["first_robot_contact_step_by_body"].get("robot0_link5")
    output = {
        "schema_version": CLASSIFICATION_SCHEMA,
        "case_id": row["case_id"], "episode_index": row["episode_index"],
        "logical_task_index": row["logical_task_index"],
        "policy_noise_seed": row["policy_noise_seed"],
        "trajectory_group_id": row["factorized_trajectory_group_id"],
        "split": row["factorized_split"], "classification": classification,
        "eligible": eligible, "task_success": task_success,
        "complete_scientific_result": complete,
        "active_obstacle_name": result["obstacle"]["active_name"],
        "first_L5_contact_step": first_l5,
        "geometry": geometry, "contacts": contacts,
        "result_path": str(result_path),
        "result_file_sha256": file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "contact_path": str(contact_path),
        "contact_file_sha256": file_sha256(contact_path),
    }
    output["payload_sha256"] = payload_sha256(output)
    return output


def selected_case(
    *, row: Mapping[str, Any], classification: Mapping[str, Any],
    archived_relative_path: str,
) -> dict[str, Any]:
    if not classification["eligible"] or classification["first_L5_contact_step"] is None:
        raise ValueError("factorized boundary case is not eligible")
    return {
        "schema_version": "vlsa_distal_clean_action_risk_case.v1",
        "case_id": row["case_id"], "episode_index": row["episode_index"],
        "suite": row["suite"], "task_level_group_id": row["task_level_group_id"],
        "active_obstacle_name": classification["active_obstacle_name"],
        "archived_result_relative_path": archived_relative_path,
        "archived_result_file_sha256": classification["result_file_sha256"],
        "archived_result_payload_sha256": classification["result_payload_sha256"],
        "first_relevant_contact_step": classification["first_L5_contact_step"],
        "paper_car_step": classification["first_L5_contact_step"],
        "protected_contact_body": "robot0_link5",
        "split": row["factorized_split"],
        "policy_noise_seed": row["policy_noise_seed"],
        "trajectory_group_id": row["factorized_trajectory_group_id"],
        "source_static_eligibility": {
            "aegis_native_task_success": True,
            "aegis_paper_car_failure": True,
            "geometry_complete": True,
            "mvee_detached_from_retained_cloud": False,
            "mvee_materially_invalid": False,
            "mvee_solver_inaccurate": False,
            "postcontrol_dynamic_other_contact": False,
            "postcontrol_dynamic_task_contact": False,
            "postcontrol_robot_contact": True,
            "refined_collision_mechanism": "positive_h_unmodelled_arm_link_contact",
            "settled_relevant_contact": False,
        },
    }


def phase_coverage(result: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for candidate in result.get("candidates", []):
        prefix = [float(value) for value in candidate["candidate_prefix_risk"][:2]]
        backup_raw = candidate.get("backup_risk")
        backup = None if backup_raw is None else [float(value) for value in backup_raw[:2]]
        combined = [float(value) for value in candidate["combined_risk"][:2]]
        phases = [
            "prefix" if backup is None or prefix[row] >= backup[row] else "backup"
            for row in range(2)
        ]
        records.append({
            "candidate_name": candidate["name"],
            "candidate_order": int(candidate["order"]),
            "prefix": prefix, "backup": backup, "combined": combined,
            "active_phase_by_row": phases,
            "exact_safe": bool(candidate["exact_safe"]),
            "terminal_status": candidate["terminal_status"],
            "correction_l2_action": float(
                candidate["residual_binding"]["applied_residual_l2_action"]
            ),
        })
    known = [item for item in records if item["terminal_status"] != "UNKNOWN_TIMEOUT"]
    nominal = next((item for item in known if item["candidate_order"] == 0), None)
    phase_boundary = {"prefix": False, "backup": False}
    for phase in phase_boundary:
        for row in range(2):
            values = [
                item["combined"][row] for item in known
                if item["active_phase_by_row"][row] == phase
            ]
            if values and min(values) <= 0.0 < max(values):
                phase_boundary[phase] = True
    return {
        "known_candidate_count": len(known),
        "known_nominal": nominal is not None,
        "known_response_count": max(0, len(known) - int(nominal is not None)),
        "prefix_dominated_boundary": phase_boundary["prefix"],
        "backup_dominated_boundary": phase_boundary["backup"],
        "records": records,
    }
