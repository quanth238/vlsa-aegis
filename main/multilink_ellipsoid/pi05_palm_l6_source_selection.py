"""Contracts for selecting natural pi0.5 palm/L6 warning sources.

This audit reads only immutable Table-1 natural-policy results and raw contact
ledgers.  It never reads counterfactual candidate outcomes.  Its output is a
source pool; a separate, committed manifest freezes train/validation/test
identities before boundary collection.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_pi05_palm_l6_source_audit.v1"
RESULT_SCHEMA = "vlsa_distal_pi05_palm_l6_source_audit_result.v1"
TARGET_GROUPS = ("palm", "L6")


def cpu_allocation_record() -> dict[str, Any]:
    """Record a strict Slurm CPU allocation without requiring a GPU."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise ValueError("pi05 source audit requires a Slurm allocation")
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("pi05 source audit must remain CPU-only")
    return {
        "slurm_job_id": job_id,
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": socket.gethostname(),
        "device": {"type": "cpu"},
        "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
    }


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    public = dict(value)
    public.pop("result_payload_sha256", None)
    return hashlib.sha256(canonical(public)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "population", "constraint_groups", "target_groups", "eligibility",
        "previous_candidate_manifests", "warning_state_rule", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("pi05 palm/L6 source-audit config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-pi05-palm-l6-source-audit-v1"
        or value["source"]["arm"] != "pi05_translational"
        or value["source"]["nominal_action_source"]
        != "raw_pi05_nominal_translational"
        or value["source"]["released_AEGIS_EE_QP_enabled"] is not False
        or value["population"]["expected_case_count"] != 1600
        or value["target_groups"] != list(TARGET_GROUPS)
        or not set(TARGET_GROUPS).issubset(
            {row["name"] for row in value["constraint_groups"]}
        )
        or value["warning_state_rule"]["one_state_per_root_episode"] is not True
        or value["warning_state_rule"]["candidate_outcomes_accessed"] is not False
    ):
        raise ValueError("pi05 palm/L6 source-audit protocol differs")
    manifest = repo_root / value["source"]["population_manifest"]
    if file_sha256(manifest) != value["source"]["population_manifest_file_sha256"]:
        raise ValueError("pi05 source population manifest differs")
    for binding in value["previous_candidate_manifests"]:
        bound = repo_root / binding["path"]
        if file_sha256(bound) != binding["file_sha256"]:
            raise ValueError("pi05 previous candidate manifest differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def load_population(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(config["population"]["expected_case_count"]):
        raise ValueError("pi05 source population count differs")
    identities = [str(row["case_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("pi05 source population identities repeat")
    return sorted(rows, key=lambda row: int(row["case_ordinal"]))


def excluded_case_ids(repo_root: Path, config: Mapping[str, Any]) -> set[str]:
    identities: set[str] = set()
    for binding in config["previous_candidate_manifests"]:
        path = repo_root / binding["path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("case_id") is not None:
                identities.add(str(row["case_id"]))
    return identities


def result_path(table1_root: Path, row: Mapping[str, Any], arm: str) -> Path:
    task_ordinal = int(row["case_ordinal"]) // 50
    return (
        Path(table1_root) / "tasks" / f"task-{task_ordinal}" / "results"
        / arm / str(row["case_id"]) / "result.json"
    )


def _result_payload_valid(value: Mapping[str, Any]) -> bool:
    expected = value.get("result_payload_sha256")
    if not isinstance(expected, str):
        return False
    public = dict(value)
    public.pop("result_payload_sha256", None)
    return hashlib.sha256(canonical(public)).hexdigest() == expected


def _warning_step(first_contact_step: int) -> int:
    # Register the latest five-action query boundary whose nominal prefix ends
    # immediately before the first raw target contact.
    return ((int(first_contact_step) - 5) // 5) * 5


def _raw_action_contract(result: Mapping[str, Any], state_step: int) -> dict[str, Any]:
    actions = result.get("actions", [])
    indexed = {int(row.get("step", -1)): row for row in actions}
    required = set(range(int(state_step) + 5))
    complete = required.issubset(indexed)
    invariant = bool(complete)
    for step in sorted(required) if complete else []:
        row = indexed[step]
        invariant = bool(
            invariant
            and row.get("control_path") == "pi05_translational_nominal"
            and row.get("qp") is None
            and row.get("modified") is False
            and float(row.get("correction_l2", -1.0)) == 0.0
            and row.get("executed") == row.get("nominal_translational")
            and row.get("env_step_input") == row.get("nominal_translational")
        )
    query_index = int(state_step) // 5
    queries = result.get("policy_queries", [])
    query_bound = bool(
        query_index < len(queries)
        and int(queries[query_index].get("query_index", -1)) == query_index
    )
    return {
        "complete_through_five_action_chunk": complete,
        "raw_pi05_action_invariant": invariant,
        "query_index": query_index,
        "query_bound": query_bound,
        "action_count": len(actions),
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classifications = Counter(str(row["classification"]) for row in records)
    eligible = [row for row in records if row["eligible_target_groups"]]
    per_group: dict[str, Any] = {}
    for group in TARGET_GROUPS:
        rows = [row for row in eligible if group in row["eligible_target_groups"]]
        per_group[group] = {
            "eligible_episode_count": len(rows),
            "task_level_group_count": len({row["task_level_group_id"] for row in rows}),
            "task_success_count": sum(bool(row["task_success"]) for row in rows),
            "first_contact_step_minimum": min(
                [int(row["first_contact_step_by_group"][group]) for row in rows],
                default=None,
            ),
            "first_contact_step_maximum": max(
                [int(row["first_contact_step_by_group"][group]) for row in rows],
                default=None,
            ),
            "case_ids": [str(row["case_id"]) for row in rows],
        }
    return {
        "record_count": len(records),
        "classification_counts": dict(sorted(classifications.items())),
        "eligible_episode_count": len(eligible),
        "eligible_by_target_group": per_group,
        "candidate_outcomes_accessed": False,
    }


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the allocation-independent source-selection evidence."""
    return {
        "schema_version": result["schema_version"],
        "status": result["status"],
        "scientific_result": result["scientific_result"],
        "claim_scope": result["claim_scope"],
        "config_payload_sha256": result["config"]["config_payload_sha256"],
        "summary": result["summary"],
        "excluded_case_count": result["excluded_case_count"],
        "records": result["records"],
        "new_simulation_performed": result["new_simulation_performed"],
        "candidate_outcomes_accessed": result["candidate_outcomes_accessed"],
        "boundary_collection_authorized": result["boundary_collection_authorized"],
        "training_authorized": result["training_authorized"],
        "correction_authorized": result["correction_authorized"],
    }
