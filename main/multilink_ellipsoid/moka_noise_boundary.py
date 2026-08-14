"""Contracts for boundary collection on validated task-0 moka trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONFIG_SCHEMA = "vlsa_distal_l5_moka_noise_boundary.v1"
CASE_SCHEMA = "vlsa_distal_l5_moka_noise_boundary_case.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source", "cohort",
        "method_bindings", "coverage_rule", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("moka boundary config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("moka boundary config schema differs")
    if value["protocol_id"] != "vlsa-distal-l5-moka-noise-boundary-v1":
        raise ValueError("moka boundary protocol differs")
    if value["source"]["eligible_case_count"] != 4:
        raise ValueError("moka boundary eligible count differs")
    cohort = value["cohort"]
    if cohort["development_cases"] != [
        "vlsa-t1-goal-ii-t0-e00", "vlsa-t1-goal-ii-t0-e45"
    ]:
        raise ValueError("moka boundary development cohort differs")
    if cohort["diagnostic_cases"] != [
        "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10"
    ]:
        raise ValueError("moka boundary diagnostic cohort differs")
    root = Path(repo_root)
    for name in (
        "population_manifest", "query_coverage_config", "adaptive_config",
        "grouped_config", "base_config", "geometry_config",
    ):
        relative = value["method_bindings"][name]
        expected = value["method_bindings"][name + "_file_sha256"]
        if _sha256(root / relative) != expected:
            raise ValueError("moka boundary %s differs" % name)
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def load_cases(path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    rows = [
        json.loads(line) for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(config["source"]["eligible_case_count"]):
        raise ValueError("moka boundary case count differs")
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    development = set(config["cohort"]["development_cases"])
    diagnostic = set(config["cohort"]["diagnostic_cases"])
    for index, row in enumerate(rows):
        required = {
            "active_obstacle_name", "archived_result_file_sha256",
            "archived_result_payload_sha256", "archived_result_relative_path",
            "case_id", "case_index", "episode_index",
            "first_relevant_contact_step", "policy_noise_seed",
            "schema_version", "split", "trajectory_group_id",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("moka boundary case keys differ")
        if row["schema_version"] != CASE_SCHEMA:
            raise ValueError("moka boundary case schema differs")
        if row["case_id"] in seen_ids or row["trajectory_group_id"] in seen_groups:
            raise ValueError("moka boundary identity is duplicated")
        seen_ids.add(row["case_id"])
        seen_groups.add(row["trajectory_group_id"])
        if row["case_id"] in development:
            expected_split = "train" if row["case_id"].endswith("e00") else "validation"
        elif row["case_id"] in diagnostic:
            expected_split = "diagnostic"
        else:
            raise ValueError("moka boundary case is outside cohort")
        if row["split"] != expected_split:
            raise ValueError("moka boundary split differs")
        if int(row["case_index"]) not in (0, 1, 2, 9):
            raise ValueError("moka boundary source index differs")
        if int(row["first_relevant_contact_step"]) <= 0:
            raise ValueError("moka boundary contact step differs")
        if row["active_obstacle_name"] != "moka_pot_obstacle_1":
            raise ValueError("moka boundary obstacle differs")
        if index > 0 and rows[index - 1]["episode_index"] >= row["episode_index"]:
            raise ValueError("moka boundary cases are not ordered")
    return rows


def validate_discovery_binding(
    *, config: dict[str, Any], cases: list[dict[str, Any]],
    discovery_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(discovery_root)
    summary_path = root / "summary.json"
    validation_path = root / "validation.json"
    source = config["source"]
    if _sha256(summary_path) != source["summary_file_sha256"]:
        raise ValueError("moka discovery summary file differs")
    if _sha256(validation_path) != source["validation_file_sha256"]:
        raise ValueError("moka discovery validation file differs")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if summary["result_payload_sha256"] != source["summary_payload_sha256"]:
        raise ValueError("moka discovery summary payload differs")
    if validation["payload_sha256"] != source["validation_payload_sha256"]:
        raise ValueError("moka discovery validation payload differs")
    if validation["status"] != "validated" or not validation["scientific_result"]:
        raise ValueError("moka discovery is not validated")
    eligible = {
        record["case_id"]: record for record in summary["records"]
        if record["eligible"]
    }
    if set(eligible) != {case["case_id"] for case in cases}:
        raise ValueError("moka discovery eligible cases differ")
    for case in cases:
        record = eligible[case["case_id"]]
        if record["result_file_sha256"] != case["archived_result_file_sha256"]:
            raise ValueError("moka discovery case file differs")
        if record["result_payload_sha256"] != case["archived_result_payload_sha256"]:
            raise ValueError("moka discovery case payload differs")
        if record["trajectory_group_id"] != case["trajectory_group_id"]:
            raise ValueError("moka discovery trajectory group differs")
    return summary, validation
