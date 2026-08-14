"""Contracts for the preregistered grouped L5 boundary extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .clean_action_risk import CASE_SCHEMA


CONFIG_SCHEMA = "vlsa_distal_l5_targeted_extension.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "target_population",
        "parent_method", "parent_population", "requested_coverage",
        "sealed_test_cases", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("targeted L5 extension config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("targeted L5 extension schema differs")
    if value["protocol_id"] != "vlsa-distal-l5-targeted-extension-v1":
        raise ValueError("targeted L5 extension protocol differs")
    target = value["target_population"]
    if (
        int(target["episode_count"]) != 5
        or target["split_counts"] != {"train": 3, "validation": 2}
        or target["one_state_per_episode"] is not True
        or target["split_assignment_fixed_before_rollout_labels"] is not True
        or target["no_row_label_used_for_selection"] is not True
        or len(target["case_ids_in_registered_order"]) != 5
        or len(set(target["case_ids_in_registered_order"])) != 5
    ):
        raise ValueError("targeted L5 population contract differs")
    if value["sealed_test_cases"] != [
        "vlsa-t1-goal-ii-t2-e42",
        "vlsa-t1-goal-ii-t3-e42",
        "vlsa-t1-goal-ii-t3-e44",
    ]:
        raise ValueError("targeted L5 sealed tests differ")
    if set(target["case_ids_in_registered_order"]) & set(value["sealed_test_cases"]):
        raise ValueError("targeted L5 extension opens a sealed test")
    if value["forbidden"] != {
        "Table1_artifact_change": True,
        "sealed_test_episode_opening": True,
        "model_training": True,
        "calibration": True,
        "QP": True,
        "closed_loop": True,
        "timeout_as_safe": True,
        "dropping_failed_states": True,
    }:
        raise ValueError("targeted L5 forbidden set differs")
    if repo_root is not None:
        root = Path(repo_root)
        parent = value["parent_method"]
        for name in ("clean", "coverage", "grouped", "adaptive"):
            config_path = root / parent[name + "_config"]
            if _file_sha256(config_path) != parent[name + "_config_file_sha256"]:
                raise ValueError("targeted L5 parent %s config differs" % name)
        manifest = root / target["manifest"]
        if _file_sha256(manifest) != target["manifest_file_sha256"]:
            raise ValueError("targeted L5 manifest differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def load_cases(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    target = config["target_population"]
    if hashlib.sha256(raw).hexdigest() != target["manifest_file_sha256"]:
        raise ValueError("targeted L5 manifest hash differs")
    rows = [
        json.loads(line) for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(target["episode_count"]):
        raise ValueError("targeted L5 episode count differs")
    if [row.get("case_id") for row in rows] != target["case_ids_in_registered_order"]:
        raise ValueError("targeted L5 case order differs")
    counts = {
        split: sum(row.get("split") == split for row in rows)
        for split in target["split_counts"]
    }
    if counts != target["split_counts"]:
        raise ValueError("targeted L5 split counts differ")
    required = {
        "schema_version", "case_id", "split", "suite", "task_level_group_id",
        "episode_index", "case_ordinal", "active_obstacle_name",
        "protected_contact_body", "first_relevant_contact_step",
        "paper_car_step", "archived_result_relative_path",
        "archived_result_file_sha256", "archived_result_payload_sha256",
        "source_static_eligibility",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("targeted L5 case keys differ")
        if row["schema_version"] != CASE_SCHEMA:
            raise ValueError("targeted L5 case schema differs")
        if row["split"] not in ("train", "validation"):
            raise ValueError("targeted L5 split differs")
        if row["protected_contact_body"] != "robot0_link5":
            raise ValueError("targeted extension is not an L5 contact case")
        eligibility = row["source_static_eligibility"]
        if not (
            eligibility["aegis_native_task_success"]
            and eligibility["aegis_paper_car_failure"]
            and eligibility["geometry_complete"]
            and eligibility["postcontrol_robot_contact"]
            and not eligibility["settled_relevant_contact"]
            and not eligibility["postcontrol_dynamic_task_contact"]
            and not eligibility["postcontrol_dynamic_other_contact"]
            and not eligibility["mvee_materially_invalid"]
            and not eligibility["mvee_detached_from_retained_cloud"]
            and not eligibility["mvee_solver_inaccurate"]
            and eligibility["refined_collision_mechanism"]
            == "positive_h_unmodelled_arm_link_contact"
        ):
            raise ValueError("targeted L5 static eligibility differs")
    return rows
