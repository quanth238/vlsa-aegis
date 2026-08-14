"""Validation helpers for the outcome-blind L5 moka trajectory discovery."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_moka_noise_discovery.v1"
SUMMARY_SCHEMA = "vlsa_distal_l5_moka_noise_discovery_summary.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_moka_noise_discovery_validation.v1"
RESULT_SCHEMA = "vlsa_table1_episode_result.v1"
CONTACT_SCHEMA = "vlsa_table1_active_obstacle_contacts.v3"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("moka discovery config schema differs")
    expected_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    if value["selection"]["episode_indices"] != expected_indices:
        raise ValueError("moka discovery episode selection differs")
    if value["selection"]["maximum_concurrent_h100s"] != 2:
        raise ValueError("moka discovery GPU ceiling differs")
    if not value["execution"]["original_aegis_ee_correction_enabled"]:
        raise ValueError("original AEGIS EE correction must remain enabled")
    if value["classification"]["proxy_material_enclosure_limit"] != 1.01:
        raise ValueError("proxy material enclosure threshold differs")
    return value


def load_manifest(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = config["selection"]["episode_indices"]
    if len(rows) != len(expected):
        raise ValueError("moka discovery manifest row count differs")
    seen_ids: set[str] = set()
    for row, episode in zip(rows, expected):
        expected_seed = 2026081400 + 1024 * int(episode)
        if row["episode_index"] != episode or row["policy_noise_seed"] != expected_seed:
            raise ValueError("moka discovery outcome-blind seed binding differs")
        if row["case_id"] in seen_ids:
            raise ValueError("moka discovery repeats a case")
        seen_ids.add(row["case_id"])
        if not (
            row["suite"] == "safelibero_goal"
            and row["safety_level"] == "II"
            and row["logical_task_index"] == 0
            and row["max_steps"] == 300
            and row["replan_steps"] == 5
            and row["required_arms"]
            == ["pi05_translational", "pi05_plus_aegis_translational"]
        ):
            raise ValueError("moka discovery case protocol differs")
        expected_group = f"{row['case_id']}-noise-{expected_seed}"
        if row.get("discovery_trajectory_group_id") != expected_group:
            raise ValueError("moka discovery trajectory group differs")
    return rows


def validate_result_payload(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("episode result schema differs")
    claimed = result.get("result_payload_sha256")
    payload = dict(result)
    payload.pop("result_payload_sha256", None)
    if claimed != sha256_bytes(canonical(payload)):
        raise ValueError("episode result payload digest differs")


def geometry_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    record = result["failure_diagnostics"]["geometry"]["record"]
    filtering = record["filtering"]
    mvee = record["mvee"]
    if record.get("status") != "complete":
        raise ValueError("geometry diagnostics are incomplete")
    cloud_center = [float(item) for item in filtering["centroid_trim"]["centroid"]]
    cloud_radius = float(filtering["centroid_trim"]["distance_max"])
    mvee_center = [float(item) for item in mvee["center"]]
    semiaxes = [float(item) for item in mvee["semiaxes"]]
    obstacle = [float(item) for item in result["obstacle"]["initial_position"]]
    q_max = float(mvee["maximum_hull_quadratic_value"])
    center_cloud = math.sqrt(sum((a - b) ** 2 for a, b in zip(mvee_center, cloud_center)))
    invalid = q_max > 1.01
    detached = invalid and center_cloud - cloud_radius > max(semiaxes)
    statuses = [str(call.get("status", "")) for call in mvee.get("solver_calls", [])]
    cloud_active_xy = math.hypot(cloud_center[0] - obstacle[0], cloud_center[1] - obstacle[1])
    return {
        "maximum_hull_quadratic_value": q_max,
        "mvee_materially_invalid": invalid,
        "mvee_detached_from_retained_cloud": detached,
        "mvee_solver_inaccurate": "optimal_inaccurate" in statuses,
        "cloud_active_xy_distance_m": cloud_active_xy,
        "proxy_valid": bool(
            not invalid
            and not detached
            and "optimal_inaccurate" not in statuses
            and cloud_active_xy <= 0.1
        ),
    }


def contact_evidence(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("schema_version") != CONTACT_SCHEMA:
        raise ValueError("contact artifact schema differs")
    bodies: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    first_step: dict[str, int] = {}
    settled_relevant = False
    for snapshot in artifact["snapshots"]:
        for event in snapshot["events"]:
            other = event["other"]
            role = str(other["classification"])
            body = str(other["body_name"])
            step = int(event["step"])
            roles[role] += 1
            if role == "robot":
                bodies[body] += 1
                first_step[body] = min(first_step.get(body, step), step)
            if step < 0 and role in {"robot", "dynamic_task_object", "dynamic_other"}:
                settled_relevant = True
    return {
        "event_counts_by_role": dict(sorted(roles.items())),
        "robot_contact_counts_by_body": dict(sorted(bodies.items())),
        "first_robot_contact_step_by_body": dict(sorted(first_step.items())),
        "settled_relevant_contact": settled_relevant,
    }


def classify_case(
    *, row: Mapping[str, Any], result_path: Path, run_root: Path,
) -> dict[str, Any]:
    if not result_path.is_file():
        return {
            "case_id": row["case_id"],
            "episode_index": row["episode_index"],
            "trajectory_group_id": row["discovery_trajectory_group_id"],
            "classification": "MISSING_OR_FAILED",
            "eligible": False,
            "result_path": str(result_path),
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_result_payload(result)
    if result["case_id"] != row["case_id"]:
        raise ValueError("result case binding differs")
    if result["case"]["policy_noise_seed"] != row["policy_noise_seed"]:
        raise ValueError("result noise seed binding differs")
    contact_rel = Path(result["failure_diagnostics"]["contacts"]["path"])
    contact_path = run_root / f"case-{int(row['episode_index'] // 5):02d}" / "results" / contact_rel
    if not contact_path.is_file():
        raise ValueError("registered contact artifact is missing")
    if sha256_path(contact_path) != result["failure_diagnostics"]["contacts"]["sha256"]:
        raise ValueError("contact artifact digest differs")
    geometry = geometry_evidence(result)
    contacts = contact_evidence(contact_path)
    bodies = contacts["robot_contact_counts_by_body"]
    task_success = bool(result.get("task_success"))
    complete = bool(result.get("status") == "complete" and result.get("scientific_result"))
    dynamic_bad = bool(
        contacts["event_counts_by_role"].get("dynamic_task_object", 0)
        or contacts["event_counts_by_role"].get("dynamic_other", 0)
    )
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
    return {
        "case_id": row["case_id"],
        "episode_index": row["episode_index"],
        "policy_noise_seed": row["policy_noise_seed"],
        "trajectory_group_id": row["discovery_trajectory_group_id"],
        "classification": classification,
        "eligible": eligible,
        "task_success": task_success,
        "complete_scientific_result": complete,
        "geometry": geometry,
        "contacts": contacts,
        "result_file_sha256": sha256_path(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "contact_file_sha256": sha256_path(contact_path),
        "result_path": str(result_path),
    }


def summarize(
    *, rows: Sequence[Mapping[str, Any]], run_root: Path,
    config: Mapping[str, Any], source_commit: str,
) -> dict[str, Any]:
    records = []
    for array_index, row in enumerate(rows):
        result_path = (
            run_root / f"case-{array_index:02d}" / "results" / "aegis"
            / str(row["case_id"]) / "result.json"
        )
        records.append(classify_case(row=row, result_path=result_path, run_root=run_root))
    counts = Counter(record["classification"] for record in records)
    eligible = [record for record in records if record["eligible"]]
    threshold = int(config["downstream_gate"]["minimum_new_eligible_distinct_initial_states"])
    output = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated_discovery_population",
        "scientific_result": True,
        "source_commit": source_commit,
        "case_count": len(records),
        "classification_counts": dict(sorted(counts.items())),
        "eligible_case_count": len(eligible),
        "eligible_case_ids": [record["case_id"] for record in eligible],
        "downstream_boundary_collection_authorized": len(eligible) >= threshold,
        "records": records,
    }
    output["payload_sha256"] = sha256_bytes(canonical(output))
    return output


def validate_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise ValueError("discovery summary schema differs")
    payload = dict(summary)
    claimed = payload.pop("payload_sha256", None)
    if claimed != sha256_bytes(canonical(payload)):
        raise ValueError("discovery summary payload differs")
    if int(summary["case_count"]) != len(summary["records"]):
        raise ValueError("discovery summary case count differs")
    if int(summary["eligible_case_count"]) != sum(
        bool(record["eligible"]) for record in summary["records"]
    ):
        raise ValueError("discovery eligible count differs")
