#!/usr/bin/env python3
"""Build the physical link-5/6 contact manifest from complete Table-1 results.

The report's existing 109-case manifest covers link contact observed through
the first CAR crossing.  This builder deliberately scans the complete contact
pair union for all 1,600 AEGIS episodes and emits a separate 165-case physical
contact population.  Actual contacted bodies are the first lineage element;
ancestors are never relabelled as the contacting link.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


SCHEMA = "vlsa_poisson_physical_link56_case.v1"
RECEIPT_SCHEMA = "vlsa_poisson_physical_link56_manifest_receipt.v1"
CONFIG_SCHEMA = "vlsa_poisson_arm_contact_study_protocol.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LINKS = {"robot0_link5", "robot0_link6"}


class PhysicalManifestError(RuntimeError):
    """Raised when source evidence or a generated row is inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PhysicalManifestError("%s must be a lowercase SHA-256" % label)
    return value


def _load_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PhysicalManifestError("%s is missing or symlinked" % label)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhysicalManifestError("%s is invalid JSON" % label) from error
    if not isinstance(value, dict):
        raise PhysicalManifestError("%s must contain one object" % label)
    return value


def _load_jsonl(path: Path, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n"):
                raise PhysicalManifestError(
                    "%s line %d lacks a terminal newline" % (label, line_number)
                )
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PhysicalManifestError(
                    "%s line %d is invalid" % (label, line_number)
                ) from error
            if not isinstance(row, dict):
                raise PhysicalManifestError("%s rows must be objects" % label)
            rows.append(row)
    return rows


def _obstacle_side(lineage: Sequence[str], active_name: str) -> bool:
    return any(
        body == active_name or body.startswith(active_name + "_")
        for body in lineage
    )


def extract_active_obstacle_robot_contacts(
    unique_pairs: Any, active_name: str
) -> Tuple[List[Dict[str, Any]], Tuple[str, ...]]:
    """Return direct link pairs and every actual robot body in the pair union."""

    if not isinstance(active_name, str) or not active_name:
        raise PhysicalManifestError("active obstacle name is invalid")
    if not isinstance(unique_pairs, list):
        raise PhysicalManifestError("unique contact pairs must be a list")
    direct: List[Dict[str, Any]] = []
    robot_bodies: Set[str] = set()
    for pair_index, pair in enumerate(unique_pairs):
        if not isinstance(pair, dict) or set(pair) != {
            "geom1",
            "geom2",
            "body_lineage1",
            "body_lineage2",
        }:
            raise PhysicalManifestError("contact pair %d has invalid fields" % pair_index)
        lineages = (pair["body_lineage1"], pair["body_lineage2"])
        geoms = (pair["geom1"], pair["geom2"])
        if any(
            not isinstance(lineage, list)
            or not lineage
            or any(not isinstance(body, str) or not body for body in lineage)
            for lineage in lineages
        ):
            raise PhysicalManifestError("contact pair %d has invalid lineage" % pair_index)
        orientations = []
        for robot_side in (0, 1):
            obstacle_side = 1 - robot_side
            robot_lineage = lineages[robot_side]
            obstacle_lineage = lineages[obstacle_side]
            if "robot0_base" not in robot_lineage or not _obstacle_side(
                obstacle_lineage, active_name
            ):
                continue
            actual_body = robot_lineage[0]
            robot_bodies.add(actual_body)
            if actual_body in _LINKS:
                orientations.append(
                    {
                        "actual_link_body_name": actual_body,
                        "link_geom_name": geoms[robot_side],
                        "active_obstacle_name": active_name,
                        "active_obstacle_actual_body_name": obstacle_lineage[0],
                        "active_obstacle_geom_name": geoms[obstacle_side],
                        "link_body_lineage": list(robot_lineage),
                        "active_obstacle_body_lineage": list(obstacle_lineage),
                        "historical_unique_contact_pair_sha256": sha256_bytes(
                            canonical_json_bytes(pair)
                        ),
                    }
                )
        if len(orientations) > 1:
            raise PhysicalManifestError("contact pair %d has ambiguous sides" % pair_index)
        direct.extend(orientations)
    return sorted(direct, key=canonical_json_bytes), tuple(sorted(robot_bodies))


def _payload_hash(result: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in result.items()
                if key != "result_payload_sha256" and not str(key).startswith("_")
            }
        )
    )


def _load_source_rows(path: Path, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    source = config["source"]
    if sha256_path(path) != source["table1_manifest_sha256"]:
        raise PhysicalManifestError("Table-1 source manifest hash differs")
    rows = _load_jsonl(path, "Table-1 source manifest")
    if len(rows) != source["table1_manifest_rows"]:
        raise PhysicalManifestError("Table-1 source manifest row count differs")
    if [row.get("case_ordinal") for row in rows] != list(range(len(rows))):
        raise PhysicalManifestError("Table-1 case ordinals are not contiguous")
    case_ids = [row.get("case_id") for row in rows]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise PhysicalManifestError("Table-1 source contains an invalid case ID")
    if len(case_ids) != len(set(case_ids)):
        raise PhysicalManifestError("Table-1 case IDs are not unique")
    return rows


def _load_audit(path: Path, expected_hash: str) -> Dict[str, Dict[str, str]]:
    if sha256_path(path) != expected_hash:
        raise PhysicalManifestError("case audit hash differs")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if len(rows) != 1600:
        raise PhysicalManifestError("case audit must contain 1,600 rows")
    by_id = {row.get("case_id", ""): row for row in rows}
    if len(by_id) != len(rows) or "" in by_id:
        raise PhysicalManifestError("case audit IDs are invalid or duplicated")
    return by_id


def _load_ledger(path: Path, expected_hash: str) -> Dict[str, Dict[str, Any]]:
    if sha256_path(path) != expected_hash:
        raise PhysicalManifestError("failure ledger hash differs")
    rows = _load_jsonl(path, "failure ledger")
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        declared = _require_hash(row.get("record_payload_sha256"), "ledger payload")
        payload = {
            key: value for key, value in row.items() if key != "record_payload_sha256"
        }
        if sha256_bytes(canonical_json_bytes(payload)) != declared:
            raise PhysicalManifestError("ledger record payload hash differs")
        if not isinstance(case_id, str) or not case_id or case_id in by_id:
            raise PhysicalManifestError("failure ledger case IDs are invalid")
        by_id[case_id] = row
    if len(by_id) != 1600:
        raise PhysicalManifestError("failure ledger must contain 1,600 cases")
    return by_id


def _result_index(root: Path, pattern: str) -> Dict[str, Path]:
    paths = sorted(root.glob(pattern))
    by_id = {path.parent.name: path for path in paths}
    if len(paths) != 1600 or len(by_id) != 1600:
        raise PhysicalManifestError("historical root must have 1,600 unique AEGIS results")
    return by_id


def _link_pattern(links: Iterable[str]) -> str:
    value = set(links)
    if value == {"robot0_link5"}:
        return "link5_only"
    if value == {"robot0_link6"}:
        return "link6_only"
    if value == _LINKS:
        return "link5_and_link6"
    raise PhysicalManifestError("selected row has an invalid link pattern")


def build_rows(
    *,
    config_path: Path,
    source_manifest: Path,
    case_audit: Path,
    historical_root: Path,
    report_manifest: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = _load_object(config_path, "study protocol")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise PhysicalManifestError("study protocol schema differs")
    source = config.get("source")
    if not isinstance(source, dict):
        raise PhysicalManifestError("study source contract is missing")
    for field in (
        "table1_manifest_sha256",
        "case_audit_sha256",
        "failure_case_ledger_sha256",
        "population_summary_sha256",
        "accepted_result_payloads_sha256",
        "accepted_aegis_payload_bindings_sha256",
        "report_link56_manifest_sha256",
    ):
        _require_hash(source.get(field), field)
    if sha256_path(report_manifest) != source["report_link56_manifest_sha256"]:
        raise PhysicalManifestError("report-aligned 109-case manifest hash differs")
    report_rows = _load_jsonl(report_manifest, "report-aligned manifest")
    report_ids = {row.get("case_id") for row in report_rows}
    if len(report_rows) != 109 or len(report_ids) != 109 or None in report_ids:
        raise PhysicalManifestError("report-aligned manifest must contain 109 cases")

    source_rows = _load_source_rows(source_manifest, config)
    audit = _load_audit(case_audit, source["case_audit_sha256"])
    ledger_path = historical_root / source["failure_case_ledger_relative_path"]
    ledger = _load_ledger(ledger_path, source["failure_case_ledger_sha256"])
    summary_path = historical_root / source["population_summary_relative_path"]
    if sha256_path(summary_path) != source["population_summary_sha256"]:
        raise PhysicalManifestError("population summary hash differs")
    population_summary = _load_object(summary_path, "population summary")
    if (
        population_summary.get("schema_version")
        != "vlsa_table1_population_summary.v1"
        or population_summary.get("status") != "complete_population_validated"
        or population_summary.get("population")
        != {
            "arms": 2,
            "cases": 1600,
            "no_results_dropped": True,
            "results": 3200,
            "task_level_groups": 32,
        }
        or population_summary.get("accepted_result_payloads_sha256")
        != source["accepted_result_payloads_sha256"]
    ):
        raise PhysicalManifestError("population summary completeness contract differs")
    results = _result_index(historical_root, source["historical_result_relative_glob"])
    source_ids = {row["case_id"] for row in source_rows}
    if set(audit) != source_ids or set(ledger) != source_ids or set(results) != source_ids:
        raise PhysicalManifestError("source, audit, ledger, and result joins differ")

    rows: List[Dict[str, Any]] = []
    accepted_aegis_payload_bindings: List[Dict[str, str]] = []
    for source_row in source_rows:
        case_id = source_row["case_id"]
        path = results[case_id]
        raw = path.read_bytes()
        try:
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PhysicalManifestError("%s result is invalid JSON" % case_id) from error
        if (
            not isinstance(result, dict)
            or result.get("schema_version") != source["historical_result_schema_version"]
            or result.get("status") != "complete"
            or result.get("scientific_result") is not True
            or result.get("case_id") != case_id
            or result.get("case_ordinal") != source_row["case_ordinal"]
            or result.get("case") != source_row
            or result.get("arm") != "pi05_plus_aegis_translational"
            or result.get("mode") != "aegis"
        ):
            raise PhysicalManifestError("%s historical result contract differs" % case_id)
        git = result.get("source", {}).get("git", {})
        if (
            git.get("commit") != source["historical_source_git_commit"]
            or git.get("dirty") is not False
        ):
            raise PhysicalManifestError("%s historical Git identity differs" % case_id)
        declared_payload = _require_hash(
            result.get("result_payload_sha256"), case_id + " result payload"
        )
        if _payload_hash(result) != declared_payload:
            raise PhysicalManifestError("%s result payload hash differs" % case_id)
        ledger_row = ledger[case_id]
        ledger_outcome = ledger_row.get("outcomes", {}).get(
            "pi05_plus_aegis_translational"
        )
        if (
            not isinstance(ledger_outcome, dict)
            or ledger_outcome.get("status") != "complete"
            or ledger_outcome.get("result_payload_sha256") != declared_payload
        ):
            raise PhysicalManifestError(
                "%s result is not bound to the authoritative failure ledger"
                % case_id
            )
        accepted_aegis_payload_bindings.append(
            {
                "case_id": case_id,
                "arm": "pi05_plus_aegis_translational",
                "result_payload_sha256": declared_payload,
            }
        )
        actions = result.get("actions")
        action_ledger = result.get("action_invariance_ledger")
        if not isinstance(actions, list) or not isinstance(action_ledger, dict):
            raise PhysicalManifestError("%s action ledger is invalid" % case_id)
        action_count = action_ledger.get("action_count")
        sequence_hash = _require_hash(
            action_ledger.get("executed_sequence_sha256"), case_id + " action sequence"
        )
        if (
            isinstance(action_count, bool)
            or not isinstance(action_count, int)
            or action_count != len(actions)
            or sha256_bytes(canonical_json_bytes([row.get("executed") for row in actions]))
            != sequence_hash
        ):
            raise PhysicalManifestError("%s action sequence differs" % case_id)

        active_name = result.get("obstacle", {}).get("active_name")
        contact = result.get("contact_telemetry")
        if not isinstance(contact, dict) or contact.get("status") != "available":
            raise PhysicalManifestError("%s contact telemetry is unavailable" % case_id)
        direct_pairs, robot_bodies = extract_active_obstacle_robot_contacts(
            contact.get("unique_contact_pairs"), active_name
        )
        if not direct_pairs:
            continue
        links = sorted({pair["actual_link_body_name"] for pair in direct_pairs})
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise PhysicalManifestError("%s metrics are invalid" % case_id)
        paper_collision = metrics.get("paper_collision")
        if not isinstance(paper_collision, bool):
            raise PhysicalManifestError("%s paper collision value is invalid" % case_id)
        if case_id in report_ids:
            phase = "through_car"
            if paper_collision is not True:
                raise PhysicalManifestError("%s report case is CAR-safe" % case_id)
        elif paper_collision:
            phase = "post_car"
        else:
            phase = "car_safe"

        link_only = set(robot_bodies).issubset(_LINKS)
        first_link_step = contact.get("first_contact_step") if link_only else None
        barrier_h = None
        correction_l2 = None
        aegis_modified = None
        constraint_lhs = None
        if first_link_step is not None:
            if (
                isinstance(first_link_step, bool)
                or not isinstance(first_link_step, int)
                or first_link_step < 0
                or first_link_step >= len(actions)
            ):
                raise PhysicalManifestError("%s first link step is invalid" % case_id)
            action = actions[first_link_step]
            qp = action.get("qp")
            if not isinstance(qp, dict):
                raise PhysicalManifestError("%s first link action lacks AEGIS QP" % case_id)
            barrier_h = qp.get("barrier_h")
            correction_l2 = action.get("correction_l2")
            aegis_modified = action.get("modified")
            constraint_lhs = qp.get("constraint_lhs")
            values = (barrier_h, correction_l2, constraint_lhs)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ) or not isinstance(aegis_modified, bool):
                raise PhysicalManifestError("%s first link AEGIS values are invalid" % case_id)

        audit_row = audit[case_id]
        settled = audit_row.get("aegis_settled_relevant_contact") == "True"
        if audit_row.get("aegis_settled_relevant_contact") not in {"True", "False"}:
            raise PhysicalManifestError("%s settled audit flag is invalid" % case_id)
        task_success = result.get("task_success")
        if not isinstance(task_success, bool):
            raise PhysicalManifestError("%s task success value is invalid" % case_id)
        clean = bool(
            link_only
            and not settled
            and barrier_h is not None
            and float(barrier_h) > 0.0
            and task_success
        )
        detailed = contact.get("detailed_artifact")
        video = result.get("video")
        pairing = result.get("pairing")
        if not isinstance(detailed, dict) or not isinstance(video, dict) or not isinstance(pairing, dict):
            raise PhysicalManifestError("%s artifact bindings are invalid" % case_id)
        if (
            detailed.get("role_authority_complete") is not True
            or detailed.get("snapshot_count") != action_count + 1
            or video.get("complete_episode") is not True
            or video.get("frames") != action_count + 1
        ):
            raise PhysicalManifestError(
                "%s complete contact/video coverage differs" % case_id
            )
        artifact_hashes = {
            "detailed_contact_gzip_sha256": _require_hash(
                detailed.get("sha256"), case_id + " contact gzip"
            ),
            "detailed_contact_payload_sha256": _require_hash(
                detailed.get("uncompressed_payload_sha256"), case_id + " contact payload"
            ),
            "video_sha256": _require_hash(video.get("sha256"), case_id + " video"),
        }
        pairing_fields = {}
        for field in (
            "initial_state_sha256",
            "initial_observation_sha256",
            "settled_simulator_state_sha256",
            "policy_noise_schedule_sha256",
            "manifest_row_sha256",
            "semantic_label_record_sha256",
        ):
            pairing_fields[field] = _require_hash(
                pairing.get(field), case_id + " pairing " + field
            )
        expected_manifest_hash = sha256_bytes(canonical_json_bytes(source_row))
        if pairing_fields["manifest_row_sha256"] != expected_manifest_hash:
            raise PhysicalManifestError("%s paired source row hash differs" % case_id)
        row = {
            "schema_version": SCHEMA,
            "protocol_id": config["protocol_id"],
            "case_id": case_id,
            "source_case_ordinal": source_row["case_ordinal"],
            "source_case": source_row,
            "source_case_sha256": expected_manifest_hash,
            "active_obstacle_name": active_name,
            "contact_vs_car_phase": phase,
            "sample_phase": "post_env_step_control_endpoint",
            "literal_link_contact_pattern": _link_pattern(links),
            "literal_link_contact_bodies": links,
            "selected_obstacle_robot_contact_bodies": list(robot_bodies),
            "only_link56_selected_obstacle_robot_contact": link_only,
            "direct_link_active_obstacle_pairs": direct_pairs,
            "direct_link_active_obstacle_pairs_sha256": sha256_bytes(
                canonical_json_bytes(direct_pairs)
            ),
            "first_sampled_link_contact_control_step": first_link_step,
            "first_link_step_is_timing_resolved": link_only,
            "paired_precontrol_aegis_at_first_link_step": {
                "barrier_h": barrier_h,
                "constraint_lhs": constraint_lhs,
                "correction_l2": correction_l2,
                "modified": aegis_modified,
            },
            "historical_outcome": {
                "paper_car_failure": paper_collision,
                "paper_car_first_crossing_step": metrics.get("collision_first_step"),
                "maximum_active_obstacle_l1_displacement_m": metrics.get(
                    "maximum_active_obstacle_l1_displacement_m"
                ),
                "task_success": task_success,
                "executed_action_count": action_count,
            },
            "settled_relevant_contact": settled,
            "clean_task_success_canary_eligible": clean,
            "historical_result": {
                "relative_path": path.relative_to(historical_root).as_posix(),
                "file_sha256": sha256_bytes(raw),
                "payload_sha256": declared_payload,
                "executed_action_sequence_sha256": sequence_hash,
                "ledger_record_payload_sha256": _require_hash(
                    ledger_row.get("record_payload_sha256"), case_id + " ledger"
                ),
            },
            "artifact_bindings": {
                **artifact_hashes,
                "detailed_contact_relative_path": detailed.get("path"),
                "video_relative_path": video.get("path"),
                "video_frames": video.get("frames"),
                "video_complete_episode": video.get("complete_episode"),
            },
            "pairing": pairing_fields,
            "claim_scope": "post_hoc_physical_contact_mechanism_population_not_unbiased_benchmark",
        }
        rows.append(row)

    rows.sort(key=lambda row: row["source_case_ordinal"])
    accepted_aegis_payload_bindings_sha256 = sha256_bytes(
        canonical_json_bytes(accepted_aegis_payload_bindings)
    )
    if (
        accepted_aegis_payload_bindings_sha256
        != source["accepted_aegis_payload_bindings_sha256"]
    ):
        raise PhysicalManifestError("accepted AEGIS payload binding hash differs")
    expected = config["populations"]["full_episode_physical_link56_contact"]
    if len(rows) != expected["expected_cases"]:
        raise PhysicalManifestError("physical link population count differs")
    if not report_ids.issubset({row["case_id"] for row in rows}):
        raise PhysicalManifestError("report-aligned population is not a physical subset")
    clean_rows = [row for row in rows if row["clean_task_success_canary_eligible"]]
    clean_contract = config["populations"]["clean_task_success_canary_population"]
    if len(clean_rows) != clean_contract["expected_cases"]:
        raise PhysicalManifestError("clean canary population count differs")
    all_case_ids_sha256 = sha256_bytes(
        canonical_json_bytes([row["case_id"] for row in rows])
    )
    clean_case_ids_sha256 = sha256_bytes(
        canonical_json_bytes([row["case_id"] for row in clean_rows])
    )
    if all_case_ids_sha256 != expected["expected_case_ids_sha256"]:
        raise PhysicalManifestError("physical case-ID ledger hash differs")
    if clean_case_ids_sha256 != clean_contract["expected_case_ids_sha256"]:
        raise PhysicalManifestError("clean case-ID ledger hash differs")

    def counts(field: str, values: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for value in values:
            key = str(value[field])
            result[key] = result.get(key, 0) + 1
        return result

    phase_counts = counts("contact_vs_car_phase", rows)
    suite_counts = counts("suite", [row["source_case"] for row in rows])
    pattern_counts = counts("literal_link_contact_pattern", rows)
    if phase_counts != expected["phase_counts"]:
        raise PhysicalManifestError("physical phase counts differ")
    if suite_counts != expected["suite_counts"]:
        raise PhysicalManifestError("physical suite counts differ")
    if pattern_counts != expected["link_pattern_counts"]:
        raise PhysicalManifestError("physical link-pattern counts differ")
    if sum(row["historical_outcome"]["task_success"] for row in rows) != expected[
        "task_success_cases"
    ]:
        raise PhysicalManifestError("physical task-success count differs")
    if sum(not row["historical_outcome"]["task_success"] for row in rows) != expected[
        "task_failure_cases"
    ]:
        raise PhysicalManifestError("physical task-failure count differs")
    if sum(row["only_link56_selected_obstacle_robot_contact"] for row in rows) != expected[
        "only_link56_robot_contact_cases"
    ]:
        raise PhysicalManifestError("link-only robot-contact count differs")
    if sum(not row["only_link56_selected_obstacle_robot_contact"] for row in rows) != expected[
        "mixed_robot_contact_cases"
    ]:
        raise PhysicalManifestError("mixed robot-contact count differs")
    if counts("contact_vs_car_phase", clean_rows) != clean_contract["phase_counts"]:
        raise PhysicalManifestError("clean phase counts differ")
    clean_suite_counts = {
        suite: sum(row["source_case"]["suite"] == suite for row in clean_rows)
        for suite in clean_contract["suite_counts"]
    }
    if clean_suite_counts != clean_contract["suite_counts"]:
        raise PhysicalManifestError("clean suite counts differ")
    if counts("literal_link_contact_pattern", clean_rows) != clean_contract["link_pattern_counts"]:
        raise PhysicalManifestError("clean link-pattern counts differ")

    rows_by_id = {row["case_id"]: row for row in rows}
    representative_ids: Set[str] = set()
    for representative in config.get("representative_cases", []):
        if not isinstance(representative, dict):
            raise PhysicalManifestError("representative case contract is invalid")
        case_id = representative.get("case_id")
        if (
            not isinstance(case_id, str)
            or case_id in representative_ids
            or case_id not in rows_by_id
        ):
            raise PhysicalManifestError("representative case identity is invalid")
        representative_ids.add(case_id)
        row = rows_by_id[case_id]
        expected_body = representative.get("expected_first_link_body")
        evidence_kind = representative.get("expected_body_evidence")
        if (
            representative.get("expected_phase") != row["contact_vs_car_phase"]
            or representative.get("historical_task_success")
            is not row["historical_outcome"]["task_success"]
            or expected_body not in _LINKS
            or expected_body not in row["literal_link_contact_bodies"]
        ):
            raise PhysicalManifestError(
                "%s representative outcome contract differs" % case_id
            )
        if evidence_kind == "case_audit_robot_contact_bodies_through_car":
            audited_bodies = {
                value.strip()
                for value in audit[case_id]["robot_contact_bodies"].split(";")
                if value.strip()
            }
            if audited_bodies != {expected_body}:
                raise PhysicalManifestError(
                    "%s representative through-CAR body differs" % case_id
                )
            audited_step = audit[case_id]["first_relevant_contact_step"]
            if (
                audited_step == ""
                or int(audited_step)
                != row["first_sampled_link_contact_control_step"]
            ):
                raise PhysicalManifestError(
                    "%s representative first-contact step differs" % case_id
                )
        elif evidence_kind == "full_episode_actual_body_union":
            if (
                row["literal_link_contact_bodies"] != [expected_body]
                or row["first_link_step_is_timing_resolved"] is not True
            ):
                raise PhysicalManifestError(
                    "%s representative full-episode body differs" % case_id
                )
        else:
            raise PhysicalManifestError(
                "%s representative body evidence is unknown" % case_id
            )

    metadata = {
        "config_sha256": sha256_path(config_path),
        "source_manifest_sha256": sha256_path(source_manifest),
        "case_audit_sha256": sha256_path(case_audit),
        "failure_ledger_sha256": sha256_path(ledger_path),
        "population_summary_sha256": sha256_path(summary_path),
        "accepted_result_payloads_sha256": source[
            "accepted_result_payloads_sha256"
        ],
        "accepted_aegis_payload_bindings_sha256": (
            accepted_aegis_payload_bindings_sha256
        ),
        "report_manifest_sha256": sha256_path(report_manifest),
        "phase_counts": phase_counts,
        "suite_counts": suite_counts,
        "link_pattern_counts": pattern_counts,
        "clean_cases": len(clean_rows),
        "clean_case_ids_sha256": clean_case_ids_sha256,
        "all_case_ids_sha256": all_case_ids_sha256,
    }
    return rows, metadata


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--case-audit", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--report-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    rows, metadata = build_rows(
        config_path=arguments.config.resolve(),
        source_manifest=arguments.source_manifest.resolve(),
        case_audit=arguments.case_audit.resolve(),
        historical_root=arguments.historical_root.resolve(),
        report_manifest=arguments.report_manifest.resolve(),
    )
    manifest_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    manifest_sha256 = sha256_bytes(manifest_payload)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_id": rows[0]["protocol_id"],
        "manifest_schema_version": SCHEMA,
        "manifest_rows": len(rows),
        "manifest_sha256": manifest_sha256,
        "first_case_id": rows[0]["case_id"],
        "last_case_id": rows[-1]["case_id"],
        **metadata,
    }
    _atomic_write(arguments.output.resolve(), manifest_payload)
    _atomic_write(
        arguments.receipt.resolve(),
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
