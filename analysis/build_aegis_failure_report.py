#!/usr/bin/env python3
"""Build an exhaustive, evidence-scoped AEGIS failure report.

This module intentionally separates three claims:

* the paper-compatible outcome (CAR / TSR / legacy ETS);
* directly observed diagnostic evidence;
* causal hypotheses that the evidence cannot uniquely identify.

The population publisher is responsible for deeply validating every result
and diagnostic artifact.  This module consumes those validated result
records one pair at a time and produces one compact row per SafeLIBERO case.
Every AEGIS CAR failure receives exactly one observed primary class.  A class
named ``*_without_sampled_contact`` is an explicit evidence limit, not a
claim that no physical contact occurred between control-step snapshots.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

try:
    from analysis import aggregate_safelibero_aegis as aggregate
except ImportError:  # pragma: no cover - direct script execution fallback
    import aggregate_safelibero_aegis as aggregate  # type: ignore[no-redef]


CASE_SCHEMA = "vlsa_table1_aegis_failure_case.v1"
REPORT_SCHEMA = "vlsa_table1_aegis_failure_report.v1"
CONTACT_SCHEMA = "vlsa_table1_active_obstacle_contacts.v3"
CONTACT_MODEL_AUTHORITY_SCHEMA = "vlsa_table1_contact_model_authority.v2"
SUMMARY_SCHEMA = "vlsa_table1_population_summary.v1"
PREPUBLISH_SCHEMA = "vlsa_table1_population_prepublish_validation.v2"
EXPECTED_CASES = 1600
EXPECTED_RESULTS = 3200
STRICT_ZERO_TRANSLATION_EPS = 1e-9

BASELINE_ARM = "pi05_translational"
AEGIS_ARM = "pi05_plus_aegis_translational"

CONTACT_ROLES = (
    "robot",
    "static_support",
    "dynamic_task_object",
    "dynamic_other",
    "unknown",
)
COLLISION_RELEVANT_CONTACT_ROLES = (
    "robot",
    "dynamic_task_object",
    "dynamic_other",
)

CAR_FAILURE_CLASSES = {
    "not_car_failure",
    "grounding_no_points_fail_open_collision",
    "filter_removed_all_points_fail_open_collision",
    "preexisting_settled_robot_or_dynamic_contact_then_collision",
    "valid_qp_mixed_collision_relevant_contact_roles",
    "valid_qp_robot_contact_positive_proxy_barrier",
    "valid_qp_robot_contact_nonpositive_proxy_barrier",
    "valid_qp_dynamic_task_object_contact_collision",
    "valid_qp_dynamic_other_contact_collision",
    "valid_qp_displacement_without_robot_or_dynamic_sampled_contact",
}

EVIDENCE_LIMITED_CAR_CLASSES = {
    "valid_qp_displacement_without_robot_or_dynamic_sampled_contact",
}

INTERPRETATION_SCOPE = {
    "car": (
        "Paper-compatible CAR is active-obstacle L1 displacement strictly "
        "greater than 1 mm, not direct physical contact."
    ),
    "contacts": (
        "MuJoCo contacts are sampled after each control step. Absence of a "
        "sampled event does not prove absence during internal physics substeps."
    ),
    "contact_roles": (
        "Active-obstacle contact roles come from MuJoCo body/joint ancestry "
        "plus native task-object and BDDL-goal membership. Static support "
        "contact is reported but is not treated as a collision mechanism."
    ),
    "proxy_barrier": (
        "AEGIS barrier_h evaluates the released static perceived-obstacle "
        "ellipsoid against its end-effector proxy; it is not simulator "
        "whole-arm clearance."
    ),
    "causality": (
        "Primary classes are observed pipeline/outcome strata. They do not "
        "uniquely separate perception error, stale geometry, unmodelled arm "
        "geometry, and discrete-time controller effects."
    ),
}


class FailureReportError(RuntimeError):
    """Raised when a supposedly complete report lacks authoritative evidence."""


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


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FailureReportError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FailureReportError(f"{label} must be a list")
    return value


def _bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise FailureReportError(f"{label} must be Boolean")
    return value


def _int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise FailureReportError(f"{label} must be an integer")
    return value


def _number(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise FailureReportError(f"{label} must be finite")
    return float(value)


def _sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FailureReportError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FailureReportError(f"{label} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise FailureReportError(f"{label} must remain relative")
    return path


def _vector_norm(values: Any, *, label: str) -> float:
    sequence = values
    if not isinstance(sequence, (list, tuple)) or len(sequence) < 3:
        raise FailureReportError(f"{label} must contain three coordinates")
    coordinates = [_number(sequence[index], label=f"{label}[{index}]") for index in range(3)]
    return math.sqrt(sum(value * value for value in coordinates))


def _result_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(result.get("case_id"))
    metrics = _mapping(result.get("metrics"), label=f"{case_id}/metrics")
    return {
        "paper_collision": _bool(
            metrics.get("paper_collision"),
            label=f"{case_id}/paper_collision",
        ),
        "task_success": _bool(
            metrics.get("task_success"),
            label=f"{case_id}/task_success",
        ),
        "legacy_ets_steps": _int(
            metrics.get("legacy_ets_steps"),
            label=f"{case_id}/legacy_ets_steps",
        ),
        "executed_action_count": _int(
            metrics.get("executed_action_count"),
            label=f"{case_id}/executed_action_count",
        ),
        "collision_first_step": (
            None
            if metrics.get("collision_first_step") is None
            else _int(
                metrics.get("collision_first_step"),
                label=f"{case_id}/collision_first_step",
            )
        ),
        "maximum_active_obstacle_l1_displacement_m": _number(
            metrics.get("maximum_active_obstacle_l1_displacement_m"),
            label=f"{case_id}/maximum displacement",
        ),
        "safety_by_no_execution": _bool(
            metrics.get("safety_by_no_execution"),
            label=f"{case_id}/safety_by_no_execution",
        ),
    }


def _diagnostics_record(result: Mapping[str, Any]) -> Mapping[str, Any]:
    case_id = str(result.get("case_id"))
    record = _mapping(
        result.get("failure_diagnostics"),
        label=f"{case_id}/failure_diagnostics",
    )
    if (
        record.get("schema_version")
        != "vlsa_table1_aegis_failure_diagnostics.v1"
        or record.get("enabled") is not True
        or record.get("status") != "published"
        or record.get("control_effect") != "read_only_observation"
    ):
        raise FailureReportError(
            f"{case_id}: failure diagnostics are not validated/published"
        )
    return record


def load_contact_payload(
    result: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    """Load the hash-bound detailed contact artifact for one result."""

    case_id = str(result.get("case_id"))
    descriptor = _mapping(
        _diagnostics_record(result).get("contacts"),
        label=f"{case_id}/contacts descriptor",
    )
    obstacle = _mapping(
        result.get("obstacle"), label=f"{case_id}/active obstacle"
    )
    active_obstacle_name = obstacle.get("active_name")
    if (
        not isinstance(active_obstacle_name, str)
        or not active_obstacle_name
        or descriptor.get("schema_version") != CONTACT_SCHEMA
        or descriptor.get("active_obstacle_name") != active_obstacle_name
        or descriptor.get("format") != "canonical_json_gzip_mtime_zero"
        or descriptor.get("role_taxonomy") != list(CONTACT_ROLES)
        or descriptor.get("role_authority_complete") is not True
    ):
        raise FailureReportError(f"{case_id}: contact descriptor changed")
    relative = _safe_relative_path(
        descriptor.get("path"), label=f"{case_id}/contacts path"
    )
    root = artifact_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FailureReportError(
            f"{case_id}: contact artifact escapes its result root"
        ) from error
    if (
        not path.is_file()
        or sha256_path(path)
        != _sha(descriptor.get("sha256"), label=f"{case_id}/contacts SHA")
    ):
        raise FailureReportError(f"{case_id}: contact artifact hash mismatch")
    try:
        raw = gzip.decompress(path.read_bytes())
        payload = json.loads(raw)
    except Exception as error:
        raise FailureReportError(
            f"{case_id}: contact artifact cannot be decoded"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CONTACT_SCHEMA
        or payload.get("case_id") != case_id
        or payload.get("active_obstacle_name") != active_obstacle_name
        or sha256_bytes(raw)
        != _sha(
            descriptor.get("uncompressed_payload_sha256"),
            label=f"{case_id}/contact payload SHA",
        )
    ):
        raise FailureReportError(f"{case_id}: contact payload binding changed")
    model_authority = payload.get("model_authority")
    if not isinstance(model_authority, Mapping):
        raise FailureReportError(
            f"{case_id}: contact model authority is missing"
        )
    authority_hash = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in model_authority.items()
                if key != "authority_sha256"
            }
        )
    )
    task_context = model_authority.get("task_context")
    task_context_hash = (
        sha256_bytes(canonical_json_bytes(task_context))
        if isinstance(task_context, Mapping)
        else None
    )
    if (
        model_authority.get("schema_version")
        != CONTACT_MODEL_AUTHORITY_SCHEMA
        or model_authority.get("active_obstacle_name")
        != active_obstacle_name
        or model_authority.get("authority_sha256") != authority_hash
        or descriptor.get("model_authority_sha256") != authority_hash
        or descriptor.get("active_obstacle_root_body_id")
        != model_authority.get("active_obstacle_root_body_id")
        or descriptor.get("task_context_sha256") != task_context_hash
        or model_authority.get("task_context_sha256") != task_context_hash
    ):
        raise FailureReportError(
            f"{case_id}: contact model authority binding changed"
        )
    snapshots = _list(
        payload.get("snapshots"), label=f"{case_id}/contact snapshots"
    )
    if (
        len(snapshots) != descriptor.get("snapshot_count")
        or any(
            not isinstance(snapshot, Mapping)
            or snapshot.get("active_obstacle_name") != active_obstacle_name
            for snapshot in snapshots
        )
    ):
        raise FailureReportError(f"{case_id}: contact snapshot count changed")
    return payload


def _contact_evidence(
    payload: Mapping[str, Any],
    *,
    collision_step: int | None,
) -> dict[str, Any]:
    snapshots = _list(payload.get("snapshots"), label="contact snapshots")
    all_events: list[Mapping[str, Any]] = []
    through_collision: list[Mapping[str, Any]] = []
    settled_events: list[Mapping[str, Any]] = []
    for snapshot in snapshots:
        snapshot_map = _mapping(snapshot, label="contact snapshot")
        role_authority = _mapping(
            snapshot_map.get("role_authority"),
            label="contact role authority",
        )
        if (
            role_authority.get("status") != "complete"
            or role_authority.get("role_classes") != list(CONTACT_ROLES)
        ):
            raise FailureReportError(
                "contact snapshot lacks complete authoritative roles"
            )
        step = _int(snapshot_map.get("step"), label="contact snapshot step")
        events = _list(snapshot_map.get("events"), label="contact events")
        for event in events:
            event_map = _mapping(event, label="contact event")
            if _int(event_map.get("step"), label="contact event step") != step:
                raise FailureReportError("contact event/snapshot step mismatch")
            all_events.append(event_map)
            if step == -1:
                settled_events.append(event_map)
            if collision_step is not None and 0 <= step <= collision_step:
                through_collision.append(event_map)

    def classes(events: Sequence[Mapping[str, Any]]) -> Counter[str]:
        output: Counter[str] = Counter()
        for event in events:
            other = _mapping(event.get("other"), label="contact other")
            classification = str(other.get("classification"))
            if (
                classification not in CONTACT_ROLES
                or classification == "unknown"
                or other.get("classification_authority") != "complete"
            ):
                raise FailureReportError(
                    "contact other role is not complete and canonical"
                )
            output[classification] += 1
        return output

    relevant_classes = classes(through_collision)
    settled_classes = classes(settled_events)
    bodies_by_role: dict[str, list[str]] = {}
    for role in CONTACT_ROLES[:-1]:
        body_names: set[str] = set()
        for event in through_collision:
            other = _mapping(event.get("other"), label="contact other")
            if other.get("classification") != role:
                continue
            body_name = other.get("body_name")
            if not isinstance(body_name, str) or not body_name:
                raise FailureReportError(
                    f"{role} contact lacks a canonical body name"
                )
            body_names.add(body_name)
        bodies_by_role[role] = sorted(body_names)
    settled_relevant_count = sum(
        settled_classes[role]
        for role in COLLISION_RELEVANT_CONTACT_ROLES
    )
    through_collision_relevant_count = sum(
        relevant_classes[role]
        for role in COLLISION_RELEVANT_CONTACT_ROLES
    )
    return {
        "total_event_count": len(all_events),
        "settled_event_count": len(settled_events),
        "settled_event_counts_by_role": {
            role: settled_classes[role] for role in CONTACT_ROLES
        },
        "settled_collision_relevant_event_count": (
            settled_relevant_count
        ),
        "events_through_collision_count": len(through_collision),
        "event_counts_through_collision_by_role": {
            role: relevant_classes[role] for role in CONTACT_ROLES
        },
        "collision_relevant_events_through_collision": (
            through_collision_relevant_count
        ),
        "bodies_through_collision_by_role": bodies_by_role,
        "static_support_events_excluded_from_collision_role": (
            relevant_classes["static_support"]
        ),
        "role_authority_complete": True,
    }


def _geometry_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(result.get("case_id"))
    diagnostics = _diagnostics_record(result)
    descriptor = _mapping(
        diagnostics.get("geometry"), label=f"{case_id}/geometry descriptor"
    )
    record = _mapping(
        descriptor.get("record"), label=f"{case_id}/geometry record"
    )
    status = str(record.get("status"))
    if status not in {
        "complete",
        "failure",
        "method_failure_passthrough",
    }:
        raise FailureReportError(
            f"{case_id}: AEGIS geometry status is not terminal"
        )
    views = _mapping(record.get("views"), label=f"{case_id}/geometry views")
    view_rows: dict[str, Any] = {}
    for view in ("agentview", "backview"):
        value = _mapping(views.get(view), label=f"{case_id}/{view}")
        detections = value.get("detections")
        detection_count = None
        selected_index = None
        if isinstance(detections, Mapping):
            detection_count = _int(
                detections.get("count"),
                label=f"{case_id}/{view}/detection count",
            )
            selected_index = detections.get("selected_index")
        view_rows[view] = {
            "status": value.get("status"),
            "detection_count": detection_count,
            "selected_index": selected_index,
        }
    filtering = record.get("filtering")
    filtering_counts = None
    if isinstance(filtering, Mapping) and isinstance(
        filtering.get("counts"), Mapping
    ):
        filtering_counts = {
            str(key): _int(
                value, label=f"{case_id}/filtering/{key}"
            )
            for key, value in filtering["counts"].items()
        }
    mvee = record.get("mvee")
    failure = record.get("failure")
    return {
        "status": status,
        "views": view_rows,
        "filtering_status": (
            filtering.get("status")
            if isinstance(filtering, Mapping)
            else None
        ),
        "filtering_counts": filtering_counts,
        "mvee_status": (
            mvee.get("status") if isinstance(mvee, Mapping) else None
        ),
        "failure_component": (
            failure.get("component")
            if isinstance(failure, Mapping)
            else None
        ),
        "failure_type": (
            failure.get("type")
            if isinstance(failure, Mapping)
            else None
        ),
    }


def _control_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(result.get("case_id"))
    actions = _list(result.get("actions"), label=f"{case_id}/actions")
    metrics = _result_metrics(result)
    if len(actions) != metrics["executed_action_count"]:
        raise FailureReportError(f"{case_id}: action count differs from metrics")
    control_paths: Counter[str] = Counter()
    qp_solved = 0
    qp_barriers: list[float] = []
    qp_constraint_lhs: list[float] = []
    nominal_translation_sum = 0.0
    executed_translation_sum = 0.0
    any_nominal_nonzero = False
    all_executed_zero = True
    collision_action: Mapping[str, Any] | None = None

    for index, raw_action in enumerate(actions):
        action = _mapping(raw_action, label=f"{case_id}/action {index}")
        if _int(action.get("step"), label=f"{case_id}/action step") != index:
            raise FailureReportError(f"{case_id}: action steps are not contiguous")
        path = str(action.get("control_path"))
        control_paths[path] += 1
        nominal_norm = _vector_norm(
            action.get("nominal_translational"),
            label=f"{case_id}/action {index}/nominal",
        )
        executed_norm = _vector_norm(
            action.get("executed"),
            label=f"{case_id}/action {index}/executed",
        )
        nominal_translation_sum += nominal_norm
        executed_translation_sum += executed_norm
        any_nominal_nonzero = (
            any_nominal_nonzero
            or nominal_norm > STRICT_ZERO_TRANSLATION_EPS
        )
        all_executed_zero = (
            all_executed_zero
            and executed_norm <= STRICT_ZERO_TRANSLATION_EPS
        )
        qp = action.get("qp")
        if path == "aegis_qp":
            qp_map = _mapping(qp, label=f"{case_id}/action {index}/qp")
            if (
                qp_map.get("status") != "solved"
                or str(qp_map.get("solver_status")).lower()
                not in {"optimal", "optimal_inaccurate"}
            ):
                raise FailureReportError(
                    f"{case_id}: executed AEGIS QP is not solved"
                )
            qp_solved += 1
            qp_barriers.append(
                _number(
                    qp_map.get("barrier_h"),
                    label=f"{case_id}/action {index}/barrier_h",
                )
            )
            qp_constraint_lhs.append(
                _number(
                    qp_map.get("constraint_lhs"),
                    label=f"{case_id}/action {index}/constraint_lhs",
                )
            )
        if metrics["collision_first_step"] == index:
            collision_action = action

    failure = result.get("method_failure")
    terminal_qp_failure = None
    if isinstance(failure, Mapping) and failure.get("component") == "aegis_qp":
        context = _mapping(
            failure.get("diagnostics"),
            label=f"{case_id}/terminal QP diagnostics",
        )
        terminal_qp_failure = {
            "step": failure.get("step"),
            "failure_type": context.get("failure_type"),
            "solver_status": context.get("solver_status"),
        }

    collision_control_path = (
        None if collision_action is None else collision_action.get("control_path")
    )
    collision_barrier_h = None
    collision_constraint_lhs = None
    if (
        collision_action is not None
        and isinstance(collision_action.get("qp"), Mapping)
    ):
        collision_barrier_h = _number(
            collision_action["qp"].get("barrier_h"),
            label=f"{case_id}/collision barrier_h",
        )
        collision_constraint_lhs = _number(
            collision_action["qp"].get("constraint_lhs"),
            label=f"{case_id}/collision constraint_lhs",
        )
    return {
        "control_path_counts": dict(sorted(control_paths.items())),
        "qp_solved_action_count": qp_solved,
        "qp_barrier_h_min": min(qp_barriers) if qp_barriers else None,
        "qp_constraint_lhs_min": (
            min(qp_constraint_lhs) if qp_constraint_lhs else None
        ),
        "terminal_qp_failure": terminal_qp_failure,
        "collision_step_control_path": collision_control_path,
        "collision_step_barrier_h": collision_barrier_h,
        "collision_step_constraint_lhs": collision_constraint_lhs,
        "nominal_translation_l2_sum": nominal_translation_sum,
        "executed_translation_l2_sum": executed_translation_sum,
        "translation_retention_ratio": (
            None
            if nominal_translation_sum <= STRICT_ZERO_TRANSLATION_EPS
            else executed_translation_sum / nominal_translation_sum
        ),
        "strict_zero_translation": (
            bool(actions)
            and any_nominal_nonzero
            and all_executed_zero
        ),
    }


def _goal_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(result.get("case_id"))
    goal = _mapping(
        result.get("goal_progress"), label=f"{case_id}/goal progress"
    )
    summary = _mapping(
        goal.get("summary"), label=f"{case_id}/goal progress summary"
    )
    return {
        "goal_definition_sha256": _sha(
            goal.get("goal_definition_sha256"),
            label=f"{case_id}/goal definition SHA",
        ),
        "initial_satisfied_count": _int(
            summary.get("initial_satisfied_count"),
            label=f"{case_id}/initial goal count",
        ),
        "final_satisfied_count": _int(
            summary.get("final_satisfied_count"),
            label=f"{case_id}/final goal count",
        ),
        "maximum_satisfied_count": _int(
            summary.get("maximum_satisfied_count"),
            label=f"{case_id}/maximum goal count",
        ),
        "initial_fraction": _number(
            summary.get("initial_fraction"),
            label=f"{case_id}/initial goal fraction",
        ),
        "final_fraction": _number(
            summary.get("final_fraction"),
            label=f"{case_id}/final goal fraction",
        ),
        "maximum_fraction": _number(
            summary.get("maximum_fraction"),
            label=f"{case_id}/maximum goal fraction",
        ),
        "regression_count": _int(
            summary.get("regression_count"),
            label=f"{case_id}/goal regression count",
        ),
    }


def _video_evidence(
    result: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    case_id = str(result.get("case_id"))
    video = _mapping(result.get("video"), label=f"{case_id}/video")
    relative = _safe_relative_path(
        video.get("path"), label=f"{case_id}/video path"
    )
    root = artifact_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FailureReportError(
            f"{case_id}: video escapes its artifact root"
        ) from error
    expected = _sha(video.get("sha256"), label=f"{case_id}/video SHA")
    if not path.is_file() or sha256_path(path) != expected:
        raise FailureReportError(f"{case_id}: video file/hash mismatch")
    if video.get("complete_episode") is not True:
        raise FailureReportError(f"{case_id}: video is not a complete episode")
    return {
        "relative_path": relative.as_posix(),
        "absolute_path": str(path),
        "sha256": expected,
        "bytes": path.stat().st_size,
        "frames": _int(video.get("frames"), label=f"{case_id}/video frames"),
        "fps": _int(video.get("fps"), label=f"{case_id}/video fps"),
        "terminal_source_array_sha256": _sha(
            video.get("terminal_source_array_sha256"),
            label=f"{case_id}/video terminal source SHA",
        ),
        "hash_verified": True,
    }


def classify_car_failure(
    *,
    metrics: Mapping[str, Any],
    geometry: Mapping[str, Any],
    control: Mapping[str, Any],
    contacts: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """Return one exhaustive observed class and non-causal hypothesis tags."""

    if metrics["paper_collision"] is False:
        return "not_car_failure", []
    failure_type = geometry.get("failure_type")
    if (
        geometry.get("status") == "method_failure_passthrough"
        and failure_type == "no_grounded_points"
    ):
        return (
            "grounding_no_points_fail_open_collision",
            ["perception_fail_open"],
        )
    if (
        geometry.get("status") == "method_failure_passthrough"
        and failure_type == "point_filter_removed_all_points"
    ):
        return (
            "filter_removed_all_points_fail_open_collision",
            ["fixed_point_filter_fail_open"],
        )
    if contacts["settled_collision_relevant_event_count"] > 0:
        return (
            "preexisting_settled_robot_or_dynamic_contact_then_collision",
            [
                "benchmark_starts_in_sampled_active_obstacle_contact_with_"
                "robot_or_dynamic_body"
            ],
        )
    if control.get("collision_step_control_path") != "aegis_qp":
        raise FailureReportError(
            "colliding AEGIS case has no registered QP/fail-open path"
        )
    counts = _mapping(
        contacts.get("event_counts_through_collision_by_role"),
        label="contact role counts through collision",
    )
    robot = _int(
        counts.get("robot"), label="robot contact events through collision"
    )
    dynamic_task = _int(
        counts.get("dynamic_task_object"),
        label="dynamic task-object events through collision",
    )
    dynamic_other = _int(
        counts.get("dynamic_other"),
        label="dynamic other events through collision",
    )
    active_roles = sum(
        value > 0 for value in (robot, dynamic_task, dynamic_other)
    )
    if active_roles > 1:
        return (
            "valid_qp_mixed_collision_relevant_contact_roles",
            [
                "released_proxy_does_not_cover_all_observed_interactions",
                "multiple_robot_or_dynamic_contact_roles_observed",
            ],
        )
    if robot:
        barrier = control.get("collision_step_barrier_h")
        if barrier is None:
            raise FailureReportError(
                "QP collision lacks its proxy barrier value"
            )
        if float(barrier) > 0.0:
            return (
                "valid_qp_robot_contact_positive_proxy_barrier",
                [
                    "proxy_simulator_geometry_mismatch_observed",
                    "whole_arm_or_stale_perception_hypothesis",
                ],
            )
        return (
            "valid_qp_robot_contact_nonpositive_proxy_barrier",
            [
                "proxy_was_already_nonpositive_at_collision_step",
                "discrete_recovery_limit_hypothesis",
            ],
        )
    if dynamic_task:
        return (
            "valid_qp_dynamic_task_object_contact_collision",
            [
                "active_obstacle_contact_with_native_goal_object_observed",
                "task_object_interaction_outside_released_proxy_hypothesis",
            ],
        )
    if dynamic_other:
        return (
            "valid_qp_dynamic_other_contact_collision",
            [
                "active_obstacle_contact_with_other_dynamic_body_observed",
                "dynamic_scene_interaction_outside_released_proxy_hypothesis",
            ],
        )
    return (
        "valid_qp_displacement_without_robot_or_dynamic_sampled_contact",
        [
            "paper_displacement_contact_disagreement_observed",
            "metric_proxy_or_contact_substep_sampling_hypothesis",
        ],
    )


def classify_task_failure(
    *,
    baseline_metrics: Mapping[str, Any],
    aegis_metrics: Mapping[str, Any],
    control: Mapping[str, Any],
) -> str:
    if aegis_metrics["task_success"]:
        return "not_task_failure"
    if aegis_metrics["executed_action_count"] == 0:
        return "no_execution_method_failure"
    if (
        not aegis_metrics["paper_collision"]
        and control["strict_zero_translation"]
    ):
        return "safe_strict_zero_translation_task_failure"
    if not aegis_metrics["paper_collision"]:
        return (
            "safe_task_regression_vs_successful_baseline"
            if baseline_metrics["task_success"]
            else "safe_shared_task_failure"
        )
    return (
        "collision_task_regression_vs_successful_baseline"
        if baseline_metrics["task_success"]
        else "collision_shared_task_failure"
    )


def _joint_outcome(metrics: Mapping[str, Any]) -> str:
    safe = not metrics["paper_collision"]
    success = metrics["task_success"]
    return "_".join(
        (
            "safe" if safe else "collision",
            "task_success" if success else "task_failure",
        )
    )


def build_case_record(
    *,
    manifest: Mapping[str, Any],
    baseline: Mapping[str, Any],
    aegis: Mapping[str, Any],
    baseline_artifact_root: Path,
    aegis_artifact_root: Path,
) -> dict[str, Any]:
    """Build one validated compact pair row from full episode results."""

    case_id = str(manifest.get("case_id"))
    if (
        baseline.get("case_id") != case_id
        or aegis.get("case_id") != case_id
        or baseline.get("arm") != BASELINE_ARM
        or aegis.get("arm") != AEGIS_ARM
    ):
        raise FailureReportError(f"{case_id}: paired result identity changed")
    baseline_pairing = _mapping(
        baseline.get("pairing"), label=f"{case_id}/baseline pairing"
    )
    aegis_pairing = _mapping(
        aegis.get("pairing"), label=f"{case_id}/AEGIS pairing"
    )
    for field in (
        "initial_state_sha256",
        "initial_observation_sha256",
        "settled_simulator_state_sha256",
        "policy_noise_schedule_sha256",
        "semantic_label_record_sha256",
    ):
        if baseline_pairing.get(field) != aegis_pairing.get(field):
            raise FailureReportError(f"{case_id}: pair differs for {field}")

    baseline_metrics = _result_metrics(baseline)
    aegis_metrics = _result_metrics(aegis)
    _diagnostics_record(baseline)
    geometry = _geometry_evidence(aegis)
    control = _control_evidence(aegis)
    contacts = _contact_evidence(
        load_contact_payload(aegis, artifact_root=aegis_artifact_root),
        collision_step=aegis_metrics["collision_first_step"],
    )
    car_class, hypotheses = classify_car_failure(
        metrics=aegis_metrics,
        geometry=geometry,
        control=control,
        contacts=contacts,
    )
    if car_class not in CAR_FAILURE_CLASSES:
        raise FailureReportError(f"{case_id}: unregistered CAR failure class")
    if (car_class == "not_car_failure") is aegis_metrics["paper_collision"]:
        raise FailureReportError(f"{case_id}: CAR class/outcome mismatch")
    task_class = classify_task_failure(
        baseline_metrics=baseline_metrics,
        aegis_metrics=aegis_metrics,
        control=control,
    )
    record = {
        "schema_version": CASE_SCHEMA,
        "case_id": case_id,
        "case_ordinal": manifest.get("case_ordinal"),
        "task_level_group_id": manifest.get("task_level_group_id"),
        "suite": manifest.get("suite"),
        "safety_level": manifest.get("safety_level"),
        "logical_task_index": manifest.get("logical_task_index"),
        "resolved_task_index": manifest.get("resolved_task_index"),
        "task_name": manifest.get("task_name"),
        "episode_index": manifest.get("episode_index"),
        "active_obstacle_name": aegis.get("obstacle", {}).get("active_name"),
        "frozen_obstacle_label": aegis_pairing.get(
            "semantic_obstacle_label"
        ),
        "pair_binding": {
            field: baseline_pairing.get(field)
            for field in (
                "initial_state_sha256",
                "initial_observation_sha256",
                "settled_simulator_state_sha256",
                "policy_noise_schedule_sha256",
                "semantic_label_record_sha256",
            )
        },
        "outcomes": {
            BASELINE_ARM: {
                **baseline_metrics,
                "joint_outcome": _joint_outcome(baseline_metrics),
                "status": baseline.get("status"),
                "result_payload_sha256": _sha(
                    baseline.get("result_payload_sha256"),
                    label=f"{case_id}/baseline result payload",
                ),
            },
            AEGIS_ARM: {
                **aegis_metrics,
                "joint_outcome": _joint_outcome(aegis_metrics),
                "status": aegis.get("status"),
                "result_payload_sha256": _sha(
                    aegis.get("result_payload_sha256"),
                    label=f"{case_id}/AEGIS result payload",
                ),
            },
        },
        "paired_transition": {
            "car": (
                ("baseline_collision" if baseline_metrics["paper_collision"] else "baseline_safe")
                + "_to_"
                + ("aegis_collision" if aegis_metrics["paper_collision"] else "aegis_safe")
            ),
            "task": (
                ("baseline_success" if baseline_metrics["task_success"] else "baseline_failure")
                + "_to_"
                + ("aegis_success" if aegis_metrics["task_success"] else "aegis_failure")
            ),
        },
        "aegis_diagnostics": {
            "geometry": geometry,
            "control": control,
            "contacts": contacts,
            "goal_progress": _goal_evidence(aegis),
            "intervention": dict(
                _mapping(
                    aegis.get("intervention"),
                    label=f"{case_id}/intervention",
                )
            ),
        },
        "failure_analysis": {
            "is_aegis_car_failure": aegis_metrics["paper_collision"],
            "primary_observed_car_class": car_class,
            "evidence_limited": car_class in EVIDENCE_LIMITED_CAR_CLASSES,
            "causal_hypothesis_tags": hypotheses,
            "aegis_task_failure_class": task_class,
            "strict_safety_by_stopping": (
                not aegis_metrics["paper_collision"]
                and not aegis_metrics["task_success"]
                and (
                    aegis_metrics["executed_action_count"] == 0
                    or control["strict_zero_translation"]
                )
            ),
        },
        "videos": {
            BASELINE_ARM: _video_evidence(
                baseline, artifact_root=baseline_artifact_root
            ),
            AEGIS_ARM: _video_evidence(
                aegis, artifact_root=aegis_artifact_root
            ),
        },
        "interpretation_scope": INTERPRETATION_SCOPE,
    }
    record["record_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(record)
    )
    return record


def validate_report_against_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    expected_cases: int = EXPECTED_CASES,
) -> dict[str, Any]:
    """Require exhaustive rows and exact alignment with the population table."""

    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status") != "complete_population_validated"
        or summary.get("population", {}).get("no_results_dropped") is not True
    ):
        raise FailureReportError("summary is not a complete strict population")
    if len(records) != expected_cases:
        raise FailureReportError(
            f"failure report has {len(records)} rows, expected {expected_cases}"
        )
    case_ids = [str(record.get("case_id")) for record in records]
    if len(set(case_ids)) != expected_cases:
        raise FailureReportError("failure report case identities are not unique")
    primary = Counter(
        str(record["failure_analysis"]["primary_observed_car_class"])
        for record in records
    )
    if set(primary) - CAR_FAILURE_CLASSES:
        raise FailureReportError("failure report contains unknown CAR classes")
    for record in records:
        outcome_collision = record["outcomes"][AEGIS_ARM][
            "paper_collision"
        ]
        registered_failure = record["failure_analysis"][
            "is_aegis_car_failure"
        ]
        primary_class = record["failure_analysis"][
            "primary_observed_car_class"
        ]
        if (
            type(outcome_collision) is not bool
            or type(registered_failure) is not bool
            or registered_failure is not outcome_collision
            or (primary_class != "not_car_failure")
            is not outcome_collision
        ):
            raise FailureReportError(
                f"{record.get('case_id')}: CAR outcome/class mismatch"
            )
    car_failures = sum(
        bool(record["failure_analysis"]["is_aegis_car_failure"])
        for record in records
    )
    classified_failures = sum(
        count
        for key, count in primary.items()
        if key != "not_car_failure"
    )
    if classified_failures != car_failures:
        raise FailureReportError(
            "not every AEGIS CAR failure has exactly one primary class"
        )

    suite_rows: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        suite_rows[str(record.get("suite"))].append(record)
    summary_suites = _mapping(
        summary.get("suites"), label="summary suites"
    )
    aegis_suites = _mapping(
        summary_suites.get(AEGIS_ARM), label="AEGIS suite summary"
    )
    for suite, rows in suite_rows.items():
        expected = _mapping(
            aegis_suites.get(suite), label=f"{suite}/summary"
        )
        if len(rows) != expected.get("episodes"):
            raise FailureReportError(f"{suite}: episode count differs")
        observed_car = (
            100.0
            * sum(
                not row["outcomes"][AEGIS_ARM]["paper_collision"]
                for row in rows
            )
            / len(rows)
        )
        observed_tsr = (
            100.0
            * sum(
                row["outcomes"][AEGIS_ARM]["task_success"]
                for row in rows
            )
            / len(rows)
        )
        observed_ets = (
            sum(
                row["outcomes"][AEGIS_ARM]["legacy_ets_steps"]
                for row in rows
            )
            / len(rows)
        )
        for key, observed in (
            ("car_percent", observed_car),
            ("tsr_percent", observed_tsr),
            ("legacy_ets_steps_mean", observed_ets),
        ):
            target = _number(
                expected.get(key), label=f"{suite}/{key}"
            )
            if not math.isclose(
                observed, target, rel_tol=0.0, abs_tol=1e-9
            ):
                raise FailureReportError(
                    f"{suite}/{key} differs from the table summary"
                )

    video_rows = [
        video
        for record in records
        for video in record["videos"].values()
    ]
    if (
        len(video_rows) != expected_cases * 2
        or any(video.get("hash_verified") is not True for video in video_rows)
    ):
        raise FailureReportError("complete hash-verified video index is absent")

    task_classes = Counter(
        str(record["failure_analysis"]["aegis_task_failure_class"])
        for record in records
    )
    car_hypotheses = Counter(
        tag
        for record in records
        if record["failure_analysis"]["is_aegis_car_failure"]
        for tag in record["failure_analysis"]["causal_hypothesis_tags"]
    )
    return {
        "case_count": expected_cases,
        "video_count": len(video_rows),
        "aegis_car_failure_count": car_failures,
        "aegis_car_percent": 100.0 * (expected_cases - car_failures) / expected_cases,
        "primary_car_failure_class_counts": dict(sorted(primary.items())),
        "evidence_limited_car_failure_count": sum(
            primary[key] for key in EVIDENCE_LIMITED_CAR_CLASSES
        ),
        "aegis_task_failure_class_counts": dict(sorted(task_classes.items())),
        "car_failure_hypothesis_tag_counts": dict(
            sorted(car_hypotheses.items())
        ),
        "unclassified_aegis_car_failures": 0,
        "all_videos_hash_verified": True,
    }


def build_report(
    records: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    summary_sha256: str,
    validation_receipt_sha256: str,
    expected_cases: int = EXPECTED_CASES,
    streaming_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    counts = validate_report_against_summary(
        records,
        summary=summary,
        expected_cases=expected_cases,
    )
    ordered_rows = sorted(
        records, key=lambda row: int(row.get("case_ordinal"))
    )
    row_ledger = [
        {
            "case_id": row["case_id"],
            "record_payload_sha256": row["record_payload_sha256"],
        }
        for row in ordered_rows
    ]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete_population_failure_analysis",
        "protocol_id": summary.get("protocol_id"),
        "source": {
            "population_summary_sha256": _sha(
                summary_sha256, label="population summary SHA"
            ),
            "population_validation_receipt_sha256": _sha(
                validation_receipt_sha256,
                label="population validation receipt SHA",
            ),
            "accepted_result_payloads_sha256": _sha(
                summary.get("accepted_result_payloads_sha256"),
                label="accepted result payload ledger SHA",
            ),
        },
        "population": {
            "cases": expected_cases,
            "results": expected_cases * 2,
            "no_cases_dropped": True,
            "all_car_failures_classified": True,
            "causal_limits_preserved": True,
        },
        "counts": counts,
        "validation_memory_shape": {
            "streaming_unit": "one_case_pair",
            "maximum_live_full_result_records": 2,
            "full_result_records_retained": 0,
            "retained_records": "compact_failure_rows_only",
            **({} if streaming_stats is None else dict(streaming_stats)),
        },
        "case_record_ledger_sha256": sha256_bytes(
            canonical_json_bytes(row_ledger)
        ),
        "interpretation_scope": INTERPRETATION_SCOPE,
    }
    report["report_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(report)
    )
    return report


def jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        canonical_json_bytes(record) + b"\n"
        for record in records
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    counts = _mapping(report.get("counts"), label="report counts")
    classes = _mapping(
        counts.get("primary_car_failure_class_counts"),
        label="CAR class counts",
    )
    task_classes = _mapping(
        counts.get("aegis_task_failure_class_counts"),
        label="task-failure class counts",
    )
    lines = [
        "# AEGIS SafeLIBERO failure analysis",
        "",
        (
            f"Validated cases: **{counts.get('case_count')}**; "
            f"hash-verified videos: **{counts.get('video_count')}**."
        ),
        (
            f"Observed AEGIS CAR: **{float(counts.get('aegis_car_percent')):.2f}%** "
            f"({counts.get('aegis_car_failure_count')} displacement-defined "
            "collision failures)."
        ),
        "",
        "## Exhaustive observed CAR-failure strata",
        "",
        "| Primary observed class | Cases |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{key}` | {value} |"
        for key, value in sorted(classes.items())
        if key != "not_car_failure"
    )
    lines.extend(
        [
            "",
            (
                "Every AEGIS CAR failure has exactly one row above. "
                f"Evidence-limited cases: "
                f"**{counts.get('evidence_limited_car_failure_count')}**."
            ),
            "",
            "## Joint safety and task-progress strata",
            "",
            "| AEGIS task outcome class | Cases |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{key}` | {value} |"
        for key, value in sorted(task_classes.items())
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
        ]
    )
    lines.extend(
        f"- **{key}:** {value}"
        for key, value in INTERPRETATION_SCOPE.items()
    )
    lines.append("")
    return "\n".join(lines)


def _load_exactly_one_validated_result(
    path: Path,
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Decode and validate one immutable per-arm result object."""

    if path.is_symlink() or not path.is_file():
        raise FailureReportError(f"result is missing or symlinked: {path}")
    iterator = aggregate.iter_result_records(path)
    try:
        raw = next(iterator)
    except StopIteration as error:
        raise FailureReportError(f"result is empty: {path}") from error
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise FailureReportError(
            f"per-arm result file contains multiple records: {path}"
        )
    try:
        return aggregate.validate_result(
            raw,
            config=config,
            manifest=manifest,
        )
    except aggregate.AggregationError as error:
        raise FailureReportError(
            f"strict result validation failed for {path}: {error}"
        ) from error


def _require_population_validation_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_cases: int,
) -> None:
    if (
        receipt.get("schema_version") != PREPUBLISH_SCHEMA
        or receipt.get("status") != "validated"
        or receipt.get("complete_paired_population") is not True
        or receipt.get("no_results_dropped") is not True
    ):
        raise FailureReportError(
            "population validation receipt is not complete and terminal"
        )
    result_artifacts = _mapping(
        receipt.get("result_artifacts"),
        label="population validation result artifacts",
    )
    diagnostics = _mapping(
        receipt.get("failure_diagnostics"),
        label="population validation failure diagnostics",
    )
    if (
        result_artifacts.get("count") != expected_cases * 2
        or diagnostics.get("count") != expected_cases * 2
        or diagnostics.get("all_results_deep_validated") is not True
        or diagnostics.get("contact_schema_version") != CONTACT_SCHEMA
        or diagnostics.get("contact_model_authority_schema_version")
        != CONTACT_MODEL_AUTHORITY_SCHEMA
        or diagnostics.get("contact_role_taxonomy")
        != list(CONTACT_ROLES)
        or diagnostics.get("contact_role_authority_complete") is not True
        or diagnostics.get("unknown_role_event_count") != 0
    ):
        raise FailureReportError(
            "population validation receipt lacks complete contact-v3 evidence"
        )


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        delete=False,
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def build_population_failure_artifacts(
    *,
    config_path: Path,
    manifest_receipt_path: Path,
    manifest_path: Path,
    results_root: Path,
    summary_path: Path,
    validation_receipt_path: Path,
    cases_output_path: Path,
    report_output_path: Path,
    markdown_output_path: Path,
    expected_cases: int = EXPECTED_CASES,
) -> dict[str, Any]:
    """Stream one full pair at a time and publish exhaustive compact evidence."""

    config, manifests = aggregate.load_protocol(
        config_path,
        manifest_receipt_path,
        manifest_path,
    )
    if config.get("arms") != [BASELINE_ARM, AEGIS_ARM]:
        raise FailureReportError("failure report arm order changed")
    if len(manifests) != expected_cases:
        raise FailureReportError(
            f"manifest has {len(manifests)} cases, expected {expected_cases}"
        )
    summary = aggregate.load_json(summary_path)
    validation_receipt = aggregate.load_json(validation_receipt_path)
    _require_population_validation_receipt(
        validation_receipt,
        expected_cases=expected_cases,
    )
    del validation_receipt

    root = results_root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise FailureReportError("population results root is unavailable")
    records: list[dict[str, Any]] = []
    max_live_full_results = 0
    for manifest in manifests:
        case_id = str(manifest["case_id"])
        ordinal = _int(
            manifest.get("case_ordinal"),
            label=f"{case_id}/case ordinal",
        )
        task_index = ordinal // int(
            config["population"]["expected_cases_per_task_level_group"]
        )
        task_results_root = root / f"task-{task_index}" / "results"
        baseline_path = (
            task_results_root / "pi05" / case_id / "result.json"
        )
        aegis_path = (
            task_results_root / "aegis" / case_id / "result.json"
        )
        baseline = _load_exactly_one_validated_result(
            baseline_path,
            config=config,
            manifest=manifest,
        )
        aegis = _load_exactly_one_validated_result(
            aegis_path,
            config=config,
            manifest=manifest,
        )
        max_live_full_results = max(max_live_full_results, 2)
        try:
            aggregate.validate_pairs(
                config=config,
                manifests=[manifest],
                results={
                    (case_id, BASELINE_ARM): baseline,
                    (case_id, AEGIS_ARM): aegis,
                },
            )
        except aggregate.AggregationError as error:
            raise FailureReportError(
                f"{case_id}: strict cross-arm pairing failed: {error}"
            ) from error
        records.append(
            build_case_record(
                manifest=manifest,
                baseline=baseline,
                aegis=aegis,
                baseline_artifact_root=task_results_root,
                aegis_artifact_root=task_results_root,
            )
        )
        del baseline
        del aegis

    streaming_stats = {
        "maximum_live_full_result_records": max_live_full_results,
        "compact_case_records_retained": len(records),
    }
    report = build_report(
        records,
        summary=summary,
        summary_sha256=sha256_path(summary_path),
        validation_receipt_sha256=sha256_path(validation_receipt_path),
        expected_cases=expected_cases,
        streaming_stats=streaming_stats,
    )
    ordered_records = sorted(
        records, key=lambda row: int(row["case_ordinal"])
    )
    _write_atomic_bytes(cases_output_path, jsonl_bytes(ordered_records))
    _write_atomic_bytes(
        report_output_path,
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )
    _write_atomic_bytes(
        markdown_output_path,
        render_markdown(report).encode("utf-8"),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exhaustive, evidence-scoped AEGIS population "
            "failure report"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--population-validation-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--cases-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = build_population_failure_artifacts(
        config_path=args.config,
        manifest_receipt_path=args.receipt,
        manifest_path=args.manifest,
        results_root=args.results,
        summary_path=args.summary,
        validation_receipt_path=args.population_validation_receipt,
        cases_output_path=args.cases_output,
        report_output_path=args.report_output,
        markdown_output_path=args.markdown_output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_count": report["population"]["cases"],
                "report_payload_sha256": report[
                    "report_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
