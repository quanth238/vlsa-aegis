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
CASE_SCHEMA_V2 = "vlsa_table1_aegis_failure_case.v2"
REPORT_SCHEMA_V2 = "vlsa_table1_aegis_failure_report.v2"
CONTACT_SCHEMA = "vlsa_table1_active_obstacle_contacts.v3"
CONTACT_MODEL_AUTHORITY_SCHEMA = "vlsa_table1_contact_model_authority.v2"
SUMMARY_SCHEMA = "vlsa_table1_population_summary.v1"
SUMMARY_SCHEMA_V2 = "vlsa_table1_population_summary.v2"
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
CAR_FAILURE_CLASSES_V2 = (
    CAR_FAILURE_CLASSES
    | {"both_views_no_usable_points_fail_open_collision"}
)
NO_USABLE_POINTS_SUBCLASSES = (
    "not_applicable",
    "both_views_no_detection",
    "agentview_detection_without_usable_3d_points",
    "backview_detection_without_usable_3d_points",
    "both_views_detections_without_usable_3d_points",
)
PAPER_CAR_CONTACT_STRATA = (
    "paper_safe_no_sampled_robot_contact",
    "paper_safe_sampled_robot_contact",
    "paper_collision_no_sampled_robot_contact",
    "paper_collision_sampled_robot_contact",
)
ROBOT_CONTACT_TRANSITIONS = (
    "baseline_no_contact_to_aegis_no_contact",
    "baseline_no_contact_to_aegis_contact",
    "baseline_contact_to_aegis_no_contact",
    "baseline_contact_to_aegis_contact",
)
PAPER_CAR_CONTACT_TRANSITIONS = tuple(
    f"baseline_{baseline}_to_aegis_{aegis}"
    for baseline in PAPER_CAR_CONTACT_STRATA
    for aegis in PAPER_CAR_CONTACT_STRATA
)

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
INTERPRETATION_SCOPE_V2 = {
    **INTERPRETATION_SCOPE,
    "method": (
        "The AEGIS arm is conditioned on one frozen Codex obstacle label per "
        "case; it is not an API-backed reproduction of the paper's semantic "
        "selector."
    ),
    "table_scope": (
        "Only pi0.5 and pi0.5+AEGIS translational rows are included. "
        "OpenVLA-OFT is not evaluated."
    ),
    "clearance": (
        "No continuous signed-distance or minimum-clearance signal exists in "
        "the immutable v1 artifacts."
    ),
    "stopping": (
        "Strict zero translation is an observed action pattern. A safe case "
        "with that pattern does not by itself prove that stopping caused "
        "safety."
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


def _contact_evidence_v2(
    payload: Mapping[str, Any],
    *,
    collision_step: int | None,
) -> dict[str, Any]:
    """Extend validated v1 contact evidence over the complete episode."""

    evidence = _contact_evidence(
        payload, collision_step=collision_step
    )
    counts: Counter[str] = Counter()
    postcontrol_counts: Counter[str] = Counter()
    snapshots = _list(payload.get("snapshots"), label="contact snapshots")
    for snapshot in snapshots:
        snapshot_map = _mapping(snapshot, label="contact snapshot")
        step = _int(
            snapshot_map.get("step"), label="contact snapshot step"
        )
        for event in _list(
            snapshot_map.get("events"), label="contact events"
        ):
            other = _mapping(
                _mapping(event, label="contact event").get("other"),
                label="contact other",
            )
            role = str(other.get("classification"))
            if (
                role not in CONTACT_ROLES
                or role == "unknown"
                or other.get("classification_authority") != "complete"
            ):
                raise FailureReportError(
                    "contact other role is not complete and canonical"
                )
            counts[role] += 1
            if step >= 0:
                postcontrol_counts[role] += 1
    evidence.update(
        {
            "event_counts_by_role": {
                role: counts[role] for role in CONTACT_ROLES
            },
            "postcontrol_event_counts_by_role": {
                role: postcontrol_counts[role] for role in CONTACT_ROLES
            },
            "robot_active_obstacle_contact": counts["robot"] > 0,
            "postcontrol_robot_active_obstacle_contact": (
                postcontrol_counts["robot"] > 0
            ),
            "collision_relevant_event_count": sum(
                counts[role]
                for role in COLLISION_RELEVANT_CONTACT_ROLES
            ),
            "postcontrol_collision_relevant_event_count": sum(
                postcontrol_counts[role]
                for role in COLLISION_RELEVANT_CONTACT_ROLES
            ),
            "sampling_scope": (
                "settled snapshot at step -1 plus post-control-step "
                "snapshots; internal physics substeps are not observed"
            ),
        }
    )
    return evidence


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


def _geometry_evidence_v2(result: Mapping[str, Any]) -> dict[str, Any]:
    """Add exact detector/point-cloud evidence without changing v1 rows."""

    case_id = str(result.get("case_id"))
    evidence = _geometry_evidence(result)
    diagnostics = _diagnostics_record(result)
    descriptor = _mapping(
        diagnostics.get("geometry"), label=f"{case_id}/geometry descriptor"
    )
    record = _mapping(
        descriptor.get("record"), label=f"{case_id}/geometry record"
    )
    views = _mapping(record.get("views"), label=f"{case_id}/geometry views")
    failure = record.get("failure")
    no_grounded_points = (
        isinstance(failure, Mapping)
        and failure.get("type") == "no_grounded_points"
    )
    for view_name in ("agentview", "backview"):
        value = _mapping(
            views.get(view_name), label=f"{case_id}/{view_name}"
        )
        returned_points = value.get("returned_point_cloud")
        returned_point_shape = None
        raw_point_count = None
        usable_point_count = None
        if isinstance(returned_points, Mapping):
            shape = returned_points.get("shape")
            if (
                not isinstance(shape, list)
                or not shape
                or any(
                    type(dimension) is not int or dimension < 0
                    for dimension in shape
                )
            ):
                raise FailureReportError(
                    f"{case_id}/{view_name}: returned point-cloud shape "
                    "changed"
                )
            returned_point_shape = list(shape)
            raw_point_count = shape[0]
            if len(shape) == 2 and shape[1] == 3:
                if returned_points.get("finite") is True:
                    usable_point_count = shape[0]
                elif shape[0] == 0:
                    usable_point_count = 0
        if no_grounded_points:
            usable_point_count = 0
        evidence["views"][view_name].update(
            {
                "returned_point_shape": returned_point_shape,
                "raw_point_count": raw_point_count,
                "usable_point_count": usable_point_count,
            }
        )
    return evidence


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


def _no_usable_points_evidence(
    geometry: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """Separate detector-box evidence from empty reconstructed 3-D points."""

    views = geometry.get("views")
    if not isinstance(views, Mapping):
        raise FailureReportError(
            "no-usable-points failure lacks per-view evidence"
        )
    detections: dict[str, int] = {}
    for view_name in ("agentview", "backview"):
        view = views.get(view_name)
        if not isinstance(view, Mapping):
            raise FailureReportError(
                f"no-usable-points failure lacks {view_name}"
            )
        detection_count = view.get("detection_count")
        usable_point_count = view.get("usable_point_count")
        if (
            type(detection_count) is not int
            or detection_count < 0
            or usable_point_count != 0
        ):
            raise FailureReportError(
                f"{view_name}: no-usable-points evidence is inconsistent"
            )
        expected_status = (
            "no_detection" if detection_count == 0 else "detected"
        )
        if view.get("status") != expected_status:
            raise FailureReportError(
                f"{view_name}: detector status/count mismatch"
            )
        detections[view_name] = detection_count

    agent_detected = detections["agentview"] > 0
    back_detected = detections["backview"] > 0
    if not agent_detected and not back_detected:
        return (
            "both_views_no_detection",
            [
                "semantic_label_or_detector_grounding_failure_hypothesis",
                "viewpoint_or_detector_threshold_failure_hypothesis",
            ],
        )
    if agent_detected and not back_detected:
        return (
            "agentview_detection_without_usable_3d_points",
            [
                "selected_crop_depth_or_point_reconstruction_failure_"
                "hypothesis",
            ],
        )
    if not agent_detected and back_detected:
        return (
            "backview_detection_without_usable_3d_points",
            [
                "selected_crop_depth_or_point_reconstruction_failure_"
                "hypothesis",
            ],
        )
    return (
        "both_views_detections_without_usable_3d_points",
        [
            "selected_crop_depth_or_point_reconstruction_failure_"
            "hypothesis",
        ],
    )


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


def _paper_car_contact_stratum(
    *,
    paper_collision: bool,
    robot_contact: bool,
) -> str:
    return "_".join(
        (
            "paper_collision" if paper_collision else "paper_safe",
            "sampled_robot_contact" if robot_contact else "no_sampled_robot_contact",
        )
    )


def _paired_goal_progress(
    baseline: Mapping[str, Any],
    aegis: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_goal = _goal_evidence(baseline)
    aegis_goal = _goal_evidence(aegis)
    if (
        baseline_goal["goal_definition_sha256"]
        != aegis_goal["goal_definition_sha256"]
    ):
        raise FailureReportError(
            "paired arms use different native goal definitions"
        )
    delta_fields = (
        "initial_satisfied_count",
        "final_satisfied_count",
        "maximum_satisfied_count",
        "initial_fraction",
        "final_fraction",
        "maximum_fraction",
        "regression_count",
    )
    deltas = {
        field: aegis_goal[field] - baseline_goal[field]
        for field in delta_fields
    }
    if (
        deltas["initial_satisfied_count"] != 0
        or not math.isclose(
            float(deltas["initial_fraction"]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise FailureReportError(
            "paired arms start from different native goal progress"
        )
    return {
        BASELINE_ARM: baseline_goal,
        AEGIS_ARM: aegis_goal,
        "same_goal_definition": True,
        "same_initial_goal_progress": True,
        "aegis_minus_pi05": deltas,
    }


def _observed_evidence_tags_v2(
    *,
    metrics: Mapping[str, Any],
    geometry: Mapping[str, Any],
    control: Mapping[str, Any],
    contacts: Mapping[str, Any],
    no_points_subclass: str | None,
) -> list[str]:
    tags = {
        (
            "paper_car_collision"
            if metrics["paper_collision"]
            else "paper_car_safe"
        ),
        (
            "task_success"
            if metrics["task_success"]
            else "task_failure"
        ),
        f"geometry_status_{geometry.get('status')}",
        (
            "sampled_robot_active_obstacle_contact_present"
            if contacts["robot_active_obstacle_contact"]
            else "sampled_robot_active_obstacle_contact_absent"
        ),
        (
            "settled_collision_relevant_contact_present"
            if contacts["settled_collision_relevant_event_count"] > 0
            else "settled_collision_relevant_contact_absent"
        ),
    }
    control_path = control.get("collision_step_control_path")
    if control_path is not None:
        tags.add(f"collision_step_control_path_{control_path}")
    barrier = control.get("collision_step_barrier_h")
    if barrier is not None:
        tags.add(
            "collision_step_proxy_barrier_positive"
            if float(barrier) > 0.0
            else "collision_step_proxy_barrier_nonpositive"
        )
    if (
        no_points_subclass is not None
        and no_points_subclass != "not_applicable"
    ):
        tags.add(no_points_subclass)
        tags.add("no_usable_3d_points_fail_open_path_observed")
    return sorted(tags)


def _refine_no_usable_points_v2(
    *,
    metrics: Mapping[str, Any],
    geometry: Mapping[str, Any],
    primary_class: str,
    causal_hypotheses: Sequence[str],
) -> tuple[str, str, list[str]]:
    subclass = "not_applicable"
    hypotheses = list(causal_hypotheses)
    if (
        geometry.get("status") == "method_failure_passthrough"
        and geometry.get("failure_type") == "no_grounded_points"
    ):
        subclass, hypotheses = _no_usable_points_evidence(geometry)
        if metrics["paper_collision"]:
            primary_class = (
                "both_views_no_usable_points_fail_open_collision"
            )
    return primary_class, subclass, hypotheses


def build_case_record_v2(
    *,
    manifest: Mapping[str, Any],
    baseline: Mapping[str, Any],
    aegis: Mapping[str, Any],
    baseline_artifact_root: Path,
    aegis_artifact_root: Path,
) -> dict[str, Any]:
    """Enrich a validated v1 pair without mutating its immutable artifacts."""

    record = build_case_record(
        manifest=manifest,
        baseline=baseline,
        aegis=aegis,
        baseline_artifact_root=baseline_artifact_root,
        aegis_artifact_root=aegis_artifact_root,
    )
    record.pop("record_payload_sha256", None)
    baseline_metrics = _result_metrics(baseline)
    aegis_metrics = _result_metrics(aegis)
    baseline_contacts = _contact_evidence_v2(
        load_contact_payload(
            baseline, artifact_root=baseline_artifact_root
        ),
        collision_step=baseline_metrics["collision_first_step"],
    )
    aegis_contacts = _contact_evidence_v2(
        load_contact_payload(
            aegis, artifact_root=aegis_artifact_root
        ),
        collision_step=aegis_metrics["collision_first_step"],
    )
    record["aegis_diagnostics"]["contacts"] = aegis_contacts
    for arm, arm_metrics, arm_contacts in (
        (BASELINE_ARM, baseline_metrics, baseline_contacts),
        (AEGIS_ARM, aegis_metrics, aegis_contacts),
    ):
        robot_contact = bool(
            arm_contacts["robot_active_obstacle_contact"]
        )
        record["outcomes"][arm].update(
            {
                "sampled_robot_active_obstacle_contact": robot_contact,
                "sampled_postcontrol_robot_active_obstacle_contact": (
                    arm_contacts[
                        "postcontrol_robot_active_obstacle_contact"
                    ]
                ),
                "sampled_active_obstacle_contact_any": (
                    arm_contacts["total_event_count"] > 0
                ),
                "sampled_collision_relevant_active_obstacle_contact": (
                    arm_contacts["collision_relevant_event_count"] > 0
                ),
                "contact_event_counts_by_role": dict(
                    arm_contacts["event_counts_by_role"]
                ),
                "postcontrol_contact_event_counts_by_role": dict(
                    arm_contacts["postcontrol_event_counts_by_role"]
                ),
                "paper_car_contact_stratum": (
                    _paper_car_contact_stratum(
                        paper_collision=arm_metrics["paper_collision"],
                        robot_contact=robot_contact,
                    )
                ),
            }
        )
    record["physical_contacts"] = {
        BASELINE_ARM: baseline_contacts,
        AEGIS_ARM: aegis_contacts,
        "scope": (
            "settled and post-control-step sampled MuJoCo contacts "
            "involving the settled active obstacle; not continuous "
            "substep contact or clearance"
        ),
    }
    record["paired_goal_progress"] = _paired_goal_progress(
        baseline, aegis
    )
    record["paired_transition"]["joint"] = (
        f"baseline_{record['outcomes'][BASELINE_ARM]['joint_outcome']}"
        f"_to_aegis_{record['outcomes'][AEGIS_ARM]['joint_outcome']}"
    )
    geometry = _geometry_evidence_v2(aegis)
    record["aegis_diagnostics"]["geometry"] = geometry
    failure_analysis = record["failure_analysis"]
    (
        failure_analysis["primary_observed_car_class"],
        no_points_subclass,
        failure_analysis["causal_hypothesis_tags"],
    ) = _refine_no_usable_points_v2(
        metrics=aegis_metrics,
        geometry=geometry,
        primary_class=failure_analysis[
            "primary_observed_car_class"
        ],
        causal_hypotheses=failure_analysis[
            "causal_hypothesis_tags"
        ],
    )
    primary_class = failure_analysis["primary_observed_car_class"]
    failure_analysis.update(
        {
            "primary_car_failure_class": primary_class,
            "evidence_limited": (
                primary_class in EVIDENCE_LIMITED_CAR_CLASSES
            ),
            "observed_evidence_tags": _observed_evidence_tags_v2(
                metrics=aegis_metrics,
                geometry=geometry,
                control=record["aegis_diagnostics"]["control"],
                contacts=aegis_contacts,
                no_points_subclass=no_points_subclass,
            ),
            "causal_hypotheses": list(
                failure_analysis["causal_hypothesis_tags"]
            ),
            "aegis_task_outcome_class": failure_analysis[
                "aegis_task_failure_class"
            ],
            "no_usable_points_observed_subclass": no_points_subclass,
            "legacy_v1_strict_safety_by_stopping_flag": bool(
                failure_analysis["strict_safety_by_stopping"]
            ),
            "safe_with_strict_zero_translation": (
                not aegis_metrics["paper_collision"]
                and bool(
                    record["aegis_diagnostics"]["control"][
                        "strict_zero_translation"
                    ]
                )
            ),
            "stopping_causal_claim_supported": False,
        }
    )
    failure_analysis.pop("strict_safety_by_stopping", None)
    record.update(
        {
            "schema_version": CASE_SCHEMA_V2,
            "claim_scope": {
                "method_label": (
                    "pi0.5 + AEGIS translational conditioned on frozen "
                    "per-case Codex obstacle labels"
                ),
                "baseline_method_label": "pi0.5 translational",
                "aegis_method_label": (
                    "pi0.5 + AEGIS translational conditioned on frozen "
                    "per-case Codex obstacle labels"
                ),
                "table_scope": (
                    "two-row translational Table-1 reproduction: "
                    "pi0.5 and pi0.5+AEGIS only"
                ),
                "openvla_oft_included": False,
                "paper_semantic_selector_reproduced": False,
                "paper_exact_end_to_end_reproduction_claimed": False,
                "clearance_available": False,
                "minimum_clearance_claimed": False,
            },
            "interpretation_scope": INTERPRETATION_SCOPE_V2,
        }
    )
    record["record_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(record)
    )
    return record


def _validate_report_against_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    expected_cases: int = EXPECTED_CASES,
    allowed_car_failure_classes: set[str],
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
    if set(primary) - allowed_car_failure_classes:
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


def validate_report_against_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    expected_cases: int = EXPECTED_CASES,
) -> dict[str, Any]:
    """Require exhaustive rows and exact alignment with the population table."""

    return _validate_report_against_summary(
        records,
        summary=summary,
        expected_cases=expected_cases,
        allowed_car_failure_classes=CAR_FAILURE_CLASSES,
    )


def _joint_outcome_counts(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> dict[str, int]:
    counts = Counter(
        str(record["outcomes"][arm]["joint_outcome"])
        for record in records
    )
    return {
        key: counts[key] for key in aggregate.JOINT_OUTCOMES
    }


def _physical_contact_counts(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> dict[str, Any]:
    strata = Counter(
        str(record["outcomes"][arm]["paper_car_contact_stratum"])
        for record in records
    )
    robot_contacts = sum(
        bool(
            record["outcomes"][arm][
                "sampled_robot_active_obstacle_contact"
            ]
        )
        for record in records
    )
    any_contacts = sum(
        bool(
            record["outcomes"][arm][
                "sampled_active_obstacle_contact_any"
            ]
        )
        for record in records
    )
    collision_relevant_contacts = sum(
        bool(
            record["outcomes"][arm][
                "sampled_collision_relevant_active_obstacle_contact"
            ]
        )
        for record in records
    )
    postcontrol_robot_contacts = sum(
        bool(
            record["outcomes"][arm][
                "sampled_postcontrol_robot_active_obstacle_contact"
            ]
        )
        for record in records
    )
    complete_strata = {
        key: strata[key] for key in PAPER_CAR_CONTACT_STRATA
    }
    if sum(complete_strata.values()) != len(records):
        raise FailureReportError(
            f"{arm}: unregistered paper-CAR/contact stratum"
        )
    disagreement_count = (
        complete_strata[
            "paper_safe_sampled_robot_contact"
        ]
        + complete_strata[
            "paper_collision_no_sampled_robot_contact"
        ]
    )
    return {
        "denominator": len(records),
        "sampled_any_contact_count": any_contacts,
        "sampled_collision_relevant_contact_count": (
            collision_relevant_contacts
        ),
        "sampled_robot_contact_count": robot_contacts,
        "no_sampled_robot_contact_count": len(records) - robot_contacts,
        "sampled_postcontrol_robot_contact_count": (
            postcontrol_robot_contacts
        ),
        "paper_car_sampled_robot_contact_agreement_count": (
            len(records) - disagreement_count
        ),
        "paper_car_sampled_robot_contact_disagreement_count": (
            disagreement_count
        ),
        "paper_car_contact_strata": complete_strata,
    }


def _paired_transition_counts_v2(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    car = Counter(
        str(record["paired_transition"]["car"]) for record in records
    )
    task = Counter(
        str(record["paired_transition"]["task"]) for record in records
    )
    joint = Counter(
        str(record["paired_transition"]["joint"]) for record in records
    )
    robot_contact: Counter[str] = Counter()
    paper_car_contact: Counter[str] = Counter()
    for record in records:
        baseline_contact = bool(
            record["outcomes"][BASELINE_ARM][
                "sampled_robot_active_obstacle_contact"
            ]
        )
        aegis_contact = bool(
            record["outcomes"][AEGIS_ARM][
                "sampled_robot_active_obstacle_contact"
            ]
        )
        robot_contact[
            "_to_".join(
                (
                    (
                        "baseline_contact"
                        if baseline_contact
                        else "baseline_no_contact"
                    ),
                    (
                        "aegis_contact"
                        if aegis_contact
                        else "aegis_no_contact"
                    ),
                )
            )
        ] += 1
        paper_car_contact[
            (
                "baseline_"
                + str(
                    record["outcomes"][BASELINE_ARM][
                        "paper_car_contact_stratum"
                    ]
                )
                + "_to_aegis_"
                + str(
                    record["outcomes"][AEGIS_ARM][
                        "paper_car_contact_stratum"
                    ]
                )
            )
        ] += 1
    complete = {
        "car": {key: car[key] for key in aggregate.CAR_TRANSITIONS},
        "task": {
            key: task[key] for key in aggregate.TASK_TRANSITIONS
        },
        "joint": {
            key: joint[key] for key in aggregate.JOINT_TRANSITIONS
        },
        "sampled_robot_contact": {
            key: robot_contact[key] for key in ROBOT_CONTACT_TRANSITIONS
        },
        "paper_car_sampled_robot_contact": {
            key: paper_car_contact[key]
            for key in PAPER_CAR_CONTACT_TRANSITIONS
        },
    }
    if any(
        sum(counts.values()) != len(records)
        for counts in complete.values()
    ):
        raise FailureReportError(
            "paired transitions contain an unregistered state"
        )
    return {"denominator": len(records), **complete}


def _failure_stratum(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary = Counter(
        str(row["failure_analysis"]["primary_observed_car_class"])
        for row in rows
    )
    task = Counter(
        str(row["failure_analysis"]["aegis_task_failure_class"])
        for row in rows
    )
    car_failures = sum(
        bool(row["failure_analysis"]["is_aegis_car_failure"])
        for row in rows
    )
    no_points = Counter(
        str(
            row["failure_analysis"][
                "no_usable_points_observed_subclass"
            ]
        )
        for row in rows
    )
    complete_no_points = {
        key: no_points[key] for key in NO_USABLE_POINTS_SUBCLASSES
    }
    if sum(complete_no_points.values()) != len(rows):
        raise FailureReportError(
            "cross-tab contains an unregistered no-points subclass"
        )
    return {
        "denominator": len(rows),
        "aegis_car_failure_count": car_failures,
        "aegis_car_success_count": len(rows) - car_failures,
        "primary_observed_car_class_counts": dict(sorted(primary.items())),
        "aegis_task_failure_class_counts": dict(sorted(task.items())),
        "no_usable_points_observed_subclass_counts": (
            complete_no_points
        ),
        "joint_outcome_counts": {
            BASELINE_ARM: _joint_outcome_counts(rows, arm=BASELINE_ARM),
            AEGIS_ARM: _joint_outcome_counts(rows, arm=AEGIS_ARM),
        },
        "physical_contacts": {
            BASELINE_ARM: _physical_contact_counts(
                rows, arm=BASELINE_ARM
            ),
            AEGIS_ARM: _physical_contact_counts(rows, arm=AEGIS_ARM),
        },
        "paired_transitions": _paired_transition_counts_v2(rows),
        "paired_goal_progress": _paired_goal_progress_counts_v2(rows),
        "intervention": _intervention_counts_v2(rows),
    }


def _cross_tabs_v2(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def grouped(field: str) -> dict[str, Any]:
        values: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            values[str(record.get(field))].append(record)
        return {
            key: _failure_stratum(rows)
            for key, rows in sorted(values.items())
        }

    tasks: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = "|".join(
            (
                str(record.get("suite")),
                str(record.get("safety_level")),
                str(record.get("logical_task_index")),
                str(record.get("task_name")),
            )
        )
        tasks[key].append(record)
    return {
        "suite": grouped("suite"),
        "safety_level": grouped("safety_level"),
        "task": {
            key: _failure_stratum(rows)
            for key, rows in sorted(tasks.items())
        },
        "frozen_obstacle_label": grouped("frozen_obstacle_label"),
    }


def _paired_goal_progress_counts_v2(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise FailureReportError(
            "paired goal-progress summary requires a nonempty stratum"
        )
    count_fields = (
        "initial_satisfied_count",
        "final_satisfied_count",
        "maximum_satisfied_count",
        "regression_count",
    )
    fraction_fields = (
        "initial_fraction",
        "final_fraction",
        "maximum_fraction",
    )

    def arm_summary(arm: str) -> dict[str, Any]:
        sums: dict[str, int | float] = {
            field: sum(
                _int(
                    row["paired_goal_progress"][arm][field],
                    label=f"{arm}/{field}",
                )
                for row in records
            )
            for field in count_fields
        }
        sums.update(
            {
                field: sum(
                    _number(
                        row["paired_goal_progress"][arm][field],
                        label=f"{arm}/{field}",
                    )
                    for row in records
                )
                for field in fraction_fields
            }
        )
        return {
            "sum": sums,
            "mean": {
                field: value / len(records)
                for field, value in sums.items()
            },
        }

    delta_sums: dict[str, int | float] = {
        field: sum(
            _int(
                row["paired_goal_progress"]["aegis_minus_pi05"][
                    field
                ],
                label=f"goal delta/{field}",
            )
            for row in records
        )
        for field in count_fields
    }
    delta_sums.update(
        {
            field: sum(
                _number(
                    row["paired_goal_progress"]["aegis_minus_pi05"][
                        field
                    ],
                    label=f"goal delta/{field}",
                )
                for row in records
            )
            for field in fraction_fields
        }
    )
    exact_binding_count = sum(
        (
            row["paired_goal_progress"].get("same_goal_definition")
            is True
            and row["paired_goal_progress"].get(
                "same_initial_goal_progress"
            )
            is True
        )
        for row in records
    )
    if exact_binding_count != len(records):
        raise FailureReportError(
            "paired goal-progress binding is incomplete"
        )
    return {
        "denominator": len(records),
        "same_goal_definition_and_initial_progress_count": (
            exact_binding_count
        ),
        "arms": {
            BASELINE_ARM: arm_summary(BASELINE_ARM),
            AEGIS_ARM: arm_summary(AEGIS_ARM),
        },
        "aegis_minus_pi05_sum": delta_sums,
        "aegis_minus_pi05_mean": {
            field: value / len(records)
            for field, value in delta_sums.items()
        },
    }


def _intervention_counts_v2(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise FailureReportError(
            "intervention summary requires a nonempty stratum"
        )
    interventions = [
        _mapping(
            row["aegis_diagnostics"]["intervention"],
            label=f"{row.get('case_id')}/intervention",
        )
        for row in records
    ]
    eligible_steps = sum(
        _int(value.get("eligible_steps"), label="eligible steps")
        for value in interventions
    )
    modified_actions = sum(
        _int(
            value.get("intervention_count"),
            label="intervention count",
        )
        for value in interventions
    )
    correction_l2_sum = sum(
        _number(
            value.get("correction_l2_sum"),
            label="correction L2 sum",
        )
        for value in interventions
    )
    modified_cases = sum(
        _int(
            value.get("intervention_count"),
            label="intervention count",
        )
        > 0
        for value in interventions
    )
    no_execution_cases = sum(
        int(
            row["outcomes"][AEGIS_ARM]["executed_action_count"]
        )
        == 0
        for row in records
    )
    return {
        "denominator": len(records),
        "eligible_step_count": eligible_steps,
        "modified_action_count": modified_actions,
        "modified_case_count": modified_cases,
        "unmodified_case_count": len(records) - modified_cases,
        "correction_l2_sum": correction_l2_sum,
        "modified_action_rate": (
            None
            if eligible_steps == 0
            else modified_actions / eligible_steps
        ),
        "correction_l2_mean_per_modified_action": (
            None
            if modified_actions == 0
            else correction_l2_sum / modified_actions
        ),
        "no_execution_case_count": no_execution_cases,
        "legacy_v1_strict_safety_by_stopping_flag_count": sum(
            bool(
                row["failure_analysis"][
                    "legacy_v1_strict_safety_by_stopping_flag"
                ]
            )
            for row in records
        ),
        "safe_with_strict_zero_translation_case_count": sum(
            bool(
                row["failure_analysis"][
                    "safe_with_strict_zero_translation"
                ]
            )
            for row in records
        ),
        "stopping_causal_claim_supported": False,
    }


def validate_report_against_summary_v2(
    records: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    expected_cases: int = EXPECTED_CASES,
) -> dict[str, Any]:
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA_V2
        or summary.get("status")
        != "complete_postpublication_analysis_v2"
    ):
        raise FailureReportError("analysis-v2 summary is not complete")
    claim_scope = _mapping(
        summary.get("claim_scope"), label="analysis-v2 claim scope"
    )
    if (
        claim_scope.get("openvla_oft_included") is not False
        or claim_scope.get("paper_semantic_selector_reproduced")
        is not False
        or claim_scope.get("paper_exact_end_to_end_reproduction_claimed")
        is not False
        or claim_scope.get("clearance_available") is not False
        or claim_scope.get("minimum_clearance_claimed") is not False
    ):
        raise FailureReportError(
            "analysis-v2 claim boundary is incomplete or overstated"
        )
    v1_projection = dict(summary)
    v1_projection["schema_version"] = SUMMARY_SCHEMA
    v1_projection["status"] = "complete_population_validated"
    base = _validate_report_against_summary(
        records,
        summary=v1_projection,
        expected_cases=expected_cases,
        allowed_car_failure_classes=CAR_FAILURE_CLASSES_V2,
    )
    for arm in (BASELINE_ARM, AEGIS_ARM):
        expected = _mapping(
            summary["average"][arm],
            label=f"analysis-v2/{arm}/average",
        )
        joint = _joint_outcome_counts(records, arm=arm)
        if joint != expected.get("joint_outcome_counts"):
            raise FailureReportError(
                f"{arm}: joint outcomes differ from analysis-v2 summary"
            )
        arm_outcomes = [record["outcomes"][arm] for record in records]
        exact = {
            "car": {
                "success_count": sum(
                    not outcome["paper_collision"]
                    for outcome in arm_outcomes
                ),
                "failure_count": sum(
                    outcome["paper_collision"]
                    for outcome in arm_outcomes
                ),
                "denominator": len(records),
            },
            "tsr": {
                "success_count": sum(
                    outcome["task_success"] for outcome in arm_outcomes
                ),
                "failure_count": sum(
                    not outcome["task_success"]
                    for outcome in arm_outcomes
                ),
                "denominator": len(records),
            },
            "legacy_ets_steps": {
                "sum": sum(
                    outcome["legacy_ets_steps"]
                    for outcome in arm_outcomes
                ),
                "denominator": len(records),
            },
            "executed_action_count": {
                "sum": sum(
                    outcome["executed_action_count"]
                    for outcome in arm_outcomes
                ),
                "denominator": len(records),
            },
        }
        for metric, observed in exact.items():
            expected_metric = _mapping(
                expected.get(metric),
                label=f"analysis-v2/{arm}/{metric}",
            )
            for field, value in observed.items():
                if expected_metric.get(field) != value:
                    raise FailureReportError(
                        f"{arm}/{metric}/{field} differs from "
                        "analysis-v2 summary"
                    )
    paired = _paired_transition_counts_v2(records)
    expected_paired = _mapping(
        summary.get("paired_transitions"),
        label="analysis-v2 paired transitions",
    )
    expected_overall = _mapping(
        expected_paired.get("overall"),
        label="analysis-v2 overall paired transitions",
    )
    for field in ("denominator", "car", "task", "joint"):
        if paired[field] != expected_overall.get(field):
            raise FailureReportError(
                f"paired {field} differs from analysis-v2 summary"
            )
    expected_case_scope = {
        key: claim_scope[key]
        for key in (
            "method_label",
            "baseline_method_label",
            "aegis_method_label",
            "table_scope",
            "openvla_oft_included",
            "paper_semantic_selector_reproduced",
            "paper_exact_end_to_end_reproduction_claimed",
            "clearance_available",
            "minimum_clearance_claimed",
        )
    }
    for record in records:
        if record.get("claim_scope") != expected_case_scope:
            raise FailureReportError(
                f"{record.get('case_id')}: claim scope differs"
            )
    no_points = Counter(
        str(
            record["failure_analysis"][
                "no_usable_points_observed_subclass"
            ]
        )
        for record in records
    )
    no_points_counts = {
        key: no_points[key] for key in NO_USABLE_POINTS_SUBCLASSES
    }
    if sum(no_points_counts.values()) != len(records):
        raise FailureReportError(
            "analysis-v2 has an unregistered no-points subclass"
        )
    return {
        **base,
        "joint_outcome_counts": {
            BASELINE_ARM: _joint_outcome_counts(
                records, arm=BASELINE_ARM
            ),
            AEGIS_ARM: _joint_outcome_counts(records, arm=AEGIS_ARM),
        },
        "physical_contacts": {
            BASELINE_ARM: _physical_contact_counts(
                records, arm=BASELINE_ARM
            ),
            AEGIS_ARM: _physical_contact_counts(
                records, arm=AEGIS_ARM
            ),
        },
        "paired_transitions": paired,
        "no_usable_points_observed_subclass_counts": (
            no_points_counts
        ),
        "paired_goal_progress": _paired_goal_progress_counts_v2(records),
        "intervention": _intervention_counts_v2(records),
        "cross_tabs": _cross_tabs_v2(records),
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


def build_report_v2(
    records: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    summary_sha256: str,
    validation_receipt_sha256: str,
    source_publication_receipt_sha256: str,
    expected_cases: int = EXPECTED_CASES,
    streaming_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    counts = validate_report_against_summary_v2(
        records,
        summary=summary,
        expected_cases=expected_cases,
    )
    source_v1 = _mapping(
        summary.get("source_v1"), label="analysis-v2 source-v1 binding"
    )
    source_receipt_sha = _sha(
        source_publication_receipt_sha256,
        label="source v1 publication receipt SHA",
    )
    if (
        _sha(
            source_v1.get("v1_publication_receipt_sha256"),
            label="summary source v1 publication receipt SHA",
        )
        != source_receipt_sha
    ):
        raise FailureReportError(
            "failure report v2 source differs from its summary"
        )
    ordered_rows = sorted(
        records, key=lambda row: int(row.get("case_ordinal"))
    )
    ledger = [
        {
            "case_id": row["case_id"],
            "record_payload_sha256": row["record_payload_sha256"],
        }
        for row in ordered_rows
    ]
    report = {
        "schema_version": REPORT_SCHEMA_V2,
        "status": "complete_postpublication_failure_analysis_v2",
        "protocol_id": summary.get("protocol_id"),
        "claim_scope": dict(
            _mapping(
                summary.get("claim_scope"),
                label="analysis-v2 claim scope",
            )
        ),
        "source": {
            "v1_publication_receipt_sha256": source_receipt_sha,
            "population_summary_v2_sha256": _sha(
                summary_sha256, label="population summary v2 SHA"
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
            "retained_records": "compact_failure_rows_v2_only",
            **({} if streaming_stats is None else dict(streaming_stats)),
        },
        "case_record_ledger_sha256": sha256_bytes(
            canonical_json_bytes(ledger)
        ),
        "interpretation_scope": INTERPRETATION_SCOPE_V2,
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


def render_markdown_v2(report: Mapping[str, Any]) -> str:
    counts = _mapping(report.get("counts"), label="analysis-v2 counts")
    classes = _mapping(
        counts.get("primary_car_failure_class_counts"),
        label="analysis-v2 CAR class counts",
    )
    joint = _mapping(
        counts.get("joint_outcome_counts"),
        label="analysis-v2 joint outcomes",
    )
    contacts = _mapping(
        counts.get("physical_contacts"),
        label="analysis-v2 physical contacts",
    )
    no_points = _mapping(
        counts.get("no_usable_points_observed_subclass_counts"),
        label="analysis-v2 no-points subclasses",
    )
    paired = _mapping(
        counts.get("paired_transitions"),
        label="analysis-v2 paired transitions",
    )
    goal = _mapping(
        counts.get("paired_goal_progress"),
        label="analysis-v2 paired goal progress",
    )
    intervention = _mapping(
        counts.get("intervention"),
        label="analysis-v2 intervention summary",
    )
    scope = _mapping(
        report.get("claim_scope"), label="analysis-v2 claim scope"
    )
    lines = [
        "# SafeLIBERO post-publication analysis v2",
        "",
        f"**Method:** {scope.get('method_label')}.",
        "",
        f"**Scope:** {scope.get('table_scope')}.",
        "",
        (
            f"Validated cases: **{counts.get('case_count')}**; "
            f"hash-verified videos: **{counts.get('video_count')}**."
        ),
        (
            f"Observed CAR: **{float(counts.get('aegis_car_percent')):.2f}%** "
            f"({counts.get('aegis_car_failure_count')} "
            "displacement-defined failures)."
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
            "## Joint paper-CAR and task outcomes",
            "",
            "| Arm | Safe + success | Safe + failure | Collision + success | Collision + failure |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in (BASELINE_ARM, AEGIS_ARM):
        arm_counts = _mapping(joint.get(arm), label=f"{arm}/joint")
        lines.append(
            "| "
            + " | ".join(
                (
                    arm,
                    str(arm_counts.get("safe_task_success")),
                    str(arm_counts.get("safe_task_failure")),
                    str(arm_counts.get("collision_task_success")),
                    str(arm_counts.get("collision_task_failure")),
                )
            )
            + " |"
        )
    car_transitions = _mapping(
        paired.get("car"), label="analysis-v2 CAR transitions"
    )
    lines.extend(
        [
            "",
            "## Exact paired CAR transitions",
            "",
            "| Transition | Cases |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{key}` | {value} |"
        for key, value in car_transitions.items()
    )
    lines.extend(
        [
            "",
            "## Sampled robot-active-obstacle contacts",
            "",
            (
                "| Arm | Any sampled active-obstacle contact | "
                "Collision-relevant sampled contact | "
                "Robot-active-obstacle contact | Denominator |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in (BASELINE_ARM, AEGIS_ARM):
        arm_counts = _mapping(contacts.get(arm), label=f"{arm}/contacts")
        lines.append(
            f"| {arm} | {arm_counts.get('sampled_any_contact_count')} "
            "| "
            f"{arm_counts.get('sampled_collision_relevant_contact_count')} "
            f"| {arm_counts.get('sampled_robot_contact_count')} "
            f"| {arm_counts.get('denominator')} |"
        )
    lines.extend(
        [
            "",
            "### Paper-CAR/contact disagreements",
            "",
            "| Arm | Disagreements | Denominator |",
            "|---|---:|---:|",
        ]
    )
    for arm in (BASELINE_ARM, AEGIS_ARM):
        arm_counts = _mapping(contacts.get(arm), label=f"{arm}/contacts")
        lines.append(
            f"| {arm} | "
            f"{arm_counts.get('paper_car_sampled_robot_contact_disagreement_count')} "
            f"| {arm_counts.get('denominator')} |"
        )
    lines.extend(
        [
            "",
            "## No-usable-3-D-points observed subclasses",
            "",
            "| Subclass | Cases |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{key}` | {value} |"
        for key, value in no_points.items()
        if key != "not_applicable"
    )
    goal_delta = _mapping(
        goal.get("aegis_minus_pi05_sum"),
        label="analysis-v2 goal deltas",
    )
    lines.extend(
        [
            "",
            "## Goal progress and interventions",
            "",
            (
                f"- Paired native-goal denominator: "
                f"**{goal.get('denominator')}**."
            ),
            (
                "- Sum of AEGIS-minus-pi0.5 final goal fraction: "
                f"**{goal_delta.get('final_fraction')}**."
            ),
            (
                f"- Modified AEGIS actions: "
                f"**{intervention.get('modified_action_count')} / "
                f"{intervention.get('eligible_step_count')}** eligible "
                "actions."
            ),
            (
                "- Safe cases with strict zero translation: "
                f"**{intervention.get('safe_with_strict_zero_translation_case_count')}**; "
                "this is not a causal stopping claim."
            ),
        ]
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
        for key, value in INTERPRETATION_SCOPE_V2.items()
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


def build_population_failure_artifacts_v2(
    *,
    config_path: Path,
    manifest_receipt_path: Path,
    manifest_path: Path,
    results_root: Path,
    summary_path: Path,
    validation_receipt_path: Path,
    source_publication_receipt_sha256: str,
    cases_output_path: Path,
    report_output_path: Path,
    markdown_output_path: Path,
    expected_cases: int = EXPECTED_CASES,
) -> dict[str, Any]:
    """Publish derived v2 rows while preserving the immutable v1 run.

    The result files are reopened and deeply validated one pair at a time.
    This is deliberately a separate entry point from the v1 publisher so a
    post-publication analysis cannot silently change the accepted v1 output.
    """

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
        baseline = _load_exactly_one_validated_result(
            task_results_root
            / "pi05"
            / case_id
            / "result.json",
            config=config,
            manifest=manifest,
        )
        aegis = _load_exactly_one_validated_result(
            task_results_root
            / "aegis"
            / case_id
            / "result.json",
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
            build_case_record_v2(
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
    report = build_report_v2(
        records,
        summary=summary,
        summary_sha256=sha256_path(summary_path),
        validation_receipt_sha256=sha256_path(validation_receipt_path),
        source_publication_receipt_sha256=(
            source_publication_receipt_sha256
        ),
        expected_cases=expected_cases,
        streaming_stats=streaming_stats,
    )
    ordered_records = sorted(
        records, key=lambda row: int(row["case_ordinal"])
    )
    _write_atomic_bytes(
        cases_output_path, jsonl_bytes(ordered_records)
    )
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
        render_markdown_v2(report).encode("utf-8"),
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
