#!/usr/bin/env python3
"""Decision-aligned, evidence-scoped AEGIS analysis-v3.

This module is a derived analysis over the immutable v1 population.  It
imports the accepted v2 implementation without changing it, then adds three
scientific views that are required to decide whether task-aware flow steering
is the appropriate next intervention:

* fail-closed kinematic scope for robot contacts;
* paired temporal ordering of intervention, nominal-policy divergence, and
  native-goal divergence;
* an exact, mutually-exclusive decision partition.

The end-effector subtree is an authoritative kinematic classification, not a
certificate that the released proxy ellipsoid encloses every physical geom.
Likewise, temporal precedence is association rather than causal proof.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from analysis import aggregate_safelibero_aegis as aggregate
    from analysis import build_aegis_failure_report as v2
except ImportError:  # pragma: no cover - direct script fallback
    import aggregate_safelibero_aegis as aggregate  # type: ignore[no-redef]
    import build_aegis_failure_report as v2  # type: ignore[no-redef]


CASE_SCHEMA = "vlsa_table1_aegis_failure_case.v3"
REPORT_SCHEMA = "vlsa_table1_aegis_failure_report.v3"
SUMMARY_SCHEMA = "vlsa_table1_population_summary.v3"
EXPECTED_CASES = v2.EXPECTED_CASES
BASELINE_ARM = v2.BASELINE_ARM
AEGIS_ARM = v2.AEGIS_ARM

ROBOT_CONTACT_SCOPES = (
    "intended_eef_subtree",
    "unprotected_robot_link",
    "diagnostic_eef_marker",
)
DECISION_PARTITION = (
    "not_baseline_success_to_aegis_safe_failure",
    "excluded_baseline_already_safe",
    "excluded_no_valid_execution",
    "excluded_pipeline_not_complete_or_not_all_qp",
    "excluded_no_intervention",
    "excluded_unprotected_robot_link_contact",
    "excluded_sampled_physical_safety_unconfirmed",
    "excluded_no_negative_final_goal_delta",
    "excluded_temporal_precedence_unavailable",
    "strong_flow_candidate_association",
)

INTERPRETATION_SCOPE = {
    **v2.INTERPRETATION_SCOPE_V2,
    "robot_contact_scope": (
        "The intended-EFF class is the exact MuJoCo descendant subtree of "
        "robot0_right_hand. It is a kinematic-scope authority, not proof that "
        "the released ellipsoid encloses every subtree geom."
    ),
    "temporal_precedence": (
        "Intervention preceding nominal-policy and native-goal divergence is "
        "an observed paired ordering. It does not prove the intervention "
        "caused the later task outcome."
    ),
    "flow_candidate": (
        "A strong flow candidate is an observational stratum after explicit "
        "baseline-collision, perception, QP, execution, contact, progress, "
        "intervention, and temporal-chain gates. Baseline-already-safe "
        "degradations remain separate. This is not a causal efficacy claim "
        "for a learned flow method."
    ),
    "stale_geometry": (
        "The released obstacle geometry is fitted once before control. The "
        "immutable artifacts do not contain a refreshed true obstacle surface "
        "or continuous clearance, so stale-geometry causality is unresolved."
    ),
}


FailureReportV3Error = v2.FailureReportError


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    return v2._mapping(value, label=label)


def _list(value: Any, *, label: str) -> list[Any]:
    return v2._list(value, label=label)


def _int(value: Any, *, label: str) -> int:
    return v2._int(value, label=label)


def _number(value: Any, *, label: str) -> float:
    return v2._number(value, label=label)


def _bool(value: Any, *, label: str) -> bool:
    return v2._bool(value, label=label)


def _body_ancestry(
    parent_ids: Sequence[int],
    body_id: int,
) -> tuple[int, ...]:
    if body_id < 0 or body_id >= len(parent_ids):
        raise FailureReportV3Error(
            f"robot-scope body ID is out of range: {body_id}"
        )
    output: list[int] = []
    seen: set[int] = set()
    current = body_id
    while True:
        if current in seen:
            raise FailureReportV3Error(
                "robot-scope body ancestry contains a cycle"
            )
        if current < 0 or current >= len(parent_ids):
            raise FailureReportV3Error(
                "robot-scope body ancestry escapes the model"
            )
        seen.add(current)
        output.append(current)
        if current == 0:
            break
        current = parent_ids[current]
    return tuple(output)


def derive_robot_scope_authority(
    contact_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive an exact hand-subtree authority from frozen MuJoCo topology."""

    model = _mapping(
        contact_payload.get("model_authority"),
        label="contact model authority",
    )
    body_names_raw = _list(
        model.get("body_names"), label="contact body names"
    )
    parent_raw = _list(
        model.get("body_parent_ids"), label="contact body parents"
    )
    robot_raw = _list(
        model.get("robot_body_ids"), label="contact robot body IDs"
    )
    if (
        len(body_names_raw) != len(parent_raw)
        or not body_names_raw
        or any(
            not isinstance(name, str) or not name
            for name in body_names_raw
        )
        or any(type(value) is not int for value in parent_raw)
        or any(type(value) is not int for value in robot_raw)
    ):
        raise FailureReportV3Error(
            "contact topology cannot authorize robot contact scope"
        )
    body_names = [str(value) for value in body_names_raw]
    parent_ids = [int(value) for value in parent_raw]
    robot_ids = set(int(value) for value in robot_raw)
    if parent_ids[0] != 0:
        raise FailureReportV3Error(
            "contact topology world body must be its own root"
        )
    for body_id, parent_id in enumerate(parent_ids):
        if not 0 <= parent_id < len(parent_ids):
            raise FailureReportV3Error(
                f"body {body_id} has an invalid parent"
            )
        _body_ancestry(parent_ids, body_id)
    if not robot_ids or any(
        body_id < 0 or body_id >= len(body_names)
        for body_id in robot_ids
    ):
        raise FailureReportV3Error(
            "contact topology has invalid robot body IDs"
        )

    hand_ids = [
        index
        for index, name in enumerate(body_names)
        if name == "robot0_right_hand"
    ]
    marker_ids = [
        index for index, name in enumerate(body_names)
        if name == "eef_marker"
    ]
    if len(hand_ids) != 1 or len(marker_ids) != 1:
        raise FailureReportV3Error(
            "robot scope requires exactly one robot0_right_hand and "
            "one eef_marker"
        )
    hand_id = hand_ids[0]
    marker_id = marker_ids[0]
    if hand_id not in robot_ids or marker_id not in robot_ids:
        raise FailureReportV3Error(
            "hand root and EEF marker must be present in robot authority"
        )
    body_ancestries = {
        body_id: _body_ancestry(parent_ids, body_id)
        for body_id in range(len(body_names))
    }
    robot_contact_universe = {
        body_id
        for body_id, ancestry in body_ancestries.items()
        if set(ancestry).intersection(robot_ids)
    }
    eef_subtree = {
        body_id
        for body_id, ancestry in body_ancestries.items()
        if hand_id in ancestry
    }
    unprotected = robot_contact_universe - eef_subtree - {marker_id}
    if (
        not eef_subtree
        or hand_id not in eef_subtree
        or eef_subtree & unprotected
        or marker_id in eef_subtree
        or eef_subtree | unprotected | {marker_id}
        != robot_contact_universe
        or not robot_ids.issubset(robot_contact_universe)
    ):
        raise FailureReportV3Error(
            "robot contact scopes do not exactly partition robot bodies"
        )
    source_sha = model.get("authority_sha256")
    v2._sha(source_sha, label="contact model authority SHA")
    authority: dict[str, Any] = {
        "schema_version": "vlsa_table1_robot_contact_scope_authority.v1",
        "source_contact_model_authority_sha256": source_sha,
        "classification_source": (
            "exact MuJoCo body IDs and body_parent_ids"
        ),
        "hand_root": {
            "body_id": hand_id,
            "body_name": body_names[hand_id],
        },
        "diagnostic_marker": {
            "body_id": marker_id,
            "body_name": body_names[marker_id],
        },
        "robot_contact_body_universe_ids": sorted(
            robot_contact_universe
        ),
        "robot_contact_body_universe_names": [
            body_names[value]
            for value in sorted(robot_contact_universe)
        ],
        "robot_contact_universe_definition": (
            "all direct MuJoCo bodies whose ancestry intersects the "
            "frozen robot_body_ids role authority"
        ),
        "body_ids_by_scope": {
            "intended_eef_subtree": sorted(eef_subtree),
            "unprotected_robot_link": sorted(unprotected),
            "diagnostic_eef_marker": [marker_id],
        },
        "body_names_by_scope": {
            "intended_eef_subtree": [
                body_names[value] for value in sorted(eef_subtree)
            ],
            "unprotected_robot_link": [
                body_names[value] for value in sorted(unprotected)
            ],
            "diagnostic_eef_marker": [body_names[marker_id]],
        },
        "exact_partition": True,
        "ellipsoid_geometric_enclosure_certified": False,
        "claim": (
            "intended end-effector kinematic subtree only; not a "
            "physical-geom enclosure certificate"
        ),
    }
    authority["authority_sha256"] = v2.sha256_bytes(
        v2.canonical_json_bytes(authority)
    )
    return authority


def _robot_scope_for_body(
    authority: Mapping[str, Any],
    body_id: int,
) -> str:
    scopes = _mapping(
        authority.get("body_ids_by_scope"),
        label="robot scope body IDs",
    )
    matches = [
        scope
        for scope in ROBOT_CONTACT_SCOPES
        if body_id in _list(
            scopes.get(scope), label=f"robot scope/{scope}"
        )
    ]
    if len(matches) != 1:
        raise FailureReportV3Error(
            f"robot body {body_id} lacks one exact contact scope"
        )
    return matches[0]


def contact_scope_evidence_v3(
    contact_payload: Mapping[str, Any],
    *,
    collision_step: int | None,
) -> dict[str, Any]:
    """Classify every robot-active-obstacle event by exact kinematic scope."""

    authority = derive_robot_scope_authority(contact_payload)
    model = _mapping(
        contact_payload.get("model_authority"),
        label="contact model authority",
    )
    body_names = _list(
        model.get("body_names"), label="contact body names"
    )
    counts: Counter[str] = Counter()
    postcontrol: Counter[str] = Counter()
    through_collision: Counter[str] = Counter()
    body_histograms = {
        scope: Counter() for scope in ROBOT_CONTACT_SCOPES
    }
    geom_histograms = {
        scope: Counter() for scope in ROBOT_CONTACT_SCOPES
    }
    through_body_histograms = {
        scope: Counter() for scope in ROBOT_CONTACT_SCOPES
    }
    through_geom_histograms = {
        scope: Counter() for scope in ROBOT_CONTACT_SCOPES
    }
    robot_event_count = 0
    for snapshot_raw in _list(
        contact_payload.get("snapshots"), label="contact snapshots"
    ):
        snapshot = _mapping(snapshot_raw, label="contact snapshot")
        step = _int(snapshot.get("step"), label="contact snapshot step")
        for event_raw in _list(
            snapshot.get("events"), label="contact events"
        ):
            event = _mapping(event_raw, label="contact event")
            other = _mapping(event.get("other"), label="contact other")
            if other.get("classification") != "robot":
                continue
            if other.get("classification_authority") != "complete":
                raise FailureReportV3Error(
                    "robot contact lacks complete role authority"
                )
            body_id = _int(
                other.get("body_id"), label="robot contact body ID"
            )
            if not 0 <= body_id < len(body_names):
                raise FailureReportV3Error(
                    "robot contact body ID is outside frozen topology"
                )
            body_name = other.get("body_name")
            geom_name = other.get("geom_name")
            if (
                body_name != body_names[body_id]
                or not isinstance(geom_name, str)
                or not geom_name
            ):
                raise FailureReportV3Error(
                    "robot contact body/geom identity differs from authority"
                )
            scope = _robot_scope_for_body(authority, body_id)
            robot_event_count += 1
            counts[scope] += 1
            body_histograms[scope][str(body_name)] += 1
            geom_histograms[scope][geom_name] += 1
            if step >= 0:
                postcontrol[scope] += 1
            if (
                collision_step is not None
                and 0 <= step <= collision_step
            ):
                through_collision[scope] += 1
                through_body_histograms[scope][str(body_name)] += 1
                through_geom_histograms[scope][geom_name] += 1
    if sum(counts.values()) != robot_event_count:
        raise FailureReportV3Error(
            "robot contact-scope event denominator changed"
        )

    def complete(counter_by_scope: Mapping[str, Counter[str]]) -> dict[str, Any]:
        return {
            scope: dict(sorted(counter_by_scope[scope].items()))
            for scope in ROBOT_CONTACT_SCOPES
        }

    return {
        "schema_version": "vlsa_table1_robot_contact_scope_evidence.v1",
        "authority": authority,
        "robot_event_count": robot_event_count,
        "event_counts_by_scope": {
            scope: counts[scope] for scope in ROBOT_CONTACT_SCOPES
        },
        "postcontrol_event_counts_by_scope": {
            scope: postcontrol[scope]
            for scope in ROBOT_CONTACT_SCOPES
        },
        "events_through_paper_collision_by_scope": {
            scope: through_collision[scope]
            for scope in ROBOT_CONTACT_SCOPES
        },
        "body_event_histograms_by_scope": complete(body_histograms),
        "geom_event_histograms_by_scope": complete(geom_histograms),
        "body_event_histograms_through_paper_collision_by_scope": (
            complete(through_body_histograms)
        ),
        "geom_event_histograms_through_paper_collision_by_scope": (
            complete(through_geom_histograms)
        ),
        "sampled_unprotected_robot_link_contact": (
            counts["unprotected_robot_link"] > 0
        ),
        "sampled_intended_eef_subtree_contact": (
            counts["intended_eef_subtree"] > 0
        ),
        "sampled_diagnostic_eef_marker_contact": (
            counts["diagnostic_eef_marker"] > 0
        ),
        "scope_partition_complete": True,
        "ellipsoid_geometric_enclosure_certified": False,
    }


def geometry_evidence_v3(result: Mapping[str, Any]) -> dict[str, Any]:
    """Expose compact geometry quality while preserving causal limits."""

    base = v2._geometry_evidence_v2(result)
    case_id = str(result.get("case_id"))
    diagnostics = v2._diagnostics_record(result)
    descriptor = _mapping(
        diagnostics.get("geometry"), label=f"{case_id}/geometry"
    )
    record = _mapping(
        descriptor.get("record"), label=f"{case_id}/geometry record"
    )
    views = _mapping(record.get("views"), label=f"{case_id}/views")
    for view_name in ("agentview", "backview"):
        view = _mapping(views.get(view_name), label=f"{case_id}/{view_name}")
        detections = view.get("detections")
        selected_box = None
        selected_logit = None
        selected_crop = None
        if isinstance(detections, Mapping):
            boxes = detections.get("returned_order_boxes_xyxy")
            logits = detections.get("returned_order_logits")
            index = detections.get("selected_index")
            if (
                type(index) is int
                and isinstance(boxes, list)
                and 0 <= index < len(boxes)
            ):
                selected_box = boxes[index]
            if (
                type(index) is int
                and isinstance(logits, list)
                and 0 <= index < len(logits)
            ):
                selected_logit = logits[index]
            selected_crop = detections.get("selected_crop")
        base["views"][view_name].update(
            {
                "selected_box_xyxy_normalized": selected_box,
                "selected_logit": selected_logit,
                "selected_crop": selected_crop,
                "box_overlap_with_true_obstacle_certified": False,
            }
        )
    mvee = record.get("mvee")
    base["mvee_quality"] = (
        None
        if not isinstance(mvee, Mapping)
        else {
            key: mvee.get(key)
            for key in (
                "status",
                "center",
                "semiaxes",
                "eigenvalues",
                "maximum_hull_quadratic_value",
                "maximum_enclosure_violation",
            )
        }
    )
    initial_direction = record.get("initial_direction")
    formula = (
        initial_direction.get("formula")
        if isinstance(initial_direction, Mapping)
        else None
    )
    if base["status"] == "complete" and formula != (
        "normalize(mvee_center - stale_pre_settle_eef_proxy_center)"
    ):
        raise FailureReportV3Error(
            f"{case_id}: complete geometry lacks pre-settle proxy provenance"
        )
    base["geometry_lifecycle"] = {
        "successful_precontrol_fit_count": (
            1 if base["status"] == "complete" else 0
        ),
        "reperception_or_refit_during_control": False,
        "first_direction_proxy_source": (
            "pre_settle_eef_proxy" if formula is not None else None
        ),
        "stale_geometry_causal_claim_supported": False,
    }
    base["true_obstacle_mesh_enclosure_certified"] = False
    return base


def _numeric_vector(value: Any, *, label: str) -> list[float]:
    raw = _list(value, label=label)
    if len(raw) != 7:
        raise FailureReportV3Error(f"{label} must have seven values")
    return [_number(item, label=f"{label}[{index}]") for index, item in enumerate(raw)]


def _l2_difference(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(
        sum((left - right) ** 2 for left, right in zip(first, second))
    )


def paired_temporal_evidence_v3(
    baseline: Mapping[str, Any],
    aegis: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare paired nominal-policy and native-goal traces read-only."""

    case_id = str(baseline.get("case_id"))
    if aegis.get("case_id") != case_id:
        raise FailureReportV3Error("temporal evidence pair identity differs")
    baseline_actions = _list(
        baseline.get("actions"), label=f"{case_id}/baseline actions"
    )
    aegis_actions = _list(
        aegis.get("actions"), label=f"{case_id}/AEGIS actions"
    )
    first_intervention: int | None = None
    aegis_regressions_after_intervention = 0
    for index, raw in enumerate(aegis_actions):
        action = _mapping(raw, label=f"{case_id}/AEGIS action {index}")
        if _int(action.get("step"), label="AEGIS action step") != index:
            raise FailureReportV3Error("AEGIS action steps are not contiguous")
        modified = _bool(action.get("modified"), label="AEGIS modified")
        if modified and first_intervention is None:
            first_intervention = index
    common = min(len(baseline_actions), len(aegis_actions))
    first_nominal_divergence: int | None = None
    xyz_divergences: list[float] = []
    full_divergences: list[float] = []
    for index in range(common):
        baseline_action = _mapping(
            baseline_actions[index],
            label=f"{case_id}/baseline action {index}",
        )
        aegis_action = _mapping(
            aegis_actions[index],
            label=f"{case_id}/AEGIS action {index}",
        )
        if (
            _int(
                baseline_action.get("step"),
                label=f"{case_id}/baseline action step",
            )
            != index
            or _int(
                aegis_action.get("step"),
                label=f"{case_id}/AEGIS action step",
            )
            != index
        ):
            raise FailureReportV3Error(
                "paired temporal action steps are not contiguous"
            )
        first = _numeric_vector(
            baseline_action.get("nominal_translational"),
            label=f"{case_id}/baseline nominal {index}",
        )
        second = _numeric_vector(
            aegis_action.get("nominal_translational"),
            label=f"{case_id}/AEGIS nominal {index}",
        )
        xyz = _l2_difference(first[:3], second[:3])
        full = _l2_difference(first, second)
        xyz_divergences.append(xyz)
        full_divergences.append(full)
        if (
            first_nominal_divergence is None
            and v2.canonical_json_bytes(first)
            != v2.canonical_json_bytes(second)
        ):
            first_nominal_divergence = index

    baseline_goal = _mapping(
        baseline.get("goal_progress"), label=f"{case_id}/baseline goal"
    )
    aegis_goal = _mapping(
        aegis.get("goal_progress"), label=f"{case_id}/AEGIS goal"
    )
    baseline_initial = _list(
        _mapping(
            baseline_goal.get("initial"),
            label=f"{case_id}/baseline initial goal",
        ).get("values"),
        label=f"{case_id}/baseline initial goal values",
    )
    aegis_initial = _list(
        _mapping(
            aegis_goal.get("initial"),
            label=f"{case_id}/AEGIS initial goal",
        ).get("values"),
        label=f"{case_id}/AEGIS initial goal values",
    )
    if baseline_initial != aegis_initial:
        raise FailureReportV3Error(
            "paired temporal evidence starts from different native goals"
        )
    first_goal_divergence: int | None = None
    goal_hamming_divergences: list[int] = []
    goal_fraction_divergences: list[float] = []
    for index in range(common):
        baseline_snapshot = _mapping(
            _mapping(
                baseline_actions[index],
                label=f"{case_id}/baseline action {index}",
            ).get("goal_progress"),
            label=f"{case_id}/baseline goal {index}",
        )
        aegis_snapshot = _mapping(
            _mapping(
                aegis_actions[index],
                label=f"{case_id}/AEGIS action {index}",
            ).get("goal_progress"),
            label=f"{case_id}/AEGIS goal {index}",
        )
        if (
            _int(
                baseline_snapshot.get("step"),
                label=f"{case_id}/baseline goal step {index}",
            )
            != index
            or _int(
                aegis_snapshot.get("step"),
                label=f"{case_id}/AEGIS goal step {index}",
            )
            != index
        ):
            raise FailureReportV3Error(
                "paired native-goal snapshots are not step aligned"
            )
        baseline_values = _list(
            baseline_snapshot.get("values"),
            label=f"{case_id}/baseline goal values {index}",
        )
        aegis_values = _list(
            aegis_snapshot.get("values"),
            label=f"{case_id}/AEGIS goal values {index}",
        )
        if (
            len(baseline_values) != len(aegis_values)
            or not baseline_values
            or any(type(value) is not bool for value in baseline_values)
            or any(type(value) is not bool for value in aegis_values)
        ):
            raise FailureReportV3Error(
                "paired native-goal vectors differ in shape or type"
            )
        hamming = sum(
            left is not right
            for left, right in zip(baseline_values, aegis_values)
        )
        goal_hamming_divergences.append(hamming)
        goal_fraction_divergences.append(
            hamming / len(baseline_values)
        )
        if (
            first_goal_divergence is None
            and hamming > 0
        ):
            first_goal_divergence = index
        if first_intervention is not None and index >= first_intervention:
            aegis_regressions_after_intervention += len(
                _list(
                    aegis_snapshot.get("regressed_indices"),
                    label=f"{case_id}/AEGIS regressions {index}",
                )
            )

    baseline_queries = _list(
        baseline.get("policy_queries"),
        label=f"{case_id}/baseline queries",
    )
    aegis_queries = _list(
        aegis.get("policy_queries"), label=f"{case_id}/AEGIS queries"
    )
    first_query_divergence: int | None = None
    for index in range(min(len(baseline_queries), len(aegis_queries))):
        baseline_query = _mapping(
            baseline_queries[index],
            label=f"{case_id}/baseline query {index}",
        )
        aegis_query = _mapping(
            aegis_queries[index],
            label=f"{case_id}/AEGIS query {index}",
        )
        if (
            _int(
                baseline_query.get("query_index"),
                label=f"{case_id}/baseline query index",
            )
            != index
            or _int(
                aegis_query.get("query_index"),
                label=f"{case_id}/AEGIS query index",
            )
            != index
        ):
            raise FailureReportV3Error(
                "paired policy-query records are not contiguous"
            )
        if (
            baseline_query.get("returned_actions_sha256")
            != aegis_query.get("returned_actions_sha256")
        ):
            first_query_divergence = index
            break

    post_start = common if first_intervention is None else first_intervention
    post_xyz = xyz_divergences[post_start:]
    post_full = full_divergences[post_start:]
    post_goal_hamming = goal_hamming_divergences[post_start:]
    post_goal_fraction = goal_fraction_divergences[post_start:]
    intervention_before_nominal = (
        None
        if first_intervention is None
        or first_nominal_divergence is None
        else first_intervention <= first_nominal_divergence
    )
    intervention_before_goal = (
        None
        if first_intervention is None
        or first_goal_divergence is None
        else first_intervention <= first_goal_divergence
    )
    intervention_strictly_before_nominal = (
        first_intervention is not None
        and first_nominal_divergence is not None
        and first_intervention < first_nominal_divergence
    )
    nominal_before_goal = (
        first_nominal_divergence is not None
        and first_goal_divergence is not None
        and first_nominal_divergence <= first_goal_divergence
    )
    registered_temporal_chain = (
        intervention_strictly_before_nominal and nominal_before_goal
    )
    return {
        "schema_version": "vlsa_table1_paired_temporal_evidence.v1",
        "source_immutable_records": {
            "baseline_result_payload_sha256": v2._sha(
                baseline.get("result_payload_sha256"),
                label=f"{case_id}/baseline result payload SHA",
            ),
            "aegis_result_payload_sha256": v2._sha(
                aegis.get("result_payload_sha256"),
                label=f"{case_id}/AEGIS result payload SHA",
            ),
            "baseline_actions_sha256": v2.sha256_bytes(
                v2.canonical_json_bytes(baseline_actions)
            ),
            "aegis_actions_sha256": v2.sha256_bytes(
                v2.canonical_json_bytes(aegis_actions)
            ),
            "baseline_policy_queries_sha256": v2.sha256_bytes(
                v2.canonical_json_bytes(baseline_queries)
            ),
            "aegis_policy_queries_sha256": v2.sha256_bytes(
                v2.canonical_json_bytes(aegis_queries)
            ),
        },
        "paired_common_action_steps": common,
        "baseline_executed_action_count": len(baseline_actions),
        "aegis_executed_action_count": len(aegis_actions),
        "first_intervention_step": first_intervention,
        "first_nominal_action_exact_divergence_step": (
            first_nominal_divergence
        ),
        "first_policy_query_action_hash_divergence_index": (
            first_query_divergence
        ),
        "first_native_goal_vector_divergence_step": (
            first_goal_divergence
        ),
        "post_intervention_common_step_count": len(post_xyz),
        "post_intervention_nominal_xyz_l2_sum": sum(post_xyz),
        "post_intervention_nominal_xyz_l2_max": (
            max(post_xyz, default=0.0)
        ),
        "post_intervention_nominal_full_action_l2_sum": sum(post_full),
        "post_intervention_nominal_full_action_l2_max": (
            max(post_full, default=0.0)
        ),
        "post_intervention_native_goal_hamming_sum": sum(
            post_goal_hamming
        ),
        "post_intervention_native_goal_hamming_max": max(
            post_goal_hamming, default=0
        ),
        "post_intervention_native_goal_fraction_sum": sum(
            post_goal_fraction
        ),
        "post_intervention_native_goal_fraction_max": max(
            post_goal_fraction, default=0.0
        ),
        "first_native_goal_divergence_hamming": (
            None
            if first_goal_divergence is None
            else goal_hamming_divergences[first_goal_divergence]
        ),
        "first_native_goal_divergence_fraction": (
            None
            if first_goal_divergence is None
            else goal_fraction_divergences[first_goal_divergence]
        ),
        "aegis_goal_regression_count_at_or_after_first_intervention": (
            aegis_regressions_after_intervention
        ),
        "intervention_precedes_or_coincides_nominal_divergence": (
            intervention_before_nominal
        ),
        "intervention_precedes_or_coincides_goal_divergence": (
            intervention_before_goal
        ),
        "intervention_precedes_or_coincides_both_divergences": (
            intervention_before_nominal is True
            and intervention_before_goal is True
        ),
        "intervention_strictly_precedes_nominal_divergence": (
            intervention_strictly_before_nominal
        ),
        "nominal_divergence_precedes_or_coincides_goal_divergence": (
            nominal_before_goal
        ),
        "registered_intervention_nominal_goal_chain": (
            registered_temporal_chain
        ),
        "causal_claim_supported": False,
        "interpretation": (
            "same-noise paired temporal association; later divergence may "
            "reflect state feedback after intervention"
        ),
    }


def decision_evidence_v3(record: Mapping[str, Any]) -> dict[str, Any]:
    outcomes = _mapping(record.get("outcomes"), label="decision outcomes")
    baseline = _mapping(
        outcomes.get(BASELINE_ARM), label="decision baseline"
    )
    aegis = _mapping(outcomes.get(AEGIS_ARM), label="decision AEGIS")
    diagnostics = _mapping(
        record.get("aegis_diagnostics"), label="decision diagnostics"
    )
    geometry = _mapping(
        diagnostics.get("geometry_v3"), label="decision geometry"
    )
    control = _mapping(
        diagnostics.get("control"), label="decision control"
    )
    intervention = _mapping(
        diagnostics.get("intervention"), label="decision intervention"
    )
    contacts = _mapping(
        record.get("physical_contacts"), label="decision contacts"
    )
    aegis_contact = _mapping(
        contacts.get(AEGIS_ARM), label="decision AEGIS contacts"
    )
    scope = _mapping(
        aegis_contact.get("robot_contact_scope_v3"),
        label="decision robot contact scope",
    )
    progress = _mapping(
        record.get("paired_goal_progress"), label="decision progress"
    )
    delta = _mapping(
        progress.get("aegis_minus_pi05"), label="decision progress delta"
    )
    temporal = _mapping(
        record.get("paired_temporal_evidence_v3"),
        label="decision temporal evidence",
    )
    executed = _int(
        aegis.get("executed_action_count"), label="decision executed actions"
    )
    control_counts = _mapping(
        control.get("control_path_counts"), label="decision control paths"
    )
    all_qp = (
        executed > 0
        and set(control_counts) == {"aegis_qp"}
        and _int(
            control_counts.get("aegis_qp"),
            label="decision QP action count",
        )
        == executed
        and control.get("terminal_qp_failure") is None
    )
    geometry_complete = (
        geometry.get("status") == "complete"
        and geometry.get("mvee_status") == "captured"
    )
    stratum = (
        _bool(
            baseline.get("task_success"),
            label="decision baseline success",
        )
        and not _bool(
            aegis.get("paper_collision"),
            label="decision AEGIS collision",
        )
        and not _bool(
            aegis.get("task_success"),
            label="decision AEGIS success",
        )
    )
    baseline_paper_collision = _bool(
        baseline.get("paper_collision"),
        label="decision baseline collision",
    )
    baseline_safe_degradation = (
        stratum and not baseline_paper_collision
    )
    nonzero_execution = executed > 0
    intervened = (
        _int(
            intervention.get("intervention_count"),
            label="decision intervention count",
        )
        > 0
    )
    collision_relevant_events = _int(
        aegis_contact.get("collision_relevant_event_count"),
        label="decision all sampled collision-relevant contacts",
    )
    postcontrol_collision_relevant_events = _int(
        aegis_contact.get("postcontrol_collision_relevant_event_count"),
        label="decision postcontrol collision-relevant contacts",
    )
    physical_safety_confirmed = (
        collision_relevant_events == 0
        and postcontrol_collision_relevant_events == 0
    )
    unprotected_contact = _bool(
        scope.get("sampled_unprotected_robot_link_contact"),
        label="decision unprotected-link contact",
    )
    negative_final_goal_delta = (
        _number(
            delta.get("final_fraction"),
            label="decision final goal delta",
        )
        < 0.0
    )
    nominal_temporal_precedence = (
        temporal.get(
            "intervention_strictly_precedes_nominal_divergence"
        )
        is True
    )
    nominal_before_goal = (
        temporal.get(
            "nominal_divergence_precedes_or_coincides_goal_divergence"
        )
        is True
    )
    temporal_precedence = (
        temporal.get(
            "registered_intervention_nominal_goal_chain"
        )
        is True
    )
    if temporal_precedence != (
        nominal_temporal_precedence and nominal_before_goal
    ):
        raise FailureReportV3Error(
            "registered intervention/nominal/goal chain is inconsistent"
        )
    if not stratum:
        partition = "not_baseline_success_to_aegis_safe_failure"
    elif baseline_safe_degradation:
        partition = "excluded_baseline_already_safe"
    elif not nonzero_execution:
        partition = "excluded_no_valid_execution"
    elif not geometry_complete or not all_qp:
        partition = "excluded_pipeline_not_complete_or_not_all_qp"
    elif not intervened:
        partition = "excluded_no_intervention"
    elif unprotected_contact:
        partition = "excluded_unprotected_robot_link_contact"
    elif not physical_safety_confirmed:
        partition = "excluded_sampled_physical_safety_unconfirmed"
    elif not negative_final_goal_delta:
        partition = "excluded_no_negative_final_goal_delta"
    elif not temporal_precedence:
        partition = "excluded_temporal_precedence_unavailable"
    else:
        partition = "strong_flow_candidate_association"
    if partition not in DECISION_PARTITION:
        raise FailureReportV3Error("flow decision partition is invalid")
    return {
        "schema_version": "vlsa_table1_flow_direction_evidence.v1",
        "baseline_success_to_aegis_safe_failure": stratum,
        "baseline_paper_collision": baseline_paper_collision,
        "baseline_safe_success_to_aegis_safe_failure_degradation": (
            baseline_safe_degradation
        ),
        "geometry_complete": geometry_complete,
        "all_executed_actions_used_solved_aegis_qp": all_qp,
        "nonzero_execution": nonzero_execution,
        "intervention_observed": intervened,
        "sampled_physical_safety_confirmed": (
            physical_safety_confirmed
        ),
        "sampled_collision_relevant_event_count": (
            collision_relevant_events
        ),
        "sampled_postcontrol_collision_relevant_event_count": (
            postcontrol_collision_relevant_events
        ),
        "sampled_unprotected_robot_link_contact": (
            unprotected_contact
        ),
        "negative_final_goal_fraction_delta": (
            negative_final_goal_delta
        ),
        "negative_maximum_goal_fraction_delta": (
            _number(
                delta.get("maximum_fraction"),
                label="decision maximum goal delta",
            )
            < 0.0
        ),
        "increased_goal_regression_count": (
            _int(
                delta.get("regression_count"),
                label="decision regression delta",
            )
            > 0
        ),
        "intervention_precedes_or_coincides_goal_divergence": (
            temporal.get(
                "intervention_precedes_or_coincides_goal_divergence"
            )
            is True
        ),
        "intervention_strictly_precedes_nominal_divergence": (
            nominal_temporal_precedence
        ),
        "nominal_divergence_precedes_or_coincides_goal_divergence": (
            nominal_before_goal
        ),
        "registered_intervention_nominal_goal_chain": (
            temporal_precedence
        ),
        "decision_partition_class": partition,
        "strong_flow_candidate_association": (
            partition == "strong_flow_candidate_association"
        ),
        "causal_flow_claim_supported": False,
    }


def build_case_record_v3(
    *,
    manifest: Mapping[str, Any],
    baseline: Mapping[str, Any],
    aegis: Mapping[str, Any],
    baseline_artifact_root: Path,
    aegis_artifact_root: Path,
) -> dict[str, Any]:
    record = v2.build_case_record_v2(
        manifest=manifest,
        baseline=baseline,
        aegis=aegis,
        baseline_artifact_root=baseline_artifact_root,
        aegis_artifact_root=aegis_artifact_root,
    )
    record.pop("record_payload_sha256", None)
    baseline_metrics = v2._result_metrics(baseline)
    aegis_metrics = v2._result_metrics(aegis)
    baseline_payload = v2.load_contact_payload(
        baseline, artifact_root=baseline_artifact_root
    )
    aegis_payload = v2.load_contact_payload(
        aegis, artifact_root=aegis_artifact_root
    )
    for arm, payload, metrics in (
        (BASELINE_ARM, baseline_payload, baseline_metrics),
        (AEGIS_ARM, aegis_payload, aegis_metrics),
    ):
        record["physical_contacts"][arm][
            "robot_contact_scope_v3"
        ] = contact_scope_evidence_v3(
            payload, collision_step=metrics["collision_first_step"]
        )
    record["aegis_diagnostics"]["geometry_v3"] = (
        geometry_evidence_v3(aegis)
    )
    record["paired_temporal_evidence_v3"] = (
        paired_temporal_evidence_v3(baseline, aegis)
    )
    record["schema_version"] = CASE_SCHEMA
    record["analysis_revision"] = {
        "name": "decision_aligned_analysis_v3",
        "accepted_v2_reused_without_mutation": True,
    }
    record["interpretation_scope"] = INTERPRETATION_SCOPE
    record["flow_direction_evidence_v3"] = decision_evidence_v3(record)
    record["record_payload_sha256"] = v2.sha256_bytes(
        v2.canonical_json_bytes(record)
    )
    return record


def _merge_histogram(
    target: Counter[str],
    raw: Mapping[str, Any],
) -> None:
    for key, value in raw.items():
        target[str(key)] += _int(value, label=f"histogram/{key}")


def contact_scope_counts_v3(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"denominator": len(records), "arms": {}}
    for arm in (BASELINE_ARM, AEGIS_ARM):
        event_counts: Counter[str] = Counter()
        post_counts: Counter[str] = Counter()
        through_counts: Counter[str] = Counter()
        case_counts: Counter[str] = Counter()
        body_hist = {
            scope: Counter() for scope in ROBOT_CONTACT_SCOPES
        }
        through_body_hist = {
            scope: Counter() for scope in ROBOT_CONTACT_SCOPES
        }
        authority_hashes: set[str] = set()
        for record in records:
            scope_record = _mapping(
                record["physical_contacts"][arm][
                    "robot_contact_scope_v3"
                ],
                label=f"{arm}/robot contact scope",
            )
            authority = _mapping(
                scope_record.get("authority"),
                label=f"{arm}/robot scope authority",
            )
            authority_hashes.add(
                v2._sha(
                    authority.get("authority_sha256"),
                    label=f"{arm}/robot scope authority SHA",
                )
            )
            for scope in ROBOT_CONTACT_SCOPES:
                count = _int(
                    scope_record["event_counts_by_scope"][scope],
                    label=f"{arm}/{scope}/events",
                )
                event_counts[scope] += count
                post_counts[scope] += _int(
                    scope_record["postcontrol_event_counts_by_scope"][
                        scope
                    ],
                    label=f"{arm}/{scope}/postcontrol",
                )
                through_counts[scope] += _int(
                    scope_record[
                        "events_through_paper_collision_by_scope"
                    ][scope],
                    label=f"{arm}/{scope}/through collision",
                )
                case_counts[scope] += count > 0
                _merge_histogram(
                    body_hist[scope],
                    _mapping(
                        scope_record[
                            "body_event_histograms_by_scope"
                        ][scope],
                        label=f"{arm}/{scope}/body histogram",
                    ),
                )
                _merge_histogram(
                    through_body_hist[scope],
                    _mapping(
                        scope_record[
                            "body_event_histograms_through_"
                            "paper_collision_by_scope"
                        ][scope],
                        label=f"{arm}/{scope}/collision body histogram",
                    ),
                )
        output["arms"][arm] = {
            "event_counts_by_scope": {
                scope: event_counts[scope]
                for scope in ROBOT_CONTACT_SCOPES
            },
            "postcontrol_event_counts_by_scope": {
                scope: post_counts[scope]
                for scope in ROBOT_CONTACT_SCOPES
            },
            "events_through_paper_collision_by_scope": {
                scope: through_counts[scope]
                for scope in ROBOT_CONTACT_SCOPES
            },
            "case_counts_by_scope": {
                scope: case_counts[scope]
                for scope in ROBOT_CONTACT_SCOPES
            },
            "body_event_histograms_by_scope": {
                scope: dict(sorted(body_hist[scope].items()))
                for scope in ROBOT_CONTACT_SCOPES
            },
            "body_event_histograms_through_paper_collision_by_scope": {
                scope: dict(sorted(through_body_hist[scope].items()))
                for scope in ROBOT_CONTACT_SCOPES
            },
            "authority_instance_count": len(authority_hashes),
            "scope_partition_complete": True,
            "ellipsoid_geometric_enclosure_certified": False,
        }
    return output


def decision_counts_v3(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    partition = Counter(
        str(
            record["flow_direction_evidence_v3"][
                "decision_partition_class"
            ]
        )
        for record in records
    )
    complete_partition = {
        key: partition[key] for key in DECISION_PARTITION
    }
    if (
        set(partition) - set(DECISION_PARTITION)
        or sum(complete_partition.values()) != len(records)
    ):
        raise FailureReportV3Error(
            "flow decision classes do not exactly partition cases"
        )
    stratum = [
        record
        for record in records
        if record["flow_direction_evidence_v3"][
            "baseline_success_to_aegis_safe_failure"
        ]
    ]
    candidates = [
        record
        for record in records
        if record["flow_direction_evidence_v3"][
            "strong_flow_candidate_association"
        ]
    ]

    def correction_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        corrections = [
            _number(
                row["aegis_diagnostics"]["intervention"][
                    "correction_l2_sum"
                ],
                label="decision correction L2 sum",
            )
            for row in rows
        ]
        retention = [
            _number(
                row["aegis_diagnostics"]["control"][
                    "translation_retention_ratio"
                ],
                label="decision translation retention",
            )
            for row in rows
            if row["aegis_diagnostics"]["control"][
                "translation_retention_ratio"
            ]
            is not None
        ]
        return {
            "denominator": len(rows),
            "correction_l2_sum": sum(corrections),
            "correction_l2_mean": (
                None if not rows else sum(corrections) / len(rows)
            ),
            "correction_l2_max": max(corrections, default=0.0),
            "translation_retention_ratio_count": len(retention),
            "translation_retention_ratio_mean": (
                None if not retention else sum(retention) / len(retention)
            ),
            "translation_retention_ratio_min": (
                None if not retention else min(retention)
            ),
            "translation_retention_ratio_max": (
                None if not retention else max(retention)
            ),
        }

    return {
        "denominator": len(records),
        "decision_partition_class_counts": complete_partition,
        "exact_partition": True,
        "baseline_success_to_aegis_safe_failure": {
            "count": len(stratum),
            "baseline_collision_count": sum(
                bool(
                    row["flow_direction_evidence_v3"][
                        "baseline_paper_collision"
                    ]
                )
                for row in stratum
            ),
            "baseline_safe_count": sum(
                not bool(
                    row["flow_direction_evidence_v3"][
                        "baseline_paper_collision"
                    ]
                )
                for row in stratum
            ),
            "geometry_complete_and_all_qp_count": sum(
                row["flow_direction_evidence_v3"][
                    "geometry_complete"
                ]
                and row["flow_direction_evidence_v3"][
                    "all_executed_actions_used_solved_aegis_qp"
                ]
                for row in stratum
            ),
            "intervention_observed_count": sum(
                bool(
                    row["flow_direction_evidence_v3"][
                        "intervention_observed"
                    ]
                )
                for row in stratum
            ),
            "sampled_physical_safety_confirmed_count": sum(
                bool(
                    row["flow_direction_evidence_v3"][
                        "sampled_physical_safety_confirmed"
                    ]
                )
                for row in stratum
            ),
            "negative_final_goal_fraction_delta_count": sum(
                bool(
                    row["flow_direction_evidence_v3"][
                        "negative_final_goal_fraction_delta"
                    ]
                )
                for row in stratum
            ),
            "strict_zero_translation_count": sum(
                bool(
                    row["aegis_diagnostics"]["control"][
                        "strict_zero_translation"
                    ]
                )
                for row in stratum
            ),
            "correction": correction_summary(stratum),
        },
        "baseline_safe_success_to_aegis_safe_failure_degradation": {
            "count": sum(
                bool(
                    row["flow_direction_evidence_v3"][
                        "baseline_safe_success_to_aegis_safe_"
                        "failure_degradation"
                    ]
                )
                for row in stratum
            ),
            "strong_flow_candidate_count": 0,
            "interpretation": (
                "baseline was already paper-safe; inspect false-positive "
                "perception or gating before collision-avoidance flow"
            ),
        },
        "strong_flow_candidate_association": {
            "count": len(candidates),
            "correction": correction_summary(candidates),
            "causal_claim_supported": False,
        },
        "causal_flow_claim_supported": False,
    }


def pipeline_counts_v3(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    geometry_status = Counter()
    mvee_status = Counter()
    failure_type = Counter()
    all_qp = 0
    for record in records:
        geometry = record["aegis_diagnostics"]["geometry_v3"]
        geometry_status[str(geometry.get("status"))] += 1
        mvee_status[str(geometry.get("mvee_status"))] += 1
        if geometry.get("failure_type") is not None:
            failure_type[str(geometry.get("failure_type"))] += 1
        all_qp += bool(
            record["flow_direction_evidence_v3"][
                "all_executed_actions_used_solved_aegis_qp"
            ]
        )
    return {
        "denominator": len(records),
        "geometry_status_counts": dict(sorted(geometry_status.items())),
        "mvee_status_counts": dict(sorted(mvee_status.items())),
        "geometry_failure_type_counts": dict(sorted(failure_type.items())),
        "all_executed_actions_solved_aegis_qp_case_count": all_qp,
        "detector_box_correctness_certified": False,
        "true_obstacle_mesh_enclosure_certified": False,
        "stale_geometry_causal_claim_supported": False,
    }


def validate_report_against_summary_v3(
    records: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    expected_cases: int = EXPECTED_CASES,
) -> dict[str, Any]:
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status")
        != "complete_postpublication_analysis_v3"
    ):
        raise FailureReportV3Error("analysis-v3 summary is not complete")
    projection = dict(summary)
    projection["schema_version"] = v2.SUMMARY_SCHEMA_V2
    projection["status"] = "complete_postpublication_analysis_v2"
    base = v2.validate_report_against_summary_v2(
        records,
        summary=projection,
        expected_cases=expected_cases,
    )
    if len(records) != expected_cases:
        raise FailureReportV3Error(
            "analysis-v3 case denominator changed"
        )
    for record in records:
        if record.get("schema_version") != CASE_SCHEMA:
            raise FailureReportV3Error(
                f"{record.get('case_id')}: wrong analysis-v3 schema"
            )
        expected_hash = v2._sha(
            record.get("record_payload_sha256"),
            label=f"{record.get('case_id')}/v3 payload SHA",
        )
        payload = dict(record)
        payload.pop("record_payload_sha256", None)
        if v2.sha256_bytes(v2.canonical_json_bytes(payload)) != expected_hash:
            raise FailureReportV3Error(
                f"{record.get('case_id')}: v3 payload hash differs"
            )
    return {
        **base,
        "robot_contact_scope_v3": contact_scope_counts_v3(records),
        "pipeline_v3": pipeline_counts_v3(records),
        "flow_direction_decision_v3": decision_counts_v3(records),
    }


def build_report_v3(
    records: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    summary_sha256: str,
    validation_receipt_sha256: str,
    source_publication_receipt_sha256: str,
    expected_cases: int = EXPECTED_CASES,
    streaming_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    counts = validate_report_against_summary_v3(
        records, summary=summary, expected_cases=expected_cases
    )
    ordered = sorted(
        records, key=lambda row: int(row.get("case_ordinal"))
    )
    ledger = [
        {
            "case_id": row["case_id"],
            "record_payload_sha256": row["record_payload_sha256"],
        }
        for row in ordered
    ]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete_postpublication_failure_analysis_v3",
        "protocol_id": summary.get("protocol_id"),
        "claim_scope": dict(
            _mapping(summary.get("claim_scope"), label="v3 claim scope")
        ),
        "source": {
            "v1_publication_receipt_sha256": v2._sha(
                source_publication_receipt_sha256,
                label="v3 source publication SHA",
            ),
            "population_summary_v3_sha256": v2._sha(
                summary_sha256, label="v3 summary SHA"
            ),
            "population_validation_receipt_sha256": v2._sha(
                validation_receipt_sha256,
                label="v3 validation receipt SHA",
            ),
            "accepted_result_payloads_sha256": v2._sha(
                summary.get("accepted_result_payloads_sha256"),
                label="v3 result ledger SHA",
            ),
        },
        "population": {
            "cases": expected_cases,
            "results": expected_cases * 2,
            "no_cases_dropped": True,
            "exact_decision_partition": True,
            "causal_limits_preserved": True,
        },
        "counts": counts,
        "validation_memory_shape": {
            "streaming_unit": "one_case_pair",
            "maximum_live_full_result_records": 2,
            "full_result_records_retained": 0,
            "retained_records": "compact_failure_rows_v3_only",
            **({} if streaming_stats is None else dict(streaming_stats)),
        },
        "case_record_ledger_sha256": v2.sha256_bytes(
            v2.canonical_json_bytes(ledger)
        ),
        "interpretation_scope": INTERPRETATION_SCOPE,
    }
    report["report_payload_sha256"] = v2.sha256_bytes(
        v2.canonical_json_bytes(report)
    )
    return report


def render_markdown_v3(report: Mapping[str, Any]) -> str:
    counts = _mapping(report.get("counts"), label="v3 counts")
    decision = _mapping(
        counts.get("flow_direction_decision_v3"),
        label="v3 flow decision",
    )
    partition = _mapping(
        decision.get("decision_partition_class_counts"),
        label="v3 decision partition",
    )
    contacts = _mapping(
        counts.get("robot_contact_scope_v3"),
        label="v3 robot contact scopes",
    )
    lines = [
        "# SafeLIBERO decision-aligned analysis v3",
        "",
        (
            "This report derives observational evidence from the exact "
            "immutable paired population. It makes no causal flow-steering, "
            "continuous-clearance, or proxy-enclosure claim."
        ),
        "",
        "## Robot-contact kinematic scope",
        "",
        (
            "`intended_eef_subtree` means exact descendants of "
            "`robot0_right_hand`; it does not certify ellipsoid enclosure."
        ),
        "",
        "| Arm | Intended EFF events | Unprotected-link events | EEF-marker events |",
        "|---|---:|---:|---:|",
    ]
    for arm in (BASELINE_ARM, AEGIS_ARM):
        arm_counts = contacts["arms"][arm]["event_counts_by_scope"]
        lines.append(
            f"| {arm} | {arm_counts['intended_eef_subtree']} | "
            f"{arm_counts['unprotected_robot_link']} | "
            f"{arm_counts['diagnostic_eef_marker']} |"
        )
    stratum = _mapping(
        decision.get("baseline_success_to_aegis_safe_failure"),
        label="v3 decision stratum",
    )
    candidate = _mapping(
        decision.get("strong_flow_candidate_association"),
        label="v3 flow candidate",
    )
    baseline_safe_degradation = _mapping(
        decision.get(
            "baseline_safe_success_to_aegis_safe_failure_degradation"
        ),
        label="v3 baseline-safe degradation",
    )
    lines.extend(
        [
            "",
            "## Task-aware flow decision stratum",
            "",
            (
                "Baseline-task-success to AEGIS-paper-safe task failure: "
                f"**{stratum.get('count')}** cases."
            ),
            (
                "Strong post-gate flow-candidate associations: "
                f"**{candidate.get('count')}** cases."
            ),
            (
                "Baseline-already-safe degradations (reported separately, "
                "never strong flow candidates): "
                f"**{baseline_safe_degradation.get('count')}** cases."
            ),
            "",
            "| Exact decision partition | Cases |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{key}` | {partition[key]} |"
        for key in DECISION_PARTITION
    )
    lines.extend(["", "## Causal boundaries", ""])
    lines.extend(
        f"- **{key}:** {value}"
        for key, value in INTERPRETATION_SCOPE.items()
    )
    lines.append("")
    return "\n".join(lines)


def build_population_failure_artifacts_v3(
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
    config, manifests = aggregate.load_protocol(
        config_path, manifest_receipt_path, manifest_path
    )
    if config.get("arms") != [BASELINE_ARM, AEGIS_ARM]:
        raise FailureReportV3Error("analysis-v3 arm order changed")
    if len(manifests) != expected_cases:
        raise FailureReportV3Error(
            f"manifest has {len(manifests)} cases, expected {expected_cases}"
        )
    summary = aggregate.load_json(summary_path)
    validation_receipt = aggregate.load_json(validation_receipt_path)
    v2._require_population_validation_receipt(
        validation_receipt, expected_cases=expected_cases
    )
    root = results_root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise FailureReportV3Error("population results root is unavailable")
    records: list[dict[str, Any]] = []
    for manifest in manifests:
        case_id = str(manifest["case_id"])
        ordinal = _int(
            manifest.get("case_ordinal"),
            label=f"{case_id}/case ordinal",
        )
        task_index = ordinal // int(
            config["population"]["expected_cases_per_task_level_group"]
        )
        task_root = root / f"task-{task_index}" / "results"
        baseline = v2._load_exactly_one_validated_result(
            task_root / "pi05" / case_id / "result.json",
            config=config,
            manifest=manifest,
        )
        aegis = v2._load_exactly_one_validated_result(
            task_root / "aegis" / case_id / "result.json",
            config=config,
            manifest=manifest,
        )
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
            raise FailureReportV3Error(
                f"{case_id}: strict cross-arm pairing failed: {error}"
            ) from error
        records.append(
            build_case_record_v3(
                manifest=manifest,
                baseline=baseline,
                aegis=aegis,
                baseline_artifact_root=task_root,
                aegis_artifact_root=task_root,
            )
        )
    report = build_report_v3(
        records,
        summary=summary,
        summary_sha256=v2.sha256_path(summary_path),
        validation_receipt_sha256=v2.sha256_path(
            validation_receipt_path
        ),
        source_publication_receipt_sha256=(
            source_publication_receipt_sha256
        ),
        expected_cases=expected_cases,
        streaming_stats={
            "maximum_live_full_result_records": 2,
            "compact_case_records_retained": len(records),
        },
    )
    ordered = sorted(
        records, key=lambda row: int(row["case_ordinal"])
    )
    v2._write_atomic_bytes(
        cases_output_path, v2.jsonl_bytes(ordered)
    )
    v2._write_atomic_bytes(
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
    v2._write_atomic_bytes(
        markdown_output_path,
        render_markdown_v3(report).encode("utf-8"),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build decision-aligned AEGIS analysis-v3 artifacts"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--source-publication-sha256", required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = build_population_failure_artifacts_v3(
        config_path=args.config,
        manifest_receipt_path=args.manifest_receipt,
        manifest_path=args.manifest,
        results_root=args.results_root,
        summary_path=args.summary,
        validation_receipt_path=args.validation_receipt,
        source_publication_receipt_sha256=(
            args.source_publication_sha256
        ),
        cases_output_path=args.cases_output,
        report_output_path=args.report_output,
        markdown_output_path=args.markdown_output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
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
