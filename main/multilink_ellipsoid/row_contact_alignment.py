"""Read-only alignment of ellipsoid rows with raw MuJoCo contact samples."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_row_contact_alignment.v1"
AUDIT_SCHEMA = "vlsa_distal_row_contact_alignment_audit.v1"
VALIDATION_SCHEMA = "vlsa_distal_row_contact_alignment_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source_populations",
        "geometry_contract", "trace_contract", "decision_contract", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("row-contact config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("row-contact config schema differs")
    populations = value["source_populations"]
    if not isinstance(populations, list) or len(populations) != 2:
        raise ValueError("row-contact populations differ")
    if [item["expected_state_count"] for item in populations] != [15, 5]:
        raise ValueError("row-contact population sizes differ")
    geometry = value["geometry_contract"]
    if geometry["row_groups"] != {
        "L5": [0, 1, 2], "L6": [3, 4], "L7": [5, 6]
    }:
        raise ValueError("row-contact row groups differ")
    if geometry["protected_geom_to_group"] != {
        "robot0_link5_collision": "L5",
        "robot0_link6_collision": "L6",
        "robot0_link7_collision": "L7",
    }:
        raise ValueError("row-contact geom mapping differs")
    for key in (
        "physical_contact_distance_threshold_m", "proxy_overlap_threshold_m",
        "certification_buffer_m", "near_boundary_absolute_clearance_m",
    ):
        number = float(geometry[key])
        if not math.isfinite(number):
            raise ValueError("row-contact threshold is nonfinite")
    if float(geometry["certification_buffer_m"]) != 0.001:
        raise ValueError("row-contact certification buffer differs")
    trace = value["trace_contract"]
    if not all(isinstance(item, bool) for item in trace.values()):
        raise ValueError("row-contact trace flags differ")
    forbidden = value["forbidden"]
    if not forbidden or not all(item is True for item in forbidden.values()):
        raise ValueError("row-contact forbidden set differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _action_offset(index: int, substep_counts: Sequence[int]) -> tuple[int, int]:
    """Map a future trace index (initial sample excluded) to action/substep."""

    remaining = int(index)
    for action_offset, count in enumerate(substep_counts):
        count = int(count)
        if count <= 0:
            raise ValueError("substep count is nonpositive")
        if remaining < count:
            return action_offset, remaining
        remaining -= count
    raise ValueError("trace index exceeds substep contract")


def segment_samples(
    segment: Mapping[str, Any], *, start_step: int, row_count: int = 7,
) -> list[dict[str, Any]]:
    """Return future samples with raw contacts aligned to the same substep."""

    trace = segment["clearance_trace_m"]
    counts = [int(item) for item in segment["substep_counts"]]
    if len(trace) != 1 + sum(counts):
        raise ValueError("clearance trace length differs")
    events_by_identity: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for raw in segment.get("protected_contacts", []):
        event = dict(raw)
        step = int(event["step"])
        substep = int(event["substep"])
        offset = step - int(start_step)
        if not 0 <= offset < len(counts):
            raise ValueError("contact identity is outside trace")
        # The instrumentation measures each action boundary twice: once as the
        # preceding action's final MuJoCo substep and once as substep=-1 before
        # the next command.  The combined trace retains only the former.
        if substep == -1 and offset > 0:
            offset -= 1
            substep = counts[offset] - 1
        elif not 0 <= substep < counts[offset]:
            raise ValueError("contact identity is outside trace")
        events_by_identity.setdefault((offset, substep), []).append(event)
    output = []
    for flat_index, row in enumerate(trace[1:]):
        values = [float(item) for item in row[:row_count]]
        if len(values) != row_count or not all(math.isfinite(item) for item in values):
            raise ValueError("clearance sample differs")
        offset, substep = _action_offset(flat_index, counts)
        output.append({
            "step": int(start_step) + offset,
            "action_offset": offset,
            "substep": substep,
            "clearance_m": values,
            "contacts": events_by_identity.get((offset, substep), []),
        })
    observed_events = sum(len(item["contacts"]) for item in output)
    if observed_events != sum(len(items) for items in events_by_identity.values()):
        raise ValueError("contact event alignment differs")
    return output


def contact_group(event: Mapping[str, Any], mapping: Mapping[str, str]) -> str:
    name = str(event.get("protected_geom_name", ""))
    return str(mapping.get(name, "UNMAPPED"))


def _empty_group(group: str, rows: Sequence[int]) -> dict[str, Any]:
    return {
        "group": group,
        "rows": [int(item) for item in rows],
        "observed_sample_count": 0,
        "raw_contact_event_count": 0,
        "raw_contact_sample_count": 0,
        "proxy_overlap_sample_count": 0,
        "proxy_overlap_without_contact_sample_count": 0,
        "proxy_false_safe_contact_sample_count": 0,
        "buffer_false_safe_contact_sample_count": 0,
        "active_contact_witness_count_by_row": [0] * len(rows),
        "contact_overlap_count_by_row": [0] * len(rows),
        "contact_within_buffer_count_by_row": [0] * len(rows),
        "contact_clearance_by_row_m": [[] for _ in rows],
        "contact_state_ids": [],
        "unmapped_contact_event_count": 0,
    }


def summarize_samples(
    records: Iterable[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    geometry = config["geometry_contract"]
    row_groups = geometry["row_groups"]
    mapping = geometry["protected_geom_to_group"]
    overlap = float(geometry["proxy_overlap_threshold_m"])
    buffer_m = float(geometry["certification_buffer_m"])
    by_group = {name: _empty_group(name, rows) for name, rows in row_groups.items()}
    unmapped_events = []
    total_samples = 0
    for record in records:
        state_id = str(record["state_id"])
        for sample in record["samples"]:
            total_samples += 1
            clearances = [float(item) for item in sample["clearance_m"]]
            grouped_events: dict[str, list[Mapping[str, Any]]] = {
                name: [] for name in row_groups
            }
            for event in sample["contacts"]:
                group = contact_group(event, mapping)
                if group == "UNMAPPED":
                    unmapped_events.append({
                        "state_id": state_id,
                        "protected_geom_name": event.get("protected_geom_name"),
                    })
                    continue
                grouped_events[group].append(event)
            for group, rows in row_groups.items():
                summary = by_group[group]
                summary["observed_sample_count"] += 1
                values = [clearances[int(row)] for row in rows]
                has_overlap = min(values) <= overlap
                contacts = grouped_events[group]
                has_contact = bool(contacts)
                summary["raw_contact_event_count"] += len(contacts)
                summary["proxy_overlap_sample_count"] += int(has_overlap)
                summary["proxy_overlap_without_contact_sample_count"] += int(
                    has_overlap and not has_contact
                )
                if not has_contact:
                    continue
                summary["raw_contact_sample_count"] += 1
                summary["contact_state_ids"].append(state_id)
                summary["proxy_false_safe_contact_sample_count"] += int(not has_overlap)
                summary["buffer_false_safe_contact_sample_count"] += int(
                    min(values) > buffer_m
                )
                active = min(range(len(values)), key=lambda index: values[index])
                summary["active_contact_witness_count_by_row"][active] += 1
                for local, value in enumerate(values):
                    summary["contact_overlap_count_by_row"][local] += int(value <= overlap)
                    summary["contact_within_buffer_count_by_row"][local] += int(
                        value <= buffer_m
                    )
                    summary["contact_clearance_by_row_m"][local].append(value)
    for summary in by_group.values():
        summary["contact_state_ids"] = sorted(set(summary["contact_state_ids"]))
        distributions = []
        for values in summary.pop("contact_clearance_by_row_m"):
            ordered = sorted(values)
            distributions.append({
                "count": len(ordered),
                "minimum_m": None if not ordered else ordered[0],
                "median_m": None if not ordered else ordered[len(ordered) // 2],
                "maximum_m": None if not ordered else ordered[-1],
            })
        summary["contact_clearance_distribution_by_row"] = distributions
    proxy_false_safe = sum(
        item["proxy_false_safe_contact_sample_count"] for item in by_group.values()
    )
    physical_witness_rows = {
        group: [
            int(rows[index])
            for index, count in enumerate(
                by_group[group]["active_contact_witness_count_by_row"]
            ) if int(count) > 0
        ] for group, rows in row_groups.items()
    }
    return {
        "observed_future_sample_count": total_samples,
        "groups": [by_group[name] for name in ("L5", "L6", "L7")],
        "unmapped_contact_events": unmapped_events,
        "proxy_false_safe_contact_sample_count": proxy_false_safe,
        "geometry_revision_required": bool(proxy_false_safe or unmapped_events),
        "poisson_authorized": False,
        "physical_active_contact_witness_rows": physical_witness_rows,
    }


def decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    l5_rows = summary["physical_active_contact_witness_rows"]["L5"]
    if summary["geometry_revision_required"]:
        action = "audit_and_refit_proxy_geometry_before_learning"
        root = "proxy_contact_alignment_failure"
    elif any(row in (1, 2) for row in l5_rows):
        action = "collect_same_mechanism_grouped_states_for_physically_witnessed_rows"
        root = "population_coverage_not_ellipsoid_geometry"
    else:
        action = "learn_only_physically_supported_rows_and_monitor_others_exactly"
        root = "deployment_population_lacks_independent_distal_row_witnesses"
    return {
        "primary_root_cause": root,
        "next_action": action,
        "ellipsoid_geometry_change_authorized": bool(
            summary["geometry_revision_required"]
        ),
        "Poisson_change_authorized": False,
        "model_training_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
