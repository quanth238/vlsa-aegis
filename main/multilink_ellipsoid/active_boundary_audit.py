"""Pure row-identifiability metrics over a finite searched action set."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.grouped_query_action_risk_freeze import candidate_outcome


CONFIG_SCHEMA = "vlsa_distal_active_boundary_audit.v1"
RESULT_SCHEMA = "vlsa_distal_active_boundary_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_active_boundary_audit_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "finite_search", "row_questions", "classification", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("active-boundary config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("active-boundary config schema differs")
    if value["protocol_id"] != "vlsa-distal-active-boundary-audit-v1":
        raise ValueError("active-boundary protocol differs")
    if value["source"]["primary_splits"] != ["train", "validation"]:
        raise ValueError("active-boundary primary splits differ")
    if int(value["source"]["finite_candidate_count_per_state"]) != 37:
        raise ValueError("active-boundary candidate count differs")
    search = value["finite_search"]
    if float(search["robust_boundary_margin_m"]) != 0.001:
        raise ValueError("active-boundary robust margin differs")
    if float(search["minimum_meaningful_variation_m"]) != 0.002:
        raise ValueError("active-boundary variation threshold differs")
    if not value["classification"]["unknown_timeouts_excluded_from_extrema"]:
        raise ValueError("active-boundary timeout handling differs")
    if not value["classification"]["active_witness_not_required_for_per_row_information"]:
        raise ValueError("active-boundary active-witness rule differs")
    if not value["classification"]["mathematical_redundancy_claim_forbidden"]:
        raise ValueError("active-boundary redundancy wording differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def audit_state_row(
    candidates: Sequence[Mapping[str, Any]], row: int, *,
    robust_margin_m: float, minimum_variation_m: float,
) -> dict[str, Any]:
    """Audit one row without treating finite search as a global proof."""
    if not 0 <= int(row) < 7:
        raise ValueError("active-boundary row differs")
    if robust_margin_m <= 0.0 or minimum_variation_m <= 0.0:
        raise ValueError("active-boundary thresholds differ")
    known = []
    unknown_count = 0
    for candidate in candidates:
        outcome = candidate_outcome(dict(candidate))
        if outcome == "unknown":
            unknown_count += 1
            continue
        risk = [float(item) for item in candidate["combined_risk"]]
        if len(risk) != 7 or any(not math.isfinite(item) for item in risk):
            raise ValueError("active-boundary risk differs")
        known.append((candidate, outcome, risk))
    if not known:
        return {
            "known_candidate_count": 0,
            "unknown_timeout_candidate_count": unknown_count,
            "witnessed_minimum_Q_m": None,
            "witnessed_maximum_Q_m": None,
            "witnessed_variation_m": None,
            "meaningful_variation": False,
            "robust_negative_side_observed": False,
            "robust_positive_side_observed": False,
            "robust_crossing_observed": False,
            "globally_safe_robust_negative_side_observed": False,
            "useful_boundary_observed": False,
            "independent_violation_observed": False,
            "proxy_invalid_candidate_count": 0,
            "minimum_candidate_name": None,
            "maximum_candidate_name": None,
            "global_safe_candidate_name": None,
            "independent_violation_candidate_name": None,
        }
    values = [item[2][int(row)] for item in known]
    minimum_index = min(range(len(known)), key=lambda index: values[index])
    maximum_index = max(range(len(known)), key=lambda index: values[index])
    negative = [item for item in known if item[2][int(row)] <= -robust_margin_m]
    positive = [item for item in known if item[2][int(row)] >= robust_margin_m]
    global_safe = [
        item for item in negative
        if item[1] == "safe"
        and bool(item[0]["exact_safe"])
        and not bool(item[0]["physical_veto"])
        and max(item[2]) <= 0.0
    ]
    independent = [
        item for item in positive
        if all(value <= 0.0 for index, value in enumerate(item[2]) if index != int(row))
    ]
    proxy_invalid = [
        item for item in known
        if max(item[2]) <= 0.0 and bool(item[0]["physical_veto"])
    ]
    variation = max(values) - min(values)
    return {
        "known_candidate_count": len(known),
        "unknown_timeout_candidate_count": unknown_count,
        "witnessed_minimum_Q_m": min(values),
        "witnessed_maximum_Q_m": max(values),
        "witnessed_variation_m": variation,
        "meaningful_variation": variation > minimum_variation_m,
        "robust_negative_side_observed": bool(negative),
        "robust_positive_side_observed": bool(positive),
        "robust_crossing_observed": bool(negative and positive),
        "globally_safe_robust_negative_side_observed": bool(global_safe),
        "useful_boundary_observed": bool(global_safe and positive),
        "independent_violation_observed": bool(independent),
        "proxy_invalid_candidate_count": len(proxy_invalid),
        "minimum_candidate_name": str(known[minimum_index][0]["name"]),
        "maximum_candidate_name": str(known[maximum_index][0]["name"]),
        "global_safe_candidate_name": (
            str(global_safe[0][0]["name"]) if global_safe else None
        ),
        "independent_violation_candidate_name": (
            str(independent[0][0]["name"]) if independent else None
        ),
    }


def row_interpretation(records: Sequence[Mapping[str, Any]]) -> str:
    if any(bool(item["proxy_invalid_candidate_count"]) for item in records):
        return "PROXY_INVALID_IN_AUDITED_SET"
    if any(bool(item["useful_boundary_observed"]) for item in records):
        return "USEFUL_BOUNDARY_WITNESSED_IN_FINITE_SET"
    if any(bool(item["robust_crossing_observed"]) for item in records):
        return "ROW_CROSSING_WITHOUT_GLOBAL_SAFE_SIDE_WITNESSED"
    if any(bool(item["meaningful_variation"]) for item in records):
        return "MEANINGFUL_VARIATION_BUT_NO_ROBUST_CROSSING_WITNESSED"
    return "NO_MEANINGFUL_VARIATION_WITNESSED_IN_FINITE_SET"
