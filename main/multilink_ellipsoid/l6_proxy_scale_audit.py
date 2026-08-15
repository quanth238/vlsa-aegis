"""Contracts for the empirical L6 robot-proxy scale calibration.

This is deliberately a development heuristic.  Uniformly scaling an enclosing
ellipsoid below one invalidates its mesh-enclosure certificate, so the result
must be judged against raw MuJoCo contacts and must not be presented as a
formal conservative bound.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l6_proxy_scale_audit.v1"
RESULT_SCHEMA = "vlsa_distal_l6_proxy_scale_audit_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version",
        "protocol_id",
        "claim_scope",
        "source_result",
        "source_result_file_sha256",
        "source_result_payload_sha256",
        "source_validation",
        "source_validation_file_sha256",
        "source_validation_payload_sha256",
        "calibration",
        "gate",
        "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("L6 proxy-scale config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("L6 proxy-scale config schema differs")
    if value["protocol_id"] != "vlsa-distal-l6-proxy-scale-audit-v1":
        raise ValueError("L6 proxy-scale protocol differs")
    calibration = value["calibration"]
    if calibration != {
        "candidate_scales_descending": [1.0, 0.995, 0.99, 0.985, 0.98],
        "scaled_rows": [3, 4],
        "scaled_physical_group": "L6",
        "transformation": "d_scaled=(d_original+1)/scale-1",
        "selection": "largest_scale_passing_all_opened_development_controls",
        "certificate_status": "heuristic_non_enclosing_empirical_proxy",
    }:
        raise ValueError("L6 proxy-scale calibration contract differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def rescale_row_slacks(
    row_slacks: Sequence[float], *, scale: float, scaled_rows: Sequence[int]
) -> list[float]:
    values = [float(item) for item in row_slacks]
    if len(values) != 7:
        raise ValueError("L6 proxy-scale audit requires seven source rows")
    scale = float(scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("L6 proxy scale must lie in (0, 1]")
    if list(scaled_rows) != [3, 4]:
        raise ValueError("L6 proxy-scale rows differ")
    for row in scaled_rows:
        values[int(row)] = (values[int(row)] + 1.0) / scale - 1.0
    return values


def evaluate_scale(
    cases: Sequence[Mapping[str, Any]], *, scale: float, scaled_rows: Sequence[int]
) -> dict[str, Any]:
    records = []
    for case in cases:
        for candidate in case["candidates"]:
            row_slacks = rescale_row_slacks(
                candidate["compiled_box_row_minimum_normalized_radial_slack"],
                scale=scale,
                scaled_rows=scaled_rows,
            )
            any_overlap = min(row_slacks) < 0.0
            status = candidate["source_terminal_status"]
            raw_contact = int(candidate["source_raw_protected_contact_count"]) > 0
            stable_control = status == "SAFE_TERMINAL" and not bool(
                candidate["source_physical_veto"]
            )
            records.append(
                {
                    "case_id": case["case_id"],
                    "name": candidate["name"],
                    "source_terminal_status": status,
                    "raw_contact_control": raw_contact,
                    "stable_contact_free_control": stable_control,
                    "row_minimum_normalized_radial_slack": row_slacks,
                    "minimum_normalized_radial_slack": min(row_slacks),
                    "any_proxy_overlap": any_overlap,
                    "known_safe_terminal": stable_control and not any_overlap,
                }
            )
    contact_controls = [item for item in records if item["raw_contact_control"]]
    stable_controls = [
        item for item in records if item["stable_contact_free_control"]
    ]
    timeouts = [
        item for item in records
        if item["source_terminal_status"] == "UNKNOWN_TIMEOUT"
    ]
    states_with_safe_support = {
        item["case_id"] for item in records if item["known_safe_terminal"]
    }
    gates = {
        "all_raw_contact_controls_overlap": all(
            item["any_proxy_overlap"] for item in contact_controls
        ),
        "all_stable_contact_free_controls_nonoverlap": all(
            not item["any_proxy_overlap"] for item in stable_controls
        ),
        "safe_support_in_both_states": len(states_with_safe_support) == len(cases),
        "timeouts_remain_unknown": all(
            not item["known_safe_terminal"] for item in timeouts
        ),
    }
    return {
        "scale": float(scale),
        "records": records,
        "gates": gates,
        "passes_controls": bool(all(gates.values())),
        "raw_contact_control_count": len(contact_controls),
        "stable_contact_free_control_count": len(stable_controls),
        "stable_contact_free_recall_count": sum(
            item["known_safe_terminal"] for item in stable_controls
        ),
        "states_with_safe_support": sorted(states_with_safe_support),
    }


def audit_source(source: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    calibration = config["calibration"]
    scales = [
        evaluate_scale(
            source["cases"],
            scale=float(scale),
            scaled_rows=calibration["scaled_rows"],
        )
        for scale in calibration["candidate_scales_descending"]
    ]
    passing = [item for item in scales if item["passes_controls"]]
    selected = None if not passing else max(passing, key=lambda item: item["scale"])
    baseline = scales[0]
    gate = config["gate"]
    gates = {
        "source_strict_gate_is_no_go": source["strict_gate_pass"] is False,
        "source_case_count": len(source["cases"]) == int(gate["require_case_count"]),
        "source_candidate_count": sum(
            len(case["candidates"]) for case in source["cases"]
        )
        == int(gate["require_candidate_count"]),
        "baseline_reproduces_incomplete_stable_recall": (
            baseline["stable_contact_free_recall_count"]
            == int(gate["require_baseline_stable_recall_count"])
        ),
        "selected_scale_exists": selected is not None,
        "selected_scale_within_maximum_shrink": (
            selected is not None
            and float(selected["scale"]) >= float(gate["minimum_allowed_scale"])
        ),
        "selected_detects_all_contact_controls": (
            selected is not None
            and selected["gates"]["all_raw_contact_controls_overlap"]
        ),
        "selected_accepts_all_stable_controls": (
            selected is not None
            and selected["stable_contact_free_recall_count"]
            == int(gate["require_stable_contact_free_recall_count"])
        ),
        "selected_has_safe_support_in_both_states": (
            selected is not None
            and selected["gates"]["safe_support_in_both_states"]
        ),
        "timeouts_remain_unknown": (
            selected is not None and selected["gates"]["timeouts_remain_unknown"]
        ),
    }
    return {
        "scale_evaluations": scales,
        "selected_scale": None if selected is None else float(selected["scale"]),
        "selected_scale_record": selected,
        "gates": gates,
        "strict_gate_pass": bool(all(gates.values())),
    }
