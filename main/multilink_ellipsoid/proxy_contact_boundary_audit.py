"""Audit seven ellipsoid rollout margins against protected MuJoCo contact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import _canonical, _numpy


CONFIG_SCHEMA = "vlsa_distal_proxy_contact_boundary_audit_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_proxy_contact_boundary_audit_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_proxy_contact_boundary_audit_moka10_validation.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("proxy-contact boundary config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "comparison", "population", "gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("proxy-contact boundary config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-proxy-contact-boundary-audit-moka10-v1"
    ):
        raise ValueError("proxy-contact boundary protocol differs")
    source = config["immutable_source"]
    if set(source) != {
        "expanded_dataset_file_sha256", "expanded_dataset_payload_sha256",
        "expanded_oracle_file_sha256", "expanded_oracle_payload_sha256",
        "validation_fresh_file_sha256", "validation_fresh_payload_sha256",
        "target_authority_result_file_sha256",
        "target_authority_result_payload_sha256",
        "target_authority_validation_file_sha256",
    }:
        raise ValueError("proxy-contact boundary immutable source differs")
    if config["comparison"] != {
        "D_opt": "minimum_of_seven_L5_L7_ellipsoid_rollout_margins_m",
        "D_sim": "any_registered_raw_protected_L5_L6_L7_contact",
        "proxy_safe": "D_opt_at_least_threshold",
        "false_safe": "proxy_safe_and_D_sim_contact",
        "false_unsafe": "not_proxy_safe_and_no_D_sim_contact",
        "thresholds_m": [
            -0.02, -0.01, -0.005, -0.002, -0.001, 0.0,
            0.001, 0.002, 0.005, 0.008, 0.01, 0.02,
        ],
        "absolute_proxy_boundary_bands_m": [0.001, 0.002, 0.005, 0.01, 0.02],
        "native_positive_distance_used": False,
    }:
        raise ValueError("proxy-contact boundary comparison differs")
    if config["population"] != {
        "state_count": 85, "fit_actions_per_state": 125,
        "fit_action_count": 10625, "fresh_state_count": 25,
        "fresh_actions_per_state": 96, "fresh_action_count": 2400,
        "total_action_count": 13025, "horizon_actions": 2,
        "temporal_resolution": "every_internal_cloned_OSC_substep",
        "no_state_or_action_discarded": True,
    }:
        raise ValueError("proxy-contact boundary population differs")
    if config["gate"] != {
        "maximum_zero_threshold_false_safe_count": 0,
        "minimum_observed_contact_action_count": 1,
        "minimum_five_mm_boundary_action_count": 1,
        "initial_state_audit_authorized_only_if_complete": True,
        "training_in_this_gate": False, "QP_in_this_gate": False,
        "simulation_in_this_gate": False, "closed_loop_E05_authorized": False,
    }:
        raise ValueError("proxy-contact boundary gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _confusion(records: Sequence[Mapping[str, Any]], threshold_m: float) -> dict[str, Any]:
    false_safe = false_unsafe = true_safe = true_unsafe = accepted = 0
    for record in records:
        predicted_safe = float(record["D_opt_m"]) >= float(threshold_m)
        contact = bool(record["D_sim_contact"])
        accepted += int(predicted_safe)
        false_safe += int(predicted_safe and contact)
        false_unsafe += int((not predicted_safe) and (not contact))
        true_safe += int(predicted_safe and (not contact))
        true_unsafe += int((not predicted_safe) and contact)
    no_contact = true_safe + false_unsafe
    contacts = true_unsafe + false_safe
    return {
        "threshold_m": float(threshold_m), "action_count": len(records),
        "accepted_action_count": accepted, "contact_action_count": contacts,
        "no_contact_action_count": no_contact, "false_safe_count": false_safe,
        "false_unsafe_count": false_unsafe, "true_safe_count": true_safe,
        "true_unsafe_count": true_unsafe,
        "no_contact_safe_recall": (
            float(true_safe / no_contact) if no_contact else None
        ),
        "contact_detection_recall": (
            float(true_unsafe / contacts) if contacts else None
        ),
    }


def _record(
    *, margins: Sequence[float], contact_count: int, source: str,
    split: str, case_id: str, state_index: int, state_step: int,
) -> dict[str, Any]:
    np = _numpy()
    values = np.asarray(margins, dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("proxy-contact margin row differs")
    if int(contact_count) < 0:
        raise ValueError("proxy-contact count differs")
    return {
        "source": source, "split": split, "case_id": case_id,
        "state_index": int(state_index), "state_step": int(state_step),
        "D_opt_m": float(np.min(values)),
        "D_sim_contact": bool(int(contact_count) > 0),
        "raw_protected_contact_count": int(contact_count),
    }


def paired_records(
    dataset: Mapping[str, Any], oracle_states: Sequence[Mapping[str, Any]],
    validation_fresh: Mapping[str, Any], config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    np = _numpy()
    expected = config["population"]
    states = dataset["state_records"]
    if len(states) != int(expected["state_count"]):
        raise ValueError("proxy-contact state population differs")
    records: list[dict[str, Any]] = []
    for state in states:
        margins = np.asarray(
            state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        contacts = list(state["candidate_raw_contact_count"])
        if margins.shape != (125, 7) or len(contacts) != 125:
            raise ValueError("proxy-contact fit population differs")
        for action_index in range(125):
            records.append(_record(
                margins=margins[action_index], contact_count=contacts[action_index],
                source="fit_grid", split=str(state["split"]),
                case_id=str(state["case_id"]), state_index=int(state["state_index"]),
                state_step=int(state["state_step"]),
            ))

    combined = {int(item["state_index"]): dict(item) for item in oracle_states}
    for item in validation_fresh["state_results"]:
        combined[int(item["state_index"])]["fresh_actions"] = item["fresh_actions"]
    fresh_states = [
        state for state in states if str(state["split"]) in ("validation", "test")
    ]
    if len(fresh_states) != int(expected["fresh_state_count"]):
        raise ValueError("proxy-contact fresh state population differs")
    for state in fresh_states:
        actions = combined[int(state["state_index"])].get("fresh_actions", [])
        if len(actions) != int(expected["fresh_actions_per_state"]):
            raise ValueError("proxy-contact fresh action population differs")
        for action in actions:
            records.append(_record(
                margins=action["minimum_distal_margin_m"],
                contact_count=int(action["raw_protected_contact_count"]),
                source="fresh_off_grid", split=str(state["split"]),
                case_id=str(state["case_id"]), state_index=int(state["state_index"]),
                state_step=int(state["state_step"]),
            ))
    if len(records) != int(expected["total_action_count"]):
        raise ValueError("proxy-contact total action population differs")
    return records


def analyze(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    comparison = config["comparison"]
    thresholds = [
        _confusion(records, float(value))
        for value in comparison["thresholds_m"]
    ]
    zero = next(item for item in thresholds if item["threshold_m"] == 0.0)
    boundary = []
    for width in comparison["absolute_proxy_boundary_bands_m"]:
        selected = [
            item for item in records if abs(float(item["D_opt_m"])) <= float(width)
        ]
        summary = _confusion(selected, 0.0)
        summary["absolute_band_m"] = float(width)
        boundary.append(summary)
    by_split = {
        split: _confusion(
            [item for item in records if item["split"] == split], 0.0
        )
        for split in ("train", "validation", "test")
    }
    gate = config["gate"]
    five_mm = next(
        item for item in boundary if item["absolute_band_m"] == 0.005
    )
    complete = bool(
        int(zero["false_safe_count"])
        <= int(gate["maximum_zero_threshold_false_safe_count"])
        and int(zero["contact_action_count"])
        >= int(gate["minimum_observed_contact_action_count"])
        and int(five_mm["action_count"])
        >= int(gate["minimum_five_mm_boundary_action_count"])
    )
    return {
        "population": {
            "total_action_count": len(records),
            "fit_action_count": sum(item["source"] == "fit_grid" for item in records),
            "fresh_action_count": sum(
                item["source"] == "fresh_off_grid" for item in records
            ),
        },
        "zero_threshold_confusion": zero,
        "threshold_sweep": thresholds,
        "absolute_proxy_boundary_bands": boundary,
        "split_zero_threshold_confusion": by_split,
        "decision": {
            "audit_complete": complete,
            "proxy_has_observed_false_safe": bool(zero["false_safe_count"]),
            "proxy_has_observed_false_unsafe": bool(zero["false_unsafe_count"]),
            "initial_state_audit_authorized": complete,
            "training_executed": False, "QP_executed": False,
            "new_simulation_executed": False,
            "closed_loop_E05_remains_blocked": True,
        },
    }


__all__ = [
    "CONFIG_SCHEMA", "RESULT_SCHEMA", "VALIDATION_SCHEMA", "analyze",
    "load_config", "paired_records",
]
