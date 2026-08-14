"""Contracts for the frozen L5 registered correction-mode audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_registered_mode_audit.v1"
RESULT_SCHEMA = "vlsa_distal_l5_registered_mode_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_registered_mode_audit_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    payload.pop("validation_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "protocol_id", "claim_scope", "sources",
        "population", "comparisons", "decision", "forbidden",
    }:
        raise ValueError("registered-mode audit config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-l5-registered-mode-audit-v1"
        or value["population"]["expected_unique_proxy_valid_state_count"] != 16
        or value["population"]["coarse_candidates_per_state"] != 12
        or len(value["comparisons"]["matched_requested_radius_1_modes"]) != 6
        or value["decision"]["learned_selector_authorized"] is not False
    ):
        raise ValueError("registered-mode audit protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _family(name: str) -> str:
    if name.startswith("normal_"):
        return "normal"
    if name.startswith("tangent_up_"):
        return "tangent_up"
    if name.startswith("tangent_side_"):
        return "tangent_side"
    if name.startswith("mixture_"):
        return "mixture"
    if name == "nominal":
        return "nominal"
    raise ValueError("registered mode family differs")


def audit_states(states: Sequence[Mapping[str, Any]],
                 config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    matched_names = list(config["comparisons"]["matched_requested_radius_1_modes"])
    strongest_name = str(config["comparisons"]["strongest_analytical_repulsion"])
    improvement = float(config["comparisons"]["material_improvement_m"])
    records = []
    matched_winner_counts = {name: 0 for name in matched_names}
    matched_family_counts = {name: 0 for name in ("normal", "tangent_up", "tangent_side")}
    all_winner_counts: dict[str, int] = {}
    any_safe_count = strongest_safe_count = alternative_rescue_count = 0
    matched_tangent_rescue_count = matched_alternative_win_count = 0
    clipped_count = 0
    for state in states:
        modes = {str(item["name"]): dict(item) for item in state["modes"]}
        if len(modes) != 12 or strongest_name not in modes or any(
            name not in modes for name in matched_names
        ):
            raise ValueError("registered state mode bank differs")
        non_nominal = [item for name, item in modes.items() if name != "nominal"]
        strongest = modes[strongest_name]
        valid_non_nominal = [item for item in non_nominal if not item["physical_veto"]]
        if not valid_non_nominal:
            best_all = min(non_nominal, key=lambda item: (item["risk_m"], item["order"]))
        else:
            best_all = min(valid_non_nominal, key=lambda item: (item["risk_m"], item["order"]))
        matched = [modes[name] for name in matched_names]
        valid_matched = [item for item in matched if not item["physical_veto"]]
        best_matched = min(
            valid_matched if valid_matched else matched,
            key=lambda item: (item["risk_m"], item["order"]),
        )
        normal_r1 = modes["normal_pos_front_loaded_r1.0"]
        matched_tangents = [
            modes[name] for name in matched_names
            if _family(name) in ("tangent_up", "tangent_side")
        ]
        best_tangent = min(
            [item for item in matched_tangents if not item["physical_veto"]]
            or matched_tangents,
            key=lambda item: (item["risk_m"], item["order"]),
        )
        any_safe = any(item["zero_margin_safe"] for item in non_nominal)
        strongest_safe = bool(strongest["zero_margin_safe"])
        alternatives = [
            item for item in non_nominal
            if item["name"] != strongest_name and item["zero_margin_safe"]
        ]
        alternative_rescue = bool(any_safe and not strongest_safe and alternatives)
        tangent_rescue = bool(
            best_tangent["zero_margin_safe"] and not normal_r1["zero_margin_safe"]
        )
        alternative_win = bool(
            _family(best_matched["name"]) != "normal"
            and best_matched["risk_m"] <= normal_r1["risk_m"] - improvement
        )
        any_safe_count += int(any_safe)
        strongest_safe_count += int(strongest_safe)
        alternative_rescue_count += int(alternative_rescue)
        matched_tangent_rescue_count += int(tangent_rescue)
        matched_alternative_win_count += int(alternative_win)
        clipped_count += sum(bool(item["clipped"]) for item in non_nominal)
        matched_winner_counts[best_matched["name"]] += 1
        matched_family_counts[_family(best_matched["name"])] += 1
        all_winner_counts[best_all["name"]] = all_winner_counts.get(best_all["name"], 0) + 1
        records.append({
            "state_id": state["state_id"], "case_id": state["case_id"],
            "split": state["split"], "target_row": state["target_row"],
            "any_registered_mode_zero_margin_safe": any_safe,
            "strongest_normal_zero_margin_safe": strongest_safe,
            "strongest_normal_registered_buffer_safe": bool(strongest["screen_safe"]),
            "strongest_normal_risk_m": float(strongest["risk_m"]),
            "strongest_normal_applied_correction_l2": float(strongest["applied_norm"]),
            "strongest_normal_clipped": bool(strongest["clipped"]),
            "best_matched_radius_1_mode": best_matched["name"],
            "best_matched_radius_1_family": _family(best_matched["name"]),
            "best_matched_radius_1_risk_m": float(best_matched["risk_m"]),
            "normal_pos_radius_1_risk_m": float(normal_r1["risk_m"]),
            "matched_tangent_rescues_normal_pos_radius_1": tangent_rescue,
            "matched_alternative_beats_normal_by_at_least_1mm": alternative_win,
            "best_all_registered_mode": best_all["name"],
            "best_all_registered_family": _family(best_all["name"]),
            "best_all_registered_risk_m": float(best_all["risk_m"]),
            "alternative_rescues_strongest_normal": alternative_rescue,
            "alternative_safe_modes": [item["name"] for item in alternatives],
        })
    if len({item["state_id"] for item in records}) != len(records):
        raise ValueError("registered audit state identity repeats")
    states_where_any_safe = [item for item in records if item["any_registered_mode_zero_margin_safe"]]
    analytical_sufficient = all(
        item["strongest_normal_zero_margin_safe"] for item in states_where_any_safe
    )
    winning_families = [name for name, count in matched_family_counts.items() if count]
    state_dependent_modes = bool(
        matched_tangent_rescue_count >= 1
        or (matched_alternative_win_count >= 2 and len(winning_families) >= 2)
    )
    return {
        "state_count": len(records),
        "state_count_with_any_zero_margin_safe_registered_mode": any_safe_count,
        "state_count_strongest_normal_zero_margin_safe": strongest_safe_count,
        "state_count_alternative_rescues_strongest_normal": alternative_rescue_count,
        "state_count_matched_tangent_rescues_radius1_normal": matched_tangent_rescue_count,
        "state_count_matched_alternative_beats_radius1_normal_by_1mm": matched_alternative_win_count,
        "matched_radius1_winner_counts": matched_winner_counts,
        "matched_radius1_winner_family_counts": matched_family_counts,
        "all_registered_winner_counts": dict(sorted(all_winner_counts.items())),
        "non_nominal_clipped_candidate_count": clipped_count,
        "analytical_repulsion_sufficient_on_observed_prefix_population": bool(analytical_sufficient),
        "state_dependent_prefix_modes_observed": state_dependent_modes,
        "complete_policy_value_mode_comparison_available": False,
        "learned_selector_authorized": False,
        "state_records": records,
    }
