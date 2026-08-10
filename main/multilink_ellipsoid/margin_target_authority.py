"""Audit ellipsoid rollout-margin labels against raw distal MuJoCo contact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import _canonical, _numpy


CONFIG_SCHEMA = "vlsa_distal_margin_target_authority_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_margin_target_authority_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_margin_target_authority_moka10_validation.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("margin-target audit config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "target_definition", "population", "gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("margin-target audit config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-margin-target-authority-moka10-v1"
    ):
        raise ValueError("margin-target audit protocol differs")
    source = config["immutable_source"]
    source_keys = {
        "expanded_dataset_file_sha256", "expanded_dataset_payload_sha256",
        "expanded_oracle_file_sha256", "expanded_oracle_payload_sha256",
        "validation_fresh_file_sha256", "validation_fresh_payload_sha256",
        "action_conditioned_result_file_sha256",
        "action_conditioned_result_payload_sha256",
        "action_conditioned_validation_file_sha256", "expected_state_count",
        "fit_actions_per_state", "expected_fit_action_count",
        "expected_fresh_state_count", "fresh_actions_per_state",
        "expected_fresh_action_count",
    }
    if (
        set(source) != source_keys
        or int(source["expected_state_count"]) != 85
        or int(source["fit_actions_per_state"]) != 125
        or int(source["expected_fit_action_count"]) != 10625
        or int(source["expected_fresh_state_count"]) != 25
        or int(source["fresh_actions_per_state"]) != 96
        or int(source["expected_fresh_action_count"]) != 2400
    ):
        raise ValueError("margin-target immutable population differs")
    if config["target_definition"] != {
        "ellipsoid_safe": "all_seven_minimum_substep_margins_nonnegative",
        "raw_distal_safe": (
            "D_sim_raw_safe_and_zero_raw_protected_L5_L6_L7_contacts"
        ),
        "horizon_actions": 2, "controller": "unchanged_OSC_POSE",
        "temporal_resolution": "every_internal_MuJoCo_substep",
        "robot_geometry": "certified_seven_slab_L5_L7_ellipsoid_union",
        "obstacle_geometry": "exact_MuJoCo_moka_pot_collision_primitives",
        "excluded_from_gate": [
            "released_AEGIS_EE_proxy", "CAR_obstacle_motion", "task_success"
        ],
    }:
        raise ValueError("margin-target definition differs")
    if config["gate"] != {
        "maximum_ellipsoid_safe_raw_distal_unsafe_count": 0,
        "require_expected_population_counts": True,
        "local_residual_calibration_authorized_only_if_gate_passes": True,
        "training_in_this_gate": False, "QP_in_this_gate": False,
        "simulation_in_this_gate": False,
        "closed_loop_E05_authorized": False,
    }:
        raise ValueError("margin-target gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _classify(
    margins: Sequence[float], raw_safe: bool, raw_contact_count: int,
) -> tuple[bool, bool, bool]:
    np = _numpy()
    values = np.asarray(margins, dtype=np.float64)
    if (
        values.shape != (7,) or not np.all(np.isfinite(values))
        or int(raw_contact_count) < 0
    ):
        raise ValueError("margin-target rollout record differs")
    ellipsoid_safe = bool(np.all(values >= 0.0))
    distal_raw_safe = bool(raw_safe and int(raw_contact_count) == 0)
    return ellipsoid_safe, distal_raw_safe, bool(
        ellipsoid_safe and not distal_raw_safe
    )


def analyze(
    dataset: Mapping[str, Any], oracle_states: Sequence[Mapping[str, Any]],
    validation_fresh: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    source = config["immutable_source"]
    states = dataset["state_records"]
    if len(states) != int(source["expected_state_count"]):
        raise ValueError("margin-target state count differs")
    case_records: dict[str, dict[str, Any]] = {}

    def case_record(case_id: str) -> dict[str, Any]:
        if case_id not in case_records:
            case_records[case_id] = {
                "case_id": case_id, "fit_action_count": 0,
                "fresh_action_count": 0, "ellipsoid_safe_count": 0,
                "raw_distal_unsafe_count": 0,
                "ellipsoid_safe_raw_distal_unsafe_count": 0,
                "dangerous_records": [],
            }
        return case_records[case_id]

    fit_count = 0
    fit_dangerous = 0
    for state in states:
        margins = np.asarray(
            state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        raw_safe = list(state["candidate_raw_safe"])
        contacts = list(state["candidate_raw_contact_count"])
        if margins.shape != (125, 7) or len(raw_safe) != 125 or len(contacts) != 125:
            raise ValueError("margin-target fit population differs")
        record = case_record(str(state["case_id"]))
        for action_index in range(125):
            ellipsoid_safe, distal_safe, dangerous = _classify(
                margins[action_index], bool(raw_safe[action_index]),
                int(contacts[action_index]),
            )
            fit_count += 1
            record["fit_action_count"] += 1
            record["ellipsoid_safe_count"] += int(ellipsoid_safe)
            record["raw_distal_unsafe_count"] += int(not distal_safe)
            if dangerous:
                fit_dangerous += 1
                record["ellipsoid_safe_raw_distal_unsafe_count"] += 1
                record["dangerous_records"].append({
                    "source": "fit_grid", "state_index": int(state["state_index"]),
                    "state_step": int(state["state_step"]),
                    "action_index": action_index,
                    "minimum_margin_m": float(np.min(margins[action_index])),
                    "raw_contact_count": int(contacts[action_index]),
                })

    combined = {
        int(item["state_index"]): dict(item) for item in oracle_states
    }
    for item in validation_fresh["state_results"]:
        combined[int(item["state_index"])]["fresh_actions"] = item["fresh_actions"]
    state_by_index = {int(item["state_index"]): item for item in states}
    fresh_states = [
        state for state in states if state["split"] in ("validation", "test")
    ]
    if len(fresh_states) != int(source["expected_fresh_state_count"]):
        raise ValueError("margin-target fresh state count differs")
    fresh_count = 0
    fresh_dangerous = 0
    for state in fresh_states:
        state_index = int(state["state_index"])
        actions = combined[state_index].get("fresh_actions", [])
        if len(actions) != int(source["fresh_actions_per_state"]):
            raise ValueError("margin-target fresh action count differs")
        record = case_record(str(state["case_id"]))
        for action in actions:
            margins = action["minimum_distal_margin_m"]
            ellipsoid_safe, distal_safe, dangerous = _classify(
                margins, bool(action["D_sim_raw_safe"]),
                int(action["raw_protected_contact_count"]),
            )
            fresh_count += 1
            record["fresh_action_count"] += 1
            record["ellipsoid_safe_count"] += int(ellipsoid_safe)
            record["raw_distal_unsafe_count"] += int(not distal_safe)
            if dangerous:
                fresh_dangerous += 1
                record["ellipsoid_safe_raw_distal_unsafe_count"] += 1
                record["dangerous_records"].append({
                    "source": "fresh_off_grid", "state_index": state_index,
                    "state_step": int(state["state_step"]),
                    "fresh_index": int(action["fresh_index"]),
                    "minimum_margin_m": float(np.min(margins)),
                    "raw_contact_count": int(
                        action["raw_protected_contact_count"]
                    ),
                })
    if (
        fit_count != int(source["expected_fit_action_count"])
        or fresh_count != int(source["expected_fresh_action_count"])
    ):
        raise ValueError("margin-target total population differs")
    dangerous_total = fit_dangerous + fresh_dangerous
    gate_pass = dangerous_total <= int(
        config["gate"]["maximum_ellipsoid_safe_raw_distal_unsafe_count"]
    )
    records = sorted(case_records.values(), key=lambda item: item["case_id"])
    return {
        "population": {
            "state_count": len(states), "fit_action_count": fit_count,
            "fresh_state_count": len(fresh_states),
            "fresh_action_count": fresh_count,
            "total_action_count": fit_count + fresh_count,
        },
        "aggregates": {
            "fit_ellipsoid_safe_raw_distal_unsafe_count": fit_dangerous,
            "fresh_ellipsoid_safe_raw_distal_unsafe_count": fresh_dangerous,
            "ellipsoid_safe_raw_distal_unsafe_count": dangerous_total,
            "case_count_with_dangerous_false_safe": sum(
                item["ellipsoid_safe_raw_distal_unsafe_count"] > 0
                for item in records
            ),
            "target_authority_gate_pass": gate_pass,
            "local_residual_calibration_preregistration_authorized": gate_pass,
            "closed_loop_E05_authorized": False,
        },
        "case_results": records,
        "decision": {
            "next_action": (
                "preregister_state_conditioned_local_residual_calibration"
                if gate_pass else
                "stop_and_repair_ellipsoid_target_geometry_before_learning"
            ),
            "training_executed": False, "QP_executed": False,
            "new_simulation_executed": False,
            "closed_loop_E05_remains_blocked": True,
        },
    }


__all__ = [
    "CONFIG_SCHEMA", "RESULT_SCHEMA", "VALIDATION_SCHEMA", "analyze",
    "load_config",
]
