"""Preregistered rounded-box early-trigger heuristic for primary E05."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


INFLATED_TRIGGER_SCHEMA = "vlsa_distal_exact_box_inflated_trigger_e05.v1"
INFLATED_TRIGGER_RESULT_SCHEMA = (
    "vlsa_distal_exact_box_inflated_trigger_e05_result.v1"
)
_CASE_ID = "vlsa-t1-goal-ii-t0-e05"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_inflated_trigger_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("inflated-trigger config is invalid JSON") from error
    required = {
        "schema_version",
        "protocol_id",
        "case_ids",
        "claim_scope",
        "base_audit_config",
        "exact_box_config",
        "prior_results",
        "rounded_box_shell",
        "trigger_scan",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("inflated-trigger config keys differ")
    if config["schema_version"] != INFLATED_TRIGGER_SCHEMA:
        raise ValueError("inflated-trigger schema differs")
    if config["protocol_id"] != "vlsa-distal-exact-box-inflated-trigger-e05-v1":
        raise ValueError("inflated-trigger protocol differs")
    if config["case_ids"] != [_CASE_ID]:
        raise ValueError("inflated-trigger audit must select only primary E05")
    if config["base_audit_config"] != {
        "config_file_sha256": (
            "c7019c176e0e8b6379cdb1b83e09d7129c1237a4f28ec2f3daba49a7b57ddc95"
        ),
        "config_payload_sha256": (
            "dc0a9d52fe75da297a187c36a0a6c9f069948511afa940c9268ff43ed392c98d"
        ),
        "schema_version": "vlsa_distal_oracle_affine_e05.v1",
    }:
        raise ValueError("inflated-trigger base audit identity differs")
    if config["exact_box_config"] != {
        "config_file_sha256": (
            "cc568a85c2acf215beda1cef4abcc31a92b3f6d3772d6c36147410afd471bf9f"
        ),
        "config_payload_sha256": (
            "d3e0ab883eb3b3de160417fc9547012db9154dae7015ff80bc739219b728013b"
        ),
        "schema_version": "vlsa_distal_oracle_exact_box_obstacle_e05.v1",
    }:
        raise ValueError("inflated-trigger exact-box config identity differs")
    if config["prior_results"] != {
        "fixed_margin": {
            "file_sha256": (
                "8493d3cdd81ffb4b09cfe2c71af23e3f63d489bb34acbb7603877a011c1ca170"
            ),
            "result_payload_sha256": (
                "3088c8222add6a2d6fe3964edd39f84293f1dd9e25958897d1bca6cfd4832764"
            ),
            "schema_version": (
                "vlsa_distal_oracle_affine_margin8mm_e05_result.v1"
            ),
            "slurm_job_id": "37175",
        },
        "exact_box": {
            "file_sha256": (
                "d57850d8d8b04e3601d0cb870cca84f15bc41684eae71dc2cb57c9414c1d10fb"
            ),
            "result_payload_sha256": (
                "30d48753b457d1b52f935a9d5a5d4912f9b2aad651751f76b1e9edf6ab442f11"
            ),
            "schema_version": (
                "vlsa_distal_oracle_exact_box_obstacle_e05_result.v1"
            ),
            "slurm_job_id": "37180",
        },
    }:
        raise ValueError("inflated-trigger prior-result identities differ")
    if config["rounded_box_shell"] != {
        "applies_to": "all_15_exact_live_mujoco_obstacle_boxes",
        "clearance_equivalence": "h_inflated_equals_h_exact_minus_shell_m",
        "half_axis_inflation": False,
        "minkowski_shape": "sphere",
        "shell_m": 0.008,
        "support_radius": "exact_oriented_box_support_plus_shell_m",
    }:
        raise ValueError("inflated-trigger rounded-box shell differs")
    if config["trigger_scan"] != {
        "action_source": "immutable_job_37109_executed_sitl_action_ledger",
        "end_step_inclusive": 192,
        "nominal_crossing": (
            "any_exact_minimum_substep_clearance_strictly_below_shell_m"
        ),
        "prefix_execution": "exact_immutable_actions_before_selected_trigger",
        "selection": "first_step_satisfying_current_safe_and_nominal_crossing",
        "start_step": 0,
        "current_safe": "all_exact_interval_start_clearances_at_least_shell_m",
    }:
        raise ValueError("inflated-trigger scan contract differs")
    if config["decision_gate"] != {
        "affine_candidate_false_safe_count": 0,
        "exact_qp_transition": (
            "zero_raw_L5_L6_L7_contact_and_obstacle_motion_at_most_0.1mm_"
            "and_all_exact_minimum_substep_clearances_at_least_shell_m"
        ),
        "local_controllability": (
            "at_least_one_raw_safe_and_inflated_proxy_safe_candidate"
        ),
        "trigger_required": True,
    }:
        raise ValueError("inflated-trigger decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
    return output


def is_first_crossing_candidate(
    current_clearance_m: Sequence[float],
    nominal_minimum_substep_clearance_m: Sequence[float],
    shell_m: float,
) -> bool:
    """Return the frozen early-trigger predicate."""

    current = [float(value) for value in current_clearance_m]
    nominal = [float(value) for value in nominal_minimum_substep_clearance_m]
    shell = float(shell_m)
    if len(current) != 8 or len(nominal) != 8:
        raise ValueError("inflated-trigger clearance dimensions differ")
    if not math.isfinite(shell) or shell <= 0.0:
        raise ValueError("inflated-trigger shell must be finite and positive")
    if not all(math.isfinite(value) for value in current + nominal):
        raise ValueError("inflated-trigger clearances are nonfinite")
    return all(value >= shell for value in current) and any(
        value < shell for value in nominal
    )


def intervention_pass(audit: Mapping[str, Any]) -> bool:
    """Return the preregistered early-trigger QP efficacy decision."""

    decision = audit.get("decision", {})
    return bool(
        decision.get("local_jointly_raw_and_proxy_safe_candidate_exists")
        and decision.get("affine_candidate_gate_pass")
        and decision.get("qp_valid")
        and decision.get("qp_exact_raw_safe")
        and decision.get("qp_exact_proxy_safe")
    )
