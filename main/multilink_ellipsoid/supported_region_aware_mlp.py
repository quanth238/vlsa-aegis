"""Supported-state retraining contract for the unchanged regional MLP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .execution_margin_nn import _canonical
from .region_aware_mlp import REGION_FEATURE_NAMES


CONFIG_SCHEMA = "vlsa_distal_supported_region_aware_mlp_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_supported_region_aware_mlp_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_supported_region_aware_mlp_moka10_validation.v1"
VALIDATION_FRESH_SCHEMA = (
    "vlsa_distal_supported_region_aware_mlp_validation_fresh_moka10.v1"
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return _sha256(_canonical(payload))


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("supported region-aware config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "source_population_manifest_sha256", "test_selected_manifest_sha256",
        "validation_selected_manifest_sha256", "geometry_config_file_sha256",
        "exact_box_config_file_sha256", "split", "validation_fresh_actions",
        "partition", "features", "model", "training", "uncertainty",
        "projection", "learned_gate", "exact_verification", "decision",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("supported region-aware config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-supported-region-aware-mlp-moka10-v1"
    ):
        raise ValueError("supported region-aware protocol differs")
    source_keys = {
        "expanded_dataset_file_sha256", "expanded_dataset_payload_sha256",
        "expanded_oracle_file_sha256", "expanded_oracle_payload_sha256",
        "support_result_file_sha256", "support_result_payload_sha256",
        "support_validation_file_sha256",
        "decision_stability_result_file_sha256",
        "decision_stability_result_payload_sha256",
        "decision_stability_validation_file_sha256",
        "reference_region_aware_config_file_sha256",
        "archived_e05_file_sha256", "archived_e05_payload_sha256",
        "expected_state_count", "fit_actions_per_state",
        "off_grid_actions_per_state",
    }
    source = config["immutable_source"]
    if (
        set(source) != source_keys
        or int(source["expected_state_count"]) != 85
        or int(source["fit_actions_per_state"]) != 125
        or int(source["off_grid_actions_per_state"]) != 96
    ):
        raise ValueError("supported region-aware immutable source differs")
    if config["split"] != {
        "unit": "complete_episode_and_task_level_group",
        "expected_state_counts": {"train": 60, "validation": 10, "test": 15},
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "new_validation_case_id": "vlsa-t1-goal-ii-t0-e30",
        "new_validation_state_indexes": [60, 61, 62, 63, 64],
        "held_out_test_never_used_for_training_calibration_or_early_stopping": True,
    }:
        raise ValueError("supported region-aware grouped split differs")
    if config["validation_fresh_actions"] != {
        "distribution": "independent_uniform_inside_registered_action_box",
        "count_per_state": 96, "seed": 20260810,
        "second_action": "immutable_released_AEGIS_nominal",
        "simulator": "cloned_OSC_with_every_internal_substep",
        "purpose": "validation_only_one_sided_calibration",
    }:
        raise ValueError("supported region-aware validation actions differ")
    if int(config["features"].get("input_dimension", -1)) != len(
        REGION_FEATURE_NAMES
    ):
        raise ValueError("supported region-aware input dimension differs")
    if config["decision"] != {
        "closed_loop_in_this_gate": False,
        "if_learned_gate_passes": "preregister_receding_QP_closed_loop_E05",
        "if_supported_MLP_fails": (
            "replace_coefficient_output_with_action_conditioned_conservative_"
            "safety_value_model"
        ),
    }:
        raise ValueError("supported region-aware decision differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def require_unchanged_model_sections(
    config: Mapping[str, Any], reference: Mapping[str, Any],
) -> None:
    """Fail if any learned-method or learned-gate section changed."""

    for key in (
        "partition", "features", "model", "training", "uncertainty",
        "projection", "learned_gate",
    ):
        reference_value = reference[key]
        if key == "learned_gate":
            reference_value = dict(reference_value)
            reference_value.pop("closed_loop_authorized_only_if_every_gate_passes")
        if config[key] != reference_value:
            raise ValueError("supported region-aware section changed: %s" % key)


def fresh_payload(value: Mapping[str, Any]) -> str:
    return _hash_without(value, "validation_fresh_payload_sha256")
