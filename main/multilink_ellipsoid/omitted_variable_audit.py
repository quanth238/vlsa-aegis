"""Controlled omitted-variable audit for the historical 56D margin input."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_conditioned_margin import action_feature_matrix
from .complete_osc_margin import _canonical, _numpy
from .two_step_margin import feature_context, feature_vectors


CONFIG_SCHEMA = "vlsa_distal_omitted_variable_audit_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_omitted_variable_audit_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_omitted_variable_audit_moka10_validation.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("omitted-variable config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "split", "determinism_test", "candidate_selection", "interventions",
        "old_input_gate", "complete_model_gate", "forbidden_actions",
        "decision",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("omitted-variable config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-omitted-variable-audit-moka10-v1"
    ):
        raise ValueError("omitted-variable protocol differs")
    source = config["immutable_source"]
    if set(source) != {
        "dataset_file_sha256", "dataset_payload_sha256",
        "collection_result_file_sha256", "collection_result_payload_sha256",
        "matched_result_file_sha256", "matched_result_payload_sha256",
        "matched_validation_file_sha256", "matched_validation_payload_sha256",
        "complete_model_file_sha256", "population_manifest_sha256",
        "selected_manifest_sha256", "same_task_manifest_sha256",
        "targeted_manifest_sha256", "geometry_config_sha256",
        "exact_box_config_sha256", "expected_state_count",
        "expected_candidate_pair_count", "expected_duplicate_replay_count",
    }:
        raise ValueError("omitted-variable immutable source differs")
    if (
        int(source["expected_state_count"]) != 85
        or int(source["expected_candidate_pair_count"]) != 10625
        or int(source["expected_duplicate_replay_count"]) != 21250
    ):
        raise ValueError("omitted-variable population differs")
    if config["split"] != {
        "unit": "complete_episode",
        "expected_state_counts": {"train": 60, "validation": 10, "test": 15},
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "test_only": True,
    }:
        raise ValueError("omitted-variable split differs")
    if config["determinism_test"] != {
        "source": "immutable_duplicate_complete_snapshot_action_replays",
        "margin_match": "bitwise_float64_equal",
        "contact_match": "canonical_receipt_equal",
        "next_state_match": "both_action_hashes_equal",
        "required_mismatch_count": 0,
    }:
        raise ValueError("omitted-variable determinism test differs")
    if config["candidate_selection"] != {
        "population": "all_85_states_no_drop",
        "per_state": ["closest_safe", "closest_unsafe"],
        "deduplicate_candidate_indexes": True,
        "fallback_when_class_absent": "closest_absolute_margin",
    }:
        raise ValueError("omitted-variable candidate selection differs")
    interventions = config["interventions"]
    if interventions != {
        "baseline": True,
        "goal_orientation": {
            "axes": [0, 1, 2], "angles_degrees": [-15.0, 15.0],
            "changes_only": "controller.goal_ori",
        },
        "rotation_action": {
            "first_action_axes": [0, 1, 2], "absolute_values": [-1.0, 1.0],
            "changes_only": "first_action_rotation_component",
        },
        "gripper_command": {
            "two_action_values": [-1.0, 1.0],
            "changes_only": "both_gripper_command_scalars",
        },
        "controller_memory": {
            "variants": [
                "toggle_new_update", "zero_relative_ori",
                "current_orientation_as_ori_ref", "zero_previous_torques",
            ],
            "changes_only": "named_controller_memory_field",
        },
        "two_action_horizon": True,
        "old56_recomputed_after_every_intervention": True,
    }:
        raise ValueError("omitted-variable interventions differ")
    if config["old_input_gate"] != {
        "all_effective_interventions_preserve_old56_bytes": True,
        "insufficient_if_identical_hash_has_safe_and_unsafe_proxy_margins": True,
        "proxy_safe_definition": "all_seven_rollout_minimum_margins_nonnegative",
        "report_raw_contact_conflicts_separately": True,
    }:
        raise ValueError("omitted-variable old-input gate differs")
    if config["complete_model_gate"] != {
        "evaluation_source": "immutable_matched_ablation_test_metrics",
        "required_test_false_safe_action_count": 0,
        "required_test_state_safe_support_count": 15,
    }:
        raise ValueError("omitted-variable complete-model gate differs")
    if config["forbidden_actions"] != {
        "QP": True, "calibration": True, "closed_loop_E05": True,
        "model_retraining": True, "new_policy_inference": True,
    }:
        raise ValueError("omitted-variable forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def select_candidate_indexes(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select the closest safe and unsafe actions without dropping a state."""

    np = _numpy()
    candidates = state["candidates"]
    values = np.asarray([
        np.min(np.asarray(item["rollout_minimum_ellipsoid_margin_m"], dtype=np.float64))
        for item in candidates
    ], dtype=np.float64)
    if values.shape != (125,) or not np.all(np.isfinite(values)):
        raise ValueError("omitted-variable candidate margins differ")
    selected: list[tuple[int, str]] = []
    safe = np.flatnonzero(values >= 0.0)
    unsafe = np.flatnonzero(values < 0.0)
    if len(safe):
        selected.append((int(safe[np.argmin(values[safe])]), "closest_safe"))
    if len(unsafe):
        selected.append((int(unsafe[np.argmax(values[unsafe])]), "closest_unsafe"))
    if not safe.size or not unsafe.size:
        selected.append((int(np.argmin(np.abs(values))), "closest_absolute_margin"))
    output = []
    seen = set()
    for index, source in selected:
        if index not in seen:
            seen.add(index)
            output.append({
                "candidate_index": int(index), "selection_source": source,
                "source_worst_margin_m": float(values[index]),
            })
    if not output or len(output) > 2:
        raise ValueError("omitted-variable selected candidate count differs")
    return output


def intervention_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = [{"family": "baseline", "name": "baseline"}]
    settings = config["interventions"]
    for axis in settings["goal_orientation"]["axes"]:
        for angle in settings["goal_orientation"]["angles_degrees"]:
            output.append({
                "family": "goal_orientation",
                "name": "goal_axis_%d_%+gdeg" % (axis, angle),
                "axis": int(axis), "angle_degrees": float(angle),
            })
    for axis in settings["rotation_action"]["first_action_axes"]:
        for value in settings["rotation_action"]["absolute_values"]:
            output.append({
                "family": "rotation_action",
                "name": "first_rotation_%d_%+g" % (axis, value),
                "axis": int(axis), "value": float(value),
            })
    for value in settings["gripper_command"]["two_action_values"]:
        output.append({
            "family": "gripper_command",
            "name": "both_grippers_%+g" % value, "value": float(value),
        })
    for name in settings["controller_memory"]["variants"]:
        output.append({"family": "controller_memory", "name": str(name)})
    if len(output) != 19:
        raise ValueError("omitted-variable intervention count differs")
    return output


def axis_rotation(axis: int, angle_degrees: float) -> Any:
    np = _numpy()
    angle = math.radians(float(angle_degrees))
    c, s = math.cos(angle), math.sin(angle)
    output = np.eye(3, dtype=np.float64)
    first, second = ((1, 2), (0, 2), (0, 1))[int(axis)]
    if int(axis) == 0:
        output[1, 1], output[1, 2], output[2, 1], output[2, 2] = c, -s, s, c
    elif int(axis) == 1:
        output[0, 0], output[0, 2], output[2, 0], output[2, 2] = c, s, -s, c
    elif int(axis) == 2:
        output[0, 0], output[0, 1], output[1, 0], output[1, 1] = c, -s, s, c
    else:
        raise ValueError("omitted-variable rotation axis differs")
    del first, second
    return output


def old56_rows(
    env: Any, probe: Any, nominal_first_xyz: Sequence[float],
    candidate_xyz: Sequence[float], nominal_second_xyz: Sequence[float],
    action_lower: Sequence[float], action_upper: Sequence[float],
) -> Any:
    context = feature_context(env, probe)
    _, pair = feature_vectors(
        context, nominal_first_xyz, candidate_xyz, nominal_second_xyz
    )
    return action_feature_matrix(pair, candidate_xyz, action_lower, action_upper)


def array_sha256(value: Any) -> str:
    np = _numpy()
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return _sha256(array.tobytes())


def analyze_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Find safe/unsafe conflicts among exactly identical old inputs."""

    np = _numpy()
    if not records:
        raise ValueError("omitted-variable record population is empty")
    effective = [item for item in records if item["effective_intervention"]]
    all_old_equal = all(item["old56_hash_equal_to_baseline"] for item in effective)
    by_key: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for item in records:
        if item["rollout_executed"]:
            key = (
                int(item["state_index"]), int(item["candidate_index"]),
                str(item["old56_sha256"]),
            )
            by_key.setdefault(key, []).append(item)
    conflict_keys = []
    raw_contact_conflicts = []
    for key, items in by_key.items():
        proxy_values = {bool(item["proxy_safe"]) for item in items}
        contact_values = {bool(item["raw_contact_free"]) for item in items}
        if len(proxy_values) > 1:
            conflict_keys.append({
                "state_index": key[0], "candidate_index": key[1],
                "old56_sha256": key[2],
                "safe_variant_names": sorted(
                    item["variant_name"] for item in items if item["proxy_safe"]
                ),
                "unsafe_variant_names": sorted(
                    item["variant_name"] for item in items if not item["proxy_safe"]
                ),
            })
        if len(contact_values) > 1:
            raw_contact_conflicts.append({
                "state_index": key[0], "candidate_index": key[1],
                "old56_sha256": key[2],
            })
    families = {}
    baseline = {
        (int(item["state_index"]), int(item["candidate_index"])): item
        for item in records if item["variant_family"] == "baseline"
    }
    for family in (
        "goal_orientation", "rotation_action", "gripper_command",
        "controller_memory",
    ):
        items = [
            item for item in records
            if item["variant_family"] == family and item["rollout_executed"]
        ]
        changed = 0
        crossed = set()
        contact_crossed = set()
        next_changed = 0
        maximum_difference = 0.0
        for item in items:
            key = (int(item["state_index"]), int(item["candidate_index"]))
            reference = baseline[key]
            difference = float(np.max(np.abs(
                np.asarray(item["rollout_minimum_ellipsoid_margin_m"], dtype=np.float64)
                - np.asarray(reference["rollout_minimum_ellipsoid_margin_m"], dtype=np.float64)
            )))
            maximum_difference = max(maximum_difference, difference)
            changed += int(difference > 0.0)
            crossed.add(key) if bool(item["proxy_safe"]) != bool(reference["proxy_safe"]) else None
            contact_crossed.add(key) if bool(item["raw_contact_free"]) != bool(reference["raw_contact_free"]) else None
            next_changed += int(
                item["next_state_sha256_per_action"]
                != reference["next_state_sha256_per_action"]
            )
        families[family] = {
            "executed_variant_count": len(items),
            "margin_changed_variant_count": int(changed),
            "baseline_proxy_class_crossing_candidate_count": len(crossed),
            "baseline_raw_contact_class_crossing_candidate_count": len(contact_crossed),
            "next_state_changed_variant_count": int(next_changed),
            "maximum_margin_absolute_difference_m": maximum_difference,
        }
    return {
        "record_count": len(records), "effective_record_count": len(effective),
        "all_effective_interventions_preserve_old56_bytes": all_old_equal,
        "identical_old56_proxy_class_conflict_count": len(conflict_keys),
        "identical_old56_raw_contact_class_conflict_count": len(raw_contact_conflicts),
        "old56_provably_insufficient": bool(all_old_equal and conflict_keys),
        "conflicts": conflict_keys, "raw_contact_conflicts": raw_contact_conflicts,
        "family_results": families,
    }


def complete_model_gate(matched_result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = matched_result["arms"]["completeOSC"]["metrics"]["test"]
    false_safe = int(metrics["proxy_false_safe_action_count"])
    support = int(metrics["state_safe_support_count"])
    return {
        "test_false_safe_action_count": false_safe,
        "test_state_safe_support_count": support,
        "zero_false_safe_pass": bool(false_safe == 0),
        "all_15_state_support_pass": bool(support == 15),
        "complete_model_prediction_gate_pass": bool(false_safe == 0 and support == 15),
    }


def final_decision(audit: Mapping[str, Any], complete: Mapping[str, Any]) -> dict[str, Any]:
    insufficient = bool(audit["old56_provably_insufficient"])
    complete_pass = bool(complete["complete_model_prediction_gate_pass"])
    if insufficient and complete_pass:
        conclusion = "omitted_inputs_were_causal_and_complete_model_passes"
    elif insufficient and not complete_pass:
        conclusion = "old56_insufficient_but_complete_plain_MLP_still_fails"
    elif not insufficient and complete_pass:
        conclusion = "no_observed_old56_conflict_complete_model_passes"
    else:
        conclusion = "no_observed_old56_conflict_and_complete_model_fails"
    return {
        "determinism_pass": True,
        "old56_provably_insufficient": insufficient,
        "complete_model_prediction_gate_pass": complete_pass,
        "conclusion": conclusion,
        "QP_calibration_or_closed_loop_authorized": False,
    }

