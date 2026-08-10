"""Test-label-free targeted boundary episode expansion."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .multi_region_affine_oracle import fit_region_target, fixed_regions
from .state_support_smoothness import analyze


CONFIG_SCHEMA = "vlsa_distal_targeted_boundary_expansion_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_targeted_boundary_expansion_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_targeted_boundary_expansion_moka10_validation.v1"
DATASET_SCHEMA = "vlsa_distal_targeted_boundary_expansion_moka10_dataset.v1"
ORACLE_SCHEMA = "vlsa_distal_targeted_boundary_expansion_moka10_oracle.v1"
SCAN_SCHEMA = "vlsa_distal_targeted_boundary_expansion_moka10_scan.v1"
SELECTION_SCHEMA = "vlsa_distal_targeted_boundary_expansion_moka10_selection.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


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
        raise ValueError("targeted expansion config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "episode_split", "state_selection", "collection", "multi_region",
        "support_audit", "decision",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("targeted expansion config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-targeted-boundary-expansion-moka10-v1"
    ):
        raise ValueError("targeted expansion protocol differs")
    if set(config["immutable_source"]) != {
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "affine_collection_config_file_sha256",
        "multi_region_config_file_sha256",
        "base_expanded_dataset_file_sha256",
        "base_expanded_dataset_payload_sha256",
        "base_expanded_oracle_file_sha256",
        "base_expanded_oracle_payload_sha256",
        "prior_result_file_sha256", "prior_result_payload_sha256",
        "prior_validation_file_sha256",
    }:
        raise ValueError("targeted expansion immutable source differs")
    split = config["episode_split"]
    if split != {
        "unit": "complete_episode",
        "additional_train_case_ids": [
            "vlsa-t1-goal-ii-t0-e25", "vlsa-t1-goal-ii-t0-e35",
            "vlsa-t1-goal-ii-t0-e40", "vlsa-t1-goal-ii-t0-e45",
        ],
        "immutable_test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "test_inputs_or_labels_used_for_selection": False,
        "expected_additional_state_counts": {
            "train": 20, "validation": 0, "test": 0,
        },
        "expected_expanded_state_counts": {
            "train": 60, "validation": 10, "test": 15,
        },
    }:
        raise ValueError("targeted expansion split differs")
    if config["state_selection"] != {
        "scan_start_step": 20, "window_state_count": 5,
        "require_every_start_clearance_nonnegative": True,
        "anchor": (
            "smallest_nonnegative_current_seven_row_clearance_with_safe_"
            "four_state_prefix"
        ),
        "maximum_anchor_clearance_m": 0.1,
        "tie_break": "smallest_anchor_step",
        "uses_future_rollout_margin": False,
        "uses_test_episode_features_or_labels": False,
    }:
        raise ValueError("targeted expansion state selection differs")
    if config["collection"] != {
        "reuse_affine_collection_action_grid_without_change": True,
        "expected_episode_count": 4, "expected_states_per_episode": 5,
        "expected_additional_state_count": 20,
        "expected_grid_actions_per_state": 125,
        "simulator": "cloned OSC with every internal substep",
        "device": "H100 allocation only",
    }:
        raise ValueError("targeted expansion collection differs")
    if (
        set(config["multi_region"]) != {"partition", "ridge_huber"}
        or int(config["multi_region"]["partition"].get("region_count", -1))
        != 27
    ):
        raise ValueError("targeted expansion regional settings differ")
    if set(config["support_audit"]) != {
        "feature_shift", "state_support", "oracle_smoothness",
    }:
        raise ValueError("targeted expansion support audit differs")
    if config["support_audit"]["state_support"].get("context") != (
        "frozen_43D_pair_features_excluding_constraint_one_hot_and_"
        "candidate_action"
    ):
        raise ValueError("targeted expansion context representation differs")
    if config["decision"] != {
        "current_region_aware_MLP_retraining_requires_support_and_smoothness_15_of_15": True,
        "if_supported_MLP_fails": (
            "replace coefficient output with action-conditioned conservative "
            "safety-value prediction"
        ),
        "training_in_this_gate": False,
        "closed_loop_E05_authorized": False,
    }:
        raise ValueError("targeted expansion decision differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def load_selected_manifest(
    path: Path, config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    if _sha256(raw) != config["immutable_source"]["selected_manifest_sha256"]:
        raise ValueError("targeted expansion manifest hash differs")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    expected = config["episode_split"]["additional_train_case_ids"]
    if (
        len(rows) != len(expected)
        or [item.get("case_id") for item in rows] != expected
        or any(item.get("split") != "train" for item in rows)
        or any(item.get("first_robot_contact_step") is not None for item in rows)
        or any(
            item.get("task_level_group_id") != "vlsa-t1-goal-ii-t0"
            for item in rows
        )
        or set(expected) & set(config["episode_split"]["immutable_test_case_ids"])
    ):
        raise ValueError("targeted expansion manifest population differs")
    return rows


def collection_config(
    config: Mapping[str, Any], affine_config: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(affine_config))
    source = config["immutable_source"]
    output.update({
        "claim_scope": config["claim_scope"],
        "selected_manifest_sha256": source["selected_manifest_sha256"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
    })
    for key in (
        "source_population_manifest_sha256", "geometry_config_file_sha256",
        "exact_box_config_file_sha256",
    ):
        if output.get(key) != source[key]:
            raise ValueError("targeted expansion collector source differs")
    return output


def select_boundary_steps(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one five-state window per episode without future/test labels."""

    settings = config["state_selection"]
    count = int(settings["window_state_count"])
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_case.setdefault(str(record["case_id"]), []).append(record)
    selections = []
    for case_id in config["episode_split"]["additional_train_case_ids"]:
        ordered = sorted(by_case.get(case_id, []), key=lambda item: item["state_step"])
        by_step = {int(item["state_step"]): item for item in ordered}
        choices = []
        for anchor in sorted(by_step):
            steps = list(range(anchor - count + 1, anchor + 1))
            if steps[0] < int(settings["scan_start_step"]):
                continue
            window = [by_step.get(step) for step in steps]
            if any(item is None for item in window):
                continue
            minima = [float(item["minimum_current_clearance_m"]) for item in window]
            if (
                settings["require_every_start_clearance_nonnegative"]
                and min(minima) < 0.0
            ):
                continue
            anchor_clearance = minima[-1]
            if anchor_clearance > float(settings["maximum_anchor_clearance_m"]):
                continue
            choices.append((anchor_clearance, anchor, steps, minima))
        if not choices:
            selections.append({
                "case_id": case_id, "valid": False,
                "reason": "no_eligible_positive_boundary_window",
            })
            continue
        anchor_clearance, anchor, steps, minima = min(
            choices, key=lambda item: (item[0], item[1])
        )
        selections.append({
            "case_id": case_id, "valid": True,
            "registered_state_steps": steps, "anchor_step": int(anchor),
            "anchor_minimum_current_clearance_m": float(anchor_clearance),
            "window_minimum_current_clearance_m": float(min(minima)),
            "eligible_window_count": len(choices),
        })
    return {
        "selection_uses_future_rollout_margin": False,
        "selection_uses_test_episode_features_or_labels": False,
        "episode_selections": selections,
        "all_episode_windows_valid": all(item["valid"] for item in selections),
        "registered_state_steps_by_case": {
            item["case_id"]: item["registered_state_steps"]
            for item in selections if item["valid"]
        },
    }


def _counts(states: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        name: sum(item["split"] == name for item in states)
        for name in ("train", "validation", "test")
    }


def build_dataset(
    base: Mapping[str, Any], additional: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    base_states = copy.deepcopy(base["state_records"])
    new_states = copy.deepcopy(additional["state_records"])
    if len(base_states) != 65 or len(new_states) != 20:
        raise ValueError("targeted expansion dataset state count differs")
    if _counts(new_states) != config["episode_split"][
        "expected_additional_state_counts"
    ]:
        raise ValueError("targeted expansion new split differs")
    for index, state in enumerate(base_states):
        if int(state["state_index"]) != index:
            raise ValueError("targeted expansion base order differs")
    for offset, state in enumerate(new_states):
        state["additional_source_state_index"] = int(state["state_index"])
        state["state_index"] = len(base_states) + offset
    states = base_states + new_states
    if _counts(states) != config["episode_split"]["expected_expanded_state_counts"]:
        raise ValueError("targeted expansion combined split differs")
    output = {
        "schema_version": DATASET_SCHEMA,
        "source_base_dataset_payload_sha256": base["dataset_payload_sha256"],
        "source_additional_dataset_payload_sha256": additional[
            "dataset_payload_sha256"
        ],
        "constraint_order": copy.deepcopy(base["constraint_order"]),
        "state_records": states,
        "summary": {
            "base_state_count": 65, "additional_state_count": 20,
            "state_count": 85, "split_counts": _counts(states),
            "complete_episode_split": True,
        },
    }
    output["dataset_payload_sha256"] = _hash_without(
        output, "dataset_payload_sha256"
    )
    return output


def build_oracle(
    dataset: Mapping[str, Any], base: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    state_results = copy.deepcopy(base["state_results"])
    if len(state_results) != 65:
        raise ValueError("targeted expansion base oracle count differs")
    for state in dataset["state_records"][65:]:
        regions = fixed_regions(
            state["candidate_first_xyz"], state["action_lower"],
            state["action_upper"], config["multi_region"]["partition"],
        )
        targets = [
            fit_region_target(
                state["candidate_first_xyz"],
                state["candidate_minimum_distal_margin_m"], region,
                config["multi_region"]["ridge_huber"],
            ) for region in regions
        ]
        state_results.append({
            "state_index": int(state["state_index"]),
            "case_id": state["case_id"], "state_step": int(state["state_step"]),
            "regions": regions, "regional_targets": targets,
        })
    output = {
        "schema_version": ORACLE_SCHEMA,
        "source_base_oracle_payload_sha256": base["oracle_payload_sha256"],
        "source_dataset_payload_sha256": dataset["dataset_payload_sha256"],
        "state_results": state_results,
        "summary": {"state_count": 85, "region_count_per_state": 27,
                    "constraint_count": 7},
    }
    output["oracle_payload_sha256"] = _hash_without(
        output, "oracle_payload_sha256"
    )
    return output


def analysis_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "immutable_source": {"expected_state_count": 85},
        "split": {
            "expected_state_counts": config["episode_split"][
                "expected_expanded_state_counts"
            ],
            "test_case_ids": config["episode_split"]["immutable_test_case_ids"],
        },
        "feature_shift": config["support_audit"]["feature_shift"],
        "state_support": config["support_audit"]["state_support"],
        "oracle_smoothness": config["support_audit"]["oracle_smoothness"],
        "decision": {
            "insufficient_support": (
                "same-task archived moka-pot episode coverage exhausted; "
                "collect new complete episodes before learning"
            ),
            "supported_but_nonsmooth": (
                "reduce region size or collect denser local boundary labels"
            ),
            "supported_and_smooth_after_prior_MLP_failure": (
                "preregister current region-aware MLP retraining"
            ),
        },
    }


def evaluate(
    base_dataset: Mapping[str, Any], additional_dataset: Mapping[str, Any],
    base_oracle: Mapping[str, Any], config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset = build_dataset(base_dataset, additional_dataset, config)
    oracle = build_oracle(dataset, base_oracle, config)
    audit = analyze(dataset, oracle, analysis_config(config))
    authorized = bool(
        audit["aggregates"]["support_sufficient"]
        and audit["aggregates"]["oracle_smoothness_sufficient"]
    )
    audit["aggregates"][
        "current_region_aware_MLP_retraining_preregistration_authorized"
    ] = authorized
    audit["aggregates"][
        "action_conditioned_model_preregistration_authorized"
    ] = False
    audit["decision"].update({
        "boundary_collection_simulation_executed": True,
        "new_simulation_executed": True,
        "current_region_aware_MLP_retraining_authorized": authorized,
        "action_conditioned_model_authorized": False,
        "closed_loop_E05_remains_blocked": True,
    })
    return dataset, oracle, audit
