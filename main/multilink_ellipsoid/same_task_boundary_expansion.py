"""Same-task complete-episode boundary coverage expansion.

This opt-in diagnostic appends three newly measured episodes to the immutable
50-state dataset, fits the already validated fixed multi-region oracle on the
new states, and reruns the frozen unseen-state support audit.  It does not
train a model or execute a closed-loop controller.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .multi_region_affine_oracle import fit_region_target, fixed_regions
from .state_support_smoothness import analyze


CONFIG_SCHEMA = "vlsa_distal_same_task_boundary_expansion_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_same_task_boundary_expansion_moka10_result.v1"
VALIDATION_SCHEMA = (
    "vlsa_distal_same_task_boundary_expansion_moka10_validation.v1"
)
EXPANDED_DATASET_SCHEMA = (
    "vlsa_distal_same_task_boundary_expansion_moka10_dataset.v1"
)
EXPANDED_ORACLE_SCHEMA = (
    "vlsa_distal_same_task_boundary_expansion_moka10_oracle.v1"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
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
        raise ValueError("same-task expansion config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "episode_split", "collection", "multi_region", "support_audit",
        "decision",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("same-task expansion config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-same-task-boundary-expansion-moka10-v1"
    ):
        raise ValueError("same-task expansion protocol differs")
    source = config["immutable_source"]
    if set(source) != {
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "affine_collection_config_file_sha256",
        "multi_region_config_file_sha256", "original_dataset_file_sha256",
        "original_dataset_payload_sha256",
        "original_multi_region_file_sha256",
        "original_multi_region_payload_sha256",
        "original_multi_region_validation_file_sha256",
        "prior_support_result_file_sha256",
        "prior_support_result_payload_sha256",
        "prior_support_validation_file_sha256",
    }:
        raise ValueError("same-task expansion immutable source differs")
    split = config["episode_split"]
    if split != {
        "unit": "complete_episode",
        "additional_train_case_ids": [
            "vlsa-t1-goal-ii-t0-e00", "vlsa-t1-goal-ii-t0-e20",
        ],
        "additional_validation_case_ids": ["vlsa-t1-goal-ii-t0-e30"],
        "immutable_test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "same_task_generalization_only": True,
        "test_labels_never_used_for_collection_fitting_or_thresholds": True,
        "expected_additional_state_counts": {
            "train": 10, "validation": 5, "test": 0,
        },
        "expected_expanded_state_counts": {
            "train": 40, "validation": 10, "test": 15,
        },
    }:
        raise ValueError("same-task expansion episode split differs")
    collection = config["collection"]
    if collection != {
        "reuse_affine_collection_state_and_action_sampling_without_change": True,
        "expected_episode_count": 3, "expected_states_per_episode": 5,
        "expected_additional_state_count": 15,
        "expected_grid_actions_per_state": 125,
        "simulator": "cloned OSC with every internal substep",
        "device": "H100 allocation only",
    }:
        raise ValueError("same-task expansion collection differs")
    multi = config["multi_region"]
    if (
        set(multi) != {
            "reuse_fixed_partition_and_ridge_huber_settings_without_change",
            "partition", "ridge_huber",
        }
        or multi[
            "reuse_fixed_partition_and_ridge_huber_settings_without_change"
        ] is not True
        or int(multi["partition"].get("region_count", -1)) != 27
        or int(multi["partition"].get("fit_actions_per_region", -1)) != 27
    ):
        raise ValueError("same-task expansion regional fit differs")
    audit = config["support_audit"]
    if set(audit) != {"feature_shift", "state_support", "oracle_smoothness"}:
        raise ValueError("same-task expansion support audit differs")
    decision = config["decision"]
    if decision != {
        "authorize_current_region_aware_MLP_retraining_only_if_support_and_smoothness_pass": True,
        "if_supported_model_still_fails": (
            "replace coefficient output with action-conditioned conservative "
            "safety-value prediction"
        ),
        "if_support_fails": (
            "collect more complete same-task episodes without touching "
            "E05_E10_E15"
        ),
        "training_in_this_gate": False,
        "closed_loop_E05_authorized": False,
    }:
        raise ValueError("same-task expansion decision differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def load_selected_manifest(
    path: Path, config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    if _sha256(raw) != config["immutable_source"]["selected_manifest_sha256"]:
        raise ValueError("same-task expansion selected manifest hash differs")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(rows) != 3 or len({item.get("case_id") for item in rows}) != 3:
        raise ValueError("same-task expansion selected cases differ")
    split = config["episode_split"]
    expected = {
        **{case_id: "train" for case_id in split["additional_train_case_ids"]},
        **{
            case_id: "validation"
            for case_id in split["additional_validation_case_ids"]
        },
    }
    observed = {str(item.get("case_id")): str(item.get("split")) for item in rows}
    if observed != expected or set(observed) & set(split["immutable_test_case_ids"]):
        raise ValueError("same-task expansion selected split differs")
    if any(
        item.get("task_level_group_id") != "vlsa-t1-goal-ii-t0"
        for item in rows
    ):
        raise ValueError("same-task expansion selected task differs")
    return rows


def collection_config(
    config: Mapping[str, Any], affine_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt the frozen collector without changing its sampling contract."""

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
            raise ValueError("same-task expansion frozen collector source differs")
    return output


def _split_counts(states: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        name: sum(item["split"] == name for item in states)
        for name in ("train", "validation", "test")
    }


def build_expanded_dataset(
    original: Mapping[str, Any], additional: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    original_states = copy.deepcopy(original["state_records"])
    additional_states = copy.deepcopy(additional["state_records"])
    if len(original_states) != 50 or len(additional_states) != 15:
        raise ValueError("same-task expansion state count differs")
    if _split_counts(additional_states) != config["episode_split"][
        "expected_additional_state_counts"
    ]:
        raise ValueError("same-task expansion additional split count differs")
    test_ids = set(config["episode_split"]["immutable_test_case_ids"])
    if any(item["case_id"] in test_ids for item in additional_states):
        raise ValueError("same-task expansion test episode leaked")
    for expected_index, state in enumerate(original_states):
        if int(state["state_index"]) != expected_index:
            raise ValueError("same-task expansion original state order differs")
    for offset, state in enumerate(additional_states):
        state["additional_source_state_index"] = int(state["state_index"])
        state["state_index"] = 50 + offset
    states = original_states + additional_states
    if _split_counts(states) != config["episode_split"][
        "expected_expanded_state_counts"
    ]:
        raise ValueError("same-task expansion combined split count differs")
    output = {
        "schema_version": EXPANDED_DATASET_SCHEMA,
        "source_original_dataset_payload_sha256": original[
            "dataset_payload_sha256"
        ],
        "source_additional_dataset_payload_sha256": additional[
            "dataset_payload_sha256"
        ],
        "constraint_order": copy.deepcopy(original["constraint_order"]),
        "state_records": states,
        "summary": {
            "original_state_count": 50, "additional_state_count": 15,
            "state_count": 65, "split_counts": _split_counts(states),
            "complete_episode_split": True,
            "immutable_test_case_ids": sorted(test_ids),
        },
    }
    output["dataset_payload_sha256"] = _hash_without(
        output, "dataset_payload_sha256"
    )
    return output


def build_expanded_oracle(
    expanded_dataset: Mapping[str, Any], original_multi: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    original_states = copy.deepcopy(original_multi["state_results"])
    if len(original_states) != 50:
        raise ValueError("same-task expansion original oracle count differs")
    state_results = original_states
    for state in expanded_dataset["state_records"][50:]:
        regions = fixed_regions(
            state["candidate_first_xyz"], state["action_lower"],
            state["action_upper"], config["multi_region"]["partition"],
        )
        targets = [
            fit_region_target(
                state["candidate_first_xyz"],
                state["candidate_minimum_distal_margin_m"], region,
                config["multi_region"]["ridge_huber"],
            )
            for region in regions
        ]
        state_results.append({
            "state_index": int(state["state_index"]),
            "case_id": state["case_id"],
            "state_step": int(state["state_step"]),
            "regions": regions, "regional_targets": targets,
        })
    output = {
        "schema_version": EXPANDED_ORACLE_SCHEMA,
        "source_original_multi_region_payload_sha256": original_multi[
            "result_payload_sha256"
        ],
        "source_expanded_dataset_payload_sha256": expanded_dataset[
            "dataset_payload_sha256"
        ],
        "state_results": state_results,
        "summary": {
            "state_count": len(state_results), "region_count_per_state": 27,
            "constraint_count": 7,
        },
    }
    output["oracle_payload_sha256"] = _hash_without(
        output, "oracle_payload_sha256"
    )
    return output


def support_analysis_config(config: Mapping[str, Any]) -> dict[str, Any]:
    split = config["episode_split"]
    audit = config["support_audit"]
    return {
        "immutable_source": {
            "expected_state_count": 65, "expected_region_count": 27,
            "expected_constraint_count": 7, "fit_actions_per_state": 125,
        },
        "split": {
            "unit": "complete_episode",
            "expected_state_counts": split["expected_expanded_state_counts"],
            "test_case_ids": split["immutable_test_case_ids"],
            "reference_neighbor_exclusion": "same_case_id",
            "test_labels_never_used_to_define_thresholds": True,
        },
        "feature_shift": audit["feature_shift"],
        "state_support": audit["state_support"],
        "oracle_smoothness": audit["oracle_smoothness"],
        "decision": {
            "insufficient_support": config["decision"]["if_support_fails"],
            "supported_but_nonsmooth": (
                "collect denser complete-episode boundary states or reduce "
                "the fixed region size"
            ),
            "supported_and_smooth_after_prior_MLP_failure": (
                "preregister current region-aware MLP retraining on the "
                "expanded complete-episode split"
            ),
        },
    }


def evaluate_expansion(
    original_dataset: Mapping[str, Any], additional_dataset: Mapping[str, Any],
    original_multi: Mapping[str, Any], config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expanded = build_expanded_dataset(original_dataset, additional_dataset, config)
    oracle = build_expanded_oracle(expanded, original_multi, config)
    analysis = analyze(expanded, oracle, support_analysis_config(config))
    authorized = bool(
        analysis["aggregates"]["support_sufficient"]
        and analysis["aggregates"]["oracle_smoothness_sufficient"]
    )
    analysis["aggregates"][
        "action_conditioned_model_preregistration_authorized"
    ] = False
    analysis["aggregates"][
        "current_region_aware_MLP_retraining_preregistration_authorized"
    ] = authorized
    analysis["decision"].update({
        "boundary_collection_simulation_executed": True,
        "current_region_aware_MLP_retraining_authorized": authorized,
        "action_conditioned_model_authorized": False,
        "closed_loop_E05_remains_blocked": True,
    })
    return expanded, oracle, analysis
