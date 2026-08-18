#!/usr/bin/env python3
"""Execute one symmetric normal/tangent observability case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.audit_distal_compiled_box_risk_target import _evaluate_case
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def collect(
    *, repo_root: Path, config_path: Path, case_index: int,
    expected_commit: str,
) -> dict:
    import numpy as np
    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        safety_coordinate_feature,
    )
    from main.multilink_ellipsoid.tight_prefix_action_observability import (
        CASE_SCHEMA, active_frame, load_config, payload_sha256,
        symmetric_action_overrides,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        CASE_SCHEMA as TIGHT_CASE_SCHEMA,
        VALIDATION_SCHEMA as TIGHT_VALIDATION_SCHEMA,
        load_config as load_tight_config,
        payload_sha256 as tight_payload,
    )

    config = load_config(config_path)
    _require(0 <= int(case_index) < len(config["cases"]),
             "action-observability case index differs")
    source = config["source"]
    tight_config_path = repo_root / source["tight_dataset_config"]
    _require(
        _file_sha256(tight_config_path)
        == source["tight_dataset_config_file_sha256"],
        "action-observability tight config file differs",
    )
    tight_config = load_tight_config(tight_config_path, repo_root=repo_root)
    _require(
        tight_config["config_payload_sha256"]
        == source["tight_dataset_config_payload_sha256"],
        "action-observability tight config payload differs",
    )
    validation_path = Path(source["tight_dataset_validation"])
    _require(
        _file_sha256(validation_path)
        == source["tight_dataset_validation_file_sha256"],
        "action-observability tight validation file differs",
    )
    validation = _load(validation_path)
    _require(
        validation.get("schema_version") == TIGHT_VALIDATION_SCHEMA
        and validation.get("validation_payload_sha256")
        == source["tight_dataset_validation_payload_sha256"]
        == tight_payload(validation)
        and validation.get("dataset_gate_pass") is True,
        "action-observability tight validation differs",
    )

    selected = config["cases"][int(case_index)]
    tight_index = int(selected["tight_dataset_case_index"])
    tight_selection = dict(tight_config["cases"][tight_index])
    _require(
        tight_selection["case_id"] == selected["case_id"]
        and tight_selection["split"] == selected["split"] == "validation",
        "action-observability selected case differs",
    )
    tight_case_path = (
        Path(source["tight_dataset_producer_dir"])
        / (selected["case_id"] + ".json")
    )
    _require(
        _file_sha256(tight_case_path) == selected["tight_case_file_sha256"],
        "action-observability tight case file differs",
    )
    tight_case = _load(tight_case_path)
    _require(
        tight_case.get("schema_version") == TIGHT_CASE_SCHEMA
        and tight_case.get("case_id") == selected["case_id"]
        and tight_case.get("result_payload_sha256") == tight_payload(tight_case),
        "action-observability tight case differs",
    )
    exact = tight_case["case"]
    perturbation = config["perturbation"]
    frame = active_frame(exact, perturbation["primary_rows"])
    overrides, symmetry = symmetric_action_overrides(
        exact["source_nominal_five_action_chunk"], frame,
        maximum_radius=float(perturbation["maximum_translation_l2_action"]),
        minimum_radius=float(perturbation["minimum_translation_l2_action"]),
        headroom_fraction=float(perturbation["symmetric_headroom_fraction"]),
        action_limit=float(perturbation["action_limit"]),
    )
    _require(
        symmetry["maximum_pair_symmetry_error"]
        <= float(config["gate"]["maximum_post_clipping_symmetry_error"]),
        "action-observability symmetry differs",
    )
    active_row = int(frame["active_row"])
    features = {}
    for item in overrides:
        features[item["name"]] = safety_coordinate_feature(
            exact, {"source_executed_actions": item["actions"]}, active_row,
            translation_scale=0.05, model_rows=tuple(range(10)),
        )
    nominal_feature = np.asarray(features["nominal"], dtype=np.float64)
    tangent_change = max(
        float(np.max(np.abs(
            np.asarray(features[name], dtype=np.float64) - nominal_feature
        )))
        for name in (
            "tangent_up_pos", "tangent_up_neg",
            "tangent_side_pos", "tangent_side_neg",
        )
    )

    tight_selection["candidate_names"] = [item["name"] for item in overrides]
    tight_selection["slab_initialization"] = "query_state_matching_source"
    evaluated = _evaluate_case(
        repo_root=repo_root,
        population_manifest=repo_root / tight_config["source"]["population_manifest"],
        geometry_config_path=(
            repo_root / tight_config["source"]["legacy_replay_geometry_config"]
        ),
        case_config=tight_selection,
        audit_config={
            "rollout_scope": tight_config["rollout_scope"],
            "gate": tight_config["gate"],
            "exact_group_target": tight_config["exact_group_target"],
            "candidate_action_overrides": overrides,
        },
    )
    value = {
        "schema_version": CASE_SCHEMA,
        "status": "complete_symmetric_action_observability_case",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "case_index": int(case_index),
        "case_id": selected["case_id"],
        "split": "validation",
        "config": config,
        "source_artifact": {
            "path": str(tight_case_path),
            "file_sha256": selected["tight_case_file_sha256"],
            "payload_sha256": tight_case["result_payload_sha256"],
        },
        "active_frame": frame,
        "action_definitions": overrides,
        "symmetry_audit": symmetry,
        "active_row_7D_features": features,
        "maximum_active_row_tangent_7D_feature_change": tangent_change,
        "case": evaluated,
        "model_training_performed": False,
        "policy_query_performed": False,
        "correction_or_QP_executed": False,
    }
    value["result_payload_sha256"] = payload_sha256(
        value, "result_payload_sha256",
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = collect(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        case_index=args.case_index, expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "case_id": value["case_id"],
        "active_row": value["active_frame"]["active_row"],
        "radius": value["symmetry_audit"]["common_translation_l2_action"],
        "tangent_7D_feature_change": value[
            "maximum_active_row_tangent_7D_feature_change"
        ],
        "result_payload_sha256": value["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
