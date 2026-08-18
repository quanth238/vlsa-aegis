#!/usr/bin/env python3
"""Prepare an exact nominal witness at one fixed earlier VLA query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.audit_distal_compiled_box_risk_target import _evaluate_case
from scripts.evaluate_distal_query_action_risk_e05 import evaluate
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _nominal_only(nominal: Any, _frame: Any, _config: Any) -> list[dict[str, Any]]:
    import numpy as np

    actions = np.asarray(nominal, dtype=np.float64)
    _require(actions.shape == (5, 7), "early gradient nominal shape differs")
    return [{
        "name": "nominal", "order": 0,
        "requested_correction_l2_action": 0.0,
        "applied_correction_l2_action": 0.0,
        "actions": actions.tolist(),
    }]


def prepare(
    *, repo_root: Path, config_path: Path, case_index: int,
    expected_commit: str, run_root: Path,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_early_gradient_probe import (
        PREP_SCHEMA, file_sha256, load_config, normalized_exact_case,
        payload_sha256,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        load_config as load_dataset_config,
    )

    config = load_config(config_path)
    _require(0 <= int(case_index) < len(config["cases"]),
             "early gradient prep case index differs")
    selected = config["cases"][int(case_index)]
    source = config["source"]
    for relative, expected, label in (
        (source["population_manifest"], source["population_manifest_file_sha256"], "population"),
        (source["legacy_replay_geometry_config"], source["legacy_replay_geometry_config_file_sha256"], "legacy geometry"),
        (source["base_risk_config"], source["base_risk_config_file_sha256"], "base risk"),
        (source["tight_dataset_config"], source["tight_dataset_config_file_sha256"], "tight dataset"),
    ):
        _require(file_sha256(repo_root / relative) == expected,
                 "early gradient prep %s differs" % label)
    dataset = load_dataset_config(
        repo_root / source["tight_dataset_config"], repo_root=repo_root,
    )
    _require(
        dataset["config_payload_sha256"]
        == source["tight_dataset_config_payload_sha256"],
        "early gradient prep tight dataset payload differs",
    )
    archived_path = Path(source["table1_root"]) / selected[
        "archived_result_relative_path"
    ]
    _require(
        _file_sha256(archived_path) == selected["archived_result_file_sha256"],
        "early gradient prep archive file differs",
    )
    archived = _load(archived_path)
    _require(
        archived.get("result_payload_sha256")
        == selected["archived_result_payload_sha256"],
        "early gradient prep archive payload differs",
    )
    run_root.mkdir(parents=True, exist_ok=False)
    source_path = run_root / "source-curve.json"
    raw = evaluate(
        repo_root=repo_root,
        population_manifest_path=repo_root / source["population_manifest"],
        archived_path=archived_path,
        geometry_config_path=repo_root / source["legacy_replay_geometry_config"],
        experiment_config_path=repo_root / source["base_risk_config"],
        expected_commit=expected_commit, output_path=source_path,
        case_id_override=selected["case_id"],
        state_step_override=int(selected["early_state_step"]),
        query_index_override=int(selected["query_index"]),
        result_schema_override="vlsa_tight_prefix_early_gradient_source.v1",
        claim_scope_override=config["claim_scope"],
        population_binding={
            "case_index": int(case_index), "selection": selected,
            "split": "validation", "test_opened": False,
        },
        candidate_definitions_override=_nominal_only,
        candidate_protocol_binding={
            "candidate_basis": "archived_matched_nominal_only_for_support_witness",
            "radii_or_alpha": [0.0], "temporal_profile": "archived",
            "released_AEGIS_EE_applied_to_every_candidate": False,
            "learned_correction_QP_enabled": False,
        },
        apply_released_aegis_ee_to_all_proposed_actions=False,
        capture_physical_context=True,
        allow_initial_proxy_unsafe_for_empirical_relabel=True,
        require_archived_task_success=True,
    )
    _atomic_write(source_path, raw)
    exact_case = _evaluate_case(
        repo_root=repo_root,
        population_manifest=repo_root / source["population_manifest"],
        geometry_config_path=repo_root / source["legacy_replay_geometry_config"],
        case_config={
            "case_id": selected["case_id"],
            "state_step": int(selected["early_state_step"]),
            "source_result": str(source_path),
            "source_result_file_sha256": _file_sha256(source_path),
            "source_result_payload_sha256": raw["result_payload_sha256"],
            "slab_initialization": "query_state_matching_source",
            "candidate_names": ["nominal"],
        },
        audit_config={
            "rollout_scope": "candidate_five_action_prefix_only",
            "gate": dataset["gate"],
            "exact_group_target": dataset["exact_group_target"],
        },
    )
    result = {
        "schema_version": PREP_SCHEMA,
        "status": "complete_early_query_exact_nominal_prep",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": raw["allocation"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "case_index": int(case_index), "case_id": selected["case_id"],
        "split": "validation", "early_state_step": selected["early_state_step"],
        "source_curve": {
            "path": str(source_path), "file_sha256": _file_sha256(source_path),
            "result_payload_sha256": raw["result_payload_sha256"],
        },
        "exact_case": exact_case,
        "scientific_view": {
            "case_index": int(case_index), "case_id": selected["case_id"],
            "split": "validation", "early_state_step": selected["early_state_step"],
            "config_file_sha256": config["config_file_sha256"],
            "config_payload_sha256": config["config_payload_sha256"],
            "exact_case": normalized_exact_case(exact_case),
        },
        "new_policy_query_count": 0,
        "model_training_performed": False,
        "QP_or_analytical_guidance_performed": False,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        case_index=args.case_index, expected_commit=args.expected_commit,
        run_root=args.run_root.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case_id"],
        "early_state_step": result["early_state_step"],
        "source_replay_exact": result["exact_case"]["source_replay_exact"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
