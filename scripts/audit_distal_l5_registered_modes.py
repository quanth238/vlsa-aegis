#!/usr/bin/env python3
"""Audit frozen coarse-screen correction modes without new simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def load_states(config: Mapping[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    from main.multilink_ellipsoid.adaptive_query_action_risk_v2 import load_config_v3
    from main.multilink_ellipsoid.l5_row01_selection import load_config as load_row01_config
    from scripts.train_distal_l5_row01_selection import load_samples

    source = config["sources"]
    row01_path = repo_root / source["row01_selection_config"]
    adaptive_path = repo_root / source["adaptive_v3_config"]
    _require(_file_sha256(row01_path) == source["row01_selection_config_file_sha256"],
             "registered audit row01 config differs")
    _require(_file_sha256(adaptive_path) == source["adaptive_v3_config_file_sha256"],
             "registered audit adaptive config differs")
    row01 = load_row01_config(row01_path)
    adaptive = load_config_v3(adaptive_path)
    samples = load_samples(row01)
    by_state: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        for sample in samples[split]:
            state_id = str(sample["state_id"])
            record = by_state.setdefault(state_id, {
                "state_id": state_id, "case_id": sample["case_id"],
                "split": split, "result_path": sample["source_result_path"],
                "result_file_sha256": sample["source_result_file_sha256"],
            })
            _require(
                record["result_path"] == sample["source_result_path"]
                and record["split"] == split,
                "registered audit state source differs",
            )
    _require(len(by_state) == int(config["population"]["expected_unique_proxy_valid_state_count"]),
             "registered audit state count differs")
    states = []
    recipes = adaptive["screening"]["candidate_recipes"]
    for state_id, binding in sorted(by_state.items()):
        path = Path(binding["result_path"])
        _require(_file_sha256(path) == binding["result_file_sha256"],
                 "registered audit source result differs")
        result = _load(path)
        records = result["adaptive_boundary_sampling"]["coarse_screening_records"]
        _require(len(records) == len(recipes) == 12,
                 "registered audit coarse bank differs")
        modes = []
        for order, (record, recipe) in enumerate(zip(records, recipes)):
            definition = record["definition"]
            _require(
                int(definition["order"]) == order
                and definition["name"] == recipe["name"]
                and definition["direction_coefficients"] == recipe["coefficients"]
                and definition["temporal_profile"] == recipe["temporal_profile"],
                "registered audit candidate definition differs",
            )
            prefix_risk = [float(value) for value in record["prefix_risk"]]
            risk = max(prefix_risk[:3])
            physical_veto = bool(record["physical_veto"])
            modes.append({
                "name": definition["name"], "order": order,
                "requested_radius": float(definition["requested_correction_l2_action"]),
                "applied_norm": float(definition["applied_correction_l2_action"]),
                "clipped": bool(definition["clipped"]),
                "risk_m": risk, "per_row_risk_m": prefix_risk[:3],
                "physical_veto": physical_veto,
                "zero_margin_safe": bool(risk <= 0.0 and not physical_veto),
                "screen_safe": bool(record["screen_safe"]),
                "screen_unsafe": bool(record["screen_unsafe"]),
            })
        target_row = result["adaptive_boundary_sampling"]["target_row"]
        states.append({
            **binding,
            "target_row": None if target_row is None else int(target_row),
            "modes": modes,
        })
    return states


def run(repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_registered_mode_audit import (
        RESULT_SCHEMA, audit_states, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    states = load_states(config, repo_root)
    audit = audit_states(states, config)
    if audit["analytical_repulsion_sufficient_on_observed_prefix_population"]:
        interpretation = "strongest_analytical_normal_sufficient_for_observed_prefix_safe_support"
    elif audit["state_dependent_prefix_modes_observed"]:
        interpretation = "state_dependent_prefix_modes_exist_but_complete_value_labels_missing"
    else:
        interpretation = "observed_prefix_artifacts_do_not_justify_learned_mode_selection"
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "source": source,
        "allocation": allocation_record(), "claim_scope": config["claim_scope"],
        "config": config, "audit": audit, "interpretation": interpretation,
        "new_simulation_performed": False, "new_labels_created": False,
        "model_training_performed": False,
        "active_perturbation_collection_authorized": False,
        "learned_selector_authorized": False,
        "QP_authorized": False, "closed_loop_authorized": False,
        "reserved_test_episodes_unopened": True,
    }
    result["result_payload_sha256"] = payload_sha256(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.repo_root.resolve(), args.config.resolve(), args.expected_commit)
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"], "audit": result["audit"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
