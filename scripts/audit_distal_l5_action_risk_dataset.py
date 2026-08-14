#!/usr/bin/env python3
"""Authorize only the restricted three-output L5 prediction pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import _allocation_record
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_action_risk import (
        AUDIT_SCHEMA, load_config, payload_sha256,
    )

    config = load_config(config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = _allocation_record()
    paths = config["source"]
    grouped_path = Path(paths["grouped_validation"])
    active_path = Path(paths["active_boundary_audit"])
    active_validation_path = Path(paths["active_boundary_validation"])
    grouped = _load(grouped_path)
    active = _load(active_path)
    active_validation = _load(active_validation_path)
    _require(_file_sha256(grouped_path) == paths["grouped_validation_file_sha256"]
             and grouped["result_payload_sha256"]
             == paths["grouped_validation_payload_sha256"],
             "L5 action-risk grouped validation differs")
    _require(_file_sha256(active_path) == paths["active_boundary_audit_file_sha256"]
             and active["result_payload_sha256"]
             == paths["active_boundary_audit_payload_sha256"],
             "L5 action-risk active audit differs")
    _require(_file_sha256(active_validation_path)
             == paths["active_boundary_validation_file_sha256"]
             and active_validation["result_payload_sha256"]
             == paths["active_boundary_validation_payload_sha256"]
             and all(active_validation["gates"].values()),
             "L5 action-risk active validation differs")
    frozen_root = Path(paths["frozen_dataset_root"])
    candidate_path = frozen_root / "candidate_manifest.jsonl"
    state_path = frozen_root / "state_manifest.jsonl"
    split_path = frozen_root / "split_manifest.json"
    _require(_file_sha256(candidate_path) == paths["candidate_manifest_file_sha256"]
             and _file_sha256(state_path) == paths["state_manifest_file_sha256"]
             and _file_sha256(split_path) == paths["split_manifest_file_sha256"],
             "L5 action-risk frozen manifest differs")
    _require(grouped["frozen_dataset"]["candidate_manifest_file_sha256"]
             == paths["candidate_manifest_file_sha256"],
             "L5 action-risk grouped freeze binding differs")
    candidates = _read_jsonl(candidate_path)
    states = _read_jsonl(state_path)
    split_manifest = _load(split_path)
    fit = [item for item in candidates if item["split"] in {"train", "validation"}]
    train = [item for item in fit if item["split"] == "train"]
    validation = [item for item in fit if item["split"] == "validation"]
    reserved = set(split_manifest["episode_groups"]["test"])
    _require(not any(item["case_id"] in reserved for item in candidates),
             "L5 action-risk reserved test labels were opened")
    near = float(config["dataset_gate"]["near_boundary_absolute_risk_m"])
    row_records = []
    for row in config["learned_scope"]["rows"]:
        useful_train = [
            item["state_id"] for item in active["state_records"]
            if item["split"] == "train"
            and item["row_records"][row]["useful_boundary_observed"]
        ]
        useful_validation = [
            item["state_id"] for item in active["state_records"]
            if item["split"] == "validation"
            and item["row_records"][row]["useful_boundary_observed"]
        ]
        record = {
            "row": row,
            "known_safe_candidate_count": sum(bool(item["exact_safe"]) for item in fit),
            "known_unsafe_candidate_count": sum(float(item["risk"][row]) > 0.0 for item in fit),
            "near_boundary_candidate_count": sum(abs(float(item["risk"][row])) <= near for item in fit),
            "train_useful_boundary_state_ids": sorted(useful_train),
            "validation_useful_boundary_state_ids": sorted(useful_validation),
        }
        gate = config["dataset_gate"]
        record["adequate"] = bool(
            record["known_safe_candidate_count"]
            >= int(gate["minimum_known_safe_candidates_per_row"])
            and record["known_unsafe_candidate_count"]
            >= int(gate["minimum_known_unsafe_candidates_per_row"])
            and record["near_boundary_candidate_count"]
            >= int(gate["minimum_near_boundary_candidates_per_row"])
            and len(useful_train)
            >= int(gate["minimum_train_useful_boundary_states_per_row"])
            and len(useful_validation)
            >= int(gate["minimum_validation_useful_boundary_states_per_row"])
        )
        row_records.append(record)
    gates = {
        "grouped_and_active_sources_valid": True,
        "candidate_manifest_contains_no_unknown_timeouts": all(
            item["outcome"] != "unknown" for item in candidates
        ),
        "train_sample_count": len(train)
        >= int(config["dataset_gate"]["minimum_train_samples"]),
        "validation_sample_count": len(validation)
        >= int(config["dataset_gate"]["minimum_validation_samples"]),
        "all_three_L5_rows_adequate": all(item["adequate"] for item in row_records),
        "diagnostic_excluded_from_fit": all(
            item["split"] != "diagnostic" for item in fit
        ),
        "reserved_test_episodes_unopened": not any(
            item["case_id"] in reserved for item in candidates
        ),
        "complete_episode_split": split_manifest["unit"] == "complete_episode",
    }
    authorized = all(gates.values())
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation,
        "config": config,
        "source_artifacts": {
            "grouped_validation_file_sha256": _file_sha256(grouped_path),
            "active_audit_file_sha256": _file_sha256(active_path),
            "active_validation_file_sha256": _file_sha256(active_validation_path),
            "candidate_manifest_file_sha256": _file_sha256(candidate_path),
            "state_manifest_file_sha256": _file_sha256(state_path),
            "split_manifest_file_sha256": _file_sha256(split_path),
        },
        "sample_counts": {
            "train": len(train), "validation": len(validation),
            "diagnostic_excluded": sum(item["split"] == "diagnostic" for item in candidates),
            "state_manifest": len(states),
        },
        "row_coverage": row_records,
        "gates": gates,
        "MLP_training_authorized": authorized,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": (
            "three_output_L5_action_risk_training_authorized"
            if authorized else "three_output_L5_action_risk_dataset_no_go"
        ),
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
    result = audit(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "sample_counts": result["sample_counts"],
        "row_coverage": result["row_coverage"],
        "gates": result["gates"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
