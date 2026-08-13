#!/usr/bin/env python3
"""Validate and aggregate the immutable clean action-risk array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from main.multilink_ellipsoid.clean_action_risk import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, load_cases, load_config, sha256,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _git_identity(args.repo_root.resolve(), args.validator_commit)
    config = load_config(args.experiment_config.resolve())
    cases = load_cases(args.selection_manifest.resolve(), config)
    accepted = []
    hashes = []
    for index, case in enumerate(cases):
        path = args.run_root.resolve() / ("case-%02d" % index) / "result.json"
        result = _load(path)
        _require(result["schema_version"] == RESULT_SCHEMA, "case result schema differs")
        _require(result["case_index"] == index and result["case"]["case_id"] == case["case_id"], "case result identity differs")
        _require(result["source"]["commit"] == args.producer_commit, "producer commit differs")
        payload = dict(result)
        claimed = payload.pop("result_payload_sha256")
        _require(sha256(canonical(payload)) == claimed, "case result self-hash differs")
        _require(result["requested_state_count"] == 4, "requested state count differs")
        _require(all(state["candidate_count"] == 26 for state in result["states"]), "candidate count differs")
        _require(all(
            candidate["sample_count"] == 651
            for state in result["states"] for candidate in state["candidates"]
        ), "candidate internal sample count differs")
        hashes.append({
            "case_id": case["case_id"], "file_sha256": _file_sha256(path),
            "result_payload_sha256": claimed,
        })
        accepted.append(result)
    split_summary = {}
    rejected = []
    for result in accepted:
        split = result["case"]["split"]
        entry = split_summary.setdefault(split, {
            "episode_count": 0, "requested_state_count": 0,
            "accepted_state_count": 0, "candidate_count": 0,
            "exact_safe_candidate_count": 0,
        })
        entry["episode_count"] += 1
        entry["requested_state_count"] += result["requested_state_count"]
        entry["accepted_state_count"] += result["accepted_state_count"]
        entry["candidate_count"] += sum(state["candidate_count"] for state in result["states"])
        entry["exact_safe_candidate_count"] += sum(state["exact_safe_candidate_count"] for state in result["states"])
        rejected.extend({"case_id": result["case"]["case_id"], **item} for item in result["rejected_states"])
    gates = {
        "all_case_artifacts_complete": len(accepted) == len(cases),
        "all_source_task_success_car_failures_reproduced": all(
            result["gates"]["source_task_success_and_car_failure"] for result in accepted
        ),
        "all_requested_states_strictly_initially_safe": all(
            result["gates"]["all_requested_states_strictly_initially_safe"] for result in accepted
        ),
        "all_requested_states_have_exact_safe_support": all(
            result["gates"]["all_requested_states_have_exact_safe_support"] for result in accepted
        ),
        "candidate_labels_complete_and_finite": all(
            result["gates"]["candidate_count_and_order_exact"]
            and result["gates"]["all_candidate_labels_finite"] for result in accepted
        ),
        "source_states_unmodified_by_cloned_rollouts": all(
            result["gates"]["source_states_unmodified_by_cloned_rollouts"] for result in accepted
        ),
    }
    passed = all(gates.values())
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "validated",
        "scientific_result": True, "producer_commit": args.producer_commit,
        "validator_commit": args.validator_commit, "config": config,
        "case_artifacts": hashes, "split_summary": split_summary,
        "rejected_states": rejected, "gates": gates,
        "interpretation": (
            "clean_exact_action_risk_dataset_pass_training_authorized"
            if passed else "clean_exact_action_risk_dataset_no_go"
        ),
        "MLP_training_authorized": passed,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    validation["validation_payload_sha256"] = sha256(canonical(validation))
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
