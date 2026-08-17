#!/usr/bin/env python3
"""Freshly execute MLP-ranked candidates until exact verification accepts one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.collect_distal_exact_group_boundary import collect
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def run(
    *, repo_root: Path, table1_root: Path, config_path: Path, case_index: int,
    expected_commit: str, run_root: Path,
    fresh_bank_result: Optional[Path] = None,
    fresh_bank_result_file_sha256: Optional[str] = None,
    fresh_bank_result_commit: Optional[str] = None,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.top5_exact_verified_pilot import (
        RESULT_SCHEMA, load_config, payload_sha256, ranked_prefix_summary,
    )

    config = load_config(config_path)
    _require(0 <= int(case_index) < len(config["cases"]),
             "top-five pilot case index differs")
    case = config["cases"][int(case_index)]
    rank_source = config["rank_source"]
    rank_path = Path(rank_source["path"])
    _require(_file_sha256(rank_path) == rank_source["file_sha256"],
             "top-five pilot rank file differs")
    ranking = _load(rank_path)
    _require(ranking.get("result_payload_sha256") == rank_source["payload_sha256"],
             "top-five pilot rank payload differs")
    rank_state = next(
        row for row in ranking["ranked_exact_verification"][case["split"]]["states"]
        if row["state_id"] == case["case_id"]
    )
    frozen_prefix = [
        row["candidate_name"] for row in rank_state["ranked_candidate_prefix"]
    ]
    _require(frozen_prefix == case["ranked_candidate_names"],
             "top-five pilot frozen rank differs")

    source_config = repo_root / case["source_config"]
    _require(_file_sha256(source_config) == case["source_config_file_sha256"],
             "top-five pilot source config differs")
    physical_groups = config["physical_acceptance_groups"]
    if fresh_bank_result is None:
        _require(
            fresh_bank_result_file_sha256 is None
            and fresh_bank_result_commit is None,
            "top-five pilot partial fresh-bank binding differs",
        )
        candidate_runtime = run_root / "fresh-bank-runtime"
        candidate_runtime.mkdir(parents=True, exist_ok=False)
        fresh = collect(
            repo_root=repo_root, table1_root=table1_root,
            config_path=source_config,
            case_index=int(case["source_case_index"]),
            expected_commit=expected_commit,
            run_root=candidate_runtime, candidate_workers=1,
            candidate_shard_name=None,
        )
        fresh_path = run_root / "fresh-bank-result.json"
        _atomic_write(fresh_path, fresh)
        fresh_file_sha256 = _file_sha256(fresh_path)
        execution_source = {
            "mode": "fresh_complete_bank_execution",
            "path": str(fresh_path),
            "file_sha256": fresh_file_sha256,
            "result_payload_sha256": fresh["result_payload_sha256"],
            "execution_commit": expected_commit,
        }
    else:
        _require(
            fresh_bank_result_file_sha256 is not None
            and fresh_bank_result_commit is not None,
            "top-five pilot reused fresh-bank binding is incomplete",
        )
        _require(
            _file_sha256(fresh_bank_result) == fresh_bank_result_file_sha256,
            "top-five pilot reused fresh-bank file differs",
        )
        fresh = _load(fresh_bank_result)
        _require(
            fresh.get("source", {}).get("commit") == fresh_bank_result_commit,
            "top-five pilot reused fresh-bank commit differs",
        )
        fresh_file_sha256 = fresh_bank_result_file_sha256
        execution_source = {
            "mode": "validated_complete_bank_reuse",
            "path": str(fresh_bank_result),
            "file_sha256": fresh_file_sha256,
            "result_payload_sha256": fresh["result_payload_sha256"],
            "execution_commit": fresh_bank_result_commit,
        }

    _require(
        fresh.get("status") == "complete"
        and fresh.get("case_id") == case["case_id"]
        and fresh.get("exact_case", {}).get("source_replay_exact") is True,
        "top-five pilot fresh case differs",
    )
    ranked = ranked_prefix_summary(
        fresh, frozen_prefix,
        [
            float(row["predicted_global"])
            for row in rank_state["ranked_candidate_prefix"]
        ],
        physical_groups, fresh_file_sha256,
    )
    attempts = ranked["attempts"]
    selected = ranked["selected"]

    top_one_unsafe = bool(
        attempts and attempts[0]["known_outcome"]
        and not attempts[0]["physical_safe"]
    )
    selected_safe = bool(selected is not None and selected["physical_safe"])
    value = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete" if selected_safe else "no_verified_safe_top5",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "case_index": int(case_index),
        "case_id": case["case_id"],
        "split": case["split"],
        "rank_source": rank_source,
        "fresh_bank_execution": execution_source,
        "attempts": attempts,
        "selected_rank": None if selected is None else selected["rank"],
        "selected_candidate": None if selected is None else selected["candidate_name"],
        "selected_action_chunk": None if selected is None else selected["executed_actions"],
        "top_one_freshly_unsafe": top_one_unsafe,
        "verified_safe_selection": selected_safe,
        "method_avoids_top_one_collision": bool(top_one_unsafe and selected_safe),
        "model_only_execution_authorized": False,
        "closed_loop_execution_authorized": False,
    }
    value["result_payload_sha256"] = payload_sha256(
        value, "result_payload_sha256",
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-bank-result", type=Path)
    parser.add_argument("--fresh-bank-result-file-sha256")
    parser.add_argument("--fresh-bank-result-commit")
    args = parser.parse_args(argv)
    value = run(
        repo_root=args.repo_root.resolve(), table1_root=args.table1_root.resolve(),
        config_path=args.config.resolve(), case_index=args.case_index,
        expected_commit=args.expected_commit, run_root=args.run_root.resolve(),
        fresh_bank_result=(
            None if args.fresh_bank_result is None
            else args.fresh_bank_result.resolve()
        ),
        fresh_bank_result_file_sha256=args.fresh_bank_result_file_sha256,
        fresh_bank_result_commit=args.fresh_bank_result_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "case_id": value["case_id"], "status": value["status"],
        "selected_rank": value["selected_rank"],
        "selected_candidate": value["selected_candidate"],
        "method_avoids_top_one_collision": value["method_avoids_top_one_collision"],
        "result_payload_sha256": value["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
