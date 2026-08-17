#!/usr/bin/env python3
"""Execute only nominal and the frozen compact-model selected candidate."""

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
) -> dict[str, Any]:
    from main.multilink_ellipsoid.compact_inference_execution import (
        RESULT_SCHEMA, case_definition, frozen_model_selection, load_config,
        payload_sha256, summarize_pair,
    )

    config = load_config(config_path)
    case = case_definition(config, case_index)
    compact_binding = config["compact_result"]
    compact_path = Path(compact_binding["path"])
    _require(
        _file_sha256(compact_path) == compact_binding["file_sha256"],
        "compact inference execution result file differs",
    )
    compact = _load(compact_path)
    _require(
        compact.get("result_payload_sha256") == compact_binding["payload_sha256"]
        and compact["compact_shared_7D"]["model"]["model_sha256"]
        == compact_binding["model_sha256"],
        "compact inference execution model binding differs",
    )
    selection = frozen_model_selection(
        compact, case, config["selection_rule"],
    )

    source_config = repo_root / case["source_config"]
    _require(
        _file_sha256(source_config) == case["source_config_file_sha256"],
        "compact inference execution source config differs",
    )
    fresh = collect(
        repo_root=repo_root, table1_root=table1_root,
        config_path=source_config,
        case_index=int(case["source_case_index"]),
        expected_commit=expected_commit, run_root=run_root,
        candidate_workers=1, source_only=False,
        candidate_subset_names=["nominal", selection["selected_candidate"]],
    )
    _require(
        fresh.get("status") == "complete"
        and fresh.get("case_id") == case["case_id"]
        and fresh.get("original_AEGIS_EE_QP_enabled") is False
        and fresh.get("learned_correction_QP_enabled") is False,
        "compact inference execution fresh case differs",
    )
    pair = summarize_pair(
        fresh, selection["selected_candidate"],
        config["physical_outcome_groups"],
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": fresh["allocation"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "case_index": int(case_index),
        "case_id": str(case["case_id"]),
        "split": str(case["split"]),
        "stratum": str(case["stratum"]),
        "compact_result": compact_binding,
        "model_selection": selection,
        "fresh_pair": fresh,
        "pair": pair,
        "inference_used_future_outcome": False,
        "released_AEGIS_EE_QP_enabled": False,
        "learned_QP_enabled": False,
        "exact_verifier_used_before_execution": False,
        "correction_safety_authorized": False,
        "closed_loop_authorized": False,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repo_root=args.repo_root.resolve(),
        table1_root=args.table1_root.resolve(),
        config_path=args.config.resolve(), case_index=args.case_index,
        expected_commit=args.expected_commit, run_root=args.run_root.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case_id"],
        "nominal_physical_safe": result["pair"]["nominal_physical_safe"],
        "selected_physical_safe": result["pair"]["selected_physical_safe"],
        "collision_avoided": result["pair"]["collision_avoided"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
