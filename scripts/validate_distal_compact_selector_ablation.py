#!/usr/bin/env python3
"""Validate independent compact-selector ablation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def run(*, repo_root: Path, config_path: Path, producer_path: Path,
        replay_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.compact_selector_ablation import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, load_config, payload_sha256,
        scientific_view,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    config = load_config(config_path)
    producer = _load(producer_path)
    replay = _load(replay_path)
    for result in (producer, replay):
        _require(
            result.get("schema_version") == RESULT_SCHEMA
            and result.get("status")
            == "complete_posthoc_inference_only_diagnostic"
            and result.get("result_payload_sha256")
            == payload_sha256(result, "result_payload_sha256")
            and result.get("new_simulator_rollout_count") == 0
            and result.get("retraining_performed") is False
            and result.get("test_used_for_margin_or_arm_choice") is False
            and result.get("prospective_paper_test_authorized") is False
            and result.get("correction_safety_authorized") is False,
            "compact selector result contract differs",
        )
    exact = scientific_view(producer) == scientific_view(replay)
    _require(exact, "compact selector independent result differs")
    margins = producer["frozen_validation_parameters"]
    _require(
        margins["fit_split"] == "validation"
        and margins["test_metrics_used"] is False
        and margins["UNKNOWN_censored"] is True,
        "compact selector margin provenance differs",
    )
    # The first valid inference artifacts stored shorthand focus keys but used
    # exact manifest state IDs internally. Resolve those rows read-only rather
    # than rerunning an otherwise valid H100 audit.
    focus = {}
    for state_id in config["evaluation"]["report_states"]:
        suffix = "-" + state_id.lower()
        focus[state_id] = {
            name: next((
                row for row in producer["arms"]["test"][name]["states"]
                if str(row["state_id"]).lower().endswith(suffix)
            ), None)
            for name in config["arms"]
        }
    _require(
        set(focus) == {"E15", "E42"}
        and all(all(value is not None for value in rows.values())
                for rows in focus.values()),
        "compact selector focus-state audit differs",
    )
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passing_posthoc_inference_only_diagnostic",
        "scientific_result": True,
        "source": _git_identity(repo_root, expected_commit),
        "allocation": allocation_record(),
        "config": config,
        "producer": {
            "path": str(producer_path),
            "file_sha256": _file_sha256(producer_path),
            "payload_sha256": producer["result_payload_sha256"],
            "source_commit": producer["source"]["commit"],
        },
        "independent_replay": {
            "path": str(replay_path),
            "file_sha256": _file_sha256(replay_path),
            "payload_sha256": replay["result_payload_sha256"],
            "source_commit": replay["source"]["commit"],
            "exact_scientific_reproduction": exact,
        },
        "frozen_validation_parameters": margins,
        "validation_only_arm_choice": producer[
            "validation_only_arm_choice"
        ],
        "arms": producer["arms"],
        "focus_states": focus,
        "diagnostic_comparison": producer["diagnostic_comparison"],
        "new_simulator_rollout_count": 0,
        "retraining_performed": False,
        "test_used_for_margin_or_arm_choice": False,
        "prospective_paper_test_authorized": False,
        "correction_safety_authorized": False,
        "QP_authorized": False,
        "denoising_authorized": False,
        "CBF_claim_authorized": False,
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256",
    )
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    validation = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_path=args.producer.resolve(), replay_path=args.replay.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps({
        "status": validation["status"],
        "comparison": validation["diagnostic_comparison"],
        "validation_payload_sha256": validation[
            "validation_payload_sha256"
        ],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
