#!/usr/bin/env python3
"""Audit train/validation support for the three frozen L5 clearances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_row01_clearance_endpoint_ablation import load_samples


def _load_frozen(
    config: Mapping[str, Any], repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from main.multilink_ellipsoid.l5_row01_clearance_endpoint_ablation import (
        load_config as load_clearance_config,
        payload_sha256 as clearance_payload_sha256,
    )
    from main.multilink_ellipsoid.l5_row01_obstacle_endpoint_ablation import (
        payload_sha256 as obstacle_payload_sha256,
    )

    source = config["source"]
    clearance_config_path = repo_root / source["clearance_endpoint_config"]
    _require(
        _file_sha256(clearance_config_path)
        == source["clearance_endpoint_config_file_sha256"],
        "clearance audit source config differs",
    )
    clearance_config = load_clearance_config(clearance_config_path)
    samples, _complete, _prior = load_samples(clearance_config, repo_root)
    obstacle_path = Path(source["frozen_obstacle_endpoint_result"])
    clearance_path = Path(source["frozen_clearance_endpoint_result"])
    _require(
        _file_sha256(obstacle_path)
        == source["frozen_obstacle_endpoint_result_file_sha256"],
        "clearance audit frozen 9D file differs",
    )
    _require(
        _file_sha256(clearance_path)
        == source["frozen_clearance_endpoint_result_file_sha256"],
        "clearance audit frozen 12D file differs",
    )
    obstacle = _load(obstacle_path)
    clearance = _load(clearance_path)
    _require(
        obstacle["result_payload_sha256"]
        == source["frozen_obstacle_endpoint_result_payload_sha256"]
        == obstacle_payload_sha256(obstacle),
        "clearance audit frozen 9D payload differs",
    )
    _require(
        clearance["result_payload_sha256"]
        == source["frozen_clearance_endpoint_result_payload_sha256"]
        == clearance_payload_sha256(clearance),
        "clearance audit frozen 12D payload differs",
    )
    return samples, obstacle, clearance


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_row01_clearance_distribution_audit import (
        RESULT_SCHEMA, audit_distribution, classify, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples, obstacle, clearance = _load_frozen(config, repo_root)
    arm9 = obstacle["obstacle_relative_endpoint_9D"]
    arm12 = clearance["clearance_endpoint_12D"]
    settings = config["audit"]
    audit = audit_distribution(
        train_samples=samples["train"],
        validation_samples=samples["validation"],
        train_predictions_9d=arm9["predictions"]["train"],
        validation_predictions_9d=arm9["predictions"]["validation"],
        train_predictions_12d=arm12["predictions"]["train"],
        validation_predictions_12d=arm12["predictions"]["validation"],
        standard_deviation_floor_m=float(
            settings["training_standard_deviation_floor_m"]
        ),
        range_tolerance_m=float(settings["range_tolerance_m"]),
        high_distance_threshold=float(
            settings["high_nearest_training_state_z_distance"]
        ),
    )
    interpretation = classify(audit)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": identity,
        "allocation": allocation_record(),
        "config": config,
        "frozen_metrics": {
            "obstacle_relative_endpoint_9D": arm9["metrics"],
            "clearance_endpoint_12D": arm12["metrics"],
        },
        "audit": audit,
        "interpretation": interpretation,
        "causal_root_cause_proven": False,
        "retain_9D_as_best_predictive_representation": True,
        "training_authorized": False,
        "feature_addition_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
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
    result = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "audit": result["audit"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
