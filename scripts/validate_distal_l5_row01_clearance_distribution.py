#!/usr/bin/env python3
"""Independently reproduce the frozen clearance distribution audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.audit_distal_l5_row01_clearance_distribution import _load_frozen
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path,
    producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_row01_clearance_distribution_audit import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, audit_distribution, classify,
        load_config, payload_sha256,
    )

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(
        result["schema_version"] == RESULT_SCHEMA
        and result["source"]["commit"] == producer_commit,
        "clearance distribution result identity differs",
    )
    _require(
        result["result_payload_sha256"] == payload_sha256(result)
        and result["config"] == config,
        "clearance distribution result payload differs",
    )
    samples, obstacle, clearance = _load_frozen(config, repo_root)
    settings = config["audit"]
    recomputed = audit_distribution(
        train_samples=samples["train"],
        validation_samples=samples["validation"],
        train_predictions_9d=obstacle["obstacle_relative_endpoint_9D"]
        ["predictions"]["train"],
        validation_predictions_9d=obstacle["obstacle_relative_endpoint_9D"]
        ["predictions"]["validation"],
        train_predictions_12d=clearance["clearance_endpoint_12D"]
        ["predictions"]["train"],
        validation_predictions_12d=clearance["clearance_endpoint_12D"]
        ["predictions"]["validation"],
        standard_deviation_floor_m=float(
            settings["training_standard_deviation_floor_m"]
        ),
        range_tolerance_m=float(settings["range_tolerance_m"]),
        high_distance_threshold=float(
            settings["high_nearest_training_state_z_distance"]
        ),
    )
    interpretation = classify(recomputed)
    _require(
        recomputed == result["audit"]
        and interpretation == result["interpretation"],
        "clearance distribution audit replay differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "audit": recomputed,
        "interpretation": interpretation,
        "causal_root_cause_proven": False,
        "training_authorized": False,
        "feature_addition_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["validation_payload_sha256"] = payload_sha256(output)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root.resolve(), result_path=args.result.resolve(),
        config_path=args.config.resolve(), producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "interpretation": output["interpretation"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
