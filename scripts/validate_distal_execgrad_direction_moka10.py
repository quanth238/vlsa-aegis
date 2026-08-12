#!/usr/bin/env python3
"""Validate the immutable direction-first ExecGrad result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.multilink_ellipsoid.execgrad import load_execgrad_config
from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from scripts.evaluate_distal_execgrad_direction_moka10 import RESULT_SCHEMA
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


VALIDATION_SCHEMA = "vlsa_distal_execgrad_direction_validation.v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--preprocess", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_execgrad_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("config", {}).get("config_payload_sha256")
        == config["config_payload_sha256"]
        and result.get("apparatus", {}).get("apparatus_gate_pass") is True
        and result.get("apparatus", {}).get("artifact", {}).get("file_sha256")
        == _file_sha256(args.preprocess.resolve())
        and result.get("model", {}).get("file_sha256")
        == _file_sha256(args.model.resolve())
        and result.get("predictions", {}).get("file_sha256")
        == _file_sha256(args.predictions.resolve()),
        "ExecGrad result identity differs",
    )
    comparisons = result["direction"]["comparisons"]
    _require(
        set(comparisons)
        == {
            "kinematic_nominal",
            "trajectory_only",
            "trajectory_plus_link_JVP",
            "frozen_direct_normalized_joint_secant",
        }
        and all(
            set(value) == {"train", "validation", "test"}
            for value in comparisons.values()
        )
        and result["decision"]["classification"]
        in {"GO_direction_mechanism", "NO_GO_direction_mechanism"}
        and bool(result["decision"]["fitted_direction_gate_pass"])
        == (result["decision"]["classification"] == "GO_direction_mechanism")
        and result["decision"]["absolute_false_safe_metrics_are_diagnostic_not_gate"] is True,
        "ExecGrad decision structure differs",
    )
    forbidden = result["forbidden_action_receipt"]
    _require(
        not any(bool(value) for value in forbidden.values())
        and result["future_untouched_episode_receipt"]["opened_or_evaluated"] is False,
        "ExecGrad forbidden action was executed",
    )
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "valid": True,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "preprocess_file_sha256": _file_sha256(args.preprocess.resolve()),
        "model_file_sha256": _file_sha256(args.model.resolve()),
        "predictions_file_sha256": _file_sha256(args.predictions.resolve()),
        "expected_commit": args.expected_commit,
        "validator_source": _git_identity(
            args.repo_root.resolve(), args.validator_commit
        ),
        "classification": result["decision"]["classification"],
        "conditional_new_episode_opening_authorized": bool(
            result["decision"]["fitted_direction_gate_pass"]
        ),
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
