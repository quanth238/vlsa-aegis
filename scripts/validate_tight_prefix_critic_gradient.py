#!/usr/bin/env python3
"""Validate independent compact-critic gradient-audit replicas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def run(
    *, repo_root: Path, config_path: Path, producer_path: Path,
    replay_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        cpu_allocation_record,
    )
    from main.multilink_ellipsoid.tight_prefix_critic_gradient_audit import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, load_config, payload_sha256,
        scientific_view,
    )

    config = load_config(config_path)
    producer = _load(producer_path)
    replay = _load(replay_path)
    for value in (producer, replay):
        _require(
            value.get("schema_version") == RESULT_SCHEMA
            and value.get("status")
            == "complete_zero_simulation_critic_gradient_audit"
            and value.get("scientific_result") is True
            and value.get("result_payload_sha256")
            == payload_sha256(value, "result_payload_sha256")
            and value.get("config", {}).get("config_payload_sha256")
            == config["config_payload_sha256"]
            and value.get("source", {}).get("commit") == expected_commit
            and value.get("new_simulator_rollout_count") == 0
            and value.get("model_retraining_performed") is False
            and value.get("action_or_flow_correction_executed") is False
            and value.get("test_used_for_gate_or_tuning") is False
            and value.get("paper_or_safety_claim_authorized") is False,
            "critic gradient result contract differs",
        )
    exact = scientific_view(producer) == scientific_view(replay)
    _require(exact, "critic gradient independent scientific views differ")
    numerical = producer["numerical_gradient_audit"]
    _require(
        numerical.get("stored_prediction_pass") is True
        and numerical.get("autograd_finite_difference_pass") is True,
        "critic gradient numerical apparatus fails",
    )
    verdict = producer["feasibility_verdict_from_validation_only"]
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated_zero_simulation_critic_gradient_audit",
        "scientific_result": True,
        "source": _git_identity(repo_root, expected_commit),
        "allocation": cpu_allocation_record(),
        "config": config,
        "producer": {
            "path": str(producer_path),
            "file_sha256": _file_sha256(producer_path),
            "payload_sha256": producer["result_payload_sha256"],
        },
        "replay": {
            "path": str(replay_path),
            "file_sha256": _file_sha256(replay_path),
            "payload_sha256": replay["result_payload_sha256"],
        },
        "independent_scientific_view_exact": exact,
        "numerical_gradient_audit": numerical,
        "validation_unsafe_trigger": producer["unsafe_trigger_by_split"][
            "validation"
        ],
        "validation_pair_metrics": producer["pair_metrics_by_split"][
            "validation"
        ],
        "diagnostic_test_unsafe_trigger": producer["unsafe_trigger_by_split"][
            "test"
        ],
        "diagnostic_test_pair_metrics": producer["pair_metrics_by_split"][
            "test"
        ],
        "feasibility_verdict_from_validation_only": verdict,
        "action_space_exact_gradient_probe_authorized": bool(
            verdict["action_space_exact_gradient_probe_authorized"]
        ),
        "flow_guidance_authorized": False,
        "new_simulator_rollout_count": 0,
        "paper_or_safety_claim_authorized": False,
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256",
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_path=args.producer.resolve(), replay_path=args.replay.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "validation_unsafe_trigger": result["validation_unsafe_trigger"],
        "validation_pair_metrics": result["validation_pair_metrics"],
        "action_space_exact_gradient_probe_authorized": result[
            "action_space_exact_gradient_probe_authorized"
        ],
        "flow_guidance_authorized": result["flow_guidance_authorized"],
        "validation_payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
