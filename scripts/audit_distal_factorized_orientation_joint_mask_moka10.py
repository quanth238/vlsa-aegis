#!/usr/bin/env python3
"""Confirm the duplicated-obstacle-orientation cause with fixed joint masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_orientation_joint_mask import (
    RESULT_SCHEMA, load_joint_mask_config, payload_sha256,
    run_joint_mask_confirmation,
)
from main.multilink_ellipsoid.shadow import allocation_record
from scripts.audit_distal_factorized_input_representation_moka10 import (
    add_arguments, load_inputs, resolved_paths,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def add_joint_arguments(parser: argparse.ArgumentParser) -> None:
    add_arguments(parser)
    parser.add_argument("--joint-config", type=Path, required=True)
    parser.add_argument("--audit-records", type=Path, required=True)
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--audit-validation", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    add_joint_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = resolved_paths(args)
    paths.update({
        "joint_config": args.joint_config.resolve(),
        "audit_records": args.audit_records.resolve(),
        "audit_result": args.audit_result.resolve(),
        "audit_validation": args.audit_validation.resolve(),
        "output": args.output.resolve(),
    })
    joint_config = load_joint_mask_config(paths["joint_config"])
    source = joint_config["immutable_source"]
    for key, path_key in (
        ("input_audit_config_file_sha256", "config"),
        ("input_audit_records_file_sha256", "audit_records"),
        ("input_audit_result_file_sha256", "audit_result"),
        ("input_audit_validation_file_sha256", "audit_validation"),
    ):
        _require(
            _file_sha256(paths[path_key]) == source[key],
            "orientation joint-mask immutable audit source differs",
        )
    audit_result = _load(paths["audit_result"])
    audit_validation = _load(paths["audit_validation"])
    _require(
        audit_result.get("result_payload_sha256")
        == source["input_audit_result_payload_sha256"]
        and audit_validation.get("validation_payload_sha256")
        == source["input_audit_validation_payload_sha256"]
        and audit_validation.get("valid") is True,
        "orientation joint-mask validated audit source differs",
    )
    audit_config, complete_dataset, metadata, archive = load_inputs(paths)
    audit = run_joint_mask_confirmation(
        config=joint_config, audit_config=audit_config,
        complete_dataset=complete_dataset, metadata=metadata, archive=archive,
        flat_model_path=paths["flat_model"],
        flat_predictions_path=paths["flat_predictions"],
        time_model_path=paths["time_model"],
        time_predictions_path=paths["time_predictions"],
    )
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": joint_config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": joint_config,
        "source_audit": {
            "records_file_sha256": _file_sha256(paths["audit_records"]),
            "result_file_sha256": _file_sha256(paths["audit_result"]),
            "validation_file_sha256": _file_sha256(paths["audit_validation"]),
        },
        "audit": audit,
        "forbidden_action_receipt": {
            key: False for key in joint_config["forbidden_actions"]
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "reductions": audit["joint_mask_explosion_RMSE_reduction_fraction"],
        "test_change": audit["joint_mask_test_RMSE_relative_increase"],
        "decision": audit["decision"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
