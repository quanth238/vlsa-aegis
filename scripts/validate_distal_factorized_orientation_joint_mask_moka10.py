#!/usr/bin/env python3
"""Independently replay the fixed orientation joint-mask confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.multilink_ellipsoid.execution_margin_nn import _canonical
from main.multilink_ellipsoid.factorized_orientation_joint_mask import (
    VALIDATION_SCHEMA, load_joint_mask_config, payload_sha256,
    run_joint_mask_confirmation,
)
from scripts.audit_distal_factorized_input_representation_moka10 import (
    load_inputs, resolved_paths,
)
from scripts.audit_distal_factorized_orientation_joint_mask_moka10 import (
    add_joint_arguments,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    add_joint_arguments(parser)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = resolved_paths(args)
    paths.update({
        "joint_config": args.joint_config.resolve(),
        "audit_records": args.audit_records.resolve(),
        "audit_result": args.audit_result.resolve(),
        "audit_validation": args.audit_validation.resolve(),
        "result": Path(args.result).resolve(),
        "output": Path(args.output).resolve(),
    })
    joint_config = load_joint_mask_config(paths["joint_config"])
    audit_config, complete_dataset, metadata, archive = load_inputs(paths)
    fresh = run_joint_mask_confirmation(
        config=joint_config, audit_config=audit_config,
        complete_dataset=complete_dataset, metadata=metadata, archive=archive,
        flat_model_path=paths["flat_model"],
        flat_predictions_path=paths["flat_predictions"],
        time_model_path=paths["time_model"],
        time_predictions_path=paths["time_predictions"],
    )
    result = _load(paths["result"])
    immutable = joint_config["immutable_source"]
    tests = {
        "source_identity_exact": result.get("source", {}).get("commit")
        == args.expected_commit,
        "result_payload_exact": result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256"),
        "audit_exact": _canonical(result.get("audit", {})) == _canonical(fresh),
        "source_audit_hashes_exact": result.get("source_audit") == {
            "records_file_sha256": _file_sha256(paths["audit_records"]),
            "result_file_sha256": _file_sha256(paths["audit_result"]),
            "validation_file_sha256": _file_sha256(paths["audit_validation"]),
        },
        "source_audit_immutable": bool(
            _file_sha256(paths["config"])
            == immutable["input_audit_config_file_sha256"]
            and _file_sha256(paths["audit_records"])
            == immutable["input_audit_records_file_sha256"]
            and _file_sha256(paths["audit_result"])
            == immutable["input_audit_result_file_sha256"]
            and _file_sha256(paths["audit_validation"])
            == immutable["input_audit_validation_file_sha256"]
        ),
        "forbidden_actions_respected": result.get("forbidden_action_receipt")
        == {key: False for key in joint_config["forbidden_actions"]},
    }
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "source": _git_identity(paths["repo"], args.expected_commit),
        "result_file_sha256": _file_sha256(paths["result"]),
        "tests": tests, "valid": bool(all(tests.values())),
        "decision": fresh["decision"],
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
