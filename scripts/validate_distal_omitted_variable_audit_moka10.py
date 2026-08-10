#!/usr/bin/env python3
"""Independently validate the omitted-variable audit artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.multilink_ellipsoid.complete_osc_margin import payload_sha256
from main.multilink_ellipsoid.omitted_variable_audit import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, analyze_records, complete_model_gate,
    final_decision, load_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--matched-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {
        "config": args.config, "result": args.result,
        "matched": args.matched_result, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    result = _load(paths["result"])
    matched = _load(paths["matched"])
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"]
        and _file_sha256(paths["matched"])
        == config["immutable_source"]["matched_result_file_sha256"],
        "omitted-variable result identity differs",
    )
    determinism = result["determinism_test"]
    _require(
        determinism == {
            "pair_count": 10625, "duplicate_replay_count": 21250,
            "mismatch_count": 0,
            "maximum_margin_absolute_difference_m": 0.0,
            "margins_contacts_and_next_state_exact": True,
        },
        "omitted-variable determinism receipt differs",
    )
    audit = analyze_records(result["variant_records"])
    complete = complete_model_gate(matched)
    decision = final_decision(audit, complete)
    _require(
        audit == result["omitted_variable_audit"]
        and complete == result["complete_model_test"]
        and decision == result["decision"]
        and int(result["state_count"]) == 85
        and result["forbidden_action_receipt"] == {
            "QP_executed": False, "calibration_executed": False,
            "closed_loop_E05_executed": False,
            "model_retraining_executed": False,
            "new_policy_inference_executed": False,
        },
        "omitted-variable audit decision differs",
    )
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "determinism_test": determinism,
        "recomputed_omitted_variable_audit": audit,
        "recomputed_complete_model_test": complete,
        "recomputed_decision": decision,
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
