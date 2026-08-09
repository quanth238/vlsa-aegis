#!/usr/bin/env python3
"""Independently validate the two-step oracle affine representation result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.oracle_affine_safe_set import (
    ORACLE_AFFINE_SAFE_SET_RESULT_SCHEMA,
    ORACLE_AFFINE_SAFE_SET_VALIDATION_SCHEMA,
    load_oracle_affine_safe_set_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_oracle_affine_safe_set_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == ORACLE_AFFINE_SAFE_SET_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "oracle affine safe-set result identity differs",
    )
    _require(
        result.get("config", {}).get("config_payload_sha256")
        == config["config_payload_sha256"],
        "oracle affine safe-set result config differs",
    )
    allocation = result.get("allocation", {})
    _require(
        "H100" in str(allocation.get("device", {}).get("name"))
        and str(allocation.get("slurm_job_id", "")).isdigit(),
        "oracle affine safe-set result is not H100 allocation-backed",
    )
    cases = result.get("case_results", {})
    _require(
        list(cases) == config["test_case_ids"],
        "oracle affine safe-set result cases differ",
    )
    observed = {}
    for case_id, item in cases.items():
        certificate = item.get("certificate", {})
        qp = item.get("qp", {})
        exact = item.get("exact_two_step_verification")
        certificate_pass = bool(
            certificate.get("valid") is True
            and certificate.get("grid_record_count")
            == config["source_dataset"]["expected_grid_count_per_case"]
            and certificate.get("exact_proxy_raw_safe_grid_candidate_count", 0) > 0
            and certificate.get("sampled_grid_false_safe_candidate_count") == 0
            and float(certificate.get("maximum_sampled_grid_overbound_m", 1.0))
            <= float(
                config["affine_certificate"]["coefficient_postcheck_tolerance_m"]
            )
        )
        qp_pass = bool(
            qp.get("valid") is True
            and item.get("valid_seven_input_row_qp") is True
            and qp.get("diagnostics", {}).get("input_constraint_count") == 7
        )
        exact_pass = bool(
            exact is not None
            and exact.get("D_opt_seven_distal_safe") is True
            and exact.get("released_AEGIS_EE_proxy_safe") is True
            and exact.get("D_sim_raw_safe") is True
            and exact.get("raw_protected_contact_count") == 0
            and float(exact.get("minimum_distal_substep_clearance_m")) >= 0.0
            and float(exact.get("minimum_released_AEGIS_EE_proxy_substep_clearance_m"))
            >= 0.0
            and float(exact.get("maximum_within_step_obstacle_l1_displacement_m"))
            <= float(
                config["exact_verification"]
                ["maximum_per_step_obstacle_l1_displacement_m"]
            )
        )
        gate = bool(certificate_pass and qp_pass and exact_pass)
        _require(
            item.get("case_gate_pass") is gate,
            "oracle affine safe-set case decision differs: %s" % case_id,
        )
        observed[case_id] = {
            "certificate_pass": certificate_pass,
            "qp_pass": qp_pass,
            "exact_pass": exact_pass,
            "case_gate_pass": gate,
        }
    representation_go = bool(all(item["case_gate_pass"] for item in observed.values()))
    decision = result.get("decision", {})
    _require(
        decision.get("representation_go") is representation_go
        and decision.get("affine_coefficient_target_collection_authorized")
        is representation_go
        and decision.get("neural_training_authorized") is False
        and decision.get("closed_loop_e05_authorized") is False,
        "oracle affine safe-set final decision differs",
    )
    output = {
        "schema_version": ORACLE_AFFINE_SAFE_SET_VALIDATION_SCHEMA,
        "status": "validated", "scientific_result": False,
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "case_validation": observed,
        "representation_go": representation_go,
        "affine_coefficient_target_collection_authorized": representation_go,
        "neural_training_authorized": False,
        "closed_loop_e05_authorized": False,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
