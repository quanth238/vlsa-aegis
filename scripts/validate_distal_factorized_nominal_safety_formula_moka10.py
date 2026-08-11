#!/usr/bin/env python3
"""Independently replay the immutable nominal-safety formula audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.factorized_nominal_safety_audit import (
    RESULT_SCHEMA, VALIDATION_SCHEMA,
)
from scripts.audit_distal_factorized_nominal_safety_formula_moka10 import (
    add_arguments, load_and_audit,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def main() -> int:
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths, _, audit = load_and_audit(args)
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256"),
        "nominal-safety result identity differs",
    )
    checks = {
        "source_commit_exact": result.get("source", {}).get("commit")
        == args.expected_commit,
        "formula_receipt_exact": result.get("formula_receipt")
        == audit["formula_receipt"],
        "split_metrics_exact": result.get("split_metrics")
        == audit["split_metrics"],
        "current_models_exact": result.get("current_structured_models")
        == audit["current_structured_models"],
        "decomposition_exact": result.get(
            "decomposition_identity_maximum_absolute_error_m"
        ) == audit["decomposition_identity_maximum_absolute_error_m"],
        "ratio_exact": result.get(
            "test_joint_prediction_to_geometry_boundary_RMSE_ratio"
        ) == audit["test_joint_prediction_to_geometry_boundary_RMSE_ratio"],
        "decision_exact": result.get("decision") == audit["decision"],
        "forbidden_actions_absent": all(
            value is False for value in result.get(
                "forbidden_action_receipt", {}
            ).values()
        ),
    }
    valid = bool(all(checks.values()))
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "complete",
        "valid": valid, "scientific_result": False,
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(),
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "checks": checks, "recomputed_decision": audit["decision"],
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
