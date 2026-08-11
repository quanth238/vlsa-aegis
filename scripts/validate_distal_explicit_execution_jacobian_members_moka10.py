#!/usr/bin/env python3
"""Independently validate the frozen explicit-J per-member audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.factorized_explicit_execution_jacobian import (
    load_explicit_jacobian_config, load_explicit_jacobian_weights,
)
from main.multilink_ellipsoid.factorized_explicit_jacobian_audit import (
    load_explicit_jacobian_audit_config,
)
from scripts.audit_distal_explicit_execution_jacobian_members_moka10 import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, add_arguments, audit_paths,
    perform_audit, validate_explicit_source,
)
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    direct_paths, validate_direct_sources,
)
from scripts.evaluate_distal_direct_horizon_normalized_secant_moka10 import (
    validate_matched_sources,
)
from scripts.evaluate_distal_explicit_execution_jacobian_moka10 import (
    validate_normalized_source,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.validate_distal_direct_horizon_displacement_moka10 import (
    _array_equal_with_nan,
)


def main() -> int:
    import numpy as np
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-commit", required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, _, _, _, _, _, arrays, sensitivities, _, representation, _,
    ) = prepare(args)
    paths.update(direct_paths(args))
    paths.update(audit_paths(args))
    paths["result"] = args.result.resolve()
    explicit_config = load_explicit_jacobian_config(paths["direct_config"])
    audit_config = load_explicit_jacobian_audit_config(paths["audit_config"])
    validate_direct_sources(paths, explicit_config)
    validate_matched_sources(paths, explicit_config)
    validate_normalized_source(paths, explicit_config)
    source_result = validate_explicit_source(paths, audit_config)
    result = _load(paths["result"])
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit")
        == args.expected_result_commit
        and result.get("records", {}).get("file_sha256")
        == _file_sha256(paths["records"])
        and int(representation["structured_input_dimension"])
        == int(arrays["features"].shape[1]),
        "explicit-J audit result identity differs",
    )
    models, state = load_explicit_jacobian_weights(
        paths["experimental_model"]
    )
    audit, recomputed = perform_audit(
        models=models, state=state, arrays=arrays,
        sensitivities=sensitivities, config=audit_config,
        source_result=source_result,
    )
    stored = np.load(paths["records"], allow_pickle=False)
    array_audit = {}
    for name, value in recomputed.items():
        equal, maximum = _array_equal_with_nan(value, stored[name])
        array_audit[name] = {
            "equal_with_matching_nan_mask": bool(equal),
            "maximum_finite_difference": float(maximum),
        }
    arrays_equal = bool(all(
        item["equal_with_matching_nan_mask"]
        for item in array_audit.values()
    ))
    audit_equal = bool(audit == result["audit"])
    valid = bool(arrays_equal and audit_equal)
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "complete",
        "valid": valid,
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(),
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "records_file_sha256": _file_sha256(paths["records"]),
        "audit": {
            "stored_arrays_exactly_reproduced": arrays_equal,
            "stored_array_audit": array_audit,
            "all_metrics_and_decision_exactly_reproduced": audit_equal,
            "decision": audit["decision"],
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
