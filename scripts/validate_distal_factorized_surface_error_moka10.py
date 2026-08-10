#!/usr/bin/env python3
"""Independently replay the factorized surface-error diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.surface_error_diagnostic import (
    SURFACE_RESULT_SCHEMA, SURFACE_VALIDATION_SCHEMA, load_surface_config,
)
from scripts.analyze_distal_factorized_surface_error_moka10 import (
    add_common_arguments, compute_surface_error_records, resolved_paths,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = resolved_paths(args)
    records_path = args.records.resolve()
    result_path = args.result.resolve()
    config = load_surface_config(paths["config"])
    result = _load(result_path)
    _require(
        result.get("schema_version") == SURFACE_RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("records", {}).get("file_sha256")
        == _file_sha256(records_path),
        "surface-error result identity differs",
    )
    recomputed_records, recomputed = compute_surface_error_records(paths, config)
    stored_archive = np.load(records_path, allow_pickle=False)
    keys_match = set(stored_archive.files) == set(recomputed_records)
    def array_matches(key: str) -> bool:
        stored = stored_archive[key]
        fresh = np.asarray(recomputed_records[key])
        if stored.dtype.kind in "fc" or fresh.dtype.kind in "fc":
            return bool(np.array_equal(stored, fresh, equal_nan=True))
        return bool(np.array_equal(stored, fresh))
    records_match = bool(keys_match and all(
        array_matches(key) for key in recomputed_records
    ))
    metrics_match = bool(
        recomputed["metrics"] == result["metrics"]
        and recomputed["decision"] == result["decision"]
    )
    forbidden = result.get("forbidden_action_receipt", {})
    forbidden_clean = bool(forbidden and not any(forbidden.values()))
    valid = bool(records_match and metrics_match and forbidden_clean)
    validation = {
        "schema_version": SURFACE_VALIDATION_SCHEMA, "status": "complete",
        "valid": valid, "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "audit": {
            "surface_records_exactly_reproduced": records_match,
            "metrics_and_decision_exactly_reproduced": metrics_match,
            "forbidden_actions_absent": forbidden_clean,
            "decision": recomputed["decision"],
        },
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
