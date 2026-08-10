#!/usr/bin/env python3
"""Validate terminal-bias metrics from immutable recorded geometry traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.factorized_terminal_bias_audit import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_terminal_bias_config,
)
from scripts.audit_distal_factorized_terminal_bias_moka10 import (
    add_common_arguments, analyze_records, resolved_paths,
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
    config = load_terminal_bias_config(paths["config"])
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("records", {}).get("file_sha256")
        == _file_sha256(args.records.resolve()),
        "terminal-bias result identity differs",
    )
    archive = np.load(args.records.resolve(), allow_pickle=False)
    records = {key: archive[key] for key in archive.files}
    analysis = analyze_records(records, config)
    metrics_match = bool(
        analysis["residual"] == result["metrics"]["residual"]
        and analysis["geometry_signal_metrics"]
        == result["metrics"]["geometry_signal_metrics"]
        and analysis["geometry_signal_decision"]
        == result["metrics"]["geometry_signal_decision"]
        and analysis["ensemble_disagreement"]
        == result["metrics"]["ensemble_disagreement"]
        and analysis["decision"] == result["decision"]
    )
    forbidden = result.get("forbidden_action_receipt", {})
    forbidden_clean = bool(forbidden and not any(forbidden.values()))
    valid = bool(metrics_match and forbidden_clean)
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "complete",
        "valid": valid, "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "audit": {
            "all_metrics_and_decision_recomputed": metrics_match,
            "forbidden_actions_absent": forbidden_clean,
            "training_submitted": False,
            "decision": analysis["decision"],
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
