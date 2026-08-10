#!/usr/bin/env python3
"""Validate unsupported-state audit metrics from recorded exact geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.multilink_ellipsoid.factorized_execution_pilot import (
    load_weights, payload_sha256, predict,
)
from main.multilink_ellipsoid.factorized_unsupported_state_audit import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_unsupported_state_config,
)
from scripts.audit_distal_factorized_unsupported_states_moka10 import (
    add_audit_arguments, analyze_records, audit_paths,
    _validate_experimental_sources,
)
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import load_inputs
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    add_audit_arguments(parser)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = audit_paths(args)
    config = load_unsupported_state_config(paths["audit_config"])
    _validate_experimental_sources(paths, config)
    (
        _, _, complete_dataset, _, _, _, _, _, _, arrays, _, baseline_q,
    ) = load_inputs(paths)
    models, state = load_weights(paths["experimental_model"])
    experimental_q = predict(models, state, arrays)
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("records", {}).get("file_sha256")
        == _file_sha256(args.records.resolve()),
        "unsupported-state result identity differs",
    )
    archive = np.load(args.records.resolve(), allow_pickle=False)
    records = {key: archive[key] for key in archive.files}
    trajectory_archive = np.load(paths["array"], allow_pickle=False)
    analysis = analyze_records(
        records=records, arrays=arrays, baseline_q=baseline_q,
        experimental_q=experimental_q, complete_dataset=complete_dataset,
        trajectory_archive=trajectory_archive, config=config,
    )
    audit_match = bool(
        analysis["states"] == result["audit"]["states"]
        and analysis["training_state_support"]
        == result["audit"]["training_state_support"]
        and analysis["primary_explanation_counts"]
        == result["audit"]["primary_explanation_counts"]
        and analysis["unsupported_state_count"]
        == result["audit"]["unsupported_state_count"]
    )
    forbidden = result.get("forbidden_action_receipt", {})
    forbidden_clean = bool(forbidden and not any(forbidden.values()))
    valid = bool(audit_match and forbidden_clean)
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "complete",
        "valid": valid, "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "audit": {
            "metrics_and_interpretation_recomputed": audit_match,
            "forbidden_actions_absent": forbidden_clean,
            "verdict": "strict_NO_GO_unchanged",
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
