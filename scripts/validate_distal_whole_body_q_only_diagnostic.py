#!/usr/bin/env python3
"""Independently retrain the train/validation-only whole-body diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _load, _require,
)
from scripts.train_distal_whole_body_q_only_diagnostic import run


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from main.multilink_ellipsoid.whole_body_q_only_diagnostic import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, payload_sha256,
    )

    result = _load(args.result.resolve())
    _require(result.get("schema_version") == RESULT_SCHEMA,
             "whole-body diagnostic result schema differs")
    _require(result.get("result_payload_sha256") == payload_sha256(
        result, "result_payload_sha256",
    ), "whole-body diagnostic result payload differs")
    replay = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    keys = ("config", "source_artifacts", "dataset", "arms", "diagnostic_only",
            "training_coverage_gate_pass", "test_authorized",
            "correction_authorized", "QP_authorized", "closed_loop_authorized")
    _require(
        canonical({key: replay[key] for key in keys})
        == canonical({key: result[key] for key in keys}),
        "whole-body diagnostic independent retrain differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passed_independent_diagnostic_retrain",
        "scientific_result": True,
        "result_payload_sha256": result["result_payload_sha256"],
        "independent_model_and_prediction_exact": True,
        "test_artifacts_accessed": False,
        "training_authorized": False,
        "correction_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256",
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
