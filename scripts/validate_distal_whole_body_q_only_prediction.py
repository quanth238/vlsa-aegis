#!/usr/bin/env python3
"""Independently retrain and validate the held-out Q-only prediction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _load, _require,
)
from scripts.train_distal_whole_body_q_only_prediction import run


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from main.multilink_ellipsoid.whole_body_q_only_prediction import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, payload_sha256,
    )

    result = _load(args.result.resolve())
    _require(result.get("schema_version") == RESULT_SCHEMA,
             "whole-body prediction result schema differs")
    _require(
        result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256"),
        "whole-body prediction result payload differs",
    )
    replay = run(
        repo_root=args.repo_root.resolve(), protocol_path=args.protocol.resolve(),
        binding_path=args.binding.resolve(), expected_commit=args.expected_commit,
    )
    keys = (
        "protocol", "binding", "coverage_audit", "source_artifacts", "dataset",
        "arms", "prediction_gate", "coverage_gate_pass",
        "heldout_prediction_gate_pass", "diagnostic_only",
        "correction_authorized", "QP_authorized", "denoising_authorized",
    )
    _require(
        canonical({key: replay[key] for key in keys})
        == canonical({key: result[key] for key in keys}),
        "whole-body prediction independent retrain differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passed_independent_Q_only_retrain",
        "scientific_result": True,
        "result_payload_sha256": result["result_payload_sha256"],
        "independent_model_and_prediction_exact": True,
        "coverage_gate_pass": bool(result["coverage_gate_pass"]),
        "heldout_prediction_gate_pass": bool(
            result["heldout_prediction_gate_pass"]
        ),
        "diagnostic_only": bool(result["diagnostic_only"]),
        "correction_authorized": bool(result["correction_authorized"]),
        "QP_authorized": False,
        "denoising_authorized": False,
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256",
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

