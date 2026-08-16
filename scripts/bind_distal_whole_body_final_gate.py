#!/usr/bin/env python3
"""Write a hash-checked final coverage config or Q-only binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    combined = subparsers.add_parser("combined-audit")
    combined.add_argument("--base", type=Path, required=True)
    combined.add_argument("--targeted-cohort", type=Path, required=True)
    combined.add_argument("--targeted-artifact-root", type=Path, required=True)
    combined.add_argument("--targeted-artifact-commit", required=True)
    combined.add_argument("--targeted-validation", type=Path, required=True)
    combined.add_argument("--output", type=Path, required=True)
    prediction = subparsers.add_parser("q-prediction")
    prediction.add_argument("--protocol", type=Path, required=True)
    prediction.add_argument("--coverage-audit", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from main.multilink_ellipsoid.whole_body_final_bindings import (
        bind_combined_audit,
        bind_q_prediction,
        binding_payload_sha256,
    )

    if args.mode == "combined-audit":
        value = bind_combined_audit(
            base_path=args.base,
            targeted_cohort_path=args.targeted_cohort,
            targeted_artifact_root=args.targeted_artifact_root,
            targeted_artifact_commit=args.targeted_artifact_commit,
            targeted_validation_path=args.targeted_validation,
        )
    else:
        value = bind_q_prediction(
            protocol_path=args.protocol,
            coverage_audit_path=args.coverage_audit,
        )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "mode": args.mode,
        "output": str(args.output.resolve()),
        "binding_payload_sha256": binding_payload_sha256(value),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
