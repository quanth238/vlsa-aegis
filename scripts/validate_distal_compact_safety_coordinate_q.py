#!/usr/bin/env python3
"""Independently retrain and validate the compact inference-only Q model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _load, _require,
)
from scripts.train_distal_compact_safety_coordinate_q import run


def validate(
    *, repo_root: Path, config_path: Path, result_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, file_sha256, payload_sha256,
        scientific_view,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    result_file_sha256 = file_sha256(result_path)
    source = _load(result_path)
    _require(
        source.get("schema_version") == RESULT_SCHEMA
        and source.get("result_payload_sha256")
        == payload_sha256(source, "result_payload_sha256"),
        "compact safety-coordinate result payload differs",
    )
    reproduced = run(
        repo_root=repo_root, config_path=config_path,
        expected_commit=expected_commit,
    )
    exact = scientific_view(source) == scientific_view(reproduced)
    _require(exact, "compact safety-coordinate independent retrain differs")
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "allocation": allocation_record(),
        "source_result": {
            "path": str(result_path),
            "file_sha256": result_file_sha256,
            "payload_sha256": source["result_payload_sha256"],
        },
        "independent_retrain": {
            "exact_scientific_reproduction": exact,
            "reproduced_model_sha256": reproduced["compact_shared_7D"][
                "model"
            ]["model_sha256"],
        },
        "simulation_rollouts": False,
        "inference_only": True,
        "diagnostic_only": True,
        "correction_authorized": False,
    }
    value["validation_payload_sha256"] = payload_sha256(
        value, "validation_payload_sha256",
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        result_path=args.result.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "independent_retrain": value["independent_retrain"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
