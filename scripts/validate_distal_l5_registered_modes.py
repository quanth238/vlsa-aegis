#!/usr/bin/env python3
"""Independent replay of the frozen registered-mode audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def validate(repo_root: Path, config_path: Path, result_path: Path,
             producer_commit: str, validator_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_registered_mode_audit import (
        VALIDATION_SCHEMA, audit_states, load_config, payload_sha256,
    )
    from scripts.audit_distal_l5_registered_modes import load_states

    config = load_config(config_path)
    result = _load(result_path)
    _require(result["source"]["commit"] == producer_commit,
             "registered-mode producer commit differs")
    _require(result["result_payload_sha256"] == payload_sha256(result),
             "registered-mode result payload differs")
    audit = audit_states(load_states(config, repo_root), config)
    _require(audit == result["audit"], "registered-mode audit replay differs")
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "passing",
        "producer_commit": producer_commit, "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "audit_exactly_reproduced": True, "audit": audit,
        "interpretation": result["interpretation"],
    }
    validation["validation_payload_sha256"] = payload_sha256(validation)
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        args.repo_root.resolve(), args.config.resolve(), args.result.resolve(),
        args.producer_commit, args.validator_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "interpretation": output["interpretation"],
        "audit_exactly_reproduced": output["audit_exactly_reproduced"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
