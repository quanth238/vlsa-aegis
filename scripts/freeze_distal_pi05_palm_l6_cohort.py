#!/usr/bin/env python3
"""Freeze a palm/L6 cohort before counterfactual outcomes are observed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
    manifest_output: Path,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.pi05_palm_l6_cohort import (
        RESULT_SCHEMA, choose_records, load_config, manifest_bytes,
        manifest_rows, selection_summary,
    )
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        cpu_allocation_record, file_sha256, payload_sha256, scientific_view,
    )

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    source = config["source_audit"]
    producer_path = Path(source["producer_path"])
    replay_path = Path(source["replay_path"])
    validation_path = Path(source["validation_path"])
    for path, key in (
        (producer_path, "producer_file_sha256"),
        (replay_path, "replay_file_sha256"),
        (validation_path, "validation_file_sha256"),
    ):
        _require(file_sha256(path) == source[key], "pi05 cohort source file differs")
    producer = _load(producer_path)
    replay = _load(replay_path)
    validation = _load(validation_path)
    _require(
        producer["result_payload_sha256"] == source["producer_payload_sha256"]
        and replay["result_payload_sha256"] == source["replay_payload_sha256"]
        and validation["validation_payload_sha256"]
        == source["validation_payload_sha256"]
        and bool(validation["source_pool_apparatus_pass"])
        and scientific_view(producer) == scientific_view(replay),
        "pi05 cohort source validation differs",
    )
    chosen = choose_records(producer["records"], config)
    table1_root = Path(producer["config"]["source"]["table1_root"])
    rows = manifest_rows(chosen, table1_root=table1_root)
    payload = manifest_bytes(rows)
    _atomic_write_bytes(manifest_output, payload)
    value = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": identity,
        "allocation": cpu_allocation_record(),
        "config": config,
        "source_validation_payload_sha256": validation[
            "validation_payload_sha256"
        ],
        "manifest_path": str(manifest_output),
        "manifest_file_sha256": hashlib.sha256(payload).hexdigest(),
        "summary": selection_summary(rows),
        "selected_rows": rows,
        "candidate_outcomes_accessed": False,
        "boundary_collection_authorized": False,
        "training_authorized": False,
        "correction_authorized": False,
    }
    value["result_payload_sha256"] = payload_sha256(value)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        manifest_output=args.manifest_output.resolve(),
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "summary": value["summary"],
        "manifest_file_sha256": value["manifest_file_sha256"],
        "result_payload_sha256": value["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
