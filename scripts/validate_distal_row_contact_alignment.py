#!/usr/bin/env python3
"""Independently recompute the row/contact alignment audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.audit_distal_row_contact_alignment import analyze
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from main.multilink_ellipsoid.row_contact_alignment import (
        AUDIT_SCHEMA, VALIDATION_SCHEMA, load_config, payload_sha256,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    config = load_config(args.config.resolve())
    audit = _load(args.audit_result.resolve())
    _require(audit["schema_version"] == AUDIT_SCHEMA,
             "row-contact audit schema differs")
    _require(audit["result_payload_sha256"]
             == payload_sha256(audit, "result_payload_sha256"),
             "row-contact audit payload differs")
    replay = analyze(config, [item.resolve() for item in args.summary])
    _require(replay == audit["analysis"], "row-contact replay differs")
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "source": source,
        "audit_result_file_sha256": _file_sha256(args.audit_result.resolve()),
        "audit_result_payload_sha256": audit["result_payload_sha256"],
        "analysis_exactly_reproduced": True,
        "decision": replay["decision"],
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256"
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "analysis_exactly_reproduced": True,
        "decision": replay["decision"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
