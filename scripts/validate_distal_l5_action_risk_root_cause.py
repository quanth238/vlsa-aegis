#!/usr/bin/env python3
"""Independently recompute the frozen L5 root-cause analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.audit_distal_l5_action_risk_root_cause import analyze
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_action_risk_diagnostic import load_samples


def main(argv: Optional[Sequence[str]] = None) -> int:
    from main.multilink_ellipsoid.l5_action_risk_diagnostic import load_config
    from main.multilink_ellipsoid.l5_action_risk_root_cause import (
        AUDIT_SCHEMA, VALIDATION_SCHEMA, payload_sha256,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-result", type=Path, required=True)
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    config = load_config(args.config.resolve())
    model = _load(args.model_result.resolve())
    audit = _load(args.audit_result.resolve())
    _require(audit["schema_version"] == AUDIT_SCHEMA, "root-cause audit schema differs")
    _require(audit["result_payload_sha256"] == payload_sha256(audit),
             "root-cause audit payload differs")
    replay = analyze(model, load_samples(config))
    _require(replay == audit["analysis"], "root-cause analysis replay differs")
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "source": source,
        "audit_result_file_sha256": _file_sha256(args.audit_result.resolve()),
        "audit_result_payload_sha256": audit["result_payload_sha256"],
        "analysis_exactly_reproduced": True,
        "decision": replay["decision"],
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["validation_payload_sha256"] = payload_sha256(output)
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "analysis_exactly_reproduced": True,
        "primary_root_cause": replay["decision"]["primary_root_cause"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
