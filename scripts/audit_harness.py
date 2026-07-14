#!/usr/bin/env python3
"""Dependency-free structural audit for the CRFS research harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)


def main() -> int:
    required = [
        "AGENTS.md",
        "README.md",
        "PROGRESS.md",
        "DECISIONS.md",
        "feature_list.json",
        "docs/architecture.md",
        "docs/experiment_protocol.md",
        "docs/oracle_acceptance_gates.md",
        "docs/infrastructure/vinuni_h100_runbook.md",
        "schemas/case-manifest.schema.json",
        "schemas/case-result.schema.json",
        "schemas/provenance.schema.json",
        "scripts/hpc/preflight.sh",
        "scripts/hpc/convert_checkpoint.sh",
        "scripts/hpc/run_oracle_case.sh",
        "scripts/hpc/submit_oracle_smoke.sh",
        "scripts/hpc/submit_oracle_array.sh",
        "slurm/oracle_mig.sbatch",
        "slurm/convert_checkpoint_mig.sbatch",
        "slurm/oracle_main_array.sbatch",
    ]
    errors = []
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required artifact: {relative}")

    features_path = ROOT / "feature_list.json"
    try:
        features = json.loads(features_path.read_text(encoding="utf-8"))["features"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        errors.append(f"feature_list.json is invalid: {error}")
        features = []
    states = {"not_started", "active", "blocked", "passing"}
    active = 0
    seen = set()
    for feature in features:
        identity = feature.get("id")
        if not identity or identity in seen:
            errors.append(f"duplicate or empty feature id: {identity!r}")
        seen.add(identity)
        if feature.get("status") not in states:
            errors.append(f"feature {identity} has noncanonical status {feature.get('status')!r}")
        active += feature.get("status") == "active"
        if feature.get("status") == "passing" and not feature.get("evidence"):
            errors.append(f"passing feature {identity} has no evidence")
    if active > 1:
        errors.append(f"WIP limit violated: {active} active features")

    baseline = "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b"
    if baseline not in (ROOT / "README.md").read_text(encoding="utf-8"):
        errors.append("README does not pin the VLSA-Aegis baseline commit")
    sampler = (ROOT / "openpi/src/openpi/models_pytorch/pi0_pytorch.py").read_text(encoding="utf-8")
    if 'crfs_intervention_mode="none"' not in sampler:
        errors.append("baseline sampler lacks a backward-compatible CRFS-off default")
    runner = (ROOT / "main/crfs_oracle/runner.py").read_text(encoding="utf-8")
    if "real_safelibero_preliminary" not in runner or "research_limitations" not in runner:
        errors.append("real runner does not disclose preliminary evidence limitations")

    for script in [
        "scripts/hpc/preflight.sh",
        "scripts/hpc/convert_checkpoint.sh",
        "scripts/hpc/run_oracle_case.sh",
        "scripts/hpc/submit_oracle_smoke.sh",
        "scripts/hpc/submit_oracle_array.sh",
    ]:
        path = ROOT / script
        if path.exists() and "set -euo pipefail" not in path.read_text(encoding="utf-8"):
            errors.append(f"{script} does not fail closed")

    if errors:
        for error in errors:
            fail(error)
        return 1
    print(f"PASS: {len(required)} artifacts, {len(features)} gates, WIP <= 1, baseline pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
