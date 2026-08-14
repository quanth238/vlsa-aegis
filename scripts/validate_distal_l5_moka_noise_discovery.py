#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.multilink_ellipsoid.moka_noise_discovery import (
    VALIDATION_SCHEMA, canonical, load_config, load_manifest, sha256_bytes,
    sha256_path, summarize, validate_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/vlsa_distal_l5_moka_noise_discovery.v1.json")
    parser.add_argument("--manifest", type=Path, default=root / "manifests/vlsa_distal_l5_moka_noise_discovery.v1.jsonl")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = load_manifest(args.manifest, config)
    observed = json.loads(args.summary.read_text(encoding="utf-8"))
    validate_summary(observed)
    recomputed = summarize(rows=rows, run_root=args.run_root, config=config, source_commit=args.source_commit)
    if canonical(observed) != canonical(recomputed):
        raise ValueError("independent discovery recomputation differs")
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "source_commit": args.source_commit,
        "summary_file_sha256": sha256_path(args.summary),
        "summary_payload_sha256": observed["payload_sha256"],
        "case_count": observed["case_count"],
        "eligible_case_count": observed["eligible_case_count"],
        "downstream_boundary_collection_authorized": observed["downstream_boundary_collection_authorized"],
    }
    value["payload_sha256"] = sha256_bytes(canonical(value))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
