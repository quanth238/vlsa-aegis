#!/usr/bin/env python3
"""Generate one allocation-only source bundle for the custom task-0 pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from crfs_harness.artifacts import atomic_write_json, file_sha256
from crfs_oracle.generated_source import (
    allocation_provenance,
    generate_source_group,
    load_generated_source_config,
    rejection_bundle,
    valid_generated_source_completion,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group-index", required=True, type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser().resolve()
    config_sha256 = file_sha256(config_path)
    if config_sha256 != args.expected_config_sha256:
        raise SystemExit("generated-source command config differs from expected SHA-256")
    config = load_generated_source_config(
        config_path,
        repo_root=root,
        expected_config_sha256=args.expected_config_sha256,
    )
    if config.get("ready_to_run") is not True or config.get("blocked_on") != []:
        raise SystemExit("generated-source config remains blocked pending independent review")
    groups = config["source_groups"]
    if not 0 <= args.group_index < len(groups):
        raise SystemExit("group-index lies outside the frozen source schedule")
    request_id = groups[args.group_index]["generation_request_id"]
    output = Path(args.output_root).expanduser().resolve() / args.run_id / request_id / "source-bundle.json"
    if output.exists():
        if valid_generated_source_completion(output):
            print(f"skipped {output}")
            return 0
        raise SystemExit(f"refusing to overwrite invalid existing final artifact: {output}")

    try:
        provenance = allocation_provenance(root)
        bundle = generate_source_group(
            config,
            config_file_sha256=config_sha256,
            group_index=args.group_index,
            run_id=args.run_id,
            repo_root=root,
        )
    except Exception as error:
        try:
            provenance
        except UnboundLocalError:
            provenance = None
        rejected = rejection_bundle(
            config=config,
            config_file_sha256=config_sha256,
            group_index=args.group_index,
            run_id=args.run_id,
            stage="source_generation",
            code=type(error).__name__,
            message=str(error),
            provenance=provenance,
        )
        atomic_write_json(output, rejected)
        print(f"rejected {output}")
        return 1
    atomic_write_json(output, bundle)
    print(f"accepted {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
