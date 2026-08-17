#!/usr/bin/env python3
"""Validate exact scientific equality of sequential/four-worker candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any, Optional, Sequence

from main.multilink_ellipsoid.candidate_worker_equivalence import (
    RESULT_SCHEMA, compare,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    os.replace(temporary, path)


def validate(*, sequential_path: Path, parallel_path: Path) -> dict[str, Any]:
    sequential_case = _load(sequential_path)
    parallel_case = _load(parallel_path)
    sequential_source_path = Path(sequential_case["source_curve"]["path"])
    parallel_source_path = Path(parallel_case["source_curve"]["path"])
    sequential_source = _load(sequential_source_path)
    parallel_source = _load(parallel_source_path)
    result = compare(
        sequential_case, sequential_source, parallel_case, parallel_source,
    )
    value = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": False,
        "claim_scope": (
            "Single-opened-training-state apparatus equivalence of sequential "
            "and four-process candidate execution; not model or safety evidence."
        ),
        "allocation": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(),
            "device": "cpu_validator",
        },
        "sequential_case": str(sequential_path),
        "parallel_case": str(parallel_path),
        **result,
        "parallel_collection_authorized": bool(result["strict_equivalence_pass"]),
    }
    payload = dict(value)
    value["result_payload_sha256"] = _sha256(_canonical(payload))
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", type=Path, required=True)
    parser.add_argument("--parallel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        sequential_path=args.sequential.resolve(),
        parallel_path=args.parallel.resolve(),
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "strict_equivalence_pass": value["strict_equivalence_pass"],
        "observed_speedup": value["observed_speedup"],
        "checks": value["checks"],
    }, sort_keys=True), flush=True)
    return 0 if value["strict_equivalence_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
