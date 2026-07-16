#!/usr/bin/env python3
"""Generate the read-only offline diagnostic for sealed AF-00A run C."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from crfs_oracle.r05a_actual_forward_population_diagnostic import (
    ActualForwardPopulationDiagnosticError,
    diagnose_bound_run,
    deterministic_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--output",
        default="-",
        help="JSON destination outside the sealed run root, or '-' for stdout",
    )
    return parser


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    arguments = _parser().parse_args()
    run_root = Path(arguments.run_root).resolve()
    report = diagnose_bound_run(run_root)
    encoded = deterministic_json(report)
    if arguments.output == "-":
        print(encoded, end="")
        return 0
    output = Path(arguments.output).resolve()
    try:
        output.relative_to(run_root)
    except ValueError:
        pass
    else:
        raise ActualForwardPopulationDiagnosticError(
            "diagnostic output must remain outside the sealed AF-00A run root"
        )
    _atomic_write(output, encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
