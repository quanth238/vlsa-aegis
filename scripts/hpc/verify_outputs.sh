#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?usage: verify_outputs.sh RESULTS_ROOT}
PYTHONPATH=${PYTHONPATH:-src} python3 -m crfs_harness.cli aggregate --results-root "$ROOT" --output "$ROOT/aggregate.json"
