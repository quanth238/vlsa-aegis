#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

test "$(git rev-parse HEAD)" = "$(git rev-parse agent/crfs-oracle-harness)" || {
  echo "WHAT: current HEAD is not the CRFS harness branch tip" >&2
  echo "WHY: verification evidence must correspond to the active implementation" >&2
  echo "FIX: switch to agent/crfs-oracle-harness or update the branch intentionally" >&2
  exit 1
}

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m compileall -q src main tests safelibero/libero/libero/envs/env_wrapper.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/audit_harness.py
git diff --check

echo "Harness verification passed. Read feature_list.json and PROGRESS.md before selecting the next gate."
