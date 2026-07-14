#!/usr/bin/env bash
set -euo pipefail

MANIFEST_LOCAL=${1:?usage: submit_endpoint_free_smoke.sh MANIFEST_JSONL}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
RUN_ID=${RUN_ID:-endpoint-free-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
MANIFEST=$REMOTE_REPO/manifests/$(basename "$MANIFEST_LOCAL")
CONFIG_LOCAL=${ENDPOINT_FREE_CONFIG_LOCAL:-$ROOT/configs/experiments/endpoint_free_reach_h5.json}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$REMOTE_REPO/configs/experiments/endpoint_free_reach_h5.json}

scripts/hpc/preflight.sh
test -f "$MANIFEST_LOCAL"
test -f "$CONFIG_LOCAL"
COUNT=$(awk 'NF {count++} END {print count+0}' "$MANIFEST_LOCAL")
test "$COUNT" -eq 20 || { echo "endpoint-free smoke requires the frozen 20-case evaluation manifest" >&2; exit 2; }
python3 - "$CONFIG_LOCAL" "$MANIFEST_LOCAL" <<'PY'
import json
import pathlib
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not config.get("ready_to_run", False):
    raise SystemExit("endpoint-free config is scaffold-only; freeze progress calibration before submission")
if pathlib.Path(config.get("manifest", "")).name != pathlib.Path(sys.argv[2]).name:
    raise SystemExit("smoke must index case 0 from the manifest frozen in the endpoint-free config")
PY

ssh "$HOST" "test -f '$REMOTE_REPO/slurm/endpoint_free_mig.sbatch' && test -f '$MANIFEST' && test -f '$EXPERIMENT_CONFIG' && grep -q 'CRFS_RUNNER_MODE' '$REMOTE_REPO/scripts/hpc/run_oracle_case.sh'"
ssh "$HOST" "mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle && cd '$REMOTE_REPO' && RUN_ID='$RUN_ID' MANIFEST='$MANIFEST' EXPERIMENT_CONFIG='$EXPERIMENT_CONFIG' REMOTE_REPO='$REMOTE_REPO' sbatch --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' slurm/endpoint_free_mig.sbatch"
