#!/usr/bin/env bash
set -euo pipefail

MANIFEST_LOCAL=${1:?usage: submit_reach_progress_smoke.sh ONE_CASE_MANIFEST_JSONL}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
RUN_ID=${RUN_ID:-r00-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
MANIFEST=$REMOTE_REPO/manifests/$(basename "$MANIFEST_LOCAL")
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$REMOTE_REPO/configs/experiments/reach_progress_calibration.json}

scripts/hpc/preflight.sh
COUNT=$(awk 'NF {count++} END {print count+0}' "$MANIFEST_LOCAL")
test "$COUNT" -eq 1 || { echo "R00 smoke manifest must contain exactly one case" >&2; exit 2; }
ssh "$HOST" "test -f '$REMOTE_REPO/slurm/reach_progress_calibration_mig.sbatch' && test -f '$MANIFEST' && test -f '$EXPERIMENT_CONFIG'"
ssh "$HOST" "mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle && cd '$REMOTE_REPO' && RUN_ID='$RUN_ID' MANIFEST='$MANIFEST' EXPERIMENT_CONFIG='$EXPERIMENT_CONFIG' REMOTE_REPO='$REMOTE_REPO' sbatch --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' slurm/reach_progress_calibration_mig.sbatch"
