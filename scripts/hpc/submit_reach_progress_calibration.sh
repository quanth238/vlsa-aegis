#!/usr/bin/env bash
set -euo pipefail

MANIFEST_LOCAL=${1:?usage: submit_reach_progress_calibration.sh MANIFEST_JSONL [BATCH_SIZE]}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
BATCH_SIZE=${2:-60}
case "$BATCH_SIZE" in 1|2|3|4|5|6|10|12|15|20|30|40|60) ;; *) echo "unsupported R00 batch size" >&2; exit 2 ;; esac
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
RUN_ID=${RUN_ID:-r00-calibration-$(date -u +%Y%m%dT%H%M%SZ)}
MANIFEST=$REMOTE_REPO/manifests/$(basename "$MANIFEST_LOCAL")
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$REMOTE_REPO/configs/experiments/reach_progress_calibration.json}

scripts/hpc/preflight.sh
COUNT=$(awk 'NF {count++} END {print count+0}' "$MANIFEST_LOCAL")
test "$COUNT" -gt 0
TASKS=$(((COUNT + BATCH_SIZE - 1) / BATCH_SIZE))
test "$TASKS" -le 2 || { echo "R00 calibration may use at most two GPU-equivalent array tasks" >&2; exit 2; }
LAST=$((TASKS - 1))
ssh "$HOST" "test -f '$REMOTE_REPO/slurm/reach_progress_calibration_main.sbatch' && test -f '$MANIFEST' && test -f '$EXPERIMENT_CONFIG'"
ssh "$HOST" "mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle && cd '$REMOTE_REPO' && RUN_ID='$RUN_ID' MANIFEST='$MANIFEST' EXPERIMENT_CONFIG='$EXPERIMENT_CONFIG' REMOTE_REPO='$REMOTE_REPO' CASE_COUNT='$COUNT' BATCH_SIZE='$BATCH_SIZE' sbatch --array='0-$LAST%2' --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' slurm/reach_progress_calibration_main.sbatch"
