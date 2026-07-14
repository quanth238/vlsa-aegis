#!/usr/bin/env bash
set -euo pipefail

MANIFEST_LOCAL=${1:?usage: submit_oracle_array.sh MANIFEST_JSONL [CONCURRENCY]}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
CONCURRENCY=${2:-1}
case "$CONCURRENCY" in 1|2) ;; *) echo "full-H100 concurrency must be 1 or 2" >&2; exit 2 ;; esac
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
RUN_ID=${RUN_ID:-oracle-validation-$(date -u +%Y%m%dT%H%M%SZ)}
MANIFEST=$REMOTE_REPO/manifests/$(basename "$MANIFEST_LOCAL")
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$REMOTE_REPO/configs/experiments/oracle_validation.json}

scripts/hpc/preflight.sh
COUNT=$(awk 'NF {count++} END {print count+0}' "$MANIFEST_LOCAL")
test "$COUNT" -gt 0
LAST=$((COUNT - 1))
ssh "$HOST" "test -f '$REMOTE_REPO/slurm/oracle_main_array.sbatch' && test -f '$MANIFEST'"
ssh "$HOST" "mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle && cd '$REMOTE_REPO' && RUN_ID='$RUN_ID' MANIFEST='$MANIFEST' EXPERIMENT_CONFIG='$EXPERIMENT_CONFIG' REMOTE_REPO='$REMOTE_REPO' sbatch --array='0-$LAST%$CONCURRENCY' --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' slurm/oracle_main_array.sbatch"
