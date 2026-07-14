#!/usr/bin/env bash
set -euo pipefail

MANIFEST_LOCAL=${1:?usage: submit_oracle_smoke.sh MANIFEST_JSONL}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
RUN_ID=${RUN_ID:-oracle-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
MANIFEST=$REMOTE_REPO/manifests/$(basename "$MANIFEST_LOCAL")

scripts/hpc/preflight.sh
test -f "$MANIFEST_LOCAL"
COUNT=$(awk 'NF {count++} END {print count+0}' "$MANIFEST_LOCAL")
test "$COUNT" -eq 1 || { echo "smoke manifest must contain exactly one case" >&2; exit 2; }

ssh "$HOST" "test -f '$REMOTE_REPO/slurm/oracle_mig.sbatch' && test -f '$MANIFEST'"
ssh "$HOST" "mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle && cd '$REMOTE_REPO' && RUN_ID='$RUN_ID' MANIFEST='$MANIFEST' REMOTE_REPO='$REMOTE_REPO' sbatch --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' slurm/oracle_mig.sbatch"
