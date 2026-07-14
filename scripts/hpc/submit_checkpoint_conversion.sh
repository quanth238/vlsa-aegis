#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}

scripts/hpc/preflight.sh
ssh "$HOST" "test -f '$REMOTE_REPO/slurm/convert_checkpoint_mig.sbatch'"
ssh "$HOST" "mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle && cd '$REMOTE_REPO' && REMOTE_REPO='$REMOTE_REPO' sbatch --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' slurm/convert_checkpoint_mig.sbatch"
