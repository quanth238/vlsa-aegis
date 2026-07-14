#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"

HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
RUN_ID=${1:-sampler-parity-$(date -u +%Y%m%dT%H%M%SZ)}
case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
    exit 2
    ;;
esac

test -f main/run_sampler_parity.py
test -f slurm/sampler_parity_mig.sbatch
test -f manifests/oracle_h05_colliding.jsonl
test -f configs/experiments/endpoint_free_reach_h5.json
test "$(shasum -a 256 evidence/r01/r01-summary.json | awk '{print $1}')" = \
  715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5

# Live cluster/QOS/storage state is authoritative.  This also rejects any
# experiment-like Python/CUDA process left on the login node.
scripts/hpc/preflight.sh

ssh "$HOST" "test -f '$REMOTE_REPO/main/run_sampler_parity.py' && \
test -f '$REMOTE_REPO/slurm/sampler_parity_mig.sbatch' && \
test -f '$REMOTE_REPO/manifests/oracle_h05_colliding.jsonl' && \
test -d /mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero/params && \
test -f /mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors && \
mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle && \
cd '$REMOTE_REPO' && \
RUN_ID='$RUN_ID' REMOTE_REPO='$REMOTE_REPO' \
sbatch --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' slurm/sampler_parity_mig.sbatch"
