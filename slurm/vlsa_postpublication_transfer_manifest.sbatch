#!/usr/bin/env bash
#SBATCH --job-name=vlsa-transfer-v2
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail

# This is a CPU-only postpublication verifier.  It does not copy videos and
# must be submitted with the exact dependency --dependency=afterok:28610.
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
exec "$REMOTE_REPO/slurm/run_vlsa_postpublication_transfer_manifest.sh"
