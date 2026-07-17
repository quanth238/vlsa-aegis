#!/usr/bin/env bash
#SBATCH --job-name=vlsa-pi05-hash
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-0%1
#SBATCH --exclude=worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
exec "$REMOTE_REPO/slurm/run_pi05_hash_receipt.sh"
