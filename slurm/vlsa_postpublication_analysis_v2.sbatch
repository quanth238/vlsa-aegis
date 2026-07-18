#!/usr/bin/env bash
#SBATCH --job-name=vlsa-a2-p28610
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

# Four CPUs and 32 GiB are sufficient because the existing analysis-v2
# builder streams compact results and holds no rollout model or renderer.
# Submit only through the reviewed held-afterok helper.
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
exec "$REMOTE_REPO/slurm/run_vlsa_postpublication_analysis_v2.sh"
