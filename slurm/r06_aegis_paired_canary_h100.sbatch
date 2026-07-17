#!/usr/bin/env bash
#SBATCH --job-name=crfs-r06-aegis-pair
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-0%1
#SBATCH --nodelist=worker-1
#SBATCH --no-requeue

set -euo pipefail
exec "${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}/scripts/hpc/run_r06_aegis_paired_canary.sh"
