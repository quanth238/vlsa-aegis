#!/usr/bin/env bash
#SBATCH --job-name=crfs-r06-aegis-val
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
exec "${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}/scripts/hpc/validate_r06_aegis_paired_canary.sh"
