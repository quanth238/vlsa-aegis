#!/usr/bin/env bash
#SBATCH --job-name=crfs-oracle-smoke
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-0%1

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
exec "$REMOTE_REPO/scripts/hpc/run_oracle_case.sh"
