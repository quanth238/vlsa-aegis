#!/usr/bin/env bash
#SBATCH --job-name=crfs-oracle
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# The submission wrapper supplies --array with an enforced maximum throttle of 2.
set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
exec "$REMOTE_REPO/scripts/hpc/run_oracle_case.sh"
