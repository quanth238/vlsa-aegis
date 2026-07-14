#!/usr/bin/env bash
#SBATCH --job-name=crfs-r00-smoke
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
export CRFS_RUNNER_MODE=reach_calibration
export CASE_START=${CASE_START:-0}
export CASE_END=${CASE_END:-0}
exec "$REMOTE_REPO/scripts/hpc/run_oracle_case.sh"
