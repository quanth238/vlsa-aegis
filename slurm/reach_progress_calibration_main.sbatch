#!/usr/bin/env bash
#SBATCH --job-name=crfs-r00-calibration
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${CASE_COUNT:?CASE_COUNT is required}"
: "${BATCH_SIZE:=60}"
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?array task id is required}
CASE_START=$((TASK_INDEX * BATCH_SIZE))
CASE_END=$((CASE_START + BATCH_SIZE - 1))
if [ "$CASE_END" -ge "$CASE_COUNT" ]; then CASE_END=$((CASE_COUNT - 1)); fi
test "$CASE_START" -le "$CASE_END"
export CRFS_RUNNER_MODE=reach_calibration CASE_START CASE_END
exec "$REMOTE_REPO/scripts/hpc/run_oracle_case.sh"
