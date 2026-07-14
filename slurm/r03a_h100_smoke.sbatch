#!/usr/bin/env bash
#SBATCH --job-name=crfs-r03a-h100-smoke
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-0%1

# This is an apparatus-only one-case fallback when the registered MIG profile
# has insufficient live memory. It matches one full-population H100 task.
set -euo pipefail
: "${SLURM_JOB_ID:?R03A H100 smoke must execute inside Slurm}"
: "${SLURM_ARRAY_TASK_ID:?R03A H100 smoke requires an exact case index}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/run_r03a_case.sh
test -x "$RUNNER" || {
  echo "missing executable R03A allocation worker: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
