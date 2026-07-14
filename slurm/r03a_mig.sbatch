#!/usr/bin/env bash
#SBATCH --job-name=crfs-r03a-smoke
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --array=0-0%1

set -euo pipefail
: "${SLURM_JOB_ID:?R03A smoke must execute inside Slurm}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/run_r03a_case.sh
test -x "$RUNNER" || {
  echo "missing executable R03A allocation worker: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
