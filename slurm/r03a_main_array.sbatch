#!/usr/bin/env bash
#SBATCH --job-name=crfs-r03a
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# The registered grouped submitter creates exactly two source-pinned arrays:
# worker-1 gets --array=0-4,6-16%1 and worker-2 gets --array=5%1.  Their joint
# ceiling is two GPU-equivalents, 16 CPUs, and 256 GiB RAM.
set -euo pipefail
: "${SLURM_JOB_ID:?R03A array must execute inside Slurm}"
: "${SLURM_ARRAY_TASK_ID:?R03A array requires an exact case index}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/run_r03a_case.sh
test -x "$RUNNER" || {
  echo "missing executable R03A allocation worker: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
