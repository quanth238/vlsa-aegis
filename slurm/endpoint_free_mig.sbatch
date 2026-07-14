#!/usr/bin/env bash
#SBATCH --job-name=crfs-endpoint-free-smoke
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

RUNNER=$REMOTE_REPO/scripts/hpc/run_oracle_case.sh
grep -q 'CRFS_RUNNER_MODE' "$RUNNER" || {
  echo "endpoint-free runner mode is not integrated in $RUNNER" >&2
  exit 2
}
export CRFS_RUNNER_MODE=endpoint_free
exec "$RUNNER"
