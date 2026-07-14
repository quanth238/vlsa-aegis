#!/usr/bin/env bash
#SBATCH --job-name=crfs-endpoint-free
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# The submission wrapper supplies --array and restricts concurrency to one or two.
set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/run_oracle_case.sh
grep -q 'CRFS_RUNNER_MODE' "$RUNNER" || {
  echo "endpoint-free runner mode is not integrated in $RUNNER" >&2
  exit 2
}
export CRFS_RUNNER_MODE=endpoint_free
exec "$RUNNER"
