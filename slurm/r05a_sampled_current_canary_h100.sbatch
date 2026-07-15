#!/usr/bin/env bash
#SBATCH --job-name=crfs-r05a-current-canary
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-0%1
#SBATCH --no-requeue

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/run_r05a_sampled_current_canary.sh
test -x "$RUNNER" || {
  echo "missing executable R05A sampled-current H100 wrapper: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
