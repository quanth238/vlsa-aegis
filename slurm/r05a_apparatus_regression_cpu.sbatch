#!/usr/bin/env bash
#SBATCH --job-name=crfs-r05a-apparatus-cpu
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-0%1
#SBATCH --no-requeue

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/run_r05a_apparatus_regression.sh
test -x "$RUNNER" || {
  echo "missing executable R05A CPU apparatus runner: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
