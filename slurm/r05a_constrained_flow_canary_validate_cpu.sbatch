#!/usr/bin/env bash
#SBATCH --job-name=crfs-r05a-cfs-validate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/validate_r05a_constrained_flow_canary.sh
test -x "$RUNNER" || {
  echo "missing executable CFS-00A CPU publisher: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
