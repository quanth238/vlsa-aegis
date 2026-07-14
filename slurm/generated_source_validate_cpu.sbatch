#!/usr/bin/env bash
#SBATCH --job-name=crfs-generated-source-validate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/run_generated_source_validation.sh
test -x "$RUNNER" || {
  echo "missing executable generated-source validation runner: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
