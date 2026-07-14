#!/usr/bin/env bash
#SBATCH --job-name=crfs-generated-source-pilot
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-0%1

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/run_generated_source_pilot.sh
test -x "$RUNNER" || {
  echo "missing executable generated-source allocation runner: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
