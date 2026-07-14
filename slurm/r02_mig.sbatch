#!/usr/bin/env bash
#SBATCH --job-name=crfs-r02-smoke
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

RUNNER=$REMOTE_REPO/scripts/hpc/run_r02_case.sh
test -x "$RUNNER" || {
  echo "missing executable opt-in R02 allocation runner: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
