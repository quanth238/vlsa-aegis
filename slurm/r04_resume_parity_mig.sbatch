#!/usr/bin/env bash
#SBATCH --job-name=crfs-r04b-resume-parity
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

RUNNER=$REMOTE_REPO/scripts/hpc/run_r04_resume_parity.sh
test -x "$RUNNER" || {
  echo "missing executable opt-in R04B allocation runner: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
