#!/usr/bin/env bash
#SBATCH --job-name=crfs-r02
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# The submission wrapper supplies --array and caps aggregate use at two GPUs,
# 16 CPUs, and 256 GB of RAM.
set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/run_r02_case.sh
test -x "$RUNNER" || {
  echo "missing executable opt-in R02 allocation runner: $RUNNER" >&2
  exit 2
}
exec "$RUNNER"
