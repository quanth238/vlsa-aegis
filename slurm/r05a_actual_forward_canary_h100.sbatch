#!/usr/bin/env bash
#SBATCH --job-name=crfs-r05a-af00a
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-0%1
#SBATCH --nodelist=worker-1
#SBATCH --no-requeue

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNNER=$REMOTE_REPO/scripts/hpc/run_r05a_actual_forward_canary.sh
test -x "$RUNNER" || { echo "missing executable AF-00A H100 wrapper: $RUNNER" >&2; exit 2; }
exec "$RUNNER"
