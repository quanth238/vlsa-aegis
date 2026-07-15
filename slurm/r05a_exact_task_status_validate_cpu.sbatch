#!/usr/bin/env bash
#SBATCH --job-name=crfs-r05a-task-validate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:02:00
#SBATCH --no-requeue

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
RUNNER=$REMOTE_REPO/scripts/hpc/validate_r05a_exact_task_status_regression.sh
test -x "$RUNNER" || { echo "missing exact-task validator runner" >&2; exit 2; }
exec "$RUNNER"
