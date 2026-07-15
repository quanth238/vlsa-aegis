#!/usr/bin/env bash
#SBATCH --job-name=crfs-r05a-cgroup-current
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:02:00
#SBATCH --array=0-0%1
#SBATCH --no-requeue

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
RUNNER=$REMOTE_REPO/scripts/hpc/run_r05a_cgroup_v2_current_capability.sh
test -x "$RUNNER" || { echo "missing executable capability runner" >&2; exit 2; }
exec "$RUNNER"
