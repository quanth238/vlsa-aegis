#!/usr/bin/env bash
#SBATCH --job-name=crfs-r05a-runtime-id
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --nodelist=worker-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:02:00
#SBATCH --no-requeue

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
RUNNER=$REMOTE_REPO/scripts/hpc/run_r05a_runtime_identity_regression.sh
test -x "$RUNNER" || { echo "missing runtime-identity regression runner" >&2; exit 2; }
exec "$RUNNER"
