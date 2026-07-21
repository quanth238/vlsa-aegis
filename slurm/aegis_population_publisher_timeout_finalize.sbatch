#!/usr/bin/env bash
#SBATCH --job-name=vlsa-aegis-finalize-recovery
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
RECOVERY_RELEASE_REPO=${RECOVERY_RELEASE_REPO:-/home/quanth/working_space/vlsa-aegis-publication-timeout-finalize}
exec "$RECOVERY_RELEASE_REPO/slurm/run_aegis_population_publisher_timeout_finalize.sh"
