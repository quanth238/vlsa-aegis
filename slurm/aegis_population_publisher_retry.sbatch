#!/usr/bin/env bash
#SBATCH --job-name=vlsa-aegis-publisher-retry
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
PUBLISHER_RELEASE_REPO=${PUBLISHER_RELEASE_REPO:-/home/quanth/working_space/vlsa-aegis-publication-retry}
exec "$PUBLISHER_RELEASE_REPO/slurm/run_aegis_population_publisher.sh"
