#!/usr/bin/env bash
#SBATCH --job-name=vlsa-a3-p28610
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail

# Submit only through the reviewed held-before-release v3 helper. The exact
# analysis-v2 dependency and all terminal input hashes are exported there.
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
exec "$REMOTE_REPO/slurm/run_vlsa_postpublication_analysis_v3.sh"
