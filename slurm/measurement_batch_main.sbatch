#!/usr/bin/env bash
#SBATCH --job-name=crfs-h03-batch
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${CASE_START:?CASE_START is required}"
: "${CASE_END:?CASE_END is required}"
exec "$REMOTE_REPO/scripts/hpc/run_oracle_case.sh"
