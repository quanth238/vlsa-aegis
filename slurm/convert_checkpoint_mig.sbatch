#!/usr/bin/env bash
#SBATCH --job-name=crfs-convert-pi05
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:30:00

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
exec "$REMOTE_REPO/scripts/hpc/convert_checkpoint.sh"
