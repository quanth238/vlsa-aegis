#!/usr/bin/env bash
#SBATCH --job-name=crfs-measurement-calibration
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"
: "${OUTPUT:=/mnt/data/quanth/experiments/crfs-oracle/measurement-calibration-${SLURM_JOB_ID}.json}"

export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main
cd "$REMOTE_REPO"
echo "host=$(hostname)"
echo "commit=$(git rev-parse HEAD)"
"$LIBERO_PYTHON" main/audit_mujoco_sphere_box.py --samples 261 --output "$OUTPUT"
echo "output=$OUTPUT"
