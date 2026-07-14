#!/usr/bin/env bash
#SBATCH --job-name=crfs-h04-calibration
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=00:30:00

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${OUTPUT:=/mnt/data/quanth/experiments/crfs-oracle/h04-calibration-${SLURM_JOB_ID}.json}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH=/mnt/data/quanth/experiments/crfs-oracle/h04-libero-config-$SLURM_JOB_ID
mkdir -p "$LIBERO_CONFIG_PATH"
cp /mnt/data/quanth/experiments/crfs-oracle/h03-population-20260714a/batch-25-49/libero-config/config.yaml "$LIBERO_CONFIG_PATH/config.yaml"
export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero
cd "$REMOTE_REPO"
/mnt/data/quanth/venvs/openpi-libero-client/bin/python main/run_h04_calibration.py \
  --manifest manifests/measurement_audit.jsonl \
  --config configs/experiments/measurement_audit.json \
  --output "$OUTPUT"
