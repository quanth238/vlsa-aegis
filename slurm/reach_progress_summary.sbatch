#!/usr/bin/env bash
#SBATCH --job-name=crfs-r00-summary
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${RUN_ID:?RUN_ID is required}"
: "${MANIFEST:=$REMOTE_REPO/manifests/reach_progress_calibration.jsonl}"
: "${EVALUATION_MANIFEST:=$REMOTE_REPO/manifests/oracle_h05_colliding.jsonl}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${OUTPUT:=$EXPERIMENT_ROOT/$RUN_ID/r00-summary.json}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"

export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main
cd "$REMOTE_REPO"
"$LIBERO_PYTHON" main/summarize_reach_progress.py \
  --manifest "$MANIFEST" \
  --evaluation-manifest "$EVALUATION_MANIFEST" \
  --results-root "$EXPERIMENT_ROOT/$RUN_ID" \
  --output "$OUTPUT" \
  --minimum-positive-examples 50
