#!/usr/bin/env bash
#SBATCH --job-name=crfs-r02-summary
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
: "${SLURM_JOB_ID:?r02_summary.sbatch must execute inside a Slurm allocation}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${RUN_ID:?RUN_ID is required}"
: "${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is required}"
: "${R01_SUMMARY:?R01_SUMMARY is required}"
: "${CHECKPOINT_ID:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
: "${CHECKPOINT_SHA256:?CHECKPOINT_SHA256 is required}"
: "${MANIFEST:=$REMOTE_REPO/manifests/oracle_h05_colliding.jsonl}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${RESULTS_ROOT:=$EXPERIMENT_ROOT/$RUN_ID}"
: "${OUTPUT:=$RESULTS_ROOT/r03-summary.json}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"

test -f "$MANIFEST"
test -f "$EXPERIMENT_CONFIG"
test -f "$R01_SUMMARY"
test -d "$RESULTS_ROOT"
export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main
cd "$REMOTE_REPO"
"$LIBERO_PYTHON" main/summarize_r02.py \
  --manifest "$MANIFEST" \
  --config "$EXPERIMENT_CONFIG" \
  --r01-summary "$R01_SUMMARY" \
  --results-root "$RESULTS_ROOT" \
  --run-id "$RUN_ID" \
  --checkpoint-id "$CHECKPOINT_ID" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --output "$OUTPUT"
