#!/usr/bin/env bash
#SBATCH --job-name=crfs-r01-summary
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${RUN_ID:?RUN_ID is required}"
: "${CONFIG_HASH:?CONFIG_HASH is required}"
: "${CHECKPOINT_SHA256:?CHECKPOINT_SHA256 is required}"
: "${R00_SUMMARY_SHA256:?R00_SUMMARY_SHA256 is required}"
: "${MANIFEST:=$REMOTE_REPO/manifests/oracle_h05_colliding.jsonl}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${RESULTS_ROOT:=$EXPERIMENT_ROOT/$RUN_ID}"
: "${OUTPUT:=$RESULTS_ROOT/r01-summary.json}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"

export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main
cd "$REMOTE_REPO"
"$LIBERO_PYTHON" main/summarize_endpoint_free.py \
  --manifest "$MANIFEST" \
  --results-root "$RESULTS_ROOT" \
  --run-id "$RUN_ID" \
  --config-hash "$CONFIG_HASH" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --r00-summary-sha256 "$R00_SUMMARY_SHA256" \
  --output "$OUTPUT"
