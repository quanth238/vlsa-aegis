#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?checkpoint conversion must execute inside Slurm}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${OPENPI_PYTHON:=/mnt/data/quanth/venvs/openpi/bin/python}"
: "${SOURCE_CHECKPOINT:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero}"
: "${CHECKPOINT_DIR:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"

MODEL=$CHECKPOINT_DIR/model.safetensors
if [ -f "$MODEL" ] && [ -f "$CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json" ]; then
  echo "valid converted checkpoint already exists"
  sha256sum "$MODEL"
  exit 0
fi
test ! -e "$CHECKPOINT_DIR" || {
  echo "partial checkpoint destination exists; inspect it before cleanup: $CHECKPOINT_DIR" >&2
  exit 2
}
TMP=$CHECKPOINT_DIR.incomplete-$SLURM_JOB_ID
test ! -e "$TMP" || { echo "temporary conversion path already exists: $TMP" >&2; exit 2; }

export OPENPI_DATA_HOME=/mnt/data/quanth/cache/openpi
export HF_HOME=/mnt/data/quanth/cache/huggingface
export XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
export UV_CACHE_DIR=/mnt/data/quanth/cache/uv
export PYTHONUNBUFFERED=1
TRANSFORMERS_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh)
export PYTHONPATH=$TRANSFORMERS_OVERLAY:$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "source_commit=$(git -C "$REMOTE_REPO" rev-parse HEAD)"
echo "source_checkpoint=$SOURCE_CHECKPOINT"
echo "destination=$CHECKPOINT_DIR"
df -h /mnt/data

cd "$REMOTE_REPO/openpi"
"$OPENPI_PYTHON" examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir "$SOURCE_CHECKPOINT" \
  --config-name pi05_libero \
  --output-path "$TMP" \
  --precision bfloat16

test -s "$TMP/model.safetensors"
test -f "$TMP/assets/physical-intelligence/libero/norm_stats.json"
mv "$TMP" "$CHECKPOINT_DIR"
sha256sum "$MODEL" | tee "$CHECKPOINT_DIR/model.safetensors.sha256"
