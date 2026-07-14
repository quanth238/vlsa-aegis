#!/usr/bin/env bash
#SBATCH --job-name=crfs-sampler-parity
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail

: "${SLURM_JOB_ID:?sampler parity must execute inside Slurm}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${OPENPI_PYTHON:=/mnt/data/quanth/venvs/openpi/bin/python}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"
: "${JAX_CHECKPOINT:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero}"
: "${PYTORCH_CHECKPOINT:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${RUN_ID:?RUN_ID is required}"

MANIFEST=$REMOTE_REPO/manifests/oracle_h05_colliding.jsonl
EXPERIMENT_CONFIG=$REMOTE_REPO/configs/experiments/endpoint_free_reach_h5.json
RUNNER=$REMOTE_REPO/main/run_sampler_parity.py
OUTPUT_DIR=$EXPERIMENT_ROOT/$RUN_ID
OUTPUT=$OUTPUT_DIR/sampler-parity.json

test -f "$RUNNER"
test -f "$MANIFEST"
test -f "$EXPERIMENT_CONFIG"
test -d "$JAX_CHECKPOINT/params"
test -f "$JAX_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json"
test -f "$PYTORCH_CHECKPOINT/model.safetensors"
test -f "$PYTORCH_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json"
test -x "$OPENPI_PYTHON"
test -x "$LIBERO_PYTHON"

mkdir -p "$OUTPUT_DIR"
export OPENPI_DATA_HOME=/mnt/data/quanth/cache/openpi
export HF_HOME=/mnt/data/quanth/cache/huggingface
export XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
export UV_CACHE_DIR=/mnt/data/quanth/cache/uv
export TORCHINDUCTOR_CACHE_DIR=/mnt/data/quanth/cache/torchinductor
export TRITON_CACHE_DIR=/mnt/data/quanth/cache/triton
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH=$OUTPUT_DIR/libero-config
mkdir -p "$LIBERO_CONFIG_PATH"

SAFELIBERO_ROOT=$REMOTE_REPO/safelibero/libero/libero
"$LIBERO_PYTHON" - "$LIBERO_CONFIG_PATH/config.yaml" "$SAFELIBERO_ROOT" <<'PY'
import os
import pathlib
import sys
import tempfile

destination = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2]).resolve()
paths = {
    "benchmark_root": root,
    "bddl_files": root / "bddl_files",
    "init_states": root / "init_files",
    "datasets": root.parent / "datasets",
    "assets": root / "assets",
}
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as handle:
    for key, value in paths.items():
        handle.write(f"{key}: {value}\n")
    temporary = handle.name
os.replace(temporary, destination)
PY

TRANSFORMERS_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh)
export PYTHONPATH=$TRANSFORMERS_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "commit=$(git -C "$REMOTE_REPO" rev-parse HEAD)"
echo "dirty=$(test -n "$(git -C "$REMOTE_REPO" status --porcelain)" && echo true || echo false)"
echo "output=$OUTPUT"

cd "$REMOTE_REPO"
"$OPENPI_PYTHON" "$RUNNER" \
  --jax-checkpoint "$JAX_CHECKPOINT" \
  --pytorch-checkpoint "$PYTORCH_CHECKPOINT" \
  --config-name pi05_libero \
  --manifest "$MANIFEST" \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --case-index 0 \
  --noise-seed 1179198633 \
  --libero-python "$LIBERO_PYTHON" \
  --step-x-max 0.10 \
  --step-x-rms 0.025 \
  --step-v-max 0.20 \
  --step-v-rms 0.050 \
  --final-model-max 0.10 \
  --final-model-rms 0.025 \
  --physical-xyz5-max 0.010 \
  --physical-xyz5-rms 0.005 \
  --physical-action7-max 0.050 \
  --physical-action7-rms 0.015 \
  --output "$OUTPUT"
