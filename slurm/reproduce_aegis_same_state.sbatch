#!/usr/bin/env bash
#SBATCH --job-name=aegis-same-state
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --nodelist=worker-1
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/aegis-baseline-repro/%x-%j.out

set -euo pipefail

: "${RUN_ID:?RUN_ID is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"

SOURCE=/home/quanth/working_space/vlsa-aegis-baseline-repro
PYTHON=/mnt/data/quanth/venvs/safety_vla/main/bin/python
OUTPUT=/mnt/data/quanth/experiments/aegis-baseline-repro/$RUN_ID

test "$(hostname -s)" = worker-1
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
test ! -e "$OUTPUT"
test -x "$PYTHON"

mkdir -p /mnt/data/quanth/experiments/aegis-baseline-repro

export PYTHONPATH="$SOURCE/main:$SOURCE/safelibero:$SOURCE/openpi/packages/openpi-client/src"
export LIBERO_CONFIG_PATH="$SOURCE/configs/libero_runtime"
export HF_HOME=/mnt/data/quanth/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/data/quanth/cache/huggingface/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

cd "$SOURCE"
exec "$PYTHON" main/reproduce_aegis_same_state.py \
  --config configs/aegis_same_state_canary.json \
  --output-root "$OUTPUT"
