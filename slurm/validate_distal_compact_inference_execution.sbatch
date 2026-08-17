#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-q7-exec
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out
set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}; EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
PRODUCER_DIR=${PRODUCER_DIR:?}; REPLAY_DIR=${REPLAY_DIR:?}; OUTPUT=${OUTPUT:?}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_distal_compact_inference_execution.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test -d "$PRODUCER_DIR"; test -d "$REPLAY_DIR"; test ! -e "$OUTPUT" || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_compact_inference_execution -v
"$AEGIS_PYTHON" -m scripts.validate_distal_compact_inference_execution \
  --repo-root "$REMOTE_REPO" --config "$CONFIG_PATH" \
  --producer-dir "$PRODUCER_DIR" --replay-dir "$REPLAY_DIR" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$OUTPUT"
