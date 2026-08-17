#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-tight-prefix
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
PRODUCER_DIR=${PRODUCER_DIR:?PRODUCER_DIR is required}
REPLAY_DIR=${REPLAY_DIR:?REPLAY_DIR is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_tight_prefix_risk_dataset.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}

[[ -n "${SLURM_JOB_ID:-}" ]]
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]]
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]]
[[ ! -e "$OUTPUT" ]]
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" scripts/validate_tight_prefix_risk.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG_PATH" \
  --producer-dir "$PRODUCER_DIR" --replay-dir "$REPLAY_DIR" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$OUTPUT"
