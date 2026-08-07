#!/usr/bin/env bash
#SBATCH --job-name=vlsa-distal-3ell-verify
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --exclude=worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?VALIDATOR_COMMIT is required}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?PRODUCER_COMMIT is required}
CANDIDATE=${CANDIDATE:?CANDIDATE is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
ARCHIVED_TABLE1_RESULT=${ARCHIVED_TABLE1_RESULT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json}

[[ -n "${SLURM_JOB_ID:-}" ]] || exit 1
gpu_line=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 1
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$VALIDATOR_COMMIT" ]] || exit 1
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 1
[[ -f "$CANDIDATE" && ! -L "$CANDIDATE" && ! -e "$OUTPUT" ]] || exit 1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
exec "$AEGIS_PYTHON" scripts/validate_distal_three_ellipsoid_replay.py \
  --candidate "$CANDIDATE" \
  --archived "$ARCHIVED_TABLE1_RESULT" \
  --config "$REMOTE_REPO/configs/vlsa_distal_three_ellipsoid_shadow_e05.v2.json" \
  --repo-root "$REMOTE_REPO" \
  --producer-commit "$PRODUCER_COMMIT" \
  --validator-commit "$VALIDATOR_COMMIT" \
  --output "$OUTPUT"
