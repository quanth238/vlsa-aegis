#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-l5-pop
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-14%4
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
PRODUCER_ROOT=${PRODUCER_ROOT:?PRODUCER_ROOT is required}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?PRODUCER_COMMIT is required}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?VALIDATOR_COMMIT is required}
PYTHON=${PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "population validation requires Slurm" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$VALIDATOR_COMMIT" ]] || {
  echo "population validator source differs" >&2; exit 2;
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "population validator source is dirty" >&2; exit 2;
}
case_dir=$PRODUCER_ROOT/case-$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$PYTHON" scripts/validate_distal_l5_aegis_consistent_risk.py \
  --repo-root "$REMOTE_REPO" --result "$case_dir/result.json" \
  --producer-commit "$PRODUCER_COMMIT" --validator-commit "$VALIDATOR_COMMIT" \
  --output "$case_dir/validation.json" --grouped-collection
