#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-l5-aegis
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
RESULT_PATH=${RESULT_PATH:?RESULT_PATH is required}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?PRODUCER_COMMIT is required}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?VALIDATOR_COMMIT is required}
OUTPUT_PATH=${OUTPUT_PATH:?OUTPUT_PATH is required}
GROUPED_COLLECTION=${GROUPED_COLLECTION:-0}
PYTHON=${PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "validator requires Slurm" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$VALIDATOR_COMMIT" ]] || {
  echo "validator source differs" >&2; exit 2;
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "validator source is dirty" >&2; exit 2;
}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
extra_args=()
if [[ "$GROUPED_COLLECTION" == 1 ]]; then
  extra_args+=(--grouped-collection)
elif [[ "$GROUPED_COLLECTION" != 0 ]]; then
  echo "invalid grouped collection switch" >&2
  exit 2
fi
"$PYTHON" scripts/validate_distal_l5_aegis_consistent_risk.py \
  --repo-root "$REMOTE_REPO" --result "$RESULT_PATH" \
  --producer-commit "$PRODUCER_COMMIT" --validator-commit "$VALIDATOR_COMMIT" \
  --output "$OUTPUT_PATH" "${extra_args[@]}"
