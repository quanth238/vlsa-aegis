#!/usr/bin/env bash
#SBATCH --job-name=vlsa-sum-l5-pop
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
PRODUCER_ROOT=${PRODUCER_ROOT:?PRODUCER_ROOT is required}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?PRODUCER_COMMIT is required}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?VALIDATOR_COMMIT is required}
SUMMARY_COMMIT=${SUMMARY_COMMIT:?SUMMARY_COMMIT is required}
OUTPUT_PATH=${OUTPUT_PATH:?OUTPUT_PATH is required}
PYTHON=${PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
ADAPTIVE=${ADAPTIVE:-0}
ADAPTIVE_V2=${ADAPTIVE_V2:-0}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "population summary requires Slurm" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$SUMMARY_COMMIT" ]] || {
  echo "population summary source differs" >&2; exit 2;
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "population summary source is dirty" >&2; exit 2;
}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
args=(
  --repo-root "$REMOTE_REPO" --producer-root "$PRODUCER_ROOT"
  --producer-commit "$PRODUCER_COMMIT"
  --validator-commit "$VALIDATOR_COMMIT"
  --summary-commit "$SUMMARY_COMMIT" --output "$OUTPUT_PATH"
)
if [[ "$ADAPTIVE" == 1 ]]; then
  args+=(--adaptive)
fi
if [[ "$ADAPTIVE_V2" == 1 ]]; then
  args+=(--adaptive-v2)
fi
"$PYTHON" scripts/summarize_distal_l5_aegis_grouped_population.py "${args[@]}"
