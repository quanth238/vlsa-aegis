#!/usr/bin/env bash
#SBATCH --job-name=vlsa-sum-l5-target
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
PRODUCER_ROOT=${PRODUCER_ROOT:?PRODUCER_ROOT is required}
PARENT_SUMMARY=${PARENT_SUMMARY:?PARENT_SUMMARY is required}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?PRODUCER_COMMIT is required}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?VALIDATOR_COMMIT is required}
SUMMARY_COMMIT=${SUMMARY_COMMIT:?SUMMARY_COMMIT is required}
PYTHON=${PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "targeted summary requires Slurm" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$SUMMARY_COMMIT" ]] || {
  echo "targeted summary source differs" >&2; exit 2;
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "targeted summary source is dirty" >&2; exit 2;
}
args=()
for index in 0 1 2 3 4; do
  case_dir=$PRODUCER_ROOT/case-$(printf '%02d' "$index")
  args+=(--result "$case_dir/result.json" --validation "$case_dir/validation.json")
done
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$PYTHON" scripts/summarize_distal_l5_targeted_extension.py \
  --repo-root "$REMOTE_REPO" --parent-summary "$PARENT_SUMMARY" \
  "${args[@]}" --producer-commit "$PRODUCER_COMMIT" \
  --validator-commit "$VALIDATOR_COMMIT" --summary-commit "$SUMMARY_COMMIT" \
  --output "$PRODUCER_ROOT/summary.json"
