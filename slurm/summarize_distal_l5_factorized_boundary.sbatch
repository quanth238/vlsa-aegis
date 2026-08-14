#!/usr/bin/env bash
#SBATCH --job-name=vlsa-sum-l5-factor
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}
PRODUCER_ROOT=${PRODUCER_ROOT:?}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?}
SUMMARY_COMMIT=${SUMMARY_COMMIT:?}
OUTPUT=${OUTPUT:?}
PYTHON=${PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$SUMMARY_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test ! -e "$OUTPUT" || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$PYTHON" scripts/summarize_distal_l5_factorized_boundary.py \
  --repo-root "$REMOTE_REPO" \
  --config "$REMOTE_REPO/configs/vlsa_distal_l5_factorized_boundary.v1.json" \
  --manifest "$REMOTE_REPO/manifests/vlsa_distal_l5_factorized_boundary.v1.jsonl" \
  --producer-root "$PRODUCER_ROOT" --producer-commit "$PRODUCER_COMMIT" \
  --validator-commit "$VALIDATOR_COMMIT" --summary-commit "$SUMMARY_COMMIT" \
  --output "$OUTPUT"
