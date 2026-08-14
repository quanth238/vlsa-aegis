#!/usr/bin/env bash
#SBATCH --job-name=vlsa-sum-l5-moka-adapt
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
PRODUCER_ROOT=${PRODUCER_ROOT:?}
BOUNDARY_ROOT=${BOUNDARY_ROOT:?}
DISCOVERY_ROOT=${DISCOVERY_ROOT:?}
PARENT_SUMMARY=${PARENT_SUMMARY:?}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?}
OUTPUT=${OUTPUT:?}
PYTHON=${PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || {
  echo "requires exactly one H100" >&2; exit 2;
}
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]]
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]]
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$PYTHON" scripts/summarize_distal_l5_moka_noise_adaptive.py \
  --repo-root "$REMOTE_REPO" --producer-root "$PRODUCER_ROOT" \
  --boundary-root "$BOUNDARY_ROOT" --discovery-root "$DISCOVERY_ROOT" \
  --parent-summary "$PARENT_SUMMARY" \
  --config "$REMOTE_REPO/configs/vlsa_distal_l5_moka_noise_adaptive_summary.v1.json" \
  --manifest "$REMOTE_REPO/manifests/vlsa_distal_l5_moka_noise_boundary.v1.jsonl" \
  --producer-commit "$PRODUCER_COMMIT" --validator-commit "$VALIDATOR_COMMIT" \
  --summary-commit "$EXPECTED_GIT_COMMIT" --output "$OUTPUT"
