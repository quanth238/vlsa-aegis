#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-l5-diag
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
PRODUCER_GIT_COMMIT=${PRODUCER_GIT_COMMIT:?PRODUCER_GIT_COMMIT is required}
MODEL_RESULT=${MODEL_RESULT:?MODEL_RESULT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "L5 diagnostic validation requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || {
  echo "requires exactly one H100" >&2; exit 2;
}
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || {
  echo "L5 diagnostic validator source differs" >&2; exit 2;
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "L5 diagnostic validator source is dirty" >&2; exit 2;
}
test -f "$MODEL_RESULT" || { echo "L5 diagnostic model is missing" >&2; exit 2; }
test ! -e "$OUTPUT" || { echo "L5 diagnostic validation output exists" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_l5_action_risk_diagnostic -v
"$AEGIS_PYTHON" scripts/validate_distal_l5_action_risk_diagnostic.py \
  --repo-root "$REMOTE_REPO" --model-result "$MODEL_RESULT" \
  --config "$REMOTE_REPO/configs/vlsa_distal_l5_action_risk_diagnostic.v1.json" \
  --producer-commit "$PRODUCER_GIT_COMMIT" \
  --validator-commit "$EXPECTED_GIT_COMMIT" --output "$OUTPUT"
