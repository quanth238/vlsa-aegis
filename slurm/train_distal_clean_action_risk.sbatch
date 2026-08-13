#!/usr/bin/env bash
#SBATCH --job-name=vlsa-train-clean-risk
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-clean-action-risk}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
VALIDATION_RUN_ID=${VALIDATION_RUN_ID:?VALIDATION_RUN_ID is required}
MODEL_RUN_ID=${MODEL_RUN_ID:?MODEL_RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-clean-action-risk}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "training requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || { echo "requires exactly one H100" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || { echo "source commit differs" >&2; exit 2; }
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || { echo "source tree is dirty" >&2; exit 2; }
MODEL_ROOT=$EXPERIMENT_ROOT/$MODEL_RUN_ID
mkdir "$MODEL_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=20260813
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" scripts/train_distal_clean_action_risk.py \
  --repo-root "$REMOTE_REPO" \
  --run-root "$EXPERIMENT_ROOT/$RUN_ID" \
  --validation "$EXPERIMENT_ROOT/$VALIDATION_RUN_ID/validation.json" \
  --config "$REMOTE_REPO/configs/vlsa_distal_clean_action_risk.v1.json" \
  --manifest "$REMOTE_REPO/manifests/vlsa_distal_clean_action_risk.v1.jsonl" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$MODEL_ROOT/result.json"
