#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-clean-model
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-clean-action-risk}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
PRODUCER_GIT_COMMIT=${PRODUCER_GIT_COMMIT:-$EXPECTED_GIT_COMMIT}
RUN_ID=${RUN_ID:?RUN_ID is required}
MODEL_RUN_ID=${MODEL_RUN_ID:?MODEL_RUN_ID is required}
MODEL_VALIDATION_RUN_ID=${MODEL_VALIDATION_RUN_ID:?MODEL_VALIDATION_RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-clean-action-risk}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "validation requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || { echo "requires exactly one H100" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || { echo "source commit differs" >&2; exit 2; }
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || { echo "source tree is dirty" >&2; exit 2; }
VALIDATION_ROOT=$EXPERIMENT_ROOT/$MODEL_VALIDATION_RUN_ID
mkdir "$VALIDATION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" scripts/validate_distal_clean_action_risk_model.py \
  --repo-root "$REMOTE_REPO" \
  --run-root "$EXPERIMENT_ROOT/$RUN_ID" \
  --model-result "$EXPERIMENT_ROOT/$MODEL_RUN_ID/result.json" \
  --config "$REMOTE_REPO/configs/vlsa_distal_clean_action_risk.v1.json" \
  --manifest "$REMOTE_REPO/manifests/vlsa_distal_clean_action_risk.v1.jsonl" \
  --producer-commit "$PRODUCER_GIT_COMMIT" \
  --validator-commit "$EXPECTED_GIT_COMMIT" \
  --output "$VALIDATION_ROOT/validation.json"
