#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-warning
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_PRODUCER_COMMIT=${EXPECTED_PRODUCER_COMMIT:?EXPECTED_PRODUCER_COMMIT is required}
EXPECTED_VALIDATION_COMMIT=${EXPECTED_VALIDATION_COMMIT:?EXPECTED_VALIDATION_COMMIT is required}
PRODUCER_ROOT=${PRODUCER_ROOT:?PRODUCER_ROOT is required}
VALIDATION_ROOT=${VALIDATION_ROOT:?VALIDATION_ROOT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "validation requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || { echo "requires exactly one H100" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_VALIDATION_COMMIT" ]] || { echo "validation source differs" >&2; exit 2; }
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || { echo "validation source is dirty" >&2; exit 2; }
mkdir "$VALIDATION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_clean_action_risk -v >"$VALIDATION_ROOT/evaluation-preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/validate_distal_clean_warning_states.py \
  --repo-root "$REMOTE_REPO" --producer-root "$PRODUCER_ROOT" \
  --selection-manifest "$REMOTE_REPO/manifests/vlsa_distal_clean_action_risk.v1.jsonl" \
  --experiment-config "$REMOTE_REPO/configs/vlsa_distal_clean_action_risk.v1.json" \
  --expected-producer-commit "$EXPECTED_PRODUCER_COMMIT" \
  --expected-validation-commit "$EXPECTED_VALIDATION_COMMIT" \
  --output "$VALIDATION_ROOT/validation.json"
