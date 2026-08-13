#!/usr/bin/env bash
#SBATCH --job-name=vlsa-group-risk-val
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_PRODUCER_COMMIT=${EXPECTED_PRODUCER_COMMIT:?EXPECTED_PRODUCER_COMMIT is required}
EXPECTED_VALIDATION_COMMIT=${EXPECTED_VALIDATION_COMMIT:?EXPECTED_VALIDATION_COMMIT is required}
PRODUCER_ROOT=${PRODUCER_ROOT:?PRODUCER_ROOT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
FREEZE_ROOT=${FREEZE_ROOT:?FREEZE_ROOT is required}
COVERAGE_ROOT=${COVERAGE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-query-boundary-coverage/query-coverage-population-20260814a}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "grouped risk validation requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || { echo "requires exactly one H100" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_VALIDATION_COMMIT" ]] || { echo "validation source differs" >&2; exit 2; }
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || { echo "validation source is dirty" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_grouped_query_action_risk tests.test_distal_query_action_risk -v
"$AEGIS_PYTHON" scripts/validate_distal_grouped_query_action_risk.py \
  --repo-root "$REMOTE_REPO" --producer-root "$PRODUCER_ROOT" \
  --coverage-root "$COVERAGE_ROOT" \
  --grouped-config "$REMOTE_REPO/configs/vlsa_distal_grouped_query_action_risk.v1.json" \
  --base-config "$REMOTE_REPO/configs/vlsa_distal_query_action_risk_e05.v1.json" \
  --freeze-config "$REMOTE_REPO/configs/vlsa_distal_grouped_query_action_risk_freeze.v1.json" \
  --selection-manifest "$REMOTE_REPO/manifests/vlsa_distal_clean_action_risk.v1.jsonl" \
  --table1-root "/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a" \
  --expected-producer-commit "$EXPECTED_PRODUCER_COMMIT" \
  --expected-validation-commit "$EXPECTED_VALIDATION_COMMIT" \
  --freeze-root "$FREEZE_ROOT" --output "$OUTPUT"
