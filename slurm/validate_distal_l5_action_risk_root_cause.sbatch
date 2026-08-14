#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-l5-root
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
CONFIG=${CONFIG:?CONFIG is required}
MODEL_RESULT=${MODEL_RESULT:?MODEL_RESULT is required}
AUDIT_RESULT=${AUDIT_RESULT:?AUDIT_RESULT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "root-cause validation requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test ! -e "$OUTPUT" || { echo "root-cause validation exists" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests \
  -p 'test_distal_l5_action_risk_root_cause.py' -v
"$AEGIS_PYTHON" scripts/validate_distal_l5_action_risk_root_cause.py \
  --repo-root "$REMOTE_REPO" --expected-commit "$EXPECTED_GIT_COMMIT" \
  --config "$CONFIG" --model-result "$MODEL_RESULT" \
  --audit-result "$AUDIT_RESULT" --output "$OUTPUT"
