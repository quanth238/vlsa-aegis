#!/usr/bin/env bash
#SBATCH --job-name=vlsa-train-l5-risk
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
DATASET_AUDIT=${DATASET_AUDIT:?DATASET_AUDIT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "L5 risk training requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || { echo "requires exactly one H100" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || { echo "source differs" >&2; exit 2; }
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || { echo "source is dirty" >&2; exit 2; }
test ! -e "$OUTPUT" || { echo "output exists" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover \
  -s tests -p 'test_distal_l5_action_risk.py' -v
"$AEGIS_PYTHON" scripts/train_distal_l5_action_risk.py \
  --repo-root "$REMOTE_REPO" \
  --config "$REMOTE_REPO/configs/vlsa_distal_l5_action_risk_mlp.v1.json" \
  --dataset-audit "$DATASET_AUDIT" --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$OUTPUT"
