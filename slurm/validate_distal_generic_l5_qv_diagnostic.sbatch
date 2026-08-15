#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-l5-qv
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
REMOTE_REPO=${REMOTE_REPO:?}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
RESULT=${RESULT:?}
OUTPUT=${OUTPUT:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test -f "$RESULT" && test ! -e "$OUTPUT" || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_generic_l5_qv_diagnostic -v
"$AEGIS_PYTHON" scripts/validate_distal_generic_l5_qv_diagnostic.py \
  --repo-root "$REMOTE_REPO" \
  --config "$REMOTE_REPO/configs/vlsa_distal_generic_l5_qv_diagnostic.v1.json" \
  --result "$RESULT" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$OUTPUT"
