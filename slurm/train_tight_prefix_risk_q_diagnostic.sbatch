#!/usr/bin/env bash
#SBATCH --job-name=vlsa-tight-prefix-q
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=worker-0,worker-3
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_tight_prefix_risk_q_diagnostic.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}

[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || exit 2
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
[[ ! -e "$OUTPUT" ]] || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests \
  -p 'test_tight_prefix_risk_q_diagnostic.py' -v
"$AEGIS_PYTHON" scripts/train_tight_prefix_risk_q_diagnostic.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$OUTPUT"
