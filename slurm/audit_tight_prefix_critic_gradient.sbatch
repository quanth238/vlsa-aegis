#!/usr/bin/env bash
#SBATCH --job-name=vlsa-tight-prefix-grad
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_tight_prefix_critic_gradient_audit.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}

[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
[[ ! -e "$OUTPUT" ]] || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests \
  -p 'test_tight_prefix_critic_gradient_audit.py' -v
"$AEGIS_PYTHON" scripts/audit_tight_prefix_critic_gradient.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$OUTPUT"
