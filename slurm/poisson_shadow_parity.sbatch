#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-shadow
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-shadow-%j.out

set -euo pipefail

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${RESULT_PATH:?RESULT_PATH is required}"
: "${EVALUATION_PYTHON:?EVALUATION_PYTHON is required}"
: "${HISTORICAL_RESULT_ROOT:?HISTORICAL_RESULT_ROOT is required}"

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]]; then
  echo "shadow parity requires a clean source tree" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "shadow parity source commit differs from EXPECTED_GIT_COMMIT" >&2
  exit 2
fi

mkdir -p "$(dirname "${RESULT_PATH}")"
export LIBERO_CONFIG_PATH="${REMOTE_REPO}/configs/libero_runtime"
export PYTHONPATH="${REMOTE_REPO}/openpi/src:${REMOTE_REPO}/openpi/packages/openpi-client/src:${REMOTE_REPO}/main:${REMOTE_REPO}/safelibero${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
"${EVALUATION_PYTHON}" scripts/run_poisson_shadow_parity.py \
  --repo-root "${REMOTE_REPO}" \
  --historical-result-root "${HISTORICAL_RESULT_ROOT}" \
  --output "${RESULT_PATH}"
