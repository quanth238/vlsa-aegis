#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-numerics
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-numerics-%j.out

set -euo pipefail

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${RESULT_PATH:?RESULT_PATH is required}"
: "${EVALUATION_PYTHON:?EVALUATION_PYTHON is required}"

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]]; then
  echo "numeric validation requires a clean source tree" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "numeric validation source commit differs from EXPECTED_GIT_COMMIT" >&2
  exit 2
fi

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export VLSA_POISSON_RUN_PRODUCTION_GRID_VALIDATION=1
"${EVALUATION_PYTHON}" scripts/run_poisson_numeric_validation.py \
  --repo-root "${REMOTE_REPO}" \
  --output "${RESULT_PATH}" \
  --minimum-tests 30
