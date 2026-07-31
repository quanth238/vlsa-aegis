#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-active-canary
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-active-canary-%j.out

set -euo pipefail

EVALUATION_PYTHON="${EVALUATION_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}"
CASE_ID="${CASE_ID:-vlsa-t1-goal-ii-t0-e05}"
REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-feasibility"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-feasibility"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${HISTORICAL_RESULT_ROOT:?HISTORICAL_RESULT_ROOT is required}"
: "${CHECKPOINT_RECEIPT:?CHECKPOINT_RECEIPT is required}"
: "${NUMERIC_VALIDATION_RESULT:?NUMERIC_VALIDATION_RESULT is required}"
: "${PARITY_RESULT:?PARITY_RESULT is required}"
: "${SHADOW_IDENTIFICATION_RESULT:?SHADOW_IDENTIFICATION_RESULT is required}"

: "${SLURM_JOB_ID:?poisson active canary must run inside a Slurm allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]]; then
  echo "REMOTE_REPO differs from the registered Poisson-feasibility checkout" >&2
  exit 2
fi
if [[ "${RESULT_ROOT}" != "${REGISTERED_RESULT_ROOT}" ]]; then
  echo "RESULT_ROOT differs from the registered Poisson-feasibility artifact root" >&2
  exit 2
fi

if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || (( SLURM_CPUS_PER_TASK > 8 )); then
  echo "poisson active canary permits at most 8 allocated CPUs" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "poisson active canary requires exactly one visible GPU" >&2
  exit 2
fi

if [[ ! -d "${REMOTE_REPO}" || -L "${REMOTE_REPO}" ]]; then
  echo "REMOTE_REPO must be a real directory" >&2
  exit 2
fi
if [[ ! -x "${EVALUATION_PYTHON}" ]]; then
  echo "EVALUATION_PYTHON must resolve to an executable" >&2
  exit 2
fi
if [[ ! -d "${HISTORICAL_RESULT_ROOT}" || -L "${HISTORICAL_RESULT_ROOT}" ]]; then
  echo "HISTORICAL_RESULT_ROOT must be a real directory" >&2
  exit 2
fi

require_regular_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "${label} must be a real prerequisite artifact: ${path}" >&2
    exit 2
  fi
}

require_regular_file "CHECKPOINT_RECEIPT" "${CHECKPOINT_RECEIPT}"
require_regular_file "NUMERIC_VALIDATION_RESULT" "${NUMERIC_VALIDATION_RESULT}"
require_regular_file "PARITY_RESULT" "${PARITY_RESULT}"
require_regular_file "SHADOW_IDENTIFICATION_RESULT" "${SHADOW_IDENTIFICATION_RESULT}"

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]]; then
  echo "poisson active canary requires a clean source tree" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "poisson active canary source commit differs from EXPECTED_GIT_COMMIT" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"
export LIBERO_CONFIG_PATH="${REMOTE_REPO}/configs/libero_runtime"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${REMOTE_REPO}/openpi/src:${REMOTE_REPO}/openpi/packages/openpi-client/src:${REMOTE_REPO}:${REMOTE_REPO}/main:${REMOTE_REPO}/safelibero${PYTHONPATH:+:${PYTHONPATH}}"

exec "${EVALUATION_PYTHON}" scripts/run_poisson_active_canary.py \
  --repo-root "${REMOTE_REPO}" \
  --historical-result-root "${HISTORICAL_RESULT_ROOT}" \
  --checkpoint-receipt "${CHECKPOINT_RECEIPT}" \
  --numeric-validation-result "${NUMERIC_VALIDATION_RESULT}" \
  --parity-result "${PARITY_RESULT}" \
  --shadow-identification-result "${SHADOW_IDENTIFICATION_RESULT}" \
  --output-root "${RESULT_ROOT}" \
  --run-id "${RUN_ID}" \
  --case-id "${CASE_ID}"
