#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-full-episode
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-full-episode-%j.out

set -euo pipefail

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-fast-feasibility"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-full-episode-feasibility"
EVALUATION_PYTHON="${EVALUATION_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}"
CASE_ID="${CASE_ID:-vlsa-t1-goal-ii-t0-e05}"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${HISTORICAL_RESULT_ROOT:?HISTORICAL_RESULT_ROOT is required}"
: "${SLURM_JOB_ID:?full-episode feasibility must run inside Slurm}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]]; then
  echo "REMOTE_REPO differs from the registered feasibility checkout" >&2
  exit 2
fi
if [[ "${RESULT_ROOT}" != "${REGISTERED_RESULT_ROOT}" ]]; then
  echo "RESULT_ROOT differs from the registered full-episode artifact root" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "RUN_ID is not portable" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || (( SLURM_CPUS_PER_TASK > 8 )); then
  echo "full-episode feasibility permits at most 8 CPUs" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "full-episode feasibility requires exactly one visible GPU" >&2
  exit 2
fi
for directory in "${REMOTE_REPO}" "${RESULT_ROOT}" "${HISTORICAL_RESULT_ROOT}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "required directory is missing or symlinked: ${directory}" >&2
    exit 2
  fi
done
if [[ -e "${RESULT_ROOT}/${RUN_ID}" || -L "${RESULT_ROOT}/${RUN_ID}" ]]; then
  echo "run root already exists; use a new immutable RUN_ID" >&2
  exit 2
fi
if [[ ! -x "${EVALUATION_PYTHON}" ]]; then
  echo "EVALUATION_PYTHON is not executable" >&2
  exit 2
fi

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]]; then
  echo "full-episode feasibility requires a clean source tree" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "source commit differs from EXPECTED_GIT_COMMIT" >&2
  exit 2
fi

export LIBERO_CONFIG_PATH="${REMOTE_REPO}/configs/libero_runtime"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${REMOTE_REPO}/openpi/src:${REMOTE_REPO}/openpi/packages/openpi-client/src:${REMOTE_REPO}:${REMOTE_REPO}/main:${REMOTE_REPO}/safelibero${PYTHONPATH:+:${PYTHONPATH}}"

exec "${EVALUATION_PYTHON}" scripts/run_poisson_fast_feasibility.py \
  --repo-root "${REMOTE_REPO}" \
  --protocol configs/vlsa_poisson_full_episode_feasibility.v1.json \
  --manifest manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl \
  --historical-result-root "${HISTORICAL_RESULT_ROOT}" \
  --output-root "${RESULT_ROOT}" \
  --run-id "${RUN_ID}" \
  --case-id "${CASE_ID}"
