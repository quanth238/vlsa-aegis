#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-full-validate
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-full-validate-%j.out

set -euo pipefail

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-fast-feasibility"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-full-episode-feasibility"
REGISTERED_EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"
PROTOCOL_RELATIVE_PATH="configs/vlsa_poisson_full_episode_feasibility.v1.json"
EVALUATION_PYTHON="${EVALUATION_PYTHON:-${REGISTERED_EVALUATION_PYTHON}}"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${FULL_EPISODE_RESULT:?FULL_EPISODE_RESULT is required}"
: "${EXPECTED_PRODUCER_GIT_COMMIT:?EXPECTED_PRODUCER_GIT_COMMIT is required}"
: "${EXPECTED_CONSUMER_GIT_COMMIT:?EXPECTED_CONSUMER_GIT_COMMIT is required}"
: "${EXPECTED_PRODUCER_JOB_ID:?EXPECTED_PRODUCER_JOB_ID is required}"
: "${SLURM_JOB_ID:?independent consumer must run inside Slurm}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]]; then
  echo "REMOTE_REPO differs from the registered feasibility checkout" >&2
  exit 2
fi
for commit_variable in EXPECTED_PRODUCER_GIT_COMMIT EXPECTED_CONSUMER_GIT_COMMIT; do
  commit_value="${!commit_variable}"
  if [[ ! "${commit_value}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "${commit_variable} must be one lowercase 40-character commit" >&2
    exit 2
  fi
done
if [[ ! "${EXPECTED_PRODUCER_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "EXPECTED_PRODUCER_JOB_ID must identify one exact Slurm job" >&2
  exit 2
fi
if [[ "${SLURM_JOB_ID}" == "${EXPECTED_PRODUCER_JOB_ID}" ]]; then
  echo "producer and independent consumer must use distinct jobs" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
   (( SLURM_CPUS_PER_TASK < 1 || SLURM_CPUS_PER_TASK > 2 )); then
  echo "independent consumer permits one or two allocated CPUs" >&2
  exit 2
fi
if [[ "${SLURM_GPUS:-}" != "" && "${SLURM_GPUS:-}" != "0" && "${SLURM_GPUS:-}" != "(null)" ]] || \
   [[ "${SLURM_GPUS_ON_NODE:-}" != "" && "${SLURM_GPUS_ON_NODE:-}" != "0" && "${SLURM_GPUS_ON_NODE:-}" != "(null)" ]]; then
  echo "independent full-episode consumer must use a CPU-only allocation" >&2
  exit 2
fi
if [[ "${EVALUATION_PYTHON}" != "${REGISTERED_EVALUATION_PYTHON}" ]] || \
   [[ ! -x "${EVALUATION_PYTHON}" ]]; then
  echo "EVALUATION_PYTHON must resolve to the registered executable" >&2
  exit 2
fi

require_canonical_real_directory() {
  local label="$1"
  local path="$2"
  local resolved
  if [[ "${path}" != /* ]] || [[ "${path}" == *"/../"* ]] || \
     [[ "${path}" == */.. ]] || [[ "${path}" == *"/./"* ]]; then
    echo "${label} must be an absolute canonical path without traversal" >&2
    exit 2
  fi
  resolved="$(realpath -e -- "${path}")"
  if [[ "${resolved}" != "${path}" || ! -d "${path}" || -L "${path}" ]]; then
    echo "${label} must be one canonical real directory" >&2
    exit 2
  fi
}

require_canonical_real_file() {
  local label="$1"
  local path="$2"
  local resolved
  if [[ "${path}" != /* ]] || [[ "${path}" == *"/../"* ]] || \
     [[ "${path}" == */.. ]] || [[ "${path}" == *"/./"* ]]; then
    echo "${label} must be an absolute canonical path without traversal" >&2
    exit 2
  fi
  resolved="$(realpath -e -- "${path}")"
  if [[ "${resolved}" != "${path}" || ! -f "${path}" || -L "${path}" ]]; then
    echo "${label} must be one canonical nonsymlink regular file" >&2
    exit 2
  fi
}

require_canonical_real_directory "REMOTE_REPO" "${REMOTE_REPO}"
require_canonical_real_file "FULL_EPISODE_RESULT" "${FULL_EPISODE_RESULT}"
case "${FULL_EPISODE_RESULT}" in
  "${REGISTERED_RESULT_ROOT}"/*/result.json) ;;
  *)
    echo "FULL_EPISODE_RESULT must be one terminal result under the registered root" >&2
    exit 2
    ;;
esac

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]]; then
  echo "independent consumer requires a clean source tree" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_CONSUMER_GIT_COMMIT}" ]]; then
  echo "independent consumer source commit differs from EXPECTED_CONSUMER_GIT_COMMIT" >&2
  exit 2
fi
require_canonical_real_file \
  "full-episode protocol" \
  "${REMOTE_REPO}/${PROTOCOL_RELATIVE_PATH}"

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REMOTE_REPO}:${REMOTE_REPO}/main${PYTHONPATH:+:${PYTHONPATH}}"

exec "${EVALUATION_PYTHON}" \
  scripts/validate_poisson_full_episode_feasibility_artifact.py \
  --result "${FULL_EPISODE_RESULT}" \
  --protocol "${REMOTE_REPO}/${PROTOCOL_RELATIVE_PATH}" \
  --expected-commit "${EXPECTED_PRODUCER_GIT_COMMIT}" \
  --consumer-commit "${EXPECTED_CONSUMER_GIT_COMMIT}" \
  --expected-job-id "${EXPECTED_PRODUCER_JOB_ID}"
