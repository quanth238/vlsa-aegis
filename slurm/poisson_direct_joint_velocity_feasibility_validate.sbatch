#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-direct-jv-validate
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-direct-jv-validate-%j.out

# Read-only, allocation-backed consumer for one completed direct-JV producer.
# It requests no GPU and publishes no experiment file; the validated JSON
# summary is captured in this consumer's distinct Slurm log.

set -euo pipefail

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-arm-link-study"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-direct-jv"
REGISTERED_EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"

EVALUATION_PYTHON="${EVALUATION_PYTHON:-${REGISTERED_EVALUATION_PYTHON}}"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${CASE_ID:?CASE_ID is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${EXPECTED_PRODUCER_JOB_ID:?EXPECTED_PRODUCER_JOB_ID is required}"
: "${SLURM_JOB_ID:?consumer must run in its own Slurm allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]] || \
   [[ "${RESULT_ROOT}" != "${REGISTERED_RESULT_ROOT}" ]]; then
  echo "consumer source or result root differs from registration" >&2
  exit 2
fi
if [[ "${EVALUATION_PYTHON}" != "${REGISTERED_EVALUATION_PYTHON}" ]] || \
   [[ ! -x "${EVALUATION_PYTHON}" ]]; then
  echo "consumer Python runtime differs or is not executable" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
   [[ ! "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
   [[ ! "${EXPECTED_PRODUCER_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "RUN_ID, commit, or producer job ID is malformed" >&2
  exit 2
fi
case "${CASE_ID}" in
  vlsa-t1-goal-ii-t0-e05|vlsa-t1-goal-ii-t3-e42) ;;
  *)
    echo "CASE_ID is outside the frozen two-case direct experiment" >&2
    exit 2
    ;;
esac
if [[ "${SLURM_JOB_ID}" == "${EXPECTED_PRODUCER_JOB_ID}" ]]; then
  echo "consumer and producer must be distinct Slurm jobs" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
   (( SLURM_CPUS_PER_TASK < 1 || SLURM_CPUS_PER_TASK > 2 )); then
  echo "consumer permits one or two allocated CPUs" >&2
  exit 2
fi
if [[ "${SLURM_GPUS:-}" != "" && "${SLURM_GPUS:-}" != "0" && \
      "${SLURM_GPUS:-}" != "(null)" ]] || \
   [[ "${SLURM_GPUS_ON_NODE:-}" != "" && \
      "${SLURM_GPUS_ON_NODE:-}" != "0" && \
      "${SLURM_GPUS_ON_NODE:-}" != "(null)" ]]; then
  echo "independent consumer must be CPU-only" >&2
  exit 2
fi

RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
RESULT_PATH="${RUN_ROOT}/result.json"
for directory in "${REMOTE_REPO}" "${RESULT_ROOT}" "${RUN_ROOT}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "required directory is missing or symlinked: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -f "${RESULT_PATH}" || -L "${RESULT_PATH}" ]]; then
  echo "complete producer result must be a nonsymlink regular file" >&2
  exit 2
fi

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]] || \
   [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "consumer requires the exact clean producer source commit" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REMOTE_REPO}:${REMOTE_REPO}/main${PYTHONPATH:+:${PYTHONPATH}}"

exec "${EVALUATION_PYTHON}" \
  scripts/validate_poisson_direct_joint_velocity_feasibility_artifact.py \
  --result "${RESULT_PATH}" \
  --expected-commit "${EXPECTED_GIT_COMMIT}" \
  --expected-job-id "${EXPECTED_PRODUCER_JOB_ID}"
