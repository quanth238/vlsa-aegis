#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-triggered-rescue-validate
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-triggered-rescue-validate-%j.out

set -euo pipefail

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-arm-link-study"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-triggered-rescue"
REGISTERED_HISTORICAL_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a"
REGISTERED_EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${HISTORICAL_RESULT_ROOT:?HISTORICAL_RESULT_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${CASE_ID:?CASE_ID is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${EXPECTED_PRODUCER_JOB_ID:?EXPECTED_PRODUCER_JOB_ID is required}"
: "${EXPECTED_PRODUCER_HOST:?EXPECTED_PRODUCER_HOST is required}"
: "${SLURM_JOB_ID:?independent validation requires a distinct Slurm job}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]] || \
   [[ "${RESULT_ROOT}" != "${REGISTERED_RESULT_ROOT}" ]] || \
   [[ "${HISTORICAL_RESULT_ROOT}" != "${REGISTERED_HISTORICAL_ROOT}" ]]; then
  echo "independent-consumer source or artifact root differs" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
   [[ ! "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
   [[ ! "${EXPECTED_PRODUCER_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "consumer identity input is malformed" >&2
  exit 2
fi
if [[ "${SLURM_JOB_ID}" == "${EXPECTED_PRODUCER_JOB_ID}" ]]; then
  echo "consumer must use a different Slurm job from the producer" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
   (( SLURM_CPUS_PER_TASK > 2 )); then
  echo "independent consumer permits at most two CPUs" >&2
  exit 2
fi
case "${CASE_ID}" in
  vlsa-t1-goal-ii-t0-e05)
    HISTORICAL_RELATIVE_PATH="tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json"
    ;;
  vlsa-t1-goal-ii-t3-e42)
    HISTORICAL_RELATIVE_PATH="tasks/task-15/results/aegis/vlsa-t1-goal-ii-t3-e42/result.json"
    ;;
  *)
    echo "CASE_ID is outside the registered two-case study" >&2
    exit 2
    ;;
esac

RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
RESULT_PATH="${RUN_ROOT}/result.json"
RECEIPT_PATH="${RUN_ROOT}/validation_receipt.json"
HISTORICAL_RESULT_PATH="${HISTORICAL_RESULT_ROOT}/${HISTORICAL_RELATIVE_PATH}"
for directory in "${REMOTE_REPO}" "${RESULT_ROOT}" "${RUN_ROOT}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "required directory is missing or symlinked: ${directory}" >&2
    exit 2
  fi
done
for path in \
  "${RESULT_PATH}" \
  "${HISTORICAL_RESULT_PATH}" \
  "${REMOTE_REPO}/configs/vlsa_poisson_triggered_rescue.v1.json" \
  "${REMOTE_REPO}/configs/vlsa_poisson_runtime_protocol.canary.v3.json"
do
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "required file is missing or symlinked: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${RECEIPT_PATH}" || -L "${RECEIPT_PATH}" ]]; then
  echo "validation receipt already exists; preserve the immutable run" >&2
  exit 2
fi
if [[ ! -x "${REGISTERED_EVALUATION_PYTHON}" ]]; then
  echo "registered evaluation Python is not executable" >&2
  exit 2
fi

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]] || \
   [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "independent consumer requires the exact clean source commit" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REMOTE_REPO}:${REMOTE_REPO}/main${PYTHONPATH:+:${PYTHONPATH}}"

exec "${REGISTERED_EVALUATION_PYTHON}" \
  scripts/validate_poisson_triggered_rescue_artifact.py \
  --result "${RESULT_PATH}" \
  --receipt "${RECEIPT_PATH}" \
  --protocol configs/vlsa_poisson_triggered_rescue.v1.json \
  --runtime-protocol configs/vlsa_poisson_runtime_protocol.canary.v3.json \
  --historical-result "${HISTORICAL_RESULT_PATH}" \
  --expected-code-commit "${EXPECTED_GIT_COMMIT}" \
  --expected-run-id "${RUN_ID}" \
  --expected-slurm-job-id "${EXPECTED_PRODUCER_JOB_ID}" \
  --expected-host "${EXPECTED_PRODUCER_HOST}"
