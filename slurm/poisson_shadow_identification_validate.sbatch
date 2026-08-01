#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-identify-validate
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-identify-validate-%j.out

# Submit only after the exact producer has reached a terminal state, normally
# with --dependency=afterany:<producer-job-id>.  This consumer requests no GPU
# and writes no experiment artifact; Slurm captures its one JSON report.

set -euo pipefail

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-feasibility"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-feasibility"
REGISTERED_HISTORICAL_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a"
REGISTERED_EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"
MANIFEST_RELATIVE_PATH="manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"
SELECTION_RELATIVE_PATH="configs/vlsa_poisson_link56_feasibility.v1.json"
RUNTIME_RELATIVE_PATH="configs/vlsa_poisson_runtime_protocol.canary.v2.json"

EVALUATION_PYTHON="${EVALUATION_PYTHON:-${REGISTERED_EVALUATION_PYTHON}}"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${IDENTIFICATION_RESULT_PATH:?IDENTIFICATION_RESULT_PATH is required}"
: "${NUMERIC_VALIDATION_RESULT:?NUMERIC_VALIDATION_RESULT is required}"
: "${PARITY_RESULT_PATH:?PARITY_RESULT_PATH is required}"
: "${HISTORICAL_RESULT_ROOT:?HISTORICAL_RESULT_ROOT is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${EXPECTED_PRODUCER_JOB_ID:?EXPECTED_PRODUCER_JOB_ID is required}"
: "${EXPECTED_PRODUCER_HOST:?EXPECTED_PRODUCER_HOST is required}"
: "${EXPECTED_PRODUCER_DEVICE:?EXPECTED_PRODUCER_DEVICE is required}"
: "${EXPECTED_NUMERIC_JOB_ID:?EXPECTED_NUMERIC_JOB_ID is required}"
: "${EXPECTED_NUMERIC_HOST:?EXPECTED_NUMERIC_HOST is required}"
: "${EXPECTED_NUMERIC_DEVICE:?EXPECTED_NUMERIC_DEVICE is required}"
: "${EXPECTED_PARITY_JOB_ID:?EXPECTED_PARITY_JOB_ID is required}"
: "${EXPECTED_PARITY_HOST:?EXPECTED_PARITY_HOST is required}"
: "${EXPECTED_PARITY_DEVICE:?EXPECTED_PARITY_DEVICE is required}"

: "${SLURM_JOB_ID:?independent consumer must run inside Slurm}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]]; then
  echo "REMOTE_REPO differs from the registered Poisson checkout" >&2
  exit 2
fi
if [[ "${HISTORICAL_RESULT_ROOT}" != "${REGISTERED_HISTORICAL_ROOT}" ]]; then
  echo "HISTORICAL_RESULT_ROOT differs from the registered Table-1 run" >&2
  exit 2
fi
if [[ ! "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_GIT_COMMIT must be one lowercase 40-character commit" >&2
  exit 2
fi
if [[ ! "${EXPECTED_PRODUCER_JOB_ID}" =~ ^[0-9]+$ ]] || \
   [[ ! "${EXPECTED_NUMERIC_JOB_ID}" =~ ^[0-9]+$ ]] || \
   [[ ! "${EXPECTED_PARITY_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "all expected prerequisite IDs must identify exact Slurm jobs" >&2
  exit 2
fi
if [[ "${SLURM_JOB_ID}" == "${EXPECTED_PRODUCER_JOB_ID}" ]] || \
   [[ "${SLURM_JOB_ID}" == "${EXPECTED_NUMERIC_JOB_ID}" ]] || \
   [[ "${SLURM_JOB_ID}" == "${EXPECTED_PARITY_JOB_ID}" ]] || \
   [[ "${EXPECTED_PRODUCER_JOB_ID}" == "${EXPECTED_NUMERIC_JOB_ID}" ]] || \
   [[ "${EXPECTED_PRODUCER_JOB_ID}" == "${EXPECTED_PARITY_JOB_ID}" ]] || \
   [[ "${EXPECTED_NUMERIC_JOB_ID}" == "${EXPECTED_PARITY_JOB_ID}" ]]; then
  echo "consumer, numeric, parity, and producer must use distinct Slurm jobs" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || (( SLURM_CPUS_PER_TASK < 1 || SLURM_CPUS_PER_TASK > 2 )); then
  echo "independent consumer permits one or two allocated CPUs" >&2
  exit 2
fi
if [[ "${SLURM_GPUS:-}" != "" && "${SLURM_GPUS:-}" != "0" && "${SLURM_GPUS:-}" != "(null)" ]] || \
   [[ "${SLURM_GPUS_ON_NODE:-}" != "" && "${SLURM_GPUS_ON_NODE:-}" != "0" && "${SLURM_GPUS_ON_NODE:-}" != "(null)" ]]; then
  echo "independent identification consumer must use a CPU-only allocation" >&2
  exit 2
fi
if [[ "${EVALUATION_PYTHON}" != "${REGISTERED_EVALUATION_PYTHON}" ]] || [[ ! -x "${EVALUATION_PYTHON}" ]]; then
  echo "EVALUATION_PYTHON must resolve to an executable" >&2
  exit 2
fi

require_canonical_real_directory() {
  local label="$1"
  local path="$2"
  local resolved
  if [[ "${path}" != /* ]] || [[ "${path}" == *"/../"* ]] || [[ "${path}" == */.. ]] || [[ "${path}" == *"/./"* ]]; then
    echo "${label} must be an absolute canonical path without traversal" >&2
    exit 2
  fi
  resolved="$(realpath -e -- "${path}")"
  if [[ "${resolved}" != "${path}" || ! -d "${path}" || -L "${path}" ]]; then
    echo "${label} must be one canonical real directory with no symlink path" >&2
    exit 2
  fi
}

require_canonical_real_file() {
  local label="$1"
  local path="$2"
  local resolved
  if [[ "${path}" != /* ]] || [[ "${path}" == *"/../"* ]] || [[ "${path}" == */.. ]] || [[ "${path}" == *"/./"* ]]; then
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
require_canonical_real_directory "HISTORICAL_RESULT_ROOT" "${HISTORICAL_RESULT_ROOT}"
require_canonical_real_file "IDENTIFICATION_RESULT_PATH" "${IDENTIFICATION_RESULT_PATH}"
require_canonical_real_file "NUMERIC_VALIDATION_RESULT" "${NUMERIC_VALIDATION_RESULT}"
require_canonical_real_file "PARITY_RESULT_PATH" "${PARITY_RESULT_PATH}"

case "${IDENTIFICATION_RESULT_PATH}" in
  "${REGISTERED_RESULT_ROOT}"/*) ;;
  *)
    echo "IDENTIFICATION_RESULT_PATH must remain under the registered result root" >&2
    exit 2
    ;;
esac
case "${NUMERIC_VALIDATION_RESULT}" in
  "${REGISTERED_RESULT_ROOT}"/*) ;;
  *)
    echo "NUMERIC_VALIDATION_RESULT must remain under the registered result root" >&2
    exit 2
    ;;
esac
case "${PARITY_RESULT_PATH}" in
  "${REGISTERED_RESULT_ROOT}"/*) ;;
  *)
    echo "PARITY_RESULT_PATH must remain under the registered result root" >&2
    exit 2
    ;;
esac

require_canonical_real_file "case manifest" "${REMOTE_REPO}/${MANIFEST_RELATIVE_PATH}"
require_canonical_real_file "selection protocol" "${REMOTE_REPO}/${SELECTION_RELATIVE_PATH}"
require_canonical_real_file "runtime protocol" "${REMOTE_REPO}/${RUNTIME_RELATIVE_PATH}"

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]]; then
  echo "independent consumer requires a clean source tree" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "independent consumer source commit differs from EXPECTED_GIT_COMMIT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export EVALUATION_PYTHON
export PYTHONPATH="${REMOTE_REPO}:${REMOTE_REPO}/main${PYTHONPATH:+:${PYTHONPATH}}"

# Intentionally no --output option and no mkdir: this process is read-only.
exec "${EVALUATION_PYTHON}" scripts/validate_poisson_shadow_identification_artifact.py \
  --repo-root "${REMOTE_REPO}" \
  --identification-result "${IDENTIFICATION_RESULT_PATH}" \
  --numeric-prerequisite "${NUMERIC_VALIDATION_RESULT}" \
  --parity-prerequisite "${PARITY_RESULT_PATH}" \
  --manifest "${MANIFEST_RELATIVE_PATH}" \
  --selection-protocol "${SELECTION_RELATIVE_PATH}" \
  --runtime-protocol "${RUNTIME_RELATIVE_PATH}" \
  --historical-result-root "${HISTORICAL_RESULT_ROOT}" \
  --expected-code-commit "${EXPECTED_GIT_COMMIT}" \
  --expected-case-id "vlsa-t1-goal-ii-t0-e05" \
  --expected-producer-job-id "${EXPECTED_PRODUCER_JOB_ID}" \
  --expected-producer-host "${EXPECTED_PRODUCER_HOST}" \
  --expected-producer-device "${EXPECTED_PRODUCER_DEVICE}" \
  --expected-numeric-job-id "${EXPECTED_NUMERIC_JOB_ID}" \
  --expected-numeric-host "${EXPECTED_NUMERIC_HOST}" \
  --expected-numeric-device "${EXPECTED_NUMERIC_DEVICE}" \
  --expected-parity-job-id "${EXPECTED_PARITY_JOB_ID}" \
  --expected-parity-host "${EXPECTED_PARITY_HOST}" \
  --expected-parity-device "${EXPECTED_PARITY_DEVICE}"
