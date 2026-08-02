#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-post-osc-validate
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-post-osc-validate-%j.out

set -euo pipefail
umask 077

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-arm-link-study"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-arm-link-study"
REGISTERED_HISTORICAL_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a"
EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RESULT:?RESULT is required}"
: "${HISTORICAL_RESULT_ROOT:?HISTORICAL_RESULT_ROOT is required}"
: "${CASE_ID:?CASE_ID is required}"
: "${PROTOCOL_RELATIVE_PATH:?PROTOCOL_RELATIVE_PATH is required}"
: "${EXPECTED_PRODUCER_GIT_COMMIT:?EXPECTED_PRODUCER_GIT_COMMIT is required}"
: "${EXPECTED_CONSUMER_GIT_COMMIT:?EXPECTED_CONSUMER_GIT_COMMIT is required}"
: "${EXPECTED_PRODUCER_JOB_ID:?EXPECTED_PRODUCER_JOB_ID is required}"
: "${NUMERIC_VALIDATION_RESULT:?NUMERIC_VALIDATION_RESULT is required}"
: "${EXPECTED_NUMERIC_JOB_ID:?EXPECTED_NUMERIC_JOB_ID is required}"
: "${SLURM_JOB_ID:?validator must run inside Slurm}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"
if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]] || \
   [[ "${HISTORICAL_RESULT_ROOT}" != "${REGISTERED_HISTORICAL_ROOT}" ]] || \
   [[ ! "${CASE_ID}" =~ ^vlsa-t1-[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
   [[ ! "${PROTOCOL_RELATIVE_PATH}" =~ ^configs/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json$ ]] || \
   [[ ! "${EXPECTED_PRODUCER_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
   [[ ! "${EXPECTED_CONSUMER_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
   [[ ! "${EXPECTED_PRODUCER_JOB_ID}" =~ ^[0-9]+$ ]] || \
   [[ ! "${EXPECTED_NUMERIC_JOB_ID}" =~ ^[0-9]+$ ]] || \
   [[ ! "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || \
   [[ "${EXPECTED_PRODUCER_JOB_ID}" == "${SLURM_JOB_ID}" ]] || \
   [[ "${EXPECTED_NUMERIC_JOB_ID}" == "${SLURM_JOB_ID}" ]] || \
   [[ "${EXPECTED_NUMERIC_JOB_ID}" == "${EXPECTED_PRODUCER_JOB_ID}" ]]; then
  echo "validator registration differs" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
   (( SLURM_CPUS_PER_TASK < 1 || SLURM_CPUS_PER_TASK > 2 )); then
  echo "validator permits one or two allocated CPUs" >&2
  exit 2
fi
[[ -x "${EVALUATION_PYTHON}" ]] || { echo "evaluation Python is absent" >&2; exit 2; }

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
require_canonical_real_directory \
  "registered result root" \
  "${REGISTERED_RESULT_ROOT}"
require_canonical_real_directory \
  "HISTORICAL_RESULT_ROOT" \
  "${HISTORICAL_RESULT_ROOT}"
require_canonical_real_file "complete result" "${RESULT}"
require_canonical_real_file \
  "post-OSC protocol" \
  "${REMOTE_REPO}/${PROTOCOL_RELATIVE_PATH}"
require_canonical_real_file \
  "post-OSC validator" \
  "${REMOTE_REPO}/scripts/validate_poisson_osc_arm_link_canary_artifact.py"
require_canonical_real_file \
  "numeric prerequisite" \
  "${NUMERIC_VALIDATION_RESULT}"
NUMERIC_RUN_ROOT="$(dirname -- "${NUMERIC_VALIDATION_RESULT}")"
if [[ "$(dirname -- "${NUMERIC_RUN_ROOT}")" != "${REGISTERED_RESULT_ROOT}" ]] || \
   [[ "$(basename -- "${NUMERIC_VALIDATION_RESULT}")" != "result.json" ]] || \
   [[ "$(basename -- "${NUMERIC_RUN_ROOT}")" != vlsa-poisson-post-osc-numeric-* ]]; then
  echo "numeric prerequisite is not a direct registered numeric run result" >&2
  exit 2
fi
require_canonical_real_directory "numeric prerequisite run root" "${NUMERIC_RUN_ROOT}"
RUN_ROOT="$(dirname -- "${RESULT}")"
if [[ "$(dirname -- "${RUN_ROOT}")" != "${REGISTERED_RESULT_ROOT}" ]] || \
   [[ "$(basename -- "${RESULT}")" != "result.json" ]]; then
  echo "RESULT is not a direct registered run result" >&2
  exit 2
fi
require_canonical_real_directory "post-OSC run root" "${RUN_ROOT}"
OUTPUT="${RUN_ROOT}/validation_receipt_schema_v4_target_link.json"
[[ ! -e "${OUTPUT}" && ! -L "${OUTPUT}" ]] || { echo "validation output exists" >&2; exit 2; }

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]] || \
   [[ "$(git rev-parse HEAD)" != "${EXPECTED_CONSUMER_GIT_COMMIT}" ]]; then
  echo "validator requires the exact clean source commit" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REMOTE_REPO}:${REMOTE_REPO}/main"

"${EVALUATION_PYTHON}" scripts/validate_poisson_osc_arm_link_canary_artifact.py \
  --result "${RESULT}" \
  --expected-case-id "${CASE_ID}" \
  --expected-protocol-relative-path "${PROTOCOL_RELATIVE_PATH}" \
  --historical-result-root "${HISTORICAL_RESULT_ROOT}" \
  --numeric-validation-result "${NUMERIC_VALIDATION_RESULT}" \
  --expected-numeric-job-id "${EXPECTED_NUMERIC_JOB_ID}" \
  --protocol "${REMOTE_REPO}/${PROTOCOL_RELATIVE_PATH}" \
  --expected-producer-job-id "${EXPECTED_PRODUCER_JOB_ID}" \
  --expected-producer-commit "${EXPECTED_PRODUCER_GIT_COMMIT}" \
  --expected-consumer-job-id "${SLURM_JOB_ID}" \
  --expected-consumer-commit "${EXPECTED_CONSUMER_GIT_COMMIT}" \
  --output "${OUTPUT}"
