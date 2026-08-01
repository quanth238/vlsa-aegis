#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-one-step-validate
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-one-step-validate-%j.out

# This is a post-producer, allocation-side consumer.  It intentionally has no
# partition or GPU request: choose a live CPU-capable partition at submission
# time.  If the cluster exposes no CPU allocation, live preflight may supply
# the minimum partition resource required by policy without changing this
# consumer's CPU-only implementation.

set -euo pipefail

EVALUATION_PYTHON="${EVALUATION_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}"
REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-feasibility"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-feasibility"
PROTOCOL_RELATIVE_PATH="configs/vlsa_poisson_one_step_counterfactual.v1.json"
SELECTION_RELATIVE_PATH="configs/vlsa_poisson_link56_feasibility.v1.json"
RUNTIME_RELATIVE_PATH="configs/vlsa_poisson_runtime_protocol.canary.v2.json"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${HISTORICAL_RESULT_PATH:?HISTORICAL_RESULT_PATH is required}"
: "${NUMERIC_VALIDATION_RESULT:?NUMERIC_VALIDATION_RESULT is required}"
: "${PARITY_RESULT:?PARITY_RESULT is required}"
: "${SHADOW_IDENTIFICATION_RESULT:?SHADOW_IDENTIFICATION_RESULT is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${EXPECTED_PRODUCER_JOB_ID:?EXPECTED_PRODUCER_JOB_ID is required}"
: "${EXPECTED_PRODUCER_HOST:?EXPECTED_PRODUCER_HOST is required}"
: "${EXPECTED_PRODUCER_DEVICE:?EXPECTED_PRODUCER_DEVICE is required}"

: "${SLURM_JOB_ID:?independent validation must run inside a separate Slurm allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]]; then
  echo "REMOTE_REPO differs from the registered Poisson-feasibility checkout" >&2
  exit 2
fi
if [[ "${RESULT_ROOT}" != "${REGISTERED_RESULT_ROOT}" ]]; then
  echo "RESULT_ROOT differs from the registered Poisson-feasibility artifact root" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "RUN_ID must use portable alphanumeric/dot/underscore/hyphen characters" >&2
  exit 2
fi
if [[ ! "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_GIT_COMMIT must be one lowercase 40-character commit" >&2
  exit 2
fi
if [[ ! "${EXPECTED_PRODUCER_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "EXPECTED_PRODUCER_JOB_ID must identify one exact Slurm job" >&2
  exit 2
fi
if [[ "${SLURM_JOB_ID}" == "${EXPECTED_PRODUCER_JOB_ID}" ]]; then
  echo "independent validation must use a different Slurm job from the producer" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || (( SLURM_CPUS_PER_TASK > 2 )); then
  echo "independent validation permits at most 2 allocated CPUs" >&2
  exit 2
fi
if [[ ! -d "${REMOTE_REPO}" || -L "${REMOTE_REPO}" ]]; then
  echo "REMOTE_REPO must be a real directory" >&2
  exit 2
fi
if [[ ! -d "${RESULT_ROOT}" || -L "${RESULT_ROOT}" ]]; then
  echo "RESULT_ROOT must be a real directory" >&2
  exit 2
fi
if [[ ! -x "${EVALUATION_PYTHON}" ]]; then
  echo "EVALUATION_PYTHON must resolve to an executable" >&2
  exit 2
fi

RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
RESULT_PATH="${RUN_ROOT}/result.json"
RECEIPT_PATH="${RUN_ROOT}/validation_receipt.json"
if [[ ! -d "${RUN_ROOT}" || -L "${RUN_ROOT}" ]]; then
  echo "the immutable producer run root must be a real directory" >&2
  exit 2
fi

require_regular_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "${label} must be a nonsymlink regular file: ${path}" >&2
    exit 2
  fi
}

require_regular_file "RESULT_PATH" "${RESULT_PATH}"
require_regular_file "RECEIPT_PATH" "${RECEIPT_PATH}"
require_regular_file "HISTORICAL_RESULT_PATH" "${HISTORICAL_RESULT_PATH}"
require_regular_file "NUMERIC_VALIDATION_RESULT" "${NUMERIC_VALIDATION_RESULT}"
require_regular_file "PARITY_RESULT" "${PARITY_RESULT}"
require_regular_file "SHADOW_IDENTIFICATION_RESULT" "${SHADOW_IDENTIFICATION_RESULT}"
require_regular_file "ONE_STEP_PROTOCOL" "${REMOTE_REPO}/${PROTOCOL_RELATIVE_PATH}"
require_regular_file "SELECTION_PROTOCOL" "${REMOTE_REPO}/${SELECTION_RELATIVE_PATH}"
require_regular_file "RUNTIME_PROTOCOL" "${REMOTE_REPO}/${RUNTIME_RELATIVE_PATH}"

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]]; then
  echo "independent validation requires a clean source tree" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "independent validation source commit differs from EXPECTED_GIT_COMMIT" >&2
  exit 2
fi

export PYTHONPATH="${REMOTE_REPO}:${REMOTE_REPO}/main${PYTHONPATH:+:${PYTHONPATH}}"

# No artifact path is opened for writing.  The only success record is stdout,
# which Slurm captures in its scheduler log.  The exact producer job and this
# consumer job must each be checked externally with sacct for COMPLETED|0:0.
exec "${EVALUATION_PYTHON}" scripts/validate_poisson_one_step_counterfactual_artifact.py \
  --result "${RESULT_PATH}" \
  --receipt "${RECEIPT_PATH}" \
  --historical-result "${HISTORICAL_RESULT_PATH}" \
  --protocol "${PROTOCOL_RELATIVE_PATH}" \
  --selection-protocol "${SELECTION_RELATIVE_PATH}" \
  --runtime-protocol "${RUNTIME_RELATIVE_PATH}" \
  --numeric-prerequisite "${NUMERIC_VALIDATION_RESULT}" \
  --parity-prerequisite "${PARITY_RESULT}" \
  --identification-prerequisite "${SHADOW_IDENTIFICATION_RESULT}" \
  --expected-code-commit "${EXPECTED_GIT_COMMIT}" \
  --expected-run-id "${RUN_ID}" \
  --expected-slurm-job-id "${EXPECTED_PRODUCER_JOB_ID}" \
  --expected-host "${EXPECTED_PRODUCER_HOST}" \
  --expected-device "${EXPECTED_PRODUCER_DEVICE}"
