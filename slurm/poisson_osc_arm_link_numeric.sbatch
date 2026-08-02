#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-post-osc-numeric
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-post-osc-numeric-%j.out

# Allocation-backed implementation gate only.  This job does not execute a
# SafeLIBERO episode and is not scientific efficacy evidence.

set -euo pipefail
umask 077

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-arm-link-study"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-arm-link-study"
REGISTERED_EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${SLURM_JOB_ID:?numeric validation must run inside Slurm}"
: "${SLURM_JOB_PARTITION:?SLURM_JOB_PARTITION is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"
: "${CUDA_VISIBLE_DEVICES:?one allocated H100 MIG device is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]] || \
   [[ "${RESULT_ROOT}" != "${REGISTERED_RESULT_ROOT}" ]]; then
  echo "numeric source or artifact root differs from registration" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
   [[ ! "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
   [[ ! "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "RUN_ID, EXPECTED_GIT_COMMIT, or SLURM_JOB_ID is malformed" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
   (( SLURM_CPUS_PER_TASK != 4 )); then
  echo "numeric validation requires exactly four allocated CPUs" >&2
  exit 2
fi
if [[ "${SLURM_JOB_PARTITION}" != "mig" ]]; then
  echo "numeric validation requires the registered MIG partition" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "numeric validation requires exactly one visible GPU" >&2
  exit 2
fi
if [[ ! -x "${REGISTERED_EVALUATION_PYTHON}" ]]; then
  echo "registered evaluation Python is absent" >&2
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
require_canonical_real_directory "RESULT_ROOT" "${RESULT_ROOT}"
require_canonical_real_file \
  "numeric validation runner" \
  "${REMOTE_REPO}/scripts/run_poisson_numeric_validation.py"

RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
RESULT_PATH="${RUN_ROOT}/result.json"
if [[ -e "${RUN_ROOT}" || -L "${RUN_ROOT}" ]]; then
  echo "immutable numeric run root already exists" >&2
  exit 2
fi
mkdir --mode=0700 -- "${RUN_ROOT}"
require_canonical_real_directory "numeric run root" "${RUN_ROOT}"
if [[ -e "${RESULT_PATH}" || -L "${RESULT_PATH}" ]]; then
  echo "immutable numeric result already exists" >&2
  exit 2
fi

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]] || \
   [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "numeric validation requires the exact clean source commit" >&2
  exit 2
fi

# Do not inherit operator or user-site import paths.  The module list and
# repository import roots are fixed by this checked-in allocation contract.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REMOTE_REPO}:${REMOTE_REPO}/main:${REMOTE_REPO}/safelibero"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export VLSA_POISSON_RUN_PRODUCTION_GRID_VALIDATION=1

"${REGISTERED_EVALUATION_PYTHON}" scripts/run_poisson_numeric_validation.py \
  --repo-root "${REMOTE_REPO}" \
  --output "${RESULT_PATH}" \
  --minimum-tests 120 \
  --test-module tests.test_poisson_geometry \
  --test-module tests.test_poisson_field \
  --test-module tests.test_poisson_field_bundle \
  --test-module tests.test_poisson_production_grid \
  --test-module tests.test_poisson_surface_sampling \
  --test-module tests.test_poisson_measurement \
  --test-module tests.test_poisson_jacobians \
  --test-module tests.test_poisson_cbf_qp \
  --test-module tests.test_poisson_prephysics_control_wrapper \
  --test-module tests.test_poisson_post_osc_torque_sensitivity \
  --test-module tests.test_poisson_post_osc_torque_shield \
  --test-module tests.test_poisson_osc_arm_link_canary \
  --test-module tests.test_poisson_osc_target_link_protocol \
  --test-module tests.test_poisson_osc_numeric_prerequisite \
  --test-module tests.test_poisson_osc_arm_link_slurm_contract \
  --test-module tests.test_poisson_numeric_validation_runner

require_canonical_real_file "immutable numeric result" "${RESULT_PATH}"
