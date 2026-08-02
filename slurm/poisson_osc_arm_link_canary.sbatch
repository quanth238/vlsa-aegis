#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-post-osc
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-post-osc-%j.out

set -euo pipefail
umask 077

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-arm-link-study"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-arm-link-study"
REGISTERED_HISTORICAL_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a"
REGISTERED_CHECKPOINT="/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero"
OPENPI_PYTHON="/mnt/data/quanth/venvs/openpi/bin/python"
EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${HISTORICAL_RESULT_ROOT:?HISTORICAL_RESULT_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${NUMERIC_VALIDATION_RESULT:?NUMERIC_VALIDATION_RESULT is required}"
: "${EXPECTED_NUMERIC_JOB_ID:?EXPECTED_NUMERIC_JOB_ID is required}"
: "${SLURM_JOB_ID:?post-OSC canary must run inside Slurm}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"
: "${CUDA_VISIBLE_DEVICES:?one allocated GPU is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]] || \
   [[ "${RESULT_ROOT}" != "${REGISTERED_RESULT_ROOT}" ]] || \
   [[ "${HISTORICAL_RESULT_ROOT}" != "${REGISTERED_HISTORICAL_ROOT}" ]]; then
  echo "post-OSC source or artifact root differs from registration" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
   [[ ! "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
   [[ ! "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || \
   [[ ! "${EXPECTED_NUMERIC_JOB_ID}" =~ ^[0-9]+$ ]] || \
   [[ "${EXPECTED_NUMERIC_JOB_ID}" == "${SLURM_JOB_ID}" ]]; then
  echo "run, commit, producer, or numeric job registration is malformed" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
   (( SLURM_CPUS_PER_TASK != 8 )); then
  echo "post-OSC canary requires exactly eight allocated CPUs" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "post-OSC canary requires exactly one visible GPU" >&2
  exit 2
fi
for executable in "${OPENPI_PYTHON}" "${EVALUATION_PYTHON}"; do
  [[ -x "${executable}" ]] || { echo "missing Python: ${executable}" >&2; exit 2; }
done

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
require_canonical_real_directory \
  "HISTORICAL_RESULT_ROOT" \
  "${HISTORICAL_RESULT_ROOT}"
require_canonical_real_directory "registered checkpoint" "${REGISTERED_CHECKPOINT}"
require_canonical_real_directory \
  "LIBERO runtime configuration" \
  "${REMOTE_REPO}/configs/libero_runtime"
require_canonical_real_directory \
  "Slurm log root" \
  "/mnt/data/quanth/slurm_logs"
require_canonical_real_file \
  "post-OSC protocol" \
  "${REMOTE_REPO}/configs/vlsa_poisson_osc_arm_link_canary.v3.json"
require_canonical_real_file \
  "arm-contact manifest" \
  "${REMOTE_REPO}/manifests/vlsa_poisson_arm_contact_165.v1.jsonl"
require_canonical_real_file \
  "post-OSC runner" \
  "${REMOTE_REPO}/scripts/run_poisson_osc_arm_link_canary.py"
require_canonical_real_file \
  "numeric prerequisite" \
  "${NUMERIC_VALIDATION_RESULT}"
NUMERIC_RUN_ROOT="$(dirname -- "${NUMERIC_VALIDATION_RESULT}")"
if [[ "$(dirname -- "${NUMERIC_RUN_ROOT}")" != "${RESULT_ROOT}" ]] || \
   [[ "$(basename -- "${NUMERIC_VALIDATION_RESULT}")" != "result.json" ]] || \
   [[ "$(basename -- "${NUMERIC_RUN_ROOT}")" != vlsa-poisson-post-osc-numeric-* ]]; then
  echo "numeric prerequisite is not a direct registered numeric run result" >&2
  exit 2
fi
require_canonical_real_directory "numeric prerequisite run root" "${NUMERIC_RUN_ROOT}"
if [[ -e "${RESULT_ROOT}/${RUN_ID}" || -L "${RESULT_ROOT}/${RUN_ID}" ]]; then
  echo "run root already exists" >&2
  exit 2
fi

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]] || [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "post-OSC canary requires the exact clean source commit" >&2
  exit 2
fi

# Prevent the allocation from inheriting operator-controlled import roots.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LIBERO_CONFIG_PATH="${REMOTE_REPO}/configs/libero_runtime"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${REMOTE_REPO}/openpi/src:${REMOTE_REPO}/openpi/packages/openpi-client/src:${REMOTE_REPO}:${REMOTE_REPO}/main:${REMOTE_REPO}/safelibero"

POLICY_PORT=$((20000 + 10#${SLURM_JOB_ID} % 30000))
SERVER_LOG="/mnt/data/quanth/slurm_logs/vlsa-poisson-post-osc-server-${SLURM_JOB_ID}.log"
[[ ! -e "${SERVER_LOG}" && ! -L "${SERVER_LOG}" ]] || { echo "server log exists" >&2; exit 2; }
POLICY_SERVER_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${POLICY_SERVER_PID}" ]] && kill -0 "${POLICY_SERVER_PID}" 2>/dev/null; then
    kill "${POLICY_SERVER_PID}" 2>/dev/null || true
    wait "${POLICY_SERVER_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

(
  cd "${REMOTE_REPO}/openpi"
  exec "${OPENPI_PYTHON}" scripts/serve_policy.py \
    --port "${POLICY_PORT}" policy:checkpoint \
    --policy.config pi05_libero --policy.dir "${REGISTERED_CHECKPOINT}"
) >"${SERVER_LOG}" 2>&1 &
POLICY_SERVER_PID=$!

healthy=false
for _ in $(seq 1 600); do
  kill -0 "${POLICY_SERVER_PID}" 2>/dev/null || { tail -n 200 "${SERVER_LOG}" >&2 || true; exit 2; }
  if curl --fail --silent "http://127.0.0.1:${POLICY_PORT}/healthz" >/dev/null 2>&1; then
    healthy=true
    break
  fi
  sleep 1
done
[[ "${healthy}" == true ]] || { echo "pi0.5 server health check failed" >&2; exit 2; }

"${EVALUATION_PYTHON}" scripts/run_poisson_osc_arm_link_canary.py \
  --repo-root "${REMOTE_REPO}" \
  --protocol configs/vlsa_poisson_osc_arm_link_canary.v3.json \
  --manifest manifests/vlsa_poisson_arm_contact_165.v1.jsonl \
  --historical-result-root "${HISTORICAL_RESULT_ROOT}" \
  --numeric-validation-result "${NUMERIC_VALIDATION_RESULT}" \
  --expected-numeric-job-id "${EXPECTED_NUMERIC_JOB_ID}" \
  --output-root "${RESULT_ROOT}" \
  --run-id "${RUN_ID}" \
  --case-id vlsa-t1-spatial-i-t3-e03 \
  --host 127.0.0.1 \
  --port "${POLICY_PORT}"
