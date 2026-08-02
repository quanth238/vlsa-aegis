#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-closed-loop
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-closed-loop-%j.out

set -euo pipefail

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-closed-loop-canary"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-closed-loop-canary"
REGISTERED_HISTORICAL_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a"
REGISTERED_CHECKPOINT="/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero"
REGISTERED_OPENPI_PYTHON="/mnt/data/quanth/venvs/openpi/bin/python"
REGISTERED_EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"

OPENPI_PYTHON="${OPENPI_PYTHON:-${REGISTERED_OPENPI_PYTHON}}"
EVALUATION_PYTHON="${EVALUATION_PYTHON:-${REGISTERED_EVALUATION_PYTHON}}"
CASE_ID="${CASE_ID:-vlsa-t1-goal-ii-t0-e05}"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${HISTORICAL_RESULT_ROOT:?HISTORICAL_RESULT_ROOT is required}"
: "${SLURM_JOB_ID:?closed-loop canary must run inside Slurm}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]] || \
   [[ "${RESULT_ROOT}" != "${REGISTERED_RESULT_ROOT}" ]] || \
   [[ "${HISTORICAL_RESULT_ROOT}" != "${REGISTERED_HISTORICAL_ROOT}" ]]; then
  echo "closed-loop source or artifact root differs from the registered path" >&2
  exit 2
fi
if [[ "${OPENPI_PYTHON}" != "${REGISTERED_OPENPI_PYTHON}" ]] || \
   [[ "${EVALUATION_PYTHON}" != "${REGISTERED_EVALUATION_PYTHON}" ]]; then
  echo "closed-loop Python runtime differs from the registered executables" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
   [[ ! "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "RUN_ID or EXPECTED_GIT_COMMIT is malformed" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
   (( SLURM_CPUS_PER_TASK > 8 )); then
  echo "closed-loop canary permits at most eight CPUs" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "closed-loop canary requires exactly one visible GPU" >&2
  exit 2
fi
for executable in "${OPENPI_PYTHON}" "${EVALUATION_PYTHON}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "registered Python is not executable: ${executable}" >&2
    exit 2
  fi
done
for directory in \
  "${REMOTE_REPO}" \
  "${RESULT_ROOT}" \
  "${HISTORICAL_RESULT_ROOT}" \
  "${REGISTERED_CHECKPOINT}"
do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "required directory is missing or symlinked: ${directory}" >&2
    exit 2
  fi
done
if [[ -e "${RESULT_ROOT}/${RUN_ID}" || -L "${RESULT_ROOT}/${RUN_ID}" ]]; then
  echo "run root already exists; choose a new immutable RUN_ID" >&2
  exit 2
fi

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]] || \
   [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "closed-loop canary requires the exact clean source commit" >&2
  exit 2
fi

export LIBERO_CONFIG_PATH="${REMOTE_REPO}/configs/libero_runtime"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${REMOTE_REPO}/openpi/src:${REMOTE_REPO}/openpi/packages/openpi-client/src:${REMOTE_REPO}:${REMOTE_REPO}/main:${REMOTE_REPO}/safelibero${PYTHONPATH:+:${PYTHONPATH}}"

POLICY_PORT=$((20000 + 10#${SLURM_JOB_ID} % 30000))
SERVER_LOG="/mnt/data/quanth/slurm_logs/vlsa-poisson-closed-loop-server-${SLURM_JOB_ID}.log"
if [[ -e "${SERVER_LOG}" || -L "${SERVER_LOG}" ]]; then
  echo "policy-server log target already exists" >&2
  exit 2
fi

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
    --port "${POLICY_PORT}" \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "${REGISTERED_CHECKPOINT}"
) >"${SERVER_LOG}" 2>&1 &
POLICY_SERVER_PID=$!

healthy=false
for _ in $(seq 1 600); do
  if ! kill -0 "${POLICY_SERVER_PID}" 2>/dev/null; then
    tail -n 200 "${SERVER_LOG}" >&2 || true
    echo "pi0.5 server exited before healthz" >&2
    exit 2
  fi
  if curl --fail --silent --show-error \
    "http://127.0.0.1:${POLICY_PORT}/healthz" >/dev/null 2>&1; then
    healthy=true
    break
  fi
  sleep 1
done
if [[ "${healthy}" != true ]]; then
  tail -n 200 "${SERVER_LOG}" >&2 || true
  echo "pi0.5 server did not become healthy" >&2
  exit 2
fi

"${EVALUATION_PYTHON}" scripts/run_poisson_closed_loop_canary.py \
  --repo-root "${REMOTE_REPO}" \
  --protocol configs/vlsa_poisson_closed_loop_canary.v1.json \
  --manifest manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl \
  --historical-result-root "${HISTORICAL_RESULT_ROOT}" \
  --output-root "${RESULT_ROOT}" \
  --run-id "${RUN_ID}" \
  --case-id "${CASE_ID}" \
  --host 127.0.0.1 \
  --port "${POLICY_PORT}"
