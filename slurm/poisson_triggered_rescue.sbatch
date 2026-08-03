#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-triggered-rescue
#SBATCH --partition=mig
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-triggered-rescue-%j.out

set -euo pipefail

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-arm-link-study"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-triggered-rescue"
REGISTERED_HISTORICAL_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a"
REGISTERED_EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"
REGISTERED_PROTOCOL="configs/vlsa_poisson_triggered_rescue.v1.json"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${HISTORICAL_RESULT_ROOT:?HISTORICAL_RESULT_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${CASE_ID:?CASE_ID is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${SLURM_JOB_ID:?triggered rescue must run inside Slurm}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]] || \
   [[ "${RESULT_ROOT}" != "${REGISTERED_RESULT_ROOT}" ]] || \
   [[ "${HISTORICAL_RESULT_ROOT}" != "${REGISTERED_HISTORICAL_ROOT}" ]]; then
  echo "triggered-rescue source or artifact root differs" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
   [[ ! "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "RUN_ID or EXPECTED_GIT_COMMIT is malformed" >&2
  exit 2
fi
case "${CASE_ID}" in
  vlsa-t1-goal-ii-t0-e05) ;;
  vlsa-t1-goal-ii-t3-e42)
    : "${E05_VALIDATION_RECEIPT:?e42 requires the validated e05 receipt}"
    if [[ ! -f "${E05_VALIDATION_RECEIPT}" || -L "${E05_VALIDATION_RECEIPT}" ]]; then
      echo "e05 validation receipt is missing or symlinked" >&2
      exit 2
    fi
    PYTHONPATH="${REMOTE_REPO}:${REMOTE_REPO}/main" \
      "${REGISTERED_EVALUATION_PYTHON}" - "${E05_VALIDATION_RECEIPT}" <<'PY'
import json
import sys
from pathlib import Path
from main.poisson_fullbody.contracts import load_hashed_json

value = load_hashed_json(Path(sys.argv[1]))
if not (
    value.get("status") == "validated"
    and value.get("artifact_valid") is True
    and value.get("feasible") is True
    and value.get("case_id") == "vlsa-t1-goal-ii-t0-e05"
    and value.get("independent_classification")
    == "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"
):
    raise SystemExit("e05 is not a validated positive result")
PY
    ;;
  *)
    echo "CASE_ID is outside the registered two-case study" >&2
    exit 2
    ;;
esac
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
   (( SLURM_CPUS_PER_TASK != 8 )); then
  echo "triggered rescue requires exactly eight allocated CPUs" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "triggered rescue requires exactly one visible H100" >&2
  exit 2
fi
if [[ ! -x "${REGISTERED_EVALUATION_PYTHON}" ]]; then
  echo "registered evaluation Python is not executable" >&2
  exit 2
fi
for directory in \
  "${REMOTE_REPO}" \
  "${HISTORICAL_RESULT_ROOT}"
do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "required directory is missing or symlinked: ${directory}" >&2
    exit 2
  fi
done
mkdir -p "${REGISTERED_RESULT_ROOT}"
if [[ ! -d "${REGISTERED_RESULT_ROOT}" || -L "${REGISTERED_RESULT_ROOT}" ]]; then
  echo "registered result root is unavailable" >&2
  exit 2
fi
if [[ -e "${RESULT_ROOT}/${RUN_ID}" || -L "${RESULT_ROOT}/${RUN_ID}" ]]; then
  echo "run root already exists; choose a new immutable RUN_ID" >&2
  exit 2
fi

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]] || \
   [[ "$(git rev-parse HEAD)" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "triggered rescue requires the exact clean source commit" >&2
  exit 2
fi
if [[ ! -f "${REGISTERED_PROTOCOL}" || -L "${REGISTERED_PROTOCOL}" ]]; then
  echo "triggered-rescue protocol is absent or symlinked" >&2
  exit 2
fi

export LIBERO_CONFIG_PATH="${REMOTE_REPO}/configs/libero_runtime"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REMOTE_REPO}/openpi/src:${REMOTE_REPO}/openpi/packages/openpi-client/src:${REMOTE_REPO}:${REMOTE_REPO}/main:${REMOTE_REPO}/safelibero${PYTHONPATH:+:${PYTHONPATH}}"

# One compact allocation-backed gate exercises the exact method contract.
"${REGISTERED_EVALUATION_PYTHON}" -m unittest -v \
  tests.test_poisson_triggered_rescue \
  tests.test_poisson_triggered_rescue_runner \
  tests.test_poisson_registered_contact_monitor

exec "${REGISTERED_EVALUATION_PYTHON}" scripts/run_poisson_triggered_rescue.py \
  --repo-root "${REMOTE_REPO}" \
  --protocol "${REGISTERED_PROTOCOL}" \
  --manifest manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl \
  --historical-result-root "${HISTORICAL_RESULT_ROOT}" \
  --output-root "${RESULT_ROOT}" \
  --run-id "${RUN_ID}" \
  --case-id "${CASE_ID}"
