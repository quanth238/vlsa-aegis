#!/usr/bin/env bash
#SBATCH --job-name=vlsa-poisson-closed-loop-validate
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-poisson-closed-loop-validate-%j.out

# Submit only after the exact producer is terminal, normally with
# --dependency=afterok:<producer-job-id>. This independent consumer requests
# no GPU. It publishes one receipt only after complete validation succeeds.

set -euo pipefail

REGISTERED_REMOTE_REPO="/home/quanth/working_space/vlsa-aegis-poisson-closed-loop-canary"
REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-poisson-closed-loop-canary"
REGISTERED_HISTORICAL_ROOT="/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a"
REGISTERED_EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"
REGISTERED_PRODUCER_DEVICE="NVIDIA H100 80GB HBM3"
PROTOCOL_RELATIVE_PATH="configs/vlsa_poisson_closed_loop_canary.v1.json"

EVALUATION_PYTHON="${EVALUATION_PYTHON:-${REGISTERED_EVALUATION_PYTHON}}"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${CLOSED_LOOP_RESULT:?CLOSED_LOOP_RESULT is required}"
: "${HISTORICAL_RESULT_ROOT:?HISTORICAL_RESULT_ROOT is required}"
: "${EXPECTED_PRODUCER_GIT_COMMIT:?EXPECTED_PRODUCER_GIT_COMMIT is required}"
: "${EXPECTED_CONSUMER_GIT_COMMIT:?EXPECTED_CONSUMER_GIT_COMMIT is required}"
: "${EXPECTED_PRODUCER_JOB_ID:?EXPECTED_PRODUCER_JOB_ID is required}"
: "${EXPECTED_PRODUCER_HOST:?EXPECTED_PRODUCER_HOST is required}"
: "${EXPECTED_PRODUCER_DEVICE:?EXPECTED_PRODUCER_DEVICE is required}"
: "${SLURM_JOB_ID:?independent consumer must run inside Slurm}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}"

if [[ "${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}" ]]; then
  echo "REMOTE_REPO differs from the registered closed-loop checkout" >&2
  exit 2
fi
if [[ "${HISTORICAL_RESULT_ROOT}" != "${REGISTERED_HISTORICAL_ROOT}" ]]; then
  echo "HISTORICAL_RESULT_ROOT differs from the registered Table-1 run" >&2
  exit 2
fi
if [[ "${EXPECTED_PRODUCER_DEVICE}" != "${REGISTERED_PRODUCER_DEVICE}" ]]; then
  echo "EXPECTED_PRODUCER_DEVICE differs from the registered H100 device" >&2
  exit 2
fi
for commit_variable in \
  EXPECTED_PRODUCER_GIT_COMMIT \
  EXPECTED_CONSUMER_GIT_COMMIT
do
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
if [[ ! "${EXPECTED_PRODUCER_HOST}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "EXPECTED_PRODUCER_HOST is malformed" >&2
  exit 2
fi
if [[ "${SLURM_JOB_ID}" == "${EXPECTED_PRODUCER_JOB_ID}" ]]; then
  echo "producer and independent consumer must use distinct Slurm jobs" >&2
  exit 2
fi
if [[ ! "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
   (( SLURM_CPUS_PER_TASK < 1 || SLURM_CPUS_PER_TASK > 2 )); then
  echo "independent consumer permits one or two allocated CPUs" >&2
  exit 2
fi
if [[ "${SLURM_GPUS:-}" != "" && "${SLURM_GPUS:-}" != "0" && "${SLURM_GPUS:-}" != "(null)" ]] || \
   [[ "${SLURM_GPUS_ON_NODE:-}" != "" && "${SLURM_GPUS_ON_NODE:-}" != "0" && "${SLURM_GPUS_ON_NODE:-}" != "(null)" ]]; then
  echo "independent closed-loop consumer must use a CPU-only allocation" >&2
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
require_canonical_real_directory "registered result root" "${REGISTERED_RESULT_ROOT}"
require_canonical_real_directory "HISTORICAL_RESULT_ROOT" "${HISTORICAL_RESULT_ROOT}"
require_canonical_real_file "CLOSED_LOOP_RESULT" "${CLOSED_LOOP_RESULT}"
require_canonical_real_file \
  "closed-loop protocol" \
  "${REMOTE_REPO}/${PROTOCOL_RELATIVE_PATH}"
require_canonical_real_file \
  "closed-loop validator" \
  "${REMOTE_REPO}/scripts/validate_poisson_closed_loop_canary_artifact.py"

RUN_ROOT="$(dirname -- "${CLOSED_LOOP_RESULT}")"
RUN_ID="$(basename -- "${RUN_ROOT}")"
if [[ "$(dirname -- "${RUN_ROOT}")" != "${REGISTERED_RESULT_ROOT}" ]] || \
   [[ "$(basename -- "${CLOSED_LOOP_RESULT}")" != "result.json" ]] || \
   [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "CLOSED_LOOP_RESULT must be one direct immutable run result" >&2
  exit 2
fi
require_canonical_real_directory "closed-loop run root" "${RUN_ROOT}"

RECEIPT_PATH="${RUN_ROOT}/validation_receipt.json"
PARTIAL_PATH="${RUN_ROOT}/validation_receipt.partial.${SLURM_JOB_ID}"
for target in "${RECEIPT_PATH}" "${PARTIAL_PATH}"; do
  if [[ -e "${target}" || -L "${target}" ]]; then
    echo "validation receipt target already exists: ${target}" >&2
    exit 2
  fi
done

cd "${REMOTE_REPO}"
if [[ -n "$(git status --short)" ]]; then
  echo "independent consumer requires a clean source tree" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${EXPECTED_CONSUMER_GIT_COMMIT}" ]]; then
  echo "consumer source commit differs from EXPECTED_CONSUMER_GIT_COMMIT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REMOTE_REPO}:${REMOTE_REPO}/main${PYTHONPATH:+:${PYTHONPATH}}"

cleanup_partial() {
  if [[ -f "${PARTIAL_PATH}" && ! -L "${PARTIAL_PATH}" ]]; then
    rm -f -- "${PARTIAL_PATH}"
  fi
}
trap cleanup_partial EXIT INT TERM

"${EVALUATION_PYTHON}" \
  scripts/validate_poisson_closed_loop_canary_artifact.py \
  --repo-root "${REMOTE_REPO}" \
  --result "${CLOSED_LOOP_RESULT}" \
  --protocol "${REMOTE_REPO}/${PROTOCOL_RELATIVE_PATH}" \
  --historical-result-root "${HISTORICAL_RESULT_ROOT}" \
  --expected-producer-commit "${EXPECTED_PRODUCER_GIT_COMMIT}" \
  --consumer-commit "${EXPECTED_CONSUMER_GIT_COMMIT}" \
  --expected-producer-job-id "${EXPECTED_PRODUCER_JOB_ID}" \
  --expected-producer-host "${EXPECTED_PRODUCER_HOST}" \
  --expected-producer-device "${EXPECTED_PRODUCER_DEVICE}" \
  >"${PARTIAL_PATH}"

# The validator exits zero for both positive and valid-negative scientific
# outcomes. Before publication, require its complete typed success summary.
"${EVALUATION_PYTHON}" - "${PARTIAL_PATH}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(value, dict):
    raise SystemExit("closed-loop validation summary must be an object")
if value.get("schema_version") != "vlsa_poisson_closed_loop_canary_consumer.v1":
    raise SystemExit("closed-loop validation summary schema differs")
if value.get("status") != "validated" or value.get("artifact_valid") is not True:
    raise SystemExit("closed-loop validation summary did not validate")
if value.get("discrepancies") != []:
    raise SystemExit("closed-loop validation summary has discrepancies")
PY

if [[ -e "${RECEIPT_PATH}" || -L "${RECEIPT_PATH}" ]]; then
  echo "validation receipt appeared before atomic publication" >&2
  exit 2
fi
mv -- "${PARTIAL_PATH}" "${RECEIPT_PATH}"
trap - EXIT INT TERM

require_canonical_real_file "validation receipt" "${RECEIPT_PATH}"
cat -- "${RECEIPT_PATH}"
