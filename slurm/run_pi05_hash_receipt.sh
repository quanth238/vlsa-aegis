#!/usr/bin/env bash
set -euo pipefail

# One-time, allocation-only, content hash of the 15 GB JAX Orbax checkpoint.
# This script does not load the policy or execute a simulator.

: "${HASH_RECEIPT_ID:?set a unique immutable HASH_RECEIPT_ID}"
: "${EXPECTED_GIT_COMMIT:?set the reviewed 40-character source commit}"

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1}
PI05_CHECKPOINT=${PI05_CHECKPOINT:-/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}

case "$HASH_RECEIPT_ID" in
  ""|*[!A-Za-z0-9._-]*)
    echo "unsafe HASH_RECEIPT_ID: $HASH_RECEIPT_ID" >&2
    exit 2
    ;;
esac
case "$EXPECTED_GIT_COMMIT" in
  *[!0-9a-f]*|"")
    echo "EXPECTED_GIT_COMMIT is not lowercase hexadecimal" >&2
    exit 2
    ;;
esac
[[ ${#EXPECTED_GIT_COMMIT} -eq 40 ]] || {
  echo "EXPECTED_GIT_COMMIT must contain exactly 40 characters" >&2
  exit 2
}

for name in SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID \
  SLURM_ARRAY_TASK_COUNT SLURMD_NODENAME SLURM_CPUS_PER_TASK \
  SLURM_MEM_PER_NODE; do
  [[ -n "${!name:-}" ]] || {
    echo "checkpoint hash requires Slurm variable $name" >&2
    exit 2
  }
done
case "$SLURMD_NODENAME" in
  worker-3|login*|login-restricted*)
    echo "checkpoint hash cannot execute on $SLURMD_NODENAME" >&2
    exit 2
    ;;
esac
[[ "$SLURM_ARRAY_TASK_ID" == 0 && "$SLURM_ARRAY_TASK_COUNT" == 1 ]] || {
  echo "checkpoint hash must be an exact one-task array" >&2
  exit 2
}
[[ "$SLURM_CPUS_PER_TASK" == 2 && "$SLURM_MEM_PER_NODE" == 16384 ]] || {
  echo "checkpoint hash allocation changed from 2 CPUs / 16384 MiB" >&2
  exit 2
}
[[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES:-}" == NoDevFiles ]] || {
  echo "checkpoint hashing must not consume a GPU" >&2
  exit 2
}
[[ -x "$AEGIS_PYTHON" ]] || {
  echo "missing AEGIS Python interpreter: $AEGIS_PYTHON" >&2
  exit 2
}
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || {
  echo "remote source commit differs from EXPECTED_GIT_COMMIT" >&2
  exit 2
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "remote source tree must be clean" >&2
  exit 2
}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
RECEIPT_ROOT=$EXPERIMENT_ROOT/checkpoint-receipts
mkdir -p "$RECEIPT_ROOT"
RECEIPT_PATH=$RECEIPT_ROOT/$HASH_RECEIPT_ID.json

exec "$AEGIS_PYTHON" "$REMOTE_REPO/scripts/compute_pi05_tree_receipt.py" \
  --repo-root "$REMOTE_REPO" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --checkpoint "$PI05_CHECKPOINT" \
  --output "$RECEIPT_PATH"
