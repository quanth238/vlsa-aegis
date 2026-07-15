#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?R05A validation must run inside Slurm}"
: "${SLURM_JOB_PARTITION:?R05A validation must record its partition}"
: "${SOURCE_JOB_ID:?SOURCE_JOB_ID is required}"
: "${SOURCE_RESULT:?SOURCE_RESULT is required}"
: "${VALIDATION_RECEIPT:?VALIDATION_RECEIPT is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"

test "$SLURM_JOB_PARTITION" = main || { echo "R05A CPU validation is frozen to main" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 2 || { echo "R05A CPU validator requires two CPUs" >&2; exit 2; }
test "${CUDA_VISIBLE_DEVICES:-NoDevFiles}" = NoDevFiles || {
  echo "R05A independent validator must not receive a GPU" >&2
  exit 2
}
test -x "$LIBERO_PYTHON" || { echo "missing validator Python" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || {
  echo "validator source commit differs" >&2
  exit 2
}
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || {
  echo "validator requires a clean remote tree" >&2
  exit 2
}

mkdir -p "$(dirname "$VALIDATION_RECEIPT")"
fallback_receipt() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ ! -f "$VALIDATION_RECEIPT" ]; then
    "$LIBERO_PYTHON" - "$VALIDATION_RECEIPT" "$status" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
value = {
    "schema_version": "1.0",
    "artifact_role": "r05a_inverse_flow_canary_cpu_afterany_validation",
    "source_job_id": os.environ.get("SOURCE_JOB_ID"),
    "result_path": os.environ.get("SOURCE_RESULT"),
    "result_sha256": None,
    "passed": False,
    "errors": [f"validator wrapper failed with exit code {sys.argv[2]}"],
    "scientific_claim_allowed": False,
    "probe_training_authorized": False,
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, path)
PY
  fi
  return "$status"
}
trap fallback_receipt EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

JSONSCHEMA_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_jsonschema_overlay.sh)
export PYTHONPATH=$JSONSCHEMA_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero
set +e
"$LIBERO_PYTHON" "$REMOTE_REPO/main/validate_crfs_r05a_canary.py" \
  --result "$SOURCE_RESULT" \
  --receipt "$VALIDATION_RECEIPT" \
  --source-job-id "$SOURCE_JOB_ID"
validation_status=$?
set -e

test -f "$VALIDATION_RECEIPT" || { echo "CPU validator wrote no receipt" >&2; exit 3; }
echo "source_job_id=$SOURCE_JOB_ID"
echo "source_result=$SOURCE_RESULT"
echo "validation_receipt=$VALIDATION_RECEIPT"
echo "validation_receipt_sha256=$(sha256sum "$VALIDATION_RECEIPT" | awk '{print $1}')"
exit "$validation_status"
