#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?sampled-current publication must run inside Slurm}"
: "${SLURM_JOB_PARTITION:?publisher partition provenance is required}"
: "${SOURCE_JOB_ID:?exact H100 array job id is required}"
: "${SOURCE_CONTRACT:?source-contract path is required}"
: "${EXPECTED_SOURCE_CONTRACT_SHA256:?externally frozen source-contract digest is required}"
: "${SUBMISSION:?atomic submission receipt is required}"
: "${PAYLOAD:?scientific payload path is required}"
: "${HOST_TELEMETRY:?sealed host telemetry path is required}"
: "${GPU_SAMPLES:?GPU sample path is required}"
: "${ALLOCATION_TEST_LOG:?allocation test log is required}"
: "${RESULT:?final result path is required}"
: "${VALIDATION_RECEIPT:?validation receipt path is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed source commit is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"

test "$SLURM_JOB_PARTITION" = main || { echo "sampled-current publisher is frozen to main" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 2 || { echo "sampled-current publisher requires two CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 8192 || { echo "sampled-current publisher requires exactly 8 GiB" >&2; exit 2; }
test "${CUDA_VISIBLE_DEVICES:-NoDevFiles}" = NoDevFiles || {
  echo "sampled-current publisher must not receive a GPU" >&2
  exit 2
}
case "$SOURCE_JOB_ID" in *[!0-9]*|'') echo "invalid source array job id" >&2; exit 2 ;; esac
case "$EXPECTED_SOURCE_CONTRACT_SHA256" in
  *[!0-9a-f]*|'') echo "invalid external source-contract digest" >&2; exit 2 ;;
esac
test "${#EXPECTED_SOURCE_CONTRACT_SHA256}" = 64 || { echo "invalid external digest length" >&2; exit 2; }
test "$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')" = "$EXPECTED_SOURCE_CONTRACT_SHA256" || {
  echo "source contract differs from the externally supplied digest" >&2
  exit 2
}
test -x "$LIBERO_PYTHON" || { echo "missing publisher Python" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || {
  echo "publisher source commit differs" >&2
  exit 2
}
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || {
  echo "publisher requires a clean remote tree" >&2
  exit 2
}
publisher_job_record=$(scontrol show job "$SLURM_JOB_ID" -o)
for field in \
  "JobState=RUNNING" \
  "Partition=main" \
  "Account=normal" \
  "QOS=normal" \
  "TimeLimit=00:15:00" \
  "Requeue=0"; do
  case " $publisher_job_record " in
    *" $field "*) ;;
    *) echo "running CPU publisher field changed: $field" >&2; exit 2 ;;
  esac
done
case " $publisher_job_record " in *gres/gpu*) echo "running CPU publisher requests a GPU" >&2; exit 2 ;; esac
publisher_req_tres=$(printf '%s\n' "$publisher_job_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test -n "$publisher_req_tres" || { echo "cannot parse running CPU publisher ReqTRES" >&2; exit 2; }
publisher_req_cpus=$(printf '%s\n' "$publisher_req_tres" | tr ',' '\n' | sed -n 's/^cpu=//p')
publisher_req_mem=$(printf '%s\n' "$publisher_req_tres" | tr ',' '\n' | sed -n 's/^mem=//p')
test "$publisher_req_cpus" = 2 || { echo "running CPU publisher CPU request changed" >&2; exit 2; }
case "$publisher_req_mem" in 8G|8192M) ;; *) echo "running CPU publisher memory request changed" >&2; exit 2 ;; esac

mkdir -p "$(dirname "$VALIDATION_RECEIPT")"
fallback_receipt() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ ! -f "$VALIDATION_RECEIPT" ]; then
    "$LIBERO_PYTHON" - "$VALIDATION_RECEIPT" "$status" <<'PY'
import json
import os
import pathlib
import tempfile
import sys

path = pathlib.Path(sys.argv[1])
value = {
    "schema_version": "2.0",
    "artifact_role": "r05a_sampled_current_canary_cpu_publication",
    "source_job_id": os.environ.get("SOURCE_JOB_ID"),
    "source_contract_path": os.environ.get("SOURCE_CONTRACT"),
    "source_contract_sha256_external": os.environ.get("EXPECTED_SOURCE_CONTRACT_SHA256"),
    "result_path": os.environ.get("RESULT"),
    "result_sha256": None,
    "passed": False,
    "published": False,
    "errors": [f"publisher wrapper failed with exit code {sys.argv[2]}"],
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

# The afterany job must distinguish a successful source from an apparatus
# failure.  Query only the exact array task and give accounting a bounded window
# to become visible.
source_state=
source_exit=
for _ in $(seq 1 30); do
  while IFS='|' read -r job_raw state exit_code rest; do
    if [ "$job_raw" = "${SOURCE_JOB_ID}_0" ]; then
      source_state=$state
      source_exit=$exit_code
      break
    fi
  done < <(sacct -X -n -P -j "$SOURCE_JOB_ID" --format=JobIDRaw,State,ExitCode 2>/dev/null || true)
  [ -n "$source_state" ] && break
  sleep 1
done
test "$source_state" = COMPLETED || {
  echo "source H100 task did not complete successfully: state=${source_state:-missing}" >&2
  exit 3
}
test "$source_exit" = 0:0 || {
  echo "source H100 task exit code is not 0:0: ${source_exit:-missing}" >&2
  exit 3
}

JSONSCHEMA_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_jsonschema_overlay.sh)
export PYTHONPATH=$JSONSCHEMA_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero
"$LIBERO_PYTHON" "$REMOTE_REPO/main/publish_crfs_r05a_sampled_current_canary.py" \
  --payload "$PAYLOAD" \
  --host-telemetry "$HOST_TELEMETRY" \
  --gpu-samples "$GPU_SAMPLES" \
  --allocation-tests-log "$ALLOCATION_TEST_LOG" \
  --source-contract "$SOURCE_CONTRACT" \
  --expected-source-contract-sha256 "$EXPECTED_SOURCE_CONTRACT_SHA256" \
  --submission "$SUBMISSION" \
  --source-job-id "$SOURCE_JOB_ID" \
  --source-job-state "$source_state" \
  --source-exit-code "$source_exit" \
  --publisher-job-id "$SLURM_JOB_ID" \
  --result "$RESULT" \
  --receipt "$VALIDATION_RECEIPT"

test -f "$RESULT" || { echo "publisher returned without results.json" >&2; exit 4; }
test -f "$VALIDATION_RECEIPT" || { echo "publisher returned without receipt" >&2; exit 4; }
echo "source_job_id=$SOURCE_JOB_ID"
echo "result=$RESULT"
echo "result_sha256=$(sha256sum "$RESULT" | awk '{print $1}')"
echo "validation_receipt=$VALIDATION_RECEIPT"
echo "validation_receipt_sha256=$(sha256sum "$VALIDATION_RECEIPT" | awk '{print $1}')"
