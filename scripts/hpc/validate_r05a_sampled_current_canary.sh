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

APPARATUS_CONFIG=$REMOTE_REPO/configs/experiments/r05a_sampled_current_canary_apparatus.json
SOURCE_STATUS_HELPER=$REMOTE_REPO/scripts/hpc/lib/slurm_exact_array_task_status.sh

mkdir -p "$(dirname "$VALIDATION_RECEIPT")"
export PUBLISHER_FAILURE_STAGE=wrapper_preflight
export SOURCE_TASK_ID=${SOURCE_JOB_ID}_0
export SOURCE_TASK_STATE=
export SOURCE_TASK_EXIT_CODE=
export SOURCE_QUERY_STATUS=
fallback_receipt() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ ! -f "$VALIDATION_RECEIPT" ]; then
    temporary=$(mktemp "$(dirname "$VALIDATION_RECEIPT")/.cpu-afterany-validation.XXXXXX") || return "$status"
    jq -n \
      --arg publisher_job_id "$SLURM_JOB_ID" \
      --arg source_job_id "$SOURCE_JOB_ID" \
      --arg source_task_id "$SOURCE_TASK_ID" \
      --arg source_state "$SOURCE_TASK_STATE" \
      --arg source_exit "$SOURCE_TASK_EXIT_CODE" \
      --arg query_status "$SOURCE_QUERY_STATUS" \
      --arg failure_stage "$PUBLISHER_FAILURE_STAGE" \
      --arg source_contract "$SOURCE_CONTRACT" \
      --arg source_contract_sha "$EXPECTED_SOURCE_CONTRACT_SHA256" \
      --arg result "$RESULT" \
      --arg wrapper_status "$status" \
      '{
        schema_version: "2.0",
        artifact_role: "r05a_sampled_current_canary_cpu_publication",
        publisher_job_id: $publisher_job_id,
        source_job_id: $source_job_id,
        source_task_id: $source_task_id,
        source_job_state: (if $source_state == "" then null else $source_state end),
        source_exit_code: (if $source_exit == "" then null else $source_exit end),
        source_query_status: (if $query_status == "" then null else ($query_status | tonumber) end),
        failure_stage: $failure_stage,
        source_contract_path: $source_contract,
        source_contract_sha256_external: $source_contract_sha,
        result_path: $result,
        result_sha256: null,
        passed: false,
        published: false,
        errors: ["publisher wrapper failed with exit code " + $wrapper_status],
        scientific_claim_allowed: false,
        probe_training_authorized: false
      }' >"$temporary" && mv "$temporary" "$VALIDATION_RECEIPT"
  fi
  return "$status"
}
trap fallback_receipt EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
test -f "$APPARATUS_CONFIG" || { echo "publisher apparatus config is missing" >&2; exit 2; }
test -f "$SOURCE_STATUS_HELPER" && test ! -L "$SOURCE_STATUS_HELPER" || {
  echo "exact source-task status helper is missing or symlinked" >&2
  exit 2
}
. "$SOURCE_STATUS_HELPER"
REGISTERED_RUN_ID=$(jq -er '.execution_release.run_id' "$APPARATUS_CONFIG")
ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$APPARATUS_CONFIG")
SOURCE_RUN_ID=$(jq -er '.run_id' "$SOURCE_CONTRACT")
test "$SOURCE_RUN_ID" = "$REGISTERED_RUN_ID" || {
  echo "publisher source run ID differs from the exact execution release" >&2
  exit 2
}
jq -e --arg run "$REGISTERED_RUN_ID" --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" \
  '.ready_to_run == true and .blocked_on == []
   and ((.execution_release | keys | sort) == (["schema_version","artifact_role","decision_artifact","accepted_implementation_commit","run_id","single_submission","source_host","resources","release_only_parent_required","allowed_release_diff_paths","automatic_resubmission_allowed","automatic_next_experiment_allowed"] | sort))
   and .execution_release.schema_version == "1.0"
   and .execution_release.artifact_role == "r05a_single_canary_execution_release"
   and .execution_release.decision_artifact == "docs/decisions/0037-require-exact-single-canary-release-identity.md"
   and .execution_release.accepted_implementation_commit == $implementation
   and .execution_release.run_id == $run
   and .execution_release.single_submission == true
   and .execution_release.source_host == "worker-1"
   and .execution_release.resources == {partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}
   and .execution_release.release_only_parent_required == true
   and .execution_release.allowed_release_diff_paths == ["configs/experiments/r05a_sampled_current_canary_apparatus.json","docs/decisions/0037-require-exact-single-canary-release-identity.md"]
   and .execution_release.automatic_resubmission_allowed == false
   and .execution_release.automatic_next_experiment_allowed == false' \
  "$APPARATUS_CONFIG" >/dev/null || {
    echo "publisher execution release contract changed" >&2
    exit 2
  }
test "$(git -C "$REMOTE_REPO" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$EXPECTED_GIT_COMMIT" || {
  echo "publisher origin release ref changed" >&2
  exit 2
}
test "$(git -C "$REMOTE_REPO" rev-list --parents -n 1 "$EXPECTED_GIT_COMMIT")" = "$EXPECTED_GIT_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || {
  echo "publisher release parent changed" >&2
  exit 2
}
APPARATUS_CONFIG_SHA256=$(sha256sum "$APPARATUS_CONFIG" | awk '{print $1}')
jq -e --arg run "$REGISTERED_RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" --arg config_sha "$APPARATUS_CONFIG_SHA256" \
  '.run_id == $run
   and .git_commit == $commit
   and .repository_file_sha256["configs/experiments/r05a_sampled_current_canary_apparatus.json"] == $config_sha' \
  "$SOURCE_CONTRACT" >/dev/null || {
    echo "publisher source contract does not bind the execution release" >&2
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

# The afterany job must distinguish a successful source from an apparatus
# failure. Query the exact display task ID; JobIDRaw is only the parent numeric
# allocation on VinUni and therefore cannot identify singleton task 0.
export PUBLISHER_FAILURE_STAGE=exact_source_task_accounting
source_record=
if source_record=$(crfs_wait_for_exact_completed_array_task "$SOURCE_JOB_ID" 30 1); then
  source_query_status=0
else
  source_query_status=$?
fi
export SOURCE_QUERY_STATUS=$source_query_status
IFS='|' read -r source_state source_exit <<EOF
$source_record
EOF
export SOURCE_TASK_STATE=$source_state
export SOURCE_TASK_EXIT_CODE=$source_exit
if [ "$source_query_status" -ne 0 ] || [ "$source_state" != COMPLETED ]; then
  echo "source H100 task did not complete successfully: state=${source_state:-missing}" >&2
  exit 3
fi
test "$source_exit" = 0:0 || {
  echo "source H100 task exit code is not 0:0: ${source_exit:-missing}" >&2
  exit 3
}

export PUBLISHER_FAILURE_STAGE=publication_runtime_preflight
JSONSCHEMA_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_jsonschema_overlay.sh)
export PYTHONPATH=$JSONSCHEMA_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero
export PUBLISHER_FAILURE_STAGE=python_publication
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
