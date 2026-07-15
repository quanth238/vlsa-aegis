#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?exact-task regression validator must run inside Slurm}"
: "${SLURM_JOB_PARTITION:?validator partition provenance is required}"
: "${RUN_ID:?immutable regression run id is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed source commit is required}"
: "${SOURCE_JOB_ID:?source array parent job id is required}"
: "${SOURCE_CONTRACT:?source contract is required}"
: "${EXPECTED_SOURCE_CONTRACT_SHA256:?source contract digest is required}"
: "${SUBMISSION:?submission receipt is required}"
: "${SOURCE_MARKER:?source marker is required}"
: "${RESULT:?result path is required}"
: "${VALIDATION_RECEIPT:?validation receipt path is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

EXPECTED_RUN_ID=r05a-exact-array-task-afterany-regression-20260715a
HELPER=$REMOTE_REPO/scripts/hpc/lib/slurm_exact_array_task_status.sh
PRODUCTION_VALIDATOR=$REMOTE_REPO/scripts/hpc/validate_r05a_sampled_current_canary.sh
CANDIDATE=${RESULT%/*}/.results.pending.json
FAILURE_STAGE=wrapper_preflight
SOURCE_TASK_ID=${SOURCE_JOB_ID}_0
SOURCE_TASK_STATE=
SOURCE_TASK_EXIT_CODE=
SOURCE_QUERY_STATUS=
RESULT_CREATED=false

validation_failure() {
  status=$?
  trap - EXIT INT TERM
  rm -f "$CANDIDATE"
  if [ "$RESULT_CREATED" = true ] && [ ! -e "$VALIDATION_RECEIPT" ]; then
    rm -f "$RESULT"
  fi
  if [ "$status" -ne 0 ] && [ ! -e "$VALIDATION_RECEIPT" ]; then
    temporary=$(mktemp "${VALIDATION_RECEIPT%/*}/.validation.XXXXXX") || return "$status"
    jq -n \
      --arg run_id "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" \
      --arg validator_job_id "$SLURM_JOB_ID" --arg source_job_id "$SOURCE_JOB_ID" \
      --arg source_task_id "$SOURCE_TASK_ID" --arg source_state "$SOURCE_TASK_STATE" \
      --arg source_exit "$SOURCE_TASK_EXIT_CODE" --arg query_status "$SOURCE_QUERY_STATUS" \
      --arg stage "$FAILURE_STAGE" --arg exit_code "$status" \
      '{
        schema_version: "1.0",
        artifact_role: "r05a_exact_task_status_afterany_validation",
        status: "failed_closed",
        run_id: $run_id,
        git_commit: $commit,
        validator_job_id: $validator_job_id,
        source_job_id: $source_job_id,
        source_task_id: $source_task_id,
        source_job_state: (if $source_state == "" then null else $source_state end),
        source_exit_code: (if $source_exit == "" then null else $source_exit end),
        source_query_status: (if $query_status == "" then null else ($query_status | tonumber) end),
        failure_stage: $stage,
        wrapper_exit_code: ($exit_code | tonumber),
        passed: false,
        published: false,
        shell_only: true,
        gpu_allocated: false,
        python_executed: false,
        model_or_simulator_executed: false,
        scientific_claim_allowed: false,
        h100_retry_authorized: false
      }' >"$temporary" && mv "$temporary" "$VALIDATION_RECEIPT"
  fi
  return "$status"
}
mkdir -p "${VALIDATION_RECEIPT%/*}"
trap validation_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test "$RUN_ID" = "$EXPECTED_RUN_ID" || { echo "exact-task regression run id changed" >&2; exit 2; }
test "$SLURM_JOB_PARTITION" = main || { echo "validator partition changed" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "validator must run on worker-1" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 1 || { echo "validator requires one CPU" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 256 || { echo "validator requires exactly 256 MiB" >&2; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-NoDevFiles}" in ''|NoDevFiles) ;; *) echo "validator received a GPU" >&2; exit 2 ;; esac
test -z "${SLURM_JOB_GPUS:-}" || { echo "validator received SLURM_JOB_GPUS" >&2; exit 2; }
case "$SOURCE_JOB_ID" in *[!0-9]*|'') echo "invalid source job id" >&2; exit 2 ;; esac
for path in "$HELPER" "$PRODUCTION_VALIDATOR" "$SOURCE_CONTRACT" "$SUBMISSION" "$SOURCE_MARKER"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked validator input: $path" >&2; exit 2; }
done
test ! -e "$RESULT" && test ! -e "$CANDIDATE" && test ! -e "$VALIDATION_RECEIPT" || {
  echo "immutable regression publication already exists" >&2; exit 2;
}
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || { echo "validator commit changed" >&2; exit 2; }
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || { echo "validator source tree is dirty" >&2; exit 2; }
test "$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')" = "$EXPECTED_SOURCE_CONTRACT_SHA256" || {
  echo "source contract digest changed" >&2; exit 2;
}

FAILURE_STAGE=exact_source_task_accounting
. "$HELPER"
source_record=
if source_record=$(crfs_wait_for_exact_completed_array_task "$SOURCE_JOB_ID" 30 1); then
  source_query_status=0
else
  source_query_status=$?
fi
SOURCE_QUERY_STATUS=$source_query_status
IFS='|' read -r source_state source_exit <<EOF
$source_record
EOF
SOURCE_TASK_STATE=$source_state
SOURCE_TASK_EXIT_CODE=$source_exit
test "$source_query_status" -eq 0 || { echo "exact source task query failed: status=$source_query_status" >&2; exit 3; }
test "$source_state" = COMPLETED && test "$source_exit" = 0:0 || {
  echo "exact source task did not complete successfully" >&2; exit 3;
}

FAILURE_STAGE=artifact_validation
SOURCE_CONTRACT_SHA256=$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')
SUBMISSION_SHA256=$(sha256sum "$SUBMISSION" | awk '{print $1}')
SOURCE_MARKER_SHA256=$(sha256sum "$SOURCE_MARKER" | awk '{print $1}')
HELPER_SHA256=$(sha256sum "$HELPER" | awk '{print $1}')
PRODUCTION_VALIDATOR_SHA256=$(sha256sum "$PRODUCTION_VALIDATOR" | awk '{print $1}')
jq -e \
  --arg run_id "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg source_job_id "$SOURCE_JOB_ID" --arg source_task_id "$SOURCE_TASK_ID" \
  --arg source_contract "$SOURCE_CONTRACT" --arg source_contract_sha "$SOURCE_CONTRACT_SHA256" \
  --arg submission "$SUBMISSION" --arg submission_sha "$SUBMISSION_SHA256" \
  '.status == "body_completed_before_process_exit"
   and .run_id == $run_id
   and .git_commit == $commit
   and .slurm_array_job_id == $source_job_id
   and .slurm_array_task_id == 0
   and .exact_source_task_id == $source_task_id
   and .host == "worker-1"
   and .source_contract_path == $source_contract
   and .source_contract_sha256 == $source_contract_sha
   and .submission_path == $submission
   and .submission_sha256 == $submission_sha
   and .shell_only == true
   and .gpu_allocated == false
   and .python_executed == false
   and .model_inference_executed == false
   and .simulator_steps_executed == 0
   and .teacher_searches_executed == 0
   and .training_executed == false
   and .scientific_claim_allowed == false' "$SOURCE_MARKER" >/dev/null || {
  echo "source marker content changed" >&2; exit 4;
}
jq -e \
  --arg run_id "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg source_job_id "$SOURCE_JOB_ID" --arg validator_job_id "$SLURM_JOB_ID" \
  --arg source_contract_sha "$SOURCE_CONTRACT_SHA256" \
  '.status == "held_source_and_afterany_registered"
   and .run_id == $run_id
   and .git_commit == $commit
   and .source_array_job_id == $source_job_id
   and .source_array_task_id == 0
   and .validator_job_id == $validator_job_id
   and .dependency == ("afterany:" + $source_job_id)
   and .source_contract_sha256 == $source_contract_sha
   and .released_at_receipt_time == false
   and .shell_only == true
   and .scientific_claim_allowed == false' "$SUBMISSION" >/dev/null || {
  echo "submission receipt content changed" >&2; exit 4;
}
jq -e \
  --arg helper_sha "$HELPER_SHA256" --arg production_validator_sha "$PRODUCTION_VALIDATOR_SHA256" \
  '.bound_file_sha256["scripts/hpc/lib/slurm_exact_array_task_status.sh"] == $helper_sha
   and .bound_file_sha256["scripts/hpc/validate_r05a_sampled_current_canary.sh"] == $production_validator_sha' \
  "$SOURCE_CONTRACT" >/dev/null || { echo "production accounting bindings changed" >&2; exit 4; }

FAILURE_STAGE=result_publication
temporary=$(mktemp "${RESULT%/*}/.results.XXXXXX")
jq -n \
  --arg run_id "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg source_job_id "$SOURCE_JOB_ID" --arg source_task_id "$SOURCE_TASK_ID" \
  --arg source_state "$source_state" --arg source_exit "$source_exit" \
  --arg validator_job_id "$SLURM_JOB_ID" \
  --arg source_contract "$SOURCE_CONTRACT" --arg source_contract_sha "$SOURCE_CONTRACT_SHA256" \
  --arg submission "$SUBMISSION" --arg submission_sha "$SUBMISSION_SHA256" \
  --arg source_marker "$SOURCE_MARKER" --arg source_marker_sha "$SOURCE_MARKER_SHA256" \
  --arg helper_sha "$HELPER_SHA256" --arg production_validator_sha "$PRODUCTION_VALIDATOR_SHA256" \
  --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_exact_task_status_afterany_regression",
    status: "passed",
    run_id: $run_id,
    git_commit: $commit,
    source_job_id: $source_job_id,
    exact_source_task_id: $source_task_id,
    observed_source_state: $source_state,
    observed_source_exit_code: $source_exit,
    source_query_status: 0,
    validator_job_id: $validator_job_id,
    dependency: ("afterany:" + $source_job_id),
    source_host: "worker-1",
    validator_host: "worker-1",
    resources: {source:{cpus:1,host_memory_mib:256,gpus:0,time_limit:"00:02:00",array:"0-0%1"},validator:{cpus:1,host_memory_mib:256,gpus:0,time_limit:"00:02:00"}},
    source_contract_path: $source_contract,
    source_contract_sha256: $source_contract_sha,
    submission_path: $submission,
    submission_sha256: $submission_sha,
    source_marker_path: $source_marker,
    source_marker_sha256: $source_marker_sha,
    production_helper_sha256: $helper_sha,
    production_validator_sha256: $production_validator_sha,
    shell_only: true,
    gpu_allocated: false,
    python_executed: false,
    checkpoint_loaded: false,
    model_inference_executed: false,
    simulator_steps_executed: 0,
    teacher_searches_executed: 0,
    training_executed: false,
    scientific_claim_allowed: false,
    h100_retry_authorized_by_this_artifact_alone: false,
    passed: true,
    timestamp_utc: $timestamp
  }' >"$temporary"
mv "$temporary" "$CANDIDATE"
jq -e '.status == "passed" and .passed == true and .scientific_claim_allowed == false' "$CANDIDATE" >/dev/null
RESULT_SHA256=$(sha256sum "$CANDIDATE" | awk '{print $1}')
mv "$CANDIDATE" "$RESULT"
RESULT_CREATED=true
temporary=$(mktemp "${VALIDATION_RECEIPT%/*}/.validation.XXXXXX")
jq -n \
  --arg run_id "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg validator_job_id "$SLURM_JOB_ID" --arg source_job_id "$SOURCE_JOB_ID" \
  --arg source_task_id "$SOURCE_TASK_ID" --arg source_state "$source_state" \
  --arg source_exit "$source_exit" --arg result "$RESULT" --arg result_sha "$RESULT_SHA256" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_exact_task_status_afterany_validation",
    status: "passed",
    run_id: $run_id,
    git_commit: $commit,
    validator_job_id: $validator_job_id,
    source_job_id: $source_job_id,
    source_task_id: $source_task_id,
    source_job_state: $source_state,
    source_exit_code: $source_exit,
    source_query_status: 0,
    failure_stage: null,
    result_path: $result,
    result_sha256: $result_sha,
    passed: true,
    published: true,
    shell_only: true,
    gpu_allocated: false,
    python_executed: false,
    scientific_claim_allowed: false,
    h100_retry_authorized: false
  }' >"$temporary"
mv "$temporary" "$VALIDATION_RECEIPT"
echo "result=$RESULT"
echo "result_sha256=$RESULT_SHA256"
echo "validation_receipt=$VALIDATION_RECEIPT"
echo "validation_receipt_sha256=$(sha256sum "$VALIDATION_RECEIPT" | awk '{print $1}')"
