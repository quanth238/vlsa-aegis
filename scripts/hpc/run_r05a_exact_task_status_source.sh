#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?exact-task regression source must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?source array job id is required}"
: "${SLURM_ARRAY_TASK_ID:?source array task id is required}"
: "${SLURM_JOB_PARTITION:?source partition provenance is required}"
: "${RUN_ID:?immutable regression run id is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed source commit is required}"
: "${SOURCE_CONTRACT:?source contract is required}"
: "${SUBMISSION:?atomic submission receipt is required}"
: "${SOURCE_MARKER:?source marker path is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

EXPECTED_RUN_ID=r05a-exact-array-task-afterany-regression-20260715a
FAILURE=${SOURCE_MARKER%/*}/source-failure.json
FAILURE_STAGE=allocation_contract

source_failure() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ ! -e "$FAILURE" ]; then
    temporary=$(mktemp "${FAILURE%/*}/.source-failure.XXXXXX") || return "$status"
    jq -n \
      --arg run_id "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" \
      --arg stage "$FAILURE_STAGE" --arg exit_code "$status" \
      --arg job_id "$SLURM_JOB_ID" --arg array_job_id "$SLURM_ARRAY_JOB_ID" \
      --arg task_id "$SLURM_ARRAY_TASK_ID" --arg host "$(hostname -s)" \
      '{
        schema_version: "1.0",
        artifact_role: "r05a_exact_task_status_source_failure",
        status: "failed_closed",
        run_id: $run_id,
        git_commit: $commit,
        failure_stage: $stage,
        exit_code: ($exit_code | tonumber),
        slurm_job_id: $job_id,
        slurm_array_job_id: $array_job_id,
        slurm_array_task_id: ($task_id | tonumber),
        host: $host,
        shell_only: true,
        gpu_allocated: false,
        python_executed: false,
        model_or_simulator_executed: false,
        scientific_claim_allowed: false,
        h100_retry_authorized: false
      }' >"$temporary" && mv "$temporary" "$FAILURE"
  fi
  return "$status"
}
trap source_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test "$RUN_ID" = "$EXPECTED_RUN_ID" || { echo "exact-task regression run id changed" >&2; exit 2; }
test "$SLURM_JOB_PARTITION" = main || { echo "source partition changed" >&2; exit 2; }
test "$SLURM_ARRAY_TASK_ID" = 0 || { echo "source requires singleton task zero" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "source must run on worker-1" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 1 || { echo "source requires one CPU" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 256 || { echo "source requires exactly 256 MiB" >&2; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-NoDevFiles}" in ''|NoDevFiles) ;; *) echo "source received a GPU" >&2; exit 2 ;; esac
test -z "${SLURM_JOB_GPUS:-}" || { echo "source received SLURM_JOB_GPUS" >&2; exit 2; }
for path in "$SOURCE_CONTRACT" "$SUBMISSION"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked source input: $path" >&2; exit 2; }
done
test ! -e "$SOURCE_MARKER" && test ! -e "$FAILURE" || { echo "source output already exists" >&2; exit 2; }
SOURCE_CONTRACT_SHA256=$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')
SUBMISSION_SHA256=$(sha256sum "$SUBMISSION" | awk '{print $1}')
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || { echo "source commit changed" >&2; exit 2; }
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || { echo "source tree is dirty" >&2; exit 2; }

FAILURE_STAGE=source_contract
jq -e \
  --arg run_id "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg array_job_id "$SLURM_ARRAY_JOB_ID" --arg marker "$SOURCE_MARKER" \
  --arg submission "$SUBMISSION" \
  '.status == "held_source_bound_before_afterany_submission"
   and .run_id == $run_id
   and .git_commit == $commit
   and .source_array_job_id == $array_job_id
   and .source_array_task_id == 0
   and .source_node == "worker-1"
   and .resources == {partition:"main",account:"normal",qos:"normal",cpus:1,host_memory_mib:256,time_limit:"00:02:00",array:"0-0%1",gpus:0,requeue:false}
   and .artifact_paths.source_marker == $marker
   and .artifact_paths.submission == $submission
   and .shell_only == true
   and .scientific_claim_allowed == false' "$SOURCE_CONTRACT" >/dev/null || {
  echo "source contract content changed" >&2; exit 2;
}

FAILURE_STAGE=source_marker
temporary=$(mktemp "${SOURCE_MARKER%/*}/.source-marker.XXXXXX")
jq -n \
  --arg run_id "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg job_id "$SLURM_JOB_ID" --arg array_job_id "$SLURM_ARRAY_JOB_ID" \
  --arg task_id "$SLURM_ARRAY_TASK_ID" --arg host "$(hostname -s)" \
  --arg source_contract "$SOURCE_CONTRACT" \
  --arg source_contract_sha "$SOURCE_CONTRACT_SHA256" \
  --arg submission "$SUBMISSION" --arg submission_sha "$SUBMISSION_SHA256" \
  --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_exact_task_status_source_marker",
    status: "body_completed_before_process_exit",
    run_id: $run_id,
    git_commit: $commit,
    slurm_job_id: $job_id,
    slurm_array_job_id: $array_job_id,
    slurm_array_task_id: ($task_id | tonumber),
    exact_source_task_id: ($array_job_id + "_0"),
    host: $host,
    source_contract_path: $source_contract,
    source_contract_sha256: $source_contract_sha,
    submission_path: $submission,
    submission_sha256: $submission_sha,
    shell_only: true,
    gpu_allocated: false,
    python_executed: false,
    checkpoint_loaded: false,
    model_inference_executed: false,
    simulator_steps_executed: 0,
    teacher_searches_executed: 0,
    training_executed: false,
    scientific_claim_allowed: false,
    h100_retry_authorized: false,
    timestamp_utc: $timestamp
  }' >"$temporary"
mv "$temporary" "$SOURCE_MARKER"
echo "source_marker=$SOURCE_MARKER"
echo "source_marker_sha256=$(sha256sum "$SOURCE_MARKER" | awk '{print $1}')"
