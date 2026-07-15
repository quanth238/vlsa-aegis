#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?capability gate must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?capability gate requires its array job id}"
: "${SLURM_ARRAY_TASK_ID:?capability gate requires array row zero}"
: "${SLURM_JOB_PARTITION:?capability gate requires partition provenance}"
: "${RUN_ID:?immutable capability run id is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed commit is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"

EXPECTED_RUN_ID=r05a-cgroup-v2-current-capability-20260715a
HELPER=$REMOTE_REPO/scripts/hpc/lib/cgroup_v2_current_capability.sh
RUNNER=$REMOTE_REPO/scripts/hpc/run_r05a_cgroup_v2_current_capability.sh
SUBMITTER=$REMOTE_REPO/scripts/hpc/submit_r05a_cgroup_v2_current_capability.sh
SBATCH=$REMOTE_REPO/slurm/r05a_cgroup_v2_current_capability_cpu.sbatch
ADR=$REMOTE_REPO/docs/decisions/0035-preregister-r05a-sampled-current-capability-gate.md
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
SUBMISSION=$RUN_ROOT/submission.json
SIDECAR=$RUN_ROOT/cgroup-v2-current-capability.tsv
RESULT=$RUN_ROOT/results.json
RESULT_CANDIDATE=$RUN_ROOT/.results.pending.json
FAILURE=$RUN_ROOT/failure.json
FAILURE_STAGE=allocation_contract

cleanup() {
  status=$?
  trap - EXIT INT TERM
  rm -f "$RESULT_CANDIDATE"
  if [ "$status" -ne 0 ] && [ -d "$RUN_ROOT" ] && [ ! -e "$FAILURE" ]; then
    temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX") || return "$status"
    jq -n \
      --arg run_id "$RUN_ID" \
      --arg stage "$FAILURE_STAGE" \
      --arg exit_code "$status" \
      --arg job_id "$SLURM_JOB_ID" \
      --arg array_job_id "$SLURM_ARRAY_JOB_ID" \
      --arg task_id "$SLURM_ARRAY_TASK_ID" \
      --arg host "$(hostname -s)" \
      '{
        schema_version: "1.0",
        artifact_role: "r05a_cgroup_v2_current_capability_failure",
        status: "failed_closed",
        run_id: $run_id,
        failure_stage: $stage,
        exit_code: ($exit_code | tonumber),
        slurm_job_id: $job_id,
        slurm_array_job_id: $array_job_id,
        slurm_array_task_id: ($task_id | tonumber),
        host: $host,
        shell_only: true,
        checkpoint_loaded: false,
        policy_server_started: false,
        model_inference_executed: false,
        simulator_steps_executed: 0,
        teacher_searches_executed: 0,
        sampled_current_never_labeled_peak: true,
        scientific_claim_allowed: false,
        retry_c_authorized: false,
        ift01_authorized: false,
        probe_training_authorized: false
      }' >"$temporary" && mv "$temporary" "$FAILURE"
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe capability RUN_ID" >&2; exit 2 ;; esac
test "$RUN_ID" = "$EXPECTED_RUN_ID" || { echo "capability run id changed" >&2; exit 2; }
test "$SLURM_JOB_PARTITION" = main || { echo "capability gate is frozen to main" >&2; exit 2; }
test "$SLURM_ARRAY_TASK_ID" = 0 || { echo "capability gate requires row zero" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "capability gate must run on worker-1" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 1 || { echo "capability gate requires one CPU" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 256 || { echo "capability gate requires 256 MiB" >&2; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-NoDevFiles}" in ''|NoDevFiles) ;; *) echo "capability gate received a GPU" >&2; exit 2 ;; esac
test -z "${SLURM_JOB_GPUS:-}" || { echo "capability gate received SLURM_JOB_GPUS" >&2; exit 2; }
test -f "$SUBMISSION" || { echo "held submission receipt is missing" >&2; exit 2; }
test ! -e "$SIDECAR" && test ! -e "$RESULT" && test ! -e "$FAILURE" || {
  echo "immutable capability run already contains execution output" >&2
  exit 2
}
for path in "$HELPER" "$RUNNER" "$SUBMITTER" "$SBATCH" "$ADR"; do
  test -f "$path" || { echo "missing capability source: $path" >&2; exit 2; }
done

SUBMISSION_SHA256=$(sha256sum "$SUBMISSION" | awk '{print $1}')
HELPER_SHA256=$(sha256sum "$HELPER" | awk '{print $1}')
RUNNER_SHA256=$(sha256sum "$RUNNER" | awk '{print $1}')
SUBMITTER_SHA256=$(sha256sum "$SUBMITTER" | awk '{print $1}')
SBATCH_SHA256=$(sha256sum "$SBATCH" | awk '{print $1}')
ADR_SHA256=$(sha256sum "$ADR" | awk '{print $1}')
jq -e \
  --arg run_id "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg array_job_id "$SLURM_ARRAY_JOB_ID" \
  --arg helper_sha "$HELPER_SHA256" \
  --arg runner_sha "$RUNNER_SHA256" \
  --arg submitter_sha "$SUBMITTER_SHA256" \
  --arg slurm_sha "$SBATCH_SHA256" \
  --arg adr_sha "$ADR_SHA256" \
  '.status == "reviewed_job_held_and_receipted"
   and .run_id == $run_id
   and .git_commit == $commit
   and .slurm_array_job_id == $array_job_id
   and .slurm_array_task_id == 0
   and .source_node == "worker-1"
   and .partition == "main"
   and .account == "normal"
   and .qos == "normal"
   and .time_limit == "00:02:00"
   and .requeue == false
   and .requested_cpus == 1
   and .requested_host_memory_mib == 256
   and .requested_gpus == 0
   and .helper_sha256 == $helper_sha
   and .runner_sha256 == $runner_sha
   and .submitter_sha256 == $submitter_sha
   and .slurm_sha256 == $slurm_sha
   and .adr_sha256 == $adr_sha
   and .scientific_claim_allowed == false' "$SUBMISSION" >/dev/null || {
  echo "capability submission receipt contract changed" >&2
  exit 2
}

test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || {
  echo "capability reviewed commit mismatch" >&2
  exit 2
}
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || {
  echo "capability gate requires a clean remote tree" >&2
  exit 2
}

. "$HELPER"
FAILURE_STAGE=bounded_cgroup_v2_capability_probe
crfs_probe_cgroup_v2_current_capability \
  /proc/self/cgroup /proc/self/mountinfo /proc/sys/kernel/osrelease \
  "$SIDECAR" "$SLURM_ARRAY_JOB_ID" 20 0.1
test -f "$SIDECAR" || { echo "capability probe wrote no sidecar" >&2; exit 3; }

single_field() {
  local section=$1 key=$2
  awk -F '\t' -v section="$section" -v key="$key" '
    $1 == section && $2 == key { value=$3; found++ }
    END { if (found != 1) exit 1; print value }
  ' "$SIDECAR"
}
test "$(single_field record status)" = completed || { echo "capability sidecar is incomplete" >&2; exit 3; }
OUTCOME=$(single_field summary outcome)
case "$OUTCOME" in
  sampled_current_contract_supported|native_memory_peak_available|sampled_current_contract_unsupported) ;;
  *) echo "capability outcome changed" >&2; exit 3 ;;
esac
KERNEL_OSRELEASE=$(single_field record kernel_osrelease)
MEMBERSHIP_PATH=$(single_field record membership_path)
MOUNT_ROOT=$(single_field record mount_root)
MOUNT_POINT=$(single_field record mount_point)
JOB_SCOPE_PATH=$(single_field summary job_scope_path)
JOB_CURRENT_STATE=$(single_field summary job_memory_current_state)
JOB_MAX_STATE=$(single_field summary job_memory_max_state)
JOB_MAX_VALUE=$(single_field summary job_memory_max_value)
JOB_EVENTS_STATE=$(single_field summary job_memory_events_state)
JOB_LOCAL_EVENTS_STATE=$(single_field summary job_memory_events_local_state)
EVENTS_HIERARCHY_STATE=$(single_field summary memory_events_hierarchy_state)
SAMPLE_COUNT=$(single_field summary sample_count_observed)
SAMPLE_INTERVAL=$(single_field summary sample_interval_requested_seconds)
SAMPLE_MAX_GAP_NS=$(single_field summary sample_max_gap_ns)
SAMPLED_HIGH_WATER=$(single_field summary sampled_memory_current_high_water_bytes)
EVENT_MAX_DELTA=$(single_field summary memory_events_max_delta)
EVENT_OOM_DELTA=$(single_field summary memory_events_oom_delta)
EVENT_OOM_KILL_DELTA=$(single_field summary memory_events_oom_kill_delta)
SUPPORTED=$(single_field summary sampled_current_contract_capability_supported)
PEAK_STATE=$(awk -F '\t' '$1 == "capability" && $2 == "job_memory_peak" { print $3; found++ } END { if (found != 1) exit 1 }' "$SIDECAR")
PEAK_VALUE=$(awk -F '\t' '$1 == "capability" && $2 == "job_memory_peak" { print $4; found++ } END { if (found != 1) exit 1 }' "$SIDECAR")
for value in "$SAMPLE_COUNT" "$SAMPLE_MAX_GAP_NS" "$SAMPLED_HIGH_WATER"; do
  case "$value" in *[!0-9]*|'') echo "capability numeric summary is invalid" >&2; exit 3 ;; esac
done
for value in "$EVENT_MAX_DELTA" "$EVENT_OOM_DELTA" "$EVENT_OOM_KILL_DELTA"; do
  case "$value" in
    '') ;;
    -*) unsigned=${value#-}; case "$unsigned" in *[!0-9]*|'') echo "capability event delta is invalid" >&2; exit 3 ;; esac ;;
    *[!0-9]*) echo "capability event delta is invalid" >&2; exit 3 ;;
  esac
done

RAW_SAMPLE_AUDIT=$(awk -F '\t' '
  BEGIN { count=0; high=0; max_gap=0; increasing="true" }
  $1 == "sample" {
    if (NF != 4 || $2 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/ || $4 !~ /^[0-9]+$/) exit 2
    if (($2 + 0) != count) exit 2
    timestamp=$3 + 0
    sample=$4 + 0
    if (sample > high) high=sample
    if (count > 0) {
      gap=timestamp - previous
      if (gap <= 0) increasing="false"
      if (gap > max_gap) max_gap=gap
    }
    previous=timestamp
    count++
  }
  END { printf "%d\t%.0f\t%.0f\t%s\n", count, high, max_gap, increasing }
' "$SIDECAR") || { echo "capability raw sample trace is malformed" >&2; exit 3; }
IFS=$'\t' read -r RAW_SAMPLE_COUNT RAW_SAMPLE_HIGH_WATER RAW_SAMPLE_MAX_GAP_NS RAW_TIMESTAMPS_INCREASING <<<"$RAW_SAMPLE_AUDIT"
test "$RAW_SAMPLE_COUNT" = "$SAMPLE_COUNT" || { echo "capability sample count summary mismatch" >&2; exit 3; }
test "$RAW_SAMPLE_HIGH_WATER" = "$SAMPLED_HIGH_WATER" || { echo "capability sample high-water summary mismatch" >&2; exit 3; }
test "$RAW_SAMPLE_MAX_GAP_NS" = "$SAMPLE_MAX_GAP_NS" || { echo "capability sample gap summary mismatch" >&2; exit 3; }
case "$RAW_TIMESTAMPS_INCREASING" in true|false) ;; *) echo "capability raw timestamp state is invalid" >&2; exit 3 ;; esac

single_event_field() {
  local phase=$1 key=$2
  awk -F '\t' -v phase="$phase" -v key="$key" '
    $1 == "event" && $2 == phase && $3 == key { value=$4; found++ }
    END { if (found != 1) exit 1; print value }
  ' "$SIDECAR"
}
for key in max oom oom_kill; do
  before=$(single_event_field before "$key") || { echo "capability raw before-event row is invalid" >&2; exit 3; }
  after=$(single_event_field after "$key") || { echo "capability raw after-event row is invalid" >&2; exit 3; }
  case "$key" in
    max) summary_delta=$EVENT_MAX_DELTA ;;
    oom) summary_delta=$EVENT_OOM_DELTA ;;
    oom_kill) summary_delta=$EVENT_OOM_KILL_DELTA ;;
  esac
  case "$before:$after" in
    *[!0-9:]*|:*|*:) raw_delta= ;;
    *) raw_delta=$((after - before)) ;;
  esac
  test "$raw_delta" = "$summary_delta" || { echo "capability event delta summary mismatch for $key" >&2; exit 3; }
done

case "$SUPPORTED" in true|false) ;; *) echo "capability support flag is invalid" >&2; exit 3 ;; esac
case "$EVENTS_HIERARCHY_STATE" in
  hierarchical|local_only_memory_localevents) ;;
  *) echo "capability event-hierarchy state is invalid" >&2; exit 3 ;;
esac
test "$SAMPLE_INTERVAL" = 0.1 || { echo "capability sample interval changed" >&2; exit 3; }
if [ "$OUTCOME" = sampled_current_contract_supported ]; then
  test "$SUPPORTED" = true || { echo "supported outcome lost its capability flag" >&2; exit 3; }
  test "$PEAK_STATE" = missing || { echo "sampled route cannot replace an available native peak" >&2; exit 3; }
  test "$JOB_CURRENT_STATE" = readable_integer || { echo "supported route lost memory.current" >&2; exit 3; }
  test "$JOB_MAX_STATE" = readable_integer || { echo "supported route has no hard ceiling" >&2; exit 3; }
  test "$JOB_MAX_VALUE" -gt 0 || { echo "supported route has a nonpositive hard ceiling" >&2; exit 3; }
  test "$JOB_EVENTS_STATE" = readable_flat_keys || { echo "supported route has no hierarchical events" >&2; exit 3; }
  test "$EVENTS_HIERARCHY_STATE" = hierarchical || { echo "supported route has local-only memory.events" >&2; exit 3; }
  test "$SAMPLE_COUNT" = 20 || { echo "supported route has an incomplete sample trace" >&2; exit 3; }
  test "$RAW_TIMESTAMPS_INCREASING" = true || { echo "supported route has non-increasing timestamps" >&2; exit 3; }
  test "$SAMPLED_HIGH_WATER" -gt 0 || { echo "supported route has no positive memory.current observation" >&2; exit 3; }
  test "$SAMPLE_MAX_GAP_NS" -le 500000000 || { echo "supported route exceeded the registered sample gap" >&2; exit 3; }
  for value in "$EVENT_MAX_DELTA" "$EVENT_OOM_DELTA" "$EVENT_OOM_KILL_DELTA"; do
    test "$value" = 0 || { echo "supported outcome observed a limit or OOM event" >&2; exit 3; }
  done
elif [ "$OUTCOME" = native_memory_peak_available ]; then
  test "$PEAK_STATE" = readable_integer || { echo "native peak outcome has no readable peak" >&2; exit 3; }
  test "$PEAK_VALUE" -gt 0 || { echo "native peak outcome has no positive peak" >&2; exit 3; }
  test "$SUPPORTED" = false || { echo "native peak outcome cannot select sampled fallback" >&2; exit 3; }
else
  test "$SUPPORTED" = false || { echo "unsupported outcome claims sampled support" >&2; exit 3; }
fi
if [ "$PEAK_STATE" = readable_integer ]; then
  case "$PEAK_VALUE" in *[!0-9]*|'') echo "native peak capability value is invalid" >&2; exit 3 ;; esac
fi

SIDECAR_SHA256=$(sha256sum "$SIDECAR" | awk '{print $1}')
FAILURE_STAGE=atomic_result_publication
jq -n \
  --arg run_id "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg submission "$SUBMISSION" \
  --arg submission_sha "$SUBMISSION_SHA256" \
  --arg sidecar "$SIDECAR" \
  --arg sidecar_sha "$SIDECAR_SHA256" \
  --arg job_id "$SLURM_JOB_ID" \
  --arg array_job_id "$SLURM_ARRAY_JOB_ID" \
  --arg host "$(hostname -s)" \
  --arg outcome "$OUTCOME" \
  --arg kernel "$KERNEL_OSRELEASE" \
  --arg membership "$MEMBERSHIP_PATH" \
  --arg mount_root "$MOUNT_ROOT" \
  --arg mount_point "$MOUNT_POINT" \
  --arg job_scope "$JOB_SCOPE_PATH" \
  --arg current_state "$JOB_CURRENT_STATE" \
  --arg max_state "$JOB_MAX_STATE" \
  --arg max_value "$JOB_MAX_VALUE" \
  --arg events_state "$JOB_EVENTS_STATE" \
  --arg local_events_state "$JOB_LOCAL_EVENTS_STATE" \
  --arg events_hierarchy_state "$EVENTS_HIERARCHY_STATE" \
  --arg sample_count "$SAMPLE_COUNT" \
  --arg max_gap "$SAMPLE_MAX_GAP_NS" \
  --arg sampled_high "$SAMPLED_HIGH_WATER" \
  --arg raw_timestamps_increasing "$RAW_TIMESTAMPS_INCREASING" \
  --arg event_max "$EVENT_MAX_DELTA" \
  --arg event_oom "$EVENT_OOM_DELTA" \
  --arg event_oom_kill "$EVENT_OOM_KILL_DELTA" \
  --arg supported "$SUPPORTED" \
  --arg peak_state "$PEAK_STATE" \
  --arg peak_value "$PEAK_VALUE" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_cgroup_v2_current_capability_result",
    status: "completed",
    run_id: $run_id,
    git_commit: $commit,
    git_dirty: false,
    submission_receipt_path: $submission,
    submission_receipt_sha256: $submission_sha,
    diagnostic_path: $sidecar,
    diagnostic_sha256: $sidecar_sha,
    host: $host,
    slurm_job_id: $job_id,
    slurm_array_job_id: $array_job_id,
    slurm_array_task_id: 0,
    partition: "main",
    account: "normal",
    qos: "normal",
    time_limit: "00:02:00",
    requeue: false,
    requested_cpus: 1,
    requested_host_memory_mib: 256,
    requested_gpus: 0,
    outcome: $outcome,
    kernel_osrelease: $kernel,
    cgroup_version: 2,
    membership_path: $membership,
    mount_root: $mount_root,
    mount_point: $mount_point,
    allocation_owned_job_scope_path: $job_scope,
    native_memory_peak_state: $peak_state,
    native_memory_peak_value_bytes: (if $peak_state == "readable_integer" then ($peak_value | tonumber) else null end),
    memory_current_state: $current_state,
    sampled_memory_current_high_water_bytes: (if ($sample_count | tonumber) > 0 then ($sampled_high | tonumber) else null end),
    sampled_memory_current_sample_count: ($sample_count | tonumber),
    sampled_memory_current_requested_interval_ms: 100,
    sampled_memory_current_max_gap_ns: ($max_gap | tonumber),
    sampled_memory_current_timestamps_strictly_increasing: ($raw_timestamps_increasing == "true"),
    sampled_memory_current_is_lower_bound_not_peak: true,
    diagnostic_raw_trace_recomputed: true,
    memory_max_state: $max_state,
    memory_max_job_scope_hard_limit_bytes: (if $max_state == "readable_integer" then ($max_value | tonumber) else null end),
    memory_max_is_job_scope_hard_limit_not_usage: true,
    memory_events_state: $events_state,
    memory_events_local_state: $local_events_state,
    memory_events_hierarchy_state: $events_hierarchy_state,
    memory_events_max_delta: (if ($event_max | length) > 0 then ($event_max | tonumber) else null end),
    memory_events_oom_delta: (if ($event_oom | length) > 0 then ($event_oom | tonumber) else null end),
    memory_events_oom_kill_delta: (if ($event_oom_kill | length) > 0 then ($event_oom_kill | tonumber) else null end),
    zero_event_deltas_mean_no_new_cgroup_recorded_limit_or_oom_event_during_sampling_window_not_exact_peak: true,
    sampled_current_contract_capability_supported: ($supported == "true"),
    shell_only: true,
    checkpoint_loaded: false,
    policy_server_started: false,
    model_inference_executed: false,
    simulator_steps_executed: 0,
    teacher_searches_executed: 0,
    scientific_claim_allowed: false,
    retry_c_authorized_by_this_artifact_alone: false,
    ift01_authorized: false,
    probe_training_authorized: false
  }' >"$RESULT_CANDIDATE"
jq -e \
  --arg run_id "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg submission_sha "$SUBMISSION_SHA256" \
  --arg sidecar_sha "$SIDECAR_SHA256" \
  '.status == "completed"
   and .run_id == $run_id
   and .git_commit == $commit
   and .submission_receipt_sha256 == $submission_sha
   and .diagnostic_sha256 == $sidecar_sha
   and .host == "worker-1"
   and .requested_cpus == 1
   and .requested_host_memory_mib == 256
   and .requested_gpus == 0
   and .sampled_memory_current_is_lower_bound_not_peak == true
   and .diagnostic_raw_trace_recomputed == true
   and .memory_max_is_job_scope_hard_limit_not_usage == true
   and .shell_only == true
   and .scientific_claim_allowed == false
   and .retry_c_authorized_by_this_artifact_alone == false
   and .ift01_authorized == false
   and .probe_training_authorized == false' "$RESULT_CANDIDATE" >/dev/null || {
  echo "capability result guard failed" >&2
  exit 4
}
mv "$RESULT_CANDIDATE" "$RESULT"
FAILURE_STAGE=complete
echo "result=$RESULT"
echo "result_sha256=$(sha256sum "$RESULT" | awk '{print $1}')"
echo "outcome=$OUTCOME"
echo "sampled_memory_current_high_water_bytes=$SAMPLED_HIGH_WATER (lower_bound_not_peak)"
