#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?TRL-00A publisher must run inside Slurm}"
: "${SLURM_JOB_PARTITION:?TRL-00A publisher requires partition provenance}"
: "${SOURCE_JOB_ID:?exact TRL-00A H100 job id is required}"
: "${SOURCE_CONTRACT:?TRL-00A source contract is required}"
: "${EXPECTED_SOURCE_CONTRACT_SHA256:?TRL-00A source-contract digest is required}"
: "${HELD_GPU_SUBMISSION:?TRL-00A held GPU receipt is required}"
: "${EXPECTED_HELD_GPU_SUBMISSION_SHA256:?TRL-00A held GPU receipt digest is required}"
: "${SUBMISSION:?TRL-00A atomic submission receipt is required}"
: "${RELEASE_FINGERPRINT:?TRL-00A release fingerprint is required}"
: "${RELEASE_FINGERPRINT_SHA256_FILE:?TRL-00A release fingerprint digest file is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed TRL-00A release commit is required}"
: "${RUN_ID:?immutable TRL-00A run id is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"

CONFIG=$REMOTE_REPO/configs/experiments/r05a_reference_trajectory_lift_canary.json
LEGACY_CONFIG=$REMOTE_REPO/configs/experiments/r05a_inverse_flow_canary.json
STATUS_HELPER=$REMOTE_REPO/scripts/hpc/lib/slurm_exact_array_task_status.sh
RUNTIME_IDENTITY_HELPER=$REMOTE_REPO/scripts/hpc/lib/r05a_runtime_identity.sh
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
CASE_DIR=$RUN_ROOT/crfs-1069f29a8d76463a
RESULT=$CASE_DIR/results.json
RECEIPT=$RUN_ROOT/cpu-afterany-validation.json

mkdir -p "$RUN_ROOT"
FAILURE_STAGE=wrapper_preflight
SOURCE_STATE=
SOURCE_EXIT=
fallback_receipt() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ ! -f "$RECEIPT" ]; then
    temporary=$(mktemp "$RUN_ROOT/.cpu-afterany-validation.XXXXXX") || return "$status"
    jq -n --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" \
      --arg publisher "$SLURM_JOB_ID" --arg source "$SOURCE_JOB_ID" \
      --arg state "$SOURCE_STATE" --arg exit_code "$SOURCE_EXIT" --arg stage "$FAILURE_STAGE" \
      --arg result "$RESULT" --arg wrapper_status "$status" \
      '{schema_version:"1.0",artifact_role:"r05a_reference_trajectory_lift_canary_cpu_publication",status:"apparatus_inconclusive",outcome:"apparatus_inconclusive",run_id:$run,case_id:"crfs-1069f29a8d76463a",git_commit:$commit,source_node:"worker-1",publisher_job_id:$publisher,source_job_id:$source,source_task_id:($source+"_0"),source_job_state:(if $state=="" then null else $state end),source_exit_code:(if $exit_code=="" then null else $exit_code end),dependency:("afterany:"+$source),failure_stage:$stage,result_path:$result,result_sha256:null,passed:false,published:false,errors:["publisher wrapper failed with exit code "+$wrapper_status],scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,collision_or_progress_claim_allowed:false,infeasibility_claim_allowed:false,raw_authority_is_minimum_required_claim_allowed:false,probe_training_authorized:false,automatic_next_gate_authorized:false}' >"$temporary" && mv "$temporary" "$RECEIPT"
  fi
  return "$status"
}
trap fallback_receipt EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test "$SLURM_JOB_PARTITION" = main || { echo "TRL-00A publisher is frozen to main" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 2 || { echo "TRL-00A publisher requires two CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 8192 || { echo "TRL-00A publisher requires exactly 8 GiB" >&2; exit 2; }
test "${CUDA_VISIBLE_DEVICES:-NoDevFiles}" = NoDevFiles || { echo "TRL-00A publisher must not receive a GPU" >&2; exit 2; }
case "$SOURCE_JOB_ID" in *[!0-9]*|'') echo "invalid TRL-00A source job id" >&2; exit 2 ;; esac
test "$SOURCE_CONTRACT" = "$RUN_ROOT/source-contract.json" || { echo "TRL-00A source-contract path changed" >&2; exit 2; }
test "$HELD_GPU_SUBMISSION" = "$RUN_ROOT/held-gpu-submission.json" || { echo "TRL-00A held GPU receipt path changed" >&2; exit 2; }
test "$SUBMISSION" = "$RUN_ROOT/submission.json" || { echo "TRL-00A atomic submission path changed" >&2; exit 2; }
test "$RELEASE_FINGERPRINT" = "$RUN_ROOT/final-pre-release-fingerprint.json" || { echo "TRL-00A release-fingerprint path changed" >&2; exit 2; }
test "$RELEASE_FINGERPRINT_SHA256_FILE" = "$RUN_ROOT/final-pre-release-fingerprint.sha256" || { echo "TRL-00A release-fingerprint digest path changed" >&2; exit 2; }
for digest in "$EXPECTED_SOURCE_CONTRACT_SHA256" "$EXPECTED_HELD_GPU_SUBMISSION_SHA256"; do
  case "$digest" in *[!0-9a-f]*|'') echo "invalid TRL-00A transaction digest" >&2; exit 2 ;; esac
  test "${#digest}" = 64 || { echo "invalid TRL-00A transaction digest length" >&2; exit 2; }
done
for path in "$CONFIG" "$LEGACY_CONFIG" "$SOURCE_CONTRACT" "$HELD_GPU_SUBMISSION" "$SUBMISSION" "$RELEASE_FINGERPRINT" "$RELEASE_FINGERPRINT_SHA256_FILE" "$STATUS_HELPER" "$RUNTIME_IDENTITY_HELPER" "$REMOTE_REPO/main/publish_crfs_r05a_reference_trajectory_lift_canary.py"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked TRL-00A publisher input: $path" >&2; exit 2; }
done
. "$RUNTIME_IDENTITY_HELPER"
test "$LIBERO_PYTHON" = "$CRFS_R05A_LIBERO_PYTHON" || { echo "noncanonical TRL-00A publisher launcher" >&2; exit 2; }
crfs_validate_r05a_libero_python "$LIBERO_PYTHON"
test "$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')" = "$EXPECTED_SOURCE_CONTRACT_SHA256" || { echo "TRL-00A source contract changed" >&2; exit 2; }
test "$(sha256sum "$HELD_GPU_SUBMISSION" | awk '{print $1}')" = "$EXPECTED_HELD_GPU_SUBMISSION_SHA256" || { echo "TRL-00A held GPU receipt changed" >&2; exit 2; }
EXPECTED_RELEASE_FINGERPRINT_SHA256=$(cat "$RELEASE_FINGERPRINT_SHA256_FILE")
case "$EXPECTED_RELEASE_FINGERPRINT_SHA256" in *[!0-9a-f]*|'') echo "invalid TRL-00A release-fingerprint digest" >&2; exit 2 ;; esac
test "${#EXPECTED_RELEASE_FINGERPRINT_SHA256}" = 64 || { echo "invalid TRL-00A release-fingerprint digest length" >&2; exit 2; }
test "$(wc -l <"$RELEASE_FINGERPRINT_SHA256_FILE" | tr -d ' ')" = 1 || { echo "TRL-00A release-fingerprint digest must be one line" >&2; exit 2; }
test "$(sha256sum "$RELEASE_FINGERPRINT" | awk '{print $1}')" = "$EXPECTED_RELEASE_FINGERPRINT_SHA256" || { echo "TRL-00A release fingerprint changed" >&2; exit 2; }
EXPECTED_SUBMISSION_SHA256=$(jq -er '.submission_sha256' "$RELEASE_FINGERPRINT")
case "$EXPECTED_SUBMISSION_SHA256" in *[!0-9a-f]*|'') echo "invalid TRL-00A submission digest" >&2; exit 2 ;; esac
test "${#EXPECTED_SUBMISSION_SHA256}" = 64 || { echo "invalid TRL-00A submission digest length" >&2; exit 2; }
test "$(sha256sum "$SUBMISSION" | awk '{print $1}')" = "$EXPECTED_SUBMISSION_SHA256" || { echo "TRL-00A atomic submission changed" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || { echo "TRL-00A publisher commit changed" >&2; exit 2; }
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || { echo "TRL-00A publisher tree is dirty" >&2; exit 2; }

ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$CONFIG")
RELEASE_ADR=$(jq -er '.execution_release.decision_artifact' "$CONFIG")
case "$RELEASE_ADR" in
  docs/decisions/*-release-reference-trajectory-lift-canary.md) ;;
  *) echo "TRL-00A publisher release decision path changed" >&2; exit 2 ;;
esac
jq -e --arg run "$RUN_ID" --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" --arg release "$RELEASE_ADR" '
  .ready_to_run == true and .blocked_on == []
  and .execution_release.artifact_role == "r05a_reference_trajectory_lift_canary_execution_release"
  and .execution_release.accepted_implementation_commit == $implementation
  and .execution_release.run_id == $run
  and .execution_release.single_submission == true
  and .execution_release.source_host == "worker-1"
  and .execution_release.resources.validator_cpus == 2
  and .execution_release.resources.validator_host_memory_mib == 8192
  and .execution_release.resources.validator_time_limit == "00:15:00"
  and .execution_release.resources.validator_gpus == 0
  and .execution_release.resources.validator_dependency == "afterany"
  and .execution_release.decision_artifact == $release
  and .execution_release.allowed_release_diff_paths == [
    "configs/experiments/r05a_reference_trajectory_lift_canary.json",
    $release
  ]
  and .execution_release.automatic_resubmission_allowed == false
  and .execution_release.automatic_next_experiment_allowed == false
  and .request_ledger.complete_finite_exact_policy_request_count == 18
  and .reference_lift_contract.raw_nonfinite_result == "apparatus_inconclusive_no_scientific_results_json"
  and .artifact_contract.gpu_may_publish_results_json == false
  and .artifact_contract.cpu_afterany_is_sole_publisher == true
' "$CONFIG" >/dev/null || { echo "TRL-00A publisher release changed" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-list --parents -n 1 "$EXPECTED_GIT_COMMIT")" = "$EXPECTED_GIT_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || { echo "TRL-00A publisher release parent changed" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" diff --name-only "$ACCEPTED_IMPLEMENTATION_COMMIT" "$EXPECTED_GIT_COMMIT" | LC_ALL=C sort)" = \
  "$(printf '%s\n' configs/experiments/r05a_reference_trajectory_lift_canary.json "$RELEASE_ADR" | LC_ALL=C sort)" || {
  echo "TRL-00A publisher release diff changed" >&2; exit 2
}

jq -e --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" --arg source "$SOURCE_JOB_ID" '
  .artifact_role == "r05a_reference_trajectory_lift_canary_source_contract"
  and .run_id == $run and .git_commit == $commit
  and .gpu_slurm_array_job_id == $source and .exact_gpu_task_id == ($source+"_0")
  and .scientific_claim_allowed == false
  and .simulator_efficacy_claim_allowed == false
  and .probe_training_authorized == false
' "$SOURCE_CONTRACT" >/dev/null || { echo "TRL-00A publisher source binding changed" >&2; exit 2; }

jq -e --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" --arg gpu "$SOURCE_JOB_ID" --arg cpu "$SLURM_JOB_ID" \
  --arg source "$SOURCE_CONTRACT" --arg source_sha "$EXPECTED_SOURCE_CONTRACT_SHA256" \
  --arg held "$HELD_GPU_SUBMISSION" --arg held_sha "$EXPECTED_HELD_GPU_SUBMISSION_SHA256" \
  --arg submission "$SUBMISSION" --arg submission_sha "$EXPECTED_SUBMISSION_SHA256" \
  --arg result "$RESULT" --arg receipt "$RECEIPT" '
  .artifact_role == "r05a_reference_trajectory_lift_canary_final_pre_release_fingerprint"
  and .status == "all_receipts_bound_gpu_still_held"
  and .run_id == $run and .git_commit == $commit and .source_node == "worker-1"
  and .gpu_slurm_array_job_id == $gpu and .gpu_slurm_array_task_id == 0
  and .exact_gpu_task_id == ($gpu+"_0") and .cpu_afterany_job_id == $cpu
  and .dependency == ("afterany:"+$gpu)
  and .source_contract_path == $source and .source_contract_sha256 == $source_sha
  and .held_gpu_submission_path == $held and .held_gpu_submission_sha256 == $held_sha
  and .submission_path == $submission and .submission_sha256 == $submission_sha
  and .expected_result == $result and .expected_validation_receipt == $receipt
  and .released_at_fingerprint_time == false
  and .scientific_claim_allowed == false and .simulator_efficacy_claim_allowed == false
  and .infeasibility_claim_allowed == false and .probe_training_authorized == false
  ' "$RELEASE_FINGERPRINT" >/dev/null || { echo "TRL-00A publisher release fingerprint changed" >&2; exit 2; }
jq -e --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" --arg gpu "$SOURCE_JOB_ID" --arg cpu "$SLURM_JOB_ID" \
  --arg source "$SOURCE_CONTRACT" --arg source_sha "$EXPECTED_SOURCE_CONTRACT_SHA256" \
  --arg held "$HELD_GPU_SUBMISSION" --arg held_sha "$EXPECTED_HELD_GPU_SUBMISSION_SHA256" \
  --arg fingerprint "$RELEASE_FINGERPRINT" '
  .artifact_role == "r05a_reference_trajectory_lift_canary_atomic_submission"
  and .run_id == $run and .git_commit == $commit and .source_node == "worker-1"
  and .gpu_slurm_array_job_id == $gpu and .gpu_slurm_array_task_id == 0
  and .exact_gpu_task_id == ($gpu+"_0") and .cpu_afterany_job_id == $cpu
  and .dependency == ("afterany:"+$gpu)
  and .source_contract_path == $source and .source_contract_sha256 == $source_sha
  and .held_gpu_submission_path == $held and .held_gpu_submission_sha256 == $held_sha
  and .release_fingerprint_path == $fingerprint and .released_at_receipt_time == false
  ' "$SUBMISSION" >/dev/null || { echo "TRL-00A publisher atomic submission changed" >&2; exit 2; }
while IFS= read -r key; do
  path=$(jq -er --arg key "$key" '.bindings[$key].path' "$RELEASE_FINGERPRINT")
  digest=$(jq -er --arg key "$key" '.bindings[$key].sha256' "$RELEASE_FINGERPRINT")
  test -f "$path" && test ! -L "$path" || { echo "TRL-00A publisher fingerprint binding is missing: $key" >&2; exit 2; }
  test "$(sha256sum "$path" | awk '{print $1}')" = "$digest" || { echo "TRL-00A publisher fingerprint binding changed: $key" >&2; exit 2; }
done <<'EOF'
provisional_gpu_job_id
held_gpu_submission
source_contract
provisional_cpu_job_id
atomic_submission
held_gpu_task_job_record
held_gpu_parent_job_record
cpu_afterany_job_record
EOF

publisher_job=$(scontrol show job "$SLURM_JOB_ID" -o)
for field in "JobId=$SLURM_JOB_ID" JobState=RUNNING Partition=main Account=normal QOS=normal TimeLimit=00:15:00 Requeue=0 NumNodes=1 NumCPUs=2 CPUs/Task=2; do
  case " $publisher_job " in *" $field "*) ;; *) echo "TRL-00A publisher field changed: $field" >&2; exit 2 ;; esac
done
publisher_req_tres=$(printf '%s\n' "$publisher_job" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test "$(printf '%s\n' "$publisher_req_tres" | tr ',' '\n' | awk -F= 'NF && $1!="billing" {print $1}' | LC_ALL=C sort)" = "$(printf '%s\n' cpu mem node | LC_ALL=C sort)" || { echo "TRL-00A publisher ReqTRES key set changed" >&2; exit 2; }
test "$(printf '%s\n' "$publisher_req_tres" | tr ',' '\n' | awk -F= '$1=="cpu" {print $2}')" = 2 || { echo "TRL-00A publisher CPU request changed" >&2; exit 2; }
case "$(printf '%s\n' "$publisher_req_tres" | tr ',' '\n' | awk -F= '$1=="mem" {print $2}')" in 8G|8192M) ;; *) echo "TRL-00A publisher memory request changed" >&2; exit 2 ;; esac
test "$(printf '%s\n' "$publisher_req_tres" | tr ',' '\n' | awk -F= '$1=="node" {print $2}')" = 1 || { echo "TRL-00A publisher node request changed" >&2; exit 2; }
if printf '%s\n' "$publisher_req_tres" | tr ',' '\n' | awk -F= '$1 ~ /^gres\/gpu/ {found=1} END {exit !found}'; then
  echo "TRL-00A publisher requests a GPU" >&2; exit 2
fi
case " $publisher_job " in *" TRESPerNode=gres/gpu:"*) echo "TRL-00A publisher has per-node GPU resources" >&2; exit 2 ;; esac

. "$STATUS_HELPER"
FAILURE_STAGE=exact_source_task_accounting
source_record=
if source_record=$(crfs_wait_for_exact_completed_array_task "$SOURCE_JOB_ID" 30 1); then source_status=0; else source_status=$?; fi
IFS='|' read -r SOURCE_STATE SOURCE_EXIT <<EOF
$source_record
EOF
test "$source_status" -eq 0 && test "$SOURCE_STATE" = COMPLETED && test "$SOURCE_EXIT" = 0:0 || {
  echo "TRL-00A source task failed: state=${SOURCE_STATE:-missing} exit=${SOURCE_EXIT:-missing}" >&2; exit 3
}

FAILURE_STAGE=python_publication
JSONSCHEMA_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_jsonschema_overlay.sh)
export PYTHONPATH=$JSONSCHEMA_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero
"$LIBERO_PYTHON" "$REMOTE_REPO/main/publish_crfs_r05a_reference_trajectory_lift_canary.py" \
  --run-root "$RUN_ROOT" --config "$CONFIG" --legacy-config "$LEGACY_CONFIG" \
  --output "$RESULT" --receipt "$RECEIPT" --source-contract "$SOURCE_CONTRACT" \
  --expected-source-contract-sha256 "$EXPECTED_SOURCE_CONTRACT_SHA256" \
  --held-gpu-submission "$HELD_GPU_SUBMISSION" \
  --expected-held-gpu-submission-sha256 "$EXPECTED_HELD_GPU_SUBMISSION_SHA256" \
  --submission "$SUBMISSION" --expected-submission-sha256 "$EXPECTED_SUBMISSION_SHA256" \
  --release-fingerprint "$RELEASE_FINGERPRINT" \
  --expected-release-fingerprint-sha256 "$EXPECTED_RELEASE_FINGERPRINT_SHA256" \
  --source-job-id "$SOURCE_JOB_ID" \
  --publisher-job-id "$SLURM_JOB_ID"
test -f "$RESULT" && test -f "$RECEIPT" || { echo "TRL-00A publisher omitted final artifacts" >&2; exit 4; }
echo "source_job_id=$SOURCE_JOB_ID"
echo "result=$RESULT"
echo "result_sha256=$(sha256sum "$RESULT" | awk '{print $1}')"
echo "validation_receipt=$RECEIPT"
echo "validation_receipt_sha256=$(sha256sum "$RECEIPT" | awk '{print $1}')"
