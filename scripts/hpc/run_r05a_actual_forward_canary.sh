#!/usr/bin/env bash
set -euo pipefail

# Control-plane guard for the exact held AF-00A allocation.  The actual model
# work is delegated only after the released config, allocation, and immutable
# source contract agree.
: "${SLURM_JOB_ID:?AF-00A must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?AF-00A requires an array job id}"
: "${SLURM_ARRAY_TASK_ID:?AF-00A requires an array task id}"
: "${EXPECTED_GIT_COMMIT:?reviewed AF-00A release commit is required}"
: "${RUN_ID:?immutable AF-00A run id is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

CONFIG=$REMOTE_REPO/configs/experiments/r05a_actual_forward_canary.json
WORKLOAD=$REMOTE_REPO/scripts/hpc/run_r05a_actual_forward_workload.sh
RUN_ROOT=/mnt/data/quanth/experiments/crfs-oracle/$RUN_ID
SOURCE_CONTRACT=${SOURCE_CONTRACT:-$RUN_ROOT/source-contract.json}
HELD_GPU_SUBMISSION=$RUN_ROOT/held-gpu-submission.json
SUBMISSION=${SUBMISSION:-$RUN_ROOT/submission.json}
RELEASE_FINGERPRINT=${RELEASE_FINGERPRINT:-$RUN_ROOT/final-pre-release-fingerprint.json}
RELEASE_FINGERPRINT_SHA256_FILE=${RELEASE_FINGERPRINT_SHA256_FILE:-$RUN_ROOT/final-pre-release-fingerprint.sha256}

case "$SLURM_ARRAY_JOB_ID" in *[!0-9]*|'') echo "invalid AF-00A array id" >&2; exit 2 ;; esac
test "$SLURM_ARRAY_TASK_ID" = 0 || { echo "AF-00A is fixed to task zero" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "AF-00A is source-pinned to worker-1" >&2; exit 2; }
test -f "$CONFIG" && test ! -L "$CONFIG" || { echo "missing released AF-00A config" >&2; exit 2; }
test -x "$WORKLOAD" || { echo "missing AF-00A workload" >&2; exit 2; }
test "$SOURCE_CONTRACT" = "$RUN_ROOT/source-contract.json" || { echo "AF-00A source-contract path changed" >&2; exit 2; }
test "$SUBMISSION" = "$RUN_ROOT/submission.json" || { echo "AF-00A submission path changed" >&2; exit 2; }
test "$RELEASE_FINGERPRINT" = "$RUN_ROOT/final-pre-release-fingerprint.json" || { echo "AF-00A release-fingerprint path changed" >&2; exit 2; }
test "$RELEASE_FINGERPRINT_SHA256_FILE" = "$RUN_ROOT/final-pre-release-fingerprint.sha256" || { echo "AF-00A release-fingerprint digest path changed" >&2; exit 2; }
for path in "$SOURCE_CONTRACT" "$HELD_GPU_SUBMISSION" "$SUBMISSION" "$RELEASE_FINGERPRINT" "$RELEASE_FINGERPRINT_SHA256_FILE"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked AF-00A transaction artifact: $path" >&2; exit 2; }
done
EXPECTED_RELEASE_FINGERPRINT_SHA256=$(cat "$RELEASE_FINGERPRINT_SHA256_FILE")
case "$EXPECTED_RELEASE_FINGERPRINT_SHA256" in *[!0-9a-f]*|'') echo "invalid AF-00A release-fingerprint digest" >&2; exit 2 ;; esac
test "${#EXPECTED_RELEASE_FINGERPRINT_SHA256}" = 64 || { echo "invalid AF-00A release-fingerprint digest length" >&2; exit 2; }
test "$(wc -l <"$RELEASE_FINGERPRINT_SHA256_FILE" | tr -d ' ')" = 1 || { echo "AF-00A release-fingerprint digest must be one line" >&2; exit 2; }
test "$(sha256sum "$RELEASE_FINGERPRINT" | awk '{print $1}')" = "$EXPECTED_RELEASE_FINGERPRINT_SHA256" || { echo "AF-00A release fingerprint changed" >&2; exit 2; }
EXPECTED_SOURCE_CONTRACT_SHA256=$(jq -er '.source_contract_sha256' "$RELEASE_FINGERPRINT")
EXPECTED_HELD_GPU_SUBMISSION_SHA256=$(jq -er '.held_gpu_submission_sha256' "$RELEASE_FINGERPRINT")
EXPECTED_SUBMISSION_SHA256=$(jq -er '.submission_sha256' "$RELEASE_FINGERPRINT")
for digest in "$EXPECTED_SOURCE_CONTRACT_SHA256" "$EXPECTED_HELD_GPU_SUBMISSION_SHA256" "$EXPECTED_SUBMISSION_SHA256"; do
  case "$digest" in *[!0-9a-f]*|'') echo "invalid AF-00A transaction digest" >&2; exit 2 ;; esac
  test "${#digest}" = 64 || { echo "invalid AF-00A transaction digest length" >&2; exit 2; }
done
test "$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')" = "$EXPECTED_SOURCE_CONTRACT_SHA256" || {
  echo "AF-00A source-contract digest changed" >&2; exit 2
}
test "$(sha256sum "$HELD_GPU_SUBMISSION" | awk '{print $1}')" = "$EXPECTED_HELD_GPU_SUBMISSION_SHA256" || { echo "AF-00A held GPU receipt changed" >&2; exit 2; }
test "$(sha256sum "$SUBMISSION" | awk '{print $1}')" = "$EXPECTED_SUBMISSION_SHA256" || { echo "AF-00A submission receipt changed" >&2; exit 2; }

CPU_PUBLISHER_JOB_ID=$(jq -er '.cpu_afterany_job_id' "$RELEASE_FINGERPRINT")
case "$CPU_PUBLISHER_JOB_ID" in *[!0-9]*|'') echo "invalid AF-00A CPU publisher identity" >&2; exit 2 ;; esac
jq -e --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" --arg gpu "$SLURM_ARRAY_JOB_ID" --arg cpu "$CPU_PUBLISHER_JOB_ID" \
  --arg source "$SOURCE_CONTRACT" --arg source_sha "$EXPECTED_SOURCE_CONTRACT_SHA256" \
  --arg held "$HELD_GPU_SUBMISSION" --arg held_sha "$EXPECTED_HELD_GPU_SUBMISSION_SHA256" \
  --arg submission "$SUBMISSION" --arg submission_sha "$EXPECTED_SUBMISSION_SHA256" \
  --arg result "$RUN_ROOT/crfs-1069f29a8d76463a/results.json" --arg receipt "$RUN_ROOT/cpu-afterany-validation.json" '
  .artifact_role == "r05a_actual_forward_canary_final_pre_release_fingerprint"
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
  ' "$RELEASE_FINGERPRINT" >/dev/null || { echo "AF-00A release fingerprint differs from this allocation" >&2; exit 2; }
jq -e --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" --arg gpu "$SLURM_ARRAY_JOB_ID" --arg cpu "$CPU_PUBLISHER_JOB_ID" \
  --arg source "$SOURCE_CONTRACT" --arg source_sha "$EXPECTED_SOURCE_CONTRACT_SHA256" \
  --arg held "$HELD_GPU_SUBMISSION" --arg held_sha "$EXPECTED_HELD_GPU_SUBMISSION_SHA256" \
  --arg fingerprint "$RELEASE_FINGERPRINT" '
  .artifact_role == "r05a_actual_forward_canary_atomic_submission"
  and .run_id == $run and .git_commit == $commit and .source_node == "worker-1"
  and .gpu_slurm_array_job_id == $gpu and .gpu_slurm_array_task_id == 0
  and .exact_gpu_task_id == ($gpu+"_0") and .cpu_afterany_job_id == $cpu
  and .dependency == ("afterany:"+$gpu)
  and .source_contract_path == $source and .source_contract_sha256 == $source_sha
  and .held_gpu_submission_path == $held and .held_gpu_submission_sha256 == $held_sha
  and .release_fingerprint_path == $fingerprint and .released_at_receipt_time == false
  ' "$SUBMISSION" >/dev/null || { echo "AF-00A submission receipt differs from this allocation" >&2; exit 2; }
while IFS='|' read -r key; do
  path=$(jq -er --arg key "$key" '.bindings[$key].path' "$RELEASE_FINGERPRINT")
  digest=$(jq -er --arg key "$key" '.bindings[$key].sha256' "$RELEASE_FINGERPRINT")
  test -f "$path" && test ! -L "$path" || { echo "AF-00A fingerprint binding is missing: $key" >&2; exit 2; }
  test "$(sha256sum "$path" | awk '{print $1}')" = "$digest" || { echo "AF-00A fingerprint binding changed: $key" >&2; exit 2; }
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
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || {
  echo "AF-00A remote HEAD differs from release" >&2; exit 2
}
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || {
  echo "AF-00A requires a clean remote source tree" >&2; exit 2
}

ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$CONFIG")
jq -e --arg run "$RUN_ID" --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" '
  .ready_to_run == true and .blocked_on == []
  and .preregistration.h100_submission_authorized == true
  and .execution_release.artifact_role == "r05a_actual_forward_canary_execution_release"
  and .execution_release.accepted_implementation_commit == $implementation
  and .execution_release.run_id == $run
  and .execution_release.single_submission == true
  and .execution_release.source_host == "worker-1"
  and .execution_release.resources == {
    partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,
    host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,
    validator_partition:"main",validator_account:"normal",validator_qos:"normal",
    validator_cpus:2,validator_host_memory_mib:8192,
    validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"
  }
  and .execution_release.release_only_parent_required == true
  and .execution_release.decision_artifact == "docs/decisions/0053-release-corrected-actual-forward-cem-canary.md"
  and .execution_release.allowed_release_diff_paths == [
    "configs/experiments/r05a_actual_forward_canary.json",
    "docs/decisions/0053-release-corrected-actual-forward-cem-canary.md"
  ]
  and .execution_release.automatic_resubmission_allowed == false
  and .execution_release.automatic_next_experiment_allowed == false
  and .flow_contract.ordinary_residual_schedule_path_only == true
  and .flow_contract.autograd_allowed == false
  and .flow_contract.inverse_flow_teacher_allowed == false
  and .request_ledger.complete_run_exact_policy_request_count == 534
' "$CONFIG" >/dev/null || { echo "AF-00A execution release changed" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-list --parents -n 1 "$EXPECTED_GIT_COMMIT")" = \
  "$EXPECTED_GIT_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || {
  echo "AF-00A release is not the direct child of its implementation" >&2; exit 2
}

job_record=$(scontrol show job "${SLURM_ARRAY_JOB_ID}_0" -o)
task_job_id=$(printf '%s\n' "$job_record" | sed -n 's/.* JobId=\([^ ]*\).*/\1/p')
case "$task_job_id" in
  "$SLURM_ARRAY_JOB_ID"|"${SLURM_ARRAY_JOB_ID}_0") ;;
  *) echo "AF-00A job field changed: JobId=$task_job_id" >&2; exit 2 ;;
esac
for field in "ArrayJobId=$SLURM_ARRAY_JOB_ID" "ArrayTaskId=0" JobState=RUNNING Partition=main Account=normal QOS=normal TimeLimit=02:00:00 Requeue=0 ReqNodeList=worker-1 NumNodes=1 NumCPUs=8 CPUs/Task=8; do
  case " $job_record " in *" $field "*) ;; *) echo "AF-00A job field changed: $field" >&2; exit 2 ;; esac
done
req_tres=$(printf '%s\n' "$job_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test "$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= 'NF && $1!="billing" {print $1}' | LC_ALL=C sort)" = "$(printf '%s\n' cpu gres/gpu mem node | LC_ALL=C sort)" || { echo "AF-00A ReqTRES key set changed" >&2; exit 2; }
test "$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= '$1=="cpu" {print $2}')" = 8 || { echo "AF-00A CPU request changed" >&2; exit 2; }
case "$(printf '%s\n' "$req_tres" | tr ',' '\n' | sed -n 's/^mem=//p')" in 64G|65536M) ;; *) echo "AF-00A memory request changed" >&2; exit 2 ;; esac
test "$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= '$1=="node" {print $2}')" = 1 || { echo "AF-00A node request changed" >&2; exit 2; }
test "$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= '$1=="gres/gpu" {print $2}')" = 1 || { echo "AF-00A GPU request changed" >&2; exit 2; }
test "$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= '$1 ~ /^gres\/gpu/ {print}')" = gres/gpu=1 || { echo "AF-00A exact GPU request changed" >&2; exit 2; }

jq -e --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" --arg gpu "$SLURM_ARRAY_JOB_ID" '
  .schema_version == "1.0"
  and .artifact_role == "r05a_actual_forward_canary_source_contract"
  and .status == "gpu_held_sources_bound_before_cpu_submission"
  and .run_id == $run and .git_commit == $commit and .git_dirty == false
  and .source_node == "worker-1"
  and .gpu_slurm_array_job_id == $gpu and .gpu_slurm_array_task_id == 0
  and .exact_gpu_task_id == ($gpu + "_0")
  and .resources.host_memory_mib == 65536
  and .resources.cpus_per_task == 8 and .resources.gpus == 1
  and .scientific_claim_allowed == false
  and .simulator_efficacy_claim_allowed == false
  and .probe_training_authorized == false
' "$SOURCE_CONTRACT" >/dev/null || { echo "AF-00A source contract differs from allocation" >&2; exit 2; }

for relative in \
  configs/experiments/r05a_actual_forward_canary.json \
  main/run_crfs_r05a_actual_forward_canary.py \
  main/publish_crfs_r05a_actual_forward_canary.py \
  main/crfs_oracle/r05a_actual_forward_canary.py \
  main/crfs_oracle/r05a_actual_forward_search.py \
  main/crfs_oracle/r05a_actual_forward_validation.py \
  openpi/scripts/serve_policy.py \
  openpi/src/openpi/policies/policy.py \
  openpi/src/openpi/models_pytorch/pi0_pytorch.py \
  scripts/hpc/lib/r05a_runtime_identity.sh \
  scripts/hpc/run_r05a_actual_forward_workload.sh; do
  expected=$(jq -er --arg path "$relative" '.repository_file_sha256[$path]' "$SOURCE_CONTRACT")
  actual=$(sha256sum "$REMOTE_REPO/$relative" | awk '{print $1}')
  test "$actual" = "$expected" || { echo "AF-00A source hash changed: $relative" >&2; exit 2; }
done

export AF_SOURCE_CONTRACT=$SOURCE_CONTRACT
exec "$WORKLOAD"
