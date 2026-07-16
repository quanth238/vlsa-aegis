#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... EXPECTED_RELEASE_COMMIT=... submit_r05a_actual_forward_canary.sh'
: "${RUN_ID:?$usage}"
: "${EXPECTED_RELEASE_COMMIT:?$usage}"
case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe AF-00A run id" >&2; exit 2 ;; esac
case "$RUN_ID" in [A-Za-z0-9]*) ;; *) echo "AF-00A run id must start alphanumeric" >&2; exit 2 ;; esac
case "$EXPECTED_RELEASE_COMMIT" in *[!0-9a-f]*|'') echo "invalid AF-00A release commit" >&2; exit 2 ;; esac
test "${#EXPECTED_RELEASE_COMMIT}" = 40 || { echo "AF-00A release commit must have 40 hex characters" >&2; exit 2; }

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
CONFIG=configs/experiments/r05a_actual_forward_canary.json
PREREG_ADR=docs/decisions/0050-preregister-actual-forward-cem-teacher-canary.md
RELEASE_ADR=docs/decisions/0051-release-actual-forward-cem-canary.md

for path in "$CONFIG" "$PREREG_ADR" \
  main/run_crfs_r05a_actual_forward_canary.py \
  main/publish_crfs_r05a_actual_forward_canary.py \
  main/crfs_oracle/r05a_actual_forward_canary.py \
  main/crfs_oracle/r05a_actual_forward_search.py \
  main/crfs_oracle/r05a_actual_forward_validation.py \
  scripts/hpc/run_r05a_actual_forward_workload.sh \
  scripts/hpc/run_r05a_actual_forward_canary.sh \
  scripts/hpc/validate_r05a_actual_forward_canary.sh \
  scripts/hpc/submit_r05a_actual_forward_canary.sh \
  slurm/r05a_actual_forward_canary_h100.sbatch \
  slurm/r05a_actual_forward_canary_validate_cpu.sbatch; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked AF-00A source: $path" >&2; exit 2; }
done
jq -e '
  .config_status == "released_exact_single_canary"
  and .ready_to_run == true and .blocked_on == []
  and .preregistration.h100_submission_authorized == true
  and (.execution_release | type=="object")
' "$CONFIG" >/dev/null || {
  echo "AF-00A is implemented but not released for H100 submission" >&2
  exit 2
}
test -z "$(git status --porcelain)" || { echo "AF-00A submission requires a clean reviewed tree" >&2; exit 2; }
test "$(git rev-parse HEAD)" = "$EXPECTED_RELEASE_COMMIT" || { echo "local AF-00A HEAD differs from release" >&2; exit 2; }
test "$(git rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$EXPECTED_RELEASE_COMMIT" || { echo "origin AF-00A release ref differs" >&2; exit 2; }

ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$CONFIG")
REGISTERED_RUN_ID=$(jq -er '.execution_release.run_id' "$CONFIG")
test "$REGISTERED_RUN_ID" = "$RUN_ID" || { echo "AF-00A caller run id differs from release" >&2; exit 2; }
test "$(git rev-list --parents -n 1 "$EXPECTED_RELEASE_COMMIT")" = "$EXPECTED_RELEASE_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || { echo "AF-00A release is not the direct implementation child" >&2; exit 2; }
jq -e --arg run "$RUN_ID" --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" '
  .config_status == "released_exact_single_canary"
  and .ready_to_run == true and .blocked_on == []
  and .preregistration.implementation_authorized == true
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
  and .execution_release.decision_artifact == "docs/decisions/0051-release-actual-forward-cem-canary.md"
  and .execution_release.allowed_release_diff_paths == [
    "configs/experiments/r05a_actual_forward_canary.json",
    "docs/decisions/0051-release-actual-forward-cem-canary.md"
  ]
  and .execution_release.automatic_resubmission_allowed == false
  and .execution_release.automatic_next_experiment_allowed == false
  and .provisional_resource_contract.host_memory_mib == 65536
  and .flow_contract.ordinary_residual_schedule_path_only == true
  and .flow_contract.autograd_allowed == false
  and .flow_contract.inverse_flow_teacher_allowed == false
  and .request_ledger.complete_run_exact_policy_request_count == 534
' "$CONFIG" >/dev/null || { echo "AF-00A exact release contract changed" >&2; exit 2; }
test -f "$RELEASE_ADR" && test ! -L "$RELEASE_ADR" || { echo "missing or symlinked AF-00A release artifact: $RELEASE_ADR" >&2; exit 2; }

PARENT_CONFIG=$(git show "$ACCEPTED_IMPLEMENTATION_COMMIT:$CONFIG")
jq -e '.ready_to_run == false and (.blocked_on|length)>0 and .preregistration.h100_submission_authorized == false and .execution_release == null' <<<"$PARENT_CONFIG" >/dev/null || { echo "AF-00A implementation parent was not fail-closed" >&2; exit 2; }
test "$(jq -cS 'del(.config_status,.ready_to_run,.blocked_on,.preregistration.h100_submission_authorized,.execution_release)' <<<"$PARENT_CONFIG")" = \
  "$(jq -cS 'del(.config_status,.ready_to_run,.blocked_on,.preregistration.h100_submission_authorized,.execution_release)' "$CONFIG")" || {
  echo "AF-00A release changed scientific content" >&2; exit 2
}
RELEASE_DIFF=$(git diff --name-only "$ACCEPTED_IMPLEMENTATION_COMMIT" "$EXPECTED_RELEASE_COMMIT")
EXPECTED_RELEASE_DIFF=$(printf '%s\n' "$CONFIG" "$RELEASE_ADR" | LC_ALL=C sort)
test "$(printf '%s\n' "$RELEASE_DIFF" | LC_ALL=C sort)" = "$EXPECTED_RELEASE_DIFF" || {
  echo "AF-00A release diff must contain exactly the config and ADR-0051" >&2; exit 2
}

# This local command performs shell-only control-plane inspection.  It does
# not require a currently free H100; a healthy busy worker may queue AF-00A.
scripts/hpc/preflight.sh

ssh "$HOST" bash -s -- "$REMOTE_REPO" "$RUN_ID" "$EXPECTED_RELEASE_COMMIT" "$ACCEPTED_IMPLEMENTATION_COMMIT" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
expected_commit=$3
accepted_implementation=$4
experiment_root=/mnt/data/quanth/experiments/crfs-oracle
run_root=$experiment_root/$run_id
case_dir=$run_root/crfs-1069f29a8d76463a
config=$remote_repo/configs/experiments/r05a_actual_forward_canary.json
legacy_config=$remote_repo/configs/experiments/r05a_inverse_flow_canary.json
manifest=$remote_repo/manifests/r05a_inverse_flow_teacher_smoke.jsonl
source_r02=/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a/crfs-1069f29a8d76463a/r02-paired.json
checkpoint=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors
r02_config=$remote_repo/configs/experiments/r02_oracle_flow.json
checkpoint_config=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/config.json
normalization_asset=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json
gpu_slurm=$remote_repo/slurm/r05a_actual_forward_canary_h100.sbatch
cpu_slurm=$remote_repo/slurm/r05a_actual_forward_canary_validate_cpu.sbatch
release_adr=$remote_repo/docs/decisions/0051-release-actual-forward-cem-canary.md
provisional_gpu_receipt=$run_root/provisional-gpu-job-id.json
held_gpu_submission=$run_root/held-gpu-submission.json
source_contract=$run_root/source-contract.json
provisional_cpu_receipt=$run_root/provisional-cpu-job-id.json
submission_receipt=$run_root/submission.json
release_fingerprint=$run_root/final-pre-release-fingerprint.json
release_fingerprint_sha_file=$run_root/final-pre-release-fingerprint.sha256
gpu_task_job_record_path=$run_root/held-gpu-task-job-record.txt
gpu_parent_job_record_path=$run_root/held-gpu-parent-job-record.txt
cpu_job_record_path=$run_root/cpu-afterany-job-record.txt

require_job_field() {
  local record=$1 field=$2 label=$3
  case " $record " in
    *" $field "*) ;;
    *) echo "$label changed: $field" >&2; return 1 ;;
  esac
}

tres_values() {
  local record=$1 key=$2 req_tres
  req_tres=$(printf '%s\n' "$record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
  printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= -v key="$key" '$1==key {print $2}'
}

require_single_tres() {
  local record=$1 key=$2 expected=$3 label=$4 values count
  values=$(tres_values "$record" "$key")
  count=$(printf '%s\n' "$values" | awk 'NF {count += 1} END {print count + 0}')
  test "$count" = 1 || { echo "$label must contain exactly one $key ReqTRES" >&2; return 1; }
  test "$values" = "$expected" || { echo "$label $key ReqTRES changed: $values" >&2; return 1; }
}

require_single_memory_tres() {
  local record=$1 expected_mib=$2 label=$3 values count
  values=$(tres_values "$record" mem)
  count=$(printf '%s\n' "$values" | awk 'NF {count += 1} END {print count + 0}')
  test "$count" = 1 || { echo "$label must contain exactly one mem ReqTRES" >&2; return 1; }
  case "$expected_mib:$values" in
    65536:64G|65536:65536M|8192:8G|8192:8192M) ;;
    *) echo "$label mem ReqTRES changed: $values" >&2; return 1 ;;
  esac
}

require_exact_tres_keys() {
  local record=$1 expected=$2 label=$3 req_tres observed billing_count
  req_tres=$(printf '%s\n' "$record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
  observed=$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= 'NF && $1!="billing" {print $1}' | LC_ALL=C sort)
  billing_count=$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= '$1=="billing" {count += 1} END {print count + 0}')
  test "$billing_count" -le 1 || { echo "$label has duplicate billing ReqTRES" >&2; return 1; }
  test "$observed" = "$expected" || { echo "$label ReqTRES key set changed: $observed" >&2; return 1; }
}

validate_gpu_records() {
  local gpu_id=$1 task_record=$2 parent_record=$3 array_task_id
  for field in \
    "JobId=${gpu_id}_0" "ArrayJobId=$gpu_id" "ArrayTaskId=0" \
    "JobState=PENDING" "Reason=JobHeldUser" "ReqNodeList=worker-1" \
    "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=02:00:00" \
    "Requeue=0" "NumNodes=1" "NumCPUs=8" "CPUs/Task=8"; do
    require_job_field "$task_record" "$field" "held AF-00A GPU task"
  done
  require_job_field "$parent_record" "ArrayJobId=$gpu_id" "held AF-00A GPU parent"
  array_task_id=$(printf '%s\n' "$parent_record" | sed -n 's/.* ArrayTaskId=\([^ ]*\).*/\1/p')
  case "$array_task_id" in 0|0%1) ;; *) echo "held AF-00A GPU parent ArrayTaskId changed: $array_task_id" >&2; return 1 ;; esac
  require_job_field "$parent_record" "ArrayTaskThrottle=1" "held AF-00A GPU parent"
  require_single_tres "$task_record" cpu 8 "held AF-00A GPU task"
  require_single_memory_tres "$task_record" 65536 "held AF-00A GPU task"
  require_single_tres "$task_record" node 1 "held AF-00A GPU task"
  require_single_tres "$task_record" gres/gpu 1 "held AF-00A GPU task"
  require_exact_tres_keys "$task_record" "$(printf '%s\n' cpu gres/gpu mem node | LC_ALL=C sort)" "held AF-00A GPU task"
  test "$(printf '%s\n' "$task_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p' | tr ',' '\n' | awk -F= '$1 ~ /^gres\/gpu/ {print}')" = "gres/gpu=1" || {
    echo "held AF-00A GPU task has non-exact GPU ReqTRES" >&2; return 1
  }
}

validate_cpu_record() {
  local cpu_id=$1 gpu_id=$2 record=$3 observed_dependency
  for field in \
    "JobId=$cpu_id" "JobState=PENDING" "Partition=main" "Account=normal" \
    "QOS=normal" "TimeLimit=00:15:00" "Requeue=0" "NumNodes=1" \
    "NumCPUs=2" "CPUs/Task=2"; do
    require_job_field "$record" "$field" "AF-00A CPU publisher"
  done
  case " $record " in
    *" ArrayJobId="*|*" ArrayTaskId="*|*" ArrayTaskThrottle="*)
      echo "AF-00A CPU publisher unexpectedly became an array" >&2; return 1 ;;
  esac
  require_single_tres "$record" cpu 2 "AF-00A CPU publisher"
  require_single_memory_tres "$record" 8192 "AF-00A CPU publisher"
  require_single_tres "$record" node 1 "AF-00A CPU publisher"
  require_exact_tres_keys "$record" "$(printf '%s\n' cpu mem node | LC_ALL=C sort)" "AF-00A CPU publisher"
  if printf '%s\n' "$record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p' | tr ',' '\n' | awk -F= '$1 ~ /^gres\/gpu/ {found=1} END {exit !found}'; then
    echo "AF-00A CPU publisher unexpectedly requests a GPU" >&2; return 1
  fi
  case " $record " in *" TRESPerNode=gres/gpu:"*) echo "AF-00A CPU publisher has per-node GPU resources" >&2; return 1 ;; esac
  observed_dependency=$(printf '%s\n' "$record" | sed -n 's/.* Dependency=\([^ ]*\).*/\1/p' | sed 's/(unfulfilled)//g; s/_\*//g')
  test "$observed_dependency" = "afterany:$gpu_id" || {
    echo "AF-00A CPU dependency changed: $observed_dependency" >&2; return 1
  }
}

bound_paths=(
  configs/experiments/r05a_actual_forward_canary.json
  configs/experiments/r05a_inverse_flow_canary.json
  configs/experiments/r02_oracle_flow.json
  manifests/r05a_inverse_flow_teacher_smoke.jsonl
  docs/decisions/0050-preregister-actual-forward-cem-teacher-canary.md
  docs/decisions/0051-release-actual-forward-cem-canary.md
  src/crfs_harness/artifacts.py
  src/crfs_harness/manifest.py
  main/crfs_oracle/r02_runner.py
  main/crfs_oracle/r03a_runner.py
  main/crfs_oracle/r05a_canary.py
  main/crfs_oracle/r05a_actual_forward_search.py
  main/crfs_oracle/r05a_actual_forward_canary.py
  main/crfs_oracle/r05a_actual_forward_validation.py
  main/crfs_oracle/r05a_full_lifetime_telemetry.py
  main/crfs_oracle/progress_calibration.py
  main/crfs_oracle/reach_progress.py
  main/crfs_oracle/runner.py
  main/run_crfs_r05a_actual_forward_canary.py
  main/publish_crfs_r05a_actual_forward_canary.py
  schemas/r05a-actual-forward-canary-envelope.schema.json
  tests/test_r05a_actual_forward_preregistration.py
  tests/test_r05a_actual_forward_search.py
  tests/test_r05a_actual_forward_canary.py
  tests/test_r05a_actual_forward_validation.py
  openpi/scripts/serve_policy.py
  openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py
  openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py
  openpi/src/openpi/serving/websocket_policy_server.py
  openpi/src/openpi/policies/policy.py
  openpi/src/openpi/models_pytorch/pi0_pytorch.py
  scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh
  scripts/hpc/lib/slurm_exact_array_task_status.sh
  scripts/hpc/lib/r05a_runtime_identity.sh
  scripts/hpc/prepare_transformers_overlay.sh
  scripts/hpc/prepare_jsonschema_overlay.sh
  scripts/hpc/run_r05a_actual_forward_workload.sh
  scripts/hpc/run_r05a_actual_forward_canary.sh
  scripts/hpc/validate_r05a_actual_forward_canary.sh
  scripts/hpc/submit_r05a_actual_forward_canary.sh
  slurm/r05a_actual_forward_canary_h100.sbatch
  slurm/r05a_actual_forward_canary_validate_cpu.sbatch
)
for relative in "${bound_paths[@]}"; do
  test -f "$remote_repo/$relative" && test ! -L "$remote_repo/$relative" || { echo "missing remote AF-00A source: $relative" >&2; exit 2; }
done
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || { echo "remote AF-00A commit mismatch" >&2; exit 2; }
test -z "$(git -C "$remote_repo" status --porcelain)" || { echo "remote AF-00A tree is dirty" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$expected_commit" || { echo "remote origin AF-00A release differs" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-list --parents -n 1 "$expected_commit")" = "$expected_commit $accepted_implementation" || { echo "remote AF-00A release parent differs" >&2; exit 2; }
remote_release_diff=$(git -C "$remote_repo" diff --name-only "$accepted_implementation" "$expected_commit" | LC_ALL=C sort)
expected_remote_release_diff=$(printf '%s\n' \
  configs/experiments/r05a_actual_forward_canary.json \
  docs/decisions/0051-release-actual-forward-cem-canary.md | LC_ALL=C sort)
test "$remote_release_diff" = "$expected_remote_release_diff" || { echo "remote AF-00A release diff is not the exact two-path release" >&2; exit 2; }
test -f "$release_adr" && test ! -L "$release_adr" || { echo "remote AF-00A release ADR is missing or symlinked" >&2; exit 2; }
test "$(jq -er '.execution_release.run_id' "$config")" = "$run_id" || { echo "remote AF-00A run id changed" >&2; exit 2; }
test "$(jq -er '.execution_release.accepted_implementation_commit' "$config")" = "$accepted_implementation" || { echo "remote AF-00A implementation changed" >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633
test "$(sha256sum "$source_r02" | awk '{print $1}')" = 055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593
test -f "$checkpoint" && test ! -L "$checkpoint" && test -r "$checkpoint" || { echo "canonical AF-00A checkpoint is missing, symlinked, or unreadable" >&2; exit 2; }
test "$(sha256sum "$r02_config" | awk '{print $1}')" = c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e
for path in "$checkpoint_config" "$normalization_asset"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked AF-00A checkpoint metadata: $path" >&2; exit 2; }
done
test "$(sha256sum "$checkpoint_config" | awk '{print $1}')" = 5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a
test "$(sha256sum "$normalization_asset" | awk '{print $1}')" = b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84
inherited_sbatch_options=$(env | sed -n 's/^\(SBATCH_[A-Za-z0-9_]*\)=.*/\1/p')
test -z "$inherited_sbatch_options" || { echo "inherited SBATCH options could change AF-00A: $inherited_sbatch_options" >&2; exit 2; }
test ! -e "$run_root" || { echo "immutable AF-00A run id is already used" >&2; exit 2; }
if [ -n "$(squeue -h -u "$(whoami)" -o '%i')" ]; then echo "AF-00A not submitted: user queue is not empty" >&2; exit 2; fi

node_record=$(scontrol show node worker-1 -o)
node_state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$node_state" in *down*|*drain*|*fail*|*maint*|*not_resp*|*unknown*|*power*) echo "worker-1 unhealthy: $node_state" >&2; exit 2 ;; esac
free_mem=$(printf '%s\n' "$node_record" | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
case "$free_mem" in *[!0-9]*|'') free_mem=0 ;; esac
case " $node_record " in *" Gres="*"gpu:nvidia_h100_80gb_hbm3:8"*) ;; *) echo "worker-1 does not advertise H100" >&2; exit 2 ;; esac

mkdir "$run_root"
preflight_tmp=$(mktemp "$run_root/.vinuni-preflight.XXXXXX")
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "[jobs]"
  squeue -u "$(whoami)" -o '%.18i %.32j %.2t %.10M %.12l %.6D %R'
  echo "[worker-1]"
  printf '%s\n' "$node_record"
  echo "[storage]"
  df -h /mnt/data
} >"$preflight_tmp"
mv "$preflight_tmp" "$run_root/vinuni-preflight.txt"
storage_percent=$(df -P /mnt/data | awk 'NR==2 {gsub(/%/,"",$5); print $5}')
case "$storage_percent" in *[!0-9]*|'') echo "cannot parse /mnt/data utilization" >&2; exit 2 ;; esac
test "$storage_percent" -lt 90 || { echo "/mnt/data is ${storage_percent}% full" >&2; exit 2; }

resources=$(jq -n '{partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}')
reservation_tmp=$(mktemp "$run_root/.launch-reservation.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg implementation "$accepted_implementation" --arg free_mem "$free_mem" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson resources "$resources" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_canary_launch_reservation",status:"preflight_verified_before_queueing",run_id:$run,git_commit:$commit,accepted_implementation_commit:$implementation,source_node:"worker-1",observed_free_mem_mib:($free_mem|tonumber),resources:$resources,single_submission:true,scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$reservation_tmp"
mv "$reservation_tmp" "$run_root/launch-reservation.json"

gpu_submission=$(RUN_ID="$run_id" EXPECTED_GIT_COMMIT="$expected_commit" REMOTE_REPO="$remote_repo" \
  SOURCE_CONTRACT="$source_contract" SUBMISSION="$submission_receipt" \
  RELEASE_FINGERPRINT="$release_fingerprint" RELEASE_FINGERPRINT_SHA256_FILE="$release_fingerprint_sha_file" \
  sbatch --parsable --hold --export=ALL --partition=main --account=normal --qos=normal --gres=gpu:1 \
  --cpus-per-task=8 --mem=64G --time=02:00:00 --no-requeue --nodelist=worker-1 \
  --array=0-0%1 --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' "$gpu_slurm")
gpu_job_id=${gpu_submission%%;*}
case "$gpu_job_id" in *[!0-9]*|'') echo "invalid AF-00A GPU job id" >&2; exit 2 ;; esac
printf 'provisional_gpu_job_id=%s\n' "$gpu_job_id" >&2
provisional_gpu_tmp=$(mktemp "$run_root/.provisional-gpu-job-id.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_canary_provisional_gpu_job_id",status:"sbatch_returned_numeric_id_before_field_validation",run_id:$run,git_commit:$commit,gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),held:true,source_node:"worker-1",automatic_cancellation_allowed:false,scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$provisional_gpu_tmp"
mv "$provisional_gpu_tmp" "$provisional_gpu_receipt"

gpu_task_record=$(scontrol show job "${gpu_job_id}_0" -o)
gpu_parent_record=$(scontrol show job "$gpu_job_id" -o)
validate_gpu_records "$gpu_job_id" "$gpu_task_record" "$gpu_parent_record"

held_tmp=$(mktemp "$run_root/.held-gpu.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson resources "$resources" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_canary_held_gpu_submission",status:"sbatch_returned_held_gpu_id",run_id:$run,git_commit:$commit,gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),source_node:"worker-1",resources:$resources,released_at_receipt_time:false,scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$held_tmp"
mv "$held_tmp" "$held_gpu_submission"
held_gpu_submission_sha=$(sha256sum "$held_gpu_submission" | awk '{print $1}')

repository_hashes='{}'
for relative in "${bound_paths[@]}"; do
  digest=$(sha256sum "$remote_repo/$relative" | awk '{print $1}')
  repository_hashes=$(jq -c --arg key "$relative" --arg value "$digest" '. + {($key):$value}' <<<"$repository_hashes")
done
while IFS= read -r relative; do
  test -n "$relative" || continue
  digest=$(sha256sum "$remote_repo/$relative" | awk '{print $1}')
  repository_hashes=$(jq -c --arg key "$relative" --arg value "$digest" '. + {($key):$value}' <<<"$repository_hashes")
done < <(git -C "$remote_repo" ls-files 'tests/test_r05a_actual_forward*.py' 'schemas/*actual-forward*')

artifact_paths=$(jq -n --arg run_root "$run_root" --arg case_dir "$case_dir" \
  --arg held "$held_gpu_submission" --arg submission "$submission_receipt" --arg fingerprint "$release_fingerprint" \
  '{run_root:$run_root,case_dir:$case_dir,payload:($case_dir+"/af00a-raw-payload.json"),raw_tensors:($case_dir+"/af00a-tensors.npz"),query_ledger:($case_dir+"/query-ledger.json"),host_telemetry:($case_dir+"/host-cgroup-sampled-current.tsv"),gpu_samples:($case_dir+"/gpu-memory-samples.csv"),allocation_tests_log:($case_dir+"/allocation-focused-tests.log"),hidden_candidate:($case_dir+"/.results.candidate.json"),result:($case_dir+"/results.json"),validation_receipt:($run_root+"/cpu-afterany-validation.json"),held_gpu_submission:$held,submission:$submission,release_fingerprint:$fingerprint,live_preflight:($run_root+"/vinuni-preflight.txt")}')
source_tmp=$(mktemp "$run_root/.source-contract.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg config_sha "$(sha256sum "$config" | awk '{print $1}')" --arg legacy_sha "$(sha256sum "$legacy_config" | awk '{print $1}')" \
  --arg manifest_sha "$(sha256sum "$manifest" | awk '{print $1}')" --arg source_sha "$(sha256sum "$source_r02" | awk '{print $1}')" \
  --arg r02_config_sha "$(sha256sum "$r02_config" | awk '{print $1}')" \
  --arg checkpoint_sha 988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed \
  --arg checkpoint_config_sha "$(sha256sum "$checkpoint_config" | awk '{print $1}')" \
  --arg normalization_sha "$(sha256sum "$normalization_asset" | awk '{print $1}')" \
  --arg preflight_sha "$(sha256sum "$run_root/vinuni-preflight.txt" | awk '{print $1}')" \
  --arg held "$held_gpu_submission" --arg held_sha "$held_gpu_submission_sha" \
  --arg provisional_gpu "$provisional_gpu_receipt" --arg provisional_gpu_sha "$(sha256sum "$provisional_gpu_receipt" | awk '{print $1}')" \
  --argjson resources "$resources" --argjson paths "$artifact_paths" --argjson hashes "$repository_hashes" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_canary_source_contract",status:"gpu_held_sources_bound_before_cpu_submission",run_id:$run,git_commit:$commit,git_dirty:false,source_node:"worker-1",gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),resources:$resources,artifact_paths:$paths,repository_file_sha256:$hashes,provisional_gpu_job_id_path:$provisional_gpu,provisional_gpu_job_id_sha256:$provisional_gpu_sha,held_gpu_submission_path:$held,held_gpu_submission_sha256:$held_sha,frozen_bindings:{actual_config_sha256:$config_sha,legacy_config_sha256:$legacy_sha,manifest_sha256:$manifest_sha,source_r02_sha256:$source_sha,source_r02_config_sha256:$r02_config_sha,checkpoint_sha256:$checkpoint_sha,checkpoint_config_sha256:$checkpoint_config_sha,normalization_asset_sha256:$normalization_sha,live_preflight_sha256:$preflight_sha},scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$source_tmp"
mv "$source_tmp" "$source_contract"
source_contract_sha=$(sha256sum "$source_contract" | awk '{print $1}')

dependency=afterany:$gpu_job_id
cpu_submission=$(RUN_ID="$run_id" SOURCE_JOB_ID="$gpu_job_id" SOURCE_CONTRACT="$source_contract" \
  EXPECTED_SOURCE_CONTRACT_SHA256="$source_contract_sha" HELD_GPU_SUBMISSION="$held_gpu_submission" \
  EXPECTED_HELD_GPU_SUBMISSION_SHA256="$held_gpu_submission_sha" SUBMISSION="$submission_receipt" \
  RELEASE_FINGERPRINT="$release_fingerprint" RELEASE_FINGERPRINT_SHA256_FILE="$release_fingerprint_sha_file" \
  EXPECTED_GIT_COMMIT="$expected_commit" REMOTE_REPO="$remote_repo" \
  sbatch --parsable --export=ALL --partition=main --account=normal --qos=normal --cpus-per-task=2 --mem=8G \
  --time=00:15:00 --no-requeue --dependency="$dependency" \
  --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' "$cpu_slurm")
cpu_job_id=${cpu_submission%%;*}
case "$cpu_job_id" in *[!0-9]*|'') echo "invalid AF-00A CPU job id" >&2; exit 2 ;; esac
printf 'provisional_cpu_publisher_job_id=%s\n' "$cpu_job_id" >&2
provisional_cpu_tmp=$(mktemp "$run_root/.provisional-cpu-job-id.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg cpu "$cpu_job_id" --arg dependency "$dependency" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_canary_provisional_cpu_job_id",status:"sbatch_returned_numeric_id_before_field_validation",run_id:$run,git_commit:$commit,gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),cpu_afterany_job_id:$cpu,dependency:$dependency,automatic_cancellation_allowed:false,scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$provisional_cpu_tmp"
mv "$provisional_cpu_tmp" "$provisional_cpu_receipt"
cpu_record=$(scontrol show job "$cpu_job_id" -o)
validate_cpu_record "$cpu_job_id" "$gpu_job_id" "$cpu_record"

submission_tmp=$(mktemp "$run_root/.submission.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg cpu "$cpu_job_id" --arg dependency "$dependency" \
  --arg source_contract "$source_contract" --arg source_sha "$source_contract_sha" \
  --arg held "$held_gpu_submission" --arg held_sha "$held_gpu_submission_sha" \
  --arg provisional_gpu "$provisional_gpu_receipt" --arg provisional_gpu_sha "$(sha256sum "$provisional_gpu_receipt" | awk '{print $1}')" \
  --arg provisional_cpu "$provisional_cpu_receipt" --arg provisional_cpu_sha "$(sha256sum "$provisional_cpu_receipt" | awk '{print $1}')" \
  --arg fingerprint "$release_fingerprint" --arg result "$case_dir/results.json" --arg receipt "$run_root/cpu-afterany-validation.json" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_canary_atomic_submission",status:"cpu_afterany_registered_gpu_held",run_id:$run,git_commit:$commit,source_node:"worker-1",gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),cpu_afterany_job_id:$cpu,dependency:$dependency,source_contract_path:$source_contract,source_contract_sha256:$source_sha,held_gpu_submission_path:$held,held_gpu_submission_sha256:$held_sha,provisional_gpu_job_id_path:$provisional_gpu,provisional_gpu_job_id_sha256:$provisional_gpu_sha,provisional_cpu_job_id_path:$provisional_cpu,provisional_cpu_job_id_sha256:$provisional_cpu_sha,release_fingerprint_path:$fingerprint,expected_result:$result,expected_validation_receipt:$receipt,released_at_receipt_time:false,scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$submission_tmp"
mv "$submission_tmp" "$submission_receipt"
submission_sha=$(sha256sum "$submission_receipt" | awk '{print $1}')

# Final pre-release fingerprint.  Re-query and bind the still-held exact GPU
# array task, its parent/throttle record, and the exact CPU afterany job before
# the sole release operation.
gpu_task_record=$(scontrol show job "${gpu_job_id}_0" -o)
gpu_parent_record=$(scontrol show job "$gpu_job_id" -o)
cpu_record=$(scontrol show job "$cpu_job_id" -o)
validate_gpu_records "$gpu_job_id" "$gpu_task_record" "$gpu_parent_record"
validate_cpu_record "$cpu_job_id" "$gpu_job_id" "$cpu_record"
gpu_task_record_tmp=$(mktemp "$run_root/.held-gpu-task-job-record.XXXXXX")
gpu_parent_record_tmp=$(mktemp "$run_root/.held-gpu-parent-job-record.XXXXXX")
cpu_job_record_tmp=$(mktemp "$run_root/.cpu-afterany-job-record.XXXXXX")
printf '%s\n' "$gpu_task_record" >"$gpu_task_record_tmp"
printf '%s\n' "$gpu_parent_record" >"$gpu_parent_record_tmp"
printf '%s\n' "$cpu_record" >"$cpu_job_record_tmp"
mv "$gpu_task_record_tmp" "$gpu_task_job_record_path"
mv "$gpu_parent_record_tmp" "$gpu_parent_job_record_path"
mv "$cpu_job_record_tmp" "$cpu_job_record_path"

fingerprint_tmp=$(mktemp "$run_root/.release-fingerprint.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg cpu "$cpu_job_id" --arg dependency "$dependency" \
  --arg source "$source_contract" --arg source_sha "$source_contract_sha" \
  --arg held "$held_gpu_submission" --arg held_sha "$held_gpu_submission_sha" \
  --arg submission "$submission_receipt" --arg submission_sha "$submission_sha" \
  --arg provisional_gpu "$provisional_gpu_receipt" --arg provisional_gpu_sha "$(sha256sum "$provisional_gpu_receipt" | awk '{print $1}')" \
  --arg provisional_cpu "$provisional_cpu_receipt" --arg provisional_cpu_sha "$(sha256sum "$provisional_cpu_receipt" | awk '{print $1}')" \
  --arg gpu_task_record "$gpu_task_job_record_path" --arg gpu_task_record_sha "$(sha256sum "$gpu_task_job_record_path" | awk '{print $1}')" \
  --arg gpu_parent_record "$gpu_parent_job_record_path" --arg gpu_parent_record_sha "$(sha256sum "$gpu_parent_job_record_path" | awk '{print $1}')" \
  --arg cpu_record "$cpu_job_record_path" --arg cpu_record_sha "$(sha256sum "$cpu_job_record_path" | awk '{print $1}')" \
  --arg result "$case_dir/results.json" --arg receipt "$run_root/cpu-afterany-validation.json" \
  --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_canary_final_pre_release_fingerprint",status:"all_receipts_bound_gpu_still_held",run_id:$run,git_commit:$commit,source_node:"worker-1",gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),cpu_afterany_job_id:$cpu,dependency:$dependency,source_contract_path:$source,source_contract_sha256:$source_sha,held_gpu_submission_path:$held,held_gpu_submission_sha256:$held_sha,submission_path:$submission,submission_sha256:$submission_sha,expected_result:$result,expected_validation_receipt:$receipt,released_at_fingerprint_time:false,bindings:{provisional_gpu_job_id:{path:$provisional_gpu,sha256:$provisional_gpu_sha},held_gpu_submission:{path:$held,sha256:$held_sha},source_contract:{path:$source,sha256:$source_sha},provisional_cpu_job_id:{path:$provisional_cpu,sha256:$provisional_cpu_sha},atomic_submission:{path:$submission,sha256:$submission_sha},held_gpu_task_job_record:{path:$gpu_task_record,sha256:$gpu_task_record_sha},held_gpu_parent_job_record:{path:$gpu_parent_record,sha256:$gpu_parent_record_sha},cpu_afterany_job_record:{path:$cpu_record,sha256:$cpu_record_sha}},scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$fingerprint_tmp"
mv "$fingerprint_tmp" "$release_fingerprint"
release_fingerprint_sha=$(sha256sum "$release_fingerprint" | awk '{print $1}')
fingerprint_sha_tmp=$(mktemp "$run_root/.release-fingerprint-sha256.XXXXXX")
printf '%s\n' "$release_fingerprint_sha" >"$fingerprint_sha_tmp"
mv "$fingerprint_sha_tmp" "$release_fingerprint_sha_file"

jq -e --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg cpu "$cpu_job_id" \
  --arg dependency "$dependency" --arg source "$source_contract" --arg source_sha "$source_contract_sha" \
  --arg held "$held_gpu_submission" --arg held_sha "$held_gpu_submission_sha" \
  --arg submission "$submission_receipt" --arg submission_sha "$submission_sha" '
  .artifact_role == "r05a_actual_forward_canary_final_pre_release_fingerprint"
  and .status == "all_receipts_bound_gpu_still_held"
  and .run_id == $run and .git_commit == $commit and .source_node == "worker-1"
  and .gpu_slurm_array_job_id == $gpu and .gpu_slurm_array_task_id == 0
  and .exact_gpu_task_id == ($gpu+"_0") and .cpu_afterany_job_id == $cpu
  and .dependency == $dependency and .source_contract_path == $source
  and .source_contract_sha256 == $source_sha and .held_gpu_submission_path == $held
  and .held_gpu_submission_sha256 == $held_sha and .submission_path == $submission
  and .submission_sha256 == $submission_sha and .released_at_fingerprint_time == false
  and .expected_result == ($source | sub("/source-contract.json$"; "/crfs-1069f29a8d76463a/results.json"))
  and .expected_validation_receipt == ($source | sub("/source-contract.json$"; "/cpu-afterany-validation.json"))
  and .scientific_claim_allowed == false and .simulator_efficacy_claim_allowed == false
  and .infeasibility_claim_allowed == false and .probe_training_authorized == false
  ' "$release_fingerprint" >/dev/null || { echo "AF-00A final pre-release fingerprint changed" >&2; exit 2; }
while IFS='|' read -r key path; do
  expected_path=$(jq -er --arg key "$key" '.bindings[$key].path' "$release_fingerprint")
  expected_sha=$(jq -er --arg key "$key" '.bindings[$key].sha256' "$release_fingerprint")
  test "$expected_path" = "$path" || { echo "AF-00A fingerprint path changed: $key" >&2; exit 2; }
  test -f "$path" && test ! -L "$path" || { echo "AF-00A fingerprint input missing or symlinked: $key" >&2; exit 2; }
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected_sha" || { echo "AF-00A fingerprint digest changed: $key" >&2; exit 2; }
done <<EOF
provisional_gpu_job_id|$provisional_gpu_receipt
held_gpu_submission|$held_gpu_submission
source_contract|$source_contract
provisional_cpu_job_id|$provisional_cpu_receipt
atomic_submission|$submission_receipt
held_gpu_task_job_record|$gpu_task_job_record_path
held_gpu_parent_job_record|$gpu_parent_job_record_path
cpu_afterany_job_record|$cpu_job_record_path
EOF
test "$(cat "$release_fingerprint_sha_file")" = "$release_fingerprint_sha" || { echo "AF-00A fingerprint digest sidecar changed" >&2; exit 2; }
test "$(wc -l <"$release_fingerprint_sha_file" | tr -d ' ')" = 1 || { echo "AF-00A fingerprint digest sidecar is not one line" >&2; exit 2; }

# One final semantic scheduler check closes the gap between fingerprinting and
# release without mutating any job or receipt.
validate_gpu_records "$gpu_job_id" "$(scontrol show job "${gpu_job_id}_0" -o)" "$(scontrol show job "$gpu_job_id" -o)"
validate_cpu_record "$cpu_job_id" "$gpu_job_id" "$(scontrol show job "$cpu_job_id" -o)"
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$remote_repo" status --porcelain)"
test "$(git -C "$remote_repo" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$expected_commit"
for relative in "${bound_paths[@]}"; do
  test "$(sha256sum "$remote_repo/$relative" | awk '{print $1}')" = "$(jq -er --arg path "$relative" '.repository_file_sha256[$path]' "$source_contract")" || {
    echo "AF-00A final source fingerprint changed: $relative" >&2; exit 2
  }
done
test "$(sha256sum "$source_contract" | awk '{print $1}')" = "$source_contract_sha"
test "$(sha256sum "$submission_receipt" | awk '{print $1}')" = "$submission_sha"

scontrol release "$gpu_job_id"
echo "submitted_gpu_job_id=$gpu_job_id"
echo "submitted_cpu_publisher_job_id=$cpu_job_id"
echo "dependency=$dependency"
echo "source_contract_sha256=$source_contract_sha"
echo "submission_receipt=$submission_receipt"
echo "submission_receipt_sha256=$submission_sha"
echo "release_fingerprint=$release_fingerprint"
echo "release_fingerprint_sha256=$release_fingerprint_sha"
REMOTE
