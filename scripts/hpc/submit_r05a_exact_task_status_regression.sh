#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?usage: RUN_ID=r05a-exact-array-task-afterany-regression-20260715a scripts/hpc/submit_r05a_exact_task_status_regression.sh}"
EXPECTED_RUN_ID=r05a-exact-array-task-afterany-regression-20260715a
test "$RUN_ID" = "$EXPECTED_RUN_ID" || { echo "exact-task regression run id changed" >&2; exit 2; }

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
EXPERIMENT_ROOT=/mnt/data/quanth/experiments/crfs-oracle
SLURM_LOG_ROOT=/mnt/data/quanth/slurm_logs/crfs-oracle
BOUND_PATHS=(
  docs/decisions/0039-preserve-sampled-current-launch-b-and-repair-exact-task-accounting.md
  scripts/hpc/lib/slurm_exact_array_task_status.sh
  scripts/hpc/validate_r05a_sampled_current_canary.sh
  scripts/hpc/run_r05a_exact_task_status_source.sh
  scripts/hpc/validate_r05a_exact_task_status_regression.sh
  scripts/hpc/submit_r05a_exact_task_status_regression.sh
  slurm/r05a_exact_task_status_source_cpu.sbatch
  slurm/r05a_exact_task_status_validate_cpu.sbatch
  tests/test_r05a_exact_task_status_regression.py
)
for path in "${BOUND_PATHS[@]}"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked regression source: $path" >&2; exit 2; }
done
git ls-files --error-unmatch "${BOUND_PATHS[@]}" >/dev/null || {
  echo "exact-task regression sources must be committed" >&2; exit 2;
}
test -z "$(git status --porcelain)" || { echo "exact-task regression requires a clean worktree" >&2; exit 2; }
EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)
test "$EXPECTED_GIT_COMMIT" = "$(git rev-parse '@{upstream}')" || {
  echo "exact-task regression source is not pushed" >&2; exit 2;
}

# Shell-only control-plane preflight. It starts no experiment on the login node.
scripts/hpc/preflight.sh

ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$EXPECTED_GIT_COMMIT" "$EXPERIMENT_ROOT" "$SLURM_LOG_ROOT" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
expected_commit=$3
experiment_root=$4
slurm_log_root=$5
run_root=$experiment_root/$run_id
source_runner=$remote_repo/scripts/hpc/run_r05a_exact_task_status_source.sh
validator_runner=$remote_repo/scripts/hpc/validate_r05a_exact_task_status_regression.sh
submitter=$remote_repo/scripts/hpc/submit_r05a_exact_task_status_regression.sh
helper=$remote_repo/scripts/hpc/lib/slurm_exact_array_task_status.sh
production_validator=$remote_repo/scripts/hpc/validate_r05a_sampled_current_canary.sh
source_slurm=$remote_repo/slurm/r05a_exact_task_status_source_cpu.sbatch
validator_slurm=$remote_repo/slurm/r05a_exact_task_status_validate_cpu.sbatch
adr=$remote_repo/docs/decisions/0039-preserve-sampled-current-launch-b-and-repair-exact-task-accounting.md
test_file=$remote_repo/tests/test_r05a_exact_task_status_regression.py
for path in "$source_runner" "$validator_runner" "$submitter" "$helper" "$production_validator" "$source_slurm" "$validator_slurm" "$adr" "$test_file"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked remote regression source: $path" >&2; exit 2; }
done
command -v jq >/dev/null 2>&1 || { echo "remote jq is required" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || { echo "remote source commit differs" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$expected_commit" || { echo "remote origin ref differs" >&2; exit 2; }
test -z "$(git -C "$remote_repo" status --porcelain)" || { echo "remote source tree is dirty" >&2; exit 2; }
test ! -e "$run_root" || { echo "immutable exact-task regression run id is already used" >&2; exit 2; }
test -z "$(squeue -h -u "$(whoami)" -o '%i')" || { echo "exact-task regression requires an empty user queue" >&2; exit 2; }

node_record=$(scontrol show node worker-1 -o)
node_state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$node_state" in *down*|*drain*|*fail*|*maint*|*not_resp*|*unknown*|*power*) echo "worker-1 is unhealthy: $node_state" >&2; exit 2 ;; esac
free_mem=$(printf '%s\n' "$node_record" | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
case "$free_mem" in *[!0-9]*|'') echo "cannot parse worker-1 FreeMem" >&2; exit 2 ;; esac
test "$free_mem" -ge 512 || { echo "worker-1 has less than 512 MiB free" >&2; exit 2; }

mkdir -p "$slurm_log_root"
source_job_id=
validator_job_id=
source_released=false
run_root_created=false
control_stage=held_source_submission
control_failure() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ "$run_root_created" = true ] && [ ! -e "$run_root/submission-failure.json" ]; then
    temporary=$(mktemp "$run_root/.submission-failure.XXXXXX") || return "$status"
    jq -n \
      --arg run_id "$run_id" --arg commit "$expected_commit" \
      --arg stage "$control_stage" --arg exit_code "$status" \
      --arg source_job_id "$source_job_id" --arg validator_job_id "$validator_job_id" \
      --arg released "$source_released" \
      '{
        schema_version: "1.0",
        artifact_role: "r05a_exact_task_status_submission_failure",
        status: "failed_closed",
        run_id: $run_id,
        git_commit: $commit,
        control_stage: $stage,
        exit_code: ($exit_code | tonumber),
        source_array_job_id: (if $source_job_id == "" then null else $source_job_id end),
        validator_job_id: (if $validator_job_id == "" then null else $validator_job_id end),
        source_released: ($released == "true"),
        exact_jobs_must_be_inspected_before_cleanup: true,
        automatic_cancel_allowed: false,
        scientific_claim_allowed: false,
        h100_retry_authorized: false
      }' >"$temporary" && mv "$temporary" "$run_root/submission-failure.json"
  fi
  if [ -n "$source_job_id" ] && [ "$source_released" = false ]; then
    echo "source job $source_job_id remains held; inspect exact jobs before cleanup" >&2
  fi
  return "$status"
}
trap control_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$run_root"
run_root_created=true

source_marker=$run_root/source-marker.json
source_contract=$run_root/source-contract.json
submission_receipt=$run_root/submission.json
result=$run_root/results.json
validation_receipt=$run_root/cpu-afterany-validation.json
source_submission=$(RUN_ID="$run_id" EXPECTED_GIT_COMMIT="$expected_commit" \
  SOURCE_CONTRACT="$source_contract" SUBMISSION="$submission_receipt" SOURCE_MARKER="$source_marker" \
  REMOTE_REPO="$remote_repo" \
  sbatch --parsable --hold --partition=main --account=normal --qos=normal \
    --cpus-per-task=1 --mem=256M --time=00:02:00 --no-requeue \
    --nodelist=worker-1 --array=0-0%1 \
    --output="$slurm_log_root/%x-%A_%a.out" "$source_slurm")
source_job_id=${source_submission%%;*}
case "$source_job_id" in *[!0-9]*|'') echo "invalid source job id: $source_submission" >&2; exit 2 ;; esac

control_stage=held_source_contract_validation
source_record=$(scontrol show job "$source_job_id" -o)
for field in "JobState=PENDING" "Reason=JobHeldUser" "ReqNodeList=worker-1" "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=00:02:00" "Requeue=0"; do
  case " $source_record " in *" $field "*) ;; *) echo "held source field changed: $field" >&2; exit 2 ;; esac
done
case " $source_record " in *gres/gpu*) echo "held source requests a GPU" >&2; exit 2 ;; esac
source_req_tres=$(printf '%s\n' "$source_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test "$(printf '%s\n' "$source_req_tres" | tr ',' '\n' | sed -n 's/^cpu=//p')" = 1 || { echo "source CPU request changed" >&2; exit 2; }
case "$(printf '%s\n' "$source_req_tres" | tr ',' '\n' | sed -n 's/^mem=//p')" in 256M) ;; *) echo "source memory request changed" >&2; exit 2 ;; esac

control_stage=held_source_receipt
temporary=$(mktemp "$run_root/.held-source.XXXXXX")
jq -n --arg run_id "$run_id" --arg commit "$expected_commit" --arg source_job_id "$source_job_id" \
  '{schema_version:"1.0",artifact_role:"r05a_exact_task_status_held_source",status:"sbatch_returned_held_source",run_id:$run_id,git_commit:$commit,source_array_job_id:$source_job_id,source_array_task_id:0,exact_source_task_id:($source_job_id+"_0"),source_node:"worker-1",resources:{partition:"main",account:"normal",qos:"normal",cpus:1,host_memory_mib:256,time_limit:"00:02:00",array:"0-0%1",gpus:0,requeue:false},released_at_receipt_time:false,shell_only:true,scientific_claim_allowed:false}' >"$temporary"
mv "$temporary" "$run_root/held-source-submission.json"

bound_hashes='{}'
for path in "$adr" "$helper" "$production_validator" "$source_runner" "$validator_runner" "$submitter" "$source_slurm" "$validator_slurm" "$test_file"; do
  relative=${path#"$remote_repo/"}
  digest=$(sha256sum "$path" | awk '{print $1}')
  bound_hashes=$(jq -c --arg key "$relative" --arg value "$digest" '. + {($key):$value}' <<<"$bound_hashes")
done
artifact_paths=$(jq -n --arg marker "$source_marker" --arg submission "$submission_receipt" --arg result "$result" --arg receipt "$validation_receipt" '{source_marker:$marker,submission:$submission,result:$result,validation_receipt:$receipt}')
temporary=$(mktemp "$run_root/.source-contract.XXXXXX")
jq -n --arg run_id "$run_id" --arg commit "$expected_commit" --arg source_job_id "$source_job_id" \
  --arg held "$run_root/held-source-submission.json" --arg held_sha "$(sha256sum "$run_root/held-source-submission.json" | awk '{print $1}')" \
  --argjson hashes "$bound_hashes" --argjson paths "$artifact_paths" \
  '{schema_version:"1.0",artifact_role:"r05a_exact_task_status_source_contract",status:"held_source_bound_before_afterany_submission",run_id:$run_id,git_commit:$commit,git_dirty:false,source_array_job_id:$source_job_id,source_array_task_id:0,exact_source_task_id:($source_job_id+"_0"),source_node:"worker-1",resources:{partition:"main",account:"normal",qos:"normal",cpus:1,host_memory_mib:256,time_limit:"00:02:00",array:"0-0%1",gpus:0,requeue:false},held_source_path:$held,held_source_sha256:$held_sha,bound_file_sha256:$hashes,artifact_paths:$paths,shell_only:true,scientific_claim_allowed:false,h100_retry_authorized:false}' >"$temporary"
mv "$temporary" "$source_contract"
source_contract_sha=$(sha256sum "$source_contract" | awk '{print $1}')

control_stage=afterany_validator_submission
dependency=afterany:$source_job_id
validator_submission=$(RUN_ID="$run_id" EXPECTED_GIT_COMMIT="$expected_commit" \
  SOURCE_JOB_ID="$source_job_id" SOURCE_CONTRACT="$source_contract" \
  EXPECTED_SOURCE_CONTRACT_SHA256="$source_contract_sha" SUBMISSION="$submission_receipt" \
  SOURCE_MARKER="$source_marker" RESULT="$result" VALIDATION_RECEIPT="$validation_receipt" \
  REMOTE_REPO="$remote_repo" \
  sbatch --parsable --partition=main --account=normal --qos=normal \
    --cpus-per-task=1 --mem=256M --time=00:02:00 --no-requeue \
    --nodelist=worker-1 --dependency="$dependency" \
    --output="$slurm_log_root/%x-%j.out" "$validator_slurm")
validator_job_id=${validator_submission%%;*}
case "$validator_job_id" in *[!0-9]*|'') echo "invalid validator job id: $validator_submission" >&2; exit 2 ;; esac

control_stage=afterany_validator_contract_validation
validator_record=$(scontrol show job "$validator_job_id" -o)
for field in "JobState=PENDING" "ReqNodeList=worker-1" "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=00:02:00" "Requeue=0"; do
  case " $validator_record " in *" $field "*) ;; *) echo "validator field changed: $field" >&2; exit 2 ;; esac
done
case " $validator_record " in *gres/gpu*) echo "validator requests a GPU" >&2; exit 2 ;; esac
validator_req_tres=$(printf '%s\n' "$validator_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test "$(printf '%s\n' "$validator_req_tres" | tr ',' '\n' | sed -n 's/^cpu=//p')" = 1 || { echo "validator CPU request changed" >&2; exit 2; }
case "$(printf '%s\n' "$validator_req_tres" | tr ',' '\n' | sed -n 's/^mem=//p')" in 256M) ;; *) echo "validator memory request changed" >&2; exit 2 ;; esac
observed_dependency=$(printf '%s\n' "$validator_record" | sed -n 's/.* Dependency=\([^ ]*\).*/\1/p' | sed 's/(unfulfilled)//g; s/_\*//g')
test "$observed_dependency" = "$dependency" || { echo "validator dependency changed: $observed_dependency" >&2; exit 2; }

control_stage=atomic_submission_receipt
temporary=$(mktemp "$run_root/.submission.XXXXXX")
jq -n --arg run_id "$run_id" --arg commit "$expected_commit" \
  --arg source_job_id "$source_job_id" --arg validator_job_id "$validator_job_id" \
  --arg dependency "$dependency" --arg source_contract "$source_contract" \
  --arg source_contract_sha "$source_contract_sha" --arg marker "$source_marker" \
  --arg result "$result" --arg receipt "$validation_receipt" \
  '{schema_version:"1.0",artifact_role:"r05a_exact_task_status_atomic_submission",status:"held_source_and_afterany_registered",run_id:$run_id,git_commit:$commit,source_array_job_id:$source_job_id,source_array_task_id:0,exact_source_task_id:($source_job_id+"_0"),validator_job_id:$validator_job_id,dependency:$dependency,source_node:"worker-1",validator_node:"worker-1",source_contract_path:$source_contract,source_contract_sha256:$source_contract_sha,artifact_paths:{source_marker:$marker,result:$result,validation_receipt:$receipt},released_at_receipt_time:false,shell_only:true,gpu_allocated:false,scientific_claim_allowed:false,h100_retry_authorized:false}' >"$temporary"
mv "$temporary" "$submission_receipt"

control_stage=release_exact_source
scontrol release "$source_job_id"
source_released=true
control_stage=complete
echo "source_array_job_id=$source_job_id"
echo "source_array_task_id=${source_job_id}_0"
echo "validator_job_id=$validator_job_id"
echo "dependency=$dependency"
echo "source_contract_sha256=$source_contract_sha"
echo "submission_receipt=$submission_receipt"
echo "submission_receipt_sha256=$(sha256sum "$submission_receipt" | awk '{print $1}')"
REMOTE
