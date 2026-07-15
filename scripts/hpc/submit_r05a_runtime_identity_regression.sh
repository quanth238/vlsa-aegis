#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... EXPECTED_RELEASE_COMMIT=... scripts/hpc/submit_r05a_runtime_identity_regression.sh'
: "${RUN_ID:?$usage}"
: "${EXPECTED_RELEASE_COMMIT:?$usage}"
case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe runtime-identity RUN_ID" >&2; exit 2 ;; esac
case "$RUN_ID" in [A-Za-z0-9]*) ;; *) echo "runtime-identity RUN_ID must start alphanumeric" >&2; exit 2 ;; esac
test "${#RUN_ID}" -le 128 || { echo "runtime-identity RUN_ID is too long" >&2; exit 2; }
case "$EXPECTED_RELEASE_COMMIT" in *[!0-9a-f]*|'') echo "invalid runtime-identity release commit" >&2; exit 2 ;; esac
test "${#EXPECTED_RELEASE_COMMIT}" = 40 || { echo "invalid runtime-identity release commit length" >&2; exit 2; }

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
EXPERIMENT_ROOT=/mnt/data/quanth/experiments/crfs-oracle
SLURM_LOG_ROOT=/mnt/data/quanth/slurm_logs/crfs-oracle
CONFIG_LOCAL=configs/experiments/r05a_runtime_identity_regression.json
ADR0043_LOCAL=docs/decisions/0043-preregister-runtime-identity-regression.md
ADR0044_LOCAL=docs/decisions/0044-release-runtime-identity-regression.md

BOUND_PATHS=(
  configs/experiments/r05a_runtime_identity_regression.json
  docs/decisions/0043-preregister-runtime-identity-regression.md
  docs/decisions/0044-release-runtime-identity-regression.md
  scripts/hpc/lib/r05a_runtime_identity.sh
  scripts/hpc/run_r05a_runtime_identity_regression.sh
  scripts/hpc/submit_r05a_runtime_identity_regression.sh
  slurm/r05a_runtime_identity_regression_cpu.sbatch
  tests/test_r05a_runtime_identity_regression.py
)
for path in "${BOUND_PATHS[@]}"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked runtime-identity source: $path" >&2; exit 2; }
done
git ls-files --error-unmatch "${BOUND_PATHS[@]}" >/dev/null || { echo "runtime-identity sources must be committed" >&2; exit 2; }

REGISTERED_RUN_ID=$(jq -er '.execution_release.run_id' "$CONFIG_LOCAL")
ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$CONFIG_LOCAL")
case "$ACCEPTED_IMPLEMENTATION_COMMIT" in *[!0-9a-f]*|'') echo "invalid accepted implementation commit" >&2; exit 2 ;; esac
test "${#ACCEPTED_IMPLEMENTATION_COMMIT}" = 40 || { echo "invalid accepted implementation commit length" >&2; exit 2; }
test "$RUN_ID" = "$REGISTERED_RUN_ID" || { echo "caller run ID differs from exact runtime-identity release" >&2; exit 2; }
jq -e --arg run "$RUN_ID" --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" '
  .ready_to_run == true and .blocked_on == []
  and .accepted_runtime_repair_commit == "5a5f3306b96f1491a4baa52657193afc1bfad2b5"
  and .resource_contract == {partition:"main",account:"normal",qos:"normal",source_host:"worker-1",cpus_per_task:1,host_memory_mib:256,time_limit:"00:02:00",gpus:0,requeue:false}
  and .runtime_identity_contract.validation_helper == "scripts/hpc/lib/r05a_runtime_identity.sh"
  and .runtime_identity_contract.validation_is_shell_only == true
  and .runtime_identity_contract.interpreter_invocation_during_validation_allowed == false
  and .claim_boundary == {shell_only:true,python_executed:false,model_or_checkpoint_loaded:false,simulator_executed:false,scientific_claim_allowed:false,transport_hypothesis_conclusion_allowed:false,h100_submission_authorized:false,automatic_resubmission_allowed:false,automatic_cancellation_allowed:false,automatic_next_gate_allowed:false}
  and ((.execution_release | keys | sort) == (["schema_version","artifact_role","decision_artifact","accepted_implementation_commit","run_id","single_submission","release_only_parent_required","resources","allowed_release_diff_paths","automatic_cancellation_allowed","automatic_resubmission_allowed","h100_submission_authorized","automatic_next_gate_allowed"] | sort))
  and .execution_release.schema_version == "1.0"
  and .execution_release.artifact_role == "r05a_runtime_identity_regression_execution_release"
  and .execution_release.decision_artifact == "docs/decisions/0044-release-runtime-identity-regression.md"
  and .execution_release.accepted_implementation_commit == $implementation
  and .execution_release.run_id == $run
  and .execution_release.single_submission == true
  and .execution_release.release_only_parent_required == true
  and .execution_release.resources == {partition:"main",account:"normal",qos:"normal",source_host:"worker-1",cpus_per_task:1,host_memory_mib:256,time_limit:"00:02:00",gpus:0,requeue:false}
  and .execution_release.allowed_release_diff_paths == ["configs/experiments/r05a_runtime_identity_regression.json","docs/decisions/0044-release-runtime-identity-regression.md"]
  and .execution_release.automatic_cancellation_allowed == false
  and .execution_release.automatic_resubmission_allowed == false
  and .execution_release.h100_submission_authorized == false
  and .execution_release.automatic_next_gate_allowed == false' "$CONFIG_LOCAL" >/dev/null || {
    echo "runtime-identity execution release changed" >&2
    exit 2
  }

test -z "$(git status --porcelain)" || { echo "runtime-identity submission requires a clean worktree" >&2; exit 2; }
test "$(git rev-parse HEAD)" = "$EXPECTED_RELEASE_COMMIT" || { echo "local HEAD differs from authorized runtime-identity release" >&2; exit 2; }
test "$(git rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$EXPECTED_RELEASE_COMMIT" || { echo "local origin differs from authorized runtime-identity release" >&2; exit 2; }
test "$(git rev-list --parents -n 1 "$EXPECTED_RELEASE_COMMIT")" = "$EXPECTED_RELEASE_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || {
  echo "runtime-identity release is not a direct implementation child" >&2
  exit 2
}
PARENT_CONFIG=$(git show "$ACCEPTED_IMPLEMENTATION_COMMIT:$CONFIG_LOCAL")
jq -e '.ready_to_run == false and (.blocked_on | type == "array" and length > 0) and .execution_release == null' <<<"$PARENT_CONFIG" >/dev/null || {
  echo "runtime-identity implementation config was not fail closed" >&2
  exit 2
}
test "$(jq -cS 'del(.ready_to_run,.blocked_on,.execution_release)' <<<"$PARENT_CONFIG")" = "$(jq -cS 'del(.ready_to_run,.blocked_on,.execution_release)' "$CONFIG_LOCAL")" || {
  echo "runtime-identity release changed non-release config content" >&2
  exit 2
}
RELEASE_DIFF=$(git diff --name-only "$ACCEPTED_IMPLEMENTATION_COMMIT" "$EXPECTED_RELEASE_COMMIT")
test "$RELEASE_DIFF" = "$(printf '%s\n' "$CONFIG_LOCAL" "$ADR0044_LOCAL")" || { echo "runtime-identity release-only path set changed" >&2; exit 2; }
grep -F "Accepted implementation commit: \`$ACCEPTED_IMPLEMENTATION_COMMIT\`." "$ADR0044_LOCAL" >/dev/null || { echo "release decision does not bind implementation commit" >&2; exit 2; }
grep -F "Immutable run ID: \`$RUN_ID\`." "$ADR0044_LOCAL" >/dev/null || { echo "release decision does not bind run ID" >&2; exit 2; }
grep -F 'H100 submission authorized: `false`.' "$ADR0044_LOCAL" >/dev/null || { echo "release decision does not prohibit H100 submission" >&2; exit 2; }

# Login use is control plane only.  This preflight invokes no Python or experiment.
scripts/hpc/preflight.sh
ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$EXPECTED_RELEASE_COMMIT" "$ACCEPTED_IMPLEMENTATION_COMMIT" \
  "$EXPERIMENT_ROOT" "$SLURM_LOG_ROOT" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
expected_commit=$3
accepted_implementation_commit=$4
experiment_root=$5
slurm_log_root=$6
run_root=$experiment_root/$run_id
config=$remote_repo/configs/experiments/r05a_runtime_identity_regression.json
adr0043=$remote_repo/docs/decisions/0043-preregister-runtime-identity-regression.md
adr0044=$remote_repo/docs/decisions/0044-release-runtime-identity-regression.md
helper=$remote_repo/scripts/hpc/lib/r05a_runtime_identity.sh
runner=$remote_repo/scripts/hpc/run_r05a_runtime_identity_regression.sh
submitter=$remote_repo/scripts/hpc/submit_r05a_runtime_identity_regression.sh
slurm_file=$remote_repo/slurm/r05a_runtime_identity_regression_cpu.sbatch
test_file=$remote_repo/tests/test_r05a_runtime_identity_regression.py
bound_paths=(
  configs/experiments/r05a_runtime_identity_regression.json
  docs/decisions/0043-preregister-runtime-identity-regression.md
  docs/decisions/0044-release-runtime-identity-regression.md
  scripts/hpc/lib/r05a_runtime_identity.sh
  scripts/hpc/run_r05a_runtime_identity_regression.sh
  scripts/hpc/submit_r05a_runtime_identity_regression.sh
  slurm/r05a_runtime_identity_regression_cpu.sbatch
  tests/test_r05a_runtime_identity_regression.py
)
for relative in "${bound_paths[@]}"; do
  test -f "$remote_repo/$relative" && test ! -L "$remote_repo/$relative" || { echo "missing remote runtime-identity source: $relative" >&2; exit 2; }
done
command -v jq >/dev/null 2>&1 || { echo "remote jq is required" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || { echo "remote runtime-identity commit differs" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$expected_commit" || { echo "remote runtime-identity origin differs" >&2; exit 2; }
test -z "$(git -C "$remote_repo" status --porcelain)" || { echo "remote runtime-identity tree is dirty" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-list --parents -n 1 "$expected_commit")" = "$expected_commit $accepted_implementation_commit" || { echo "remote release parent changed" >&2; exit 2; }
test "$(jq -er '.execution_release.run_id' "$config")" = "$run_id" || { echo "remote released run ID changed" >&2; exit 2; }
test "$(jq -er '.execution_release.accepted_implementation_commit' "$config")" = "$accepted_implementation_commit" || { echo "remote accepted implementation changed" >&2; exit 2; }
jq -e '.ready_to_run == true and .blocked_on == [] and .claim_boundary.shell_only == true and .claim_boundary.h100_submission_authorized == false and .execution_release.single_submission == true and .execution_release.release_only_parent_required == true and .execution_release.resources == {partition:"main",account:"normal",qos:"normal",source_host:"worker-1",cpus_per_task:1,host_memory_mib:256,time_limit:"00:02:00",gpus:0,requeue:false} and .execution_release.automatic_cancellation_allowed == false and .execution_release.automatic_resubmission_allowed == false and .execution_release.h100_submission_authorized == false and .execution_release.automatic_next_gate_allowed == false' "$config" >/dev/null || { echo "remote release contract changed" >&2; exit 2; }
test ! -e "$run_root" || { echo "immutable runtime-identity run ID is already used" >&2; exit 2; }
test -z "$(squeue -h -u "$(whoami)" -o '%i')" || { echo "runtime-identity regression requires an empty user queue" >&2; exit 2; }

node_record=$(scontrol show node worker-1 -o)
node_state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$node_state" in *down*|*drain*|*fail*|*maint*|*not_resp*|*unknown*|*power*) echo "worker-1 is unhealthy: $node_state" >&2; exit 2 ;; esac
free_mem=$(printf '%s\n' "$node_record" | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
case "$free_mem" in *[!0-9]*|'') echo "cannot parse worker-1 FreeMem" >&2; exit 2 ;; esac
test "$free_mem" -ge 256 || { echo "worker-1 has less than 256 MiB free" >&2; exit 2; }

mkdir -p "$slurm_log_root"
mkdir "$run_root"
held_job_id=
release_attempted=false
release_confirmed=false
control_stage=held_submission
control_failure() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ ! -e "$run_root/submission-failure.json" ]; then
    temporary=$(mktemp "$run_root/.submission-failure.XXXXXX") || return "$status"
    jq -n \
      --arg run "$run_id" --arg commit "$expected_commit" --arg implementation "$accepted_implementation_commit" \
      --arg stage "$control_stage" --arg exit_code "$status" --arg job "$held_job_id" \
      --arg release_attempted "$release_attempted" --arg release_confirmed "$release_confirmed" \
      --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
      {
        schema_version:"1.0",
        artifact_role:"r05a_runtime_identity_regression_submission_failure",
        status:"failed_closed",
        run_id:$run,
        git_commit:$commit,
        accepted_implementation_commit:$implementation,
        control_stage:$stage,
        exit_code:($exit_code|tonumber),
        slurm_job_id:(if $job == "" then null else $job end),
        release_attempted:($release_attempted == "true"),
        release_confirmed:($release_confirmed == "true"),
        exact_job_must_be_inspected_before_cleanup:true,
        shell_only:true,
        automatic_cancellation_allowed:false,
        automatic_resubmission_allowed:false,
        h100_submission_authorized:false,
        scientific_claim_allowed:false,
        timestamp_utc:$now
      }' >"$temporary" && mv "$temporary" "$run_root/submission-failure.json"
  fi
  if [ -n "$held_job_id" ] && [ "$release_confirmed" = false ]; then
    echo "runtime-identity job $held_job_id requires exact inspection; no automatic cancellation or resubmission" >&2
  fi
  return "$status"
}
trap control_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

held=$run_root/held-source-submission.json
source_contract=$run_root/source-contract.json
submission_receipt=$run_root/submission.json
result=$run_root/results.json
failure=$run_root/failure.json
job_submission=$(RUN_ID="$run_id" EXPECTED_GIT_COMMIT="$expected_commit" \
  RUNTIME_IDENTITY_CONFIG="$config" SOURCE_CONTRACT="$source_contract" \
  SUBMISSION="$submission_receipt" RESULT="$result" FAILURE="$failure" \
  REMOTE_REPO="$remote_repo" \
  sbatch --parsable --hold --partition=main --account=normal --qos=normal \
    --nodelist=worker-1 --cpus-per-task=1 --mem=256M --time=00:02:00 --no-requeue \
    --output="$slurm_log_root/%x-%j.out" "$slurm_file")
held_job_id=${job_submission%%;*}
case "$held_job_id" in *[!0-9]*|'') echo "invalid runtime-identity job ID: $job_submission" >&2; exit 2 ;; esac

resources=$(jq -n '{partition:"main",account:"normal",qos:"normal",source_host:"worker-1",cpus_per_task:1,host_memory_mib:256,time_limit:"00:02:00",gpus:0,requeue:false}')
control_stage=held_receipt
temporary=$(mktemp "$run_root/.held-source-submission.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg job "$held_job_id" --argjson resources "$resources" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
  {schema_version:"1.0",artifact_role:"r05a_runtime_identity_regression_held_submission",status:"sbatch_returned_held_job",run_id:$run,git_commit:$commit,slurm_job_id:$job,source_node:"worker-1",resources:$resources,released_at_receipt_time:false,shell_only:true,gpu_allocated:false,scientific_claim_allowed:false,h100_submission_authorized:false,timestamp_utc:$now}' >"$temporary"
mv "$temporary" "$held"
held_sha=$(sha256sum "$held" | awk '{print $1}')

validate_held_job() {
  local record lower req_tres req_cpu req_mem
  record=$(scontrol show job "$held_job_id" -o)
  for field in "JobState=PENDING" "Reason=JobHeldUser" "ReqNodeList=worker-1" "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=00:02:00" "Requeue=0"; do
    case " $record " in *" $field "*) ;; *) echo "held runtime-identity field changed: $field" >&2; return 2 ;; esac
  done
  lower=$(printf '%s\n' "$record" | tr '[:upper:]' '[:lower:]')
  case "$lower" in *gpu*) echo "held runtime-identity job requests a GPU" >&2; return 2 ;; esac
  req_tres=$(printf '%s\n' "$record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
  req_cpu=$(printf '%s\n' "$req_tres" | tr ',' '\n' | sed -n 's/^cpu=//p')
  req_mem=$(printf '%s\n' "$req_tres" | tr ',' '\n' | sed -n 's/^mem=//p')
  test "$req_cpu" = 1 || { echo "held runtime-identity CPU request changed: $req_cpu" >&2; return 2; }
  case "$req_mem" in 256M) ;; *) echo "held runtime-identity memory request changed: $req_mem" >&2; return 2 ;; esac
}
control_stage=held_job_contract
validate_held_job

bound_hashes='{}'
for relative in "${bound_paths[@]}"; do
  digest=$(sha256sum "$remote_repo/$relative" | awk '{print $1}')
  bound_hashes=$(jq -c --arg key "$relative" --arg value "$digest" '. + {($key):$value}' <<<"$bound_hashes")
done
runtime_identity=$(jq -c '.runtime_identity_contract | {openpi_python,libero_python}' "$config")
artifact_paths=$(jq -n --arg held "$held" --arg source "$source_contract" --arg submission "$submission_receipt" --arg result "$result" --arg failure "$failure" '{held_submission:$held,source_contract:$source,submission:$submission,result:$result,failure:$failure}')
control_stage=source_contract
temporary=$(mktemp "$run_root/.source-contract.XXXXXX")
jq -n \
  --arg run "$run_id" --arg commit "$expected_commit" --arg implementation "$accepted_implementation_commit" \
  --arg job "$held_job_id" --arg held "$held" --arg held_sha "$held_sha" \
  --argjson resources "$resources" --argjson paths "$artifact_paths" \
  --argjson runtime "$runtime_identity" --argjson hashes "$bound_hashes" \
  --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
  {schema_version:"1.0",artifact_role:"r05a_runtime_identity_regression_source_contract",status:"held_job_bound_before_release",run_id:$run,git_commit:$commit,git_dirty:false,accepted_implementation_commit:$implementation,slurm_job_id:$job,source_node:"worker-1",resources:$resources,artifact_paths:$paths,held_submission_path:$held,held_submission_sha256:$held_sha,runtime_identity:$runtime,bound_file_sha256:$hashes,shell_only:true,scientific_claim_allowed:false,h100_submission_authorized:false,timestamp_utc:$now}' >"$temporary"
mv "$temporary" "$source_contract"
source_sha=$(sha256sum "$source_contract" | awk '{print $1}')

control_stage=atomic_submission_token
temporary=$(mktemp "$run_root/.submission.XXXXXX")
jq -n \
  --arg run "$run_id" --arg commit "$expected_commit" --arg implementation "$accepted_implementation_commit" \
  --arg job "$held_job_id" --arg held "$held" --arg held_sha "$held_sha" \
  --arg source "$source_contract" --arg source_sha "$source_sha" \
  --arg result "$result" --arg failure "$failure" --argjson resources "$resources" \
  --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
  {schema_version:"1.0",artifact_role:"r05a_runtime_identity_regression_atomic_submission",status:"held_job_receipted_before_release",run_id:$run,git_commit:$commit,accepted_implementation_commit:$implementation,slurm_job_id:$job,source_node:"worker-1",resources:$resources,held_submission_path:$held,held_submission_sha256:$held_sha,source_contract_path:$source,source_contract_sha256:$source_sha,expected_result:$result,expected_failure:$failure,released_at_receipt_time:false,shell_only:true,gpu_allocated:false,scientific_claim_allowed:false,h100_submission_authorized:false,automatic_cancellation_allowed:false,automatic_resubmission_allowed:false,timestamp_utc:$now}' >"$temporary"
mv "$temporary" "$submission_receipt"
submission_sha=$(sha256sum "$submission_receipt" | awk '{print $1}')

control_stage=final_pre_release_validation
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$remote_repo" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$expected_commit"
test -z "$(git -C "$remote_repo" status --porcelain)"
validate_held_job
test "$(sha256sum "$held" | awk '{print $1}')" = "$held_sha"
test "$(sha256sum "$source_contract" | awk '{print $1}')" = "$source_sha"
test "$(sha256sum "$submission_receipt" | awk '{print $1}')" = "$submission_sha"
for relative in "${bound_paths[@]}"; do
  test "$(sha256sum "$remote_repo/$relative" | awk '{print $1}')" = "$(jq -er --arg path "$relative" '.bound_file_sha256[$path]' "$source_contract")" || { echo "final bound source changed: $relative" >&2; exit 2; }
done
jq -e --arg run "$run_id" --arg commit "$expected_commit" --arg job "$held_job_id" --arg held_sha "$held_sha" --arg source_sha "$source_sha" '.run_id == $run and .git_commit == $commit and .slurm_job_id == $job and .held_submission_sha256 == $held_sha and .source_contract_sha256 == $source_sha and .released_at_receipt_time == false and .gpu_allocated == false and .h100_submission_authorized == false' "$submission_receipt" >/dev/null

control_stage=release_requested_state_unknown
release_attempted=true
scontrol release "$held_job_id"
release_confirmed=true
control_stage=complete
echo "runtime_identity_job_id=$held_job_id"
echo "source_contract_sha256=$source_sha"
echo "submission_receipt=$submission_receipt"
echo "submission_receipt_sha256=$submission_sha"
REMOTE
