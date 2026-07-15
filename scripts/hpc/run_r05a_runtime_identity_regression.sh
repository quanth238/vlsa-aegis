#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?runtime-identity regression must run inside Slurm}"
: "${SLURM_JOB_PARTITION:?partition provenance is required}"
: "${RUN_ID:?immutable runtime-identity run id is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed release commit is required}"
: "${RUNTIME_IDENTITY_CONFIG:?released runtime-identity config is required}"
: "${SOURCE_CONTRACT:?source contract is required}"
: "${SUBMISSION:?atomic submission receipt is required}"
: "${RESULT:?immutable result path is required}"
: "${FAILURE:?immutable failure path is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

RUNTIME_IDENTITY_HELPER=$REMOTE_REPO/scripts/hpc/lib/r05a_runtime_identity.sh
RUN_ROOT=/mnt/data/quanth/experiments/crfs-oracle/$RUN_ID
FAILURE_STAGE=allocation_contract

write_failure() {
  local status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ ! -e "$FAILURE" ] && [ ! -e "$RESULT" ]; then
    local temporary
    temporary=$(mktemp "${FAILURE%/*}/.runtime-identity-failure.XXXXXX") || return "$status"
    jq -n \
      --arg run_id "$RUN_ID" \
      --arg commit "$EXPECTED_GIT_COMMIT" \
      --arg job_id "$SLURM_JOB_ID" \
      --arg host "$(hostname -s)" \
      --arg stage "$FAILURE_STAGE" \
      --arg exit_code "$status" \
      '{
        schema_version:"1.0",
        artifact_role:"r05a_runtime_identity_regression_failure",
        status:"failed_closed",
        run_id:$run_id,
        git_commit:$commit,
        slurm_job_id:$job_id,
        host:$host,
        failure_stage:$stage,
        exit_code:($exit_code|tonumber),
        shell_only:true,
        openpi_python_executed:false,
        libero_python_executed:false,
        model_or_checkpoint_loaded:false,
        simulator_executed:false,
        scientific_claim_allowed:false,
        h100_submission_authorized:false,
        automatic_resubmission_allowed:false
      }' >"$temporary" && mv "$temporary" "$FAILURE"
  fi
  return "$status"
}
trap write_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

case "$SLURM_JOB_ID" in *[!0-9]*|'') echo "invalid Slurm job id" >&2; exit 2 ;; esac
case "$EXPECTED_GIT_COMMIT" in *[!0-9a-f]*|'') echo "invalid release commit" >&2; exit 2 ;; esac
test "${#EXPECTED_GIT_COMMIT}" = 40 || { echo "invalid release commit length" >&2; exit 2; }
test "$SLURM_JOB_PARTITION" = main || { echo "runtime-identity regression is frozen to main" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "runtime-identity regression must run on worker-1" >&2; exit 2; }
test -z "${SLURM_ARRAY_JOB_ID:-}" && test -z "${SLURM_ARRAY_TASK_ID:-}" || { echo "runtime-identity regression must be a non-array job" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 1 || { echo "runtime-identity regression requires one CPU" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 256 || { echo "runtime-identity regression requires exactly 256 MiB" >&2; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-NoDevFiles}" in ''|NoDevFiles) ;; *) echo "runtime-identity regression received a GPU" >&2; exit 2 ;; esac
test -z "${SLURM_JOB_GPUS:-}" || { echo "runtime-identity regression received SLURM_JOB_GPUS" >&2; exit 2; }
JOB_RECORD=$(scontrol show job "$SLURM_JOB_ID" -o)
for field in "JobState=RUNNING" "ReqNodeList=worker-1" "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=00:02:00" "Requeue=0"; do
  case " $JOB_RECORD " in *" $field "*) ;; *) echo "running Slurm contract changed: $field" >&2; exit 2 ;; esac
done
case "$(printf '%s\n' "$JOB_RECORD" | tr '[:upper:]' '[:lower:]')" in *gpu*) echo "running Slurm contract contains a GPU" >&2; exit 2 ;; esac
REQ_TRES=$(printf '%s\n' "$JOB_RECORD" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test "$(printf '%s\n' "$REQ_TRES" | tr ',' '\n' | sed -n 's/^cpu=//p')" = 1 || { echo "running CPU request changed" >&2; exit 2; }
case "$(printf '%s\n' "$REQ_TRES" | tr ',' '\n' | sed -n 's/^mem=//p')" in 256M) ;; *) echo "running memory request changed" >&2; exit 2 ;; esac
test ! -e "$RESULT" && test ! -e "$FAILURE" || { echo "immutable runtime-identity output already exists" >&2; exit 2; }
test "$RUNTIME_IDENTITY_CONFIG" = "$REMOTE_REPO/configs/experiments/r05a_runtime_identity_regression.json" || { echo "runtime-identity config path changed" >&2; exit 2; }
test "$SOURCE_CONTRACT" = "$RUN_ROOT/source-contract.json" || { echo "source-contract path changed" >&2; exit 2; }
test "$SUBMISSION" = "$RUN_ROOT/submission.json" || { echo "submission path changed" >&2; exit 2; }
test "$RESULT" = "$RUN_ROOT/results.json" || { echo "result path changed" >&2; exit 2; }
test "$FAILURE" = "$RUN_ROOT/failure.json" || { echo "failure path changed" >&2; exit 2; }

for path in "$RUNTIME_IDENTITY_CONFIG" "$SOURCE_CONTRACT" "$SUBMISSION" "$RUNTIME_IDENTITY_HELPER"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked runtime-identity input: $path" >&2; exit 2; }
done
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || { echo "runtime-identity source commit changed" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$EXPECTED_GIT_COMMIT" || { echo "runtime-identity origin ref changed" >&2; exit 2; }
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || { echo "runtime-identity source tree is dirty" >&2; exit 2; }

FAILURE_STAGE=release_contract
REGISTERED_RUN_ID=$(jq -er '.execution_release.run_id' "$RUNTIME_IDENTITY_CONFIG")
ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$RUNTIME_IDENTITY_CONFIG")
test "$RUN_ID" = "$REGISTERED_RUN_ID" || { echo "runtime-identity run id differs from release" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-list --parents -n 1 "$EXPECTED_GIT_COMMIT")" = "$EXPECTED_GIT_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || {
  echo "runtime-identity release is not a direct implementation child" >&2
  exit 2
}
jq -e --arg run "$RUN_ID" --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" '
  .ready_to_run == true and .blocked_on == []
  and .accepted_runtime_repair_commit == "5a5f3306b96f1491a4baa52657193afc1bfad2b5"
  and .resource_contract == {partition:"main",account:"normal",qos:"normal",source_host:"worker-1",cpus_per_task:1,host_memory_mib:256,time_limit:"00:02:00",gpus:0,requeue:false}
  and .claim_boundary.shell_only == true
  and .claim_boundary.python_executed == false
  and .claim_boundary.h100_submission_authorized == false
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
  and .execution_release.automatic_next_gate_allowed == false' "$RUNTIME_IDENTITY_CONFIG" >/dev/null || {
    echo "runtime-identity execution release changed" >&2
    exit 2
  }

FAILURE_STAGE=submission_contract
SUBMISSION_SHA256=$(sha256sum "$SUBMISSION" | awk '{print $1}')
SOURCE_CONTRACT_SHA256=$(jq -er '.source_contract_sha256' "$SUBMISSION")
HELD_SUBMISSION=$RUN_ROOT/held-source-submission.json
HELD_SUBMISSION_SHA256=$(jq -er '.held_submission_sha256' "$SUBMISSION")
jq -e \
  --arg run "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" \
  --arg job "$SLURM_JOB_ID" \
  --arg source "$SOURCE_CONTRACT" \
  --arg held "$HELD_SUBMISSION" \
  --arg result "$RESULT" \
  --arg failure "$FAILURE" '
  .schema_version == "1.0"
  and .artifact_role == "r05a_runtime_identity_regression_atomic_submission"
  and .status == "held_job_receipted_before_release"
  and .run_id == $run and .git_commit == $commit
  and .accepted_implementation_commit == $implementation
  and .slurm_job_id == $job and .source_node == "worker-1"
  and .resources == {partition:"main",account:"normal",qos:"normal",source_host:"worker-1",cpus_per_task:1,host_memory_mib:256,time_limit:"00:02:00",gpus:0,requeue:false}
  and .source_contract_path == $source
  and (.source_contract_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and .held_submission_path == $held
  and (.held_submission_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and .expected_result == $result and .expected_failure == $failure
  and .released_at_receipt_time == false
  and .shell_only == true and .gpu_allocated == false
  and .scientific_claim_allowed == false
  and .h100_submission_authorized == false
  and .automatic_cancellation_allowed == false
  and .automatic_resubmission_allowed == false' "$SUBMISSION" >/dev/null || {
    echo "runtime-identity final submission token changed" >&2
    exit 2
  }
test "$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')" = "$SOURCE_CONTRACT_SHA256" || { echo "source contract differs from final submission token" >&2; exit 2; }
test "$(sha256sum "$HELD_SUBMISSION" | awk '{print $1}')" = "$HELD_SUBMISSION_SHA256" || { echo "held receipt differs from final submission token" >&2; exit 2; }

FAILURE_STAGE=source_contract
jq -e \
  --arg run "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" \
  --arg job "$SLURM_JOB_ID" \
  --arg held "${SOURCE_CONTRACT%/*}/held-source-submission.json" \
  --arg result "$RESULT" \
  --arg failure "$FAILURE" \
  --arg submission "$SUBMISSION" '
  .schema_version == "1.0"
  and .artifact_role == "r05a_runtime_identity_regression_source_contract"
  and .status == "held_job_bound_before_release"
  and .run_id == $run and .git_commit == $commit and .git_dirty == false
  and .accepted_implementation_commit == $implementation
  and .slurm_job_id == $job and .source_node == "worker-1"
  and .resources == {partition:"main",account:"normal",qos:"normal",source_host:"worker-1",cpus_per_task:1,host_memory_mib:256,time_limit:"00:02:00",gpus:0,requeue:false}
  and .artifact_paths.result == $result
  and .artifact_paths.failure == $failure
  and .artifact_paths.submission == $submission
  and .held_submission_path == $held
  and (.held_submission_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and .runtime_identity == {
    openpi_python:{public_path:"/mnt/data/quanth/venvs/openpi/bin/python",direct_link_target:"/mnt/data/quanth/anaconda3/bin/python",resolved_executable:"/mnt/data/quanth/anaconda3/bin/python3.11",resolved_sha256:"c71718900fe84a9124d39abdd9d68d029930e0dcff1764686d8d6aad97216bc9"},
    libero_python:{public_path:"/mnt/data/quanth/venvs/openpi-libero-client/bin/python",direct_link_target:"/home/quanth/.local/share/uv/python/cpython-3.8-linux-x86_64-gnu/bin/python3.8",resolved_executable:"/home/quanth/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3.8",resolved_sha256:"c70efda0ee43d9a0014ee570cad3abb4f46b0c11f6ea88f7c467a91faafd4f62"}
  }
  and .shell_only == true and .scientific_claim_allowed == false
  and .h100_submission_authorized == false' "$SOURCE_CONTRACT" >/dev/null || {
    echo "runtime-identity source contract changed" >&2
    exit 2
  }

EXPECTED_BOUND_KEYS='["configs/experiments/r05a_runtime_identity_regression.json","docs/decisions/0043-preregister-runtime-identity-regression.md","docs/decisions/0044-release-runtime-identity-regression.md","scripts/hpc/lib/r05a_runtime_identity.sh","scripts/hpc/run_r05a_runtime_identity_regression.sh","scripts/hpc/submit_r05a_runtime_identity_regression.sh","slurm/r05a_runtime_identity_regression_cpu.sbatch","tests/test_r05a_runtime_identity_regression.py"]'
test "$(jq -cS '.bound_file_sha256 | keys' "$SOURCE_CONTRACT")" = "$(jq -cS . <<<"$EXPECTED_BOUND_KEYS")" || {
  echo "bound runtime-identity source set changed" >&2
  exit 2
}
while IFS=$'\t' read -r relative expected_sha; do
  case "$relative" in /*|*..*) echo "unsafe bound runtime-identity path: $relative" >&2; exit 2 ;; esac
  path=$REMOTE_REPO/$relative
  test -f "$path" && test ! -L "$path" || { echo "missing bound runtime-identity source: $relative" >&2; exit 2; }
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected_sha" || { echo "bound runtime-identity source changed: $relative" >&2; exit 2; }
done < <(jq -er '.bound_file_sha256 | to_entries[] | [.key,.value] | @tsv' "$SOURCE_CONTRACT")

test "$(jq -er '.held_submission_path' "$SOURCE_CONTRACT")" = "$HELD_SUBMISSION" || { echo "source contract held path changed" >&2; exit 2; }
test -f "$HELD_SUBMISSION" && test ! -L "$HELD_SUBMISSION" || { echo "held submission receipt is missing or symlinked" >&2; exit 2; }
test "$(sha256sum "$HELD_SUBMISSION" | awk '{print $1}')" = "$(jq -er '.held_submission_sha256' "$SOURCE_CONTRACT")" || {
  echo "held submission receipt changed" >&2
  exit 2
}
jq -e \
  --arg run "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg job "$SLURM_JOB_ID" '
  .schema_version == "1.0"
  and .artifact_role == "r05a_runtime_identity_regression_held_submission"
  and .status == "sbatch_returned_held_job"
  and .run_id == $run and .git_commit == $commit and .slurm_job_id == $job
  and .source_node == "worker-1"
  and .resources == {partition:"main",account:"normal",qos:"normal",source_host:"worker-1",cpus_per_task:1,host_memory_mib:256,time_limit:"00:02:00",gpus:0,requeue:false}
  and .released_at_receipt_time == false
  and .shell_only == true and .gpu_allocated == false
  and .scientific_claim_allowed == false
  and .h100_submission_authorized == false' "$HELD_SUBMISSION" >/dev/null || {
    echo "held submission receipt semantics changed" >&2
    exit 2
  }

jq -e \
  --arg run "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" \
  --arg job "$SLURM_JOB_ID" \
  --arg source "$SOURCE_CONTRACT" \
  --arg source_sha "$SOURCE_CONTRACT_SHA256" \
  --arg held "$HELD_SUBMISSION" \
  --arg held_sha "$HELD_SUBMISSION_SHA256" \
  --arg result "$RESULT" \
  --arg failure "$FAILURE" '
  .schema_version == "1.0"
  and .artifact_role == "r05a_runtime_identity_regression_atomic_submission"
  and .status == "held_job_receipted_before_release"
  and .run_id == $run and .git_commit == $commit and .slurm_job_id == $job
  and .accepted_implementation_commit == $implementation
  and .source_node == "worker-1"
  and .resources == {partition:"main",account:"normal",qos:"normal",source_host:"worker-1",cpus_per_task:1,host_memory_mib:256,time_limit:"00:02:00",gpus:0,requeue:false}
  and .source_contract_path == $source
  and .source_contract_sha256 == $source_sha
  and .held_submission_path == $held
  and .held_submission_sha256 == $held_sha
  and .expected_result == $result
  and .expected_failure == $failure
  and .released_at_receipt_time == false
  and .shell_only == true and .gpu_allocated == false
  and .scientific_claim_allowed == false
  and .h100_submission_authorized == false' "$SUBMISSION" >/dev/null || {
    echo "runtime-identity submission receipt changed" >&2
    exit 2
  }

FAILURE_STAGE=runtime_identity
. "$RUNTIME_IDENTITY_HELPER"
crfs_validate_r05a_openpi_python "$CRFS_R05A_OPENPI_PYTHON"
crfs_validate_r05a_libero_python "$CRFS_R05A_LIBERO_PYTHON"

OPENPI_DIRECT=$(readlink -- "$CRFS_R05A_OPENPI_PYTHON")
OPENPI_RESOLVED=$(readlink -f -- "$CRFS_R05A_OPENPI_PYTHON")
OPENPI_DIGEST_RECORD=$(sha256sum -- "$OPENPI_RESOLVED")
OPENPI_SHA256=${OPENPI_DIGEST_RECORD%% *}
LIBERO_DIRECT=$(readlink -- "$CRFS_R05A_LIBERO_PYTHON")
LIBERO_RESOLVED=$(readlink -f -- "$CRFS_R05A_LIBERO_PYTHON")
LIBERO_DIGEST_RECORD=$(sha256sum -- "$LIBERO_RESOLVED")
LIBERO_SHA256=${LIBERO_DIGEST_RECORD%% *}
test "$CRFS_R05A_OPENPI_PYTHON" = /mnt/data/quanth/venvs/openpi/bin/python
test "$OPENPI_DIRECT" = /mnt/data/quanth/anaconda3/bin/python
test "$OPENPI_RESOLVED" = /mnt/data/quanth/anaconda3/bin/python3.11
test "$OPENPI_SHA256" = c71718900fe84a9124d39abdd9d68d029930e0dcff1764686d8d6aad97216bc9
test "$CRFS_R05A_LIBERO_PYTHON" = /mnt/data/quanth/venvs/openpi-libero-client/bin/python
test "$LIBERO_DIRECT" = /home/quanth/.local/share/uv/python/cpython-3.8-linux-x86_64-gnu/bin/python3.8
test "$LIBERO_RESOLVED" = /home/quanth/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3.8
test "$LIBERO_SHA256" = c70efda0ee43d9a0014ee570cad3abb4f46b0c11f6ea88f7c467a91faafd4f62
crfs_validate_r05a_openpi_python "$CRFS_R05A_OPENPI_PYTHON"
crfs_validate_r05a_libero_python "$CRFS_R05A_LIBERO_PYTHON"

FAILURE_STAGE=result_publication
temporary=$(mktemp "${RESULT%/*}/.runtime-identity-results.XXXXXX")
jq -n \
  --arg run "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg job "$SLURM_JOB_ID" \
  --arg host "$(hostname -s)" \
  --arg source "$SOURCE_CONTRACT" \
  --arg source_sha "$SOURCE_CONTRACT_SHA256" \
  --arg held "$HELD_SUBMISSION" \
  --arg held_sha "$HELD_SUBMISSION_SHA256" \
  --arg submission "$SUBMISSION" \
  --arg submission_sha "$SUBMISSION_SHA256" \
  --arg openpi_public "$CRFS_R05A_OPENPI_PYTHON" \
  --arg openpi_direct "$OPENPI_DIRECT" \
  --arg openpi_resolved "$OPENPI_RESOLVED" \
  --arg openpi_sha "$OPENPI_SHA256" \
  --arg libero_public "$CRFS_R05A_LIBERO_PYTHON" \
  --arg libero_direct "$LIBERO_DIRECT" \
  --arg libero_resolved "$LIBERO_RESOLVED" \
  --arg libero_sha "$LIBERO_SHA256" \
  --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
  {
    schema_version:"1.0",
    artifact_role:"r05a_runtime_identity_regression_result",
    status:"passed",
    run_id:$run,
    git_commit:$commit,
    git_dirty:false,
    slurm_job_id:$job,
    source_node:$host,
    resources:{partition:"main",account:"normal",qos:"normal",source_host:"worker-1",cpus_per_task:1,host_memory_mib:256,time_limit:"00:02:00",gpus:0,requeue:false},
    source_contract_path:$source,
    source_contract_sha256:$source_sha,
    held_submission_path:$held,
    held_submission_sha256:$held_sha,
    submission_path:$submission,
    submission_sha256:$submission_sha,
    observed_runtime_identity:{
      openpi_python:{public_path:$openpi_public,direct_link_target:$openpi_direct,resolved_executable:$openpi_resolved,resolved_sha256:$openpi_sha,public_is_symlink:true,resolved_is_regular_non_symlink_executable:true,passed:true},
      libero_python:{public_path:$libero_public,direct_link_target:$libero_direct,resolved_executable:$libero_resolved,resolved_sha256:$libero_sha,public_is_symlink:true,resolved_is_regular_non_symlink_executable:true,passed:true}
    },
    shell_only:true,
    openpi_python_executed:false,
    libero_python_executed:false,
    checkpoint_loaded:false,
    model_inference_executed:false,
    simulator_executed:false,
    rendering_executed:false,
    metrics_executed:false,
    training_executed:false,
    scientific_claim_allowed:false,
    transport_hypothesis_conclusion_allowed:false,
    h100_submission_authorized:false,
    automatic_resubmission_allowed:false,
    automatic_next_gate_allowed:false,
    timestamp_utc:$timestamp
  }' >"$temporary"
mv "$temporary" "$RESULT"
test -f "$RESULT" && test ! -L "$RESULT" || { echo "runtime-identity result was not atomically published" >&2; exit 3; }
FAILURE_STAGE=complete
echo "runtime_identity_result=$RESULT"
echo "runtime_identity_result_sha256=$(sha256sum "$RESULT" | awk '{print $1}')"
