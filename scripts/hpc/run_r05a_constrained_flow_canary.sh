#!/usr/bin/env bash
set -euo pipefail

# Exact-release H100 wrapper.  It validates the held transaction before any
# dependency setup or model process, then enters the isolated CFS workload.
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${EXPECTED_GIT_COMMIT:?reviewed release commit is required}"
: "${RUN_ID:?immutable CFS-00A run id is required}"
: "${SLURM_ARRAY_JOB_ID:?source array job id is required}"
: "${SLURM_ARRAY_TASK_ID:?source array task id is required}"

case "$SLURM_ARRAY_JOB_ID" in *[!0-9]*|'') echo "invalid source array job id" >&2; exit 2 ;; esac
test "$SLURM_ARRAY_TASK_ID" = 0 || { echo "CFS-00A is fixed to task zero" >&2; exit 2; }

WORKLOAD=$REMOTE_REPO/scripts/hpc/run_r05a_constrained_flow_workload.sh
RUN_ROOT=/mnt/data/quanth/experiments/crfs-oracle/$RUN_ID
CASE_DIR=$RUN_ROOT/crfs-1069f29a8d76463a
SOURCE_CONTRACT=$RUN_ROOT/source-contract.json
SUBMISSION=$RUN_ROOT/submission.json
HELD_GPU_SUBMISSION=$RUN_ROOT/held-gpu-submission.json
SCIENTIFIC_CONFIG=$REMOTE_REPO/configs/experiments/r05a_constrained_flow_canary.json
APPARATUS_CONFIG=$REMOTE_REPO/configs/experiments/r05a_constrained_flow_canary_apparatus.json

for path in "$WORKLOAD" "$SOURCE_CONTRACT" "$SUBMISSION" "$HELD_GPU_SUBMISSION" "$SCIENTIFIC_CONFIG" "$APPARATUS_CONFIG"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked CFS-00A release input: $path" >&2; exit 2; }
done
test -x "$WORKLOAD" || { echo "CFS-00A workload is not executable" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || { echo "CFS-00A source commit changed" >&2; exit 2; }
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || { echo "CFS-00A requires a clean remote tree" >&2; exit 2; }

REGISTERED_RUN_ID=$(jq -er '.execution_release.run_id' "$APPARATUS_CONFIG")
ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$APPARATUS_CONFIG")
test "$RUN_ID" = "$REGISTERED_RUN_ID" || { echo "run ID differs from exact execution release" >&2; exit 2; }
jq -e --arg run "$RUN_ID" --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" '
  .ready_to_run == true and .blocked_on == []
  and ((.execution_release | keys | sort) == (["schema_version","artifact_role","decision_artifact","accepted_implementation_commit","run_id","single_submission","source_host","resources","release_only_parent_required","allowed_release_diff_paths","automatic_resubmission_allowed","automatic_next_experiment_allowed"] | sort))
  and .execution_release.schema_version == "1.0"
  and .execution_release.artifact_role == "r05a_constrained_flow_canary_execution_release"
  and .execution_release.decision_artifact == "docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md"
  and .execution_release.accepted_implementation_commit == $implementation
  and .execution_release.run_id == $run
  and .execution_release.single_submission == true
  and .execution_release.source_host == "worker-1"
  and .execution_release.resources == {partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}
  and .execution_release.release_only_parent_required == true
  and .execution_release.allowed_release_diff_paths == ["configs/experiments/r05a_constrained_flow_canary.json","configs/experiments/r05a_constrained_flow_canary_apparatus.json","docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md"]
  and .execution_release.automatic_resubmission_allowed == false
  and .execution_release.automatic_next_experiment_allowed == false' "$APPARATUS_CONFIG" >/dev/null || {
    echo "CFS-00A execution release contract changed" >&2
    exit 2
  }
jq -e '
  .config_status == "released_exact_single_canary"
  and .ready_to_run == true and .blocked_on == []
  and .preregistration.h100_submission_authorized == true
  and (.execution_release | type == "object")' "$SCIENTIFIC_CONFIG" >/dev/null || { echo "scientific constrained-flow config is not exactly released" >&2; exit 2; }
test "$(jq -cS '.execution_release' "$SCIENTIFIC_CONFIG")" = "$(jq -cS '.execution_release' "$APPARATUS_CONFIG")" || { echo "scientific and apparatus execution releases differ" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$EXPECTED_GIT_COMMIT" || { echo "origin release ref changed" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-list --parents -n 1 "$EXPECTED_GIT_COMMIT")" = "$EXPECTED_GIT_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || { echo "release parent changed" >&2; exit 2; }

job_record=$(scontrol show job "$SLURM_ARRAY_JOB_ID" -o)
for field in "JobState=RUNNING" "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=02:00:00" "Requeue=0" "ReqNodeList=worker-1"; do
  case " $job_record " in *" $field "*) ;; *) echo "running H100 job field changed: $field" >&2; exit 2 ;; esac
done
req_tres=$(printf '%s\n' "$job_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test -n "$req_tres" || { echo "cannot parse running H100 ReqTRES" >&2; exit 2; }
req_cpus=$(printf '%s\n' "$req_tres" | tr ',' '\n' | sed -n 's/^cpu=//p')
req_mem=$(printf '%s\n' "$req_tres" | tr ',' '\n' | sed -n 's/^mem=//p')
req_gpus=$(printf '%s\n' "$req_tres" | tr ',' '\n' | sed -n 's#^gres/gpu=##p')
test "$req_cpus" = 8 || { echo "running H100 CPU request changed" >&2; exit 2; }
case "$req_mem" in 64G|65536M) ;; *) echo "running H100 memory request changed" >&2; exit 2 ;; esac
test "$req_gpus" = 1 || { echo "running H100 GPU request changed" >&2; exit 2; }

SOURCE_CONTRACT_SHA256=$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')
HELD_GPU_SUBMISSION_SHA256=$(sha256sum "$HELD_GPU_SUBMISSION" | awk '{print $1}')
jq -e --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" --arg gpu "$SLURM_ARRAY_JOB_ID" --arg held "$HELD_GPU_SUBMISSION" --arg held_sha "$HELD_GPU_SUBMISSION_SHA256" --arg run_root "$RUN_ROOT" --arg case_dir "$CASE_DIR" '
  .schema_version == "1.0"
  and .artifact_role == "r05a_constrained_flow_canary_source_contract"
  and .status == "gpu_held_sources_bound_before_cpu_submission"
  and .run_id == $run and .git_commit == $commit and .git_dirty == false
  and .source_node == "worker-1"
  and .gpu_slurm_array_job_id == $gpu and .gpu_slurm_array_task_id == 0
  and .exact_gpu_task_id == ($gpu + "_0")
  and .resources == {partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}
  and .artifact_paths == {run_root:$run_root,case_dir:$case_dir,legacy_payload:($case_dir+"/canary-payload.json"),constrained_flow_payload:($case_dir+"/constrained-flow-payload.json"),host_telemetry:($case_dir+"/host-cgroup-sampled-current.tsv"),gpu_samples:($case_dir+"/gpu-memory-samples.csv"),allocation_tests_log:($case_dir+"/allocation-focused-tests.log"),hidden_candidate:($case_dir+"/.results.candidate.json"),result:($case_dir+"/results.json"),validation_receipt:($run_root+"/cpu-afterany-validation.json")}
  and .held_gpu_submission_path == $held and .held_gpu_submission_sha256 == $held_sha
  and .scientific_claim_allowed == false and .infeasibility_claim_allowed == false
  and .simulator_efficacy_claim_allowed == false and .probe_training_authorized == false' "$SOURCE_CONTRACT" >/dev/null || {
    echo "source contract differs from exact held task" >&2
    exit 2
  }
jq -e --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" --arg gpu "$SLURM_ARRAY_JOB_ID" '
  .schema_version == "1.0"
  and .artifact_role == "r05a_constrained_flow_canary_held_gpu_submission"
  and .status == "sbatch_returned_held_gpu_id"
  and .run_id == $run and .git_commit == $commit
  and .gpu_slurm_array_job_id == $gpu and .gpu_slurm_array_task_id == 0
  and .exact_gpu_task_id == ($gpu + "_0") and .source_node == "worker-1"
  and .released_at_receipt_time == false
  and .scientific_claim_allowed == false and .infeasibility_claim_allowed == false
  and .simulator_efficacy_claim_allowed == false and .probe_training_authorized == false' "$HELD_GPU_SUBMISSION" >/dev/null || {
    echo "held-GPU receipt differs from exact task" >&2
    exit 2
  }
jq -e --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" --arg gpu "$SLURM_ARRAY_JOB_ID" --arg contract "$SOURCE_CONTRACT" --arg source_sha "$SOURCE_CONTRACT_SHA256" --arg result "$CASE_DIR/results.json" --arg receipt "$RUN_ROOT/cpu-afterany-validation.json" '
  .schema_version == "1.0"
  and .artifact_role == "r05a_constrained_flow_canary_atomic_submission"
  and .status == "cpu_afterany_registered_gpu_held"
  and .run_id == $run and .git_commit == $commit and .source_node == "worker-1"
  and .gpu_slurm_array_job_id == $gpu and .gpu_slurm_array_task_id == 0
  and (.cpu_afterany_job_id | type == "string" and test("^[1-9][0-9]*$"))
  and .dependency == ("afterany:" + $gpu)
  and .source_contract_path == $contract and .source_contract_sha256 == $source_sha
  and .expected_result == $result and .expected_validation_receipt == $receipt
  and .scientific_claim_allowed == false and .infeasibility_claim_allowed == false
  and .simulator_efficacy_claim_allowed == false and .probe_training_authorized == false' "$SUBMISSION" >/dev/null || {
    echo "atomic submission receipt differs from exact task" >&2
    exit 2
  }

export R05A_SOURCE_CONTRACT=$SOURCE_CONTRACT
exec "$WORKLOAD"
