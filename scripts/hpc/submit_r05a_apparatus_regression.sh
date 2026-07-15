#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?usage: RUN_ID=r05a-adr0031-apparatus-cpu-20260715a submit_r05a_apparatus_regression.sh}"
EXPECTED_RUN_ID=r05a-adr0031-apparatus-cpu-20260715a
test "$RUN_ID" = "$EXPECTED_RUN_ID" || { echo "R05A apparatus run id changed" >&2; exit 2; }

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
EXPERIMENT_ROOT=/mnt/data/quanth/experiments/crfs-oracle
SLURM_LOG_ROOT=/mnt/data/quanth/slurm_logs/crfs-oracle

for path in \
  main/crfs_oracle/r05a_allocation_tests.json \
  main/crfs_oracle/r05a_canary.py \
  main/validate_crfs_r05a_apparatus_regression.py \
  docs/decisions/0032-preregister-r05a-cpu-apparatus-regression.md \
  docs/decisions/0033-accept-r05a-artifact-singleton-scalar-decoding.md \
  docs/decisions/0034-recompute-r05a-status-from-exact-artifact-traces.md \
  scripts/hpc/lib/cgroup_memory.sh \
  scripts/hpc/lib/r05a_allocation_tests.sh \
  scripts/hpc/run_r05a_apparatus_regression.sh \
  scripts/hpc/submit_r05a_apparatus_regression.sh \
  slurm/r05a_apparatus_regression_cpu.sbatch \
  tests/test_cgroup_memory_resolver.py \
  tests/test_r05a_allocation_test_runner.py \
  tests/test_r05a_apparatus_regression.py \
  tests/test_r05a_apparatus_result_validator.py \
  tests/test_r05a_canary.py; do
  test -f "$path" || { echo "missing R05A apparatus source: $path" >&2; exit 2; }
done
git ls-files --error-unmatch \
  main/crfs_oracle/r05a_allocation_tests.json \
  main/crfs_oracle/r05a_canary.py \
  main/validate_crfs_r05a_apparatus_regression.py \
  docs/decisions/0032-preregister-r05a-cpu-apparatus-regression.md \
  docs/decisions/0033-accept-r05a-artifact-singleton-scalar-decoding.md \
  docs/decisions/0034-recompute-r05a-status-from-exact-artifact-traces.md \
  scripts/hpc/lib/cgroup_memory.sh \
  scripts/hpc/lib/r05a_allocation_tests.sh \
  scripts/hpc/run_r05a_apparatus_regression.sh \
  scripts/hpc/submit_r05a_apparatus_regression.sh \
  slurm/r05a_apparatus_regression_cpu.sbatch \
  tests/test_cgroup_memory_resolver.py \
  tests/test_r05a_allocation_test_runner.py \
  tests/test_r05a_apparatus_regression.py \
  tests/test_r05a_apparatus_result_validator.py \
  tests/test_r05a_canary.py >/dev/null || {
  echo "R05A apparatus sources must be committed before submission" >&2
  exit 2
}
test -z "$(git status --porcelain)" || {
  echo "R05A apparatus submission requires a clean reviewed worktree" >&2
  exit 2
}
EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)
UPSTREAM_COMMIT=$(git rev-parse '@{upstream}')
test "$EXPECTED_GIT_COMMIT" = "$UPSTREAM_COMMIT" || {
  echo "R05A apparatus source is not pushed to its upstream" >&2
  exit 2
}

# Shell-only control-plane preflight; no Python or experiment runs on login.
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
slurm_file=$remote_repo/slurm/r05a_apparatus_regression_cpu.sbatch
runner=$remote_repo/scripts/hpc/run_r05a_apparatus_regression.sh
registry=$remote_repo/main/crfs_oracle/r05a_allocation_tests.json
test_helper=$remote_repo/scripts/hpc/lib/r05a_allocation_tests.sh
cgroup_helper=$remote_repo/scripts/hpc/lib/cgroup_memory.sh
validator=$remote_repo/main/validate_crfs_r05a_apparatus_regression.py

for path in "$slurm_file" "$runner" "$registry" "$test_helper" "$cgroup_helper" "$validator"; do
  test -e "$path" || { echo "missing remote R05A apparatus input: $path" >&2; exit 2; }
done
command -v jq >/dev/null 2>&1 || { echo "remote jq is required" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || {
  echo "remote R05A apparatus source is not synchronized" >&2
  exit 2
}
test -z "$(git -C "$remote_repo" status --porcelain)" || {
  echo "remote R05A apparatus source tree is dirty" >&2
  exit 2
}
test ! -e "$run_root" || { echo "immutable R05A apparatus run id is already used" >&2; exit 2; }
test -z "$(squeue -h -u "$(whoami)" -o '%i')" || {
  echo "R05A apparatus regression requires an empty user queue" >&2
  exit 2
}

node_record=$(scontrol show node worker-1 -o)
node_state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$node_state" in
  *down*|*drain*|*fail*|*maint*|*not_resp*|*unknown*|*power*)
    echo "worker-1 is unhealthy: $node_state" >&2
    exit 2
    ;;
esac
free_mem=$(printf '%s\n' "$node_record" | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
case "$free_mem" in *[!0-9]*|'') echo "cannot parse worker-1 FreeMem" >&2; exit 2 ;; esac
test "$free_mem" -ge 8192 || {
  echo "R05A apparatus not submitted: worker-1 FreeMem=${free_mem}MiB < 8192MiB" >&2
  exit 2
}

mkdir "$run_root"
control_stage=held_sbatch
held_job_id=
released=false
control_failure() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ ! -e "$run_root/submission-failure.json" ]; then
    failure_tmp=$(mktemp "$run_root/.submission-failure.XXXXXX") || return "$status"
    jq -n \
      --arg run_id "$run_id" \
      --arg commit "$expected_commit" \
      --arg stage "$control_stage" \
      --arg status "$status" \
      --arg job_id "$held_job_id" \
      --arg released "$released" \
      --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      '{
        schema_version: "1.0",
        artifact_role: "r05a_adr0031_cpu_apparatus_submission_failure",
        status: "failed_closed",
        run_id: $run_id,
        git_commit: $commit,
        control_stage: $stage,
        exit_code: ($status | tonumber),
        slurm_array_job_id: (if ($job_id | length) > 0 then $job_id else null end),
        released: ($released == "true"),
        exact_job_must_be_inspected_before_cleanup: true,
        scientific_claim_allowed: false,
        timestamp_utc: $timestamp
      }' >"$failure_tmp" && mv "$failure_tmp" "$run_root/submission-failure.json"
    if [ -n "$held_job_id" ] && [ "$released" = false ]; then
      echo "R05A apparatus job $held_job_id remains held; inspect the exact receipt before cleanup" >&2
    fi
  fi
  return "$status"
}
trap control_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$slurm_log_root"

submission=$(
  RUN_ID="$run_id" \
  EXPECTED_GIT_COMMIT="$expected_commit" \
  REMOTE_REPO="$remote_repo" \
  EXPERIMENT_ROOT="$experiment_root" \
  sbatch --parsable --hold \
    --partition=main \
    --account=normal \
    --qos=normal \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:20:00 \
    --no-requeue \
    --nodelist=worker-1 \
    --array=0-0%1 \
    --output="$slurm_log_root/%x-%A_%a.out" \
    "$slurm_file"
)
job_id=${submission%%;*}
case "$job_id" in *[!0-9]*|'') echo "invalid R05A apparatus job id: $submission" >&2; exit 2 ;; esac
held_job_id=$job_id
control_stage=held_submission_receipt

held_tmp=$(mktemp "$run_root/.held-submission.XXXXXX")
jq -n \
  --arg run_id "$run_id" \
  --arg job_id "$job_id" \
  --arg commit "$expected_commit" \
  --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_adr0031_cpu_apparatus_held_submission",
    status: "sbatch_returned_held_job_id",
    run_id: $run_id,
    git_commit: $commit,
    slurm_array_job_id: $job_id,
    slurm_array_task_id: 0,
    source_node: "worker-1",
    partition: "main",
    account: "normal",
    qos: "normal",
    time_limit: "00:20:00",
    requeue: false,
    requested_cpus: 2,
    requested_host_memory_mib: 8192,
    requested_gpus: 0,
    released_at_receipt_time: false,
    scientific_claim_allowed: false,
    timestamp_utc: $timestamp
  }' >"$held_tmp"
mv "$held_tmp" "$run_root/held-submission.json"

control_stage=held_job_contract_validation
job_record=$(scontrol show job "$job_id" -o)
case " $job_record " in *" JobState=PENDING "*) ;; *) echo "apparatus job is not pending while held" >&2; exit 2 ;; esac
case " $job_record " in *" Reason=JobHeldUser "*) ;; *) echo "apparatus job is not held" >&2; exit 2 ;; esac
case " $job_record " in *" ReqNodeList=worker-1 "*) ;; *) echo "apparatus job lost worker-1 pin" >&2; exit 2 ;; esac
case " $job_record " in *" Partition=main "*) ;; *) echo "apparatus job partition changed" >&2; exit 2 ;; esac
case " $job_record " in *" Account=normal "*) ;; *) echo "apparatus job account changed" >&2; exit 2 ;; esac
case " $job_record " in *" QOS=normal "*) ;; *) echo "apparatus job QOS changed" >&2; exit 2 ;; esac
case " $job_record " in *" TimeLimit=00:20:00 "*) ;; *) echo "apparatus job time limit changed" >&2; exit 2 ;; esac
case " $job_record " in *" Requeue=0 "*) ;; *) echo "apparatus job requeue setting changed" >&2; exit 2 ;; esac
if printf '%s\n' "$job_record" | tr '[:upper:]' '[:lower:]' | grep -q 'gpu'; then
  echo "apparatus job record unexpectedly contains a GPU request" >&2
  exit 2
fi
req_tres=$(printf '%s\n' "$job_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test -n "$req_tres" || { echo "cannot parse apparatus ReqTRES" >&2; exit 2; }
req_cpus=$(printf '%s\n' "$req_tres" | tr ',' '\n' | sed -n 's/^cpu=\([0-9][0-9]*\)$/\1/p')
req_mem=$(printf '%s\n' "$req_tres" | tr ',' '\n' | sed -n 's/^mem=\([^,]*\)$/\1/p')
test "$req_cpus" = 2 || { echo "apparatus ReqTRES CPU count changed: $req_cpus" >&2; exit 2; }
case "$req_mem" in 8G|8192M) ;; *) echo "apparatus ReqTRES memory changed: $req_mem" >&2; exit 2 ;; esac

control_stage=atomic_submission_receipt
receipt_tmp=$(mktemp "$run_root/.submission.XXXXXX")
jq -n \
  --arg run_id "$run_id" \
  --arg job_id "$job_id" \
  --arg commit "$expected_commit" \
  --arg registry_sha "$(sha256sum "$registry" | awk '{print $1}')" \
  --arg test_helper_sha "$(sha256sum "$test_helper" | awk '{print $1}')" \
  --arg cgroup_helper_sha "$(sha256sum "$cgroup_helper" | awk '{print $1}')" \
  --arg runner_sha "$(sha256sum "$runner" | awk '{print $1}')" \
  --arg slurm_sha "$(sha256sum "$slurm_file" | awk '{print $1}')" \
  --arg validator_sha "$(sha256sum "$validator" | awk '{print $1}')" \
  --arg held_sha "$(sha256sum "$run_root/held-submission.json" | awk '{print $1}')" \
  --arg result "$run_root/results.json" \
  --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_adr0031_cpu_apparatus_submission",
    status: "reviewed_job_held_and_receipted",
    run_id: $run_id,
    git_commit: $commit,
    slurm_array_job_id: $job_id,
    slurm_array_task_id: 0,
    source_node: "worker-1",
    array: "0-0%1",
    partition: "main",
    account: "normal",
    qos: "normal",
    time_limit: "00:20:00",
    requeue: false,
    requested_cpus: 2,
    requested_host_memory_mib: 8192,
    requested_gpus: 0,
    registry_sha256: $registry_sha,
    allocation_test_helper_sha256: $test_helper_sha,
    cgroup_helper_sha256: $cgroup_helper_sha,
    runner_sha256: $runner_sha,
    slurm_sha256: $slurm_sha,
    validator_sha256: $validator_sha,
    held_submission_sha256: $held_sha,
    expected_result: $result,
    scientific_claim_allowed: false,
    timestamp_utc: $timestamp
  }' >"$receipt_tmp"
mv "$receipt_tmp" "$run_root/submission.json"

control_stage=release_exact_held_job
scontrol release "$job_id"
released=true
control_stage=complete
echo "submitted_r05a_apparatus_job_id=$job_id"
echo "submission_receipt=$run_root/submission.json"
echo "submission_receipt_sha256=$(sha256sum "$run_root/submission.json" | awk '{print $1}')"
REMOTE
