#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?usage: RUN_ID=r05a-cgroup-v2-current-capability-20260715a submit_r05a_cgroup_v2_current_capability.sh}"
EXPECTED_RUN_ID=r05a-cgroup-v2-current-capability-20260715a
test "$RUN_ID" = "$EXPECTED_RUN_ID" || { echo "capability run id changed" >&2; exit 2; }

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
EXPERIMENT_ROOT=/mnt/data/quanth/experiments/crfs-oracle
SLURM_LOG_ROOT=/mnt/data/quanth/slurm_logs/crfs-oracle
for path in \
  docs/decisions/0035-preregister-r05a-sampled-current-capability-gate.md \
  scripts/hpc/lib/cgroup_v2_current_capability.sh \
  scripts/hpc/run_r05a_cgroup_v2_current_capability.sh \
  scripts/hpc/submit_r05a_cgroup_v2_current_capability.sh \
  slurm/r05a_cgroup_v2_current_capability_cpu.sbatch \
  tests/test_r05a_cgroup_v2_current_capability.py; do
  test -f "$path" || { echo "missing capability source: $path" >&2; exit 2; }
done
git ls-files --error-unmatch \
  docs/decisions/0035-preregister-r05a-sampled-current-capability-gate.md \
  scripts/hpc/lib/cgroup_v2_current_capability.sh \
  scripts/hpc/run_r05a_cgroup_v2_current_capability.sh \
  scripts/hpc/submit_r05a_cgroup_v2_current_capability.sh \
  slurm/r05a_cgroup_v2_current_capability_cpu.sbatch \
  tests/test_r05a_cgroup_v2_current_capability.py >/dev/null || {
  echo "capability sources must be committed before submission" >&2
  exit 2
}
test -z "$(git status --porcelain)" || { echo "capability submission requires a clean worktree" >&2; exit 2; }
EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)
test "$EXPECTED_GIT_COMMIT" = "$(git rev-parse '@{upstream}')" || {
  echo "capability source is not pushed" >&2
  exit 2
}
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
adr=$remote_repo/docs/decisions/0035-preregister-r05a-sampled-current-capability-gate.md
helper=$remote_repo/scripts/hpc/lib/cgroup_v2_current_capability.sh
runner=$remote_repo/scripts/hpc/run_r05a_cgroup_v2_current_capability.sh
submitter=$remote_repo/scripts/hpc/submit_r05a_cgroup_v2_current_capability.sh
slurm_file=$remote_repo/slurm/r05a_cgroup_v2_current_capability_cpu.sbatch
for path in "$adr" "$helper" "$runner" "$submitter" "$slurm_file"; do
  test -f "$path" || { echo "missing remote capability source: $path" >&2; exit 2; }
done
command -v jq >/dev/null 2>&1 || { echo "remote jq is required" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || { echo "remote source mismatch" >&2; exit 2; }
test -z "$(git -C "$remote_repo" status --porcelain)" || { echo "remote source is dirty" >&2; exit 2; }
test ! -e "$run_root" || { echo "immutable capability run id is already used" >&2; exit 2; }
test -z "$(squeue -h -u "$(whoami)" -o '%i')" || { echo "capability gate requires an empty user queue" >&2; exit 2; }

node_record=$(scontrol show node worker-1 -o)
node_state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$node_state" in *down*|*drain*|*fail*|*maint*|*not_resp*|*unknown*|*power*) echo "worker-1 is unhealthy: $node_state" >&2; exit 2 ;; esac
free_mem=$(printf '%s\n' "$node_record" | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
case "$free_mem" in *[!0-9]*|'') echo "cannot parse worker-1 FreeMem" >&2; exit 2 ;; esac
test "$free_mem" -ge 256 || { echo "worker-1 has less than 256 MiB free" >&2; exit 2; }

mkdir "$run_root"
mkdir -p "$slurm_log_root"
control_stage=held_sbatch
held_job_id=
released=false
control_failure() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ ! -e "$run_root/submission-failure.json" ]; then
    temporary=$(mktemp "$run_root/.submission-failure.XXXXXX") || return "$status"
    jq -n \
      --arg run_id "$run_id" --arg commit "$expected_commit" \
      --arg stage "$control_stage" --arg exit_code "$status" \
      --arg job_id "$held_job_id" --arg released "$released" \
      '{
        schema_version: "1.0",
        artifact_role: "r05a_cgroup_v2_current_capability_submission_failure",
        status: "failed_closed",
        run_id: $run_id,
        git_commit: $commit,
        control_stage: $stage,
        exit_code: ($exit_code | tonumber),
        slurm_array_job_id: (if ($job_id | length) > 0 then $job_id else null end),
        released: ($released == "true"),
        exact_job_must_be_inspected_before_cleanup: true,
        scientific_claim_allowed: false
      }' >"$temporary" && mv "$temporary" "$run_root/submission-failure.json"
    if [ -n "$held_job_id" ] && [ "$released" = false ]; then
      echo "capability job $held_job_id remains held; inspect exact receipt before cleanup" >&2
    fi
  fi
  return "$status"
}
trap control_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

submission=$(
  RUN_ID="$run_id" EXPECTED_GIT_COMMIT="$expected_commit" REMOTE_REPO="$remote_repo" EXPERIMENT_ROOT="$experiment_root" \
  sbatch --parsable --hold \
    --partition=main --account=normal --qos=normal \
    --cpus-per-task=1 --mem=256M --time=00:02:00 --no-requeue \
    --nodelist=worker-1 --array=0-0%1 \
    --output="$slurm_log_root/%x-%A_%a.out" "$slurm_file"
)
job_id=${submission%%;*}
case "$job_id" in *[!0-9]*|'') echo "invalid capability job id: $submission" >&2; exit 2 ;; esac
held_job_id=$job_id
control_stage=held_submission_receipt
temporary=$(mktemp "$run_root/.held-submission.XXXXXX")
jq -n \
  --arg run_id "$run_id" --arg job_id "$job_id" --arg commit "$expected_commit" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_cgroup_v2_current_capability_held_submission",
    status: "sbatch_returned_held_job_id",
    run_id: $run_id,
    git_commit: $commit,
    slurm_array_job_id: $job_id,
    slurm_array_task_id: 0,
    source_node: "worker-1",
    partition: "main",
    account: "normal",
    qos: "normal",
    time_limit: "00:02:00",
    requeue: false,
    requested_cpus: 1,
    requested_host_memory_mib: 256,
    requested_gpus: 0,
    released_at_receipt_time: false,
    scientific_claim_allowed: false
  }' >"$temporary"
mv "$temporary" "$run_root/held-submission.json"

control_stage=held_job_contract_validation
job_record=$(scontrol show job "$job_id" -o)
case " $job_record " in *" JobState=PENDING "*) ;; *) echo "capability job is not pending while held" >&2; exit 2 ;; esac
case " $job_record " in *" Reason=JobHeldUser "*) ;; *) echo "capability job is not held" >&2; exit 2 ;; esac
case " $job_record " in *" ReqNodeList=worker-1 "*) ;; *) echo "capability job lost worker-1 pin" >&2; exit 2 ;; esac
case " $job_record " in *" Partition=main "*) ;; *) echo "capability partition changed" >&2; exit 2 ;; esac
case " $job_record " in *" Account=normal "*) ;; *) echo "capability account changed" >&2; exit 2 ;; esac
case " $job_record " in *" QOS=normal "*) ;; *) echo "capability QOS changed" >&2; exit 2 ;; esac
case " $job_record " in *" TimeLimit=00:02:00 "*) ;; *) echo "capability time limit changed" >&2; exit 2 ;; esac
case " $job_record " in *" Requeue=0 "*) ;; *) echo "capability requeue setting changed" >&2; exit 2 ;; esac
printf '%s\n' "$job_record" | tr '[:upper:]' '[:lower:]' | grep -q gpu && { echo "capability job unexpectedly requests a GPU" >&2; exit 2; }
req_tres=$(printf '%s\n' "$job_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
req_cpus=$(printf '%s\n' "$req_tres" | tr ',' '\n' | sed -n 's/^cpu=\([0-9][0-9]*\)$/\1/p')
req_mem=$(printf '%s\n' "$req_tres" | tr ',' '\n' | sed -n 's/^mem=\([^,]*\)$/\1/p')
test "$req_cpus" = 1 || { echo "capability CPU request changed" >&2; exit 2; }
case "$req_mem" in 256M) ;; *) echo "capability memory request changed: $req_mem" >&2; exit 2 ;; esac

control_stage=atomic_submission_receipt
temporary=$(mktemp "$run_root/.submission.XXXXXX")
jq -n \
  --arg run_id "$run_id" --arg job_id "$job_id" --arg commit "$expected_commit" \
  --arg helper_sha "$(sha256sum "$helper" | awk '{print $1}')" \
  --arg runner_sha "$(sha256sum "$runner" | awk '{print $1}')" \
  --arg submitter_sha "$(sha256sum "$submitter" | awk '{print $1}')" \
  --arg slurm_sha "$(sha256sum "$slurm_file" | awk '{print $1}')" \
  --arg adr_sha "$(sha256sum "$adr" | awk '{print $1}')" \
  --arg held_sha "$(sha256sum "$run_root/held-submission.json" | awk '{print $1}')" \
  --arg expected_result "$run_root/results.json" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_cgroup_v2_current_capability_submission",
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
    time_limit: "00:02:00",
    requeue: false,
    requested_cpus: 1,
    requested_host_memory_mib: 256,
    requested_gpus: 0,
    helper_sha256: $helper_sha,
    runner_sha256: $runner_sha,
    submitter_sha256: $submitter_sha,
    slurm_sha256: $slurm_sha,
    adr_sha256: $adr_sha,
    held_submission_sha256: $held_sha,
    expected_result: $expected_result,
    scientific_claim_allowed: false
  }' >"$temporary"
mv "$temporary" "$run_root/submission.json"
control_stage=release_exact_held_job
scontrol release "$job_id"
released=true
control_stage=complete
echo "submitted_r05a_cgroup_capability_job_id=$job_id"
echo "submission_receipt=$run_root/submission.json"
echo "submission_receipt_sha256=$(sha256sum "$run_root/submission.json" | awk '{print $1}')"
REMOTE
