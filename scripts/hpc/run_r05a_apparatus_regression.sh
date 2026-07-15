#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?R05A apparatus regression must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?R05A apparatus regression must expose its array job id}"
: "${SLURM_ARRAY_TASK_ID:?R05A apparatus regression must be a one-row array}"
: "${SLURM_JOB_PARTITION:?R05A apparatus regression must record its partition}"
: "${RUN_ID:?RUN_ID must be the preregistered immutable apparatus id}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${OPENPI_PYTHON:=/mnt/data/quanth/venvs/openpi/bin/python}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"

EXPECTED_RUN_ID=r05a-adr0031-apparatus-cpu-20260715a
case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe R05A apparatus RUN_ID" >&2; exit 2 ;; esac
test "$RUN_ID" = "$EXPECTED_RUN_ID" || { echo "R05A apparatus run id changed" >&2; exit 2; }
ALLOCATION_TEST_REGISTRY=$REMOTE_REPO/main/crfs_oracle/r05a_allocation_tests.json
CGROUP_HELPER=$REMOTE_REPO/scripts/hpc/lib/cgroup_memory.sh
TEST_HELPER=$REMOTE_REPO/scripts/hpc/lib/r05a_allocation_tests.sh
SBATCH_FILE=$REMOTE_REPO/slurm/r05a_apparatus_regression_cpu.sbatch
RUNNER_FILE=$REMOTE_REPO/scripts/hpc/run_r05a_apparatus_regression.sh
VALIDATOR_FILE=$REMOTE_REPO/main/validate_crfs_r05a_apparatus_regression.py
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
SUBMISSION=$RUN_ROOT/submission.json
SUITE_LOG_DIR=$RUN_ROOT/suites
TEST_LOG=$RUN_ROOT/allocation-focused-tests.log
CGROUP_DIAGNOSTIC=$RUN_ROOT/host-cgroup-memory.tsv
RESULT=$RUN_ROOT/results.json
RESULT_CANDIDATE=$RUN_ROOT/.results.pending.json
FAILURE=$RUN_ROOT/failure.json
FAILURE_STAGE=allocation_contract

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ]; then
    rm -f "$RESULT_CANDIDATE"
  fi
  if [ "$status" -ne 0 ] && [ ! -e "$FAILURE" ] && [ -d "$RUN_ROOT" ]; then
    temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX") || return "$status"
    jq -n \
      --arg run_id "$RUN_ID" \
      --arg stage "$FAILURE_STAGE" \
      --arg status "$status" \
      --arg job_id "$SLURM_JOB_ID" \
      --arg array_job_id "$SLURM_ARRAY_JOB_ID" \
      --arg task_id "$SLURM_ARRAY_TASK_ID" \
      --arg host "$(hostname -s)" \
      '{
        schema_version: "1.0",
        artifact_role: "r05a_adr0031_cpu_apparatus_failure",
        status: "failed",
        run_id: $run_id,
        failure_stage: $stage,
        exit_code: ($status | tonumber),
        slurm_job_id: $job_id,
        slurm_array_job_id: $array_job_id,
        slurm_array_task_id: ($task_id | tonumber),
        host: $host,
        scientific_claim_allowed: false,
        simulator_efficacy_evaluated: false,
        checkpoint_loaded: false,
        policy_server_started: false,
        real_pi05_teacher_searches_executed: 0,
        real_pi05_teacher_observations_produced: 0,
        synthetic_unit_test_solver_calls_are_not_real_teacher_searches: true,
        simulator_steps_executed: 0,
        probe_training_authorized: false
      }' >"$temporary" && mv "$temporary" "$FAILURE"
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test "$SLURM_JOB_PARTITION" = main || { echo "R05A apparatus regression is frozen to main" >&2; exit 2; }
test "$SLURM_ARRAY_TASK_ID" = 0 || { echo "R05A apparatus regression requires array row zero" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "R05A apparatus regression must run on worker-1" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 2 || { echo "R05A apparatus regression requires two CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 8192 || { echo "R05A apparatus regression requires exactly 8 GiB" >&2; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-NoDevFiles}" in
  ''|NoDevFiles) ;;
  *) echo "R05A apparatus regression must not receive a GPU" >&2; exit 2 ;;
esac
test -z "${SLURM_JOB_GPUS:-}" || { echo "R05A apparatus regression received SLURM_JOB_GPUS" >&2; exit 2; }
test -f "$SUBMISSION" || { echo "R05A apparatus regression requires its held submission receipt" >&2; exit 2; }
test ! -e "$RESULT" && test ! -e "$RESULT_CANDIDATE" && test ! -e "$FAILURE" && test ! -e "$SUITE_LOG_DIR" || {
  echo "immutable R05A apparatus run already contains execution output" >&2
  exit 2
}
for path in \
  "$OPENPI_PYTHON" "$ALLOCATION_TEST_REGISTRY" "$CGROUP_HELPER" "$TEST_HELPER" \
  "$SBATCH_FILE" "$RUNNER_FILE" \
  "$VALIDATOR_FILE" \
  "$REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh"; do
  test -e "$path" || { echo "missing R05A apparatus input: $path" >&2; exit 2; }
done
SUBMISSION_SHA256=$(sha256sum "$SUBMISSION" | awk '{print $1}')
REGISTRY_SHA256=$(sha256sum "$ALLOCATION_TEST_REGISTRY" | awk '{print $1}')
TEST_HELPER_SHA256=$(sha256sum "$TEST_HELPER" | awk '{print $1}')
CGROUP_HELPER_SHA256=$(sha256sum "$CGROUP_HELPER" | awk '{print $1}')
RUNNER_SHA256=$(sha256sum "$RUNNER_FILE" | awk '{print $1}')
SBATCH_SHA256=$(sha256sum "$SBATCH_FILE" | awk '{print $1}')
VALIDATOR_SHA256=$(sha256sum "$VALIDATOR_FILE" | awk '{print $1}')
jq -e \
  --arg run_id "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg submission_sha "$SUBMISSION_SHA256" \
  --arg array_job_id "$SLURM_ARRAY_JOB_ID" \
  --arg registry_sha "$REGISTRY_SHA256" \
  --arg test_helper_sha "$TEST_HELPER_SHA256" \
  --arg cgroup_helper_sha "$CGROUP_HELPER_SHA256" \
  --arg runner_sha "$RUNNER_SHA256" \
  --arg slurm_sha "$SBATCH_SHA256" \
  --arg validator_sha "$VALIDATOR_SHA256" \
  '.status == "reviewed_job_held_and_receipted"
   and .run_id == $run_id
   and .git_commit == $commit
   and .slurm_array_job_id == $array_job_id
   and .slurm_array_task_id == 0
   and .source_node == "worker-1"
   and .partition == "main"
   and .account == "normal"
   and .qos == "normal"
   and .time_limit == "00:20:00"
   and .requeue == false
   and .requested_cpus == 2
   and .requested_host_memory_mib == 8192
   and .requested_gpus == 0
   and .registry_sha256 == $registry_sha
   and .allocation_test_helper_sha256 == $test_helper_sha
   and .cgroup_helper_sha256 == $cgroup_helper_sha
   and .runner_sha256 == $runner_sha
   and .slurm_sha256 == $slurm_sha
   and .validator_sha256 == $validator_sha
   and .scientific_claim_allowed == false' "$SUBMISSION" >/dev/null || {
  echo "R05A apparatus submission receipt contract changed" >&2
  exit 2
}

mkdir "$SUITE_LOG_DIR"

GIT_COMMIT=$(git -C "$REMOTE_REPO" rev-parse HEAD)
GIT_DIRTY=$(test -n "$(git -C "$REMOTE_REPO" status --porcelain)" && echo true || echo false)
test "$GIT_COMMIT" = "$EXPECTED_GIT_COMMIT" || { echo "R05A apparatus reviewed commit mismatch" >&2; exit 2; }
test "$GIT_DIRTY" = false || { echo "R05A apparatus requires a clean remote tree" >&2; exit 2; }

. "$CGROUP_HELPER"
. "$TEST_HELPER"
cd "$REMOTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

FAILURE_STAGE=dependency_backed_focused_tests
TRANSFORMERS_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh)
export PYTHONPATH=$TRANSFORMERS_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src
crfs_run_r05a_allocation_tests \
  "$ALLOCATION_TEST_REGISTRY" "$OPENPI_PYTHON" tests "$TEST_LOG" "$SUITE_LOG_DIR"

FAILURE_STAGE=live_cgroup_memory_peak
if ! host_cgroup_peak_bytes=$(crfs_read_live_cgroup_memory_peak \
  /proc/self/cgroup /proc/self/mountinfo "$CGROUP_DIAGNOSTIC"); then
  echo "R05A apparatus could not read the live worker-1 cgroup peak" >&2
  exit 3
fi
case "$host_cgroup_peak_bytes" in *[!0-9]*|'') echo "invalid cgroup peak" >&2; exit 3 ;; esac
test "$host_cgroup_peak_bytes" -gt 0 || { echo "nonpositive cgroup peak" >&2; exit 3; }
test "$host_cgroup_peak_bytes" -le $((8192 * 1024 * 1024)) || {
  echo "cgroup peak exceeds the exact 8 GiB apparatus allocation" >&2
  exit 3
}

FAILURE_STAGE=independent_result_finalization
"$OPENPI_PYTHON" - \
  "$TEST_LOG" "$CGROUP_DIAGNOSTIC" "$RESULT_CANDIDATE" "$RUN_ID" "$EXPECTED_GIT_COMMIT" \
  "$SUBMISSION" "$SUBMISSION_SHA256" \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" \
  "$(hostname -s)" "$host_cgroup_peak_bytes" <<'PY'
from datetime import datetime, timezone
import sys

from crfs_harness.artifacts import atomic_write_json
from crfs_oracle.r05a_canary import (
    ALLOCATION_TEST_COUNTS,
    ALLOCATION_TEST_REGISTRY_SHA256,
    _parse_allocation_test_log,
    _parse_cgroup_memory_diagnostic,
)

(
    test_log,
    cgroup_path,
    output,
    run_id,
    expected_commit,
    submission_path,
    submission_sha256,
    job_id,
    array_job_id,
    task_id,
    host,
    shell_peak,
) = sys.argv[1:]
test_sha, observed_counts = _parse_allocation_test_log(test_log)
cgroup_sha, cgroup_record = _parse_cgroup_memory_diagnostic(cgroup_path)
shell_peak_value = int(shell_peak)
if observed_counts != dict(ALLOCATION_TEST_COUNTS):
    raise SystemExit("allocation test counts differ from the authoritative registry")
if int(cgroup_record["peak_bytes"]) != shell_peak_value:
    raise SystemExit("parsed cgroup peak differs from the shell resolver result")
if shell_peak_value <= 0 or shell_peak_value > 8192 * 1024 * 1024:
    raise SystemExit("parsed cgroup peak is outside the exact CPU allocation")

result = {
    "schema_version": "1.0",
    "artifact_role": "r05a_adr0031_cpu_apparatus_regression",
    "status": "passed",
    "run_id": run_id,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": expected_commit,
    "git_dirty": False,
    "submission_receipt_path": submission_path,
    "submission_receipt_sha256": submission_sha256,
    "host": host,
    "slurm_job_id": job_id,
    "slurm_array_job_id": array_job_id,
    "slurm_array_task_id": int(task_id),
    "requested_cpus": 2,
    "requested_host_memory_mib": 8192,
    "partition": "main",
    "account": "normal",
    "qos": "normal",
    "time_limit": "00:20:00",
    "requeue": False,
    "gpu_allocated": False,
    "allocation_tests": {
        "registry_path": "main/crfs_oracle/r05a_allocation_tests.json",
        "registry_sha256": ALLOCATION_TEST_REGISTRY_SHA256,
        "expected_counts": dict(ALLOCATION_TEST_COUNTS),
        "observed_counts": dict(observed_counts),
        "zero_skips": True,
        "log_path": test_log,
        "log_sha256": test_sha,
    },
    "host_cgroup_memory": {
        "diagnostic_path": cgroup_path,
        "diagnostic_sha256": cgroup_sha,
        **cgroup_record,
    },
    "scientific_claim_allowed": False,
    "simulator_efficacy_evaluated": False,
    "checkpoint_loaded": False,
    "policy_server_started": False,
    "real_pi05_teacher_searches_executed": 0,
    "real_pi05_teacher_observations_produced": 0,
    "synthetic_unit_test_solver_calls_excluded_from_guard": True,
    "simulator_steps_executed": 0,
    "probe_training_authorized": False,
    "retry_c_authorized_by_this_artifact_alone": False,
}
atomic_write_json(output, result)
PY

test -f "$RESULT_CANDIDATE" || { echo "R05A apparatus finalizer wrote no candidate" >&2; exit 4; }
FAILURE_STAGE=independent_result_validation
"$OPENPI_PYTHON" "$VALIDATOR_FILE" \
  --result "$RESULT_CANDIDATE" \
  --submission "$SUBMISSION" \
  --expected-run-root "$RUN_ROOT" \
  --expected-result-path "$RESULT" \
  --repo-root "$REMOTE_REPO" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  --expected-job-id "$SLURM_JOB_ID" \
  --expected-array-job-id "$SLURM_ARRAY_JOB_ID" \
  --expected-task-id "$SLURM_ARRAY_TASK_ID" \
  --expected-submission-sha256 "$SUBMISSION_SHA256"
jq -e \
  --arg run_id "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg submission_sha "$SUBMISSION_SHA256" \
  '.status == "passed"
   and .run_id == $run_id
   and .git_commit == $commit
   and .submission_receipt_sha256 == $submission_sha
   and .host == "worker-1"
   and .slurm_array_task_id == 0
   and .requested_cpus == 2
   and .requested_host_memory_mib == 8192
   and .partition == "main"
   and .account == "normal"
   and .qos == "normal"
   and .time_limit == "00:20:00"
   and .requeue == false
   and .gpu_allocated == false
   and .allocation_tests.zero_skips == true
   and .scientific_claim_allowed == false
   and .simulator_efficacy_evaluated == false
   and .checkpoint_loaded == false
   and .policy_server_started == false
   and .real_pi05_teacher_searches_executed == 0
   and .real_pi05_teacher_observations_produced == 0
   and .synthetic_unit_test_solver_calls_excluded_from_guard == true
   and .simulator_steps_executed == 0
   and .probe_training_authorized == false
   and .retry_c_authorized_by_this_artifact_alone == false' "$RESULT_CANDIDATE" >/dev/null || {
  echo "R05A apparatus result guard validation failed" >&2
  exit 4
}
mv "$RESULT_CANDIDATE" "$RESULT"
test -f "$RESULT" || { echo "R05A apparatus result publication failed" >&2; exit 4; }

FAILURE_STAGE=complete
echo "result=$RESULT"
echo "result_sha256=$(sha256sum "$RESULT" | awk '{print $1}')"
echo "host_cgroup_peak_bytes=$host_cgroup_peak_bytes"
