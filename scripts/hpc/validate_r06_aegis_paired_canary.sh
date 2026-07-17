#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?R06 paired validator must run inside Slurm}"
: "${SOURCE_JOB_ID:?R06 paired source GPU job id is required}"
: "${RUN_ID:?R06 paired run id is required}"
: "${EXPECTED_GIT_COMMIT:?R06 paired release commit is required}"
: "${SOURCE_CONTRACT:?R06 paired source contract is required}"
: "${SUBMISSION_RECEIPT:?R06 paired submission receipt is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"

LIBERO_PYTHON=/mnt/data/quanth/venvs/openpi-libero-client/bin/python
CONFIG=$REMOTE_REPO/configs/experiments/r06_aegis_collision_conditioned.json
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
RESULT=$RUN_ROOT/crfs-1069f29a8d76463a/results.json
RECEIPT=$RUN_ROOT/cpu-afterany-validation.json
VALIDATOR=$REMOTE_REPO/main/validate_crfs_r06_aegis_paired_canary.py
STATUS_HELPER=$REMOTE_REPO/scripts/hpc/lib/slurm_exact_array_task_status.sh

test "${SLURM_CPUS_PER_TASK:-}" = 2 || { echo "R06 paired validator requires two CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 8192 || { echo "R06 paired validator requires 8 GiB" >&2; exit 2; }
test "${CUDA_VISIBLE_DEVICES:-NoDevFiles}" = NoDevFiles || {
  echo "R06 paired validator must not receive a GPU" >&2; exit 2
}
for path in "$CONFIG" "$SOURCE_CONTRACT" "$SUBMISSION_RECEIPT" "$VALIDATOR" "$STATUS_HELPER"; do
  test -f "$path" && test ! -L "$path" || { echo "missing validator input: $path" >&2; exit 2; }
done
test -x "$LIBERO_PYTHON"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
SOURCE_CONTRACT_SHA256=$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')
SUBMISSION_SHA256=$(sha256sum "$SUBMISSION_RECEIPT" | awk '{print $1}')
jq -e --arg run "$RUN_ID" --arg gpu "$SOURCE_JOB_ID" --arg cpu "$SLURM_JOB_ID" '
  .artifact_role == "r06_aegis_paired_canary_atomic_submission"
  and .run_id == $run and .gpu_slurm_array_job_id == $gpu
  and .exact_gpu_task_id == ($gpu+"_0")
  and .cpu_afterany_job_id == $cpu
  and .dependency == ("afterany:"+$gpu)
  and .released_at_receipt_time == false
' "$SUBMISSION_RECEIPT" >/dev/null

publisher_record=$(scontrol show job "$SLURM_JOB_ID" -o)
for field in "JobId=$SLURM_JOB_ID" JobState=RUNNING Partition=main Account=normal \
  QOS=normal TimeLimit=00:10:00 Requeue=0 NumCPUs=2 CPUs/Task=2; do
  case " $publisher_record " in *" $field "*) ;; *) echo "R06 validator field changed: $field" >&2; exit 2 ;; esac
done
case " $publisher_record " in
  *" TRESPerNode=gres/gpu:"*) echo "R06 validator unexpectedly has a GPU" >&2; exit 2 ;;
esac

. "$STATUS_HELPER"
source_record=
set +e
source_record=$(crfs_wait_for_exact_completed_array_task "$SOURCE_JOB_ID" 30 1)
source_query_status=$?
set -e
IFS='|' read -r SOURCE_STATE SOURCE_EXIT_CODE <<EOF
$source_record
EOF
test -n "$SOURCE_STATE" || SOURCE_STATE=missing
test -n "$SOURCE_EXIT_CODE" || SOURCE_EXIT_CODE=missing

export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main
set +e
"$LIBERO_PYTHON" "$VALIDATOR" \
  --run-root "$RUN_ROOT" --config "$CONFIG" --result "$RESULT" --receipt "$RECEIPT" \
  --source-contract "$SOURCE_CONTRACT" --source-contract-sha256 "$SOURCE_CONTRACT_SHA256" \
  --submission "$SUBMISSION_RECEIPT" --submission-sha256 "$SUBMISSION_SHA256" \
  --source-job-id "$SOURCE_JOB_ID" --validator-job-id "$SLURM_JOB_ID" \
  --source-state "$SOURCE_STATE" --source-exit-code "$SOURCE_EXIT_CODE"
validator_status=$?
set -e
test -f "$RECEIPT" && test ! -L "$RECEIPT" || {
  echo "R06 paired validator did not write its receipt" >&2; exit 4
}
echo "source_query_status=$source_query_status"
echo "source_state=$SOURCE_STATE"
echo "source_exit_code=$SOURCE_EXIT_CODE"
echo "validation_receipt=$RECEIPT"
echo "validation_receipt_sha256=$(sha256sum "$RECEIPT" | awk '{print $1}')"
exit "$validator_status"
