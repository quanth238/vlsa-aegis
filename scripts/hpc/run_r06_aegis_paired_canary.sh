#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?R06 paired canary must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?R06 paired canary requires an array job id}"
: "${SLURM_ARRAY_TASK_ID:?R06 paired canary requires an array task id}"
: "${RUN_ID:?immutable R06 paired run id is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed R06 paired release commit is required}"
: "${EXPECTED_CONFIG_SHA256:?released R06 config hash is required}"
: "${SOURCE_CONTRACT:?R06 paired source contract is required}"
: "${SUBMISSION_RECEIPT:?R06 paired atomic submission receipt is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

CONFIG=$REMOTE_REPO/configs/experiments/r06_aegis_collision_conditioned.json
WORKLOAD=$REMOTE_REPO/scripts/hpc/run_r06_aegis_paired_workload.sh
MANIFEST=$REMOTE_REPO/manifests/oracle_h05_colliding.jsonl
LABEL_MANIFEST=$REMOTE_REPO/manifests/r06_codex_obstacle_labels_canary.jsonl
R02_CONFIG=$REMOTE_REPO/configs/experiments/r02_oracle_flow.json

test "$SLURM_ARRAY_TASK_ID" = 0 || { echo "R06 paired canary is task zero only" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "R06 paired canary is pinned to worker-1" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 8 || { echo "R06 paired canary requires eight CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 65536 || { echo "R06 paired canary requires 64 GiB" >&2; exit 2; }
test -n "${CUDA_VISIBLE_DEVICES:-}" && test "$CUDA_VISIBLE_DEVICES" != NoDevFiles || {
  echo "R06 paired canary has no allocation-visible GPU" >&2; exit 2
}
case "$CUDA_VISIBLE_DEVICES" in *,*) echo "R06 paired canary requires one GPU" >&2; exit 2 ;; esac

for path in "$CONFIG" "$WORKLOAD" "$MANIFEST" "$LABEL_MANIFEST" "$R02_CONFIG" \
  "$SOURCE_CONTRACT" "$SUBMISSION_RECEIPT"; do
  test -f "$path" && test ! -L "$path" || { echo "missing paired input: $path" >&2; exit 2; }
done
test -x "$WORKLOAD"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256"
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41
test "$(sha256sum "$R02_CONFIG" | awk '{print $1}')" = c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e
test "$(sha256sum "$LABEL_MANIFEST" | awk '{print $1}')" = 6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f
SOURCE_CONTRACT_SHA256=$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')
SUBMISSION_SHA256=$(sha256sum "$SUBMISSION_RECEIPT" | awk '{print $1}')
CPU_VALIDATOR_JOB_ID=$(jq -er '.cpu_afterany_job_id' "$SUBMISSION_RECEIPT")

ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$CONFIG")
RELEASE_ADR=$(jq -er '.execution_release.decision_artifact' "$CONFIG")
case "$RELEASE_ADR" in
  docs/decisions/*-release-aegis-paired-canary.md) ;;
  *) echo "R06 paired release ADR path changed" >&2; exit 2 ;;
esac
test "$(git -C "$REMOTE_REPO" rev-list --parents -n 1 "$EXPECTED_GIT_COMMIT")" = \
  "$EXPECTED_GIT_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT"
test "$(git -C "$REMOTE_REPO" diff --name-only "$ACCEPTED_IMPLEMENTATION_COMMIT" "$EXPECTED_GIT_COMMIT" | LC_ALL=C sort)" = \
  "$(printf '%s\n' configs/experiments/r06_aegis_collision_conditioned.json "$RELEASE_ADR" | LC_ALL=C sort)"

jq -e --arg run "$RUN_ID" '
  .ready_to_run == true and .blocked_on == []
  and .execution_release.artifact_role == "r06_aegis_paired_codex_label_canary_execution_release"
  and .execution_release.stage == "paired_codex_label_canary"
  and .execution_release.run_id == $run
  and .execution_release.single_case_index == 0
  and .execution_release.case_id == "crfs-1069f29a8d76463a"
  and .execution_release.aegis_execution_allowed == true
  and .execution_release.semantic_label_required == true
  and .execution_release.groundingdino_execution_allowed == true
  and .execution_release.qp_execution_allowed == true
  and .execution_release.original_glm_execution_allowed == false
  and .execution_release.probe_or_mlp_training_authorized == false
  and .execution_release.automatic_population_launch_authorized == false
  and .execution_release.codex_label_manifest.sha256 == "6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f"
  and .execution_release.codex_label_manifest.freeze_commit == "d9cf569d619e014c9e6423cd9ddb40f592435a71"
  and .execution_release.codex_label_manifest.capture_artifact_sha256 == "f2d32024ac6fac5aa5d133b5138df72501f1590b8cb0927b732edde69dabaaac"
  and .execution_release.codex_label_manifest.capture_completed_at == "2026-07-17T06:55:59Z"
  and .execution_release.resources.validator_dependency == "afterany"
  and .execution_release.resources.validator_gpus == 0
' "$CONFIG" >/dev/null

jq -e --arg run "$RUN_ID" --arg commit "$EXPECTED_GIT_COMMIT" '
  .artifact_role == "r06_aegis_paired_canary_source_contract"
  and .status == "sources_bound_before_held_submission"
  and .run_id == $run and .git_commit == $commit
  and .stage == "paired_codex_label_canary"
  and .collision_conditioned_only == true
  and .probe_or_mlp_training_authorized == false
' "$SOURCE_CONTRACT" >/dev/null
jq -e --arg run "$RUN_ID" --arg gpu "$SLURM_ARRAY_JOB_ID" \
  --arg cpu "$CPU_VALIDATOR_JOB_ID" --arg source "$SOURCE_CONTRACT_SHA256" '
  .artifact_role == "r06_aegis_paired_canary_atomic_submission"
  and .status == "gpu_and_afterany_registered_before_release"
  and .run_id == $run
  and .gpu_slurm_array_job_id == $gpu
  and .exact_gpu_task_id == ($gpu+"_0")
  and .cpu_afterany_job_id == $cpu
  and .dependency == ("afterany:"+$gpu)
  and .source_contract_sha256 == $source
  and .released_at_receipt_time == false
' "$SUBMISSION_RECEIPT" >/dev/null

job_record=$(scontrol show job "${SLURM_ARRAY_JOB_ID}_0" -o)
for field in "ArrayJobId=$SLURM_ARRAY_JOB_ID" "ArrayTaskId=0" JobState=RUNNING \
  Partition=main Account=normal QOS=normal TimeLimit=00:30:00 Requeue=0 \
  ReqNodeList=worker-1 NumCPUs=8 CPUs/Task=8; do
  case " $job_record " in *" $field "*) ;; *) echo "R06 paired job field changed: $field" >&2; exit 2 ;; esac
done

export R06_PAIRED_SOURCE_CONTRACT=$SOURCE_CONTRACT
export R06_PAIRED_SUBMISSION_RECEIPT=$SUBMISSION_RECEIPT
exec "$WORKLOAD"
