#!/usr/bin/env bash
set -euo pipefail

# Allocation-side release guard. The capture workload cannot run AEGIS,
# GroundingDINO, a semantic selector, or a QP.
: "${SLURM_JOB_ID:?R06 capture must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?R06 capture requires an array job id}"
: "${SLURM_ARRAY_TASK_ID:?R06 capture requires an array task id}"
: "${RUN_ID:?immutable R06 capture run id is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed R06 capture commit is required}"
: "${EXPECTED_CONFIG_SHA256:?released R06 capture config hash is required}"
: "${SOURCE_CONTRACT:?R06 capture source contract is required}"
: "${SUBMISSION_RECEIPT:?R06 capture submission receipt is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"

CONFIG=$REMOTE_REPO/configs/experiments/r06_aegis_collision_conditioned.json
WORKLOAD=$REMOTE_REPO/scripts/hpc/run_r06_aegis_capture_workload.sh
MANIFEST=$REMOTE_REPO/manifests/oracle_h05_colliding.jsonl
R02_CONFIG=$REMOTE_REPO/configs/experiments/r02_oracle_flow.json

case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe R06 capture run id" >&2; exit 2 ;; esac
case "$SLURM_ARRAY_JOB_ID" in *[!0-9]*|'') echo "invalid R06 array id" >&2; exit 2 ;; esac
test "$SLURM_ARRAY_TASK_ID" = 0 || { echo "R06 capture is fixed to task zero" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "R06 capture is source-pinned to worker-1" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 8 || { echo "R06 capture requires eight CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 65536 || { echo "R06 capture requires exactly 64 GiB" >&2; exit 2; }
test -n "${CUDA_VISIBLE_DEVICES:-}" && test "${CUDA_VISIBLE_DEVICES:-}" != NoDevFiles || {
  echo "R06 capture has no allocation-visible GPU" >&2
  exit 2
}
case "$CUDA_VISIBLE_DEVICES" in *,*) echo "R06 capture requires exactly one visible GPU" >&2; exit 2 ;; esac

for path in "$CONFIG" "$WORKLOAD" "$MANIFEST" "$R02_CONFIG" "$SOURCE_CONTRACT" "$SUBMISSION_RECEIPT"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked R06 capture input: $path" >&2; exit 2; }
done
test -x "$WORKLOAD" || { echo "R06 capture workload is not executable" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || {
  echo "R06 capture source commit changed" >&2
  exit 2
}
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || {
  echo "R06 capture requires a clean remote source tree" >&2
  exit 2
}
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256" || {
  echo "R06 capture config changed after release" >&2
  exit 2
}
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41
test "$(sha256sum "$R02_CONFIG" | awk '{print $1}')" = c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e
ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$CONFIG")
RELEASE_ADR=$(jq -er '.execution_release.decision_artifact' "$CONFIG")
case "$ACCEPTED_IMPLEMENTATION_COMMIT" in *[!0-9a-f]*|'') echo "invalid R06 accepted implementation commit" >&2; exit 2 ;; esac
test "${#ACCEPTED_IMPLEMENTATION_COMMIT}" = 40 || { echo "invalid R06 accepted implementation length" >&2; exit 2; }
case "$RELEASE_ADR" in
  docs/decisions/*-release-aegis-label-capture-canary.md) ;;
  *) echo "R06 capture release ADR path changed" >&2; exit 2 ;;
esac
test -f "$REMOTE_REPO/$RELEASE_ADR" && test ! -L "$REMOTE_REPO/$RELEASE_ADR" || {
  echo "R06 capture release ADR is missing or symlinked" >&2
  exit 2
}
RELEASE_ADR_SHA256=$(sha256sum "$REMOTE_REPO/$RELEASE_ADR" | awk '{print $1}')
test "$(git -C "$REMOTE_REPO" rev-list --parents -n 1 "$EXPECTED_GIT_COMMIT")" = \
  "$EXPECTED_GIT_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || {
  echo "R06 capture release is not the direct child of its implementation" >&2
  exit 2
}
observed_release_diff=$(git -C "$REMOTE_REPO" diff --name-only "$ACCEPTED_IMPLEMENTATION_COMMIT" "$EXPECTED_GIT_COMMIT" | LC_ALL=C sort)
expected_release_diff=$(printf '%s\n' \
  configs/experiments/r06_aegis_collision_conditioned.json \
  "$RELEASE_ADR" | LC_ALL=C sort)
test "$observed_release_diff" = "$expected_release_diff" || {
  echo "R06 capture release diff is not exact config plus release ADR" >&2
  exit 2
}

jq -e \
  --arg run "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" \
  --arg release_adr "$RELEASE_ADR" \
  --arg release_adr_sha "$RELEASE_ADR_SHA256" \
  --arg config_sha "$EXPECTED_CONFIG_SHA256" '
  .schema_version == "1.0"
  and .artifact_role == "r06_aegis_label_capture_source_contract"
  and .status == "sources_bound_before_held_submission"
  and .run_id == $run
  and .git_commit == $commit
  and .accepted_implementation_commit == $implementation
  and .release_decision_artifact == $release_adr
  and .release_decision_sha256 == $release_adr_sha
  and .release_only_parent_required == true
  and .git_dirty == false
  and .config_sha256 == $config_sha
  and .manifest_sha256 == "b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41"
  and .r02_config_sha256 == "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e"
  and .source_r02_pair_sha256 == "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593"
  and .checkpoint_sha256 == "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
  and .checkpoint_config_sha256 == "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a"
  and .normalization_asset_sha256 == "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
  and .robosuite_image_convention == "opengl"
  and .robosuite_version == "1.4.1"
  and .stage == "codex_label_capture"
  and .aegis_execution_allowed == false
  and .groundingdino_execution_allowed == false
  and .qp_execution_allowed == false
  and .probe_or_mlp_training_authorized == false
  ' "$SOURCE_CONTRACT" >/dev/null || { echo "R06 capture source contract changed" >&2; exit 2; }
SOURCE_CONTRACT_SHA256=$(sha256sum "$SOURCE_CONTRACT" | awk '{print $1}')

jq -e \
  --arg run "$RUN_ID" \
  --arg commit "$EXPECTED_GIT_COMMIT" \
  --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" \
  --arg release_adr "$RELEASE_ADR" \
  --arg release_adr_sha "$RELEASE_ADR_SHA256" \
  --arg config_sha "$EXPECTED_CONFIG_SHA256" \
  --arg source_sha "$SOURCE_CONTRACT_SHA256" \
  --arg array "$SLURM_ARRAY_JOB_ID" '
  .schema_version == "1.0"
  and .artifact_role == "r06_aegis_label_capture_submission"
  and .status == "held_task_validated_before_release"
  and .run_id == $run
  and .git_commit == $commit
  and .accepted_implementation_commit == $implementation
  and .release_decision_artifact == $release_adr
  and .release_decision_sha256 == $release_adr_sha
  and .release_only_parent_required == true
  and .config_sha256 == $config_sha
  and .source_contract_sha256 == $source_sha
  and .slurm_array_job_id == $array
  and .slurm_array_task_id == 0
  and .exact_task_id == ($array + "_0")
  and .source_host == "worker-1"
  and .released_at_receipt_time == false
  ' "$SUBMISSION_RECEIPT" >/dev/null || { echo "R06 capture submission receipt changed" >&2; exit 2; }

job_record=$(scontrol show job "${SLURM_ARRAY_JOB_ID}_0" -o)
for field in \
  "ArrayJobId=$SLURM_ARRAY_JOB_ID" \
  "ArrayTaskId=0" \
  JobState=RUNNING \
  Partition=main \
  Account=normal \
  QOS=normal \
  TimeLimit=00:30:00 \
  Requeue=0 \
  ReqNodeList=worker-1 \
  NumCPUs=8 \
  CPUs/Task=8; do
  case " $job_record " in *" $field "*) ;; *) echo "R06 capture job field changed: $field" >&2; exit 2 ;; esac
done

export R06_CAPTURE_SOURCE_CONTRACT=$SOURCE_CONTRACT
export R06_CAPTURE_SUBMISSION_RECEIPT=$SUBMISSION_RECEIPT
exec "$WORKLOAD"
