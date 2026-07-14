#!/usr/bin/env bash
#SBATCH --job-name=crfs-r03a-summary
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
: "${SLURM_JOB_ID:?R03A summary must execute inside Slurm}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${MANIFEST:?MANIFEST is required}"
: "${MANIFEST_SHA256:?MANIFEST_SHA256 is required}"
: "${CONFIG:?CONFIG is required}"
: "${CONFIG_SHA256:?CONFIG_SHA256 is required}"
: "${R02_RAW_ROOT:?R02_RAW_ROOT is required}"
: "${R03_SUMMARY:?R03_SUMMARY is required}"
: "${R03_SUMMARY_SHA256:?R03_SUMMARY_SHA256 is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${SOURCE_WORKER_1_SLURM_ARRAY_JOB_ID:?SOURCE_WORKER_1_SLURM_ARRAY_JOB_ID is required}"
: "${SOURCE_WORKER_2_SLURM_ARRAY_JOB_ID:?SOURCE_WORKER_2_SLURM_ARRAY_JOB_ID is required}"
: "${CHECKPOINT_ID:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
: "${CHECKPOINT_SHA256:?CHECKPOINT_SHA256 is required}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"

test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || {
  echo "R03A summary allocation checkout differs from reviewed commit" >&2
  exit 2
}
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || {
  echo "R03A summary allocation checkout is dirty" >&2
  exit 2
}
test -f "$MANIFEST"
test -f "$CONFIG"
test -d "$R02_RAW_ROOT"
test -f "$R03_SUMMARY"
test -d "$RESULTS_ROOT"
test ! -e "$OUTPUT" || {
  echo "immutable R03A population summary already exists: $OUTPUT" >&2
  exit 2
}
GROUPED_LAUNCH=$RESULTS_ROOT/grouped-launch.json
SUMMARY_SUBMISSION=$RESULTS_ROOT/summary-submission.json
test -f "$GROUPED_LAUNCH" && test -f "$SUMMARY_SUBMISSION" || {
  echo "R03A grouped launch or summary-submission evidence is missing" >&2
  exit 2
}
test -x "$REMOTE_REPO/scripts/hpc/prepare_jsonschema_overlay.sh" || {
  echo "missing JSON Schema overlay helper" >&2
  exit 2
}
JSONSCHEMA_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_jsonschema_overlay.sh)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$JSONSCHEMA_OVERLAY "$LIBERO_PYTHON" -c \
  'import jsonschema; jsonschema.Draft202012Validator.check_schema({"type": "object"})'
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA256" || {
  echo "R03A summary allocation manifest hash mismatch" >&2
  exit 2
}
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$CONFIG_SHA256" || {
  echo "R03A summary allocation config hash mismatch" >&2
  exit 2
}
test "$(sha256sum "$R03_SUMMARY" | awk '{print $1}')" = "$R03_SUMMARY_SHA256" || {
  echo "R03A summary allocation source R03 hash mismatch" >&2
  exit 2
}
GROUPED_LAUNCH_SHA256=$(sha256sum "$GROUPED_LAUNCH" | awk '{print $1}')
jq -e \
  --arg run_id "$RUN_ID" \
  --arg git_commit "$EXPECTED_GIT_COMMIT" \
  --arg manifest_sha256 "$MANIFEST_SHA256" \
  --arg config_sha256 "$CONFIG_SHA256" \
  --arg r03_summary_sha256 "$R03_SUMMARY_SHA256" \
  --arg checkpoint_sha256 "$CHECKPOINT_SHA256" \
  --arg grouped_launch_sha256 "$GROUPED_LAUNCH_SHA256" \
  --arg worker_1_job_id "$SOURCE_WORKER_1_SLURM_ARRAY_JOB_ID" \
  --arg worker_2_job_id "$SOURCE_WORKER_2_SLURM_ARRAY_JOB_ID" \
  --arg summary_job_id "$SLURM_JOB_ID" \
  --arg dependency "afterany:$SOURCE_WORKER_1_SLURM_ARRAY_JOB_ID:$SOURCE_WORKER_2_SLURM_ARRAY_JOB_ID" '
    .artifact_role == "r03a_population_summary_submission" and
    .status == "dependency_registered" and
    .scientific_claim_allowed == false and
    .run_id == $run_id and .git_commit == $git_commit and
    .manifest_sha256 == $manifest_sha256 and
    .config_sha256 == $config_sha256 and
    .r03_summary_sha256 == $r03_summary_sha256 and
    .checkpoint_sha256 == $checkpoint_sha256 and
    .grouped_launch_sha256 == $grouped_launch_sha256 and
    .source_slurm_array_job_ids_by_host == {
      "worker-1": $worker_1_job_id,
      "worker-2": $worker_2_job_id
    } and
    .slurm_summary_job_id == $summary_job_id and
    .dependency == $dependency
  ' "$SUMMARY_SUBMISSION" >/dev/null || {
  echo "R03A summary execution differs from the registered dependency" >&2
  exit 2
}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$JSONSCHEMA_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main
cd "$REMOTE_REPO"
"$LIBERO_PYTHON" -c 'import importlib.metadata; print("jsonschema", importlib.metadata.version("jsonschema"))'
"$LIBERO_PYTHON" main/summarize_r03a.py \
  --manifest "$MANIFEST" \
  --config "$CONFIG" \
  --r02-raw-root "$R02_RAW_ROOT" \
  --r03-summary "$R03_SUMMARY" \
  --results-root "$RESULTS_ROOT" \
  --output "$OUTPUT" \
  --run-id "$RUN_ID" \
  --checkpoint-id "$CHECKPOINT_ID" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  --source-worker-1-slurm-array-job-id "$SOURCE_WORKER_1_SLURM_ARRAY_JOB_ID" \
  --source-worker-2-slurm-array-job-id "$SOURCE_WORKER_2_SLURM_ARRAY_JOB_ID" \
  --summary-slurm-job-id "$SLURM_JOB_ID"
