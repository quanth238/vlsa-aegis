#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?generated-source validation must run inside Slurm}"
: "${SLURM_JOB_PARTITION:?generated-source validation must record its partition}"
: "${SOURCE_JOB_ID:?SOURCE_JOB_ID is required}"
: "${SOURCE_ARRAY_JOB_ID:?SOURCE_ARRAY_JOB_ID is required}"
: "${SOURCE_ARRAY_TASK_ID:?SOURCE_ARRAY_TASK_ID is required}"
: "${SOURCE_ARTIFACT:?SOURCE_ARTIFACT is required}"
: "${EXPECTED_CONFIG_SHA256:?EXPECTED_CONFIG_SHA256 is required}"
: "${GENERATION_REQUEST_ID:?GENERATION_REQUEST_ID is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"

test "$SLURM_JOB_PARTITION" = main || {
  echo "independent generated-source validation is frozen to the CPU main partition" >&2
  exit 2
}
test -x "$LIBERO_PYTHON" || {
  echo "missing validation Python: $LIBERO_PYTHON" >&2
  exit 2
}
test -f "$SOURCE_ARTIFACT" || {
  echo "missing generated-source canary artifact: $SOURCE_ARTIFACT" >&2
  exit 2
}

VALIDATOR=$REMOTE_REPO/main/validate_task0_single_obstacle_source.py
test -f "$VALIDATOR" || {
  echo "missing generated-source validator: $VALIDATOR" >&2
  exit 2
}
GIT_COMMIT=$(git -C "$REMOTE_REPO" rev-parse HEAD)
GIT_DIRTY=$(test -n "$(git -C "$REMOTE_REPO" status --porcelain)" && echo true || echo false)
test "$GIT_COMMIT" = "$EXPECTED_GIT_COMMIT" || {
  echo "validation source differs from the reviewed generated-source commit" >&2
  exit 2
}
test "$GIT_DIRTY" = false || {
  echo "generated-source validation requires a clean remote worktree" >&2
  exit 2
}

export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero
REPORT=${SOURCE_ARTIFACT%.json}.cpu-validation.json
TEMP=$REPORT.$SLURM_JOB_ID.tmp
trap 'rm -f "$TEMP"' EXIT INT TERM

set +e
"$LIBERO_PYTHON" "$VALIDATOR" \
  --artifact "$SOURCE_ARTIFACT" \
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256" \
  --expected-source-slurm-job-id "$SOURCE_JOB_ID" \
  --expected-source-slurm-array-job-id "$SOURCE_ARRAY_JOB_ID" \
  --expected-source-slurm-array-task-id "$SOURCE_ARRAY_TASK_ID" \
  --expected-generation-request-id "$GENERATION_REQUEST_ID" >"$TEMP"
validation_status=$?
set -e
mv "$TEMP" "$REPORT"
trap - EXIT INT TERM

echo "source_artifact=$SOURCE_ARTIFACT"
echo "source_artifact_sha256=$(sha256sum "$SOURCE_ARTIFACT" | awk '{print $1}')"
echo "independent_validation_report=$REPORT"
echo "independent_validation_report_sha256=$(sha256sum "$REPORT" | awk '{print $1}')"
echo "validation_git_commit=$GIT_COMMIT"
echo "validation_git_dirty=$GIT_DIRTY"
exit "$validation_status"
