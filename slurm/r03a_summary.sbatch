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
: "${SOURCE_SLURM_ARRAY_JOB_ID:?SOURCE_SLURM_ARRAY_JOB_ID is required}"
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
"$LIBERO_PYTHON" -c 'import jsonschema' || {
  echo "R03A production summary requires jsonschema in the allocation venv" >&2
  exit 2
}
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

export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main
cd "$REMOTE_REPO"
"$LIBERO_PYTHON" -c 'import jsonschema; print("jsonschema", jsonschema.__version__)'
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
  --source-slurm-array-job-id "$SOURCE_SLURM_ARRAY_JOB_ID"
