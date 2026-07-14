#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?run_generated_source_pilot.sh must execute inside a Slurm allocation}"
: "${SLURM_ARRAY_JOB_ID:?generated-source pilot must expose its exact Slurm array job id}"
: "${SLURM_ARRAY_TASK_ID:?generated-source pilot must expose its exact serialized-array task id}"
: "${SLURM_JOB_PARTITION:?generated-source pilot must record its Slurm partition}"
: "${CUDA_VISIBLE_DEVICES:?generated-source pilot requires a rendering-only MIG allocation}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"
: "${OUTPUT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${RUN_ID:?RUN_ID must be an explicit immutable generated-source pilot identifier}"
: "${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is required; the pilot has no implicit config}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT must bind the reviewed source commit}"
: "${GROUP_INDEX:=$SLURM_ARRAY_TASK_ID}"

EXPECTED_CONFIG_SHA256=332c90fdb9e4560ab522e6333fbe5f0846d1c7caca9190f1730436036856f22a
GENERATOR=$REMOTE_REPO/main/generate_task0_single_obstacle_source.py
VALIDATOR=$REMOTE_REPO/main/validate_task0_single_obstacle_source.py

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
    exit 2
    ;;
esac
case "$GROUP_INDEX" in
  *[!0-9]*|'') echo "GROUP_INDEX must be a nonnegative integer" >&2; exit 2 ;;
esac
test "$GROUP_INDEX" -ge 0 && test "$GROUP_INDEX" -le 9 || {
  echo "retired source-integrity pilot is frozen to group indices 0 through 9" >&2
  exit 2
}
test "$GROUP_INDEX" = "$SLURM_ARRAY_TASK_ID" || {
  echo "GROUP_INDEX must equal SLURM_ARRAY_TASK_ID" >&2
  exit 2
}
test "$SLURM_JOB_PARTITION" = mig || {
  echo "generated-source rendering pilot is frozen to the mig partition" >&2
  exit 2
}
test "$CUDA_VISIBLE_DEVICES" != NoDevFiles || {
  echo "generated-source pilot has no visible allocated rendering device" >&2
  exit 2
}
GENERATION_REQUEST_ID=$(printf 'greq-task0-single-obstacle-v1-%04d' "$GROUP_INDEX")

# Select an atomic failure writer before installing the trap.  This records a
# failed launch even when the requested SafeLIBERO interpreter is unavailable.
REQUESTED_FAILURE_PYTHON=${FAILURE_PYTHON:-}
SYSTEM_PYTHON=$(command -v python3 2>/dev/null || true)
FAILURE_PYTHON=
for candidate in "$REQUESTED_FAILURE_PYTHON" "$LIBERO_PYTHON" "$SYSTEM_PYTHON"; do
  test -n "$candidate" && test -x "$candidate" || continue
  if "$candidate" -c 'import json, os, pathlib, tempfile' >/dev/null 2>&1; then
    FAILURE_PYTHON=$candidate
    break
  fi
done
test -n "$FAILURE_PYTHON" || {
  echo "no verified Python interpreter is available for atomic failure artifacts" >&2
  exit 2
}

export RUN_ID GROUP_INDEX GENERATION_REQUEST_ID
FAILURE_STAGE=allocation_contract
RUN_DIR=$OUTPUT_ROOT/$RUN_ID
FAILURE_DIR=$RUN_DIR/failures
mkdir -p "$FAILURE_DIR"
FAILURE=$FAILURE_DIR/group-index-$GROUP_INDEX.json
GENERATOR_LOG=$FAILURE_DIR/generator-group-index-$GROUP_INDEX.log
CONFIG_SHA256=unavailable
GENERATOR_SHA256=unavailable
VALIDATOR_SHA256=unavailable
ARTIFACT_SHA256=unavailable
VALIDATION_REPORT_SHA256=unavailable
GIT_COMMIT=unavailable
GIT_DIRTY=unavailable
GENERATION_ATTEMPTED=false

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ]; then
    "$FAILURE_PYTHON" - \
      "$FAILURE" "$status" "$FAILURE_STAGE" \
      "$CONFIG_SHA256" "$GENERATOR_SHA256" "$VALIDATOR_SHA256" \
      "$ARTIFACT_SHA256" "$VALIDATION_REPORT_SHA256" \
      "$GIT_COMMIT" "$GIT_DIRTY" "$GENERATOR_LOG" \
      "$GENERATION_ATTEMPTED" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
value = {
    "schema_version": "1.0",
    "status": "failed",
    "artifact_role": "generated_source_pilot_launch_failure",
    "retired_pilot": True,
    "pilot_role": "retired_source_integrity_only",
    "scientific_claim_allowed": False,
    "probe_training_authorized": False,
    "allocation_purpose": "mujoco_offscreen_osmesa_render_and_observation_hashing_only",
    "policy_server_started": False,
    "policy_inference_calls": 0,
    "model_checkpoint_loaded": False,
    "training_executed": False,
    "statistics_aggregation_executed": False,
    "generation_attempted": sys.argv[12] == "true",
    "exit_code": int(sys.argv[2]),
    "stage": sys.argv[3],
    "run_id": os.environ["RUN_ID"],
    "group_index": int(os.environ["GROUP_INDEX"]),
    "generation_request_id": os.environ["GENERATION_REQUEST_ID"],
    "provenance": {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "allocation_visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "render_backend": os.environ.get("MUJOCO_GL", "osmesa"),
        "mujoco_gl": os.environ.get("MUJOCO_GL", "osmesa"),
    },
    "config_sha256": sys.argv[4],
    "expected_config_sha256": "332c90fdb9e4560ab522e6333fbe5f0846d1c7caca9190f1730436036856f22a",
    "generator_sha256": sys.argv[5],
    "validator_sha256": sys.argv[6],
    "artifact_sha256": sys.argv[7],
    "validation_report_sha256": sys.argv[8],
    "git_commit": sys.argv[9],
    "reviewed_git_commit": os.environ.get("EXPECTED_GIT_COMMIT"),
    "git_dirty": sys.argv[10],
    "generator_log": sys.argv[11],
    "failure_writer_python": sys.executable,
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, path)
PY
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

FAILURE_STAGE=source_binding
for path in "$GENERATOR" "$VALIDATOR" "$EXPERIMENT_CONFIG"; do
  test -f "$path" || {
    echo "missing frozen generated-source pilot input: $path" >&2
    exit 2
  }
done
test -x "$LIBERO_PYTHON" || {
  echo "missing SafeLIBERO allocation Python: $LIBERO_PYTHON" >&2
  exit 2
}

CONFIG_SHA256=$(sha256sum "$EXPERIMENT_CONFIG" | awk '{print $1}')
GENERATOR_SHA256=$(sha256sum "$GENERATOR" | awk '{print $1}')
VALIDATOR_SHA256=$(sha256sum "$VALIDATOR" | awk '{print $1}')
GIT_COMMIT=$(git -C "$REMOTE_REPO" rev-parse HEAD)
GIT_DIRTY=$(test -n "$(git -C "$REMOTE_REPO" status --porcelain)" && echo true || echo false)
test "$CONFIG_SHA256" = "$EXPECTED_CONFIG_SHA256" || {
  echo "generated-source pilot config hash mismatch" >&2
  exit 2
}
test "$GIT_COMMIT" = "$EXPECTED_GIT_COMMIT" || {
  echo "source commit differs from reviewed generated-source pilot submission" >&2
  exit 2
}
test "$GIT_DIRTY" = false || {
  echo "generated-source pilot requires a clean remote source tree" >&2
  exit 2
}

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH=$RUN_DIR/libero-config
export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero
mkdir -p "$LIBERO_CONFIG_PATH"

SAFELIBERO_ROOT=$REMOTE_REPO/safelibero/libero/libero
"$LIBERO_PYTHON" - "$LIBERO_CONFIG_PATH/config.yaml" "$SAFELIBERO_ROOT" <<'PY'
import os
import pathlib
import sys
import tempfile

destination = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2]).resolve()
lines = {
    "benchmark_root": root,
    "bddl_files": root / "bddl_files",
    "init_states": root / "init_files",
    "datasets": root.parent / "datasets",
    "assets": root / "assets",
}
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as handle:
    for key, value in lines.items():
        handle.write(f"{key}: {value}\n")
    temporary = handle.name
os.replace(temporary, destination)
PY

FAILURE_STAGE=config_schedule_binding
"$LIBERO_PYTHON" - \
  "$EXPERIMENT_CONFIG" "$GENERATION_REQUEST_ID" "$GROUP_INDEX" <<'PY'
import json
import pathlib
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if config.get("ready_to_run") is not True:
    raise SystemExit("generated-source pilot config is not ready_to_run")
if config.get("blocked_on") != []:
    raise SystemExit("generated-source pilot config still has blocked dependencies")
if config.get("pilot_only") is not True:
    raise SystemExit("generated-source pilot must remain pilot_only")
if config.get("ready_for_training") is not False or config.get("ready_for_claims") is not False:
    raise SystemExit("generated-source pilot cannot authorize training or claims")
groups = config.get("source_groups") if isinstance(config, dict) else None
if not isinstance(groups, list) or len(groups) != 10:
    raise SystemExit("generated-source pilot config must freeze exactly 10 source groups")
index = int(sys.argv[3])
selected = groups[index]
if not isinstance(selected, dict) or selected.get("generation_request_id") != sys.argv[2]:
    raise SystemExit(f"generated-source group index {index} generation request changed")
PY

SOURCE_DIR=$RUN_DIR/$GENERATION_REQUEST_ID
RESULT=$SOURCE_DIR/source-bundle.json
VALIDATION_LOG=$SOURCE_DIR/source-bundle.validation.json
mkdir -p "$SOURCE_DIR"
GENERATOR_LOG=$SOURCE_DIR/generator.log

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "partition=$SLURM_JOB_PARTITION"
echo "allocation_visible_gpu=$CUDA_VISIBLE_DEVICES"
echo "render_device=cpu_osmesa"
echo "mujoco_gl=$MUJOCO_GL"
echo "allocation_purpose=mujoco_offscreen_osmesa_render_and_observation_hashing_only"
echo "policy_server_started=false"
echo "policy_inference_calls=0"
echo "model_checkpoint_loaded=false"
echo "training_executed=false"
echo "statistics_aggregation_executed=false"
echo "repo=$REMOTE_REPO"
echo "commit=$GIT_COMMIT"
echo "dirty=$GIT_DIRTY"
echo "run_id=$RUN_ID"
echo "group_index=$GROUP_INDEX"
echo "generation_request_id=$GENERATION_REQUEST_ID"
echo "config_sha256=$CONFIG_SHA256"
echo "generator_sha256=$GENERATOR_SHA256"
echo "validator_sha256=$VALIDATOR_SHA256"
"$LIBERO_PYTHON" -V

FAILURE_STAGE=safelibero_render_dependency_contract
"$LIBERO_PYTHON" -c 'import libero, numpy; from libero.libero.envs import OffScreenRenderEnv; print("offscreen_render_env_import=ok")'

if [ -f "$RESULT" ]; then
  echo "resume_candidate=$RESULT"
else
  FAILURE_STAGE=generated_source_runner
  GENERATION_ATTEMPTED=true
  "$LIBERO_PYTHON" "$GENERATOR" \
    --config "$EXPERIMENT_CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$RUN_ID" \
    --group-index "$GROUP_INDEX" \
    --expected-config-sha256 "$CONFIG_SHA256" >"$GENERATOR_LOG" 2>&1
fi

test -f "$RESULT" || {
  echo "generated-source runner returned without final artifact: $RESULT" >&2
  exit 5
}

# Resume is allowed only when the exact validator accepts the final artifact
# for this same Slurm task.  The validator itself never instantiates MuJoCo.
FAILURE_STAGE=exact_job_artifact_validation
VALIDATION_TEMP=$SOURCE_DIR/.source-bundle.validation.$SLURM_JOB_ID.tmp
"$LIBERO_PYTHON" "$VALIDATOR" \
  --artifact "$RESULT" \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --expected-source-slurm-job-id "$SLURM_JOB_ID" \
  --expected-source-slurm-array-job-id "$SLURM_ARRAY_JOB_ID" \
  --expected-source-slurm-array-task-id "$SLURM_ARRAY_TASK_ID" \
  --expected-generation-request-id "$GENERATION_REQUEST_ID" >"$VALIDATION_TEMP"
mv "$VALIDATION_TEMP" "$VALIDATION_LOG"

ARTIFACT_SHA256=$(sha256sum "$RESULT" | awk '{print $1}')
VALIDATION_REPORT_SHA256=$(sha256sum "$VALIDATION_LOG" | awk '{print $1}')
echo "artifact=$RESULT"
echo "artifact_sha256=$ARTIFACT_SHA256"
echo "validation_report=$VALIDATION_LOG"
echo "validation_report_sha256=$VALIDATION_REPORT_SHA256"
echo "exact_validator_sha256=$VALIDATOR_SHA256"
FAILURE_STAGE=complete
