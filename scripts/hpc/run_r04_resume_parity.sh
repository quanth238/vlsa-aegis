#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?run_r04_resume_parity.sh must execute inside a Slurm allocation}"
: "${SLURM_ARRAY_JOB_ID:?R04B smoke must expose its exact Slurm array job id}"
: "${SLURM_ARRAY_TASK_ID:?R04B smoke must be a one-element Slurm array}"
: "${SLURM_JOB_PARTITION:?R04B smoke must record its Slurm partition}"
: "${CUDA_VISIBLE_DEVICES:?R04B smoke must run inside a GPU allocation}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${OPENPI_PYTHON:=/mnt/data/quanth/venvs/openpi/bin/python}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"
: "${OPENPI_DATA_HOME:=/mnt/data/quanth/cache/openpi}"
: "${CHECKPOINT_DIR:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${RUN_ID:?RUN_ID must be an explicit immutable R04B smoke identifier}"
: "${MANIFEST:?MANIFEST is required; R04B has no implicit manifest}"
: "${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is required; R04B has no implicit config}"
: "${PARITY_ARTIFACT:?PARITY_ARTIFACT is required as passed sampler evidence}"
: "${R04A_ARTIFACT:?R04A_ARTIFACT is required as passed apparatus evidence}"
: "${R04A_VALIDATION:?R04A_VALIDATION is required as passed independent validation evidence}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT must bind the reviewed source commit}"
: "${CASE_INDEX:=$SLURM_ARRAY_TASK_ID}"

# Select and verify the atomic failure-artifact writer before installing the
# cleanup trap. The LIBERO client environment is preferred, but a missing or
# broken client interpreter must not suppress the launch-failure artifact.
REQUESTED_FAILURE_PYTHON=${FAILURE_PYTHON:-}
SYSTEM_PYTHON=$(command -v python3 2>/dev/null || true)
FAILURE_PYTHON=
for candidate in \
  "$REQUESTED_FAILURE_PYTHON" \
  "$LIBERO_PYTHON" \
  "$OPENPI_PYTHON" \
  "$SYSTEM_PYTHON"; do
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

EXPECTED_MANIFEST_SHA256=3180836991e64b774670218f5fe4edaa10775fbb33fcdb330cc781b8b8937dad
EXPECTED_CONFIG_SHA256=a0edb7edac86d2abd887218002d14e28e072cd472d631797d18d540f5140be33
EXPECTED_R00_SUMMARY_SHA256=90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f
EXPECTED_R03_SUMMARY_SHA256=dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e
EXPECTED_PARITY_SHA256=26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a
EXPECTED_R04A_ARTIFACT_SHA256=b820793ec42a5228c858e297d13806d8ae7f02e5cc7a765769316473d795f285
EXPECTED_R04A_VALIDATION_SHA256=cf821650c48d30f6ebf2bbf7afe06b21938e0b137aa5acf1921f571e409e05ea
EXPECTED_CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
RUNNER=$REMOTE_REPO/main/run_r04_resume_parity.py
MODEL=$CHECKPOINT_DIR/model.safetensors
R00_SUMMARY=$REMOTE_REPO/evidence/r00/r00-summary.json
R03_SUMMARY=$REMOTE_REPO/evidence/r03/r03-summary.json

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2; exit 2 ;;
esac
case "$CASE_INDEX" in
  *[!0-9]*|'') echo "CASE_INDEX must be a nonnegative integer" >&2; exit 2 ;;
esac
test "$CASE_INDEX" -eq 0 || { echo "R04B is frozen to apparatus case index 0" >&2; exit 2; }
test "$CASE_INDEX" = "$SLURM_ARRAY_TASK_ID" || { echo "CASE_INDEX must equal SLURM_ARRAY_TASK_ID" >&2; exit 2; }
test "$CUDA_VISIBLE_DEVICES" != NoDevFiles || { echo "R04B has no visible allocated GPU" >&2; exit 2; }
if [ -z "${PORT:-}" ]; then
  PORT=$((20000 + SLURM_JOB_ID % 30000))
fi
case "$PORT" in
  *[!0-9]*|'') echo "PORT must be an integer" >&2; exit 2 ;;
esac
test "$PORT" -ge 1024 && test "$PORT" -le 65535 || { echo "PORT must be between 1024 and 65535" >&2; exit 2; }

export RUN_ID CASE_INDEX
FAILURE_STAGE=allocation_contract
CASE_ID=unresolved
FAILURE_DIR=$EXPERIMENT_ROOT/$RUN_ID/failures
mkdir -p "$FAILURE_DIR"
FAILURE=$FAILURE_DIR/case-index-$CASE_INDEX.json
SERVER_LOG=$FAILURE_DIR/policy-server-case-$CASE_INDEX.log
CLIENT_LOG=$FAILURE_DIR/r04-resume-client-case-$CASE_INDEX.log
VALIDATION_LOG=$FAILURE_DIR/r04-resume-validation-case-$CASE_INDEX.json
SERVER_PID=
MANIFEST_SHA256=unavailable
CONFIG_SHA256=unavailable
CHECKPOINT_SHA256=unavailable
R00_SUMMARY_SHA256=unavailable
R03_SUMMARY_SHA256=unavailable
PARITY_SHA256=unavailable
R04A_ARTIFACT_SHA256=unavailable
R04A_VALIDATION_SHA256=unavailable
GIT_COMMIT=unavailable
GIT_DIRTY=unavailable
export CASE_ID

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$status" -ne 0 ]; then
    "$FAILURE_PYTHON" - \
      "$FAILURE" "$status" "$FAILURE_STAGE" \
      "$MANIFEST_SHA256" "$CONFIG_SHA256" "$CHECKPOINT_SHA256" \
      "$R00_SUMMARY_SHA256" "$R03_SUMMARY_SHA256" "$PARITY_SHA256" \
      "$R04A_ARTIFACT_SHA256" "$R04A_VALIDATION_SHA256" \
      "$GIT_COMMIT" "$GIT_DIRTY" "$SERVER_LOG" "$CLIENT_LOG" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
value = {
    "schema_version": "1.0",
    "status": "failed",
    "artifact_role": "r04b_resume_parity_launch_failure",
    "scientific_claim_allowed": False,
    "probe_training_authorized": False,
    "planned_simulator_setup": "one_reset_plus_20_dummy_settle_control_steps",
    "simulator_setup_completion": "unknown_or_not_completed_due_failure",
    "policy_generated_action_steps_executed": 0,
    "efficacy_rollouts_executed": 0,
    "simulator_efficacy_evaluated": False,
    "failure_writer_python": sys.executable,
    "exit_code": int(sys.argv[2]),
    "stage": sys.argv[3],
    "run_id": os.environ["RUN_ID"],
    "case_index": int(os.environ["CASE_INDEX"]),
    "case_id": os.environ.get("CASE_ID", "unresolved"),
    "provenance": {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
    },
    "manifest_sha256": sys.argv[4],
    "expected_manifest_sha256": "3180836991e64b774670218f5fe4edaa10775fbb33fcdb330cc781b8b8937dad",
    "config_sha256": sys.argv[5],
    "expected_config_sha256": "a0edb7edac86d2abd887218002d14e28e072cd472d631797d18d540f5140be33",
    "checkpoint_sha256": sys.argv[6],
    "expected_checkpoint_sha256": "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
    "r00_summary_sha256": sys.argv[7],
    "expected_r00_summary_sha256": "90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f",
    "r03_summary_sha256": sys.argv[8],
    "expected_r03_summary_sha256": "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e",
    "sampler_parity_sha256": sys.argv[9],
    "expected_sampler_parity_sha256": "26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a",
    "r04a_artifact_sha256": sys.argv[10],
    "expected_r04a_artifact_sha256": "b820793ec42a5228c858e297d13806d8ae7f02e5cc7a765769316473d795f285",
    "r04a_validation_sha256": sys.argv[11],
    "expected_r04a_validation_sha256": "cf821650c48d30f6ebf2bbf7afe06b21938e0b137aa5acf1921f571e409e05ea",
    "git_commit": sys.argv[12],
    "reviewed_git_commit": os.environ.get("EXPECTED_GIT_COMMIT"),
    "git_dirty": sys.argv[13],
    "policy_server_log": sys.argv[14],
    "client_log": sys.argv[15],
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
for path in \
  "$RUNNER" \
  "$MANIFEST" \
  "$EXPERIMENT_CONFIG" \
  "$R00_SUMMARY" \
  "$R03_SUMMARY" \
  "$PARITY_ARTIFACT" \
  "$R04A_ARTIFACT" \
  "$R04A_VALIDATION" \
  "$MODEL"; do
  test -f "$path" || { echo "missing frozen R04B input: $path" >&2; exit 2; }
done
test -x "$OPENPI_PYTHON" || { echo "missing OpenPI Python: $OPENPI_PYTHON" >&2; exit 2; }
test -x "$LIBERO_PYTHON" || { echo "missing LIBERO Python: $LIBERO_PYTHON" >&2; exit 2; }

MANIFEST_SHA256=$(sha256sum "$MANIFEST" | awk '{print $1}')
CONFIG_SHA256=$(sha256sum "$EXPERIMENT_CONFIG" | awk '{print $1}')
CHECKPOINT_SHA256=$(sha256sum "$MODEL" | awk '{print $1}')
R00_SUMMARY_SHA256=$(sha256sum "$R00_SUMMARY" | awk '{print $1}')
R03_SUMMARY_SHA256=$(sha256sum "$R03_SUMMARY" | awk '{print $1}')
PARITY_SHA256=$(sha256sum "$PARITY_ARTIFACT" | awk '{print $1}')
R04A_ARTIFACT_SHA256=$(sha256sum "$R04A_ARTIFACT" | awk '{print $1}')
R04A_VALIDATION_SHA256=$(sha256sum "$R04A_VALIDATION" | awk '{print $1}')
GIT_COMMIT=$(git -C "$REMOTE_REPO" rev-parse HEAD)
GIT_DIRTY=$(test -n "$(git -C "$REMOTE_REPO" status --porcelain)" && echo true || echo false)

test "$MANIFEST_SHA256" = "$EXPECTED_MANIFEST_SHA256" || { echo "R04B manifest hash mismatch" >&2; exit 2; }
test "$CONFIG_SHA256" = "$EXPECTED_CONFIG_SHA256" || { echo "R04B config hash mismatch" >&2; exit 2; }
test "$CHECKPOINT_SHA256" = "$EXPECTED_CHECKPOINT_SHA256" || { echo "R04B checkpoint hash mismatch" >&2; exit 2; }
test "$R00_SUMMARY_SHA256" = "$EXPECTED_R00_SUMMARY_SHA256" || { echo "R00 summary hash mismatch" >&2; exit 2; }
test "$R03_SUMMARY_SHA256" = "$EXPECTED_R03_SUMMARY_SHA256" || { echo "R03 summary hash mismatch" >&2; exit 2; }
test "$PARITY_SHA256" = "$EXPECTED_PARITY_SHA256" || { echo "sampler parity hash mismatch" >&2; exit 2; }
test "$R04A_ARTIFACT_SHA256" = "$EXPECTED_R04A_ARTIFACT_SHA256" || { echo "R04A artifact hash mismatch" >&2; exit 2; }
test "$R04A_VALIDATION_SHA256" = "$EXPECTED_R04A_VALIDATION_SHA256" || { echo "R04A validation hash mismatch" >&2; exit 2; }
test "$GIT_COMMIT" = "$EXPECTED_GIT_COMMIT" || { echo "source commit differs from reviewed R04B submission" >&2; exit 2; }
test "$GIT_DIRTY" = false || { echo "R04B requires a clean remote source tree" >&2; exit 2; }

CASE_ID=$("$LIBERO_PYTHON" - "$MANIFEST" "$CASE_INDEX" <<'PY'
import json
import pathlib
import sys

records = [
    json.loads(line)
    for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
index = int(sys.argv[2])
if len(records) != 120:
    raise SystemExit(f"R04B source manifest must contain 120 rows, found {len(records)}")
if index != 0:
    raise SystemExit("R04B apparatus parity is frozen to case index 0")
case = records[index]
if case.get("case_id") != "crfs-93365b8b851365f2":
    raise SystemExit("R04B fixed case identity changed")
if case.get("group_id") != "safelibero_spatial:II:0:2":
    raise SystemExit("R04B fixed source group changed")
print(case["case_id"])
PY
)
export CASE_ID
CASE_DIR=$EXPERIMENT_ROOT/$RUN_ID/$CASE_ID
mkdir -p "$CASE_DIR"
FAILURE=$CASE_DIR/launch-failure.json
SERVER_LOG=$CASE_DIR/policy-server.log
CLIENT_LOG=$CASE_DIR/r04-resume-client.log
VALIDATION_LOG=$CASE_DIR/r04-resume-validation.json
RESULT=$CASE_DIR/r04-resume-parity.json

export OPENPI_DATA_HOME
export HF_HOME=/mnt/data/quanth/cache/huggingface
export XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
export PIP_CACHE_DIR=/mnt/data/quanth/cache/pip
export UV_CACHE_DIR=/mnt/data/quanth/cache/uv
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH=$CASE_DIR/libero-config
export PYTHONUNBUFFERED=1
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

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "partition=$SLURM_JOB_PARTITION"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "slurm_mem_per_node=${SLURM_MEM_PER_NODE:-unset}"
echo "repo=$REMOTE_REPO"
echo "commit=$GIT_COMMIT"
echo "dirty=$GIT_DIRTY"
echo "run_id=$RUN_ID"
echo "case_index=$CASE_INDEX"
echo "case_id=$CASE_ID"
echo "manifest_sha256=$MANIFEST_SHA256"
echo "config_sha256=$CONFIG_SHA256"
echo "r00_summary_sha256=$R00_SUMMARY_SHA256"
echo "r03_summary_sha256=$R03_SUMMARY_SHA256"
echo "sampler_parity_sha256=$PARITY_SHA256"
echo "r04a_artifact_sha256=$R04A_ARTIFACT_SHA256"
echo "r04a_validation_sha256=$R04A_VALIDATION_SHA256"
echo "checkpoint_sha256=$CHECKPOINT_SHA256"
"$OPENPI_PYTHON" -V
"$OPENPI_PYTHON" -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'

FAILURE_STAGE=policy_server_startup
TRANSFORMERS_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh)
export PYTHONPATH=$TRANSFORMERS_OVERLAY:$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src

# Exercise the resume-control validation against the actual allocation
# PyTorch/OpenPI dependency stack before loading the checkpoint.  Local
# dependency-free checks cannot cover these imports.
FAILURE_STAGE=dependency_backed_sampler_policy_contract
cd "$REMOTE_REPO"
"$OPENPI_PYTHON" -m unittest discover \
  -s tests \
  -p 'test_r04b_sampler_policy_contract.py' \
  -v

FAILURE_STAGE=policy_server_startup
(
  cd "$REMOTE_REPO/openpi"
  "$OPENPI_PYTHON" scripts/serve_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$CHECKPOINT_DIR"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 200 "$SERVER_LOG" >&2 || true
    exit 3
  fi
  if grep -q "server listening" "$SERVER_LOG"; then
    ready=1
    break
  fi
  sleep 1
done
test "$ready" = 1 || { tail -n 200 "$SERVER_LOG" >&2 || true; exit 4; }

FAILURE_STAGE=r04b_resume_parity_runner
export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/packages/openpi-client/src
cd "$REMOTE_REPO"
"$LIBERO_PYTHON" main/run_r04_resume_parity.py \
  --manifest "$MANIFEST" \
  --config "$EXPERIMENT_CONFIG" \
  --output-root "$EXPERIMENT_ROOT" \
  --run-id "$RUN_ID" \
  --case-index "$CASE_INDEX" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --checkpoint-id "$CHECKPOINT_DIR" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" >"$CLIENT_LOG" 2>&1

test -f "$RESULT" || { echo "R04B runner returned without final artifact: $RESULT" >&2; exit 5; }

# Validate the final artifact inside the exact GPU array task that generated it.
# The apparatus performs one reset plus 20 dummy settle-control steps, then
# executes no policy-generated action and no efficacy rollout.
FAILURE_STAGE=exact_job_artifact_validation
"$LIBERO_PYTHON" main/run_r04_resume_parity.py \
  --validate-artifact "$RESULT" \
  --expected-config-file-sha256 "$CONFIG_SHA256" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --expected-r04a-validation-summary-sha256 "$R04A_VALIDATION_SHA256" \
  --expected-source-slurm-job-id "$SLURM_JOB_ID" \
  --expected-source-slurm-array-job-id "$SLURM_ARRAY_JOB_ID" \
  --expected-source-slurm-array-task-id "$SLURM_ARRAY_TASK_ID" >"$VALIDATION_LOG"
"$LIBERO_PYTHON" - "$RESULT" "$RUN_ID" "$CASE_ID" <<'PY'
import json
import os
import pathlib
import sys

from crfs_oracle.r04_resume import validate_r04_resume_result

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
errors = validate_r04_resume_result(value)
if errors:
    raise SystemExit("R04B artifact validation failed: " + "; ".join(errors))
if value.get("run_id") != sys.argv[2] or value.get("case_id") != sys.argv[3]:
    raise SystemExit("R04B artifact identity differs from the exact submitted case")
provenance = value.get("provenance")
if not isinstance(provenance, dict):
    raise SystemExit("R04B artifact has no provenance object")
expected = {
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
    "slurm_array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
    "partition": os.environ["SLURM_JOB_PARTITION"],
    "device": os.environ["CUDA_VISIBLE_DEVICES"],
}
mismatches = {
    key: {"artifact": provenance.get(key), "allocation": expected_value}
    for key, expected_value in expected.items()
    if provenance.get(key) != expected_value
}
if mismatches:
    raise SystemExit(f"R04B artifact provenance differs from exact job: {mismatches}")
print("validation_error_count=0")
print(f"validated_exact_job_id={expected['slurm_job_id']}")
print(f"validated_exact_array_task={expected['slurm_array_job_id']}_{expected['slurm_array_task_id']}")
PY

echo "result=$RESULT"
echo "result_sha256=$(sha256sum "$RESULT" | awk '{print $1}')"
echo "validation_report=$VALIDATION_LOG"
FAILURE_STAGE=complete
