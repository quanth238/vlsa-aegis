#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?R05A canary must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?R05A canary must expose its array job id}"
: "${SLURM_ARRAY_TASK_ID:?R05A canary must be a one-row array}"
: "${SLURM_JOB_PARTITION:?R05A canary must record its partition}"
: "${CUDA_VISIBLE_DEVICES:?R05A canary requires an allocated GPU}"
: "${RUN_ID:?RUN_ID must be an explicit immutable canary id}"
: "${MANIFEST:?MANIFEST is required}"
: "${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is required}"
: "${R02_RAW_ROOT:?R02_RAW_ROOT is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${OPENPI_PYTHON:=/mnt/data/quanth/venvs/openpi/bin/python}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"
: "${OPENPI_DATA_HOME:=/mnt/data/quanth/cache/openpi}"
: "${CHECKPOINT_DIR:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"

EXPECTED_MANIFEST_SHA256=bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633
EXPECTED_CONFIG_SHA256=c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb
EXPECTED_SCHEMA_SHA256=e1681f865f81f2986945d10fb14073fe4cd9e78cbc5026f6749ab321b9cfb7a7
EXPECTED_DECISION_SHA256=d2a00b1b049e92bb1ec8f11d60fa447e5e8bd5109cb8cf3f3fbb08f89498656f
EXPECTED_R02_SHA256=055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593
EXPECTED_CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
CASE_ID=crfs-1069f29a8d76463a
CASE_INDEX=$SLURM_ARRAY_TASK_ID
MODEL=$CHECKPOINT_DIR/model.safetensors
SOURCE_R02=$R02_RAW_ROOT/$CASE_ID/r02-paired.json
SCHEMA=$REMOTE_REPO/schemas/r05a-inverse-flow-canary.schema.json
DECISION=$REMOTE_REPO/docs/decisions/0028-pivot-to-inverse-flow-transport.md

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'') echo "unsafe R05A RUN_ID" >&2; exit 2 ;;
esac
test "$SLURM_JOB_PARTITION" = main || { echo "R05A canary is frozen to main" >&2; exit 2; }
test "$CASE_INDEX" = 0 || { echo "R05A canary is fixed to array row zero" >&2; exit 2; }
test "$CUDA_VISIBLE_DEVICES" != NoDevFiles || { echo "R05A canary has no visible GPU" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "R05A canary must remain on worker-1" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 8 || { echo "R05A canary requires eight CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" -ge 65536 || { echo "R05A canary requires the registered 64 GiB allocation" >&2; exit 2; }

if [ -z "${PORT:-}" ]; then
  PORT=$((20000 + SLURM_JOB_ID % 30000))
fi
case "$PORT" in
  *[!0-9]*|'') echo "PORT must be an integer" >&2; exit 2 ;;
esac
test "$PORT" -ge 1024 && test "$PORT" -le 65535 || { echo "PORT outside 1024..65535" >&2; exit 2; }

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
CASE_DIR=$RUN_ROOT/$CASE_ID
FAILURE_DIR=$RUN_ROOT/failures
test -f "$RUN_ROOT/submission.json" || {
  echo "R05A allocation requires the atomic held-job submission receipt" >&2
  exit 2
}
test ! -e "$CASE_DIR" && test ! -e "$FAILURE_DIR" || {
  echo "immutable R05A run id already contains allocation output" >&2
  exit 2
}
mkdir "$CASE_DIR" "$FAILURE_DIR"
FAILURE=$FAILURE_DIR/case-index-0.json
SERVER_LOG=$CASE_DIR/policy-server.log
CLIENT_LOG=$CASE_DIR/canary-client.log
TEST_LOG=$CASE_DIR/allocation-focused-tests.log
GPU_SAMPLES=$CASE_DIR/gpu-memory-samples.csv
PAYLOAD=$CASE_DIR/canary-payload.json
RESULT=$CASE_DIR/results.json
SERVER_PID=
MONITOR_PID=
FAILURE_STAGE=allocation_contract

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ -n "${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$status" -ne 0 ]; then
    "$LIBERO_PYTHON" - "$FAILURE" "$status" "$FAILURE_STAGE" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
value = {
    "schema_version": "1.0",
    "artifact_role": "r05a_inverse_flow_canary_launch_failure",
    "status": "failed",
    "exit_code": int(sys.argv[2]),
    "stage": sys.argv[3],
    "run_id": os.environ["RUN_ID"],
    "case_id": "crfs-1069f29a8d76463a",
    "scientific_claim_allowed": False,
    "probe_training_authorized": False,
    "teacher_generated_action_steps_executed": 0,
    "simulator_efficacy_evaluated": False,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "host": os.uname().nodename,
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

for path in \
  "$OPENPI_PYTHON" \
  "$LIBERO_PYTHON" \
  "$MANIFEST" \
  "$EXPERIMENT_CONFIG" \
  "$SCHEMA" \
  "$DECISION" \
  "$SOURCE_R02" \
  "$MODEL" \
  "$REMOTE_REPO/main/run_crfs_r05a_canary.py" \
  "$REMOTE_REPO/main/finalize_crfs_r05a_canary.py"; do
  test -e "$path" || { echo "missing R05A input: $path" >&2; exit 2; }
done
GIT_COMMIT=$(git -C "$REMOTE_REPO" rev-parse HEAD)
GIT_DIRTY=$(test -n "$(git -C "$REMOTE_REPO" status --porcelain)" && echo true || echo false)
MANIFEST_SHA256=$(sha256sum "$MANIFEST" | awk '{print $1}')
CONFIG_SHA256=$(sha256sum "$EXPERIMENT_CONFIG" | awk '{print $1}')
SCHEMA_SHA256=$(sha256sum "$SCHEMA" | awk '{print $1}')
DECISION_SHA256=$(sha256sum "$DECISION" | awk '{print $1}')
R02_SHA256=$(sha256sum "$SOURCE_R02" | awk '{print $1}')
CHECKPOINT_SHA256=$(sha256sum "$MODEL" | awk '{print $1}')
test "$GIT_COMMIT" = "$EXPECTED_GIT_COMMIT" || { echo "reviewed commit mismatch" >&2; exit 2; }
test "$GIT_DIRTY" = false || { echo "R05A canary requires a clean remote tree" >&2; exit 2; }
test "$MANIFEST_SHA256" = "$EXPECTED_MANIFEST_SHA256" || { echo "manifest hash mismatch" >&2; exit 2; }
test "$CONFIG_SHA256" = "$EXPECTED_CONFIG_SHA256" || { echo "config hash mismatch" >&2; exit 2; }
test "$SCHEMA_SHA256" = "$EXPECTED_SCHEMA_SHA256" || { echo "schema hash mismatch" >&2; exit 2; }
test "$DECISION_SHA256" = "$EXPECTED_DECISION_SHA256" || { echo "decision hash mismatch" >&2; exit 2; }
test "$R02_SHA256" = "$EXPECTED_R02_SHA256" || { echo "source R02 hash mismatch" >&2; exit 2; }
test "$CHECKPOINT_SHA256" = "$EXPECTED_CHECKPOINT_SHA256" || { echo "checkpoint hash mismatch" >&2; exit 2; }

cd "$REMOTE_REPO"

export OPENPI_DATA_HOME
export HF_HOME=/mnt/data/quanth/cache/huggingface
export XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
export PIP_CACHE_DIR=/mnt/data/quanth/cache/pip
export UV_CACHE_DIR=/mnt/data/quanth/cache/uv
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH=$CASE_DIR/libero-config
export PYTHONUNBUFFERED=1
case "$CUDA_VISIBLE_DEVICES" in
  *,*) echo "R05A canary requires exactly one visible GPU" >&2; exit 2 ;;
esac
ALLOCATED_GPU_UUID=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
case "$ALLOCATED_GPU_UUID" in
  GPU-*) ;;
  *) echo "cannot bind the allocation-visible GPU UUID" >&2; exit 2 ;;
esac
export CRFS_ALLOCATED_GPU_UUID=$ALLOCATED_GPU_UUID
mkdir -p "$LIBERO_CONFIG_PATH"

SAFELIBERO_ROOT=$REMOTE_REPO/safelibero/libero/libero
"$LIBERO_PYTHON" - "$LIBERO_CONFIG_PATH/config.yaml" "$SAFELIBERO_ROOT" <<'PY'
import os
import pathlib
import sys
import tempfile

destination = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2]).resolve()
values = {
    "benchmark_root": root,
    "bddl_files": root / "bddl_files",
    "init_states": root / "init_files",
    "datasets": root.parent / "datasets",
    "assets": root / "assets",
}
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as handle:
    for key, value in values.items():
        handle.write(f"{key}: {value}\n")
    temporary = handle.name
os.replace(temporary, destination)
PY

TRANSFORMERS_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh)
FAILURE_STAGE=dependency_backed_focused_tests
export PYTHONPATH=$TRANSFORMERS_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src
: >"$TEST_LOG"
for suite in \
  test_inverse_flow_control.py:17 \
  test_inverse_flow_sampler.py:8 \
  test_inverse_flow_policy.py:10 \
  test_r05a_canary.py:10; do
  pattern=${suite%%:*}
  expected=${suite##*:}
  suite_log=$CASE_DIR/allocation-$pattern.log
  "$OPENPI_PYTHON" -m unittest discover -s tests -p "$pattern" -v >"$suite_log" 2>&1
  observed=$(sed -n 's/^Ran \([0-9][0-9]*\) tests\{0,1\} in .*/\1/p' "$suite_log")
  test "$observed" = "$expected" || {
    echo "$pattern ran ${observed:-unknown} tests; expected $expected" >&2
    exit 2
  }
  test "$(grep -xc 'OK' "$suite_log")" = 1 || {
    echo "$pattern did not finish with exactly one OK" >&2
    exit 2
  }
  if grep -Eq 'skipped=|^FAILED|^ERROR' "$suite_log"; then
    echo "$pattern skipped or failed allocation-backed tests" >&2
    exit 2
  fi
  printf 'verified_test_suite=%s expected=%s observed=%s skips=0 status=passed\n' \
    "$pattern" "$expected" "$observed" >>"$TEST_LOG"
  sed "s/^/[$pattern] /" "$suite_log" >>"$TEST_LOG"
done

printf 'timestamp_ns,gpu_uuid,compute_mib,device_mib\n' >"$GPU_SAMPLES"
monitor_gpu() {
  while :; do
    timestamp=$(date +%s%N)
    compute=$(nvidia-smi --query-compute-apps=gpu_uuid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
      awk -F',' -v uuid="$ALLOCATED_GPU_UUID" '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); if ($1 == uuid && $2 ~ /^[0-9]+([.][0-9]+)?$/) sum += $2} END {printf "%.0f", sum + 0}')
    device=$(nvidia-smi --id="$ALLOCATED_GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | \
      awk '$1 ~ /^[0-9]+([.][0-9]+)?$/ {sum += $1} END {printf "%.0f", sum + 0}')
    printf '%s,%s,%s,%s\n' "$timestamp" "$ALLOCATED_GPU_UUID" "$compute" "$device" >>"$GPU_SAMPLES"
    sleep 1
  done
}
monitor_gpu &
MONITOR_PID=$!

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

FAILURE_STAGE=r05a_canary_payload
export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/packages/openpi-client/src
cd "$REMOTE_REPO"
"$LIBERO_PYTHON" main/run_crfs_r05a_canary.py \
  --manifest "$MANIFEST" \
  --config "$EXPERIMENT_CONFIG" \
  --r02-raw-root "$R02_RAW_ROOT" \
  --output-root "$EXPERIMENT_ROOT" \
  --run-id "$RUN_ID" \
  --case-index 0 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --checkpoint-id "$CHECKPOINT_DIR" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" >"$CLIENT_LOG" 2>&1
test -f "$PAYLOAD" || { echo "canary payload was not written" >&2; exit 5; }

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
MONITOR_PID=

host_cgroup_peak_bytes=
v2_path=$(awk -F: '$1 == "0" && $2 == "" {print $3; exit}' /proc/self/cgroup)
if [ -n "$v2_path" ] && [ -r "/sys/fs/cgroup${v2_path}/memory.peak" ]; then
  host_cgroup_peak_bytes=$(cat "/sys/fs/cgroup${v2_path}/memory.peak")
else
  v1_path=$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3; exit}' /proc/self/cgroup)
  if [ -n "$v1_path" ] && [ -r "/sys/fs/cgroup/memory${v1_path}/memory.max_usage_in_bytes" ]; then
    host_cgroup_peak_bytes=$(cat "/sys/fs/cgroup/memory${v1_path}/memory.max_usage_in_bytes")
  fi
fi
case "$host_cgroup_peak_bytes" in
  *[!0-9]*|'') echo "live Slurm cgroup memory peak is unavailable" >&2; exit 6 ;;
esac
test "$host_cgroup_peak_bytes" -gt 0 || { echo "live cgroup peak is not positive" >&2; exit 6; }

FAILURE_STAGE=atomic_memory_finalization
JSONSCHEMA_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_jsonschema_overlay.sh)
export PYTHONPATH=$JSONSCHEMA_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero
"$LIBERO_PYTHON" main/finalize_crfs_r05a_canary.py \
  --payload "$PAYLOAD" \
  --output "$RESULT" \
  --host-cgroup-peak-bytes "$host_cgroup_peak_bytes" \
  --gpu-samples "$GPU_SAMPLES" \
  --allocation-tests-log "$TEST_LOG" \
  --allocation-tests-exit-code 0
test -f "$RESULT" || { echo "finalizer returned without results.json" >&2; exit 7; }

FAILURE_STAGE=exact_gpu_job_validation
"$LIBERO_PYTHON" - "$RESULT" "$SLURM_JOB_ID" <<'PY'
import json
import pathlib
import sys

from crfs_oracle.r05a_canary import validate_r05a_canary_result, validate_r05a_canary_schema

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
errors = validate_r05a_canary_result(value) + validate_r05a_canary_schema(value, require_jsonschema=True)
if value.get("provenance", {}).get("slurm_job_id") != sys.argv[2]:
    errors.append("result does not bind the exact GPU job id")
if errors:
    raise SystemExit("R05A GPU-job validation failed: " + "; ".join(errors))
print("validation_error_count=0")
PY

echo "result=$RESULT"
echo "result_sha256=$(sha256sum "$RESULT" | awk '{print $1}')"
echo "host_cgroup_peak_bytes=$host_cgroup_peak_bytes"
FAILURE_STAGE=complete
