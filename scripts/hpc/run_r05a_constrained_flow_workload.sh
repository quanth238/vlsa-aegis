#!/usr/bin/env bash
set -euo pipefail

# CFS-00A allocation workload.  This is intentionally separate from every
# historical R05A runner because it installs the opt-in constrained-flow policy
# adapter.  It writes raw payloads and telemetry only; the CPU afterany job is
# the sole results.json publisher.
: "${SLURM_JOB_ID:?CFS-00A must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?CFS-00A requires its array job id}"
: "${SLURM_ARRAY_TASK_ID:?CFS-00A requires its array task id}"
: "${SLURM_JOB_PARTITION:?CFS-00A requires partition provenance}"
: "${CUDA_VISIBLE_DEVICES:?CFS-00A requires an allocated GPU}"
: "${RUN_ID:?immutable CFS-00A run id is required}"
: "${MANIFEST:?frozen manifest is required}"
: "${CFS_CONFIG:?frozen constrained-flow config is required}"
: "${LEGACY_CONFIG:?frozen historical R05A config is required}"
: "${R02_RAW_ROOT:?frozen R02 raw root is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed release commit is required}"
: "${R05A_SOURCE_CONTRACT:?source contract is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${CHECKPOINT_DIR:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"

require_canonical_runtime_path() {
  local variable=${1:?runtime variable is required}
  local expected=${2:?canonical runtime path is required}
  local observed=${!variable-}
  if [ -n "$observed" ] && [ "$observed" != "$expected" ]; then
    echo "noncanonical inherited CFS-00A runtime path: $variable=$observed" >&2
    return 2
  fi
  printf -v "$variable" '%s' "$expected"
}

require_canonical_runtime_path OPENPI_PYTHON /mnt/data/quanth/venvs/openpi/bin/python
require_canonical_runtime_path LIBERO_PYTHON /mnt/data/quanth/venvs/openpi-libero-client/bin/python
require_canonical_runtime_path OPENPI_DATA_HOME /mnt/data/quanth/cache/openpi
require_canonical_runtime_path TRANSFORMERS_SITE_PACKAGES /mnt/data/quanth/venvs/openpi/lib/python3.11/site-packages
require_canonical_runtime_path TRANSFORMERS_OVERLAY /mnt/data/quanth/cache/crfs/transformers-openpi-4.53.2-exact-24be8ac6749a
require_canonical_runtime_path PYTHONDONTWRITEBYTECODE 1

EXPECTED_MANIFEST_SHA256=bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633
EXPECTED_LEGACY_CONFIG_SHA256=c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb
EXPECTED_R02_SHA256=055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593
EXPECTED_CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
CASE_ID=crfs-1069f29a8d76463a
CASE_INDEX=$SLURM_ARRAY_TASK_ID
MODEL=$CHECKPOINT_DIR/model.safetensors
SOURCE_R02=$R02_RAW_ROOT/$CASE_ID/r02-paired.json
ALLOCATION_TEST_REGISTRY=$REMOTE_REPO/main/crfs_oracle/r05a_constrained_flow_allocation_tests.json

case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe CFS-00A RUN_ID" >&2; exit 2 ;; esac
case "$SLURM_ARRAY_JOB_ID" in *[!0-9]*|'') echo "invalid array job id" >&2; exit 2 ;; esac
test "$SLURM_JOB_PARTITION" = main || { echo "CFS-00A is frozen to main" >&2; exit 2; }
test "$CASE_INDEX" = 0 || { echo "CFS-00A is fixed to array task zero" >&2; exit 2; }
test "$CUDA_VISIBLE_DEVICES" != NoDevFiles || { echo "CFS-00A has no visible GPU" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "CFS-00A must remain on worker-1" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 8 || { echo "CFS-00A requires eight CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 65536 || { echo "CFS-00A requires exactly 64 GiB" >&2; exit 2; }

if [ -z "${PORT:-}" ]; then
  PORT=$((20000 + SLURM_JOB_ID % 30000))
fi
case "$PORT" in *[!0-9]*|'') echo "PORT must be an integer" >&2; exit 2 ;; esac
test "$PORT" -ge 1024 && test "$PORT" -le 65535 || { echo "PORT outside 1024..65535" >&2; exit 2; }

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
CASE_DIR=$RUN_ROOT/$CASE_ID
FAILURE_DIR=$RUN_ROOT/failures
test -f "$RUN_ROOT/submission.json" || { echo "missing atomic submission receipt" >&2; exit 2; }
test ! -e "$CASE_DIR" && test ! -e "$FAILURE_DIR" || { echo "immutable run already has allocation output" >&2; exit 2; }
mkdir "$CASE_DIR" "$FAILURE_DIR"

FAILURE=$FAILURE_DIR/case-index-0.json
SERVER_LOG=$CASE_DIR/cfs-policy-server.log
CLIENT_LOG=$CASE_DIR/constrained-flow-client.log
TEST_LOG=$CASE_DIR/allocation-focused-tests.log
GPU_SAMPLES=$CASE_DIR/gpu-memory-samples.csv
GPU_MONITOR_STOP=$CASE_DIR/.gpu-monitor-stop
HOST_TELEMETRY=$CASE_DIR/host-cgroup-sampled-current.tsv
HOST_TELEMETRY_READY=$CASE_DIR/.host-telemetry-ready
HOST_TELEMETRY_STOP=$CASE_DIR/.host-telemetry-stop
POLICY_SERVER_LAUNCH=$CASE_DIR/.policy-server-launch
POLICY_SERVER_CLEANUP_COMPLETE=$CASE_DIR/.policy-server-cleanup-complete
GPU_MONITOR_CLEANUP_COMPLETE=$CASE_DIR/.gpu-monitor-cleanup-complete
WORKLOAD_CLEANUP_COMPLETE=$CASE_DIR/.workload-cleanup-complete
LEGACY_PAYLOAD=$CASE_DIR/canary-payload.json
CFS_PAYLOAD=$CASE_DIR/constrained-flow-payload.json
HIDDEN_CANDIDATE=$CASE_DIR/.results.candidate.json
RESULT=$CASE_DIR/results.json
SERVER_PID=
GPU_MONITOR_PID=
HOST_MONITOR_PID=
FAILURE_STAGE=allocation_contract

crfs_stop_policy_server_exact_sigterm() {
  local pid=${1:?policy server pid is required}
  local wait_status
  kill "$pid" 2>/dev/null || return 1
  if wait "$pid" 2>/dev/null; then
    wait_status=0
  else
    wait_status=$?
  fi
  if [ "$wait_status" -ne 143 ]; then
    echo "CFS policy server wait status was $wait_status, expected reviewed SIGTERM status 143" >&2
    return 1
  fi
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
  if [ -n "${GPU_MONITOR_PID:-}" ] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    : >"$GPU_MONITOR_STOP"
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  GPU_MONITOR_PID=
  if [ -n "${HOST_MONITOR_PID:-}" ]; then
    if [ ! -e "$WORKLOAD_CLEANUP_COMPLETE" ] && command -v crfs_write_cgroup_v2_full_lifetime_marker >/dev/null 2>&1; then
      crfs_write_cgroup_v2_full_lifetime_marker "$WORKLOAD_CLEANUP_COMPLETE" || true
    fi
    : >"$HOST_TELEMETRY_STOP"
    wait "$HOST_MONITOR_PID" 2>/dev/null || true
  fi
  HOST_MONITOR_PID=
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
    "artifact_role": "r05a_constrained_flow_canary_launch_failure",
    "status": "failed",
    "exit_code": int(sys.argv[2]),
    "stage": sys.argv[3],
    "run_id": os.environ["RUN_ID"],
    "case_id": "crfs-1069f29a8d76463a",
    "scientific_claim_allowed": False,
    "infeasibility_claim_allowed": False,
    "probe_training_authorized": False,
    "policy_generated_action_steps_executed": 0,
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
  "$CFS_CONFIG" \
  "$LEGACY_CONFIG" \
  "$R05A_SOURCE_CONTRACT" \
  "$ALLOCATION_TEST_REGISTRY" \
  "$SOURCE_R02" \
  "$MODEL" \
  "$REMOTE_REPO/scripts/hpc/lib/r05a_allocation_tests.sh" \
  "$REMOTE_REPO/scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh" \
  "$REMOTE_REPO/main/run_crfs_r05a_constrained_flow_canary.py" \
  "$REMOTE_REPO/openpi/scripts/serve_cfs_policy.py"; do
  test -e "$path" && test ! -L "$path" || { echo "missing or symlinked CFS-00A input: $path" >&2; exit 2; }
done
. "$REMOTE_REPO/scripts/hpc/lib/r05a_allocation_tests.sh"
. "$REMOTE_REPO/scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh"

GIT_COMMIT=$(git -C "$REMOTE_REPO" rev-parse HEAD)
GIT_DIRTY=$(test -n "$(git -C "$REMOTE_REPO" status --porcelain)" && echo true || echo false)
test "$GIT_COMMIT" = "$EXPECTED_GIT_COMMIT" || { echo "reviewed commit mismatch" >&2; exit 2; }
test "$GIT_DIRTY" = false || { echo "CFS-00A requires a clean remote tree" >&2; exit 2; }
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA256" || { echo "manifest hash mismatch" >&2; exit 2; }
test "$(sha256sum "$LEGACY_CONFIG" | awk '{print $1}')" = "$EXPECTED_LEGACY_CONFIG_SHA256" || { echo "legacy config hash mismatch" >&2; exit 2; }
test "$(sha256sum "$SOURCE_R02" | awk '{print $1}')" = "$EXPECTED_R02_SHA256" || { echo "source R02 hash mismatch" >&2; exit 2; }
test "$(sha256sum "$MODEL" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA256" || { echo "checkpoint hash mismatch" >&2; exit 2; }

for relative in \
  configs/experiments/r05a_constrained_flow_canary.json \
  configs/experiments/r05a_inverse_flow_canary.json \
  main/crfs_oracle/r05a_constrained_flow_allocation_tests.json \
  main/crfs_oracle/r05a_constrained_flow_canary.py \
  main/run_crfs_r05a_constrained_flow_canary.py \
  openpi/scripts/serve_cfs_policy.py \
  openpi/src/openpi/models_pytorch/crfs_linearized_control.py \
  openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py \
  openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py \
  openpi/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py \
  openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/check.py \
  openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py \
  openpi/src/openpi/policies/crfs_constrained_flow_adapter.py; do
  expected=$(jq -er --arg path "$relative" '.repository_file_sha256[$path]' "$R05A_SOURCE_CONTRACT")
  actual=$(sha256sum "$REMOTE_REPO/$relative" | awk '{print $1}')
  test "$actual" = "$expected" || { echo "source-contract hash mismatch: $relative" >&2; exit 2; }
done

FAILURE_STAGE=host_telemetry_startup
crfs_monitor_cgroup_v2_full_lifetime \
  /proc/self/cgroup \
  /proc/self/mountinfo \
  /proc/sys/kernel/osrelease \
  "$HOST_TELEMETRY" \
  "$HOST_TELEMETRY_READY" \
  "$HOST_TELEMETRY_STOP" \
  "$POLICY_SERVER_LAUNCH" \
  "$POLICY_SERVER_CLEANUP_COMPLETE" \
  "$GPU_MONITOR_CLEANUP_COMPLETE" \
  "$WORKLOAD_CLEANUP_COMPLETE" \
  "$SLURM_ARRAY_JOB_ID" \
  0.1 \
  500000000 &
HOST_MONITOR_PID=$!
telemetry_ready=false
for _ in $(seq 1 100); do
  if [ -f "$HOST_TELEMETRY_READY" ]; then telemetry_ready=true; break; fi
  kill -0 "$HOST_MONITOR_PID" 2>/dev/null || break
  sleep 0.1
done
test "$telemetry_ready" = true || { echo "host telemetry did not signal ready" >&2; exit 6; }

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
export PYTHONDONTWRITEBYTECODE
case "$CUDA_VISIBLE_DEVICES" in *,*) echo "CFS-00A requires exactly one visible GPU" >&2; exit 2 ;; esac
ALLOCATED_GPU_UUID=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
case "$ALLOCATED_GPU_UUID" in GPU-*) ;; *) echo "cannot bind allocated GPU UUID" >&2; exit 2 ;; esac
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

prepared_transformers_overlay=$(
  TRANSFORMERS_SITE_PACKAGES="$TRANSFORMERS_SITE_PACKAGES" \
  TRANSFORMERS_OVERLAY="$TRANSFORMERS_OVERLAY" \
  "$REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh"
)
test "$prepared_transformers_overlay" = "$TRANSFORMERS_OVERLAY" || {
  echo "Transformers overlay helper returned a noncanonical runtime path" >&2
  exit 3
}
FAILURE_STAGE=dependency_backed_focused_tests
export PYTHONPATH=$TRANSFORMERS_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src
crfs_run_r05a_allocation_tests "$ALLOCATION_TEST_REGISTRY" "$OPENPI_PYTHON" tests "$TEST_LOG" "$CASE_DIR"

printf 'timestamp_ns,gpu_uuid,compute_mib,device_mib\n' >"$GPU_SAMPLES"
monitor_gpu() {
  while [ ! -e "$GPU_MONITOR_STOP" ]; do
    timestamp=$(date +%s%N)
    compute=$(nvidia-smi --query-compute-apps=gpu_uuid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' -v uuid="$ALLOCATED_GPU_UUID" '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); if ($1 == uuid && $2 ~ /^[0-9]+([.][0-9]+)?$/) sum += $2} END {printf "%.0f", sum + 0}')
    device=$(nvidia-smi --id="$ALLOCATED_GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+([.][0-9]+)?$/ {sum += $1} END {printf "%.0f", sum + 0}')
    printf '%s,%s,%s,%s\n' "$timestamp" "$ALLOCATED_GPU_UUID" "$compute" "$device" >>"$GPU_SAMPLES"
    sleep 1
  done
}
monitor_gpu &
GPU_MONITOR_PID=$!

FAILURE_STAGE=cfs_policy_server_startup
kill -0 "$HOST_MONITOR_PID" 2>/dev/null || { echo "host monitor stopped before policy launch" >&2; exit 6; }
crfs_write_cgroup_v2_full_lifetime_marker "$POLICY_SERVER_LAUNCH"
(
  cd "$REMOTE_REPO/openpi"
  exec "$OPENPI_PYTHON" scripts/serve_cfs_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$CHECKPOINT_DIR"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 200 "$SERVER_LOG" >&2 || true; exit 3; fi
  if grep -q "server listening" "$SERVER_LOG"; then ready=1; break; fi
  sleep 1
done
test "$ready" = 1 || { tail -n 200 "$SERVER_LOG" >&2 || true; exit 4; }

FAILURE_STAGE=constrained_flow_payload
export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/packages/openpi-client/src
cd "$REMOTE_REPO"
"$LIBERO_PYTHON" main/run_crfs_r05a_constrained_flow_canary.py \
  --manifest "$MANIFEST" \
  --config "$CFS_CONFIG" \
  --legacy-config "$LEGACY_CONFIG" \
  --r02-raw-root "$R02_RAW_ROOT" \
  --output-root "$EXPERIMENT_ROOT" \
  --run-id "$RUN_ID" \
  --case-index 0 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --checkpoint-id "$CHECKPOINT_DIR" \
  --checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256" >"$CLIENT_LOG" 2>&1
test -f "$CFS_PAYLOAD" || { echo "constrained-flow payload was not written" >&2; exit 5; }
kill -0 "$SERVER_PID" 2>/dev/null || {
  echo "CFS policy server died during the paired client; refusing clean source completion" >&2
  tail -n 200 "$SERVER_LOG" >&2 || true
  exit 5
}
payload_variant=$(jq -er '.payload_variant' "$CFS_PAYLOAD")
case "$payload_variant" in
  complete_comparison)
    test -f "$LEGACY_PAYLOAD" || { echo "complete comparison lacks legacy payload" >&2; exit 5; }
    ;;
  terminal_apparatus_failure)
    # The raw failure records whether the legacy payload was reached.  The CPU
    # validator owns the consistency check and scientific classification.
    terminal_failure=$(jq -er '[.failure.error_type,.failure.message] | join("\n")' "$CFS_PAYLOAD")
    if printf '%s\n' "$terminal_failure" | grep -Eiq \
      'out[ -]?of[ -]?memory|cuda([^[:alnum:]]|_)*(oom|error[^[:alnum:]]*2)|cublas_status_alloc_failed|cudnn_status_alloc_failed|outofmemoryerror|memoryerror|std::bad_alloc|cannot allocate memory'; then
      echo "terminal CFS payload indicates an out-of-memory failure; refusing clean source completion" >&2
      exit 5
    fi
    ;;
  *) echo "unknown constrained-flow payload variant" >&2; exit 5 ;;
esac
if grep -Eiq \
  'out[ -]?of[ -]?memory|cuda([^[:alnum:]]|_)*(oom|error[^[:alnum:]]*2)|cublas_status_alloc_failed|cudnn_status_alloc_failed|outofmemoryerror|memoryerror|std::bad_alloc|cannot allocate memory' \
  "$SERVER_LOG"; then
  echo "CFS policy server log indicates an out-of-memory failure; refusing clean source completion" >&2
  exit 5
fi

if crfs_stop_policy_server_exact_sigterm "$SERVER_PID"; then
  server_shutdown_status=0
else
  server_shutdown_status=$?
fi
SERVER_PID=
test "$server_shutdown_status" -eq 0 || {
  echo "CFS policy server did not complete the reviewed SIGTERM cleanup" >&2
  exit 5
}
crfs_write_cgroup_v2_full_lifetime_marker "$POLICY_SERVER_CLEANUP_COMPLETE"
: >"$GPU_MONITOR_STOP"
wait "$GPU_MONITOR_PID" || { GPU_MONITOR_PID=; echo "GPU monitor failed" >&2; exit 6; }
GPU_MONITOR_PID=
crfs_write_cgroup_v2_full_lifetime_marker "$GPU_MONITOR_CLEANUP_COMPLETE"

FAILURE_STAGE=host_telemetry_seal
crfs_write_cgroup_v2_full_lifetime_marker "$WORKLOAD_CLEANUP_COMPLETE"
: >"$HOST_TELEMETRY_STOP"
wait "$HOST_MONITOR_PID" || { HOST_MONITOR_PID=; echo "host telemetry monitor failed" >&2; exit 6; }
HOST_MONITOR_PID=
test -s "$HOST_TELEMETRY" || { echo "host telemetry was not atomically sealed" >&2; exit 6; }

FAILURE_STAGE=host_telemetry_allocation_revalidation
"$LIBERO_PYTHON" - "$HOST_TELEMETRY" "$SLURM_ARRAY_JOB_ID" <<'PY'
import sys
from crfs_oracle.r05a_full_lifetime_telemetry import parse_full_lifetime_telemetry

summary = parse_full_lifetime_telemetry(sys.argv[1], expected_job_id=sys.argv[2])
if summary.get("contract_passed") is not True:
    raise SystemExit("allocation-side host telemetry contract did not pass")
print("allocation_host_telemetry_revalidation=passed")
print(f"allocation_host_telemetry_sha256={summary['raw_trace_sha256']}")
PY
test ! -e "$HIDDEN_CANDIDATE" && test ! -e "$RESULT" || { echo "GPU source must not publish a candidate or results.json" >&2; exit 7; }

echo "legacy_payload=$LEGACY_PAYLOAD"
echo "legacy_payload_exists=$([ -f "$LEGACY_PAYLOAD" ] && echo true || echo false)"
echo "constrained_flow_payload=$CFS_PAYLOAD"
echo "constrained_flow_payload_sha256=$(sha256sum "$CFS_PAYLOAD" | awk '{print $1}')"
echo "host_telemetry=$HOST_TELEMETRY"
echo "host_telemetry_sha256=$(sha256sum "$HOST_TELEMETRY" | awk '{print $1}')"
FAILURE_STAGE=complete
