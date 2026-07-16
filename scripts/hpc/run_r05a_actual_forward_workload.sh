#!/usr/bin/env bash
set -euo pipefail

# AF-00A allocation workload.  It uses the ordinary Pi0.5 policy server and
# the existing opt-in residual_schedule envelope.  It never invokes the
# inverse-flow teacher, autograd, the retired CFS adapter, or simulator actions.
: "${SLURM_JOB_ID:?AF-00A must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?AF-00A requires an array job id}"
: "${SLURM_ARRAY_TASK_ID:?AF-00A requires an array task id}"
: "${SLURM_JOB_PARTITION:?AF-00A requires partition provenance}"
: "${CUDA_VISIBLE_DEVICES:?AF-00A requires an allocated GPU}"
: "${RUN_ID:?immutable AF-00A run id is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed release commit is required}"
: "${AF_SOURCE_CONTRACT:?AF-00A source contract is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${R02_RAW_ROOT:=/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a}"
: "${CHECKPOINT_DIR:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"

MANIFEST=$REMOTE_REPO/manifests/r05a_inverse_flow_teacher_smoke.jsonl
CONFIG=$REMOTE_REPO/configs/experiments/r05a_actual_forward_canary.json
LEGACY_CONFIG=$REMOTE_REPO/configs/experiments/r05a_inverse_flow_canary.json
OPENPI_PYTHON=/mnt/data/quanth/venvs/openpi/bin/python
LIBERO_PYTHON=/mnt/data/quanth/venvs/openpi-libero-client/bin/python
RUNTIME_IDENTITY_HELPER=$REMOTE_REPO/scripts/hpc/lib/r05a_runtime_identity.sh
CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
CASE_ID=crfs-1069f29a8d76463a
SOURCE_R02=$R02_RAW_ROOT/$CASE_ID/r02-paired.json
MODEL=$CHECKPOINT_DIR/model.safetensors

case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe AF-00A run id" >&2; exit 2 ;; esac
test "$SLURM_ARRAY_TASK_ID" = 0 || { echo "AF-00A is fixed to task zero" >&2; exit 2; }
test "$SLURM_JOB_PARTITION" = main || { echo "AF-00A is frozen to main" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "AF-00A must remain on worker-1" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 8 || { echo "AF-00A requires eight CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 65536 || { echo "AF-00A requires exactly 64 GiB" >&2; exit 2; }
test "$CUDA_VISIBLE_DEVICES" != NoDevFiles || { echo "AF-00A has no visible GPU" >&2; exit 2; }
case "$CUDA_VISIBLE_DEVICES" in *,*) echo "AF-00A requires one visible GPU" >&2; exit 2 ;; esac

for path in \
  "$MANIFEST" "$CONFIG" "$LEGACY_CONFIG" "$SOURCE_R02" "$MODEL" \
  "$AF_SOURCE_CONTRACT" "$RUNTIME_IDENTITY_HELPER" \
  "$REMOTE_REPO/main/run_crfs_r05a_actual_forward_canary.py" \
  "$REMOTE_REPO/openpi/scripts/serve_policy.py" \
  "$REMOTE_REPO/scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked AF-00A input: $path" >&2; exit 2; }
done
. "$RUNTIME_IDENTITY_HELPER"
test "$OPENPI_PYTHON" = "$CRFS_R05A_OPENPI_PYTHON" || { echo "noncanonical AF-00A OpenPI launcher" >&2; exit 2; }
test "$LIBERO_PYTHON" = "$CRFS_R05A_LIBERO_PYTHON" || { echo "noncanonical AF-00A LIBERO launcher" >&2; exit 2; }
crfs_validate_r05a_openpi_python "$OPENPI_PYTHON"
crfs_validate_r05a_libero_python "$LIBERO_PYTHON"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || { echo "AF-00A commit changed" >&2; exit 2; }
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || { echo "AF-00A remote tree is dirty" >&2; exit 2; }
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633 || { echo "AF-00A manifest changed" >&2; exit 2; }
test "$(sha256sum "$SOURCE_R02" | awk '{print $1}')" = 055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593 || { echo "AF-00A R02 source changed" >&2; exit 2; }
test "$(sha256sum "$MODEL" | awk '{print $1}')" = "$CHECKPOINT_SHA256" || { echo "AF-00A checkpoint changed" >&2; exit 2; }
jq -e '
  .ready_to_run == true and .blocked_on == []
  and .flow_contract.ordinary_residual_schedule_path_only == true
  and .flow_contract.inverse_flow_teacher_allowed == false
  and .flow_contract.autograd_allowed == false
  and .flow_contract.terminal_action_overwrite_allowed == false
  and .request_ledger.complete_run_exact_policy_request_count == 534
  and .request_ledger.pre_search_paired_requests == 4
  and .request_ledger.equal_split_repeatability_requests == 2
  and .request_ledger.cem_search_requests == 520
  and .request_ledger.selected_B_replay_requests == 2
  and .request_ledger.reversed_B_replay_requests == 2
  and .request_ledger.post_search_paired_requests == 4
  and .request_ledger.identical_schedule_requires_identical_scientific_output == true
  and .request_ledger.duplicate_replay_requests_execute_even_if_schedule_matches_A == true
  and .request_ledger.non_apparatus_outcome_requires_complete_ledger == true
  and .request_ledger.caught_apparatus_fault_must_issue_no_later_requests == true
  and .request_ledger.hard_failure_may_preserve_wrapper_evidence_without_partial_ledger == true
  and .request_ledger.missing_extra_reordered_deduplicated_or_post_failure_request_invalid == true
  and .execution_boundary.policy_generated_action_steps_executed == 0
  and .execution_boundary.teacher_generated_action_steps_executed == 0
  and .execution_boundary.efficacy_rollouts_executed == 0
' "$CONFIG" >/dev/null || { echo "AF-00A scientific contract changed" >&2; exit 2; }

if [ -z "${PORT:-}" ]; then PORT=$((20000 + SLURM_JOB_ID % 30000)); fi
case "$PORT" in *[!0-9]*|'') echo "invalid AF-00A port" >&2; exit 2 ;; esac
test "$PORT" -ge 1024 && test "$PORT" -le 65535 || { echo "AF-00A port out of range" >&2; exit 2; }

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
CASE_DIR=$RUN_ROOT/$CASE_ID
FAILURE_DIR=$RUN_ROOT/failures
test -f "$RUN_ROOT/submission.json" || { echo "missing AF-00A submission receipt" >&2; exit 2; }
test ! -e "$CASE_DIR" && test ! -e "$FAILURE_DIR" || { echo "immutable AF-00A output already exists" >&2; exit 2; }
mkdir "$CASE_DIR" "$FAILURE_DIR"

PAYLOAD=$CASE_DIR/af00a-raw-payload.json
TENSORS=$CASE_DIR/af00a-tensors.npz
QUERY_LEDGER=$CASE_DIR/query-ledger.json
RESULT=$CASE_DIR/results.json
SERVER_LOG=$CASE_DIR/policy-server.log
CLIENT_LOG=$CASE_DIR/actual-forward-client.log
TEST_LOG=$CASE_DIR/allocation-focused-tests.log
GPU_SAMPLES=$CASE_DIR/gpu-memory-samples.csv
GPU_STOP=$CASE_DIR/.gpu-monitor-stop
HOST_TELEMETRY=$CASE_DIR/host-cgroup-sampled-current.tsv
HOST_READY=$CASE_DIR/.host-telemetry-ready
HOST_STOP=$CASE_DIR/.host-telemetry-stop
POLICY_LAUNCH=$CASE_DIR/.policy-server-launch
POLICY_CLEAN=$CASE_DIR/.policy-server-cleanup-complete
GPU_CLEAN=$CASE_DIR/.gpu-monitor-cleanup-complete
WORKLOAD_CLEAN=$CASE_DIR/.workload-cleanup-complete
FAILURE=$FAILURE_DIR/case-index-0.json
SERVER_PID=
GPU_PID=
HOST_PID=
FAILURE_STAGE=allocation_contract

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi
  if [ -n "${GPU_PID:-}" ]; then : >"$GPU_STOP"; wait "$GPU_PID" 2>/dev/null || true; fi
  if [ -n "${HOST_PID:-}" ]; then : >"$HOST_STOP"; wait "$HOST_PID" 2>/dev/null || true; fi
  if [ "$status" -ne 0 ]; then
    temporary=$(mktemp "$FAILURE_DIR/.failure.XXXXXX") || return "$status"
    jq -n --arg run "$RUN_ID" --arg stage "$FAILURE_STAGE" --arg job "$SLURM_ARRAY_JOB_ID" --arg status "$status" \
      '{schema_version:"1.0",artifact_role:"r05a_actual_forward_canary_launch_failure",status:"failed",run_id:$run,case_id:"crfs-1069f29a8d76463a",stage:$stage,exit_code:($status|tonumber),slurm_array_job_id:$job,slurm_array_task_id:0,scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,policy_generated_action_steps_executed:0,teacher_generated_action_steps_executed:0}' >"$temporary" && mv "$temporary" "$FAILURE"
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

. "$REMOTE_REPO/scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh"
FAILURE_STAGE=host_telemetry_startup
crfs_monitor_cgroup_v2_full_lifetime \
  /proc/self/cgroup /proc/self/mountinfo /proc/sys/kernel/osrelease \
  "$HOST_TELEMETRY" "$HOST_READY" "$HOST_STOP" "$POLICY_LAUNCH" "$POLICY_CLEAN" \
  "$GPU_CLEAN" "$WORKLOAD_CLEAN" "$SLURM_ARRAY_JOB_ID" 0.1 500000000 &
HOST_PID=$!
ready=false
for _ in $(seq 1 100); do
  if [ -f "$HOST_READY" ]; then ready=true; break; fi
  kill -0 "$HOST_PID" 2>/dev/null || break
  sleep 0.1
done
test "$ready" = true || { echo "AF-00A host telemetry failed to start" >&2; exit 6; }

cd "$REMOTE_REPO"
export OPENPI_DATA_HOME=/mnt/data/quanth/cache/openpi
export HF_HOME=/mnt/data/quanth/cache/huggingface
export XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
export PIP_CACHE_DIR=/mnt/data/quanth/cache/pip
export UV_CACHE_DIR=/mnt/data/quanth/cache/uv
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export LIBERO_CONFIG_PATH=$CASE_DIR/libero-config
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
with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as handle:
    for key, value in values.items():
        handle.write(f"{key}: {value}\n")
    temporary = handle.name
os.replace(temporary, destination)
PY

TRANSFORMERS_OVERLAY=$(
  TRANSFORMERS_SITE_PACKAGES=/mnt/data/quanth/venvs/openpi/lib/python3.11/site-packages \
  TRANSFORMERS_OVERLAY=/mnt/data/quanth/cache/crfs/transformers-openpi-4.53.2-exact-24be8ac6749a \
  "$REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh"
)
export PYTHONPATH=$TRANSFORMERS_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src

FAILURE_STAGE=dependency_backed_focused_tests
"$LIBERO_PYTHON" -m unittest discover -s tests -p 'test_r05a_actual_forward*.py' -v >"$TEST_LOG" 2>&1
observed_tests=$(sed -n 's/^Ran \([0-9][0-9]*\) tests\{0,1\} in .*/\1/p' "$TEST_LOG")
case "$observed_tests" in ''|0) echo "AF-00A focused tests did not execute" >&2; exit 2 ;; esac
test "$(grep -xc 'OK' "$TEST_LOG")" = 1 || { echo "AF-00A focused tests failed" >&2; exit 2; }
if grep -Eq 'skipped=|^FAILED|^ERROR' "$TEST_LOG"; then echo "AF-00A focused tests skipped or failed" >&2; exit 2; fi

ALLOCATED_GPU_UUID=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
case "$ALLOCATED_GPU_UUID" in GPU-*) ;; *) echo "cannot bind AF-00A GPU UUID" >&2; exit 2 ;; esac
printf 'timestamp_ns,gpu_uuid,compute_mib,device_mib\n' >"$GPU_SAMPLES"
monitor_gpu() {
  while [ ! -e "$GPU_STOP" ]; do
    timestamp=$(date +%s%N)
    compute=$(nvidia-smi --query-compute-apps=gpu_uuid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' -v uuid="$ALLOCATED_GPU_UUID" '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); if ($1 == uuid && $2 ~ /^[0-9]+([.][0-9]+)?$/) sum += $2} END {printf "%.0f", sum + 0}')
    device=$(nvidia-smi --id="$ALLOCATED_GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+([.][0-9]+)?$/ {sum += $1} END {printf "%.0f", sum + 0}')
    printf '%s,%s,%s,%s\n' "$timestamp" "$ALLOCATED_GPU_UUID" "$compute" "$device" >>"$GPU_SAMPLES"
    sleep 1
  done
}
monitor_gpu &
GPU_PID=$!

FAILURE_STAGE=ordinary_policy_server_startup
crfs_write_cgroup_v2_full_lifetime_marker "$POLICY_LAUNCH"
(
  cd "$REMOTE_REPO/openpi"
  exec "$OPENPI_PYTHON" scripts/serve_policy.py --port "$PORT" \
    policy:checkpoint --policy.config pi05_libero --policy.dir "$CHECKPOINT_DIR"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
server_ready=false
for _ in $(seq 1 600); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 200 "$SERVER_LOG" >&2 || true; exit 3; fi
  if grep -q 'server listening' "$SERVER_LOG"; then server_ready=true; break; fi
  sleep 1
done
test "$server_ready" = true || { tail -n 200 "$SERVER_LOG" >&2 || true; exit 4; }

FAILURE_STAGE=actual_forward_payload
"$LIBERO_PYTHON" "$REMOTE_REPO/main/run_crfs_r05a_actual_forward_canary.py" \
  --manifest "$MANIFEST" --config "$CONFIG" --legacy-config "$LEGACY_CONFIG" \
  --r02-raw-root "$R02_RAW_ROOT" --output-root "$EXPERIMENT_ROOT" --run-id "$RUN_ID" \
  --case-index 0 --host 127.0.0.1 --port "$PORT" --checkpoint-id "$CHECKPOINT_DIR" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" >"$CLIENT_LOG" 2>&1
test -f "$PAYLOAD" || { echo "AF-00A payload was not written" >&2; exit 5; }
test -f "$TENSORS" || { echo "AF-00A tensor archive was not written" >&2; exit 5; }
test -f "$QUERY_LEDGER" || { echo "AF-00A query ledger was not written" >&2; exit 5; }
jq -e --arg run "$RUN_ID" --arg job "$SLURM_ARRAY_JOB_ID" --arg tensors "$TENSORS" --arg tensor_sha "$(sha256sum "$TENSORS" | awk '{print $1}')" '
  .run_id == $run and .case_id == "crfs-1069f29a8d76463a"
  and (.request_ledger | type=="array" and length==534)
  and .tensor_archive.path == $tensors and .tensor_archive.sha256 == $tensor_sha
  and (.provenance.slurm_array_job_id|tostring) == $job
  and (.provenance.slurm_array_task_id|tostring) == "0"
  and .execution_boundary.policy_generated_action_steps_executed == 0
  and .execution_boundary.teacher_generated_action_steps_executed == 0
  and .execution_boundary.efficacy_rollouts_executed == 0
  and .execution_boundary.simulator_efficacy_evaluated == false
' "$PAYLOAD" >/dev/null || { echo "AF-00A raw payload contract changed" >&2; exit 5; }
jq -e --arg tensor_sha "$(sha256sum "$TENSORS" | awk '{print $1}')" '
  .case_id == "crfs-1069f29a8d76463a"
  and .exact_policy_request_count == 534
  and (.rows | type=="array" and length==534)
  and .tensor_archive_sha256 == $tensor_sha
' "$QUERY_LEDGER" >/dev/null || { echo "AF-00A query ledger contract changed" >&2; exit 5; }
kill -0 "$SERVER_PID" 2>/dev/null || { echo "ordinary policy server died during AF-00A" >&2; exit 5; }
if grep -Eiq 'out[ -]?of[ -]?memory|outofmemoryerror|cuda([^[:alnum:]]|_)*(oom|error[^[:alnum:]]*2)|cublas_status_alloc_failed|cudnn_status_alloc_failed|std::bad_alloc|cannot allocate memory' "$SERVER_LOG" "$CLIENT_LOG"; then
  echo "AF-00A log indicates an out-of-memory failure" >&2; exit 5
fi

kill "$SERVER_PID"
if wait "$SERVER_PID"; then server_status=0; else server_status=$?; fi
SERVER_PID=
case "$server_status" in 0|143) ;; *) echo "ordinary policy server cleanup status changed: $server_status" >&2; exit 5 ;; esac
crfs_write_cgroup_v2_full_lifetime_marker "$POLICY_CLEAN"
: >"$GPU_STOP"
wait "$GPU_PID" || { GPU_PID=; echo "AF-00A GPU monitor failed" >&2; exit 6; }
GPU_PID=
crfs_write_cgroup_v2_full_lifetime_marker "$GPU_CLEAN"

FAILURE_STAGE=host_telemetry_seal
crfs_write_cgroup_v2_full_lifetime_marker "$WORKLOAD_CLEAN"
: >"$HOST_STOP"
wait "$HOST_PID" || { HOST_PID=; echo "AF-00A host telemetry failed" >&2; exit 6; }
HOST_PID=
test -s "$HOST_TELEMETRY" || { echo "AF-00A host telemetry was not sealed" >&2; exit 6; }
"$LIBERO_PYTHON" - "$HOST_TELEMETRY" "$SLURM_ARRAY_JOB_ID" <<'PY'
import sys

from crfs_oracle.r05a_full_lifetime_telemetry import parse_full_lifetime_telemetry

summary = parse_full_lifetime_telemetry(sys.argv[1], expected_job_id=sys.argv[2])
if summary.get("contract_passed") is not True:
    raise SystemExit("AF-00A allocation-side host telemetry contract failed")
print(f"allocation_host_telemetry_sha256={summary['raw_trace_sha256']}")
PY
test ! -e "$RESULT" && test ! -e "$CASE_DIR/.results.candidate.json" || { echo "AF-00A GPU job must not publish results.json" >&2; exit 7; }

echo "payload=$PAYLOAD"
echo "payload_sha256=$(sha256sum "$PAYLOAD" | awk '{print $1}')"
echo "tensors=$TENSORS"
echo "tensors_sha256=$(sha256sum "$TENSORS" | awk '{print $1}')"
echo "query_ledger=$QUERY_LEDGER"
echo "query_ledger_sha256=$(sha256sum "$QUERY_LEDGER" | awk '{print $1}')"
echo "focused_tests=$observed_tests"
echo "host_telemetry_sha256=$(sha256sum "$HOST_TELEMETRY" | awk '{print $1}')"
FAILURE_STAGE=complete
