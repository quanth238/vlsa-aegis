#!/usr/bin/env bash
set -euo pipefail

# TRL-00A allocation workload.  It uses the ordinary Pi0.5 policy server and
# the opt-in reference-trajectory lift envelope.  It never invokes the
# inverse-flow teacher, autograd, a planner, or generated simulator actions.
: "${SLURM_JOB_ID:?TRL-00A must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?TRL-00A requires an array job id}"
: "${SLURM_ARRAY_TASK_ID:?TRL-00A requires an array task id}"
: "${SLURM_JOB_PARTITION:?TRL-00A requires partition provenance}"
: "${CUDA_VISIBLE_DEVICES:?TRL-00A requires an allocated GPU}"
: "${RUN_ID:?immutable TRL-00A run id is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed release commit is required}"
: "${TRL_SOURCE_CONTRACT:?TRL-00A source contract is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${R02_RAW_ROOT:=/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a}"
: "${CHECKPOINT_DIR:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"

MANIFEST=$REMOTE_REPO/manifests/r05a_inverse_flow_teacher_smoke.jsonl
CONFIG=$REMOTE_REPO/configs/experiments/r05a_reference_trajectory_lift_canary.json
LEGACY_CONFIG=$REMOTE_REPO/configs/experiments/r05a_inverse_flow_canary.json
OPENPI_PYTHON=/mnt/data/quanth/venvs/openpi/bin/python
LIBERO_PYTHON=/mnt/data/quanth/venvs/openpi-libero-client/bin/python
RUNTIME_IDENTITY_HELPER=$REMOTE_REPO/scripts/hpc/lib/r05a_runtime_identity.sh
CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
CASE_ID=crfs-1069f29a8d76463a
SOURCE_R02=$R02_RAW_ROOT/$CASE_ID/r02-paired.json
MODEL=$CHECKPOINT_DIR/model.safetensors

case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe TRL-00A run id" >&2; exit 2 ;; esac
test "$SLURM_ARRAY_TASK_ID" = 0 || { echo "TRL-00A is fixed to task zero" >&2; exit 2; }
test "$SLURM_JOB_PARTITION" = main || { echo "TRL-00A is frozen to main" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "TRL-00A must remain on worker-1" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 8 || { echo "TRL-00A requires eight CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 65536 || { echo "TRL-00A requires exactly 64 GiB" >&2; exit 2; }
test "$CUDA_VISIBLE_DEVICES" != NoDevFiles || { echo "TRL-00A has no visible GPU" >&2; exit 2; }
case "$CUDA_VISIBLE_DEVICES" in *,*) echo "TRL-00A requires one visible GPU" >&2; exit 2 ;; esac

for path in \
  "$MANIFEST" "$CONFIG" "$LEGACY_CONFIG" "$SOURCE_R02" "$MODEL" \
  "$TRL_SOURCE_CONTRACT" "$RUNTIME_IDENTITY_HELPER" \
  "$REMOTE_REPO/main/run_crfs_r05a_reference_trajectory_lift_canary.py" \
  "$REMOTE_REPO/openpi/scripts/serve_policy.py" \
  "$REMOTE_REPO/scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked TRL-00A input: $path" >&2; exit 2; }
done
. "$RUNTIME_IDENTITY_HELPER"
test "$OPENPI_PYTHON" = "$CRFS_R05A_OPENPI_PYTHON" || { echo "noncanonical TRL-00A OpenPI launcher" >&2; exit 2; }
test "$LIBERO_PYTHON" = "$CRFS_R05A_LIBERO_PYTHON" || { echo "noncanonical TRL-00A LIBERO launcher" >&2; exit 2; }
crfs_validate_r05a_openpi_python "$OPENPI_PYTHON"
crfs_validate_r05a_libero_python "$LIBERO_PYTHON"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT" || { echo "TRL-00A commit changed" >&2; exit 2; }
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || { echo "TRL-00A remote tree is dirty" >&2; exit 2; }
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633 || { echo "TRL-00A manifest changed" >&2; exit 2; }
test "$(sha256sum "$SOURCE_R02" | awk '{print $1}')" = 055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593 || { echo "TRL-00A R02 source changed" >&2; exit 2; }
test "$(sha256sum "$MODEL" | awk '{print $1}')" = "$CHECKPOINT_SHA256" || { echo "TRL-00A checkpoint changed" >&2; exit 2; }
jq -e '
  .ready_to_run == true and .blocked_on == []
  and .preregistration.h100_submission_authorized == true
  and .flow_contract.sampler_steps == 10
  and .flow_contract.intervention_step == 5
  and .flow_contract.inverse_flow_teacher_allowed == false
  and .flow_contract.autograd_allowed == false
  and .flow_contract.cem_or_empirical_population_fit_allowed == false
  and .flow_contract.terminal_action_overwrite_allowed == false
  and .reference_lift_contract.mode == "reference_trajectory_lift"
  and .reference_lift_contract.raw_nonfinite_result == "apparatus_inconclusive_no_scientific_results_json"
  and .request_ledger.complete_finite_exact_policy_request_count == 18
  and .request_ledger.complete_ordered_phases == [
    {"name":"compiled_frozen_pre","start":0,"stop_inclusive":0},
    {"name":"eager_source_trace_pre","start":1,"stop_inclusive":1},
    {"name":"eager_normalized_final_pre","start":2,"stop_inclusive":2},
    {"name":"zero_schedule_pre","start":3,"stop_inclusive":3},
    {"name":"arm_a_equal_split","start":4,"stop_inclusive":5},
    {"name":"budgeted_lift_generation","start":6,"stop_inclusive":7},
    {"name":"budgeted_lift_replay","start":8,"stop_inclusive":9},
    {"name":"raw_lift_generation","start":10,"stop_inclusive":11},
    {"name":"raw_lift_replay","start":12,"stop_inclusive":13},
    {"name":"zero_schedule_post","start":14,"stop_inclusive":14},
    {"name":"eager_normalized_final_post","start":15,"stop_inclusive":15},
    {"name":"eager_source_trace_post","start":16,"stop_inclusive":16},
    {"name":"compiled_frozen_post","start":17,"stop_inclusive":17}
  ]
  and .request_ledger.missing_extra_reordered_or_deduplicated_request_invalid == true
  and .artifact_contract.raw_payload_name == "trl00a-raw-payload.json"
  and .artifact_contract.tensor_archive_name == "trl00a-tensors.npz"
  and .artifact_contract.request_ledger_name == "request-ledger.json"
  and .artifact_contract.gpu_may_publish_results_json == false
  and .artifact_contract.cpu_afterany_is_sole_publisher == true
  and .execution_boundary.policy_generated_action_steps_executed == 0
  and .execution_boundary.teacher_generated_action_steps_executed == 0
  and .execution_boundary.efficacy_rollouts_executed == 0
' "$CONFIG" >/dev/null || { echo "TRL-00A scientific contract changed" >&2; exit 2; }

if [ -z "${PORT:-}" ]; then PORT=$((20000 + SLURM_JOB_ID % 30000)); fi
case "$PORT" in *[!0-9]*|'') echo "invalid TRL-00A port" >&2; exit 2 ;; esac
test "$PORT" -ge 1024 && test "$PORT" -le 65535 || { echo "TRL-00A port out of range" >&2; exit 2; }

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
CASE_DIR=$RUN_ROOT/$CASE_ID
FAILURE_DIR=$RUN_ROOT/failures
test -f "$RUN_ROOT/submission.json" || { echo "missing TRL-00A submission receipt" >&2; exit 2; }
test ! -e "$CASE_DIR" && test ! -e "$FAILURE_DIR" || { echo "immutable TRL-00A output already exists" >&2; exit 2; }
mkdir "$CASE_DIR" "$FAILURE_DIR"

PAYLOAD=$CASE_DIR/trl00a-raw-payload.json
TENSORS=$CASE_DIR/trl00a-tensors.npz
QUERY_LEDGER=$CASE_DIR/request-ledger.json
RESULT=$CASE_DIR/results.json
SERVER_LOG=$CASE_DIR/policy-server.log
CLIENT_LOG=$CASE_DIR/reference-trajectory-lift-client.log
TEST_LOG=$CASE_DIR/allocation-focused-tests.log
OPENPI_TEST_LOG=$CASE_DIR/allocation-openpi-tests.log
LIBERO_TEST_LOG=$CASE_DIR/allocation-libero-tests.log
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
      '{schema_version:"1.0",artifact_role:"r05a_reference_trajectory_lift_canary_launch_failure",status:"failed",run_id:$run,case_id:"crfs-1069f29a8d76463a",stage:$stage,exit_code:($status|tonumber),slurm_array_job_id:$job,slurm_array_task_id:0,scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,policy_generated_action_steps_executed:0,teacher_generated_action_steps_executed:0}' >"$temporary" && mv "$temporary" "$FAILURE"
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
test "$ready" = true || { echo "TRL-00A host telemetry failed to start" >&2; exit 6; }

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
"$OPENPI_PYTHON" -m unittest discover -s tests -p 'test_reference_trajectory_lift.py' -v >"$OPENPI_TEST_LOG" 2>&1
"$LIBERO_PYTHON" -m unittest \
  tests.test_r05a_reference_trajectory_lift_preregistration \
  tests.test_r05a_reference_trajectory_lift_canary \
  tests.test_r05a_reference_trajectory_lift_validation \
  tests.test_r05a_reference_trajectory_lift_hpc_contract \
  -v >"$LIBERO_TEST_LOG" 2>&1
{
  printf '%s\n' '[openpi-python]'
  cat "$OPENPI_TEST_LOG"
  printf '%s\n' '[libero-python]'
  cat "$LIBERO_TEST_LOG"
} >"$TEST_LOG"
observed_openpi_tests=$(sed -n 's/^Ran \([0-9][0-9]*\) tests\{0,1\} in .*/\1/p' "$OPENPI_TEST_LOG")
observed_libero_tests=$(sed -n 's/^Ran \([0-9][0-9]*\) tests\{0,1\} in .*/\1/p' "$LIBERO_TEST_LOG")
for observed in "$observed_openpi_tests" "$observed_libero_tests"; do
  case "$observed" in ''|0) echo "TRL-00A focused tests did not execute" >&2; exit 2 ;; esac
done
for log in "$OPENPI_TEST_LOG" "$LIBERO_TEST_LOG"; do
  test "$(grep -xc 'OK' "$log")" = 1 || { echo "TRL-00A focused tests failed" >&2; exit 2; }
  if grep -Eq 'skipped=|^FAILED|^ERROR' "$log"; then echo "TRL-00A focused tests skipped or failed" >&2; exit 2; fi
done
observed_tests=$((observed_openpi_tests + observed_libero_tests))

ALLOCATED_GPU_UUID=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
case "$ALLOCATED_GPU_UUID" in GPU-*) ;; *) echo "cannot bind TRL-00A GPU UUID" >&2; exit 2 ;; esac
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

FAILURE_STAGE=reference_trajectory_lift_payload
"$LIBERO_PYTHON" "$REMOTE_REPO/main/run_crfs_r05a_reference_trajectory_lift_canary.py" \
  --manifest "$MANIFEST" --config "$CONFIG" --legacy-config "$LEGACY_CONFIG" \
  --r02-raw-root "$R02_RAW_ROOT" --output-root "$EXPERIMENT_ROOT" --run-id "$RUN_ID" \
  --case-index 0 --host 127.0.0.1 --port "$PORT" --checkpoint-id "$CHECKPOINT_DIR" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" >"$CLIENT_LOG" 2>&1
test -f "$PAYLOAD" || { echo "TRL-00A payload was not written" >&2; exit 5; }
test -f "$TENSORS" || { echo "TRL-00A tensor archive was not written" >&2; exit 5; }
test -f "$QUERY_LEDGER" || { echo "TRL-00A query ledger was not written" >&2; exit 5; }
jq -e --arg run "$RUN_ID" --arg job "$SLURM_ARRAY_JOB_ID" --arg tensor_sha "$(sha256sum "$TENSORS" | awk '{print $1}')" '
  .schema_version == "1.0"
  and .payload_type == "r05a_reference_trajectory_lift_raw_payload"
  and .status == "raw_evidence_complete"
  and .branch == "finite"
  and .run_id == $run and .case_id == "crfs-1069f29a8d76463a"
  and .group_id == "safelibero_spatial:II:0:46"
  and (.request_ledger | type=="array" and length==18)
  and .request_accounting == {exact:true,total:18,finite_complete:true}
  and (.raw | keys | sort) == [
    "branch",
    "duplicate_exact",
    "evaluations",
    "ordinary_replay_exact",
    "replay_envelope"
  ]
  and .raw.branch == "finite"
  and (.raw.evaluations | type=="array" and length==2)
  and .raw.duplicate_exact == true
  and .raw.ordinary_replay_exact == true
  and .artifacts.tensor_archive == "trl00a-tensors.npz"
  and .artifacts.tensor_archive_sha256 == $tensor_sha
  and .artifacts.request_ledger == "request-ledger.json"
  and .artifacts.results_json_published == false
  and (.provenance.slurm_array_job_id|tostring) == $job
  and (.provenance.slurm_array_task_id|tostring) == "0"
  and .provenance.host == "worker-1"
  and .execution_boundary.policy_requests == 18
  and .execution_boundary.policy_generated_action_steps_executed == 0
  and .execution_boundary.teacher_generated_action_steps_executed == 0
  and .execution_boundary.efficacy_rollouts_executed == 0
  and .execution_boundary.simulator_efficacy_evaluated == false
' "$PAYLOAD" >/dev/null || { echo "TRL-00A raw payload contract changed" >&2; exit 5; }
jq -e --arg tensor_sha "$(sha256sum "$TENSORS" | awk '{print $1}')" '
  .schema_version == "1.0"
  and .payload_type == "r05a_reference_trajectory_lift_request_ledger"
  and .case_id == "crfs-1069f29a8d76463a"
  and .branch == "finite"
  and .exact_policy_request_count == 18
  and (.rows | type=="array" and length==18)
  and .tensor_archive_sha256 == $tensor_sha
' "$QUERY_LEDGER" >/dev/null || { echo "TRL-00A query ledger contract changed" >&2; exit 5; }
kill -0 "$SERVER_PID" 2>/dev/null || { echo "ordinary policy server died during TRL-00A" >&2; exit 5; }
if grep -Eiq 'out[ -]?of[ -]?memory|outofmemoryerror|cuda([^[:alnum:]]|_)*(oom|error[^[:alnum:]]*2)|cublas_status_alloc_failed|cudnn_status_alloc_failed|std::bad_alloc|cannot allocate memory' "$SERVER_LOG" "$CLIENT_LOG"; then
  echo "TRL-00A log indicates an out-of-memory failure" >&2; exit 5
fi

kill "$SERVER_PID"
if wait "$SERVER_PID"; then server_status=0; else server_status=$?; fi
SERVER_PID=
case "$server_status" in 0|143) ;; *) echo "ordinary policy server cleanup status changed: $server_status" >&2; exit 5 ;; esac
crfs_write_cgroup_v2_full_lifetime_marker "$POLICY_CLEAN"
: >"$GPU_STOP"
wait "$GPU_PID" || { GPU_PID=; echo "TRL-00A GPU monitor failed" >&2; exit 6; }
GPU_PID=
crfs_write_cgroup_v2_full_lifetime_marker "$GPU_CLEAN"

FAILURE_STAGE=host_telemetry_seal
crfs_write_cgroup_v2_full_lifetime_marker "$WORKLOAD_CLEAN"
: >"$HOST_STOP"
wait "$HOST_PID" || { HOST_PID=; echo "TRL-00A host telemetry failed" >&2; exit 6; }
HOST_PID=
test -s "$HOST_TELEMETRY" || { echo "TRL-00A host telemetry was not sealed" >&2; exit 6; }
"$LIBERO_PYTHON" - "$HOST_TELEMETRY" "$SLURM_ARRAY_JOB_ID" <<'PY'
import sys

from crfs_oracle.r05a_full_lifetime_telemetry import parse_full_lifetime_telemetry

summary = parse_full_lifetime_telemetry(sys.argv[1], expected_job_id=sys.argv[2])
if summary.get("contract_passed") is not True:
    raise SystemExit("TRL-00A allocation-side host telemetry contract failed")
print(f"allocation_host_telemetry_sha256={summary['raw_trace_sha256']}")
PY
test ! -e "$RESULT" && test ! -e "$CASE_DIR/.results.candidate.json" || { echo "TRL-00A GPU job must not publish results.json" >&2; exit 7; }

echo "payload=$PAYLOAD"
echo "payload_sha256=$(sha256sum "$PAYLOAD" | awk '{print $1}')"
echo "tensors=$TENSORS"
echo "tensors_sha256=$(sha256sum "$TENSORS" | awk '{print $1}')"
echo "query_ledger=$QUERY_LEDGER"
echo "query_ledger_sha256=$(sha256sum "$QUERY_LEDGER" | awk '{print $1}')"
echo "focused_tests=$observed_tests"
echo "host_telemetry_sha256=$(sha256sum "$HOST_TELEMETRY" | awk '{print $1}')"
FAILURE_STAGE=complete
