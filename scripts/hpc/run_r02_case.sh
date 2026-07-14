#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?run_r02_case.sh must execute inside a Slurm allocation}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${OPENPI_PYTHON:=/mnt/data/quanth/venvs/openpi/bin/python}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"
: "${OPENPI_DATA_HOME:=/mnt/data/quanth/cache/openpi}"
: "${CHECKPOINT_DIR:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${RUN_ID:?RUN_ID must be an explicit immutable R02 run identifier}"
: "${MANIFEST:?MANIFEST is required}"
: "${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is required; R02 has no implicit config}"
: "${R01_RAW_ROOT:?R01_RAW_ROOT is required}"
: "${R01_SUMMARY:?R01_SUMMARY is required}"
: "${R01_SUMMARY_SHA256:?R01_SUMMARY_SHA256 is required}"
: "${PARITY_ARTIFACT:?PARITY_ARTIFACT is required}"
: "${PARITY_ARTIFACT_SHA256:?PARITY_ARTIFACT_SHA256 is required}"
: "${CASE_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"
export RUN_ID CASE_INDEX R01_SUMMARY_SHA256 PARITY_ARTIFACT_SHA256
if [ -z "${PORT:-}" ]; then
  PORT=$((8230 + ${SLURM_ARRAY_TASK_ID:-0}))
fi

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
    exit 2
    ;;
esac
case "$CASE_INDEX" in
  *[!0-9]*|'') echo "CASE_INDEX must be a nonnegative integer" >&2; exit 2 ;;
esac
case "$R01_SUMMARY_SHA256" in
  *[!0-9a-f]*|'') echo "R01_SUMMARY_SHA256 must be lowercase hexadecimal" >&2; exit 2 ;;
esac
case "$PARITY_ARTIFACT_SHA256" in
  *[!0-9a-f]*|'') echo "PARITY_ARTIFACT_SHA256 must be lowercase hexadecimal" >&2; exit 2 ;;
esac
test "${#R01_SUMMARY_SHA256}" -eq 64 || { echo "R01_SUMMARY_SHA256 must have 64 characters" >&2; exit 2; }
test "${#PARITY_ARTIFACT_SHA256}" -eq 64 || { echo "PARITY_ARTIFACT_SHA256 must have 64 characters" >&2; exit 2; }

RUNNER=$REMOTE_REPO/main/run_crfs_r02.py
MODEL=$CHECKPOINT_DIR/model.safetensors
test -f "$RUNNER" || { echo "missing R02 runner: $RUNNER" >&2; exit 2; }
test -f "$MANIFEST" || { echo "missing frozen manifest: $MANIFEST" >&2; exit 2; }
test -f "$EXPERIMENT_CONFIG" || { echo "missing external R02 config: $EXPERIMENT_CONFIG" >&2; exit 2; }
test -d "$R01_RAW_ROOT" || { echo "missing R01 raw root: $R01_RAW_ROOT" >&2; exit 2; }
test -f "$R01_SUMMARY" || { echo "missing R01 summary: $R01_SUMMARY" >&2; exit 2; }
test -f "$PARITY_ARTIFACT" || { echo "missing parity artifact: $PARITY_ARTIFACT" >&2; exit 2; }
test -f "$MODEL" || { echo "missing converted checkpoint: $MODEL" >&2; exit 2; }

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
if not 0 <= index < len(records):
    raise SystemExit(f"case index {index} outside manifest length {len(records)}")
case_id = records[index].get("case_id")
if not isinstance(case_id, str) or not case_id:
    raise SystemExit("manifest case has no case_id")
print(case_id)
PY
)
export CASE_ID

OBSERVED_R01_SUMMARY_SHA256=$(sha256sum "$R01_SUMMARY" | awk '{print $1}')
OBSERVED_PARITY_ARTIFACT_SHA256=$(sha256sum "$PARITY_ARTIFACT" | awk '{print $1}')
test "$OBSERVED_R01_SUMMARY_SHA256" = "$R01_SUMMARY_SHA256" || {
  echo "R01 summary hash mismatch" >&2
  exit 2
}
test "$OBSERVED_PARITY_ARTIFACT_SHA256" = "$PARITY_ARTIFACT_SHA256" || {
  echo "sampler parity artifact hash mismatch" >&2
  exit 2
}

CHECKPOINT_SHA256=$(sha256sum "$MODEL" | awk '{print $1}')
MANIFEST_SHA256=$(sha256sum "$MANIFEST" | awk '{print $1}')
CONFIG_SHA256=$(sha256sum "$EXPERIMENT_CONFIG" | awk '{print $1}')
CASE_DIR=$EXPERIMENT_ROOT/$RUN_ID/case-$CASE_INDEX
mkdir -p "$CASE_DIR"
SERVER_LOG=$CASE_DIR/policy-server.log
CLIENT_LOG=$CASE_DIR/r02-client.log
FAILURE=$CASE_DIR/launch-failure.json

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
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-none}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "repo=$REMOTE_REPO"
echo "commit=$(git -C "$REMOTE_REPO" rev-parse HEAD)"
echo "dirty=$(test -n "$(git -C "$REMOTE_REPO" status --porcelain)" && echo true || echo false)"
echo "run_id=$RUN_ID"
echo "case_index=$CASE_INDEX"
echo "manifest_sha256=$MANIFEST_SHA256"
echo "config_sha256=$CONFIG_SHA256"
echo "r01_summary_sha256=$R01_SUMMARY_SHA256"
echo "parity_artifact_sha256=$PARITY_ARTIFACT_SHA256"
echo "checkpoint_sha256=$CHECKPOINT_SHA256"
"$OPENPI_PYTHON" -V
"$OPENPI_PYTHON" -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'

SERVER_PID=
FAILURE_STAGE=policy_server_startup
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$status" -ne 0 ]; then
    "$LIBERO_PYTHON" - "$FAILURE" "$status" "$FAILURE_STAGE" "$MANIFEST_SHA256" "$CONFIG_SHA256" "$CHECKPOINT_SHA256" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
value = {
    "status": "failed",
    "exit_code": int(sys.argv[2]),
    "stage": sys.argv[3],
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "run_id": os.environ["RUN_ID"],
    "case_index": int(os.environ["CASE_INDEX"]),
    "case_id": os.environ["CASE_ID"],
    "manifest_sha256": sys.argv[4],
    "config_sha256": sys.argv[5],
    "checkpoint_sha256": sys.argv[6],
    "r01_summary_sha256": os.environ["R01_SUMMARY_SHA256"],
    "parity_artifact_sha256": os.environ["PARITY_ARTIFACT_SHA256"],
    "policy_server_log": str(path.parent / "policy-server.log"),
    "client_log": str(path.parent / "r02-client.log"),
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

TRANSFORMERS_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh)
export PYTHONPATH=$TRANSFORMERS_OVERLAY:$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src
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

FAILURE_STAGE=r02_runner
export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/packages/openpi-client/src
cd "$REMOTE_REPO"
"$LIBERO_PYTHON" main/run_crfs_r02.py \
  --manifest "$MANIFEST" \
  --config "$EXPERIMENT_CONFIG" \
  --r01-raw-root "$R01_RAW_ROOT" \
  --r01-summary "$R01_SUMMARY" \
  --r01-summary-sha256 "$R01_SUMMARY_SHA256" \
  --parity-artifact "$PARITY_ARTIFACT" \
  --parity-artifact-sha256 "$PARITY_ARTIFACT_SHA256" \
  --output-root "$EXPERIMENT_ROOT" \
  --run-id "$RUN_ID" \
  --case-index "$CASE_INDEX" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --checkpoint-id "$CHECKPOINT_DIR" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" >"$CLIENT_LOG" 2>&1

FAILURE_STAGE=complete
