#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?R06 paired workload must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?R06 paired workload requires an array job id}"
: "${SLURM_ARRAY_TASK_ID:?R06 paired workload requires an array task id}"
: "${RUN_ID:?R06 paired run id is required}"
: "${EXPECTED_GIT_COMMIT:?R06 paired release commit is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${CHECKPOINT_DIR:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"

OPENPI_PYTHON=/mnt/data/quanth/venvs/openpi/bin/python
LIBERO_PYTHON=/mnt/data/quanth/venvs/openpi-libero-client/bin/python
CONFIG=$REMOTE_REPO/configs/experiments/r06_aegis_collision_conditioned.json
MANIFEST=$REMOTE_REPO/manifests/oracle_h05_colliding.jsonl
LABEL_MANIFEST=$REMOTE_REPO/manifests/r06_codex_obstacle_labels_canary.jsonl
R02_CONFIG=$REMOTE_REPO/configs/experiments/r02_oracle_flow.json
SOURCE_R02=/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a/crfs-1069f29a8d76463a/r02-paired.json
CAPTURE=/mnt/data/quanth/experiments/crfs-oracle/r06-aegis-label-capture-canary-20260717b/crfs-1069f29a8d76463a/capture.json
DINO_CONFIG=/mnt/data/quanth/cache/uv/archive-v0/hHOpLbugg_lAlUaF/groundingdino/config/GroundingDINO_SwinT_OGC.py
DINO_CHECKPOINT=/mnt/data/quanth/cache/aegis/groundingdino/groundingdino_swint_ogc.pth
MODEL=$CHECKPOINT_DIR/model.safetensors
CHECKPOINT_CONFIG=$CHECKPOINT_DIR/config.json
NORMALIZATION=$CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json
RUNNER=$REMOTE_REPO/main/run_crfs_r06_aegis_paired_canary.py
CASE_ID=crfs-1069f29a8d76463a
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
CASE_DIR=$RUN_ROOT/$CASE_ID
RESULT=$CASE_DIR/results.json
PERCEPTION_DIR=$CASE_DIR/aegis-perception
SERVER_LOG=$CASE_DIR/policy-server.log
CLIENT_LOG=$CASE_DIR/paired-client.log
TEST_LOG=$CASE_DIR/allocation-focused-tests.log
FAILURE=$RUN_ROOT/gpu-apparatus-failure-task-0.json
SERVER_PID=
FAILURE_STAGE=allocation_contract

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$status" -ne 0 ] && test ! -e "$FAILURE"; then
    temporary=$(mktemp "$RUN_ROOT/.r06-paired-failure.XXXXXX")
    jq -n --arg run "$RUN_ID" --arg stage "$FAILURE_STAGE" \
      --arg task "${SLURM_ARRAY_JOB_ID}_0" --arg host "$(hostname -s)" \
      --arg commit "$EXPECTED_GIT_COMMIT" --argjson exit_code "$status" \
      --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{
        schema_version:"1.0",
        artifact_role:"r06_aegis_paired_gpu_apparatus_failure",
        status:"apparatus_inconclusive",
        run_id:$run,
        case_id:"crfs-1069f29a8d76463a",
        failure_stage:$stage,
        exact_gpu_task_id:$task,
        host:$host,
        git_commit:$commit,
        exit_code:$exit_code,
        scientific_claim_allowed:false,
        probe_or_mlp_training_authorized:false,
        timestamp_utc:$now
      }' >"$temporary"
    mv "$temporary" "$FAILURE"
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test "$SLURM_ARRAY_TASK_ID" = 0
test "$(hostname -s)" = worker-1
for path in "$CONFIG" "$MANIFEST" "$LABEL_MANIFEST" "$R02_CONFIG" "$SOURCE_R02" \
  "$CAPTURE" "$DINO_CONFIG" "$DINO_CHECKPOINT" "$MODEL" "$CHECKPOINT_CONFIG" \
  "$NORMALIZATION" "$RUNNER"; do
  test -f "$path" && test ! -L "$path" || { echo "missing paired runtime input: $path" >&2; exit 2; }
done
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41
test "$(sha256sum "$LABEL_MANIFEST" | awk '{print $1}')" = 6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f
test "$(sha256sum "$R02_CONFIG" | awk '{print $1}')" = c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e
test "$(sha256sum "$SOURCE_R02" | awk '{print $1}')" = 055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593
test "$(sha256sum "$CAPTURE" | awk '{print $1}')" = f2d32024ac6fac5aa5d133b5138df72501f1590b8cb0927b732edde69dabaaac
test "$(sha256sum "$DINO_CONFIG" | awk '{print $1}')" = 172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1
test "$(sha256sum "$DINO_CHECKPOINT" | awk '{print $1}')" = 3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799
test "$(sha256sum "$MODEL" | awk '{print $1}')" = 988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
test "$(sha256sum "$CHECKPOINT_CONFIG" | awk '{print $1}')" = 5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a
test "$(sha256sum "$NORMALIZATION" | awk '{print $1}')" = b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84
test -d "$RUN_ROOT" && test ! -L "$RUN_ROOT"
test ! -e "$CASE_DIR" || { echo "R06 paired case directory is already used" >&2; exit 2; }
mkdir "$CASE_DIR"

export OPENPI_DATA_HOME=/mnt/data/quanth/cache/openpi
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
export AEGIS_GROUNDING_DINO_CONFIG=$DINO_CONFIG
export AEGIS_GROUNDING_DINO_CHECKPOINT=$DINO_CHECKPOINT
mkdir "$LIBERO_CONFIG_PATH"
cd "$REMOTE_REPO"

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
with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as stream:
    for key, value in values.items():
        stream.write(f"{key}: {value}\n")
    temporary = stream.name
os.replace(temporary, destination)
PY

export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/packages/openpi-client/src
FAILURE_STAGE=allocation_dependency_preflight
"$LIBERO_PYTHON" - >"$TEST_LOG" 2>&1 <<'PY'
from importlib import metadata

import cv2
import cvxpy
import groundingdino
import matplotlib
import numpy
import open3d
import osqp
import robosuite.macros as macros
import scipy
import torch

assert torch.cuda.is_available()
assert getattr(macros, "IMAGE_CONVENTION", None) == "opengl"
assert metadata.version("robosuite") == "1.4.1"
from groundingdino.util.inference import load_model, predict
from crfs_oracle.aegis_perception import resolve_groundingdino_only_dependencies
from crfs_oracle.aegis_baseline import verify_upstream_sources
import os

dependencies = resolve_groundingdino_only_dependencies(
    os.environ,
    expected_config_sha256="172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1",
    expected_checkpoint_sha256="3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
    require_packages=True,
    require_cuda=True,
)
assert all(dependencies.packages.values())
verify_upstream_sources(".")
print("allocation_dependency_preflight=passed")
print("torch_cuda_available=true")
print("focused_test_skips=0")
PY

FAILURE_STAGE=allocation_focused_tests
"$LIBERO_PYTHON" -m unittest -v \
  tests.test_aegis_baseline tests.test_aegis_pairing tests.test_aegis_perception \
  tests.test_aegis_runner >>"$TEST_LOG" 2>&1
test "$(grep -Eic 'skipped[= :]|skip=' "$TEST_LOG")" = 0 || {
  echo "R06 allocation-focused tests contain skips" >&2; exit 2
}

PORT=$((20000 + SLURM_ARRAY_JOB_ID % 30000))
TRANSFORMERS_OVERLAY=$("$REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh")
export PYTHONPATH=$TRANSFORMERS_OVERLAY:$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src
FAILURE_STAGE=policy_server_startup
(
  cd "$REMOTE_REPO/openpi"
  "$OPENPI_PYTHON" scripts/serve_policy.py --port "$PORT" \
    policy:checkpoint --policy.config pi05_libero --policy.dir "$CHECKPOINT_DIR"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 200 "$SERVER_LOG" >&2; exit 3; fi
  if grep -q "server listening" "$SERVER_LOG"; then ready=1; break; fi
  sleep 1
done
test "$ready" = 1 || { tail -n 200 "$SERVER_LOG" >&2; exit 4; }

export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/packages/openpi-client/src
FAILURE_STAGE=paired_runner
"$LIBERO_PYTHON" "$RUNNER" \
  --manifest "$MANIFEST" --config "$CONFIG" --r02-config "$R02_CONFIG" \
  --source-r02 "$SOURCE_R02" --output "$RESULT" \
  --perception-output-dir "$PERCEPTION_DIR" --run-id "$RUN_ID" --case-index 0 \
  --host 127.0.0.1 --port "$PORT" --checkpoint-id "$CHECKPOINT_DIR" \
  --checkpoint-model "$MODEL" --checkpoint-config "$CHECKPOINT_CONFIG" \
  --normalization-asset "$NORMALIZATION" --label-manifest "$LABEL_MANIFEST" \
  --capture-artifact "$CAPTURE" --dino-config "$DINO_CONFIG" \
  --dino-checkpoint "$DINO_CHECKPOINT" >"$CLIENT_LOG" 2>&1

test -f "$RESULT" && test ! -L "$RESULT"
echo "paired_result=$RESULT"
echo "paired_result_sha256=$(sha256sum "$RESULT" | awk '{print $1}')"
echo "allocation_focused_tests=$TEST_LOG"
FAILURE_STAGE=complete
