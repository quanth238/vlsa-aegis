#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?R06 capture workload must run inside Slurm}"
: "${SLURM_ARRAY_JOB_ID:?R06 capture workload requires an array job id}"
: "${SLURM_ARRAY_TASK_ID:?R06 capture workload requires an array task id}"
: "${RUN_ID:?immutable R06 capture run id is required}"
: "${EXPECTED_GIT_COMMIT:?reviewed R06 capture commit is required}"
: "${EXPECTED_CONFIG_SHA256:?released R06 capture config hash is required}"
: "${R06_CAPTURE_SOURCE_CONTRACT:?R06 capture source contract is required}"
: "${R06_CAPTURE_SUBMISSION_RECEIPT:?R06 capture submission receipt is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${CHECKPOINT_DIR:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
: "${R02_RAW_ROOT:=/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a}"

OPENPI_PYTHON=/mnt/data/quanth/venvs/openpi/bin/python
LIBERO_PYTHON=/mnt/data/quanth/venvs/openpi-libero-client/bin/python
CONFIG=$REMOTE_REPO/configs/experiments/r06_aegis_collision_conditioned.json
MANIFEST=$REMOTE_REPO/manifests/oracle_h05_colliding.jsonl
R02_CONFIG=$REMOTE_REPO/configs/experiments/r02_oracle_flow.json
RUNNER=$REMOTE_REPO/main/run_crfs_r06_aegis_capture.py
MODEL=$CHECKPOINT_DIR/model.safetensors
CHECKPOINT_CONFIG=$CHECKPOINT_DIR/config.json
NORMALIZATION_ASSET=$CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json
SOURCE_R02=$R02_RAW_ROOT/crfs-1069f29a8d76463a/r02-paired.json
CASE_ID=crfs-1069f29a8d76463a
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
CASE_DIR=$RUN_ROOT/$CASE_ID
ASSET_DIR=$CASE_DIR/codex-label-capture
RESULT=$CASE_DIR/capture.json
SERVER_LOG=$CASE_DIR/policy-server.log
CLIENT_LOG=$CASE_DIR/capture-client.log
VALIDATION_LOG=$CASE_DIR/capture-validation.log
FAILURE=$RUN_ROOT/capture-failure-task-0.json
SERVER_PID=
FAILURE_STAGE=allocation_contract

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$status" -ne 0 ] && test -d "$RUN_ROOT" && test ! -e "$FAILURE"; then
    failure_tmp=$(mktemp "$RUN_ROOT/.capture-failure.XXXXXX")
    jq -n \
      --arg run "$RUN_ID" \
      --arg case "$CASE_ID" \
      --arg stage "$FAILURE_STAGE" \
      --arg job "$SLURM_JOB_ID" \
      --arg array "$SLURM_ARRAY_JOB_ID" \
      --arg task "$SLURM_ARRAY_TASK_ID" \
      --arg commit "$EXPECTED_GIT_COMMIT" \
      --arg config_sha "$EXPECTED_CONFIG_SHA256" \
      --arg host "$(hostname -s)" \
      --argjson exit_code "$status" \
      --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
      {
        schema_version:"1.0",
        artifact_role:"r06_aegis_label_capture_failure",
        status:"apparatus_inconclusive",
        run_id:$run,
        case_id:$case,
        failure_stage:$stage,
        exit_code:$exit_code,
        git_commit:$commit,
        config_sha256:$config_sha,
        slurm_job_id:$job,
        slurm_array_job_id:$array,
        slurm_array_task_id:$task,
        host:$host,
        aegis_executed:false,
        qp_steps:0,
        collision_or_safety_claim_allowed:false,
        probe_or_mlp_training_authorized:false,
        timestamp_utc:$now
      }' >"$failure_tmp"
    mv "$failure_tmp" "$FAILURE"
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test "$SLURM_ARRAY_TASK_ID" = 0 || { echo "R06 capture is fixed to row zero" >&2; exit 2; }
test "$(hostname -s)" = worker-1 || { echo "R06 capture must remain on worker-1" >&2; exit 2; }
for launcher in "$OPENPI_PYTHON" "$LIBERO_PYTHON"; do
  test -x "$launcher" || { echo "missing canonical R06 interpreter launcher: $launcher" >&2; exit 2; }
done
for path in \
  "$CONFIG" "$MANIFEST" "$R02_CONFIG" \
  "$RUNNER" "$MODEL" "$CHECKPOINT_CONFIG" "$NORMALIZATION_ASSET" "$SOURCE_R02" \
  "$R06_CAPTURE_SOURCE_CONTRACT" "$R06_CAPTURE_SUBMISSION_RECEIPT"; do
  test -e "$path" && test ! -L "$path" || { echo "missing or symlinked R06 capture input: $path" >&2; exit 2; }
done
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256"
test "$(sha256sum "$MODEL" | awk '{print $1}')" = 988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
test "$(sha256sum "$CHECKPOINT_CONFIG" | awk '{print $1}')" = 5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a
test "$(sha256sum "$NORMALIZATION_ASSET" | awk '{print $1}')" = b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84
test "$(sha256sum "$SOURCE_R02" | awk '{print $1}')" = 055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593
image_convention_record=$(MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa "$LIBERO_PYTHON" - <<'PY'
from importlib import metadata

import robosuite.macros as macros

value = getattr(macros, "IMAGE_CONVENTION", None)
if not isinstance(value, str):
    raise SystemExit("robosuite IMAGE_CONVENTION is unavailable")
print(f"R06_IMAGE_CONVENTION={value}")
print(f"R06_ROBOSUITE_VERSION={metadata.version('robosuite')}")
PY
)
R06_ROBOSUITE_IMAGE_CONVENTION=$(printf '%s\n' "$image_convention_record" | sed -n 's/^R06_IMAGE_CONVENTION=//p')
R06_ROBOSUITE_VERSION=$(printf '%s\n' "$image_convention_record" | sed -n 's/^R06_ROBOSUITE_VERSION=//p')
test "$(printf '%s\n' "$R06_ROBOSUITE_IMAGE_CONVENTION" | wc -l | tr -d ' ')" = 1 || {
  echo "R06 capture could not extract one live robosuite image convention" >&2
  exit 2
}
test "$R06_ROBOSUITE_IMAGE_CONVENTION" = opengl || {
  echo "R06 capture requires the frozen robosuite OpenGL image convention, found: $R06_ROBOSUITE_IMAGE_CONVENTION" >&2
  exit 2
}
test "$R06_ROBOSUITE_VERSION" = 1.4.1 || {
  echo "R06 capture requires robosuite 1.4.1, found: $R06_ROBOSUITE_VERSION" >&2
  exit 2
}
export R06_ROBOSUITE_IMAGE_CONVENTION R06_ROBOSUITE_VERSION
jq -e --arg run "$RUN_ID" '
  .ready_to_run == true
  and .blocked_on == []
  and .execution_release.artifact_role == "r06_aegis_codex_label_capture_execution_release"
  and .execution_release.stage == "codex_label_capture"
  and .execution_release.run_id == $run
  and (.execution_release.accepted_implementation_commit | type == "string" and length == 40)
  and (.execution_release.decision_artifact | type == "string" and endswith("-release-aegis-label-capture-canary.md"))
  and .execution_release.allowed_release_diff_paths == [
    "configs/experiments/r06_aegis_collision_conditioned.json",
    .execution_release.decision_artifact
  ]
  and .execution_release.single_case_index == 0
  and .execution_release.case_id == "crfs-1069f29a8d76463a"
  and .execution_release.aegis_execution_allowed == false
  and .execution_release.semantic_label_required == false
  and .execution_release.groundingdino_execution_allowed == false
  and .execution_release.qp_execution_allowed == false
  and .execution_release.robosuite_image_convention == "opengl"
  and .execution_release.robosuite_version == "1.4.1"
  and .execution_release.probe_or_mlp_training_authorized == false
  and .execution_release.automatic_population_launch_authorized == false
  and .execution_release.release_only_parent_required == true
  ' "$CONFIG" >/dev/null || { echo "R06 capture release is not capture-only" >&2; exit 2; }

test -d "$RUN_ROOT" && test ! -L "$RUN_ROOT" || { echo "R06 immutable run root was not reserved" >&2; exit 2; }
test ! -e "$CASE_DIR" || { echo "R06 capture case directory is already used" >&2; exit 2; }
mkdir "$CASE_DIR"

PORT=$((20000 + SLURM_ARRAY_JOB_ID % 30000))
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
mkdir "$LIBERO_CONFIG_PATH"

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

TRANSFORMERS_OVERLAY=$("$REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh")
export PYTHONPATH=$TRANSFORMERS_OVERLAY:$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src
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

export PYTHONPATH=$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/packages/openpi-client/src
cd "$REMOTE_REPO"
FAILURE_STAGE=capture_runner
"$LIBERO_PYTHON" "$RUNNER" \
  --manifest "$MANIFEST" \
  --config "$CONFIG" \
  --r02-config "$R02_CONFIG" \
  --output "$RESULT" \
  --asset-dir "$ASSET_DIR" \
  --run-id "$RUN_ID" \
  --case-index 0 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --checkpoint-id "$CHECKPOINT_DIR" \
  --checkpoint-model "$MODEL" \
  --checkpoint-sha256 988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed \
  --checkpoint-config "$CHECKPOINT_CONFIG" \
  --checkpoint-config-sha256 5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a \
  --normalization-asset "$NORMALIZATION_ASSET" \
  --normalization-asset-sha256 b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84 \
  >"$CLIENT_LOG" 2>&1

FAILURE_STAGE=capture_validation
"$LIBERO_PYTHON" - "$RESULT" "$ASSET_DIR" >"$VALIDATION_LOG" <<'PY'
import hashlib
import json
import math
import os
import pathlib
import sys

import imageio.v3 as imageio
import numpy as np

from crfs_oracle.aegis_pairing import exact_array_record
from crfs_oracle.aegis_runner import validate_aegis_label_capture

result_path = pathlib.Path(sys.argv[1])
asset_dir = pathlib.Path(sys.argv[2]).resolve()
value = json.loads(result_path.read_text(encoding="utf-8"))
errors = list(validate_aegis_label_capture(value))
execution = value.get("execution", {})
if not (
    isinstance(execution, dict)
    and execution.get("run_id") == os.environ["RUN_ID"]
    and execution.get("slurm_job_id") == os.environ["SLURM_JOB_ID"]
    and execution.get("slurm_array_job_id") == os.environ["SLURM_ARRAY_JOB_ID"]
    and str(execution.get("slurm_array_task_id")) == os.environ["SLURM_ARRAY_TASK_ID"]
    and execution.get("stage") == "codex_label_capture"
):
    errors.append("capture execution identity differs from the allocation")
ledger = value.get("settle_ledger", {}).get("ledger", [])
selection = value.get("settle_selection", {})
binding = value.get("selected_branch_binding", {})
if not (
    isinstance(ledger, list)
    and len(ledger) == 21
    and [row.get("boundary_index") for row in ledger] == list(range(21))
    and selection.get("selected_boundary_index") == 20
    and binding.get("checks")
    and all(binding["checks"].values())
    and binding.get("selected_integration_state") == ledger[20].get("integration_state")
    and binding.get("selected_active_obstacle")
    == value.get("settle_ledger", {}).get("active_obstacle")
    and binding.get("selected_D_sim_m") == ledger[20].get("D_sim_m")
    and binding.get("selected_contact") == ledger[20].get("contact")
):
    errors.append("capture selected-ledger/reference binding is incomplete")
baseline = value.get("pi05_baseline", {})
repeats = baseline.get("repeats", []) if isinstance(baseline, dict) else []
if not (
    len(repeats) == 2
    and repeats[0] == repeats[1]
    and baseline.get("repeat_exact") is True
    and baseline.get("collision_reproduced_both") is True
):
    errors.append("capture does not contain two exact baseline repeats")
else:
    for index, repeat in enumerate(repeats):
        nominal = np.asarray(repeat.get("nominal_actions"))
        executed = np.asarray(repeat.get("executed_actions"))
        if not (
            repeat.get("measurement_samples") == 126
            and repeat.get("measurement", {}).get("samples") == 126
            and repeat.get("frozen_collision_definition") is True
            and isinstance(repeat.get("contact"), bool)
            and isinstance(repeat.get("minimum_D_sim_m"), (int, float))
            and math.isfinite(float(repeat["minimum_D_sim_m"]))
            and nominal.shape == (5, 7)
            and executed.shape == (5, 7)
            and np.isfinite(nominal).all()
            and np.isfinite(executed).all()
            and isinstance(repeat.get("reach"), dict)
            and math.isfinite(float(repeat["reach"]["reach_progress_m"]))
            and isinstance(repeat.get("outcomes"), dict)
        ):
            errors.append(f"baseline repeat {index} lacks full 126-sample/action/metric evidence")
orientation = value.get("perception_render", {}).get("camera_orientation", {})
anchors = orientation.get("anchors", {}) if isinstance(orientation, dict) else {}
if not (
    orientation.get("robosuite_version") == "1.4.1"
    and orientation.get("live_image_convention") == "opengl"
    and orientation.get("same_state_224_observable_anchor_passed") is True
    and orientation.get("same_state_224_policy_agent_anchor_passed") is True
    and set(anchors) == {"agentview", "backview"}
    and all(
        anchors[camera].get("image_bytes_equal") is True
        and anchors[camera].get("depth_bytes_equal") is True
        for camera in ("agentview", "backview")
    )
):
    errors.append("capture lacks the allocation-anchored camera orientation mapping")
for record in value.get("assets", {}).values():
    if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
        errors.append("capture asset record is incomplete")
        continue
    path = pathlib.Path(record["path"]).resolve()
    try:
        path.relative_to(asset_dir)
    except ValueError:
        errors.append(f"capture asset escapes asset directory: {path}")
        continue
    if not path.is_file():
        errors.append(f"capture asset is missing: {path}")
        continue
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        errors.append(f"capture asset hash changed: {path}")
for key in (
    "agentview_image",
    "agentview_depth",
    "backview_image",
    "backview_depth",
):
    asset_record = value.get("assets", {}).get(f"{key}_npy", {})
    render_record = value.get("perception_render", {}).get("views", {}).get(key)
    if not isinstance(asset_record, dict) or not isinstance(render_record, dict):
        errors.append(f"{key} lacks both asset and perception-render records")
        continue
    source = np.load(asset_record["path"], allow_pickle=False)
    reconstructed = exact_array_record(source, label=key)
    if asset_record.get("array") != reconstructed:
        errors.append(f"{key} NPY differs from its asset exact-array record")
    if render_record != reconstructed:
        errors.append(f"{key} NPY differs from its perception-render exact-array record")
for camera in ("agentview", "backview"):
    npy_record = value.get("assets", {}).get(f"{camera}_image_npy", {})
    png_record = value.get("assets", {}).get(f"{camera}_image_png", {})
    if isinstance(npy_record, dict) and isinstance(png_record, dict):
        source = np.load(npy_record["path"], allow_pickle=False)
        decoded = imageio.imread(png_record["path"])
        if source.dtype != decoded.dtype or source.shape != decoded.shape or not np.array_equal(source, decoded):
            errors.append(f"{camera} PNG is not a lossless representation of the captured RGB array")
if errors:
    raise SystemExit("; ".join(errors))
print("capture_valid=true")
print("aegis_executed=false")
print("qp_steps=0")
print(
    "codex_review_agentview_array_sha256="
    + value["perception_render"]["views"]["agentview_image"]["sha256"]
)
PY

echo "capture_result=$RESULT"
echo "capture_result_sha256=$(sha256sum "$RESULT" | awk '{print $1}')"
echo "capture_validation=$VALIDATION_LOG"
FAILURE_STAGE=complete
