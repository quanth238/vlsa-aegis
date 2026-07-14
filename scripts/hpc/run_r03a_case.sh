#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?run_r03a_case.sh must execute inside a Slurm allocation}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${OPENPI_PYTHON:=/mnt/data/quanth/venvs/openpi/bin/python}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"
: "${OPENPI_DATA_HOME:=/mnt/data/quanth/cache/openpi}"
: "${CHECKPOINT_DIR:=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${RUN_ID:?RUN_ID must be an explicit immutable R03A run identifier}"
: "${MANIFEST:?MANIFEST is required; R03A has no implicit manifest}"
: "${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is required; R03A has no implicit config}"
: "${R02_RAW_ROOT:?R02_RAW_ROOT is required}"
: "${R03_SUMMARY:?R03_SUMMARY is required}"
: "${R03_SUMMARY_SHA256:?R03_SUMMARY_SHA256 is required}"
: "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT must bind the reviewed source commit}"
: "${CASE_INDEX:=${SLURM_ARRAY_TASK_ID:-0}}"

EXPECTED_MANIFEST_SHA256=241f1b94a5434973dcfb235b43b9386ad15046ff87d35d5b0c34e295c2132916
EXPECTED_R03_SUMMARY_SHA256=dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e
EXPECTED_CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
    exit 2
    ;;
esac
case "$CASE_INDEX" in
  *[!0-9]*|'') echo "CASE_INDEX must be a nonnegative integer" >&2; exit 2 ;;
esac
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "$CASE_INDEX" != "$SLURM_ARRAY_TASK_ID" ]; then
  echo "CASE_INDEX must equal SLURM_ARRAY_TASK_ID when running as an array" >&2
  exit 2
fi
for binding in "$R03_SUMMARY_SHA256" "$EXPECTED_GIT_COMMIT"; do
  case "$binding" in
    *[!0-9a-f]*|'') echo "hash bindings must be lowercase hexadecimal" >&2; exit 2 ;;
  esac
done
test "${#R03_SUMMARY_SHA256}" -eq 64 || {
  echo "R03_SUMMARY_SHA256 must have 64 characters" >&2
  exit 2
}
case "${#EXPECTED_GIT_COMMIT}" in
  40|64) ;;
  *) echo "EXPECTED_GIT_COMMIT must be a 40- or 64-character Git object id" >&2; exit 2 ;;
esac
test "$R03_SUMMARY_SHA256" = "$EXPECTED_R03_SUMMARY_SHA256" || {
  echo "R03_SUMMARY_SHA256 differs from the accepted R03 evidence" >&2
  exit 2
}

if [ -z "${PORT:-}" ]; then
  PORT=$((20000 + SLURM_JOB_ID % 30000))
fi
case "$PORT" in
  *[!0-9]*|'') echo "PORT must be an integer" >&2; exit 2 ;;
esac
test "$PORT" -ge 1024 && test "$PORT" -le 65535 || {
  echo "PORT must be between 1024 and 65535" >&2
  exit 2
}

RUNNER=$REMOTE_REPO/main/run_crfs_r03a.py
MODEL=$CHECKPOINT_DIR/model.safetensors
CASE_DIR=$EXPERIMENT_ROOT/$RUN_ID/case-$CASE_INDEX
mkdir -p "$CASE_DIR"
SERVER_LOG=$CASE_DIR/policy-server.log
CLIENT_LOG=$CASE_DIR/r03a-client.log
FAILURE=$CASE_DIR/launch-failure.json

export RUN_ID CASE_INDEX R03_SUMMARY_SHA256
CASE_ID=unresolved
MANIFEST_SHA256=unavailable
CONFIG_SHA256=unavailable
CHECKPOINT_SHA256=unavailable
GIT_COMMIT=unavailable
GIT_DIRTY=unavailable
SOURCE_R02_RESULT=unavailable
SOURCE_R02_SHA256=unavailable
EXPECTED_SOURCE_R02_SHA256=unavailable
EXPECTED_RESULT=unresolved
SERVER_PID=
FAILURE_STAGE=allocation_contract
export CASE_ID

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$status" -ne 0 ]; then
    "$LIBERO_PYTHON" - \
      "$FAILURE" "$status" "$FAILURE_STAGE" \
      "$MANIFEST" "$MANIFEST_SHA256" \
      "$EXPERIMENT_CONFIG" "$CONFIG_SHA256" \
      "$CHECKPOINT_DIR" "$CHECKPOINT_SHA256" \
      "$R02_RAW_ROOT" "$SOURCE_R02_RESULT" "$SOURCE_R02_SHA256" \
      "$EXPECTED_SOURCE_R02_SHA256" \
      "$R03_SUMMARY" "$R03_SUMMARY_SHA256" \
      "$GIT_COMMIT" "$GIT_DIRTY" \
      "$SERVER_LOG" "$CLIENT_LOG" "$EXPECTED_RESULT" <<'PY'
from datetime import datetime, timezone
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
value = {
    "schema_version": "1.0",
    "artifact_role": "r03a_analytic_kill_test_launch_failure",
    "status": "failed",
    "scientific_claim_allowed": False,
    "exit_code": int(sys.argv[2]),
    "stage": sys.argv[3],
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "host": os.uname().nodename,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "run_id": os.environ["RUN_ID"],
    "case_index": int(os.environ["CASE_INDEX"]),
    "case_id": os.environ.get("CASE_ID", "unresolved"),
    "manifest": sys.argv[4],
    "manifest_sha256": sys.argv[5],
    "expected_manifest_sha256": "241f1b94a5434973dcfb235b43b9386ad15046ff87d35d5b0c34e295c2132916",
    "config": sys.argv[6],
    "config_sha256": sys.argv[7],
    "checkpoint_id": sys.argv[8],
    "checkpoint_sha256": sys.argv[9],
    "expected_checkpoint_sha256": "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
    "r02_raw_root": sys.argv[10],
    "source_r02_result": sys.argv[11],
    "source_r02_sha256": sys.argv[12],
    "expected_source_r02_sha256": sys.argv[13],
    "r03_summary": sys.argv[14],
    "r03_summary_sha256": sys.argv[15],
    "expected_r03_summary_sha256": "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e",
    "git_commit": sys.argv[16],
    "reviewed_git_commit": os.environ.get("EXPECTED_GIT_COMMIT"),
    "git_dirty": sys.argv[17],
    "policy_server_log": sys.argv[18],
    "client_log": sys.argv[19],
    "expected_result": sys.argv[20],
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
test -x "$OPENPI_PYTHON" || { echo "missing OpenPI Python: $OPENPI_PYTHON" >&2; exit 2; }
test -x "$LIBERO_PYTHON" || { echo "missing LIBERO Python: $LIBERO_PYTHON" >&2; exit 2; }
test -x "$REMOTE_REPO/scripts/hpc/prepare_transformers_overlay.sh" || {
  echo "missing transformers overlay helper" >&2
  exit 2
}
test -x "$REMOTE_REPO/scripts/hpc/prepare_jsonschema_overlay.sh" || {
  echo "missing JSON Schema overlay helper" >&2
  exit 2
}
test -f "$RUNNER" || { echo "missing R03A runner: $RUNNER" >&2; exit 2; }
test -f "$MANIFEST" || { echo "missing frozen R03A manifest: $MANIFEST" >&2; exit 2; }
test -f "$EXPERIMENT_CONFIG" || { echo "missing external R03A config: $EXPERIMENT_CONFIG" >&2; exit 2; }
test -d "$R02_RAW_ROOT" || { echo "missing raw R02 source root: $R02_RAW_ROOT" >&2; exit 2; }
test -f "$R03_SUMMARY" || { echo "missing accepted R03 summary: $R03_SUMMARY" >&2; exit 2; }
test -f "$MODEL" || { echo "missing converted checkpoint: $MODEL" >&2; exit 2; }

MANIFEST_SHA256=$(sha256sum "$MANIFEST" | awk '{print $1}')
CONFIG_SHA256=$(sha256sum "$EXPERIMENT_CONFIG" | awk '{print $1}')
CHECKPOINT_SHA256=$(sha256sum "$MODEL" | awk '{print $1}')
OBSERVED_R03_SUMMARY_SHA256=$(sha256sum "$R03_SUMMARY" | awk '{print $1}')
GIT_COMMIT=$(git -C "$REMOTE_REPO" rev-parse HEAD)
GIT_DIRTY=$(test -n "$(git -C "$REMOTE_REPO" status --porcelain)" && echo true || echo false)

test "$MANIFEST_SHA256" = "$EXPECTED_MANIFEST_SHA256" || {
  echo "R03A eligible manifest hash mismatch" >&2
  exit 2
}
test "$OBSERVED_R03_SUMMARY_SHA256" = "$R03_SUMMARY_SHA256" || {
  echo "R03 summary hash mismatch" >&2
  exit 2
}
test "$CHECKPOINT_SHA256" = "$EXPECTED_CHECKPOINT_SHA256" || {
  echo "checkpoint hash differs from the R03-bound checkpoint" >&2
  exit 2
}
test "$GIT_COMMIT" = "$EXPECTED_GIT_COMMIT" || {
  echo "source commit differs from reviewed submission" >&2
  exit 2
}
test "$GIT_DIRTY" = false || {
  echo "R03A requires a clean remote source tree" >&2
  exit 2
}

CASE_BINDING=$("$LIBERO_PYTHON" - \
  "$MANIFEST" "$CASE_INDEX" "$EXPERIMENT_CONFIG" "$R03_SUMMARY" \
  "$R02_RAW_ROOT" "$MANIFEST_SHA256" "$R03_SUMMARY_SHA256" \
  "$CHECKPOINT_SHA256" <<'PY'
import json
import pathlib
import re
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve()
index = int(sys.argv[2])
config_path = pathlib.Path(sys.argv[3]).resolve()
summary_path = pathlib.Path(sys.argv[4]).resolve()
r02_raw_root = pathlib.Path(sys.argv[5]).resolve()
manifest_sha256, summary_sha256, checkpoint_sha256 = sys.argv[6:9]

records = [
    json.loads(line)
    for line in manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(records) != 17:
    raise SystemExit(f"R03A manifest must contain 17 rows, found {len(records)}")
case_ids = [record.get("case_id") for record in records]
group_ids = [record.get("group_id") for record in records]
if not all(isinstance(item, str) and item for item in case_ids):
    raise SystemExit("R03A manifest contains an invalid case identity")
if not all(isinstance(item, str) and item for item in group_ids):
    raise SystemExit("R03A manifest contains an invalid state-group identity")
if len(set(case_ids)) != 17 or len(set(group_ids)) != 17:
    raise SystemExit("R03A manifest case/state groups must be unique")
if not 0 <= index < len(records):
    raise SystemExit(f"case index {index} outside manifest length {len(records)}")

config = json.loads(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit("R03A config must be an object")
if config.get("name") != "r03a_analytic_kill_test":
    raise SystemExit("R03A config name differs from the frozen experiment")
if config.get("ready_to_run") is not True or config.get("blocked_on") != []:
    raise SystemExit("R03A config is not ready with no blocked dependencies")
declared_manifest = config.get("manifest")
if not isinstance(declared_manifest, str) or pathlib.Path(declared_manifest).name != manifest_path.name:
    raise SystemExit("R03A config is not bound to the supplied manifest")
settings = config.get("r03a")
if not isinstance(settings, dict):
    raise SystemExit("R03A config has no r03a object")
if settings.get("eligible_case_count") != 17:
    raise SystemExit("R03A config eligible population size differs")
if settings.get("eligible_manifest_sha256") != manifest_sha256:
    raise SystemExit("R03A config eligible manifest hash differs")
if settings.get("source_r03_summary_sha256") != summary_sha256:
    raise SystemExit("R03A config R03 summary hash differs")
if settings.get("source_checkpoint_sha256") != checkpoint_sha256:
    raise SystemExit("R03A config checkpoint hash differs")
if settings.get("source_r02_result_filename") != "r02-paired.json":
    raise SystemExit("R03A config raw R02 filename differs")
declared_r02_root = settings.get("source_r02_results_root")
if not isinstance(declared_r02_root, str) or pathlib.Path(declared_r02_root).resolve() != r02_raw_root:
    raise SystemExit("R03A config raw R02 root differs from R02_RAW_ROOT")
declared_summary = settings.get("source_r03_summary_artifact")
repo_root = config_path.parents[2]
if not isinstance(declared_summary, str):
    raise SystemExit("R03A config has no R03 summary path")
declared_summary_path = pathlib.Path(declared_summary)
if not declared_summary_path.is_absolute():
    declared_summary_path = repo_root / declared_summary_path
if declared_summary_path.resolve() != summary_path:
    raise SystemExit("R03A config R03 summary path differs from R03_SUMMARY")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
if not isinstance(summary, dict):
    raise SystemExit("R03 summary must be an object")
if not (
    summary.get("gate") == "R03"
    and summary.get("status") == "passed"
    and summary.get("gate_passed") is True
    and summary.get("r02_apparatus_passed") is True
):
    raise SystemExit("R03 summary is not accepted passing evidence")
population = summary.get("population")
if not isinstance(population, dict) or population.get("eligible_case_ids") != case_ids:
    raise SystemExit("R03 summary eligible identities/order differ from the manifest")
result_hashes = summary.get("result_hashes")
if not isinstance(result_hashes, list):
    raise SystemExit("R03 summary has no result hash list")
hash_by_case = {}
for item in result_hashes:
    if not isinstance(item, dict):
        raise SystemExit("R03 result hash entry must be an object")
    case_id = item.get("case_id")
    digest = item.get("sha256")
    if (
        not isinstance(case_id, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or case_id in hash_by_case
    ):
        raise SystemExit("R03 result hash entry is invalid or duplicated")
    hash_by_case[case_id] = digest
selected = str(case_ids[index])
expected = hash_by_case.get(selected)
if expected is None:
    raise SystemExit("selected R03A case has no raw R02 content hash")
print(f"{selected}\t{expected}")
PY
)
IFS=$'\t' read -r CASE_ID EXPECTED_SOURCE_R02_SHA256 <<<"$CASE_BINDING"
case "$CASE_ID" in
  *[!A-Za-z0-9._-]*|'') echo "selected R03A case_id is not path safe" >&2; exit 2 ;;
esac
export CASE_ID
SOURCE_R02_RESULT=$R02_RAW_ROOT/$CASE_ID/r02-paired.json
EXPECTED_RESULT=$EXPERIMENT_ROOT/$RUN_ID/$CASE_ID/r03a-analytic-kill-test.json
test -f "$SOURCE_R02_RESULT" || {
  echo "missing selected raw R02 source: $SOURCE_R02_RESULT" >&2
  exit 2
}
SOURCE_R02_SHA256=$(sha256sum "$SOURCE_R02_RESULT" | awk '{print $1}')
test "$SOURCE_R02_SHA256" = "$EXPECTED_SOURCE_R02_SHA256" || {
  echo "selected raw R02 source hash differs from the accepted R03 summary" >&2
  exit 2
}

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
test -d "$SAFELIBERO_ROOT" || { echo "missing SafeLIBERO root: $SAFELIBERO_ROOT" >&2; exit 2; }
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

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [ "$CUDA_VISIBLE_DEVICES" = NoDevFiles ]; then
  echo "R03A requires an allocation-visible GPU" >&2
  exit 2
fi

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-none}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "repo=$REMOTE_REPO"
echo "commit=$GIT_COMMIT"
echo "dirty=$GIT_DIRTY"
echo "run_id=$RUN_ID"
echo "case_index=$CASE_INDEX"
echo "case_id=$CASE_ID"
echo "manifest_sha256=$MANIFEST_SHA256"
echo "config_sha256=$CONFIG_SHA256"
echo "r03_summary_sha256=$R03_SUMMARY_SHA256"
echo "source_r02_sha256=$SOURCE_R02_SHA256"
echo "checkpoint_sha256=$CHECKPOINT_SHA256"
echo "expected_result=$EXPECTED_RESULT"
"$OPENPI_PYTHON" -V
"$OPENPI_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__); print(torch.cuda.get_device_name(0))'
FAILURE_STAGE=schema_dependency
JSONSCHEMA_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_jsonschema_overlay.sh)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$JSONSCHEMA_OVERLAY "$LIBERO_PYTHON" - \
  "$REMOTE_REPO/schemas/r03a-analytic-kill-test.schema.json" <<'PY'
import importlib.metadata
import json
import pathlib
import sys

import jsonschema

schema = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
jsonschema.Draft202012Validator.check_schema(schema)
print("jsonschema", importlib.metadata.version("jsonschema"))
PY
echo "jsonschema_overlay=$JSONSCHEMA_OVERLAY"
echo "jsonschema_bundle_sha256=$(tr -d '[:space:]' < "$JSONSCHEMA_OVERLAY/.source-bundle-sha256")"

FAILURE_STAGE=policy_server_startup
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

FAILURE_STAGE=r03a_runner
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$JSONSCHEMA_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero:$REMOTE_REPO/openpi/packages/openpi-client/src
cd "$REMOTE_REPO"
"$LIBERO_PYTHON" main/run_crfs_r03a.py \
  --manifest "$MANIFEST" \
  --config "$EXPERIMENT_CONFIG" \
  --r02-raw-root "$R02_RAW_ROOT" \
  --r03-summary "$R03_SUMMARY" \
  --r03-summary-sha256 "$R03_SUMMARY_SHA256" \
  --output-root "$EXPERIMENT_ROOT" \
  --run-id "$RUN_ID" \
  --case-index "$CASE_INDEX" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --checkpoint-id "$CHECKPOINT_DIR" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" >"$CLIENT_LOG" 2>&1

FAILURE_STAGE=r03a_result_contract
test -f "$EXPECTED_RESULT" || {
  echo "R03A runner returned without final artifact: $EXPECTED_RESULT" >&2
  exit 5
}
FAILURE_STAGE=r03a_apparatus_acceptance
"$LIBERO_PYTHON" - "$EXPECTED_RESULT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("status") != "completed":
    raise SystemExit(
        "R03A apparatus rejection: fresh nominal collision was not reconfirmed"
    )
outcome = value.get("outcome")
if not isinstance(outcome, dict) or outcome.get(
    "fresh_nominal_collision_reproduced"
) is not True:
    raise SystemExit(
        "R03A apparatus rejection: fresh_nominal_collision_reproduced is not true"
    )
arms = value.get("arms")
if not isinstance(arms, dict):
    raise SystemExit("R03A apparatus rejection: arms object is missing")
for name in ("analytic_trajectory_mid", "analytic_trajectory_early"):
    arm = arms.get(name)
    if not isinstance(arm, dict):
        raise SystemExit(f"R03A apparatus rejection: {name} arm is missing")
    status = arm.get("status")
    if status in {
        "policy_failure",
        "not_evaluated_after_nominal_collision_not_reconfirmed",
    }:
        raise SystemExit(
            f"R03A apparatus rejection: {name} has unacceptable status {status}"
        )
PY
echo "result=$EXPECTED_RESULT"
echo "result_sha256=$(sha256sum "$EXPECTED_RESULT" | awk '{print $1}')"
FAILURE_STAGE=complete
