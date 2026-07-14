#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... submit_r02_smoke.sh MANIFEST_JSONL CONFIG_JSON R01_RAW_ROOT_REMOTE R01_SUMMARY_REMOTE R01_SUMMARY_SHA256 PARITY_ARTIFACT_REMOTE PARITY_ARTIFACT_SHA256'
MANIFEST_LOCAL=${1:?$usage}
CONFIG_LOCAL=${2:?$usage}
R01_RAW_ROOT=${3:?$usage}
R01_SUMMARY=${4:?$usage}
R01_SUMMARY_SHA256=${5:?$usage}
PARITY_ARTIFACT=${6:?$usage}
PARITY_ARTIFACT_SHA256=${7:?$usage}
: "${RUN_ID:?set RUN_ID to an immutable exact R02 smoke identifier}"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
MANIFEST=$REMOTE_REPO/manifests/$(basename "$MANIFEST_LOCAL")
EXPERIMENT_CONFIG=$REMOTE_REPO/configs/experiments/$(basename "$CONFIG_LOCAL")

scripts/hpc/preflight.sh
test -f "$MANIFEST_LOCAL" || { echo "missing local frozen manifest: $MANIFEST_LOCAL" >&2; exit 2; }
test -f "$CONFIG_LOCAL" || { echo "missing external local R02 config: $CONFIG_LOCAL" >&2; exit 2; }
COUNT=$(awk 'NF {count++} END {print count+0}' "$MANIFEST_LOCAL")
test "$COUNT" -eq 20 || { echo "R02 smoke must index the frozen original 20-case manifest" >&2; exit 2; }

python3 - "$CONFIG_LOCAL" "$MANIFEST_LOCAL" "$R01_SUMMARY_SHA256" "$PARITY_ARTIFACT_SHA256" <<'PY'
import json
import pathlib
import re
import sys

config_path, manifest_path = map(pathlib.Path, sys.argv[1:3])
r01_hash, parity_hash = sys.argv[3:5]
for name, value in (("R01 summary", r01_hash), ("parity artifact", parity_hash)):
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SystemExit(f"{name} SHA256 must be 64 lowercase hexadecimal characters")
config = json.loads(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict) or not config.get("ready_to_run", False):
    raise SystemExit("R02 config is absent, invalid, or not ready_to_run")
if config.get("blocked_on", []) != []:
    raise SystemExit("R02 config still has blocked dependencies")
serialized = json.dumps(config, sort_keys=True)
if manifest_path.name not in serialized:
    raise SystemExit("R02 config is not bound to the passed frozen manifest")
if r01_hash not in serialized:
    raise SystemExit("R02 config is not bound to the expected R01 summary hash")
if parity_hash not in serialized:
    raise SystemExit("R02 config is not bound to the expected parity artifact hash")
PY

ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$MANIFEST" "$EXPERIMENT_CONFIG" \
  "$R01_RAW_ROOT" "$R01_SUMMARY" "$R01_SUMMARY_SHA256" \
  "$PARITY_ARTIFACT" "$PARITY_ARTIFACT_SHA256" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
manifest=$3
experiment_config=$4
r01_raw_root=$5
r01_summary=$6
r01_summary_sha256=$7
parity_artifact=$8
parity_artifact_sha256=$9

test -x "$remote_repo/scripts/hpc/run_r02_case.sh"
test -f "$remote_repo/slurm/r02_mig.sbatch"
test -f "$remote_repo/main/run_crfs_r02.py"
test -f "$manifest"
test -f "$experiment_config"
test -d "$r01_raw_root"
test -f "$r01_summary"
test -f "$parity_artifact"
test "$(sha256sum "$r01_summary" | awk '{print $1}')" = "$r01_summary_sha256"
test "$(sha256sum "$parity_artifact" | awk '{print $1}')" = "$parity_artifact_sha256"

mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle
cd "$remote_repo"
RUN_ID="$run_id" \
MANIFEST="$manifest" \
EXPERIMENT_CONFIG="$experiment_config" \
R01_RAW_ROOT="$r01_raw_root" \
R01_SUMMARY="$r01_summary" \
R01_SUMMARY_SHA256="$r01_summary_sha256" \
PARITY_ARTIFACT="$parity_artifact" \
PARITY_ARTIFACT_SHA256="$parity_artifact_sha256" \
REMOTE_REPO="$remote_repo" \
sbatch --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' slurm/r02_mig.sbatch
REMOTE
