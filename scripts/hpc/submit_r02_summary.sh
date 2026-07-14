#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... submit_r02_summary.sh MANIFEST_JSONL CONFIG_JSON CONFIG_SHA256 R01_SUMMARY_JSON R01_SUMMARY_SHA256 CHECKPOINT_SHA256'
MANIFEST_LOCAL=${1:?$usage}
CONFIG_LOCAL=${2:?$usage}
CONFIG_SHA256=${3:?$usage}
R01_SUMMARY_LOCAL=${4:?$usage}
R01_SUMMARY_SHA256=${5:?$usage}
CHECKPOINT_SHA256=${6:?$usage}
: "${RUN_ID:?set RUN_ID to the immutable exact 20-case R02 run identifier}"

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
    exit 2
    ;;
esac
for name_and_value in \
  "config:$CONFIG_SHA256" \
  "R01 summary:$R01_SUMMARY_SHA256" \
  "checkpoint:$CHECKPOINT_SHA256"; do
  name=${name_and_value%%:*}
  value=${name_and_value#*:}
  case "$value" in
    *[!0-9a-f]*|'')
      echo "$name SHA256 must be lowercase hexadecimal" >&2
      exit 2
      ;;
  esac
  test "${#value}" -eq 64 || {
    echo "$name SHA256 must have 64 characters" >&2
    exit 2
  }
done

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
CHECKPOINT_ID=${CHECKPOINT_ID:-/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}

test -f "$MANIFEST_LOCAL" || { echo "missing local frozen manifest: $MANIFEST_LOCAL" >&2; exit 2; }
test -f "$CONFIG_LOCAL" || { echo "missing local frozen R02 config: $CONFIG_LOCAL" >&2; exit 2; }
test -f "$R01_SUMMARY_LOCAL" || { echo "missing local passed R01 summary: $R01_SUMMARY_LOCAL" >&2; exit 2; }

OBSERVED_CONFIG_SHA256=$(shasum -a 256 "$CONFIG_LOCAL" | awk '{print $1}')
OBSERVED_R01_SUMMARY_SHA256=$(shasum -a 256 "$R01_SUMMARY_LOCAL" | awk '{print $1}')
MANIFEST_SHA256=$(shasum -a 256 "$MANIFEST_LOCAL" | awk '{print $1}')
test "$OBSERVED_CONFIG_SHA256" = "$CONFIG_SHA256" || {
  echo "local R02 config hash mismatch" >&2
  exit 2
}
test "$OBSERVED_R01_SUMMARY_SHA256" = "$R01_SUMMARY_SHA256" || {
  echo "local R01 summary hash mismatch" >&2
  exit 2
}

python3 - "$MANIFEST_LOCAL" "$CONFIG_LOCAL" "$R01_SUMMARY_LOCAL" "$R01_SUMMARY_SHA256" <<'PY'
import json
import pathlib
import re
import sys

manifest_path, config_path, summary_path = map(pathlib.Path, sys.argv[1:4])
expected_summary_sha256 = sys.argv[4]

records = []
for line_number, line in enumerate(
    manifest_path.read_text(encoding="utf-8").splitlines(), start=1
):
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid manifest JSON on line {line_number}: {error}")
    case_id = record.get("case_id") if isinstance(record, dict) else None
    if not isinstance(case_id, str) or re.fullmatch(r"[A-Za-z0-9._-]+", case_id) is None:
        raise SystemExit(f"invalid case_id on manifest line {line_number}")
    records.append(record)
case_ids = [record["case_id"] for record in records]
if len(case_ids) != 20 or len(set(case_ids)) != 20:
    raise SystemExit("R02 summary requires the frozen 20-case manifest with unique identities")

config = json.loads(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict) or config.get("ready_to_run") is not True:
    raise SystemExit("R02 config is absent, invalid, or not ready_to_run")
if config.get("blocked_on", []) != []:
    raise SystemExit("R02 config still has blocked dependencies")
if config.get("manifest") != f"manifests/{manifest_path.name}":
    raise SystemExit("R02 config is not bound to the passed frozen manifest")
settings = config.get("r02")
if not isinstance(settings, dict):
    raise SystemExit("R02 config has no r02 settings")
if settings.get("r01_summary_artifact") != f"evidence/r01/{summary_path.name}":
    raise SystemExit("R02 config is not bound to the passed R01 summary path")
if settings.get("r01_summary_sha256") != expected_summary_sha256:
    raise SystemExit("R02 config is not bound to the passed R01 summary hash")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
if not isinstance(summary, dict) or summary.get("gate") != "R01":
    raise SystemExit("supplied predecessor is not an R01 summary")
if summary.get("status") != "passed" or summary.get("gate_passed") is not True:
    raise SystemExit("supplied R01 summary is not passing")
PY

MANIFEST=$REMOTE_REPO/manifests/$(basename "$MANIFEST_LOCAL")
EXPERIMENT_CONFIG=$REMOTE_REPO/configs/experiments/$(basename "$CONFIG_LOCAL")
R01_SUMMARY=$REMOTE_REPO/evidence/r01/$(basename "$R01_SUMMARY_LOCAL")
RESULTS_ROOT=/mnt/data/quanth/experiments/crfs-oracle/$RUN_ID

# The live cluster state is authoritative and must be checked immediately before
# asking the login node to submit the allocation-backed CPU verifier.
scripts/hpc/preflight.sh

ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$MANIFEST" "$MANIFEST_SHA256" \
  "$EXPERIMENT_CONFIG" "$CONFIG_SHA256" \
  "$R01_SUMMARY" "$R01_SUMMARY_SHA256" \
  "$CHECKPOINT_ID" "$CHECKPOINT_SHA256" "$RESULTS_ROOT" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
manifest=$3
manifest_sha256=$4
experiment_config=$5
config_sha256=$6
r01_summary=$7
r01_summary_sha256=$8
checkpoint_id=$9
checkpoint_sha256=${10}
results_root=${11}

test -f "$remote_repo/slurm/r02_summary.sbatch"
test -f "$remote_repo/main/summarize_r02.py"
test -f "$manifest"
test -f "$experiment_config"
test -f "$r01_summary"
test -f "$checkpoint_id/model.safetensors"
test -d "$results_root"
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$manifest_sha256"
test "$(sha256sum "$experiment_config" | awk '{print $1}')" = "$config_sha256"
test "$(sha256sum "$r01_summary" | awk '{print $1}')" = "$r01_summary_sha256"

if grep -Eq '^#SBATCH[[:space:]]+--gres([=[:space:]]|$)' "$remote_repo/slurm/r02_summary.sbatch"; then
  echo "R02 summary must use a CPU-only Slurm allocation" >&2
  exit 2
fi

manifest_count=$(awk 'NF {count++} END {print count+0}' "$manifest")
test "$manifest_count" -eq 20 || {
  echo "remote R02 manifest is not the full 20-case population" >&2
  exit 2
}
case_ids=$(sed -n 's/.*"case_id":"\([A-Za-z0-9._-]*\)".*/\1/p' "$manifest")
case_id_count=$(printf '%s\n' "$case_ids" | awk 'NF {count++} END {print count+0}')
unique_case_id_count=$(printf '%s\n' "$case_ids" | awk 'NF' | sort -u | awk 'END {print NR+0}')
test "$case_id_count" -eq 20 && test "$unique_case_id_count" -eq 20 || {
  echo "remote R02 manifest case identities are incomplete or duplicated" >&2
  exit 2
}
while IFS= read -r case_id; do
  test -n "$case_id" || continue
  test -f "$results_root/$case_id/r02-paired.json" || {
    echo "missing final R02 result for $case_id" >&2
    exit 2
  }
done <<EOF
$case_ids
EOF
result_count=$(find "$results_root" -mindepth 2 -maxdepth 2 -type f -name r02-paired.json -print | awk 'END {print NR+0}')
test "$result_count" -eq 20 || {
  echo "R02 result root must contain exactly 20 final case artifacts" >&2
  exit 2
}

mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle
cd "$remote_repo"
submission=$(
  RUN_ID="$run_id" \
  MANIFEST="$manifest" \
  EXPERIMENT_CONFIG="$experiment_config" \
  CONFIG_SHA256="$config_sha256" \
  R01_SUMMARY="$r01_summary" \
  R01_SUMMARY_SHA256="$r01_summary_sha256" \
  CHECKPOINT_ID="$checkpoint_id" \
  CHECKPOINT_SHA256="$checkpoint_sha256" \
  RESULTS_ROOT="$results_root" \
  OUTPUT="$results_root/r03-summary.json" \
  REMOTE_REPO="$remote_repo" \
  sbatch --parsable \
    --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' \
    slurm/r02_summary.sbatch
)
job_id=${submission%%;*}
case "$job_id" in
  *[!0-9]*|'')
    echo "sbatch did not return an exact numeric job id: $submission" >&2
    exit 2
    ;;
esac
printf '%s\n' "$job_id"
REMOTE
