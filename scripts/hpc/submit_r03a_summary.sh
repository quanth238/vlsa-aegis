#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... submit_r03a_summary.sh SOURCE_ARRAY_JOB_ID MANIFEST_JSONL CONFIG_JSON R02_RAW_ROOT_REMOTE R03_SUMMARY_REMOTE R03_SUMMARY_SHA256'
SOURCE_ARRAY_JOB_ID=${1:?$usage}
MANIFEST_LOCAL=${2:?$usage}
CONFIG_LOCAL=${3:?$usage}
R02_RAW_ROOT=${4:?$usage}
R03_SUMMARY=${5:?$usage}
R03_SUMMARY_SHA256=${6:?$usage}
: "${RUN_ID:?set RUN_ID to the exact immutable full R03A array run identifier}"

EXPECTED_MANIFEST_SHA256=241f1b94a5434973dcfb235b43b9386ad15046ff87d35d5b0c34e295c2132916
EXPECTED_CONFIG_SHA256=445c7f227b780f87f5393f893a0ebba96e77a83d1f3690e6eee7559effceaa8a
EXPECTED_R03_SUMMARY_SHA256=dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e
CHECKPOINT_ID=${CHECKPOINT_ID:-/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:-988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/crfs-oracle}

case "$SOURCE_ARRAY_JOB_ID" in
  *[!0-9]*|'') echo "SOURCE_ARRAY_JOB_ID must be one exact numeric Slurm job id" >&2; exit 2 ;;
esac
case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_ID contains unsafe characters" >&2; exit 2 ;;
esac
test "$R03_SUMMARY_SHA256" = "$EXPECTED_R03_SUMMARY_SHA256" || {
  echo "R03 summary hash differs from accepted evidence" >&2
  exit 2
}

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
MANIFEST=$REMOTE_REPO/manifests/r03a_analytic_kill_test_eligible.jsonl
CONFIG=$REMOTE_REPO/configs/experiments/r03a_analytic_kill_test.json
RESULTS_ROOT=$EXPERIMENT_ROOT/$RUN_ID
OUTPUT=$RESULTS_ROOT/r03a-summary.json

test "$(basename "$MANIFEST_LOCAL")" = r03a_analytic_kill_test_eligible.jsonl
test "$(basename "$CONFIG_LOCAL")" = r03a_analytic_kill_test.json
test "$(shasum -a 256 "$MANIFEST_LOCAL" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA256" || {
  echo "local R03A manifest hash mismatch" >&2; exit 2;
}
test "$(shasum -a 256 "$CONFIG_LOCAL" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256" || {
  echo "local R03A config hash mismatch" >&2; exit 2;
}
test "$(awk 'NF {count++} END {print count+0}' "$MANIFEST_LOCAL")" -eq 17 || {
  echo "R03A population summary requires all 17 registered cases" >&2; exit 2;
}
jq -e \
  --arg manifest_sha "$EXPECTED_MANIFEST_SHA256" \
  --arg r03_sha "$EXPECTED_R03_SUMMARY_SHA256" '
    .ready_to_run == true and .blocked_on == [] and
    .r03a.eligible_case_count == 17 and
    .r03a.eligible_manifest_sha256 == $manifest_sha and
    .r03a.source_r03_summary_sha256 == $r03_sha and
    .r03a.probe_training_authorized == false
  ' "$CONFIG_LOCAL" >/dev/null || {
  echo "local R03A summary contract is not frozen" >&2; exit 2;
}

SOURCE_FILES=(
  configs/experiments/r03a_analytic_kill_test.json
  main/crfs_oracle/r03a_validation.py
  main/summarize_r03a.py
  manifests/r03a_analytic_kill_test_eligible.jsonl
  scripts/hpc/prepare_jsonschema_overlay.sh
  scripts/hpc/submit_r03a_summary.sh
  slurm/r03a_summary.sbatch
)
for source_file in "${SOURCE_FILES[@]}"; do
  git ls-files --error-unmatch "$source_file" >/dev/null 2>&1 || {
    echo "R03A summary source must be committed: $source_file" >&2; exit 2;
  }
done
LOCAL_CHANGES=$(git status --porcelain --untracked-files=all | awk '$0 !~ /^\?\? tmp\// {print}')
test -z "$LOCAL_CHANGES" || {
  echo "R03A summary submission requires a clean reviewed tree (only user-owned ?? tmp/ is ignored)" >&2
  exit 2
}
EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)

scripts/hpc/preflight.sh

ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$EXPECTED_GIT_COMMIT" "$SOURCE_ARRAY_JOB_ID" "$RUN_ID" \
  "$MANIFEST" "$EXPECTED_MANIFEST_SHA256" "$CONFIG" "$EXPECTED_CONFIG_SHA256" \
  "$R02_RAW_ROOT" "$R03_SUMMARY" "$R03_SUMMARY_SHA256" "$RESULTS_ROOT" \
  "$OUTPUT" "$CHECKPOINT_ID" "$CHECKPOINT_SHA256" <<'REMOTE'
set -euo pipefail
remote_repo=$1
expected_commit=$2
source_array_job_id=$3
run_id=$4
manifest=$5
manifest_sha256=$6
config=$7
config_sha256=$8
r02_raw_root=$9
r03_summary=${10}
r03_summary_sha256=${11}
results_root=${12}
output=${13}
checkpoint_id=${14}
checkpoint_sha256=${15}

for path in \
  "$manifest" "$config" "$r03_summary" \
  "$remote_repo/main/summarize_r03a.py" \
  "$remote_repo/main/crfs_oracle/r03a_validation.py" \
  "$remote_repo/scripts/hpc/prepare_jsonschema_overlay.sh" \
  "$remote_repo/slurm/r03a_summary.sbatch"; do
  test -f "$path" || { echo "missing remote R03A summary input: $path" >&2; exit 2; }
done
test -d "$r02_raw_root" || { echo "missing raw R02 source root" >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$manifest_sha256" || {
  echo "remote R03A manifest hash mismatch" >&2; exit 2;
}
test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha256" || {
  echo "remote R03A config hash mismatch" >&2; exit 2;
}
test "$(sha256sum "$r03_summary" | awk '{print $1}')" = "$r03_summary_sha256" || {
  echo "remote R03 source hash mismatch" >&2; exit 2;
}
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || {
  echo "remote checkout differs from reviewed R03A summary commit" >&2; exit 2;
}
test -z "$(git -C "$remote_repo" status --porcelain)" || {
  echo "remote R03A summary checkout is dirty" >&2; exit 2;
}
test ! -e "$output" || {
  echo "immutable R03A population summary already exists" >&2; exit 2;
}

source_record=$(scontrol show job "$source_array_job_id" -o 2>/dev/null || true)
test -n "$source_record" || {
  echo "cannot inspect exact source array job $source_array_job_id" >&2; exit 2;
}
case "$source_record" in
  *"JobName=crfs-r03a "*) ;;
  *) echo "source job is not the registered R03A full array" >&2; exit 2 ;;
esac
case "$source_record" in
  *"ArrayTaskId=0-16%"*|*"ArrayTaskId=0-16"*) ;;
  *) echo "source job does not bind the complete 0-16 R03A array" >&2; exit 2 ;;
esac

excluded_nodes=()
while IFS='|' read -r node state; do
  test -n "$node" || continue
  normalized=$(printf '%s' "$state" | tr '[:upper:]' '[:lower:]')
  case "$normalized" in
    *down*|*drain*|*drng*|*fail*|*maint*|*not_resp*|*notresponding*|*no_resp*|*power*|*unknown*)
      excluded_nodes+=("$node")
      ;;
  esac
done < <(sinfo -h -p main -N -o '%N|%T' | sort -u)
sbatch_args=()
if [ "${#excluded_nodes[@]}" -gt 0 ]; then
  excluded_csv=$(IFS=,; echo "${excluded_nodes[*]}")
  sbatch_args+=(--exclude="$excluded_csv")
fi

mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle
cd "$remote_repo"
submission=$(
  EXPECTED_GIT_COMMIT="$expected_commit" \
  RUN_ID="$run_id" \
  MANIFEST="$manifest" \
  MANIFEST_SHA256="$manifest_sha256" \
  CONFIG="$config" \
  CONFIG_SHA256="$config_sha256" \
  R02_RAW_ROOT="$r02_raw_root" \
  R03_SUMMARY="$r03_summary" \
  R03_SUMMARY_SHA256="$r03_summary_sha256" \
  RESULTS_ROOT="$results_root" \
  OUTPUT="$output" \
  SOURCE_SLURM_ARRAY_JOB_ID="$source_array_job_id" \
  CHECKPOINT_ID="$checkpoint_id" \
  CHECKPOINT_SHA256="$checkpoint_sha256" \
  REMOTE_REPO="$remote_repo" \
  sbatch --parsable "${sbatch_args[@]}" \
    --dependency="afterok:$source_array_job_id" \
    --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' \
    slurm/r03a_summary.sbatch
)
job_id=${submission%%;*}
case "$job_id" in
  *[!0-9]*|'') echo "sbatch returned an invalid R03A summary job id: $submission" >&2; exit 2 ;;
esac
echo "submitted_summary_job_id=$job_id"
echo "summary_dependency=afterok:$source_array_job_id"
echo "expected_summary=$output"
REMOTE
