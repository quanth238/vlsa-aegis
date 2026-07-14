#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... submit_r04_label_smoke.sh manifests/reach_progress_calibration.jsonl configs/experiments/r04_continuation_labels.json'
MANIFEST_LOCAL=${1:?$usage}
CONFIG_LOCAL=${2:?$usage}
: "${RUN_ID:?set RUN_ID to an immutable exact R04A apparatus-smoke identifier}"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
MANIFEST=$REMOTE_REPO/manifests/$(basename "$MANIFEST_LOCAL")
EXPERIMENT_CONFIG=$REMOTE_REPO/configs/experiments/$(basename "$CONFIG_LOCAL")
CHECKPOINT_DIR=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
PARITY_ARTIFACT=/mnt/data/quanth/experiments/crfs-oracle/sampler-parity-r02-20260714c/sampler-parity.json

EXPECTED_MANIFEST_SHA256=3180836991e64b774670218f5fe4edaa10775fbb33fcdb330cc781b8b8937dad
EXPECTED_CONFIG_SHA256=561128a5a05710e50b282582463127ee3f8cd87c9ebbaee822f8c4602fda1c25
EXPECTED_R00_SUMMARY_SHA256=90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f
EXPECTED_R03_SUMMARY_SHA256=dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e
EXPECTED_PARITY_SHA256=26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a
EXPECTED_CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed

scripts/hpc/preflight.sh
test -f "$MANIFEST_LOCAL" || { echo "missing local frozen R00 manifest: $MANIFEST_LOCAL" >&2; exit 2; }
test -f "$CONFIG_LOCAL" || { echo "missing local frozen R04A config: $CONFIG_LOCAL" >&2; exit 2; }
test -f evidence/r00/r00-summary.json || { echo "missing local R00 summary" >&2; exit 2; }
test -f evidence/r03/r03-summary.json || { echo "missing local R03 summary" >&2; exit 2; }
git ls-files --error-unmatch \
  "$MANIFEST_LOCAL" \
  "$CONFIG_LOCAL" \
  scripts/hpc/run_r04_label_case.sh \
  scripts/hpc/submit_r04_label_smoke.sh \
  slurm/r04_label_mig.sbatch \
  main/run_r04_label_contract.py >/dev/null || {
  echo "R04A source files must be committed before submission" >&2
  exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "R04A submission requires a clean tracked worktree" >&2
  exit 2
}

MANIFEST_SHA256=$(shasum -a 256 "$MANIFEST_LOCAL" | awk '{print $1}')
CONFIG_SHA256=$(shasum -a 256 "$CONFIG_LOCAL" | awk '{print $1}')
R00_SUMMARY_SHA256=$(shasum -a 256 evidence/r00/r00-summary.json | awk '{print $1}')
R03_SUMMARY_SHA256=$(shasum -a 256 evidence/r03/r03-summary.json | awk '{print $1}')
test "$MANIFEST_SHA256" = "$EXPECTED_MANIFEST_SHA256" || { echo "local R00 manifest hash mismatch" >&2; exit 2; }
test "$CONFIG_SHA256" = "$EXPECTED_CONFIG_SHA256" || { echo "local frozen R04A config hash mismatch" >&2; exit 2; }
test "$R00_SUMMARY_SHA256" = "$EXPECTED_R00_SUMMARY_SHA256" || { echo "local R00 summary hash mismatch" >&2; exit 2; }
test "$R03_SUMMARY_SHA256" = "$EXPECTED_R03_SUMMARY_SHA256" || { echo "local R03 summary hash mismatch" >&2; exit 2; }

python3 - "$CONFIG_LOCAL" "$MANIFEST_LOCAL" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
config = json.loads(config_path.read_text(encoding="utf-8"))
records = [
    json.loads(line)
    for line in manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
expected = {
    "manifest_sha256": "3180836991e64b774670218f5fe4edaa10775fbb33fcdb330cc781b8b8937dad",
    "checkpoint_sha256": "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
}
if config_path.name != "r04_continuation_labels.json":
    raise SystemExit("R04A smoke requires the frozen config filename")
if manifest_path.name != "reach_progress_calibration.jsonl":
    raise SystemExit("R04A smoke requires the frozen R00 manifest filename")
if config.get("ready_to_run") is not True or config.get("blocked_on") != []:
    raise SystemExit("R04A apparatus config is not ready_to_run")
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"R04A config {key} binding changed")
if len(records) != 120 or len({row.get("group_id") for row in records}) != 30:
    raise SystemExit("R04A source must remain the 120-row, 30-group R00 manifest")
if records[0].get("case_id") != "crfs-93365b8b851365f2":
    raise SystemExit("R04A case-index-0 identity changed")
settings = config.get("r04a", {})
required = {
    "enabled": True,
    "phase": "pregrasp_reach",
    "apparatus_scope": "label_contract_smoke_only",
    "trace_steps": [1, 2, 3, 4, 5],
    "trace_times": [0.9, 0.8, 0.7, 0.6, 0.5],
    "duplicate_trace_requests": 2,
    "continuation": "deterministic_frozen_eager_no_intervention",
    "intervention_mode": "none",
    "correction": None,
    "training": False,
    "guidance": False,
    "executed_action_horizon": 5,
    "simulator_repeats": 2,
    "geometry_frame": "world",
    "geometry_representation": "geom_name_sorted_padded_obb",
    "maximum_obbs": 21,
    "retain_all_outcomes": True,
    "r00_summary_sha256": "90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f",
    "r03_summary_sha256": "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e",
    "sampler_parity_sha256": "26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a",
    "r02_claim_cases_excluded": True,
}
for key, value in required.items():
    if settings.get(key) != value:
        raise SystemExit(f"R04A frozen field changed: {key}")
coverage = settings.get("coverage_audit", {})
if coverage.get("state_groups") != 30 or coverage.get("rows_in_boundary_interval") != 0:
    raise SystemExit("R04A coverage limitation changed")
claim = settings.get("claim_bearing_requirements", {})
if claim.get("new_immutable_state_groups") is not True:
    raise SystemExit("R04A cannot authorize reuse for claim-bearing training")
if claim.get("split_complete_groups_before_label_generation") is not True:
    raise SystemExit("R04A split must be frozen before labels")
PY

EXPECTED_COMMIT=$(git rev-parse HEAD)
ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$MANIFEST" "$EXPERIMENT_CONFIG" \
  "$EXPECTED_COMMIT" "$MANIFEST_SHA256" "$CONFIG_SHA256" \
  "$CHECKPOINT_DIR" "$PARITY_ARTIFACT" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
manifest=$3
experiment_config=$4
expected_commit=$5
manifest_sha256=$6
config_sha256=$7
checkpoint_dir=$8
parity_artifact=$9

expected_r00_summary_sha256=90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f
expected_config_sha256=561128a5a05710e50b282582463127ee3f8cd87c9ebbaee822f8c4602fda1c25
expected_r03_summary_sha256=dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e
expected_parity_sha256=26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a
expected_checkpoint_sha256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
r00_summary=$remote_repo/evidence/r00/r00-summary.json
r03_summary=$remote_repo/evidence/r03/r03-summary.json
checkpoint=$checkpoint_dir/model.safetensors

for path in \
  "$remote_repo/scripts/hpc/run_r04_label_case.sh" \
  "$remote_repo/slurm/r04_label_mig.sbatch" \
  "$remote_repo/main/run_r04_label_contract.py" \
  "$manifest" \
  "$experiment_config" \
  "$r00_summary" \
  "$r03_summary" \
  "$parity_artifact" \
  "$checkpoint"; do
  test -f "$path" || { echo "missing remote R04A input: $path" >&2; exit 2; }
done
test -x "$remote_repo/scripts/hpc/run_r04_label_case.sh" || { echo "remote R04A runner is not executable" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || {
  echo "remote source is not synchronized to the reviewed R04A commit" >&2
  exit 2
}
test -z "$(git -C "$remote_repo" status --porcelain)" || {
  echo "remote source tree is dirty" >&2
  exit 2
}
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$manifest_sha256"
test "$(sha256sum "$experiment_config" | awk '{print $1}')" = "$config_sha256"
test "$config_sha256" = "$expected_config_sha256"
test "$(sha256sum "$r00_summary" | awk '{print $1}')" = "$expected_r00_summary_sha256"
test "$(sha256sum "$r03_summary" | awk '{print $1}')" = "$expected_r03_summary_sha256"
test "$(sha256sum "$parity_artifact" | awk '{print $1}')" = "$expected_parity_sha256"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint_sha256"

# Count current allocations before adding this 1-GPU/6-CPU/80-GiB smoke.  QOS
# is still authoritative at scheduling time, but the submission fails closed
# rather than relying on QOS to correct a known aggregate-ceiling violation.
memory_to_mb() {
  value=$1
  case "$value" in
    *K) number=${value%K}; echo $(((number + 1023) / 1024)) ;;
    *M) number=${value%M}; echo "$number" ;;
    *G) number=${value%G}; echo $((number * 1024)) ;;
    *T) number=${value%T}; echo $((number * 1024 * 1024)) ;;
    *[!0-9]*|'') return 1 ;;
    *) echo "$value" ;;
  esac
}

allocated_gpus=0
allocated_cpus=0
allocated_mem_mb=0
while IFS= read -r job_id; do
  test -n "$job_id" || continue
  job_record=$(scontrol show job "$job_id" -o)
  alloc_tres=$(printf '%s\n' "$job_record" | sed -n 's/.* AllocTRES=\([^ ]*\).*/\1/p')
  test -n "$alloc_tres" || {
    echo "cannot audit allocated TRES for live job $job_id" >&2
    exit 2
  }
  job_cpus=$(printf '%s' "$alloc_tres" | tr ',' '\n' | sed -n 's/^cpu=\([0-9][0-9]*\)$/\1/p')
  job_mem=$(printf '%s' "$alloc_tres" | tr ',' '\n' | sed -n 's/^mem=\([^,]*\)$/\1/p')
  job_gpus=$(printf '%s' "$alloc_tres" | tr ',' '\n' | sed -n 's#^gres/gpu[^=]*=\([0-9][0-9]*\)$#\1#p' | awk '{sum += $1} END {print sum+0}')
  test -n "$job_cpus" && test -n "$job_mem" || {
    echo "cannot parse CPU or memory allocation for live job $job_id: $alloc_tres" >&2
    exit 2
  }
  job_mem_mb=$(memory_to_mb "$job_mem") || {
    echo "cannot parse allocated memory for live job $job_id: $job_mem" >&2
    exit 2
  }
  allocated_cpus=$((allocated_cpus + job_cpus))
  allocated_mem_mb=$((allocated_mem_mb + job_mem_mb))
  allocated_gpus=$((allocated_gpus + job_gpus))
done < <(squeue -h -u "$(whoami)" -t RUNNING,COMPLETING,CONFIGURING,SUSPENDED -o '%i')

projected_gpus=$((allocated_gpus + 1))
projected_cpus=$((allocated_cpus + 6))
projected_mem_mb=$((allocated_mem_mb + 80 * 1024))
echo "current_allocated_gpus=$allocated_gpus current_allocated_cpus=$allocated_cpus current_allocated_mem_mb=$allocated_mem_mb"
echo "projected_gpus=$projected_gpus projected_cpus=$projected_cpus projected_mem_mb=$projected_mem_mb"
test "$projected_gpus" -le 2 || { echo "R04A smoke would exceed the 2-GPU-equivalent user ceiling" >&2; exit 2; }
test "$projected_cpus" -le 16 || { echo "R04A smoke would exceed the 16-CPU user ceiling" >&2; exit 2; }
test "$projected_mem_mb" -le $((256 * 1024)) || { echo "R04A smoke would exceed the 256-GiB user ceiling" >&2; exit 2; }

# Live state is authoritative.  Refuse submission unless a non-drained MIG node
# currently has at least the requested 80 GiB of free host memory.  Pin the
# submission to the checked nodes and explicitly exclude every unhealthy node.
minimum_free_mem_mb=$((80 * 1024))
eligible_nodes=()
excluded_nodes=()
while IFS='|' read -r node state; do
  test -n "$node" || continue
  normalized_state=$(printf '%s' "$state" | tr '[:upper:]' '[:lower:]')
  case "$normalized_state" in
    *down*|*drain*|*drng*|*fail*|*maint*|*not_resp*|*notresponding*|*no_resp*|*power*|*unknown*|*reserved*)
      excluded_nodes+=("$node")
      continue
      ;;
  esac
  record=$(scontrol show node "$node" -o)
  free_mem=$(printf '%s\n' "$record" | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
  test -n "$free_mem" || {
    echo "cannot determine fresh FreeMem for MIG node $node" >&2
    exit 2
  }
  echo "live_mig_node=$node state=$state free_mem_mb=$free_mem"
  if [ "$free_mem" -ge "$minimum_free_mem_mb" ]; then
    eligible_nodes+=("$node")
  fi
done < <(sinfo -h -p mig -N -o '%N|%T' | sort -u)

test "${#eligible_nodes[@]}" -gt 0 || {
  echo "R04A smoke not submitted: no healthy MIG node has a fresh 80 GiB of free host memory" >&2
  exit 2
}
eligible_csv=$(IFS=,; echo "${eligible_nodes[*]}")
sbatch_args=(--nodelist="$eligible_csv")
if [ "${#excluded_nodes[@]}" -gt 0 ]; then
  excluded_csv=$(IFS=,; echo "${excluded_nodes[*]}")
  echo "excluded_unhealthy_mig_nodes=$excluded_csv"
  sbatch_args+=(--exclude="$excluded_csv")
fi

mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle
cd "$remote_repo"
RUN_ID="$run_id" \
MANIFEST="$manifest" \
EXPERIMENT_CONFIG="$experiment_config" \
REMOTE_REPO="$remote_repo" \
EXPECTED_GIT_COMMIT="$expected_commit" \
sbatch "${sbatch_args[@]}" \
  --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' \
  slurm/r04_label_mig.sbatch
REMOTE
