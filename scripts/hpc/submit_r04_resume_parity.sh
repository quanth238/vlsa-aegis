#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... submit_r04_resume_parity.sh manifests/reach_progress_calibration.jsonl configs/experiments/r04_resume_parity.json'
MANIFEST_LOCAL=${1:?$usage}
CONFIG_LOCAL=${2:?$usage}
: "${RUN_ID:?set RUN_ID to an immutable exact R04B resume-parity smoke identifier}"
case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
    exit 2
    ;;
esac

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
MANIFEST=$REMOTE_REPO/manifests/$(basename "$MANIFEST_LOCAL")
EXPERIMENT_CONFIG=$REMOTE_REPO/configs/experiments/$(basename "$CONFIG_LOCAL")
CHECKPOINT_DIR=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
PARITY_ARTIFACT=/mnt/data/quanth/experiments/crfs-oracle/sampler-parity-r02-20260714c/sampler-parity.json
R04A_ARTIFACT=/mnt/data/quanth/experiments/crfs-oracle/r04a-label-contract-smoke-20260714a/crfs-93365b8b851365f2/r04-label-contract.json
R04A_VALIDATION_LOCAL=evidence/r04a/r04a-validation.json
R04A_VALIDATION=$REMOTE_REPO/$R04A_VALIDATION_LOCAL
R00_SUMMARY_LOCAL=evidence/r00/r00-summary.json
R03_SUMMARY_LOCAL=evidence/r03/r03-summary.json
R00_SUMMARY=$REMOTE_REPO/$R00_SUMMARY_LOCAL
R03_SUMMARY=$REMOTE_REPO/$R03_SUMMARY_LOCAL

EXPECTED_MANIFEST_SHA256=3180836991e64b774670218f5fe4edaa10775fbb33fcdb330cc781b8b8937dad
EXPECTED_CONFIG_SHA256=a0edb7edac86d2abd887218002d14e28e072cd472d631797d18d540f5140be33
EXPECTED_R00_SUMMARY_SHA256=90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f
EXPECTED_R03_SUMMARY_SHA256=dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e
EXPECTED_PARITY_SHA256=26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a
EXPECTED_R04A_ARTIFACT_SHA256=b820793ec42a5228c858e297d13806d8ae7f02e5cc7a765769316473d795f285
EXPECTED_R04A_VALIDATION_SHA256=cf821650c48d30f6ebf2bbf7afe06b21938e0b137aa5acf1921f571e409e05ea
EXPECTED_CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed

# Live Slurm, QOS, storage, and login-process state is authoritative.
scripts/hpc/preflight.sh
for path in \
  "$MANIFEST_LOCAL" \
  "$CONFIG_LOCAL" \
  "$R00_SUMMARY_LOCAL" \
  "$R03_SUMMARY_LOCAL" \
  "$R04A_VALIDATION_LOCAL"; do
  test -f "$path" || { echo "missing local frozen R04B input: $path" >&2; exit 2; }
done
git ls-files --error-unmatch \
  "$MANIFEST_LOCAL" \
  "$CONFIG_LOCAL" \
  "$R00_SUMMARY_LOCAL" \
  "$R03_SUMMARY_LOCAL" \
  "$R04A_VALIDATION_LOCAL" \
  scripts/hpc/run_r04_resume_parity.sh \
  scripts/hpc/submit_r04_resume_parity.sh \
  slurm/r04_resume_parity_mig.sbatch \
  schemas/r04-resume-parity.schema.json \
  main/run_r04_resume_parity.py \
  main/crfs_oracle/r04_resume.py >/dev/null || {
  echo "R04B source files must be committed before submission" >&2
  exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "R04B submission requires a clean reviewed tracked worktree" >&2
  exit 2
}

MANIFEST_SHA256=$(shasum -a 256 "$MANIFEST_LOCAL" | awk '{print $1}')
CONFIG_SHA256=$(shasum -a 256 "$CONFIG_LOCAL" | awk '{print $1}')
R00_SUMMARY_SHA256=$(shasum -a 256 "$R00_SUMMARY_LOCAL" | awk '{print $1}')
R03_SUMMARY_SHA256=$(shasum -a 256 "$R03_SUMMARY_LOCAL" | awk '{print $1}')
R04A_VALIDATION_SHA256=$(shasum -a 256 "$R04A_VALIDATION_LOCAL" | awk '{print $1}')
test "$MANIFEST_SHA256" = "$EXPECTED_MANIFEST_SHA256" || { echo "local R04B manifest hash mismatch" >&2; exit 2; }
test "$CONFIG_SHA256" = "$EXPECTED_CONFIG_SHA256" || { echo "local frozen R04B config hash mismatch" >&2; exit 2; }
test "$R00_SUMMARY_SHA256" = "$EXPECTED_R00_SUMMARY_SHA256" || { echo "local R00 summary hash mismatch" >&2; exit 2; }
test "$R03_SUMMARY_SHA256" = "$EXPECTED_R03_SUMMARY_SHA256" || { echo "local R03 summary hash mismatch" >&2; exit 2; }
test "$R04A_VALIDATION_SHA256" = "$EXPECTED_R04A_VALIDATION_SHA256" || { echo "local R04A validation hash mismatch" >&2; exit 2; }

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
if config_path.name != "r04_resume_parity.json":
    raise SystemExit("R04B smoke requires the frozen config filename")
if manifest_path.name != "reach_progress_calibration.jsonl":
    raise SystemExit("R04B smoke requires the frozen R00 manifest filename")
if config.get("ready_to_run") is not True or config.get("blocked_on") != []:
    raise SystemExit("R04B resume-parity config is not ready_to_run")
if config.get("manifest_sha256") != "3180836991e64b774670218f5fe4edaa10775fbb33fcdb330cc781b8b8937dad":
    raise SystemExit("R04B config manifest binding changed")
if config.get("checkpoint_sha256") != "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed":
    raise SystemExit("R04B config checkpoint binding changed")
if len(records) != 120 or len({row.get("group_id") for row in records}) != 30:
    raise SystemExit("R04B source must remain the 120-row, 30-group R00 manifest")
case = records[0]
if case.get("case_id") != "crfs-93365b8b851365f2" or case.get("group_id") != "safelibero_spatial:II:0:2":
    raise SystemExit("R04B fixed apparatus case identity changed")
settings = config.get("r04b", {})
required = {
    "enabled": True,
    "apparatus_scope": "exact_saved_latent_resume_edit_parity_smoke_only",
    "smoke_case_index": 0,
    "source_trace_steps": [1, 2, 3, 4, 5],
    "source_eager_calls_per_step": 2,
    "zero_resume_calls_per_step": 2,
    "nonzero_resume_calls_per_step": 2,
    "compiled_default_calls": 2,
    "compiled_default_call_placement": "one_before_and_one_after_eager_resume_sequence",
    "compiled_default_request_contract": "reserved_envelope_explicit_paired_noise_mode_none_no_trace_no_correction_or_resume",
    "compiled_default_comparison_contract": "exact_current_eager_R04A_golden_hashes_and_compiled_physical_within_frozen_R02_limits",
    "compiled_default_physical_limits": {
        "source": "docs/decisions/0011-use-eager-path-for-r02-parity.md",
        "source_sha256": "11646bec37bdb2d15e9507080156755300782e0a4eabf91449dcc5105ca10d36",
        "physical_xyz5_max": 0.010,
        "physical_xyz5_rms": 0.005,
        "physical_action7_max": 0.050,
        "physical_action7_rms": 0.015,
        "normalized_model_comparison_status": "not_exposed_by_ordinary_compiled_physical_policy_reply_validated_R02_predecessor_remains_authority",
    },
    "total_policy_calls": 32,
    "source_intervention_mode": "none",
    "resume_intervention_mode": "latent_resume_edit",
    "latent_edit_space": "model",
    "return_normalized_final": True,
    "simulator_setup": "one_reset_plus_20_dummy_settle_control_steps",
    "policy_generated_action_steps_executed": 0,
    "efficacy_rollouts_executed": 0,
    "simulator_efficacy_evaluated": False,
    "training": False,
    "guidance": False,
    "learned_probe": False,
    "scientific_claim_authorized": False,
}
for key, value in required.items():
    if settings.get(key) != value:
        raise SystemExit(f"R04B frozen field changed: {key}")
if settings.get("zero_edit") != {
    "dtype": "float32",
    "construction": "all_zeros",
    "shape": [10, 32],
}:
    raise SystemExit("R04B zero-edit contract changed")
if settings.get("nonzero_edit") != {
    "dtype": "float32",
    "construction": "single_model_coordinate",
    "shape": [10, 32],
    "index": [0, 0],
    "value": 0.03125,
    "role": "implementation_sentinel_only_never_support_or_dose_evidence",
}:
    raise SystemExit("R04B deterministic nonzero edit contract changed")
PY

EXPECTED_COMMIT=$(git rev-parse HEAD)
ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$MANIFEST" "$EXPERIMENT_CONFIG" \
  "$EXPECTED_COMMIT" "$MANIFEST_SHA256" "$CONFIG_SHA256" \
  "$CHECKPOINT_DIR" "$R00_SUMMARY" "$R03_SUMMARY" "$PARITY_ARTIFACT" \
  "$R04A_ARTIFACT" "$R04A_VALIDATION" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
manifest=$3
experiment_config=$4
expected_commit=$5
manifest_sha256=$6
config_sha256=$7
checkpoint_dir=$8
r00_summary=$9
r03_summary=${10}
parity_artifact=${11}
r04a_artifact=${12}
r04a_validation=${13}

expected_config_sha256=a0edb7edac86d2abd887218002d14e28e072cd472d631797d18d540f5140be33
expected_r00_summary_sha256=90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f
expected_r03_summary_sha256=dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e
expected_parity_sha256=26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a
expected_r04a_artifact_sha256=b820793ec42a5228c858e297d13806d8ae7f02e5cc7a765769316473d795f285
expected_r04a_validation_sha256=cf821650c48d30f6ebf2bbf7afe06b21938e0b137aa5acf1921f571e409e05ea
expected_checkpoint_sha256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
checkpoint=$checkpoint_dir/model.safetensors

for path in \
  "$remote_repo/scripts/hpc/run_r04_resume_parity.sh" \
  "$remote_repo/slurm/r04_resume_parity_mig.sbatch" \
  "$remote_repo/schemas/r04-resume-parity.schema.json" \
  "$remote_repo/main/run_r04_resume_parity.py" \
  "$remote_repo/main/crfs_oracle/r04_resume.py" \
  "$manifest" \
  "$experiment_config" \
  "$r00_summary" \
  "$r03_summary" \
  "$parity_artifact" \
  "$r04a_artifact" \
  "$r04a_validation" \
  "$checkpoint"; do
  test -f "$path" || { echo "missing remote R04B input: $path" >&2; exit 2; }
done
test -x "$remote_repo/scripts/hpc/run_r04_resume_parity.sh" || { echo "remote R04B runner is not executable" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || {
  echo "remote source is not synchronized to the reviewed R04B commit" >&2
  exit 2
}
test -z "$(git -C "$remote_repo" status --porcelain)" || {
  echo "remote R04B source tree is dirty" >&2
  exit 2
}
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$manifest_sha256"
test "$(sha256sum "$experiment_config" | awk '{print $1}')" = "$config_sha256"
test "$config_sha256" = "$expected_config_sha256"
test "$(sha256sum "$r00_summary" | awk '{print $1}')" = "$expected_r00_summary_sha256"
test "$(sha256sum "$r03_summary" | awk '{print $1}')" = "$expected_r03_summary_sha256"
test "$(sha256sum "$parity_artifact" | awk '{print $1}')" = "$expected_parity_sha256"
test "$(sha256sum "$r04a_artifact" | awk '{print $1}')" = "$expected_r04a_artifact_sha256"
test "$(sha256sum "$r04a_validation" | awk '{print $1}')" = "$expected_r04a_validation_sha256"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint_sha256"

# Fail closed if this 1-GPU/6-CPU/80-GiB smoke would exceed the aggregate
# user ceiling. QOS remains authoritative at scheduling time.
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
  test -n "$alloc_tres" || { echo "cannot audit allocated TRES for live job $job_id" >&2; exit 2; }
  job_cpus=$(printf '%s' "$alloc_tres" | tr ',' '\n' | sed -n 's/^cpu=\([0-9][0-9]*\)$/\1/p')
  job_mem=$(printf '%s' "$alloc_tres" | tr ',' '\n' | sed -n 's/^mem=\([^,]*\)$/\1/p')
  job_gpus=$(printf '%s' "$alloc_tres" | tr ',' '\n' | sed -n 's#^gres/gpu[^=]*=\([0-9][0-9]*\)$#\1#p' | awk '{sum += $1} END {print sum+0}')
  test -n "$job_cpus" && test -n "$job_mem" || { echo "cannot parse CPU or memory allocation for live job $job_id: $alloc_tres" >&2; exit 2; }
  job_mem_mb=$(memory_to_mb "$job_mem") || { echo "cannot parse allocated memory for live job $job_id: $job_mem" >&2; exit 2; }
  allocated_cpus=$((allocated_cpus + job_cpus))
  allocated_mem_mb=$((allocated_mem_mb + job_mem_mb))
  allocated_gpus=$((allocated_gpus + job_gpus))
done < <(squeue -h -u "$(whoami)" -t RUNNING,COMPLETING,CONFIGURING,SUSPENDED -o '%i')

projected_gpus=$((allocated_gpus + 1))
projected_cpus=$((allocated_cpus + 6))
projected_mem_mb=$((allocated_mem_mb + 80 * 1024))
echo "current_allocated_gpus=$allocated_gpus current_allocated_cpus=$allocated_cpus current_allocated_mem_mb=$allocated_mem_mb"
echo "projected_gpus=$projected_gpus projected_cpus=$projected_cpus projected_mem_mb=$projected_mem_mb"
test "$projected_gpus" -le 2 || { echo "R04B smoke would exceed the 2-GPU-equivalent user ceiling" >&2; exit 2; }
test "$projected_cpus" -le 16 || { echo "R04B smoke would exceed the 16-CPU user ceiling" >&2; exit 2; }
test "$projected_mem_mb" -le $((256 * 1024)) || { echo "R04B smoke would exceed the 256-GiB user ceiling" >&2; exit 2; }

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
  test -n "$free_mem" || { echo "cannot determine fresh FreeMem for MIG node $node" >&2; exit 2; }
  echo "live_mig_node=$node state=$state free_mem_mb=$free_mem"
  if [ "$free_mem" -ge "$minimum_free_mem_mb" ]; then
    eligible_nodes+=("$node")
  fi
done < <(sinfo -h -p mig -N -o '%N|%T' | sort -u)

test "${#eligible_nodes[@]}" -gt 0 || {
  echo "R04B smoke not submitted: no healthy MIG node has a fresh 80 GiB of free host memory" >&2
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
submission=$(
  RUN_ID="$run_id" \
  MANIFEST="$manifest" \
  EXPERIMENT_CONFIG="$experiment_config" \
  PARITY_ARTIFACT="$parity_artifact" \
  R04A_ARTIFACT="$r04a_artifact" \
  R04A_VALIDATION="$r04a_validation" \
  REMOTE_REPO="$remote_repo" \
  EXPECTED_GIT_COMMIT="$expected_commit" \
  sbatch --parsable "${sbatch_args[@]}" \
    --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' \
    slurm/r04_resume_parity_mig.sbatch
)
job_id=${submission%%;*}
case "$job_id" in
  *[!0-9]*|'')
    echo "sbatch did not return an exact numeric R04B job id: $submission" >&2
    exit 2
    ;;
esac
echo "submitted_exact_job_id=$job_id"
echo "expected_array_task=${job_id}_0"
echo "expected_artifact=/mnt/data/quanth/experiments/crfs-oracle/$run_id/crfs-93365b8b851365f2/r04-resume-parity.json"
echo "standalone_validator_source_slurm_job_id=$job_id"
echo "standalone_validator_source_slurm_array_job_id=$job_id"
echo "standalone_validator_source_slurm_array_task_id=0"
REMOTE
