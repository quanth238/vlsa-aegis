#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... submit_r05a_canary.sh [manifest] [config]'
: "${RUN_ID:?$usage}"
case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'') echo "unsafe R05A RUN_ID" >&2; exit 2 ;;
esac

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
MANIFEST_LOCAL=${1:-manifests/r05a_inverse_flow_teacher_smoke.jsonl}
CONFIG_LOCAL=${2:-configs/experiments/r05a_inverse_flow_canary.json}
SCHEMA_LOCAL=schemas/r05a-inverse-flow-canary.schema.json
DECISION_LOCAL=docs/decisions/0028-pivot-to-inverse-flow-transport.md
R03_SUMMARY_LOCAL=evidence/r03/r03-summary.json
REMOTE_MANIFEST=$REMOTE_REPO/manifests/$(basename "$MANIFEST_LOCAL")
REMOTE_CONFIG=$REMOTE_REPO/configs/experiments/$(basename "$CONFIG_LOCAL")
R02_RAW_ROOT=/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a
EXPERIMENT_ROOT=/mnt/data/quanth/experiments/crfs-oracle
CHECKPOINT_DIR=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

EXPECTED_MANIFEST_SHA256=bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633
EXPECTED_CONFIG_SHA256=c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb
EXPECTED_SCHEMA_SHA256=e1681f865f81f2986945d10fb14073fe4cd9e78cbc5026f6749ab321b9cfb7a7
EXPECTED_DECISION_SHA256=d2a00b1b049e92bb1ec8f11d60fa447e5e8bd5109cb8cf3f3fbb08f89498656f
EXPECTED_R03_SUMMARY_SHA256=dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e
EXPECTED_CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed

for path in "$MANIFEST_LOCAL" "$CONFIG_LOCAL" "$SCHEMA_LOCAL" "$DECISION_LOCAL" "$R03_SUMMARY_LOCAL"; do
  test -f "$path" || { echo "missing frozen R05A input: $path" >&2; exit 2; }
done
test "$(shasum -a 256 "$MANIFEST_LOCAL" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA256" || {
  echo "local R05A manifest hash mismatch" >&2; exit 2
}
test "$(shasum -a 256 "$CONFIG_LOCAL" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256" || {
  echo "local R05A config hash mismatch" >&2; exit 2
}
test "$(shasum -a 256 "$SCHEMA_LOCAL" | awk '{print $1}')" = "$EXPECTED_SCHEMA_SHA256" || {
  echo "local R05A schema hash mismatch" >&2; exit 2
}
test "$(shasum -a 256 "$DECISION_LOCAL" | awk '{print $1}')" = "$EXPECTED_DECISION_SHA256" || {
  echo "local R05A decision hash mismatch" >&2; exit 2
}
test "$(shasum -a 256 "$R03_SUMMARY_LOCAL" | awk '{print $1}')" = "$EXPECTED_R03_SUMMARY_SHA256" || {
  echo "local R03 summary hash mismatch" >&2; exit 2
}

git ls-files --error-unmatch \
  "$MANIFEST_LOCAL" \
  "$CONFIG_LOCAL" \
  "$SCHEMA_LOCAL" \
  "$DECISION_LOCAL" \
  main/crfs_oracle/r05a_canary.py \
  main/run_crfs_r05a_canary.py \
  main/finalize_crfs_r05a_canary.py \
  main/validate_crfs_r05a_canary.py \
  tests/test_r05a_canary.py \
  scripts/hpc/run_r05a_canary.sh \
  scripts/hpc/validate_r05a_canary.sh \
  scripts/hpc/submit_r05a_canary.sh \
  slurm/r05a_canary_h100.sbatch \
  slurm/r05a_canary_validate_cpu.sbatch >/dev/null || {
  echo "R05A canary source files must be committed before submission" >&2
  exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "R05A canary submission requires a clean reviewed worktree" >&2
  exit 2
}
EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)

# Login-node inspection only; this helper never starts Python remotely.
scripts/hpc/preflight.sh

ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$REMOTE_MANIFEST" "$REMOTE_CONFIG" \
  "$EXPECTED_GIT_COMMIT" "$EXPERIMENT_ROOT" "$R02_RAW_ROOT" "$CHECKPOINT_DIR" \
  "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_CONFIG_SHA256" "$EXPECTED_SCHEMA_SHA256" \
  "$EXPECTED_DECISION_SHA256" "$EXPECTED_R03_SUMMARY_SHA256" "$EXPECTED_CHECKPOINT_SHA256" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
manifest=$3
config=$4
expected_commit=$5
experiment_root=$6
r02_raw_root=$7
checkpoint_dir=$8
expected_manifest_sha=$9
expected_config_sha=${10}
expected_schema_sha=${11}
expected_decision_sha=${12}
expected_r03_summary_sha=${13}
expected_checkpoint_sha=${14}

schema=$remote_repo/schemas/r05a-inverse-flow-canary.schema.json
decision=$remote_repo/docs/decisions/0028-pivot-to-inverse-flow-transport.md
r03_summary=$remote_repo/evidence/r03/r03-summary.json
source_r02=$r02_raw_root/crfs-1069f29a8d76463a/r02-paired.json
checkpoint=$checkpoint_dir/model.safetensors
gpu_slurm=$remote_repo/slurm/r05a_canary_h100.sbatch
cpu_slurm=$remote_repo/slurm/r05a_canary_validate_cpu.sbatch
run_root=$experiment_root/$run_id
result=$run_root/crfs-1069f29a8d76463a/results.json
receipt=$run_root/cpu-afterany-validation.json

for path in \
  "$manifest" "$config" "$schema" "$decision" "$r03_summary" \
  "$source_r02" "$checkpoint" "$gpu_slurm" "$cpu_slurm" \
  "$remote_repo/scripts/hpc/run_r05a_canary.sh" \
  "$remote_repo/scripts/hpc/validate_r05a_canary.sh"; do
  test -e "$path" || { echo "missing remote R05A input: $path" >&2; exit 2; }
done
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || {
  echo "remote repository is not synchronized to reviewed commit" >&2; exit 2
}
test -z "$(git -C "$remote_repo" status --porcelain)" || {
  echo "remote repository is dirty" >&2; exit 2
}
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$expected_manifest_sha"
test "$(sha256sum "$config" | awk '{print $1}')" = "$expected_config_sha"
test "$(sha256sum "$schema" | awk '{print $1}')" = "$expected_schema_sha"
test "$(sha256sum "$decision" | awk '{print $1}')" = "$expected_decision_sha"
test "$(sha256sum "$r03_summary" | awk '{print $1}')" = "$expected_r03_summary_sha"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint_sha"
test "$(sha256sum "$source_r02" | awk '{print $1}')" = 055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593
test ! -e "$run_root" || { echo "immutable R05A run id is already used" >&2; exit 2; }

# Conservative exact canary launch: no competing user allocations or pending
# jobs are allowed, so its one GPU / eight CPU / 64 GiB request is auditable.
if [ -n "$(squeue -h -u "$(whoami)" -o '%i')" ]; then
  echo "R05A canary not submitted: user queue is not empty" >&2
  exit 2
fi
node_record=$(scontrol show node worker-1 -o)
node_state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$node_state" in
  *down*|*drain*|*fail*|*maint*|*not_resp*|*unknown*|*power*)
    echo "worker-1 is unhealthy: $node_state" >&2
    exit 2
    ;;
esac
free_mem=$(printf '%s\n' "$node_record" | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
case "$free_mem" in *[!0-9]*|'') echo "cannot parse worker-1 FreeMem" >&2; exit 2 ;; esac
test "$free_mem" -ge 65536 || {
  echo "R05A canary not submitted: worker-1 FreeMem=${free_mem}MiB < 65536MiB" >&2
  exit 2
}
case " $node_record " in
  *" Gres="*"gpu:nvidia_h100_80gb_hbm3:8"*) ;;
  *) echo "worker-1 does not advertise an H100 GRES" >&2; exit 2 ;;
esac
cfg_tres=$(printf '%s\n' "$node_record" | sed -n 's/.* CfgTRES=\([^ ]*\).*/\1/p')
alloc_tres=$(printf '%s\n' "$node_record" | sed -n 's/.* AllocTRES=\([^ ]*\).*/\1/p')
configured_gpus=$(printf '%s\n' "$cfg_tres" | tr ',' '\n' | sed -n 's/^gres\/gpu=\([0-9][0-9]*\)$/\1/p')
allocated_gpus=$(printf '%s\n' "$alloc_tres" | tr ',' '\n' | sed -n 's/^gres\/gpu=\([0-9][0-9]*\)$/\1/p')
case "$configured_gpus" in *[!0-9]*|'') echo "cannot parse worker-1 configured GPUs" >&2; exit 2 ;; esac
case "$allocated_gpus" in *[!0-9]*|'') allocated_gpus=0 ;; esac
free_gpus=$((configured_gpus - allocated_gpus))
test "$free_gpus" -ge 1 || {
  echo "R05A canary not submitted: worker-1 has ${free_gpus} free GPUs" >&2
  exit 2
}

mkdir "$run_root"
reservation_tmp=$(mktemp "$run_root/.launch-reservation.XXXXXX")
jq -n \
  --arg run_id "$run_id" \
  --arg commit "$expected_commit" \
  --arg free_mem "$free_mem" \
  --arg free_gpus "$free_gpus" \
  --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_canary_launch_reservation",
    status: "capacity_verified_before_submission",
    run_id: $run_id,
    git_commit: $commit,
    source_node: "worker-1",
    observed_free_mem_mib: ($free_mem | tonumber),
    observed_free_gpus: ($free_gpus | tonumber),
    requested_gpus: 1,
    requested_cpus: 8,
    requested_host_memory_mib: 65536,
    array: "0-0%1",
    scientific_claim_allowed: false,
    timestamp_utc: $timestamp
  }' >"$reservation_tmp"
mv "$reservation_tmp" "$run_root/launch-reservation.json"

gpu_submission=$(
  RUN_ID="$run_id" \
  MANIFEST="$manifest" \
  EXPERIMENT_CONFIG="$config" \
  R02_RAW_ROOT="$r02_raw_root" \
  CHECKPOINT_DIR="$checkpoint_dir" \
  EXPERIMENT_ROOT="$experiment_root" \
  EXPECTED_GIT_COMMIT="$expected_commit" \
  REMOTE_REPO="$remote_repo" \
  sbatch --parsable --hold \
    --nodelist=worker-1 \
    --array=0-0%1 \
    --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' \
    "$gpu_slurm"
)
gpu_job_id=${gpu_submission%%;*}
case "$gpu_job_id" in *[!0-9]*|'') echo "invalid GPU job id: $gpu_submission" >&2; exit 2 ;; esac

# Persist the exact held GPU id before any dependent submission or
# verification.  If a later control-plane step fails, this receipt is the
# only cleanup authority; the held job is deliberately left held.
held_tmp=$(mktemp "$run_root/.held-gpu-submission.XXXXXX")
jq -n \
  --arg run_id "$run_id" \
  --arg gpu_job_id "$gpu_job_id" \
  --arg commit "$expected_commit" \
  --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_canary_held_gpu_submission",
    status: "sbatch_returned_held_gpu_id",
    run_id: $run_id,
    git_commit: $commit,
    gpu_slurm_array_job_id: $gpu_job_id,
    gpu_slurm_array_task_id: 0,
    source_node: "worker-1",
    released_at_receipt_time: false,
    scientific_claim_allowed: false,
    timestamp_utc: $timestamp
  }' >"$held_tmp"
mv "$held_tmp" "$run_root/held-gpu-submission.json"

gpu_record=$(scontrol show job "$gpu_job_id" -o)
case " $gpu_record " in *" JobState=PENDING "*" Reason=JobHeldUser "*) ;; *) echo "GPU canary is not held" >&2; exit 2 ;; esac
case " $gpu_record " in *" ReqNodeList=worker-1 "*) ;; *) echo "GPU canary lost worker-1 pin" >&2; exit 2 ;; esac

dependency=afterany:$gpu_job_id
cpu_submission=$(
  SOURCE_JOB_ID="$gpu_job_id" \
  SOURCE_RESULT="$result" \
  VALIDATION_RECEIPT="$receipt" \
  EXPECTED_GIT_COMMIT="$expected_commit" \
  REMOTE_REPO="$remote_repo" \
  sbatch --parsable \
    --dependency="$dependency" \
    --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' \
    "$cpu_slurm"
)
cpu_job_id=${cpu_submission%%;*}
case "$cpu_job_id" in *[!0-9]*|'') echo "invalid CPU validator job id: $cpu_submission" >&2; exit 2 ;; esac
cpu_record=$(scontrol show job "$cpu_job_id" -o)
case " $cpu_record " in *" JobState=PENDING "*) ;; *) echo "CPU validator is not pending" >&2; exit 2 ;; esac
case " $cpu_record " in *" Partition=main "*) ;; *) echo "CPU validator is not on main" >&2; exit 2 ;; esac
case " $cpu_record " in *"gres/gpu"*) echo "CPU validator unexpectedly requests a GPU" >&2; exit 2 ;; esac
observed_dependency=$(printf '%s\n' "$cpu_record" | sed -n 's/.* Dependency=\([^ ]*\).*/\1/p' | sed 's/(unfulfilled)//g; s/_\*//g')
test "$observed_dependency" = "$dependency" || {
  echo "CPU validator dependency differs: $observed_dependency" >&2; exit 2
}

submission_tmp=$(mktemp "$run_root/.submission.XXXXXX")
jq -n \
  --arg run_id "$run_id" \
  --arg commit "$expected_commit" \
  --arg gpu_job_id "$gpu_job_id" \
  --arg cpu_job_id "$cpu_job_id" \
  --arg dependency "$dependency" \
  --arg result "$result" \
  --arg receipt "$receipt" \
  --arg reservation_sha "$(sha256sum "$run_root/launch-reservation.json" | awk '{print $1}')" \
  --arg held_gpu_sha "$(sha256sum "$run_root/held-gpu-submission.json" | awk '{print $1}')" \
  --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    schema_version: "1.0",
    artifact_role: "r05a_canary_atomic_submission",
    status: "validator_registered_gpu_held",
    run_id: $run_id,
    git_commit: $commit,
    source_node: "worker-1",
    gpu_slurm_array_job_id: $gpu_job_id,
    gpu_slurm_array_task_id: 0,
    cpu_afterany_job_id: $cpu_job_id,
    dependency: $dependency,
    expected_result: $result,
    expected_validation_receipt: $receipt,
    launch_reservation_sha256: $reservation_sha,
    held_gpu_submission_sha256: $held_gpu_sha,
    scientific_claim_allowed: false,
    timestamp_utc: $timestamp
  }' >"$submission_tmp"
mv "$submission_tmp" "$run_root/submission.json"

# Release only after the independent afterany job and immutable receipt exist.
scontrol release "$gpu_job_id"
echo "submitted_gpu_job_id=$gpu_job_id"
echo "submitted_cpu_validator_job_id=$cpu_job_id"
echo "dependency=$dependency"
echo "submission_receipt=$run_root/submission.json"
echo "submission_receipt_sha256=$(sha256sum "$run_root/submission.json" | awk '{print $1}')"
REMOTE
