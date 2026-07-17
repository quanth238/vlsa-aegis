#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... submit_r06_aegis_paired_canary.sh configs/experiments/r06_aegis_collision_conditioned.json'
CONFIG_LOCAL=${1:?$usage}
: "${RUN_ID:?$usage}"
case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe R06 paired run id" >&2; exit 2 ;; esac

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
EXPECTED_COMMIT=$(git rev-parse HEAD)
CONFIG_SHA256=$(shasum -a 256 "$CONFIG_LOCAL" | awk '{print $1}')
test "$(basename "$CONFIG_LOCAL")" = r06_aegis_collision_conditioned.json

required=(
  "$CONFIG_LOCAL"
  main/run_crfs_r06_aegis_paired_canary.py
  main/validate_crfs_r06_aegis_paired_canary.py
  main/crfs_oracle/aegis_baseline.py
  main/crfs_oracle/aegis_pairing.py
  main/crfs_oracle/aegis_perception.py
  main/crfs_oracle/aegis_runner.py
  main/main_aegis.py
  main/main_aegis_translational.py
  main/utils.py
  manifests/oracle_h05_colliding.jsonl
  manifests/r06_codex_obstacle_labels_canary.jsonl
  scripts/hpc/run_r06_aegis_paired_canary.sh
  scripts/hpc/run_r06_aegis_paired_workload.sh
  scripts/hpc/validate_r06_aegis_paired_canary.sh
  scripts/hpc/submit_r06_aegis_paired_canary.sh
  slurm/r06_aegis_paired_canary_h100.sbatch
  slurm/r06_aegis_paired_canary_validate_cpu.sbatch
  tests/test_r06_aegis_paired_hpc_contract.py
)
git ls-files --error-unmatch "${required[@]}" >/dev/null || {
  echo "R06 paired sources must be committed before submission" >&2; exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "R06 paired submission requires no tracked local changes" >&2; exit 2
}

python3 - "$CONFIG_LOCAL" "$RUN_ID" "$ROOT" <<'PY'
import importlib.util
import json
import pathlib
import sys

config_path, run_id, root = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
spec = importlib.util.spec_from_file_location(
    "r06_paired_cli", root / "main/run_crfs_r06_aegis_paired_canary.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
value = json.loads(config_path.read_text(encoding="utf-8"))
module._require_paired_release(value, run_id=run_id)
PY
ACCEPTED_IMPLEMENTATION_COMMIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["execution_release"]["accepted_implementation_commit"])' "$CONFIG_LOCAL")
RELEASE_ADR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["execution_release"]["decision_artifact"])' "$CONFIG_LOCAL")
test -f "$RELEASE_ADR" && test ! -L "$RELEASE_ADR"
git ls-files --error-unmatch "$RELEASE_ADR" >/dev/null
test "$(git rev-list --parents -n 1 "$EXPECTED_COMMIT")" = \
  "$EXPECTED_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT"
test "$(git diff --name-only "$ACCEPTED_IMPLEMENTATION_COMMIT" "$EXPECTED_COMMIT" | LC_ALL=C sort)" = \
  "$(printf '%s\n' configs/experiments/r06_aegis_collision_conditioned.json "$RELEASE_ADR" | LC_ALL=C sort)"

ssh "$HOST" bash -s -- "$REMOTE_REPO" "$RUN_ID" "$EXPECTED_COMMIT" \
  "$ACCEPTED_IMPLEMENTATION_COMMIT" "$RELEASE_ADR" "$CONFIG_SHA256" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
expected_commit=$3
accepted_implementation=$4
release_adr_relative=$5
config_sha256=$6
run_root=/mnt/data/quanth/experiments/crfs-oracle/$run_id
config=$remote_repo/configs/experiments/r06_aegis_collision_conditioned.json
manifest=$remote_repo/manifests/oracle_h05_colliding.jsonl
label=$remote_repo/manifests/r06_codex_obstacle_labels_canary.jsonl
r02_config=$remote_repo/configs/experiments/r02_oracle_flow.json
source_r02=/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a/crfs-1069f29a8d76463a/r02-paired.json
capture=/mnt/data/quanth/experiments/crfs-oracle/r06-aegis-label-capture-canary-20260717b/crfs-1069f29a8d76463a/capture.json
checkpoint=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors
checkpoint_config=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/config.json
normalization=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json
dino_config=/mnt/data/quanth/cache/uv/archive-v0/hHOpLbugg_lAlUaF/groundingdino/config/GroundingDINO_SwinT_OGC.py
dino_checkpoint=/mnt/data/quanth/cache/aegis/groundingdino/groundingdino_swint_ogc.pth
gpu_slurm=$remote_repo/slurm/r06_aegis_paired_canary_h100.sbatch
cpu_slurm=$remote_repo/slurm/r06_aegis_paired_canary_validate_cpu.sbatch
source_contract=$run_root/source-contract.json
submission=$run_root/submission.json
release_adr=$remote_repo/$release_adr_relative

for path in "$config" "$manifest" "$label" "$r02_config" "$source_r02" "$capture" \
  "$checkpoint" "$checkpoint_config" "$normalization" "$dino_config" "$dino_checkpoint" \
  "$gpu_slurm" "$cpu_slurm" "$release_adr"; do
  test -f "$path" && test ! -L "$path" || { echo "missing remote paired input: $path" >&2; exit 2; }
done
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$remote_repo" status --porcelain)"
test "$(git -C "$remote_repo" rev-list --parents -n 1 "$expected_commit")" = \
  "$expected_commit $accepted_implementation"
test "$(git -C "$remote_repo" diff --name-only "$accepted_implementation" "$expected_commit" | LC_ALL=C sort)" = \
  "$(printf '%s\n' configs/experiments/r06_aegis_collision_conditioned.json "$release_adr_relative" | LC_ALL=C sort)"
test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha256"
test "$(sha256sum "$manifest" | awk '{print $1}')" = b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41
test "$(sha256sum "$label" | awk '{print $1}')" = 6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f
test "$(sha256sum "$r02_config" | awk '{print $1}')" = c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e
test "$(sha256sum "$source_r02" | awk '{print $1}')" = 055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593
test "$(sha256sum "$capture" | awk '{print $1}')" = f2d32024ac6fac5aa5d133b5138df72501f1590b8cb0927b732edde69dabaaac
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = 988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
test "$(sha256sum "$checkpoint_config" | awk '{print $1}')" = 5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a
test "$(sha256sum "$normalization" | awk '{print $1}')" = b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84
test "$(sha256sum "$dino_config" | awk '{print $1}')" = 172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1
test "$(sha256sum "$dino_checkpoint" | awk '{print $1}')" = 3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799
test ! -e "$run_root" || { echo "immutable R06 paired run id is already used" >&2; exit 2; }
inherited=$(env | sed -n 's/^\(SBATCH_[A-Za-z0-9_]*\)=.*/\1/p')
test -z "$inherited" || { echo "inherited SBATCH options could alter R06: $inherited" >&2; exit 2; }
node_record=$(scontrol show node worker-1 -o)
node_state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$node_state" in *down*|*drain*|*fail*|*maint*|*not_resp*|*unknown*|*power*) echo "worker-1 unhealthy: $node_state" >&2; exit 2 ;; esac

mkdir "$run_root"
preflight_tmp=$(mktemp "$run_root/.preflight.XXXXXX")
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  squeue -u "$(whoami)" -o '%.18i %.32j %.2t %.10M %.12l %.6D %R'
  printf '%s\n' "$node_record"
  df -h /mnt/data
} >"$preflight_tmp"
mv "$preflight_tmp" "$run_root/vinuni-preflight.txt"

tree=$(git -C "$remote_repo" rev-parse "$expected_commit^{tree}")
release_adr_sha=$(sha256sum "$release_adr" | awk '{print $1}')
source_tmp=$(mktemp "$run_root/.source-contract.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" \
  --arg tree "$tree" --arg implementation "$accepted_implementation" \
  --arg release_adr "$release_adr_relative" --arg release_adr_sha "$release_adr_sha" \
  --arg config_sha "$config_sha256" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{
    schema_version:"1.0",
    artifact_role:"r06_aegis_paired_canary_source_contract",
    status:"sources_bound_before_held_submission",
    run_id:$run,
    git_commit:$commit,
    git_tree:$tree,
    git_dirty:false,
    accepted_implementation_commit:$implementation,
    release_decision_artifact:$release_adr,
    release_decision_sha256:$release_adr_sha,
    release_only_parent_required:true,
    config_sha256:$config_sha,
    manifest_sha256:"b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41",
    r02_config_sha256:"c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e",
    source_r02_pair_sha256:"055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
    checkpoint_sha256:"988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
    checkpoint_config_sha256:"5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a",
    normalization_asset_sha256:"b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
    groundingdino_config_sha256:"172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1",
    groundingdino_checkpoint_sha256:"3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
    codex_label_manifest_sha256:"6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f",
    codex_label_freeze_commit:"d9cf569d619e014c9e6423cd9ddb40f592435a71",
    capture_artifact_sha256:"f2d32024ac6fac5aa5d133b5138df72501f1590b8cb0927b732edde69dabaaac",
    capture_completed_at:"2026-07-17T06:55:59Z",
    stage:"paired_codex_label_canary",
    resources:{partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"00:30:00",array:"0-0%1",source_host:"worker-1",requeue:false,validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:10:00",validator_gpus:0,validator_dependency:"afterany"},
    collision_conditioned_only:true,
    original_end_to_end_aegis:false,
    probe_or_mlp_training_authorized:false,
    automatic_population_launch_authorized:false,
    timestamp_utc:$now
  }' >"$source_tmp"
mv "$source_tmp" "$source_contract"
source_sha=$(sha256sum "$source_contract" | awk '{print $1}')

mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle
gpu_output=$(RUN_ID="$run_id" EXPECTED_GIT_COMMIT="$expected_commit" \
  EXPECTED_CONFIG_SHA256="$config_sha256" SOURCE_CONTRACT="$source_contract" \
  SUBMISSION_RECEIPT="$submission" REMOTE_REPO="$remote_repo" \
  sbatch --parsable --hold --export=ALL \
  --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' "$gpu_slurm")
gpu_id=${gpu_output%%;*}
case "$gpu_id" in *[!0-9]*|'') echo "invalid R06 paired GPU job id" >&2; exit 2 ;; esac
released=false
cpu_id=
cleanup_exact_jobs() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ "$released" = false ]; then
    if [ -n "$cpu_id" ] && scontrol show job "$cpu_id" -o >&2; then scancel "$cpu_id"; fi
    if scontrol show job "${gpu_id}_0" -o >&2; then scancel "${gpu_id}_0"; fi
  fi
  return "$status"
}
trap cleanup_exact_jobs EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

gpu_record=$(scontrol show job "${gpu_id}_0" -o)
for field in "ArrayJobId=$gpu_id" "ArrayTaskId=0" JobState=PENDING Reason=JobHeldUser \
  Partition=main Account=normal QOS=normal TimeLimit=00:30:00 Requeue=0 \
  ReqNodeList=worker-1 NumCPUs=8 CPUs/Task=8; do
  case " $gpu_record " in *" $field "*) ;; *) echo "held R06 GPU field changed: $field" >&2; exit 2 ;; esac
done

cpu_output=$(RUN_ID="$run_id" SOURCE_JOB_ID="$gpu_id" EXPECTED_GIT_COMMIT="$expected_commit" \
  SOURCE_CONTRACT="$source_contract" SUBMISSION_RECEIPT="$submission" \
  REMOTE_REPO="$remote_repo" sbatch --parsable --export=ALL \
  --dependency="afterany:$gpu_id" \
  --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' "$cpu_slurm")
cpu_id=${cpu_output%%;*}
case "$cpu_id" in *[!0-9]*|'') echo "invalid R06 paired CPU job id" >&2; exit 2 ;; esac
cpu_record=$(scontrol show job "$cpu_id" -o)
for field in "JobId=$cpu_id" JobState=PENDING Partition=main Account=normal QOS=normal \
  TimeLimit=00:10:00 Requeue=0 NumCPUs=2 CPUs/Task=2; do
  case " $cpu_record " in *" $field "*) ;; *) echo "R06 CPU field changed: $field" >&2; exit 2 ;; esac
done
dependency=$(printf '%s\n' "$cpu_record" | sed -n 's/.* Dependency=\([^ ]*\).*/\1/p' | sed 's/(unfulfilled)//g; s/_\*//g')
test "$dependency" = "afterany:$gpu_id" || { echo "R06 CPU dependency changed: $dependency" >&2; exit 2; }
case " $cpu_record " in *" TRESPerNode=gres/gpu:"*) echo "R06 CPU validator requests a GPU" >&2; exit 2 ;; esac

submission_tmp=$(mktemp "$run_root/.submission.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg source "$source_sha" \
  --arg gpu "$gpu_id" --arg cpu "$cpu_id" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{
    schema_version:"1.0",
    artifact_role:"r06_aegis_paired_canary_atomic_submission",
    status:"gpu_and_afterany_registered_before_release",
    run_id:$run,
    git_commit:$commit,
    source_contract_sha256:$source,
    gpu_slurm_array_job_id:$gpu,
    gpu_slurm_array_task_id:0,
    exact_gpu_task_id:($gpu+"_0"),
    cpu_afterany_job_id:$cpu,
    dependency:("afterany:"+$gpu),
    source_host:"worker-1",
    released_at_receipt_time:false,
    collision_conditioned_only:true,
    population_launch_authorized:false,
    probe_or_mlp_training_authorized:false,
    timestamp_utc:$now
  }' >"$submission_tmp"
mv "$submission_tmp" "$submission"
submission_sha=$(sha256sum "$submission" | awk '{print $1}')
scontrol release "$gpu_id"
released=true
trap - EXIT INT TERM
printf 'r06_paired_gpu_job_id=%s\n' "$gpu_id"
printf 'r06_paired_exact_gpu_task_id=%s_0\n' "$gpu_id"
printf 'r06_paired_cpu_validator_job_id=%s\n' "$cpu_id"
printf 'r06_paired_source_contract_sha256=%s\n' "$source_sha"
printf 'r06_paired_submission_sha256=%s\n' "$submission_sha"
printf 'r06_paired_run_root=%s\n' "$run_root"
REMOTE
