#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... submit_r06_aegis_capture_canary.sh configs/experiments/r06_aegis_collision_conditioned.json'
CONFIG_LOCAL=${1:?$usage}
: "${RUN_ID:?set an immutable R06 capture RUN_ID}"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
CONFIG_REMOTE=$REMOTE_REPO/configs/experiments/$(basename "$CONFIG_LOCAL")
EXPECTED_COMMIT=$(git rev-parse HEAD)
CONFIG_SHA256=$(shasum -a 256 "$CONFIG_LOCAL" | awk '{print $1}')

case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe R06 capture run id" >&2; exit 2 ;; esac
test "$(basename "$CONFIG_LOCAL")" = r06_aegis_collision_conditioned.json || {
  echo "R06 capture requires the registered config filename" >&2
  exit 2
}
git ls-files --error-unmatch \
  "$CONFIG_LOCAL" \
  main/run_crfs_r06_aegis_capture.py \
  main/crfs_oracle/aegis_baseline.py \
  main/crfs_oracle/aegis_perception.py \
  main/crfs_oracle/aegis_runner.py \
  main/crfs_oracle/aegis_pairing.py \
  main/main_aegis.py \
  main/main_aegis_translational.py \
  main/utils.py \
  docs/decisions/0064-preregister-aegis-collision-conditioned-baseline.md \
  docs/decisions/0065-use-codex-frozen-semantic-labels-for-aegis.md \
  evidence/r06/groundingdino-setup-28391.json \
  scripts/hpc/run_r06_aegis_capture_canary.sh \
  scripts/hpc/run_r06_aegis_capture_workload.sh \
  scripts/hpc/submit_r06_aegis_capture_canary.sh \
  slurm/r06_aegis_capture_canary_h100.sbatch \
  tests/test_aegis_baseline.py \
  tests/test_aegis_pairing.py \
  tests/test_aegis_perception.py \
  tests/test_aegis_runner.py \
  tests/test_r06_aegis_capture_hpc_contract.py >/dev/null || {
  echo "R06 capture sources must be committed before submission" >&2
  exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "R06 capture submission requires no tracked local changes" >&2
  exit 2
}
python3 - "$CONFIG_LOCAL" "$RUN_ID" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
run_id = sys.argv[2]
release = value.get("execution_release")
if value.get("ready_to_run") is not True or value.get("blocked_on") != []:
    raise SystemExit("R06 capture config is not released")
required = {
    "artifact_role": "r06_aegis_codex_label_capture_execution_release",
    "stage": "codex_label_capture",
    "run_id": run_id,
    "single_case_index": 0,
    "case_id": "crfs-1069f29a8d76463a",
    "aegis_execution_allowed": False,
    "semantic_label_required": False,
    "groundingdino_execution_allowed": False,
    "qp_execution_allowed": False,
    "robosuite_image_convention": "opengl",
    "robosuite_version": "1.4.1",
    "probe_or_mlp_training_authorized": False,
    "automatic_population_launch_authorized": False,
    "release_only_parent_required": True,
}
if not isinstance(release, dict):
    raise SystemExit("R06 capture execution_release is missing")
for key, expected in required.items():
    if release.get(key) != expected:
        raise SystemExit(f"R06 capture release field changed: {key}")
expected_resources = {
    "partition": "main",
    "account": "normal",
    "qos": "normal",
    "gpus": 1,
    "cpus_per_task": 8,
    "host_memory_mib": 65536,
    "time_limit": "00:30:00",
    "array": "0-0%1",
    "source_host": "worker-1",
    "requeue": False,
}
if release.get("resources") != expected_resources:
    raise SystemExit("R06 capture release resources changed")
accepted = release.get("accepted_implementation_commit")
if not (
    isinstance(accepted, str)
    and len(accepted) == 40
    and all(character in "0123456789abcdef" for character in accepted)
):
    raise SystemExit("R06 capture accepted implementation commit is invalid")
decision = release.get("decision_artifact")
if not (
    isinstance(decision, str)
    and decision.startswith("docs/decisions/")
    and decision.endswith("-release-aegis-label-capture-canary.md")
):
    raise SystemExit("R06 capture release ADR path changed")
if release.get("allowed_release_diff_paths") != [
    "configs/experiments/r06_aegis_collision_conditioned.json",
    decision,
]:
    raise SystemExit("R06 capture release diff allowlist changed")
PY
ACCEPTED_IMPLEMENTATION_COMMIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["execution_release"]["accepted_implementation_commit"])' "$CONFIG_LOCAL")
RELEASE_ADR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["execution_release"]["decision_artifact"])' "$CONFIG_LOCAL")
test -f "$RELEASE_ADR" && test ! -L "$RELEASE_ADR" || { echo "local R06 release ADR is missing or symlinked" >&2; exit 2; }
git ls-files --error-unmatch "$RELEASE_ADR" >/dev/null || {
  echo "local R06 release ADR must be committed before submission" >&2
  exit 2
}
test "$(git rev-list --parents -n 1 "$EXPECTED_COMMIT")" = "$EXPECTED_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || {
  echo "local R06 release is not the direct child of its implementation" >&2
  exit 2
}
observed_release_diff=$(git diff --name-only "$ACCEPTED_IMPLEMENTATION_COMMIT" "$EXPECTED_COMMIT" | LC_ALL=C sort)
expected_release_diff=$(printf '%s\n' \
  configs/experiments/r06_aegis_collision_conditioned.json \
  "$RELEASE_ADR" | LC_ALL=C sort)
test "$observed_release_diff" = "$expected_release_diff" || {
  echo "local R06 release diff is not exact config plus release ADR" >&2
  exit 2
}

ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$CONFIG_REMOTE" "$RUN_ID" "$EXPECTED_COMMIT" "$CONFIG_SHA256" <<'REMOTE'
set -euo pipefail
remote_repo=$1
config=$2
run_id=$3
expected_commit=$4
config_sha256=$5

manifest=$remote_repo/manifests/oracle_h05_colliding.jsonl
r02_config=$remote_repo/configs/experiments/r02_oracle_flow.json
checkpoint=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors
checkpoint_config=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/config.json
normalization=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json
source_r02=/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a/crfs-1069f29a8d76463a/r02-paired.json
run_root=/mnt/data/quanth/experiments/crfs-oracle/$run_id
source_contract=$run_root/source-contract.json
submission=$run_root/submission.json
slurm_file=$remote_repo/slurm/r06_aegis_capture_canary_h100.sbatch

for path in \
  "$config" "$manifest" "$r02_config" "$checkpoint" "$checkpoint_config" "$normalization" "$source_r02" \
  "$slurm_file" "$remote_repo/scripts/hpc/run_r06_aegis_capture_canary.sh" \
  "$remote_repo/scripts/hpc/run_r06_aegis_capture_workload.sh" \
  "$remote_repo/main/run_crfs_r06_aegis_capture.py"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked remote R06 input: $path" >&2; exit 2; }
done
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || { echo "remote R06 commit differs" >&2; exit 2; }
test -z "$(git -C "$remote_repo" status --porcelain)" || { echo "remote R06 tree is dirty" >&2; exit 2; }
test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha256"
test "$(sha256sum "$manifest" | awk '{print $1}')" = b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41
test "$(sha256sum "$r02_config" | awk '{print $1}')" = c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = 988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed
test "$(sha256sum "$checkpoint_config" | awk '{print $1}')" = 5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a
test "$(sha256sum "$normalization" | awk '{print $1}')" = b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84
test "$(sha256sum "$source_r02" | awk '{print $1}')" = 055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593
accepted_implementation=$(jq -er '.execution_release.accepted_implementation_commit' "$config")
release_adr=$(jq -er '.execution_release.decision_artifact' "$config")
case "$release_adr" in docs/decisions/*-release-aegis-label-capture-canary.md) ;; *) echo "remote R06 release ADR changed" >&2; exit 2 ;; esac
test -f "$remote_repo/$release_adr" && test ! -L "$remote_repo/$release_adr" || { echo "remote R06 release ADR missing" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-list --parents -n 1 "$expected_commit")" = \
  "$expected_commit $accepted_implementation" || { echo "remote R06 release parent changed" >&2; exit 2; }
observed_release_diff=$(git -C "$remote_repo" diff --name-only "$accepted_implementation" "$expected_commit" | LC_ALL=C sort)
expected_release_diff=$(printf '%s\n' \
  configs/experiments/r06_aegis_collision_conditioned.json \
  "$release_adr" | LC_ALL=C sort)
test "$observed_release_diff" = "$expected_release_diff" || { echo "remote R06 release diff changed" >&2; exit 2; }
release_adr_sha=$(sha256sum "$remote_repo/$release_adr" | awk '{print $1}')
test ! -e "$run_root" || { echo "immutable R06 capture run id is already used" >&2; exit 2; }

node_record=$(scontrol show node worker-1 -o)
state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$state" in *down*|*drain*|*fail*|*maint*|*not_resp*|*unknown*|*power*) echo "worker-1 is unhealthy: $state" >&2; exit 2 ;; esac

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

source_tmp=$(mktemp "$run_root/.source-contract.XXXXXX")
jq -n \
  --arg run "$run_id" \
  --arg commit "$expected_commit" \
  --arg implementation "$accepted_implementation" \
  --arg release_adr "$release_adr" \
  --arg release_adr_sha "$release_adr_sha" \
  --arg config_sha "$config_sha256" \
  --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
  {
    schema_version:"1.0",
    artifact_role:"r06_aegis_label_capture_source_contract",
    status:"sources_bound_before_held_submission",
    run_id:$run,
    git_commit:$commit,
    accepted_implementation_commit:$implementation,
    release_decision_artifact:$release_adr,
    release_decision_sha256:$release_adr_sha,
    release_only_parent_required:true,
    git_dirty:false,
    config_sha256:$config_sha,
    manifest_sha256:"b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41",
    r02_config_sha256:"c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e",
    source_r02_pair_sha256:"055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
    checkpoint_sha256:"988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
    checkpoint_config_sha256:"5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a",
    normalization_asset_sha256:"b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
    robosuite_image_convention:"opengl",
    robosuite_version:"1.4.1",
    stage:"codex_label_capture",
    resources:{partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"00:30:00",array:"0-0%1",source_host:"worker-1",requeue:false},
    aegis_execution_allowed:false,
    groundingdino_execution_allowed:false,
    qp_execution_allowed:false,
    probe_or_mlp_training_authorized:false,
    timestamp_utc:$now
  }' >"$source_tmp"
mv "$source_tmp" "$source_contract"
source_sha=$(sha256sum "$source_contract" | awk '{print $1}')

mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle
submission_output=$(RUN_ID="$run_id" EXPECTED_GIT_COMMIT="$expected_commit" \
  EXPECTED_CONFIG_SHA256="$config_sha256" SOURCE_CONTRACT="$source_contract" \
  SUBMISSION_RECEIPT="$submission" REMOTE_REPO="$remote_repo" \
  sbatch --parsable --hold --export=ALL \
  --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' "$slurm_file")
job_id=${submission_output%%;*}
case "$job_id" in *[!0-9]*|'') echo "invalid R06 capture Slurm job id" >&2; exit 2 ;; esac
released=false
cleanup_exact_held_task() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ "$released" = false ]; then
    echo "R06 capture submission failed after reserving exact held task ${job_id}_0" >&2
    if scontrol show job "${job_id}_0" -o >&2; then
      scancel "${job_id}_0"
      echo "cancelled_exact_unreleased_task=${job_id}_0" >&2
    fi
  fi
  return "$status"
}
trap cleanup_exact_held_task EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
task_record=$(scontrol show job "${job_id}_0" -o)
for field in \
  "ArrayJobId=$job_id" "ArrayTaskId=0" JobState=PENDING Reason=JobHeldUser \
  Partition=main Account=normal QOS=normal TimeLimit=00:30:00 Requeue=0 \
  ReqNodeList=worker-1 NumCPUs=8 CPUs/Task=8; do
  case " $task_record " in *" $field "*) ;; *) echo "held R06 task field changed: $field" >&2; exit 2 ;; esac
done

submission_tmp=$(mktemp "$run_root/.submission.XXXXXX")
jq -n \
  --arg run "$run_id" \
  --arg commit "$expected_commit" \
  --arg implementation "$accepted_implementation" \
  --arg release_adr "$release_adr" \
  --arg release_adr_sha "$release_adr_sha" \
  --arg config_sha "$config_sha256" \
  --arg source_sha "$source_sha" \
  --arg job "$job_id" \
  --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
  {
    schema_version:"1.0",
    artifact_role:"r06_aegis_label_capture_submission",
    status:"held_task_validated_before_release",
    run_id:$run,
    git_commit:$commit,
    accepted_implementation_commit:$implementation,
    release_decision_artifact:$release_adr,
    release_decision_sha256:$release_adr_sha,
    release_only_parent_required:true,
    config_sha256:$config_sha,
    source_contract_sha256:$source_sha,
    slurm_array_job_id:$job,
    slurm_array_task_id:0,
    exact_task_id:($job+"_0"),
    source_host:"worker-1",
    released_at_receipt_time:false,
    timestamp_utc:$now
  }' >"$submission_tmp"
mv "$submission_tmp" "$submission"
scontrol release "$job_id"
released=true
trap - EXIT INT TERM
printf 'r06_capture_job_id=%s\n' "$job_id"
printf 'r06_capture_exact_task_id=%s_0\n' "$job_id"
printf 'r06_capture_run_root=%s\n' "$run_root"
REMOTE
