#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... submit_generated_source_pilot.sh configs/experiments/task0_single_obstacle_generated_v1.json'
CONFIG_LOCAL=${1:?$usage}
: "${RUN_ID:?set RUN_ID to an immutable retired source-integrity pilot identifier}"
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
OUTPUT_ROOT=/mnt/data/quanth/experiments/crfs-oracle
EXPERIMENT_CONFIG=$REMOTE_REPO/configs/experiments/task0_single_obstacle_generated_v1.json
EXPECTED_CONFIG_SHA256=332c90fdb9e4560ab522e6333fbe5f0846d1c7caca9190f1730436036856f22a

test "$(basename "$CONFIG_LOCAL")" = task0_single_obstacle_generated_v1.json || {
  echo "generated-source submission requires the frozen task0_single_obstacle_generated_v1 config" >&2
  exit 2
}
test -f "$CONFIG_LOCAL" || {
  echo "missing local generated-source config: $CONFIG_LOCAL" >&2
  exit 2
}
CONFIG_SHA256=$(shasum -a 256 "$CONFIG_LOCAL" | awk '{print $1}')
test "$CONFIG_SHA256" = "$EXPECTED_CONFIG_SHA256" || {
  echo "local frozen generated-source config hash mismatch" >&2
  exit 2
}

python3 - "$CONFIG_LOCAL" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
config = json.loads(path.read_text(encoding="utf-8"))
if config.get("ready_to_run") is not True:
    raise SystemExit("generated-source pilot config is not ready_to_run")
if config.get("blocked_on") != []:
    raise SystemExit("generated-source pilot config still has blocked dependencies")
if config.get("pilot_only") is not True:
    raise SystemExit("generated-source pilot must remain pilot_only")
if config.get("ready_for_training") is not False or config.get("ready_for_claims") is not False:
    raise SystemExit("generated-source pilot cannot authorize training or claims")
groups = config.get("source_groups") if isinstance(config, dict) else None
if not isinstance(groups, list) or len(groups) != 10:
    raise SystemExit("retired source-integrity config must contain 10 request slots")
expected_ids = [f"greq-task0-single-obstacle-v1-{index:04d}" for index in range(10)]
observed_ids = [
    group.get("generation_request_id") if isinstance(group, dict) else None
    for group in groups
]
if observed_ids != expected_ids:
    raise SystemExit("retired source-integrity generation-request identities changed")
PY

SOURCE_FILES=(
  configs/experiments/task0_single_obstacle_generated_v1.json
  main/crfs_oracle/generated_source.py
  main/generate_task0_single_obstacle_source.py
  main/validate_task0_single_obstacle_source.py
  schemas/generated-source-state.schema.json
  scripts/hpc/run_generated_source_pilot.sh
  scripts/hpc/run_generated_source_validation.sh
  scripts/hpc/submit_generated_source_pilot.sh
  slurm/generated_source_pilot_main.sbatch
  slurm/generated_source_validate_cpu.sbatch
)
for source_file in "${SOURCE_FILES[@]}"; do
  git ls-files --error-unmatch "$source_file" >/dev/null 2>&1 || {
    echo "generated-source pilot source files must be committed before submission: $source_file" >&2
    exit 2
  }
done
git diff --quiet && git diff --cached --quiet || {
  echo "generated-source submission requires a clean reviewed tracked worktree" >&2
  exit 2
}
EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)

# The live cluster state is authoritative immediately before the remote
# allocation audit and submission.  This command performs no experiment work.
scripts/hpc/preflight.sh

ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$EXPERIMENT_CONFIG" "$CONFIG_SHA256" \
  "$EXPECTED_GIT_COMMIT" "$OUTPUT_ROOT" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
experiment_config=$3
config_sha256=$4
expected_commit=$5
output_root=$6

for path in \
  "$experiment_config" \
  "$remote_repo/main/crfs_oracle/generated_source.py" \
  "$remote_repo/main/generate_task0_single_obstacle_source.py" \
  "$remote_repo/main/validate_task0_single_obstacle_source.py" \
  "$remote_repo/schemas/generated-source-state.schema.json" \
  "$remote_repo/scripts/hpc/run_generated_source_pilot.sh" \
  "$remote_repo/scripts/hpc/run_generated_source_validation.sh" \
  "$remote_repo/slurm/generated_source_pilot_main.sbatch" \
  "$remote_repo/slurm/generated_source_validate_cpu.sbatch"; do
  test -f "$path" || { echo "missing remote generated-source source: $path" >&2; exit 2; }
done
test -x "$remote_repo/scripts/hpc/run_generated_source_pilot.sh" || {
  echo "remote generated-source allocation runner is not executable" >&2
  exit 2
}
test -x "$remote_repo/scripts/hpc/run_generated_source_validation.sh" || {
  echo "remote generated-source validation runner is not executable" >&2
  exit 2
}
test "$(sha256sum "$experiment_config" | awk '{print $1}')" = "$config_sha256" || {
  echo "remote generated-source config hash mismatch" >&2
  exit 2
}
remote_commit=$(git -C "$remote_repo" rev-parse HEAD)
remote_dirty=$(test -n "$(git -C "$remote_repo" status --porcelain)" && echo true || echo false)
test "$remote_commit" = "$expected_commit" || {
  echo "remote source is not synchronized to the reviewed generated-source commit" >&2
  exit 2
}
test "$remote_dirty" = false || {
  echo "remote generated-source source tree is dirty" >&2
  exit 2
}
test ! -e "$output_root/$run_id" || {
  echo "immutable generated-source RUN_ID already exists; choose a new identifier" >&2
  exit 2
}

if grep -Eq 'serve_policy\.py|policy:checkpoint|WebsocketClientPolicy|CHECKPOINT_DIR|OPENPI_PYTHON' \
  "$remote_repo/scripts/hpc/run_generated_source_pilot.sh"; then
  echo "generated-source runner unexpectedly contains model-inference machinery" >&2
  exit 2
fi

memory_to_mb() {
  value=$1
  case "$value" in
    *[Kk]) number=${value%?}; echo $(((number + 1023) / 1024)) ;;
    *[Mm]) echo "${value%?}" ;;
    *[Gg]) echo $((${value%?} * 1024)) ;;
    *[Tt]) echo $((${value%?} * 1024 * 1024)) ;;
    *[!0-9]*|'') return 1 ;;
    *) echo "$value" ;;
  esac
}

allocated_cpus=0
allocated_mem_mb=0
allocated_gpus=0
while IFS= read -r job_id; do
  test -n "$job_id" || continue
  job_record=$(scontrol show job "$job_id" -o)
  alloc_tres=$(printf '%s\n' "$job_record" | sed -n 's/.* AllocTRES=\([^ ]*\).*/\1/p')
  test -n "$alloc_tres" || {
    echo "cannot determine fresh AllocTRES for live job $job_id" >&2
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
projected_cpus=$((allocated_cpus + 4))
projected_mem_mb=$((allocated_mem_mb + 32 * 1024))
echo "current_allocated_gpus=$allocated_gpus current_allocated_cpus=$allocated_cpus current_allocated_mem_mb=$allocated_mem_mb"
echo "projected_gpus=$projected_gpus projected_cpus=$projected_cpus projected_mem_mb=$projected_mem_mb"
test "$projected_gpus" -le 2 || {
  echo "generated-source pilot would exceed the 2-GPU-equivalent user ceiling" >&2
  exit 2
}
test "$projected_cpus" -le 16 || {
  echo "generated-source pilot would exceed the 16-CPU user ceiling" >&2
  exit 2
}
test "$projected_mem_mb" -le $((256 * 1024)) || {
  echo "generated-source pilot would exceed the 256-GiB user ceiling" >&2
  exit 2
}

minimum_free_mem_mb=$((32 * 1024))
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
  echo "generated-source pilot not submitted: no healthy MIG node has a fresh 32 GiB of free host memory" >&2
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
  EXPERIMENT_CONFIG="$experiment_config" \
  OUTPUT_ROOT="$output_root" \
  REMOTE_REPO="$remote_repo" \
  EXPECTED_GIT_COMMIT="$expected_commit" \
  sbatch --parsable "${sbatch_args[@]}" \
    --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' \
    slurm/generated_source_pilot_main.sbatch
)
job_id=${submission%%;*}
case "$job_id" in
  *[!0-9]*|'')
    echo "sbatch did not return an exact numeric generated-source job id: $submission" >&2
    exit 2
    ;;
esac
echo "submitted_exact_job_id=$job_id"
echo "pilot_phase=allocation_canary"
echo "expected_array_task=${job_id}_0"
artifact=$output_root/$run_id/greq-task0-single-obstacle-v1-0000/source-bundle.json
echo "expected_artifact=$artifact"

validation_submission=$(
  SOURCE_JOB_ID="$job_id" \
  SOURCE_ARRAY_JOB_ID="$job_id" \
  SOURCE_ARRAY_TASK_ID=0 \
  SOURCE_ARTIFACT="$artifact" \
  EXPECTED_CONFIG_SHA256="$config_sha256" \
  EXPECTED_GIT_COMMIT="$expected_commit" \
  GENERATION_REQUEST_ID=greq-task0-single-obstacle-v1-0000 \
  REMOTE_REPO="$remote_repo" \
  sbatch --parsable --dependency="afterok:$job_id" \
    --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' \
    slurm/generated_source_validate_cpu.sbatch
)
validation_job_id=${validation_submission%%;*}
case "$validation_job_id" in
  *[!0-9]*|'')
    echo "sbatch did not return an exact numeric validation job id: $validation_submission" >&2
    exit 2
    ;;
esac
echo "submitted_independent_validation_job_id=$validation_job_id"
echo "validation_dependency=afterok:$job_id"
echo "expected_cpu_validation=${artifact%.json}.cpu-validation.json"
REMOTE
