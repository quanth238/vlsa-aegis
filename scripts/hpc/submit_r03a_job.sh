#!/usr/bin/env bash
set -euo pipefail

MODE=${R03A_SUBMISSION_MODE:?use submit_r03a_smoke.sh, submit_r03a_h100_smoke.sh, or submit_r03a_array.sh}
usage="usage: RUN_ID=... submit_r03a_${MODE}.sh MANIFEST_JSONL CONFIG_JSON R02_RAW_ROOT_REMOTE R03_SUMMARY_REMOTE R03_SUMMARY_SHA256 [CONCURRENCY]"
MANIFEST_LOCAL=${1:?$usage}
CONFIG_LOCAL=${2:?$usage}
R02_RAW_ROOT=${3:?$usage}
R03_SUMMARY=${4:?$usage}
R03_SUMMARY_SHA256=${5:?$usage}
CONCURRENCY=${6:-1}
: "${RUN_ID:?set RUN_ID to an immutable exact R03A identifier}"

EXPECTED_MANIFEST_SHA256=241f1b94a5434973dcfb235b43b9386ad15046ff87d35d5b0c34e295c2132916
EXPECTED_CONFIG_SHA256=445c7f227b780f87f5393f893a0ebba96e77a83d1f3690e6eee7559effceaa8a
EXPECTED_SCHEMA_SHA256=2f0f9db28ae9d62582e88bc70afe40ef4e37245a60190863e2f3419a135d70c3
EXPECTED_R03_SUMMARY_SHA256=dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e
EXPECTED_DECISION_SHA256=b1b717856ebee8d3e93a21b35606e62ad6cc4baa2b9346f7bd1f415c84b25ead
CHECKPOINT_ID=${CHECKPOINT_ID:-/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:-988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/crfs-oracle}

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
    exit 2
    ;;
esac
case "$MODE" in
  smoke)
    test "$CONCURRENCY" -eq 1 || {
      echo "R03A smoke concurrency is fixed at one" >&2
      exit 2
    }
    partition=mig
    cpus_per_task=6
    memory_per_task_mb=$((80 * 1024))
    required_case_count=1
    slurm_file=slurm/r03a_mig.sbatch
    ;;
  h100_smoke)
    test "$CONCURRENCY" -eq 1 || {
      echo "R03A H100 smoke concurrency is fixed at one" >&2
      exit 2
    }
    partition=main
    cpus_per_task=8
    memory_per_task_mb=$((128 * 1024))
    required_case_count=1
    slurm_file=slurm/r03a_h100_smoke.sbatch
    ;;
  array)
    case "$CONCURRENCY" in
      1|2) ;;
      *) echo "R03A full-H100 concurrency must be 1 or 2" >&2; exit 2 ;;
    esac
    partition=main
    cpus_per_task=8
    memory_per_task_mb=$((128 * 1024))
    required_case_count=17
    slurm_file=slurm/r03a_main_array.sbatch
    ;;
  *)
    echo "unregistered R03A submission mode: $MODE" >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke)
    echo "ADR-0024 retires the cross-node MIG smoke; use the source-node H100 smoke" >&2
    exit 2
    ;;
  array)
    echo "ADR-0024 blocks the ungrouped full array until a hash-derived source-node launcher is registered" >&2
    exit 2
    ;;
esac

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
MANIFEST=$REMOTE_REPO/manifests/r03a_analytic_kill_test_eligible.jsonl
CONFIG=$REMOTE_REPO/configs/experiments/r03a_analytic_kill_test.json
SCHEMA=$REMOTE_REPO/schemas/r03a-analytic-kill-test.schema.json
DECISION=$REMOTE_REPO/docs/decisions/0023-run-strong-analytic-kill-test.md

test "$(basename "$MANIFEST_LOCAL")" = r03a_analytic_kill_test_eligible.jsonl || {
  echo "R03A requires the frozen eligible manifest filename" >&2
  exit 2
}
test "$(basename "$CONFIG_LOCAL")" = r03a_analytic_kill_test.json || {
  echo "R03A requires the frozen analytic kill-test config filename" >&2
  exit 2
}
test -f "$MANIFEST_LOCAL" || { echo "missing R03A manifest" >&2; exit 2; }
test -f "$CONFIG_LOCAL" || { echo "missing R03A config" >&2; exit 2; }
test "$R03_SUMMARY_SHA256" = "$EXPECTED_R03_SUMMARY_SHA256" || {
  echo "R03A must bind the accepted R03 summary hash" >&2
  exit 2
}
test "$(shasum -a 256 "$MANIFEST_LOCAL" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA256" || {
  echo "local R03A eligible manifest hash mismatch" >&2
  exit 2
}
test "$(shasum -a 256 "$CONFIG_LOCAL" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256" || {
  echo "local R03A config hash mismatch" >&2
  exit 2
}
test "$(shasum -a 256 schemas/r03a-analytic-kill-test.schema.json | awk '{print $1}')" = "$EXPECTED_SCHEMA_SHA256" || {
  echo "local R03A artifact schema hash mismatch" >&2
  exit 2
}
test "$(shasum -a 256 docs/decisions/0023-run-strong-analytic-kill-test.md | awk '{print $1}')" = "$EXPECTED_DECISION_SHA256" || {
  echo "local ADR-0023 hash mismatch" >&2
  exit 2
}
COUNT=$(awk 'NF {count++} END {print count+0}' "$MANIFEST_LOCAL")
test "$COUNT" -eq 17 || {
  echo "R03A requires the complete frozen 17-case population" >&2
  exit 2
}
jq -e \
  --arg manifest_sha "$EXPECTED_MANIFEST_SHA256" \
  --arg schema_sha "$EXPECTED_SCHEMA_SHA256" \
  --arg r03_sha "$EXPECTED_R03_SUMMARY_SHA256" \
  --arg decision_sha "$EXPECTED_DECISION_SHA256" '
    .ready_to_run == true and .blocked_on == [] and
    .manifest == "manifests/r03a_analytic_kill_test_eligible.jsonl" and
    .r03a.eligible_case_count == 17 and
    .r03a.eligible_manifest_sha256 == $manifest_sha and
    .r03a.artifact_schema_sha256 == $schema_sha and
    .r03a.source_r03_summary_sha256 == $r03_sha and
    .r03a.decision_sha256 == $decision_sha and
    .r03a.required_arms == ["analytic_trajectory_mid", "analytic_trajectory_early"] and
    .r03a.probe_training_authorized == false and
    .r03a.confirmatory_r04_unblocked_by_r03a_alone == false
  ' "$CONFIG_LOCAL" >/dev/null || {
  echo "local R03A config violates the frozen kill-test contract" >&2
  exit 2
}

SOURCE_FILES=(
  configs/experiments/r03a_analytic_kill_test.json
  docs/decisions/0023-run-strong-analytic-kill-test.md
  docs/decisions/0024-preserve-source-node-trace-pairing.md
  docs/decisions/0025-bind-r03a-trace-gate-to-native-leaf-evidence.md
  docs/decisions/0026-validate-r03a-scalars-in-recorded-dtype.md
  main/crfs_oracle/r03a_runner.py
  main/crfs_oracle/r03a_validation.py
  main/run_crfs_r03a.py
  main/summarize_r03a.py
  manifests/r03a_analytic_kill_test_eligible.jsonl
  openpi/src/openpi/models_pytorch/crfs_analytic.py
  openpi/src/openpi/models_pytorch/pi0_pytorch.py
  openpi/src/openpi/policies/policy.py
  schemas/r03a-analytic-kill-test.schema.json
  scripts/hpc/prepare_jsonschema_overlay.sh
  scripts/hpc/run_r03a_case.sh
  scripts/hpc/submit_r03a_array.sh
  scripts/hpc/submit_r03a_h100_smoke.sh
  scripts/hpc/submit_r03a_job.sh
  scripts/hpc/submit_r03a_smoke.sh
  slurm/r03a_h100_smoke.sbatch
  slurm/r03a_main_array.sbatch
  slurm/r03a_mig.sbatch
)
for source_file in "${SOURCE_FILES[@]}"; do
  git ls-files --error-unmatch "$source_file" >/dev/null 2>&1 || {
    echo "R03A source must be committed before submission: $source_file" >&2
    exit 2
  }
done
LOCAL_CHANGES=$(git status --porcelain --untracked-files=all | awk '$0 !~ /^\?\? tmp\// {print}')
test -z "$LOCAL_CHANGES" || {
  echo "R03A submission requires a clean reviewed tree (only user-owned ?? tmp/ is ignored)" >&2
  exit 2
}
EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)

# This is the last control-plane action before the remote live audit/submission.
scripts/hpc/preflight.sh

ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$EXPECTED_GIT_COMMIT" "$MODE" "$partition" \
  "$cpus_per_task" "$memory_per_task_mb" "$CONCURRENCY" "$required_case_count" \
  "$RUN_ID" "$MANIFEST" "$EXPECTED_MANIFEST_SHA256" "$CONFIG" \
  "$EXPECTED_CONFIG_SHA256" "$SCHEMA" "$EXPECTED_SCHEMA_SHA256" \
  "$DECISION" "$EXPECTED_DECISION_SHA256" "$R02_RAW_ROOT" \
  "$R03_SUMMARY" "$R03_SUMMARY_SHA256" "$CHECKPOINT_ID" \
  "$CHECKPOINT_SHA256" "$OUTPUT_ROOT" "$slurm_file" <<'REMOTE'
set -euo pipefail
remote_repo=$1
expected_commit=$2
mode=$3
partition=$4
cpus_per_task=$5
memory_per_task_mb=$6
concurrency=$7
required_case_count=$8
run_id=$9
manifest=${10}
manifest_sha256=${11}
config=${12}
config_sha256=${13}
schema=${14}
schema_sha256=${15}
decision=${16}
decision_sha256=${17}
r02_raw_root=${18}
r03_summary=${19}
r03_summary_sha256=${20}
checkpoint_id=${21}
checkpoint_sha256=${22}
output_root=${23}
slurm_file=${24}

for path in \
  "$manifest" "$config" "$schema" "$decision" "$r03_summary" \
  "$remote_repo/main/run_crfs_r03a.py" \
  "$remote_repo/main/crfs_oracle/r03a_runner.py" \
  "$remote_repo/main/crfs_oracle/r03a_validation.py" \
  "$remote_repo/scripts/hpc/run_r03a_case.sh" \
  "$remote_repo/$slurm_file"; do
  test -f "$path" || { echo "missing remote R03A source/input: $path" >&2; exit 2; }
done
test -x "$remote_repo/scripts/hpc/run_r03a_case.sh" || {
  echo "remote R03A allocation worker is not executable" >&2
  exit 2
}
test -d "$r02_raw_root" || { echo "missing remote raw R02 root" >&2; exit 2; }
test -f "$checkpoint_id/model.safetensors" || { echo "missing remote checkpoint" >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$manifest_sha256" || {
  echo "remote R03A manifest hash mismatch" >&2; exit 2;
}
test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha256" || {
  echo "remote R03A config hash mismatch" >&2; exit 2;
}
test "$(sha256sum "$schema" | awk '{print $1}')" = "$schema_sha256" || {
  echo "remote R03A schema hash mismatch" >&2; exit 2;
}
test "$(sha256sum "$decision" | awk '{print $1}')" = "$decision_sha256" || {
  echo "remote ADR-0023 hash mismatch" >&2; exit 2;
}
test "$(sha256sum "$r03_summary" | awk '{print $1}')" = "$r03_summary_sha256" || {
  echo "remote R03 source summary hash mismatch" >&2; exit 2;
}
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || {
  echo "remote checkout differs from the reviewed local R03A commit" >&2; exit 2;
}
test -z "$(git -C "$remote_repo" status --porcelain)" || {
  echo "remote R03A checkout is dirty" >&2; exit 2;
}
test ! -e "$output_root/$run_id" || {
  echo "immutable R03A RUN_ID already exists: $run_id" >&2; exit 2;
}

case_ids=$(sed -n 's/.*"case_id":"\([A-Za-z0-9._-]*\)".*/\1/p' "$manifest")
case_id_count=$(printf '%s\n' "$case_ids" | awk 'NF {count++} END {print count+0}')
unique_count=$(printf '%s\n' "$case_ids" | awk 'NF' | sort -u | awk 'END {print NR+0}')
test "$case_id_count" -eq 17 && test "$unique_count" -eq 17 || {
  echo "remote R03A manifest identities are incomplete or duplicated" >&2; exit 2;
}
checked=0
while IFS= read -r case_id; do
  test -n "$case_id" || continue
  test -f "$r02_raw_root/$case_id/r02-paired.json" || {
    echo "missing immutable R02 source for $case_id" >&2; exit 2;
  }
  checked=$((checked + 1))
  test "$checked" -lt "$required_case_count" || break
done <<EOF
$case_ids
EOF
test "$checked" -eq "$required_case_count" || {
  echo "R03A source-case audit did not reach the required population" >&2; exit 2;
}

required_source_node=
if [ "$mode" = h100_smoke ]; then
  selected_case_id=$(printf '%s\n' "$case_ids" | awk 'NF {print; exit}')
  selected_r02_result=$r02_raw_root/$selected_case_id/r02-paired.json
  expected_selected_sha=$(jq -r --arg case_id "$selected_case_id" '
    [.result_hashes[] | select(.case_id == $case_id) | .sha256] as $matches |
    if ($matches | length) == 1 then $matches[0] else empty end
  ' "$r03_summary")
  test -n "$expected_selected_sha" || {
    echo "R03 summary has no unique hash for H100 smoke case" >&2; exit 2;
  }
  test "$(sha256sum "$selected_r02_result" | awk '{print $1}')" = "$expected_selected_sha" || {
    echo "selected R02 source hash differs before source-node pinning" >&2; exit 2;
  }
  required_source_node=$(jq -r '.provenance.host // empty' "$selected_r02_result")
  case "$required_source_node" in
    *[!A-Za-z0-9._-]*|'')
      echo "selected R02 source host is missing or unsafe" >&2
      exit 2
      ;;
  esac
  test "$(sinfo -h -p main -N -n "$required_source_node" -o '%N' | sort -u)" = "$required_source_node" || {
    echo "selected R02 source host is not an exact main-partition node" >&2
    exit 2
  }
  echo "required_source_node=$required_source_node"
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

pending_job_ids=$(squeue -h -r -u "$(whoami)" -t PENDING -o '%i')
allocated_job_ids=$(squeue -h -u "$(whoami)" -t RUNNING,COMPLETING,CONFIGURING,SUSPENDED -o '%i')
allocated_cpus=0
allocated_mem_mb=0
allocated_gpus=0
while IFS= read -r job_id; do
  test -n "$job_id" || continue
  alloc_tres=$(scontrol show job "$job_id" -o | sed -n 's/.* AllocTRES=\([^ ]*\).*/\1/p')
  test -n "$alloc_tres" || { echo "cannot determine AllocTRES for $job_id" >&2; exit 2; }
  job_cpus=$(printf '%s' "$alloc_tres" | tr ',' '\n' | sed -n 's/^cpu=\([0-9][0-9]*\)$/\1/p')
  job_mem=$(printf '%s' "$alloc_tres" | tr ',' '\n' | sed -n 's/^mem=\([^,]*\)$/\1/p')
  job_gpus=$(printf '%s' "$alloc_tres" | tr ',' '\n' | sed -n 's#^gres/gpu[^=]*=\([0-9][0-9]*\)$#\1#p' | awk '{sum += $1} END {print sum+0}')
  test -n "$job_cpus" && test -n "$job_mem" || {
    echo "cannot parse live allocation for $job_id: $alloc_tres" >&2; exit 2;
  }
  job_mem_mb=$(memory_to_mb "$job_mem") || exit 2
  allocated_cpus=$((allocated_cpus + job_cpus))
  allocated_mem_mb=$((allocated_mem_mb + job_mem_mb))
  allocated_gpus=$((allocated_gpus + job_gpus))
done < <(printf '%s\n' "$allocated_job_ids")

pending_cpus=0
pending_mem_mb=0
pending_gpus=0
while IFS= read -r job_id; do
  test -n "$job_id" || continue
  req_tres=$(scontrol show job "$job_id" -o | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
  test -n "$req_tres" && test "$req_tres" != "(null)" || {
    echo "cannot determine ReqTRES for pending job $job_id" >&2; exit 2;
  }
  job_cpus=$(printf '%s' "$req_tres" | tr ',' '\n' | sed -n 's/^cpu=\([0-9][0-9]*\)$/\1/p')
  job_mem=$(printf '%s' "$req_tres" | tr ',' '\n' | sed -n 's/^mem=\([^,]*\)$/\1/p')
  job_gpus=$(printf '%s' "$req_tres" | tr ',' '\n' | sed -n 's#^gres/gpu[^=]*=\([0-9][0-9]*\)$#\1#p' | awk '{sum += $1} END {print sum+0}')
  test -n "$job_cpus" && test -n "$job_mem" || {
    echo "cannot parse pending request for $job_id: $req_tres" >&2; exit 2;
  }
  job_mem_mb=$(memory_to_mb "$job_mem") || exit 2
  pending_cpus=$((pending_cpus + job_cpus))
  pending_mem_mb=$((pending_mem_mb + job_mem_mb))
  pending_gpus=$((pending_gpus + job_gpus))
done < <(printf '%s\n' "$pending_job_ids")

projected_gpus=$((allocated_gpus + pending_gpus + concurrency))
projected_cpus=$((allocated_cpus + pending_cpus + cpus_per_task * concurrency))
projected_mem_mb=$((allocated_mem_mb + pending_mem_mb + memory_per_task_mb * concurrency))
echo "existing_allocated_gpus=$allocated_gpus existing_pending_gpus=$pending_gpus"
echo "existing_allocated_cpus=$allocated_cpus existing_pending_cpus=$pending_cpus"
echo "existing_allocated_mem_mb=$allocated_mem_mb existing_pending_mem_mb=$pending_mem_mb"
echo "projected_gpus=$projected_gpus projected_cpus=$projected_cpus projected_mem_mb=$projected_mem_mb"
test "$projected_gpus" -le 2 || { echo "R03A would exceed the two-GPU ceiling" >&2; exit 2; }
test "$projected_cpus" -le 16 || { echo "R03A would exceed the 16-CPU ceiling" >&2; exit 2; }
test "$projected_mem_mb" -le $((256 * 1024)) || { echo "R03A would exceed the 256-GiB ceiling" >&2; exit 2; }

eligible_nodes=()
excluded_nodes=()
node_states=$(sinfo -h -p "$partition" -N -o '%N|%T' | sort -u)
while IFS='|' read -r node state; do
  test -n "$node" || continue
  if [ -n "$required_source_node" ] && [ "$node" != "$required_source_node" ]; then
    continue
  fi
  normalized=$(printf '%s' "$state" | tr '[:upper:]' '[:lower:]')
  case "$normalized" in
    *down*|*drain*|*drng*|*fail*|*maint*|*not_resp*|*notresponding*|*no_resp*|*power*|*unknown*|*\**)
      excluded_nodes+=("$node")
      continue
      ;;
  esac
  free_mem=$(scontrol show node "$node" -o | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
  test -n "$free_mem" || { echo "cannot determine FreeMem for $node" >&2; exit 2; }
  if [ "$free_mem" -ge "$memory_per_task_mb" ]; then
    eligible_nodes+=("$node")
  fi
done < <(printf '%s\n' "$node_states")
test "${#eligible_nodes[@]}" -gt 0 || {
  if [ -n "$required_source_node" ]; then
    echo "required source node $required_source_node is unhealthy or lacks requested live free memory" >&2
  else
    echo "no healthy $partition node has the requested live free memory" >&2
  fi
  exit 2
}
eligible_csv=$(IFS=,; echo "${eligible_nodes[*]}")
sbatch_args=(--nodelist="$eligible_csv")
if [ "${#excluded_nodes[@]}" -gt 0 ]; then
  excluded_csv=$(IFS=,; echo "${excluded_nodes[*]}")
  sbatch_args+=(--exclude="$excluded_csv")
fi
if [ "$mode" = array ]; then
  sbatch_args+=(--array="0-16%$concurrency")
fi

mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle
cd "$remote_repo"
submission=$(
  RUN_ID="$run_id" \
  MANIFEST="$manifest" \
  CONFIG="$config" \
  EXPERIMENT_CONFIG="$config" \
  R02_RAW_ROOT="$r02_raw_root" \
  R03_SUMMARY="$r03_summary" \
  R03_SUMMARY_SHA256="$r03_summary_sha256" \
  CHECKPOINT_ID="$checkpoint_id" \
  CHECKPOINT_DIR="$checkpoint_id" \
  CHECKPOINT_SHA256="$checkpoint_sha256" \
  OUTPUT_ROOT="$output_root" \
  EXPERIMENT_ROOT="$output_root" \
  EXPECTED_GIT_COMMIT="$expected_commit" \
  REMOTE_REPO="$remote_repo" \
  sbatch --parsable "${sbatch_args[@]}" \
    --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' \
    "$slurm_file"
)
job_id=${submission%%;*}
case "$job_id" in
  *[!0-9]*|'') echo "sbatch returned an invalid R03A job id: $submission" >&2; exit 2 ;;
esac
echo "submitted_exact_job_id=$job_id"
echo "submission_mode=$mode"
echo "expected_commit=$expected_commit"
REMOTE
