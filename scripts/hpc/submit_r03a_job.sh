#!/usr/bin/env bash
set -euo pipefail

MODE=${R03A_SUBMISSION_MODE:?use submit_r03a_smoke.sh, submit_r03a_h100_smoke.sh, submit_r03a_grouped_array.sh, or submit_r03a_array.sh}
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
  grouped_array)
    test "$CONCURRENCY" -eq 2 || {
      echo "R03A source-node grouped full run has fixed total concurrency two" >&2
      exit 2
    }
    partition=main
    cpus_per_task=8
    memory_per_task_mb=$((128 * 1024))
    required_case_count=17
    slurm_file=slurm/r03a_main_array.sbatch
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
  docs/decisions/0027-register-source-node-grouped-r03a-population.md
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
  scripts/hpc/submit_r03a_grouped_array.sh
  scripts/hpc/submit_r03a_h100_smoke.sh
  scripts/hpc/submit_r03a_job.sh
  scripts/hpc/submit_r03a_smoke.sh
  scripts/hpc/submit_r03a_summary.sh
  slurm/r03a_h100_smoke.sbatch
  slurm/r03a_main_array.sbatch
  slurm/r03a_mig.sbatch
  slurm/r03a_summary.sbatch
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
  "$remote_repo/main/summarize_r03a.py" \
  "$remote_repo/main/crfs_oracle/r03a_runner.py" \
  "$remote_repo/main/crfs_oracle/r03a_validation.py" \
  "$remote_repo/scripts/hpc/prepare_jsonschema_overlay.sh" \
  "$remote_repo/scripts/hpc/run_r03a_case.sh" \
  "$remote_repo/$slurm_file" \
  "$remote_repo/slurm/r03a_summary.sbatch"; do
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

case_ids=$(jq -r '.case_id // empty' "$manifest")
case_id_count=$(printf '%s\n' "$case_ids" | awk 'NF {count++} END {print count+0}')
unique_count=$(printf '%s\n' "$case_ids" | awk 'NF' | sort -u | awk 'END {print NR+0}')
test "$case_id_count" -eq 17 && test "$unique_count" -eq 17 || {
  echo "remote R03A manifest identities are incomplete or duplicated" >&2; exit 2;
}

# Pass 1 is deliberately complete before any provenance host is read.  Host
# grouping is meaningful only after every source byte string in the requested
# population has been bound to the accepted R03 result set.
hash_checked=0
while IFS= read -r case_id; do
  test -n "$case_id" || continue
  source_result=$r02_raw_root/$case_id/r02-paired.json
  test -f "$source_result" || {
    echo "missing immutable R02 source for $case_id" >&2; exit 2;
  }
  expected_source_sha=$(jq -r --arg case_id "$case_id" '
    [.result_hashes[] | select(.case_id == $case_id) | .sha256] as $matches |
    if ($matches | length) == 1 then $matches[0] else empty end
  ' "$r03_summary")
  test -n "$expected_source_sha" || {
    echo "R03 summary has no unique hash for $case_id" >&2; exit 2;
  }
  test "$(sha256sum "$source_result" | awk '{print $1}')" = "$expected_source_sha" || {
    echo "R02 source hash differs before source-node pinning for $case_id" >&2
    exit 2
  }
  hash_checked=$((hash_checked + 1))
  test "$hash_checked" -lt "$required_case_count" || break
done <<EOF
$case_ids
EOF
test "$hash_checked" -eq "$required_case_count" || {
  echo "R03A source hash audit did not reach the required population" >&2; exit 2;
}

# Pass 2 may now read the already hash-bound provenance and derive the exact
# source-node grouping.  Do not combine these loops: ADR-0027 requires the
# complete hash audit to finish before the first host is observed.
host_checked=0
index=0
required_source_node=
required_source_nodes=()
worker_1_indices=
worker_2_indices=
while IFS= read -r case_id; do
  test -n "$case_id" || continue
  source_result=$r02_raw_root/$case_id/r02-paired.json
  source_host=$(jq -r '.provenance.host // empty' "$source_result")
  case "$source_host" in
    *[!A-Za-z0-9._-]*|'')
      echo "R02 source host is missing or unsafe for $case_id" >&2
      exit 2
      ;;
  esac
  if [ "$mode" = h100_smoke ]; then
    required_source_node=$source_host
    required_source_nodes=("$source_host")
  elif [ "$mode" = grouped_array ]; then
    case "$source_host" in
      worker-1)
        worker_1_indices=${worker_1_indices:+$worker_1_indices,}$index
        ;;
      worker-2)
        worker_2_indices=${worker_2_indices:+$worker_2_indices,}$index
        ;;
      *)
        echo "R03A source host $source_host is outside the registered grouping" >&2
        exit 2
        ;;
    esac
  fi
  host_checked=$((host_checked + 1))
  index=$((index + 1))
  test "$host_checked" -lt "$required_case_count" || break
done <<EOF
$case_ids
EOF
test "$host_checked" -eq "$required_case_count" || {
  echo "R03A source-host audit did not reach the required population" >&2; exit 2;
}

if [ "$mode" = h100_smoke ]; then
  echo "required_source_node=$required_source_node"
elif [ "$mode" = grouped_array ]; then
  expected_worker_1_indices=0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16
  expected_worker_2_indices=5
  test "$worker_1_indices" = "$expected_worker_1_indices" || {
    echo "worker-1 source index group differs from ADR-0024" >&2; exit 2;
  }
  test "$worker_2_indices" = "$expected_worker_2_indices" || {
    echo "worker-2 source index group differs from ADR-0024" >&2; exit 2;
  }
  required_source_nodes=(worker-1 worker-2)
  echo "worker_1_source_indices=$worker_1_indices"
  echo "worker_2_source_indices=$worker_2_indices"
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
  if [ "${#required_source_nodes[@]}" -gt 0 ]; then
    source_node_required=false
    for required_node in "${required_source_nodes[@]}"; do
      if [ "$node" = "$required_node" ]; then
        source_node_required=true
        break
      fi
    done
    $source_node_required || continue
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
for required_node in "${required_source_nodes[@]}"; do
  required_node_eligible=false
  for eligible_node in "${eligible_nodes[@]}"; do
    if [ "$eligible_node" = "$required_node" ]; then
      required_node_eligible=true
      break
    fi
  done
  $required_node_eligible || {
    echo "required source node $required_node is unhealthy or lacks requested live free memory" >&2
    exit 2
  }
done
test "${#eligible_nodes[@]}" -gt 0 || {
  echo "no healthy $partition node has the requested live free memory" >&2
  exit 2
}
eligible_csv=$(IFS=,; echo "${eligible_nodes[*]}")
sbatch_args=(--nodelist="$eligible_csv")
if [ "${#excluded_nodes[@]}" -gt 0 ]; then
  excluded_csv=$(IFS=,; echo "${excluded_nodes[*]}")
  sbatch_args+=(--exclude="$excluded_csv")
fi
mkdir -p /mnt/data/quanth/slurm_logs/crfs-oracle
cd "$remote_repo"
if [ "$mode" = grouped_array ]; then
  run_root=$output_root/$run_id
  mkdir "$run_root" || {
    echo "failed to reserve immutable grouped R03A run id: $run_id" >&2
    exit 2
  }
  reservation_tmp=$(mktemp "$run_root/.launch-reservation.XXXXXX")
  jq -n \
    --arg run_id "$run_id" \
    --arg git_commit "$expected_commit" \
    --arg manifest_sha256 "$manifest_sha256" \
    --arg config_sha256 "$config_sha256" \
    --arg schema_sha256 "$schema_sha256" \
    --arg decision_sha256 "$decision_sha256" \
    --arg r03_summary_sha256 "$r03_summary_sha256" \
    --arg checkpoint_sha256 "$checkpoint_sha256" \
    --arg timestamp_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{
      schema_version: "1.0",
      artifact_role: "r03a_source_node_grouped_launch_reservation",
      status: "reserved",
      scientific_claim_allowed: false,
      run_id: $run_id,
      git_commit: $git_commit,
      manifest_sha256: $manifest_sha256,
      config_sha256: $config_sha256,
      schema_sha256: $schema_sha256,
      decision_sha256: $decision_sha256,
      r03_summary_sha256: $r03_summary_sha256,
      checkpoint_sha256: $checkpoint_sha256,
      timestamp_utc: $timestamp_utc,
      groups: {
        "worker-1": {indices: "0-4,6-16", concurrency: 1},
        "worker-2": {indices: "5", concurrency: 1}
      }
    }' >"$reservation_tmp"
  mv "$reservation_tmp" "$run_root/launch-reservation.json"

  submit_group() {
    group_node=$1
    group_indices=$2
    group_job_name=$3
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
      EXPECTED_SOURCE_NODE="$group_node" \
      REMOTE_REPO="$remote_repo" \
      sbatch --parsable --hold \
        --nodelist="$group_node" \
        --array="$group_indices%1" \
        --job-name="$group_job_name" \
        --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' \
        "$slurm_file"
    )
    submitted_job_id=${submission%%;*}
    case "$submitted_job_id" in
      *[!0-9]*|'')
        echo "sbatch returned an invalid grouped R03A job id: $submission" >&2
        exit 2
        ;;
    esac
    receipt_tmp=$(mktemp "$run_root/.${group_node}-submission.XXXXXX")
    jq -n \
      --arg run_id "$run_id" \
      --arg git_commit "$expected_commit" \
      --arg node "$group_node" \
      --arg indices "$group_indices" \
      --arg job_id "$submitted_job_id" \
      --arg timestamp_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      '{
        schema_version: "1.0",
        artifact_role: "r03a_source_node_group_submission",
        status: "held",
        scientific_claim_allowed: false,
        run_id: $run_id,
        git_commit: $git_commit,
        source_node: $node,
        array_indices: $indices,
        concurrency: 1,
        slurm_array_job_id: $job_id,
        timestamp_utc: $timestamp_utc
      }' >"$receipt_tmp"
    mv "$receipt_tmp" "$run_root/$group_node-submission.json"
    echo "held_${group_node}_job_id=$submitted_job_id"
  }

  submit_group worker-1 0-4,6-16 crfs-r03a-worker-1
  worker_1_job_id=$submitted_job_id
  submit_group worker-2 5 crfs-r03a-worker-2
  worker_2_job_id=$submitted_job_id
  test "$worker_1_job_id" != "$worker_2_job_id" || {
    echo "grouped R03A submissions returned the same Slurm job id" >&2
    exit 2
  }

  validate_held_group() {
    held_job_id=$1
    held_name=$2
    held_node=$3
    held_indices=$4
    held_record=$(scontrol show job "$held_job_id" -o)
    case " $held_record " in
      *" JobName=$held_name "*) ;;
      *) echo "held grouped job $held_job_id has the wrong name" >&2; exit 2 ;;
    esac
    case " $held_record " in
      *" ReqNodeList=$held_node "*) ;;
      *) echo "held grouped job $held_job_id has the wrong source node" >&2; exit 2 ;;
    esac
    case " $held_record " in
      *" ArrayTaskId=$held_indices%1 "*|*" ArrayTaskId=$held_indices "*) ;;
      *) echo "held grouped job $held_job_id has the wrong indices" >&2; exit 2 ;;
    esac
    case " $held_record " in
      *" ArrayTaskThrottle=1 "*) ;;
      *) echo "held grouped job $held_job_id does not prove the %1 throttle" >&2; exit 2 ;;
    esac
    case " $held_record " in
      *" JobState=PENDING "*) ;;
      *) echo "grouped job $held_job_id is not pending" >&2; exit 2 ;;
    esac
    case " $held_record " in
      *" Reason=JobHeldUser "*) ;;
      *) echo "grouped job $held_job_id is not on an explicit user hold" >&2; exit 2 ;;
    esac
    case " $held_record " in
      *" Priority=0 "*) ;;
      *) echo "grouped job $held_job_id does not have held priority zero" >&2; exit 2 ;;
    esac
  }
  validate_held_group "$worker_1_job_id" crfs-r03a-worker-1 worker-1 0-4,6-16
  validate_held_group "$worker_2_job_id" crfs-r03a-worker-2 worker-2 5

  launch_tmp=$(mktemp "$run_root/.grouped-launch.XXXXXX")
  jq -n \
    --arg run_id "$run_id" \
    --arg git_commit "$expected_commit" \
    --arg manifest_sha256 "$manifest_sha256" \
    --arg config_sha256 "$config_sha256" \
    --arg schema_sha256 "$schema_sha256" \
    --arg decision_sha256 "$decision_sha256" \
    --arg r03_summary_sha256 "$r03_summary_sha256" \
    --arg checkpoint_sha256 "$checkpoint_sha256" \
    --arg reservation_sha256 "$(sha256sum "$run_root/launch-reservation.json" | awk '{print $1}')" \
    --arg worker_1_job_id "$worker_1_job_id" \
    --arg worker_2_job_id "$worker_2_job_id" \
    --arg timestamp_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{
      schema_version: "1.0",
      artifact_role: "r03a_source_node_grouped_launch",
      status: "held_validated",
      scientific_claim_allowed: false,
      run_id: $run_id,
      git_commit: $git_commit,
      manifest_sha256: $manifest_sha256,
      config_sha256: $config_sha256,
      schema_sha256: $schema_sha256,
      decision_sha256: $decision_sha256,
      r03_summary_sha256: $r03_summary_sha256,
      checkpoint_sha256: $checkpoint_sha256,
      launch_reservation_sha256: $reservation_sha256,
      timestamp_utc: $timestamp_utc,
      groups: {
        "worker-1": {
          indices: "0-4,6-16", concurrency: 1,
          slurm_array_job_id: $worker_1_job_id
        },
        "worker-2": {
          indices: "5", concurrency: 1,
          slurm_array_job_id: $worker_2_job_id
        }
      }
    }' >"$launch_tmp"
  mv "$launch_tmp" "$run_root/grouped-launch.json"

  # Register the fixed-denominator verifier while both GPU arrays are still
  # held.  If this submission, its validation, or its receipt fails, set -e
  # leaves both source arrays held and the immutable run explicitly partial.
  summary_dependency=afterany:$worker_1_job_id:$worker_2_job_id
  summary_output=$run_root/r03a-summary.json
  summary_submission=$(
    EXPECTED_GIT_COMMIT="$expected_commit" \
    RUN_ID="$run_id" \
    MANIFEST="$manifest" \
    MANIFEST_SHA256="$manifest_sha256" \
    CONFIG="$config" \
    CONFIG_SHA256="$config_sha256" \
    R02_RAW_ROOT="$r02_raw_root" \
    R03_SUMMARY="$r03_summary" \
    R03_SUMMARY_SHA256="$r03_summary_sha256" \
    RESULTS_ROOT="$run_root" \
    OUTPUT="$summary_output" \
    SOURCE_WORKER_1_SLURM_ARRAY_JOB_ID="$worker_1_job_id" \
    SOURCE_WORKER_2_SLURM_ARRAY_JOB_ID="$worker_2_job_id" \
    CHECKPOINT_ID="$checkpoint_id" \
    CHECKPOINT_SHA256="$checkpoint_sha256" \
    REMOTE_REPO="$remote_repo" \
    sbatch --parsable \
      --dependency="$summary_dependency" \
      --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' \
      slurm/r03a_summary.sbatch
  )
  summary_job_id=${summary_submission%%;*}
  case "$summary_job_id" in
    *[!0-9]*|'')
      echo "sbatch returned an invalid R03A summary job id: $summary_submission" >&2
      exit 2
      ;;
  esac
  test "$summary_job_id" != "$worker_1_job_id" && \
    test "$summary_job_id" != "$worker_2_job_id" || {
    echo "R03A summary and source arrays must have distinct Slurm job ids" >&2
    exit 2
  }
  summary_record=$(scontrol show job "$summary_job_id" -o)
  case " $summary_record " in
    *" JobName=crfs-r03a-summary "*) ;;
    *) echo "registered R03A summary job has the wrong name" >&2; exit 2 ;;
  esac
  case " $summary_record " in
    *" JobState=PENDING "*) ;;
    *) echo "registered R03A summary is not pending on its source arrays" >&2; exit 2 ;;
  esac
  case " $summary_record " in
    *" Partition=main "*) ;;
    *) echo "registered R03A summary is not on the main partition" >&2; exit 2 ;;
  esac
  case " $summary_record " in
    *" ReqTRES=cpu=2,mem=16G,"*) ;;
    *) echo "registered R03A summary has the wrong CPU/memory request" >&2; exit 2 ;;
  esac
  case " $summary_record " in
    *"gres/gpu"*)
      echo "registered R03A summary unexpectedly requests a GPU" >&2
      exit 2
      ;;
  esac
  observed_summary_dependency=$(printf '%s\n' "$summary_record" | \
    sed -n 's/.* Dependency=\([^ ]*\).*/\1/p')
  normalized_summary_dependency=$(printf '%s' "$observed_summary_dependency" | \
    sed -e 's/(unfulfilled)//g' -e 's/_\*//g' -e 's/,afterany:/:/g')
  test "$normalized_summary_dependency" = "$summary_dependency" || {
    echo "registered R03A summary has the wrong Slurm dependency" >&2
    exit 2
  }
  summary_receipt_tmp=$(mktemp "$run_root/.summary-submission.XXXXXX")
  jq -n \
    --arg run_id "$run_id" \
    --arg git_commit "$expected_commit" \
    --arg manifest_sha256 "$manifest_sha256" \
    --arg config_sha256 "$config_sha256" \
    --arg schema_sha256 "$schema_sha256" \
    --arg decision_sha256 "$decision_sha256" \
    --arg r03_summary_sha256 "$r03_summary_sha256" \
    --arg checkpoint_sha256 "$checkpoint_sha256" \
    --arg grouped_launch_sha256 "$(sha256sum "$run_root/grouped-launch.json" | awk '{print $1}')" \
    --arg worker_1_job_id "$worker_1_job_id" \
    --arg worker_2_job_id "$worker_2_job_id" \
    --arg summary_job_id "$summary_job_id" \
    --arg dependency "$summary_dependency" \
    --arg timestamp_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{
      schema_version: "1.0",
      artifact_role: "r03a_population_summary_submission",
      status: "dependency_registered",
      scientific_claim_allowed: false,
      run_id: $run_id,
      git_commit: $git_commit,
      manifest_sha256: $manifest_sha256,
      config_sha256: $config_sha256,
      schema_sha256: $schema_sha256,
      decision_sha256: $decision_sha256,
      r03_summary_sha256: $r03_summary_sha256,
      checkpoint_sha256: $checkpoint_sha256,
      grouped_launch_sha256: $grouped_launch_sha256,
      source_slurm_array_job_ids_by_host: {
        "worker-1": $worker_1_job_id,
        "worker-2": $worker_2_job_id
      },
      slurm_summary_job_id: $summary_job_id,
      dependency: $dependency,
      timestamp_utc: $timestamp_utc
    }' >"$summary_receipt_tmp"
  mv "$summary_receipt_tmp" "$run_root/summary-submission.json"

  # The dependency and its immutable receipt now exist; only now may the
  # exact held source arrays become runnable.
  scontrol release "$worker_1_job_id" "$worker_2_job_id"
  echo "submitted_worker_1_job_id=$worker_1_job_id"
  echo "submitted_worker_2_job_id=$worker_2_job_id"
  echo "submitted_summary_job_id=$summary_job_id"
  echo "summary_dependency=$summary_dependency"
  echo "summary_submission_sha256=$(sha256sum "$run_root/summary-submission.json" | awk '{print $1}')"
  echo "grouped_launch_sha256=$(sha256sum "$run_root/grouped-launch.json" | awk '{print $1}')"
else
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
    EXPECTED_SOURCE_NODE="$required_source_node" \
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
fi
echo "submission_mode=$mode"
echo "expected_commit=$expected_commit"
REMOTE
