#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... submit_r05a_sampled_current_canary.sh'
: "${RUN_ID:?$usage}"
case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe sampled-current RUN_ID" >&2; exit 2 ;; esac
case "$RUN_ID" in [A-Za-z0-9]*) ;; *) echo "sampled-current RUN_ID must start with an alphanumeric character" >&2; exit 2 ;; esac
test "${#RUN_ID}" -le 128 || { echo "sampled-current RUN_ID is too long" >&2; exit 2; }

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
EXPERIMENT_ROOT=/mnt/data/quanth/experiments/crfs-oracle
R02_RAW_ROOT=/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a
CHECKPOINT_DIR=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
MANIFEST_LOCAL=manifests/r05a_inverse_flow_teacher_smoke.jsonl
SCIENTIFIC_CONFIG_LOCAL=configs/experiments/r05a_inverse_flow_canary.json
APPARATUS_CONFIG_LOCAL=configs/experiments/r05a_sampled_current_canary_apparatus.json
ENVELOPE_SCHEMA_LOCAL=schemas/r05a-sampled-current-canary-envelope.schema.json
ADR0036_LOCAL=docs/decisions/0036-preregister-r05a-full-lifetime-sampled-current-canary.md
ADR0037_LOCAL=docs/decisions/0037-require-exact-single-canary-release-identity.md

EXPECTED_MANIFEST_SHA256=bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633
EXPECTED_SCIENTIFIC_CONFIG_SHA256=c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb
EXPECTED_R02_SHA256=055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593
EXPECTED_CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed

BOUND_REPOSITORY_PATHS=(
  configs/experiments/r05a_inverse_flow_canary.json
  configs/experiments/r05a_sampled_current_canary_apparatus.json
  manifests/r05a_inverse_flow_teacher_smoke.jsonl
  docs/decisions/0028-pivot-to-inverse-flow-transport.md
  docs/decisions/0036-preregister-r05a-full-lifetime-sampled-current-canary.md
  docs/decisions/0037-require-exact-single-canary-release-identity.md
  docs/decisions/0038-preserve-sampled-current-launch-a-and-repair-token-parsing.md
  docs/decisions/0039-preserve-sampled-current-launch-b-and-repair-exact-task-accounting.md
  evidence/r05a/ift00a-sampled-current-launch-a.json
  evidence/r05a/ift00a-sampled-current-launch-b.json
  evidence/r05a/exact-array-task-afterany-regression-a.json
  schemas/r05a-inverse-flow-canary.schema.json
  schemas/r05a-sampled-current-canary-envelope.schema.json
  evidence/r03/r03-summary.json
  main/crfs_oracle/r05a_canary.py
  main/crfs_oracle/r05a_allocation_tests.json
  main/crfs_oracle/r05a_full_lifetime_telemetry.py
  main/crfs_oracle/r05a_sampled_current_canary.py
  main/run_crfs_r05a_canary.py
  main/publish_crfs_r05a_sampled_current_canary.py
  openpi/src/openpi/models_pytorch/crfs_inverse_control.py
  openpi/src/openpi/models_pytorch/pi0_pytorch.py
  openpi/src/openpi/policies/policy.py
  scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh
  scripts/hpc/lib/r05a_allocation_tests.sh
  scripts/hpc/lib/slurm_exact_array_task_status.sh
  scripts/hpc/run_r05a_canary.sh
  scripts/hpc/run_r05a_sampled_current_canary.sh
  scripts/hpc/validate_r05a_sampled_current_canary.sh
  scripts/hpc/submit_r05a_sampled_current_canary.sh
  slurm/r05a_sampled_current_canary_h100.sbatch
  slurm/r05a_sampled_current_canary_validate_cpu.sbatch
)

for path in "${BOUND_REPOSITORY_PATHS[@]}"; do
  test -f "$path" && test ! -L "$path" || {
    echo "missing or symlinked sampled-current bound source: $path" >&2
    exit 2
  }
done
test "$(shasum -a 256 "$MANIFEST_LOCAL" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA256" || {
  echo "frozen manifest changed" >&2; exit 2
}
test "$(shasum -a 256 "$SCIENTIFIC_CONFIG_LOCAL" | awk '{print $1}')" = "$EXPECTED_SCIENTIFIC_CONFIG_SHA256" || {
  echo "frozen scientific config changed" >&2; exit 2
}
test "$(shasum -a 256 main/crfs_oracle/r05a_canary.py | awk '{print $1}')" = 9d4402ccb92835bc1af2a3e04fe54c0b94839f41beb2971022216f73ec67fede
test "$(shasum -a 256 main/run_crfs_r05a_canary.py | awk '{print $1}')" = 247bd20e48ffe2228d2633d2f667a2791371163cca1c43ad7d278f4fbfd7439a
test "$(shasum -a 256 openpi/src/openpi/models_pytorch/crfs_inverse_control.py | awk '{print $1}')" = 965082822466774e0a86eeb6f2d178c9e5f3d6d02bca5090c4e1fcc77bc52aa8
test "$(shasum -a 256 openpi/src/openpi/models_pytorch/pi0_pytorch.py | awk '{print $1}')" = 80366dcc7b2ddc598717d4c71c0e68e46312a1e5ffd3fc04479d5599433f4c55
test "$(shasum -a 256 openpi/src/openpi/policies/policy.py | awk '{print $1}')" = d16767ff2073d5c177cdfcc06dc05dcbf7cdb9ef2a2a0150023f7953b03508b9

jq -e '.ready_to_run == true and .blocked_on == []' "$APPARATUS_CONFIG_LOCAL" >/dev/null || {
  echo "sampled-current apparatus is implemented but not released for H100 submission" >&2
  exit 2
}
: "${EXPECTED_RELEASE_COMMIT:?exact reviewed release commit is required}"
case "$EXPECTED_RELEASE_COMMIT" in *[!0-9a-f]*|'') echo "invalid expected release commit" >&2; exit 2 ;; esac
test "${#EXPECTED_RELEASE_COMMIT}" = 40 || { echo "invalid expected release commit length" >&2; exit 2; }
REGISTERED_RUN_ID=$(jq -er '.execution_release.run_id' "$APPARATUS_CONFIG_LOCAL")
ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$APPARATUS_CONFIG_LOCAL")
case "$ACCEPTED_IMPLEMENTATION_COMMIT" in *[!0-9a-f]*|'') echo "invalid accepted implementation commit" >&2; exit 2 ;; esac
test "${#ACCEPTED_IMPLEMENTATION_COMMIT}" = 40 || { echo "invalid accepted implementation commit length" >&2; exit 2; }
test "$RUN_ID" = "$REGISTERED_RUN_ID" || {
  echo "caller RUN_ID is not the exact registered canary identity" >&2
  exit 2
}
jq -e \
  --arg run "$RUN_ID" \
  --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" \
  '.ready_to_run == true
   and .blocked_on == []
   and (.execution_release | type == "object")
   and ((.execution_release | keys | sort) == (["schema_version","artifact_role","decision_artifact","accepted_implementation_commit","run_id","single_submission","source_host","resources","release_only_parent_required","allowed_release_diff_paths","automatic_resubmission_allowed","automatic_next_experiment_allowed"] | sort))
   and .execution_release.schema_version == "1.0"
   and .execution_release.artifact_role == "r05a_single_canary_execution_release"
   and .execution_release.decision_artifact == "docs/decisions/0037-require-exact-single-canary-release-identity.md"
   and .execution_release.accepted_implementation_commit == $implementation
   and .execution_release.run_id == $run
   and .execution_release.single_submission == true
   and .execution_release.source_host == "worker-1"
   and .execution_release.resources == {partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}
   and .execution_release.release_only_parent_required == true
   and .execution_release.allowed_release_diff_paths == ["configs/experiments/r05a_sampled_current_canary_apparatus.json","docs/decisions/0037-require-exact-single-canary-release-identity.md"]
   and .execution_release.automatic_resubmission_allowed == false
   and .execution_release.automatic_next_experiment_allowed == false
   and .resource_contract == {source_host:"worker-1",partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}' \
  "$APPARATUS_CONFIG_LOCAL" >/dev/null || {
    echo "sampled-current exact execution release contract changed" >&2
    exit 2
  }
test "$(jq -r '.envelope_schema_sha256' "$APPARATUS_CONFIG_LOCAL")" = "$(shasum -a 256 "$ENVELOPE_SCHEMA_LOCAL" | awk '{print $1}')" || {
  echo "apparatus config envelope-schema binding changed" >&2; exit 2
}
git ls-files --error-unmatch "${BOUND_REPOSITORY_PATHS[@]}" >/dev/null || {
  echo "all sampled-current sources must be committed before release" >&2; exit 2
}
test -z "$(git status --porcelain)" || {
  echo "sampled-current submission requires a clean reviewed worktree" >&2; exit 2
}
test "$(git rev-parse HEAD)" = "$EXPECTED_RELEASE_COMMIT" || {
  echo "local HEAD differs from the externally authorized release commit" >&2; exit 2
}
test "$(git rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$EXPECTED_RELEASE_COMMIT" || {
  echo "local origin release ref differs from the authorized release commit" >&2; exit 2
}
test "$(git rev-list --parents -n 1 "$EXPECTED_RELEASE_COMMIT")" = "$EXPECTED_RELEASE_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || {
  echo "release commit is not the direct child of the accepted implementation" >&2; exit 2
}
for commit in "$ACCEPTED_IMPLEMENTATION_COMMIT" "$EXPECTED_RELEASE_COMMIT"; do
  test "$(git ls-tree "$commit" -- "$APPARATUS_CONFIG_LOCAL" | awk '{print $1" "$2}')" = "100644 blob" || {
    echo "apparatus config git mode changed in release ancestry" >&2; exit 2
  }
done
PARENT_APPARATUS_CONFIG=$(git show "$ACCEPTED_IMPLEMENTATION_COMMIT:$APPARATUS_CONFIG_LOCAL")
jq -e '.ready_to_run == false and (.blocked_on | type == "array" and length > 0) and (has("execution_release") | not)' <<<"$PARENT_APPARATUS_CONFIG" >/dev/null || {
  echo "accepted implementation parent was not an unreleased apparatus" >&2; exit 2
}
test "$(jq -S -c 'del(.ready_to_run,.blocked_on,.execution_release)' <<<"$PARENT_APPARATUS_CONFIG")" = "$(jq -S -c 'del(.ready_to_run,.blocked_on,.execution_release)' "$APPARATUS_CONFIG_LOCAL")" || {
  echo "release commit changed non-release apparatus content" >&2; exit 2
}
RELEASE_RESOURCE_JSON=$(jq -cS '.execution_release.resources' "$APPARATUS_CONFIG_LOCAL")
EXPECTED_RELEASE_DECISION_FILE=$(mktemp)
cleanup_expected_release_decision() { rm -f "$EXPECTED_RELEASE_DECISION_FILE"; }
trap cleanup_expected_release_decision EXIT INT TERM
git show "$ACCEPTED_IMPLEMENTATION_COMMIT:$ADR0037_LOCAL" >"$EXPECTED_RELEASE_DECISION_FILE"
{
  printf '\n## Exact execution release\n\n'
  printf 'Execution authorization: one preregistered IFT-00A canary submission only.\n\n'
  printf -- '- Accepted implementation commit: `%s`.\n' "$ACCEPTED_IMPLEMENTATION_COMMIT"
  printf -- '- Immutable run ID: `%s`.\n' "$RUN_ID"
  printf -- '- Source host: `worker-1`.\n'
  printf -- '- Resources (canonical JSON): `%s`.\n' "$RELEASE_RESOURCE_JSON"
  printf -- '- Single submission: `true`.\n'
  printf -- '- Automatic resubmission: `false`.\n'
  printf -- '- Automatic next experiment: `false`.\n'
  printf -- '- Simulator efficacy claim authorized: `false`.\n'
  printf -- '- Probe or MLP training authorized: `false`.\n\n'
  printf '%s\n' 'This appendix authorizes only the frozen one-case mechanism canary. It does not authorize IFT-01, solver tuning, a simulator efficacy claim, label collection, probe training, or MLP training.'
} >>"$EXPECTED_RELEASE_DECISION_FILE"
cmp -s "$EXPECTED_RELEASE_DECISION_FILE" "$ADR0037_LOCAL" || {
  echo "release decision is not the exact canonical appendix" >&2; exit 2
}
cleanup_expected_release_decision
trap - EXIT INT TERM
RELEASE_DIFF=$(git diff --name-only "$ACCEPTED_IMPLEMENTATION_COMMIT" "$EXPECTED_RELEASE_COMMIT")
test -n "$RELEASE_DIFF" || { echo "release-only commit has no changes" >&2; exit 2; }
release_config_seen=false
release_decision_seen=false
while IFS= read -r changed_path; do
  case "$changed_path" in
    configs/experiments/r05a_sampled_current_canary_apparatus.json) release_config_seen=true ;;
    docs/decisions/0037-require-exact-single-canary-release-identity.md) release_decision_seen=true ;;
    *) echo "non-release path changed in release commit: $changed_path" >&2; exit 2 ;;
  esac
done <<<"$RELEASE_DIFF"
test "$release_config_seen" = true && test "$release_decision_seen" = true || {
  echo "release commit is missing its config or decision binding" >&2; exit 2
}
EXPECTED_GIT_COMMIT=$EXPECTED_RELEASE_COMMIT
APPARATUS_CONFIG_SHA256=$(shasum -a 256 "$APPARATUS_CONFIG_LOCAL" | awk '{print $1}')
ENVELOPE_SCHEMA_SHA256=$(shasum -a 256 "$ENVELOPE_SCHEMA_LOCAL" | awk '{print $1}')
ADR0036_SHA256=$(shasum -a 256 "$ADR0036_LOCAL" | awk '{print $1}')
ADR0037_SHA256=$(shasum -a 256 "$ADR0037_LOCAL" | awk '{print $1}')

# Login-node work remains shell control-plane inspection only.
scripts/hpc/preflight.sh

ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$EXPECTED_GIT_COMMIT" "$EXPERIMENT_ROOT" \
  "$R02_RAW_ROOT" "$CHECKPOINT_DIR" "$EXPECTED_MANIFEST_SHA256" \
  "$EXPECTED_SCIENTIFIC_CONFIG_SHA256" "$EXPECTED_R02_SHA256" \
  "$EXPECTED_CHECKPOINT_SHA256" "$APPARATUS_CONFIG_SHA256" \
  "$ENVELOPE_SCHEMA_SHA256" "$ADR0036_SHA256" "$ADR0037_SHA256" \
  "$ACCEPTED_IMPLEMENTATION_COMMIT" <<'REMOTE'
set -euo pipefail
remote_repo=$1
run_id=$2
expected_commit=$3
experiment_root=$4
r02_raw_root=$5
checkpoint_dir=$6
expected_manifest_sha=$7
expected_scientific_config_sha=$8
expected_r02_sha=$9
expected_checkpoint_sha=${10}
expected_apparatus_config_sha=${11}
expected_envelope_schema_sha=${12}
expected_adr0036_sha=${13}
expected_adr0037_sha=${14}
accepted_implementation_commit=${15}

manifest=$remote_repo/manifests/r05a_inverse_flow_teacher_smoke.jsonl
scientific_config=$remote_repo/configs/experiments/r05a_inverse_flow_canary.json
apparatus_config=$remote_repo/configs/experiments/r05a_sampled_current_canary_apparatus.json
envelope_schema=$remote_repo/schemas/r05a-sampled-current-canary-envelope.schema.json
adr0036=$remote_repo/docs/decisions/0036-preregister-r05a-full-lifetime-sampled-current-canary.md
adr0037=$remote_repo/docs/decisions/0037-require-exact-single-canary-release-identity.md
source_r02=$r02_raw_root/crfs-1069f29a8d76463a/r02-paired.json
checkpoint=$checkpoint_dir/model.safetensors
gpu_slurm=$remote_repo/slurm/r05a_sampled_current_canary_h100.sbatch
cpu_slurm=$remote_repo/slurm/r05a_sampled_current_canary_validate_cpu.sbatch
run_root=$experiment_root/$run_id
case_dir=$run_root/crfs-1069f29a8d76463a
payload=$case_dir/canary-payload.json
host_telemetry=$case_dir/host-cgroup-sampled-current.tsv
gpu_samples=$case_dir/gpu-memory-samples.csv
test_log=$case_dir/allocation-focused-tests.log
candidate=$case_dir/.results.candidate.json
result=$case_dir/results.json
validation_receipt=$run_root/cpu-afterany-validation.json

bound_paths=(
  configs/experiments/r05a_inverse_flow_canary.json
  configs/experiments/r05a_sampled_current_canary_apparatus.json
  manifests/r05a_inverse_flow_teacher_smoke.jsonl
  docs/decisions/0028-pivot-to-inverse-flow-transport.md
  docs/decisions/0036-preregister-r05a-full-lifetime-sampled-current-canary.md
  docs/decisions/0037-require-exact-single-canary-release-identity.md
  docs/decisions/0038-preserve-sampled-current-launch-a-and-repair-token-parsing.md
  docs/decisions/0039-preserve-sampled-current-launch-b-and-repair-exact-task-accounting.md
  evidence/r05a/ift00a-sampled-current-launch-a.json
  evidence/r05a/ift00a-sampled-current-launch-b.json
  evidence/r05a/exact-array-task-afterany-regression-a.json
  schemas/r05a-inverse-flow-canary.schema.json
  schemas/r05a-sampled-current-canary-envelope.schema.json
  evidence/r03/r03-summary.json
  main/crfs_oracle/r05a_canary.py
  main/crfs_oracle/r05a_allocation_tests.json
  main/crfs_oracle/r05a_full_lifetime_telemetry.py
  main/crfs_oracle/r05a_sampled_current_canary.py
  main/run_crfs_r05a_canary.py
  main/publish_crfs_r05a_sampled_current_canary.py
  openpi/src/openpi/models_pytorch/crfs_inverse_control.py
  openpi/src/openpi/models_pytorch/pi0_pytorch.py
  openpi/src/openpi/policies/policy.py
  scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh
  scripts/hpc/lib/r05a_allocation_tests.sh
  scripts/hpc/lib/slurm_exact_array_task_status.sh
  scripts/hpc/run_r05a_canary.sh
  scripts/hpc/run_r05a_sampled_current_canary.sh
  scripts/hpc/validate_r05a_sampled_current_canary.sh
  scripts/hpc/submit_r05a_sampled_current_canary.sh
  slurm/r05a_sampled_current_canary_h100.sbatch
  slurm/r05a_sampled_current_canary_validate_cpu.sbatch
)
for relative in "${bound_paths[@]}"; do
  test -f "$remote_repo/$relative" && test ! -L "$remote_repo/$relative" || {
    echo "missing or symlinked remote bound source: $relative" >&2
    exit 2
  }
done
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || { echo "remote commit mismatch" >&2; exit 2; }
test -z "$(git -C "$remote_repo" status --porcelain)" || { echo "remote tree is dirty" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$expected_commit" || { echo "remote origin release ref mismatch" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-list --parents -n 1 "$expected_commit")" = "$expected_commit $accepted_implementation_commit" || { echo "remote release is not the direct implementation child" >&2; exit 2; }
for commit in "$accepted_implementation_commit" "$expected_commit"; do
  test "$(git -C "$remote_repo" ls-tree "$commit" -- configs/experiments/r05a_sampled_current_canary_apparatus.json | awk '{print $1" "$2}')" = "100644 blob" || { echo "remote apparatus config git mode changed" >&2; exit 2; }
done
remote_parent_apparatus=$(git -C "$remote_repo" show "$accepted_implementation_commit:configs/experiments/r05a_sampled_current_canary_apparatus.json")
jq -e '.ready_to_run == false and (.blocked_on | type == "array" and length > 0) and (has("execution_release") | not)' <<<"$remote_parent_apparatus" >/dev/null || { echo "remote implementation parent was not unreleased" >&2; exit 2; }
test "$(jq -S -c 'del(.ready_to_run,.blocked_on,.execution_release)' <<<"$remote_parent_apparatus")" = "$(jq -S -c 'del(.ready_to_run,.blocked_on,.execution_release)' "$apparatus_config")" || { echo "remote release changed non-release apparatus content" >&2; exit 2; }
remote_release_resource_json=$(jq -cS '.execution_release.resources' "$apparatus_config")
remote_appendix_implementation=$(jq -er '.execution_release.accepted_implementation_commit' "$apparatus_config")
remote_appendix_run_id=$(jq -er '.execution_release.run_id' "$apparatus_config")
remote_expected_decision=$(mktemp)
cleanup_remote_expected_decision() { rm -f "$remote_expected_decision"; }
trap cleanup_remote_expected_decision EXIT INT TERM
git -C "$remote_repo" show "$accepted_implementation_commit:docs/decisions/0037-require-exact-single-canary-release-identity.md" >"$remote_expected_decision"
{
  printf '\n## Exact execution release\n\n'
  printf 'Execution authorization: one preregistered IFT-00A canary submission only.\n\n'
  printf -- '- Accepted implementation commit: `%s`.\n' "$remote_appendix_implementation"
  printf -- '- Immutable run ID: `%s`.\n' "$remote_appendix_run_id"
  printf -- '- Source host: `worker-1`.\n'
  printf -- '- Resources (canonical JSON): `%s`.\n' "$remote_release_resource_json"
  printf -- '- Single submission: `true`.\n'
  printf -- '- Automatic resubmission: `false`.\n'
  printf -- '- Automatic next experiment: `false`.\n'
  printf -- '- Simulator efficacy claim authorized: `false`.\n'
  printf -- '- Probe or MLP training authorized: `false`.\n\n'
  printf '%s\n' 'This appendix authorizes only the frozen one-case mechanism canary. It does not authorize IFT-01, solver tuning, a simulator efficacy claim, label collection, probe training, or MLP training.'
} >>"$remote_expected_decision"
cmp -s "$remote_expected_decision" "$adr0037" || { echo "remote release decision is not the exact canonical appendix" >&2; exit 2; }
cleanup_remote_expected_decision
trap - EXIT INT TERM
remote_release_diff=$(git -C "$remote_repo" diff --name-only "$accepted_implementation_commit" "$expected_commit")
test -n "$remote_release_diff" || { echo "remote release-only commit has no changes" >&2; exit 2; }
remote_config_seen=false
remote_decision_seen=false
while IFS= read -r changed_path; do
  case "$changed_path" in
    configs/experiments/r05a_sampled_current_canary_apparatus.json) remote_config_seen=true ;;
    docs/decisions/0037-require-exact-single-canary-release-identity.md) remote_decision_seen=true ;;
    *) echo "remote non-release path changed: $changed_path" >&2; exit 2 ;;
  esac
done <<<"$remote_release_diff"
test "$remote_config_seen" = true && test "$remote_decision_seen" = true || { echo "remote release binding is incomplete" >&2; exit 2; }
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$expected_manifest_sha"
test "$(sha256sum "$scientific_config" | awk '{print $1}')" = "$expected_scientific_config_sha"
test "$(sha256sum "$apparatus_config" | awk '{print $1}')" = "$expected_apparatus_config_sha"
test "$(sha256sum "$envelope_schema" | awk '{print $1}')" = "$expected_envelope_schema_sha"
test "$(sha256sum "$adr0036" | awk '{print $1}')" = "$expected_adr0036_sha"
test "$(sha256sum "$adr0037" | awk '{print $1}')" = "$expected_adr0037_sha"
test "$(jq -er '.execution_release.run_id' "$apparatus_config")" = "$run_id" || { echo "remote registered run ID mismatch" >&2; exit 2; }
test "$(jq -er '.execution_release.accepted_implementation_commit' "$apparatus_config")" = "$accepted_implementation_commit" || { echo "remote accepted implementation mismatch" >&2; exit 2; }
jq -e --arg run "$run_id" --arg implementation "$accepted_implementation_commit" \
  '.ready_to_run == true and .blocked_on == []
   and ((.execution_release | keys | sort) == (["schema_version","artifact_role","decision_artifact","accepted_implementation_commit","run_id","single_submission","source_host","resources","release_only_parent_required","allowed_release_diff_paths","automatic_resubmission_allowed","automatic_next_experiment_allowed"] | sort))
   and .execution_release.schema_version == "1.0"
   and .execution_release.artifact_role == "r05a_single_canary_execution_release"
   and .execution_release.decision_artifact == "docs/decisions/0037-require-exact-single-canary-release-identity.md"
   and .execution_release.accepted_implementation_commit == $implementation
   and .execution_release.run_id == $run
   and .execution_release.single_submission == true
   and .execution_release.source_host == "worker-1"
   and .execution_release.resources == {partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}
   and .execution_release.release_only_parent_required == true
   and .execution_release.allowed_release_diff_paths == ["configs/experiments/r05a_sampled_current_canary_apparatus.json","docs/decisions/0037-require-exact-single-canary-release-identity.md"]
   and .execution_release.automatic_resubmission_allowed == false
   and .execution_release.automatic_next_experiment_allowed == false
   and .resource_contract == {source_host:"worker-1",partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}' \
  "$apparatus_config" >/dev/null || { echo "remote execution release contract changed" >&2; exit 2; }
test "$(sha256sum "$source_r02" | awk '{print $1}')" = "$expected_r02_sha"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint_sha"
test ! -e "$run_root" || { echo "immutable sampled-current run id is already used" >&2; exit 2; }
if [ -n "$(squeue -h -u "$(whoami)" -o '%i')" ]; then
  echo "sampled-current canary not submitted: user queue is not empty" >&2
  exit 2
fi
node_record=$(scontrol show node worker-1 -o)
node_state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$node_state" in *down*|*drain*|*fail*|*maint*|*not_resp*|*unknown*|*power*) echo "worker-1 unhealthy: $node_state" >&2; exit 2 ;; esac
free_mem=$(printf '%s\n' "$node_record" | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
case "$free_mem" in *[!0-9]*|'') echo "cannot parse worker-1 FreeMem" >&2; exit 2 ;; esac
test "$free_mem" -ge 65536 || { echo "worker-1 FreeMem=${free_mem}MiB < 65536MiB" >&2; exit 2; }
case " $node_record " in *" Gres="*"gpu:nvidia_h100_80gb_hbm3:8"*) ;; *) echo "worker-1 does not advertise H100" >&2; exit 2 ;; esac

mkdir "$run_root"
reservation_tmp=$(mktemp "$run_root/.launch-reservation.XXXXXX")
jq -n --arg run_id "$run_id" --arg commit "$expected_commit" --arg implementation "$accepted_implementation_commit" --arg decision "docs/decisions/0037-require-exact-single-canary-release-identity.md" --arg free_mem "$free_mem" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"2.0",artifact_role:"r05a_sampled_current_canary_launch_reservation",status:"preflight_verified_before_queueing",run_id:$run_id,git_commit:$commit,accepted_implementation_commit:$implementation,release_decision_path:$decision,single_submission:true,source_node:"worker-1",observed_free_mem_mib:($free_mem|tonumber),requested_gpus:1,requested_cpus:8,requested_host_memory_mib:65536,array:"0-0%1",scientific_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$reservation_tmp"
mv "$reservation_tmp" "$run_root/launch-reservation.json"

gpu_submission=$(RUN_ID="$run_id" MANIFEST="$manifest" EXPERIMENT_CONFIG="$scientific_config" R02_RAW_ROOT="$r02_raw_root" CHECKPOINT_DIR="$checkpoint_dir" EXPERIMENT_ROOT="$experiment_root" EXPECTED_GIT_COMMIT="$expected_commit" REMOTE_REPO="$remote_repo" sbatch --parsable --hold --partition=main --account=normal --qos=normal --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --no-requeue --nodelist=worker-1 --array=0-0%1 --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' "$gpu_slurm")
gpu_job_id=${gpu_submission%%;*}
case "$gpu_job_id" in *[!0-9]*|'') echo "invalid GPU job id" >&2; exit 2 ;; esac
gpu_record=$(scontrol show job "$gpu_job_id" -o)
for field in "JobState=PENDING" "Reason=JobHeldUser" "ReqNodeList=worker-1"; do
  case " $gpu_record " in *" $field "*) ;; *) echo "held GPU job field changed: $field" >&2; exit 2 ;; esac
done
for field in "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=02:00:00" "Requeue=0"; do
  case " $gpu_record " in *" $field "*) ;; *) echo "held GPU job field changed: $field" >&2; exit 2 ;; esac
done
gpu_req_tres=$(printf '%s\n' "$gpu_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test -n "$gpu_req_tres" || { echo "cannot parse held GPU ReqTRES" >&2; exit 2; }
gpu_req_cpus=$(printf '%s\n' "$gpu_req_tres" | tr ',' '\n' | sed -n 's/^cpu=//p')
gpu_req_mem=$(printf '%s\n' "$gpu_req_tres" | tr ',' '\n' | sed -n 's/^mem=//p')
gpu_req_count=$(printf '%s\n' "$gpu_req_tres" | tr ',' '\n' | sed -n 's#^gres/gpu=##p')
test "$gpu_req_cpus" = 8 || { echo "held GPU CPU request changed: $gpu_req_cpus" >&2; exit 2; }
case "$gpu_req_mem" in 64G|65536M) ;; *) echo "held GPU memory request changed: $gpu_req_mem" >&2; exit 2 ;; esac
test "$gpu_req_count" = 1 || { echo "held GPU count changed: $gpu_req_count" >&2; exit 2; }

resources=$(jq -n '{partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}')
held_tmp=$(mktemp "$run_root/.held-gpu.XXXXXX")
jq -n --arg run_id "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --argjson resources "$resources" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"2.0",artifact_role:"r05a_sampled_current_canary_held_gpu_submission",status:"sbatch_returned_held_gpu_id",run_id:$run_id,git_commit:$commit,gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),source_node:"worker-1",resources:$resources,released_at_receipt_time:false,scientific_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$held_tmp"
mv "$held_tmp" "$run_root/held-gpu-submission.json"

repository_hashes='{}'
for relative in "${bound_paths[@]}"; do
  digest=$(sha256sum "$remote_repo/$relative" | awk '{print $1}')
  repository_hashes=$(jq -c --arg key "$relative" --arg value "$digest" '. + {($key):$value}' <<<"$repository_hashes")
done
frozen_bindings=$(jq -n '{scientific_config_file_sha256:"c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb",scientific_config_projection_sha256:"9e2ff74cda8ac3b5d4098942a82cc3352cebea3f4b8e1fc1d326c2dc24d2887d",manifest_sha256:"bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633",adr0028_sha256:"d2a00b1b049e92bb1ec8f11d60fa447e5e8bd5109cb8cf3f3fbb08f89498656f",historical_scientific_schema_sha256:"e1681f865f81f2986945d10fb14073fe4cd9e78cbc5026f6749ab321b9cfb7a7",source_r02_sha256:"055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",source_r02_config_sha256:"c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e",source_r03_summary_sha256:"dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e",source_r03_ordered_result_set_digest:"fa79a132f2fecbedab5917111ef71f022e52c933d8e535aaca118add5f5b7895",checkpoint_sha256:"988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",normalization_asset_sha256:"b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",baseline_revision:"57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",r05a_canary_module_sha256:"9d4402ccb92835bc1af2a3e04fe54c0b94839f41beb2971022216f73ec67fede",r05a_live_entrypoint_sha256:"247bd20e48ffe2228d2633d2f667a2791371163cca1c43ad7d278f4fbfd7439a",inverse_control_implementation_sha256:"965082822466774e0a86eeb6f2d178c9e5f3d6d02bca5090c4e1fcc77bc52aa8",pi05_sampler_sha256:"80366dcc7b2ddc598717d4c71c0e68e46312a1e5ffd3fc04479d5599433f4c55",policy_boundary_sha256:"d16767ff2073d5c177cdfcc06dc05dcbf7cdb9ef2a2a0150023f7953b03508b9"}')
artifact_paths=$(jq -n --arg run_root "$run_root" --arg case_dir "$case_dir" --arg payload "$payload" --arg host "$host_telemetry" --arg gpu "$gpu_samples" --arg tests "$test_log" --arg candidate "$candidate" --arg result "$result" --arg receipt "$validation_receipt" '{run_root:$run_root,case_dir:$case_dir,payload:$payload,host_telemetry:$host,gpu_samples:$gpu,allocation_tests_log:$tests,hidden_candidate:$candidate,result:$result,validation_receipt:$receipt}')
source_tmp=$(mktemp "$run_root/.source-contract.XXXXXX")
jq -n --arg run_id "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --argjson resources "$resources" --argjson paths "$artifact_paths" --argjson frozen "$frozen_bindings" --argjson hashes "$repository_hashes" --arg held_path "$run_root/held-gpu-submission.json" --arg held_sha "$(sha256sum "$run_root/held-gpu-submission.json" | awk '{print $1}')" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"2.0",artifact_role:"r05a_sampled_current_canary_source_contract",status:"gpu_held_sources_bound_before_cpu_submission",run_id:$run_id,git_commit:$commit,git_dirty:false,source_node:"worker-1",gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),resources:$resources,artifact_paths:$paths,frozen_bindings:$frozen,repository_file_sha256:$hashes,held_gpu_submission_path:$held_path,held_gpu_submission_sha256:$held_sha,scientific_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$source_tmp"
mv "$source_tmp" "$run_root/source-contract.json"
source_contract_sha=$(sha256sum "$run_root/source-contract.json" | awk '{print $1}')

dependency=afterany:$gpu_job_id
cpu_submission=$(SOURCE_JOB_ID="$gpu_job_id" SOURCE_CONTRACT="$run_root/source-contract.json" EXPECTED_SOURCE_CONTRACT_SHA256="$source_contract_sha" SUBMISSION="$run_root/submission.json" PAYLOAD="$payload" HOST_TELEMETRY="$host_telemetry" GPU_SAMPLES="$gpu_samples" ALLOCATION_TEST_LOG="$test_log" RESULT="$result" VALIDATION_RECEIPT="$validation_receipt" EXPECTED_GIT_COMMIT="$expected_commit" REMOTE_REPO="$remote_repo" sbatch --parsable --partition=main --account=normal --qos=normal --cpus-per-task=2 --mem=8G --time=00:15:00 --no-requeue --dependency="$dependency" --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' "$cpu_slurm")
cpu_job_id=${cpu_submission%%;*}
case "$cpu_job_id" in *[!0-9]*|'') echo "invalid CPU publisher job id" >&2; exit 2 ;; esac
cpu_record=$(scontrol show job "$cpu_job_id" -o)
for field in "JobState=PENDING" "Partition=main"; do
  case " $cpu_record " in *" $field "*) ;; *) echo "CPU publisher job field changed: $field" >&2; exit 2 ;; esac
done
case " $cpu_record " in *gres/gpu*) echo "CPU publisher unexpectedly requests GPU" >&2; exit 2 ;; esac
for field in "Account=normal" "QOS=normal" "TimeLimit=00:15:00" "Requeue=0"; do
  case " $cpu_record " in *" $field "*) ;; *) echo "CPU publisher job field changed: $field" >&2; exit 2 ;; esac
done
cpu_req_tres=$(printf '%s\n' "$cpu_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test -n "$cpu_req_tres" || { echo "cannot parse CPU publisher ReqTRES" >&2; exit 2; }
cpu_req_cpus=$(printf '%s\n' "$cpu_req_tres" | tr ',' '\n' | sed -n 's/^cpu=//p')
cpu_req_mem=$(printf '%s\n' "$cpu_req_tres" | tr ',' '\n' | sed -n 's/^mem=//p')
test "$cpu_req_cpus" = 2 || { echo "CPU publisher CPU request changed: $cpu_req_cpus" >&2; exit 2; }
case "$cpu_req_mem" in 8G|8192M) ;; *) echo "CPU publisher memory request changed: $cpu_req_mem" >&2; exit 2 ;; esac
observed_dependency=$(printf '%s\n' "$cpu_record" | sed -n 's/.* Dependency=\([^ ]*\).*/\1/p' | sed 's/(unfulfilled)//g; s/_\*//g')
test "$observed_dependency" = "$dependency" || { echo "CPU dependency changed" >&2; exit 2; }

submission_tmp=$(mktemp "$run_root/.submission.XXXXXX")
jq -n --arg run_id "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg cpu "$cpu_job_id" --arg dependency "$dependency" --arg source_contract "$run_root/source-contract.json" --arg source_sha "$source_contract_sha" --arg result "$result" --arg receipt "$validation_receipt" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"2.0",artifact_role:"r05a_sampled_current_canary_atomic_submission",status:"cpu_afterany_registered_gpu_held",run_id:$run_id,git_commit:$commit,source_node:"worker-1",gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,cpu_afterany_job_id:$cpu,dependency:$dependency,source_contract_path:$source_contract,source_contract_sha256:$source_sha,expected_result:$result,expected_validation_receipt:$receipt,scientific_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$submission_tmp"
mv "$submission_tmp" "$run_root/submission.json"

scontrol release "$gpu_job_id"
echo "submitted_gpu_job_id=$gpu_job_id"
echo "submitted_cpu_publisher_job_id=$cpu_job_id"
echo "dependency=$dependency"
echo "source_contract_sha256=$source_contract_sha"
echo "submission_receipt=$run_root/submission.json"
echo "submission_receipt_sha256=$(sha256sum "$run_root/submission.json" | awk '{print $1}')"
REMOTE
