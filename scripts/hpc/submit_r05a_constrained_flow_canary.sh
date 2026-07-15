#!/usr/bin/env bash
set -euo pipefail

usage='usage: RUN_ID=... EXPECTED_RELEASE_COMMIT=... submit_r05a_constrained_flow_canary.sh'
: "${RUN_ID:?$usage}"
: "${EXPECTED_RELEASE_COMMIT:?$usage}"
case "$RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "unsafe CFS-00A RUN_ID" >&2; exit 2 ;; esac
case "$RUN_ID" in [A-Za-z0-9]*) ;; *) echo "CFS-00A RUN_ID must start alphanumeric" >&2; exit 2 ;; esac
test "${#RUN_ID}" -le 128 || { echo "CFS-00A RUN_ID too long" >&2; exit 2; }
case "$EXPECTED_RELEASE_COMMIT" in *[!0-9a-f]*|'') echo "invalid release commit" >&2; exit 2 ;; esac
test "${#EXPECTED_RELEASE_COMMIT}" = 40 || { echo "invalid release commit length" >&2; exit 2; }

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
EXPERIMENT_ROOT=/mnt/data/quanth/experiments/crfs-oracle
R02_RAW_ROOT=/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a
CHECKPOINT_DIR=/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
MANIFEST_LOCAL=manifests/r05a_inverse_flow_teacher_smoke.jsonl
CFS_CONFIG_LOCAL=configs/experiments/r05a_constrained_flow_canary.json
LEGACY_CONFIG_LOCAL=configs/experiments/r05a_inverse_flow_canary.json
APPARATUS_CONFIG_LOCAL=configs/experiments/r05a_constrained_flow_canary_apparatus.json
ENVELOPE_SCHEMA_LOCAL=schemas/r05a-constrained-flow-canary-envelope.schema.json
ADR0040_LOCAL=docs/decisions/0040-preregister-same-budget-constrained-flow-diagnostic.md
ADR0041_LOCAL=docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md

EXPECTED_MANIFEST_SHA256=bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633
EXPECTED_LEGACY_CONFIG_SHA256=c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb
EXPECTED_R02_SHA256=055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593
EXPECTED_CHECKPOINT_SHA256=988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed

BOUND_REPOSITORY_PATHS=(
  configs/experiments/r05a_constrained_flow_canary.json
  configs/experiments/r05a_constrained_flow_canary_apparatus.json
  configs/experiments/r05a_inverse_flow_canary.json
  manifests/r05a_inverse_flow_teacher_smoke.jsonl
  docs/decisions/0040-preregister-same-budget-constrained-flow-diagnostic.md
  docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md
  schemas/r05a-inverse-flow-canary.schema.json
  schemas/r05a-constrained-flow-canary-envelope.schema.json
  main/crfs_oracle/r05a_canary.py
  main/crfs_oracle/r05a_constrained_flow_allocation_tests.json
  main/crfs_oracle/r05a_constrained_flow_canary.py
  main/crfs_oracle/r05a_constrained_flow_validation.py
  main/crfs_oracle/r05a_constrained_flow_publication.py
  main/crfs_oracle/r05a_full_lifetime_telemetry.py
  main/crfs_oracle/r05a_sampled_current_canary.py
  main/run_crfs_r05a_canary.py
  main/run_crfs_r05a_constrained_flow_canary.py
  main/publish_crfs_r05a_constrained_flow_canary.py
  openpi/scripts/serve_cfs_policy.py
  openpi/scripts/serve_policy.py
  openpi/src/openpi/models_pytorch/crfs_inverse_control.py
  openpi/src/openpi/models_pytorch/crfs_linearized_control.py
  openpi/src/openpi/models_pytorch/pi0_pytorch.py
  openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py
  openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
  openpi/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py
  openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/check.py
  openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py
  openpi/src/openpi/policies/crfs_constrained_flow_adapter.py
  openpi/src/openpi/policies/policy.py
  scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh
  scripts/hpc/lib/r05a_allocation_tests.sh
  scripts/hpc/lib/slurm_exact_array_task_status.sh
  scripts/hpc/prepare_jsonschema_overlay.sh
  scripts/hpc/prepare_transformers_overlay.sh
  scripts/hpc/run_r05a_constrained_flow_workload.sh
  scripts/hpc/run_r05a_constrained_flow_canary.sh
  scripts/hpc/validate_r05a_constrained_flow_canary.sh
  scripts/hpc/submit_r05a_constrained_flow_canary.sh
  slurm/r05a_constrained_flow_canary_h100.sbatch
  slurm/r05a_constrained_flow_canary_validate_cpu.sbatch
  tests/test_inverse_flow_control.py
  tests/test_inverse_flow_sampler.py
  tests/test_inverse_flow_policy.py
  tests/test_r05a_canary.py
  tests/test_linearized_flow_control.py
  tests/test_constrained_flow_adapter.py
  tests/test_r05a_constrained_flow_canary.py
  tests/test_r05a_constrained_flow_hpc_contract.py
  tests/test_r05a_constrained_flow_publication.py
  tests/test_r05a_constrained_flow_validation.py
)

for path in "${BOUND_REPOSITORY_PATHS[@]}"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked CFS-00A bound source: $path" >&2; exit 2; }
done
test "$(shasum -a 256 "$MANIFEST_LOCAL" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA256" || { echo "frozen manifest changed" >&2; exit 2; }
test "$(shasum -a 256 "$LEGACY_CONFIG_LOCAL" | awk '{print $1}')" = "$EXPECTED_LEGACY_CONFIG_SHA256" || { echo "legacy R05A config changed" >&2; exit 2; }
test "$(shasum -a 256 openpi/src/openpi/models_pytorch/crfs_inverse_control.py | awk '{print $1}')" = 965082822466774e0a86eeb6f2d178c9e5f3d6d02bca5090c4e1fcc77bc52aa8
test "$(shasum -a 256 openpi/src/openpi/models_pytorch/pi0_pytorch.py | awk '{print $1}')" = 80366dcc7b2ddc598717d4c71c0e68e46312a1e5ffd3fc04479d5599433f4c55
test "$(shasum -a 256 openpi/src/openpi/policies/policy.py | awk '{print $1}')" = d16767ff2073d5c177cdfcc06dc05dcbf7cdb9ef2a2a0150023f7953b03508b9
test "$(shasum -a 256 openpi/scripts/serve_policy.py | awk '{print $1}')" = eccc0448b4873fd30a1fff3355c5de7a7c227138e6db04b7dcaa5bb38a6a5809

jq -e '.config_status == "released_exact_single_canary" and .ready_to_run == true and .blocked_on == [] and .preregistration.h100_submission_authorized == true and (.execution_release|type=="object")' "$CFS_CONFIG_LOCAL" >/dev/null || { echo "scientific CFS config is not exactly released" >&2; exit 2; }
jq -e '.ready_to_run == true and .blocked_on == []' "$APPARATUS_CONFIG_LOCAL" >/dev/null || { echo "CFS apparatus is not released" >&2; exit 2; }
REGISTERED_RUN_ID=$(jq -er '.execution_release.run_id' "$APPARATUS_CONFIG_LOCAL")
ACCEPTED_IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$APPARATUS_CONFIG_LOCAL")
case "$ACCEPTED_IMPLEMENTATION_COMMIT" in *[!0-9a-f]*|'') echo "invalid implementation commit" >&2; exit 2 ;; esac
test "${#ACCEPTED_IMPLEMENTATION_COMMIT}" = 40 || { echo "invalid implementation commit length" >&2; exit 2; }
test "$RUN_ID" = "$REGISTERED_RUN_ID" || { echo "caller run ID is not exact release identity" >&2; exit 2; }

jq -e --arg run "$RUN_ID" --arg implementation "$ACCEPTED_IMPLEMENTATION_COMMIT" '
  .ready_to_run == true and .blocked_on == []
  and ((.execution_release | keys | sort) == (["schema_version","artifact_role","decision_artifact","accepted_implementation_commit","run_id","single_submission","source_host","resources","release_only_parent_required","allowed_release_diff_paths","automatic_resubmission_allowed","automatic_next_experiment_allowed"] | sort))
  and .execution_release.schema_version == "1.0"
  and .execution_release.artifact_role == "r05a_constrained_flow_canary_execution_release"
  and .execution_release.decision_artifact == "docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md"
  and .execution_release.accepted_implementation_commit == $implementation
  and .execution_release.run_id == $run and .execution_release.single_submission == true
  and .execution_release.source_host == "worker-1"
  and .execution_release.resources == {partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}
  and .execution_release.release_only_parent_required == true
  and .execution_release.allowed_release_diff_paths == ["configs/experiments/r05a_constrained_flow_canary.json","configs/experiments/r05a_constrained_flow_canary_apparatus.json","docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md"]
  and .execution_release.automatic_resubmission_allowed == false
  and .execution_release.automatic_next_experiment_allowed == false
  and .resource_contract == {partition:"main",account:"normal",qos:"normal",source_host:"worker-1",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}' "$APPARATUS_CONFIG_LOCAL" >/dev/null || { echo "exact CFS execution release changed" >&2; exit 2; }
test "$(jq -cS '.execution_release' "$CFS_CONFIG_LOCAL")" = "$(jq -cS '.execution_release' "$APPARATUS_CONFIG_LOCAL")" || { echo "scientific and apparatus execution releases differ" >&2; exit 2; }
test "$(jq -er '.scientific_config.sha256' "$APPARATUS_CONFIG_LOCAL")" = "$(shasum -a 256 "$CFS_CONFIG_LOCAL" | awk '{print $1}')" || { echo "apparatus does not bind released scientific config" >&2; exit 2; }
test "$(jq -er '.envelope_schema.sha256' "$APPARATUS_CONFIG_LOCAL")" = "$(shasum -a 256 "$ENVELOPE_SCHEMA_LOCAL" | awk '{print $1}')" || { echo "apparatus envelope binding changed" >&2; exit 2; }
test "$(jq -er '.legacy_scientific_config.sha256' "$APPARATUS_CONFIG_LOCAL")" = "$(shasum -a 256 "$LEGACY_CONFIG_LOCAL" | awk '{print $1}')" || { echo "apparatus legacy config binding changed" >&2; exit 2; }
test "$(jq -er '.semantic_validator.sha256' "$APPARATUS_CONFIG_LOCAL")" = "$(shasum -a 256 main/crfs_oracle/r05a_constrained_flow_validation.py | awk '{print $1}')" || { echo "apparatus semantic-validator binding changed" >&2; exit 2; }
test "$(jq -er '.allocation_test_contract.registry_sha256' "$APPARATUS_CONFIG_LOCAL")" = "$(shasum -a 256 main/crfs_oracle/r05a_constrained_flow_allocation_tests.json | awk '{print $1}')" || { echo "apparatus allocation registry binding changed" >&2; exit 2; }
test "$(jq -cS '.allocation_test_contract.expected_counts' "$APPARATUS_CONFIG_LOCAL")" = "$(jq -cS '[.suites[] | {key:.pattern,value:.expected_tests}] | from_entries' main/crfs_oracle/r05a_constrained_flow_allocation_tests.json)" || { echo "apparatus allocation test counts changed" >&2; exit 2; }
test "$(jq -er '.allocation_test_contract.zero_skips_required' "$APPARATUS_CONFIG_LOCAL")" = true || { echo "apparatus no longer requires zero allocation-test skips" >&2; exit 2; }

git ls-files --error-unmatch "${BOUND_REPOSITORY_PATHS[@]}" >/dev/null || { echo "all CFS sources must be committed" >&2; exit 2; }
test -z "$(git status --porcelain)" || { echo "submission requires clean reviewed worktree" >&2; exit 2; }
test "$(git rev-parse HEAD)" = "$EXPECTED_RELEASE_COMMIT" || { echo "local HEAD differs from authorized release" >&2; exit 2; }
test "$(git rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$EXPECTED_RELEASE_COMMIT" || { echo "local origin differs from authorized release" >&2; exit 2; }
test "$(git rev-list --parents -n 1 "$EXPECTED_RELEASE_COMMIT")" = "$EXPECTED_RELEASE_COMMIT $ACCEPTED_IMPLEMENTATION_COMMIT" || { echo "release is not direct implementation child" >&2; exit 2; }

PARENT_CFS_CONFIG=$(git show "$ACCEPTED_IMPLEMENTATION_COMMIT:$CFS_CONFIG_LOCAL")
PARENT_APPARATUS_CONFIG=$(git show "$ACCEPTED_IMPLEMENTATION_COMMIT:$APPARATUS_CONFIG_LOCAL")
jq -e '.config_status == "draft_preregistered_not_released" and .ready_to_run == false and (.blocked_on|type=="array" and length>0) and .preregistration.h100_submission_authorized == false and ((.execution_release? // null) == null)' <<<"$PARENT_CFS_CONFIG" >/dev/null || { echo "implementation scientific config was not fail closed" >&2; exit 2; }
jq -e '.ready_to_run == false and (.blocked_on|type=="array" and length>0) and .execution_release == null' <<<"$PARENT_APPARATUS_CONFIG" >/dev/null || { echo "implementation apparatus was not fail closed" >&2; exit 2; }
test "$(jq -S -c 'del(.config_status,.ready_to_run,.blocked_on,.preregistration.h100_submission_authorized,.execution_release)' <<<"$PARENT_CFS_CONFIG")" = "$(jq -S -c 'del(.config_status,.ready_to_run,.blocked_on,.preregistration.h100_submission_authorized,.execution_release)' "$CFS_CONFIG_LOCAL")" || { echo "release changed scientific content" >&2; exit 2; }
test "$(jq -S -c 'del(.ready_to_run,.blocked_on,.execution_release,.scientific_config.sha256)' <<<"$PARENT_APPARATUS_CONFIG")" = "$(jq -S -c 'del(.ready_to_run,.blocked_on,.execution_release,.scientific_config.sha256)' "$APPARATUS_CONFIG_LOCAL")" || { echo "release changed non-release apparatus content" >&2; exit 2; }

RELEASE_DIFF=$(git diff --name-only "$ACCEPTED_IMPLEMENTATION_COMMIT" "$EXPECTED_RELEASE_COMMIT")
test "$RELEASE_DIFF" = "$(printf '%s\n' "$CFS_CONFIG_LOCAL" "$APPARATUS_CONFIG_LOCAL" "$ADR0041_LOCAL")" || { echo "release-only path set changed" >&2; exit 2; }
RELEASE_RESOURCE_JSON=$(jq -cS '.execution_release.resources' "$APPARATUS_CONFIG_LOCAL")
EXPECTED_RELEASE_DECISION=$(mktemp)
cleanup_expected_decision() { rm -f "$EXPECTED_RELEASE_DECISION"; }
trap cleanup_expected_decision EXIT INT TERM
git show "$ACCEPTED_IMPLEMENTATION_COMMIT:$ADR0041_LOCAL" >"$EXPECTED_RELEASE_DECISION"
{
  printf '\n## Exact execution release\n\n'
  printf 'Execution authorization: one preregistered CFS-00A canary submission only.\n\n'
  printf -- '- Accepted implementation commit: `%s`.\n' "$ACCEPTED_IMPLEMENTATION_COMMIT"
  printf -- '- Immutable run ID: `%s`.\n' "$RUN_ID"
  printf -- '- Source host: `worker-1`.\n'
  printf -- '- Resources (canonical JSON): `%s`.\n' "$RELEASE_RESOURCE_JSON"
  printf -- '- Single submission: `true`.\n'
  printf -- '- Automatic resubmission: `false`.\n'
  printf -- '- Automatic next experiment: `false`.\n'
  printf -- '- Simulator efficacy claim authorized: `false`.\n'
  printf -- '- Infeasibility claim authorized: `false`.\n'
  printf -- '- Probe or MLP training authorized: `false`.\n\n'
  printf '%s\n' 'This appendix authorizes only the frozen one-case same-budget transport mechanism canary. It does not authorize IFT-01, solver or tolerance tuning, simulator execution of generated actions, an efficacy or infeasibility claim, label collection, probe training, or MLP training.'
} >>"$EXPECTED_RELEASE_DECISION"
cmp -s "$EXPECTED_RELEASE_DECISION" "$ADR0041_LOCAL" || { echo "release decision is not canonical appendix" >&2; exit 2; }
cleanup_expected_decision
trap - EXIT INT TERM

CFS_CONFIG_SHA256=$(shasum -a 256 "$CFS_CONFIG_LOCAL" | awk '{print $1}')
APPARATUS_CONFIG_SHA256=$(shasum -a 256 "$APPARATUS_CONFIG_LOCAL" | awk '{print $1}')
ENVELOPE_SCHEMA_SHA256=$(shasum -a 256 "$ENVELOPE_SCHEMA_LOCAL" | awk '{print $1}')
ADR0040_SHA256=$(shasum -a 256 "$ADR0040_LOCAL" | awk '{print $1}')
ADR0041_SHA256=$(shasum -a 256 "$ADR0041_LOCAL" | awk '{print $1}')

# Login node is control plane only.
scripts/hpc/preflight.sh
ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$EXPECTED_RELEASE_COMMIT" "$EXPERIMENT_ROOT" "$R02_RAW_ROOT" "$CHECKPOINT_DIR" \
  "$EXPECTED_MANIFEST_SHA256" "$CFS_CONFIG_SHA256" "$EXPECTED_LEGACY_CONFIG_SHA256" "$EXPECTED_R02_SHA256" "$EXPECTED_CHECKPOINT_SHA256" \
  "$APPARATUS_CONFIG_SHA256" "$ENVELOPE_SCHEMA_SHA256" "$ADR0040_SHA256" "$ADR0041_SHA256" "$ACCEPTED_IMPLEMENTATION_COMMIT" <<'REMOTE'
set -euo pipefail
remote_repo=$1; run_id=$2; expected_commit=$3; experiment_root=$4; r02_raw_root=$5; checkpoint_dir=$6
expected_manifest_sha=$7; expected_cfs_config_sha=$8; expected_legacy_config_sha=$9; expected_r02_sha=${10}; expected_checkpoint_sha=${11}
expected_apparatus_sha=${12}; expected_schema_sha=${13}; expected_adr0040_sha=${14}; expected_adr0041_sha=${15}; accepted_implementation_commit=${16}

manifest=$remote_repo/manifests/r05a_inverse_flow_teacher_smoke.jsonl
cfs_config=$remote_repo/configs/experiments/r05a_constrained_flow_canary.json
legacy_config=$remote_repo/configs/experiments/r05a_inverse_flow_canary.json
apparatus_config=$remote_repo/configs/experiments/r05a_constrained_flow_canary_apparatus.json
envelope_schema=$remote_repo/schemas/r05a-constrained-flow-canary-envelope.schema.json
adr0040=$remote_repo/docs/decisions/0040-preregister-same-budget-constrained-flow-diagnostic.md
adr0041=$remote_repo/docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md
source_r02=$r02_raw_root/crfs-1069f29a8d76463a/r02-paired.json
checkpoint=$checkpoint_dir/model.safetensors
gpu_slurm=$remote_repo/slurm/r05a_constrained_flow_canary_h100.sbatch
cpu_slurm=$remote_repo/slurm/r05a_constrained_flow_canary_validate_cpu.sbatch
run_root=$experiment_root/$run_id
case_dir=$run_root/crfs-1069f29a8d76463a
legacy_payload=$case_dir/canary-payload.json
cfs_payload=$case_dir/constrained-flow-payload.json
host_telemetry=$case_dir/host-cgroup-sampled-current.tsv
gpu_samples=$case_dir/gpu-memory-samples.csv
test_log=$case_dir/allocation-focused-tests.log
candidate=$case_dir/.results.candidate.json
result=$case_dir/results.json
validation_receipt=$run_root/cpu-afterany-validation.json
provisional_gpu_receipt=$run_root/provisional-gpu-job-id.json
provisional_cpu_receipt=$run_root/provisional-cpu-job-id.json

bound_paths=(
  configs/experiments/r05a_constrained_flow_canary.json configs/experiments/r05a_constrained_flow_canary_apparatus.json configs/experiments/r05a_inverse_flow_canary.json
  manifests/r05a_inverse_flow_teacher_smoke.jsonl
  docs/decisions/0040-preregister-same-budget-constrained-flow-diagnostic.md docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md
  schemas/r05a-inverse-flow-canary.schema.json schemas/r05a-constrained-flow-canary-envelope.schema.json
  main/crfs_oracle/r05a_canary.py main/crfs_oracle/r05a_constrained_flow_allocation_tests.json
  main/crfs_oracle/r05a_constrained_flow_canary.py main/crfs_oracle/r05a_constrained_flow_validation.py main/crfs_oracle/r05a_constrained_flow_publication.py main/crfs_oracle/r05a_full_lifetime_telemetry.py main/crfs_oracle/r05a_sampled_current_canary.py
  main/run_crfs_r05a_canary.py main/run_crfs_r05a_constrained_flow_canary.py main/publish_crfs_r05a_constrained_flow_canary.py
  openpi/scripts/serve_cfs_policy.py openpi/scripts/serve_policy.py openpi/src/openpi/models_pytorch/crfs_inverse_control.py openpi/src/openpi/models_pytorch/crfs_linearized_control.py
  openpi/src/openpi/models_pytorch/pi0_pytorch.py
  openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
  openpi/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/check.py openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py
  openpi/src/openpi/policies/crfs_constrained_flow_adapter.py openpi/src/openpi/policies/policy.py
  scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh scripts/hpc/lib/r05a_allocation_tests.sh scripts/hpc/lib/slurm_exact_array_task_status.sh
  scripts/hpc/prepare_jsonschema_overlay.sh scripts/hpc/prepare_transformers_overlay.sh
  scripts/hpc/run_r05a_constrained_flow_workload.sh scripts/hpc/run_r05a_constrained_flow_canary.sh scripts/hpc/validate_r05a_constrained_flow_canary.sh scripts/hpc/submit_r05a_constrained_flow_canary.sh
  slurm/r05a_constrained_flow_canary_h100.sbatch slurm/r05a_constrained_flow_canary_validate_cpu.sbatch
  tests/test_inverse_flow_control.py tests/test_inverse_flow_sampler.py tests/test_inverse_flow_policy.py tests/test_r05a_canary.py
  tests/test_linearized_flow_control.py tests/test_constrained_flow_adapter.py tests/test_r05a_constrained_flow_canary.py tests/test_r05a_constrained_flow_hpc_contract.py tests/test_r05a_constrained_flow_publication.py tests/test_r05a_constrained_flow_validation.py
)
for relative in "${bound_paths[@]}"; do test -f "$remote_repo/$relative" && test ! -L "$remote_repo/$relative" || { echo "missing remote bound source: $relative" >&2; exit 2; }; done
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit" || { echo "remote commit mismatch" >&2; exit 2; }
test -z "$(git -C "$remote_repo" status --porcelain)" || { echo "remote tree dirty" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$expected_commit" || { echo "remote origin mismatch" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-list --parents -n 1 "$expected_commit")" = "$expected_commit $accepted_implementation_commit" || { echo "remote release parent mismatch" >&2; exit 2; }
test "$(sha256sum "$manifest"|awk '{print $1}')" = "$expected_manifest_sha"
test "$(sha256sum "$cfs_config"|awk '{print $1}')" = "$expected_cfs_config_sha"
test "$(sha256sum "$legacy_config"|awk '{print $1}')" = "$expected_legacy_config_sha"
test "$(sha256sum "$apparatus_config"|awk '{print $1}')" = "$expected_apparatus_sha"
test "$(sha256sum "$envelope_schema"|awk '{print $1}')" = "$expected_schema_sha"
test "$(sha256sum "$adr0040"|awk '{print $1}')" = "$expected_adr0040_sha"
test "$(sha256sum "$adr0041"|awk '{print $1}')" = "$expected_adr0041_sha"
test "$(sha256sum "$source_r02"|awk '{print $1}')" = "$expected_r02_sha"
test "$(sha256sum "$checkpoint"|awk '{print $1}')" = "$expected_checkpoint_sha"
test "$(sha256sum "$remote_repo/openpi/scripts/serve_policy.py"|awk '{print $1}')" = eccc0448b4873fd30a1fff3355c5de7a7c227138e6db04b7dcaa5bb38a6a5809
test "$(jq -er '.execution_release.run_id' "$apparatus_config")" = "$run_id"
test "$(jq -er '.execution_release.accepted_implementation_commit' "$apparatus_config")" = "$accepted_implementation_commit"
jq -e '.config_status == "released_exact_single_canary" and .ready_to_run == true and .blocked_on == [] and .preregistration.h100_submission_authorized == true and (.execution_release|type=="object")' "$cfs_config" >/dev/null || { echo "remote scientific config is not exactly released" >&2; exit 2; }
test "$(jq -cS '.execution_release' "$cfs_config")" = "$(jq -cS '.execution_release' "$apparatus_config")" || { echo "remote execution releases differ" >&2; exit 2; }
test ! -e "$run_root" || { echo "immutable CFS-00A run id already used" >&2; exit 2; }
if [ -n "$(squeue -h -u "$(whoami)" -o '%i')" ]; then echo "CFS-00A not submitted: user queue is not empty" >&2; exit 2; fi
node_record=$(scontrol show node worker-1 -o)
node_state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$node_state" in *down*|*drain*|*fail*|*maint*|*not_resp*|*unknown*|*power*) echo "worker-1 unhealthy: $node_state" >&2; exit 2 ;; esac
free_mem=$(printf '%s\n' "$node_record" | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
case "$free_mem" in *[!0-9]*|'') echo "cannot parse worker-1 FreeMem" >&2; exit 2 ;; esac
test "$free_mem" -ge 65536 || { echo "worker-1 FreeMem=${free_mem}MiB < 65536MiB" >&2; exit 2; }
case " $node_record " in *" Gres="*"gpu:nvidia_h100_80gb_hbm3:8"*) ;; *) echo "worker-1 does not advertise H100" >&2; exit 2 ;; esac

mkdir "$run_root"
reservation_tmp=$(mktemp "$run_root/.launch-reservation.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg implementation "$accepted_implementation_commit" --arg free "$free_mem" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"1.0",artifact_role:"r05a_constrained_flow_canary_launch_reservation",status:"preflight_verified_before_queueing",run_id:$run,git_commit:$commit,accepted_implementation_commit:$implementation,release_decision_path:"docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md",single_submission:true,source_node:"worker-1",observed_free_mem_mib:($free|tonumber),requested_gpus:1,requested_cpus:8,requested_host_memory_mib:65536,array:"0-0%1",scientific_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$reservation_tmp"
mv "$reservation_tmp" "$run_root/launch-reservation.json"

gpu_submission=$(RUN_ID="$run_id" MANIFEST="$manifest" CFS_CONFIG="$cfs_config" LEGACY_CONFIG="$legacy_config" R02_RAW_ROOT="$r02_raw_root" CHECKPOINT_DIR="$checkpoint_dir" EXPERIMENT_ROOT="$experiment_root" EXPECTED_GIT_COMMIT="$expected_commit" REMOTE_REPO="$remote_repo" sbatch --parsable --hold --partition=main --account=normal --qos=normal --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --no-requeue --nodelist=worker-1 --array=0-0%1 --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' "$gpu_slurm")
gpu_job_id=${gpu_submission%%;*}
case "$gpu_job_id" in *[!0-9]*|'') echo "invalid GPU job id" >&2; exit 2 ;; esac
printf 'provisional_gpu_job_id=%s\n' "$gpu_job_id" >&2
provisional_gpu_tmp=$(mktemp "$run_root/.provisional-gpu-job-id.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"1.0",artifact_role:"r05a_constrained_flow_canary_provisional_gpu_job_id",status:"sbatch_returned_numeric_id_before_field_validation",run_id:$run,git_commit:$commit,gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),held:true,source_node:"worker-1",automatic_cancellation_allowed:false,scientific_claim_allowed:false,timestamp_utc:$now}' >"$provisional_gpu_tmp"
mv "$provisional_gpu_tmp" "$provisional_gpu_receipt"
gpu_record=$(scontrol show job "$gpu_job_id" -o)
for field in "JobState=PENDING" "Reason=JobHeldUser" "ReqNodeList=worker-1" "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=02:00:00" "Requeue=0"; do case " $gpu_record " in *" $field "*) ;; *) echo "held GPU field changed: $field" >&2; exit 2 ;; esac; done
gpu_req_tres=$(printf '%s\n' "$gpu_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test "$(printf '%s\n' "$gpu_req_tres"|tr ',' '\n'|sed -n 's/^cpu=//p')" = 8
case "$(printf '%s\n' "$gpu_req_tres"|tr ',' '\n'|sed -n 's/^mem=//p')" in 64G|65536M) ;; *) echo "held GPU memory changed" >&2; exit 2 ;; esac
test "$(printf '%s\n' "$gpu_req_tres"|tr ',' '\n'|sed -n 's#^gres/gpu=##p')" = 1

resources=$(jq -n '{partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}')
held_tmp=$(mktemp "$run_root/.held-gpu.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --argjson resources "$resources" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"1.0",artifact_role:"r05a_constrained_flow_canary_held_gpu_submission",status:"sbatch_returned_held_gpu_id",run_id:$run,git_commit:$commit,gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),source_node:"worker-1",resources:$resources,released_at_receipt_time:false,scientific_claim_allowed:false,infeasibility_claim_allowed:false,simulator_efficacy_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$held_tmp"
mv "$held_tmp" "$run_root/held-gpu-submission.json"

repository_hashes='{}'
for relative in "${bound_paths[@]}"; do digest=$(sha256sum "$remote_repo/$relative"|awk '{print $1}'); repository_hashes=$(jq -c --arg key "$relative" --arg value "$digest" '.+{($key):$value}' <<<"$repository_hashes"); done
frozen_bindings=$(jq -n --arg cfs "$expected_cfs_config_sha" --arg legacy "$expected_legacy_config_sha" --arg manifest "$expected_manifest_sha" --arg r02 "$expected_r02_sha" --arg checkpoint "$expected_checkpoint_sha" --arg schema "$expected_schema_sha" --arg adr0040 "$expected_adr0040_sha" --arg adr0041 "$expected_adr0041_sha" '{constrained_flow_config_sha256:$cfs,legacy_scientific_config_sha256:$legacy,manifest_sha256:$manifest,source_r02_sha256:$r02,checkpoint_sha256:$checkpoint,envelope_schema_sha256:$schema,adr0040_sha256:$adr0040,adr0041_sha256:$adr0041,normalization_asset_sha256:"b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",baseline_revision:"57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",historical_inverse_control_sha256:"965082822466774e0a86eeb6f2d178c9e5f3d6d02bca5090c4e1fcc77bc52aa8",pi05_sampler_sha256:"80366dcc7b2ddc598717d4c71c0e68e46312a1e5ffd3fc04479d5599433f4c55",policy_boundary_sha256:"d16767ff2073d5c177cdfcc06dc05dcbf7cdb9ef2a2a0150023f7953b03508b9",ordinary_policy_server_sha256:"eccc0448b4873fd30a1fff3355c5de7a7c227138e6db04b7dcaa5bb38a6a5809",transformers_source_bundle_sha256:"430b00a688e12ff457cdd65929bd164fd001ffa1716dc83589d5388806d2bb33",transformers_replacement_bundle_sha256:"2e1b546bdf42e9872c84734b2d5baf52bd411664685922458e734732c8434098",transformers_overlay_bundle_sha256:"24be8ac6749a4cf7e19c261b14b39e951a499ec61b0602d0badcc4354171d261"}')
artifact_paths=$(jq -n --arg run "$run_root" --arg case "$case_dir" --arg legacy "$legacy_payload" --arg cfs "$cfs_payload" --arg host "$host_telemetry" --arg gpu "$gpu_samples" --arg tests "$test_log" --arg candidate "$candidate" --arg result "$result" --arg receipt "$validation_receipt" '{run_root:$run,case_dir:$case,legacy_payload:$legacy,constrained_flow_payload:$cfs,host_telemetry:$host,gpu_samples:$gpu,allocation_tests_log:$tests,hidden_candidate:$candidate,result:$result,validation_receipt:$receipt}')
source_tmp=$(mktemp "$run_root/.source-contract.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --argjson resources "$resources" --argjson paths "$artifact_paths" --argjson frozen "$frozen_bindings" --argjson hashes "$repository_hashes" --arg held "$run_root/held-gpu-submission.json" --arg held_sha "$(sha256sum "$run_root/held-gpu-submission.json"|awk '{print $1}')" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"1.0",artifact_role:"r05a_constrained_flow_canary_source_contract",status:"gpu_held_sources_bound_before_cpu_submission",run_id:$run,git_commit:$commit,git_dirty:false,source_node:"worker-1",gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),resources:$resources,artifact_paths:$paths,frozen_bindings:$frozen,repository_file_sha256:$hashes,held_gpu_submission_path:$held,held_gpu_submission_sha256:$held_sha,scientific_claim_allowed:false,infeasibility_claim_allowed:false,simulator_efficacy_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$source_tmp"
mv "$source_tmp" "$run_root/source-contract.json"
source_contract_sha=$(sha256sum "$run_root/source-contract.json"|awk '{print $1}')

dependency=afterany:$gpu_job_id
cpu_submission=$(SOURCE_JOB_ID="$gpu_job_id" SOURCE_CONTRACT="$run_root/source-contract.json" EXPECTED_SOURCE_CONTRACT_SHA256="$source_contract_sha" SUBMISSION="$run_root/submission.json" LEGACY_PAYLOAD="$legacy_payload" CONSTRAINED_FLOW_PAYLOAD="$cfs_payload" HOST_TELEMETRY="$host_telemetry" GPU_SAMPLES="$gpu_samples" ALLOCATION_TEST_LOG="$test_log" RESULT="$result" VALIDATION_RECEIPT="$validation_receipt" EXPECTED_GIT_COMMIT="$expected_commit" REMOTE_REPO="$remote_repo" sbatch --parsable --partition=main --account=normal --qos=normal --cpus-per-task=2 --mem=8G --time=00:15:00 --no-requeue --dependency="$dependency" --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' "$cpu_slurm")
cpu_job_id=${cpu_submission%%;*}
case "$cpu_job_id" in *[!0-9]*|'') echo "invalid CPU publisher id" >&2; exit 2 ;; esac
printf 'provisional_cpu_publisher_job_id=%s\n' "$cpu_job_id" >&2
provisional_cpu_tmp=$(mktemp "$run_root/.provisional-cpu-job-id.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg cpu "$cpu_job_id" --arg dependency "$dependency" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"1.0",artifact_role:"r05a_constrained_flow_canary_provisional_cpu_job_id",status:"sbatch_returned_numeric_id_before_field_validation",run_id:$run,git_commit:$commit,gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,cpu_afterany_job_id:$cpu,dependency:$dependency,automatic_cancellation_allowed:false,scientific_claim_allowed:false,timestamp_utc:$now}' >"$provisional_cpu_tmp"
mv "$provisional_cpu_tmp" "$provisional_cpu_receipt"
cpu_record=$(scontrol show job "$cpu_job_id" -o)
for field in "JobState=PENDING" "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=00:15:00" "Requeue=0"; do case " $cpu_record " in *" $field "*) ;; *) echo "CPU publisher field changed: $field" >&2; exit 2 ;; esac; done
case " $cpu_record " in *gres/gpu*) echo "CPU publisher requests GPU" >&2; exit 2 ;; esac
cpu_req_tres=$(printf '%s\n' "$cpu_record"|sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test "$(printf '%s\n' "$cpu_req_tres"|tr ',' '\n'|sed -n 's/^cpu=//p')" = 2
case "$(printf '%s\n' "$cpu_req_tres"|tr ',' '\n'|sed -n 's/^mem=//p')" in 8G|8192M) ;; *) echo "CPU publisher memory changed" >&2; exit 2 ;; esac
observed_dependency=$(printf '%s\n' "$cpu_record"|sed -n 's/.* Dependency=\([^ ]*\).*/\1/p'|sed 's/(unfulfilled)//g; s/_\*//g')
test "$observed_dependency" = "$dependency" || { echo "CPU dependency changed" >&2; exit 2; }

submission_tmp=$(mktemp "$run_root/.submission.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg cpu "$cpu_job_id" --arg dependency "$dependency" --arg source "$run_root/source-contract.json" --arg source_sha "$source_contract_sha" --arg result "$result" --arg receipt "$validation_receipt" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"1.0",artifact_role:"r05a_constrained_flow_canary_atomic_submission",status:"cpu_afterany_registered_gpu_held",run_id:$run,git_commit:$commit,source_node:"worker-1",gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,cpu_afterany_job_id:$cpu,dependency:$dependency,source_contract_path:$source,source_contract_sha256:$source_sha,expected_result:$result,expected_validation_receipt:$receipt,scientific_claim_allowed:false,infeasibility_claim_allowed:false,simulator_efficacy_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$submission_tmp"
mv "$submission_tmp" "$run_root/submission.json"

# Final fingerprint immediately before release.  Any mutation consumes this
# identity while leaving the source job held for exact manual inspection.
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$remote_repo" status --porcelain)"
test "$(git -C "$remote_repo" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$expected_commit"
for relative in "${bound_paths[@]}"; do test "$(sha256sum "$remote_repo/$relative"|awk '{print $1}')" = "$(jq -er --arg path "$relative" '.repository_file_sha256[$path]' "$run_root/source-contract.json")" || { echo "final source fingerprint changed: $relative" >&2; exit 2; }; done
test "$(sha256sum "$run_root/source-contract.json"|awk '{print $1}')" = "$source_contract_sha"
jq -e --arg gpu "$gpu_job_id" --arg cpu "$cpu_job_id" --arg source_sha "$source_contract_sha" '.gpu_slurm_array_job_id==$gpu and .cpu_afterany_job_id==$cpu and .source_contract_sha256==$source_sha' "$run_root/submission.json" >/dev/null

scontrol release "$gpu_job_id"
echo "submitted_gpu_job_id=$gpu_job_id"
echo "submitted_cpu_publisher_job_id=$cpu_job_id"
echo "dependency=$dependency"
echo "source_contract_sha256=$source_contract_sha"
echo "submission_receipt=$run_root/submission.json"
echo "submission_receipt_sha256=$(sha256sum "$run_root/submission.json"|awk '{print $1}')"
REMOTE
