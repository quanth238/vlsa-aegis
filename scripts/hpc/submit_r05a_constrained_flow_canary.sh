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
HISTORICAL_ADR0041_LOCAL=docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md
ADR0045_LOCAL=docs/decisions/0045-accept-runtime-identity-regression.md
ADR0046_LOCAL=docs/decisions/0046-require-fresh-cfs00a-release-bound-to-runtime-evidence.md
RUNTIME_IDENTITY_EVIDENCE_LOCAL=evidence/r05a/runtime-identity-regression-20260716a.json
RUNTIME_IDENTITY_PREFLIGHT_LOCAL=evidence/r05a/runtime-identity-preflight-20260715T221453Z.txt
VINUNI_GUIDE_LOCAL='/Users/quanth238/Library/Mobile Documents/iCloud~md~obsidian/Documents/LLM Knowledge Base/10 Raw/articles/research-infrastructure/2026-05-03 - VinUni H100 Server Guide.md'

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
  docs/decisions/0045-accept-runtime-identity-regression.md
  docs/decisions/0046-require-fresh-cfs00a-release-bound-to-runtime-evidence.md
  evidence/r05a/runtime-identity-regression-20260716a.json
  evidence/r05a/runtime-identity-preflight-20260715T221453Z.txt
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
  scripts/hpc/lib/r05a_runtime_identity.sh
  scripts/hpc/lib/slurm_exact_array_task_status.sh
  scripts/hpc/preflight.sh
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
  tests/test_r05a_runtime_identity_regression_evidence.py
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
test "$(shasum -a 256 "$HISTORICAL_ADR0041_LOCAL" | awk '{print $1}')" = f1906b21d0fc79b44b01d7e7a4bd693835a6489f014a1cb4ea07379d31013f17 || { echo "historical ADR-0041 changed" >&2; exit 2; }
test "$(shasum -a 256 "$ADR0045_LOCAL" | awk '{print $1}')" = 3672cfac46d8ffbd5a224e837e871bdc7010884907f367b6e418471a68c1d98e || { echo "ADR-0045 changed" >&2; exit 2; }
test "$(shasum -a 256 "$RUNTIME_IDENTITY_EVIDENCE_LOCAL" | awk '{print $1}')" = ba83d7d696310456b696ddd0a846e5a6b0554b994204ecf7c568e2b565508237 || { echo "runtime identity evidence changed" >&2; exit 2; }
test "$(shasum -a 256 "$RUNTIME_IDENTITY_PREFLIGHT_LOCAL" | awk '{print $1}')" = c351ec194cf838829e82105f1343de242199e93a855b9ce45862b64a6b955221 || { echo "runtime identity historical preflight changed" >&2; exit 2; }
test -f "$VINUNI_GUIDE_LOCAL" && test ! -L "$VINUNI_GUIDE_LOCAL" || { echo "exact VinUni H100 guide is missing or symlinked" >&2; exit 2; }
test "$(shasum -a 256 "$VINUNI_GUIDE_LOCAL" | awk '{print $1}')" = acee44c535e2fc25f8986e41efe233f21683a71c7fb5fa0ae726f0dae573b108 || { echo "VinUni H100 guide bytes changed; rereview before release" >&2; exit 2; }
test "$(wc -l <"$VINUNI_GUIDE_LOCAL" | tr -d ' ')" = 1298 || { echo "VinUni H100 guide line count changed; rereview before release" >&2; exit 2; }
jq -e '
  .release_commit == "8415b659a46699757de1e99558713e56b95255b5"
  and .source_host == "worker-1"
  and .job.job_id == "28043" and .job.state == "COMPLETED" and .job.exit_code == "0:0"
  and .job.allocated_gpus == 0 and .execution.shell_only == true
  and .execution.openpi_python_executed == false and .execution.libero_python_executed == false
  and .interpretation.cfs_runtime_integration_evaluated == false
  and .interpretation.h100_submission_authorized_by_this_result == false
  and .immutable_artifacts.result.sha256 == "3bda039cd94bf283ebd2b2d9ff1839ce411037a729e6664efaabff20ce38b0de"' "$RUNTIME_IDENTITY_EVIDENCE_LOCAL" >/dev/null || { echo "runtime identity terminal semantics changed" >&2; exit 2; }

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
  and .execution_release.decision_artifact == "docs/decisions/0046-require-fresh-cfs00a-release-bound-to-runtime-evidence.md"
  and .execution_release.accepted_implementation_commit == $implementation
  and .execution_release.run_id == $run and .execution_release.single_submission == true
  and .execution_release.source_host == "worker-1"
  and .execution_release.resources == {partition:"main",account:"normal",qos:"normal",gpus:1,cpus_per_task:8,host_memory_mib:65536,time_limit:"02:00:00",array:"0-0%1",requeue:false,validator_partition:"main",validator_account:"normal",validator_qos:"normal",validator_cpus:2,validator_host_memory_mib:8192,validator_time_limit:"00:15:00",validator_gpus:0,validator_dependency:"afterany"}
  and .execution_release.release_only_parent_required == true
  and .execution_release.allowed_release_diff_paths == ["configs/experiments/r05a_constrained_flow_canary.json","configs/experiments/r05a_constrained_flow_canary_apparatus.json","docs/decisions/0046-require-fresh-cfs00a-release-bound-to-runtime-evidence.md"]
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
jq -e '
  .runtime_identity_contract == {
    validation_helper:"scripts/hpc/lib/r05a_runtime_identity.sh",
    validation_is_shell_only:true,
    interpreter_invocation_during_validation_allowed:false,
    openpi_python:{
      public_path:"/mnt/data/quanth/venvs/openpi/bin/python",
      direct_link_target:"/mnt/data/quanth/anaconda3/bin/python",
      resolved_executable:"/mnt/data/quanth/anaconda3/bin/python3.11",
      resolved_sha256:"c71718900fe84a9124d39abdd9d68d029930e0dcff1764686d8d6aad97216bc9"
    },
    libero_python:{
      public_path:"/mnt/data/quanth/venvs/openpi-libero-client/bin/python",
      direct_link_target:"/home/quanth/.local/share/uv/python/cpython-3.8-linux-x86_64-gnu/bin/python3.8",
      resolved_executable:"/home/quanth/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3.8",
      resolved_sha256:"c70efda0ee43d9a0014ee570cad3abb4f46b0c11f6ea88f7c467a91faafd4f62"
    }
  }' "$APPARATUS_CONFIG_LOCAL" >/dev/null || { echo "apparatus runtime identity changed" >&2; exit 2; }
jq -e '.runtime_identity_evidence_binding == {
  decision_path:"docs/decisions/0045-accept-runtime-identity-regression.md",
  decision_sha256:"3672cfac46d8ffbd5a224e837e871bdc7010884907f367b6e418471a68c1d98e",
  evidence_path:"evidence/r05a/runtime-identity-regression-20260716a.json",
  evidence_sha256:"ba83d7d696310456b696ddd0a846e5a6b0554b994204ecf7c568e2b565508237",
  preflight_path:"evidence/r05a/runtime-identity-preflight-20260715T221453Z.txt",
  preflight_sha256:"c351ec194cf838829e82105f1343de242199e93a855b9ce45862b64a6b955221",
  release_commit:"8415b659a46699757de1e99558713e56b95255b5",job_id:"28043",job_state:"COMPLETED",job_exit_code:"0:0",source_host:"worker-1",
  result_sha256:"3bda039cd94bf283ebd2b2d9ff1839ce411037a729e6664efaabff20ce38b0de",shell_only:true,gpus_allocated:0,
  cfs_runtime_integration_evaluated:false,h100_submission_authorized_by_evidence:false
}' "$APPARATUS_CONFIG_LOCAL" >/dev/null || { echo "apparatus runtime identity evidence binding changed" >&2; exit 2; }
jq -e '.vinuni_h100_guide_contract == {
  title:"2026-05-03 - VinUni H100 Server Guide.md",
  local_reference_path:"/Users/quanth238/Library/Mobile Documents/iCloud~md~obsidian/Documents/LLM Knowledge Base/10 Raw/articles/research-infrastructure/2026-05-03 - VinUni H100 Server Guide.md",
  sha256:"acee44c535e2fc25f8986e41efe233f21683a71c7fb5fa0ae726f0dae573b108",
  line_count:1298,login_node_role:"control_plane_only",allocation_compute_only:true,
  live_preflight_overrides_examples:true,free_h100_required_before_submission:false,
  pending_submission_allowed:true,
  reroute_when_worker_1_busy:false,shared_storage_stop_percent:90
}' "$APPARATUS_CONFIG_LOCAL" >/dev/null || { echo "apparatus VinUni H100 guide contract changed" >&2; exit 2; }

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
test "$RELEASE_DIFF" = "$(printf '%s\n' "$CFS_CONFIG_LOCAL" "$APPARATUS_CONFIG_LOCAL" "$ADR0046_LOCAL")" || { echo "release-only path set changed" >&2; exit 2; }
RELEASE_RESOURCE_JSON=$(jq -cS '.execution_release.resources' "$APPARATUS_CONFIG_LOCAL")
EXPECTED_RELEASE_DECISION=$(mktemp)
cleanup_expected_decision() { rm -f "$EXPECTED_RELEASE_DECISION"; }
trap cleanup_expected_decision EXIT INT TERM
git show "$ACCEPTED_IMPLEMENTATION_COMMIT:$ADR0046_LOCAL" >"$EXPECTED_RELEASE_DECISION"
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
cmp -s "$EXPECTED_RELEASE_DECISION" "$ADR0046_LOCAL" || { echo "release decision is not canonical appendix" >&2; exit 2; }
cleanup_expected_decision
trap - EXIT INT TERM

CFS_CONFIG_SHA256=$(shasum -a 256 "$CFS_CONFIG_LOCAL" | awk '{print $1}')
APPARATUS_CONFIG_SHA256=$(shasum -a 256 "$APPARATUS_CONFIG_LOCAL" | awk '{print $1}')
ENVELOPE_SCHEMA_SHA256=$(shasum -a 256 "$ENVELOPE_SCHEMA_LOCAL" | awk '{print $1}')
ADR0040_SHA256=$(shasum -a 256 "$ADR0040_LOCAL" | awk '{print $1}')
HISTORICAL_ADR0041_SHA256=$(shasum -a 256 "$HISTORICAL_ADR0041_LOCAL" | awk '{print $1}')
ADR0045_SHA256=$(shasum -a 256 "$ADR0045_LOCAL" | awk '{print $1}')
ADR0046_SHA256=$(shasum -a 256 "$ADR0046_LOCAL" | awk '{print $1}')
RUNTIME_IDENTITY_EVIDENCE_SHA256=$(shasum -a 256 "$RUNTIME_IDENTITY_EVIDENCE_LOCAL" | awk '{print $1}')
RUNTIME_IDENTITY_PREFLIGHT_SHA256=$(shasum -a 256 "$RUNTIME_IDENTITY_PREFLIGHT_LOCAL" | awk '{print $1}')

# Login node is control plane only.
mkdir -p .harness/live
FRESH_PREFLIGHT_LOCAL=.harness/live/cfs00a-preflight-$(date -u +%Y%m%dT%H%M%SZ).txt
PREFLIGHT_OUT="$FRESH_PREFLIGHT_LOCAL" scripts/hpc/preflight.sh
FRESH_PREFLIGHT_SHA256=$(shasum -a 256 "$FRESH_PREFLIGHT_LOCAL" | awk '{print $1}')
FRESH_PREFLIGHT_BASE64=$(base64 <"$FRESH_PREFLIGHT_LOCAL" | tr -d '\n')
ssh "$HOST" bash -s -- \
  "$REMOTE_REPO" "$RUN_ID" "$EXPECTED_RELEASE_COMMIT" "$EXPERIMENT_ROOT" "$R02_RAW_ROOT" "$CHECKPOINT_DIR" \
  "$EXPECTED_MANIFEST_SHA256" "$CFS_CONFIG_SHA256" "$EXPECTED_LEGACY_CONFIG_SHA256" "$EXPECTED_R02_SHA256" "$EXPECTED_CHECKPOINT_SHA256" \
  "$APPARATUS_CONFIG_SHA256" "$ENVELOPE_SCHEMA_SHA256" "$ADR0040_SHA256" "$HISTORICAL_ADR0041_SHA256" "$ADR0045_SHA256" "$ADR0046_SHA256" \
  "$RUNTIME_IDENTITY_EVIDENCE_SHA256" "$RUNTIME_IDENTITY_PREFLIGHT_SHA256" "$ACCEPTED_IMPLEMENTATION_COMMIT" \
  "$FRESH_PREFLIGHT_SHA256" "$FRESH_PREFLIGHT_BASE64" <<'REMOTE'
set -euo pipefail

slurm_duration_seconds() {
  local value=${1:?Slurm duration is required}
  local days=0
  local clock=$value
  local first second third
  case "$value" in
    UNLIMITED|INFINITE) printf '%s\n' 9223372036854775807; return 0 ;;
    *-*) days=${value%%-*}; clock=${value#*-} ;;
  esac
  case "$days" in *[!0-9]*|'') return 1 ;; esac
  IFS=: read -r first second third <<<"$clock"
  case "$first" in *[!0-9]*|'') return 1 ;; esac
  if [ -z "${second:-}" ]; then
    printf '%s\n' $((10#$days * 86400 + 10#$first * 60))
    return 0
  fi
  case "$second" in *[!0-9]*|'') return 1 ;; esac
  if [ -z "${third:-}" ]; then
    printf '%s\n' $((10#$days * 86400 + 10#$first * 60 + 10#$second))
    return 0
  fi
  case "$third" in *[!0-9]*|'') return 1 ;; esac
  printf '%s\n' $((10#$days * 86400 + 10#$first * 3600 + 10#$second * 60 + 10#$third))
}

require_two_hour_capacity() {
  local label=${1:?duration label is required}
  local value=${2-}
  local seconds
  [ -z "$value" ] && return 0
  seconds=$(slurm_duration_seconds "$value") || { echo "cannot parse $label duration: $value" >&2; return 2; }
  [ "$seconds" -ge 7200 ] || { echo "$label duration is below two hours: $value" >&2; return 2; }
}

tres_value() {
  local list=${1-}
  local key=${2:?TRES key is required}
  printf '%s\n' "$list" | tr ',' '\n' | awk -F= -v key="$key" '$1==key {print $2; exit}'
}

memory_mib() {
  local value=${1:?memory value is required}
  local number
  case "$value" in
    *T) number=${value%T}; case "$number" in *[!0-9]*|'') return 1 ;; esac; printf '%s\n' $((10#$number * 1024 * 1024)) ;;
    *G) number=${value%G}; case "$number" in *[!0-9]*|'') return 1 ;; esac; printf '%s\n' $((10#$number * 1024)) ;;
    *M) number=${value%M}; case "$number" in *[!0-9]*|'') return 1 ;; esac; printf '%s\n' "$number" ;;
    *K) number=${value%K}; case "$number" in *[!0-9]*|'') return 1 ;; esac; printf '%s\n' $((10#$number / 1024)) ;;
    *) return 1 ;;
  esac
}

require_tres_capacity() {
  local label=${1:?TRES label is required}
  local list=${2-}
  local key=${3:?TRES key is required}
  local minimum=${4:?TRES minimum is required}
  local required=${5:-required}
  local value
  value=$(tres_value "$list" "$key")
  if [ -z "$value" ]; then
    [ "$required" = optional ] && return 0
    echo "$label omits required $key capacity" >&2
    return 2
  fi
  if [ "$key" = mem ]; then
    value=$(memory_mib "$value") || { echo "cannot parse $label $key capacity" >&2; return 2; }
  else
    case "$value" in *[!0-9]*|'') echo "cannot parse $label $key capacity" >&2; return 2 ;; esac
  fi
  [ "$value" -ge "$minimum" ] || { echo "$label $key capacity is below $minimum" >&2; return 2; }
}

remote_repo=$1; run_id=$2; expected_commit=$3; experiment_root=$4; r02_raw_root=$5; checkpoint_dir=$6
expected_manifest_sha=$7; expected_cfs_config_sha=$8; expected_legacy_config_sha=$9; expected_r02_sha=${10}; expected_checkpoint_sha=${11}
expected_apparatus_sha=${12}; expected_schema_sha=${13}; expected_adr0040_sha=${14}; expected_historical_adr0041_sha=${15}; expected_adr0045_sha=${16}; expected_adr0046_sha=${17}
expected_runtime_identity_evidence_sha=${18}; expected_runtime_identity_preflight_sha=${19}; accepted_implementation_commit=${20}
fresh_preflight_sha=${21}; fresh_preflight_base64=${22}

manifest=$remote_repo/manifests/r05a_inverse_flow_teacher_smoke.jsonl
cfs_config=$remote_repo/configs/experiments/r05a_constrained_flow_canary.json
legacy_config=$remote_repo/configs/experiments/r05a_inverse_flow_canary.json
apparatus_config=$remote_repo/configs/experiments/r05a_constrained_flow_canary_apparatus.json
envelope_schema=$remote_repo/schemas/r05a-constrained-flow-canary-envelope.schema.json
adr0040=$remote_repo/docs/decisions/0040-preregister-same-budget-constrained-flow-diagnostic.md
historical_adr0041=$remote_repo/docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md
adr0045=$remote_repo/docs/decisions/0045-accept-runtime-identity-regression.md
adr0046=$remote_repo/docs/decisions/0046-require-fresh-cfs00a-release-bound-to-runtime-evidence.md
runtime_identity_evidence=$remote_repo/evidence/r05a/runtime-identity-regression-20260716a.json
runtime_identity_preflight=$remote_repo/evidence/r05a/runtime-identity-preflight-20260715T221453Z.txt
source_r02=$r02_raw_root/crfs-1069f29a8d76463a/r02-paired.json
checkpoint=$checkpoint_dir/model.safetensors
gpu_slurm=$remote_repo/slurm/r05a_constrained_flow_canary_h100.sbatch
cpu_slurm=$remote_repo/slurm/r05a_constrained_flow_canary_validate_cpu.sbatch
run_root=$experiment_root/$run_id
live_preflight=$run_root/vinuni-preflight.txt
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
  docs/decisions/0045-accept-runtime-identity-regression.md docs/decisions/0046-require-fresh-cfs00a-release-bound-to-runtime-evidence.md
  evidence/r05a/runtime-identity-regression-20260716a.json evidence/r05a/runtime-identity-preflight-20260715T221453Z.txt
  schemas/r05a-inverse-flow-canary.schema.json schemas/r05a-constrained-flow-canary-envelope.schema.json
  main/crfs_oracle/r05a_canary.py main/crfs_oracle/r05a_constrained_flow_allocation_tests.json
  main/crfs_oracle/r05a_constrained_flow_canary.py main/crfs_oracle/r05a_constrained_flow_validation.py main/crfs_oracle/r05a_constrained_flow_publication.py main/crfs_oracle/r05a_full_lifetime_telemetry.py main/crfs_oracle/r05a_sampled_current_canary.py
  main/run_crfs_r05a_canary.py main/run_crfs_r05a_constrained_flow_canary.py main/publish_crfs_r05a_constrained_flow_canary.py
  openpi/scripts/serve_cfs_policy.py openpi/scripts/serve_policy.py openpi/src/openpi/models_pytorch/crfs_inverse_control.py openpi/src/openpi/models_pytorch/crfs_linearized_control.py
  openpi/src/openpi/models_pytorch/pi0_pytorch.py
  openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
  openpi/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/check.py openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py
  openpi/src/openpi/policies/crfs_constrained_flow_adapter.py openpi/src/openpi/policies/policy.py
  scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh scripts/hpc/lib/r05a_allocation_tests.sh scripts/hpc/lib/r05a_runtime_identity.sh scripts/hpc/lib/slurm_exact_array_task_status.sh
  scripts/hpc/preflight.sh scripts/hpc/prepare_jsonschema_overlay.sh scripts/hpc/prepare_transformers_overlay.sh
  scripts/hpc/run_r05a_constrained_flow_workload.sh scripts/hpc/run_r05a_constrained_flow_canary.sh scripts/hpc/validate_r05a_constrained_flow_canary.sh scripts/hpc/submit_r05a_constrained_flow_canary.sh
  slurm/r05a_constrained_flow_canary_h100.sbatch slurm/r05a_constrained_flow_canary_validate_cpu.sbatch
  tests/test_inverse_flow_control.py tests/test_inverse_flow_sampler.py tests/test_inverse_flow_policy.py tests/test_r05a_canary.py
  tests/test_linearized_flow_control.py tests/test_constrained_flow_adapter.py tests/test_r05a_constrained_flow_canary.py tests/test_r05a_constrained_flow_hpc_contract.py tests/test_r05a_constrained_flow_publication.py tests/test_r05a_constrained_flow_validation.py tests/test_r05a_runtime_identity_regression_evidence.py
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
test "$(sha256sum "$historical_adr0041"|awk '{print $1}')" = "$expected_historical_adr0041_sha"
test "$(sha256sum "$adr0045"|awk '{print $1}')" = "$expected_adr0045_sha"
test "$(sha256sum "$adr0046"|awk '{print $1}')" = "$expected_adr0046_sha"
test "$(sha256sum "$runtime_identity_evidence"|awk '{print $1}')" = "$expected_runtime_identity_evidence_sha"
test "$(sha256sum "$runtime_identity_preflight"|awk '{print $1}')" = "$expected_runtime_identity_preflight_sha"
test "$(sha256sum "$source_r02"|awk '{print $1}')" = "$expected_r02_sha"
# The multi-GB checkpoint is deliberately not hashed on the login node.  The
# H100 allocation rehashes its exact bytes before starting the policy server.
test -f "$checkpoint" && test ! -L "$checkpoint" && test -r "$checkpoint" || { echo "canonical checkpoint is missing, symlinked, or unreadable" >&2; exit 2; }
test "$(sha256sum "$remote_repo/openpi/scripts/serve_policy.py"|awk '{print $1}')" = eccc0448b4873fd30a1fff3355c5de7a7c227138e6db04b7dcaa5bb38a6a5809
test "$(jq -er '.execution_release.run_id' "$apparatus_config")" = "$run_id"
test "$(jq -er '.execution_release.accepted_implementation_commit' "$apparatus_config")" = "$accepted_implementation_commit"
jq -e '.config_status == "released_exact_single_canary" and .ready_to_run == true and .blocked_on == [] and .preregistration.h100_submission_authorized == true and (.execution_release|type=="object")' "$cfs_config" >/dev/null || { echo "remote scientific config is not exactly released" >&2; exit 2; }
test "$(jq -cS '.execution_release' "$cfs_config")" = "$(jq -cS '.execution_release' "$apparatus_config")" || { echo "remote execution releases differ" >&2; exit 2; }
jq -e '
  .release_commit == "8415b659a46699757de1e99558713e56b95255b5"
  and .job.job_id == "28043" and .job.state == "COMPLETED" and .job.exit_code == "0:0"
  and .job.allocated_gpus == 0 and .execution.shell_only == true
  and .execution.openpi_python_executed == false and .execution.libero_python_executed == false
  and .interpretation.cfs_runtime_integration_evaluated == false
  and .interpretation.h100_submission_authorized_by_this_result == false
  and .immutable_artifacts.result.sha256 == "3bda039cd94bf283ebd2b2d9ff1839ce411037a729e6664efaabff20ce38b0de"' "$runtime_identity_evidence" >/dev/null || { echo "remote runtime identity evidence changed" >&2; exit 2; }
inherited_sbatch_options=$(env | sed -n 's/^\(SBATCH_[A-Za-z0-9_]*\)=.*/\1/p')
test -z "$inherited_sbatch_options" || { echo "inherited SBATCH options could change the exact transaction: $inherited_sbatch_options" >&2; exit 2; }
test ! -e "$run_root" || { echo "immutable CFS-00A run id already used" >&2; exit 2; }
if [ -n "$(squeue -h -u "$(whoami)" -o '%i')" ]; then echo "CFS-00A not submitted: user queue is not empty" >&2; exit 2; fi
qos_record=$(sacctmgr -n -P show qos normal format=Name,MaxWall,MaxTRESPerJob,MaxTRESPU,GrpTRES,MaxJobsPU,MaxSubmitJobsPU)
qos_line=$(printf '%s\n' "$qos_record" | awk -F'|' '$1=="normal" {print; exit}')
[ -n "$qos_line" ] || { echo "normal QOS record is missing" >&2; exit 2; }
IFS='|' read -r qos_name qos_max_wall qos_per_job qos_per_user qos_group qos_max_jobs qos_max_submit <<EOF
$qos_line
EOF
test "$qos_name" = normal || { echo "normal QOS identity changed" >&2; exit 2; }
require_two_hour_capacity "normal QOS MaxWall" "$qos_max_wall"
for qos_limit in "$qos_per_job" "$qos_per_user" "$qos_group"; do
  require_tres_capacity "normal QOS" "$qos_limit" cpu 8 optional
  require_tres_capacity "normal QOS" "$qos_limit" gres/gpu 1 optional
  require_tres_capacity "normal QOS" "$qos_limit" mem 65536 optional
done
for pair in "MaxJobsPU:$qos_max_jobs" "MaxSubmitJobsPU:$qos_max_submit"; do
  limit_name=${pair%%:*}; limit_value=${pair#*:}
  if [ -n "$limit_value" ]; then
    case "$limit_value" in *[!0-9]*) echo "cannot parse normal QOS $limit_name" >&2; exit 2 ;; esac
    minimum_jobs=1; [ "$limit_name" = MaxSubmitJobsPU ] && minimum_jobs=2
    test "$limit_value" -ge "$minimum_jobs" || { echo "normal QOS $limit_name is below the held-source transaction requirement" >&2; exit 2; }
  fi
done

assoc_record=$(sacctmgr -n -P show assoc user="$(whoami)" format=Cluster,Account,User,Partition,QOS,DefaultQOS,MaxWall,MaxTRESPerJob,MaxTRESPU,GrpTRES,MaxJobs,MaxSubmitJobs)
assoc_line=$(printf '%s\n' "$assoc_record" | awk -F'|' -v user="$(whoami)" '$2=="normal" && $3==user {print; exit}')
[ -n "$assoc_line" ] || { echo "normal account association is missing" >&2; exit 2; }
IFS='|' read -r assoc_cluster assoc_account assoc_user assoc_partition assoc_qos assoc_default_qos assoc_max_wall assoc_per_job assoc_per_user assoc_group assoc_max_jobs assoc_max_submit <<EOF
$assoc_line
EOF
test "$assoc_account" = normal && test "$assoc_user" = "$(whoami)" || { echo "normal account association changed" >&2; exit 2; }
case "$assoc_partition" in ''|main) ;; *) echo "association does not permit main" >&2; exit 2 ;; esac
case ",${assoc_qos},${assoc_default_qos}," in *,normal,*) ;; *) echo "association does not permit normal QOS" >&2; exit 2 ;; esac
require_two_hour_capacity "normal association MaxWall" "$assoc_max_wall"
for assoc_limit in "$assoc_per_job" "$assoc_per_user" "$assoc_group"; do
  require_tres_capacity "normal association" "$assoc_limit" cpu 8 optional
  require_tres_capacity "normal association" "$assoc_limit" gres/gpu 1 optional
  require_tres_capacity "normal association" "$assoc_limit" mem 65536 optional
done
for pair in "MaxJobs:$assoc_max_jobs" "MaxSubmitJobs:$assoc_max_submit"; do
  limit_name=${pair%%:*}; limit_value=${pair#*:}
  if [ -n "$limit_value" ]; then
    case "$limit_value" in *[!0-9]*) echo "cannot parse association $limit_name" >&2; exit 2 ;; esac
    minimum_jobs=1; [ "$limit_name" = MaxSubmitJobs ] && minimum_jobs=2
    test "$limit_value" -ge "$minimum_jobs" || { echo "association $limit_name is below the held-source transaction requirement" >&2; exit 2; }
  fi
done

partition_record=$(scontrol show partition main -o)
case " $partition_record " in *" PartitionName=main "*" State=UP "*) ;; *) echo "main partition is unavailable" >&2; exit 2 ;; esac
partition_max_time=$(printf '%s\n' "$partition_record" | sed -n 's/.* MaxTime=\([^ ]*\).*/\1/p')
require_two_hour_capacity "main partition MaxTime" "$partition_max_time"
for partition_field in AllowAccounts AllowQos; do
  partition_value=$(printf '%s\n' "$partition_record" | sed -n "s/.* ${partition_field}=\([^ ]*\).*/\1/p")
  case ",${partition_value}," in *,ALL,*|*,normal,*) ;; *) echo "main partition $partition_field rejects normal" >&2; exit 2 ;; esac
done
partition_node=$(sinfo -h -p main -n worker-1 -o '%N|%a' | head -n 1)
test "$partition_node" = 'worker-1|up' || { echo "worker-1 is not available in main: $partition_node" >&2; exit 2; }

storage_percent=$(df -P /mnt/data | awk 'NR==2 {value=$5; gsub(/%/,"",value); print value}')
case "$storage_percent" in *[!0-9]*|'') echo "cannot parse /mnt/data utilization" >&2; exit 2 ;; esac
test "$storage_percent" -lt 90 || { echo "/mnt/data is ${storage_percent}% full; stop before submission" >&2; exit 2; }
log_dir=/mnt/data/quanth/slurm_logs/crfs-oracle
test -d "$log_dir" && test -w "$log_dir" || { echo "Slurm log directory is missing or not writable: $log_dir" >&2; exit 2; }
node_record=$(scontrol show node worker-1 -o)
node_state=$(printf '%s\n' "$node_record" | sed -n 's/.* State=\([^ ]*\).*/\1/p' | tr '[:upper:]' '[:lower:]')
case "$node_state" in idle|mixed|idle+dynamic_norm|mixed+dynamic_norm) ;; *) echo "worker-1 is not an authorized idle/mixed state: $node_state" >&2; exit 2 ;; esac
free_mem=$(printf '%s\n' "$node_record" | sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p')
case "$free_mem" in *[!0-9]*|'') echo "cannot parse worker-1 FreeMem" >&2; exit 2 ;; esac
test "$free_mem" -ge 65536 || { echo "worker-1 FreeMem=${free_mem}MiB < 65536MiB" >&2; exit 2; }
gres_value=$(printf '%s\n' "$node_record" | sed -n 's/.* Gres=\([^ ]*\).*/\1/p')
h100_configured=$(printf '%s\n' "$gres_value" | tr ',' '\n' | sed -n 's#^gpu:nvidia_h100_80gb_hbm3:\([0-9][0-9]*\).*#\1#p')
case "$h100_configured" in *[!0-9]*|'') echo "cannot parse worker-1 configured H100 count" >&2; exit 2 ;; esac
cfg_tres=$(printf '%s\n' "$node_record" | sed -n 's/.* CfgTRES=\([^ ]*\).*/\1/p')
alloc_tres=$(printf '%s\n' "$node_record" | sed -n 's/.* AllocTRES=\([^ ]*\).*/\1/p')
configured_gpus=$(printf '%s\n' "$cfg_tres" | tr ',' '\n' | sed -n 's#^gres/gpu=##p')
allocated_gpus=$(printf '%s\n' "$alloc_tres" | tr ',' '\n' | sed -n 's#^gres/gpu=##p')
case "$configured_gpus" in *[!0-9]*|'') echo "cannot parse worker-1 configured GPU count" >&2; exit 2 ;; esac
case "$allocated_gpus" in '') allocated_gpus=0 ;; *[!0-9]*) echo "cannot parse worker-1 allocated GPU count" >&2; exit 2 ;; esac
test "$configured_gpus" = "$h100_configured" || { echo "worker-1 generic GPU count is not exclusively the configured H100 count" >&2; exit 2; }
free_h100=$((configured_gpus - allocated_gpus))
test "$allocated_gpus" -le "$configured_gpus" || { echo "worker-1 allocated GPUs exceed configured GPUs" >&2; exit 2; }
case "$fresh_preflight_sha" in *[!0-9a-f]*|'') echo "invalid fresh preflight digest" >&2; exit 2 ;; esac
test "${#fresh_preflight_sha}" = 64 || { echo "invalid fresh preflight digest length" >&2; exit 2; }
case "$fresh_preflight_base64" in *[!A-Za-z0-9+/=]*|'') echo "invalid fresh preflight encoding" >&2; exit 2 ;; esac

mkdir "$run_root"
preflight_tmp=$(mktemp "$run_root/.vinuni-preflight.XXXXXX")
printf '%s' "$fresh_preflight_base64" | base64 -d >"$preflight_tmp"
test "$(sha256sum "$preflight_tmp" | awk '{print $1}')" = "$fresh_preflight_sha" || { echo "fresh preflight transfer changed" >&2; exit 2; }
mv "$preflight_tmp" "$live_preflight"
reservation_tmp=$(mktemp "$run_root/.launch-reservation.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg implementation "$accepted_implementation_commit" --arg free "$free_mem" --arg free_h100 "$free_h100" --arg preflight "$live_preflight" --arg preflight_sha "$fresh_preflight_sha" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"1.0",artifact_role:"r05a_constrained_flow_canary_launch_reservation",status:"preflight_verified_before_queueing",run_id:$run,git_commit:$commit,accepted_implementation_commit:$implementation,release_decision_path:"docs/decisions/0046-require-fresh-cfs00a-release-bound-to-runtime-evidence.md",single_submission:true,source_node:"worker-1",observed_free_mem_mib:($free|tonumber),observed_unallocated_h100_count:($free_h100|tonumber),immediate_h100_capacity_available:(($free_h100|tonumber) >= 1),pending_submission_allowed:true,fresh_preflight_path:$preflight,fresh_preflight_sha256:$preflight_sha,requested_gpus:1,requested_cpus:8,requested_host_memory_mib:65536,array:"0-0%1",scientific_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$reservation_tmp"
mv "$reservation_tmp" "$run_root/launch-reservation.json"

gpu_submission=$(RUN_ID="$run_id" MANIFEST="$manifest" CFS_CONFIG="$cfs_config" LEGACY_CONFIG="$legacy_config" R02_RAW_ROOT="$r02_raw_root" CHECKPOINT_DIR="$checkpoint_dir" EXPERIMENT_ROOT="$experiment_root" EXPECTED_GIT_COMMIT="$expected_commit" REMOTE_REPO="$remote_repo" sbatch --parsable --hold --export=ALL --partition=main --account=normal --qos=normal --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --no-requeue --nodelist=worker-1 --array=0-0%1 --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%A_%a.out' "$gpu_slurm")
gpu_job_id=${gpu_submission%%;*}
case "$gpu_job_id" in *[!0-9]*|'') echo "invalid GPU job id" >&2; exit 2 ;; esac
printf 'provisional_gpu_job_id=%s\n' "$gpu_job_id" >&2
provisional_gpu_tmp=$(mktemp "$run_root/.provisional-gpu-job-id.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"1.0",artifact_role:"r05a_constrained_flow_canary_provisional_gpu_job_id",status:"sbatch_returned_numeric_id_before_field_validation",run_id:$run,git_commit:$commit,gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),held:true,source_node:"worker-1",automatic_cancellation_allowed:false,scientific_claim_allowed:false,timestamp_utc:$now}' >"$provisional_gpu_tmp"
mv "$provisional_gpu_tmp" "$provisional_gpu_receipt"
gpu_record=$(scontrol show job "${gpu_job_id}_0" -o)
for field in "JobState=PENDING" "Reason=JobHeldUser" "ReqNodeList=worker-1" "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=02:00:00" "Requeue=0"; do case " $gpu_record " in *" $field "*) ;; *) echo "held GPU field changed: $field" >&2; exit 2 ;; esac; done
for field in "ArrayTaskId=0" "ArrayTaskThrottle=1"; do case " $gpu_record " in *" $field "*) ;; *) echo "held GPU array field changed: $field" >&2; exit 2 ;; esac; done
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
frozen_bindings=$(jq -n \
  --arg cfs "$expected_cfs_config_sha" \
  --arg legacy "$expected_legacy_config_sha" \
  --arg manifest "$expected_manifest_sha" \
  --arg r02 "$expected_r02_sha" \
  --arg checkpoint "$expected_checkpoint_sha" \
  --arg schema "$expected_schema_sha" \
  --arg adr0040 "$expected_adr0040_sha" \
  --arg historical_adr0041 "$expected_historical_adr0041_sha" \
  --arg adr0045 "$expected_adr0045_sha" \
  --arg adr0046 "$expected_adr0046_sha" \
  --arg runtime_identity_evidence "$expected_runtime_identity_evidence_sha" \
  --arg runtime_identity_preflight "$expected_runtime_identity_preflight_sha" \
  '{
    constrained_flow_config_sha256:$cfs,
    legacy_scientific_config_sha256:$legacy,
    manifest_sha256:$manifest,
    source_r02_sha256:$r02,
    checkpoint_sha256:$checkpoint,
    envelope_schema_sha256:$schema,
    adr0040_sha256:$adr0040,
    historical_adr0041_sha256:$historical_adr0041,
    adr0045_sha256:$adr0045,
    adr0046_sha256:$adr0046,
    runtime_identity_evidence_sha256:$runtime_identity_evidence,
    runtime_identity_preflight_sha256:$runtime_identity_preflight,
    runtime_identity_release_commit:"8415b659a46699757de1e99558713e56b95255b5",
    runtime_identity_job_id:"28043",
    runtime_identity_result_sha256:"3bda039cd94bf283ebd2b2d9ff1839ce411037a729e6664efaabff20ce38b0de",
    vinuni_h100_guide_sha256:"acee44c535e2fc25f8986e41efe233f21683a71c7fb5fa0ae726f0dae573b108",
    normalization_asset_sha256:"b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
    baseline_revision:"57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
    historical_inverse_control_sha256:"965082822466774e0a86eeb6f2d178c9e5f3d6d02bca5090c4e1fcc77bc52aa8",
    pi05_sampler_sha256:"80366dcc7b2ddc598717d4c71c0e68e46312a1e5ffd3fc04479d5599433f4c55",
    policy_boundary_sha256:"d16767ff2073d5c177cdfcc06dc05dcbf7cdb9ef2a2a0150023f7953b03508b9",
    ordinary_policy_server_sha256:"eccc0448b4873fd30a1fff3355c5de7a7c227138e6db04b7dcaa5bb38a6a5809",
    transformers_source_bundle_sha256:"430b00a688e12ff457cdd65929bd164fd001ffa1716dc83589d5388806d2bb33",
    transformers_replacement_bundle_sha256:"2e1b546bdf42e9872c84734b2d5baf52bd411664685922458e734732c8434098",
    transformers_overlay_bundle_sha256:"24be8ac6749a4cf7e19c261b14b39e951a499ec61b0602d0badcc4354171d261",
    openpi_python_public_path:"/mnt/data/quanth/venvs/openpi/bin/python",
    openpi_python_direct_link_target:"/mnt/data/quanth/anaconda3/bin/python",
    openpi_python_resolved_executable:"/mnt/data/quanth/anaconda3/bin/python3.11",
    openpi_python_resolved_sha256:"c71718900fe84a9124d39abdd9d68d029930e0dcff1764686d8d6aad97216bc9",
    libero_python_public_path:"/mnt/data/quanth/venvs/openpi-libero-client/bin/python",
    libero_python_direct_link_target:"/home/quanth/.local/share/uv/python/cpython-3.8-linux-x86_64-gnu/bin/python3.8",
    libero_python_resolved_executable:"/home/quanth/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3.8",
    libero_python_resolved_sha256:"c70efda0ee43d9a0014ee570cad3abb4f46b0c11f6ea88f7c467a91faafd4f62"
  }')
artifact_paths=$(jq -n --arg run "$run_root" --arg case "$case_dir" --arg legacy "$legacy_payload" --arg cfs "$cfs_payload" --arg host "$host_telemetry" --arg gpu "$gpu_samples" --arg tests "$test_log" --arg candidate "$candidate" --arg result "$result" --arg receipt "$validation_receipt" '{run_root:$run,case_dir:$case,legacy_payload:$legacy,constrained_flow_payload:$cfs,host_telemetry:$host,gpu_samples:$gpu,allocation_tests_log:$tests,hidden_candidate:$candidate,result:$result,validation_receipt:$receipt}')
source_tmp=$(mktemp "$run_root/.source-contract.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --argjson resources "$resources" --argjson paths "$artifact_paths" --argjson frozen "$frozen_bindings" --argjson hashes "$repository_hashes" --arg held "$run_root/held-gpu-submission.json" --arg held_sha "$(sha256sum "$run_root/held-gpu-submission.json"|awk '{print $1}')" --arg preflight "$live_preflight" --arg preflight_sha "$fresh_preflight_sha" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"1.0",artifact_role:"r05a_constrained_flow_canary_source_contract",status:"gpu_held_sources_bound_before_cpu_submission",run_id:$run,git_commit:$commit,git_dirty:false,source_node:"worker-1",gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,exact_gpu_task_id:($gpu+"_0"),resources:$resources,artifact_paths:$paths,frozen_bindings:$frozen,repository_file_sha256:$hashes,held_gpu_submission_path:$held,held_gpu_submission_sha256:$held_sha,live_preflight_path:$preflight,live_preflight_sha256:$preflight_sha,scientific_claim_allowed:false,infeasibility_claim_allowed:false,simulator_efficacy_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$source_tmp"
mv "$source_tmp" "$run_root/source-contract.json"
source_contract_sha=$(sha256sum "$run_root/source-contract.json"|awk '{print $1}')

dependency=afterany:$gpu_job_id
cpu_submission=$(SOURCE_JOB_ID="$gpu_job_id" SOURCE_CONTRACT="$run_root/source-contract.json" EXPECTED_SOURCE_CONTRACT_SHA256="$source_contract_sha" SUBMISSION="$run_root/submission.json" LEGACY_PAYLOAD="$legacy_payload" CONSTRAINED_FLOW_PAYLOAD="$cfs_payload" HOST_TELEMETRY="$host_telemetry" GPU_SAMPLES="$gpu_samples" ALLOCATION_TEST_LOG="$test_log" RESULT="$result" VALIDATION_RECEIPT="$validation_receipt" EXPECTED_GIT_COMMIT="$expected_commit" REMOTE_REPO="$remote_repo" sbatch --parsable --export=ALL --partition=main --account=normal --qos=normal --cpus-per-task=2 --mem=8G --time=00:15:00 --no-requeue --dependency="$dependency" --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' "$cpu_slurm")
cpu_job_id=${cpu_submission%%;*}
case "$cpu_job_id" in *[!0-9]*|'') echo "invalid CPU publisher id" >&2; exit 2 ;; esac
printf 'provisional_cpu_publisher_job_id=%s\n' "$cpu_job_id" >&2
provisional_cpu_tmp=$(mktemp "$run_root/.provisional-cpu-job-id.XXXXXX")
jq -n --arg run "$run_id" --arg commit "$expected_commit" --arg gpu "$gpu_job_id" --arg cpu "$cpu_job_id" --arg dependency "$dependency" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schema_version:"1.0",artifact_role:"r05a_constrained_flow_canary_provisional_cpu_job_id",status:"sbatch_returned_numeric_id_before_field_validation",run_id:$run,git_commit:$commit,gpu_slurm_array_job_id:$gpu,gpu_slurm_array_task_id:0,cpu_afterany_job_id:$cpu,dependency:$dependency,automatic_cancellation_allowed:false,scientific_claim_allowed:false,timestamp_utc:$now}' >"$provisional_cpu_tmp"
mv "$provisional_cpu_tmp" "$provisional_cpu_receipt"
cpu_record=$(scontrol show job "$cpu_job_id" -o)
for field in "JobState=PENDING" "Partition=main" "Account=normal" "QOS=normal" "TimeLimit=00:15:00" "Requeue=0"; do case " $cpu_record " in *" $field "*) ;; *) echo "CPU publisher field changed: $field" >&2; exit 2 ;; esac; done
case " $cpu_record " in *gres/gpu*) echo "CPU publisher requests GPU" >&2; exit 2 ;; esac
case " $cpu_record " in *" ArrayJobId="*|*" ArrayTaskId="*|*" ArrayTaskThrottle="*) echo "CPU publisher unexpectedly became an array" >&2; exit 2 ;; esac
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
test "$(sha256sum "$live_preflight" | awk '{print $1}')" = "$fresh_preflight_sha"
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
