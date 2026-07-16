#!/usr/bin/env bash
set -euo pipefail

usage='usage: EXPECTED_RECOVERY_RELEASE_COMMIT=... submit_r05a_actual_forward_recovery.sh'
: "${EXPECTED_RECOVERY_RELEASE_COMMIT:?$usage}"
case "$EXPECTED_RECOVERY_RELEASE_COMMIT" in *[!0-9a-f]*|'') echo "invalid AF-00A recovery release commit" >&2; exit 2 ;; esac
test "${#EXPECTED_RECOVERY_RELEASE_COMMIT}" = 40 || { echo "AF-00A recovery release commit must have 40 hex characters" >&2; exit 2; }

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
HOST=${VINUNI_HOST:-vinuni}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-crfs}
CONFIG=configs/experiments/r05a_actual_forward_recovery_20260716c.json
RUN_ID=r05a-actual-forward-cem-canary-20260716c

for path in "$CONFIG" main/publish_crfs_r05a_actual_forward_canary.py schemas/r05a-actual-forward-recovery-publication-receipt.schema.json scripts/hpc/validate_r05a_actual_forward_recovery.sh scripts/hpc/submit_r05a_actual_forward_recovery.sh slurm/r05a_actual_forward_recovery_cpu.sbatch; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked AF-00A recovery source: $path" >&2; exit 2; }
done
jq -e '
  .config_status == "released_exact_cpu_republication"
  and .ready_to_run == true and .blocked_on == []
  and (.execution_release | type == "object")
  and .source_binding.run_id == "r05a-actual-forward-cem-canary-20260716c"
  and .source_binding.source_git_commit == "bd14f97eeffafd20525454db4d7a52614e1146c4"
  and .source_binding.source_job_id == "28281"
  and .source_binding.original_publisher_job_id == "28282"
  and .resources == {partition:"main",account:"normal",qos:"normal",cpus_per_task:2,host_memory_mib:8192,time_limit:"00:15:00",gpus:0,requeue:false,dependency:"afterany:28282"}
  and .claims == {scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,automatic_next_gate_authorized:false}
' "$CONFIG" >/dev/null || { echo "AF-00A recovery is implemented but not released" >&2; exit 2; }
test -z "$(git status --porcelain)" || { echo "AF-00A recovery requires a clean tree" >&2; exit 2; }
test "$(git rev-parse HEAD)" = "$EXPECTED_RECOVERY_RELEASE_COMMIT" || { echo "local AF-00A recovery HEAD differs" >&2; exit 2; }
test "$(git rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$EXPECTED_RECOVERY_RELEASE_COMMIT" || { echo "origin AF-00A recovery release differs" >&2; exit 2; }
IMPLEMENTATION_COMMIT=$(jq -er '.execution_release.accepted_implementation_commit' "$CONFIG")
test "$(git rev-list --parents -n 1 "$EXPECTED_RECOVERY_RELEASE_COMMIT")" = "$EXPECTED_RECOVERY_RELEASE_COMMIT $IMPLEMENTATION_COMMIT" || { echo "AF-00A recovery release is not a direct implementation child" >&2; exit 2; }
RELEASE_ADR=$(jq -er '.execution_release.decision_artifact' "$CONFIG")
test -f "$RELEASE_ADR" && test ! -L "$RELEASE_ADR" || { echo "AF-00A recovery release ADR is missing" >&2; exit 2; }
EXPECTED_DIFF=$(printf '%s\n' "$CONFIG" "$RELEASE_ADR" | LC_ALL=C sort)
test "$(git diff --name-only "$IMPLEMENTATION_COMMIT" "$EXPECTED_RECOVERY_RELEASE_COMMIT" | LC_ALL=C sort)" = "$EXPECTED_DIFF" || { echo "AF-00A recovery release diff changed" >&2; exit 2; }
PARENT_CONFIG=$(git show "$IMPLEMENTATION_COMMIT:$CONFIG")
jq -e '.ready_to_run == false and (.blocked_on | length > 0) and .execution_release == null' <<<"$PARENT_CONFIG" >/dev/null || { echo "AF-00A recovery implementation was not fail closed" >&2; exit 2; }
test "$(jq -cS 'del(.config_status,.ready_to_run,.blocked_on,.execution_release)' <<<"$PARENT_CONFIG")" = "$(jq -cS 'del(.config_status,.ready_to_run,.blocked_on,.execution_release)' "$CONFIG")" || { echo "AF-00A recovery release changed frozen content" >&2; exit 2; }

scripts/hpc/preflight.sh

ssh "$HOST" bash -s -- "$REMOTE_REPO" "$EXPECTED_RECOVERY_RELEASE_COMMIT" "$IMPLEMENTATION_COMMIT" <<'REMOTE'
set -euo pipefail
remote_repo=$1
release_commit=$2
implementation_commit=$3
run_id=r05a-actual-forward-cem-canary-20260716c
run_root=/mnt/data/quanth/experiments/crfs-oracle/$run_id
case_dir=$run_root/crfs-1069f29a8d76463a
config=$remote_repo/configs/experiments/r05a_actual_forward_recovery_20260716c.json
sbatch_file=$remote_repo/slurm/r05a_actual_forward_recovery_cpu.sbatch
provisional=$run_root/republication-provisional-cpu-job-id.json
source_contract=$run_root/republication-source-contract.json
submission=$run_root/republication-submission.json
fingerprint=$run_root/republication-final-pre-release-fingerprint.json
fingerprint_sha_file=$run_root/republication-final-pre-release-fingerprint.sha256
cpu_record_path=$run_root/republication-held-cpu-job-record.txt
source_record_path=$run_root/republication-source-terminal-job-record.txt
original_record_path=$run_root/republication-original-publisher-terminal-job-record.txt
transaction_lock=$run_root/republication-transaction-lock
result=$case_dir/results.json
original_receipt=$run_root/cpu-afterany-validation.json
recovery_receipt=$run_root/cpu-republication-validation.json

test "$(git -C "$remote_repo" rev-parse HEAD)" = "$release_commit" || { echo "remote AF-00A recovery commit differs" >&2; exit 2; }
test -z "$(git -C "$remote_repo" status --porcelain)" || { echo "remote AF-00A recovery tree is dirty" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-parse refs/remotes/origin/agent/crfs-oracle-harness)" = "$release_commit" || { echo "remote origin AF-00A recovery differs" >&2; exit 2; }
test "$(git -C "$remote_repo" rev-list --parents -n 1 "$release_commit")" = "$release_commit $implementation_commit" || { echo "remote AF-00A recovery ancestry differs" >&2; exit 2; }
for path in "$provisional" "$source_contract" "$submission" "$fingerprint" "$fingerprint_sha_file" "$cpu_record_path" "$source_record_path" "$original_record_path" "$transaction_lock" "$result" "$recovery_receipt" "$case_dir/.results.candidate.json"; do
  test ! -e "$path" || { echo "AF-00A recovery namespace is already consumed: $path" >&2; exit 2; }
done

while IFS='|' read -r relative expected; do
  test -f "$run_root/$relative" && test ! -L "$run_root/$relative" || { echo "missing immutable AF-00A run-C artifact: $relative" >&2; exit 2; }
  test "$(sha256sum "$run_root/$relative" | awk '{print $1}')" = "$expected" || { echo "immutable AF-00A run-C artifact changed: $relative" >&2; exit 2; }
done <<'EOF'
source-contract.json|dcd725f0de3797ad24f2445bd347f5a1bcc9d6044f121dc9eeb2370bcdddb0f0
held-gpu-submission.json|ddcebd30f77a95d617b7461af0c34b652fc18bfea51d6c2e00afd6f4d290463d
submission.json|0008bc0e4b55604cf6e708e4300de342aa7b927e765d64f4e90432eb2703a4b0
final-pre-release-fingerprint.json|32c9cd064ba59abc90cc279be3989894dc2e69e80de94f0beef3d9bc794e6221
cpu-afterany-validation.json|f8e9ef108630cad9a9691dba0827a8be68d3b15e41ee3de2e4e6cfaf03736d27
crfs-1069f29a8d76463a/af00a-raw-payload.json|00433437470ca7678a236c4778d5cf68160c8d3299f128b64985b1aeb142d5d1
crfs-1069f29a8d76463a/af00a-tensors.npz|ada53973653ba7c21dab33cdbc4b4498c7b2b1840f2fc284bd3d3c76baf1d383
crfs-1069f29a8d76463a/query-ledger.json|619f15ad46390d6e55e74365dc2530e58b383592ec998842d4bbd898c7266b92
crfs-1069f29a8d76463a/host-cgroup-sampled-current.tsv|277d80ac479f6db3115fe7eb110a7b324708831a82efc846b72fed5a1fd36804
crfs-1069f29a8d76463a/gpu-memory-samples.csv|2f75d0a1299104e6c2d93cf98a6c3bac13d4872ad8f3574a680cac633bbde3b1
crfs-1069f29a8d76463a/allocation-focused-tests.log|60a3eb4ae4b62f8895ade6dbd203ff49e2f30bed4c543dbbe3e975c078eb9566
vinuni-preflight.txt|57d64964aec997c97bfe8f2598e7f028f605c6b1b5b1b41ef22d1ce629e6f789
EOF

source_record=$(scontrol show job 28281_0 -o)
original_record=$(scontrol show job 28282 -o)
. "$remote_repo/scripts/hpc/lib/slurm_array_task_record_identity.sh"
crfs_validate_exact_array_task_identity "$source_record" 28281 0 || { echo "AF-00A source task identity changed" >&2; exit 2; }
for field in JobState=COMPLETED ExitCode=0:0 NodeList=worker-1; do case " $source_record " in *" $field "*) ;; *) echo "AF-00A source terminal state changed: $field" >&2; exit 2 ;; esac; done
for field in JobState=FAILED ExitCode=1:0; do case " $original_record " in *" $field "*) ;; *) echo "AF-00A original publisher state changed: $field" >&2; exit 2 ;; esac; done
mkdir "$transaction_lock" || { echo "AF-00A recovery transaction was concurrently consumed" >&2; exit 2; }
source_record_tmp=$(mktemp "$run_root/.republication-source-terminal-job-record.XXXXXX")
original_record_tmp=$(mktemp "$run_root/.republication-original-publisher-terminal-job-record.XXXXXX")
printf '%s\n' "$source_record" >"$source_record_tmp"
printf '%s\n' "$original_record" >"$original_record_tmp"
mv "$source_record_tmp" "$source_record_path"
mv "$original_record_tmp" "$original_record_path"

repository_hashes='{}'
while IFS= read -r relative; do
  test -f "$remote_repo/$relative" && test ! -L "$remote_repo/$relative" || { echo "missing recovery source: $relative" >&2; exit 2; }
  digest=$(sha256sum "$remote_repo/$relative" | awk '{print $1}')
  repository_hashes=$(jq -c --arg key "$relative" --arg value "$digest" '. + {($key):$value}' <<<"$repository_hashes")
done < <(jq -er '.recovery_repository_paths[]' "$config")
decision_artifact=$(jq -er '.execution_release.decision_artifact' "$config")
test -f "$remote_repo/$decision_artifact" && test ! -L "$remote_repo/$decision_artifact" || { echo "missing recovery release decision: $decision_artifact" >&2; exit 2; }
decision_digest=$(sha256sum "$remote_repo/$decision_artifact" | awk '{print $1}')
repository_hashes=$(jq -c --arg key "$decision_artifact" --arg value "$decision_digest" '. + {($key):$value}' <<<"$repository_hashes")

dependency=afterany:28282
cpu_submission=$(RUN_ID="$run_id" EXPECTED_RECOVERY_RELEASE_COMMIT="$release_commit" REMOTE_REPO="$remote_repo" \
  sbatch --parsable --hold --export=ALL --partition=main --account=normal --qos=normal \
  --cpus-per-task=2 --mem=8G --time=00:15:00 --no-requeue --dependency="$dependency" \
  --output='/mnt/data/quanth/slurm_logs/crfs-oracle/%x-%j.out' "$sbatch_file")
cpu_job_id=${cpu_submission%%;*}
case "$cpu_job_id" in *[!0-9]*|'') echo "invalid AF-00A recovery CPU job id" >&2; exit 2 ;; esac
test "$cpu_job_id" != 28282 || { echo "AF-00A recovery reused the failed publisher id" >&2; exit 2; }
provisional_tmp=$(mktemp "$run_root/.republication-provisional-cpu-job-id.XXXXXX")
jq -n --arg run "$run_id" --arg source_commit bd14f97eeffafd20525454db4d7a52614e1146c4 --arg recovery_commit "$release_commit" --arg cpu "$cpu_job_id" --arg dependency "$dependency" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_republication_provisional_cpu_job_id",status:"sbatch_returned_held_numeric_id_before_inspection",run_id:$run,source_git_commit:$source_commit,recovery_git_commit:$recovery_commit,source_job_id:"28281",source_task_id:"28281_0",original_publisher_job_id:"28282",recovery_publisher_job_id:$cpu,dependency:$dependency,automatic_cancellation_allowed:false,scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,timestamp_utc:$now}' >"$provisional_tmp"
mv "$provisional_tmp" "$provisional"

cpu_record=$(scontrol show job "$cpu_job_id" -o)
for field in "JobId=$cpu_job_id" JobState=PENDING Reason=JobHeldUser Partition=main Account=normal QOS=normal TimeLimit=00:15:00 Requeue=0 NumNodes=1 NumCPUs=2 CPUs/Task=2; do case " $cpu_record " in *" $field "*) ;; *) echo "held AF-00A recovery CPU changed: $field" >&2; exit 2 ;; esac; done
req_tres=$(printf '%s\n' "$cpu_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test "$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= 'NF && $1!="billing" {print $1}' | LC_ALL=C sort)" = "$(printf '%s\n' cpu mem node | LC_ALL=C sort)" || { echo "held AF-00A recovery CPU ReqTRES changed" >&2; exit 2; }
case " $cpu_record " in *" TRESPerNode=gres/gpu:"*) echo "held AF-00A recovery requested a GPU" >&2; exit 2 ;; esac
cpu_record_tmp=$(mktemp "$run_root/.republication-held-cpu-job-record.XXXXXX")
printf '%s\n' "$cpu_record" >"$cpu_record_tmp"
mv "$cpu_record_tmp" "$cpu_record_path"

source_tmp=$(mktemp "$run_root/.republication-source-contract.XXXXXX")
jq -n --arg run "$run_id" --arg source_commit bd14f97eeffafd20525454db4d7a52614e1146c4 --arg recovery_commit "$release_commit" --arg cpu "$cpu_job_id" --arg dependency "$dependency" --arg original_receipt "$original_receipt" --arg original_receipt_sha f8e9ef108630cad9a9691dba0827a8be68d3b15e41ee3de2e4e6cfaf03736d27 --arg source_record "$source_record_path" --arg source_record_sha "$(sha256sum "$source_record_path" | awk '{print $1}')" --arg original_record "$original_record_path" --arg original_record_sha "$(sha256sum "$original_record_path" | awk '{print $1}')" --argjson hashes "$repository_hashes" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_republication_source_contract",run_id:$run,source_git_commit:$source_commit,recovery_git_commit:$recovery_commit,source_job_id:"28281",source_task_id:"28281_0",original_publisher_job_id:"28282",recovery_publisher_job_id:$cpu,dependency:$dependency,resources:{partition:"main",account:"normal",qos:"normal",cpus_per_task:2,host_memory_mib:8192,time_limit:"00:15:00",gpus:0,requeue:false,dependency:$dependency},original_receipt_path:$original_receipt,original_receipt_sha256:$original_receipt_sha,terminal_job_records:{source:{path:$source_record,sha256:$source_record_sha},original_publisher:{path:$original_record,sha256:$original_record_sha}},repository_file_sha256:$hashes,scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false}' >"$source_tmp"
mv "$source_tmp" "$source_contract"
source_sha=$(sha256sum "$source_contract" | awk '{print $1}')

submission_tmp=$(mktemp "$run_root/.republication-submission.XXXXXX")
jq -n --arg run "$run_id" --arg source_commit bd14f97eeffafd20525454db4d7a52614e1146c4 --arg recovery_commit "$release_commit" --arg cpu "$cpu_job_id" --arg dependency "$dependency" --arg source "$source_contract" --arg source_sha "$source_sha" --arg result "$result" --arg receipt "$recovery_receipt" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_republication_submission",run_id:$run,source_git_commit:$source_commit,recovery_git_commit:$recovery_commit,source_job_id:"28281",source_task_id:"28281_0",original_publisher_job_id:"28282",recovery_publisher_job_id:$cpu,dependency:$dependency,recovery_source_contract_path:$source,recovery_source_contract_sha256:$source_sha,expected_result:$result,expected_recovery_receipt:$receipt,scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false}' >"$submission_tmp"
mv "$submission_tmp" "$submission"
submission_sha=$(sha256sum "$submission" | awk '{print $1}')

fingerprint_tmp=$(mktemp "$run_root/.republication-final-pre-release-fingerprint.XXXXXX")
jq -n --arg run "$run_id" --arg source_commit bd14f97eeffafd20525454db4d7a52614e1146c4 --arg recovery_commit "$release_commit" --arg cpu "$cpu_job_id" --arg dependency "$dependency" --arg source "$source_contract" --arg source_sha "$source_sha" --arg submission "$submission" --arg submission_sha "$submission_sha" --arg provisional "$provisional" --arg provisional_sha "$(sha256sum "$provisional" | awk '{print $1}')" --arg cpu_record "$cpu_record_path" --arg cpu_record_sha "$(sha256sum "$cpu_record_path" | awk '{print $1}')" --arg source_record "$source_record_path" --arg source_record_sha "$(sha256sum "$source_record_path" | awk '{print $1}')" --arg original_record "$original_record_path" --arg original_record_sha "$(sha256sum "$original_record_path" | awk '{print $1}')" --arg result "$result" --arg receipt "$recovery_receipt" \
  '{schema_version:"1.0",artifact_role:"r05a_actual_forward_republication_final_pre_release_fingerprint",run_id:$run,source_git_commit:$source_commit,recovery_git_commit:$recovery_commit,source_job_id:"28281",source_task_id:"28281_0",original_publisher_job_id:"28282",recovery_publisher_job_id:$cpu,dependency:$dependency,recovery_source_contract_path:$source,recovery_source_contract_sha256:$source_sha,recovery_submission_path:$submission,recovery_submission_sha256:$submission_sha,expected_result:$result,expected_recovery_receipt:$receipt,bindings:{provisional_cpu_job_id:{path:$provisional,sha256:$provisional_sha},held_cpu_job_record:{path:$cpu_record,sha256:$cpu_record_sha},source_terminal_job_record:{path:$source_record,sha256:$source_record_sha},original_publisher_terminal_job_record:{path:$original_record,sha256:$original_record_sha}},scientific_claim_allowed:false,simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false}' >"$fingerprint_tmp"
mv "$fingerprint_tmp" "$fingerprint"
fingerprint_sha=$(sha256sum "$fingerprint" | awk '{print $1}')
fingerprint_sha_tmp=$(mktemp "$run_root/.republication-final-pre-release-fingerprint-sha256.XXXXXX")
printf '%s\n' "$fingerprint_sha" >"$fingerprint_sha_tmp"
mv "$fingerprint_sha_tmp" "$fingerprint_sha_file"

test "$(sha256sum "$original_receipt" | awk '{print $1}')" = f8e9ef108630cad9a9691dba0827a8be68d3b15e41ee3de2e4e6cfaf03736d27
test "$(git -C "$remote_repo" rev-parse HEAD)" = "$release_commit"
test -z "$(git -C "$remote_repo" status --porcelain)"
scontrol release "$cpu_job_id"
echo "submitted_recovery_cpu_job_id=$cpu_job_id"
echo "dependency=$dependency"
echo "recovery_source_contract_sha256=$source_sha"
echo "recovery_submission_sha256=$submission_sha"
echo "recovery_fingerprint_sha256=$fingerprint_sha"
REMOTE
