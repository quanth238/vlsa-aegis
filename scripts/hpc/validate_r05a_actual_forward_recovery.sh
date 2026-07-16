#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?AF-00A recovery must run inside Slurm}"
: "${SLURM_JOB_PARTITION:?AF-00A recovery requires partition provenance}"
: "${EXPECTED_RECOVERY_RELEASE_COMMIT:?AF-00A recovery release commit is required}"
: "${RUN_ID:?AF-00A source run id is required}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${EXPERIMENT_ROOT:=/mnt/data/quanth/experiments/crfs-oracle}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"

SOURCE_JOB_ID=28281
ORIGINAL_PUBLISHER_JOB_ID=28282
SOURCE_COMMIT=bd14f97eeffafd20525454db4d7a52614e1146c4
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
CASE_DIR=$RUN_ROOT/crfs-1069f29a8d76463a
RESULT=$CASE_DIR/results.json
ORIGINAL_RECEIPT=$RUN_ROOT/cpu-afterany-validation.json
RECOVERY_RECEIPT=$RUN_ROOT/cpu-republication-validation.json
CONFIG=$REMOTE_REPO/configs/experiments/r05a_actual_forward_canary.json
LEGACY_CONFIG=$REMOTE_REPO/configs/experiments/r05a_inverse_flow_canary.json
RECOVERY_CONFIG=$REMOTE_REPO/configs/experiments/r05a_actual_forward_recovery_20260716c.json
SOURCE_CONTRACT=$RUN_ROOT/source-contract.json
HELD_GPU_SUBMISSION=$RUN_ROOT/held-gpu-submission.json
SUBMISSION=$RUN_ROOT/submission.json
RELEASE_FINGERPRINT=$RUN_ROOT/final-pre-release-fingerprint.json
RECOVERY_SOURCE_CONTRACT=$RUN_ROOT/republication-source-contract.json
RECOVERY_SUBMISSION=$RUN_ROOT/republication-submission.json
RECOVERY_FINGERPRINT=$RUN_ROOT/republication-final-pre-release-fingerprint.json
RECOVERY_FINGERPRINT_SHA_FILE=$RUN_ROOT/republication-final-pre-release-fingerprint.sha256
ARRAY_IDENTITY_HELPER=$REMOTE_REPO/scripts/hpc/lib/slurm_array_task_record_identity.sh
RUNTIME_IDENTITY_HELPER=$REMOTE_REPO/scripts/hpc/lib/r05a_runtime_identity.sh

FAILURE_STAGE=wrapper_preflight
fallback_receipt() {
  status=$?
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ ! -e "$RECOVERY_RECEIPT" ]; then
    temporary=$(mktemp "$RUN_ROOT/.cpu-republication-validation.XXXXXX") || return "$status"
    jq -n --arg publisher "$SLURM_JOB_ID" --arg stage "$FAILURE_STAGE" --arg result "$RESULT" --arg status "$status" \
      '{schema_version:"1.0",artifact_role:"r05a_actual_forward_canary_cpu_republication",source_job_id:"28281",source_task_id:"28281_0",original_publisher_job_id:"28282",recovery_publisher_job_id:$publisher,failure_stage:$stage,result_path:$result,result_sha256:null,published:false,passed:false,errors:["recovery wrapper failed with exit code "+$status],simulator_efficacy_claim_allowed:false,infeasibility_claim_allowed:false,probe_training_authorized:false,automatic_next_gate_authorized:false}' >"$temporary" && mv "$temporary" "$RECOVERY_RECEIPT"
  fi
  return "$status"
}
trap fallback_receipt EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test "$RUN_ID" = r05a-actual-forward-cem-canary-20260716c || { echo "AF-00A recovery run changed" >&2; exit 2; }
test "$SLURM_JOB_PARTITION" = main || { echo "AF-00A recovery partition changed" >&2; exit 2; }
test "${SLURM_CPUS_PER_TASK:-}" = 2 || { echo "AF-00A recovery requires two CPUs" >&2; exit 2; }
test "${SLURM_MEM_PER_NODE:-0}" = 8192 || { echo "AF-00A recovery requires exactly 8 GiB" >&2; exit 2; }
test "${CUDA_VISIBLE_DEVICES:-NoDevFiles}" = NoDevFiles || { echo "AF-00A recovery must not receive a GPU" >&2; exit 2; }
test "$SLURM_JOB_ID" != "$ORIGINAL_PUBLISHER_JOB_ID" || { echo "AF-00A recovery cannot impersonate the failed publisher" >&2; exit 2; }
for path in "$CONFIG" "$LEGACY_CONFIG" "$RECOVERY_CONFIG" "$SOURCE_CONTRACT" "$HELD_GPU_SUBMISSION" "$SUBMISSION" "$RELEASE_FINGERPRINT" "$ORIGINAL_RECEIPT" "$RECOVERY_SOURCE_CONTRACT" "$RECOVERY_SUBMISSION" "$RECOVERY_FINGERPRINT" "$RECOVERY_FINGERPRINT_SHA_FILE" "$ARRAY_IDENTITY_HELPER" "$RUNTIME_IDENTITY_HELPER" "$REMOTE_REPO/main/publish_crfs_r05a_actual_forward_canary.py"; do
  test -f "$path" && test ! -L "$path" || { echo "missing or symlinked AF-00A recovery input: $path" >&2; exit 2; }
done
. "$RUNTIME_IDENTITY_HELPER"
test "$LIBERO_PYTHON" = "$CRFS_R05A_LIBERO_PYTHON" || { echo "noncanonical AF-00A recovery Python launcher" >&2; exit 2; }
crfs_validate_r05a_libero_python "$LIBERO_PYTHON"
test ! -e "$RESULT" && test ! -e "$CASE_DIR/.results.candidate.json" && test ! -e "$RECOVERY_RECEIPT" || { echo "AF-00A recovery publication target is not fresh" >&2; exit 2; }
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_RECOVERY_RELEASE_COMMIT" || { echo "AF-00A recovery checkout changed" >&2; exit 2; }
test -z "$(git -C "$REMOTE_REPO" status --porcelain)" || { echo "AF-00A recovery checkout is dirty" >&2; exit 2; }

while IFS='|' read -r relative expected; do
  test "$(sha256sum "$RUN_ROOT/$relative" | awk '{print $1}')" = "$expected" || { echo "AF-00A run-C artifact changed: $relative" >&2; exit 2; }
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

source_record=$(scontrol show job "${SOURCE_JOB_ID}_0" -o)
original_record=$(scontrol show job "$ORIGINAL_PUBLISHER_JOB_ID" -o)
current_record=$(scontrol show job "$SLURM_JOB_ID" -o)
. "$ARRAY_IDENTITY_HELPER"
crfs_validate_exact_array_task_identity "$source_record" "$SOURCE_JOB_ID" 0 || { echo "AF-00A source task identity changed" >&2; exit 2; }
for field in JobState=COMPLETED ExitCode=0:0 NodeList=worker-1 NumCPUs=8 CPUs/Task=8; do
  case " $source_record " in *" $field "*) ;; *) echo "AF-00A source job changed: $field" >&2; exit 2 ;; esac
done
for field in JobState=FAILED ExitCode=1:0 NumCPUs=2 CPUs/Task=2; do
  case " $original_record " in *" $field "*) ;; *) echo "AF-00A original publisher changed: $field" >&2; exit 2 ;; esac
done
for field in "JobId=$SLURM_JOB_ID" JobState=RUNNING Partition=main Account=normal QOS=normal TimeLimit=00:15:00 Requeue=0 NumNodes=1 NumCPUs=2 CPUs/Task=2; do
  case " $current_record " in *" $field "*) ;; *) echo "AF-00A recovery publisher changed: $field" >&2; exit 2 ;; esac
done
req_tres=$(printf '%s\n' "$current_record" | sed -n 's/.* ReqTRES=\([^ ]*\).*/\1/p')
test "$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= 'NF && $1!="billing" {print $1}' | LC_ALL=C sort)" = "$(printf '%s\n' cpu mem node | LC_ALL=C sort)" || { echo "AF-00A recovery ReqTRES changed" >&2; exit 2; }
test "$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= '$1=="cpu" {print $2}')" = 2 || { echo "AF-00A recovery CPU ReqTRES changed" >&2; exit 2; }
case "$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= '$1=="mem" {print $2}')" in 8G|8192M) ;; *) echo "AF-00A recovery memory ReqTRES changed" >&2; exit 2 ;; esac
test "$(printf '%s\n' "$req_tres" | tr ',' '\n' | awk -F= '$1=="node" {print $2}')" = 1 || { echo "AF-00A recovery node ReqTRES changed" >&2; exit 2; }
case " $current_record " in *" TRESPerNode=gres/gpu:"*) echo "AF-00A recovery received a GPU" >&2; exit 2 ;; esac

RECOVERY_FINGERPRINT_SHA=$(cat "$RECOVERY_FINGERPRINT_SHA_FILE")
test "$(sha256sum "$RECOVERY_FINGERPRINT" | awk '{print $1}')" = "$RECOVERY_FINGERPRINT_SHA" || { echo "AF-00A recovery fingerprint changed" >&2; exit 2; }
RECOVERY_SOURCE_SHA=$(jq -er '.recovery_source_contract_sha256' "$RECOVERY_FINGERPRINT")
RECOVERY_SUBMISSION_SHA=$(jq -er '.recovery_submission_sha256' "$RECOVERY_FINGERPRINT")
test "$(sha256sum "$RECOVERY_SOURCE_CONTRACT" | awk '{print $1}')" = "$RECOVERY_SOURCE_SHA" || { echo "AF-00A recovery source contract changed" >&2; exit 2; }
test "$(sha256sum "$RECOVERY_SUBMISSION" | awk '{print $1}')" = "$RECOVERY_SUBMISSION_SHA" || { echo "AF-00A recovery submission changed" >&2; exit 2; }

FAILURE_STAGE=python_republication
JSONSCHEMA_OVERLAY=$($REMOTE_REPO/scripts/hpc/prepare_jsonschema_overlay.sh)
export PYTHONPATH=$JSONSCHEMA_OVERLAY:$REMOTE_REPO/src:$REMOTE_REPO/main:$REMOTE_REPO/safelibero
"$LIBERO_PYTHON" "$REMOTE_REPO/main/publish_crfs_r05a_actual_forward_canary.py" \
  --run-root "$RUN_ROOT" --config "$CONFIG" --legacy-config "$LEGACY_CONFIG" \
  --output "$RESULT" --receipt "$RECOVERY_RECEIPT" \
  --source-contract "$SOURCE_CONTRACT" --expected-source-contract-sha256 dcd725f0de3797ad24f2445bd347f5a1bcc9d6044f121dc9eeb2370bcdddb0f0 \
  --held-gpu-submission "$HELD_GPU_SUBMISSION" --expected-held-gpu-submission-sha256 ddcebd30f77a95d617b7461af0c34b652fc18bfea51d6c2e00afd6f4d290463d \
  --submission "$SUBMISSION" --expected-submission-sha256 0008bc0e4b55604cf6e708e4300de342aa7b927e765d64f4e90432eb2703a4b0 \
  --release-fingerprint "$RELEASE_FINGERPRINT" --expected-release-fingerprint-sha256 32c9cd064ba59abc90cc279be3989894dc2e69e80de94f0beef3d9bc794e6221 \
  --source-job-id "$SOURCE_JOB_ID" --publisher-job-id "$SLURM_JOB_ID" \
  --recovery-config "$RECOVERY_CONFIG" --original-publisher-job-id "$ORIGINAL_PUBLISHER_JOB_ID" \
  --original-receipt "$ORIGINAL_RECEIPT" --expected-original-receipt-sha256 f8e9ef108630cad9a9691dba0827a8be68d3b15e41ee3de2e4e6cfaf03736d27 \
  --recovery-source-contract "$RECOVERY_SOURCE_CONTRACT" --expected-recovery-source-contract-sha256 "$RECOVERY_SOURCE_SHA" \
  --recovery-submission "$RECOVERY_SUBMISSION" --expected-recovery-submission-sha256 "$RECOVERY_SUBMISSION_SHA" \
  --recovery-release-fingerprint "$RECOVERY_FINGERPRINT" --expected-recovery-release-fingerprint-sha256 "$RECOVERY_FINGERPRINT_SHA"

test -f "$RESULT" && test -f "$RECOVERY_RECEIPT" || { echo "AF-00A recovery omitted publication" >&2; exit 4; }
test "$(sha256sum "$ORIGINAL_RECEIPT" | awk '{print $1}')" = f8e9ef108630cad9a9691dba0827a8be68d3b15e41ee3de2e4e6cfaf03736d27 || { echo "AF-00A recovery changed original receipt" >&2; exit 4; }
echo "result=$RESULT"
echo "result_sha256=$(sha256sum "$RESULT" | awk '{print $1}')"
echo "recovery_receipt=$RECOVERY_RECEIPT"
echo "recovery_receipt_sha256=$(sha256sum "$RECOVERY_RECEIPT" | awk '{print $1}')"
