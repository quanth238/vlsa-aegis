#!/usr/bin/env bash
set -euo pipefail

# Exact shell-only control-plane launcher for the postpublication analysis-v2
# job.  The terminal publisher may be a finalize-only recovery job while the
# immutable summary and prepublish artifacts remain owned by timed-out job
# 28940.  The CPU job is submitted held, inspected, immutably receipted,
# rechecked, and only then released.  Reruns recover the one exact job instead
# of submitting another.

readonly POPULATION_ARRAY_JOB_ID=28609
readonly PUBLISHER_JOB_ID=${PUBLISHER_JOB_ID:-28610}
readonly ARTIFACT_PUBLISHER_JOB_ID=${ARTIFACT_PUBLISHER_JOB_ID:-$PUBLISHER_JOB_ID}
readonly TIMEOUT_ARTIFACT_PUBLISHER_JOB_ID=28940
readonly POPULATION_RUN_ID=vlsa-table1-contact-authority-population-20260718a
readonly POPULATION_SOURCE_GIT_COMMIT=1592aa59361f431ba96c6ddcbebcb596f6c20853
readonly EXPECTED_JOB_NAME=vlsa-a2-p${PUBLISHER_JOB_ID}
readonly EXPECTED_REMOTE_REPO=/home/quanth/working_space/vlsa-aegis-table-repro
readonly EXPECTED_EXPERIMENT_ROOT=/mnt/data/quanth/experiments/vlsa-aegis-table1
readonly EXPECTED_OUTPUT_ROOT=/mnt/data/quanth/experiments/vlsa-aegis-table1-analysis-v2

: "${EXPECTED_PUBLICATION_RECEIPT_SHA256:?set only after the terminal publisher completes}"
: "${EXPECTED_ANALYSIS_GIT_COMMIT:?set the exact reviewed analysis release commit}"
: "${EXPECTED_BUILDER_SHA256:?set the reviewed analysis-v2 builder SHA-256}"
: "${EXPECTED_RUNNER_SHA256:?set the reviewed allocation runner SHA-256}"
: "${EXPECTED_SBATCH_SHA256:?set the reviewed SBatch SHA-256}"
: "${EXPECTED_SUBMIT_HELPER_SHA256:?set the reviewed login helper SHA-256}"
: "${EXPECTED_CONFIG_SHA256:?set the reviewed protocol config SHA-256}"
: "${EXPECTED_MANIFEST_SHA256:?set the reviewed population manifest SHA-256}"
: "${EXPECTED_MANIFEST_RECEIPT_SHA256:?set the reviewed manifest receipt SHA-256}"
: "${EXPECTED_V1_SUMMARY_SHA256:?set the terminal publisher summary SHA-256}"
: "${EXPECTED_PREPUBLISH_RECEIPT_SHA256:?set the terminal prepublish receipt SHA-256}"

RUN_ID=${RUN_ID:-$POPULATION_RUN_ID}
EXPECTED_SOURCE_GIT_COMMIT=${EXPECTED_SOURCE_GIT_COMMIT:-$POPULATION_SOURCE_GIT_COMMIT}
REMOTE_REPO=${REMOTE_REPO:-$EXPECTED_REMOTE_REPO}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-$EXPECTED_EXPERIMENT_ROOT}
RUN_ROOT=${RUN_ROOT:-$EXPERIMENT_ROOT/$RUN_ID}
RESULTS_ROOT=${RESULTS_ROOT:-$RUN_ROOT/tasks}
PUBLICATION_RECEIPT=${PUBLICATION_RECEIPT:-$RUN_ROOT/population-publication-receipt.json}
V1_SUMMARY_PATH=${V1_SUMMARY_PATH:-$RUN_ROOT/publication-attempts/job-$ARTIFACT_PUBLISHER_JOB_ID/population-summary.json}
PREPUBLISH_RECEIPT_PATH=${PREPUBLISH_RECEIPT_PATH:-$RUN_ROOT/publication-attempts/job-$ARTIFACT_PUBLISHER_JOB_ID/prepublish-validation.json}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_table1_translational.json}
MANIFEST_PATH=${MANIFEST_PATH:-$REMOTE_REPO/manifests/vlsa_table1_population.jsonl}
MANIFEST_RECEIPT_PATH=${MANIFEST_RECEIPT_PATH:-$REMOTE_REPO/manifests/vlsa_table1_population.receipt.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-$EXPECTED_OUTPUT_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID}
BUILDER=$REMOTE_REPO/analysis/build_safelibero_postpublication_v2.py
RUNNER=$REMOTE_REPO/slurm/run_vlsa_postpublication_analysis_v2.sh
SBATCH=$REMOTE_REPO/slurm/vlsa_postpublication_analysis_v2.sbatch
SUBMIT_HELPER=$REMOTE_REPO/scripts/submit_vlsa_postpublication_analysis_v2_28610.sh

CONTROL_ROOT=$OUTPUT_ROOT/.analysis-v2-control
CONTROL_DIR=$CONTROL_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID
INTENT_PATH=$CONTROL_DIR/submission-intent.tsv
INTENT_SHA=$CONTROL_DIR/submission-intent.sha256
JOB_ID_PATH=$CONTROL_DIR/job-id.txt
JOB_ID_SHA=$CONTROL_DIR/job-id.sha256
SCONTROL_PATH=$CONTROL_DIR/held-job-scontrol.txt
SCONTROL_SHA=$CONTROL_DIR/held-job-scontrol.sha256
SUBMISSION_RECEIPT=$CONTROL_DIR/submission-receipt.tsv
SUBMISSION_RECEIPT_SHA=$CONTROL_DIR/submission-receipt.sha256
RELEASE_RECEIPT=$CONTROL_DIR/release-receipt.tsv
RELEASE_RECEIPT_SHA=$CONTROL_DIR/release-receipt.sha256

die() {
  echo "postpublication analysis-v2 launch rejected: $*" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_scalar() {
  local label=$1
  local value=$2
  case "$value" in
    *$'\n'*|*$'\r'*|*$'\t'*) die "$label contains a control delimiter" ;;
  esac
}

require_export_scalar() {
  local label=$1
  local value=$2
  require_scalar "$label" "$value"
  case "$value" in
    *,*) die "$label contains a comma and cannot be exported safely" ;;
  esac
}

require_regular_file() {
  local label=$1
  local path=$2
  [[ -f "$path" && ! -L "$path" ]] || \
    die "$label is missing, non-regular, or symlinked: $path"
}

require_file_hash() {
  local label=$1
  local path=$2
  local expected=$3
  require_regular_file "$label" "$path"
  [[ "$(sha256_file "$path")" == "$expected" ]] || \
    die "$label SHA-256 changed"
}

publish_exact_file() {
  local destination=$1
  local temporary
  temporary=$(mktemp "$CONTROL_DIR/.immutable.XXXXXX")
  umask 077
  if ! command cat >"$temporary"; then
    rm -f "$temporary"
    die "cannot stage immutable file $destination"
  fi
  chmod 0444 "$temporary"
  sync -f "$temporary"
  if [[ -e "$destination" || -L "$destination" ]]; then
    if [[ -L "$destination" || ! -f "$destination" ]] || \
      ! cmp -s "$temporary" "$destination"; then
      rm -f "$temporary"
      die "immutable file differs: $destination"
    fi
    rm -f "$temporary"
    return
  fi
  if ! ln "$temporary" "$destination"; then
    rm -f "$temporary"
    die "cannot atomically publish immutable file $destination"
  fi
  rm -f "$temporary"
  sync -f "$CONTROL_DIR"
}

publish_receipt_sha256() {
  local receipt=$1
  local sidecar=$2
  printf '%s  %s\n' "$(sha256_file "$receipt")" "$(basename "$receipt")" |
    publish_exact_file "$sidecar"
}

verify_receipt_sha256() {
  local receipt=$1
  local sidecar=$2
  require_regular_file immutable_receipt "$receipt"
  require_regular_file immutable_receipt_sidecar "$sidecar"
  (
    cd "$(dirname "$receipt")"
    sha256sum --check --status "$(basename "$sidecar")"
  ) || die "immutable receipt hash differs: $receipt"
}

tsv_field() {
  local path=$1
  local key=$2
  awk -F'\t' -v key="$key" \
    '$1 == key {print substr($0, length($1) + 2); exit}' "$path"
}

scontrol_field() {
  local path=$1
  local key=$2
  awk -v key="$key" '{
    for (field_index = 1; field_index <= NF; field_index++) {
      prefix = key "="
      if (substr($field_index, 1, length(prefix)) == prefix) {
        print substr($field_index, length(prefix) + 1)
        exit
      }
    }
  }' "$path"
}

require_tsv_field() {
  local path=$1
  local key=$2
  local expected=$3
  local observed
  observed=$(tsv_field "$path" "$key")
  [[ "$observed" == "$expected" ]] || \
    die "immutable receipt $key differs: observed=$observed expected=$expected"
}

require_scontrol_field() {
  local path=$1
  local key=$2
  local expected=$3
  local observed
  observed=$(scontrol_field "$path" "$key")
  [[ "$observed" == "$expected" ]] || \
    die "held job $key differs: observed=$observed expected=$expected"
}

require_output_unused() {
  [[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || \
    die "analysis-v2 output directory already exists"
}

validate_frozen_inputs() {
  [[ -d "$REMOTE_REPO/.git" || -f "$REMOTE_REPO/.git" ]] || \
    die "reviewed repository is missing"
  [[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == \
    "$EXPECTED_ANALYSIS_GIT_COMMIT" ]] || \
    die "repository is not at the reviewed analysis release commit"
  [[ -z "$(git -C "$REMOTE_REPO" status --short --untracked-files=all)" ]] || \
    die "reviewed repository is dirty"
  [[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || \
    die "immutable run root is missing or symlinked"
  [[ -d "$RESULTS_ROOT" && ! -L "$RESULTS_ROOT" ]] || \
    die "immutable results root is missing or symlinked"
  [[ -d "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || \
    die "analysis output root must already exist as a real directory"
  require_file_hash builder "$BUILDER" "$EXPECTED_BUILDER_SHA256"
  require_file_hash runner "$RUNNER" "$EXPECTED_RUNNER_SHA256"
  require_file_hash SBatch "$SBATCH" "$EXPECTED_SBATCH_SHA256"
  require_file_hash submit_helper "$SUBMIT_HELPER" \
    "$EXPECTED_SUBMIT_HELPER_SHA256"
  require_file_hash publication_receipt "$PUBLICATION_RECEIPT" \
    "$EXPECTED_PUBLICATION_RECEIPT_SHA256"
  require_file_hash v1_summary "$V1_SUMMARY_PATH" \
    "$EXPECTED_V1_SUMMARY_SHA256"
  require_file_hash prepublish_receipt "$PREPUBLISH_RECEIPT_PATH" \
    "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"
  require_file_hash config "$CONFIG_PATH" "$EXPECTED_CONFIG_SHA256"
  require_file_hash manifest "$MANIFEST_PATH" "$EXPECTED_MANIFEST_SHA256"
  require_file_hash manifest_receipt "$MANIFEST_RECEIPT_PATH" \
    "$EXPECTED_MANIFEST_RECEIPT_SHA256"

  local publisher_rows
  publisher_rows=$(
    sacct -n -X -j "$PUBLISHER_JOB_ID" \
      --format=JobIDRaw,State,ExitCode -P |
      awk -F'|' -v job="$PUBLISHER_JOB_ID" \
        '$1 == job {print $2 "|" $3}'
  )
  [[ "$publisher_rows" == "COMPLETED|0:0" ]] || \
    die "publisher $PUBLISHER_JOB_ID is not exactly COMPLETED with exit 0:0"
}

inspect_held_job() {
  local job_id=$1
  local temporary
  temporary=$(mktemp "$CONTROL_DIR/.scontrol.XXXXXX")
  if ! scontrol show job -dd -o "$job_id" >"$temporary"; then
    rm -f "$temporary"
    die "cannot inspect exact held job $job_id"
  fi
  require_scontrol_field "$temporary" JobId "$job_id"
  require_scontrol_field "$temporary" JobName "$EXPECTED_JOB_NAME"
  require_scontrol_field "$temporary" JobState PENDING
  require_scontrol_field "$temporary" Reason JobHeldUser
  local observed_dependency
  observed_dependency=$(scontrol_field "$temporary" Dependency)
  [[ "${observed_dependency%%(*}" == "afterok:$PUBLISHER_JOB_ID" ]] || {
    rm -f "$temporary"
    die "held job dependency differs: $observed_dependency"
  }
  require_scontrol_field "$temporary" Partition main
  require_scontrol_field "$temporary" Account normal
  require_scontrol_field "$temporary" QOS normal
  require_scontrol_field "$temporary" NumNodes 1
  require_scontrol_field "$temporary" NumCPUs 4
  require_scontrol_field "$temporary" NumTasks 1
  require_scontrol_field "$temporary" "CPUs/Task" 4
  require_scontrol_field "$temporary" MinMemoryNode 32G
  require_scontrol_field "$temporary" TimeLimit 04:00:00
  require_scontrol_field "$temporary" Requeue 0
  require_scontrol_field "$temporary" ExcNodeList worker-3
  require_scontrol_field "$temporary" Command "$SBATCH"
  require_scontrol_field "$temporary" WorkDir "$REMOTE_REPO"
  require_scontrol_field "$temporary" StdOut \
    "/mnt/data/quanth/slurm_logs/$EXPECTED_JOB_NAME-$job_id.out"
  local resource_field
  for resource_field in ReqTRES AllocTRES TresPerNode TresPerTask Gres; do
    case "$(scontrol_field "$temporary" "$resource_field")" in
      *[Gg][Pp][Uu]*)
        rm -f "$temporary"
        die "held job unexpectedly requests a GPU in $resource_field"
        ;;
    esac
  done

  if [[ -e "$SCONTROL_PATH" || -L "$SCONTROL_PATH" ]]; then
    verify_receipt_sha256 "$SCONTROL_PATH" "$SCONTROL_SHA"
    local field
    for field in JobId JobName JobState Reason Partition Account QOS \
      NumNodes NumCPUs NumTasks "CPUs/Task" MinMemoryNode TimeLimit \
      Requeue ExcNodeList Command WorkDir StdOut ReqTRES AllocTRES \
      TresPerNode TresPerTask Gres; do
      [[ "$(scontrol_field "$SCONTROL_PATH" "$field")" == \
        "$(scontrol_field "$temporary" "$field")" ]] || {
        rm -f "$temporary"
        die "held-job recorded $field differs during recovery"
      }
    done
    [[ "$(scontrol_field "$SCONTROL_PATH" Dependency)" == \
      "$observed_dependency" ]] || {
      rm -f "$temporary"
      die "held-job recorded dependency differs during recovery"
    }
    rm -f "$temporary"
  else
    chmod 0444 "$temporary"
    sync -f "$temporary"
    if ! ln "$temporary" "$SCONTROL_PATH"; then
      rm -f "$temporary"
      die "cannot atomically publish held-job scontrol snapshot"
    fi
    rm -f "$temporary"
    sync -f "$CONTROL_DIR"
    publish_receipt_sha256 "$SCONTROL_PATH" "$SCONTROL_SHA"
  fi
}

validate_submission_receipt() {
  local job_id=$1
  verify_receipt_sha256 "$INTENT_PATH" "$INTENT_SHA"
  verify_receipt_sha256 "$JOB_ID_PATH" "$JOB_ID_SHA"
  verify_receipt_sha256 "$SCONTROL_PATH" "$SCONTROL_SHA"
  verify_receipt_sha256 "$SUBMISSION_RECEIPT" "$SUBMISSION_RECEIPT_SHA"
  [[ "$(tr -d '\n' <"$JOB_ID_PATH")" == "$job_id" ]] || \
    die "immutable job ID record differs"
  require_tsv_field "$SUBMISSION_RECEIPT" schema_version \
    vlsa_postpublication_analysis_v2_submission_receipt.v1
  require_tsv_field "$SUBMISSION_RECEIPT" status held_validated
  require_tsv_field "$SUBMISSION_RECEIPT" job_id "$job_id"
  require_tsv_field "$SUBMISSION_RECEIPT" job_name "$EXPECTED_JOB_NAME"
  require_tsv_field "$SUBMISSION_RECEIPT" population_array_job_id \
    "$POPULATION_ARRAY_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" publisher_job_id "$PUBLISHER_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" artifact_publisher_job_id \
    "$ARTIFACT_PUBLISHER_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" dependency \
    "afterok:$PUBLISHER_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" source_git_commit \
    "$EXPECTED_SOURCE_GIT_COMMIT"
  require_tsv_field "$SUBMISSION_RECEIPT" analysis_git_commit \
    "$EXPECTED_ANALYSIS_GIT_COMMIT"
  require_tsv_field "$SUBMISSION_RECEIPT" publication_receipt_sha256 \
    "$EXPECTED_PUBLICATION_RECEIPT_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" v1_summary_sha256 \
    "$EXPECTED_V1_SUMMARY_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" prepublish_receipt_sha256 \
    "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" builder_sha256 \
    "$EXPECTED_BUILDER_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" runner_sha256 \
    "$EXPECTED_RUNNER_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" sbatch_sha256 \
    "$EXPECTED_SBATCH_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" submit_helper_sha256 \
    "$EXPECTED_SUBMIT_HELPER_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" intent_sha256 \
    "$(sha256_file "$INTENT_PATH")"
  require_tsv_field "$SUBMISSION_RECEIPT" scontrol_snapshot_sha256 \
    "$(sha256_file "$SCONTROL_PATH")"
  require_tsv_field "$SUBMISSION_RECEIPT" output_unused true
  require_tsv_field "$SUBMISSION_RECEIPT" cpu_only true
  require_tsv_field "$SUBMISSION_RECEIPT" held_before_release true
}

validate_release_receipt() {
  local job_id=$1
  validate_submission_receipt "$job_id"
  verify_receipt_sha256 "$RELEASE_RECEIPT" "$RELEASE_RECEIPT_SHA"
  require_tsv_field "$RELEASE_RECEIPT" schema_version \
    vlsa_postpublication_analysis_v2_release_receipt.v1
  require_tsv_field "$RELEASE_RECEIPT" status released
  require_tsv_field "$RELEASE_RECEIPT" job_id "$job_id"
  require_tsv_field "$RELEASE_RECEIPT" job_name "$EXPECTED_JOB_NAME"
  require_tsv_field "$RELEASE_RECEIPT" population_array_job_id \
    "$POPULATION_ARRAY_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" publisher_job_id "$PUBLISHER_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" artifact_publisher_job_id \
    "$ARTIFACT_PUBLISHER_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" dependency \
    "afterok:$PUBLISHER_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" submission_receipt_sha256 \
    "$(sha256_file "$SUBMISSION_RECEIPT")"
  require_tsv_field "$RELEASE_RECEIPT" output_unused_before_release true
  case "$(tsv_field "$RELEASE_RECEIPT" release_action)" in
    released_exact_held_job|recovered_after_prior_release) ;;
    *) die "release receipt action is invalid" ;;
  esac
  case "$(tsv_field "$RELEASE_RECEIPT" observed_reason_after_action)" in
    *Held*) die "release receipt retains a held state" ;;
  esac
}

for value in "$EXPECTED_SOURCE_GIT_COMMIT" "$EXPECTED_ANALYSIS_GIT_COMMIT"; do
  [[ "$value" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is not 40 hex"
done
[[ "$PUBLISHER_JOB_ID" =~ ^[0-9]+$ ]] || \
  die "terminal publisher job ID is not numeric"
[[ "$ARTIFACT_PUBLISHER_JOB_ID" =~ ^[0-9]+$ ]] || \
  die "artifact publisher job ID is not numeric"
if [[ "$ARTIFACT_PUBLISHER_JOB_ID" != "$PUBLISHER_JOB_ID" ]]; then
  [[ "$ARTIFACT_PUBLISHER_JOB_ID" == \
    "$TIMEOUT_ARTIFACT_PUBLISHER_JOB_ID" ]] || \
    die "split publication is allowed only for timed-out artifact publisher 28940"
fi
for value in \
  "$EXPECTED_PUBLICATION_RECEIPT_SHA256" \
  "$EXPECTED_BUILDER_SHA256" \
  "$EXPECTED_RUNNER_SHA256" \
  "$EXPECTED_SBATCH_SHA256" \
  "$EXPECTED_SUBMIT_HELPER_SHA256" \
  "$EXPECTED_CONFIG_SHA256" \
  "$EXPECTED_MANIFEST_SHA256" \
  "$EXPECTED_MANIFEST_RECEIPT_SHA256" \
  "$EXPECTED_V1_SUMMARY_SHA256" \
  "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die "expected SHA-256 is not 64 hex"
done
[[ "$RUN_ID" == "$POPULATION_RUN_ID" ]] || \
  die "run ID differs from the exact population"
[[ "$EXPECTED_SOURCE_GIT_COMMIT" == "$POPULATION_SOURCE_GIT_COMMIT" ]] || \
  die "source commit differs from the exact population"
[[ "$REMOTE_REPO" == "$EXPECTED_REMOTE_REPO" ]] || \
  die "repository path differs from the reviewed release"
[[ "$EXPERIMENT_ROOT" == "$EXPECTED_EXPERIMENT_ROOT" ]] || \
  die "experiment root differs from the exact population"
[[ "$RUN_ROOT" == "$EXPERIMENT_ROOT/$POPULATION_RUN_ID" ]] || \
  die "run root differs from the exact population"
[[ "$RESULTS_ROOT" == "$RUN_ROOT/tasks" ]] || \
  die "results root differs from the immutable task tree"
[[ "$PUBLICATION_RECEIPT" == \
  "$RUN_ROOT/population-publication-receipt.json" ]] || \
  die "publication receipt path differs from the exact population"
[[ "$V1_SUMMARY_PATH" == \
  "$RUN_ROOT/publication-attempts/job-$ARTIFACT_PUBLISHER_JOB_ID/population-summary.json" ]] || \
  die "v1 summary path differs from the exact publisher"
[[ "$PREPUBLISH_RECEIPT_PATH" == \
  "$RUN_ROOT/publication-attempts/job-$ARTIFACT_PUBLISHER_JOB_ID/prepublish-validation.json" ]] || \
  die "prepublish receipt path differs from the exact publisher"
[[ "$CONFIG_PATH" == "$REMOTE_REPO/configs/vlsa_table1_translational.json" ]] || \
  die "config path differs from the reviewed protocol"
[[ "$MANIFEST_PATH" == "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" ]] || \
  die "manifest path differs from the immutable population"
[[ "$MANIFEST_RECEIPT_PATH" == \
  "$REMOTE_REPO/manifests/vlsa_table1_population.receipt.json" ]] || \
  die "manifest receipt path differs from the immutable population"
[[ "$OUTPUT_ROOT" == "$EXPECTED_OUTPUT_ROOT" ]] || \
  die "output root differs from the reviewed destination"
[[ "$OUTPUT_DIR" == \
  "$OUTPUT_ROOT/$POPULATION_RUN_ID-publisher-$PUBLISHER_JOB_ID" ]] || \
  die "output directory differs from the reviewed destination"

for name in PUBLISHER_JOB_ID ARTIFACT_PUBLISHER_JOB_ID \
  RUN_ID EXPECTED_SOURCE_GIT_COMMIT \
  EXPECTED_PUBLICATION_RECEIPT_SHA256 EXPECTED_ANALYSIS_GIT_COMMIT \
  EXPECTED_BUILDER_SHA256 EXPECTED_RUNNER_SHA256 \
  EXPECTED_SBATCH_SHA256 EXPECTED_CONFIG_SHA256 \
  EXPECTED_MANIFEST_SHA256 EXPECTED_MANIFEST_RECEIPT_SHA256 \
  EXPECTED_V1_SUMMARY_SHA256 EXPECTED_PREPUBLISH_RECEIPT_SHA256 \
  REMOTE_REPO EXPERIMENT_ROOT RUN_ROOT RESULTS_ROOT \
  PUBLICATION_RECEIPT V1_SUMMARY_PATH PREPUBLISH_RECEIPT_PATH \
  CONFIG_PATH MANIFEST_PATH MANIFEST_RECEIPT_PATH OUTPUT_ROOT OUTPUT_DIR; do
  require_export_scalar "$name" "${!name}"
done
require_scalar EXPECTED_JOB_NAME "$EXPECTED_JOB_NAME"

# The receipt hash and all publisher-derived descriptors are accepted only
# after this exact accounting gate passes.
validate_frozen_inputs

if [[ ! -e "$CONTROL_ROOT" && ! -L "$CONTROL_ROOT" ]]; then
  mkdir "$CONTROL_ROOT"
fi
[[ -d "$CONTROL_ROOT" && ! -L "$CONTROL_ROOT" ]] || \
  die "analysis control root is not a real directory"
if [[ ! -e "$CONTROL_DIR" && ! -L "$CONTROL_DIR" ]]; then
  mkdir "$CONTROL_DIR"
fi
[[ -d "$CONTROL_DIR" && ! -L "$CONTROL_DIR" ]] || \
  die "analysis control directory is not a real directory"
[[ ! -L "$CONTROL_DIR/.lock" ]] || \
  die "analysis control lock must not be a symlink"
exec 9>"$CONTROL_DIR/.lock"
flock -n 9 || die "another exact postpublication analysis launch is active"

{
  printf 'schema_version\tvlsa_postpublication_analysis_v2_submission_intent.v1\n'
  printf 'status\tfrozen\n'
  printf 'run_id\t%s\n' "$RUN_ID"
  printf 'population_array_job_id\t%s\n' "$POPULATION_ARRAY_JOB_ID"
  printf 'publisher_job_id\t%s\n' "$PUBLISHER_JOB_ID"
  printf 'artifact_publisher_job_id\t%s\n' "$ARTIFACT_PUBLISHER_JOB_ID"
  printf 'dependency\tafterok:%s\n' "$PUBLISHER_JOB_ID"
  printf 'job_name\t%s\n' "$EXPECTED_JOB_NAME"
  printf 'population_source_git_commit\t%s\n' "$EXPECTED_SOURCE_GIT_COMMIT"
  printf 'analysis_git_commit\t%s\n' "$EXPECTED_ANALYSIS_GIT_COMMIT"
  printf 'publication_receipt\t%s\n' "$PUBLICATION_RECEIPT"
  printf 'publication_receipt_sha256\t%s\n' "$EXPECTED_PUBLICATION_RECEIPT_SHA256"
  printf 'v1_summary\t%s\n' "$V1_SUMMARY_PATH"
  printf 'v1_summary_sha256\t%s\n' "$EXPECTED_V1_SUMMARY_SHA256"
  printf 'prepublish_receipt\t%s\n' "$PREPUBLISH_RECEIPT_PATH"
  printf 'prepublish_receipt_sha256\t%s\n' "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"
  printf 'config\t%s\n' "$CONFIG_PATH"
  printf 'config_sha256\t%s\n' "$EXPECTED_CONFIG_SHA256"
  printf 'manifest\t%s\n' "$MANIFEST_PATH"
  printf 'manifest_sha256\t%s\n' "$EXPECTED_MANIFEST_SHA256"
  printf 'manifest_receipt\t%s\n' "$MANIFEST_RECEIPT_PATH"
  printf 'manifest_receipt_sha256\t%s\n' "$EXPECTED_MANIFEST_RECEIPT_SHA256"
  printf 'results_root\t%s\n' "$RESULTS_ROOT"
  printf 'builder_sha256\t%s\n' "$EXPECTED_BUILDER_SHA256"
  printf 'runner_sha256\t%s\n' "$EXPECTED_RUNNER_SHA256"
  printf 'sbatch_sha256\t%s\n' "$EXPECTED_SBATCH_SHA256"
  printf 'submit_helper_sha256\t%s\n' "$EXPECTED_SUBMIT_HELPER_SHA256"
  printf 'output_dir\t%s\n' "$OUTPUT_DIR"
  printf 'resources\tpartition=main,account=normal,qos=normal,nodes=1,ntasks=1,cpus=4,mem=32G,time=04:00:00,gpus=0,exclude=worker-3\n'
} | publish_exact_file "$INTENT_PATH"
publish_receipt_sha256 "$INTENT_PATH" "$INTENT_SHA"
verify_receipt_sha256 "$INTENT_PATH" "$INTENT_SHA"

if [[ -f "$RELEASE_RECEIPT" || -L "$RELEASE_RECEIPT" ]]; then
  existing_job_id=$(tsv_field "$RELEASE_RECEIPT" job_id)
  [[ "$existing_job_id" =~ ^[0-9]+$ ]] || \
    die "release receipt has an invalid job ID"
  validate_release_receipt "$existing_job_id"
  printf '%s\n' "$existing_job_id"
  exit 0
fi

job_id=
if [[ -f "$JOB_ID_PATH" && ! -L "$JOB_ID_PATH" ]]; then
  verify_receipt_sha256 "$JOB_ID_PATH" "$JOB_ID_SHA"
  job_id=$(tr -d '\n' <"$JOB_ID_PATH")
else
  recovered_jobs=$(
    squeue --noheader --user="${USER:?USER is required}" \
      --name="$EXPECTED_JOB_NAME" --format='%A' |
      awk 'NF {print $1}'
  )
  recovered_count=$(
    printf '%s\n' "$recovered_jobs" | awk 'NF {count++} END {print count+0}'
  )
  (( recovered_count <= 1 )) || \
    die "multiple active jobs have the exact recovery job name"
  if (( recovered_count == 1 )); then
    job_id=$(printf '%s\n' "$recovered_jobs" | awk 'NF {print; exit}')
  else
    require_output_unused
    export_spec="RUN_ID=$RUN_ID"
    for name in PUBLISHER_JOB_ID ARTIFACT_PUBLISHER_JOB_ID \
      EXPECTED_SOURCE_GIT_COMMIT \
      EXPECTED_PUBLICATION_RECEIPT_SHA256 EXPECTED_ANALYSIS_GIT_COMMIT \
      EXPECTED_BUILDER_SHA256 EXPECTED_RUNNER_SHA256 \
      EXPECTED_SBATCH_SHA256 EXPECTED_CONFIG_SHA256 \
      EXPECTED_MANIFEST_SHA256 EXPECTED_MANIFEST_RECEIPT_SHA256 \
      EXPECTED_V1_SUMMARY_SHA256 EXPECTED_PREPUBLISH_RECEIPT_SHA256 \
      REMOTE_REPO EXPERIMENT_ROOT RUN_ROOT RESULTS_ROOT \
      PUBLICATION_RECEIPT V1_SUMMARY_PATH PREPUBLISH_RECEIPT_PATH \
      CONFIG_PATH MANIFEST_PATH MANIFEST_RECEIPT_PATH OUTPUT_ROOT OUTPUT_DIR; do
      export_spec+=",${name}=${!name}"
    done
    submission=$(
      sbatch --parsable --hold \
        --job-name="$EXPECTED_JOB_NAME" \
        --dependency="afterok:$PUBLISHER_JOB_ID" \
        --chdir="$REMOTE_REPO" \
        --export="$export_spec" \
        "$SBATCH"
    )
    job_id=${submission%%;*}
  fi
  [[ "$job_id" =~ ^[0-9]+$ ]] || die "sbatch returned an invalid job ID"
  printf '%s\n' "$job_id" | publish_exact_file "$JOB_ID_PATH"
  publish_receipt_sha256 "$JOB_ID_PATH" "$JOB_ID_SHA"
fi
[[ "$job_id" =~ ^[0-9]+$ ]] || die "recorded job ID is invalid"
verify_receipt_sha256 "$JOB_ID_PATH" "$JOB_ID_SHA"

if [[ ! -f "$SUBMISSION_RECEIPT" && ! -L "$SUBMISSION_RECEIPT" ]]; then
  require_output_unused
  inspect_held_job "$job_id"
  {
    printf 'schema_version\tvlsa_postpublication_analysis_v2_submission_receipt.v1\n'
    printf 'status\theld_validated\n'
    printf 'job_id\t%s\n' "$job_id"
    printf 'job_name\t%s\n' "$EXPECTED_JOB_NAME"
    printf 'population_array_job_id\t%s\n' "$POPULATION_ARRAY_JOB_ID"
    printf 'publisher_job_id\t%s\n' "$PUBLISHER_JOB_ID"
    printf 'artifact_publisher_job_id\t%s\n' "$ARTIFACT_PUBLISHER_JOB_ID"
    printf 'dependency\tafterok:%s\n' "$PUBLISHER_JOB_ID"
    printf 'source_git_commit\t%s\n' "$EXPECTED_SOURCE_GIT_COMMIT"
    printf 'analysis_git_commit\t%s\n' "$EXPECTED_ANALYSIS_GIT_COMMIT"
    printf 'publication_receipt_sha256\t%s\n' "$EXPECTED_PUBLICATION_RECEIPT_SHA256"
    printf 'v1_summary_sha256\t%s\n' "$EXPECTED_V1_SUMMARY_SHA256"
    printf 'prepublish_receipt_sha256\t%s\n' "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"
    printf 'builder_sha256\t%s\n' "$EXPECTED_BUILDER_SHA256"
    printf 'runner_sha256\t%s\n' "$EXPECTED_RUNNER_SHA256"
    printf 'sbatch_sha256\t%s\n' "$EXPECTED_SBATCH_SHA256"
    printf 'submit_helper_sha256\t%s\n' "$EXPECTED_SUBMIT_HELPER_SHA256"
    printf 'intent_sha256\t%s\n' "$(sha256_file "$INTENT_PATH")"
    printf 'scontrol_snapshot_sha256\t%s\n' "$(sha256_file "$SCONTROL_PATH")"
    printf 'output_unused\ttrue\n'
    printf 'cpu_only\ttrue\n'
    printf 'held_before_release\ttrue\n'
  } | publish_exact_file "$SUBMISSION_RECEIPT"
  publish_receipt_sha256 "$SUBMISSION_RECEIPT" "$SUBMISSION_RECEIPT_SHA"
else
  validate_submission_receipt "$job_id"
fi

validate_frozen_inputs
validate_submission_receipt "$job_id"
submission_receipt_sha256=$(sha256_file "$SUBMISSION_RECEIPT")

current_snapshot=$(scontrol show job -dd -o "$job_id" 2>/dev/null || true)
current_state=
current_reason=
if [[ -n "$current_snapshot" ]]; then
  current_state=$(
    printf '%s\n' "$current_snapshot" |
      awk '{for(i=1;i<=NF;i++) if($i ~ /^JobState=/) {sub(/^JobState=/,"",$i); print $i; exit}}'
  )
  current_reason=$(
    printf '%s\n' "$current_snapshot" |
      awk '{for(i=1;i<=NF;i++) if($i ~ /^Reason=/) {sub(/^Reason=/,"",$i); print $i; exit}}'
  )
fi
release_action=
if [[ "$current_state" == PENDING && "$current_reason" == JobHeldUser ]]; then
  require_output_unused
  inspect_held_job "$job_id"
  scontrol release "$job_id"
  release_action=released_exact_held_job
else
  if [[ -z "$current_state" ]]; then
    current_state=$(
      sacct -n -X -j "$job_id" --format=JobIDRaw,State -P |
        awk -F'|' -v job="$job_id" '$1 == job {print $2; exit}'
    )
  fi
  case "$current_state" in
    PENDING)
      case "$current_reason" in
        *Held*) die "exact job remains held for reason $current_reason" ;;
      esac
      release_action=recovered_after_prior_release
      ;;
    CONFIGURING|RUNNING|COMPLETING|COMPLETED|FAILED|TIMEOUT|\
OUT_OF_MEMORY|CANCELLED|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
      release_action=recovered_after_prior_release
      ;;
    *)
      die "exact job is neither held nor observably released: $current_state/$current_reason"
      ;;
  esac
fi

post_action_snapshot=$(scontrol show job -dd -o "$job_id" 2>/dev/null || true)
post_action_state=
post_action_reason=
if [[ -n "$post_action_snapshot" ]]; then
  post_action_state=$(
    printf '%s\n' "$post_action_snapshot" |
      awk '{for(i=1;i<=NF;i++) if($i ~ /^JobState=/) {sub(/^JobState=/,"",$i); print $i; exit}}'
  )
  post_action_reason=$(
    printf '%s\n' "$post_action_snapshot" |
      awk '{for(i=1;i<=NF;i++) if($i ~ /^Reason=/) {sub(/^Reason=/,"",$i); print $i; exit}}'
  )
  case "$post_action_reason" in
    *Held*) die "exact job is still held after release: $post_action_reason" ;;
  esac
fi

{
  printf 'schema_version\tvlsa_postpublication_analysis_v2_release_receipt.v1\n'
  printf 'status\treleased\n'
  printf 'job_id\t%s\n' "$job_id"
  printf 'job_name\t%s\n' "$EXPECTED_JOB_NAME"
  printf 'population_array_job_id\t%s\n' "$POPULATION_ARRAY_JOB_ID"
  printf 'publisher_job_id\t%s\n' "$PUBLISHER_JOB_ID"
  printf 'artifact_publisher_job_id\t%s\n' "$ARTIFACT_PUBLISHER_JOB_ID"
  printf 'dependency\tafterok:%s\n' "$PUBLISHER_JOB_ID"
  printf 'release_action\t%s\n' "$release_action"
  printf 'observed_state_before_action\t%s\n' "${current_state:-unknown}"
  printf 'observed_reason_before_action\t%s\n' "${current_reason:-unknown}"
  printf 'observed_state_after_action\t%s\n' "${post_action_state:-not_in_scontrol}"
  printf 'observed_reason_after_action\t%s\n' "${post_action_reason:-not_in_scontrol}"
  printf 'submission_receipt_sha256\t%s\n' "$submission_receipt_sha256"
  printf 'output_unused_before_release\ttrue\n'
} | publish_exact_file "$RELEASE_RECEIPT"
publish_receipt_sha256 "$RELEASE_RECEIPT" "$RELEASE_RECEIPT_SHA"
validate_release_receipt "$job_id"

printf '%s\n' "$job_id"
