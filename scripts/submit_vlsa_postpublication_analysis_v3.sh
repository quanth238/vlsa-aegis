#!/usr/bin/env bash
set -euo pipefail

# Shell-only login-node launcher for the decision-aligned v3 derivation.
# Accepted analysis-v2 must already be terminal and immutable. The new CPU
# job is submitted held, inspected, receipted, rechecked, and then released.

readonly POPULATION_ARRAY_JOB_ID=28609
readonly PUBLISHER_JOB_ID=28610
readonly POPULATION_RUN_ID=vlsa-table1-contact-authority-population-20260718a
readonly POPULATION_SOURCE_GIT_COMMIT=1592aa59361f431ba96c6ddcbebcb596f6c20853
readonly ACCEPTED_V3_IMPLEMENTATION_COMMIT=59b7cdf42acfcb2c76e5cd7a50703ee1dd375e74
readonly EXPECTED_JOB_NAME=vlsa-a3-p28610
readonly EXPECTED_V2_JOB_NAME=vlsa-a2-p28610
readonly EXPECTED_REMOTE_REPO=/home/quanth/working_space/vlsa-aegis-table-repro
readonly EXPECTED_EXPERIMENT_ROOT=/mnt/data/quanth/experiments/vlsa-aegis-table1
readonly EXPECTED_V2_OUTPUT_ROOT=/mnt/data/quanth/experiments/vlsa-aegis-table1-analysis-v2
readonly EXPECTED_V3_OUTPUT_ROOT=/mnt/data/quanth/experiments/vlsa-aegis-table1-analysis-v3

: "${V2_ANALYSIS_JOB_ID:?set the exact terminal analysis-v2 job ID}"
: "${EXPECTED_V2_ANALYSIS_GIT_COMMIT:?set exact analysis-v2 release commit}"
: "${EXPECTED_ANALYSIS_V3_GIT_COMMIT:?set reviewed v3 launch release commit}"
: "${EXPECTED_V1_PUBLICATION_RECEIPT_SHA256:?set v1 publication receipt SHA-256}"
: "${EXPECTED_PREPUBLISH_RECEIPT_SHA256:?set prepublish receipt SHA-256}"
: "${EXPECTED_SUMMARY_V2_SHA256:?set accepted analysis-v2 summary SHA-256}"
: "${EXPECTED_ANALYSIS_V2_RECEIPT_SHA256:?set accepted analysis-v2 receipt SHA-256}"
: "${EXPECTED_V2_SUBMISSION_RECEIPT_SHA256:?set v2 submission receipt SHA-256}"
: "${EXPECTED_V2_RELEASE_RECEIPT_SHA256:?set v2 release receipt SHA-256}"
: "${EXPECTED_CONFIG_SHA256:?set protocol config SHA-256}"
: "${EXPECTED_MANIFEST_SHA256:?set population manifest SHA-256}"
: "${EXPECTED_MANIFEST_RECEIPT_SHA256:?set manifest receipt SHA-256}"
: "${EXPECTED_V3_MODULE_SHA256:?set accepted v3 module SHA-256}"
: "${EXPECTED_V3_BUILDER_SHA256:?set reviewed v3 receipt builder SHA-256}"
: "${EXPECTED_V3_RUNNER_SHA256:?set reviewed v3 runner SHA-256}"
: "${EXPECTED_V3_SBATCH_SHA256:?set reviewed v3 SBatch SHA-256}"
: "${EXPECTED_V3_SUBMIT_HELPER_SHA256:?set reviewed v3 submit helper SHA-256}"

RUN_ID=${RUN_ID:-$POPULATION_RUN_ID}
EXPECTED_SOURCE_GIT_COMMIT=${EXPECTED_SOURCE_GIT_COMMIT:-$POPULATION_SOURCE_GIT_COMMIT}
REMOTE_REPO=${REMOTE_REPO:-$EXPECTED_REMOTE_REPO}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-$EXPECTED_EXPERIMENT_ROOT}
RUN_ROOT=${RUN_ROOT:-$EXPERIMENT_ROOT/$RUN_ID}
RESULTS_ROOT=${RESULTS_ROOT:-$RUN_ROOT/tasks}
V1_PUBLICATION_RECEIPT=${V1_PUBLICATION_RECEIPT:-$RUN_ROOT/population-publication-receipt.json}
PREPUBLISH_RECEIPT=${PREPUBLISH_RECEIPT:-$RUN_ROOT/publication-attempts/job-$PUBLISHER_JOB_ID/prepublish-validation.json}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_table1_translational.json}
MANIFEST_PATH=${MANIFEST_PATH:-$REMOTE_REPO/manifests/vlsa_table1_population.jsonl}
MANIFEST_RECEIPT_PATH=${MANIFEST_RECEIPT_PATH:-$REMOTE_REPO/manifests/vlsa_table1_population.receipt.json}
V2_OUTPUT_ROOT=${V2_OUTPUT_ROOT:-$EXPECTED_V2_OUTPUT_ROOT}
V2_OUTPUT_DIR=${V2_OUTPUT_DIR:-$V2_OUTPUT_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID}
SUMMARY_V2_PATH=${SUMMARY_V2_PATH:-$V2_OUTPUT_DIR/population-summary-v2.json}
ANALYSIS_V2_RECEIPT=${ANALYSIS_V2_RECEIPT:-$V2_OUTPUT_DIR/analysis-v2-receipt.json}
V2_CONTROL_DIR=${V2_CONTROL_DIR:-$V2_OUTPUT_ROOT/.analysis-v2-control/$RUN_ID-publisher-$PUBLISHER_JOB_ID}
V2_SUBMISSION_RECEIPT=${V2_SUBMISSION_RECEIPT:-$V2_CONTROL_DIR/submission-receipt.tsv}
V2_RELEASE_RECEIPT=${V2_RELEASE_RECEIPT:-$V2_CONTROL_DIR/release-receipt.tsv}
V3_OUTPUT_ROOT=${V3_OUTPUT_ROOT:-$EXPECTED_V3_OUTPUT_ROOT}
V3_OUTPUT_DIR=${V3_OUTPUT_DIR:-$V3_OUTPUT_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID-v2job-$V2_ANALYSIS_JOB_ID}
V3_MODULE=$REMOTE_REPO/analysis/build_aegis_failure_report_v3.py
V3_BUILDER=$REMOTE_REPO/analysis/build_safelibero_postpublication_v3.py
V3_RUNNER=$REMOTE_REPO/slurm/run_vlsa_postpublication_analysis_v3.sh
V3_SBATCH=$REMOTE_REPO/slurm/vlsa_postpublication_analysis_v3.sbatch
V3_SUBMIT_HELPER=$REMOTE_REPO/scripts/submit_vlsa_postpublication_analysis_v3.sh

CONTROL_ROOT=$V3_OUTPUT_ROOT/.analysis-v3-control
CONTROL_DIR=$CONTROL_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID-v2job-$V2_ANALYSIS_JOB_ID
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
  echo "postpublication analysis-v3 launch rejected: $*" >&2
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

require_exact_terminal_job() {
  local job_id=$1
  local expected_name=$2
  local rows
  rows=$(
    sacct -n -X -j "$job_id" \
      --format=JobIDRaw,JobName,State,ExitCode -P |
      awk -F'|' -v job="$job_id" '$1 == job {print $2 "|" $3 "|" $4}'
  )
  [[ "$rows" == "$expected_name|COMPLETED|0:0" ]] || \
    die "job $job_id is not exact terminal success: $rows"
}

validate_v2_receipts() {
  require_file_hash v2_submission_receipt "$V2_SUBMISSION_RECEIPT" \
    "$EXPECTED_V2_SUBMISSION_RECEIPT_SHA256"
  require_file_hash v2_release_receipt "$V2_RELEASE_RECEIPT" \
    "$EXPECTED_V2_RELEASE_RECEIPT_SHA256"
  verify_receipt_sha256 "$V2_SUBMISSION_RECEIPT" \
    "$V2_CONTROL_DIR/submission-receipt.sha256"
  verify_receipt_sha256 "$V2_RELEASE_RECEIPT" \
    "$V2_CONTROL_DIR/release-receipt.sha256"
  require_tsv_field "$V2_SUBMISSION_RECEIPT" schema_version \
    vlsa_postpublication_analysis_v2_submission_receipt.v1
  require_tsv_field "$V2_SUBMISSION_RECEIPT" status held_validated
  require_tsv_field "$V2_SUBMISSION_RECEIPT" job_id "$V2_ANALYSIS_JOB_ID"
  require_tsv_field "$V2_SUBMISSION_RECEIPT" job_name "$EXPECTED_V2_JOB_NAME"
  require_tsv_field "$V2_SUBMISSION_RECEIPT" population_array_job_id \
    "$POPULATION_ARRAY_JOB_ID"
  require_tsv_field "$V2_SUBMISSION_RECEIPT" publisher_job_id \
    "$PUBLISHER_JOB_ID"
  require_tsv_field "$V2_SUBMISSION_RECEIPT" dependency \
    "afterok:$PUBLISHER_JOB_ID"
  require_tsv_field "$V2_SUBMISSION_RECEIPT" source_git_commit \
    "$EXPECTED_SOURCE_GIT_COMMIT"
  require_tsv_field "$V2_SUBMISSION_RECEIPT" analysis_git_commit \
    "$EXPECTED_V2_ANALYSIS_GIT_COMMIT"
  require_tsv_field "$V2_SUBMISSION_RECEIPT" cpu_only true
  require_tsv_field "$V2_SUBMISSION_RECEIPT" held_before_release true
  require_tsv_field "$V2_RELEASE_RECEIPT" schema_version \
    vlsa_postpublication_analysis_v2_release_receipt.v1
  require_tsv_field "$V2_RELEASE_RECEIPT" status released
  require_tsv_field "$V2_RELEASE_RECEIPT" job_id "$V2_ANALYSIS_JOB_ID"
  require_tsv_field "$V2_RELEASE_RECEIPT" dependency \
    "afterok:$PUBLISHER_JOB_ID"
  require_tsv_field "$V2_RELEASE_RECEIPT" submission_receipt_sha256 \
    "$EXPECTED_V2_SUBMISSION_RECEIPT_SHA256"
}

validate_frozen_inputs() {
  local output_policy=${1:-unused}
  require_exact_terminal_job "$PUBLISHER_JOB_ID" vlsa-aegis-publisher
  require_exact_terminal_job "$V2_ANALYSIS_JOB_ID" "$EXPECTED_V2_JOB_NAME"
  validate_v2_receipts
  [[ -d "$REMOTE_REPO/.git" || -f "$REMOTE_REPO/.git" ]] || \
    die "reviewed repository is missing"
  [[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == \
    "$EXPECTED_ANALYSIS_V3_GIT_COMMIT" ]] || \
    die "repository is not at reviewed v3 release"
  [[ -z "$(git -C "$REMOTE_REPO" status --short --untracked-files=all)" ]] || \
    die "reviewed repository is dirty"
  git -C "$REMOTE_REPO" merge-base --is-ancestor \
    "$ACCEPTED_V3_IMPLEMENTATION_COMMIT" "$EXPECTED_ANALYSIS_V3_GIT_COMMIT" || \
    die "reviewed release does not descend from accepted v3"
  [[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "run root unavailable"
  [[ -d "$RESULTS_ROOT" && ! -L "$RESULTS_ROOT" ]] || \
    die "results root unavailable"
  [[ -d "$V2_OUTPUT_DIR" && ! -L "$V2_OUTPUT_DIR" ]] || \
    die "analysis-v2 output unavailable"
  [[ -d "$V3_OUTPUT_ROOT" && ! -L "$V3_OUTPUT_ROOT" ]] || \
    die "analysis-v3 output root must already exist"
  case "$output_policy" in
    unused)
      [[ ! -e "$V3_OUTPUT_DIR" && ! -L "$V3_OUTPUT_DIR" ]] || \
        die "analysis-v3 output directory already exists"
      ;;
    allow_existing)
      if [[ -e "$V3_OUTPUT_DIR" || -L "$V3_OUTPUT_DIR" ]]; then
        [[ -d "$V3_OUTPUT_DIR" && ! -L "$V3_OUTPUT_DIR" ]] || \
          die "existing analysis-v3 output is not a real directory"
      fi
      ;;
    *) die "invalid output validation policy" ;;
  esac
  require_file_hash v1_publication_receipt "$V1_PUBLICATION_RECEIPT" \
    "$EXPECTED_V1_PUBLICATION_RECEIPT_SHA256"
  require_file_hash prepublish_receipt "$PREPUBLISH_RECEIPT" \
    "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"
  require_file_hash summary_v2 "$SUMMARY_V2_PATH" \
    "$EXPECTED_SUMMARY_V2_SHA256"
  require_file_hash analysis_v2_receipt "$ANALYSIS_V2_RECEIPT" \
    "$EXPECTED_ANALYSIS_V2_RECEIPT_SHA256"
  require_file_hash config "$CONFIG_PATH" "$EXPECTED_CONFIG_SHA256"
  require_file_hash manifest "$MANIFEST_PATH" "$EXPECTED_MANIFEST_SHA256"
  require_file_hash manifest_receipt "$MANIFEST_RECEIPT_PATH" \
    "$EXPECTED_MANIFEST_RECEIPT_SHA256"
  require_file_hash v3_module "$V3_MODULE" "$EXPECTED_V3_MODULE_SHA256"
  require_file_hash v3_builder "$V3_BUILDER" "$EXPECTED_V3_BUILDER_SHA256"
  require_file_hash v3_runner "$V3_RUNNER" "$EXPECTED_V3_RUNNER_SHA256"
  require_file_hash v3_sbatch "$V3_SBATCH" "$EXPECTED_V3_SBATCH_SHA256"
  require_file_hash v3_submit_helper "$V3_SUBMIT_HELPER" \
    "$EXPECTED_V3_SUBMIT_HELPER_SHA256"
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
  [[ "${observed_dependency%%(*}" == "afterok:$V2_ANALYSIS_JOB_ID" ]] || {
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
  require_scontrol_field "$temporary" Command "$V3_SBATCH"
  require_scontrol_field "$temporary" WorkDir "$REMOTE_REPO"
  require_scontrol_field "$temporary" StdOut \
    "/mnt/data/quanth/slurm_logs/$EXPECTED_JOB_NAME-$job_id.out"
  local field
  for field in ReqTRES AllocTRES TresPerNode TresPerTask Gres; do
    case "$(scontrol_field "$temporary" "$field")" in
      *[Gg][Pp][Uu]*)
        rm -f "$temporary"
        die "held job unexpectedly requests a GPU in $field"
        ;;
    esac
  done
  if [[ -e "$SCONTROL_PATH" || -L "$SCONTROL_PATH" ]]; then
    verify_receipt_sha256 "$SCONTROL_PATH" "$SCONTROL_SHA"
    local recorded_field
    for recorded_field in JobId JobName JobState Reason Partition Account QOS \
      NumNodes NumCPUs NumTasks "CPUs/Task" MinMemoryNode TimeLimit \
      Requeue ExcNodeList Command WorkDir StdOut ReqTRES AllocTRES \
      TresPerNode TresPerTask Gres; do
      [[ "$(scontrol_field "$SCONTROL_PATH" "$recorded_field")" == \
        "$(scontrol_field "$temporary" "$recorded_field")" ]] || {
        rm -f "$temporary"
        die "held-job recorded $recorded_field differs during recovery"
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
    ln "$temporary" "$SCONTROL_PATH" || {
      rm -f "$temporary"
      die "cannot publish held-job snapshot"
    }
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
    die "immutable job ID differs"
  require_tsv_field "$SUBMISSION_RECEIPT" schema_version \
    vlsa_postpublication_analysis_v3_submission_receipt.v1
  require_tsv_field "$SUBMISSION_RECEIPT" status held_validated
  require_tsv_field "$SUBMISSION_RECEIPT" run_id "$RUN_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" job_id "$job_id"
  require_tsv_field "$SUBMISSION_RECEIPT" job_name "$EXPECTED_JOB_NAME"
  require_tsv_field "$SUBMISSION_RECEIPT" population_array_job_id \
    "$POPULATION_ARRAY_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" publisher_job_id "$PUBLISHER_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" analysis_v2_job_id \
    "$V2_ANALYSIS_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" dependency \
    "afterok:$V2_ANALYSIS_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" source_git_commit \
    "$EXPECTED_SOURCE_GIT_COMMIT"
  require_tsv_field "$SUBMISSION_RECEIPT" analysis_v2_git_commit \
    "$EXPECTED_V2_ANALYSIS_GIT_COMMIT"
  require_tsv_field "$SUBMISSION_RECEIPT" analysis_v3_git_commit \
    "$EXPECTED_ANALYSIS_V3_GIT_COMMIT"
  require_tsv_field "$SUBMISSION_RECEIPT" summary_v2_sha256 \
    "$EXPECTED_SUMMARY_V2_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" analysis_v2_receipt_sha256 \
    "$EXPECTED_ANALYSIS_V2_RECEIPT_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" v2_submission_receipt_sha256 \
    "$EXPECTED_V2_SUBMISSION_RECEIPT_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" v2_release_receipt_sha256 \
    "$EXPECTED_V2_RELEASE_RECEIPT_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" v1_publication_receipt_sha256 \
    "$EXPECTED_V1_PUBLICATION_RECEIPT_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" prepublish_receipt_sha256 \
    "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" config_sha256 \
    "$EXPECTED_CONFIG_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" manifest_sha256 \
    "$EXPECTED_MANIFEST_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" manifest_receipt_sha256 \
    "$EXPECTED_MANIFEST_RECEIPT_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" v3_module_sha256 \
    "$EXPECTED_V3_MODULE_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" builder_sha256 \
    "$EXPECTED_V3_BUILDER_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" runner_sha256 \
    "$EXPECTED_V3_RUNNER_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" sbatch_sha256 \
    "$EXPECTED_V3_SBATCH_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" submit_helper_sha256 \
    "$EXPECTED_V3_SUBMIT_HELPER_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" output_dir "$V3_OUTPUT_DIR"
  require_tsv_field "$SUBMISSION_RECEIPT" partition main
  require_tsv_field "$SUBMISSION_RECEIPT" account normal
  require_tsv_field "$SUBMISSION_RECEIPT" qos normal
  require_tsv_field "$SUBMISSION_RECEIPT" nodes 1
  require_tsv_field "$SUBMISSION_RECEIPT" ntasks 1
  require_tsv_field "$SUBMISSION_RECEIPT" cpus_per_task 4
  require_tsv_field "$SUBMISSION_RECEIPT" memory 32G
  require_tsv_field "$SUBMISSION_RECEIPT" time_limit 04:00:00
  require_tsv_field "$SUBMISSION_RECEIPT" gpus 0
  require_tsv_field "$SUBMISSION_RECEIPT" exclude worker-3
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
    vlsa_postpublication_analysis_v3_release_receipt.v1
  require_tsv_field "$RELEASE_RECEIPT" status released
  require_tsv_field "$RELEASE_RECEIPT" run_id "$RUN_ID"
  require_tsv_field "$RELEASE_RECEIPT" job_id "$job_id"
  require_tsv_field "$RELEASE_RECEIPT" job_name "$EXPECTED_JOB_NAME"
  require_tsv_field "$RELEASE_RECEIPT" population_array_job_id \
    "$POPULATION_ARRAY_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" publisher_job_id "$PUBLISHER_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" analysis_v2_job_id \
    "$V2_ANALYSIS_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" dependency \
    "afterok:$V2_ANALYSIS_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" submission_receipt_sha256 \
    "$(sha256_file "$SUBMISSION_RECEIPT")"
  require_tsv_field "$RELEASE_RECEIPT" output_unused_before_release true
  case "$(tsv_field "$RELEASE_RECEIPT" release_action)" in
    released_exact_held_job|recovered_after_prior_release) ;;
    *) die "release receipt action is invalid" ;;
  esac
  local field
  for field in observed_state_before_action observed_reason_before_action \
    observed_state_after_action observed_reason_after_action; do
    [[ -n "$(tsv_field "$RELEASE_RECEIPT" "$field")" ]] || \
      die "release receipt $field is missing"
  done
  case "$(tsv_field "$RELEASE_RECEIPT" observed_reason_after_action)" in
    *Held*) die "release receipt retains a held state" ;;
  esac
}

[[ "$V2_ANALYSIS_JOB_ID" =~ ^[0-9]+$ ]] || die "v2 job ID is invalid"
for value in "$EXPECTED_SOURCE_GIT_COMMIT" \
  "$EXPECTED_V2_ANALYSIS_GIT_COMMIT" \
  "$EXPECTED_ANALYSIS_V3_GIT_COMMIT"; do
  [[ "$value" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is invalid"
done
for value in \
  "$EXPECTED_V1_PUBLICATION_RECEIPT_SHA256" \
  "$EXPECTED_PREPUBLISH_RECEIPT_SHA256" \
  "$EXPECTED_SUMMARY_V2_SHA256" \
  "$EXPECTED_ANALYSIS_V2_RECEIPT_SHA256" \
  "$EXPECTED_V2_SUBMISSION_RECEIPT_SHA256" \
  "$EXPECTED_V2_RELEASE_RECEIPT_SHA256" \
  "$EXPECTED_CONFIG_SHA256" \
  "$EXPECTED_MANIFEST_SHA256" \
  "$EXPECTED_MANIFEST_RECEIPT_SHA256" \
  "$EXPECTED_V3_MODULE_SHA256" \
  "$EXPECTED_V3_BUILDER_SHA256" \
  "$EXPECTED_V3_RUNNER_SHA256" \
  "$EXPECTED_V3_SBATCH_SHA256" \
  "$EXPECTED_V3_SUBMIT_HELPER_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die "expected SHA-256 is invalid"
done
[[ "$RUN_ID" == "$POPULATION_RUN_ID" ]] || die "run ID differs"
[[ "$EXPECTED_SOURCE_GIT_COMMIT" == "$POPULATION_SOURCE_GIT_COMMIT" ]] || \
  die "runtime source commit differs"
[[ "$REMOTE_REPO" == "$EXPECTED_REMOTE_REPO" ]] || die "repo path differs"
[[ "$EXPERIMENT_ROOT" == "$EXPECTED_EXPERIMENT_ROOT" ]] || \
  die "experiment root differs"
[[ "$RUN_ROOT" == "$EXPERIMENT_ROOT/$POPULATION_RUN_ID" ]] || \
  die "run root differs"
[[ "$RESULTS_ROOT" == "$RUN_ROOT/tasks" ]] || die "results root differs"
[[ "$V2_OUTPUT_ROOT" == "$EXPECTED_V2_OUTPUT_ROOT" ]] || \
  die "v2 output root differs"
[[ "$V2_OUTPUT_DIR" == \
  "$V2_OUTPUT_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID" ]] || \
  die "v2 output directory differs"
[[ "$SUMMARY_V2_PATH" == "$V2_OUTPUT_DIR/population-summary-v2.json" ]] || \
  die "v2 summary path differs"
[[ "$ANALYSIS_V2_RECEIPT" == "$V2_OUTPUT_DIR/analysis-v2-receipt.json" ]] || \
  die "v2 receipt path differs"
[[ "$V2_CONTROL_DIR" == \
  "$V2_OUTPUT_ROOT/.analysis-v2-control/$RUN_ID-publisher-$PUBLISHER_JOB_ID" ]] || \
  die "v2 control path differs"
[[ "$V3_OUTPUT_ROOT" == "$EXPECTED_V3_OUTPUT_ROOT" ]] || \
  die "v3 output root differs"
[[ "$V3_OUTPUT_DIR" == \
  "$V3_OUTPUT_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID-v2job-$V2_ANALYSIS_JOB_ID" ]] || \
  die "v3 output directory differs"

for name in V2_ANALYSIS_JOB_ID EXPECTED_SOURCE_GIT_COMMIT \
  EXPECTED_V2_ANALYSIS_GIT_COMMIT EXPECTED_ANALYSIS_V3_GIT_COMMIT \
  EXPECTED_V1_PUBLICATION_RECEIPT_SHA256 \
  EXPECTED_PREPUBLISH_RECEIPT_SHA256 EXPECTED_SUMMARY_V2_SHA256 \
  EXPECTED_ANALYSIS_V2_RECEIPT_SHA256 \
  EXPECTED_V2_SUBMISSION_RECEIPT_SHA256 \
  EXPECTED_V2_RELEASE_RECEIPT_SHA256 EXPECTED_CONFIG_SHA256 \
  EXPECTED_MANIFEST_SHA256 EXPECTED_MANIFEST_RECEIPT_SHA256 \
  EXPECTED_V3_MODULE_SHA256 EXPECTED_V3_BUILDER_SHA256 \
  EXPECTED_V3_RUNNER_SHA256 EXPECTED_V3_SBATCH_SHA256 \
  EXPECTED_V3_SUBMIT_HELPER_SHA256 RUN_ID REMOTE_REPO EXPERIMENT_ROOT \
  RUN_ROOT RESULTS_ROOT V1_PUBLICATION_RECEIPT PREPUBLISH_RECEIPT \
  CONFIG_PATH MANIFEST_PATH MANIFEST_RECEIPT_PATH V2_OUTPUT_ROOT \
  V2_OUTPUT_DIR SUMMARY_V2_PATH ANALYSIS_V2_RECEIPT V2_CONTROL_DIR \
  V2_SUBMISSION_RECEIPT V2_RELEASE_RECEIPT V3_OUTPUT_ROOT V3_OUTPUT_DIR; do
  require_export_scalar "$name" "${!name}"
done

initial_output_policy=unused
if [[ (-f "$JOB_ID_PATH" && ! -L "$JOB_ID_PATH") || \
  (-f "$RELEASE_RECEIPT" && ! -L "$RELEASE_RECEIPT") ]]; then
  initial_output_policy=allow_existing
fi
validate_frozen_inputs "$initial_output_policy"

if [[ ! -e "$CONTROL_ROOT" && ! -L "$CONTROL_ROOT" ]]; then
  mkdir "$CONTROL_ROOT"
fi
[[ -d "$CONTROL_ROOT" && ! -L "$CONTROL_ROOT" ]] || \
  die "v3 control root is not a real directory"
if [[ ! -e "$CONTROL_DIR" && ! -L "$CONTROL_DIR" ]]; then
  mkdir "$CONTROL_DIR"
fi
[[ -d "$CONTROL_DIR" && ! -L "$CONTROL_DIR" ]] || \
  die "v3 control directory is not a real directory"
[[ ! -L "$CONTROL_DIR/.lock" ]] || die "control lock is a symlink"
exec 9>"$CONTROL_DIR/.lock"
flock -n 9 || die "another exact v3 launch is active"

{
  printf 'schema_version\tvlsa_postpublication_analysis_v3_submission_intent.v1\n'
  printf 'status\tfrozen\n'
  printf 'run_id\t%s\n' "$RUN_ID"
  printf 'population_array_job_id\t%s\n' "$POPULATION_ARRAY_JOB_ID"
  printf 'publisher_job_id\t%s\n' "$PUBLISHER_JOB_ID"
  printf 'analysis_v2_job_id\t%s\n' "$V2_ANALYSIS_JOB_ID"
  printf 'dependency\tafterok:%s\n' "$V2_ANALYSIS_JOB_ID"
  printf 'job_name\t%s\n' "$EXPECTED_JOB_NAME"
  printf 'population_source_git_commit\t%s\n' "$EXPECTED_SOURCE_GIT_COMMIT"
  printf 'analysis_v2_git_commit\t%s\n' "$EXPECTED_V2_ANALYSIS_GIT_COMMIT"
  printf 'analysis_v3_git_commit\t%s\n' "$EXPECTED_ANALYSIS_V3_GIT_COMMIT"
  printf 'summary_v2_sha256\t%s\n' "$EXPECTED_SUMMARY_V2_SHA256"
  printf 'analysis_v2_receipt_sha256\t%s\n' "$EXPECTED_ANALYSIS_V2_RECEIPT_SHA256"
  printf 'v2_submission_receipt_sha256\t%s\n' "$EXPECTED_V2_SUBMISSION_RECEIPT_SHA256"
  printf 'v2_release_receipt_sha256\t%s\n' "$EXPECTED_V2_RELEASE_RECEIPT_SHA256"
  printf 'v1_publication_receipt_sha256\t%s\n' "$EXPECTED_V1_PUBLICATION_RECEIPT_SHA256"
  printf 'prepublish_receipt_sha256\t%s\n' "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"
  printf 'config_sha256\t%s\n' "$EXPECTED_CONFIG_SHA256"
  printf 'manifest_sha256\t%s\n' "$EXPECTED_MANIFEST_SHA256"
  printf 'manifest_receipt_sha256\t%s\n' "$EXPECTED_MANIFEST_RECEIPT_SHA256"
  printf 'v3_module_sha256\t%s\n' "$EXPECTED_V3_MODULE_SHA256"
  printf 'builder_sha256\t%s\n' "$EXPECTED_V3_BUILDER_SHA256"
  printf 'runner_sha256\t%s\n' "$EXPECTED_V3_RUNNER_SHA256"
  printf 'sbatch_sha256\t%s\n' "$EXPECTED_V3_SBATCH_SHA256"
  printf 'submit_helper_sha256\t%s\n' "$EXPECTED_V3_SUBMIT_HELPER_SHA256"
  printf 'output_dir\t%s\n' "$V3_OUTPUT_DIR"
  printf 'resources\tpartition=main,account=normal,qos=normal,nodes=1,ntasks=1,cpus=4,mem=32G,time=04:00:00,gpus=0,exclude=worker-3\n'
} | publish_exact_file "$INTENT_PATH"
publish_receipt_sha256 "$INTENT_PATH" "$INTENT_SHA"
verify_receipt_sha256 "$INTENT_PATH" "$INTENT_SHA"

if [[ -f "$RELEASE_RECEIPT" || -L "$RELEASE_RECEIPT" ]]; then
  old_job=$(tsv_field "$RELEASE_RECEIPT" job_id)
  [[ "$old_job" =~ ^[0-9]+$ ]] || die "recorded v3 job ID is invalid"
  validate_release_receipt "$old_job"
  printf '%s\n' "$old_job"
  exit 0
fi

job_id=
if [[ -f "$JOB_ID_PATH" && ! -L "$JOB_ID_PATH" ]]; then
  verify_receipt_sha256 "$JOB_ID_PATH" "$JOB_ID_SHA"
  job_id=$(tr -d '\n' <"$JOB_ID_PATH")
else
  recovered=$(
    squeue --noheader --user="${USER:?USER is required}" \
      --name="$EXPECTED_JOB_NAME" --format='%A' | awk 'NF {print $1}'
  )
  count=$(printf '%s\n' "$recovered" | awk 'NF {n++} END {print n+0}')
  (( count <= 1 )) || die "multiple active jobs share v3 recovery name"
  if (( count == 1 )); then
    job_id=$(printf '%s\n' "$recovered" | awk 'NF {print; exit}')
  else
    [[ ! -e "$V3_OUTPUT_DIR" && ! -L "$V3_OUTPUT_DIR" ]] || \
      die "analysis-v3 output directory already exists"
    export_spec="RUN_ID=$RUN_ID"
    for name in V2_ANALYSIS_JOB_ID EXPECTED_SOURCE_GIT_COMMIT \
      EXPECTED_V2_ANALYSIS_GIT_COMMIT EXPECTED_ANALYSIS_V3_GIT_COMMIT \
      EXPECTED_V1_PUBLICATION_RECEIPT_SHA256 \
      EXPECTED_PREPUBLISH_RECEIPT_SHA256 EXPECTED_SUMMARY_V2_SHA256 \
      EXPECTED_ANALYSIS_V2_RECEIPT_SHA256 \
      EXPECTED_V2_SUBMISSION_RECEIPT_SHA256 \
      EXPECTED_V2_RELEASE_RECEIPT_SHA256 EXPECTED_CONFIG_SHA256 \
      EXPECTED_MANIFEST_SHA256 EXPECTED_MANIFEST_RECEIPT_SHA256 \
      EXPECTED_V3_MODULE_SHA256 EXPECTED_V3_BUILDER_SHA256 \
      EXPECTED_V3_RUNNER_SHA256 EXPECTED_V3_SBATCH_SHA256 \
      EXPECTED_V3_SUBMIT_HELPER_SHA256 REMOTE_REPO \
      EXPERIMENT_ROOT RUN_ROOT RESULTS_ROOT V1_PUBLICATION_RECEIPT \
      PREPUBLISH_RECEIPT CONFIG_PATH MANIFEST_PATH MANIFEST_RECEIPT_PATH \
      V2_OUTPUT_ROOT V2_OUTPUT_DIR SUMMARY_V2_PATH ANALYSIS_V2_RECEIPT \
      V2_CONTROL_DIR V2_SUBMISSION_RECEIPT V2_RELEASE_RECEIPT \
      V3_OUTPUT_ROOT V3_OUTPUT_DIR; do
      export_spec+=",${name}=${!name}"
    done
    submission=$(
      sbatch --parsable --hold \
        --job-name="$EXPECTED_JOB_NAME" \
        --dependency="afterok:$V2_ANALYSIS_JOB_ID" \
        --chdir="$REMOTE_REPO" \
        --export="$export_spec" \
        "$V3_SBATCH"
    )
    job_id=${submission%%;*}
  fi
  [[ "$job_id" =~ ^[0-9]+$ ]] || die "sbatch returned invalid job ID"
  printf '%s\n' "$job_id" | publish_exact_file "$JOB_ID_PATH"
  publish_receipt_sha256 "$JOB_ID_PATH" "$JOB_ID_SHA"
fi
[[ "$job_id" =~ ^[0-9]+$ ]] || die "recorded job ID is invalid"
verify_receipt_sha256 "$JOB_ID_PATH" "$JOB_ID_SHA"

if [[ ! -f "$SUBMISSION_RECEIPT" && ! -L "$SUBMISSION_RECEIPT" ]]; then
  inspect_held_job "$job_id"
  {
    printf 'schema_version\tvlsa_postpublication_analysis_v3_submission_receipt.v1\n'
    printf 'status\theld_validated\n'
    printf 'run_id\t%s\n' "$RUN_ID"
    printf 'job_id\t%s\n' "$job_id"
    printf 'job_name\t%s\n' "$EXPECTED_JOB_NAME"
    printf 'population_array_job_id\t%s\n' "$POPULATION_ARRAY_JOB_ID"
    printf 'publisher_job_id\t%s\n' "$PUBLISHER_JOB_ID"
    printf 'analysis_v2_job_id\t%s\n' "$V2_ANALYSIS_JOB_ID"
    printf 'dependency\tafterok:%s\n' "$V2_ANALYSIS_JOB_ID"
    printf 'source_git_commit\t%s\n' "$EXPECTED_SOURCE_GIT_COMMIT"
    printf 'analysis_v2_git_commit\t%s\n' "$EXPECTED_V2_ANALYSIS_GIT_COMMIT"
    printf 'analysis_v3_git_commit\t%s\n' "$EXPECTED_ANALYSIS_V3_GIT_COMMIT"
    printf 'summary_v2_sha256\t%s\n' "$EXPECTED_SUMMARY_V2_SHA256"
    printf 'analysis_v2_receipt_sha256\t%s\n' "$EXPECTED_ANALYSIS_V2_RECEIPT_SHA256"
    printf 'v2_submission_receipt_sha256\t%s\n' "$EXPECTED_V2_SUBMISSION_RECEIPT_SHA256"
    printf 'v2_release_receipt_sha256\t%s\n' "$EXPECTED_V2_RELEASE_RECEIPT_SHA256"
    printf 'v1_publication_receipt_sha256\t%s\n' "$EXPECTED_V1_PUBLICATION_RECEIPT_SHA256"
    printf 'prepublish_receipt_sha256\t%s\n' "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"
    printf 'config_sha256\t%s\n' "$EXPECTED_CONFIG_SHA256"
    printf 'manifest_sha256\t%s\n' "$EXPECTED_MANIFEST_SHA256"
    printf 'manifest_receipt_sha256\t%s\n' "$EXPECTED_MANIFEST_RECEIPT_SHA256"
    printf 'v3_module_sha256\t%s\n' "$EXPECTED_V3_MODULE_SHA256"
    printf 'builder_sha256\t%s\n' "$EXPECTED_V3_BUILDER_SHA256"
    printf 'runner_sha256\t%s\n' "$EXPECTED_V3_RUNNER_SHA256"
    printf 'sbatch_sha256\t%s\n' "$EXPECTED_V3_SBATCH_SHA256"
    printf 'submit_helper_sha256\t%s\n' "$EXPECTED_V3_SUBMIT_HELPER_SHA256"
    printf 'output_dir\t%s\n' "$V3_OUTPUT_DIR"
    printf 'partition\tmain\n'
    printf 'account\tnormal\n'
    printf 'qos\tnormal\n'
    printf 'nodes\t1\n'
    printf 'ntasks\t1\n'
    printf 'cpus_per_task\t4\n'
    printf 'memory\t32G\n'
    printf 'time_limit\t04:00:00\n'
    printf 'gpus\t0\n'
    printf 'exclude\tworker-3\n'
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

validate_submission_receipt "$job_id"
submission_sha=$(sha256_file "$SUBMISSION_RECEIPT")

snapshot=$(scontrol show job -dd -o "$job_id" 2>/dev/null || true)
state=
reason=
if [[ -n "$snapshot" ]]; then
  state=$(printf '%s\n' "$snapshot" | \
    awk '{for(i=1;i<=NF;i++) if($i~/^JobState=/){sub(/^JobState=/,"",$i);print $i;exit}}')
  reason=$(printf '%s\n' "$snapshot" | \
    awk '{for(i=1;i<=NF;i++) if($i~/^Reason=/){sub(/^Reason=/,"",$i);print $i;exit}}')
fi
if [[ "$state" == PENDING && "$reason" == JobHeldUser ]]; then
  validate_frozen_inputs unused
  inspect_held_job "$job_id"
  scontrol release "$job_id"
  action=released_exact_held_job
else
  validate_frozen_inputs allow_existing
  if [[ -z "$state" ]]; then
    state=$(
      sacct -n -X -j "$job_id" --format=JobIDRaw,State -P |
        awk -F'|' -v job="$job_id" '$1==job {print $2;exit}'
    )
  fi
  case "$state" in
    PENDING|CONFIGURING|RUNNING|COMPLETING|COMPLETED|FAILED|TIMEOUT|\
OUT_OF_MEMORY|CANCELLED|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
      case "$reason" in *Held*) die "exact v3 job remains held" ;; esac
      action=recovered_after_prior_release
      ;;
    *) die "v3 job is neither held nor observably released: $state/$reason" ;;
  esac
fi

after=$(scontrol show job -dd -o "$job_id" 2>/dev/null || true)
after_state=
after_reason=
if [[ -n "$after" ]]; then
  after_state=$(printf '%s\n' "$after" | \
    awk '{for(i=1;i<=NF;i++) if($i~/^JobState=/){sub(/^JobState=/,"",$i);print $i;exit}}')
  after_reason=$(printf '%s\n' "$after" | \
    awk '{for(i=1;i<=NF;i++) if($i~/^Reason=/){sub(/^Reason=/,"",$i);print $i;exit}}')
  case "$after_reason" in *Held*) die "v3 job is still held after release" ;; esac
fi

{
  printf 'schema_version\tvlsa_postpublication_analysis_v3_release_receipt.v1\n'
  printf 'status\treleased\n'
  printf 'run_id\t%s\n' "$RUN_ID"
  printf 'job_id\t%s\n' "$job_id"
  printf 'job_name\t%s\n' "$EXPECTED_JOB_NAME"
  printf 'population_array_job_id\t%s\n' "$POPULATION_ARRAY_JOB_ID"
  printf 'publisher_job_id\t%s\n' "$PUBLISHER_JOB_ID"
  printf 'analysis_v2_job_id\t%s\n' "$V2_ANALYSIS_JOB_ID"
  printf 'dependency\tafterok:%s\n' "$V2_ANALYSIS_JOB_ID"
  printf 'release_action\t%s\n' "$action"
  printf 'observed_state_before_action\t%s\n' "${state:-unknown}"
  printf 'observed_reason_before_action\t%s\n' "${reason:-unknown}"
  printf 'observed_state_after_action\t%s\n' "${after_state:-not_in_scontrol}"
  printf 'observed_reason_after_action\t%s\n' "${after_reason:-not_in_scontrol}"
  printf 'submission_receipt_sha256\t%s\n' "$submission_sha"
  printf 'output_unused_before_release\ttrue\n'
} | publish_exact_file "$RELEASE_RECEIPT"
publish_receipt_sha256 "$RELEASE_RECEIPT" "$RELEASE_RECEIPT_SHA"
validate_release_receipt "$job_id"
printf '%s\n' "$job_id"
