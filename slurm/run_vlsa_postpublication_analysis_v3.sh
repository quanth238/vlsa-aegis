#!/usr/bin/env bash
set -euo pipefail

# Allocation-side runner for the decision-aligned v3 derivation. This job
# reads only terminal population and analysis-v2 artifacts. It never runs
# inference, perception, rendering, simulation, model serving, or training.

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
readonly EXPECTED_AEGIS_PYTHON=/mnt/data/quanth/venvs/safety_vla/main/bin/python

: "${V2_ANALYSIS_JOB_ID:?set the exact terminal analysis-v2 job ID}"
: "${EXPECTED_V2_ANALYSIS_GIT_COMMIT:?set the exact analysis-v2 release commit}"
: "${EXPECTED_ANALYSIS_V3_GIT_COMMIT:?set the exact reviewed v3 launch release commit}"
: "${EXPECTED_V1_PUBLICATION_RECEIPT_SHA256:?set terminal v1 publication receipt SHA-256}"
: "${EXPECTED_PREPUBLISH_RECEIPT_SHA256:?set terminal prepublish receipt SHA-256}"
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
AEGIS_PYTHON=${AEGIS_PYTHON:-$EXPECTED_AEGIS_PYTHON}
V3_MODULE=$REMOTE_REPO/analysis/build_aegis_failure_report_v3.py
V3_BUILDER=$REMOTE_REPO/analysis/build_safelibero_postpublication_v3.py
V3_RUNNER=$REMOTE_REPO/slurm/run_vlsa_postpublication_analysis_v3.sh
V3_SBATCH=$REMOTE_REPO/slurm/vlsa_postpublication_analysis_v3.sbatch
V3_SUBMIT_HELPER=$REMOTE_REPO/scripts/submit_vlsa_postpublication_analysis_v3.sh
V3_CONTROL_DIR=$V3_OUTPUT_ROOT/.analysis-v3-control/$RUN_ID-publisher-$PUBLISHER_JOB_ID-v2job-$V2_ANALYSIS_JOB_ID
V3_SUBMISSION_RECEIPT=$V3_CONTROL_DIR/submission-receipt.tsv
V3_SUBMISSION_RECEIPT_SHA=$V3_CONTROL_DIR/submission-receipt.sha256
V3_RELEASE_RECEIPT=$V3_CONTROL_DIR/release-receipt.tsv
V3_RELEASE_RECEIPT_SHA=$V3_CONTROL_DIR/release-receipt.sha256

die() {
  echo "postpublication analysis-v3 runner rejected: $*" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha256() {
  local label=$1
  local value=$2
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || \
    die "$label is not lowercase SHA-256"
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

tsv_field() {
  local path=$1
  local key=$2
  awk -F'\t' -v key="$key" \
    '$1 == key {print substr($0, length($1) + 2); exit}' "$path"
}

require_tsv_field() {
  local path=$1
  local key=$2
  local expected=$3
  local observed
  observed=$(tsv_field "$path" "$key")
  [[ "$observed" == "$expected" ]] || \
    die "immutable launch receipt $key differs: observed=$observed expected=$expected"
}

verify_receipt_sha256() {
  local receipt=$1
  local sidecar=$2
  require_regular_file immutable_launch_receipt "$receipt"
  require_regular_file immutable_launch_receipt_sidecar "$sidecar"
  (
    cd "$(dirname "$receipt")"
    sha256sum --check --status "$(basename "$sidecar")"
  ) || die "immutable launch receipt hash differs: $receipt"
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

[[ "$V2_ANALYSIS_JOB_ID" =~ ^[0-9]+$ ]] || \
  die "analysis-v2 job ID is not numeric"
[[ "$EXPECTED_V2_ANALYSIS_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "analysis-v2 Git commit is invalid"
[[ "$EXPECTED_ANALYSIS_V3_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "analysis-v3 Git commit is invalid"
[[ "$RUN_ID" == "$POPULATION_RUN_ID" ]] || die "run ID differs"
[[ "$EXPECTED_SOURCE_GIT_COMMIT" == "$POPULATION_SOURCE_GIT_COMMIT" ]] || \
  die "runtime source commit differs"
[[ "$REMOTE_REPO" == "$EXPECTED_REMOTE_REPO" ]] || \
  die "repository path differs"
[[ "$EXPERIMENT_ROOT" == "$EXPECTED_EXPERIMENT_ROOT" ]] || \
  die "experiment root differs"
[[ "$RUN_ROOT" == "$EXPECTED_EXPERIMENT_ROOT/$POPULATION_RUN_ID" ]] || \
  die "run root differs"
[[ "$RESULTS_ROOT" == "$RUN_ROOT/tasks" ]] || die "results root differs"
[[ "$V1_PUBLICATION_RECEIPT" == \
  "$RUN_ROOT/population-publication-receipt.json" ]] || \
  die "v1 publication receipt path differs"
[[ "$PREPUBLISH_RECEIPT" == \
  "$RUN_ROOT/publication-attempts/job-$PUBLISHER_JOB_ID/prepublish-validation.json" ]] || \
  die "prepublish receipt path differs"
[[ "$CONFIG_PATH" == "$REMOTE_REPO/configs/vlsa_table1_translational.json" ]] || \
  die "config path differs"
[[ "$MANIFEST_PATH" == "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" ]] || \
  die "manifest path differs"
[[ "$MANIFEST_RECEIPT_PATH" == \
  "$REMOTE_REPO/manifests/vlsa_table1_population.receipt.json" ]] || \
  die "manifest receipt path differs"
[[ "$V2_OUTPUT_ROOT" == "$EXPECTED_V2_OUTPUT_ROOT" ]] || \
  die "analysis-v2 output root differs"
[[ "$V2_OUTPUT_DIR" == \
  "$EXPECTED_V2_OUTPUT_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID" ]] || \
  die "analysis-v2 output directory differs"
[[ "$SUMMARY_V2_PATH" == "$V2_OUTPUT_DIR/population-summary-v2.json" ]] || \
  die "analysis-v2 summary path differs"
[[ "$ANALYSIS_V2_RECEIPT" == "$V2_OUTPUT_DIR/analysis-v2-receipt.json" ]] || \
  die "analysis-v2 receipt path differs"
[[ "$V2_CONTROL_DIR" == \
  "$EXPECTED_V2_OUTPUT_ROOT/.analysis-v2-control/$RUN_ID-publisher-$PUBLISHER_JOB_ID" ]] || \
  die "analysis-v2 control directory differs"
[[ "$V2_SUBMISSION_RECEIPT" == "$V2_CONTROL_DIR/submission-receipt.tsv" ]] || \
  die "analysis-v2 submission receipt path differs"
[[ "$V2_RELEASE_RECEIPT" == "$V2_CONTROL_DIR/release-receipt.tsv" ]] || \
  die "analysis-v2 release receipt path differs"
[[ "$V3_OUTPUT_ROOT" == "$EXPECTED_V3_OUTPUT_ROOT" ]] || \
  die "analysis-v3 output root differs"
[[ "$V3_OUTPUT_DIR" == \
  "$EXPECTED_V3_OUTPUT_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID-v2job-$V2_ANALYSIS_JOB_ID" ]] || \
  die "analysis-v3 output directory differs"
[[ "$V3_CONTROL_DIR" == \
  "$EXPECTED_V3_OUTPUT_ROOT/.analysis-v3-control/$RUN_ID-publisher-$PUBLISHER_JOB_ID-v2job-$V2_ANALYSIS_JOB_ID" ]] || \
  die "analysis-v3 control directory differs"
[[ "$AEGIS_PYTHON" == "$EXPECTED_AEGIS_PYTHON" ]] || \
  die "Python path differs"

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
  require_sha256 immutable_hash "$value"
done

for name in SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_DEPENDENCY \
  SLURMD_NODENAME SLURM_JOB_PARTITION SLURM_CPUS_PER_TASK \
  SLURM_MEM_PER_NODE; do
  [[ -n "${!name:-}" ]] || die "missing allocation field $name"
done
[[ "$SLURM_JOB_NAME" == "$EXPECTED_JOB_NAME" ]] || \
  die "job name differs"
[[ "$SLURM_JOB_DEPENDENCY" == "afterok:$V2_ANALYSIS_JOB_ID" ]] || \
  die "dependency must be exactly afterok:$V2_ANALYSIS_JOB_ID"
[[ "$SLURM_JOB_PARTITION" == main ]] || die "partition must be main"
[[ "$SLURM_CPUS_PER_TASK" == 4 ]] || die "requires exactly 4 CPUs"
[[ "$SLURM_MEM_PER_NODE" == 32768 ]] || \
  die "requires exactly 32768 MiB RAM"
case "$SLURMD_NODENAME" in
  worker-3|login*|login-restricted*)
    die "cannot execute on $SLURMD_NODENAME"
    ;;
esac
[[ -z "${CUDA_VISIBLE_DEVICES:-}" || \
  "${CUDA_VISIBLE_DEVICES:-}" == NoDevFiles || \
  "${CUDA_VISIBLE_DEVICES:-}" == -1 ]] || \
  die "CPU-only analysis must not receive CUDA devices"
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || \
  die "CPU-only analysis must not receive a Slurm GPU allocation"

require_exact_terminal_job "$PUBLISHER_JOB_ID" vlsa-aegis-publisher
require_exact_terminal_job "$V2_ANALYSIS_JOB_ID" "$EXPECTED_V2_JOB_NAME"

[[ -d "$REMOTE_REPO/.git" || -f "$REMOTE_REPO/.git" ]] || \
  die "reviewed repository is missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == \
  "$EXPECTED_ANALYSIS_V3_GIT_COMMIT" ]] || \
  die "repository is not at the reviewed v3 release"
[[ -z "$(git -C "$REMOTE_REPO" status --short --untracked-files=all)" ]] || \
  die "reviewed repository is dirty"
git -C "$REMOTE_REPO" merge-base --is-ancestor \
  "$ACCEPTED_V3_IMPLEMENTATION_COMMIT" "$EXPECTED_ANALYSIS_V3_GIT_COMMIT" || \
  die "reviewed release does not descend from accepted v3"
[[ -x "$AEGIS_PYTHON" ]] || die "Python interpreter is not executable"
[[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "run root unavailable"
[[ -d "$RESULTS_ROOT" && ! -L "$RESULTS_ROOT" ]] || \
  die "results root unavailable"
[[ -d "$V2_OUTPUT_DIR" && ! -L "$V2_OUTPUT_DIR" ]] || \
  die "analysis-v2 output unavailable"
[[ -d "$V3_OUTPUT_ROOT" && ! -L "$V3_OUTPUT_ROOT" ]] || \
  die "analysis-v3 output root must already exist"
[[ ! -e "$V3_OUTPUT_DIR" && ! -L "$V3_OUTPUT_DIR" ]] || \
  die "analysis-v3 output directory already exists"

require_file_hash v1_publication_receipt "$V1_PUBLICATION_RECEIPT" \
  "$EXPECTED_V1_PUBLICATION_RECEIPT_SHA256"
require_file_hash prepublish_receipt "$PREPUBLISH_RECEIPT" \
  "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"
require_file_hash summary_v2 "$SUMMARY_V2_PATH" \
  "$EXPECTED_SUMMARY_V2_SHA256"
require_file_hash analysis_v2_receipt "$ANALYSIS_V2_RECEIPT" \
  "$EXPECTED_ANALYSIS_V2_RECEIPT_SHA256"
require_file_hash v2_submission_receipt "$V2_SUBMISSION_RECEIPT" \
  "$EXPECTED_V2_SUBMISSION_RECEIPT_SHA256"
require_file_hash v2_release_receipt "$V2_RELEASE_RECEIPT" \
  "$EXPECTED_V2_RELEASE_RECEIPT_SHA256"
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

# Release can start this allocation before the login-node helper has
# published the immutable release receipt. Wait only for that bounded
# control-plane handoff; no experiment or Python work occurs in this loop.
for _ in $(seq 1 60); do
  if [[ -f "$V3_SUBMISSION_RECEIPT" && \
    -f "$V3_SUBMISSION_RECEIPT_SHA" && \
    -f "$V3_RELEASE_RECEIPT" && -f "$V3_RELEASE_RECEIPT_SHA" ]]; then
    break
  fi
  sleep 1
done
verify_receipt_sha256 "$V3_SUBMISSION_RECEIPT" \
  "$V3_SUBMISSION_RECEIPT_SHA"
verify_receipt_sha256 "$V3_RELEASE_RECEIPT" "$V3_RELEASE_RECEIPT_SHA"
require_tsv_field "$V3_SUBMISSION_RECEIPT" schema_version \
  vlsa_postpublication_analysis_v3_submission_receipt.v1
require_tsv_field "$V3_SUBMISSION_RECEIPT" status held_validated
require_tsv_field "$V3_SUBMISSION_RECEIPT" run_id "$RUN_ID"
require_tsv_field "$V3_SUBMISSION_RECEIPT" job_id "$SLURM_JOB_ID"
require_tsv_field "$V3_SUBMISSION_RECEIPT" job_name "$EXPECTED_JOB_NAME"
require_tsv_field "$V3_SUBMISSION_RECEIPT" analysis_v2_job_id \
  "$V2_ANALYSIS_JOB_ID"
require_tsv_field "$V3_SUBMISSION_RECEIPT" dependency \
  "afterok:$V2_ANALYSIS_JOB_ID"
require_tsv_field "$V3_SUBMISSION_RECEIPT" source_git_commit \
  "$EXPECTED_SOURCE_GIT_COMMIT"
require_tsv_field "$V3_SUBMISSION_RECEIPT" analysis_v3_git_commit \
  "$EXPECTED_ANALYSIS_V3_GIT_COMMIT"
require_tsv_field "$V3_SUBMISSION_RECEIPT" output_dir "$V3_OUTPUT_DIR"
require_tsv_field "$V3_SUBMISSION_RECEIPT" partition main
require_tsv_field "$V3_SUBMISSION_RECEIPT" account normal
require_tsv_field "$V3_SUBMISSION_RECEIPT" qos normal
require_tsv_field "$V3_SUBMISSION_RECEIPT" nodes 1
require_tsv_field "$V3_SUBMISSION_RECEIPT" ntasks 1
require_tsv_field "$V3_SUBMISSION_RECEIPT" cpus_per_task 4
require_tsv_field "$V3_SUBMISSION_RECEIPT" memory 32G
require_tsv_field "$V3_SUBMISSION_RECEIPT" time_limit 04:00:00
require_tsv_field "$V3_SUBMISSION_RECEIPT" gpus 0
require_tsv_field "$V3_SUBMISSION_RECEIPT" exclude worker-3
require_tsv_field "$V3_SUBMISSION_RECEIPT" output_unused true
require_tsv_field "$V3_SUBMISSION_RECEIPT" cpu_only true
require_tsv_field "$V3_SUBMISSION_RECEIPT" held_before_release true
require_tsv_field "$V3_RELEASE_RECEIPT" schema_version \
  vlsa_postpublication_analysis_v3_release_receipt.v1
require_tsv_field "$V3_RELEASE_RECEIPT" status released
require_tsv_field "$V3_RELEASE_RECEIPT" run_id "$RUN_ID"
require_tsv_field "$V3_RELEASE_RECEIPT" job_id "$SLURM_JOB_ID"
require_tsv_field "$V3_RELEASE_RECEIPT" dependency \
  "afterok:$V2_ANALYSIS_JOB_ID"
require_tsv_field "$V3_RELEASE_RECEIPT" submission_receipt_sha256 \
  "$(sha256_file "$V3_SUBMISSION_RECEIPT")"
require_tsv_field "$V3_RELEASE_RECEIPT" output_unused_before_release true
case "$(tsv_field "$V3_RELEASE_RECEIPT" release_action)" in
  released_exact_held_job|recovered_after_prior_release) ;;
  *) die "analysis-v3 release receipt action is invalid" ;;
esac
for field in observed_state_before_action observed_reason_before_action \
  observed_state_after_action observed_reason_after_action; do
  [[ -n "$(tsv_field "$V3_RELEASE_RECEIPT" "$field")" ]] || \
    die "analysis-v3 release receipt $field is missing"
done
case "$(tsv_field "$V3_RELEASE_RECEIPT" observed_reason_after_action)" in
  *Held*) die "analysis-v3 release receipt retains a held state" ;;
esac
V3_SUBMISSION_RECEIPT_SHA256=$(sha256_file "$V3_SUBMISSION_RECEIPT")
V3_RELEASE_RECEIPT_SHA256=$(sha256_file "$V3_RELEASE_RECEIPT")

[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == \
  "$EXPECTED_ANALYSIS_V3_GIT_COMMIT" ]] || \
  die "repository commit changed during preflight"
[[ -z "$(git -C "$REMOTE_REPO" status --short --untracked-files=all)" ]] || \
  die "repository changed during preflight"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec "$AEGIS_PYTHON" "$V3_BUILDER" \
  --config "$CONFIG_PATH" \
  --manifest-receipt "$MANIFEST_RECEIPT_PATH" \
  --manifest "$MANIFEST_PATH" \
  --results-root "$RESULTS_ROOT" \
  --v1-publication-receipt "$V1_PUBLICATION_RECEIPT" \
  --prepublish-receipt "$PREPUBLISH_RECEIPT" \
  --summary-v2 "$SUMMARY_V2_PATH" \
  --analysis-v2-receipt "$ANALYSIS_V2_RECEIPT" \
  --analysis-v2-submission-receipt "$V2_SUBMISSION_RECEIPT" \
  --analysis-v2-release-receipt "$V2_RELEASE_RECEIPT" \
  --analysis-v3-submission-receipt "$V3_SUBMISSION_RECEIPT" \
  --analysis-v3-release-receipt "$V3_RELEASE_RECEIPT" \
  --analysis-v2-job-id "$V2_ANALYSIS_JOB_ID" \
  --analysis-v2-git-commit "$EXPECTED_V2_ANALYSIS_GIT_COMMIT" \
  --analysis-v3-job-id "$SLURM_JOB_ID" \
  --analysis-v3-git-commit "$EXPECTED_ANALYSIS_V3_GIT_COMMIT" \
  --expected-summary-v2-sha256 "$EXPECTED_SUMMARY_V2_SHA256" \
  --expected-analysis-v2-receipt-sha256 \
    "$EXPECTED_ANALYSIS_V2_RECEIPT_SHA256" \
  --expected-analysis-v2-submission-receipt-sha256 \
    "$EXPECTED_V2_SUBMISSION_RECEIPT_SHA256" \
  --expected-analysis-v2-release-receipt-sha256 \
    "$EXPECTED_V2_RELEASE_RECEIPT_SHA256" \
  --expected-analysis-v3-submission-receipt-sha256 \
    "$V3_SUBMISSION_RECEIPT_SHA256" \
  --expected-analysis-v3-release-receipt-sha256 \
    "$V3_RELEASE_RECEIPT_SHA256" \
  --expected-v1-publication-receipt-sha256 \
    "$EXPECTED_V1_PUBLICATION_RECEIPT_SHA256" \
  --expected-prepublish-receipt-sha256 \
    "$EXPECTED_PREPUBLISH_RECEIPT_SHA256" \
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --expected-manifest-receipt-sha256 \
    "$EXPECTED_MANIFEST_RECEIPT_SHA256" \
  --expected-v3-module-sha256 "$EXPECTED_V3_MODULE_SHA256" \
  --expected-builder-sha256 "$EXPECTED_V3_BUILDER_SHA256" \
  --expected-runner-sha256 "$EXPECTED_V3_RUNNER_SHA256" \
  --expected-sbatch-sha256 "$EXPECTED_V3_SBATCH_SHA256" \
  --expected-submit-helper-sha256 "$EXPECTED_V3_SUBMIT_HELPER_SHA256" \
  --output-root "$V3_OUTPUT_DIR"
