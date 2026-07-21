#!/usr/bin/env bash
set -euo pipefail

# Allocation-side runner for the exact postpublication analysis-v2 derivation.
# This job reads only the already-published compact population and writes a
# separate derived analysis directory.  It never runs inference, perception,
# rendering, simulation, model serving, or training.

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
readonly EXPECTED_AEGIS_PYTHON=/mnt/data/quanth/venvs/safety_vla/main/bin/python

: "${RUN_ID:?set the immutable population run ID}"
: "${EXPECTED_SOURCE_GIT_COMMIT:?set the exact population source commit}"
: "${EXPECTED_PUBLICATION_RECEIPT_SHA256:?set the terminal publisher receipt SHA-256}"
: "${EXPECTED_ANALYSIS_GIT_COMMIT:?set the exact reviewed analysis release commit}"
: "${EXPECTED_BUILDER_SHA256:?set the reviewed analysis-v2 builder SHA-256}"
: "${EXPECTED_RUNNER_SHA256:?set the reviewed allocation runner SHA-256}"
: "${EXPECTED_SBATCH_SHA256:?set the reviewed SBatch SHA-256}"
: "${EXPECTED_CONFIG_SHA256:?set the reviewed protocol config SHA-256}"
: "${EXPECTED_MANIFEST_SHA256:?set the reviewed population manifest SHA-256}"
: "${EXPECTED_MANIFEST_RECEIPT_SHA256:?set the reviewed manifest receipt SHA-256}"
: "${EXPECTED_V1_SUMMARY_SHA256:?set the terminal v1 summary SHA-256}"
: "${EXPECTED_PREPUBLISH_RECEIPT_SHA256:?set the terminal prepublish receipt SHA-256}"

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
AEGIS_PYTHON=${AEGIS_PYTHON:-$EXPECTED_AEGIS_PYTHON}
BUILDER=$REMOTE_REPO/analysis/build_safelibero_postpublication_v2.py
RUNNER=$REMOTE_REPO/slurm/run_vlsa_postpublication_analysis_v2.sh
SBATCH=$REMOTE_REPO/slurm/vlsa_postpublication_analysis_v2.sbatch

die() {
  echo "postpublication analysis-v2 runner rejected: $*" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha256() {
  local name=$1
  local value=$2
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die "$name is not lowercase SHA-256"
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

[[ "$RUN_ID" == "$POPULATION_RUN_ID" ]] || \
  die "run ID differs from the exact population"
[[ "$PUBLISHER_JOB_ID" =~ ^[0-9]+$ ]] || \
  die "terminal publisher job ID is not numeric"
[[ "$ARTIFACT_PUBLISHER_JOB_ID" =~ ^[0-9]+$ ]] || \
  die "artifact publisher job ID is not numeric"
if [[ "$ARTIFACT_PUBLISHER_JOB_ID" != "$PUBLISHER_JOB_ID" ]]; then
  [[ "$ARTIFACT_PUBLISHER_JOB_ID" == \
    "$TIMEOUT_ARTIFACT_PUBLISHER_JOB_ID" ]] || \
    die "split publication is allowed only for timed-out artifact publisher 28940"
fi
[[ "$EXPECTED_SOURCE_GIT_COMMIT" == "$POPULATION_SOURCE_GIT_COMMIT" ]] || \
  die "population source commit differs from the exact runtime"
[[ "$REMOTE_REPO" == "$EXPECTED_REMOTE_REPO" ]] || \
  die "repository path differs from the reviewed release"
[[ "$EXPERIMENT_ROOT" == "$EXPECTED_EXPERIMENT_ROOT" ]] || \
  die "experiment root differs from the exact population"
[[ "$RUN_ROOT" == "$EXPECTED_EXPERIMENT_ROOT/$POPULATION_RUN_ID" ]] || \
  die "run root differs from the exact population"
[[ "$RESULTS_ROOT" == "$RUN_ROOT/tasks" ]] || \
  die "results root differs from the immutable task result tree"
[[ "$PUBLICATION_RECEIPT" == "$RUN_ROOT/population-publication-receipt.json" ]] || \
  die "publication receipt path differs from the exact population"
[[ "$V1_SUMMARY_PATH" == \
  "$RUN_ROOT/publication-attempts/job-$ARTIFACT_PUBLISHER_JOB_ID/population-summary.json" ]] || \
  die "v1 summary path differs from artifact publisher $ARTIFACT_PUBLISHER_JOB_ID"
[[ "$PREPUBLISH_RECEIPT_PATH" == \
  "$RUN_ROOT/publication-attempts/job-$ARTIFACT_PUBLISHER_JOB_ID/prepublish-validation.json" ]] || \
  die "prepublish receipt path differs from artifact publisher $ARTIFACT_PUBLISHER_JOB_ID"
[[ "$CONFIG_PATH" == "$REMOTE_REPO/configs/vlsa_table1_translational.json" ]] || \
  die "config path differs from the reviewed protocol"
[[ "$MANIFEST_PATH" == "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" ]] || \
  die "manifest path differs from the immutable population"
[[ "$MANIFEST_RECEIPT_PATH" == \
  "$REMOTE_REPO/manifests/vlsa_table1_population.receipt.json" ]] || \
  die "manifest receipt path differs from the immutable population"
[[ "$OUTPUT_ROOT" == "$EXPECTED_OUTPUT_ROOT" ]] || \
  die "analysis output root differs from the reviewed destination"
[[ "$OUTPUT_DIR" == \
  "$EXPECTED_OUTPUT_ROOT/$POPULATION_RUN_ID-publisher-$PUBLISHER_JOB_ID" ]] || \
  die "analysis output directory differs from the reviewed destination"
[[ "$AEGIS_PYTHON" == "$EXPECTED_AEGIS_PYTHON" ]] || \
  die "Python path differs from the reviewed allocation environment"

for value in \
  "$EXPECTED_PUBLICATION_RECEIPT_SHA256" \
  "$EXPECTED_BUILDER_SHA256" \
  "$EXPECTED_RUNNER_SHA256" \
  "$EXPECTED_SBATCH_SHA256" \
  "$EXPECTED_CONFIG_SHA256" \
  "$EXPECTED_MANIFEST_SHA256" \
  "$EXPECTED_MANIFEST_RECEIPT_SHA256" \
  "$EXPECTED_V1_SUMMARY_SHA256" \
  "$EXPECTED_PREPUBLISH_RECEIPT_SHA256"; do
  require_sha256 immutable_hash "$value"
done
[[ "$EXPECTED_ANALYSIS_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  die "analysis release commit is not 40 lowercase hex"

for name in SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_DEPENDENCY \
  SLURMD_NODENAME SLURM_JOB_PARTITION SLURM_CPUS_PER_TASK \
  SLURM_MEM_PER_NODE; do
  [[ -n "${!name:-}" ]] || die "missing allocation field $name"
done
[[ "$SLURM_JOB_DEPENDENCY" == "afterok:$PUBLISHER_JOB_ID" ]] || \
  die "dependency must be exactly afterok:$PUBLISHER_JOB_ID"
[[ "$SLURM_JOB_NAME" == "$EXPECTED_JOB_NAME" ]] || \
  die "job name differs from the held reviewed launch"
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

publisher_rows=$(
  sacct -n -X -j "$PUBLISHER_JOB_ID" \
    --format=JobIDRaw,State,ExitCode -P |
    awk -F'|' -v job="$PUBLISHER_JOB_ID" \
      '$1 == job {print $2 "|" $3}'
)
[[ "$publisher_rows" == "COMPLETED|0:0" ]] || \
  die "publisher $PUBLISHER_JOB_ID is not exactly COMPLETED with exit 0:0"

[[ -d "$REMOTE_REPO/.git" || -f "$REMOTE_REPO/.git" ]] || \
  die "reviewed repository is missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == \
  "$EXPECTED_ANALYSIS_GIT_COMMIT" ]] || \
  die "repository is not at the reviewed analysis release commit"
[[ -z "$(git -C "$REMOTE_REPO" status --short --untracked-files=all)" ]] || \
  die "reviewed repository is dirty"
[[ -x "$AEGIS_PYTHON" ]] || die "Python interpreter is not executable"
[[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || \
  die "immutable run root is missing or symlinked"
[[ -d "$RESULTS_ROOT" && ! -L "$RESULTS_ROOT" ]] || \
  die "immutable results root is missing or symlinked"
[[ -d "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || \
  die "analysis output root must already exist as a real directory"
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || \
  die "analysis-v2 output directory already exists"

require_file_hash builder "$BUILDER" "$EXPECTED_BUILDER_SHA256"
require_file_hash runner "$RUNNER" "$EXPECTED_RUNNER_SHA256"
require_file_hash SBatch "$SBATCH" "$EXPECTED_SBATCH_SHA256"
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

# Recheck the release identity after all immutable inputs have been hashed.
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == \
  "$EXPECTED_ANALYSIS_GIT_COMMIT" ]] || \
  die "repository commit changed during allocation preflight"
[[ -z "$(git -C "$REMOTE_REPO" status --short --untracked-files=all)" ]] || \
  die "repository changed during allocation preflight"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec "$AEGIS_PYTHON" "$BUILDER" \
  --config "$CONFIG_PATH" \
  --manifest-receipt "$MANIFEST_RECEIPT_PATH" \
  --manifest "$MANIFEST_PATH" \
  --results-root "$RESULTS_ROOT" \
  --v1-publication-receipt "$PUBLICATION_RECEIPT" \
  --v1-summary "$V1_SUMMARY_PATH" \
  --prepublish-receipt "$PREPUBLISH_RECEIPT_PATH" \
  --output-root "$OUTPUT_DIR"
