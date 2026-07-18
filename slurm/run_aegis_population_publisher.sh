#!/usr/bin/env bash
set -euo pipefail

# Submit this exact CPU publisher with:
#   sbatch --dependency=afterany:<population-array-job-id> \
#     --export=ALL,RUN_ID=...,EXPECTED_GIT_COMMIT=...,\
# POPULATION_ARRAY_JOB_ID=...,LABEL_MANIFEST_PATH=...,\
# PI05_HASH_RECEIPT_PATH=...,PAIRED_CANARY_RECEIPT_PATH=...,\
# EXPECTED_PI05_HASH_RECEIPT_SHA256=...,\
# EXPECTED_PAIRED_CANARY_RECEIPT_SHA256=...,GROUNDINGDINO_DEVICE=cpu \
#     slurm/aegis_population_publisher.sbatch
#
# It never runs inference, rendering, a simulator, or training.

: "${RUN_ID:?set the immutable population RUN_ID}"
: "${EXPECTED_GIT_COMMIT:?set the population release commit}"
: "${POPULATION_ARRAY_JOB_ID:?set the exact completed population array job ID}"
: "${LABEL_MANIFEST_PATH:?set the complete frozen label manifest}"
: "${PI05_HASH_RECEIPT_PATH:?set the allocation-backed pi0.5 hash receipt}"
: "${PAIRED_CANARY_RECEIPT_PATH:?set the validated paired-canary receipt}"
: "${EXPECTED_PI05_HASH_RECEIPT_SHA256:?set the reviewed pi0.5 receipt SHA-256}"
: "${EXPECTED_PAIRED_CANARY_RECEIPT_SHA256:?set the reviewed canary receipt SHA-256}"
: "${GROUNDINGDINO_DEVICE:?set the exact paired-canary device (cpu)}"

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ "$GROUNDINGDINO_DEVICE" == cpu ]] || {
  echo "GROUNDINGDINO_DEVICE must be exactly cpu" >&2
  exit 2
}

case "$POPULATION_ARRAY_JOB_ID" in
  ""|*[!0-9]*)
    echo "POPULATION_ARRAY_JOB_ID must be one exact numeric array job ID" >&2
    exit 2
    ;;
esac

# shellcheck source=slurm/aegis_runtime_common.sh
source "$REMOTE_REPO/slurm/aegis_runtime_common.sh"

[[ "$(aegis_sha256 "$PI05_HASH_RECEIPT_PATH")" == \
  "$EXPECTED_PI05_HASH_RECEIPT_SHA256" ]] || {
  aegis_die "pi0.5 receipt differs from its reviewed SHA-256"
}
[[ "$(aegis_sha256 "$PAIRED_CANARY_RECEIPT_PATH")" == \
  "$EXPECTED_PAIRED_CANARY_RECEIPT_SHA256" ]] || {
  aegis_die "paired-canary receipt differs from its reviewed SHA-256"
}

for name in SLURM_JOB_ID SLURM_JOB_DEPENDENCY SLURMD_NODENAME \
  SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE; do
  aegis_require_env "$name"
done
[[ "$SLURM_JOB_DEPENDENCY" == "afterany:$POPULATION_ARRAY_JOB_ID" ]] || {
  aegis_die "publisher dependency must be exactly afterany:$POPULATION_ARRAY_JOB_ID"
}
case "$SLURMD_NODENAME" in
  worker-3|login*|login-restricted*)
    aegis_die "population publisher cannot execute on $SLURMD_NODENAME"
    ;;
esac
[[ "$SLURM_CPUS_PER_TASK" == 4 && "$SLURM_MEM_PER_NODE" == 32768 ]] || {
  aegis_die "publisher allocation changed from 4 CPUs / 32768 MiB"
}
[[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES:-}" == NoDevFiles ]] || {
  aegis_die "population publisher must not consume a GPU"
}

aegis_validate_identity population
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

PUBLISH_ATTEMPTS_ROOT=$RUN_ROOT/publication-attempts
mkdir -p "$PUBLISH_ATTEMPTS_ROOT"
PUBLISH_ATTEMPT_ROOT=$PUBLISH_ATTEMPTS_ROOT/job-$SLURM_JOB_ID
mkdir "$PUBLISH_ATTEMPT_ROOT" || {
  aegis_die "publisher attempt already exists: $PUBLISH_ATTEMPT_ROOT"
}
PUBLISH_STAGE=slurm_accounting
publisher_cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ $exit_code -ne 0 && ! -e "$PUBLISH_ATTEMPT_ROOT/publisher-failure.json" ]]; then
    local temporary
    temporary=$(mktemp "$PUBLISH_ATTEMPT_ROOT/.publisher-failure.XXXXXX")
    {
      printf '{\n'
      printf '  "schema_version": "vlsa_table1_population_publisher_failure.v1",\n'
      printf '  "status": "apparatus_failure",\n'
      printf '  "scientific_result": false,\n'
      printf '  "run_id": "%s",\n' "$RUN_ID"
      printf '  "population_array_job_id": "%s",\n' "$POPULATION_ARRAY_JOB_ID"
      printf '  "publisher_job_id": "%s",\n' "$SLURM_JOB_ID"
      printf '  "host": "%s",\n' "$SLURMD_NODENAME"
      printf '  "failure_stage": "%s",\n' "$PUBLISH_STAGE"
      printf '  "exit_code": %d\n' "$exit_code"
      printf '}\n'
    } >"$temporary"
    mv "$temporary" "$PUBLISH_ATTEMPT_ROOT/publisher-failure.json"
  fi
  return "$exit_code"
}
trap publisher_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ACCOUNTING_PATH=$PUBLISH_ATTEMPT_ROOT/population-array-sacct.txt
sacct -n -j "$POPULATION_ARRAY_JOB_ID" \
  --format=JobID,JobIDRaw,State -P >"$ACCOUNTING_PATH"

VALIDATOR=$REMOTE_REPO/scripts/validate_aegis_run_artifacts.py
PREPUBLISH_RECEIPT=$PUBLISH_ATTEMPT_ROOT/prepublish-validation.json
PUBLISH_STAGE=population_prepublish_validation
"$AEGIS_PYTHON" "$VALIDATOR" population-prepublish \
  --run-root "$RUN_ROOT" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --config "$CONFIG_PATH" \
  --manifest "$MANIFEST_PATH" \
  --manifest-receipt "$MANIFEST_RECEIPT_PATH" \
  --labels "$LABEL_MANIFEST_PATH" \
  --pi05-hash-receipt "$PI05_HASH_RECEIPT_PATH" \
  --paired-canary-receipt "$PAIRED_CANARY_RECEIPT_PATH" \
  --population-array-job-id "$POPULATION_ARRAY_JOB_ID" \
  --slurm-accounting "$ACCOUNTING_PATH" \
  --output "$PREPUBLISH_RECEIPT"

SUMMARY_PATH=$PUBLISH_ATTEMPT_ROOT/population-summary.json
PUBLISH_STAGE=strict_population_aggregation
"$AEGIS_PYTHON" "$REMOTE_REPO/analysis/aggregate_safelibero_aegis.py" \
  --config "$CONFIG_PATH" \
  --receipt "$MANIFEST_RECEIPT_PATH" \
  --manifest "$MANIFEST_PATH" \
  --results "$RUN_ROOT/tasks" \
  --output "$SUMMARY_PATH"

GALLERY_ROOT=$PUBLISH_ATTEMPT_ROOT/gallery
mkdir "$GALLERY_ROOT"
GALLERY_PATH=$GALLERY_ROOT/index.html
PUBLISH_STAGE=strict_population_gallery
"$AEGIS_PYTHON" "$REMOTE_REPO/analysis/build_safelibero_video_gallery.py" \
  --summary "$SUMMARY_PATH" \
  --results "$RUN_ROOT/tasks" \
  --output-root "$GALLERY_ROOT" \
  --index-name index.html

PUBLISH_STAGE=publication_receipt
"$AEGIS_PYTHON" "$VALIDATOR" population-finalize \
  --run-root "$RUN_ROOT" \
  --prepublish-receipt "$PREPUBLISH_RECEIPT" \
  --summary "$SUMMARY_PATH" \
  --gallery "$GALLERY_PATH" \
  --output "$RUN_ROOT/population-publication-receipt.json"

PUBLISH_STAGE=complete
