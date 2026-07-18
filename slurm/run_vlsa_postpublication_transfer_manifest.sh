#!/usr/bin/env bash
set -euo pipefail

# Allocation-side runner for the exact current-v2 postpublication inventory.
# This runner performs only filesystem validation, hashing, and compact JSON
# publication.  It never runs policy inference, perception, rendering,
# simulation, metrics rollouts, model serving, or training.

readonly POPULATION_ARRAY_JOB_ID=28609
readonly PUBLISHER_JOB_ID=28610
readonly POPULATION_RUN_ID=vlsa-table1-contact-authority-population-20260718a
readonly POPULATION_SOURCE_GIT_COMMIT=1592aa59361f431ba96c6ddcbebcb596f6c20853
readonly EXPECTED_JOB_NAME=vlsa-tx-p28610

: "${RUN_ID:?set the immutable population run ID}"
: "${EXPECTED_SOURCE_GIT_COMMIT:?set the exact population source commit}"
: "${EXPECTED_PUBLICATION_RECEIPT_SHA256:?set the reviewed publication receipt SHA-256}"
: "${EXPECTED_VERIFIER_SHA256:?set the reviewed verifier script SHA-256}"
: "${EXPECTED_VERIFIER_GIT_COMMIT:?set the exact verifier repository commit}"
: "${EXPECTED_RUNNER_SHA256:?set the reviewed allocation runner SHA-256}"
: "${EXPECTED_SBATCH_SHA256:?set the reviewed SBatch SHA-256}"
: "${LABEL_MANIFEST_PATH:?set the immutable 1,600-row label manifest}"

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
RUN_ROOT=${RUN_ROOT:-$EXPERIMENT_ROOT/$RUN_ID}
PUBLICATION_RECEIPT=${PUBLICATION_RECEIPT:-$RUN_ROOT/population-publication-receipt.json}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_table1_translational.json}
MANIFEST_PATH=${MANIFEST_PATH:-$REMOTE_REPO/manifests/vlsa_table1_population.jsonl}
MANIFEST_RECEIPT_PATH=${MANIFEST_RECEIPT_PATH:-$REMOTE_REPO/manifests/vlsa_table1_population.receipt.json}
TRANSFER_OUTPUT_ROOT=${TRANSFER_OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1-transfer}
TRANSFER_OUTPUT_DIR=${TRANSFER_OUTPUT_DIR:-$TRANSFER_OUTPUT_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID}
VERIFIER=$REMOTE_REPO/scripts/vlsa_postpublication_transfer_manifest.py
RUNNER=$REMOTE_REPO/slurm/run_vlsa_postpublication_transfer_manifest.sh
SBATCH=$REMOTE_REPO/slurm/vlsa_postpublication_transfer_manifest.sbatch

die() {
  echo "postpublication transfer verifier: $*" >&2
  exit 2
}

[[ "$RUN_ID" == "$POPULATION_RUN_ID" ]] || \
  die "run ID differs from the exact population"
[[ "$EXPECTED_SOURCE_GIT_COMMIT" == "$POPULATION_SOURCE_GIT_COMMIT" ]] || \
  die "source commit differs from the exact population"

for name in SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_DEPENDENCY \
  SLURMD_NODENAME SLURM_JOB_PARTITION SLURM_CPUS_PER_TASK \
  SLURM_MEM_PER_NODE; do
  [[ -n "${!name:-}" ]] || die "missing allocation field $name"
done

case "$SLURMD_NODENAME" in
  worker-3|login*|login-restricted*)
    die "cannot execute on $SLURMD_NODENAME"
    ;;
esac
[[ "$SLURM_JOB_DEPENDENCY" == "afterok:$PUBLISHER_JOB_ID" ]] || \
  die "dependency must be exactly afterok:$PUBLISHER_JOB_ID"
[[ "$SLURM_JOB_NAME" == "$EXPECTED_JOB_NAME" ]] || \
  die "job name differs from the held reviewed launch"
[[ "$SLURM_JOB_PARTITION" == main ]] || die "partition must be main"
[[ "$SLURM_CPUS_PER_TASK" == 4 ]] || die "requires exactly 4 CPUs"
[[ "$SLURM_MEM_PER_NODE" == 32768 ]] || \
  die "requires exactly 32768 MiB RAM"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" || \
  "${CUDA_VISIBLE_DEVICES:-}" == NoDevFiles || \
  "${CUDA_VISIBLE_DEVICES:-}" == -1 ]] || \
  die "must not receive CUDA devices"
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || \
  die "must not receive a Slurm GPU allocation"

[[ -x "$AEGIS_PYTHON" ]] || die "Python interpreter is not executable"
for path in "$VERIFIER" "$RUNNER" "$SBATCH"; do
  [[ -f "$path" && ! -L "$path" ]] || \
    die "reviewed source is missing or symlinked: $path"
done
[[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || \
  die "immutable run root is missing or symlinked"
[[ -f "$PUBLICATION_RECEIPT" && ! -L "$PUBLICATION_RECEIPT" ]] || \
  die "publication receipt is missing or symlinked"
[[ -d "$TRANSFER_OUTPUT_ROOT" && ! -L "$TRANSFER_OUTPUT_ROOT" ]] || \
  die "transfer output root must already exist and be a real directory"
[[ ! -e "$TRANSFER_OUTPUT_DIR" && ! -L "$TRANSFER_OUTPUT_DIR" ]] || \
  die "transfer output directory already exists"

observed_verifier_sha256=$(sha256sum "$VERIFIER" | awk '{print $1}')
[[ "$observed_verifier_sha256" == "$EXPECTED_VERIFIER_SHA256" ]] || \
  die "verifier differs from its reviewed SHA-256"
observed_runner_sha256=$(sha256sum "$RUNNER" | awk '{print $1}')
[[ "$observed_runner_sha256" == "$EXPECTED_RUNNER_SHA256" ]] || \
  die "runner differs from its reviewed SHA-256"
observed_sbatch_sha256=$(sha256sum "$SBATCH" | awk '{print $1}')
[[ "$observed_sbatch_sha256" == "$EXPECTED_SBATCH_SHA256" ]] || \
  die "SBatch differs from its reviewed SHA-256"
observed_commit=$(git -C "$REMOTE_REPO" rev-parse HEAD)
[[ "$observed_commit" == "$EXPECTED_VERIFIER_GIT_COMMIT" ]] || \
  die "verifier repository is not at the reviewed commit"
[[ -z "$(git -C "$REMOTE_REPO" status --short --untracked-files=all)" ]] || \
  die "verifier repository is dirty"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec "$AEGIS_PYTHON" "$VERIFIER" \
  --run-root "$RUN_ROOT" \
  --publication-receipt "$PUBLICATION_RECEIPT" \
  --expected-publication-receipt-sha256 \
    "$EXPECTED_PUBLICATION_RECEIPT_SHA256" \
  --expected-run-id "$RUN_ID" \
  --expected-source-commit "$EXPECTED_SOURCE_GIT_COMMIT" \
  --expected-population-array-job-id "$POPULATION_ARRAY_JOB_ID" \
  --expected-publisher-job-id "$PUBLISHER_JOB_ID" \
  --expected-verifier-sha256 "$EXPECTED_VERIFIER_SHA256" \
  --expected-verifier-git-commit "$EXPECTED_VERIFIER_GIT_COMMIT" \
  --config "$CONFIG_PATH" \
  --manifest "$MANIFEST_PATH" \
  --manifest-receipt "$MANIFEST_RECEIPT_PATH" \
  --labels "$LABEL_MANIFEST_PATH" \
  --output-dir "$TRANSFER_OUTPUT_DIR"
