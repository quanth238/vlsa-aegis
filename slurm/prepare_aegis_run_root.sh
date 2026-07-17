#!/usr/bin/env bash
set -euo pipefail

# Control-plane-only reservation.  This script does not submit a job and does
# not execute Python, rendering, inference, metrics, or model serving.

: "${RUN_ID:?set an immutable RUN_ID}"
: "${RUN_STAGE:?set RUN_STAGE (capture-canary, capture-population, paired-canary, or population)}"
: "${EXPECTED_GIT_COMMIT:?set the reviewed 40-character source commit}"

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_table1_translational.json}
MANIFEST_PATH=${MANIFEST_PATH:-$REMOTE_REPO/manifests/vlsa_table1_population.jsonl}
MANIFEST_RECEIPT_PATH=${MANIFEST_RECEIPT_PATH:-$REMOTE_REPO/manifests/vlsa_table1_population.receipt.json}
LABEL_MANIFEST_PATH=${LABEL_MANIFEST_PATH:-}
GROUNDINGDINO_DEVICE=${GROUNDINGDINO_DEVICE:-}
CASE_ORDINAL=${CASE_ORDINAL:-0}
EXPECTED_PI05_TREE_SHA256=${EXPECTED_PI05_TREE_SHA256:-}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "run-root reservation belongs on the control plane, before sbatch" >&2
  exit 2
fi
case "$RUN_ID" in
  ""|*[!A-Za-z0-9._-]*) echo "unsafe RUN_ID: $RUN_ID" >&2; exit 2 ;;
esac
case "$RUN_STAGE" in
  capture-canary|capture-population|paired-canary|population) ;;
  *) echo "unsupported RUN_STAGE: $RUN_STAGE" >&2; exit 2 ;;
esac
if [[ "$RUN_STAGE" == capture-canary || "$RUN_STAGE" == paired-canary ]]; then
  [[ "$CASE_ORDINAL" =~ ^[0-9]+$ ]] || {
    echo "canary CASE_ORDINAL must be numeric" >&2
    exit 2
  }
  (( CASE_ORDINAL >= 0 && CASE_ORDINAL < 1600 )) || {
    echo "canary CASE_ORDINAL must be in [0, 1599]" >&2
    exit 2
  }
  contract_case_ordinal=$CASE_ORDINAL
else
  contract_case_ordinal=all
fi
case "$EXPECTED_GIT_COMMIT" in
  *[!0-9a-f]*|"") echo "EXPECTED_GIT_COMMIT is not lowercase hexadecimal" >&2; exit 2 ;;
esac
if [[ ${#EXPECTED_GIT_COMMIT} -ne 40 ]]; then
  echo "EXPECTED_GIT_COMMIT must contain exactly 40 characters" >&2
  exit 2
fi
case "$EXPERIMENT_ROOT" in
  /mnt/data/quanth/experiments/*) ;;
  *) echo "EXPERIMENT_ROOT must remain under /mnt/data/quanth/experiments" >&2; exit 2 ;;
esac

for command in git sha256sum awk date mkdir mv mktemp; do
  command -v "$command" >/dev/null || {
    echo "missing control-plane command: $command" >&2
    exit 2
  }
done
for path in "$CONFIG_PATH" "$MANIFEST_PATH" "$MANIFEST_RECEIPT_PATH"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "missing or symlinked frozen input: $path" >&2
    exit 2
  }
done
if [[ "$RUN_STAGE" == paired-canary || "$RUN_STAGE" == population ]]; then
  [[ -n "$LABEL_MANIFEST_PATH" && -f "$LABEL_MANIFEST_PATH" && ! -L "$LABEL_MANIFEST_PATH" ]] || {
    echo "evaluation run requires LABEL_MANIFEST_PATH" >&2
    exit 2
  }
  case "$GROUNDINGDINO_DEVICE" in
    cpu|cuda) ;;
    *)
      echo "evaluation requires an explicit GROUNDINGDINO_DEVICE=cpu or cuda" >&2
      exit 2
      ;;
  esac
  if [[ -n "$EXPECTED_PI05_TREE_SHA256" ]]; then
    case "$EXPECTED_PI05_TREE_SHA256" in
      *[!0-9a-f]*|"") echo "EXPECTED_PI05_TREE_SHA256 is invalid" >&2; exit 2 ;;
    esac
    [[ ${#EXPECTED_PI05_TREE_SHA256} -eq 64 ]] || {
      echo "EXPECTED_PI05_TREE_SHA256 must contain 64 characters" >&2
      exit 2
    }
    pi05_tree_sha256=$EXPECTED_PI05_TREE_SHA256
  else
    pi05_tree_sha256=structural-and-metadata-only
  fi
elif [[ -n "$LABEL_MANIFEST_PATH" ]]; then
  echo "capture reservation must not bind an outcome-stage label manifest" >&2
  exit 2
fi
if [[ "$RUN_STAGE" == capture-canary || "$RUN_STAGE" == capture-population ]]; then
  pi05_tree_sha256=none
fi

observed_commit=$(git -C "$REMOTE_REPO" rev-parse HEAD)
[[ "$observed_commit" == "$EXPECTED_GIT_COMMIT" ]] || {
  echo "remote source is $observed_commit, expected $EXPECTED_GIT_COMMIT" >&2
  exit 2
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "remote source tree must be clean before run reservation" >&2
  exit 2
}

config_sha256=$(sha256sum "$CONFIG_PATH" | awk '{print $1}')
manifest_sha256=$(sha256sum "$MANIFEST_PATH" | awk '{print $1}')
receipt_sha256=$(sha256sum "$MANIFEST_RECEIPT_PATH" | awk '{print $1}')
if [[ -n "$LABEL_MANIFEST_PATH" ]]; then
  label_manifest_sha256=$(sha256sum "$LABEL_MANIFEST_PATH" | awk '{print $1}')
else
  label_manifest_sha256=none
fi

mkdir -p "$EXPERIMENT_ROOT"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir "$RUN_ROOT" || {
  echo "immutable run root already exists: $RUN_ROOT" >&2
  exit 2
}
contract_complete=false
cleanup_incomplete_root() {
  if [[ "$contract_complete" != true ]]; then
    rmdir "$RUN_ROOT" 2>/dev/null || true
  fi
}
trap cleanup_incomplete_root EXIT

temporary=$(mktemp "$RUN_ROOT/.run-contract.XXXXXX")
{
  printf 'schema_version\t%s\n' vlsa_table1_run_contract.v1
  printf 'run_id\t%s\n' "$RUN_ID"
  printf 'run_stage\t%s\n' "$RUN_STAGE"
  printf 'case_ordinal\t%s\n' "$contract_case_ordinal"
  printf 'git_commit\t%s\n' "$EXPECTED_GIT_COMMIT"
  printf 'config_sha256\t%s\n' "$config_sha256"
  printf 'manifest_sha256\t%s\n' "$manifest_sha256"
  printf 'manifest_receipt_sha256\t%s\n' "$receipt_sha256"
  printf 'label_manifest_sha256\t%s\n' "$label_manifest_sha256"
  printf 'pi05_tree_sha256\t%s\n' "$pi05_tree_sha256"
  if [[ "$RUN_STAGE" == paired-canary || "$RUN_STAGE" == population ]]; then
    printf 'groundingdino_device\t%s\n' "$GROUNDINGDINO_DEVICE"
  else
    printf 'groundingdino_device\t%s\n' none
  fi
  printf 'created_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'job_submission_performed_by_reservation_script\t%s\n' false
} >"$temporary"
mv "$temporary" "$RUN_ROOT/run-contract.tsv"
chmod a-w "$RUN_ROOT/run-contract.tsv"
contract_complete=true
trap - EXIT

printf 'reserved_run_root=%s\n' "$RUN_ROOT"
printf 'run_contract=%s\n' "$RUN_ROOT/run-contract.tsv"
printf 'no_job_submitted=true\n'
