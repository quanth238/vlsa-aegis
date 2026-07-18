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
LABEL_PUBLICATION_RECEIPT_PATH=${LABEL_PUBLICATION_RECEIPT_PATH:-}
GROUNDINGDINO_DEVICE=${GROUNDINGDINO_DEVICE:-}
CASE_ORDINAL=${CASE_ORDINAL:-0}
EXPECTED_PI05_TREE_SHA256=${EXPECTED_PI05_TREE_SHA256:-}
PI05_HASH_RECEIPT_PATH=${PI05_HASH_RECEIPT_PATH:-}
PAIRED_CANARY_RECEIPT_PATH=${PAIRED_CANARY_RECEIPT_PATH:-}
EXPECTED_PI05_HASH_RECEIPT_SHA256=${EXPECTED_PI05_HASH_RECEIPT_SHA256:-}
EXPECTED_PAIRED_CANARY_RECEIPT_SHA256=${EXPECTED_PAIRED_CANARY_RECEIPT_SHA256:-}

json_string_value() {
  local path=$1
  local key=$2
  awk -F '"' -v wanted="$key" '
    $2 == wanted {
      if (seen++) exit 3
      value=$4
    }
    END {
      if (seen != 1) exit 4
      print value
    }
  ' "$path"
}

json_top_level_string_value() {
  local path=$1
  local key=$2
  awk -F '"' -v wanted="$key" '
    substr($0, 1, 3) == "  \"" && $2 == wanted {
      if (seen++) exit 3
      value=$4
    }
    END {
      if (seen != 1) exit 4
      print value
    }
  ' "$path"
}

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
  if [[ "$RUN_STAGE" == paired-canary && "$CASE_ORDINAL" != 100 ]]; then
    echo "paired canary is frozen to case ordinal 100" >&2
    exit 2
  fi
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
  [[ "$GROUNDINGDINO_DEVICE" == cpu ]] || {
    echo "evaluation is frozen to GROUNDINGDINO_DEVICE=cpu" >&2
    exit 2
  }
  [[ -n "$PI05_HASH_RECEIPT_PATH" && -f "$PI05_HASH_RECEIPT_PATH" && ! -L "$PI05_HASH_RECEIPT_PATH" ]] || {
    echo "evaluation requires an allocation-backed PI05_HASH_RECEIPT_PATH" >&2
    exit 2
  }
  [[ -n "$LABEL_PUBLICATION_RECEIPT_PATH" && \
    -f "$LABEL_PUBLICATION_RECEIPT_PATH" && \
    ! -L "$LABEL_PUBLICATION_RECEIPT_PATH" ]] || {
    echo "evaluation requires LABEL_PUBLICATION_RECEIPT_PATH" >&2
    exit 2
  }
  [[ "$(sha256sum "$LABEL_MANIFEST_PATH" | awk '{print $1}')" == \
    f9a862f28f168f02de4e0987e37d297de24b167ae50fb96c7f8243a76916880e ]] || {
    echo "evaluation requires the exact frozen 1,600-row label manifest" >&2
    exit 2
  }
  label_publication_receipt_sha256=$(
    sha256sum "$LABEL_PUBLICATION_RECEIPT_PATH" | awk '{print $1}'
  )
  [[ "$label_publication_receipt_sha256" == \
    e83611f46ce5fbb13c84f74db3825ab114bf7184db96b62be2965c7a0c5b9e20 ]] || {
    echo "frozen label publication receipt SHA-256 changed" >&2
    exit 2
  }
  case "$EXPECTED_PI05_HASH_RECEIPT_SHA256" in
    *[!0-9a-f]*|"")
      echo "evaluation requires a reviewed EXPECTED_PI05_HASH_RECEIPT_SHA256" >&2
      exit 2
      ;;
  esac
  [[ ${#EXPECTED_PI05_HASH_RECEIPT_SHA256} -eq 64 ]] || {
    echo "EXPECTED_PI05_HASH_RECEIPT_SHA256 must contain 64 characters" >&2
    exit 2
  }
  [[ "$(json_top_level_string_value "$PI05_HASH_RECEIPT_PATH" schema_version)" == \
    vlsa_table1_pi05_hash_receipt.v1 ]] || {
    echo "unexpected pi0.5 hash receipt schema" >&2
    exit 2
  }
  [[ "$(json_top_level_string_value "$PI05_HASH_RECEIPT_PATH" status)" == passed ]] || {
    echo "pi0.5 hash receipt did not pass" >&2
    exit 2
  }
  [[ "$(json_string_value "$PI05_HASH_RECEIPT_PATH" git_commit)" == \
    "$EXPECTED_GIT_COMMIT" ]] || {
    echo "pi0.5 hash receipt source commit differs" >&2
    exit 2
  }
  grep -Eq '^[[:space:]]*"full_content_hash_verified":[[:space:]]*true,?[[:space:]]*$' \
    "$PI05_HASH_RECEIPT_PATH" || {
    echo "pi0.5 hash receipt lacks full content verification" >&2
    exit 2
  }
  pi05_tree_sha256=$(
    json_string_value "$PI05_HASH_RECEIPT_PATH" full_content_tree_sha256
  )
  case "$pi05_tree_sha256" in
    *[!0-9a-f]*|"") echo "pi0.5 receipt tree SHA-256 is invalid" >&2; exit 2 ;;
  esac
  [[ ${#pi05_tree_sha256} -eq 64 ]] || {
    echo "pi0.5 receipt tree SHA-256 must contain 64 characters" >&2
    exit 2
  }
  if [[ -n "$EXPECTED_PI05_TREE_SHA256" && \
    "$EXPECTED_PI05_TREE_SHA256" != "$pi05_tree_sha256" ]]; then
    echo "EXPECTED_PI05_TREE_SHA256 differs from the hash receipt" >&2
    exit 2
  fi
  pi05_hash_receipt_sha256=$(sha256sum "$PI05_HASH_RECEIPT_PATH" | awk '{print $1}')
  [[ "$pi05_hash_receipt_sha256" == "$EXPECTED_PI05_HASH_RECEIPT_SHA256" ]] || {
    echo "pi0.5 hash receipt differs from its reviewed SHA-256" >&2
    exit 2
  }
elif [[ -n "$LABEL_MANIFEST_PATH" ]]; then
  echo "capture reservation must not bind an outcome-stage label manifest" >&2
  exit 2
fi
if [[ "$RUN_STAGE" == capture-canary || "$RUN_STAGE" == capture-population ]]; then
  pi05_tree_sha256=none
  pi05_hash_receipt_sha256=none
  label_publication_receipt_sha256=none
fi

if [[ "$RUN_STAGE" == population ]]; then
  [[ -n "$PAIRED_CANARY_RECEIPT_PATH" && -f "$PAIRED_CANARY_RECEIPT_PATH" && \
    ! -L "$PAIRED_CANARY_RECEIPT_PATH" ]] || {
    echo "population requires a validated PAIRED_CANARY_RECEIPT_PATH" >&2
    exit 2
  }
  case "$EXPECTED_PAIRED_CANARY_RECEIPT_SHA256" in
    *[!0-9a-f]*|"")
      echo "population requires a reviewed EXPECTED_PAIRED_CANARY_RECEIPT_SHA256" >&2
      exit 2
      ;;
  esac
  [[ ${#EXPECTED_PAIRED_CANARY_RECEIPT_SHA256} -eq 64 ]] || {
    echo "EXPECTED_PAIRED_CANARY_RECEIPT_SHA256 must contain 64 characters" >&2
    exit 2
  }
  [[ "$(json_top_level_string_value "$PAIRED_CANARY_RECEIPT_PATH" schema_version)" == \
    vlsa_table1_action_invariant_paired_canary_validation.v1 ]] || {
    echo "unexpected paired-canary receipt schema" >&2
    exit 2
  }
  [[ "$(json_top_level_string_value "$PAIRED_CANARY_RECEIPT_PATH" status)" == validated ]] || {
    echo "paired-canary receipt is not validated" >&2
    exit 2
  }
  grep -Eq '^[[:space:]]*"paired_result_valid":[[:space:]]*true,?[[:space:]]*$' \
    "$PAIRED_CANARY_RECEIPT_PATH" || {
    echo "paired-canary receipt did not validate the pair" >&2
    exit 2
  }
  for field in action_invariance_valid failure_diagnostics_valid; do
    grep -Eq "^[[:space:]]*\"$field\":[[:space:]]*true,?[[:space:]]*$" \
      "$PAIRED_CANARY_RECEIPT_PATH" || {
      echo "paired-canary receipt did not validate $field" >&2
      exit 2
    }
  done
  [[ "$(json_string_value "$PAIRED_CANARY_RECEIPT_PATH" source_git_commit)" == \
    "$EXPECTED_GIT_COMMIT" ]] || {
    echo "paired-canary source commit differs from the population release" >&2
    exit 2
  }
  [[ "$(json_string_value "$PAIRED_CANARY_RECEIPT_PATH" pi05_tree_sha256)" == \
    "$pi05_tree_sha256" ]] || {
    echo "paired-canary pi0.5 checkpoint differs from population" >&2
    exit 2
  }
  [[ "$(json_string_value "$PAIRED_CANARY_RECEIPT_PATH" groundingdino_device)" == \
    "$GROUNDINGDINO_DEVICE" ]] || {
    echo "paired-canary GroundingDINO device differs from population" >&2
    exit 2
  }
  paired_canary_receipt_sha256=$(
    sha256sum "$PAIRED_CANARY_RECEIPT_PATH" | awk '{print $1}'
  )
  [[ "$paired_canary_receipt_sha256" == \
    "$EXPECTED_PAIRED_CANARY_RECEIPT_SHA256" ]] || {
    echo "paired-canary receipt differs from its reviewed SHA-256" >&2
    exit 2
  }
else
  paired_canary_receipt_sha256=none
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
  printf 'label_publication_receipt_sha256\t%s\n' \
    "$label_publication_receipt_sha256"
  printf 'pi05_tree_sha256\t%s\n' "$pi05_tree_sha256"
  printf 'pi05_hash_receipt_sha256\t%s\n' "$pi05_hash_receipt_sha256"
  printf 'paired_canary_receipt_sha256\t%s\n' "$paired_canary_receipt_sha256"
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
