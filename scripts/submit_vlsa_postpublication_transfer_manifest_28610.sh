#!/usr/bin/env bash
set -euo pipefail

# Exact recovery-safe launch helper for the postpublication verifier.  The
# terminal publisher may be a finalize-only recovery job while immutable
# prepublication artifacts remain owned by timed-out job 28940.  The job is
# submitted held, inspected and receipted, rechecked against every frozen
# input, and only then released.  An interruption before release leaves the
# exact job held; rerunning this helper recovers it by immutable identity.

readonly POPULATION_ARRAY_JOB_ID=28609
readonly PUBLISHER_JOB_ID=${PUBLISHER_JOB_ID:-28610}
readonly ARTIFACT_PUBLISHER_JOB_ID=${ARTIFACT_PUBLISHER_JOB_ID:-$PUBLISHER_JOB_ID}
readonly TIMEOUT_ARTIFACT_PUBLISHER_JOB_ID=28940
readonly POPULATION_RUN_ID=vlsa-table1-contact-authority-population-20260718a
readonly POPULATION_SOURCE_GIT_COMMIT=1592aa59361f431ba96c6ddcbebcb596f6c20853
readonly EXPECTED_JOB_NAME=vlsa-tx-p${PUBLISHER_JOB_ID}
readonly EXPECTED_REMOTE_REPO=/home/quanth/working_space/vlsa-aegis-table-repro
readonly EXPECTED_EXPERIMENT_ROOT=/mnt/data/quanth/experiments/vlsa-aegis-table1
readonly EXPECTED_TRANSFER_OUTPUT_ROOT=/mnt/data/quanth/experiments/vlsa-aegis-table1-transfer

: "${EXPECTED_PUBLICATION_RECEIPT_SHA256:?set the reviewed publication receipt SHA-256}"
: "${EXPECTED_VERIFIER_SHA256:?set the reviewed verifier script SHA-256}"
: "${EXPECTED_VERIFIER_GIT_COMMIT:?set the exact verifier repository commit}"
: "${EXPECTED_RUNNER_SHA256:?set the reviewed allocation runner SHA-256}"
: "${EXPECTED_SBATCH_SHA256:?set the reviewed SBatch SHA-256}"
: "${LABEL_MANIFEST_PATH:?set the immutable 1,600-row label manifest}"

RUN_ID=${RUN_ID:-$POPULATION_RUN_ID}
EXPECTED_SOURCE_GIT_COMMIT=${EXPECTED_SOURCE_GIT_COMMIT:-$POPULATION_SOURCE_GIT_COMMIT}
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1}
RUN_ROOT=${RUN_ROOT:-$EXPERIMENT_ROOT/$RUN_ID}
PUBLICATION_RECEIPT=${PUBLICATION_RECEIPT:-$RUN_ROOT/population-publication-receipt.json}
TRANSFER_OUTPUT_ROOT=${TRANSFER_OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1-transfer}
TRANSFER_OUTPUT_DIR=${TRANSFER_OUTPUT_DIR:-$TRANSFER_OUTPUT_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID}
VERIFIER=$REMOTE_REPO/scripts/vlsa_postpublication_transfer_manifest.py
RUNNER=$REMOTE_REPO/slurm/run_vlsa_postpublication_transfer_manifest.sh
SBATCH=$REMOTE_REPO/slurm/vlsa_postpublication_transfer_manifest.sbatch
CONTROL_ROOT=$TRANSFER_OUTPUT_ROOT/.postpublication-control
CONTROL_DIR=$CONTROL_ROOT/$RUN_ID-publisher-$PUBLISHER_JOB_ID
INTENT_PATH=$CONTROL_DIR/submission-intent.tsv
JOB_ID_PATH=$CONTROL_DIR/job-id.txt
SCONTROL_PATH=$CONTROL_DIR/held-job-scontrol.txt
SUBMISSION_RECEIPT=$CONTROL_DIR/submission-receipt.tsv
SUBMISSION_RECEIPT_SHA=$CONTROL_DIR/submission-receipt.sha256
RELEASE_RECEIPT=$CONTROL_DIR/release-receipt.tsv
RELEASE_RECEIPT_SHA=$CONTROL_DIR/release-receipt.sha256

die() {
  echo "postpublication launch rejected: $*" >&2
  exit 2
}

require_scalar() {
  local label=$1
  local value=$2
  case "$value" in
    *$'\n'*|*$'\r'*|*$'\t'*)
      die "$label contains a control delimiter"
      ;;
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

publish_exact_file() {
  local destination=$1
  local temporary
  temporary=$(mktemp "$CONTROL_DIR/.receipt.XXXXXX")
  umask 077
  if ! command cat >"$temporary"; then
    rm -f "$temporary"
    die "cannot stage immutable receipt $destination"
  fi
  chmod 0444 "$temporary"
  sync -f "$temporary"
  if [[ -e "$destination" || -L "$destination" ]]; then
    if [[ -L "$destination" || ! -f "$destination" ]] || \
      ! cmp -s "$temporary" "$destination"; then
      rm -f "$temporary"
      die "immutable receipt differs: $destination"
    fi
    rm -f "$temporary"
    return
  fi
  if ! ln "$temporary" "$destination"; then
    rm -f "$temporary"
    die "cannot atomically publish immutable receipt $destination"
  fi
  rm -f "$temporary"
  sync -f "$CONTROL_DIR"
}

publish_receipt_sha256() {
  local receipt=$1
  local sidecar=$2
  local digest
  digest=$(sha256sum "$receipt" | awk '{print $1}')
  printf '%s  %s\n' "$digest" "$(basename "$receipt")" |
    publish_exact_file "$sidecar"
}

verify_receipt_sha256() {
  local receipt=$1
  local sidecar=$2
  [[ -f "$receipt" && ! -L "$receipt" ]] || \
    die "immutable receipt is missing: $receipt"
  [[ -f "$sidecar" && ! -L "$sidecar" ]] || \
    die "immutable receipt hash is missing: $sidecar"
  (
    cd "$(dirname "$receipt")"
    sha256sum --check --status "$(basename "$sidecar")"
  ) || die "immutable receipt hash differs: $receipt"
}

tsv_field() {
  local path=$1
  local key=$2
  awk -F'\t' -v key="$key" '$1 == key {print substr($0, length($1) + 2); exit}' "$path"
}

scontrol_field() {
  local path=$1
  local key=$2
  awk -v key="$key" '{
    for (index = 1; index <= NF; index++) {
      prefix = key "="
      if (substr($index, 1, length(prefix)) == prefix) {
        print substr($index, length(prefix) + 1)
        exit
      }
    }
  }' "$path"
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

require_tsv_field() {
  local path=$1
  local key=$2
  local expected=$3
  local observed
  observed=$(tsv_field "$path" "$key")
  [[ "$observed" == "$expected" ]] || \
    die "immutable receipt $key differs: observed=$observed expected=$expected"
}

validate_submission_receipt() {
  verify_receipt_sha256 "$SUBMISSION_RECEIPT" "$SUBMISSION_RECEIPT_SHA"
  local job_id=$1
  require_tsv_field "$SUBMISSION_RECEIPT" schema_version \
    vlsa_postpublication_submission_receipt.v1
  require_tsv_field "$SUBMISSION_RECEIPT" status held_validated
  require_tsv_field "$SUBMISSION_RECEIPT" job_id "$job_id"
  require_tsv_field "$SUBMISSION_RECEIPT" job_name "$EXPECTED_JOB_NAME"
  require_tsv_field "$SUBMISSION_RECEIPT" population_array_job_id \
    "$POPULATION_ARRAY_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" publisher_job_id \
    "$PUBLISHER_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" artifact_publisher_job_id \
    "$ARTIFACT_PUBLISHER_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" dependency \
    "afterok:$PUBLISHER_JOB_ID"
  require_tsv_field "$SUBMISSION_RECEIPT" source_git_commit \
    "$EXPECTED_SOURCE_GIT_COMMIT"
  require_tsv_field "$SUBMISSION_RECEIPT" publication_receipt_sha256 \
    "$EXPECTED_PUBLICATION_RECEIPT_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" verifier_git_commit \
    "$EXPECTED_VERIFIER_GIT_COMMIT"
  require_tsv_field "$SUBMISSION_RECEIPT" verifier_sha256 \
    "$EXPECTED_VERIFIER_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" runner_sha256 \
    "$EXPECTED_RUNNER_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" sbatch_sha256 \
    "$EXPECTED_SBATCH_SHA256"
  require_tsv_field "$SUBMISSION_RECEIPT" output_unused true
  require_tsv_field "$SUBMISSION_RECEIPT" cpu_only true
  require_tsv_field "$SUBMISSION_RECEIPT" held_before_release true
  [[ -f "$SCONTROL_PATH" && ! -L "$SCONTROL_PATH" ]] || \
    die "held-job scontrol snapshot is missing"
  require_tsv_field "$SUBMISSION_RECEIPT" scontrol_snapshot_sha256 \
    "$(sha256sum "$SCONTROL_PATH" | awk '{print $1}')"
}

validate_release_receipt() {
  local job_id=$1
  validate_submission_receipt "$job_id"
  verify_receipt_sha256 "$RELEASE_RECEIPT" "$RELEASE_RECEIPT_SHA"
  require_tsv_field "$RELEASE_RECEIPT" schema_version \
    vlsa_postpublication_release_receipt.v1
  require_tsv_field "$RELEASE_RECEIPT" status released
  require_tsv_field "$RELEASE_RECEIPT" job_id "$job_id"
  require_tsv_field "$RELEASE_RECEIPT" job_name "$EXPECTED_JOB_NAME"
  require_tsv_field "$RELEASE_RECEIPT" publisher_job_id "$PUBLISHER_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" artifact_publisher_job_id \
    "$ARTIFACT_PUBLISHER_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" dependency \
    "afterok:$PUBLISHER_JOB_ID"
  require_tsv_field "$RELEASE_RECEIPT" submission_receipt_sha256 \
    "$(sha256sum "$SUBMISSION_RECEIPT" | awk '{print $1}')"
  require_tsv_field "$RELEASE_RECEIPT" output_unused_before_release true
  case "$(tsv_field "$RELEASE_RECEIPT" release_action)" in
    released_exact_held_job|recovered_after_prior_release) ;;
    *) die "release receipt action is invalid" ;;
  esac
  case "$(tsv_field "$RELEASE_RECEIPT" observed_reason_after_action)" in
    *Held*) die "release receipt retains a held state" ;;
  esac
}

validate_frozen_inputs() {
  [[ -d "$REMOTE_REPO/.git" || -f "$REMOTE_REPO/.git" ]] || \
    die "remote repository is missing"
  [[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == \
    "$EXPECTED_VERIFIER_GIT_COMMIT" ]] || \
    die "remote repository is not at the reviewed verifier commit"
  [[ -z "$(git -C "$REMOTE_REPO" status --short --untracked-files=all)" ]] || \
    die "remote repository is dirty"
  for path in "$VERIFIER" "$RUNNER" "$SBATCH" "$LABEL_MANIFEST_PATH"; do
    [[ -f "$path" && ! -L "$path" ]] || \
      die "reviewed source/input is missing or symlinked: $path"
  done
  [[ "$(sha256sum "$VERIFIER" | awk '{print $1}')" == \
    "$EXPECTED_VERIFIER_SHA256" ]] || die "verifier SHA-256 changed"
  [[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == \
    "$EXPECTED_RUNNER_SHA256" ]] || die "runner SHA-256 changed"
  [[ "$(sha256sum "$SBATCH" | awk '{print $1}')" == \
    "$EXPECTED_SBATCH_SHA256" ]] || die "SBatch SHA-256 changed"
  [[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || \
    die "immutable population run root is missing or symlinked"
  [[ -f "$PUBLICATION_RECEIPT" && ! -L "$PUBLICATION_RECEIPT" ]] || \
    die "publication receipt is missing or symlinked"
  [[ "$(sha256sum "$PUBLICATION_RECEIPT" | awk '{print $1}')" == \
    "$EXPECTED_PUBLICATION_RECEIPT_SHA256" ]] || \
    die "publication receipt SHA-256 changed"
  [[ -d "$TRANSFER_OUTPUT_ROOT" && ! -L "$TRANSFER_OUTPUT_ROOT" ]] || \
    die "transfer output root must already exist and be a real directory"
  [[ ! -e "$TRANSFER_OUTPUT_DIR" && ! -L "$TRANSFER_OUTPUT_DIR" ]] || \
    die "transfer output directory already exists"

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
    if [[ -L "$SCONTROL_PATH" || ! -f "$SCONTROL_PATH" ]]; then
      rm -f "$temporary"
      die "held-job scontrol snapshot is not a regular file"
    fi
    local field
    for field in JobId JobName JobState Reason Partition \
      Account QOS NumNodes NumCPUs NumTasks "CPUs/Task" MinMemoryNode \
      TimeLimit Requeue ExcNodeList Command WorkDir StdOut ReqTRES \
      AllocTRES TresPerNode TresPerTask Gres; do
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
      die "cannot publish held-job scontrol snapshot"
    fi
    rm -f "$temporary"
    sync -f "$CONTROL_DIR"
  fi
}

for value in \
  "$EXPECTED_SOURCE_GIT_COMMIT" "$EXPECTED_VERIFIER_GIT_COMMIT"; do
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
  "$EXPECTED_VERIFIER_SHA256" \
  "$EXPECTED_RUNNER_SHA256" \
  "$EXPECTED_SBATCH_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die "expected SHA-256 is not 64 hex"
done
[[ "$RUN_ID" == "$POPULATION_RUN_ID" ]] || \
  die "run ID differs from the exact population"
[[ "$EXPECTED_SOURCE_GIT_COMMIT" == "$POPULATION_SOURCE_GIT_COMMIT" ]] || \
  die "source commit differs from the exact population"
[[ "$REMOTE_REPO" == "$EXPECTED_REMOTE_REPO" ]] || \
  die "remote repository path differs from the exact release"
[[ "$EXPERIMENT_ROOT" == "$EXPECTED_EXPERIMENT_ROOT" ]] || \
  die "experiment root differs from the exact population"
[[ "$RUN_ROOT" == "$EXPECTED_EXPERIMENT_ROOT/$POPULATION_RUN_ID" ]] || \
  die "run root differs from the exact population"
[[ "$PUBLICATION_RECEIPT" == \
  "$RUN_ROOT/population-publication-receipt.json" ]] || \
  die "publication receipt path differs from the exact population"
[[ "$TRANSFER_OUTPUT_ROOT" == "$EXPECTED_TRANSFER_OUTPUT_ROOT" ]] || \
  die "transfer output root differs from the reviewed destination"
[[ "$TRANSFER_OUTPUT_DIR" == \
  "$TRANSFER_OUTPUT_ROOT/$POPULATION_RUN_ID-publisher-$PUBLISHER_JOB_ID" ]] || \
  die "transfer output directory differs from the reviewed destination"
case "$LABEL_MANIFEST_PATH" in
  /*) ;;
  *) die "label manifest path must be absolute" ;;
esac
for name in PUBLISHER_JOB_ID ARTIFACT_PUBLISHER_JOB_ID \
  RUN_ID EXPECTED_SOURCE_GIT_COMMIT \
  EXPECTED_PUBLICATION_RECEIPT_SHA256 EXPECTED_VERIFIER_SHA256 \
  EXPECTED_VERIFIER_GIT_COMMIT EXPECTED_RUNNER_SHA256 \
  EXPECTED_SBATCH_SHA256 LABEL_MANIFEST_PATH REMOTE_REPO \
  EXPERIMENT_ROOT RUN_ROOT PUBLICATION_RECEIPT TRANSFER_OUTPUT_ROOT \
  TRANSFER_OUTPUT_DIR; do
  require_export_scalar "$name" "${!name}"
done
require_scalar CONTROL_DIR "$CONTROL_DIR"
require_scalar EXPECTED_JOB_NAME "$EXPECTED_JOB_NAME"

validate_frozen_inputs
if [[ ! -e "$CONTROL_ROOT" && ! -L "$CONTROL_ROOT" ]]; then
  mkdir "$CONTROL_ROOT"
fi
[[ -d "$CONTROL_ROOT" && ! -L "$CONTROL_ROOT" ]] || \
  die "postpublication control root is not a real directory"
if [[ ! -e "$CONTROL_DIR" && ! -L "$CONTROL_DIR" ]]; then
  mkdir "$CONTROL_DIR"
fi
[[ -d "$CONTROL_DIR" && ! -L "$CONTROL_DIR" ]] || \
  die "postpublication control directory is not a real directory"

[[ ! -L "$CONTROL_DIR/.lock" ]] || \
  die "postpublication control lock must not be a symlink"
exec 9>"$CONTROL_DIR/.lock"
flock -n 9 || die "another exact postpublication launch is active"

{
  printf 'schema_version\tvlsa_postpublication_submission_intent.v1\n'
  printf 'status\tfrozen\n'
  printf 'run_id\t%s\n' "$RUN_ID"
  printf 'population_array_job_id\t%s\n' "$POPULATION_ARRAY_JOB_ID"
  printf 'publisher_job_id\t%s\n' "$PUBLISHER_JOB_ID"
  printf 'artifact_publisher_job_id\t%s\n' "$ARTIFACT_PUBLISHER_JOB_ID"
  printf 'dependency\tafterok:%s\n' "$PUBLISHER_JOB_ID"
  printf 'job_name\t%s\n' "$EXPECTED_JOB_NAME"
  printf 'population_source_git_commit\t%s\n' "$EXPECTED_SOURCE_GIT_COMMIT"
  printf 'publication_receipt_sha256\t%s\n' "$EXPECTED_PUBLICATION_RECEIPT_SHA256"
  printf 'verifier_git_commit\t%s\n' "$EXPECTED_VERIFIER_GIT_COMMIT"
  printf 'verifier_sha256\t%s\n' "$EXPECTED_VERIFIER_SHA256"
  printf 'runner_sha256\t%s\n' "$EXPECTED_RUNNER_SHA256"
  printf 'sbatch_sha256\t%s\n' "$EXPECTED_SBATCH_SHA256"
  printf 'run_root\t%s\n' "$RUN_ROOT"
  printf 'transfer_output_dir\t%s\n' "$TRANSFER_OUTPUT_DIR"
  printf 'resources\tpartition=main,account=normal,qos=normal,nodes=1,ntasks=1,cpus=4,mem=32G,time=04:00:00,gpus=0,exclude=worker-3\n'
} | publish_exact_file "$INTENT_PATH"

if [[ -f "$RELEASE_RECEIPT" || -L "$RELEASE_RECEIPT" ]]; then
  existing_job_id=$(tsv_field "$RELEASE_RECEIPT" job_id)
  [[ "$existing_job_id" =~ ^[0-9]+$ ]] || \
    die "release receipt has an invalid job ID"
  publish_receipt_sha256 \
    "$SUBMISSION_RECEIPT" "$SUBMISSION_RECEIPT_SHA"
  publish_receipt_sha256 "$RELEASE_RECEIPT" "$RELEASE_RECEIPT_SHA"
  validate_release_receipt "$existing_job_id"
  printf '%s\n' "$existing_job_id"
  exit 0
fi

job_id=
if [[ -f "$JOB_ID_PATH" && ! -L "$JOB_ID_PATH" ]]; then
  job_id=$(tr -d '\n' <"$JOB_ID_PATH")
else
  mapfile -t recovered_jobs < <(
    squeue --noheader --user="${USER:?USER is required}" \
      --name="$EXPECTED_JOB_NAME" --format='%A' |
      awk 'NF {print $1}'
  )
  if (( ${#recovered_jobs[@]} > 1 )); then
    die "multiple active jobs have the exact recovery job name"
  fi
  if (( ${#recovered_jobs[@]} == 1 )); then
    job_id=${recovered_jobs[0]}
  else
    export_spec="RUN_ID=$RUN_ID"
    for name in PUBLISHER_JOB_ID ARTIFACT_PUBLISHER_JOB_ID \
      EXPECTED_SOURCE_GIT_COMMIT \
      EXPECTED_PUBLICATION_RECEIPT_SHA256 EXPECTED_VERIFIER_SHA256 \
      EXPECTED_VERIFIER_GIT_COMMIT EXPECTED_RUNNER_SHA256 \
      EXPECTED_SBATCH_SHA256 LABEL_MANIFEST_PATH REMOTE_REPO \
      EXPERIMENT_ROOT RUN_ROOT PUBLICATION_RECEIPT \
      TRANSFER_OUTPUT_ROOT TRANSFER_OUTPUT_DIR; do
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
fi
[[ "$job_id" =~ ^[0-9]+$ ]] || die "recorded job ID is invalid"

if [[ ! -f "$SUBMISSION_RECEIPT" && ! -L "$SUBMISSION_RECEIPT" ]]; then
  inspect_held_job "$job_id"
  scontrol_sha256=$(sha256sum "$SCONTROL_PATH" | awk '{print $1}')
  {
    printf 'schema_version\tvlsa_postpublication_submission_receipt.v1\n'
    printf 'status\theld_validated\n'
    printf 'job_id\t%s\n' "$job_id"
    printf 'job_name\t%s\n' "$EXPECTED_JOB_NAME"
    printf 'population_array_job_id\t%s\n' "$POPULATION_ARRAY_JOB_ID"
    printf 'publisher_job_id\t%s\n' "$PUBLISHER_JOB_ID"
    printf 'artifact_publisher_job_id\t%s\n' "$ARTIFACT_PUBLISHER_JOB_ID"
    printf 'dependency\tafterok:%s\n' "$PUBLISHER_JOB_ID"
    printf 'source_git_commit\t%s\n' "$EXPECTED_SOURCE_GIT_COMMIT"
    printf 'publication_receipt_sha256\t%s\n' "$EXPECTED_PUBLICATION_RECEIPT_SHA256"
    printf 'verifier_git_commit\t%s\n' "$EXPECTED_VERIFIER_GIT_COMMIT"
    printf 'verifier_sha256\t%s\n' "$EXPECTED_VERIFIER_SHA256"
    printf 'runner_sha256\t%s\n' "$EXPECTED_RUNNER_SHA256"
    printf 'sbatch_sha256\t%s\n' "$EXPECTED_SBATCH_SHA256"
    printf 'scontrol_snapshot_sha256\t%s\n' "$scontrol_sha256"
    printf 'output_unused\ttrue\n'
    printf 'cpu_only\ttrue\n'
    printf 'held_before_release\ttrue\n'
  } | publish_exact_file "$SUBMISSION_RECEIPT"
  publish_receipt_sha256 "$SUBMISSION_RECEIPT" "$SUBMISSION_RECEIPT_SHA"
else
  publish_receipt_sha256 \
    "$SUBMISSION_RECEIPT" "$SUBMISSION_RECEIPT_SHA"
  validate_submission_receipt "$job_id"
fi

validate_frozen_inputs
validate_submission_receipt "$job_id"
submission_receipt_sha256=$(sha256sum "$SUBMISSION_RECEIPT" | awk '{print $1}')

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
  inspect_held_job "$job_id"
  scontrol release "$job_id"
  release_action=released_exact_held_job
else
  if [[ -z "$current_state" ]]; then
    current_state=$(
      sacct -n -X -j "$job_id" --format=JobIDRaw,State -P |
        awk -F'|' -v job="$job_id" '$1 == job {print $2}' |
        awk '{print $1; exit}'
    )
  fi
  case "$current_state" in
    PENDING)
      case "$current_reason" in
        *Held*) die "exact job remains held for reason $current_reason" ;;
      esac
      release_action=recovered_after_prior_release
      ;;
    RUNNING|COMPLETING|COMPLETED|FAILED|TIMEOUT|OUT_OF_MEMORY)
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
  printf 'schema_version\tvlsa_postpublication_release_receipt.v1\n'
  printf 'status\treleased\n'
  printf 'job_id\t%s\n' "$job_id"
  printf 'job_name\t%s\n' "$EXPECTED_JOB_NAME"
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
